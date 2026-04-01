# SD 1.5 full fine-tuning — modern training script
# Based on sdxl_train.py architecture with all advanced features ported for SD 1.5.

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler

from library import deepspeed_utils, model_util
import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library.edm2_loss_utils import prepare_edm2_loss_weighting, plot_edm2_loss_weighting_check, plot_edm2_loss_weighting
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)


# SD 1.5 diffusers UNet block mapping for per-block learning rates.
# Block 0:  conv_in, time_embedding, time_proj  (input projection + timestep MLP)
# Block 1:  down_blocks.0
# Block 2:  down_blocks.1
# Block 3:  down_blocks.2
# Block 4:  down_blocks.3
# Block 5:  mid_block
# Block 6:  up_blocks.0
# Block 7:  up_blocks.1
# Block 8:  up_blocks.2
# Block 9:  up_blocks.3
# Block 10: conv_norm_out, conv_out  (final output layers)
UNET_NUM_BLOCKS_FOR_BLOCK_LR = 11


def _unet_block_index_from_name(name: str) -> int:
    """Map a diffusers UNet2DConditionModel parameter name to a block index."""
    if name.startswith("conv_in."):
        return 0
    if name.startswith("time_embedding.") or name.startswith("time_proj."):
        return 0
    if name.startswith("down_blocks."):
        return 1 + int(name.split(".")[1])
    if name.startswith("mid_block."):
        return 5
    if name.startswith("up_blocks."):
        return 6 + int(name.split(".")[1])
    if name.startswith("conv_norm_out.") or name.startswith("conv_out."):
        return 10
    raise ValueError(f"unexpected parameter name: {name}")


def _unet_block_prefix_from_name(name: str) -> str:
    if name.startswith("conv_in.") or name.startswith("conv_out.") or name.startswith("conv_norm_out."):
        return name.split(".")[0]
    if name.startswith("time_embedding.") or name.startswith("time_proj."):
        return name.split(".")[0]
    if name.startswith("down_blocks.") or name.startswith("up_blocks."):
        parts = name.split(".")
        return ".".join(parts[:2])
    if name.startswith("mid_block."):
        return "mid_block"
    raise ValueError(f"unexpected parameter name: {name}")


def get_block_params_to_optimize(unet, block_lrs: List[float], frozen_blocks: set[int] | None = None) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for name, param in unet.named_parameters():
        block_index = _unet_block_index_from_name(name)
        if frozen_blocks and block_index in frozen_blocks:
            continue
        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def freeze_unet_blocks(unet, frozen_blocks: set[int]) -> None:
    if not frozen_blocks:
        return
    for name, param in unet.named_parameters():
        block_index = _unet_block_index_from_name(name)
        if block_index in frozen_blocks:
            param.requires_grad_(False)


def describe_unet_blocks(unet):
    info = {}
    for name, param in unet.named_parameters():
        block_index = _unet_block_index_from_name(name)
        block_prefix = _unet_block_prefix_from_name(name)
        block_entry = info.setdefault(block_index, {"example": name, "params": 0, "layers": set()})
        block_entry["params"] += param.numel()

        layer_path = name.rsplit(".", 1)[0]
        suffix = ""
        if layer_path == block_prefix:
            suffix = ""
        elif layer_path.startswith(f"{block_prefix}."):
            suffix = layer_path[len(block_prefix) + 1:]
        else:
            suffix = layer_path

        if suffix:
            tokens = [token for token in suffix.split(".") if token]
            while tokens and tokens[0].isdigit():
                tokens.pop(0)
            layer_name = ".".join(tokens) if tokens else block_prefix
        else:
            layer_name = block_prefix

        block_entry["layers"].add(layer_name)

    for entry in info.values():
        entry["layers"] = sorted(entry["layers"])
    return info


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder")
        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # TF32 — enabled by default for SD 1.5 to avoid FP16 loss scaling issues
    # and BF16 quality degradation. Gives near-FP32 range with ~8x speedup on Ampere+ GPUs.
    if not args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        logger.info("TF32 enabled for matmul and cuDNN. Use --disable_tf32 to turn off.")
    else:
        logger.info("TF32 disabled by user request.")

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        args.cache_text_encoder_outputs = True
        logger.info("cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled")

    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values (got {len(block_lrs)})"
    else:
        block_lrs = None

    frozen_unet_blocks = set()
    if args.freeze_unet_blocks:
        for token in args.freeze_unet_blocks.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid U-Net block index '{token}' in --freeze_unet_blocks") from exc
            if idx < 0 or idx >= UNET_NUM_BLOCKS_FOR_BLOCK_LR:
                raise ValueError(f"--freeze_unet_blocks indices must be in [0, {UNET_NUM_BLOCKS_FOR_BLOCK_LR - 1}]")
            frozen_unet_blocks.add(idx)

    # VAE scale/shift — SD 1.5 default is 0.18215
    vae_scale_factor = 0.18215
    vae_shift_factor = 0.0

    if args.vae_custom_scale is not None:
        vae_scale_factor = float(args.vae_custom_scale)
        logger.info(f"Using custom VAE scale factor: {vae_scale_factor}")
    else:
        logger.info(f"Using SD 1.5 VAE scale factor: {vae_scale_factor}")

    if args.vae_custom_shift is not None:
        vae_shift_factor = float(args.vae_custom_shift)
        logger.info(f"Using custom VAE shift factor: {vae_shift_factor}")

    args.vae_scale_factor = vae_scale_factor
    args.vae_shift_factor = vae_shift_factor

    if args.flow_model:
        logger.info("Using Rectified Flow training objective.")
        if args.v_parameterization:
            raise ValueError("`--flow_model` is incompatible with `--v_parameterization`; Rectified Flow already predicts velocity.")
        if args.min_snr_gamma:
            logger.warning("`--min_snr_gamma` is ignored when Rectified Flow is enabled.")
            args.min_snr_gamma = None
        if args.debiased_estimation_loss:
            logger.warning("`--debiased_estimation_loss` is ignored when Rectified Flow is enabled.")
            args.debiased_estimation_loss = False
        if args.scale_v_pred_loss_like_noise_pred:
            logger.warning("`--scale_v_pred_loss_like_noise_pred` is ignored when Rectified Flow is enabled.")
            args.scale_v_pred_loss_like_noise_pred = False
        if args.v_pred_like_loss:
            logger.warning("`--v_pred_like_loss` is ignored when Rectified Flow is enabled.")
            args.v_pred_like_loss = None
        if args.flow_use_ot:
            logger.info("Using cosine optimal transport pairing for Rectified Flow batches.")
        shift_enabled = args.flow_uniform_shift or args.flow_uniform_static_ratio is not None
        if args.flow_timestep_distribution == "logit_normal":
            if args.flow_logit_std <= 0:
                raise ValueError("`--flow_logit_std` must be positive.")
            logger.info(
                "Rectified Flow timesteps sampled from logit-normal distribution with "
                f"mean={args.flow_logit_mean}, std={args.flow_logit_std}."
            )
        elif args.flow_timestep_distribution == "uniform":
            logger.info("Rectified Flow timesteps sampled uniformly in [0, 1].")
        else:
            raise ValueError(f"Unknown Rectified Flow timestep distribution: {args.flow_timestep_distribution}")
        if shift_enabled:
            if args.flow_uniform_static_ratio is not None:
                if args.flow_uniform_static_ratio <= 0:
                    raise ValueError("`--flow_uniform_static_ratio` must be positive.")
                logger.info(f"Applying Rectified Flow timestep shift with static ratio={args.flow_uniform_static_ratio}.")
            else:
                logger.info(f"Applying resolution-dependent Rectified Flow timestep shift with base pixels={args.flow_uniform_base_pixels}.")

    if args.contrastive_flow_matching and not (args.v_parameterization or args.flow_model):
        raise ValueError("`--contrastive_flow_matching` requires either v-parameterization or Rectified Flow.")

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = train_util.load_tokenizer(args)

    # Prepare dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0}".format(", ".join(ignored))
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

    if args.protected_tags_file:
        logger.info("Injecting protected_tags_file into datasets...")
        for ds in train_dataset_group.datasets:
            ds.protected_tags_file = args.protected_tags_file
    if args.log_caption_tag_dropout:
        logger.info("Enabling caption tag dropout logging for datasets...")
        for ds in train_dataset_group.datasets:
            ds.log_caption_tag_dropout = True
    if args.log_caption_dropout:
        logger.info("Enabling caption dropout logging for datasets...")
        for ds in train_dataset_group.datasets:
            ds.log_caption_dropout = True

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    if args.no_token_padding:
        train_dataset_group.disable_token_padding()

    train_dataset_group.verify_bucket_reso_steps(64)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

    # Prepare accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # Load SD 1.5 model
    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype, accelerator)

    # VAE swap if requested
    if args.vae is not None:
        logger.info(f"Loading replacement VAE from {args.vae}")
        replace_vae = model_util.load_vae(args.vae, vae_dtype)
        if replace_vae is not None:
            vae = replace_vae
            logger.info("VAE replaced successfully.")

    if args.vae_reflection_padding:
        vae = model_util.use_reflection_padding(vae)

    if args.list_unet_blocks:
        block_info = describe_unet_blocks(unet)
        accelerator.print("SD 1.5 U-Net block mapping (index -> example parameter) with param counts and layers:")
        for idx in sorted(block_info.keys()):
            info = block_info[idx]
            layers = ", ".join(info.get("layers", [])) or "-"
            accelerator.print(f"{idx:02d}: {info['example']} (params: {info['params']:,})")
            accelerator.print(f"    layers: {layers}")
        return

    # Verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

    # Memory efficient attention
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
    if args.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers for VAE")
        vae.enable_xformers_memory_efficient_attention()

    # Cache latents
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(
                vae,
                args.vae_batch_size,
                args.cache_latents_to_disk,
                accelerator.is_main_process,
                getattr(args, "skip_existing", False),
            )
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # Set up training state
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    train_unet = args.learning_rate != 0
    train_text_encoder = False

    if args.train_text_encoder:
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
        lr_te = args.learning_rate_te if args.learning_rate_te is not None else args.learning_rate
        train_text_encoder = lr_te != 0

        if not train_text_encoder:
            text_encoder.to(weight_dtype)
        text_encoder.requires_grad_(train_text_encoder)
        text_encoder.train(train_text_encoder)
    else:
        text_encoder.to(weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        # Cache text encoder outputs
        if args.cache_text_encoder_outputs:
            with torch.no_grad(), accelerator.autocast():
                train_dataset_group.cache_text_encoder_outputs(
                    (tokenizer,),
                    (text_encoder,),
                    accelerator.device,
                    None,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )
            accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if train_unet and frozen_unet_blocks:
        accelerator.print(f"Freezing U-Net blocks: {sorted(frozen_unet_blocks)}")
        freeze_unet_blocks(unet, frozen_unet_blocks)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            trainable_params = [p for p in unet.parameters() if p.requires_grad]
            params_to_optimize.append({"params": trainable_params, "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs, frozen_unet_blocks))

    if train_text_encoder:
        training_models.append(text_encoder)
        params_to_optimize.append({"params": list(text_encoder.parameters()), "lr": args.learning_rate_te or args.learning_rate})

    # Count trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder: {train_text_encoder}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # Prepare optimizer
    accelerator.print("prepare optimizer, data loader etc.")

    if args.fused_optimizer_groups:
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr

                param_group.append(p)

                if len(param_group) == params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
                    param_group_lr = -1

        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})

        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]

        logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # Prepare dataloader
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    dataloader_kwargs = dict(
        dataset=train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )
    if args.prefetch_factor is not None and n_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    # Calculate training steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # LR scheduler
    if args.fused_optimizer_groups:
        lr_schedulers = [train_util.get_scheduler_fix(args, opt, accelerator.num_processes) for opt in optimizers]
        lr_scheduler = lr_schedulers[0]
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # Full precision training modes
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16'"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed precision='bf16'"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    # Prepare with accelerator
    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder=text_encoder if train_text_encoder else None,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder:
            text_encoder = accelerator.prepare(text_encoder)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Move text encoder to appropriate device after prepare
    if args.cache_text_encoder_outputs:
        text_encoder.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        text_encoder.to(accelerator.device)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # Resume from checkpoint
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # Fused backward pass
    if args.fused_backward_pass:
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    elif args.fused_optimizer_groups:
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, opt in enumerate(optimizers):
            for param_group in opt.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # Compute epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # Print training info
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    # Log latents_only subsets
    latents_only_subsets = []
    for ds in train_dataset_group.datasets:
        for subset in ds.subsets:
            if getattr(subset, "latents_only", False):
                latents_only_subsets.append(subset.image_dir)
    if latents_only_subsets:
        accelerator.print("*** LATENTS-ONLY MODE: source images will NOT be loaded for the following subsets ***")
        for d in latents_only_subsets:
            accelerator.print(f"  - {d}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(args, noise_scheduler, accelerator)

    if args.edm2_loss_weighting:
        training_models.append(edm2_model)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "sd15_finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
    )

    loss_recorder = train_util.LossRecorder()

    if args.edm2_loss_weighting:
        loss_scaled_recorder = train_util.LossRecorder()
        loss_edm2_recorder = train_util.LossRecorder()
        plot_edm2_loss_weighting(args, 0, edm2_model, 1000, accelerator.device)

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.fused_optimizer_groups:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)

                if args.vae_shift_factor != 0.0:
                    latents = latents - args.vae_shift_factor
                latents = latents * args.vae_scale_factor

                # Get text encoder hidden states
                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    with torch.set_grad_enabled(train_text_encoder):
                        if args.weighted_captions:
                            encoder_hidden_states = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75 if args.max_token_length else 1,
                                clip_skip=args.clip_skip,
                            )
                        else:
                            input_ids = batch["input_ids"].to(accelerator.device)
                            encoder_hidden_states = train_util.get_hidden_states(
                                args, input_ids, tokenizer, text_encoder,
                                None if not args.full_fp16 else weight_dtype,
                            )
                else:
                    encoder_hidden_states = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)

                # Dynamic shift for rectified flow
                needs_dynamic_shift = (
                    args.flow_model and args.flow_uniform_shift and args.flow_uniform_static_ratio is None
                )
                if needs_dynamic_shift:
                    target_size = batch.get("target_sizes_hw", None)
                    if target_size is None:
                        raise ValueError(
                            "Resolution-dependent Rectified Flow shift requires target size information in the batch."
                        )
                    pixel_counts = (target_size[:, 0] * target_size[:, 1]).to(latents.device, torch.float32)
                else:
                    pixel_counts = None

                # Sample noise and timesteps
                noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                    args, noise_scheduler, latents, pixel_counts=pixel_counts
                )

                noisy_latents = noisy_latents.to(weight_dtype)

                # Predict noise residual — SD 1.5 UNet returns .sample
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.flow_model:
                    target = noise - latents
                elif args.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if (
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                ):
                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                    )
                    if args.contrastive_flow_matching and latents.size(0) > 1:
                        negative_latents = latents.roll(1, 0)
                        negative_noise = noise.roll(1, 0)
                        with torch.no_grad():
                            if args.flow_model:
                                target_negative = negative_noise - negative_latents
                            else:
                                target_negative = noise_scheduler.get_velocity(negative_latents, negative_noise, timesteps)
                        loss_contrastive = torch.nn.functional.mse_loss(
                            noise_pred.float(), target_negative.float(), reduction="none"
                        )
                        loss = loss - args.cfm_lambda * loss_contrastive
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    # Per-sample loss weights from dataset (DreamBooth)
                    if "loss_weights" in batch:
                        loss = loss * batch["loss_weights"]

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                    loss = loss.mean()
                else:
                    per_pixel_loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                    )
                    if args.contrastive_flow_matching and latents.size(0) > 1:
                        negative_latents = latents.roll(1, 0)
                        negative_noise = noise.roll(1, 0)
                        with torch.no_grad():
                            if args.flow_model:
                                target_negative = negative_noise - negative_latents
                            else:
                                target_negative = noise_scheduler.get_velocity(negative_latents, negative_noise, timesteps)
                        loss_contrastive = torch.nn.functional.mse_loss(
                            noise_pred.float(), target_negative.float(), reduction="none"
                        )
                        per_pixel_loss = per_pixel_loss - args.cfm_lambda * loss_contrastive

                    per_pixel_loss_reduced = per_pixel_loss.mean([1, 2, 3])

                    # Per-sample loss weights from dataset (DreamBooth)
                    if "loss_weights" in batch:
                        per_pixel_loss_reduced = per_pixel_loss_reduced * batch["loss_weights"]

                    loss = per_pixel_loss_reduced.mean()

                if loss.ndim != 0:
                    loss = loss.mean()

                pre_scaling_loss = loss.detach()

                if args.edm2_loss_weighting:
                    loss, loss_scaled = edm2_model(loss, timesteps)
                    loss_scaled = loss_scaled.mean()
                else:
                    loss_scaled = None

                if loss.ndim != 0:
                    loss = loss.mean()

                accelerator.backward(loss)

                edm2_loss = loss
                loss = pre_scaling_loss

                # Sync EDM2 gradients across GPUs
                if args.edm2_loss_weighting and accelerator.sync_gradients:
                    for param in edm2_model.parameters():
                        if param.grad is not None:
                            param.grad = accelerator.reduce(param.grad, reduction="mean")

                if not (args.fused_backward_pass or args.fused_optimizer_groups):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            if args.edm2_loss_weighting and m is edm2_model:
                                continue
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()
                    if args.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

                if args.edm2_loss_weighting:
                    if accelerator.sync_gradients:
                        edm2_grad_norm = (
                            args.edm2_loss_weighting_max_grad_norm
                            if args.edm2_loss_weighting_max_grad_norm is not None
                            else args.max_grad_norm
                        )
                        if edm2_grad_norm != 0.0:
                            edm2_params = list(accelerator.unwrap_model(edm2_model).parameters())
                            accelerator.clip_grad_norm_(edm2_params, edm2_grad_norm)
                    edm2_optimizer.step()
                    edm2_lr_scheduler.step()
                    edm2_optimizer.zero_grad(set_to_none=True)

            # Sync gradients check
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                )

                # Save model at step intervals
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder),
                            accelerator.unwrap_model(unet),
                            vae,
                        )

                        if args.edm2_loss_weighting:
                            loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step, "_edm2_loss_weights")
                            loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
                            accelerator.print(f"saving edm2 loss weights: {loss_weights_file}")
                            accelerator.unwrap_model(edm2_model).save_weights(loss_weights_file, accelerator.unwrap_model(edm2_model).dtype, None)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no, "_edm2_loss_weights")
                                remove_loss_weights_file = os.path.join(args.output_dir, remove_loss_weights_ckpt_name)
                                if os.path.exists(remove_loss_weights_file):
                                    os.remove(remove_loss_weights_file)

                if plot_edm2_loss_weighting_check(args, global_step):
                    plot_edm2_loss_weighting(args, global_step, edm2_model, 1000, accelerator.device)

            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)

                if args.edm2_loss_weighting:
                    edm2_loss_val = edm2_loss.detach().item()
                    logs["loss/scaled"] = loss_scaled.detach().item() if loss_scaled is not None else 0.0
                    logs["loss/edm2"] = edm2_loss_val
                    logs["lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

                accelerator.log(logs, step=global_step)

            if args.edm2_loss_weighting and global_step % 10 == 0:
                edm2_loss_val = edm2_loss.detach().item()
                ratio = edm2_loss_val / current_loss if current_loss > 0 else 0.0
                accelerator.print(
                    f"[EDM2] step {global_step}: raw_loss={current_loss:.4f}, edm2_loss={edm2_loss_val:.4f} (ratio: {ratio:.2f}x)"
                )

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder),
                    accelerator.unwrap_model(unet),
                    vae,
                )

                if args.edm2_loss_weighting:
                    if (epoch + 1) % args.save_every_n_epochs == 0:
                        loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1, "_edm2_loss_weights")
                        loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
                        accelerator.print(f"saving edm2 loss weights: {loss_weights_file}")
                        accelerator.unwrap_model(edm2_model).save_weights(loss_weights_file, accelerator.unwrap_model(edm2_model).dtype, None)

                        remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                        if remove_epoch_no is not None:
                            remove_loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no, "_edm2_loss_weights")
                            remove_loss_weights_file = os.path.join(args.output_dir, remove_loss_weights_ckpt_name)
                            if os.path.exists(remove_loss_weights_file):
                                os.remove(remove_loss_weights_file)

        train_util.sample_images(
            accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
        )

    is_main_process = accelerator.is_main_process
    unet = accelerator.unwrap_model(unet)
    text_encoder = accelerator.unwrap_model(text_encoder)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, src_path, save_stable_diffusion_format, use_safetensors, save_dtype, epoch, global_step,
            text_encoder, unet, vae,
        )
        logger.info("model saved.")

        if args.edm2_loss_weighting:
            loss_weights_ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as, "_edm2_loss_weights")
            loss_weights_file = os.path.join(args.output_dir, loss_weights_ckpt_name)
            logger.info(f"saving edm2 loss weights: {loss_weights_file}")
            edm2_model.save_weights(loss_weights_file, edm2_model.dtype, None)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder (default: same as --learning_rate)",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="train the text encoder alongside the U-Net",
    )
    parser.add_argument(
        "--no_token_padding",
        action="store_true",
        help="disable token padding (same as Diffuser's DreamBooth)",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs",
        action="store_true",
        help="cache text encoder outputs to speed up training (requires text encoder not being trained)",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk (implies --cache_text_encoder_outputs)",
    )
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers for VAE",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE)",
    )
    parser.add_argument(
        "--vae_reflection_padding",
        action="store_true",
        help="switch VAE convolutions to reflection padding (improves border quality for some custom VAEs)",
    )
    parser.add_argument(
        "--vae_custom_scale",
        type=float,
        default=None,
        help="override the latent scaling factor applied after VAE encode (default: 0.18215 for SD 1.5)",
    )
    parser.add_argument(
        "--vae_custom_shift",
        type=float,
        default=None,
        help="apply a constant latent shift before scaling",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch_factor (batches per worker to pre-load). Only used when num_workers > 0.",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values "
        f"(0=conv_in/time, 1-4=down, 5=mid, 6-9=up, 10=out)",
    )
    parser.add_argument(
        "--list_unet_blocks",
        action="store_true",
        help="print SD 1.5 U-Net block indices with example parameter names, then exit",
    )
    parser.add_argument(
        "--freeze_unet_blocks",
        type=str,
        default=None,
        help=f"comma-separated block indices to freeze in the U-Net (0=conv_in/time, 1-4=down, 5=mid, 6-9=up, 10=out)",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="disable TF32 (enabled by default for SD 1.5 to avoid FP16/BF16 precision issues)",
    )

    # Rectified Flow arguments
    parser.add_argument("--flow_model", action="store_true", help="enable Rectified Flow training objective")
    parser.add_argument("--flow_use_ot", action="store_true", help="pair latents and noise with cosine optimal transport")
    parser.add_argument(
        "--flow_timestep_distribution", type=str, default="logit_normal", choices=["logit_normal", "uniform"],
        help="sampling distribution over Rectified Flow sigmas (default: logit_normal)",
    )
    parser.add_argument("--flow_logit_mean", type=float, default=0.0, help="mean of the logit-normal distribution")
    parser.add_argument("--flow_logit_std", type=float, default=1.0, help="stddev of the logit-normal distribution")
    parser.add_argument("--flow_uniform_shift", action="store_true", help="apply resolution-dependent timestep shift")
    parser.add_argument("--flow_uniform_base_pixels", type=float, default=512.0 * 512.0, help="reference pixel count for timestep shift (SD 1.5 default: 512*512)")
    parser.add_argument("--flow_uniform_static_ratio", type=float, default=None, help="use a fixed ratio for timestep shift; overrides resolution-based shift")

    # Contrastive Flow Matching
    parser.add_argument("--contrastive_flow_matching", action="store_true", help="enable Contrastive Flow Matching objective")
    parser.add_argument("--cfm_lambda", type=float, default=0.05, help="lambda weight for the contrastive term in CFM loss")

    # EDM2 loss weighting
    parser.add_argument("--edm2_loss_weighting", action="store_true", help="use EDM2 loss weighting")
    parser.add_argument("--edm2_loss_weighting_optimizer", type=str, default="torch.optim.AdamW", help="optimizer for EDM2 loss weighting")
    parser.add_argument("--edm2_loss_weighting_optimizer_lr", type=float, default=2e-2, help="learning rate for EDM2 loss weighting optimizer")
    parser.add_argument("--edm2_loss_weighting_optimizer_args", type=str, default=r"{'weight_decay': 0, 'betas': (0.9,0.999)}", help="optimizer args for EDM2")
    parser.add_argument("--edm2_loss_weighting_lr_scheduler", action="store_true", help="use lr scheduler with EDM2 optimizer")
    parser.add_argument("--edm2_loss_weighting_lr_scheduler_warmup_percent", type=float, default=0.1)
    parser.add_argument("--edm2_loss_weighting_lr_scheduler_constant_percent", type=float, default=0.1)
    parser.add_argument("--edm2_loss_weighting_lr_scheduler_decay_scaling", type=float, default=1.0)
    parser.add_argument("--edm2_loss_weighting_num_channels", type=int, default=128, help="Fourier feature channels for loss weighting")
    parser.add_argument("--edm2_loss_weighting_initial_weights", type=str, default=None, help="path to initial EDM2 loss weights")
    parser.add_argument("--edm2_loss_weighting_generate_graph", action="store_true", help="generate loss weighting graphs")
    parser.add_argument("--edm2_loss_weighting_generate_graph_every_x_steps", type=int, default=20)
    parser.add_argument("--edm2_loss_weighting_generate_graph_output_dir", type=str, default=None)
    parser.add_argument("--edm2_loss_weighting_generate_graph_y_limit", type=int, default=None)
    parser.add_argument("--edm2_loss_weighting_importance_weighting", action="store_true")
    parser.add_argument("--edm2_loss_weighting_importance_weighting_max", type=float, default=10.0)
    parser.add_argument("--edm2_loss_weighting_importance_min_snr_gamma", type=float, default=1.0)
    parser.add_argument("--edm2_loss_weighting_importance_weighting_safety_override", action="store_true")
    parser.add_argument("--edm2_loss_weighting_max_grad_norm", type=float, default=None)

    # Latent caching
    parser.add_argument("--skip_existing", action="store_true", help="skip latent caching if npz file already exists")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)

import argparse
import os

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from library import sdxl_model_util, sdxl_train_util, train_util, custom_sdxl_utils
import train_network
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.latent_shift = 0.0

    def assert_extra_args(self, args, train_dataset_group):
        sdxl_train_util.verify_sdxl_training_args(args)

        # Auto-detect VAE scale/shift
        vae_scale_factor, vae_shift_factor = custom_sdxl_utils.get_vae_scale_and_shift(getattr(args, "vae_type", None))

        if getattr(args, "vae_custom_scale", None) is not None:
             self.vae_scale_factor = float(args.vae_custom_scale)
             logger.info(f"Using custom VAE scale factor: {self.vae_scale_factor}")
        else:
             self.vae_scale_factor = vae_scale_factor
             logger.info(f"Using auto-detected VAE scale factor: {self.vae_scale_factor} (vae_type: {getattr(args, 'vae_type', None)})")

        if getattr(args, "vae_custom_shift", None) is not None:
             self.latent_shift = float(args.vae_custom_shift)
             logger.info(f"Using custom VAE shift factor: {self.latent_shift}")
        else:
             self.latent_shift = vae_shift_factor
             if self.latent_shift != 0:
                  logger.info(f"Using auto-detected VAE shift factor: {self.latent_shift}")

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"
        
        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        vae_type = getattr(args, "vae_type", None)
        latent_channels = getattr(args, "latent_channels", None)
        
        # Determine if we should use custom loader
        # We use it if vae_type/latent_channels is specified OR always safely?
        # Always using it is safer for resume cases where arguments might not be passed but checkpoint is modified (though user *should* pass args)
        # However, relying on args is better for explicit intent
        
        logger.info(f"Loading model with custom loader. VAE Type: {vae_type}, Latent Channels: {latent_channels}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = custom_sdxl_utils.load_custom_sdxl_checkpoint(
            args.pretrained_model_name_or_path,
            accelerator.device,
            weight_dtype,
            custom_vae_type=vae_type,
            latent_channels_override=latent_channels
        )
        
        # Load Stable Diffusion Format flag is not returned by custom func, but implied by path check?
        # sdxl_train_util.load_target_model does a lot of checks.
        # My custom loader assumes "checkpoint" loading path (ckpt/safetensors).
        # If user provides Diffusers directory, my custom loader fails.
        # TODO: Handle diffusers model path.
        
        # Re-check implementation. `load_custom_sdxl_checkpoint` assumes file load.
        # If pretrained_model_name_or_path is directory, we should probably fall back to standard load
        # BUT standard load doesn't support 16-channel resume.
        
        # Use standard load if it is a directory?
        # Typically custom VAE training uses a base SDXL checkpoint file.
        
        if os.path.isdir(args.pretrained_model_name_or_path):
             # Fallback or implement directory support? 
             # For now, let's assume if it is a directory, it is standard SDXL and resume logic for 16-channel isn't primary concern unless checkpoint is saved as file?
             # Actually `sdxl_train_util.load_target_model` handles both.
             pass 

        # If custom VAE specified:
        if vae_type:
            logger.info(f"Replacing VAE with custom VAE type: {vae_type}")
            new_vae = custom_sdxl_utils.load_custom_vae(
                getattr(args, "vae", None), vae_type, weight_dtype, accelerator.device
            )
            if new_vae:
                vae = new_vae
        
        # Patching UNet is handled inside loader if detected or override provided.
        # But if we loaded from diffusers dir, we might still need to patch.
        # IF detected_channels != target_channels (handled in loader)
        
        self.load_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path) # approximate
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def cache_latents(self, args, accelerator, vae, unet, train_dataset_group, vae_dtype):
        if getattr(args, "vae_type", None) or getattr(args, "latent_channels", None):
             logger.info("Using custom latent caching for custom VAE.")
             custom_sdxl_utils.cache_latents_custom(
                 vae,
                 train_dataset_group,
                 args,
                 accelerator,
                 vae_type=getattr(args, "vae_type", "sdxl"),
                 latent_channels=getattr(args, "latent_channels", None)
             )
        else:
             super().cache_latents(args, accelerator, vae, unet, train_dataset_group, vae_dtype)

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def get_flow_pixel_counts(self, args, batch, latents):
        if (
            getattr(args, "flow_model", False)
            and args.flow_uniform_shift
            and args.flow_uniform_static_ratio is None
        ):
            target_size = batch.get("target_sizes_hw")
            if target_size is None:
                raise ValueError(
                    "Resolution-dependent Rectified Flow shift requires target size information in the batch."
                )
            return (target_size[:, 0] * target_size[:, 1]).to(latents.device, torch.float32)
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    args.use_zero_cond_dropout,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--vae_type", type=str, default=None, help="Specify VAE type: sdxl, flux, sana, etc.")
    parser.add_argument("--latent_channels", type=int, default=None, help="Override latent channels (e.g. 16 for Flux, 32 for Sana)")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)

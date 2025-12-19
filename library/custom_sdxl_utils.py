import os
import torch
import numpy as np
from torch import nn
from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLFlux2
from library import model_util
from tqdm import tqdm
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF
from library import model_util, sdxl_model_util, sdxl_original_unet
from library.sdxl_model_util import _load_state_dict_on_device, convert_sdxl_text_encoder_2_checkpoint
from accelerate import init_empty_weights
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTextModelWithProjection
from library.train_util import (
    IMAGE_TRANSFORMS,
    load_image,
    trim_and_resize_if_required,
    save_latents_to_disk,
    is_disk_cached_latents_is_expected,
)

def load_custom_sdxl_checkpoint(ckpt_path, map_location, dtype=None, custom_vae_type=None, latent_channels_override=None):
    """
    Custom loader that inspects checkpoint to determine if UNet needs patching BEFORE loading weights.
    Crucial for resuming training with modified UNet (16/32 channels).
    """
    print(f"[Custom Loader] Loading SDXL from {ckpt_path}")
    
    
    # Defaults
    target_channels = 4
    if latent_channels_override is not None:
        target_channels = latent_channels_override
    elif custom_vae_type is not None:
        if custom_vae_type.lower() == 'flux':
            target_channels = 16
        elif custom_vae_type.lower() in ['flux2', 'sana']:
            target_channels = 32
            
    # 1. Load State Dict
    if model_util.is_safetensors(ckpt_path):
        from safetensors.torch import load_file
        try:
            state_dict = load_file(ckpt_path, device=map_location)
        except:
            state_dict = load_file(ckpt_path)
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0

    # 2. Inspect UNet channels from state dict
    # Look for "model.diffusion_model.input_blocks.0.0.weight" or equivalent
    unet_conv_in_key = "model.diffusion_model.input_blocks.0.0.weight"
    detected_channels = 4
    if unet_conv_in_key in state_dict:
        detected_channels = state_dict[unet_conv_in_key].shape[1]
        print(f"[Custom Loader] Detected UNet input channels in checkpoint: {detected_channels}")
    
    # 3. Build UNet
    print("building U-Net")
    with init_empty_weights():
        unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    
    # 4. Patch UNet if needed
    # 5. Load UNet Weights Logic
    # We have two scenarios:
    # A. Resume/Same-Structure: detected_channels == target_channels
    #    - We want to build the expected structure (patched if needed) and then load weights directly.
    # B. New Training/Conversion: detected_channels (e.g. 4) != target_channels (e.g. 16)
    #    - We need to load the weights into the standard structure (4 channels) FIRST.
    #    - THEN we patch the UNet to 16 channels (which copies/expands the loaded weights).

    print("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)

    if detected_channels == target_channels:
        # Case A: Resume or Standard
        if target_channels != 4:
             print(f"[Custom Loader] Resume Mode: Patching UNet to {target_channels} channels BEFORE loading weights.")
             unet = patch_unet_for_latent_channels(unet, target_channels)
        
        info = _load_state_dict_on_device(unet, unet_sd, device=map_location, dtype=dtype)
        print(f"U-Net loaded: {info}")
    
    else:
        # Case B: Conversion (Standard -> Custom)
        print(f"[Custom Loader] New/Conversion Mode: Loading {detected_channels}-channel weights FIRST, then patching to {target_channels}.")
        
        # 1. Load weights into standard (unpatched) UNet
        info = _load_state_dict_on_device(unet, unet_sd, device=map_location, dtype=dtype)
        print(f"U-Net standard weights loaded: {info}")
        
        # 2. Patch (expand)
        unet = patch_unet_for_latent_channels(unet, target_channels)
    
    # 6. Load Text Encoders (Standard SDXL logic)
    print("building text encoders")
    # TE1
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
        max_position_embeddings=77, hidden_act="quick_gelu", layer_norm_eps=1e-05, dropout=0.0, attention_dropout=0.0,
        initializer_range=0.02, initializer_factor=1.0, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        model_type="clip_text_model", projection_dim=768
    )
    with init_empty_weights():
        text_model1 = CLIPTextModel._from_config(text_model1_cfg)
    
    # TE2
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32, num_attention_heads=20,
        max_position_embeddings=77, hidden_act="gelu", layer_norm_eps=1e-05, dropout=0.0, attention_dropout=0.0,
        initializer_range=0.02, initializer_factor=1.0, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        model_type="clip_text_model", projection_dim=1280
    )
    with init_empty_weights():
        text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    print("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    if "text_model.embeddings.position_ids" in te1_sd:
        te1_sd.pop("text_model.embeddings.position_ids")

    _load_state_dict_on_device(text_model1, te1_sd, device=map_location)
    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    _load_state_dict_on_device(text_model2, converted_sd, device=map_location)

    # 7. Load VAE
    # We load standard VAE from checkpoint first, then user can replace it later if args.vae_type is set
    # OR we can skip loading if we know we are replacing it? 
    # Better to load it to be safe, unless detected channels != 4, implies custom VAE was used.
    # But VAE weights in checkpoint might be standard SDXL VAE even if UNet is patched?
    # No, typically user saves "model" (unet+te+vae).
    
    print("building VAE")
    vae_config = model_util.create_vae_diffusers_config()
    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)
    
    print("loading VAE from checkpoint")
    try:
        converted_vae_checkpoint = model_util.convert_ldm_vae_checkpoint(state_dict, vae_config)
        _load_state_dict_on_device(vae, converted_vae_checkpoint, device=map_location, dtype=dtype)
    except Exception as e:
        print(f"VAE load failed (expected if checkpoint has no VAE or custom format): {e}")

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, unet, logit_scale, ckpt_info


    w, h = image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w == w and new_h == h:
        return image
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))

def load_latents_npz_custom(path: str):
    with open(path, "rb") as f:
        with np.load(f, allow_pickle=False) as data:
            return data["latents"].copy()

def use_reflection_padding(vae):
    """Set all Conv2d layers with padding to use reflection padding."""
    for module in vae.modules():
        if isinstance(module, nn.Conv2d):
            pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            if pad_h > 0 or pad_w > 0:
                module.padding_mode = "reflect"
    print('Using reflection padding in vae')
    return vae

def patch_unet_for_latent_channels(unet, latent_channels=16):
    """Ensure the SDXL UNet accepts the desired latent channel count."""
    # Handle Input Convolution
    if hasattr(unet, "conv_in"):
        conv_in = unet.conv_in
    elif hasattr(unet, "input_blocks"):
        # Custom SDXL style: input_blocks[0][0] is the Conv2d
        conv_in = unet.input_blocks[0][0]
    else:
        raise AttributeError("Could not find input convolution (conv_in) in UNet.")

    # Handle Output Convolution
    if hasattr(unet, "conv_out"):
        conv_out = unet.conv_out
    elif hasattr(unet, "out"):
         # Custom SDXL style: out is ModuleList [GroupNorm, SiLU, Conv2d]
         # usually index 2
         conv_out = unet.out[2]
    else:
        raise AttributeError("Could not find output convolution (conv_out) in UNet.")

    if conv_in.in_channels == latent_channels and conv_out.out_channels == latent_channels:
        # unet.register_to_config(in_channels=latent_channels, out_channels=latent_channels)
        # diffusers UNet config might be read-only or handled differently, but modifying conv layers is the key.
        print(f"[Model Loader] UNet already configured for {latent_channels} latent channels.")
        return unet

    new_conv_in = nn.Conv2d(
        latent_channels,
        conv_in.out_channels,
        conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding,
    )
    new_conv_out = nn.Conv2d(
        conv_out.in_channels,
        latent_channels,
        conv_out.kernel_size,
        stride=conv_out.stride,
        padding=conv_out.padding,
    )

    with torch.no_grad():
        # Only copy weights if source is not on meta device
        if conv_in.weight.device.type != 'meta':
             new_conv_in.weight.zero_()
             # Copy existing weights for the channels we have
             # If new channels > old channels, the extra input channels will be 0-init (ignored initially)
             # If new channels < old channels, we slice.
             min_in = min(conv_in.in_channels, latent_channels)
             new_conv_in.weight[:, :min_in, ...].copy_(conv_in.weight[:, :min_in, ...])
             
             if conv_in.bias is not None:
                 new_conv_in.bias.zero_()
                 new_conv_in.bias.copy_(conv_in.bias)
        else:
             print("[Model Loader] UNet on meta device: Skipping weight copy during patching (will be loaded from checkpoint).")

        if conv_out.weight.device.type != 'meta':
             new_conv_out.weight.zero_()
             # For output, if new > old, we fill first 'old' channels.
             # If new < old, we take first 'new' channels.
             min_out = min(conv_out.out_channels, latent_channels)
             new_conv_out.weight[:min_out, ...].copy_(conv_out.weight[:min_out, ...])
             
             if conv_out.bias is not None:
                 new_conv_out.bias.zero_()
                 new_conv_out.bias[:min_out].copy_(conv_out.bias[:min_out])

    if hasattr(unet, "conv_in"):
        unet.conv_in = new_conv_in
    elif hasattr(unet, "input_blocks"):
        unet.input_blocks[0][0] = new_conv_in

    if hasattr(unet, "conv_out"):
        unet.conv_out = new_conv_out
    elif hasattr(unet, "out"):
        unet.out[2] = new_conv_out
    
    # Update config if possible to avoid warnings/checks down the line
    if hasattr(unet, "config"):
        unet.config.in_channels = latent_channels
        unet.config.out_channels = latent_channels

    print(f"[Model Loader] Patched UNet for {latent_channels}-channel latents.")
    return unet

def load_custom_vae(vae_path, vae_type, dtype, device):
    if vae_path == 'sdxl_vae':
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    elif vae_path == 'flux_vae' or (vae_type == 'flux' and vae_path is None):
        print("[Model Loader] Loading Flux VAE from 'black-forest-labs/FLUX.1-dev'.")
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
        )
    elif vae_path == 'sana_vae' or (vae_type == 'sana' and vae_path is None):
        print("[Model Loader] Loading Sana VAE from 'mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers'.")
        vae = AutoencoderDC.from_pretrained(
            "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
        )
    elif vae_path == 'flux2_vae' or (vae_type == 'flux2' and vae_path is None):
        print("[Model Loader] Loading Flux2 VAE from 'black-forest-labs/FLUX.2-dev'.")
        vae = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            subfolder="vae",
            torch_dtype=torch.float32,
        )
    else:
        # Load from path
        # Check if we should use custom loader for file path if type is known
        if vae_type in ['flux', 'flux2']:
            print(f"[Model Loader] Loading custom {vae_type} VAE from: {vae_path}")
            
            # Check if it's a directory (local repo)
            if os.path.isdir(vae_path):
                # Try to load as pretrained directory
                # If it's the root repo (has 'vae' folder), use subfolder='vae'
                # If it's the vae folder itself, use subfolder=None
                subfolder = None
                if os.path.isdir(os.path.join(vae_path, "vae")):
                    subfolder = "vae"
                
                print(f"  Detected directory: {vae_path}")
                print(f"  Loading via from_pretrained (subfolder={subfolder})...")
                try:
                    if vae_type == 'flux2':
                        vae = AutoencoderKLFlux2.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch.float32)
                    else:
                        vae = AutoencoderKL.from_pretrained(vae_path, subfolder=subfolder)
                        
                    vae.to(device, dtype=dtype)
                    return vae
                except Exception as e:
                    print(f"  Error loading from directory: {e}")
                    raise e

            # If it's a file, we proceed with weight loading logic
            # Initialize appropriate class
            if vae_type == 'flux2':
                 # Fallback strategy for single file: 
                 # 1. Try to find config.json in same dir as file (or parent dir)
                 # 2. Try to load PRETRAINED structure from HuggingFace
                 
                 vae = None
                 
                 # Check parent dir for config.json (common structure: /path/to/vae_dir/diffusion_pytorch_model.safetensors)
                 parent_dir = os.path.dirname(vae_path)
                 config_path = os.path.join(parent_dir, "config.json")
                 
                 if os.path.exists(config_path):
                     print(f"  Found config.json at {config_path}. Using parent dir as pretrained source.")
                     try:
                         # Attempt to load structure from parent dir
                         vae = AutoencoderKLFlux2.from_pretrained(parent_dir, torch_dtype=torch.float32)
                         print("  Successfully initialized structure from local config.")
                     except Exception as e:
                         print(f"  Parent dir load failed: {e}. Fallback to HF config.")
                 
                 if vae is None:
                     try:
                         print("  Initializing Flux2 structure from 'black-forest-labs/FLUX.2-dev' to ensure correct config...")
                         vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.float32)
                     except Exception as e:
                         print(f"  Warning: Could not fetch config from HF ({e}). Using default init (might fail if config missing).")
                         # If we can't get it from HF, we are stuck unless we have config.
                         return None 
            else: # flux
                 print("  Initializing Flux1 structure from 'black-forest-labs/FLUX.1-dev'...")
                 vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
            
            # Load weights from file
            from safetensors.torch import load_file
            if vae_path.endswith(".safetensors"):
                sd = load_file(vae_path)
            else:
                sd = torch.load(vae_path, map_location='cpu')
                if "state_dict" in sd:
                    sd = sd["state_dict"]
            
            # Helper to strip keys if needed
            # For Flux2/1, keys usually match standard Diffusers VAE format if exported correctly.
            
            # Load
            m, u = vae.load_state_dict(sd, strict=False)
            print(f"[Model Loader] Loaded state dict. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
            if len(m) > 0:
                print(f"  First 5 missing: {m[:5]}")
            
        else:
             # Standard LDM-compatible VAE or unknown
             vae = model_util.load_vae(vae_path, dtype)
    
    vae.to(device, dtype=dtype)
    return vae

def normalize_latents(latents, mu, sigma):
    mu = mu.view(1, -1, 1, 1).to(latents.device)
    sigma = sigma.view(1, -1, 1, 1).to(latents.device)
    return (latents - mu) / sigma

def cache_latents_custom(
    vae,
    dataset_group,
    args,
    accelerator,
    vae_type='sdxl',
    latent_channels=None
):
    """
    Custom latent caching loop supporting Flux/Sana/Custom VAEs.
    Matches the standard cache format (latents + metadata) so loaders work.
    """
    image_infos = list(dataset_group.image_data.values())
    if not image_infos:
        print("No images to cache.")
        return

    # Build a lookup from image key to subset (per-dataset)
    image_to_subset = {}
    if hasattr(dataset_group, "image_to_subset"):
        image_to_subset = dataset_group.image_to_subset
    else:
        for ds in getattr(dataset_group, "datasets", []):
            if hasattr(ds, "image_to_subset"):
                image_to_subset.update(ds.image_to_subset)

    # Prepare cache paths and skip already-valid entries
    images_to_process = []
    skip_existing = getattr(args, "skip_existing", False)
    for info in image_infos:
        info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
        subset = image_to_subset.get(info.image_key)
        if subset is None:
            raise ValueError(f"subset not found for image key: {info.image_key}")
        if skip_existing and os.path.exists(info.latents_npz):
            continue
        if subset and os.path.exists(info.latents_npz) and is_disk_cached_latents_is_expected(
            info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
        ):
            continue
        images_to_process.append(info)

    if not images_to_process:
        print(f"[Custom Latents Cache] All {len(image_infos)} latents exist. Skipping encoding.")
        return

    print(
        f"[Custom Latents Cache] Caching latents for {len(images_to_process)} images "
        f"(Skipped {len(image_infos) - len(images_to_process)}). VAE Type: {vae_type}"
    )

    # Group by resolution/augment settings like the standard cache path
    class Condition:
        def __init__(self, reso, flip_aug, alpha_mask, random_crop):
            self.reso = reso
            self.flip_aug = flip_aug
            self.alpha_mask = alpha_mask
            self.random_crop = random_crop

        def __eq__(self, other):
            return (
                self.reso == other.reso
                and self.flip_aug == other.flip_aug
                and self.alpha_mask == other.alpha_mask
                and self.random_crop == other.random_crop
            )

    images_to_process.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])
    batches = []
    batch = []
    current_condition = None
    batch_size = getattr(args, "vae_batch_size", 1)

    for info in images_to_process:
        subset = image_to_subset.get(info.image_key)
        condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
        if len(batch) > 0 and current_condition != condition:
            batches.append((current_condition, batch))
            batch = []
        batch.append(info)
        current_condition = condition
        if len(batch) >= batch_size:
            batches.append((current_condition, batch))
            batch = []
            current_condition = None
    if len(batch) > 0:
        batches.append((current_condition, batch))

    vae.eval()
    vae.to(accelerator.device, dtype=torch.float32 if args.no_half_vae else vae.dtype)

    for condition, batch_infos in tqdm(batches, desc="Caching Custom Latents"):
        images = []
        alpha_masks = []
        for info in batch_infos:
            subset = image_to_subset.get(info.image_key)
            image = load_image(info.absolute_path, subset.alpha_mask)
            image, original_size, crop_ltrb = trim_and_resize_if_required(
                subset.random_crop, image, info.bucket_reso, info.resized_size
            )

            info.latents_original_size = original_size
            info.latents_crop_ltrb = crop_ltrb

            if subset.alpha_mask:
                if image.shape[2] == 4:
                    alpha_mask = image[:, :, 3].astype(np.float32) / 255.0
                else:
                    alpha_mask = np.ones_like(image[:, :, 0], dtype=np.float32)
                alpha_masks.append(torch.from_numpy(alpha_mask))
            else:
                alpha_masks.append(None)

            image = image[:, :, :3]  # strip alpha for encoding
            images.append(IMAGE_TRANSFORMS(image))

        img_tensors = torch.stack(images, dim=0).to(device=vae.device, dtype=vae.dtype)

        with torch.no_grad():
            encoded = vae.encode(img_tensors)
            latents = encoded.latent_dist.sample().to("cpu")

            if condition.flip_aug:
                flipped_imgs = torch.flip(img_tensors, dims=[3])
                flipped_latents = vae.encode(flipped_imgs).latent_dist.sample().to("cpu")
            else:
                flipped_latents = [None] * len(latents)

        for info, latent, flipped_latent, alpha_mask in zip(batch_infos, latents, flipped_latents, alpha_masks):
            save_latents_to_disk(
                info.latents_npz,
                latent,
                info.latents_original_size,
                info.latents_crop_ltrb,
                flipped_latent,
                alpha_mask,
            )

    vae.to("cpu")
    torch.cuda.empty_cache()


def get_vae_scale_and_shift(vae_type):
    """
    Returns (scale_factor, shift_factor) for the given VAE type.
    Defaults to SDXL values if unknown.
        SDXL: 0.13025, 0.0
        Flux: 0.3611, 0.0
        Sana: 0.41407, 0.0
    """
    if not vae_type:
        return sdxl_model_util.VAE_SCALE_FACTOR, 0.0
        
    vt = vae_type.lower()
    if 'flux2' in vt:
        return 1.0, 0.0
    elif 'flux' in vt:
        return 0.3611, 0.0
    elif 'sana' in vt:
         return 0.41407, 0.0
    
    return sdxl_model_util.VAE_SCALE_FACTOR, 0.0

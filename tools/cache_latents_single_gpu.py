
import argparse
import os
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from library import train_util, config_util, custom_sdxl_utils, sdxl_train_util, model_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.train_util import IMAGE_TRANSFORMS, load_image, trim_and_resize_if_required

def setup_parser():
    parser = argparse.ArgumentParser(description="Single-GPU High-Performance Latent Cacher")
    
    # Dataset and Config args
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=16, help="Number of CPU workers for data loading")
    
    # Model args
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to base SDXL model")
    parser.add_argument("--vae", type=str, default=None, help="Path to custom VAE")
    parser.add_argument("--vae_type", type=str, default=None, help="VAE type: sdxl, flux, sana")
    parser.add_argument("--no_half_vae", action="store_true", help="Use float32 VAE")
    
    # Caching args
    # vae_batch_size is added by train_util
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing .npz files")
    parser.add_argument("--max_save_workers", type=int, default=8, help="Number of threads for async disk I/O")
    
    # Scaling/Shifting args (Default: Raw)
    parser.add_argument("--vae_custom_scale", type=float, default=None, help="Custom scale factor")
    parser.add_argument("--vae_custom_shift", type=float, default=None, help="Custom shift factor")
    
    # Legacy args for compatibility with config files
    # resolution and bucket args are added by train_util.add_dataset_arguments
    
    parser.add_argument("--mixed_precision", type=str, default="bf16") # Used for weight_dtype
    parser.add_argument("--full_bf16", action="store_true")
    parser.add_argument(
        "--save_precision",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Precision used when writing latents to disk. `auto` stores fp16 for half/bfloat16 inference, otherwise fp32. `bfloat16` stores compact uint16 payload plus dtype tag."
    )

    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, help="Directory for tokenizer cache")
    parser.add_argument("--max_token_length", type=int, default=None, help="Max token length")
    # Add other necessary args required by library util calls
    train_util.add_dataset_arguments(parser, True, True, True) 
    
    return parser

def save_npz_async(file_path, latents, original_size, crop_ltrb, flipped_latents=None, alpha_mask=None, latents_dtype=None):
    """Save .npz in a separate thread."""
    save_dict = {
        "latents": latents,
        "original_size": np.array(original_size),
        "crop_ltrb": np.array(crop_ltrb),
    }
    if flipped_latents is not None:
        save_dict["flipped_latents"] = flipped_latents
    if alpha_mask is not None:
        save_dict["alpha_mask"] = alpha_mask
    if latents_dtype is not None:
        save_dict["latents_dtype"] = latents_dtype
        
    np.savez(file_path, **save_dict)

def main():
    parser = setup_parser()
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (0, 1, etc.)")
    args = parser.parse_args()
    
    # Basic Setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    if args.save_precision == "float16":
        save_mode = "float16"
    elif args.save_precision == "float32":
        save_mode = "float32"
    elif args.save_precision == "bfloat16":
        save_mode = "bfloat16"
    else:
        # auto: prefer fp16 for half/bf16 inference, fp32 otherwise
        if vae_dtype == torch.bfloat16:
            save_mode = "float16"  # bfloat16 is stored as fp16 unless explicitly requested
        elif vae_dtype == torch.float16:
            save_mode = "float16"
        else:
            save_mode = "float32"

    save_np_dtype = {"float16": np.float16, "float32": np.float32}.get(save_mode, None)

    print(f"Device: {device}, VAE Dtype: {vae_dtype}")
    print(f"Latents save mode: {save_mode}")

    # Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #torch.backends.cudnn.benchmark = True
    print("Enabled TF32 and CuDNN Benchmark.")

    # Load VAE
    print("Loading VAE...")
    vae = None
    
    # Logic adapted from cache_latents_standalone.py
    if args.vae_type:
        print(f"Loading custom VAE: {args.vae_type}")
        # Assuming model_path is passed as vae path or pretrained model path
        vae_path = args.vae if args.vae else args.pretrained_model_name_or_path
        vae = custom_sdxl_utils.load_custom_vae(vae_path, args.vae_type, vae_dtype, device)
    else:
        # Standard SDXL Load — only the VAE, never the full pipeline
        target_path = args.vae if args.vae else args.pretrained_model_name_or_path
        print(f"Loading standard VAE from: {target_path}")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(target_path, subfolder="vae" if not args.vae else None, torch_dtype=vae_dtype)
            
    vae.to(device, dtype=vae_dtype)
    # Channels Last Optimization for NVIDIA GPUs (especially 4090 with Tensor Cores)
    vae.to(memory_format=torch.channels_last)
    vae.requires_grad_(False)
    vae.eval()
    print(f"Loaded VAE Dtype: {vae.dtype}")

    # Determine Scale/Shift
    scale_factor = args.vae_custom_scale
    shift_factor = args.vae_custom_shift
    if scale_factor is None and shift_factor is None:
        print("Using RAW latents (Scale=1.0, Shift=0.0)")
        scale_factor = 1.0
        shift_factor = 0.0
    else:
        print(f"Using Custom Scale: {scale_factor}, Shift: {shift_factor}")
        scale_factor = scale_factor if scale_factor is not None else 1.0
        shift_factor = shift_factor if shift_factor is not None else 0.0


    # Load Tokenizers (only needed for SDXL path — custom VAE types don't use them)
    tokenizers = None
    if not args.vae_type:
        print("Loading tokenizers...")
        try:
            tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
            tokenizers = [tokenizer1, tokenizer2]
        except Exception as e:
            print(f"Warning: Failed to load SDXL tokenizers: {e}")
    else:
        print("Skipping tokenizer loading (not needed for latent caching with custom VAE).")
        # Dataset init accesses tokenizer.model_max_length when max_token_length is None,
        # so provide a safe default to avoid the access entirely.
        if args.max_token_length is None:
            args.max_token_length = 75

    # Prepare Dataset
    print("Preparing Dataset...")
    # Use BlueprintGenerator to load config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
    user_config = config_util.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, argparse.Namespace(**vars(args)), tokenizer=tokenizers)
    dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    
    # Collect all image infos
    all_image_infos = []
    skipped_count = 0
    
    for info in dataset_group.image_data.values():
         # Set npz path
         info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
         if args.skip_existing and os.path.exists(info.latents_npz):
             skipped_count += 1
             continue
         all_image_infos.append(info)
         
    print(f"Total images: {len(dataset_group.image_data)}")
    print(f"Skipping: {skipped_count}")
    print(f"To Cache: {len(all_image_infos)}")
    
    if len(all_image_infos) == 0:
        print("Nothing to do.")
        return

    # Group images by bucket resolution, then shuffle the groups so large and small
    # resolutions are interleaved (avoids monotonically increasing GPU time).
    import random
    from itertools import groupby
    all_image_infos.sort(key=lambda x: (x.bucket_reso[0], x.bucket_reso[1]))
    bucket_groups = [list(g) for _, g in groupby(all_image_infos, key=lambda x: x.bucket_reso)]
    random.shuffle(bucket_groups)
    all_image_infos = [info for group in bucket_groups for info in group]

    # Build global image_to_subset map since DatasetGroup doesn't expose it directly
    global_image_to_subset = {}
    for dataset in dataset_group.datasets:
        if hasattr(dataset, "image_to_subset"):
            global_image_to_subset.update(dataset.image_to_subset)

    # ----------------------------------------------------------------------
    # Optimized Dataset Class for Parallel CPU Processing
    # ----------------------------------------------------------------------
    class ImageCacheDataset(torch.utils.data.Dataset):
        def __init__(self, image_infos, image_to_subset_map):
            self.image_infos = image_infos
            self.image_to_subset_map = image_to_subset_map
            
        def __len__(self):
            return len(self.image_infos)
            
        def __getitem__(self, idx):
            info = self.image_infos[idx]
            subset = self.image_to_subset_map.get(info.image_key)
            if subset is None:
                raise ValueError(f"Could not find subset for image: {info.image_key}")
            
            # Heavy CPU Work: Load and Resize
            img = load_image(info.absolute_path)
            
            image_np, original_size, crop_ltrb = trim_and_resize_if_required(
                subset.random_crop, img, info.bucket_reso, info.resized_size
            )
            
            # Flatten to tensor
            tensor = IMAGE_TRANSFORMS(image_np) # C, H, W
            
            # Return tuple
            return tensor, info, str(original_size), str(crop_ltrb) 
            # Note: Tensors must be stacked. info, and numpy arrays can't be easily collated by default 
            # unless we use custom collate. 
            # Actually, standard default_collate works if we return basic types.
            # But ImageInfo is an object.
            # We will use a custom collate that just stacks the tensors and returns lists for others.

    # ----------------------------------------------------------------------
    # Bucket Batch Sampler
    # ----------------------------------------------------------------------
    class BucketBatchSampler(torch.utils.data.Sampler):
        def __init__(self, image_infos, batch_size):
            self.image_infos = image_infos
            self.batch_size = batch_size
            
        def __iter__(self):
            batch = []
            current_reso = None
            
            for i, info in enumerate(self.image_infos):
                # Check resolution change
                if current_reso is None:
                    current_reso = info.bucket_reso
                
                if info.bucket_reso != current_reso:
                    # Flush batch
                    if batch:
                        yield batch
                    batch = []
                    current_reso = info.bucket_reso
                
                batch.append(i)
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            
            if batch:
                yield batch
                
        def __len__(self):
            # Approximate length (not critical for TQDM if we use total images count on TQDM side or just infinite?)
            # Actually hard to calculate exactly without traversing.
            return len(self.image_infos) // self.batch_size 

    # ----------------------------------------------------------------------
    # Custom Collate
    # ----------------------------------------------------------------------
    def fast_collate(batch):
        # batch is list of tuples (tensor, info, original_size_str, crop_ltrb_str)
        # We need to stack tensors.
        
        # Unzip
        tensors = [item[0] for item in batch]
        infos = [item[1] for item in batch]
        
        # original_size and crop_ltrb were converted to str to avoid array collation issues?
        # Actually, let's just return them as is, but we need to handle them carefully.
        # If we return them as lists of standard python objects, default_collate might try to tensorize them.
        # So we construct manually.
        
        # Re-parse if we passed strings, or just assume we handle it in main
        # In __getitem__ above, I passed str(original_size). Let's fix that design.
        # Better: return original_size (tuple/list) and let collate just listify it.
        
        # Correct logic:
        # __getitem__ returns (tensor, info, original_size, crop_ltrb)
        
        # Since we use a CUSTOM collate, default_collate is NOT called.
        # So we can return whatever we want.
        
        orig_sizes = [item[2] for item in batch] # These are tuples/arrays from trim_and_resize (tuple)
        crop_ltrbs = [item[3] for item in batch] # tuple
        
        batch_tensor = torch.stack(tensors)
        
        return batch_tensor, infos, orig_sizes, crop_ltrbs

    # Re-define dataset class __getitem__ to be cleaner
    class ImageCacheDataset(torch.utils.data.Dataset):
        def __init__(self, image_infos, image_to_subset_map):
            self.image_infos = image_infos
            self.image_to_subset_map = image_to_subset_map
        def __len__(self): return len(self.image_infos)
        def __getitem__(self, idx):
            info = self.image_infos[idx]
            subset = self.image_to_subset_map.get(info.image_key)
            img = load_image(info.absolute_path)
            image_np, original_size, crop_ltrb = trim_and_resize_if_required(
                subset.random_crop, img, info.bucket_reso, info.resized_size
            )
            return IMAGE_TRANSFORMS(image_np), info, original_size, crop_ltrb

    dataset = ImageCacheDataset(all_image_infos, global_image_to_subset)
    sampler = BucketBatchSampler(all_image_infos, args.vae_batch_size)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.max_data_loader_n_workers,
        collate_fn=fast_collate,
        pin_memory=True,
        persistent_workers=True if args.max_data_loader_n_workers > 0 else False
        # prefetch_factor defaults to 2/4 usually
    )

    # Output ThreadPool
    executor = ThreadPoolExecutor(max_workers=args.max_save_workers)
    
    # Process Loop
    print(f"Starting Caching Loop (Workers: {args.max_data_loader_n_workers}, Batch: {args.vae_batch_size})...")
    start_time = time.time()
    
    autocast_ctx = torch.autocast("cuda", dtype=weight_dtype)
    
    # TQDM: Since batch sampler assumes length, but we replaced logic,
    # let's just iterate dataloader. len(dataloader) uses sampler.__len__ which is approx.
    
    t_start_data = time.time()
    log_interval = 10
    batch_count = 0
    total_data_time = 0
    total_gpu_time = 0
    import signal
    stop_event = False
    def handle_interrupt(signum, frame):
        nonlocal stop_event
        print(f"\nReceived signal {signum}. Stopping after current batch...")
        stop_event = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    with torch.inference_mode():
        for images, infos, orig_sizes, crops in tqdm(dataloader, smoothing=0.01):
            if stop_event:
                break
                
            t_data = time.time() - t_start_data
            
            t_start_gpu = time.time()
            # Move to device and ensure channels_last match
            images = images.to(device, non_blocking=True).to(vae_dtype, memory_format=torch.channels_last)
            
            with autocast_ctx:
                if hasattr(vae, "encode"):
                    dist = vae.encode(images).latent_dist
                else:
                    dist = vae.encode(images).latent_dist
                
                latents = dist.sample()
                
                if shift_factor != 0.0:
                    latents = latents - shift_factor
                if scale_factor != 1.0:
                    latents = latents * scale_factor
            
            # Convert to chosen precision before saving.
            latents_dtype_tag = save_mode
            if save_mode == "bfloat16":
                latents_to_save = latents.to(dtype=torch.bfloat16).cpu().view(torch.uint16).numpy()
            else:
                target_torch_dtype = torch.float16 if save_mode == "float16" else torch.float32
                latents_to_save = latents.to(dtype=target_torch_dtype).cpu().numpy().astype(save_np_dtype, copy=False)
            t_gpu = time.time() - t_start_gpu
            
            # Profiling Stats
            total_data_time += t_data
            total_gpu_time += t_gpu
            batch_count += 1
            if batch_count % log_interval == 0:
                # Print average per item
                avg_data = total_data_time / log_interval
                avg_gpu = total_gpu_time / log_interval
                print(f" [Profile] Data Load: {avg_data:.3f}s, GPU Infer: {avg_gpu:.3f}s ")
                # Reset
                total_data_time = 0
                total_gpu_time = 0
                
                # Check ratio
                # if avg_data > avg_gpu: print(" (CPU Bound) ", end='\r')
                # else: print(" (GPU Bound) ", end='\r')


            # Save
            for i, info in enumerate(infos):
                executor.submit(
                    save_npz_async,
                    info.latents_npz,
                    latents_to_save[i],
                    orig_sizes[i],
                    crops[i],
                    None, None,
                    latents_dtype_tag
                )
            
            t_start_data = time.time()

                
    # Wait for IO
    print("Waiting for disk I/O to finish...")
    executor.shutdown(wait=True)
    
    end_time = time.time()
    print(f"Finished. Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()

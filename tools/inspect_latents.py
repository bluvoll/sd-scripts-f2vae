import numpy as np
import argparse
import os
import sys

def inspect(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    try:
        with np.load(path) as data:
            keys = list(data.keys())
            print(f"Keys found: {keys}")
            
            if 'latents' not in data:
                print(f"Error: 'latents' key missing.")
                return
            
            latents = data['latents']
            print("--- Latent Statistics ---")
            print(f"Shape: {latents.shape}")
            print(f"Dtype: {latents.dtype}")
            print(f"Total Elements: {latents.size}")
            
            # Estimate uncompressed size
            bytes_per_elem = latents.itemsize
            total_bytes = latents.size * bytes_per_elem
            print(f"Raw Data Size: {total_bytes / 1024 / 1024:.2f} MB")
            print(f"File on Disk: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
            print(f"Min: {latents.min():.4f}")
            print(f"Max: {latents.max():.4f}")
            print(f"Mean: {latents.mean():.4f}")
            print(f"Std: {latents.std():.4f}")
            
            # Check for NaN/Inf
            if np.isnan(latents).any():
                print("WARNING: Contains NaNs!")
            if np.isinf(latents).any():
                print("WARNING: Contains Infs!")

            # Guess dimensions
            # Usually (C, H, W)
            if len(latents.shape) == 3:
                C, H, W = latents.shape
                print(f"Assumed Layout (C, H, W): Channels={C}, Height={H}, Width={W}")
                print(f"Aspect Ratio: {W/H:.2f}")
                
            elif len(latents.shape) == 4:
                B, C, H, W = latents.shape
                print(f"Assumed Layout (B, C, H, W): Batch={B}, Channels={C}, Height={H}, Width={W}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .npz latent file")
    args = parser.parse_args()
    inspect(args.path)

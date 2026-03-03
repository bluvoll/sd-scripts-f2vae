"""Scan cached latent .npz files for dimensions not aligned to bucket_reso_steps."""

import argparse
import glob
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np


def check_npz(args_tuple):
    npz_path, reso_steps_latent = args_tuple
    try:
        with np.load(npz_path) as npz:
            if "latents" not in npz:
                return npz_path, "missing_latents"
            shape = npz["latents"].shape  # [4, H, W]
            h, w = shape[1], shape[2]
            if h % reso_steps_latent != 0 or w % reso_steps_latent != 0:
                return npz_path, f"bad_shape {shape} -> pixel {h*8}x{w*8}"
    except Exception as e:
        return npz_path, f"error: {e}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs="+", help="directories to scan recursively")
    parser.add_argument("--bucket_reso_steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(32, cpu_count()))
    parser.add_argument("--delete", action="store_true", help="delete bad npz files")
    args = parser.parse_args()

    reso_steps_latent = args.bucket_reso_steps // 8  # 64 -> 8 in latent space

    npz_files = []
    for d in args.data_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".npz") and not f.endswith("_te_outputs.npz"):
                    npz_files.append(os.path.join(root, f))

    print(f"Scanning {len(npz_files)} npz files with {args.workers} workers...")

    bad_files = []
    work = [(f, reso_steps_latent) for f in npz_files]

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(check_npz, work, chunksize=256)):
            if result is not None:
                bad_files.append(result)
                print(f"  FOUND: {result[0]} — {result[1]}")
            if (i + 1) % 100000 == 0:
                print(f"  checked {i+1}/{len(npz_files)}...")

    print(f"\nDone. Found {len(bad_files)} bad file(s).")

    if bad_files and args.delete:
        for path, reason in bad_files:
            os.remove(path)
            print(f"  deleted: {path}")
        print(f"Deleted {len(bad_files)} file(s). Re-run training to re-cache them.")


if __name__ == "__main__":
    main()

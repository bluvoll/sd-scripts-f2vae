#!/usr/bin/env python3
"""
Generate a metadata JSON (caption + train_resolution) using the same loading path as sd-scripts.

This builds the dataset via config_util (so captions, repeats, wildcards, etc. match training),
runs bucket size calculation (image size reads), then emits a JSON mapping absolute image paths
to {"caption": str, "train_resolution": [w, h]}.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import List

from library import config_util
from library.config_util import BlueprintGenerator, ConfigSanitizer


class _DummyTokenizer:
    """Minimal tokenizer stub: only the attributes BaseDataset needs."""

    def __init__(self, max_length: int = 77) -> None:
        self.model_max_length = max_length
        self.eos_token_id = 0
        self.pad_token_id = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute metadata JSON (caption + train_resolution).")
    parser.add_argument(
        "--dataset_config",
        type=Path,
        required=True,
        help="Path to the dataset toml used for training (same as --dataset_config in training).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the generated metadata JSON.",
    )
    parser.add_argument(
        "--train_resolution_mode",
        choices=["actual", "bucket"],
        default="actual",
        help="What to store in train_resolution: the raw image size ('actual', default) or the bucket resolution chosen by sd-scripts ('bucket').",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=77,
        help="Tokenizer max length stub (not used for size calc; defaults to 77).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    user_config = config_util.load_user_config(args.dataset_config)

    # Build a minimal argparse namespace; most values come from the dataset_config itself.
    minimal_args = Namespace(masked_loss=False, debug_dataset=False)

    # Use dummy tokenizers so we don't need to download real ones just to measure images.
    dummy_tokens: List[_DummyTokenizer] = [
        _DummyTokenizer(args.tokenizer_max_length),
        _DummyTokenizer(args.tokenizer_max_length),
    ]

    generator = BlueprintGenerator(ConfigSanitizer(True, True, True, True))
    blueprint = generator.generate(user_config, minimal_args, tokenizer=dummy_tokens)

    # This call instantiates datasets and runs make_buckets(), which reads image sizes if missing.
    dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    meta = {}
    for dataset in dataset_group.datasets:
        for info in dataset.image_data.values():
            if info.image_size is None:
                continue  # should not happen after make_buckets, but guard just in case
            if args.train_resolution_mode == "bucket":
                reso = info.bucket_reso
            else:
                reso = info.image_size
            meta[info.absolute_path] = {
                "caption": info.caption,
                "train_resolution": list(reso),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(meta)} entries to {args.out}")


if __name__ == "__main__":
    main()

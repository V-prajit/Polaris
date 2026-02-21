#!/usr/bin/env python3
"""
Download ParkSeg12k from HuggingFace and save as images/ + masks/ layout
for SegFormer training with ParkSegDataset.

Defaults to the official HuggingFace ``train`` split only to avoid data leakage
from the held-out ``test`` set. Pass ``--include_test`` only if you explicitly
want a train+test merged corpus.

Usage
-----
    python scripts/download_parkseg.py --output_dir data/parkseg12k

This creates::

    data/parkseg12k/
        images/   train_00000.jpg, train_00001.jpg, ...
        masks/    train_00000.png, train_00001.png, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description="Download ParkSeg12k for SegFormer training")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/parkseg12k"),
        help="Where to save images/ and masks/",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Stream from HuggingFace (avoids downloading full parquet to disk)",
    )
    p.add_argument(
        "--no_streaming",
        dest="streaming",
        action="store_false",
    )
    p.add_argument(
        "--include_test",
        action="store_true",
        help="Also include HF test split (off by default to avoid leakage).",
    )
    args = p.parse_args()

    try:
        import datasets
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"]
        )
        import datasets

    img_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ParkSeg12k from HuggingFace...")
    ds = datasets.load_dataset("UTEL-UIUC/parkseg12k", streaming=args.streaming)

    count = 0
    split_names = ["train", "test"] if args.include_test else ["train"]
    for split_name in split_names:
        if split_name not in ds:
            print(f"  Split '{split_name}' not found, skipping.")
            continue

        for i, item in enumerate(ds[split_name]):
            fname = f"{split_name}_{i:05d}"

            # Save RGB image
            rgb = item["rgb"]
            if not isinstance(rgb, Image.Image):
                rgb = Image.fromarray(np.array(rgb))
            rgb.convert("RGB").save(img_dir / f"{fname}.jpg", quality=95)

            # Save mask as-is (grayscale with class IDs preserved)
            mask = item["mask"]
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.array(mask))
            # Ensure single-channel
            if mask.mode != "L":
                mask = mask.convert("L")
            mask.save(mask_dir / f"{fname}.png")

            count += 1
            if count % 500 == 0:
                print(f"  Saved {count} samples...")

    print(f"Done. {count} samples saved to {args.output_dir}")
    print(f"  Images: {img_dir}")
    print(f"  Masks:  {mask_dir}")
    print()
    print("To train SegFormer:")
    print(f"  python scripts/train_segformer.py --data_dir {args.output_dir} \\")
    print("      --output_dir checkpoints/segformer-b5-parkseg --epochs 30")


if __name__ == "__main__":
    main()

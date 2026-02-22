#!/usr/bin/env python3
"""
Prepare APKLOT dataset in YOLO format for parking spot/block detection.

Reads PASCAL VOC XML annotations from the APKLOT satellite dataset and
converts them to YOLO format (class cx cy w h, normalised).

Usage
-----
    python scripts/prepare_apklot_yolo.py \\
        --apklot_dir "data/APKLOT/1. Satellite/Dataset" \\
        --output_dir data/apklot_yolo \\
        --val_split 0.15
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def get_yolo_format(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def process_pascal_voc(apklot_dir: Path) -> list[dict]:
    xml_pattern = str(apklot_dir / "*/PASCAL_format/Annotations/*.xml")
    xml_files = sorted(glob.glob(xml_pattern))
    print(f"Found {len(xml_files)} PASCAL VOC XML files")

    samples = []
    skipped = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_name = root.find("filename").text
        xml_dir = os.path.dirname(xml_file)
        jpeg_dir = os.path.join(os.path.dirname(xml_dir), "JPEGImages")
        img_path = os.path.join(jpeg_dir, img_name)

        if not os.path.exists(img_path):
            stem = os.path.splitext(os.path.basename(xml_file))[0]
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(jpeg_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if not os.path.exists(img_path):
            skipped += 1
            continue

        size = root.find("size")
        if size is not None:
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
        else:
            from PIL import Image
            img = Image.open(img_path)
            img_w, img_h = img.size

        yolo_labels = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = max(0, float(bndbox.find("xmin").text))
            ymin = max(0, float(bndbox.find("ymin").text))
            xmax = min(img_w, float(bndbox.find("xmax").text))
            ymax = min(img_h, float(bndbox.find("ymax").text))

            if xmax <= xmin or ymax <= ymin:
                continue

            cx, cy, w, h = get_yolo_format([xmin, ymin, xmax, ymax], img_w, img_h)
            yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_labels:
            samples.append({"image_path": img_path, "labels": yolo_labels})

    print(f"Parsed {len(samples)} images with annotations ({skipped} skipped)")
    return samples


def main():
    p = argparse.ArgumentParser(description="Prepare APKLOT for YOLO training")
    p.add_argument(
        "--apklot_dir", type=Path,
        default=Path("data/APKLOT/1. Satellite/Dataset"),
        help="APKLOT satellite dataset root",
    )
    p.add_argument("--output_dir", type=Path, default=Path("data/apklot_yolo"))
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = args.output_dir
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    samples = process_pascal_voc(args.apklot_dir)
    if not samples:
        print("ERROR: No samples found. Check --apklot_dir path.")
        return

    random.seed(args.seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Split: {len(train_samples)} train / {len(val_samples)} val")

    def save_split(split_samples, split_name):
        for i, s in enumerate(split_samples):
            img_name = f"{split_name}_{i:05d}.jpg"
            img_dst = out / split_name / "images" / img_name
            txt_dst = out / split_name / "labels" / img_name.replace(".jpg", ".txt")
            shutil.copy2(s["image_path"], img_dst)
            txt_dst.write_text("\n".join(s["labels"]))

    save_split(train_samples, "train")
    save_split(val_samples, "val")

    yaml_config = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "parkingspot"},
    }
    yaml_path = out / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"Done! Dataset saved to {out}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Train: {len(train_samples)} images")
    print(f"  Val:   {len(val_samples)} images")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train YOLO on APKLOT dataset for parking spot detection.

Usage (single GPU)::

    python scripts/train_yolo_apklot.py --data_dir data/apklot_yolo --epochs 80

Usage (sbatch)::

    sbatch --wrap='python scripts/train_yolo_apklot.py \
        --data_dir data/apklot_yolo --epochs 80 --batch 64 --device 0'

If the APKLOT YOLO dataset hasn't been prepared yet, run first::

    python scripts/prepare_apklot_yolo.py --output_dir data/apklot_yolo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on APKLOT parking spots")
    p.add_argument("--data_dir", type=Path, default=Path("data/apklot_yolo"),
                    help="APKLOT YOLO dataset root (must contain data.yaml)")
    p.add_argument("--model", type=str, default="yolo11n.pt",
                    help="Base YOLO model (default: yolo11n.pt)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640,
                    help="Training image size (APKLOT images are large, 640 recommended)")
    p.add_argument("--batch", type=int, default=64,
                    help="Batch size (adjust for GPU memory)")
    p.add_argument("--device", type=str, default="0", help="CUDA device(s)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--project", type=str, default="runs")
    p.add_argument("--name", type=str, default="yolo_apklot")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    data_yaml = args.data_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"data.yaml not found at {data_yaml}")
        print("Running prepare_apklot_yolo.py first...")
        prep_script = _REPO_ROOT / "scripts" / "prepare_apklot_yolo.py"
        os.system(f"{sys.executable} {prep_script} --output_dir {args.data_dir}")
        if not data_yaml.exists():
            print("ERROR: Dataset preparation failed.")
            sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        plots=True,
        workers=args.workers,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15.0,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
    )

    # Copy best weights to models/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        dst = _REPO_ROOT / "models" / "yolo_apklot_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_pt, dst)
        print(f"Best weights copied to {dst}")


if __name__ == "__main__":
    main()

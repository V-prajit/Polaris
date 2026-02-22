#!/usr/bin/env python3
"""
Evaluate a fine-tuned SegFormer-b5 checkpoint on ParkSeg12k.

Reports per-image mIoU, pixel accuracy, and compares predicted stall count
to the ground-truth count derived from the mask.

Usage
-----
::

    python scripts/eval_segformer.py \\
        --checkpoint checkpoints/segformer-b5-parkseg/best_model \\
        --data_dir /path/to/ParkSeg12k \\
        --output_csv results/segformer_eval.csv

Output columns in CSV:
    image_path, miou, stall_iou, pixel_acc, gt_count, pred_count, abs_err, rel_err
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# ── Repo on path ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from parksight.data import ParkSegDataset, STALL_CLASS_IDS
from parksight.segment import ParkingSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_segformer")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SegFormer-b5 on ParkSeg12k")
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to fine-tuned model directory")
    p.add_argument("--data_dir", required=True, type=Path, help="ParkSeg12k root directory")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stall_class_ids", nargs="+", type=int, default=list(STALL_CLASS_IDS))
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_csv", type=Path, default=Path("results/segformer_eval.csv"), help="Where to write per-image CSV results")
    p.add_argument("--stall_area_px", type=int, default=400, help="Avg stall area in pixels (512×512 canonical)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    ious = []
    for c in range(num_classes):
        i = ((preds == c) & (labels == c)).sum()
        u = ((preds == c) | (labels == c)).sum()
        if u == 0:
            continue
        ious.append(i / u)
    return float(np.mean(ious)) if ious else 0.0


def compute_class_iou(preds: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    inter = ((preds == class_id) & (labels == class_id)).sum()
    union = ((preds == class_id) | (labels == class_id)).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_to_count(mask: np.ndarray, stall_area_px: int) -> int:
    """Area-based count from binary mask."""
    return max(0, round(int(mask.sum()) / stall_area_px))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Device: %s", device)

    # Load model
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(
        str(args.checkpoint),
        do_resize=True,
        size={"height": args.img_size, "width": args.img_size},
    )
    model = SegformerForSemanticSegmentation.from_pretrained(str(args.checkpoint)).to(device)
    model.eval()

    # Build val dataset (same split as training)
    stall_ids = set(args.stall_class_ids)
    _, val_ds = ParkSegDataset.make_splits(
        args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        stall_class_ids=stall_ids,
        img_size=args.img_size,
    )
    logger.info("Validation set: %d samples", len(val_ds))

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
    )

    # Evaluation loop
    rows: list[dict] = []
    global_preds_all: list[np.ndarray] = []
    global_labels_all: list[np.ndarray] = []
    abs_errors: list[float] = []
    rel_errors: list[float] = []

    # We need per-image metrics so we iterate sample by sample inside each batch
    for batch_idx, batch in enumerate(val_loader):
        pixel_values = batch["pixel_values"].to(device)
        labels_batch = batch["labels"]  # (B, H, W)
        image_paths = batch["image_path"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        upsampled = F.interpolate(
            outputs.logits,
            size=labels_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # (B, 2, H, W)
        preds_batch = upsampled.argmax(dim=1).cpu().numpy()  # (B, H, W)
        labels_np_batch = labels_batch.numpy()

        for i in range(len(image_paths)):
            pred = preds_batch[i]   # (H, W)
            label = labels_np_batch[i]  # (H, W)

            miou = compute_miou(pred.flatten(), label.flatten())
            stall_iou = compute_class_iou(pred.flatten(), label.flatten(), class_id=1)
            px_acc = float((pred == label).mean())

            gt_count = mask_to_count(label.astype(np.uint8), args.stall_area_px)
            pred_count = mask_to_count(pred.astype(np.uint8), args.stall_area_px)

            abs_err = abs(pred_count - gt_count)
            rel_err = abs_err / max(1, gt_count)

            rows.append({
                "image_path": image_paths[i],
                "miou": round(miou, 4),
                "stall_iou": round(stall_iou, 4),
                "pixel_acc": round(px_acc, 4),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "abs_err": abs_err,
                "rel_err": round(rel_err, 4),
            })

            global_preds_all.append(pred.flatten())
            global_labels_all.append(label.flatten())
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)

        if (batch_idx + 1) % 10 == 0:
            logger.info("  Processed %d / %d batches …", batch_idx + 1, len(val_loader))

    # Aggregate stats
    all_preds_np = np.concatenate(global_preds_all)
    all_labels_np = np.concatenate(global_labels_all)
    global_miou = compute_miou(all_preds_np, all_labels_np)
    global_stall_iou = compute_class_iou(all_preds_np, all_labels_np, class_id=1)
    global_px_acc = float((all_preds_np == all_labels_np).mean())
    mean_abs_err = float(np.mean(abs_errors))
    mean_rel_err = float(np.mean(rel_errors))
    median_abs_err = float(np.median(abs_errors))

    logger.info("=" * 50)
    logger.info("Global mIoU          : %.4f", global_miou)
    logger.info("Global stall IoU     : %.4f", global_stall_iou)
    logger.info("Global pixel accuracy: %.4f", global_px_acc)
    logger.info("Mean absolute error  : %.2f stalls", mean_abs_err)
    logger.info("Median absolute error: %.2f stalls", median_abs_err)
    logger.info("Mean relative error  : %.2f%%", mean_rel_err * 100)
    logger.info("=" * 50)

    # Write CSV
    fieldnames = ["image_path", "miou", "stall_iou", "pixel_acc", "gt_count", "pred_count", "abs_err", "rel_err"]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Append summary row
    with args.output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "image_path": "AGGREGATE",
            "miou": round(global_miou, 4),
            "stall_iou": round(global_stall_iou, 4),
            "pixel_acc": round(global_px_acc, 4),
            "gt_count": "—",
            "pred_count": "—",
            "abs_err": round(mean_abs_err, 2),
            "rel_err": round(mean_rel_err, 4),
        })

    logger.info("Results written to %s (%d rows)", args.output_csv, len(rows))


if __name__ == "__main__":
    main()

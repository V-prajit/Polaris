#!/usr/bin/env python3
"""
Fine-tune SegFormer-b5 on ParkSeg12k for binary parking-stall segmentation.

Usage
-----
Single GPU::

    python scripts/train_segformer.py \\
        --data_dir /path/to/ParkSeg12k \\
        --output_dir checkpoints/segformer-b5-parkseg \\
        --epochs 30 --batch_size 8 --lr 6e-5

Multi-GPU (torchrun)::

    torchrun --nproc_per_node=4 scripts/train_segformer.py \\
        --data_dir /path/to/ParkSeg12k \\
        --output_dir checkpoints/segformer-b5-parkseg \\
        --epochs 30 --batch_size 8 --lr 6e-5

The script will:
  1. Load ParkSeg12k via ``parksight.data.ParkSegDataset``.
  2. Fine-tune ``nvidia/segformer-b5-finetuned-ade-640-640`` (binary: stall / bg).
  3. Log mIoU, pixel accuracy, and loss per epoch to stdout + ``<output_dir>/metrics.json``.
  4. Save the best mIoU checkpoint to ``<output_dir>/best_model/``.
  5. Save the final checkpoint to ``<output_dir>/final_model/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T

# ── Make sure repo is on PYTHONPATH ────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from parksight.data import ParkSegDataset


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-layout dataset (train/images + train/labels)
# ─────────────────────────────────────────────────────────────────────────────

class YoloSegDataset(Dataset):
    """
    Reads a pre-split YOLO dataset layout::

        data_dir/
          train/images/*.jpg
          train/labels/*.txt   ← YOLO bbox format (class cx cy w h, normalised)
          val/images/*.jpg
          val/labels/*.txt

    Each sample returns:
        ``pixel_values`` : float32 tensor [3, img_size, img_size]  (ImageNet-normed)
        ``labels``       : int64  tensor [img_size, img_size]       (0=bg, 1=stall)
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: Path,
        split: str,          # "train" or "val"
        img_size: int = 512,
        augment: bool = True,
    ) -> None:
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        img_dir   = data_dir / split / "images"
        label_dir = data_dir / split / "labels"

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Expected {img_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Expected {label_dir}")

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        self.pairs: List[Tuple[Path, Path]] = []
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in exts:
                continue
            lbl_p = label_dir / (img_p.stem + ".txt")
            if lbl_p.exists():
                self.pairs.append((img_p, lbl_p))

        if not self.pairs:
            raise ValueError(f"No (image, label) pairs found under {data_dir}/{split}/")

        logger.info("YoloSegDataset [%s]: %d pairs", split, len(self.pairs))

        # Shared normalisation transform
        self._to_tensor = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _bbox_to_mask(self, label_path: Path, img_w: int, img_h: int) -> np.ndarray:
        """Render YOLO bboxes into a binary numpy mask (uint8, 0/1)."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        try:
            lines = label_path.read_text().splitlines()
        except Exception:
            return mask
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            mask[y1:y2, x1:x2] = 1
        return mask

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Build binary mask from YOLO bboxes
        mask_np = self._bbox_to_mask(lbl_path, img_w, img_h)
        mask_img = Image.fromarray(mask_np * 255)  # 0 or 255

        # Optional random horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            image    = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize image and mask to img_size
        pixel_values = self._to_tensor(image)  # [3, H, W]

        mask_resized = mask_img.resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )
        labels = torch.from_numpy(
            (np.array(mask_resized) > 127).astype(np.int64)
        )  # [H, W], values 0/1

        return {"pixel_values": pixel_values, "labels": labels}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_segformer")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SegFormer-b5 on ParkSeg12k (binary parking segmentation)"
    )

    # Data
    p.add_argument("--data_dir", required=True, type=Path, help="ParkSeg12k root directory (contains images/ and masks/)")
    p.add_argument("--val_split", type=float, default=0.15, help="Validation fraction (default 0.15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stall_class_ids", nargs="+", type=int, default=[1, 2, 3], help="ParkSeg12k class IDs to map to 'stall' (default: 1 2 3)")

    # Model
    p.add_argument("--base_model", type=str, default="nvidia/segformer-b5-finetuned-ade-640-640", help="HuggingFace model ID to start from")
    p.add_argument("--img_size", type=int, default=512, help="Training image size (default 512)")

    # Training
    p.add_argument("--output_dir", required=True, type=Path, help="Where to save checkpoints and metrics")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=6e-5, help="Peak learning rate (AdamW)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_power", type=float, default=0.9, help="PolynomialLR power")
    p.add_argument("--early_stop_patience", type=int, default=7, help="Epochs without val mIoU improvement before stopping")
    p.add_argument("--class_weight_bg", type=float, default=0.3, help="Cross-entropy weight for background class (stall weight = 1.0)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True, help="Use mixed precision (default True)")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--resume", type=Path, default=None, help="Resume from a saved checkpoint directory")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    """Compute mean IoU over *num_classes* from flat arrays."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum()
        union = (pred_c | label_c).sum()
        if union == 0:
            continue  # class absent in batch — skip
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_acc(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    scaler,
    device: str,
    amp: bool,
    grad_accum: int,
    loss_fn,
    rank: int,
) -> dict:
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(amp and "cuda" in device)):
            outputs = model(pixel_values=pixel_values, labels=labels)
            # HF SegFormer computes CE loss internally when labels provided
            loss = outputs.loss / grad_accum

        if amp and "cuda" in device:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if amp and "cuda" in device:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        steps += 1

    return {"loss": total_loss / max(steps, 1)}


@torch.no_grad()
def validate(model, loader: DataLoader, device: str, amp: bool) -> dict:
    import torch.nn.functional as F

    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    steps = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(amp and "cuda" in device)):
            outputs = model(pixel_values=pixel_values, labels=labels)

        # Upsample logits to label resolution
        logits = outputs.logits  # (B, C, H/4, W/4)
        upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = upsampled.argmax(dim=1)

        all_preds.append(preds.cpu().numpy().flatten())
        all_labels.append(labels.cpu().numpy().flatten())
        total_loss += outputs.loss.item()
        steps += 1

    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)

    return {
        "val_loss": total_loss / max(steps, 1),
        "val_miou": compute_miou(all_preds_np, all_labels_np),
        "val_pixel_acc": compute_pixel_acc(all_preds_np, all_labels_np),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Distributed setup ──────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rank = local_rank
    is_main = rank == 0

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", args.output_dir)
        logger.info("Device: %s | world_size: %d | AMP: %s", device, world_size, args.amp)

    # ── Seed ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ── DataLoaders ────────────────────────────────────────────────────
    # Auto-detect YOLO pre-split layout (data_dir/train/images exists)
    yolo_train_dir = args.data_dir / "train" / "images"
    if yolo_train_dir.is_dir():
        if is_main:
            logger.info("Detected YOLO pre-split layout — using YoloSegDataset")
        train_ds = YoloSegDataset(args.data_dir, split="train", img_size=args.img_size, augment=True)
        val_ds   = YoloSegDataset(args.data_dir, split="val",   img_size=args.img_size, augment=False)
    else:
        if is_main:
            logger.info("Using ParkSegDataset (images/ + masks/ layout)")
        stall_ids = set(args.stall_class_ids)
        train_ds, val_ds = ParkSegDataset.make_splits(
            args.data_dir,
            val_split=args.val_split,
            seed=args.seed,
            stall_class_ids=stall_ids,
            img_size=args.img_size,
        )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if is_main:
        logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────
    from transformers import SegformerForSemanticSegmentation

    id2label = {0: "background", 1: "parking_stall"}
    label2id = {"background": 0, "parking_stall": 1}

    if args.resume:
        if is_main:
            logger.info("Resuming from %s", args.resume)
        model = SegformerForSemanticSegmentation.from_pretrained(str(args.resume))
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.base_model,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,  # decoder head changes size
        )

    # Apply class weights to CrossEntropyLoss
    class_weights = torch.tensor(
        [args.class_weight_bg, 1.0], dtype=torch.float32, device=device
    )
    # Patch the model's config so HF can use our weights via a custom loss below
    # (HF SegFormer doesn't expose class_weight natively — we override the loss)
    model.to(device)

    # Wrap model for DDP
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    # ── Custom loss wrapper ────────────────────────────────────────────
    import torch.nn.functional as F

    def weighted_ce_loss(logits, labels, class_weights, ignore_index=255):
        """Apply weighted CrossEntropyLoss with upsampling."""
        upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(upsampled, labels, weight=class_weights, ignore_index=ignore_index)

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=args.lr_power)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and "cuda" in device))

    # ── Training loop ─────────────────────────────────────────────────
    best_miou = 0.0
    patience_counter = 0
    all_metrics: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        # --- Custom train loop (to use weighted loss) ---
        core_model = model.module if is_distributed else model
        core_model.train()
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(args.amp and "cuda" in device)):
                outputs = core_model(pixel_values=pixel_values)
                loss = weighted_ce_loss(outputs.logits, labels, class_weights) / args.grad_accum

            if args.amp and "cuda" in device:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if args.amp and "cuda" in device:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            steps += 1

        train_metrics = {"loss": total_loss / max(steps, 1)}

        # --- Validation ---
        core_model.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=(args.amp and "cuda" in device)):
                    outputs = core_model(pixel_values=pixel_values)
                    v_loss = weighted_ce_loss(outputs.logits, labels, class_weights)

                upsampled = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                preds = upsampled.argmax(dim=1)
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels_list.append(labels.cpu().numpy().flatten())
                val_loss += v_loss.item()
                val_steps += 1

        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels_list)
        val_miou = compute_miou(preds_np, labels_np)
        val_acc = compute_pixel_acc(preds_np, labels_np)
        val_metrics = {
            "val_loss": val_loss / max(val_steps, 1),
            "val_miou": val_miou,
            "val_pixel_acc": val_acc,
        }

        elapsed = time.time() - t0
        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr,
            "elapsed_s": round(elapsed, 1),
        }
        all_metrics.append(epoch_metrics)

        if is_main:
            logger.info(
                "Epoch %02d/%02d | loss=%.4f | val_loss=%.4f | mIoU=%.4f | pxAcc=%.4f | %.0fs",
                epoch, args.epochs,
                train_metrics["loss"],
                val_metrics["val_loss"],
                val_miou,
                val_acc,
                elapsed,
            )

            # Save best checkpoint
            if val_miou > best_miou:
                best_miou = val_miou
                patience_counter = 0
                best_dir = args.output_dir / "best_model"
                core_model.save_pretrained(best_dir)
                logger.info("  ✓ New best mIoU=%.4f → saved to %s", best_miou, best_dir)
            else:
                patience_counter += 1
                logger.info("  Patience: %d/%d", patience_counter, args.early_stop_patience)

            # Save metrics JSON
            with (args.output_dir / "metrics.json").open("w") as f:
                json.dump(all_metrics, f, indent=2)

        if patience_counter >= args.early_stop_patience:
            if is_main:
                logger.info("Early stopping at epoch %d (best mIoU=%.4f).", epoch, best_miou)
            break

    # Save final model
    if is_main:
        final_dir = args.output_dir / "final_model"
        core_model = model.module if is_distributed else model
        core_model.save_pretrained(final_dir)
        logger.info("Final model saved to %s", final_dir)
        logger.info("Training complete. Best val mIoU: %.4f", best_miou)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Dataset utilities for fine-tuning SegFormer-b5 on ParkSeg12k.

Provides:
- ``ParkSegDataset``  — PyTorch Dataset for ParkSeg12k images + masks.
- ``OSMCropDataset``  — Thin inference wrapper: PIL image + OSM bbox list.
- ``get_transforms``  — Albumentations pipelines for train / val / test splits.

Expected ParkSeg12k layout
--------------------------
::

    <data_dir>/
        images/     *.jpg | *.png   (RGB satellite tiles)
        masks/      *.png           (grayscale, same stem; pixel value = class id)

Mask remapping
--------------
ParkSeg12k stores class indices.  We remap to **binary**:

    1 = parking stall (any stall/marking class)
    0 = background   (everything else)

If ParkSeg12k uses a different schema, update ``STALL_CLASS_IDS`` below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Class remapping ────────────────────────────────────────────────────────────
# ParkSeg12k class IDs that correspond to parking stall markings.
# Adjust if the dataset uses different indices.
STALL_CLASS_IDS: set[int] = {255}  # ParkSeg12k uses 0=bg, 255=parking

# Image size fed to SegFormer
IMG_SIZE: int = 512

# ── Try importing albumentations (optional but recommended) ────────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _ALBU_AVAILABLE = True
except ImportError:
    _ALBU_AVAILABLE = False
    logger.warning(
        "albumentations not installed — using basic torchvision transforms instead. "
        "Install with: pip install albumentations"
    )

# ── Try importing torchvision as fallback ─────────────────────────────────────
try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────────────────────────────────────


def get_transforms(
    split: str = "train",
    img_size: int = IMG_SIZE,
) -> Any:
    """
    Return an augmentation pipeline appropriate for *split*.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    img_size : int
        Target spatial resolution (height = width).

    Returns
    -------
    albumentations.Compose | torchvision.transforms.Compose | None
        Transform callable.  Applied in ``ParkSegDataset.__getitem__``.
    """
    if _ALBU_AVAILABLE:
        return _get_albu_transforms(split, img_size)
    if _TORCHVISION_AVAILABLE:
        logger.warning("Falling back to basic torchvision transforms (no mask-safe augmentations).")
        return _get_torchvision_transforms(split, img_size)
    logger.warning("No augmentation library found — returning None (raw PIL images).")
    return None


def _get_albu_transforms(split: str, img_size: int) -> "A.Compose":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.333),
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05,
                    p=0.6,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.RandomShadow(p=0.15),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:  # val / test
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def _get_torchvision_transforms(split: str, img_size: int) -> T.Compose:
    """Minimal torchvision pipeline (image only — spatial augmentations are
    handled in __getitem__ to keep image/mask in sync)."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                # NOTE: RandomHorizontalFlip removed — applied manually in
                # __getitem__ so the mask can be flipped in sync.
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize])


# ─────────────────────────────────────────────────────────────────────────────
# ParkSegDataset
# ─────────────────────────────────────────────────────────────────────────────


class ParkSegDataset:
    """
    PyTorch-compatible Dataset for ParkSeg12k (or any compatible directory).

    Parameters
    ----------
    data_dir : str or Path
        Root directory with ``images/`` and ``masks/`` subdirectories.
    split : str
        ``"train"`` or ``"val"`` — controls which augmentations are applied.
    transform : callable or None
        Override the default transform pipeline.  If *None*, ``get_transforms``
        is called with *split*.
    val_split : float
        Fraction of data to hold out for validation (only used when building
        train/val splits from the same directory).
    seed : int
        Random seed for the train/val split.
    indices : list[int] or None
        If provided, use only these indices from the full file list.
    stall_class_ids : set[int]
        ParkSeg12k class IDs to remap to 1 (parking stall).
    img_size : int
        Spatial resolution for SegFormer input (default 512).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_split: float = 0.15,
        seed: int = 42,
        indices: Optional[List[int]] = None,
        stall_class_ids: Optional[set[int]] = None,
        img_size: int = IMG_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.stall_class_ids = stall_class_ids or STALL_CLASS_IDS

        # Gather image paths
        img_dir = self.data_dir / "images"
        mask_dir = self.data_dir / "masks"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Expected images directory at {img_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Expected masks directory at {mask_dir}")

        all_img_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
        if not all_img_paths:
            raise ValueError(f"No images found in {img_dir}")

        # Filter to images that have a corresponding mask
        paired: List[Tuple[Path, Path]] = []
        for img_p in all_img_paths:
            mask_p = mask_dir / (img_p.stem + ".png")
            if mask_p.exists():
                paired.append((img_p, mask_p))
            else:
                logger.debug("No mask for %s — skipping.", img_p.name)

        if not paired:
            raise ValueError(f"No (image, mask) pairs found under {self.data_dir}")

        logger.info("Found %d image-mask pairs in %s", len(paired), self.data_dir)

        # Apply indices (pre-computed train/val split)
        if indices is not None:
            paired = [paired[i] for i in indices]
        self.pairs = paired

        # Transform
        self.transform = transform if transform is not None else get_transforms(split, img_size)

    # -- Helper: build train/val split indices --------------------------------

    @classmethod
    def make_splits(
        cls,
        data_dir: str | Path,
        val_split: float = 0.15,
        seed: int = 42,
        **kwargs,
    ) -> Tuple["ParkSegDataset", "ParkSegDataset"]:
        """
        Convenience factory: returns ``(train_dataset, val_dataset)``.

        Parameters
        ----------
        data_dir : str or Path
        val_split : float
        seed : int
        **kwargs
            Forwarded to :class:`ParkSegDataset` (e.g. ``stall_class_ids``).

        Returns
        -------
        (ParkSegDataset, ParkSegDataset)
        """
        import random

        tmp = cls(data_dir, split="train", transform=lambda x: x, **kwargs)
        n = len(tmp.pairs)
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_val = max(1, int(n * val_split))
        val_idx = sorted(idx[:n_val])
        train_idx = sorted(idx[n_val:])

        train_ds = cls(data_dir, split="train", indices=train_idx, **kwargs)
        val_ds = cls(data_dir, split="val", indices=val_idx, **kwargs)
        logger.info("Split: %d train / %d val", len(train_ds), len(val_ds))
        return train_ds, val_ds

    # -- Dataset interface ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
            ``"pixel_values"`` : torch.Tensor[3, H, W]  (float, normalised)
            ``"labels"``       : torch.Tensor[H, W]     (long, 0/1)
            ``"image_path"``   : str  (for debugging)
        """
        import torch

        img_path, mask_path = self.pairs[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask and remap to binary
        raw_mask = np.array(Image.open(mask_path), dtype=np.int32)
        binary_mask = np.isin(raw_mask, sorted(self.stall_class_ids)).astype(np.uint8)

        if self.transform is not None and _ALBU_AVAILABLE and isinstance(
            self.transform, type(None).__class__.__mro__[0]  # always False, just a guard
        ):
            pass  # unreachable; kept for linting clarity

        if _ALBU_AVAILABLE and self.transform is not None:
            img_np = np.array(image)
            transformed = self.transform(image=img_np, mask=binary_mask)
            pixel_values = transformed["image"]  # Tensor[3, H, W]
            mask_out = transformed["mask"]
            labels = mask_out.long() if isinstance(mask_out, torch.Tensor) else torch.from_numpy(mask_out).long()
        elif _TORCHVISION_AVAILABLE and self.transform is not None:
            # torchvision path — apply spatial augmentations manually to keep
            # image and mask in sync, then apply the rest of the transform.
            import random as _rand
            import torchvision.transforms.functional as TF

            # Synchronized random horizontal flip
            if self.split == "train" and _rand.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                binary_mask = np.fliplr(binary_mask).copy()

            pixel_values = self.transform(image)  # Tensor[3, H, W]
            mask_pil = Image.fromarray(binary_mask).resize(
                (self.img_size, self.img_size), Image.NEAREST
            )
            labels = torch.from_numpy(np.array(mask_pil)).long()
        else:
            # Fallback: no transform, just convert
            import torchvision.transforms.functional as TF

            pixel_values = TF.to_tensor(image.resize((self.img_size, self.img_size)))
            mask_rs = np.array(
                Image.fromarray(binary_mask).resize((self.img_size, self.img_size), Image.NEAREST)
            )
            labels = torch.from_numpy(mask_rs).long()

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(img_path),
        }


# ─────────────────────────────────────────────────────────────────────────────
# OSMCropDataset  (inference — no labels)
# ─────────────────────────────────────────────────────────────────────────────


class OSMCropDataset:
    """
    Inference dataset: applies OSM parking bbox crops to a single satellite image.

    Given a large satellite PIL image (e.g. 4096×4096) and a list of pixel-space
    bounding boxes ``[(minx, miny, maxx, maxy), ...]``, iterates over crops for
    batch inference.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Full satellite tile (RGB).
    bboxes_px : list of (minx, miny, maxx, maxy)
        Pixel-space bounding boxes of OSM parking regions within *pil_image*.
    img_size : int
        Resize each crop to this size before feeding to SegFormer.
    """

    def __init__(
        self,
        pil_image: Image.Image,
        bboxes_px: Sequence[Tuple[float, float, float, float]],
        img_size: int = IMG_SIZE,
    ) -> None:
        self.pil_image = pil_image.convert("RGB")
        self.bboxes_px = list(bboxes_px)
        self.img_size = img_size

        self._transforms = get_transforms("val", img_size)

    def __len__(self) -> int:
        return len(self.bboxes_px)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
            ``"pixel_values"`` : torch.Tensor[3, H, W]
            ``"bbox"``         : (minx, miny, maxx, maxy)
            ``"crop"``         : PIL.Image.Image  (original crop, for annotation)
        """
        import torch

        bbox = self.bboxes_px[idx]
        minx, miny, maxx, maxy = [int(v) for v in bbox]
        crop = self.pil_image.crop((minx, miny, maxx, maxy))

        if _ALBU_AVAILABLE and self._transforms is not None:
            img_np = np.array(crop)
            pixel_values = self._transforms(image=img_np)["image"]
        elif _TORCHVISION_AVAILABLE and self._transforms is not None:
            pixel_values = self._transforms(crop)
        else:
            import torchvision.transforms.functional as TF

            pixel_values = TF.to_tensor(crop.resize((self.img_size, self.img_size)))

        return {"pixel_values": pixel_values, "bbox": bbox, "crop": crop}

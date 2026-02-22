"""
SegFormer-b5 parking segmentation and spot counting.

Fine-tune ``nvidia/segformer-b5-finetuned-ade-640-640`` on ParkSeg12k to produce
pixel-level parking-stall masks, then derive integer spot counts via connected-
component analysis and area estimation.

Usage (inference)
-----------------
>>> from parksight.segment import ParkingSegmenter
>>> seg = ParkingSegmenter("checkpoints/segformer-b5-parkseg/best_model")
>>> result = seg.count_spots(pil_image)
>>> print(result["count"])

Usage (zero-shot / before fine-tuning)
---------------------------------------
>>> seg = ParkingSegmenter("nvidia/segformer-b5-finetuned-ade-640-640")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ADE20K class indices that correspond loosely to road / pavement surfaces.
# Used only when running the off-the-shelf ADE20K checkpoint (no fine-tuning).
# After fine-tuning on ParkSeg12k, class 1 == parking stall directly.
_ADE20K_PAVEMENT_CLASSES: set[int] = {
    6,   # road
    11,  # sidewalk/pavement
    52,  # path
    53,  # runway
    91,  # dirt track
}

# Connected-component area thresholds (in pixels at 512×512 canonical size).
# A typical parking stall covers ~1 000–8 000 px; tuned on zoom-19 Esri tiles.
_CC_MIN_AREA_PX = 400
_CC_MAX_AREA_PX = 12_000


@dataclass
class SegmentationResult:
    """Container for a single segmentation run."""

    mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    count: int = 0
    method: str = "none"
    # Breakdown for transparency
    area_count: int = 0
    cc_count: int = 0


class ParkingSegmenter:
    """
    SegFormer-b5 semantic segmentation → parking spot count.

    Supports two modes:

    * **Fine-tuned** (recommended): pass a local checkpoint trained on ParkSeg12k
      via ``train_segformer.py``.  Class 1 = parking stall.
    * **Zero-shot** (for quick demos): pass the HuggingFace hub ID of the ADE20K
      checkpoint; pavement-class pixels are used as a proxy.

    Parameters
    ----------
    model_id_or_path : str or Path
        HuggingFace model ID **or** path to a locally saved fine-tuned checkpoint.
    device : str or None
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-detected if *None*.
    is_finetuned : bool or None
        If *True*, class 1 = stall.  If *False*, ADE20K pavement proxy is used.
        If *None* (default), inferred from whether a ``config.json`` with
        ``num_labels <= 3`` exists at the path.
    stall_area_px : int
        Average parking stall area in pixels at the canonical tile resolution
        (512×512 at zoom 19).  Used for area-based counting fallback.
    avg_car_area_px : int
        Not used internally but stored for external calibration helpers.
    """

    def __init__(
        self,
        model_id_or_path: Union[str, Path] = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str | None = None,
        is_finetuned: bool | None = None,
        stall_area_px: int = 400,
        avg_car_area_px: int = 1_200,
    ) -> None:
        self.model_id_or_path = str(model_id_or_path)
        self.device = device or self._pick_device()
        self.stall_area_px = stall_area_px
        self.avg_car_area_px = avg_car_area_px

        # Infer fine-tuned flag
        if is_finetuned is None:
            cfg_path = Path(model_id_or_path) / "config.json"
            if cfg_path.exists():
                import json

                with cfg_path.open() as f:
                    cfg = json.load(f)
                n = cfg.get("num_labels", len(cfg.get("id2label", {})) or 999)
                is_finetuned = n <= 4
            else:
                is_finetuned = False  # assume HF hub zero-shot checkpoint
        self.is_finetuned = is_finetuned

        self._processor = None
        self._model = None

    # ── Device helpers ─────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ── Lazy model load ────────────────────────────────────────────

    def _load(self) -> None:
        """Load processor + model on first inference call."""
        if self._model is not None:
            return

        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        logger.info(
            "Loading SegFormer from %s on %s (fine-tuned=%s) …",
            self.model_id_or_path,
            self.device,
            self.is_finetuned,
        )

        model_path = Path(self.model_id_or_path)
        local_only = model_path.exists()

        processor_kwargs = {
            "do_resize": True,
            "size": {"height": 512, "width": 512},
            "do_rescale": True,
            "do_normalize": True,
        }
        if local_only:
            processor_kwargs["local_files_only"] = True

        try:
            self._processor = SegformerImageProcessor.from_pretrained(
                self.model_id_or_path,
                **processor_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Could not load SegFormer processor from %s: %s. "
                "Falling back to default SegFormer image processor settings.",
                self.model_id_or_path,
                exc,
            )
            self._processor = SegformerImageProcessor(
                do_resize=True,
                size={"height": 512, "width": 512},
                do_rescale=True,
                do_normalize=True,
                rescale_factor=1 / 255,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )

        model_kwargs = {}
        if local_only:
            model_kwargs["local_files_only"] = True

        try:
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_id_or_path,
                **model_kwargs,
            ).to(self.device)
        except Exception as exc:
            # If safetensors is broken but pytorch_model.bin exists, force .bin load.
            if model_path.exists() and (model_path / "pytorch_model.bin").exists():
                logger.warning(
                    "Failed to load safetensors for %s (%s); retrying with pytorch_model.bin.",
                    self.model_id_or_path,
                    exc,
                )
                retry_kwargs = dict(model_kwargs)
                retry_kwargs["use_safetensors"] = False
                self._model = SegformerForSemanticSegmentation.from_pretrained(
                    self.model_id_or_path,
                    **retry_kwargs,
                ).to(self.device)
            else:
                raise

        self._model.eval()

    # ── Core segmentation ──────────────────────────────────────────

    def segment(self, pil_image: Image.Image) -> np.ndarray:
        """
        Run SegFormer on a PIL RGB image and return a binary parking mask.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB satellite tile (any resolution — resized internally to 512×512).

        Returns
        -------
        numpy.ndarray
            Binary uint8 mask (H×W) where 1 = parking stall pixel, 0 = background.
            Output is at the **original** image resolution (upsampled from H/4×W/4).
        """
        import torch
        import torch.nn.functional as F

        self._load()

        pil_image = pil_image.convert("RGB")
        orig_w, orig_h = pil_image.size
        inputs = self._processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # logits: (1, num_classes, H/4, W/4)
        logits = outputs.logits  # type: ignore[attr-defined]

        # Upsample to original resolution
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        predicted_classes = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # H×W

        if self.is_finetuned:
            # Class 1 = parking stall (trained label)
            mask = (predicted_classes == 1).astype(np.uint8)
        else:
            # ADE20K proxy: pavement / road classes as surrogate
            mask = np.isin(predicted_classes, list(_ADE20K_PAVEMENT_CLASSES)).astype(
                np.uint8
            )

        return mask

    def segment_with_tta(self, pil_image: Image.Image) -> np.ndarray:
        """
        Run segmentation with test-time augmentation (4 orientations).

        Averages probability maps from original, horizontal flip,
        vertical flip, and both flips for more robust predictions.
        """
        import torch
        import torch.nn.functional as F

        self._load()

        pil_image = pil_image.convert("RGB")
        orig_w, orig_h = pil_image.size

        # generate 4 augmented versions
        augmented = [
            pil_image,
            pil_image.transpose(Image.FLIP_LEFT_RIGHT),
            pil_image.transpose(Image.FLIP_TOP_BOTTOM),
            pil_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
        ]

        prob_sum = None

        for i, aug_img in enumerate(augmented):
            inputs = self._processor(images=aug_img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits = outputs.logits
            upsampled = F.interpolate(
                logits, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )
            probs = torch.softmax(upsampled, dim=1)  # (1, C, H, W)

            # undo the flip on the probability map
            if i == 1:
                probs = torch.flip(probs, dims=[3])
            elif i == 2:
                probs = torch.flip(probs, dims=[2])
            elif i == 3:
                probs = torch.flip(probs, dims=[2, 3])

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum = prob_sum + probs

        avg_probs = prob_sum / 4.0
        predicted = avg_probs.argmax(dim=1).squeeze(0).cpu().numpy()

        if self.is_finetuned:
            mask = (predicted == 1).astype(np.uint8)
        else:
            mask = np.isin(predicted, list(_ADE20K_PAVEMENT_CLASSES)).astype(np.uint8)

        return mask

    @staticmethod
    def postprocess_mask(mask, min_blob_px=200):
        """
        Clean up a binary parking mask with morphological operations.

        Uses a small kernel (5×5, 1 iteration) to fill tiny holes without
        merging adjacent parking stalls into single blobs.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # close small gaps inside parking regions (gentle — 1 iteration)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # open to remove tiny noise blobs
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, small_kernel, iterations=1)

        # remove connected components smaller than min_blob_px
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_blob_px:
                cleaned[labels == i] = 0

        return cleaned

    # ── Counting ───────────────────────────────────────────────────

    def _count_from_mask(
        self,
        mask: np.ndarray,
        geo_scale_mpp: float | None = None,
    ) -> dict[str, int]:
        """
        Convert a binary parking mask to stall counts using two strategies.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary uint8 mask (H×W) as returned by :meth:`segment`.
        geo_scale_mpp : float or None
            Metres per pixel of the tile.  If provided, ``stall_area_px`` is
            recomputed from ``STALL_AREA_USA`` (≈ 13 m²) for the actual resolution
            rather than the canonical 512×512 assumption.

        Returns
        -------
        dict
            ``{"area_count": int, "cc_count": int}``
        """
        import json
        from pathlib import Path as _Path

        # --- Area-based count ---
        stall_area_px = self.stall_area_px
        if geo_scale_mpp is not None:
            try:
                _cfg_f = _Path(__file__).resolve().parent.parent / "config.json"
                with _cfg_f.open() as _f:
                    _cfg = json.load(_f)
                stall_m2 = _cfg.get("STALL_AREA_USA", 13.0)
            except Exception:
                stall_m2 = 13.0
            stall_area_px = max(1, int(stall_m2 / (geo_scale_mpp ** 2)))

        stall_pixels = int(mask.sum())
        area_count = max(0, round(stall_pixels / stall_area_px))

        # --- Connected-component count ---
        # Scale area thresholds to actual mask size (calibrated for 512px)
        h, w = mask.shape
        scale_factor = (h * w) / (512 * 512)
        min_area = int(_CC_MIN_AREA_PX * scale_factor)
        max_area = int(_CC_MAX_AREA_PX * scale_factor)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        cc_count = 0
        for i in range(1, num_labels):  # skip background label 0
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue  # noise — skip
            if area <= max_area:
                cc_count += 1  # single stall-sized blob
            else:
                # Large merged region — estimate stall count from its area
                cc_count += max(1, round(area / stall_area_px))

        return {"area_count": area_count, "cc_count": cc_count}

    def count_spots(
        self,
        pil_image: Image.Image,
        geo_scale_mpp: float | None = None,
        apply_postprocess: bool = True,
    ) -> SegmentationResult:
        """
        End-to-end: segment → count → return structured result.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB satellite tile.
        geo_scale_mpp : float or None
            Metres per pixel.  See :meth:`_count_from_mask`.
        apply_postprocess : bool
            If True (default), apply morphological cleanup before counting.

        Returns
        -------
        SegmentationResult
            Contains ``count`` (blended integer), ``mask``, ``method``,
            ``area_count``, and ``cc_count``.
        """
        mask = self.segment(pil_image)
        if apply_postprocess:
            mask = self.postprocess_mask(mask)
        counts = self._count_from_mask(mask, geo_scale_mpp=geo_scale_mpp)
        area_count = counts["area_count"]
        cc_count = counts["cc_count"]

        # Blending strategy:
        # 1. If cc_count > 0 and mask is reasonably dense, trust CC more.
        # 2. If mask is sparse (< 2% coverage), rely on area.
        # 3. Otherwise average the two.
        coverage = float(mask.sum()) / max(1, mask.size)
        if cc_count == 0 or coverage < 0.02:
            final_count = area_count
            method = "area"
        elif abs(area_count - cc_count) <= max(2, 0.2 * cc_count):
            # Both agree within 20%: average
            final_count = round((area_count + cc_count) / 2)
            method = "blend"
        else:
            # High disagreement: trust CC (more interpretable)
            final_count = cc_count
            method = "cc"

        return SegmentationResult(
            mask=mask,
            count=final_count,
            method=method,
            area_count=area_count,
            cc_count=cc_count,
        )

    def segment_to_contours(
        self,
        pil_image: Image.Image,
        min_area: int = 200,
    ) -> list:
        """Segment the image and return parking-region contours as pixel arrays.

        Runs the full segmentation pipeline and extracts contours from the
        resulting binary mask using OpenCV.  Contours smaller than *min_area*
        pixels are discarded as noise.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB satellite tile.
        min_area : int
            Minimum contour area in pixels to include (default 200).

        Returns
        -------
        list of numpy.ndarray
            Each array has shape ``(N, 2)`` with columns ``[x, y]`` in pixel
            coordinates (origin = top-left corner of the tile).
        """
        mask = self.segment(pil_image)

        # findContours expects uint8; mask from segment() is already uint8
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        result = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            # contour shape is (N, 1, 2) — squeeze to (N, 2) for convenience
            result.append(contour.reshape(-1, 2))

        return result

    # ── Visualisation ──────────────────────────────────────────────

    @staticmethod
    def annotate(
        pil_image: Image.Image,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 80),
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Overlay a coloured parking mask on the original RGB image.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            Original RGB satellite tile.
        mask : numpy.ndarray
            Binary uint8 mask from :meth:`segment` (same H×W as *pil_image*).
        color : (R, G, B)
            Highlight colour for stall pixels (default: vivid green).
        alpha : float
            Opacity of the overlay (0 = invisible, 1 = opaque).

        Returns
        -------
        PIL.Image.Image
            Annotated copy of *pil_image*.
        """
        base = np.array(pil_image.convert("RGB"), dtype=np.float32)

        overlay = np.zeros_like(base)
        for c_idx, c_val in enumerate(color):
            overlay[..., c_idx] = c_val

        blended = base.copy()
        stall_px = mask.astype(bool)
        blended[stall_px] = (
            (1 - alpha) * base[stall_px] + alpha * overlay[stall_px]
        )

        return Image.fromarray(blended.astype(np.uint8))

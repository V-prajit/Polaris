"""
imagery.py — Derive geometric parking parameters from SegFormer stall masks.

Given a parking lot polygon and a trained SegFormer checkpoint, this module
runs satellite tile inference and returns per-lot parameters that replace
the static defaults in ``count_geometric``:

- ``efficiency``       — stall pixel fraction (how much of the lot is usable)
- ``min_m2_per_stall`` — median stall blob size in m²
- ``stall_angle_deg``  — dominant stall orientation from PCA on blob axes

Usage
-----
>>> from parksight.imagery import infer_geometric_params
>>> params = infer_geometric_params(geom, "models/segformer_best")
>>> result = count_geometric(geom, osm_tags=tags, **params.as_overrides())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

logger = logging.getLogger(__name__)

# ----- tuneable constants ---------------------------------------------------
_EFFICIENCY_MIN = 0.25   # below this → lot is essentially empty / not parking
_EFFICIENCY_MAX = 0.95   # theoretical maximum (fully striped)
_MIN_BLOB_AREA_PX = 150  # ignore tiny blobs (noise / road markings)
_MAX_BLOB_AREA_PX = 15_000  # ignore huge blobs (whole-lot artefacts)
_M2_PER_PX_DEFAULT = 0.06  # fallback: zoom-19 tile at lat~33° → ~0.06 m²/px
# ---------------------------------------------------------------------------


@dataclass
class GeometricParams:
    """
    Per-lot geometric parameters derived from a SegFormer stall mask.

    All fields have physically sensible defaults so they can be used as
    overrides on top of ``count_geometric``'s base arguments.
    """
    efficiency: float = 0.60          # usable stall fraction of lot area
    min_m2_per_stall: float = 25.0    # gross area per stall (used as area cap)
    stall_angle_deg: float | None = None  # dominant angle; None = try all

    def as_overrides(self) -> dict:
        """Return kwargs suitable for passing to ``count_geometric``."""
        d: dict = {
            "efficiency": self.efficiency,
        }
        # min_m2_per_stall is handled inside _surface_lot_estimate; expose via
        # the dedicated kwarg if count_geometric is extended to accept it.
        # For now we embed it in the returned dict for the caller to use.
        d["_min_m2_per_stall"] = self.min_m2_per_stall
        return d


def _pixel_area_to_m2(geom, img_w: int, img_h: int) -> float:
    """
    Estimate m²/pixel for a satellite tile covering *geom*.

    Uses the fact that the tile bounding box in EPSG:3857 (metres) is mapped
    to img_w × img_h pixels.
    """
    try:
        minx, miny, maxx, maxy = geom.bounds
        bbox_area_m2 = (maxx - minx) * (maxy - miny)
        px_area_m2 = bbox_area_m2 / (img_w * img_h)
        return max(px_area_m2, 1e-6)
    except Exception:
        return _M2_PER_PX_DEFAULT


def _dominant_angle_from_blobs(labels: np.ndarray, n_labels: int) -> float | None:
    """
    Estimate dominant stall orientation via PCA on connected-component bounding
    boxes.  Returns the nearest standard angle (45, 60, 90) or None if
    unreliable.
    """
    try:
        import cv2
    except ImportError:
        return None

    angles = []
    for lbl in range(1, n_labels + 1):
        mask = (labels == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < _MIN_BLOB_AREA_PX:
            continue
        if len(cnt) < 5:
            continue
        _, (w_px, h_px), angle = cv2.minAreaRect(cnt)
        if w_px < 3 or h_px < 3:
            continue
        # cv2.minAreaRect returns angle in [-90, 0); normalise to [0, 90)
        angle = abs(angle) % 90
        angles.append(angle)

    if not angles:
        return None

    median_angle = float(np.median(angles))

    # Snap to nearest standard stall angle
    for std in [45.0, 60.0, 90.0]:
        if abs(median_angle - std) <= 15.0 or abs(median_angle - (90 - std)) <= 15.0:
            return std
    return 90.0  # default fallback


def infer_geometric_params(
    geom: Union[Polygon, MultiPolygon],
    model_path: Union[str, Path],
    padding_pct: float = 0.05,
    zoom: int = 19,
) -> GeometricParams:
    """
    Run SegFormer on the satellite tile for *geom* and derive geometric params.

    Parameters
    ----------
    geom : Polygon | MultiPolygon
        Parking lot in **EPSG:3857** (metres).
    model_path : str or Path
        Path to a saved SegFormer checkpoint directory (must contain
        ``config.json`` and ``pytorch_model.bin`` / ``model.safetensors``).
    padding_pct : float
        Extra padding around the bounding box when fetching the tile.
    zoom : int
        Slippy-map zoom level (19 = ~0.3 m/px at US latitudes).

    Returns
    -------
    GeometricParams
        Efficiency, min_m2_per_stall, and stall_angle_deg derived from the
        stall segmentation mask.
    """
    try:
        import cv2
        from parksight.fetch import get_satellite_tile
        from parksight.segment import ParkingSegmenter
    except ImportError as exc:
        logger.warning("infer_geometric_params: missing dependency %s — returning defaults.", exc)
        return GeometricParams()

    # ── 1. Fetch satellite tile ───────────────────────────────────────────
    try:
        pil_tile = get_satellite_tile(geom, padding_pct=padding_pct, zoom=zoom)
    except Exception as exc:
        logger.warning("Tile fetch failed for geom: %s — returning defaults.", exc)
        return GeometricParams()

    img_w, img_h = pil_tile.size

    # ── 2. Run SegFormer inference ────────────────────────────────────────
    try:
        segmenter = ParkingSegmenter(model_path)
        stall_mask = segmenter.segment(pil_tile)  # uint8 H×W, 1=stall 0=bg
    except Exception as exc:
        logger.warning("SegFormer inference failed: %s — returning defaults.", exc)
        return GeometricParams()

    # ── 3. Build lot polygon mask (clip to actual polygon boundary) ───────
    try:
        from PIL import Image, ImageDraw
        lot_mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(lot_mask)

        # Scale geom coords to pixel space
        minx, miny, maxx, maxy = geom.bounds
        pad_x = (maxx - minx) * padding_pct
        pad_y = (maxy - miny) * padding_pct
        ext_minx, ext_miny = minx - pad_x, miny - pad_y
        ext_maxx, ext_maxy = maxx + pad_x, maxy + pad_y
        scale_x = img_w / (ext_maxx - ext_minx)
        scale_y = img_h / (ext_maxy - ext_miny)

        def _to_px(x, y):
            return (
                (x - ext_minx) * scale_x,
                img_h - (y - ext_miny) * scale_y,  # flip Y
            )

        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            px_coords = [_to_px(x, y) for x, y in poly.exterior.coords]
            draw.polygon(px_coords, fill=255)
        lot_mask_np = np.array(lot_mask) > 0  # bool H×W
    except Exception:
        lot_mask_np = np.ones(stall_mask.shape, dtype=bool)

    # ── 4. Compute efficiency ─────────────────────────────────────────────
    # Resize stall mask to match lot_mask if shapes differ
    if stall_mask.shape != lot_mask_np.shape:
        stall_mask_resized = cv2.resize(
            stall_mask, (lot_mask_np.shape[1], lot_mask_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        stall_mask_resized = stall_mask

    lot_pixels = int(lot_mask_np.sum())
    if lot_pixels == 0:
        return GeometricParams()

    stall_pixels = int((stall_mask_resized.astype(bool) & lot_mask_np).sum())
    raw_efficiency = stall_pixels / lot_pixels
    efficiency = float(np.clip(raw_efficiency, _EFFICIENCY_MIN, _EFFICIENCY_MAX))

    logger.info(
        "infer_geometric_params: efficiency=%.2f (%d/%d stall/lot px)",
        efficiency, stall_pixels, lot_pixels,
    )

    # ── 5. Estimate m²/stall from connected components ───────────────────
    px_to_m2 = _pixel_area_to_m2(geom, img_w, img_h)
    min_m2_per_stall = 25.0  # default

    try:
        stall_clipped = (stall_mask_resized.astype(bool) & lot_mask_np).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            stall_clipped, connectivity=8
        )
        # stats shape: (n+1, 5); col 4 = area in pixels
        blob_areas_px = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
        valid = blob_areas_px[
            (blob_areas_px >= _MIN_BLOB_AREA_PX) & (blob_areas_px <= _MAX_BLOB_AREA_PX)
        ]
        if len(valid) >= 3:
            median_blob_px = float(np.median(valid))
            min_m2_per_stall = float(np.clip(median_blob_px * px_to_m2, 10.0, 60.0))
            logger.info(
                "infer_geometric_params: %d valid blobs, median=%.0f px → %.1f m²/stall",
                len(valid), median_blob_px, min_m2_per_stall,
            )
    except Exception as exc:
        logger.debug("CC analysis failed: %s", exc)
        n_labels, labels = 0, np.zeros_like(stall_mask_resized)

    # ── 6. Estimate stall angle ───────────────────────────────────────────
    stall_angle_deg = _dominant_angle_from_blobs(labels, n_labels)

    return GeometricParams(
        efficiency=efficiency,
        min_m2_per_stall=min_m2_per_stall,
        stall_angle_deg=stall_angle_deg,
    )



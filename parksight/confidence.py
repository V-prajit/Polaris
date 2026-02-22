"""
Confidence bands for parking estimates.

Wraps point estimates with realistic low/high bounds based on the
estimation method used.  Wider bands for heuristic methods, tighter
bands for direct ML detections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EstimateBand:
    """A point estimate with low/high confidence bounds."""
    value: int
    low: int
    high: int
    method: str

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "method": self.method,
        }


# Method → (lower_pct, upper_pct) relative margin
# e.g. 0.20 means ±20% → [value*0.80, value*1.20]
_MARGINS = {
    # ML-based (tighter)
    "yolo_detect":   (0.15, 0.15),   # YOLO direct box count
    "yolo_car":      (0.15, 0.20),   # YOLO car detection (satellite angle adds noise)
    "segformer":     (0.18, 0.18),   # SegFormer mask → count
    "blend":         (0.18, 0.18),   # SegFormer blended count
    "cc":            (0.15, 0.20),   # connected-component count
    # Heuristic (wider)
    "area":          (0.25, 0.30),   # area / stall_size formula
    "geometric":     (0.20, 0.25),   # geometric layout estimate
    "edge":          (0.25, 0.30),   # Canny + Hough edge count
    # Structure-based (can't see inside)
    "garage":        (0.25, 0.35),   # multi-storey estimate
    "underground":   (0.30, 0.40),   # underground estimate
    "osm_capacity":  (0.10, 0.10),   # OSM capacity tag (trusted)
    # Street
    "street":        (0.15, 0.25),   # curb length formula
    # Fallback
    "default":       (0.25, 0.30),
}


def confidence_band(
    value: int,
    method: str = "default",
    floor: int = 0,
) -> EstimateBand:
    """
    Wrap a point estimate with confidence bounds.

    Parameters
    ----------
    value : int
        Point estimate (e.g. 150 spots).
    method : str
        Estimation method key (see ``_MARGINS``).
    floor : int
        Minimum allowed low bound (default 0).

    Returns
    -------
    EstimateBand
    """
    lo_pct, hi_pct = _MARGINS.get(method, _MARGINS["default"])
    low = max(floor, math.floor(value * (1.0 - lo_pct)))
    high = math.ceil(value * (1.0 + hi_pct))
    return EstimateBand(value=value, low=low, high=high, method=method)


def utilization_band(
    cars: EstimateBand,
    capacity: EstimateBand,
) -> dict:
    """
    Compute utilization rate with propagated uncertainty.

    Returns dict with value, low, high as percentages (0–100+).
    """
    if capacity.value <= 0:
        return {"value": 0.0, "low": 0.0, "high": 0.0}

    util = cars.value / capacity.value
    # Worst case: most cars / fewest spots
    util_high = cars.high / max(capacity.low, 1)
    # Best case: fewest cars / most spots
    util_low = cars.low / max(capacity.high, 1)

    return {
        "value": round(min(util * 100, 100.0), 1),
        "low": round(min(util_low * 100, 100.0), 1),
        "high": round(min(util_high * 100, 150.0), 1),  # can exceed 100% (overflow)
    }

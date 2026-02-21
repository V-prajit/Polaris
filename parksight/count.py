"""
CV-based parking space counting.

Implements two estimation strategies:

* **Edge counting** — for Polygon geometries: fetch a satellite tile,
  detect edges with Canny + Hough, and derive a stall count.
* **Line counting** — for LineString / MultiLineString (street parking):
  compute real-world length and divide by average car length.

All tuneable parameters are loaded from ``config.json`` at package level.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt
import math
import json
from typing import Union
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from parksight import config, corrected_line_length
from parksight.fetch import get_satellite_tile

logger = logging.getLogger(__name__)


def get_line_count(geometry):
    """estimate street-parking capacity from a line geometry in epsg:3857"""
    length_m = corrected_line_length(geometry)
    car_count = int(round(length_m / config["AVG_CAR_LENGTH"]))
    return max(car_count, 0)


def count_edges(geom, padding_pct=0.10, zoom=19):
    """
    Count parking stalls for a single geometry using the CV baseline.

    Pipeline (for Polygons):
    1. Fetch satellite tile covering the geometry.
    2. Convert to grayscale, gaussian blur.
    3. Canny edge detection, hough line detection.
    4. If fewer than 50 lines, fall back to area-based formula.
       Otherwise estimate N = lines / 2.
    """
    logger.info("Processing geometry: %s", geom.geom_type)

    if geom.geom_type == "Point":
        return 0

    if geom.geom_type in ("LineString", "MultiLineString"):
        return get_line_count(geom)

    # polygon / multipolygon path
    pil_img = get_satellite_tile(geom, padding_pct=padding_pct, zoom=zoom)
    img_array = np.array(pil_img)

    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(
        gray,
        tuple(config["CV2"]["gaussian_blur"]["ksize"]),
        config["CV2"]["gaussian_blur"]["sigma"],
    )

    edges = cv2.Canny(
        gray_blur,
        config["CV2"]["canny"]["threshold1"],
        config["CV2"]["canny"]["threshold2"],
        apertureSize=config["CV2"]["canny"]["aperture_size"],
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=config["CV2"]["hough_lines_p"]["rho"],
        theta=config["CV2"]["hough_lines_p"]["theta"],
        threshold=config["CV2"]["hough_lines_p"]["threshold"],
        minLineLength=config["CV2"]["hough_lines_p"]["min_line_length"],
        maxLineGap=config["CV2"]["hough_lines_p"]["max_line_gap"],
    )

    area_estimate = int(
        (geom.area / config["STALL_AREA_USA"]) * (1 - config["mu"])
    )

    if lines is not None:
        n_lines = len(lines)
        return area_estimate if n_lines < 50 else int(n_lines / 2)

    return area_estimate


def visualize_pipeline(geom, padding_pct=0.10, zoom=19):
    """
    Run the CV pipeline and return intermediate images for educational display.

    Returns a dict with keys: "satellite", "gray", "blur", "edges", "lines".
    """
    pil_img = get_satellite_tile(geom, padding_pct=padding_pct, zoom=zoom)
    img_array = np.array(pil_img)

    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(
        gray,
        tuple(config["CV2"]["gaussian_blur"]["ksize"]),
        config["CV2"]["gaussian_blur"]["sigma"],
    )
    edges = cv2.Canny(
        gray_blur,
        config["CV2"]["canny"]["threshold1"],
        config["CV2"]["canny"]["threshold2"],
        apertureSize=config["CV2"]["canny"]["aperture_size"],
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=config["CV2"]["hough_lines_p"]["rho"],
        theta=config["CV2"]["hough_lines_p"]["theta"],
        threshold=config["CV2"]["hough_lines_p"]["threshold"],
        minLineLength=config["CV2"]["hough_lines_p"]["min_line_length"],
        maxLineGap=config["CV2"]["hough_lines_p"]["max_line_gap"],
    )

    line_img = img_array.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return {
        "satellite": img_array,
        "gray": gray,
        "blur": gray_blur,
        "edges": edges,
        "lines": line_img,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometric / design-standards estimator
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class GeometricResult:
    """Result of the geometric parking-design estimator."""
    count: int            # estimated number of stalls
    best_angle_deg: float # stall angle that maximised count (90 / 60 / 45 / 0)
    layout: str           # "double-loaded", "single-loaded", "parallel", "irregular"
    lot_area_m2: float    # polygon area in m²
    gross_per_stall_m2: float  # implied gross area per stall
    solidity: float = 1.0 # polygon area / convex hull area (compactness measure)


def count_geometric(
    geom: Union[Polygon, "MultiPolygon"],
    osm_tags: dict | None = None,
    stall_w: float = 2.6,
    stall_d: float = 5.5,
    aisle_w: float = 7.3,
    efficiency: float = 0.85,
) -> GeometricResult:
    """
    Estimate parking spots by reverse-engineering standard lot design (ITE/NPA).

    Algorithm
    ---------
    1. Compute the lot's **minimum oriented bounding rectangle** (tightest OBB,
       not axis-aligned), giving ``short_side`` and ``long_side`` in metres.
    2. For each candidate stall angle (90°, 60°, 45°):
       - Double-loaded module width = 2 × stall_depth + aisle_width = 18.3 m (90°)
       - Modules across short side → 2 rows per module (+ possible edge row)
       - Stalls per row along long side = long_side / stall_pitch
       - raw_count = n_rows × stalls_per_row
    3. Pick the angle that maximises raw_count.
    4. Apply ``efficiency`` factor (dead corners, disabled bays, cart corrals).
    5. Respect ``parking:orientation`` OSM tag if present.

    Parameters
    ----------
    geom : Polygon | MultiPolygon
        Parking lot in **EPSG:3857** (metres).
    osm_tags : dict or None
        Raw OSM feature tags.
    stall_w : float
        Stall width [m] — US standard 2.6 m (8.5 ft).
    stall_d : float
        Stall depth [m] — US standard 5.5 m (18 ft).
    aisle_w : float
        Two-way drive-aisle width [m] — 7.3 m (24 ft) for 90°.
    efficiency : float
        Fraction of theoretical capacity that is usable (default 0.85).

    Returns
    -------
    GeometricResult
    """
    from shapely.ops import unary_union as _uu

    tags = osm_tags or {}

    # ── merge multi-polygons ──────────────────────────────────────────────────
    if geom.geom_type == "MultiPolygon":
        geom = _uu(geom)

    lot_area = geom.area  # m²

    # ── Solidity check: filter out campus-zone / non-parking polygons ─────────
    # Solidity = polygon area / convex hull area.
    # Real parking lots are compact (0.7–1.0).
    # Jagged campus zones wrapping buildings score 0.2–0.4.
    # We scale effective area by solidity so irregular polygons get lower counts.
    try:
        convex_area = geom.convex_hull.area
        solidity = lot_area / convex_area if convex_area > 0 else 1.0
    except Exception:
        solidity = 1.0

    # Very non-compact: almost certainly not a real parking lot
    if solidity < 0.25:
        logger.warning(
            "Polygon solidity=%.2f < 0.25 — likely a campus zone, not a parking lot. "
            "Returning 0.", solidity
        )
        return GeometricResult(
            count=0, best_angle_deg=0.0, layout="irregular",
            lot_area_m2=lot_area, gross_per_stall_m2=float("inf"),
            solidity=solidity,
        )

    # Scale the effective area by solidity (irregular areas contain less usable pavement)
    effective_area = lot_area * solidity


    # ── read OSM orientation hint ─────────────────────────────────────────────
    orientation = str(
        tags.get("parking:orientation")
        or tags.get("orientation")
        or tags.get("parking")
        or ""
    ).lower()

    # Parallel parking: car fits lengthwise, one row per ~car-width strip
    if orientation in ("parallel", "street_side", "on_street"):
        gross = stall_w * stall_d   # 2.6 × 5.5 ≈ 14.3 m²
        count = max(0, int((lot_area / gross) * efficiency))
        return GeometricResult(
            count=count, best_angle_deg=0.0, layout="parallel",
            lot_area_m2=lot_area, gross_per_stall_m2=gross,
        )

    # Limit angle search if OSM gives a hint
    if orientation in ("diagonal", "angled"):
        candidate_angles = [60.0, 45.0]
    else:
        candidate_angles = [90.0, 60.0, 45.0]

    # ── minimum oriented bounding rectangle ──────────────────────────────────
    mbr = geom.minimum_rotated_rectangle
    coords = list(mbr.exterior.coords)[:4]
    sides = sorted(
        np.linalg.norm(np.array(coords[i + 1]) - np.array(coords[i]))
        for i in range(3)
    )
    mbr_short, mbr_long = sides[0], sides[1]

    best_count = 0
    best_angle = 90.0
    best_gross = lot_area  # fallback: 1 stall

    for angle in candidate_angles:
        rad = math.radians(angle)

        # Stall pitch along the row (widens for angled stalls)
        eff_pitch = stall_w if angle == 90.0 else stall_w / math.sin(rad)

        # Double-loaded module depth across the short axis
        module_w = 2.0 * stall_d + aisle_w          # e.g. 18.3 m at 90°

        n_modules = int(mbr_short / module_w)
        n_rows = n_modules * 2

        # One extra single-loaded row if space allows
        remaining = mbr_short - n_modules * module_w
        if remaining >= stall_d + aisle_w / 2:
            n_rows += 1

        stalls_per_row = max(0, int(mbr_long / eff_pitch))
        raw = n_rows * stalls_per_row
        gross = lot_area / raw if raw > 0 else float("inf")

        if raw > best_count:
            best_count = raw
            best_angle = angle
            best_gross = gross

    # ── Cap by effective polygon area (solidity-adjusted) ────────────────────
    # OBB can be much larger than the actual polygon for irregular lots.
    # Physical upper bound: 1 stall per (stall_w × stall_d) = 14.3 m² minimum.
    # We use effective_area (= lot_area × solidity) so jagged campus zones
    # can't inflate the count beyond what their usable footprint supports.
    min_m2_per_stall = stall_w * stall_d  # 14.3 m²
    area_cap = int((effective_area / min_m2_per_stall) * efficiency)
    final = min(max(0, int(best_count * efficiency)), area_cap)

    return GeometricResult(
        count=final,
        best_angle_deg=best_angle,
        layout="double-loaded",
        lot_area_m2=lot_area,
        gross_per_stall_m2=best_gross,
        solidity=solidity,
    )


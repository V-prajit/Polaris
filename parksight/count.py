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
# Geometric / design-standards estimator (v2)
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class GeometricResult:
    """Result of the geometric parking-design estimator."""
    count: int                     # estimated number of stalls
    best_angle_deg: float          # stall angle that maximised count
    layout: str                    # "double-loaded", "single-loaded", "parallel", "irregular"
    lot_area_m2: float             # polygon area in m²
    gross_per_stall_m2: float      # implied gross area per stall
    solidity: float = 1.0          # polygon area / convex hull area
    parking_type: str = "surface"  # "surface", "multi_storey", "underground", "rooftop"
    levels: int = 1                # floors used in estimate
    osm_capacity: int | None = None  # raw OSM capacity tag if present


# ── Internal helpers ─────────────────────────────────────────────────────────

def _detect_parking_type(tags: dict) -> str:
    """Classify parking structure from OSM tags."""
    parking = str(tags.get("parking", "")).lower().replace("-", "_")
    building = str(tags.get("building", "")).lower()

    if parking in ("multi_storey",) or building in ("parking", "garage", "garages"):
        return "multi_storey"
    if parking == "underground":
        return "underground"
    if parking in ("rooftop", "roof"):
        return "rooftop"
    return "surface"


def _read_levels(tags: dict, parking_type: str) -> int:
    """Read level count from OSM tags, falling back to config defaults."""
    # Try explicit level tags
    for key in ("parking:levels", "building:levels", "levels",
                "building:levels:underground"):
        val = tags.get(key)
        if val is not None:
            try:
                n = int(float(val))
                if n > 0:
                    return n
            except (ValueError, TypeError):
                continue

    # Defaults from config
    if parking_type == "underground":
        return config.get("DEFAULT_UNDERGROUND_LEVELS", 2)
    if parking_type == "multi_storey":
        return config.get("DEFAULT_GARAGE_LEVELS", 3)
    return 1  # surface / rooftop


def _read_osm_capacity(tags: dict) -> int | None:
    """Parse the OSM capacity=* tag if present."""
    val = tags.get("capacity")
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _subtract_interior_rings(geom) -> float:
    """Return usable area: exterior minus interior ring (hole) areas."""
    if geom.geom_type == "Polygon":
        return geom.area  # shapely already subtracts holes
    # MultiPolygon: sum of all polygon areas (holes already subtracted)
    return sum(p.area for p in geom.geoms)


def _surface_lot_estimate(
    geom,
    tags: dict,
    stall_w: float,
    stall_d: float,
    aisle_w_90: float,
    efficiency: float,
    solidity: float,
    min_m2_override: float | None = None,
    forced_angle: float | None = None,
) -> GeometricResult:
    """
    OBB-based surface lot estimator (ITE/NPA design standards).

    Improvements over v1:
    - Angle-specific aisle widths (90°=7.3m, 60°=5.5m, 45°=4.0m)
    - Aspect-ratio detection: very narrow lots → parallel/single-loaded
    - Interior rings (holes) subtracted from usable area
    - Tiny lots (<50 m²) and huge lots (>50,000 m²) handled specially
    """
    lot_area = _subtract_interior_rings(geom)
    effective_area = lot_area * solidity

    # ── Tiny lot guard ────────────────────────────────────────────────────
    if lot_area < 50.0:
        return GeometricResult(
            count=0, best_angle_deg=0.0, layout="too_small",
            lot_area_m2=lot_area, gross_per_stall_m2=float("inf"),
            solidity=solidity, parking_type="surface", levels=1,
        )

    # ── Huge lot: likely a mapped zone — reduce efficiency ────────────────
    if lot_area > 50_000:
        efficiency = min(efficiency, 0.70)

    # ── Read OSM orientation hint ─────────────────────────────────────────
    orientation = str(
        tags.get("parking:orientation")
        or tags.get("orientation")
        or tags.get("parking")
        or ""
    ).lower()

    # ── Minimum oriented bounding rectangle ──────────────────────────────
    mbr = geom.minimum_rotated_rectangle
    coords = list(mbr.exterior.coords)[:4]
    sides = sorted(
        np.linalg.norm(np.array(coords[i + 1]) - np.array(coords[i]))
        for i in range(3)
    )
    mbr_short, mbr_long = sides[0], sides[1]
    aspect = mbr_long / mbr_short if mbr_short > 0 else float("inf")

    # ── Narrow lots or explicit parallel → parallel layout ───────────────
    if orientation in ("parallel", "street_side", "on_street") or aspect > 8.0:
        gross = stall_w * stall_d
        count = max(0, int((effective_area / gross) * efficiency))
        return GeometricResult(
            count=count, best_angle_deg=0.0, layout="parallel",
            lot_area_m2=lot_area, gross_per_stall_m2=gross,
            solidity=solidity, parking_type="surface", levels=1,
        )

    # ── Angle-specific aisle widths (ITE standards) ──────────────────────
    aisle_widths = {
        90.0: config.get("AISLE_WIDTH_90", 7.3),
        60.0: config.get("AISLE_WIDTH_60", 5.5),
        45.0: config.get("AISLE_WIDTH_45", 4.0),
    }

    if orientation in ("diagonal", "angled"):
        candidate_angles = [60.0, 45.0]
    elif forced_angle is not None:
        candidate_angles = [forced_angle]
    else:
        candidate_angles = [90.0, 60.0, 45.0]

    best_count = 0
    best_angle = 90.0
    best_gross = lot_area
    best_layout = "double-loaded"

    for angle in candidate_angles:
        rad = math.radians(angle)
        aisle_w = aisle_widths.get(angle, aisle_w_90)

        # Stall pitch along the row
        eff_pitch = stall_w if angle == 90.0 else stall_w / math.sin(rad)

        # Double-loaded module depth
        module_w = 2.0 * stall_d + aisle_w

        n_modules = int(mbr_short / module_w)
        n_rows = n_modules * 2
        layout = "double-loaded"

        # Extra single-loaded row in remaining space
        remaining = mbr_short - n_modules * module_w
        if remaining >= stall_d + aisle_w / 2:
            n_rows += 1
            layout = "double-loaded+edge"

        # Very narrow: can only fit single-loaded
        if n_modules == 0 and mbr_short >= stall_d + aisle_w / 2:
            n_rows = 1
            layout = "single-loaded"

        stalls_per_row = max(0, int(mbr_long / eff_pitch))
        raw = n_rows * stalls_per_row
        gross = lot_area / raw if raw > 0 else float("inf")

        if raw > best_count:
            best_count = raw
            best_angle = angle
            best_gross = gross
            best_layout = layout

    # ── Cap by effective area ────────────────────────────────────────────
    # Use ITE gross area per stall (25 m²) as a more realistic floor.
    # stall_w × stall_d = 14.3 m² is the theoretical minimum (no aisle share);
    # real lots use 25–30 m² gross including circulation.
    # min_m2_override allows imagery.py to supply a per-lot measured value.
    if min_m2_override is not None:
        min_m2_per_stall = float(np.clip(min_m2_override, 10.0, 60.0))
    else:
        min_m2_per_stall = max(stall_w * stall_d, 30.0)
    area_cap = int((effective_area / min_m2_per_stall) * efficiency)
    final = min(max(0, int(best_count * efficiency)), area_cap)

    return GeometricResult(
        count=final,
        best_angle_deg=best_angle,
        layout=best_layout,
        lot_area_m2=lot_area,
        gross_per_stall_m2=best_gross,
        solidity=solidity,
        parking_type="surface",
        levels=1,
    )


def count_geometric(
    geom: Union[Polygon, "MultiPolygon"],
    osm_tags: dict | None = None,
    stall_w: float = 2.6,
    stall_d: float = 5.5,
    aisle_w: float = 7.3,
    efficiency: float = 0.60,
    model_path: str | None = None,
    use_heuristics: bool = False,
) -> GeometricResult:
    """
    Estimate parking spots from polygon geometry + OSM tags.

    Handles:
    - Surface lots (OBB layout simulation with angle-specific aisles)
    - Multi-storey garages (floor area × levels × garage efficiency)
    - Underground parking (floor area × levels × garage efficiency)
    - Rooftop parking (single floor, surface efficiency)
    - OSM capacity tag blending (60% OSM / 40% geometric)
    - Edge cases: tiny lots, huge lots, narrow lots, interior rings
    - Dynamic parameter inference from SegFormer satellite tile (if model_path set)

    Parameters
    ----------
    geom : Polygon | MultiPolygon
        Parking lot in **EPSG:3857** (metres).
    osm_tags : dict or None
        Raw OSM feature tags.
    stall_w, stall_d, aisle_w, efficiency : float
        Design parameters (US defaults).
    model_path : str or None
        Path to a fine-tuned SegFormer checkpoint.  When provided, runs
        ``infer_geometric_params()`` on the satellite tile to obtain
        per-lot efficiency and stall size instead of the static defaults.
    use_heuristics : bool
        If True, infer parameters (efficiency, stall size, angle) based on 
        the geometry's shape and OSM tags rather than using SegFormer or defaults.

    Returns
    -------
    GeometricResult
    """
    from shapely.ops import unary_union as _uu

    tags = osm_tags or {}

    # ── Merge MultiPolygon ───────────────────────────────────────────────
    if geom.geom_type == "MultiPolygon":
        geom = _uu(geom)

    lot_area = _subtract_interior_rings(geom)

    # ── Solidity ─────────────────────────────────────────────────────────
    try:
        convex_area = geom.convex_hull.area
        solidity = lot_area / convex_area if convex_area > 0 else 1.0
    except Exception:
        solidity = 1.0

    if solidity < 0.25:
        logger.warning(
            "Polygon solidity=%.2f < 0.25 — likely a campus zone. Returning 0.",
            solidity,
        )
        return GeometricResult(
            count=0, best_angle_deg=0.0, layout="irregular",
            lot_area_m2=lot_area, gross_per_stall_m2=float("inf"),
            solidity=solidity, parking_type="surface",
        )

    # ── Detect parking type ──────────────────────────────────────────────
    parking_type = _detect_parking_type(tags)
    levels = _read_levels(tags, parking_type)
    osm_cap = _read_osm_capacity(tags)

    # ── Dynamic parameter inference ──────────────────────────────────────
    min_m2_override = None
    if use_heuristics:
        eff_est, m2_est, angle_est = _infer_hyperparams_heuristic(geom, tags)
        efficiency = eff_est
        min_m2_override = m2_est
        forced_angle = angle_est
        print(f"    [geom] Heuristic params: eff={efficiency:.2f}, m2/stall={min_m2_override:.1f}, angle={forced_angle}")
    elif model_path is not None:
        try:
            from parksight.imagery import infer_geometric_params
            params = infer_geometric_params(geom, model_path)
            efficiency = params.efficiency
            min_m2_override = params.min_m2_per_stall
            forced_angle = params.stall_angle_deg
            print(f"    [geom] Dynamic params: eff={efficiency:.2f}, m2/stall={min_m2_override:.1f}, angle={forced_angle}")
        except Exception as e:
            logger.warning("Dynamic parameter inference failed: %s", e)
            forced_angle = None
    else:
        forced_angle = None

    # ── Structured parking (garage / underground) ────────────────────────
    if parking_type in ("multi_storey", "underground"):
        stall_area = config.get("STALL_AREA_M2", 15.5)
        usable_frac = config.get("USABLE_FRACTION_GARAGE", 0.60)

        spots_per_floor = int(lot_area * usable_frac / stall_area)
        estimate = spots_per_floor * levels

        # Blend with OSM capacity if available
        if osm_cap is not None and osm_cap > 0:
            estimate = int(0.6 * osm_cap + 0.4 * estimate)

        gross = lot_area / max(spots_per_floor, 1)

        return GeometricResult(
            count=max(estimate, 0),
            best_angle_deg=90.0,
            layout="structured",
            lot_area_m2=lot_area,
            gross_per_stall_m2=gross,
            solidity=solidity,
            parking_type=parking_type,
            levels=levels,
            osm_capacity=osm_cap,
        )

    # ── Rooftop parking (single floor, surface efficiency) ───────────────
    if parking_type == "rooftop":
        stall_area = config.get("STALL_AREA_M2", 15.5)
        usable_frac = config.get("USABLE_FRACTION_SURFACE", 0.50)
        estimate = int(lot_area * usable_frac / stall_area)

        if osm_cap is not None and osm_cap > 0:
            estimate = int(0.6 * osm_cap + 0.4 * estimate)

        return GeometricResult(
            count=max(estimate, 0),
            best_angle_deg=90.0,
            layout="rooftop",
            lot_area_m2=lot_area,
            gross_per_stall_m2=lot_area / max(estimate, 1),
            solidity=solidity,
            parking_type="rooftop",
            levels=1,
            osm_capacity=osm_cap,
        )

    # ── Surface lot (OBB-based layout simulation) ────────────────────────
    result = _surface_lot_estimate(
        geom, tags, stall_w, stall_d, aisle_w, efficiency, solidity,
        min_m2_override=min_m2_override, forced_angle=forced_angle,
    )

    # Blend with OSM capacity if available
    if osm_cap is not None and osm_cap > 0:
        blended = int(0.6 * osm_cap + 0.4 * result.count)
        result = GeometricResult(
            count=blended,
            best_angle_deg=result.best_angle_deg,
            layout=result.layout,
            lot_area_m2=result.lot_area_m2,
            gross_per_stall_m2=result.gross_per_stall_m2,
            solidity=result.solidity,
            parking_type="surface",
            levels=1,
            osm_capacity=osm_cap,
        )

    return result


def _infer_hyperparams_heuristic(geom: Union[Polygon, "MultiPolygon"], tags: dict) -> tuple[float, float, float | None]:
    """
    Infers layout hyperparameters based on simple geometric and OSM metadata rules.
    Returns: (efficiency, min_m2_per_stall, stall_angle_deg)
    """
    from parksight.features import extract_geom_features
    feats = extract_geom_features(geom)
    area = feats["area_m2"]
    
    # 1. Efficiency
    # Base efficiency on how regular the shape is. Re-entrant shapes (L-shapes) waste space.
    # Lots that fill their bounding box well can be packed more tightly.
    base_eff = 0.65
    if area < 500:
        base_eff = 0.80
    elif area > 10000:
        base_eff = 0.55
        
    # Penalize efficiency for highly irregular shapes
    fill_ratio = feats.get("obb_fill_ratio", 1.0)
    convex_ratio = feats.get("convex_hull_ratio", 1.0)
    
    # A perfect rectangle has fill_ratio=1.0. A weird lot might have 0.5.
    # We blend this down slightly.
    eff = base_eff * ((fill_ratio + convex_ratio) / 2.0)
        
    # 2. Min m2 per stall
    # Garages pack spots tighter, surface lots often have wider default painted lines/aisles
    parking_type = _detect_parking_type(tags)
    if parking_type in ("multi-storey", "underground"):
        min_m2 = 25.0
    else:
        min_m2 = 30.0
        
    # --- STATIC CACHE INTEGRATION (DISABLED) ---
    # Zoning height heuristics from Atlanta_Zoning_Districts.geojson were degrading
    # stall estimates. The efficiency_modifier was fetched but never applied.
    # Disabled for Hacklytics 2026 demo.
    # ---------------------------------

    eff = max(0.40, min(0.95, eff))

    # 3. Angle
    # If the lot is extremely long and narrow, assume 0 or 90 depending on 
    # orientation, otherwise allow the optimizer to run free
    forced_angle = None
    try:
        from shapely import minimum_rotated_rectangle # type: ignore
        mbr = minimum_rotated_rectangle(geom)
        coords = list(mbr.exterior.coords)
        if len(coords) >= 4:
            e1_len = ((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)**0.5
            e2_len = ((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)**0.5
            w, h = min(e1_len, e2_len), max(e1_len, e2_len)
            
            # Very skewed aspect ratio suggests street or angled thin lot
            # We already have obb_aspect_ratio from features, which is L/W (always >= 1)
            aspect_ratio = feats.get("obb_aspect_ratio", 1.0)
            if aspect_ratio > 4.0:
                # Find the bearing of the long edge
                import math
                if e1_len > e2_len:
                    dx, dy = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
                else:
                    dx, dy = coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]
                angle_rad = math.atan2(dy, dx)
                forced_angle = (math.degrees(angle_rad) % 90.0)
    except Exception:
        pass
        
    return eff, min_m2, forced_angle

"""
features.py — Extract physical tabular features from OSM polygons.

These features are used by the lightweight ML predictor (Random Forest/XGBoost)
to estimate geometric parameters (efficiency and m² per stall) extremely
fast at inference time, avoiding heavy Computer Vision processing.
"""

from __future__ import annotations

import math
from typing import Union

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def extract_geom_features(geom: Union[Polygon, MultiPolygon]) -> dict[str, float]:
    """
    Compute intrinsic physical characteristics of a parking lot polygon.
    Input `geom` must be projected (e.g., EPSG:3857 in metres) for
    meaningful scale features.

    Returns a dictionary of float features suitable for tabular ML models.
    """
    if geom.is_empty:
        return {
            "area_m2": 0.0,
            "perimeter_m": 0.0,
            "p2a_ratio": 0.0,
            "convex_hull_ratio": 1.0,
            "num_vertices": 0.0,
            "is_multipolygon": 0.0,
            "num_rings": 0.0,
            "obb_aspect_ratio": 1.0,
            "obb_fill_ratio": 0.0,
        }

    # If MultiPolygon, treat as a single shape for most features
    poly = unary_union(geom) if geom.geom_type == "MultiPolygon" else geom

    # Base traits
    area_m2 = float(poly.area)
    perimeter_m = float(poly.length)

    # Complexity: how "stringy" or "amoeba-like" is the lot?
    # A perfect circle has minimum perimeter^2 / area (4π ≈ 12.57).
    # Higher ratios mean long, narrow, or highly irregular lots.
    p2a_ratio = (perimeter_m ** 2) / max(area_m2, 1e-6)

    # L/U-shape penalty (Concavity): how much of the convex hull does it fill?
    # Lots with deep cutouts (e.g. wrapping around a building) fill less of their hull.
    hull = poly.convex_hull
    convex_hull_ratio = area_m2 / max(hull.area, 1e-6)

    # Vertex count & rings (measures how complex/detailed the mapping is)
    if poly.geom_type == "Polygon":
        num_vertices = len(poly.exterior.coords) - 1
        num_rings = len(poly.interiors)
        is_multi = 0.0
    else:  # MultiPolygon / GeometryCollection
        num_vertices = sum(len(p.exterior.coords) - 1 for p in poly.geoms if hasattr(p, "exterior"))
        num_rings = sum(len(p.interiors) for p in poly.geoms if hasattr(p, "interiors"))
        is_multi = 1.0

    # Minimum Oriented Bounding Box (OBB)
    obb = poly.minimum_rotated_rectangle
    if obb.geom_type == "Polygon":
        # Get side lengths of OBB
        coords = list(obb.exterior.coords)
        if len(coords) >= 4:
            d1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
            d2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
            length = max(d1, d2)
            width = min(d1, d2)
            obb_aspect_ratio = length / max(width, 1e-6)
        else:
            obb_aspect_ratio = 1.0
    else: # e.g. a line
        obb_aspect_ratio = 100.0

    obb_fill_ratio = area_m2 / max(obb.area, 1e-6)

    return {
        "area_m2": area_m2,
        "perimeter_m": perimeter_m,
        "p2a_ratio": p2a_ratio,
        "convex_hull_ratio": convex_hull_ratio,
        "num_vertices": float(num_vertices),
        "is_multipolygon": is_multi,
        "num_rings": float(num_rings),
        "obb_aspect_ratio": obb_aspect_ratio,
        "obb_fill_ratio": obb_fill_ratio,
    }

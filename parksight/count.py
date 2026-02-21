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

import json
import logging
import math
from pathlib import Path
from typing import Union

import contextily as ctx
import cv2
import geopandas as gpd
import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString, MultiLineString, Point, Polygon

logger = logging.getLogger(__name__)

# ── Load config from package-relative path ─────────────────────────

_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

with _CONFIG_FILE.open(encoding="utf-8") as _f:
    config: dict = json.load(_f)


def get_line_count(geometry: Union[LineString, MultiLineString]) -> int:
    """
    Estimate street-parking capacity from a line geometry.

    Converts the geometry to WGS-84 to obtain a latitude-corrected length
    in metres, then divides by the average car length.

    Parameters
    ----------
    geometry : LineString | MultiLineString
        Street-parking geometry in EPSG:3857.

    Returns
    -------
    int
        Estimated number of cars that fit along the kerb.
    """
    geo_wgs = gpd.GeoSeries([geometry], crs=3857).to_crs(4326).geometry[0]

    lats: list[float] = []
    if geo_wgs.geom_type == "MultiLineString":
        for line in geo_wgs.geoms:
            lats.extend(y for _, y in line.coords)
    else:
        lats.extend(y for _, y in geo_wgs.coords)

    mean_lat = float(np.mean(lats))
    length_m = float(geometry.length)
    corrected = length_m * math.cos(math.radians(mean_lat))
    car_count = int(round(corrected / config["AVG_CAR_LENGTH"]))
    return max(car_count, 0)


def count_edges(
    geom: Union[Polygon, LineString, Point, MultiLineString],
    padding_pct: float = 0.10,
    zoom: int = 19,
) -> int:
    """
    Count parking stalls for a single geometry using the CV baseline.

    Pipeline (for Polygons):

    1. Fetch an Esri satellite tile covering the geometry.
    2. Convert to grayscale → Gaussian blur.
    3. Canny edge detection → Hough line detection.
    4. If fewer than 50 lines detected, fall back to an area-based formula:
       ``N = floor(area / stall_area * (1 - mu))``.
       Otherwise estimate ``N = lines / 2``.

    Parameters
    ----------
    geom : shapely geometry
        Feature geometry in EPSG:3857.
    padding_pct : float
        Fraction of bbox to add as padding (default 0.10).
    zoom : int
        Tile zoom level (default 19).

    Returns
    -------
    int
        Estimated number of parking stalls.
    """
    logger.info("Processing geometry: %s", geom.geom_type)

    if geom.geom_type == "Point":
        return 0

    if geom.geom_type in ("LineString", "MultiLineString"):
        return get_line_count(geom)

    # Polygon / MultiPolygon path
    minx, miny, maxx, maxy = geom.bounds

    width = maxx - minx
    height = maxy - miny
    x_pad = width * padding_pct
    y_pad = height * padding_pct

    img_array, _ = ctx.bounds2img(
        minx - x_pad,
        miny - y_pad,
        maxx + x_pad,
        maxy + y_pad,
        zoom=zoom,
        ll=False,
        source=ctx.providers.Esri.WorldImagery,
    )

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

    lines: npt.NDArray[np.int32] | None = cv2.HoughLinesP(
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


def visualize_pipeline(
    geom: Polygon,
    padding_pct: float = 0.10,
    zoom: int = 19,
) -> dict[str, np.ndarray]:
    """
    Run the CV pipeline and return intermediate images for educational display.

    Returns a dict with keys: ``"satellite"``, ``"gray"``, ``"blur"``,
    ``"edges"``, ``"lines"``.

    Parameters
    ----------
    geom : Polygon
        Feature geometry in EPSG:3857.
    padding_pct : float
        Fraction of bbox to add as padding.
    zoom : int
        Tile zoom level.

    Returns
    -------
    dict[str, numpy.ndarray]
        Intermediate CV images keyed by step name.
    """
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    x_pad = width * padding_pct
    y_pad = height * padding_pct

    img_array, _ = ctx.bounds2img(
        minx - x_pad, miny - y_pad, maxx + x_pad, maxy + y_pad,
        zoom=zoom, ll=False, source=ctx.providers.Esri.WorldImagery,
    )

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

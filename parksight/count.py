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
from shapely.geometry import LineString, MultiLineString, Point, Polygon

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

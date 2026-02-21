"""
ParkSight — satellite parking space counting toolkit.

Provides CV and ML baselines for estimating parking capacity
from OpenStreetMap geometries and satellite imagery.
"""

import json
import math
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# single source of truth for all constants
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
with _CONFIG_FILE.open(encoding="utf-8") as _f:
    config = json.load(_f)


def pick_device():
    """auto-select best available torch device"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_structure(tags):
    """check if osm tags indicate a garage or underground structure"""
    parking_type = str(tags.get("parking", "")).lower()
    building_type = str(tags.get("building", "")).lower()
    return (
        parking_type in ("multi-storey", "multi_storey", "underground")
        or building_type in ("garage", "garages")
    )


def corrected_line_length(geometry):
    """get latitude-corrected length in metres for a line geometry in epsg:3857"""
    geo_wgs = gpd.GeoSeries([geometry], crs=3857).to_crs(4326).geometry.iloc[0]

    lats = []
    if geo_wgs.geom_type == "MultiLineString":
        for line in geo_wgs.geoms:
            lats.extend(y for _, y in line.coords)
    else:
        lats.extend(y for _, y in geo_wgs.coords)

    mean_lat = float(np.mean(lats))
    length_m = float(geometry.length) * math.cos(math.radians(mean_lat))
    return length_m

"""
Fetch parking features from OpenStreetMap and satellite tiles.

Uses OSMnx to geocode addresses and query parking-related OSM features,
and contextily to fetch Esri World Imagery satellite tiles.
"""

from __future__ import annotations

import logging
from typing import Any

import contextily as ctx
import geopandas as gpd
import numpy as np
import osmnx as ox
from PIL import Image
from shapely.geometry import base as geom_base

logger = logging.getLogger(__name__)


# ── Default OSM tags for parking features ──────────────────────────

DEFAULT_TAGS: dict[str, Any] = {
    "amenity": ["parking", "parking_space"],
    "building": ["garage", "garages"],
    "parking:lane": True,
    "parking:left": True,
    "parking:right": True,
    "parking:both": True,
}


def get_parking_data(
    address: str,
    dist: int = 300,
    tags: dict[str, Any] | None = None,
) -> tuple[gpd.GeoDataFrame, tuple[float, float]]:
    """
    Geocode *address* and fetch OSM parking features within *dist* metres.

    Parameters
    ----------
    address : str
        Human-readable address to geocode (e.g. "Georgia Tech, Atlanta, GA").
    dist : int
        Search radius in metres (default 300).
    tags : dict, optional
        OSM tag filter.  Defaults to :data:`DEFAULT_TAGS`.

    Returns
    -------
    (gdf, (lat, lon))
        A GeoDataFrame of parking features in EPSG:4326 and the geocoded
        centre point.
    """
    if tags is None:
        tags = DEFAULT_TAGS

    geocode_result = ox.geocoder.geocode(address)
    print(geocode_result)
    if not geocode_result:
        raise ValueError(f"Could not geocode address: {address}")
    lat, lon = geocode_result
    logger.info("Geocoded %s → (%.5f, %.5f)", address, lat, lon)

    gdf = ox.features.features_from_point(
        center_point=(lat, lon),
        tags=tags,
        dist=dist,
    )

    if gdf.empty:
        logger.warning("No parking features found within %d m of %s.", dist, address)
        return gdf, (lat, lon)

    centroids = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    gdf["lat_lon"] = [(pt.y, pt.x) for pt in centroids]
    gdf = gdf[gdf.is_valid]

    return gdf, (lat, lon)


def get_satellite_tile(
    geometry: geom_base.BaseGeometry,
    padding_pct: float = 0.10,
    zoom: int = 19,
) -> Image.Image:
    """
    Fetch an Esri World Imagery satellite tile covering *geometry*.

    The geometry must be in **Web Mercator (EPSG:3857)**.  A percentage-based
    padding is added around the bounding box so that the tile has some context.

    Parameters
    ----------
    geometry : shapely geometry
        Feature geometry in EPSG:3857.
    padding_pct : float
        Fraction of the bbox size to add as padding (default 0.10).
    zoom : int
        Tile zoom level (default 19).

    Returns
    -------
    PIL.Image.Image
        RGB satellite image covering the padded bounding box.
    """
    if geometry.geom_type == "Point":
        x, y = geometry.x, geometry.y
        buffer = 50  # metres
        minx, miny, maxx, maxy = x - buffer, y - buffer, x + buffer, y + buffer
    else:
        minx, miny, maxx, maxy = geometry.bounds

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

    return Image.fromarray(img_array)

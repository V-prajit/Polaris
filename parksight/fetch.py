"""
Fetch parking features from OpenStreetMap and satellite tiles.

Uses OSMnx to geocode addresses and query parking-related OSM features,
and contextily to fetch Esri World Imagery satellite tiles.

Also provides helpers for structured parking and street parking queries.
"""

from __future__ import annotations

import logging

import contextily as ctx
import geopandas as gpd
import osmnx as ox
from PIL import Image

logger = logging.getLogger(__name__)


# default osm tags for parking features
DEFAULT_TAGS = {
    "amenity": ["parking", "parking_space"],
    "building": ["garage", "garages"],
    "parking:lane": True,
    "parking:left": True,
    "parking:right": True,
    "parking:both": True,
}


def get_parking_data(address, dist=300, tags=None):
    """
    Geocode address and fetch OSM parking features within dist metres.

    Returns (gdf, (lat, lon)).
    """
    if tags is None:
        tags = DEFAULT_TAGS

    geocode_result = ox.geocoder.geocode(address)
    if not geocode_result:
        raise ValueError(f"Could not geocode address: {address}")
    lat, lon = geocode_result
    logger.info("Geocoded %s -> (%.5f, %.5f)", address, lat, lon)

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


def get_satellite_tile(geometry, padding_pct=0.10, zoom=19):
    """
    Fetch an Esri World Imagery satellite tile covering geometry.

    The geometry must be in Web Mercator (EPSG:3857).
    """
    if geometry.geom_type == "Point":
        x, y = geometry.x, geometry.y
        buffer = 50
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


def fetch_structured_parking(address, dist=300):
    """
    Fetch garage, underground, and multi-storey parking features from OSM.

    Returns (gdf, (lat, lon)).
    """
    lat, lon = ox.geocoder.geocode(address)

    structure_tags = {
        "building": ["garage", "garages"],
        "parking": ["multi-storey", "underground"],
    }

    try:
        gdf = ox.features.features_from_point(
            center_point=(lat, lon),
            tags=structure_tags,
            dist=dist,
        )
    except Exception:
        gdf = gpd.GeoDataFrame()

    return gdf, (lat, lon)


def fetch_street_parking(address, dist=300):
    """
    Fetch road segments with parking lane tags from OSM.

    Returns (gdf, (lat, lon)).
    """
    lat, lon = ox.geocoder.geocode(address)

    street_tags = {
        "parking:lane": True,
        "parking:left": True,
        "parking:right": True,
        "parking:both": True,
    }

    try:
        gdf = ox.features.features_from_point(
            center_point=(lat, lon),
            tags=street_tags,
            dist=dist,
        )
    except Exception:
        gdf = gpd.GeoDataFrame()

    return gdf, (lat, lon)

"""
Fetch parking features from OpenStreetMap and satellite tiles.

Uses OSMnx to geocode addresses and query parking-related OSM features,
and contextily to fetch Esri World Imagery satellite tiles.

Also provides helpers for structured parking and street parking queries.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import time

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


_TILE_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "tiles"
_TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TILE_SOURCES = [("esri_world_imagery", ctx.providers.Esri.WorldImagery)]
try:
    _TILE_SOURCES.append(("usgs_usimagery", ctx.providers.USGS.USImagery))
except Exception:
    # Not all contextily versions expose USGS providers.
    pass


def _tile_cache_key(minx, miny, maxx, maxy, zoom, source_name):
    key = f"{round(minx, 2)}:{round(miny, 2)}:{round(maxx, 2)}:{round(maxy, 2)}:{zoom}:{source_name}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _load_tile_from_cache(cache_key):
    cache_path = _TILE_CACHE_DIR / f"{cache_key}.png"
    if not cache_path.exists():
        return None

    try:
        with Image.open(cache_path) as img:
            return img.convert("RGB")
    except Exception:
        logger.warning("Corrupt tile cache entry detected: %s", cache_path)
        return None


def _save_tile_to_cache(cache_key, image):
    cache_path = _TILE_CACHE_DIR / f"{cache_key}.png"
    try:
        image.save(cache_path, format="PNG")
    except Exception as exc:
        logger.warning("Failed to write tile cache %s: %s", cache_path, exc)


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

    padded_minx = minx - x_pad
    padded_miny = miny - y_pad
    padded_maxx = maxx + x_pad
    padded_maxy = maxy + y_pad

    # Generic cache key so we can fall back to stale imagery if provider calls fail.
    generic_cache_key = _tile_cache_key(
        padded_minx, padded_miny, padded_maxx, padded_maxy, zoom, "generic"
    )
    stale = _load_tile_from_cache(generic_cache_key)

    last_error = None
    for source_name, source in _TILE_SOURCES:
        provider_cache_key = _tile_cache_key(
            padded_minx, padded_miny, padded_maxx, padded_maxy, zoom, source_name
        )
        cached = _load_tile_from_cache(provider_cache_key)
        if cached is not None:
            return cached

        # Retry provider calls with short backoff for bursty tile throttling.
        for attempt in range(2):
            try:
                img_array, _ = ctx.bounds2img(
                    padded_minx,
                    padded_miny,
                    padded_maxx,
                    padded_maxy,
                    zoom=zoom,
                    ll=False,
                    source=source,
                )
                image = Image.fromarray(img_array).convert("RGB")
                _save_tile_to_cache(provider_cache_key, image)
                _save_tile_to_cache(generic_cache_key, image)
                return image
            except Exception as exc:
                last_error = exc
                wait_s = 0.8 * (attempt + 1)
                logger.warning(
                    "Tile fetch failed (source=%s, attempt=%s/2): %s",
                    source_name,
                    attempt + 1,
                    exc,
                )
                time.sleep(wait_s)

    if stale is not None:
        logger.warning("Using stale cached tile due to provider errors: %s", last_error)
        return stale

    raise RuntimeError(f"Failed to fetch satellite tile: {last_error}")


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


# --- coordinate-based variants (skip geocoding) ---

def get_parking_data_by_coords(lat, lon, dist=300, tags=None):
    """fetch osm parking features around (lat, lon) without geocoding"""
    if tags is None:
        tags = DEFAULT_TAGS

    gdf = ox.features.features_from_point(
        center_point=(lat, lon),
        tags=tags,
        dist=dist,
    )

    if gdf.empty:
        logger.warning("No parking features within %d m of (%.5f, %.5f).", dist, lat, lon)
        return gdf

    centroids = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    gdf["lat_lon"] = [(pt.y, pt.x) for pt in centroids]
    gdf = gdf[gdf.is_valid]

    return gdf


def fetch_structured_parking_by_coords(lat, lon, dist=300):
    """fetch garage / underground features from osm using coordinates"""
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

    return gdf


def fetch_street_parking_by_coords(lat, lon, dist=300):
    """fetch road segments with parking lane tags using coordinates"""
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

    return gdf

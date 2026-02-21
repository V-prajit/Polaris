"""
ParkSight FastAPI backend.

Single endpoint /api/estimate that takes lat, lon, radius and returns
a full parking breakdown (surface + structured + street) with geometry
coordinates so the frontend can draw features on a map.
"""

import sys
import time
import logging
from pathlib import Path
import math

import geopandas as gpd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# make sure parksight + yolo are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parksight import is_structure
from parksight.fetch import (
    get_parking_data_by_coords,
    get_satellite_tile,
    fetch_structured_parking_by_coords,
    fetch_street_parking_by_coords,
)
from parksight.count import get_line_count
from parksight.estimate_structured import estimate_structured_parking, estimate_street_parking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ParkSight API",
    description="Parking capacity estimation from satellite imagery and OSM data",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# lazy-loaded yolo detector singleton
_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        weights = PROJECT_ROOT / "models" / "yolo26n_run1.pt"
        if not weights.exists():
            logger.warning("YOLO weights not found at %s, surface detection disabled", weights)
            return None
        from yolo.detect import YOLOParkingDetector
        logger.info("Loading YOLO model from %s ...", weights)
        _detector = YOLOParkingDetector(str(weights))
        logger.info("YOLO model loaded.")
    return _detector


def _geometry_to_coords(geom):
    """convert a shapely geometry to a serialisable list of coordinate rings"""
    if geom.geom_type == "Point":
        return {"type": "Point", "coordinates": [geom.x, geom.y]}
    elif geom.geom_type == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [list(geom.exterior.coords)],
        }
    elif geom.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [list(poly.exterior.coords)] for poly in geom.geoms
            ],
        }
    elif geom.geom_type == "LineString":
        return {"type": "LineString", "coordinates": list(geom.coords)}
    elif geom.geom_type == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": [list(line.coords) for line in geom.geoms],
        }
    return {"type": geom.geom_type, "coordinates": []}


def _sanitize_dict(d):
    """recursively convert NaN values to None for JSON serialization"""
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(v) for v in d]
    elif isinstance(d, float) and math.isnan(d):
        return None
    return d


def _run_surface_detection(gdf_3857):
    """run yolo on surface lots, return list of per-feature dicts"""
    detector = _get_detector()
    results = []

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()
        is_struct = is_structure(tags)

        # skip structures — handled by structured estimator
        if is_struct:
            continue

        count = 0
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            if detector is not None:
                img = get_satellite_tile(geom)
                count = detector.count_spots(img, geom, osm_tags=tags)
            else:
                # fallback area estimate when no model
                from parksight import config
                stall_area = config["STALL_AREA_M2"]
                usable = config["USABLE_FRACTION_SURFACE"]
                count = int((geom.area / stall_area) * usable)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            count = get_line_count(geom)

        if count <= 0:
            continue

        # convert geometry to wgs84 for the frontend
        geom_wgs = gpd.GeoSeries([geom], crs=3857).to_crs(4326).iloc[0]
        centroid_wgs = geom_wgs.centroid

        results.append({
            "name": tags.get("name", f"Surface #{idx}"),
            "type": "surface",
            "count": count,
            "centroid": [centroid_wgs.y, centroid_wgs.x],
            "geometry": _geometry_to_coords(geom_wgs),
        })

    return results


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _detector is not None}


@app.get("/api/estimate")
def estimate(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
    radius: int = Query(300, description="Search radius in metres", ge=50, le=2000),
):
    t_start = time.time()

    # --- 1. surface lots ---
    gdf = get_parking_data_by_coords(lat, lon, dist=radius)
    surface_features = []
    surface_total = 0

    if gdf is not None and not gdf.empty:
        gdf_3857 = gdf.to_crs(epsg=3857)
        surface_features = _run_surface_detection(gdf_3857)
        surface_total = sum(f["count"] for f in surface_features)

    # --- 2. structured parking (garages / underground) ---
    struct_gdf = fetch_structured_parking_by_coords(lat, lon, dist=radius)
    structured_features = []
    structured_total = 0

    if not struct_gdf.empty:
        raw = estimate_structured_parking(struct_gdf)
        structured_total = sum(r["total_spots"] for r in raw)

        struct_3857 = struct_gdf.to_crs(epsg=3857) if struct_gdf.crs != "EPSG:3857" else struct_gdf
        struct_wgs = struct_3857.to_crs(4326)

        for r in raw:
            idx = r["index"]
            if idx in struct_wgs.index:
                geom_wgs = struct_wgs.loc[idx].geometry
                centroid = geom_wgs.centroid
                structured_features.append({
                    "name": r["name"],
                    "type": r["type"],
                    "count": r["total_spots"],
                    "levels": r["levels"],
                    "floor_area_m2": r["floor_area_m2"],
                    "centroid": [centroid.y, centroid.x],
                    "geometry": _geometry_to_coords(geom_wgs),
                })

    # --- 3. street parking ---
    street_gdf = fetch_street_parking_by_coords(lat, lon, dist=radius)
    street_features = []
    street_total = 0

    if not street_gdf.empty:
        raw = estimate_street_parking(street_gdf)
        street_total = sum(r["total_spots"] for r in raw)

        street_3857 = street_gdf.to_crs(epsg=3857) if street_gdf.crs != "EPSG:3857" else street_gdf
        street_wgs = street_3857.to_crs(4326)

        for r in raw:
            idx = r["index"]
            if idx in street_wgs.index:
                geom_wgs = street_wgs.loc[idx].geometry
                centroid = geom_wgs.centroid
                street_features.append({
                    "name": r["name"],
                    "type": "street",
                    "count": r["total_spots"],
                    "length_m": r["length_m"],
                    "sides": r["sides"],
                    "centroid": [centroid.y, centroid.x],
                    "geometry": _geometry_to_coords(geom_wgs),
                })

    grand_total = surface_total + structured_total + street_total
    elapsed = round(time.time() - t_start, 2)

    return _sanitize_dict({
        "lat": lat,
        "lon": lon,
        "radius": radius,
        "surface": {
            "total": surface_total,
            "features": surface_features,
        },
        "structured": {
            "total": structured_total,
            "features": structured_features,
        },
        "street": {
            "total": street_total,
            "features": street_features,
        },
        "grand_total": grand_total,
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

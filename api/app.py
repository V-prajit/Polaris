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

from cachetools import TTLCache, cached
import h3


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


# lazy-loaded model singletons
_detector = None
_segmenter = None

def _get_detector():
    global _detector
    if _detector is None:
        # Prefer APKLOT-trained model, fall back to ParkSeg-trained
        apklot_weights = PROJECT_ROOT / "models" / "yolo_apklot_best.pt"
        parkseg_weights = PROJECT_ROOT / "models" / "yolo26n_run1.pt"
        if apklot_weights.exists():
            from yolo.detect import YOLOParkingDetector
            logger.info("Loading APKLOT YOLO model from %s ...", apklot_weights)
            _detector = YOLOParkingDetector(str(apklot_weights), count_mode="detect")
        elif parkseg_weights.exists():
            from yolo.detect import YOLOParkingDetector
            logger.info("Loading ParkSeg YOLO model from %s ...", parkseg_weights)
            _detector = YOLOParkingDetector(str(parkseg_weights), count_mode="area")
        else:
            logger.warning("No YOLO weights found, surface detection disabled")
            return None
        logger.info("YOLO model loaded (count_mode=%s).", _detector.count_mode)
    return _detector


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg-final" / "best_model"
        if not ckpt.exists():
            ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg" / "best_model"
        if not ckpt.exists():
            logger.info("No SegFormer checkpoint found, skipping segmentation stage.")
            return None
        from parksight.segment import ParkingSegmenter
        logger.info("Loading SegFormer from %s ...", ckpt)
        _segmenter = ParkingSegmenter(str(ckpt))
        logger.info("SegFormer loaded.")
    return _segmenter


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
    """Run two-stage pipeline on surface lots: SegFormer mask → YOLO detect."""
    detector = _get_detector()
    segmenter = _get_segmenter()
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
            img = get_satellite_tile(geom)

            # Stage 1: SegFormer lot mask (optional)
            seg_mask = None
            if segmenter is not None:
                try:
                    seg_mask = segmenter.segment(img)
                except Exception as e:
                    logger.warning("SegFormer failed for feature %s: %s", idx, e)

            # Stage 2: YOLO detection (with optional mask filtering)
            if detector is not None:
                count = detector.count_spots(
                    img, geom, osm_tags=tags, segformer_mask=seg_mask
                )
            elif seg_mask is not None:
                # SegFormer-only fallback: area-based count from mask
                result = segmenter.count_spots(img)
                count = result.count
            else:
                # No models — pure area heuristic
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


# 10 minute cache, max 100 items
_estimate_cache = TTLCache(maxsize=100, ttl=600)

def _estimate_cache_key(lat: float, lon: float, radius: int):
    # cache key rounds to 3 decimal places (~111m) so nearby requests reuse cache
    return hash((round(lat, 3), round(lon, 3), radius))


@app.get("/api/estimate")
@cached(cache=_estimate_cache, key=_estimate_cache_key)
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


@app.get("/api/macro")
def macro(
    min_lat: float = Query(..., description="South boundary (WGS84)"),
    min_lon: float = Query(..., description="West boundary (WGS84)"),
    max_lat: float = Query(..., description="North boundary (WGS84)"),
    max_lon: float = Query(..., description="East boundary (WGS84)"),
    resolution: int = Query(9, description="H3 grid resolution (e.g. 9 is ~170m radius)"),
):
    """
    City-Wide Heatmap Generator.
    Splits a bounding box into H3 hexagons and quickly estimates the parking
    capacity in each hexagon using OSM data + area heuristics.
    (Skips heavy YOLO inference to allow scanning large areas fast).
    """
    t_start = time.time()
    
    # 1. build the bounding box polygon
    poly = h3.LatLngPoly([
        (min_lat, min_lon),
        (max_lat, min_lon),
        (max_lat, max_lon),
        (min_lat, max_lon),
    ])
    
    # 2. fill with h3 hexagons
    hexagons = list(h3.polygon_to_cells(poly, res=resolution))
    if len(hexagons) > 200:
        return {"error": f"Bounding box too large for resolution {resolution}. Trying to generate {len(hexagons)} cells (max 200). Reduce resolution or shrink bounding box.", "status": 400}
    
    grid_features = []
    
    # quick area fallback
    from parksight import config
    stall_area = config["STALL_AREA_M2"]
    usable = config["USABLE_FRACTION_SURFACE"]
    
    # search radius roughly matches hex size (res 9 ~ 170m radius hex)
    radius = int(math.sqrt(h3.cell_area(hexagons[0], unit='m^2') / math.pi)) if hexagons else 200
    
    for hex_id in hexagons:
        lat, lon = h3.cell_to_latlng(hex_id)
        hex_boundary = h3.cell_to_boundary(hex_id)
        # convert (lat,lon) to (lon,lat) for GeoJSON
        geojson_coords = [[ [lon, lat] for lat, lon in hex_boundary ]]
        # close the loop
        geojson_coords[0].append(geojson_coords[0][0])
        
        surface_total = 0
        structured_total = 0
        street_total = 0
        
        # --- surface ---
        gdf = get_parking_data_by_coords(lat, lon, dist=radius)
        if gdf is not None and not gdf.empty:
            gdf_3857 = gdf.to_crs(epsg=3857)
            for idx, row in gdf_3857.iterrows():
                geom = row.geometry
                tags = row.to_dict()
                if is_structure(tags):
                    continue
                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    count = int((geom.area / stall_area) * usable)
                elif geom.geom_type in ("LineString", "MultiLineString"):
                    count = get_line_count(geom)
                else:
                    count = 0
                surface_total += max(count, 0)
                
        # --- structured ---
        struct_gdf = fetch_structured_parking_by_coords(lat, lon, dist=radius)
        if not struct_gdf.empty:
            raw = estimate_structured_parking(struct_gdf)
            structured_total = sum(r["total_spots"] for r in raw)
            
        # --- street ---
        street_gdf = fetch_street_parking_by_coords(lat, lon, dist=radius)
        if not street_gdf.empty:
            raw = estimate_street_parking(street_gdf)
            street_total = sum(r["total_spots"] for r in raw)
            
        total = surface_total + structured_total + street_total
        
        grid_features.append({
            "hex_id": hex_id,
            "centroid": [lat, lon],
            "total": total,
            "surface": surface_total,
            "structured": structured_total,
            "street": street_total,
            "geometry": {
                "type": "Polygon",
                "coordinates": geojson_coords
            }
        })
        
    elapsed = round(time.time() - t_start, 2)
    return {
        "status": "ok",
        "bbox": [min_lat, min_lon, max_lat, max_lon],
        "resolution": resolution,
        "hex_count": len(hexagons),
        "radius_used": radius,
        "grid": grid_features,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

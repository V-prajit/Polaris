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
from dotenv import load_dotenv

load_dotenv()

from cachetools import TTLCache, cached
import h3


import geopandas as gpd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from parksight.confidence import confidence_band, utilization_band
from parksight.vector_search import (
    get_vector_index_status,
    index_hex_cells,
    semantic_search,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ParkSight API",
    description="Parking capacity estimation from satellite imagery and OSM data",
    version="0.1.0",
)

import os

_cors_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = ["*"] if _cors_raw.strip() == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# lazy-loaded model singletons
_detector = None
_segmenter = None
_car_detector = None

# Synthetic scan tuning for places where OSM has no surface lot polygons.
# Keep this small so inference stays fast and map scale remains usable.
_SYNTHETIC_SCAN_MIN_RADIUS_M = 40
_SYNTHETIC_SCAN_MAX_RADIUS_M = 80
_SYNTHETIC_SCAN_RADIUS_FACTOR = 0.20

# Convert detected cars to a conservative capacity estimate in synthetic mode.
_SYNTHETIC_CAPACITY_MULTIPLIER = 1.6
_SYNTHETIC_CAPACITY_BUFFER = 3
_SYNTHETIC_FALLBACK_MIN = 8
_SYNTHETIC_FALLBACK_MAX = 48

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


def _get_car_detector():
    """Load a COCO-pretrained YOLO for vehicle counting."""
    global _car_detector
    if _car_detector is None:
        try:
            from yolo.detect import YOLOParkingDetector
            weights = PROJECT_ROOT / "models" / "yolo_aerial_cars.pt"
            if not weights.exists():
                weights_str = "yolo_aerial_cars.pt"
            else:
                weights_str = str(weights)
            logger.info("Loading COCO YOLO for car counting (%s) ...", weights_str)
            _car_detector = YOLOParkingDetector(weights_str, count_mode="detect")
            logger.info("Car detector loaded.")
        except Exception as e:
            logger.warning("Could not load car detector: %s", e)
            return None
    return _car_detector


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


def _is_synthetic_scan(tags: dict) -> bool:
    """Return True if the feature was synthetically injected for scan fallback."""
    raw_flag = tags.get("is_synthetic_scan")
    if isinstance(raw_flag, bool):
        return raw_flag
    if isinstance(raw_flag, float) and math.isnan(raw_flag):
        raw_flag = False
    elif isinstance(raw_flag, str):
        return raw_flag.strip().lower() in {"1", "true", "yes"}
    elif raw_flag is None:
        raw_flag = False

    if bool(raw_flag):
        return True
    name = str(tags.get("name", "")).lower()
    return name.startswith("scan area")


def _synthetic_scan_radius_m(request_radius: int) -> int:
    """Clamp synthetic scan radius so demo responses stay realistic and fast."""
    scaled = int(request_radius * _SYNTHETIC_SCAN_RADIUS_FACTOR)
    return max(_SYNTHETIC_SCAN_MIN_RADIUS_M, min(_SYNTHETIC_SCAN_MAX_RADIUS_M, scaled))


def _synthetic_capacity_from_cars(cars_detected: int, area_count: int) -> int:
    """
    Estimate capacity for synthetic scan regions using car detections as baseline.

    When no cars are seen, use a tightly clamped fallback from area estimate.
    """
    if cars_detected > 0:
        from_cars = int(round(cars_detected * _SYNTHETIC_CAPACITY_MULTIPLIER))
        return max(from_cars, cars_detected + _SYNTHETIC_CAPACITY_BUFFER)

    fallback = int(area_count * 0.02)
    return max(_SYNTHETIC_FALLBACK_MIN, min(_SYNTHETIC_FALLBACK_MAX, fallback))


def _pixel_boxes_to_wgs84(boxes_pixel, geom_3857, img_size, padding_pct=0.10):
    """Convert pixel-space bounding boxes to WGS84 lat/lon rectangles.

    Parameters
    ----------
    boxes_pixel : list of (x1, y1, x2, y2, conf)
        Detection boxes in pixel coordinates (origin = top-left).
    geom_3857 : shapely geometry
        OSM feature geometry in EPSG:3857 (used to derive tile extents).
    img_size : (width, height)
        Pixel dimensions of the satellite tile (PIL Image.size).
    padding_pct : float
        Padding fraction applied when fetching the tile (default 0.10 = 10 %).

    Returns
    -------
    list of {"bbox": [lat1, lon1, lat2, lon2], "conf": float}
        Each dict represents one detection box in WGS84 degrees.
        lat1/lon1 = south-west corner, lat2/lon2 = north-east corner.
    """
    if not boxes_pixel:
        return []

    import pyproj

    minx, miny, maxx, maxy = geom_3857.bounds
    width = maxx - minx
    height = maxy - miny
    pad_x = width * padding_pct
    pad_y = height * padding_pct
    tile_minx = minx - pad_x
    tile_miny = miny - pad_y
    tile_maxx = maxx + pad_x
    tile_maxy = maxy + pad_y
    img_w, img_h = img_size
    m_per_px_x = (tile_maxx - tile_minx) / img_w
    m_per_px_y = (tile_maxy - tile_miny) / img_h

    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    wgs84_boxes = []
    for x1, y1, x2, y2, conf in boxes_pixel:
        # pixel → EPSG:3857
        geo_x1 = tile_minx + x1 * m_per_px_x
        geo_x2 = tile_minx + x2 * m_per_px_x
        # y-axis is flipped: pixel y=0 is at the top (north) of the tile
        geo_y1 = tile_maxy - y2 * m_per_px_y   # south edge of box
        geo_y2 = tile_maxy - y1 * m_per_px_y   # north edge of box
        # EPSG:3857 → WGS84 (lon, lat order with always_xy=True)
        lon1, lat1 = transformer.transform(geo_x1, geo_y1)   # SW corner
        lon2, lat2 = transformer.transform(geo_x2, geo_y2)   # NE corner
        wgs84_boxes.append({"bbox": [lat1, lon1, lat2, lon2], "conf": round(float(conf), 3)})
    return wgs84_boxes


def _pixel_contours_to_wgs84(contours, geom_3857, img_size, padding_pct=0.10):
    """Convert pixel-space contours to a WGS84 GeoJSON MultiPolygon.

    Parameters
    ----------
    contours : list of numpy.ndarray
        Each array is shape (N, 2) with columns [x, y] in pixel coords,
        as returned by ParkingSegmenter.segment_to_contours().
    geom_3857 : shapely geometry
        OSM feature geometry in EPSG:3857.
    img_size : (width, height)
        Pixel dimensions of the satellite tile.
    padding_pct : float
        Padding fraction used when the tile was fetched.

    Returns
    -------
    dict or None
        GeoJSON geometry dict of type "MultiPolygon", or *None* if no
        contours survive the conversion (e.g. all have fewer than 3 points).
    """
    if not contours:
        return None

    import pyproj

    minx, miny, maxx, maxy = geom_3857.bounds
    width = maxx - minx
    height = maxy - miny
    pad_x = width * padding_pct
    pad_y = height * padding_pct
    tile_minx = minx - pad_x
    tile_miny = miny - pad_y
    tile_maxx = maxx + pad_x
    tile_maxy = maxy + pad_y
    img_w, img_h = img_size
    m_per_px_x = (tile_maxx - tile_minx) / img_w
    m_per_px_y = (tile_maxy - tile_miny) / img_h

    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        ring = []
        for pt in contour:
            x, y = float(pt[0]), float(pt[1])
            geo_x = tile_minx + x * m_per_px_x
            geo_y = tile_maxy - y * m_per_px_y
            lon, lat = transformer.transform(geo_x, geo_y)
            ring.append([lon, lat])
        # GeoJSON rings must be closed
        ring.append(ring[0])
        polygons.append([ring])

    if not polygons:
        return None

    return {"type": "MultiPolygon", "coordinates": polygons}


def _run_surface_detection(gdf_3857):
    """Run two-stage pipeline on surface lots: SegFormer mask → YOLO detect + car count."""
    detector = _get_detector()
    segmenter = _get_segmenter()
    car_detector = _get_car_detector()
    results = []

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()
        is_struct = is_structure(tags)
        is_synthetic_scan = _is_synthetic_scan(tags)

        # skip structures — handled by structured estimator
        if is_struct:
            continue

        count = 0
        count_area = 0
        count_yolo = 0
        count_segformer = 0
        cars = 0
        spot_method = "area"
        segformer_available = segmenter is not None

        # Overlay data — populated only for Polygon features that go through imagery
        spot_boxes = []       # list of (x1,y1,x2,y2,conf) pixel tuples
        car_boxes = []        # list of (x1,y1,x2,y2,conf) pixel tuples
        seg_contours = []     # list of (N,2) numpy arrays
        img = None            # set below when a tile is fetched

        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)

            from parksight import config as _cfg
            stall_area = _cfg["STALL_AREA_M2"]
            usable = _cfg["USABLE_FRACTION_SURFACE"]
            count_area = int((geom.area / stall_area) * usable)

            # Stage 1: SegFormer lot mask (optional)
            seg_mask = None
            if segmenter is not None:
                try:
                    seg_mask = segmenter.segment(img)
                    result = segmenter.count_spots(img)
                    count_segformer = result.count
                    # Extract contours from the same mask for the overlay
                    seg_contours = segmenter.segment_to_contours(img)
                except Exception as e:
                    logger.warning("SegFormer failed for feature %s: %s", idx, e)

            # Stage 2: YOLO spot detection (with optional mask filtering)
            if detector is not None:
                count_yolo, spot_boxes = detector.count_spots_with_boxes(
                    img, geom, osm_tags=tags, segformer_mask=seg_mask
                )
                spot_method = "yolo_detect" if detector.count_mode == "detect" else "segformer"
                count = count_yolo
            elif seg_mask is not None:
                spot_method = "segformer"
                count = count_segformer
            else:
                spot_method = "area"
                count = count_area

            # Stage 3: Car counting (COCO-pretrained YOLO)
            if car_detector is not None:
                try:
                    cars, car_boxes = car_detector.count_cars_with_boxes(
                        img, segformer_mask=seg_mask
                    )
                except Exception as e:
                    logger.warning("Car counting failed for feature %s: %s", idx, e)

            # Synthetic scan fallback:
            # avoid massive area-based totals when no true OSM lot polygon exists.
            if is_synthetic_scan:
                baseline = _synthetic_capacity_from_cars(cars_detected=cars, area_count=count_area)
                count = baseline
                count_yolo = baseline
                spot_method = "blend"
                # SegFormer output should not be advertised if model is unavailable.
                if not segformer_available:
                    count_segformer = 0

        elif geom.geom_type in ("LineString", "MultiLineString"):
            count = get_line_count(geom)
            count_area = count
            spot_method = "street"

        if count <= 0 and count_area <= 0:
            continue

        # confidence bands
        spot_band = confidence_band(count, method=spot_method)
        car_band = confidence_band(cars, method="yolo_car")
        utilization = utilization_band(car_band, spot_band)

        # convert geometry to wgs84 for the frontend
        geom_wgs = gpd.GeoSeries([geom], crs=3857).to_crs(4326).iloc[0]
        centroid_wgs = geom_wgs.centroid

        # Derive tile_bounds from the WGS84 geometry extents [S, W, N, E]
        tb = geom_wgs.bounds  # (minx=west, miny=south, maxx=east, maxy=north)
        tile_bounds = [tb[1], tb[0], tb[3], tb[2]]

        # Convert pixel-space overlay data to WGS84 (only when we have imagery)
        spot_boxes_wgs84 = []
        car_boxes_wgs84 = []
        segformer_contours_wgs84 = None
        if img is not None:
            img_size = img.size
            spot_boxes_wgs84 = _pixel_boxes_to_wgs84(spot_boxes, geom, img_size)
            car_boxes_wgs84 = _pixel_boxes_to_wgs84(car_boxes, geom, img_size)
            segformer_contours_wgs84 = _pixel_contours_to_wgs84(seg_contours, geom, img_size)

        results.append({
            "name": tags.get("name", f"Surface #{idx}"),
            "type": "surface",
            "count": count,
            "count_area": count_area,
            "count_yolo": count_yolo,
            "count_segformer": count_segformer,
            "spots": spot_band.to_dict(),
            "cars": car_band.to_dict(),
            "utilization": utilization,
            "is_synthetic_scan": is_synthetic_scan,
            "scan_radius_m": tags.get("scan_radius_m"),
            "segformer_available": segformer_available,
            "centroid": [centroid_wgs.y, centroid_wgs.x],
            "geometry": _geometry_to_coords(geom_wgs),
            # Detection overlay data (new fields — backwards-compatible additions)
            "spot_boxes": spot_boxes_wgs84,
            "car_boxes": car_boxes_wgs84,
            "segformer_contours": segformer_contours_wgs84,
            "tile_bounds": tile_bounds,
        })

    return results


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _detector is not None}


# 10 minute cache, max 100 items
_estimate_cache = TTLCache(maxsize=100, ttl=600)


class PolarisSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=50)
    min_spots: int | None = Field(default=None, ge=0)
    require_garage: bool = False
    require_street: bool = False


ATLANTA_POLARIS_BBOX = {
    "min_lat": 33.647,
    "max_lat": 33.886,
    "min_lon": -84.552,
    "max_lon": -84.289,
    "resolution": 7,
}

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

    # Check if OSM returned any non-structure polygon features (actual surface lots)
    has_surface_polygons = False
    gdf_3857 = None
    if gdf is not None and not gdf.empty:
        gdf_3857 = gdf.to_crs(epsg=3857)
        for _, row in gdf_3857.iterrows():
            geom = row.geometry
            if geom.geom_type in ("Polygon", "MultiPolygon") and not is_structure(row.to_dict()):
                has_surface_polygons = True
                break

    if not has_surface_polygons:
        # No surface parking polygons in OSM — create a synthetic scan polygon
        # so YOLO can still detect cars/spots from satellite imagery.
        # Radius is clamped to avoid giant circles that inflate counts/zoom.
        import pyproj
        from shapely.geometry import Point as ShapelyPoint

        scan_radius = _synthetic_scan_radius_m(radius)
        transformer_to_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        cx, cy = transformer_to_3857.transform(lon, lat)
        synthetic_geom = ShapelyPoint(cx, cy).buffer(scan_radius)
        synthetic_gdf = gpd.GeoDataFrame(
            [{
                "name": f"Scan Area ({lat:.5f}, {lon:.5f})",
                "amenity": "parking",
                "parking": "surface",
                "is_synthetic_scan": True,
                "scan_radius_m": scan_radius,
            }],
            geometry=[synthetic_geom],
            crs="EPSG:3857",
        )
        if gdf is not None and not gdf.empty:
            import pandas as pd
            gdf_3857 = pd.concat([gdf_3857, synthetic_gdf], ignore_index=True)
            gdf_3857 = gpd.GeoDataFrame(gdf_3857, geometry="geometry", crs="EPSG:3857")
        else:
            gdf_3857 = synthetic_gdf
        logger.info(
            "Created synthetic scan polygon for (%.5f, %.5f) request_radius=%sm scan_radius=%sm",
            lat,
            lon,
            radius,
            scan_radius,
        )

    if gdf_3857 is not None and not gdf_3857.empty:
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
                method = r["type"]  # "garage" or "underground"
                spot_band = confidence_band(r["total_spots"], method=method)
                structured_features.append({
                    "name": r["name"],
                    "type": r["type"],
                    "count": r["total_spots"],
                    "spots": spot_band.to_dict(),
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
                spot_band = confidence_band(r["total_spots"], method="street")
                street_features.append({
                    "name": r["name"],
                    "type": "street",
                    "count": r["total_spots"],
                    "spots": spot_band.to_dict(),
                    "length_m": r["length_m"],
                    "sides": r["sides"],
                    "centroid": [centroid.y, centroid.x],
                    "geometry": _geometry_to_coords(geom_wgs),
                })

    grand_total = surface_total + structured_total + street_total
    total_cars = sum(f.get("cars", {}).get("value", 0) for f in surface_features)
    elapsed = round(time.time() - t_start, 2)

    # aggregate confidence bands
    grand_spot_band = confidence_band(grand_total, method="default")
    grand_car_band = confidence_band(total_cars, method="yolo_car")
    grand_utilization = utilization_band(grand_car_band, grand_spot_band)

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
        "spots": grand_spot_band.to_dict(),
        "cars": grand_car_band.to_dict(),
        "utilization": grand_utilization,
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
    if len(hexagons) > 10000:
        return {"error": f"Bounding box too large for resolution {resolution}. Trying to generate {len(hexagons)} cells (max 200). Reduce resolution or shrink bounding box.", "status": 400}
    
    grid_features = []
    skipped_cells = 0
    
    # quick area fallback
    from parksight import config
    stall_area = config["STALL_AREA_M2"]
    usable = config["USABLE_FRACTION_SURFACE"]
    
    # search radius roughly matches hex size (res 9 ~ 170m radius hex)
    radius = int(math.sqrt(h3.cell_area(hexagons[0], unit='m^2') / math.pi)) if hexagons else 200
    
    for hex_id in hexagons:
        try:
            lat, lon = h3.cell_to_latlng(hex_id)
            hex_boundary = h3.cell_to_boundary(hex_id)
            # convert (lat,lon) to (lon,lat) for GeoJSON
            geojson_coords = [[[lon, lat] for lat, lon in hex_boundary]]
            # close the loop
            geojson_coords[0].append(geojson_coords[0][0])

            surface_total = 0
            structured_total = 0
            street_total = 0

            # --- surface ---
            gdf = get_parking_data_by_coords(lat, lon, dist=radius)
            if gdf is not None and not gdf.empty:
                gdf_3857 = gdf.to_crs(epsg=3857)
                for _, row in gdf_3857.iterrows():
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
        except Exception as exc:
            skipped_cells += 1
            logger.warning("Skipping macro hex %s due to error: %s", hex_id, exc)
            continue
        
    elapsed = round(time.time() - t_start, 2)
    return {
        "status": "ok",
        "bbox": [min_lat, min_lon, max_lat, max_lon],
        "resolution": resolution,
        "hex_count": len(hexagons),
        "processed_cells": len(grid_features),
        "skipped_cells": skipped_cells,
        "radius_used": radius,
        "grid": grid_features,
        "elapsed_seconds": elapsed
    }


@app.post("/api/polaris/index")
async def polaris_index():
    """
    Build/refresh the Polaris semantic index from Atlanta macro parking cells.
    """
    t_start = time.time()

    macro_result = macro(
        min_lat=ATLANTA_POLARIS_BBOX["min_lat"],
        min_lon=ATLANTA_POLARIS_BBOX["min_lon"],
        max_lat=ATLANTA_POLARIS_BBOX["max_lat"],
        max_lon=ATLANTA_POLARIS_BBOX["max_lon"],
        resolution=ATLANTA_POLARIS_BBOX["resolution"],
    )

    if macro_result.get("status") == 400:
        raise HTTPException(
            status_code=400,
            detail=macro_result.get("error", "Failed to build macro hex grid."),
        )

    grid = macro_result.get("grid", [])
    if not grid:
        raise HTTPException(
            status_code=500,
            detail="Macro indexing returned an empty grid; nothing to index.",
        )

    try:
        indexed_cells = await index_hex_cells(grid)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Polaris index build failed.")
        raise HTTPException(
            status_code=500, detail=f"Failed to build Polaris index: {exc}"
        ) from exc

    return {
        "status": "ok",
        "indexed_cells": indexed_cells,
        "hex_count": len(grid),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "bbox": [
            ATLANTA_POLARIS_BBOX["min_lat"],
            ATLANTA_POLARIS_BBOX["min_lon"],
            ATLANTA_POLARIS_BBOX["max_lat"],
            ATLANTA_POLARIS_BBOX["max_lon"],
        ],
        "resolution": ATLANTA_POLARIS_BBOX["resolution"],
    }


@app.post("/api/polaris/search")
async def polaris_search(request: PolarisSearchRequest):
    """
    Semantic search over indexed Atlanta parking hex cells.
    """
    status = await get_vector_index_status()
    if not status.get("db_ready"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Actian VectorAI DB is not ready. "
                f"Details: {status.get('error', 'health check failed')}"
            ),
        )
    if int(status.get("indexed_count", 0)) <= 0:
        raise HTTPException(
            status_code=400,
            detail="No cells are indexed yet. Run POST /api/polaris/index first.",
        )

    try:
        results = await semantic_search(
            query=request.query,
            top_k=request.top_k,
            min_spots=request.min_spots,
            require_garage=request.require_garage,
            require_street=request.require_street,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Polaris semantic search failed.")
        raise HTTPException(
            status_code=500, detail=f"Polaris semantic search failed: {exc}"
        ) from exc

    return {
        "status": "ok",
        "query": request.query,
        "top_k": request.top_k,
        "filters": {
            "min_spots": request.min_spots,
            "require_garage": request.require_garage,
            "require_street": request.require_street,
        },
        "result_count": len(results),
        "results": results,
    }


@app.get("/api/polaris/status")
async def polaris_status():
    """
    Vector DB readiness and current indexed cell count.
    """
    return await get_vector_index_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

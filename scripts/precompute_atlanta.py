#!/usr/bin/env python3
import sys
import os
import json
import traceback
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parksight.fetch import (
    get_parking_data_by_coords, get_satellite_tile,
    fetch_structured_parking_by_coords, fetch_street_parking_by_coords
)
from parksight.count import get_line_count
from parksight.estimate_structured import estimate_structured_parking, estimate_street_parking
from parksight.confidence import confidence_band, utilization_band
from yolo.detect import YOLOParkingDetector
from parksight.segment import ParkingSegmenter
from parksight import is_structure
import geopandas as gpd

def geometry_to_coords(geom):
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

def sanitize_dict(d):
    import math
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    elif isinstance(d, float) and math.isnan(d):
        return None
    return d

def get_models():
    detector = None
    segmenter = None
    car_detector = None

    # Load parking detector
    apklot_weights = PROJECT_ROOT / "models" / "yolo_apklot_best.pt"
    parkseg_weights = PROJECT_ROOT / "models" / "yolo26n_run1.pt"
    if apklot_weights.exists():
        detector = YOLOParkingDetector(str(apklot_weights), count_mode="detect")
    elif parkseg_weights.exists():
        detector = YOLOParkingDetector(str(parkseg_weights), count_mode="area")

    # Load car detector
    weights = PROJECT_ROOT / "models" / "yolo_aerial_cars.pt"
    if weights.exists():
        car_detector = YOLOParkingDetector(str(weights), count_mode="detect")

    # Load segmenter
    ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg-final" / "best_model"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg" / "best_model"
    if ckpt.exists():
        segmenter = ParkingSegmenter(str(ckpt))

    return detector, segmenter, car_detector

def run_surface_detection(gdf_3857, detector, segmenter, car_detector):
    results = []

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()
        is_struct = is_structure(tags)

        if is_struct:
            continue

        count = 0
        count_area = 0
        count_yolo = 0
        count_segformer = 0
        cars = 0
        spot_method = "area"

        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)

            from parksight import config as _cfg
            stall_area = _cfg["STALL_AREA_M2"]
            usable = _cfg["USABLE_FRACTION_SURFACE"]
            count_area = int((geom.area / stall_area) * usable)

            seg_mask = None
            if segmenter is not None:
                try:
                    seg_mask = segmenter.segment(img)
                    result = segmenter.count_spots(img)
                    count_segformer = result.count
                except Exception:
                    pass

            if detector is not None:
                count_yolo = detector.count_spots(img, geom, osm_tags=tags, segformer_mask=seg_mask)
                spot_method = "yolo_detect" if detector.count_mode == "detect" else "segformer"
                count = count_yolo
            elif seg_mask is not None:
                spot_method = "segformer"
                count = count_segformer
            else:
                spot_method = "area"
                count = count_area

            if car_detector is not None:
                try:
                    cars = car_detector.count_cars(img, segformer_mask=seg_mask)
                except Exception:
                    pass

        elif geom.geom_type in ("LineString", "MultiLineString"):
            count = get_line_count(geom)
            count_area = count
            spot_method = "street"

        if count <= 0 and count_area <= 0:
            continue

        spot_band = confidence_band(count, method=spot_method)
        car_band = confidence_band(cars, method="yolo_car")
        utilization = utilization_band(car_band, spot_band)

        geom_wgs = gpd.GeoSeries([geom], crs=3857).to_crs(4326).iloc[0]
        centroid_wgs = geom_wgs.centroid

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
            "centroid": [centroid_wgs.y, centroid_wgs.x],
            "geometry": geometry_to_coords(geom_wgs),
        })

    return results

def get_estimate(lat, lon, radius, detector, segmenter, car_detector):
    t_start = time.time()

    # 1. surface lots
    gdf = get_parking_data_by_coords(lat, lon, dist=radius)
    surface_features = []
    surface_total = 0
    if gdf is not None and not gdf.empty:
        gdf_3857 = gdf.to_crs(epsg=3857)
        surface_features = run_surface_detection(gdf_3857, detector, segmenter, car_detector)
        surface_total = sum(f["count"] for f in surface_features)

    # 2. structured parking
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
                method = r["type"]
                spot_band = confidence_band(r["total_spots"], method=method)
                structured_features.append({
                    "name": r["name"],
                    "type": r["type"],
                    "count": r["total_spots"],
                    "spots": spot_band.to_dict(),
                    "levels": r["levels"],
                    "floor_area_m2": r["floor_area_m2"],
                    "centroid": [centroid.y, centroid.x],
                    "geometry": geometry_to_coords(geom_wgs),
                })

    # 3. street parking
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
                    "geometry": geometry_to_coords(geom_wgs),
                })

    grand_total = surface_total + structured_total + street_total
    total_cars = sum(f.get("cars", {}).get("value", 0) for f in surface_features)
    elapsed = round(time.time() - t_start, 2)

    grand_spot_band = confidence_band(grand_total, method="default")
    grand_car_band = confidence_band(total_cars, method="yolo_car")
    grand_utilization = utilization_band(grand_car_band, grand_spot_band)

    return sanitize_dict({
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

def main():
    parser = argparse.ArgumentParser(description="Precompute Atlanta Parking Estimates")
    parser.add_argument("--min-lat", type=float, default=33.647)
    parser.add_argument("--max-lat", type=float, default=33.886)
    parser.add_argument("--min-lon", type=float, default=-84.552)
    parser.add_argument("--max-lon", type=float, default=-84.289)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--radius", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="public/precomputed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lats = np.arange(args.min_lat, args.max_lat + args.step/2, args.step)
    lons = np.arange(args.min_lon, args.max_lon + args.step/2, args.step)
    
    grid = [(lat, lon) for lat in lats for lon in lons]
    print(f"Grid points to process: {len(grid)}")

    detector, segmenter, car_detector = get_models()

    log_file = open("precompute_errors.log", "a")
    
    for lat, lon in tqdm(grid, desc="Precomputing estimates"):
        filename = f"{lat:.3f}_{lon:.3f}_{args.radius}.json"
        filepath = os.path.join(args.output_dir, filename)
        
        if os.path.exists(filepath):
            continue
            
        try:
            result = get_estimate(lat, lon, args.radius, detector, segmenter, car_detector)
            with open(filepath, "w") as f:
                json.dump(result, f)
        except Exception as e:
            log_file.write(f"Error at lat={lat:.3f}, lon={lon:.3f}:\n")
            traceback.print_exc(file=log_file)
            log_file.flush()
            continue

    log_file.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare parking estimation methods on real parking lots.

Fetches random surface lots from OSM near known locations, then runs:
  1. Area heuristic    (area / stall_size * usable_fraction)
  2. Edge detection     (Canny + Hough lines)
  3. Geometric design   (ITE/NPA reverse-engineering)
  4. YOLO spot detect   (yolo26n_run1.pt or APKLOT model)
  5. YOLO car count     (COCO yolo11n + SAHI sliced inference)

Prints a comparison table and optionally saves satellite tiles.
"""

import sys
import time
import logging
from pathlib import Path
from tabulate import tabulate

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parksight import config, is_structure
from parksight.fetch import get_parking_data_by_coords, get_satellite_tile
from parksight.count import count_edges, count_geometric
from parksight.confidence import confidence_band

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Test locations (lat, lon, label) ─────────────────────────────────────────
TEST_LOCATIONS = [
    (33.7756, -84.3963, "Georgia Tech, Atlanta"),
    (33.7570, -84.4015, "Atlantic Station, Atlanta"),
]

RADIUS = 300  # metres
MAX_LOTS_PER_LOCATION = 3


def _load_yolo_spot_detector():
    """Load the parking-spot YOLO model."""
    from yolo.detect import YOLOParkingDetector
    apklot = ROOT / "models" / "yolo_apklot_best.pt"
    parkseg = ROOT / "models" / "yolo26n_run1.pt"
    if apklot.exists():
        return YOLOParkingDetector(str(apklot), count_mode="detect"), "APKLOT"
    elif parkseg.exists():
        return YOLOParkingDetector(str(parkseg), count_mode="area"), "ParkSeg"
    return None, None


def _load_yolo_car_detector():
    """Load the COCO-pretrained YOLO for car counting."""
    from yolo.detect import YOLOParkingDetector
    weights = ROOT / "models" / "yolo11n.pt"
    weights_str = str(weights) if weights.exists() else "yolo11n.pt"
    return YOLOParkingDetector(weights_str, count_mode="detect")


def main():
    print("=" * 80)
    print("ParkSight Method Comparison")
    print("=" * 80)

    # load models once
    print("\nLoading YOLO models...")
    spot_detector, spot_model_name = _load_yolo_spot_detector()
    if spot_detector:
        print(f"  Spot detector: {spot_model_name}")
    else:
        print("  Spot detector: NONE (no weights found)")

    car_detector = _load_yolo_car_detector()
    print(f"  Car detector:  COCO yolo11n loaded")

    stall_area = config["STALL_AREA_M2"]
    usable = config["USABLE_FRACTION_SURFACE"]

    all_rows = []

    for lat, lon, label in TEST_LOCATIONS:
        print(f"\n{'─' * 70}")
        print(f"📍 {label} ({lat}, {lon})")
        print(f"{'─' * 70}")

        try:
            gdf = get_parking_data_by_coords(lat, lon, dist=RADIUS)
        except Exception as e:
            print(f"  OSM query failed: {e}")
            continue
        if gdf is None or gdf.empty:
            print("  No parking features found, skipping.")
            continue

        gdf_3857 = gdf.to_crs(epsg=3857)

        lot_num = 0
        for idx, row in gdf_3857.iterrows():
            geom = row.geometry
            tags = row.to_dict()

            if is_structure(tags):
                continue
            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if geom.area < 200:  # skip tiny polygons
                continue

            lot_num += 1
            if lot_num > MAX_LOTS_PER_LOCATION:
                break

            raw_name = tags.get("name")
            name = str(raw_name) if raw_name and not (isinstance(raw_name, float)) else f"Lot #{idx}"
            area_m2 = geom.area

            # ── Method 1: Area heuristic ─────────────────────────────────
            area_est = int((area_m2 / stall_area) * usable)

            # ── Method 2: Edge detection (Canny + Hough) ────────────────
            try:
                edge_est = count_edges(geom)
            except Exception as e:
                edge_est = -1

            # ── Method 3: Geometric design ──────────────────────────────
            try:
                geo_result = count_geometric(geom, osm_tags=tags)
                geo_est = geo_result.count
            except Exception as e:
                geo_est = -1

            # Fetch satellite tile once, reuse for all ML methods
            print(f"    Fetching tile...", end=" ", flush=True)
            try:
                img = get_satellite_tile(geom)
                print(f"{img.size}", flush=True)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # ── Method 4: YOLO spot detection ───────────────────────────
            yolo_spots = -1
            if spot_detector is not None:
                try:
                    yolo_spots = spot_detector.count_spots(img, geom, osm_tags=tags)
                except Exception as e:
                    yolo_spots = -1

            # ── Method 5: YOLO car count (SAHI sliced inference) ─────
            yolo_cars = -1
            try:
                yolo_cars = car_detector.count_cars(img, confidence=0.15)
            except Exception as e:
                print(f"    Car count error: {e}")
                yolo_cars = -1

            # confidence bands
            area_band = confidence_band(area_est, method="area")
            geo_band = confidence_band(geo_est, method="geometric") if geo_est >= 0 else None
            car_band = confidence_band(yolo_cars, method="yolo_car") if yolo_cars >= 0 else None

            row_data = {
                "location": label[:20],
                "lot": name[:25],
                "area_m2": round(area_m2),
                "area_est": area_est,
                "edge_est": edge_est if edge_est >= 0 else "err",
                "geometric": geo_est if geo_est >= 0 else "err",
                "yolo_spots": yolo_spots if yolo_spots >= 0 else "err",
                "yolo_cars": yolo_cars if yolo_cars >= 0 else "err",
            }
            all_rows.append(row_data)

            print(f"  {name} ({round(area_m2)} m²)")
            print(f"    Area heuristic:  {area_est:>5}  [{area_band.low}-{area_band.high}]")
            print(f"    Edge detection:  {edge_est if edge_est >= 0 else 'err':>5}")
            if geo_est >= 0:
                print(f"    Geometric:       {geo_est:>5}  [{geo_band.low}-{geo_band.high}]")
            if yolo_spots >= 0:
                print(f"    YOLO spots:      {yolo_spots:>5}")
            print(f"    YOLO cars SAHI:  {yolo_cars if yolo_cars >= 0 else 'err':>5}  {'['+str(car_band.low)+'-'+str(car_band.high)+']' if car_band else ''}")

    # ── Summary table ────────────────────────────────────────────────────────
    if all_rows:
        print(f"\n{'=' * 80}")
        print("SUMMARY TABLE")
        print(f"{'=' * 80}\n")
        headers = {
            "location": "Location",
            "lot": "Lot Name",
            "area_m2": "Area m²",
            "area_est": "Area Est",
            "edge_est": "Edge Det",
            "geometric": "Geometric",
            "yolo_spots": "YOLO Spots",
            "yolo_cars": "Cars (SAHI)",
        }
        print(tabulate(all_rows, headers=headers, tablefmt="rounded_grid"))
    else:
        print("\nNo lots found across any location.")


if __name__ == "__main__":
    main()

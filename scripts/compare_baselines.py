#!/usr/bin/env python3
"""
Compare all four counting methods on the same address:
  1. CV Baseline (Tier 1) - Canny + Hough lines
  2. ML Baseline (Tier 2) - Grounding DINO zero-shot
  3. YOLO26 Pipeline       - Fine-tuned YOLOv8n + OSM metadata
  4. Enhanced Pipeline     - YOLO surface + OSM garages + OSM street parking

Usage:
    python scripts/compare_baselines.py --address "Georgia Tech, Atlanta, GA"
    python scripts/compare_baselines.py --address "Georgia Tech, Atlanta, GA" --radius 500
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parksight import is_structure
from parksight.fetch import get_parking_data, get_satellite_tile, fetch_structured_parking, fetch_street_parking
from parksight.count import count_edges, get_line_count
from parksight.estimate_structured import estimate_structured_parking, estimate_street_parking


def run_geometric_baseline(gdf_3857):
    """Tier 0: ITE/NPA design-standards estimator using OSM polygon geometry."""
    from parksight.count import count_geometric
    from shapely.ops import unary_union

    # Deduplicate: drop any polygon that overlaps >60% with a larger one
    geoms = list(gdf_3857.geometry)
    areas = [g.area for g in geoms]
    keep  = [True] * len(geoms)
    for i, gi in enumerate(geoms):
        if not keep[i]:
            continue
        for j, gj in enumerate(geoms):
            if i == j or not keep[j] or areas[j] <= areas[i]:
                continue
            try:
                if gi.intersection(gj).area / gi.area > 0.60:
                    keep[i] = False
                    break
            except Exception:
                pass

    total = 0
    for idx, (_, row) in enumerate(gdf_3857.iterrows()):
        if not keep[idx]:
            continue
        geom = row.geometry
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        try:
            result = count_geometric(geom, osm_tags=row.to_dict())
            total += result.count
        except Exception as e:
            print(f"  Geometric skip: {e}")
    return total


def run_cv_baseline(gdf_3857):
    # tier 1: canny edge detection + hough lines
    counts = []
    for _, row in gdf_3857.iterrows():
        geom = row.geometry
        try:
            c = count_edges(geom)
        except Exception as e:
            print(f"  CV skip: {e}")
            c = 0
        counts.append(c)
    return counts


def run_ml_baseline(gdf_3857):
    # tier 2: grounding dino zero-shot via parksight.detect.ParkingDetector
    # Note: ml_baseline.py was removed; use --skip-ml to bypass this tier.
    from parksight.detect import ParkingDetector
    detector = ParkingDetector()

    counts = []
    for _, row in gdf_3857.iterrows():
        geom = row.geometry
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)
            c = detector.count_spots(img)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            c = get_line_count(geom)
        else:
            c = 0
        counts.append(c)
    return counts


def run_yolo_pipeline(gdf_3857, weights_path):
    # our pipeline: fine-tuned yolo26 + osm metadata
    from yolo.detect import YOLOParkingDetector
    detector = YOLOParkingDetector(weights_path)

    counts = []
    for _, row in gdf_3857.iterrows():
        geom = row.geometry
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)
            c = detector.count_spots(img, geom, osm_tags=row.to_dict())
        elif geom.geom_type in ("LineString", "MultiLineString"):
            c = get_line_count(geom)
        else:
            c = 0
        counts.append(c)
    return counts


def run_enhanced_pipeline(gdf_3857, address, radius, weights_path):
    # enhanced: yolo for surface + osm structured + osm street
    from yolo.detect import YOLOParkingDetector
    detector = YOLOParkingDetector(weights_path)

    # surface detection (skip structures)
    surface_total = 0
    for _, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()

        if is_structure(tags):
            continue

        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)
            c = detector.count_spots(img, geom, osm_tags=tags)
            surface_total += c
        elif geom.geom_type in ("LineString", "MultiLineString"):
            surface_total += get_line_count(geom)

    # structured parking
    struct_gdf, _ = fetch_structured_parking(address, dist=radius)
    structured_total = 0
    if not struct_gdf.empty:
        results = estimate_structured_parking(struct_gdf)
        structured_total = sum(r["total_spots"] for r in results)

    # street parking
    street_gdf, _ = fetch_street_parking(address, dist=radius)
    street_total = 0
    if not street_gdf.empty:
        results = estimate_street_parking(street_gdf)
        street_total = sum(r["total_spots"] for r in results)

    grand_total = surface_total + structured_total + street_total
    return grand_total, surface_total, structured_total, street_total


def main():
    parser = argparse.ArgumentParser(description="Compare all parking counting methods")
    parser.add_argument("--address", required=True, type=str)
    parser.add_argument("--radius", default=300, type=int)
    parser.add_argument("--weights", default="models/yolo26n_run1.pt", type=str)
    parser.add_argument("--skip-ml", action="store_true", help="Skip Grounding DINO (slow to download)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PARKSIGHT BASELINE COMPARISON")
    print(f"  Address: {args.address}")
    print(f"  Radius:  {args.radius}m")
    print(f"{'='*60}\n")

    # fetch shared data
    print("[*] Fetching parking features from OSM...")
    gdf, (lat, lon) = get_parking_data(args.address, dist=args.radius)
    if gdf.empty:
        print("No parking features found. Try a larger radius.")
        return

    num_features = len(gdf)
    print(f"[*] Found {num_features} parking features\n")
    gdf_3857 = gdf.to_crs(epsg=3857)

    results = {}

    # --- tier 0: geometric baseline ---
    print("[0] Running Geometric Baseline (ITE/NPA design standards)...")
    t0 = time.time()
    geo_total = run_geometric_baseline(gdf_3857)
    geo_time = time.time() - t0
    results["Geometric (design std)"] = {"total": geo_total, "time": geo_time}
    print(f"    Total: {geo_total} spots | Time: {geo_time:.1f}s\n")

    # --- tier 1: cv baseline ---
    print("[1/4] Running CV Baseline (Canny + Hough)...")
    t0 = time.time()
    cv_counts = run_cv_baseline(gdf_3857)
    cv_time = time.time() - t0
    cv_total = sum(cv_counts)
    results["CV Baseline"] = {"total": cv_total, "time": cv_time}
    print(f"      Total: {cv_total} spots | Time: {cv_time:.1f}s\n")

    # --- tier 2: ml baseline ---
    if not args.skip_ml:
        print("[2/4] Running ML Baseline (Grounding DINO)...")
        t0 = time.time()
        ml_counts = run_ml_baseline(gdf_3857)
        ml_time = time.time() - t0
        ml_total = sum(ml_counts)
        results["ML Baseline"] = {"total": ml_total, "time": ml_time}
        print(f"      Total: {ml_total} spots | Time: {ml_time:.1f}s\n")
    else:
        print("[2/4] Skipping ML Baseline (--skip-ml)\n")

    # --- yolo26 pipeline ---
    print("[3/4] Running YOLO26 Pipeline...")
    t0 = time.time()
    yolo_counts = run_yolo_pipeline(gdf_3857, args.weights)
    yolo_time = time.time() - t0
    yolo_total = sum(yolo_counts)
    results["YOLO26 (raw)"] = {"total": yolo_total, "time": yolo_time}
    print(f"      Total: {yolo_total} spots | Time: {yolo_time:.1f}s\n")

    # --- enhanced pipeline ---
    print("[4/4] Running Enhanced Pipeline (YOLO + OSM structures + streets)...")
    t0 = time.time()
    enhanced_total, surface, structured, street = run_enhanced_pipeline(
        gdf_3857, args.address, args.radius, args.weights
    )
    enhanced_time = time.time() - t0
    results["Enhanced"] = {"total": enhanced_total, "time": enhanced_time}
    print(f"      Surface: {surface} | Garages: {structured} | Street: {street}")
    print(f"      Total: {enhanced_total} spots | Time: {enhanced_time:.1f}s\n")

    # --- summary table ---
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'Total Spots':>12} {'Time':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    for name, data in results.items():
        print(f"  {name:<30} {data['total']:>12,} {data['time']:>9.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


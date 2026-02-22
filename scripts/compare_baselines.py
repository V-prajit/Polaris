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
    """Tier 0: returns (total, {row_position: count})."""
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
    per_lot = {}
    for pos, (_, row) in enumerate(gdf_3857.iterrows()):
        if not keep[pos]:
            per_lot[pos] = 0
            continue
        geom = row.geometry
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            per_lot[pos] = 0
            continue
        try:
            result = count_geometric(geom, osm_tags=row.to_dict())
            per_lot[pos] = result.count
            total += result.count
        except Exception as e:
            print(f"  Geometric skip: {e}")
            per_lot[pos] = 0
    return total, per_lot


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


def save_lot_images(gdf_3857, out_dir: Path, address: str,
                    per_lot_counts: dict | None = None,
                    max_images: int | None = None):
    """Save satellite tiles with OSM polygon overlay and per-lot counts."""
    import random
    try:
        import contextily as ctx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError as e:
        print(f"  [images] skipped — missing dependency: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect polygon rows and optionally random-sample
    poly_rows = [
        (i, pos, idx, row)
        for i, (pos, (idx, row)) in enumerate(
            ((p, item) for p, item in enumerate(gdf_3857.iterrows())), 1
        )
        if row.geometry.geom_type in ("Polygon", "MultiPolygon")
    ]
    if max_images and len(poly_rows) > max_images:
        poly_rows = random.sample(poly_rows, max_images)
        poly_rows.sort(key=lambda x: x[0])  # keep display order

    print(f"\n[*] Saving {len(poly_rows)} lot image(s) to {out_dir}/")
    counts = per_lot_counts or {}

    for i, pos, idx, row in poly_rows:
        geom   = row.geometry
        osm_id = str(idx[1]) if isinstance(idx, tuple) else str(idx)
        name   = str(row.get("name", "") or "")[:22] or osm_id

        try:
            minx, miny, maxx, maxy = geom.bounds
            pad = max(maxx - minx, maxy - miny) * 0.15
            img_arr, ext = ctx.bounds2img(
                minx - pad, miny - pad, maxx + pad, maxy + pad,
                zoom=19, ll=False,
                source=ctx.providers.Esri.WorldImagery,
            )

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_arr, extent=ext, origin="upper")

            exterior = (
                list(geom.exterior.coords)
                if geom.geom_type == "Polygon"
                else list(geom.geoms[0].exterior.coords)
            )
            ax.add_patch(MplPolygon(
                exterior, closed=True,
                edgecolor="red", facecolor=(1, 0, 0, 0.08), linewidth=2.5,
            ))

            # Build count annotation
            lot_c = counts.get(pos, {})
            count_str = "  ".join(
                f"{k}={v}" for k, v in lot_c.items() if v
            ) or "no counts"
            ax.set_title(
                f"#{i} {name}\narea={geom.area:.0f} m\u00b2  |  {count_str}",
                fontsize=8,
            )
            ax.axis("off")
            plt.tight_layout(pad=0.5)

            fname = out_dir / f"lot_{i:02d}_{osm_id}.png"
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {fname.name}  [{count_str}]")
        except Exception as e:
            print(f"  lot {i} image failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare all parking counting methods")
    parser.add_argument("--address", required=True, type=str)
    parser.add_argument("--radius", default=300, type=int)
    parser.add_argument("--weights", default="models/yolo26n_run1.pt", type=str)
    parser.add_argument("--skip-ml", action="store_true", help="Skip Grounding DINO (slow to download)")
    parser.add_argument("--save-images", type=Path, default=None,
                        metavar="DIR",
                        help="Save satellite tile per lot to DIR (e.g. lot_images/)")
    parser.add_argument("--max-images", type=int, default=None,
                        metavar="N",
                        help="Randomly sample N lots for images (default: all)")
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
    geo_total, geo_per_lot = run_geometric_baseline(gdf_3857)
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

    # Build per-lot counts dict for image labels {row_pos: {method: count}}
    per_lot_counts: dict = {}
    for pos, c in enumerate(cv_counts):
        per_lot_counts.setdefault(pos, {})["cv"] = c
    for pos, c in geo_per_lot.items():
        per_lot_counts.setdefault(pos, {})["geo"] = c

    # --- tier 2: ml baseline ---
    if not args.skip_ml:
        print("[2/4] Running ML Baseline (Grounding DINO)...")
        t0 = time.time()
        ml_counts = run_ml_baseline(gdf_3857)
        ml_time = time.time() - t0
        ml_total = sum(ml_counts)
        results["ML Baseline"] = {"total": ml_total, "time": ml_time}
        print(f"      Total: {ml_total} spots | Time: {ml_time:.1f}s\n")
        for pos, c in enumerate(ml_counts):
            per_lot_counts.setdefault(pos, {})["ml"] = c
    else:
        print("[2/4] Skipping ML Baseline (--skip-ml)\n")


    # --- yolo26 pipeline (commented out — model not yet trained) ---
    # print("[3/4] Running YOLO26 Pipeline...")
    # t0 = time.time()
    # yolo_counts = run_yolo_pipeline(gdf_3857, args.weights)
    # yolo_time = time.time() - t0
    # yolo_total = sum(yolo_counts)
    # results["YOLO26 (raw)"] = {"total": yolo_total, "time": yolo_time}
    # print(f"      Total: {yolo_total} spots | Time: {yolo_time:.1f}s\n")

    # --- enhanced pipeline (commented out — requires trained YOLO model) ---
    # print("[4/4] Running Enhanced Pipeline (YOLO + OSM structures + streets)...")
    # t0 = time.time()
    # enhanced_total, surface, structured, street = run_enhanced_pipeline(
    #     gdf_3857, args.address, args.radius, args.weights
    # )
    # enhanced_time = time.time() - t0
    # results["Enhanced"] = {"total": enhanced_total, "time": enhanced_time}
    # print(f"      Surface: {surface} | Garages: {structured} | Street: {street}")
    # print(f"      Total: {enhanced_total} spots | Time: {enhanced_time:.1f}s\n")

    # --- summary table ---
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'Total Spots':>12} {'Time':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    for name, data in results.items():
        print(f"  {name:<30} {data['total']:>12,} {data['time']:>9.1f}s")
    print(f"{'='*60}\n")

    # --- save images ---
    if args.save_images:
        save_lot_images(
            gdf_3857, args.save_images, args.address,
            per_lot_counts=per_lot_counts,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()


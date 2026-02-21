"""
predict.py — Estimate parking spots for all OSM lots near an address.

Usage
-----
# Geometric only (fast, no model needed):
python scripts/predict.py "Georgia Tech, Atlanta GA"

# All three methods including SegFormer:
python scripts/predict.py "Georgia Tech, Atlanta GA" \
    --model checkpoints/segformer-b5-parkseg/best_model \
    --dist 500

# Output CSV:
python scripts/predict.py "Times Square, New York" --csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate parking spots near an address.")
    p.add_argument("address", help="Address or place name (geocoded via OSM)")
    p.add_argument("--dist",  type=int, default=300,
                   help="Search radius in metres (default 300)")
    p.add_argument("--model", type=Path, default=None,
                   help="Path to fine-tuned SegFormer checkpoint. "
                        "If omitted, SegFormer column is skipped.")
    p.add_argument("--csv",   type=Path, default=None,
                   help="Save results to this CSV file.")
    p.add_argument("--no-cv", action="store_true",
                   help="Skip the slow CV Hough-edge baseline.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from parksight.fetch import get_parking_data, get_satellite_tile
    from parksight.count import count_geometric, count_edges

    # ── 1. Fetch OSM parking polygons ─────────────────────────────────────────
    print(f"\n📍 Fetching parking lots within {args.dist} m of: {args.address!r}")
    gdf, (lat, lon) = get_parking_data(args.address, dist=args.dist)

    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if polys.empty:
        print("⚠  No polygon parking features found. Try increasing --dist.")
        sys.exit(0)

    polys_m = polys.to_crs(epsg=3857)
    print(f"   Found {len(polys_m)} parking polygons.\n")

    # ── 2. Load SegFormer (optional) ──────────────────────────────────────────
    segmenter = None
    if args.model is not None:
        from parksight.segment import ParkingSegmenter
        print(f"🤖 Loading SegFormer from {args.model} …")
        segmenter = ParkingSegmenter(str(args.model))
        print("   Model loaded.\n")

    # ── 3. Run estimators for each lot ────────────────────────────────────────
    rows = []
    for i, (_, row) in enumerate(polys_m.iterrows(), 1):
        geom     = row.geometry
        osm_tags = row.to_dict()
        area_m2  = geom.area

        # --- Geometric (always) ---
        geo = count_geometric(geom, osm_tags=osm_tags)
        entry = {
            "index":          i,
            "area_m2":        round(area_m2, 1),
            "stall_angle":    geo.best_angle_deg,
            "geometric":      geo.count,
        }

        # --- CV Hough (optional) ---
        if not args.no_cv:
            try:
                entry["cv_hough"] = count_edges(geom)
            except Exception as e:
                entry["cv_hough"] = f"ERR: {e}"

        # --- SegFormer (optional) ---
        if segmenter is not None:
            try:
                tile   = get_satellite_tile(geom)
                result = segmenter.count_spots(tile)
                entry["segformer"]        = result.count
                entry["segformer_method"] = result.method
            except Exception as e:
                entry["segformer"]        = f"ERR: {e}"
                entry["segformer_method"] = "-"

        rows.append(entry)
        status = f"  [{i:>3}/{len(polys_m)}]  area={area_m2:.0f} m²  → geometric={geo.count}"
        if "cv_hough" in entry:
            status += f"  cv={entry['cv_hough']}"
        if "segformer" in entry:
            status += f"  seg={entry['segformer']}"
        print(status)

    # ── 4. Display summary ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    totals = {"geometric": df["geometric"].sum()}
    if "cv_hough" in df.columns:
        totals["cv_hough"] = pd.to_numeric(df["cv_hough"], errors="coerce").sum()
    if "segformer" in df.columns:
        totals["segformer"] = pd.to_numeric(df["segformer"], errors="coerce").sum()
    print("\nTotals:")
    for k, v in totals.items():
        print(f"  {k:<18} {int(v):>6} estimated spots")

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n💾 Saved to {args.csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI: address → parking stall count + interactive map.

Usage:
    python examples/run_baseline.py --address "Georgia Tech, Atlanta, GA"
    python examples/run_baseline.py --address "1600 Amphitheatre Parkway, Mountain View, CA" --radius 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure parksight is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parksight import count, fetch, viz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParkSight baseline: estimate parking stalls from an address."
    )
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help='Address to search (e.g. "Georgia Tech, Atlanta, GA").',
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=300,
        help="Search radius in metres (default: 300).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parking_map.html",
        help="Output path for the HTML map (default: parking_map.html).",
    )
    args = parser.parse_args()

    # 1. Fetch parking features
    print(f"Searching for parking within {args.radius} m of: {args.address}")
    gdf, (lat, lon) = fetch.get_parking_data(args.address, dist=args.radius)

    if gdf.empty:
        print("No parking features found. Try a larger radius.")
        return

    print(f"Found {len(gdf)} parking features around ({lat:.5f}, {lon:.5f})")

    # 2. Count stalls (CV baseline)
    gdf_3857 = gdf.to_crs(epsg=3857)
    counts = []
    for _, row in gdf_3857.iterrows():
        try:
            c = count.count_edges(row.geometry)
        except Exception as e:
            print(f"  Skipped feature: {e}")
            c = 0
        counts.append(c)

    gdf["count"] = counts
    total = sum(counts)
    print(f"\nCV baseline total: {total} estimated stalls")

    # 3. Build and save map
    m = viz.create_parking_map(lat, lon, gdf, radius=args.radius, total_count=total)
    m.save(args.output)
    print(f"Map saved to: {args.output}")

    # 4. Export to GeoJSON for frontend
    geojson_output = "parking_data.geojson"
    # Convert lat/lon column if it exists to lists, or just drop it if it causes issues, but GeoJSON handles geometries natively.
    # lat_lon was added as a tuple, which might not serialize easily in some versions, so we stringify it or drop it
    if "lat_lon" in gdf.columns:
        gdf = gdf.drop(columns=["lat_lon"])
    
    # Also add a top-level metadata file or just save the GeoJSON
    gdf.to_file(geojson_output, driver="GeoJSON")
    
    import json
    metadata = {
        "center": [lat, lon],
        "radius": args.radius,
        "total_stalls": total,
        "features_count": len(gdf)
    }
    with open("parking_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"Data exported to {geojson_output} and parking_metadata.json")


if __name__ == "__main__":
    main()

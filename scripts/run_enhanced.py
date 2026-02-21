#!/usr/bin/env python3
"""
Enhanced parking pipeline: YOLO surface detection + structured parking + street parking.

Combines three data sources for comprehensive parking estimation:
  1. YOLO (or SegFormer) for surface lot detection from satellite imagery
  2. OSM building data for garage/underground structures
  3. OSM road geometry for on-street parking

Usage:
    python scripts/run_enhanced.py --address "Georgia Tech, Atlanta, GA"
    python scripts/run_enhanced.py --address "Georgia Tech, Atlanta, GA" --radius 500
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parksight import is_structure
from parksight.fetch import get_parking_data, get_satellite_tile, fetch_structured_parking, fetch_street_parking
from parksight.count import get_line_count
from parksight.estimate_structured import estimate_structured_parking, estimate_street_parking
from parksight.viz import create_parking_map


def run_surface_detection(gdf_3857, weights_path):
    """run yolo on surface lots only, skip structures"""
    from yolo.detect import YOLOParkingDetector
    detector = YOLOParkingDetector(weights_path)

    counts = []
    for _, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()

        # skip structures — they get handled by the structured estimator
        is_struct = is_structure(tags)

        if is_struct:
            counts.append(0)
            continue

        if geom.geom_type in ("Polygon", "MultiPolygon"):
            img = get_satellite_tile(geom)
            c = detector.count_spots(img, geom, osm_tags=tags)
            counts.append(c)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            c = get_line_count(geom)
            counts.append(c)
        else:
            counts.append(0)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Enhanced parking pipeline")
    parser.add_argument("--address", required=True, type=str)
    parser.add_argument("--radius", default=300, type=int)
    parser.add_argument("--weights", default="models/yolo26n_run1.pt", type=str)
    parser.add_argument("--output", default="parking_map_enhanced.html", type=str)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PARKSIGHT ENHANCED PIPELINE")
    print(f"  Address: {args.address}")
    print(f"  Radius:  {args.radius}m")
    print(f"{'='*60}\n")

    t_start = time.time()

    # === 1. surface lot detection ===
    print("[1/3] Surface lot detection (YOLO)...")
    gdf, (lat, lon) = get_parking_data(args.address, dist=args.radius)

    if gdf.empty:
        print("No parking features found.")
        return

    gdf_3857 = gdf.to_crs(epsg=3857)
    surface_counts = run_surface_detection(gdf_3857, args.weights)
    surface_total = sum(surface_counts)
    print(f"      Surface lots: {surface_total} spots\n")

    # === 2. structured parking (garages, underground) ===
    print("[2/3] Structured parking estimation (OSM)...")
    struct_gdf, _ = fetch_structured_parking(args.address, dist=args.radius)

    structured_results = []
    structured_total = 0
    if not struct_gdf.empty:
        structured_results = estimate_structured_parking(struct_gdf)
        structured_total = sum(r["total_spots"] for r in structured_results)

    for r in structured_results:
        print(f"      {r['name']} [{r['type']}]: {r['levels']} levels, {r['total_spots']} spots")

    if not structured_results:
        print("      No structured parking found in OSM data")
    print(f"      Structured total: {structured_total} spots\n")

    # === 3. street parking ===
    print("[3/3] Street parking estimation (OSM)...")
    street_gdf, _ = fetch_street_parking(args.address, dist=args.radius)

    street_results = []
    street_total = 0
    if not street_gdf.empty:
        street_results = estimate_street_parking(street_gdf)
        street_total = sum(r["total_spots"] for r in street_results)

    for r in street_results[:5]:  # show first 5
        print(f"      {r['name']}: {r['length_m']:.0f}m × {r['sides']} sides = {r['total_spots']} spots")
    if len(street_results) > 5:
        print(f"      ... and {len(street_results) - 5} more segments")

    if not street_results:
        print("      No street parking tags found in OSM data")
    print(f"      Street total: {street_total} spots\n")

    # === summary ===
    grand_total = surface_total + structured_total + street_total
    elapsed = time.time() - t_start

    print(f"{'='*60}")
    print(f"  RESULTS BREAKDOWN")
    print(f"{'='*60}")
    print(f"  {'Source':<30} {'Spots':>10}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Surface lots (YOLO)':<30} {surface_total:>10,}")
    print(f"  {'Garages/Underground (OSM)':<30} {structured_total:>10,}")
    print(f"  {'Street parking (OSM)':<30} {street_total:>10,}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'TOTAL':<30} {grand_total:>10,}")
    print(f"  {'Time elapsed':<30} {elapsed:>9.1f}s")
    print(f"{'='*60}\n")

    # === build map ===
    # merge structured counts back into the surface gdf
    gdf["count"] = surface_counts
    gdf["source"] = "surface"

    # for structures that exist in both gdf and struct_gdf, update the count
    for r in structured_results:
        idx = r["index"]
        if idx in gdf.index:
            gdf.loc[idx, "count"] = r["total_spots"]
            gdf.loc[idx, "source"] = r["type"]

    total_count = grand_total

    m = create_parking_map(lat, lon, gdf, radius=args.radius, total_count=total_count)

    # add structured parking features that aren't already in gdf
    if not struct_gdf.empty:
        import folium
        for r in structured_results:
            idx = r["index"]
            if idx not in gdf.index and idx in struct_gdf.index:
                geom = struct_gdf.loc[idx].geometry
                centroid_wgs = gpd.GeoSeries([geom.centroid], crs=struct_gdf.crs).to_crs(4326).iloc[0]
                color = "#ff9800" if r["type"] == "garage" else "#9c27b0"
                folium.CircleMarker(
                    location=(centroid_wgs.y, centroid_wgs.x),
                    radius=6,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=f"{r['name']} [{r['type']}]: {r['total_spots']} spots ({r['levels']} levels)",
                ).add_to(m)

    # add street parking segments
    if not street_gdf.empty and street_results:
        import folium
        street_wgs = street_gdf.to_crs(4326)
        for r in street_results:
            idx = r["index"]
            if idx in street_wgs.index:
                geom = street_wgs.loc[idx].geometry
                if geom.geom_type in ("LineString", "MultiLineString"):
                    coords = []
                    if geom.geom_type == "MultiLineString":
                        for line in geom.geoms:
                            coords.extend([(y, x) for x, y in line.coords])
                    else:
                        coords = [(y, x) for x, y in geom.coords]
                    folium.PolyLine(
                        locations=coords,
                        color="#4caf50",
                        weight=4,
                        opacity=0.8,
                        tooltip=f"{r['name']} [street]: {r['total_spots']} spots",
                    ).add_to(m)

    # add legend with breakdown
    import folium
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 20px; left: 20px; z-index: 9999;
        background: white; padding: 12px 16px;
        border: 2px solid grey; font-size: 13px;
        border-radius: 6px;
    ">
        <b>ParkSight Enhanced</b><br>
        🔴 Surface lots: {surface_total:,}<br>
        🟠 Garages/Underground: {structured_total:,}<br>
        🟢 Street parking: {street_total:,}<br>
        <b>Total: {grand_total:,}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(args.output)
    print(f"Map saved to: {args.output}")


if __name__ == "__main__":
    main()

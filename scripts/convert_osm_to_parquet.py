"""Convert raw Overpass JSON to GeoDataFrames (parquet) matching osmnx output format."""
import json
import os
import geopandas as gpd
from shapely.geometry import shape, Polygon, LineString, MultiPolygon, Point

def parse_overpass_json(filepath):
    """Parse raw Overpass JSON into a GeoDataFrame."""
    with open(filepath) as f:
        data = json.load(f)

    elements = data.get("elements", [])

    # Build node lookup
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    features = []
    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way":
            coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
            if len(coords) < 3:
                if len(coords) >= 2:
                    geom = LineString(coords)
                else:
                    continue
            else:
                if coords[0] == coords[-1]:
                    geom = Polygon(coords)
                else:
                    geom = LineString(coords)
            row = {**tags, "geometry": geom, "osmid": el["id"]}
            features.append(row)
        elif el["type"] == "relation":
            # Skip relations for now — they're complex multipolygons
            pass

    if not features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    return gdf

def main():
    os.makedirs("cache", exist_ok=True)

    for name in ["surface", "structured", "street"]:
        raw_path = f"cache/{name}_raw.json"
        parquet_path = f"cache/atlanta_{name}_parking.parquet"

        if not os.path.exists(raw_path):
            print(f"Skipping {name} — {raw_path} not found")
            continue

        if os.path.exists(parquet_path):
            print(f"Skipping {name} — {parquet_path} already exists")
            continue

        print(f"Converting {name}...")
        gdf = parse_overpass_json(raw_path)
        if gdf.empty:
            print(f"  No features found for {name}")
            # Save empty GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Convert object columns to string for parquet compatibility
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype == "object":
                gdf[col] = gdf[col].astype(str)

        gdf.to_parquet(parquet_path)
        print(f"  Saved {len(gdf)} features to {parquet_path}")

if __name__ == "__main__":
    main()

"""Convert raw Overpass JSON to GeoDataFrames (parquet) matching osmnx output format."""
import json
import os
import geopandas as gpd
from shapely.geometry import Polygon, LineString

def parse_overpass_json(filepath):
    """Parse raw Overpass JSON into a GeoDataFrame."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        print(f"  Skipping {filepath} — empty or missing")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    with open(filepath) as f:
        raw = f.read().strip()

    if not raw or not raw.startswith("{"):
        print(f"  Skipping {filepath} — not valid JSON")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    data = json.loads(raw)
    elements = data.get("elements", [])

    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    features = []
    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way":
            coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
            if len(coords) < 2:
                continue
            if len(coords) >= 3 and coords[0] == coords[-1]:
                geom = Polygon(coords)
            else:
                geom = LineString(coords)
            row = {**tags, "geometry": geom, "osmid": el["id"]}
            features.append(row)

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    return gdf

def main():
    os.makedirs("cache", exist_ok=True)

    for name in ["surface", "structured", "street"]:
        raw_path = f"cache/{name}_raw.json"
        parquet_path = f"cache/atlanta_{name}_parking.parquet"

        if os.path.exists(parquet_path):
            print(f"Skipping {name} — {parquet_path} already exists. Delete to reconvert.")
            continue

        print(f"Converting {name}...")
        gdf = parse_overpass_json(raw_path)

        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype == "object":
                gdf[col] = gdf[col].astype(str)

        gdf.to_parquet(parquet_path)
        print(f"  Saved {len(gdf)} features to {parquet_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Direct Polaris indexer for Klaus GT area.
Bypasses the slow macro/OSMnx pipeline — generates H3 hex cells,
assigns realistic parking data, and indexes straight into Actian.
"""
import asyncio
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3

from parksight.vector_search import index_hex_cells

# Klaus Advanced Computing Building, Georgia Tech
CENTER_LAT, CENTER_LON = 33.7771, -84.3963
BBOX = {
    "min_lat": 33.773, "max_lat": 33.782,
    "min_lon": -84.400, "max_lon": -84.391,
}
RESOLUTION = 10
LOCATION_HINT = "Georgia Tech campus near Klaus Advanced Computing Building, Atlanta GA"

# Realistic parking profiles for GT campus area
PROFILES = [
    {"surface": (20, 80), "structured": (100, 400), "street": (5, 30)},   # near garages
    {"surface": (5, 40),  "structured": (0, 0),     "street": (10, 40)},  # residential streets
    {"surface": (30, 120),"structured": (0, 50),     "street": (5, 20)},  # mixed
    {"surface": (0, 10),  "structured": (200, 600),  "street": (0, 10)},  # big deck
    {"surface": (10, 50), "structured": (0, 0),      "street": (15, 50)}, # surface + street
]


def generate_hex_cells():
    """Generate H3 hex cells covering the Klaus area with realistic parking data."""
    poly = h3.LatLngPoly([
        (BBOX["min_lat"], BBOX["min_lon"]),
        (BBOX["max_lat"], BBOX["min_lon"]),
        (BBOX["max_lat"], BBOX["max_lon"]),
        (BBOX["min_lat"], BBOX["max_lon"]),
    ])
    hexagons = list(h3.polygon_to_cells(poly, res=RESOLUTION))
    print(f"Generated {len(hexagons)} hex cells at resolution {RESOLUTION}")

    cells = []
    for hex_id in hexagons:
        lat, lon = h3.cell_to_latlng(hex_id)
        hex_boundary = h3.cell_to_boundary(hex_id)
        geojson_coords = [[[lo, la] for la, lo in hex_boundary]]
        geojson_coords[0].append(geojson_coords[0][0])

        # Pick a random parking profile and generate values
        profile = random.choice(PROFILES)
        surface = random.randint(*profile["surface"])
        structured = random.randint(*profile["structured"])
        street = random.randint(*profile["street"])

        # Cells closer to campus center get more structured parking
        dist_to_center = math.sqrt((lat - CENTER_LAT)**2 + (lon - CENTER_LON)**2)
        if dist_to_center < 0.002:  # very close to Klaus
            structured = max(structured, random.randint(150, 500))

        cells.append({
            "hex_id": hex_id,
            "centroid": [lat, lon],
            "total": surface + structured + street,
            "surface": surface,
            "structured": structured,
            "street": street,
            "geometry": {
                "type": "Polygon",
                "coordinates": geojson_coords,
            },
        })

    return cells


async def main():
    cells = generate_hex_cells()
    if not cells:
        print("ERROR: No hex cells generated")
        return

    print(f"Indexing {len(cells)} cells into Actian VectorAI DB...")
    print(f"Location hint: {LOCATION_HINT}")
    print("Each cell: reverse geocode + Gemini embedding + Actian upsert")
    print("This should take 1-3 minutes...\n")

    for i, cell in enumerate(cells):
        print(f"  [{i+1}/{len(cells)}] {cell['hex_id']} "
              f"({cell['centroid'][0]:.5f}, {cell['centroid'][1]:.5f}) "
              f"total={cell['total']}")

    count = await index_hex_cells(cells, location_hint=LOCATION_HINT)
    print(f"\nDone! Indexed {count} cells into Actian VectorAI DB.")


if __name__ == "__main__":
    asyncio.run(main())

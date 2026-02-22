#!/usr/bin/env python3
"""
generate_demo_overlay.py
------------------------
Build a single precomputed JSON file using the SAME logic as /api/estimate.

This script calls api.app.estimate() directly so demo cache generation stays
in sync with backend behavior (synthetic scan sizing, SegFormer fallback,
car-box overlays, and count sanity logic).

Usage:
    python scripts/generate_demo_overlay.py
    python scripts/generate_demo_overlay.py --lat 33.803 --lon -84.411 --radius 300

Output:
    public/precomputed/{lat:.3f}_{lon:.3f}_{radius}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _count_overlay_items(surface_features: list[dict]) -> tuple[int, int, int]:
    """Return (spot_boxes, car_boxes, segformer_polygons) totals."""
    spot_boxes = sum(len(f.get("spot_boxes", [])) for f in surface_features)
    car_boxes = sum(len(f.get("car_boxes", [])) for f in surface_features)
    segformer_polys = 0
    for feature in surface_features:
        contours = feature.get("segformer_contours")
        if contours and isinstance(contours, dict):
            segformer_polys += len(contours.get("coordinates", []))
    return spot_boxes, car_boxes, segformer_polys


def _default_output_path(lat: float, lon: float, radius: int) -> Path:
    cache_name = f"{lat:.3f}_{lon:.3f}_{radius}.json"
    return PROJECT_ROOT / "public" / "precomputed" / cache_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one demo precomputed JSON via /api/estimate logic")
    parser.add_argument("--lat", type=float, default=33.803)
    parser.add_argument("--lon", type=float, default=-84.411)
    parser.add_argument("--radius", type=int, default=300)
    parser.add_argument("--output", type=str, default="", help="Optional absolute/relative output file path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else _default_output_path(args.lat, args.lon, args.radius)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print("=" * 70)
    print("generate_demo_overlay.py")
    print(f"Target:  lat={args.lat}  lon={args.lon}  radius={args.radius}m")
    print(f"Output:  {output_path}")
    print("=" * 70)

    from api.app import estimate

    t0 = time.time()
    print("\n[1/2] Running api.app.estimate() ...")
    try:
        response = estimate(lat=args.lat, lon=args.lon, radius=args.radius)
    except Exception as exc:
        print(f"ERROR: estimate() failed: {exc}")
        return 1

    print("[2/2] Writing precomputed JSON ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    tmp_path.replace(output_path)

    surface = response.get("surface", {})
    structured = response.get("structured", {})
    street = response.get("street", {})
    surface_features = surface.get("features", [])

    synthetic_features = [f for f in surface_features if f.get("is_synthetic_scan")]
    segformer_available = any(f.get("segformer_available") for f in surface_features)
    spot_boxes, car_boxes, segformer_polys = _count_overlay_items(surface_features)

    print(f"\nSUCCESS: JSON written to {output_path}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Surface features      : {len(surface_features)}")
    print(f"  Structured features   : {len(structured.get('features', []))}")
    print(f"  Street features       : {len(street.get('features', []))}")
    print(f"  Surface total spots   : {surface.get('total', 0)}")
    print(f"  Structured total      : {structured.get('total', 0)}")
    print(f"  Street total          : {street.get('total', 0)}")
    print(f"  Grand total spots     : {response.get('grand_total', 0)}")
    print(f"  Total cars            : {response.get('cars', {}).get('value', 0)}")
    print(f"  Synthetic scan lots   : {len(synthetic_features)}")
    if synthetic_features:
        print(f"  Synthetic scan radius : {synthetic_features[0].get('scan_radius_m')}m")
    print(f"  SegFormer available   : {segformer_available}")
    print(f"  Overlay spot boxes    : {spot_boxes}")
    print(f"  Overlay car boxes     : {car_boxes}")
    print(f"  SegFormer polygons    : {segformer_polys}")
    print(f"  Elapsed               : {round(time.time() - t0, 2)}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

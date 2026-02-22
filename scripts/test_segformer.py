"""
Quick smoke test for the fine-tuned SegFormer model.

Loads the checkpoint from models/best_model/, runs inference on a
satellite tile fetched for a known parking lot, and prints the
segmentation result alongside the YOLO result for comparison.

Usage:
    python scripts/test_segformer.py [--lat 33.7756 --lon -84.3963 --radius 200]
    python scripts/test_segformer.py --save-masks          # saves individual + grid
    python scripts/test_segformer.py --max-examples 25     # up to 25 tiles in grid
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from parksight.fetch import get_parking_data_by_coords, get_satellite_tile
from parksight.segment import ParkingSegmenter
from parksight import is_structure


def _find_segformer_checkpoint() -> Path | None:
    """Match checkpoint search order used by the API."""
    candidates = [
        PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg-final" / "best_model",
        PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg" / "best_model",
        PROJECT_ROOT / "models" / "segformer_best",
        PROJECT_ROOT / "models" / "best_model",
        PROJECT_ROOT / "checkpoints" / "best_model",
    ]
    for path in candidates:
        if (
            path.exists()
            and (path / "config.json").exists()
            and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
            and (path / "preprocessor_config.json").exists()
        ):
            return path
    return None


def _load_spot_yolo():
    """Load parking-spot YOLO model for SegFormer comparison."""
    from yolo.detect import YOLOParkingDetector

    apklot = PROJECT_ROOT / "models" / "yolo_apklot_best.pt"
    parkseg = PROJECT_ROOT / "models" / "yolo26n_run1.pt"
    if apklot.exists():
        return YOLOParkingDetector(str(apklot), count_mode="detect"), "APKLOT", "detect"
    if parkseg.exists():
        return YOLOParkingDetector(str(parkseg), count_mode="area"), "ParkSeg", "area"
    return None, "none", "none"


def _add_label(img: Image.Image, text: str) -> Image.Image:
    """Return a copy of *img* with a label bar at the top."""
    BAR_H = 28
    labelled = Image.new("RGB", (img.width, img.height + BAR_H), (30, 30, 30))
    labelled.paste(img, (0, BAR_H))
    draw = ImageDraw.Draw(labelled)
    draw.text((4, 4), text, fill="white")
    return labelled


def _build_comparison_grid(rows, cols, tiles):
    """
    Build a grid image from *tiles*.

    Each element of *tiles* is a dict with keys:
        original, yolo_ann, seg_ann, name, yolo_count, seg_count
    The grid has 3 sub-columns per tile column (Original / YOLO / SegFormer).
    """
    if not tiles:
        return None

    # Resize every image to a common thumbnail size
    THUMB = 256
    n = len(tiles)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))

    LABEL_H = 28
    cell_w = THUMB * 3  # 3 images side-by-side per tile
    cell_h = THUMB + LABEL_H
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 20, 20))

    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x0 = c * cell_w
        y0 = r * cell_h

        resample = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", None))
        orig = t["original"].resize((THUMB, THUMB), resample)
        yolo = t["yolo_ann"].resize((THUMB, THUMB), resample)
        seg = t["seg_ann"].resize((THUMB, THUMB), resample)

        orig_l = _add_label(orig, f"{t['name']}")
        yolo_l = _add_label(yolo, f"YOLO: {t['yolo_count']} spots")
        seg_l  = _add_label(seg,  f"SegFormer: {t['seg_count']} spots")

        grid.paste(orig_l, (x0, y0))
        grid.paste(yolo_l, (x0 + THUMB, y0))
        grid.paste(seg_l,  (x0 + THUMB * 2, y0))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Test SegFormer parking segmentation")
    parser.add_argument("--lat", type=float, default=33.7756, help="Latitude (default: Georgia Tech)")
    parser.add_argument("--lon", type=float, default=-84.3963, help="Longitude")
    parser.add_argument("--radius", type=int, default=200, help="Search radius in metres")
    parser.add_argument("--save-masks", action="store_true", help="Save annotated images to disk")
    parser.add_argument("--max-examples", type=int, default=25, help="Max tiles to include in the comparison grid (default: 25)")
    args = parser.parse_args()

    model_path = _find_segformer_checkpoint()
    if model_path is None:
        print("ERROR: SegFormer checkpoint not found.")
        print("Checked:")
        print(f"  - {PROJECT_ROOT / 'checkpoints' / 'segformer-b5-parkseg-final' / 'best_model'}")
        print(f"  - {PROJECT_ROOT / 'checkpoints' / 'segformer-b5-parkseg' / 'best_model'}")
        print(f"  - {PROJECT_ROOT / 'checkpoints' / 'best_model'}")
        print(f"  - {PROJECT_ROOT / 'models' / 'segformer_best'}")
        print(f"  - {PROJECT_ROOT / 'models' / 'best_model'}")
        sys.exit(1)

    print(f"Loading SegFormer from {model_path} ...")
    segmenter = ParkingSegmenter(str(model_path))

    print(f"Fetching parking data near ({args.lat}, {args.lon}), radius={args.radius}m ...")
    gdf = get_parking_data_by_coords(args.lat, args.lon, dist=args.radius)

    if gdf is None or gdf.empty:
        print("No parking features found in this area.")
        sys.exit(0)

    gdf_3857 = gdf.to_crs(epsg=3857)

    # Also load YOLO spot detector for comparison if available
    yolo_detector, yolo_name, yolo_mode = _load_spot_yolo()
    if yolo_detector is not None:
        print(f"Loading YOLO spot detector ({yolo_name}, count_mode={yolo_mode}) for comparison ...")
    else:
        print("No YOLO spot weights found (yolo_apklot_best.pt or yolo26n_run1.pt).")

    print(f"\nFound {len(gdf_3857)} parking features. Processing surface lots...\n")
    print(f"{'#':<4} {'Name':<30} {'Area m²':<10} {'SegFormer':<12} {'Method':<8} {'YOLO':<8}")
    print("-" * 80)

    total_seg = 0
    total_yolo = 0
    comparison_tiles = []  # collect for grid

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()

        if is_structure(tags) or geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue

        img = get_satellite_tile(geom)
        area_m2 = round(geom.area, 1)

        # SegFormer
        result = segmenter.count_spots(img)
        seg_count = result.count
        total_seg += seg_count

        # YOLO
        yolo_count = "-"
        yolo_detections = None
        if yolo_detector is not None:
            yolo_detections = yolo_detector.detect(img)
            yolo_count = yolo_detector.count_spots(img, geom, osm_tags=tags)
            total_yolo += yolo_count

        name = tags.get("name", None)
        if not isinstance(name, str) or not name:
            name = f"Lot #{idx}"
        if len(name) > 28:
            name = name[:28] + ".."

        print(f"{str(idx):<4} {name:<30} {area_m2:<10} {seg_count:<12} {result.method:<8} {str(yolo_count):<8}")

        # --- save individual side-by-side images and collect for grid ---
        if args.save_masks:
            seg_ann = segmenter.annotate(img, result.mask)

            if yolo_detector is not None and yolo_detections is not None:
                yolo_ann = yolo_detector.annotate(img, yolo_detections)
            else:
                yolo_ann = img.copy()

            # Individual side-by-side: Original | YOLO | SegFormer
            w, h = img.size
            side_by_side = Image.new("RGB", (w * 3, h + 28), (20, 20, 20))
            side_by_side.paste(_add_label(img, "Original"), (0, 0))
            side_by_side.paste(_add_label(yolo_ann, f"YOLO: {yolo_count}"), (w, 0))
            side_by_side.paste(_add_label(seg_ann, f"SegFormer: {seg_count}"), (w * 2, 0))

            out_dir = PROJECT_ROOT / "cache"
            out_dir.mkdir(exist_ok=True)
            side_by_side.save(str(out_dir / f"comparison_{idx}.png"))

            # Collect for summary grid (up to max_examples)
            if len(comparison_tiles) < args.max_examples:
                comparison_tiles.append({
                    "original": img,
                    "yolo_ann": yolo_ann,
                    "seg_ann": seg_ann,
                    "name": name,
                    "yolo_count": yolo_count,
                    "seg_count": seg_count,
                })

    print("-" * 80)
    print(f"{'TOTAL':<44} {total_seg:<12} {'':8} {total_yolo:<8}")

    if args.save_masks:
        out_dir = PROJECT_ROOT / "cache"
        n = len(comparison_tiles)
        print(f"\nSide-by-side images saved to {out_dir / 'comparison_*.png'}")

        if comparison_tiles:
            # Build a 5-column summary grid
            grid = _build_comparison_grid(rows=5, cols=5, tiles=comparison_tiles)
            if grid is not None:
                grid_path = out_dir / "comparison_grid.png"
                grid.save(str(grid_path))
                print(f"Summary grid ({n} tiles, Original|YOLO|SegFormer) saved to {grid_path}")


if __name__ == "__main__":
    main()

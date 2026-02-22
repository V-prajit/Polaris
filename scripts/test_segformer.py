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
import os
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from parksight.fetch import get_parking_data_by_coords, get_satellite_tile
from parksight.segment import ParkingSegmenter
from parksight import is_structure


def _is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as f:
            head = f.read(256)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _iter_segformer_candidates() -> list[Path]:
    env_ckpt = os.getenv("SEGFORMER_CKPT", "").strip()
    candidates: list[Path] = []
    if env_ckpt:
        candidates.append(Path(env_ckpt))

    candidates.extend([
        PROJECT_ROOT / "models" / "best_model",
        PROJECT_ROOT / "models" / "segformer_best",
        PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg-final" / "best_model",
        PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg" / "best_model",
        PROJECT_ROOT / "checkpoints" / "best_model",
    ])

    for root in [PROJECT_ROOT / "models", PROJECT_ROOT / "checkpoints"]:
        if not root.exists():
            continue
        for match in sorted(root.glob("*segformer*")):
            if match.is_dir():
                candidates.append(match)
                nested_best = match / "best_model"
                if nested_best.is_dir():
                    candidates.append(nested_best)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _checkpoint_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing dir"
    if not path.is_dir():
        return False, "not a directory"
    if not (path / "config.json").exists():
        return False, "missing config.json"

    safetensors_path = path / "model.safetensors"
    pytorch_path = path / "pytorch_model.bin"
    if not (safetensors_path.exists() or pytorch_path.exists()):
        return False, "missing model weights"

    if safetensors_path.exists():
        if _is_git_lfs_pointer(safetensors_path):
            return False, "model.safetensors is a Git LFS pointer"
        try:
            size = safetensors_path.stat().st_size
            if size < 16:
                return False, "model.safetensors too small"
            with safetensors_path.open("rb") as f:
                header_raw = f.read(8)
            header_len = struct.unpack("<Q", header_raw)[0]
            if header_len <= 0 or header_len > 128 * 1024 * 1024 or header_len + 8 > size:
                return False, f"bad safetensors header len={header_len}"
        except Exception as exc:
            return False, f"safetensors header read failed: {exc}"

    # preprocessor_config.json is optional; ParkingSegmenter falls back to defaults.
    return True, "usable"


def _find_segformer_checkpoint() -> tuple[Path | None, list[tuple[Path, bool, str]]]:
    report: list[tuple[Path, bool, str]] = []
    for path in _iter_segformer_candidates():
        ok, reason = _checkpoint_status(path)
        report.append((path, ok, reason))
        if ok:
            return path, report
    return None, report


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

    model_path, ckpt_report = _find_segformer_checkpoint()
    print("SegFormer checkpoint diagnostics:")
    for path, ok, reason in ckpt_report:
        marker = "OK" if ok else "SKIP"
        print(f"  [{marker}] {path} ({reason})")

    if model_path is None:
        print("ERROR: SegFormer checkpoint not found.")
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

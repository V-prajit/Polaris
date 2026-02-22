#!/usr/bin/env python3
"""
generate_demo_overlay.py
------------------------
Standalone diagnostic script that generates ONE precomputed JSON file with
full detection overlay data (car_boxes, spot_boxes, segformer_contours) for
Atlantic Station (33.803, -84.411, radius=300).

Run this DIRECTLY on the H200 to either:
  (a) produce the file successfully, or
  (b) identify exactly which stage is crashing in the /api/estimate endpoint.

Each pipeline stage is wrapped in try/except with detailed error output.
A fallback path generates the basic precomputed file (no overlays) if the
new *_with_boxes methods fail, so we always get at least a valid JSON output.

Usage:
    python scripts/generate_demo_overlay.py

Output:
    public/precomputed/33.803_-84.411_300.json
"""

import sys
import os
import json
import math
import time
import traceback
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Demo target ─────────────────────────────────────────────────────────────
TARGET_LAT    = 33.803
TARGET_LON    = -84.411
TARGET_RADIUS = 300
OUTPUT_PATH   = PROJECT_ROOT / "public" / "precomputed" / f"{TARGET_LAT}_{TARGET_LON}_{TARGET_RADIUS}.json"

# ── Helpers copied verbatim from api/app.py ──────────────────────────────────

def _geometry_to_coords(geom):
    """Convert a shapely geometry to a serialisable coordinate dict."""
    if geom.geom_type == "Point":
        return {"type": "Point", "coordinates": [geom.x, geom.y]}
    elif geom.geom_type == "Polygon":
        return {"type": "Polygon", "coordinates": [list(geom.exterior.coords)]}
    elif geom.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [[list(poly.exterior.coords)] for poly in geom.geoms],
        }
    elif geom.geom_type == "LineString":
        return {"type": "LineString", "coordinates": list(geom.coords)}
    elif geom.geom_type == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": [list(line.coords) for line in geom.geoms],
        }
    return {"type": geom.geom_type, "coordinates": []}


def _sanitize_dict(d):
    """Recursively convert NaN values to None for JSON serialisation."""
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(v) for v in d]
    elif isinstance(d, float) and math.isnan(d):
        return None
    return d


def _pixel_boxes_to_wgs84(boxes_pixel, geom_3857, img_size, padding_pct=0.10):
    """Convert pixel-space bounding boxes to WGS84 lat/lon rectangles.

    Matches the implementation in api/app.py exactly.

    Parameters
    ----------
    boxes_pixel : list of (x1, y1, x2, y2, conf)
    geom_3857   : shapely geometry in EPSG:3857
    img_size    : (width, height) — PIL Image.size
    padding_pct : padding fraction used when the tile was fetched

    Returns
    -------
    list of {"bbox": [lat1, lon1, lat2, lon2], "conf": float}
    """
    if not boxes_pixel:
        return []

    import pyproj

    minx, miny, maxx, maxy = geom_3857.bounds
    width  = maxx - minx
    height = maxy - miny
    pad_x  = width  * padding_pct
    pad_y  = height * padding_pct
    tile_minx = minx - pad_x
    tile_miny = miny - pad_y
    tile_maxx = maxx + pad_x
    tile_maxy = maxy + pad_y
    img_w, img_h = img_size
    m_per_px_x = (tile_maxx - tile_minx) / img_w
    m_per_px_y = (tile_maxy - tile_miny) / img_h

    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    wgs84_boxes = []
    for x1, y1, x2, y2, conf in boxes_pixel:
        geo_x1 = tile_minx + x1 * m_per_px_x
        geo_x2 = tile_minx + x2 * m_per_px_x
        # y-axis is flipped: pixel y=0 is at the top (north) of the tile
        geo_y1 = tile_maxy - y2 * m_per_px_y   # south edge of box
        geo_y2 = tile_maxy - y1 * m_per_px_y   # north edge of box
        lon1, lat1 = transformer.transform(geo_x1, geo_y1)   # SW corner
        lon2, lat2 = transformer.transform(geo_x2, geo_y2)   # NE corner
        wgs84_boxes.append({"bbox": [lat1, lon1, lat2, lon2], "conf": round(float(conf), 3)})

    return wgs84_boxes


def _pixel_contours_to_wgs84(contours, geom_3857, img_size, padding_pct=0.10):
    """Convert pixel-space contours to a WGS84 GeoJSON MultiPolygon.

    Matches the implementation in api/app.py exactly.

    Parameters
    ----------
    contours  : list of numpy.ndarray, each shape (N, 2) with [x, y] columns
    geom_3857 : shapely geometry in EPSG:3857
    img_size  : (width, height)
    padding_pct : padding fraction used when the tile was fetched

    Returns
    -------
    dict or None — GeoJSON MultiPolygon, or None if no valid contours.
    """
    if not contours:
        return None

    import pyproj

    minx, miny, maxx, maxy = geom_3857.bounds
    width  = maxx - minx
    height = maxy - miny
    pad_x  = width  * padding_pct
    pad_y  = height * padding_pct
    tile_minx = minx - pad_x
    tile_miny = miny - pad_y
    tile_maxx = maxx + pad_x
    tile_maxy = maxy + pad_y
    img_w, img_h = img_size
    m_per_px_x = (tile_maxx - tile_minx) / img_w
    m_per_px_y = (tile_maxy - tile_miny) / img_h

    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        ring = []
        for pt in contour:
            x, y = float(pt[0]), float(pt[1])
            geo_x = tile_minx + x * m_per_px_x
            geo_y = tile_maxy - y * m_per_px_y
            lon, lat = transformer.transform(geo_x, geo_y)
            ring.append([lon, lat])
        # GeoJSON rings must be closed
        ring.append(ring[0])
        polygons.append([ring])

    if not polygons:
        return None

    return {"type": "MultiPolygon", "coordinates": polygons}


# ── Model loading ────────────────────────────────────────────────────────────

def load_models():
    """Load detector, segmenter, and car_detector — same logic as precompute_atlanta.py."""
    detector     = None
    segmenter    = None
    car_detector = None

    # Parking-spot detector
    apklot_weights  = PROJECT_ROOT / "models" / "yolo_apklot_best.pt"
    parkseg_weights = PROJECT_ROOT / "models" / "yolo26n_run1.pt"

    try:
        from yolo.detect import YOLOParkingDetector
        if apklot_weights.exists():
            print(f"[models] Loading APKLOT YOLO from {apklot_weights} ...")
            detector = YOLOParkingDetector(str(apklot_weights), count_mode="detect")
            print(f"[models] Spot detector loaded (count_mode={detector.count_mode})")
        elif parkseg_weights.exists():
            print(f"[models] Loading ParkSeg YOLO from {parkseg_weights} ...")
            detector = YOLOParkingDetector(str(parkseg_weights), count_mode="area")
            print(f"[models] Spot detector loaded (count_mode={detector.count_mode})")
        else:
            print("[models] WARNING: no spot-detector weights found — skipping spot detection")
    except Exception as exc:
        print(f"[models] ERROR loading spot detector: {exc}")
        traceback.print_exc()

    # Car detector
    car_weights = PROJECT_ROOT / "models" / "yolo_aerial_cars.pt"
    try:
        from yolo.detect import YOLOParkingDetector
        if car_weights.exists():
            print(f"[models] Loading car detector from {car_weights} ...")
            car_detector = YOLOParkingDetector(str(car_weights), count_mode="detect")
            print("[models] Car detector loaded")
        else:
            print("[models] WARNING: yolo_aerial_cars.pt not found — skipping car detection")
    except Exception as exc:
        print(f"[models] ERROR loading car detector: {exc}")
        traceback.print_exc()

    # SegFormer segmenter
    ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg-final" / "best_model"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "segformer-b5-parkseg" / "best_model"
    try:
        if ckpt.exists():
            from parksight.segment import ParkingSegmenter
            print(f"[models] Loading SegFormer from {ckpt} ...")
            segmenter = ParkingSegmenter(str(ckpt))
            print("[models] SegFormer loaded")
        else:
            print("[models] WARNING: no SegFormer checkpoint found — skipping segmentation")
    except Exception as exc:
        print(f"[models] ERROR loading SegFormer: {exc}")
        traceback.print_exc()

    return detector, segmenter, car_detector


# ── Surface-detection pipeline ───────────────────────────────────────────────

def run_surface_detection_with_overlays(gdf_3857, detector, segmenter, car_detector):
    """Run the full pipeline for each surface feature.

    Each stage is individually guarded:
      - If count_spots_with_boxes / count_cars_with_boxes crash, falls back to
        count_spots / count_cars and returns empty overlay arrays.
      - If count_spots / count_cars also crash, falls back to the area estimate.
    """
    import geopandas as gpd
    from parksight import is_structure, config as _cfg
    from parksight.fetch import get_satellite_tile
    from parksight.count import get_line_count
    from parksight.confidence import confidence_band, utilization_band

    stall_area = _cfg["STALL_AREA_M2"]
    usable     = _cfg["USABLE_FRACTION_SURFACE"]

    results = []

    total_features = len(gdf_3857)
    print(f"\n[surface] Processing {total_features} feature(s) from OSM ...")

    for idx, row in gdf_3857.iterrows():
        geom = row.geometry
        tags = row.to_dict()
        is_struct = is_structure(tags)

        if is_struct:
            print(f"  [{idx}] Skipping structure: {tags.get('name', idx)}")
            continue

        print(f"\n  [{idx}] Feature: {tags.get('name', f'Surface #{idx}')}  geom_type={geom.geom_type}")

        count          = 0
        count_area     = 0
        count_yolo     = 0
        count_segformer = 0
        cars           = 0
        spot_method    = "area"

        # Overlay containers (pixel-space, converted to WGS84 below)
        spot_boxes_px  = []   # list of (x1,y1,x2,y2,conf)
        car_boxes_px   = []   # list of (x1,y1,x2,y2,conf)
        seg_contours   = []   # list of (N,2) numpy arrays
        img            = None

        # ── Polygon / MultiPolygon features: run imagery pipeline ──────────
        if geom.geom_type in ("Polygon", "MultiPolygon"):

            # Stage A: Fetch satellite tile
            print(f"    [A] Fetching satellite tile ...")
            try:
                img = get_satellite_tile(geom)
                print(f"    [A] Tile fetched: size={img.size}")
            except Exception as exc:
                print(f"    [A] ERROR fetching tile: {exc}")
                traceback.print_exc()
                # Without an image we cannot run any ML stages
                img = None

            count_area = int((geom.area / stall_area) * usable)
            print(f"    [A] Area-based estimate: {count_area} spots")

            if img is not None:

                # Stage B: SegFormer segmentation + contours
                seg_mask = None
                if segmenter is not None:
                    print(f"    [B] Running SegFormer segmentation ...")
                    try:
                        seg_mask = segmenter.segment(img)
                        seg_result = segmenter.count_spots(img)
                        count_segformer = seg_result.count
                        print(f"    [B] SegFormer count: {count_segformer}  mask_coverage={seg_mask.mean():.3f}")
                    except Exception as exc:
                        print(f"    [B] ERROR in segmenter.segment/count_spots: {exc}")
                        traceback.print_exc()
                        seg_mask = None
                        count_segformer = 0

                    # Stage B2: Extract contours from the mask
                    print(f"    [B2] Extracting SegFormer contours ...")
                    try:
                        seg_contours = segmenter.segment_to_contours(img)
                        print(f"    [B2] Contours extracted: {len(seg_contours)} polygon(s)")
                    except Exception as exc:
                        print(f"    [B2] ERROR in segmenter.segment_to_contours: {exc}")
                        traceback.print_exc()
                        seg_contours = []

                # Stage C: YOLO spot detection — try with_boxes first, then plain
                if detector is not None:
                    print(f"    [C] Running YOLO spot detector (with_boxes) ...")
                    try:
                        count_yolo, spot_boxes_px = detector.count_spots_with_boxes(
                            img, geom, osm_tags=tags, segformer_mask=seg_mask
                        )
                        spot_method = (
                            "yolo_detect" if detector.count_mode == "detect" else "segformer"
                        )
                        count = count_yolo
                        print(
                            f"    [C] count_spots_with_boxes OK: "
                            f"count={count_yolo}  boxes={len(spot_boxes_px)}"
                        )
                    except Exception as exc:
                        print(f"    [C] ERROR in count_spots_with_boxes: {exc}")
                        traceback.print_exc()
                        # Fallback: plain count_spots (no boxes)
                        print(f"    [C] Falling back to count_spots (no boxes) ...")
                        try:
                            count_yolo = detector.count_spots(
                                img, geom, osm_tags=tags, segformer_mask=seg_mask
                            )
                            spot_method = (
                                "yolo_detect" if detector.count_mode == "detect" else "segformer"
                            )
                            count = count_yolo
                            spot_boxes_px = []   # empty — overlay unavailable
                            print(f"    [C] Fallback count_spots OK: count={count_yolo}")
                        except Exception as exc2:
                            print(f"    [C] ERROR in fallback count_spots: {exc2}")
                            traceback.print_exc()
                            count       = count_segformer if count_segformer else count_area
                            spot_method = "segformer" if count_segformer else "area"

                elif seg_mask is not None:
                    spot_method = "segformer"
                    count = count_segformer
                else:
                    spot_method = "area"
                    count = count_area

                # Stage D: Car counting — try with_boxes first, then plain
                if car_detector is not None:
                    print(f"    [D] Running car detector (with_boxes) ...")
                    try:
                        cars, car_boxes_px = car_detector.count_cars_with_boxes(
                            img, segformer_mask=seg_mask
                        )
                        print(
                            f"    [D] count_cars_with_boxes OK: "
                            f"count={cars}  boxes={len(car_boxes_px)}"
                        )
                    except Exception as exc:
                        print(f"    [D] ERROR in count_cars_with_boxes: {exc}")
                        traceback.print_exc()
                        # Fallback: plain count_cars (no boxes)
                        print(f"    [D] Falling back to count_cars (no boxes) ...")
                        try:
                            cars = car_detector.count_cars(img, segformer_mask=seg_mask)
                            car_boxes_px = []   # empty — overlay unavailable
                            print(f"    [D] Fallback count_cars OK: count={cars}")
                        except Exception as exc2:
                            print(f"    [D] ERROR in fallback count_cars: {exc2}")
                            traceback.print_exc()
                            cars = 0
                            car_boxes_px = []

        # ── LineString features: line-based count ───────────────────────────
        elif geom.geom_type in ("LineString", "MultiLineString"):
            count       = get_line_count(geom)
            count_area  = count
            spot_method = "street"
            print(f"    Line-based count: {count} spots")

        # Skip zero-count features
        if count <= 0 and count_area <= 0:
            print(f"    Skipping (count=0, count_area=0)")
            continue

        # Stage E: Confidence bands
        spot_band   = confidence_band(count,  method=spot_method)
        car_band    = confidence_band(cars,   method="yolo_car")
        utilization = utilization_band(car_band, spot_band)

        # Stage F: Convert geometry to WGS84
        geom_wgs    = gpd.GeoSeries([geom], crs=3857).to_crs(4326).iloc[0]
        centroid_wgs = geom_wgs.centroid
        tb           = geom_wgs.bounds   # (west, south, east, north)
        tile_bounds  = [tb[1], tb[0], tb[3], tb[2]]   # [S, W, N, E]

        # Stage G: Convert pixel-space overlays to WGS84
        spot_boxes_wgs84       = []
        car_boxes_wgs84        = []
        segformer_contours_wgs84 = None

        if img is not None:
            img_size = img.size
            print(f"    [G] Converting pixel overlays to WGS84 (img_size={img_size}) ...")

            # Spot boxes
            try:
                spot_boxes_wgs84 = _pixel_boxes_to_wgs84(spot_boxes_px, geom, img_size)
                print(f"    [G] spot_boxes_wgs84: {len(spot_boxes_wgs84)} box(es)")
            except Exception as exc:
                print(f"    [G] ERROR converting spot boxes to WGS84: {exc}")
                traceback.print_exc()
                spot_boxes_wgs84 = []

            # Car boxes
            try:
                car_boxes_wgs84 = _pixel_boxes_to_wgs84(car_boxes_px, geom, img_size)
                print(f"    [G] car_boxes_wgs84: {len(car_boxes_wgs84)} box(es)")
            except Exception as exc:
                print(f"    [G] ERROR converting car boxes to WGS84: {exc}")
                traceback.print_exc()
                car_boxes_wgs84 = []

            # SegFormer contours
            try:
                segformer_contours_wgs84 = _pixel_contours_to_wgs84(
                    seg_contours, geom, img_size
                )
                n_poly = (
                    len(segformer_contours_wgs84.get("coordinates", []))
                    if segformer_contours_wgs84
                    else 0
                )
                print(f"    [G] segformer_contours_wgs84: {n_poly} polygon(s)")
            except Exception as exc:
                print(f"    [G] ERROR converting contours to WGS84: {exc}")
                traceback.print_exc()
                segformer_contours_wgs84 = None

        results.append({
            "name":              tags.get("name", f"Surface #{idx}"),
            "type":              "surface",
            "count":             count,
            "count_area":        count_area,
            "count_yolo":        count_yolo,
            "count_segformer":   count_segformer,
            "spots":             spot_band.to_dict(),
            "cars":              car_band.to_dict(),
            "utilization":       utilization,
            "centroid":          [centroid_wgs.y, centroid_wgs.x],
            "geometry":          _geometry_to_coords(geom_wgs),
            # Detection overlay fields
            "spot_boxes":              spot_boxes_wgs84,
            "car_boxes":               car_boxes_wgs84,
            "segformer_contours":      segformer_contours_wgs84,
            "tile_bounds":             tile_bounds,
        })

        print(f"    Feature done: count={count}  cars={cars}  method={spot_method}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("=" * 70)
    print(f"generate_demo_overlay.py")
    print(f"Target:  lat={TARGET_LAT}  lon={TARGET_LON}  radius={TARGET_RADIUS}m")
    print(f"Output:  {OUTPUT_PATH}")
    print("=" * 70)

    # ── Step 1: Load models ─────────────────────────────────────────────────
    print("\n[1/5] Loading models ...")
    detector, segmenter, car_detector = load_models()

    # ── Step 2: Fetch OSM data ───────────────────────────────────────────────
    print(f"\n[2/5] Fetching OSM parking data for ({TARGET_LAT}, {TARGET_LON}, {TARGET_RADIUS}m) ...")
    try:
        from parksight.fetch import (
            get_parking_data_by_coords,
            fetch_structured_parking_by_coords,
            fetch_street_parking_by_coords,
        )
        gdf = get_parking_data_by_coords(TARGET_LAT, TARGET_LON, dist=TARGET_RADIUS)
        print(f"  Surface OSM features: {len(gdf) if gdf is not None and not gdf.empty else 0}")
    except Exception as exc:
        print(f"  ERROR fetching OSM data: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 2b: Check for polygon features; create synthetic if none ──────
    import geopandas as gpd
    from shapely.geometry import Point
    import pyproj

    has_polygons = False
    if gdf is not None and not gdf.empty:
        gdf_3857_check = gdf.to_crs(epsg=3857)
        polygon_mask = gdf_3857_check.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        has_polygons = polygon_mask.any()
        print(f"  Polygon/MultiPolygon features in OSM: {polygon_mask.sum()}")

    if not has_polygons:
        print(f"\n  *** No surface parking polygons in OSM at this location ***")
        print(f"  Creating synthetic scan polygon ({TARGET_RADIUS}m radius) for YOLO detection ...")

        # Convert lat/lon to EPSG:3857 meters
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        cx, cy = transformer.transform(TARGET_LON, TARGET_LAT)
        synthetic_geom = Point(cx, cy).buffer(TARGET_RADIUS)  # circle in meters

        # Build a minimal GeoDataFrame with the synthetic polygon
        synthetic_gdf = gpd.GeoDataFrame(
            [{"name": f"Scan Area ({TARGET_LAT}, {TARGET_LON})", "amenity": "parking", "parking": "surface"}],
            geometry=[synthetic_geom],
            crs="EPSG:3857",
        )

        if gdf is not None and not gdf.empty:
            # Merge: keep original features AND add synthetic polygon
            gdf_3857_orig = gdf.to_crs(epsg=3857)
            import pandas as pd
            gdf = pd.concat([gdf_3857_orig, synthetic_gdf], ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:3857")
        else:
            gdf = synthetic_gdf

        print(f"  Synthetic polygon added. Total features: {len(gdf)}")
        # Mark that gdf is already in EPSG:3857
        _gdf_already_3857 = True
    else:
        _gdf_already_3857 = False

    # ── Step 3: Surface detection pipeline ──────────────────────────────────
    print("\n[3/5] Running surface detection pipeline ...")
    surface_features = []
    surface_total    = 0

    if gdf is not None and not gdf.empty:
        try:
            gdf_3857 = gdf if _gdf_already_3857 else gdf.to_crs(epsg=3857)
            surface_features = run_surface_detection_with_overlays(
                gdf_3857, detector, segmenter, car_detector
            )
            surface_total = sum(f["count"] for f in surface_features)
            print(f"\n[3/5] Surface pipeline complete: {len(surface_features)} feature(s), "
                  f"{surface_total} total spots")
        except Exception as exc:
            print(f"\n[3/5] UNHANDLED ERROR in surface detection pipeline: {exc}")
            traceback.print_exc()
    else:
        print("  No surface features found.")

    # ── Step 4: Structured and street parking ────────────────────────────────
    print("\n[4/5] Fetching structured and street parking ...")

    import geopandas as gpd
    from parksight.estimate_structured import estimate_structured_parking, estimate_street_parking
    from parksight.confidence import confidence_band, utilization_band

    # Structured
    structured_features = []
    structured_total    = 0
    try:
        struct_gdf = fetch_structured_parking_by_coords(TARGET_LAT, TARGET_LON, dist=TARGET_RADIUS)
        print(f"  Structured features: {len(struct_gdf) if not struct_gdf.empty else 0}")
        if not struct_gdf.empty:
            raw = estimate_structured_parking(struct_gdf)
            structured_total = sum(r["total_spots"] for r in raw)
            struct_3857 = struct_gdf.to_crs(epsg=3857) if struct_gdf.crs != "EPSG:3857" else struct_gdf
            struct_wgs  = struct_3857.to_crs(4326)
            for r in raw:
                idx = r["index"]
                if idx in struct_wgs.index:
                    geom_wgs = struct_wgs.loc[idx].geometry
                    centroid = geom_wgs.centroid
                    method   = r["type"]
                    spot_band = confidence_band(r["total_spots"], method=method)
                    structured_features.append({
                        "name":         r["name"],
                        "type":         r["type"],
                        "count":        r["total_spots"],
                        "spots":        spot_band.to_dict(),
                        "levels":       r["levels"],
                        "floor_area_m2": r["floor_area_m2"],
                        "centroid":     [centroid.y, centroid.x],
                        "geometry":     _geometry_to_coords(geom_wgs),
                    })
    except Exception as exc:
        print(f"  ERROR fetching structured parking: {exc}")
        traceback.print_exc()

    # Street
    street_features = []
    street_total    = 0
    try:
        street_gdf = fetch_street_parking_by_coords(TARGET_LAT, TARGET_LON, dist=TARGET_RADIUS)
        print(f"  Street features: {len(street_gdf) if not street_gdf.empty else 0}")
        if not street_gdf.empty:
            raw = estimate_street_parking(street_gdf)
            street_total = sum(r["total_spots"] for r in raw)
            street_3857  = street_gdf.to_crs(epsg=3857) if street_gdf.crs != "EPSG:3857" else street_gdf
            street_wgs   = street_3857.to_crs(4326)
            for r in raw:
                idx = r["index"]
                if idx in street_wgs.index:
                    geom_wgs = street_wgs.loc[idx].geometry
                    centroid = geom_wgs.centroid
                    spot_band = confidence_band(r["total_spots"], method="street")
                    street_features.append({
                        "name":     r["name"],
                        "type":     "street",
                        "count":    r["total_spots"],
                        "spots":    spot_band.to_dict(),
                        "length_m": r["length_m"],
                        "sides":    r["sides"],
                        "centroid": [centroid.y, centroid.x],
                        "geometry": _geometry_to_coords(geom_wgs),
                    })
    except Exception as exc:
        print(f"  ERROR fetching street parking: {exc}")
        traceback.print_exc()

    # ── Step 5: Build response and write JSON ─────────────────────────────────
    print("\n[5/5] Building response and writing JSON ...")

    grand_total = surface_total + structured_total + street_total
    total_cars  = sum(f.get("cars", {}).get("value", 0) for f in surface_features)
    elapsed     = round(time.time() - t_total, 2)

    grand_spot_band  = confidence_band(grand_total, method="default")
    grand_car_band   = confidence_band(total_cars,  method="yolo_car")
    grand_utilization = utilization_band(grand_car_band, grand_spot_band)

    response = _sanitize_dict({
        "lat":    TARGET_LAT,
        "lon":    TARGET_LON,
        "radius": TARGET_RADIUS,
        "surface": {
            "total":    surface_total,
            "features": surface_features,
        },
        "structured": {
            "total":    structured_total,
            "features": structured_features,
        },
        "street": {
            "total":    street_total,
            "features": street_features,
        },
        "grand_total": grand_total,
        "spots":       grand_spot_band.to_dict(),
        "cars":        grand_car_band.to_dict(),
        "utilization": grand_utilization,
        "elapsed_seconds": elapsed,
    })

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically via a temp file, matching precompute_atlanta.py pattern
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(response, f, indent=2)
        os.rename(tmp_path, OUTPUT_PATH)
        print(f"\nSUCCESS: JSON written to {OUTPUT_PATH}")
    except Exception as exc:
        print(f"\nERROR writing output file: {exc}")
        traceback.print_exc()
        if tmp_path.exists():
            tmp_path.unlink()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Surface features : {len(surface_features)}")
    print(f"  Structured       : {len(structured_features)}")
    print(f"  Street           : {len(street_features)}")
    print(f"  Grand total spots: {grand_total}")
    print(f"  Total cars       : {total_cars}")
    print(f"  Elapsed          : {elapsed}s")

    # Report overlay coverage so we can verify it worked
    n_spot_boxes = sum(len(f.get("spot_boxes", [])) for f in surface_features)
    n_car_boxes  = sum(len(f.get("car_boxes",  [])) for f in surface_features)
    n_contours   = sum(
        len(f["segformer_contours"].get("coordinates", []))
        if f.get("segformer_contours")
        else 0
        for f in surface_features
    )
    print(f"  Overlay spot boxes    : {n_spot_boxes} total across all features")
    print(f"  Overlay car boxes     : {n_car_boxes} total across all features")
    print(f"  SegFormer contours    : {n_contours} polygon(s) total")
    print("=" * 70)


if __name__ == "__main__":
    main()

import torch
from shapely.geometry import box as shapely_box
from ultralytics import YOLO
from PIL import Image, ImageDraw

from parksight import config, pick_device, is_structure


class YOLOParkingDetector:
    def __init__(self, weights_path, device=None):
        self.device = device or pick_device()
        self.model = YOLO(weights_path)
        self.model.to(self.device)

    def detect(self, pil_image, confidence=None):
        conf = confidence if confidence is not None else config["MIN_CONFIDENCE"]
        results = self.model(pil_image, conf=conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            detections.append((xyxy, conf_score, cls_name))

        return detections

    def count_spots(self, pil_image, geometry, osm_tags=None):
        tags = osm_tags or {}
        stall_area = config["STALL_AREA_M2"]
        usable_fraction = config["USABLE_FRACTION_SURFACE"]

        # 1. direct from osm capacity tag
        if "capacity" in tags:
            try:
                return int(float(tags["capacity"]))
            except ValueError:
                pass

        # 2. floor multiplier for garages
        floor_multiplier = 1
        is_struct = is_structure(tags)

        if is_struct:
            try:
                floor_multiplier = int(float(tags.get("building:levels", 3)))
            except ValueError:
                floor_multiplier = config["DEFAULT_GARAGE_LEVELS"]

        # 3. model inference
        detections = self.detect(pil_image)

        if not detections:
            area_est = int((geometry.area / stall_area) * usable_fraction)
            return area_est * floor_multiplier

        # convert pixel coords to real-world metres using the geometry bounds
        minx, miny, maxx, maxy = geometry.bounds
        tile_width_m = maxx - minx
        tile_height_m = maxy - miny
        img_w, img_h = pil_image.size

        padding_pct = 0.10
        pad_x = tile_width_m * padding_pct
        pad_y = tile_height_m * padding_pct

        tile_minx = minx - pad_x
        tile_miny = miny - pad_y
        tile_maxx = maxx + pad_x
        tile_maxy = maxy + pad_y

        m_per_px_x = (tile_maxx - tile_minx) / img_w
        m_per_px_y = (tile_maxy - tile_miny) / img_h

        total_spots = 0

        for bbox_px, conf, _ in detections:
            x1, y1, x2, y2 = bbox_px

            geo_x1 = tile_minx + x1 * m_per_px_x
            geo_x2 = tile_minx + x2 * m_per_px_x
            geo_y1 = tile_maxy - y2 * m_per_px_y
            geo_y2 = tile_maxy - y1 * m_per_px_y

            det_box = shapely_box(geo_x1, geo_y1, geo_x2, geo_y2)
            intersection = geometry.intersection(det_box)

            if intersection.is_empty:
                continue

            clipped_area_m2 = intersection.area
            estimated = int(clipped_area_m2 * usable_fraction / stall_area)
            estimated = int(estimated * floor_multiplier)
            total_spots += max(estimated, 0)

        # fallback for known structures with no visible detections
        if total_spots == 0 and is_struct:
            area_est = int((geometry.area / stall_area) * usable_fraction)
            return area_est * floor_multiplier

        return total_spots

    def annotate(self, pil_image, detections):
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)

        for box, conf, cls_name in detections:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"{cls_name} {conf:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")

        return annotated

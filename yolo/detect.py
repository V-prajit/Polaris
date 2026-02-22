import torch
import numpy as np
from shapely.geometry import box as shapely_box
from ultralytics import YOLO
from PIL import Image, ImageDraw

from parksight import config, pick_device, is_structure

# COCO class IDs for vehicles visible from above
_VEHICLE_CLASS_IDS = {3, 4, 5, 8}  # car, van, truck, bus
_VEHICLE_CLASS_NAMES = {3: "car", 4: "van", 5: "truck", 8: "bus"}


def mask_to_bboxes(mask: "np.ndarray", min_area: int = 500) -> list:
    """Extract bounding boxes from a binary segmentation mask.

    Returns list of (x1, y1, x2, y2) pixel-space boxes for regions
    larger than *min_area* pixels.
    """
    import cv2
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    bboxes = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, x + w, y + h))
    return bboxes


class YOLOParkingDetector:
    def __init__(self, weights_path, device=None, count_mode="auto"):
        """
        Parameters
        ----------
        weights_path : str
            Path to YOLO .pt weights.
        device : str or None
            Torch device.
        count_mode : str
            ``"detect"`` — count = number of detections (APKLOT-trained).
            ``"area"``   — count = detected-region area / stall area (ParkSeg-trained).
            ``"auto"``   — detect if >=10 boxes, else area.
        """
        self.device = device or pick_device()
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        self.count_mode = count_mode

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

    def count_spots(self, pil_image, geometry, osm_tags=None, segformer_mask=None):
        """Count parking spots in an image tile.

        Parameters
        ----------
        pil_image : PIL.Image
            Satellite tile covering *geometry*.
        geometry : shapely geometry
            OSM geometry in EPSG:3857.
        osm_tags : dict or None
            OSM tags for the feature.
        segformer_mask : np.ndarray or None
            If provided, YOLO detections outside the mask are discarded.
        """
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

        # Decide counting strategy
        mode = self.count_mode
        if mode == "auto":
            mode = "detect" if len(detections) >= 10 else "area"

        # ── Detection-count mode (APKLOT-trained: each box ≈ 1 spot) ──
        if mode == "detect":
            count = 0
            img_w, img_h = pil_image.size
            for bbox_px, conf, _ in detections:
                if segformer_mask is not None:
                    cx = int((bbox_px[0] + bbox_px[2]) / 2)
                    cy = int((bbox_px[1] + bbox_px[3]) / 2)
                    cx = min(max(cx, 0), img_w - 1)
                    cy = min(max(cy, 0), img_h - 1)
                    # mask may be different resolution than image
                    mh, mw = segformer_mask.shape[:2]
                    mx = int(cx * mw / img_w)
                    my = int(cy * mh / img_h)
                    mx = min(max(mx, 0), mw - 1)
                    my = min(max(my, 0), mh - 1)
                    if segformer_mask[my, mx] == 0:
                        continue  # detection outside parking mask
                count += 1
            count *= floor_multiplier
            if count == 0 and is_struct:
                area_est = int((geometry.area / stall_area) * usable_fraction)
                return area_est * floor_multiplier
            return count

        # ── Area-count mode (ParkSeg-trained: boxes are regions) ──
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

    def count_cars(self, pil_image, segformer_mask=None, confidence=None):
        """Count vehicles visible in a satellite/aerial tile.

        Uses SAHI (Slicing Aided Hyper Inference) to slice the tile into
        small overlapping crops so that COCO-pretrained YOLO can recognise
        vehicles from overhead.  Falls back to direct inference if SAHI
        is not installed.

        Parameters
        ----------
        pil_image : PIL.Image
            Satellite tile.
        segformer_mask : np.ndarray or None
            Binary mask (H×W, 1=parking).  Detections outside are dropped.
        confidence : float or None
            Override minimum confidence threshold.

        Returns
        -------
        int
            Number of vehicles detected inside the parking area.
        """
        conf = confidence if confidence is not None else config.get("CAR_CONFIDENCE", 0.25)
        img_w, img_h = pil_image.size

        # ── Try SAHI sliced inference (much better for overhead imagery) ──
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import tempfile, os

            # SAHI needs a file path — write a temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_image.save(tmp.name)
            tmp.close()

            if not hasattr(self, "_sahi_model"):
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=self.model.ckpt_path,
                    confidence_threshold=conf,
                    device=str(self.device),
                )
            self._sahi_model.confidence_threshold = conf

            result = get_sliced_prediction(
                tmp.name,
                self._sahi_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.4,
                overlap_width_ratio=0.4,
                verbose=0,
            )
            os.unlink(tmp.name)

            count = 0
            for pred in result.object_prediction_list:
                cat = pred.category.name
                if cat not in _VEHICLE_CLASS_NAMES.values():
                    continue

                if segformer_mask is not None:
                    bbox = pred.bbox  # sahi BoundingBox
                    cx = int((bbox.minx + bbox.maxx) / 2)
                    cy = int((bbox.miny + bbox.maxy) / 2)
                    cx = min(max(cx, 0), img_w - 1)
                    cy = min(max(cy, 0), img_h - 1)
                    mh, mw = segformer_mask.shape[:2]
                    mx = int(cx * mw / img_w)
                    my = int(cy * mh / img_h)
                    mx = min(max(mx, 0), mw - 1)
                    my = min(max(my, 0), mh - 1)
                    if segformer_mask[my, mx] == 0:
                        continue

                count += 1
            return count

        except ImportError:
            pass

        # ── Fallback: direct inference (works for street-level, weak on satellite) ──
        results = self.model(pil_image, conf=conf, verbose=False)[0]
        count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _VEHICLE_CLASS_IDS:
                continue

            if segformer_mask is not None:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)
                cx = min(max(cx, 0), img_w - 1)
                cy = min(max(cy, 0), img_h - 1)
                mh, mw = segformer_mask.shape[:2]
                mx = int(cx * mw / img_w)
                my = int(cy * mh / img_h)
                mx = min(max(mx, 0), mw - 1)
                my = min(max(my, 0), mh - 1)
                if segformer_mask[my, mx] == 0:
                    continue

            count += 1

        return count

    def annotate(self, pil_image, detections):
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)

        for box, conf, cls_name in detections:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"{cls_name} {conf:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")

        return annotated

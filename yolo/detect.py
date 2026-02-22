import torch
import numpy as np
from shapely.geometry import box as shapely_box
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageOps, ImageEnhance # Added ImageEnhance for TTA

from parksight import config, pick_device, is_structure

# COCO class IDs for vehicles visible from above
_VEHICLE_CLASS_IDS = {3, 4, 5, 8}  # car, van, truck, bus
_VEHICLE_CLASS_NAMES = {3: "car", 4: "van", 5: "truck", 8: "bus"}


def _compute_iou(boxA, boxB):
    """
    Computes Intersection over Union for two bounding boxes.
    Boxes are expected in (x1, y1, x2, y2) format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    return interArea / unionArea


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

    def count_spots_with_boxes(self, pil_image, geometry, osm_tags=None, segformer_mask=None):
        """Like count_spots() but also returns bounding boxes in pixel coords.

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

        Returns
        -------
        (count, boxes)
            count : int
            boxes : list of (x1, y1, x2, y2, conf) float tuples in pixel coords.
        """
        tags = osm_tags or {}
        stall_area = config["STALL_AREA_M2"]
        usable_fraction = config["USABLE_FRACTION_SURFACE"]

        # 1. direct from osm capacity tag — no imagery needed, no boxes
        if "capacity" in tags:
            try:
                return int(float(tags["capacity"])), []
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
            return area_est * floor_multiplier, []

        # Decide counting strategy
        mode = self.count_mode
        if mode == "auto":
            mode = "detect" if len(detections) >= 10 else "area"

        # ── Detection-count mode (APKLOT-trained: each box ≈ 1 spot) ──
        if mode == "detect":
            count = 0
            boxes = []
            img_w, img_h = pil_image.size
            for bbox_px, conf, _ in detections:
                if segformer_mask is not None:
                    cx = int((bbox_px[0] + bbox_px[2]) / 2)
                    cy = int((bbox_px[1] + bbox_px[3]) / 2)
                    cx = min(max(cx, 0), img_w - 1)
                    cy = min(max(cy, 0), img_h - 1)
                    mh, mw = segformer_mask.shape[:2]
                    mx = int(cx * mw / img_w)
                    my = int(cy * mh / img_h)
                    mx = min(max(mx, 0), mw - 1)
                    my = min(max(my, 0), mh - 1)
                    if segformer_mask[my, mx] == 0:
                        continue  # detection outside parking mask
                count += 1
                boxes.append((
                    float(bbox_px[0]), float(bbox_px[1]),
                    float(bbox_px[2]), float(bbox_px[3]),
                    float(conf),
                ))
            count *= floor_multiplier
            if count == 0 and is_struct:
                area_est = int((geometry.area / stall_area) * usable_fraction)
                return area_est * floor_multiplier, []
            return count, boxes

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
        boxes = []

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
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        # fallback for known structures with no visible detections
        if total_spots == 0 and is_struct:
            area_est = int((geometry.area / stall_area) * usable_fraction)
            return area_est * floor_multiplier, []

        return total_spots, boxes

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

            # Pass 1: Original Image
            result1 = get_sliced_prediction(
                tmp.name,
                self._sahi_model,
                slice_height=128,
                slice_width=128,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                verbose=0,
            )
            os.unlink(tmp.name)
            
            # Pass 2: Inverted Image (Test Time Augmentation)
            inverted_img = ImageOps.invert(pil_image.convert("RGB"))
            tmp_inv = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            inverted_img.save(tmp_inv.name)
            tmp_inv.close()
            
            result2 = get_sliced_prediction(
                tmp_inv.name,
                self._sahi_model,
                slice_height=128,
                slice_width=128,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                verbose=0,
            )
            os.unlink(tmp_inv.name)
            
            # Pass 3: Brightened Image (Pulls dark cars out of shadows)
            bright_img = ImageEnhance.Brightness(pil_image.convert("RGB")).enhance(1.5)
            tmp_br = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            bright_img.save(tmp_br.name)
            tmp_br.close()
            
            result3 = get_sliced_prediction(
                tmp_br.name,
                self._sahi_model,
                slice_height=128,
                slice_width=128,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                verbose=0,
            )
            os.unlink(tmp_br.name)
            
            # Combine all predictions from all 3 passes
            all_preds = result1.object_prediction_list + result2.object_prediction_list + result3.object_prediction_list
            
            # Filter classes and map to format
            formatted_boxes = []
            for pred in all_preds:
                cat_name = pred.category.name.lower()
                if any(v in cat_name for v in ["car", "van", "truck", "bus"]):
                    # Keep threshold aligned with configured car confidence.
                    if pred.score.value >= conf:
                        box = pred.bbox
                        formatted_boxes.append((pred, box.minx, box.miny, box.maxx, box.maxy, pred.score.value))
                    
            # Apply NMS
            formatted_boxes.sort(key=lambda x: x[5], reverse=True)
            kept_preds = []
            for current in formatted_boxes:
                overlap = False
                for kept in kept_preds:
                    if _compute_iou(current[1:5], kept[1:5]) > 0.20:
                        overlap = True
                        break
                if not overlap:
                    kept_preds.append(current)
            
            # Count valid ones against SegFormer mask
            count = 0
            for box_item in kept_preds:
                pred = box_item[0]
                if segformer_mask is not None:
                    box = pred.bbox
                    cx = int((box.minx + box.maxx) / 2)
                    cy = int((box.miny + box.maxy) / 2)
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

    def count_cars_with_boxes(self, pil_image, segformer_mask=None, confidence=None):
        """Like count_cars() but also returns bounding boxes in pixel coords.

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
        (count, boxes)
            count : int
            boxes : list of (x1, y1, x2, y2, conf) float tuples in pixel coords.
        """
        conf = confidence if confidence is not None else config.get("CAR_CONFIDENCE", 0.25)
        img_w, img_h = pil_image.size

        # ── Try SAHI sliced inference ──
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import tempfile, os

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

            # Pass 1: Original image
            result1 = get_sliced_prediction(
                tmp.name, self._sahi_model,
                slice_height=128, slice_width=128,
                overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0,
            )
            os.unlink(tmp.name)

            # Pass 2: Inverted image (TTA)
            inverted_img = ImageOps.invert(pil_image.convert("RGB"))
            tmp_inv = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            inverted_img.save(tmp_inv.name)
            tmp_inv.close()
            result2 = get_sliced_prediction(
                tmp_inv.name, self._sahi_model,
                slice_height=128, slice_width=128,
                overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0,
            )
            os.unlink(tmp_inv.name)

            # Pass 3: Brightened image
            bright_img = ImageEnhance.Brightness(pil_image.convert("RGB")).enhance(1.5)
            tmp_br = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            bright_img.save(tmp_br.name)
            tmp_br.close()
            result3 = get_sliced_prediction(
                tmp_br.name, self._sahi_model,
                slice_height=128, slice_width=128,
                overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0,
            )
            os.unlink(tmp_br.name)

            all_preds = (
                result1.object_prediction_list
                + result2.object_prediction_list
                + result3.object_prediction_list
            )

            # Filter to vehicle classes using configured confidence threshold.
            formatted_boxes = []
            for pred in all_preds:
                cat_name = pred.category.name.lower()
                if any(v in cat_name for v in ["car", "van", "truck", "bus"]):
                    if pred.score.value >= conf:
                        box = pred.bbox
                        formatted_boxes.append(
                            (pred, box.minx, box.miny, box.maxx, box.maxy, pred.score.value)
                        )

            # NMS
            formatted_boxes.sort(key=lambda x: x[5], reverse=True)
            kept_preds = []
            for current in formatted_boxes:
                overlap = False
                for kept in kept_preds:
                    if _compute_iou(current[1:5], kept[1:5]) > 0.20:
                        overlap = True
                        break
                if not overlap:
                    kept_preds.append(current)

            # Apply SegFormer mask filter and collect boxes
            count = 0
            boxes = []
            for box_item in kept_preds:
                pred = box_item[0]
                bx1, by1, bx2, by2, bscore = (
                    box_item[1], box_item[2], box_item[3], box_item[4], box_item[5]
                )
                if segformer_mask is not None:
                    cx = int((bx1 + bx2) / 2)
                    cy = int((by1 + by2) / 2)
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
                boxes.append((float(bx1), float(by1), float(bx2), float(by2), float(bscore)))

            return count, boxes

        except ImportError:
            pass

        # ── Fallback: direct inference ──
        results = self.model(pil_image, conf=conf, verbose=False)[0]
        count = 0
        boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _VEHICLE_CLASS_IDS:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            if segformer_mask is not None:
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
            boxes.append((
                float(xyxy[0]), float(xyxy[1]),
                float(xyxy[2]), float(xyxy[3]),
                float(box.conf[0]),
            ))

        return count, boxes

    def annotate(self, pil_image, detections):
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)

        for box, conf, cls_name in detections:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"{cls_name} {conf:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")

        return annotated

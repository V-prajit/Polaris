import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

STALL_AREA_M2 = 15.5
USABLE_FRACTION = 0.62

class YOLOParkingDetector:
    def __init__(self, weights_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.model = YOLO(weights_path)
        self.model.to(self.device)

    def detect(self, pil_image, confidence=0.25):
        results = self.model(pil_image, conf=confidence, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            detections.append((xyxy, conf, cls_name))
            
        return detections

    def count_spots(self, pil_image, geometry):
        detections = self.detect(pil_image)
        
        minx, miny, maxx, maxy = geometry.bounds
        tile_width_m = maxx - minx
        tile_height_m = maxy - miny
        img_w, img_h = pil_image.size
        
        padding_pct = 0.10
        actual_width_m = tile_width_m * (1 + 2 * padding_pct)
        actual_height_m = tile_height_m * (1 + 2 * padding_pct)
        
        m_per_px_x = actual_width_m / img_w
        m_per_px_y = actual_height_m / img_h
        
        total_spots = 0
        for box, _, _ in detections:
            x1, y1, x2, y2 = box
            box_area_m2 = (x2 - x1) * m_per_px_x * (y2 - y1) * m_per_px_y
            estimated_spots = int(box_area_m2 * USABLE_FRACTION / STALL_AREA_M2)
            total_spots += estimated_spots
            
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

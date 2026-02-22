import random
import os
import requests
import math
import io
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from io import BytesIO

# Import YOLO and Segformer locally
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CITIES = [
    {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740}
]

def download_static_map(lat, lon, zoom=18):
    """Fallback to quickly download a static map image for visualization"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB").resize((512, 512))
    except Exception as e:
        print(f"Error fetching tile: {e}")
    return Image.new("RGB", (512, 512), color=(128,128,128))

def get_yolo_model():
    model_path = "models/yolo_aerial_cars.pt"
    if not os.path.exists(model_path): return None
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

def get_segformer_model():
    model_dir = "models/best_model"
    if not os.path.exists(model_dir): return None, None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    return processor, model

def nms(boxes_with_scores, iou_thresh=0.20):
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if float(boxAArea + boxBArea - interArea) == 0: return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    boxes_with_scores.sort(key=lambda x: x[4], reverse=True)
    kept_boxes = []
    for current in boxes_with_scores:
        overlap = False
        for kept in kept_boxes:
            if compute_iou(current[:4], kept[:4]) > iou_thresh:
                overlap = True; break
        if not overlap: kept_boxes.append(current)
    return kept_boxes

def run_yolo(img, sahi_model):
    if sahi_model is None: return img.copy(), 0
    
    # Save temp image for Sahi
    img.save("temp_yolo.png")
    
    res = get_sliced_prediction(
        "temp_yolo.png", sahi_model,
        slice_height=128, slice_width=128,
        overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0
    )
    
    all_preds = []
    for p in res.object_prediction_list:
        x1, y1, x2, y2 = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
        score = p.score.value
        cat = p.category.name.lower()
        if any(v in cat for v in ["car", "van", "truck", "bus"]) and score >= 0.35:
            all_preds.append((x1, y1, x2, y2, score, cat))
            
    final_boxes = nms(all_preds)
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)
    for (x1, y1, x2, y2, score, cat) in final_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
        
    return out_img, len(final_boxes)

def run_segformer(img, processor, model):
    if model is None: return img.copy(), 0
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    seg_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # Create colored overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if seg_mask[y, x] > 0: # If is parking area
                overlay.putpixel((x, y), (255, 0, 255, 128))
                
    out_img = img.convert('RGBA')
    out_img = Image.alpha_composite(out_img, overlay).convert('RGB')
    
    # Very rough stall estimate based on area
    px_area = (seg_mask > 0).sum()
    est_stalls = int((px_area / (512*512)) * 100) # Dummy rough scale factor
    
    return out_img, est_stalls

def main():
    print("Loading Models...")
    yolo_model = get_yolo_model()
    seg_processor, seg_model = get_segformer_model()
    
    print("Testing 5 Random Cities on 512x512 patches...")
    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    fig.suptitle("ParkSight 512x512 Patch Visualizer", fontsize=16)
    
    for j, col in enumerate(["Satellite Image", "YOLO V11", "Segformer B5"]):
        axs[0, j].set_title(col, fontsize=14, pad=10)
        
    for i, city in enumerate(CITIES):
        # Add slight random offset
        lat = city["lat"] + random.uniform(-0.02, 0.02)
        lon = city["lon"] + random.uniform(-0.02, 0.02)
        
        print(f"[{i+1}/5] Fetching {city['name']} at {lat:.4f}, {lon:.4f}")
        img = download_static_map(lat, lon)
        
        yolo_img, yolo_count = run_yolo(img, yolo_model)
        seg_img, seg_count = run_segformer(img, seg_processor, seg_model)
        
        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].text(10, 30, f"{city['name']}\nLat: {lat:.3f}\nLon: {lon:.3f}", color="white", backgroundcolor="black")
        
        axs[i, 1].imshow(yolo_img)
        axs[i, 1].axis('off')
        axs[i, 1].text(256, 480, f"{yolo_count} Cars Found", color="cyan", backgroundcolor="black", ha="center")
        
        axs[i, 2].imshow(seg_img)
        axs[i, 2].axis('off')
        axs[i, 2].text(256, 480, f"Seg Area Est: {seg_count}", color="magenta", backgroundcolor="black", ha="center")
        
    plt.tight_layout()
    plt.savefig("patch_comparisons.png", dpi=150)
    print("✅ Saved comparison grid to patch_comparisons.png")
    
if __name__ == "__main__":
    main()

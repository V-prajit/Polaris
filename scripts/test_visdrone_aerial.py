import contextily as ctx
from shapely.geometry import box
import numpy as np
from PIL import Image
import pyproj
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import urllib.request

from huggingface_hub import hf_hub_download
import shutil

# Download weights if not present
model_path = "yolov8s-visdrone.pt"
if not os.path.exists(model_path):
    print("Downloading yolov8s-visdrone.pt from huggingface...")
    downloaded_path = hf_hub_download(repo_id="mshamrai/yolov8s-visdrone", filename="best.pt")
    shutil.copy(downloaded_path, model_path)

def get_test_tile(lat, lon, size_m=200):
    """Fetch an Esri satellite tile centered on lat/lon."""
    transformer = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = transformer.transform(lon, lat)
    bounds = (x - size_m/2, y - size_m/2, x + size_m/2, y + size_m/2)
    img_array, extent = ctx.bounds2img(*bounds, zoom=19,
        source=ctx.providers.Esri.WorldImagery)
    return Image.fromarray(img_array[:, :, :3])  # drop alpha

# Test locations (known parking lots in Atlanta)
test_locations = [
    ("Atlantic Station", 33.7908, -84.3953),
    ("Turner Field Lot", 33.7353, -84.3894),
    ("GT Parking", 33.7756, -84.4003),
]

print("Fetching test tiles...")
for name, lat, lon in test_locations:
    tile = get_test_tile(lat, lon)
    tile.save(f"test_{name.replace(' ', '_')}.png")

print("Initializing SAHI model...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,
    device="mps", # using MPS for mac if available, else cpu
)

results_summary = {}

for name, lat, lon in test_locations:
    location_key = name.replace(' ', '_')
    img_path = f"test_{location_key}.png"
    print(f"Running inference on {name}...")
    
    result = get_sliced_prediction(
        img_path,
        sahi_model,
        slice_height=128,
        slice_width=128,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        verbose=1,
    )
    
    vehicle_count = 0
    for pred in result.object_prediction_list:
        cat = pred.category.name.lower()
        if any(v in cat for v in ["car", "van", "truck", "bus", "vehicle"]):
            vehicle_count += 1
            
    print(f"{name}: {vehicle_count} vehicles detected")
    results_summary[name] = vehicle_count
    
    # Save annotated image for visual inspection
    result.export_visuals(export_dir=f"results_{location_key}/")

print("\n--- Summary ---")
for name, count in results_summary.items():
    print(f"{name}: {count} vehicles")

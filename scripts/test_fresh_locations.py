import requests
import random
import time
import json
import os
import math
import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict

# Bounding box roughly covering Atlanta and some surrounding suburbs
ATLANTA_MIN_LAT = 33.65
ATLANTA_MAX_LAT = 33.90
ATLANTA_MIN_LON = -84.50
ATLANTA_MAX_LON = -84.25

NUM_LOCATIONS_TO_TEST = 5
RADIUS = 400

def get_random_coordinate():
    lat = random.uniform(ATLANTA_MIN_LAT, ATLANTA_MAX_LAT)
    lon = random.uniform(ATLANTA_MIN_LON, ATLANTA_MAX_LON)
    return round(lat, 5), round(lon, 5)

def download_static_map(lat, lon, zoom=17):
    """Fallback to quickly download a static map image for visualization"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"Error fetching tile: {e}")
    
    # Return a blank gray image on failure
    return Image.new("RGB", (256, 256), color=(128,128,128))

def run_test():
    print(f"============================================================")
    print(f"🚀 ParkSight Model Benchmark - Visual Grid ({NUM_LOCATIONS_TO_TEST} locations)")
    print(f"============================================================")

    fig, axs = plt.subplots(NUM_LOCATIONS_TO_TEST, 4, figsize=(16, 4 * NUM_LOCATIONS_TO_TEST))
    fig.suptitle("ParkSight Detection Comparison (Fresh Atlanta Locations)", fontsize=16)
    
    # Column headers
    cols = ["Satellite Image", "Math Heuristics (Area)", "YOLO V11", "Segformer"]
    for j in range(4):
        axs[0, j].set_title(cols[j], fontsize=14, pad=10)

    for i in range(NUM_LOCATIONS_TO_TEST):
        lat, lon = get_random_coordinate()
        print(f"\n[{i+1}/{NUM_LOCATIONS_TO_TEST}] Testing Random Coordinate: {lat}, {lon} (Radius: {RADIUS}m)")
        url = f"http://localhost:8000/api/estimate?lat={lat}&lon={lon}&radius={RADIUS}"
        
        # Download the central tile for context
        img = download_static_map(lat, lon)
        
        try:
            start_time = time.time()
            resp = requests.get(url, timeout=300)
            elapsed = time.time() - start_time
            
            if resp.status_code == 200:
                data = resp.json()
                surface_features = data.get("surface", {}).get("features", [])
                
                sum_area = sum(f.get("count_area", 0) for f in surface_features)
                sum_yolo = sum(f.get("count_yolo", 0) for f in surface_features)
                sum_seg = sum(f.get("count_segformer", 0) for f in surface_features)
                
                print(f"   ⏱️  Backend took {elapsed:.1f}s. Results: Math={sum_area}, YOLO={sum_yolo}, Seg={sum_seg}")
                
                # Draw the original image in the first col
                axs[i, 0].imshow(img)
                axs[i, 0].axis('off')
                axs[i, 0].text(10, 20, f"Lat: {lat}\nLon: {lon}", color="white", backgroundcolor="black", fontsize=10)
                
                # Draw the Math result
                axs[i, 1].imshow(img)
                axs[i, 1].axis('off')
                axs[i, 1].text(128, 128, f"{sum_area} Spots\n(Area Heuristic)", color="yellow", 
                               backgroundcolor="black", fontsize=14, ha="center", va="center")
                               
                # Draw the YOLO result
                axs[i, 2].imshow(img)
                axs[i, 2].axis('off')
                axs[i, 2].text(128, 128, f"{sum_yolo} Spots\n(YOLOv11+TTA)", color="cyan", 
                               backgroundcolor="black", fontsize=14, ha="center", va="center")
                               
                # Draw the Segformer result
                axs[i, 3].imshow(img)
                axs[i, 3].axis('off')
                axs[i, 3].text(128, 128, f"{sum_seg} Spots\n(Segformer B5)", color="magenta", 
                               backgroundcolor="black", fontsize=14, ha="center", va="center")
                               
            else:
                print(f"   ❌ Failed with status code {resp.status_code}")
                for j in range(4): axs[i, j].axis('off')
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            for j in range(4): axs[i, j].axis('off')
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    out_file = "fresh_locations_grid.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved visual comparison matrix to {out_file}")

if __name__ == "__main__":
    run_test()

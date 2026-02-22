import os
import time
import requests
import math
from tqdm import tqdm

# Default to roughly the Atlantic Station demo area
DEMO_LAT = 33.8025746
DEMO_LON = -84.4106416
RADIUS_M = 1000  # 1km radius around demo location

ZOOM_LEVELS = [16, 17, 18]

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def get_bbox_tiles(lat, lon, radius_m, zoom):
    # Rough approximation: 1 deg lat = 111km, 1 deg lon = 111km * cos(lat)
    lat_offset = radius_m / 111000.0
    lon_offset = radius_m / (111000.0 * math.cos(math.radians(lat)))
    
    min_lat, max_lat = lat - lat_offset, lat + lat_offset
    min_lon, max_lon = lon - lon_offset, lon + lon_offset
    
    min_x, max_y = lat_lon_to_tile(min_lat, min_lon, zoom)
    max_x, min_y = lat_lon_to_tile(max_lat, max_lon, zoom)
    
    return range(min_x, max_x + 1), range(min_y, max_y + 1)

def download_tiles():
    out_dir = os.path.join("public", "tiles")
    os.makedirs(out_dir, exist_ok=True)
    
    sources = {
        "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "labels": "https://a.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"
    }

    # First, calculate total tiles to download
    tasks = []
    for z in ZOOM_LEVELS:
        xs, ys = get_bbox_tiles(DEMO_LAT, DEMO_LON, RADIUS_M, z)
        for x in xs:
            for y in ys:
                for source_name, url_template in sources.items():
                    url = url_template.format(z=z, x=x, y=y)
                    tile_path = os.path.join(out_dir, source_name, str(z), str(x), f"{y}.png")
                    tasks.append((url, tile_path))
                    
    print(f"Total tiles to process: {len(tasks)}")

    total_downloaded = 0
    total_skipped = 0

    for url, tile_path in tqdm(tasks, desc="Downloading map tiles"):
        if os.path.exists(tile_path):
            total_skipped += 1
            continue
            
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(tile_path, "wb") as f:
                    f.write(resp.content)
                total_downloaded += 1
                time.sleep(0.1) # Be nice to tile servers
            else:
                tqdm.write(f"Failed to fetch {url}: {resp.status_code}")
        except Exception as e:
            tqdm.write(f"Error fetching {url}: {e}")
                        
    print(f"Downloaded {total_downloaded} new tiles to {out_dir}. Skipped {total_skipped} existing tiles.")

if __name__ == "__main__":
    download_tiles()

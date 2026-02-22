import os
import time
import requests
import math
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y_val = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    y = int(y_val)
    return x, y

def get_bbox_tiles(min_lat, max_lat, min_lon, max_lon, zoom):
    min_x, max_y = lat_lon_to_tile(min_lat, min_lon, zoom)
    max_x, min_y = lat_lon_to_tile(max_lat, max_lon, zoom)
    # Ensure min < max in case coordinates are ordered differently
    return range(min(min_x, max_x), max(min_x, max_x) + 1), range(min(min_y, max_y), max(min_y, max_y) + 1)

def download_tile(url, tile_path):
    if os.path.exists(tile_path):
        return "skipped"
    
    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(tile_path, "wb") as f:
                f.write(resp.content)
            time.sleep(0.05)
            return "downloaded"
        else:
            return f"failed: {resp.status_code}"
    except Exception as e:
        return f"error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Download satellite tiles.")
    parser.add_argument("--min-lat", type=float, default=33.647)
    parser.add_argument("--max-lat", type=float, default=33.886)
    parser.add_argument("--min-lon", type=float, default=-84.552)
    parser.add_argument("--max-lon", type=float, default=-84.289)
    parser.add_argument("--zooms", type=str, default="17,18,19")
    parser.add_argument("--output-dir", type=str, default="data/tiles")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    zooms = [int(z) for z in args.zooms.split(",")]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    tasks_by_zoom = {}
    total_tiles = 0

    print("=== Tile Download Planning ===")
    for z in zooms:
        xs, ys = get_bbox_tiles(args.min_lat, args.max_lat, args.min_lon, args.max_lon, z)
        tasks = []
        for x in xs:
            for y in ys:
                url = url_template.format(z=z, x=x, y=y)
                # Output: data/tiles/satellite/{z}/{x}/{y}.png
                tile_path = os.path.join(out_dir, "satellite", str(z), str(x), f"{y}.png")
                tasks.append((url, tile_path))
        tasks_by_zoom[z] = tasks
        total_tiles += len(tasks)
        print(f"Zoom {z}: {len(tasks)} tiles")
    
    # 1 tile = ~150KB on average maybe? Let's say ~100-200KB. Estimate 150KB.
    estimated_size_mb = total_tiles * 150 / 1024
    if estimated_size_mb > 1024:
        print(f"Estimated size: {estimated_size_mb / 1024:.2f} GB")
    else:
        print(f"Estimated size: {estimated_size_mb:.2f} MB")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for z in zooms:
            tasks = tasks_by_zoom[z]
            if not tasks:
                continue
            
            futures = {executor.submit(download_tile, url, path): url for url, path in tasks}
            
            with tqdm(total=len(tasks), desc=f"Zoom {z}") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res == "skipped":
                        total_skipped += 1
                    elif res == "downloaded":
                        total_downloaded += 1
                    else:
                        total_failed += 1
                    pbar.update(1)

    print("\n=== Summary ===")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped:    {total_skipped}")
    print(f"Failed:     {total_failed}")

if __name__ == "__main__":
    main()

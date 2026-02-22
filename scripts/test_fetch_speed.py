import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parksight.fetch import get_parking_data_by_coords

t = time.time()
gdf = get_parking_data_by_coords(33.75, -84.4, 300)
print(f"OSM fetch: {time.time()-t:.2f}s")

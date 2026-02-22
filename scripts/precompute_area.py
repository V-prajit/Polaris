import os
import json
import requests
import time

# List of demo locations for the Hackathon
CITIES = [
    {"name": "Atlanta (Atlantic Station)", "lat": 33.8025746, "lon": -84.4106416, "radius": 300},
    {"name": "Atlanta (Downtown)", "lat": 33.7553, "lon": -84.3900, "radius": 300},
    {"name": "Midtown Atlanta+", "lat": 33.7844, "lon": -84.3831, "radius": 300},
    {"name": "Buckhead", "lat": 33.8390, "lon": -84.3798, "radius": 300},
    {"name": "Decatur", "lat": 33.7748, "lon": -84.2963, "radius": 300},
]

def precompute():
    """
    Run the heavy YOLO and Segformer inferences on the backend 
    and save the results to public/precomputed for lightning-fast frontend loading.
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "precomputed")
    os.makedirs(out_dir, exist_ok=True)
    
    for city in CITIES:
        lat = city["lat"]
        lon = city["lon"]
        radius = city["radius"]
        print(f"Precomputing {city['name']} at {lat}, {lon} (radius {radius})")
        url = f"http://localhost:8000/api/estimate?lat={lat}&lon={lon}&radius={radius}"
        try:
            start = time.time()
            resp = requests.get(url, timeout=600) # Give it 10 minutes per huge query
            if resp.status_code == 200:
                data = resp.json()
                filename = f"{round(lat, 3)}_{round(lon, 3)}_{radius}.json"
                filepath = os.path.join(out_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(data, f)
                print(f"--> Saved to {filepath} in {time.time() - start:.1f}s")
            else:
                print(f"--> Failed: {resp.status_code}")
        except Exception as e:
            print(f"--> Error: {e}")

if __name__ == "__main__":
    precompute()

import requests
import os
import time

os.makedirs('cache', exist_ok=True)
overpass_url = 'https://overpass-api.de/api/interpreter'

queries = {
    "surface": """
[out:json][timeout:300];
(
  way["amenity"="parking"](33.647,-84.552,33.886,-84.289);
  relation["amenity"="parking"](33.647,-84.552,33.886,-84.289);
);
out body;
>;
out skel qt;
""",
    "structured": """
[out:json][timeout:300];
(
  way["building"="parking"](33.647,-84.552,33.886,-84.289);
  way["building"="garage"](33.647,-84.552,33.886,-84.289);
  way["parking"="multi-storey"](33.647,-84.552,33.886,-84.289);
  way["parking"="underground"](33.647,-84.552,33.886,-84.289);
  relation["building"="parking"](33.647,-84.552,33.886,-84.289);
);
out body;
>;
out skel qt;
""",
    "street": """
[out:json][timeout:300];
(
  way["parking:lane"](33.647,-84.552,33.886,-84.289);
  way["parking:left"](33.647,-84.552,33.886,-84.289);
  way["parking:right"](33.647,-84.552,33.886,-84.289);
  way["parking:both"](33.647,-84.552,33.886,-84.289);
);
out body;
>;
out skel qt;
""",
}

for name, query in queries.items():
    out_path = f"cache/{name}_raw.json"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        print(f"Skipping {name} — already have {os.path.getsize(out_path)} bytes")
        continue

    for attempt in range(3):
        print(f"Fetching {name} (attempt {attempt+1})...")
        try:
            r = requests.post(overpass_url, data={'data': query}, timeout=300)
            if r.status_code == 200 and len(r.content) > 50:
                with open(out_path, 'w') as f:
                    f.write(r.text)
                print(f"  {name}: {len(r.content)} bytes")
                break
            else:
                print(f"  {name}: HTTP {r.status_code}, {len(r.content)} bytes — retrying in 30s")
                time.sleep(30)
        except Exception as e:
            print(f"  {name}: error {e} — retrying in 30s")
            time.sleep(30)
    else:
        print(f"  {name}: FAILED after 3 attempts")

print("Done!")

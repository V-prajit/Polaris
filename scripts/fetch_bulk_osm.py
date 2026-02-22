import requests
import os

os.makedirs('cache', exist_ok=True)
overpass_url = 'https://overpass-api.de/api/interpreter'

print('Fetching surface parking...')
query = """
[out:json][timeout:300];
(
  way["amenity"="parking"](33.647,-84.552,33.886,-84.289);
  relation["amenity"="parking"](33.647,-84.552,33.886,-84.289);
);
out body;
>;
out skel qt;
"""
r = requests.post(overpass_url, data={'data': query}, timeout=300)
print(f'Surface: {r.status_code}, {len(r.content)} bytes')
with open('cache/surface_raw.json', 'w') as f:
    f.write(r.text)

print('Fetching structured parking...')
query2 = """
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
"""
r2 = requests.post(overpass_url, data={'data': query2}, timeout=300)
print(f'Structured: {r2.status_code}, {len(r2.content)} bytes')
with open('cache/structured_raw.json', 'w') as f:
    f.write(r2.text)

print('Fetching street parking...')
query3 = """
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
"""
r3 = requests.post(overpass_url, data={'data': query3}, timeout=300)
print(f'Street: {r3.status_code}, {len(r3.content)} bytes')
with open('cache/street_raw.json', 'w') as f:
    f.write(r3.text)

print('Done! Raw JSON saved to cache/')

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import re
import json

ZONING_HEURISTICS = {
  "R-1": { "description": "Single-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "R-2": { "description": "Single-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "R-3": { "description": "Single-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "R-4": { "description": "Single-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "R-5": { "description": "Two-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "RG-1": { "description": "Residential General Sector 1", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 300 },
  "RG-2": { "description": "Residential General Sector 2", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 275 },
  "RG-3": { "description": "Residential General Sector 3", "max_deck_height_ft": 80, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 250 },
  "MR-1": { "description": "Multi-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 275 },
  "MR-2": { "description": "Multi-Family Residential", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 275 },
  "MR-3": { "description": "Multi-Family Residential", "max_deck_height_ft": 80, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 250 },
  "MR-4A": { "description": "Multi-Family Residential", "max_deck_height_ft": 80, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 250 },
  "MR-4B": { "description": "Multi-Family Residential", "max_deck_height_ft": 52, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 250 },
  "MR-5A": { "description": "Multi-Family Residential", "max_deck_height_ft": 150, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 200 },
  "C-1": { "description": "Community Business", "max_deck_height_ft": 35, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 300 },
  "C-2": { "description": "Commercial Service", "max_deck_height_ft": 60, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 275 },
  "C-3": { "description": "Commercial Residential", "max_deck_height_ft": 150, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 250 },
  "I-1": { "description": "Light Industrial", "max_deck_height_ft": 60, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "I-2": { "description": "Heavy Industrial", "max_deck_height_ft": 60, "parking_standard_dims": "8.5' x 18'", "compact_allowed": false, "heuristic_sqft_per_space_incl_aisle": 300 },
  "SPI": { "description": "Special Public Interest (Downtown/Midtown/Buckhead)", "max_deck_height_ft": 200, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 200 },
  "PD-H": { "description": "Planned Development Housing", "max_deck_height_ft": 60, "parking_standard_dims": "8.5' x 18'", "compact_allowed": true, "heuristic_sqft_per_space_incl_aisle": 275 }
}

def create_coordinate_cache(coordinate_list, output_filepath="coord_zoning_cache.json"):
    print("Loading zoning map...")
    zoning_gdf = gpd.read_file("Atlanta_Zoning_Districts.geojson")
    
    # coordinate_list should be a list of dicts: [{"lat": 33.7845, "lon": -84.3755}, ...]
    df_coords = pd.DataFrame(coordinate_list)
    geometry = [Point(xy) for xy in zip(df_coords['lon'], df_coords['lat'])]
    coords_gdf = gpd.GeoDataFrame(df_coords, geometry=geometry, crs=zoning_gdf.crs)
    
    print("Performing spatial join...")
    joined_gdf = gpd.sjoin(coords_gdf, zoning_gdf[['ZONECLASS', 'geometry']], how="left", predicate="within")
    
    def map_heuristic(zoneclass):
        if pd.isna(zoneclass): return None
        clean_zone = re.sub(r'-C$', '', str(zoneclass).strip())
        if clean_zone in ZONING_HEURISTICS: return ZONING_HEURISTICS[clean_zone]
        for key in sorted(ZONING_HEURISTICS.keys(), key=len, reverse=True):
            if clean_zone.startswith(key): return ZONING_HEURISTICS[key]
        return None

    joined_gdf['heuristic_data'] = joined_gdf['ZONECLASS'].apply(map_heuristic)
    
    # Use a rounded string representation of the coordinates as the cache key
    # .6f provides ~11 centimeter precision, which is more than enough.
    print("Saving cache to file...")
    cache_dict = {}
    for _, row in joined_gdf.iterrows():
        # Create a string key like "33.784500,-84.375500"
        coord_key = f"{row['lat']:.6f},{row['lon']:.6f}"
        cache_dict[coord_key] = row['heuristic_data']
        
    with open(output_filepath, 'w') as f:
        json.dump(cache_dict, f, indent=2)
    
    print(f"Successfully cached {len(cache_dict)} coordinates to {output_filepath}")
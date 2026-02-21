import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# YOLO Pipeline Evaluation

This notebook demonstrates the YOLO backup pipeline on three different Atlanta locations:
1. Georgia Tech campus
2. Lenox Square Mall
3. A residential area
"""

code_imports = """\
import matplotlib.pyplot as plt
from IPython.display import display
import os

from parksight.fetch import get_parking_data, get_satellite_tile
from yolo.detect import YOLOParkingDetector
from parksight.viz import create_parking_map

weights_path = "models/best.pt" if os.path.exists("models/best.pt") else "runs/parksight_yolo/weights/best.pt"
detector = YOLOParkingDetector(weights_path)
"""

def demo_location(loc_name):
    return f"""\
print("Processing: {loc_name}")
gdf, (lat, lon) = get_parking_data("{loc_name}", dist=300)
gdf_3857 = gdf.to_crs(epsg=3857)

total_spots = 0
for idx, row in gdf_3857.iterrows():
    if row.geometry.geom_type in ("Polygon", "MultiPolygon"):
        img = get_satellite_tile(row.geometry)
        detector.detect(img)
        spots = detector.count_spots(img, row.geometry)
        total_spots += spots
        
        # Annotate
        annotated = detector.annotate(img, detector.detect(img))
        
        # Display
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Satellite")
        axes[0].axis('off')
        
        axes[1].imshow(annotated)
        axes[1].set_title(f"YOLO detections (est. {{spots}} spots)")
        axes[1].axis('off')
        plt.show()

print(f"Total spots estimated for {loc_name}: {{total_spots}}")

m = create_parking_map(lat, lon, gdf, radius=300, total_count=total_spots)
display(m)
"""

cells = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell("## 1. Georgia Tech campus"),
    nbf.v4.new_code_cell(demo_location("Georgia Tech campus, Atlanta, GA")),
    nbf.v4.new_markdown_cell("## 2. Lenox Square Mall"),
    nbf.v4.new_code_cell(demo_location("Lenox Square Mall, Atlanta, GA")),
    nbf.v4.new_markdown_cell("## 3. A residential area"),
    nbf.v4.new_code_cell(demo_location("Midtown residential area, Atlanta, GA"))
]

nb['cells'] = cells
os.makedirs("notebooks", exist_ok=True)
with open('notebooks/03_yolo_pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

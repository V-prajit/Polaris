# Datasets

This directory is a placeholder for training data. The baselines work
without any downloaded datasets — they use OSM + live Esri satellite tiles.

To **improve** on the baselines you'll want labelled parking data.
Here are some good starting points:

## Parking-specific

| Dataset | What it has | Size | Link |
|---------|-------------|------|------|
| **ParkSeg12k** | 12 000 aerial images with pixel-level parking segmentation masks | ~4 GB | [github.com/geohai/ParkSeg12k](https://github.com/geohai/ParkSeg12k) |
| **APKLOT** | Aerial parking lot images with bounding-box annotations | ~1 GB | [Kaggle](https://www.kaggle.com/datasets) |
| **PKLot** | 12 000+ images of occupied/empty parking spaces from fixed CCTV cameras | ~4 GB | [web.inf.ufpr.br/vri/databases/parking-lot-database](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) |

## General satellite imagery

| Dataset | Resolution | Link |
|---------|-----------|------|
| **NAIP** | 1 m aerial imagery (USA) | [naip-usdaonline.hub.arcgis.com](https://naip-usdaonline.hub.arcgis.com/) |
| **SpaceNet** | Sub-metre with building footprints | [spacenet.ai](https://spacenet.ai/) |
| **Sentinel-2** | 10 m multispectral (global, free) | [Copernicus Open Access Hub](https://scihub.copernicus.eu/) |

## Tips

- Start with **ParkSeg12k** — it's the most directly relevant.
- For training YOLO-family detectors, convert segmentation masks to bounding boxes.
- Consider augmenting with your own satellite tiles via `parksight.fetch.get_satellite_tile()`.

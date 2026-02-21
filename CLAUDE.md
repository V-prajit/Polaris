# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ParkSight** — estimates parking stall counts from satellite imagery using OSM geospatial data, computer vision, and fine-tuned ML models. Built for Hacklytics 2026.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
# or with uv (preferred):
uv sync
```

### Run the CLI baseline
```bash
python examples/run_baseline.py --address "Georgia Tech, Atlanta, GA" [--radius 300] [--output parking_map.html]
```

### Run the FastAPI backend
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# or:
python -m api.app
```

### YOLO training (requires A100 GPU)
```bash
python yolo/train.py
# Auto-downloads dataset from HuggingFace; saves best weights to runs/parksight_yolo/weights/best.pt
```

### YOLO inference
```bash
python yolo/run.py
```

### SegFormer training/evaluation
```bash
python scripts/train_segformer.py
python scripts/eval_segformer.py
```

### Jupyter notebooks
```bash
jupyter notebook notebooks/01_quickstart.ipynb    # end-to-end tutorial
jupyter notebook notebooks/02_improve_counting.ipynb
jupyter notebook notebooks/03_yolo_pipeline.ipynb
```

There is no automated test suite; validate changes by running the CLI example or hitting the API health endpoint (`GET /api/health`).

## Architecture

### Data flow
```
User address
  → OSMnx geocode → (lat, lon)
  → OSM query for parking polygons, garages, street lanes
  → Per-feature counting:
      Surface lots:   satellite tile (Esri zoom 19) → YOLO / CV / SegFormer → stall count
      Structures:     OSM area × levels ÷ stall_area (blended with capacity tags)
      Street parking: linestring length ÷ car_length × sides
  → Aggregate estimate + Folium HTML map
```

### Key modules

| Path | Responsibility |
|------|---------------|
| `parksight/__init__.py` | Config loading, shared utilities |
| `parksight/fetch.py` | OSM feature queries + satellite tile fetching |
| `parksight/count.py` | CV counting (Canny + Hough lines) |
| `parksight/estimate_structured.py` | Structured/street parking estimation |
| `parksight/segment.py` | SegFormer segmentation inference |
| `parksight/viz.py` | Folium map generation |
| `yolo/detect.py` | `YOLOParkingDetector` wrapper (singleton, lazy-loaded) |
| `api/app.py` | FastAPI server with TTL cache and H3 macro endpoint |
| `config.json` | Tunable CV/ML parameters (edge thresholds, stall areas, confidence) |

### API endpoints
- `GET /api/estimate?lat=&lon=&radius=` — per-location parking estimate with geometry
- `GET /api/macro?min_lat=&max_lat=&min_lon=&max_lon=&resolution=9` — city-wide H3 hexagon heatmap (max 200 cells)
- `GET /api/health` — model load status

### Counting strategy (in priority order)
1. **YOLO** (fine-tuned YOLOv8 Nano, `models/yolo26n_run1.pt`) — bounding-box detections on satellite tiles
2. **SegFormer** (`models/best_model/`) — pixel-level segmentation → connected-component count
3. **CV baseline** — OpenCV Canny edges + HoughLinesP → line count ÷ 2
4. **Area heuristic** — polygon area ÷ stall area (config-driven, ~32 m² USA)

### Geometry conventions
- Calculations: Web Mercator (EPSG:3857)
- API output: WGS84 (EPSG:4326)

### Caching
API responses are cached with a 10-minute TTL, keyed to 3 decimal places of lat/lon (~111 m grid). Cache files land in `cache/`.

## Pre-trained models
- `models/yolo26n_run1.pt` — YOLOv8 Nano trained on parking aerial imagery
- `models/best_model/` — SegFormer-b5 checkpoint (untracked in git)

## Configuration
`config.json` controls CV thresholds, OSM query tags, stall area defaults, and minimum detection confidence. Adjust here before touching code.

# ParkSight Starter Kit

Estimate parking capacity from satellite imagery using computer vision and
machine learning. Built for **Hacklytics 2026**.

## The Challenge

Given an address, estimate how many parking stalls exist within a radius.
You get two baselines to beat:

1. **CV baseline** (fast, no ML) — Canny edge detection + Hough line counting
2. **ML baseline** (strong) — Grounding DINO zero-shot object detection

Your job: build something better.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd parksight-starter-kit
pip install -r requirements.txt

# Run the CLI baseline
python examples/run_baseline.py --address "Georgia Tech, Atlanta, GA"

# Or open the notebook
jupyter notebook notebooks/01_quickstart.ipynb
```

No API keys required. Everything runs on free public data (OSM + Esri tiles).

## How It Works

### CV Baseline (Tier 1)

```
Address → OSMnx geocode → fetch parking features (OSM)
  → For each Polygon:
    Esri satellite tile → grayscale → GaussianBlur → Canny → HoughLinesP
    → if <50 lines: area formula  else: lines/2
  → Sum counts → Folium map
```

### ML Baseline (Tier 2)

```
Satellite tile (PIL Image)
  → CLAHE preprocessing + morphological cleanup
  → Grounding DINO zero-shot detection (labels: "parking space", "car")
  → Bounding boxes → count detections
  → Annotated image
```

## Project Structure

```
parksight-starter-kit/
├── README.md                        # You are here
├── LICENSE                          # MIT
├── requirements.txt                 # pip install -r requirements.txt
├── config.json                      # Tunable CV parameters (see below)
│
├── parksight/                       # Python package
│   ├── fetch.py                     # OSM feature fetching + satellite tiles
│   ├── count.py                     # CV baseline counting
│   ├── detect.py                    # ML baseline (Grounding DINO)
│   ├── imagery.py                   # Image tiling + optional GEE helpers
│   ├── viz.py                       # Folium interactive maps
│   └── utils.py                     # Preprocessing (CLAHE, morphology, bbox)
│
├── notebooks/
│   ├── 01_quickstart.ipynb          # End-to-end walkthrough (~5 min)
│   └── 02_improve_counting.ipynb    # ML improvement scaffolding
│
├── examples/
│   └── run_baseline.py              # CLI: address → count + map
│
└── data/
    └── README.md                    # Links to training datasets
```

## Config Tuning Guide

Edit `config.json` to adjust the CV pipeline. Key parameters:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `CV2.gaussian_blur.ksize` | `[5, 5]` | Blur kernel size. Larger = smoother, fewer false edges |
| `CV2.gaussian_blur.sigma` | `1.5` | Blur standard deviation |
| `CV2.canny.threshold1` | `50` | Lower hysteresis threshold for edge detection |
| `CV2.canny.threshold2` | `150` | Upper hysteresis threshold for edge detection |
| `CV2.hough_lines_p.threshold` | `100` | Min votes to detect a line. Lower = more lines |
| `CV2.hough_lines_p.min_line_length` | `50` | Minimum line length in pixels |
| `CV2.hough_lines_p.max_line_gap` | `10` | Max gap between line segments to merge |
| `STALL_AREA_USA` | `32.0` | Average stall area in sq metres (area fallback) |
| `mu` | `0.7` | Drive aisle / landscaping buffer fraction |
| `AVG_CAR_LENGTH` | `4` | Average car length in metres (street parking) |

## Ideas to Beat the Baselines

### Better CV
- Adaptive thresholding instead of Canny
- Contour-based counting
- Template matching for stall line patterns

### Better ML
- Fine-tune Grounding DINO or YOLOv8 on parking-specific data
- Use SAM (Segment Anything) for stall segmentation
- Ensemble CV + ML predictions

### Novel Approaches
- Density estimation (count without detecting individuals)
- Semantic segmentation → pixel ratio → count
- Multi-scale tiling for large lots
- Temporal analysis (compare occupied vs empty)

## Datasets

See [`data/README.md`](data/README.md) for download links:
- **ParkSeg12k** — 12k aerial images with segmentation masks
- **APKLOT** — Aerial parking lots with bounding boxes
- **NAIP** — 1m US aerial imagery
- **SpaceNet** — Sub-metre satellite with building footprints

## License

MIT

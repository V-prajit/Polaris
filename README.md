# ParkSight

Full-stack parking intelligence from satellite imagery, OSM data, and ML.
Built for **Hacklytics 2026**.

Given any location, ParkSight estimates **total parking capacity**, **live vehicle counts**, and **utilization rates** — with confidence bands — across surface lots, garages, underground structures, and street parking.

## Architecture

```
                         ┌─────────────┐
                         │  /api/estimate  │
                         │  (lat, lon, r)  │
                         └──────┬──────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
        Surface Lots      Structures        Street Parking
        (Polygons)      (Garages/UG)       (LineStrings)
               │                │                │
               ▼                ▼                ▼
     ┌─────────────────┐  floor_area ×     curb_length ×
     │ Stage 1: WHERE   │  levels ×         0.8 × sides /
     │ SegFormer-b5     │  usable_frac /    car_length
     │ (ParkSeg12k)     │  stall_area
     │ → binary mask    │       │                │
     └────────┬────────┘       │                │
              ▼                │                │
     ┌─────────────────┐       │                │
     │ Stage 2: HOW MANY│       │                │
     │ YOLO (APKLOT)    │       │                │
     │ → spot count     │       │                │
     │ (mask-filtered)  │       │                │
     └────────┬────────┘       │                │
              ▼                │                │
     ┌─────────────────┐       │                │
     │ Stage 3: CARS    │       │                │
     │ YOLO (VisDrone/  │       │                │
     │ COCO) + SAHI     │       │                │
     │ → vehicle count  │       │                │
     └────────┬────────┘       │                │
              ▼                ▼                ▼
     ┌──────────────────────────────────────────────┐
     │          Confidence Bands + Utilization       │
     │  spots: {value, low, high, method}            │
     │  cars:  {value, low, high, method}            │
     │  utilization: cars / spots (propagated)       │
     └──────────────────────────────────────────────┘
```

### Estimation Methods (ranked by reliability)

| Method                     | What it does                                                   | Confidence |
| -------------------------- | -------------------------------------------------------------- | ---------- |
| **YOLO spot detection**    | APKLOT-trained model, each bbox = 1 stall                      | ±15%       |
| **SegFormer mask → count** | Binary parking mask → connected components / area              | ±18%       |
| **Geometric (ITE/NPA)**    | Reverse-engineer lot design: stall angle × module depth × rows | ±20-25%    |
| **Area heuristic**         | `lot_area / 15.5m² × 0.50`                                     | ±25-30%    |
| **Edge detection**         | Canny + Hough lines / 2                                        | ±25-30%    |
| **Garage estimate**        | `floor_area × levels × 0.60 / stall_area`                      | ±25-35%    |
| **Street estimate**        | `curb_length × 0.80 / car_length × sides`                      | ±15-25%    |

### Vehicle Counting (Overhead)

COCO-pretrained YOLO can't detect cars from satellite imagery (trained on street-level photos). Our solution:

1. **SAHI (Slicing Aided Hyper Inference)** — slice satellite tiles into 128×128 crops with 25% overlap, run YOLO on each crop, merge detections. Cars become large enough relative to each crop for COCO YOLO to recognise.
2. **VisDrone fine-tuned model** (planned) — YOLO fine-tuned on 6,471 drone/aerial images with vehicle classes (car, van, truck, bus). Combined with SAHI for best results.
3. **SegFormer mask filtering** — only count vehicles inside detected parking areas, reducing false positives.

## Quick Start

```bash
pip install -r requirements.txt

# API server
python -m api.app
# → http://localhost:8000/api/estimate?lat=33.7756&lon=-84.3963&radius=300

# CLI baseline
python examples/run_baseline.py --address "Georgia Tech, Atlanta, GA"

# Compare all estimation methods
python scripts/test_methods.py
```

## Quick Start (Frontend)

First, run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
ParkSight/
├── api/
│   └── app.py                      # FastAPI backend (/api/estimate, /api/macro)
├── parksight/
│   ├── fetch.py                    # OSM feature fetching + Esri satellite tiles
│   ├── count.py                    # Edge detection + geometric estimator
│   ├── detect.py                   # Grounding DINO zero-shot detector
│   ├── segment.py                  # SegFormer parking segmentation + counting
│   ├── confidence.py               # Confidence bands + utilization calculation
│   ├── estimate_structured.py      # Garage / underground / street estimators
│   ├── data.py                     # ParkSeg12k dataset loader for training
│   ├── imagery.py                  # Image tiling helpers
│   ├── viz.py                      # Folium interactive maps
│   └── utils.py                    # Preprocessing (CLAHE, morphology)
├── yolo/
│   └── detect.py                   # YOLO parking detector + SAHI car counter
├── scripts/
│   ├── train_segformer.py          # SegFormer-b5 fine-tuning (multi-GPU)
│   ├── test_methods.py             # Compare all estimation methods
│   ├── download_parkseg.py         # Download ParkSeg12k from HuggingFace
│   └── prepare_apklot_yolo.py      # Convert APKLOT to YOLO format
├── models/
│   ├── yolo26n_run1.pt             # ParkSeg-trained YOLO
│   └── best_model/                 # SegFormer checkpoint
├── src/                            # Next.js frontend code
├── config.json                     # All tuneable parameters
└── requirements.txt
```

## Training

### SegFormer-b5 (parking segmentation)

Fine-tuned on ParkSeg12k (11k satellite image/mask pairs) for binary parking stall segmentation.

```bash
# Single GPU
python scripts/train_segformer.py \
    --data_dir data/parkseg12k \
    --output_dir checkpoints/segformer-b5-parkseg \
    --epochs 30 --batch_size 32 --lr 6e-5 --amp \
    --early_stop_patience 7 --stall_class_ids 255

# Multi-GPU (8×H200)
torchrun --nproc_per_node=8 scripts/train_segformer.py \
    --data_dir data/parkseg12k \
    --output_dir checkpoints/segformer-b5-parkseg \
    --epochs 50 --batch_size 16 --lr 6e-5 --amp \
    --early_stop_patience 10 --stall_class_ids 255
```

### YOLO Vehicle Detection (planned)

Fine-tune on VisDrone for aerial vehicle counting:

```bash
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
```

## Datasets

- **ParkSeg12k** — 12k aerial images with binary parking masks (HuggingFace)
- **APKLOT** — 1,519 aerial parking lot images with bounding box annotations
- **VisDrone** — 6,471 drone images with vehicle annotations (car/van/truck/bus)

## API Response

```json
{
  "grand_total": 500,
  "spots": {"value": 500, "low": 375, "high": 650, "method": "default"},
  "cars": {"value": 42, "low": 35, "high": 50, "method": "yolo_car"},
  "utilization": {"value": 8.4, "low": 5.4, "high": 13.3},
  "surface": {
    "total": 320,
    "features": [{
      "name": "Surface #1",
      "count": 150,
      "spots": {"value": 150, "low": 127, "high": 173, "method": "yolo_detect"},
      "cars": {"value": 22, "low": 18, "high": 26, "method": "yolo_car"},
      "utilization": {"value": 14.7, "low": 10.4, "high": 20.5}
    }]
  },
  "structured": { "total": 130, "features": ["..."] },
  "street": { "total": 50, "features": ["..."] }
}
```

## License
MIT

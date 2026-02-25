# Polaris - Satellite-Powered Parking Intelligence

> **Winner - GrowthFactor Challenge @ [Hacklytics 2026](https://hacklytics-2026.devpost.com/) (Georgia Tech, Feb 20-22)**

**Polaris** is a scalable ML platform that quantifies parking inventory from space. Drop a pin anywhere in Atlanta, and in under 2 seconds it parses satellite imagery to deliver stall counts, confidence intervals, and a **0-100 Polaris Score** for parking accessibility - across surface lots, parking garages, underground structures, and street parking.

We chose **Track B (City-Wide Mapping)** and mapped the **entirety of Atlanta**, generating a comprehensive parking heatmap across the metro area using H3 hexagonal grids.

[Devpost](https://devpost.com/software/parasite-2dri43) | [GitHub](https://github.com/V-prajit/Polaris)

---

## Table of Contents

- [The Problem](#the-problem)
- [Our Solution](#our-solution)
- [Three Detection Approaches](#three-detection-approaches)
- [Beyond Surface Lots](#beyond-surface-lots)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Hackathon Context](#hackathon-context)
- [Team](#team)
- [License](#license)

---

## The Problem

Retailers need to know one deceptively simple thing: *how much parking is available near their locations?*

Today, answering this question requires expensive manual surveys or proprietary datasets that are instantly stale. Satellite imagery is abundant, high-resolution, and constantly updated - but turning raw pixels into accurate parking spot counts is an unsolved data science challenge.

**GrowthFactor**, a company that builds software for retail and real estate decision-makers, sponsored this challenge at Hacklytics 2026. Parking availability is one of the most requested - and hardest to obtain - data points their customers ask about.

---

## Our Solution

Polaris attacks the problem from three angles simultaneously, then fuses the results:

1. **SegFormer-b5** - Fine-tuned on the [ParkSeg12k](https://github.com/UTEL-UIUC/ParkSeg12k) dataset (12,617 satellite image/mask pairs across 45 US cities) for pixel-level parking lot segmentation
2. **YOLOv8 / YOLO11** - Detects individual parking stalls and vehicles from satellite tiles using SAHI (Slicing Aided Hyper Inference) for small-object detection
3. **Geometric Heuristics** - Uses OSM polygon boundaries + ITE/NPA parking design standards (angle-specific aisle widths, OBB layout simulation) to estimate capacity from lot geometry alone - near instant and highly accurate

For structured parking invisible from above (garages, underground, street parking), we supplement with **OpenStreetMap metadata**, building height data, and municipal records.

The final estimate is a **weighted ensemble** of all methods, producing confidence intervals and the **Polaris Score** - a 0-100 composite metric for parking accessibility.

---

## Three Detection Approaches

### 1. SegFormer-b5 - Semantic Segmentation

Fine-tuned `nvidia/segformer-b5-finetuned-ade-640-640` on the **ParkSeg12k** dataset to produce pixel-level parking masks. The pipeline:

- Fetches 512×512 Esri satellite tiles at zoom 19
- Runs SegFormer inference with **test-time augmentation** (4 orientations averaged)
- Applies morphological postprocessing (close > open > filter)
- Counts stalls via **connected-component analysis** with area thresholds (400-12,000 px)
- Falls back to **area-based estimation** using real-world stall dimensions (15.5 m2/stall)

**Strengths:** Extremely accurate lot boundary detection, works across diverse geographies  
**Weaknesses:** Computationally expensive, struggles to differentiate adjacent stalls

### 2. YOLO - Object Detection

Two YOLO models run in parallel:

| Model | Purpose | Dataset |
|-------|---------|---------|
| **Custom ParkSeg YOLO** | Detect individual parking stalls | Trained on ParkSeg12k + APKLOT |
| **YOLOv8s-VisDrone** | Detect and count vehicles | Pre-trained on VisDrone aerial dataset |

Both use **SAHI** (Slicing Aided Hyper Inference) to handle the small object sizes inherent in satellite imagery. Optimal configuration after grid search: `slice_size=256×256, overlap_ratio=0.4` (MAE of 6.0 vehicles per region).

**Strengths:** Fast inference, directly counts individual stalls and vehicles  
**Weaknesses:** Lower precision on densely packed lots

### 3. Geometric Heuristics - Layout Simulation

No ML required. Uses OSM polygon boundaries and parking engineering standards:

- **OBB-based layout simulation** - Fits parking rows at 45, 60, and 90 degree angles with angle-specific aisle widths (ITE/NPA standards: 7.3m for 90, 5.5m for 60, 4.0m for 45)
- **Aspect-ratio detection** - Narrow lots automatically switch to parallel/single-loaded layouts
- **Solidity scoring** - Uses convex hull ratio to penalize irregular shapes
- **OSM capacity blending** - When available, blends geometrically estimated values with official OSM `capacity` tags

**Strengths:** Near-instant, highly accurate for regular lots, no GPU needed  
**Weaknesses:** Struggles with irregular shapes and lots without OSM boundaries

---

## Beyond Surface Lots

A key differentiator of Polaris is handling parking that satellites **cannot see**:

| Type | Method | Data Source |
|------|--------|-------------|
| **Parking Garages** | Floor area x levels x 60% usable fraction / 15.5 m2/stall | OSM `building:levels`, `parking:levels` tags |
| **Underground** | Same formula, defaults to 2 levels if tag missing | OSM `parking=underground` |
| **Street Parking** | Curb length x 80% usable / 5.5m avg car length x sides | OSM `parking:lane`, `parking:left/right/both` tags |

When OSM has a `capacity` tag, estimates are **blended** (60% OSM + 40% geometric) for higher accuracy.

---

## Tech Stack

### Backend (Python 3.13+)
| Component | Technology |
|-----------|-----------|
| API Framework | **FastAPI** + Uvicorn |
| Segmentation | **SegFormer-b5** (HuggingFace Transformers) |
| Object Detection | **Ultralytics YOLO** (v8 + v11) |
| Geospatial | **OSMnx**, GeoPandas, Shapely, PyProj |
| Satellite Tiles | **Contextily** (Esri World Imagery) |
| Image Processing | OpenCV, Pillow, NumPy |
| Vector Search | **Actian VectorAI DB** + Gemini Embeddings |
| Caching | TTLCache (in-memory) + disk tile cache |

### Frontend (TypeScript)
| Component | Technology |
|-----------|-----------|
| Framework | **Next.js 16** (App Router) |
| UI | React 19, Tailwind CSS 4, Framer Motion |
| Maps | **Leaflet** + React-Leaflet |
| Components | Radix UI, Lucide Icons |
| Globe | **COBE** (WebGL globe) |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| GPU Inference | **Brev.dev** (H200 node) |
| Hosting | **Vultr** VPS (systemd services) |
| Vector Database | **Actian VectorAI DB** (Docker) |
| Embeddings | **Google Gemini** (`text-embedding-004`, 768d) |


## Getting Started

### Prerequisites

- **Python 3.13+** (with `uv` or `pip`)
- **Node.js 20+** (LTS)
- **Docker** (for Actian VectorAI DB, optional)
- **GPU recommended** for SegFormer/YOLO inference (CUDA or MPS)

### 1. Clone & Install

```bash
git clone https://github.com/V-prajit/Polaris.git
cd Polaris

# Python dependencies
pip install -r requirements.txt
# or with uv:
uv sync

# Frontend dependencies
npm install
```

### 2. Environment Variables

```bash
cp .env.production .env
```

Edit `.env` and add your API keys:

```env
GEMINI_API_KEY=your_gemini_key        # For semantic search embeddings
GOOGLE_MAPS_API_KEY=your_maps_key     # For POI enrichment (optional)
BACKEND_URL=http://localhost:8000      # API URL
```

### 3. Start the Vector Database (optional)

```bash
docker-compose up -d
```

### 4. Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### 5. Run the Frontend

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) - you'll see the Polaris globe. Search for any Atlanta location to get a parking analysis.

### Production Deployment

For a one-click deployment on a Vultr VPS:

```bash
bash scripts/deploy_vultr.sh
```

This sets up systemd services for both the API and frontend, configures the firewall, and starts the VectorAI DB container.

---

## API Reference

### `GET /api/estimate`

Point-level parking analysis for a single location.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lat` | float | *required* | Latitude (WGS84) |
| `lon` | float | *required* | Longitude (WGS84) |
| `radius` | int | 300 | Search radius in metres (50-2000) |

**Returns:** Surface lots with stall counts (SegFormer + YOLO + geometric), structured parking estimates, street parking estimates, confidence intervals, segmentation mask contours (GeoJSON), and the Polaris Score.

### `GET /api/macro`

City-wide parking heatmap using H3 hexagonal grid.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_lat`, `max_lat` | float | *required* | Bounding box latitude |
| `min_lon`, `max_lon` | float | *required* | Bounding box longitude |
| `resolution` | int | 9 | H3 grid resolution (9 = ~170m radius hexagons) |

**Returns:** Array of H3 hexagon cells, each with parking capacity estimates and metadata.

### `POST /api/polaris/search`

Semantic search over indexed parking profiles using natural language.

```json
{
  "query": "areas with lots of garage parking near restaurants",
  "top_k": 10,
  "min_spots": 50,
  "require_garage": true
}
```

**Returns:** Top-K matching hex cells ranked by semantic similarity via Gemini embeddings and Actian VectorAI DB.

---

## Benchmarks

### Method Comparison - Atlanta Test Locations

```
Georgia Tech, Atlanta (33.7756, -84.3963)
----------------------------------------------
  Lot #301779 (3,396 m2)
    Area heuristic:    109  [81-142]
    Edge detection:     31
    Geometric:          65  [52-82]
    YOLO spots:         84
    YOLO cars (SAHI):    7  [5-9]

Atlantic Station, Atlanta (33.757, -84.4015)
----------------------------------------------
  Lot #800491500 (6,478 m2)
    Area heuristic:    208  [156-271]
    Geometric:         122  [97-152]
    YOLO spots:         87
    YOLO cars (SAHI):    2  [1-2]
```

The best results come from averaging **YOLO spot detection** with the **SegFormer-informed geometric heuristic**, while other methods serve as validation cross-checks.

### Vehicle Detection Accuracy (SAHI Tuning)

| Location | Ground Truth | Baseline (128x128) | Tuned (256x256) |
|----------|:---:|:---:|:---:|
| Atlantic Station | ~55 | 73 | 47 |
| Turner Field Lot | ~27 | 26 | 21 |
| GT Parking | ~34 | 31 | 30 |

Best SAHI config: `slice=256x256, overlap=0.4` - **MAE: 6.0 vehicles/region**

---

## Hackathon Context

This project was built in **36 hours** for the **ParkSight** challenge at [Hacklytics 2026](https://hacklytics-2026.devpost.com/) (Georgia Tech, Feb 20-22), sponsored by [GrowthFactor](https://www.growthfactor.ai/).

### The Challenge

Build a pipeline that uses satellite or aerial imagery to map and count parking spots. Two tracks were offered:

- **Track A (Point Query):** Given a lat/long, estimate nearby parking capacity
- **Track B (City-Wide Mapping):** Generate a comprehensive parking map for an entire city *(more ambitious, weighted favorably)*

We chose **Track B** and delivered a full city-wide parking map of Atlanta, while also supporting point queries.

### Judging Criteria

| Criteria | Weight | Our Approach |
|----------|:---:|---|
| **Spot Detection Accuracy** | 40% | Three-method ensemble with confidence intervals, validated against ground truth at 3 Atlanta locations |
| **Technical Approach** | 25% | SegFormer fine-tuning, YOLO with SAHI, geometric layout simulation using ITE/NPA engineering standards |
| **Scalability & Generalization** | 20% | City-wide H3 hex mapping, works across different lot types (surface, structured, street) |
| **Presentation & Insight** | 15% | Interactive globe UI, real-time map dashboard with Polaris Score, semantic search |

### Data Sources Used

- [**ParkSeg12k**](https://github.com/UTEL-UIUC/ParkSeg12k) - 12,617 satellite image/mask pairs for SegFormer training
- [**APKLOT**](https://github.com/langheran/APKLOT) - 7,000 annotated polygons for YOLO training
- [**VisDrone**](https://github.com/VisDrone/VisDrone-Dataset) - Aerial vehicle detection dataset (pre-trained model)
- **Esri World Imagery** - High-res satellite tiles via Contextily
- **OpenStreetMap** - Parking polygons, building metadata, road networks, capacity tags
- **Atlanta Zoning Districts** - GeoJSON for zoning context

---

## Team

| Name | GitHub |
|---|---|
| **Prajit Viswanadha** | [@V-prajit](https://github.com/V-prajit) |
| **Shashank Yaji** | [@SSKYAJI](https://github.com/SSKYAJI) |
| **Jeevan Ramasamy** | [@JeevanandanRamasamy](https://github.com/JeevanandanRamasamy) |

**Sponsor Mentor:** Raj - Co-founder at [GrowthFactor](https://www.growthfactor.ai/)

---

## License

[MIT](LICENSE) - Copyright (c) 2026 GrowthFactor, Inc.
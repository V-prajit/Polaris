================================================================================
ParkSight Method Comparison
================================================================================
Loading YOLO models...
  Spot detector: ParkSeg
  Car detector:  COCO yolo11n loaded
  ML baseline:   Grounding DINO loaded
  Segmenter:     ParkSeg loaded
──────────────────────────────────────────────────────────────────────
📍 Georgia Tech, Atlanta (33.7756, -84.3963)
──────────────────────────────────────────────────────────────────────
  Lot #('relation', 301779) (3396 m²)
    Area heuristic:    109  [81-142]
    Edge detection:     31
    Geometric:          65  [52-82]
    YOLO spots:         84
    YOLO cars SAHI:      7  [5-9]
  Lot #('relation', 304240) (2856 m²)
    Area heuristic:     92  [69-120]
    Edge detection:     82
    Geometric:          29  [23-37]
    YOLO spots:          5
    YOLO cars SAHI:     14  [11-17]
  Lot #('relation', 304242) (2938 m²)
    Area heuristic:     94  [70-123]
    Edge detection:     57
    Geometric:           0  [0-0]
    YOLO spots:         94
    YOLO cars SAHI:     14  [11-17]
──────────────────────────────────────────────────────────────────────
📍 Atlantic Station, Atlanta (33.757, -84.4015)
──────────────────────────────────────────────────────────────────────
  Lot #('way', 800491500) (6478 m²)
    Area heuristic:    208  [156-271]
    Edge detection:     28
    Geometric:         122  [97-152]
    YOLO spots:         87
    YOLO cars SAHI:      2  [1-2]
  Lot #('way', 800491501) (8831 m²)
    Area heuristic:    284  [213-370]
    Edge detection:     72
    Geometric:         176  [140-220]
    YOLO spots:         93
    YOLO cars SAHI:      2  [1-2]
  Lot #('way', 853099020) (4709 m²)
    Area heuristic:    151  [113-197]
    Edge detection:     44
    Geometric:          94  [75-117]
    YOLO spots:        151
    YOLO cars SAHI:      1  [0-1]


The best results came out to be with YOLO Spot and the Segformer adn the math heuristic averaged out while the other methods were not even close with the results.
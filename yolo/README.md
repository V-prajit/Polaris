# Hacklytics 2026: ParkSight YOLO Pipeline

This folder contains the complete YOLO26 end-to-end training and inference pipeline for the counting of parking spots from satellite imagery.

## 🚀 Quick Start for A100 Training

To train the YOLO detector on a fresh server (like an A100 node), simply follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/Growth-Factor-AI/GrowthFactor-Parksight-Hacklytics-2026.git
cd GrowthFactor-Parksight-Hacklytics-2026
```

**2. Install requirements**
```bash
pip install -r requirements.txt
```

**3. Run the training script**
```bash
python yolo/train.py
```

> **Note**: You do **not** need to manually download or format the datasets! Running `train.py` will automatically detect if the dataset is missing, download it via HuggingFace, format the segmentation masks to YOLO bounding boxes, and begin the training on the A100 GPU immediately.

## 🎯 Running Inference (After Training)

Once the model finishes training, the best weights will be saved to `runs/parksight_yolo/weights/best.pt`.

To test the model and generate a map visualization, run:
```bash
python yolo/run.py
```

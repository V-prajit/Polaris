#!/bin/bash
# train_visdrone.sh
# Run this on your H200 instance to fine-tune the YOLO model on VisDrone.

# Ensure ultralytics is installed
pip install ultralytics

# Run the training script we just modified for the H200
python scripts/train_visdrone.py

echo "Training complete! Your weights should be in runs/visdrone_aerial_cars/weights/best.pt"

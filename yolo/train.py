import os
from ultralytics import YOLO

def main():
    if not os.path.exists("data/yolo_dataset/data.yaml"):
        import sys
        print("YOLO dataset not found. Running prepare_dataset.py...")
        os.system(f"{sys.executable} yolo/prepare_dataset.py")
        print("Dataset preparation complete.")
        
    model = YOLO("yolo26n.pt")  # Using YOLO26 nano for best performance/speed
    
    # Updated for Nvidia A100 GPU
    results = model.train(
        data="data/yolo_dataset/data.yaml",
        epochs=100,          # Increased for fast training on A100
        imgsz=512,
        batch=128,           # Large batch size to utilize massive A100 VRAM
        device="0",          # Use Nvidia CUDA GPU
        project="runs",
        name="parksight_yolo",
        patience=10,         # Patience increased for longer training
        save=True,
        plots=True,
        workers=8,           # More workers for faster data loading
        amp=True,            # Enable automatic mixed precision
    )

if __name__ == "__main__":
    main()

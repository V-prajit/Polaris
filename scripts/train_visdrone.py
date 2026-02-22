from ultralytics import YOLO

def main():
    # Start from COCO yolov8s.pt for a clean slate, or use the downloaded HF weights.
    # The prompt says: "model = YOLO('yolov8s.pt') # or yolov8s-visdrone.pt if available"
    # Let's use yolov8s.pt since the HF weights were so bad, maybe they were corrupted or poorly trained.
    model = YOLO("yolov8s.pt")
    
    # Train as per the instructions
    results = model.train(
        data="VisDrone.yaml",
        epochs=30,
        imgsz=640,
        batch=64,
        device="0",
        project="runs",
        name="visdrone_aerial_cars",
        patience=10,
        save=True,
        amp=True,
        workers=8,
        classes=[3, 4, 5, 8], # car, van, truck, bus
    )

if __name__ == "__main__":
    main()

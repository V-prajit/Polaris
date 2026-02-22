import contextily as ctx
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ground_truths = {
    "test_Atlantic_Station.png": 55,
    "test_Turner_Field_Lot.png": 27,
    "test_GT_Parking.png": 34
}

slice_sizes = [128, 256, 320]
overlaps = [0.2, 0.3, 0.4]

model_path = "yolov8s-visdrone.pt"

print("Initializing SAHI model...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,
    device="mps",
)

best_config = None
best_mae = float('inf')
results_summary = []

for size in slice_sizes:
    for overlap in overlaps:
        print(f"\n--- Testing size={size}, overlap={overlap} ---")
        total_error = 0
        config_counts = {}
        for img_path, gt_count in ground_truths.items():
            result = get_sliced_prediction(
                img_path,
                sahi_model,
                slice_height=size,
                slice_width=size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0,
            )
            vehicle_count = 0
            for pred in result.object_prediction_list:
                cat = pred.category.name.lower()
                if any(v in cat for v in ["car", "van", "truck", "bus", "vehicle"]):
                    vehicle_count += 1
            
            error = abs(vehicle_count - gt_count)
            total_error += error
            config_counts[img_path] = vehicle_count
            print(f"{img_path}: Predicted={vehicle_count}, GT={gt_count}")
        
        mae = total_error / len(ground_truths)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_config = (size, overlap)
        
        results_summary.append({
            "size": size,
            "overlap": overlap,
            "mae": mae,
            "counts": config_counts
        })

print(f"\nBest Config: size={best_config[0]}, overlap={best_config[1]} with MAE={best_mae:.2f}")

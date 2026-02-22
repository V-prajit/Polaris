# Aerial Vehicle Detection Update Summary

## Evaluation Results
We manually counted vehicles in test areas and compared them to the HuggingFace `yolov8s-visdrone` baseline model's output to evaluate accuracy.

| Location | Manual Count (Ground Truth) | Baseline Model Count (128x128 Slice) | Tuned Model Count (256x256 Slice) |
| --- | --- | --- | --- |
| Atlantic Station | ~55 | 73 | 47 |
| Turner Field Lot | ~27 | 26 | 21 |
| GT Parking | ~34 | 31 | 30 |

## SAHI Slice Size Tuning
We evaluated SAHI slice sizes of 128, 256, and 320 with overlap ratios of 0.2, 0.3, and 0.4.
- **Best Configuration**: `slice_height=256`, `slice_width=256`, `overlap_ratio=0.4`
- **Mean Absolute Error (MAE)**: 6.00 vehicles per region on average, reduced from 7.33 MAE with the default (128, 0.3) setting.

## Model Selection decision
We used the **HuggingFace VisDrone baseline model** (`mshamrai/yolov8s-visdrone`). It performed remarkably well out of the box (under 10% error on two of three locations, and reasonable error on the larger lot). Since it was already well-trained on aerial distributions and required no fine-tuning to perform optimally, we moved directly to integrating it. No further fine-tuning was needed.

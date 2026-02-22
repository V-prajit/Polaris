import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if float(boxAArea + boxBArea - interArea) == 0: return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def nms(boxes_with_scores, iou_thresh=0.3):
    # boxes_with_scores: list of (x1, y1, x2, y2, score, cat_name)
    boxes_with_scores.sort(key=lambda x: x[4], reverse=True)
    kept_boxes = []
    for current in boxes_with_scores:
        overlap = False
        for kept in kept_boxes:
            if compute_iou(current[:4], kept[:4]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(current)
    return kept_boxes

img_path = "test_Atlantic_Station.png"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    exit(1)

model_path = "models/yolo_aerial_cars.pt"
print(f"Loading model from {model_path}...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,
    device="mps",
)

from PIL import Image, ImageDraw, ImageOps, ImageEnhance

# ... (omitted compute_iou and nms for brevity in this snippet)

original_img = Image.open(img_path).convert("RGB")
# Create inverted version
inverted_img = ImageOps.invert(original_img)
inverted_path = "test_Atlantic_Station_inv.png"
inverted_img.save(inverted_path)

# Create brightened version (lifts shadows for black cars)
bright_img = ImageEnhance.Brightness(original_img).enhance(1.5)
bright_path = "test_Atlantic_Station_bright.png"
bright_img.save(bright_path)

print(f"Running pass 1 (Original) on {img_path}...")
res1 = get_sliced_prediction(
    img_path, sahi_model,
    slice_height=128, slice_width=128,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0
)

print(f"Running pass 2 (Inverted) on {inverted_path}...")
res2 = get_sliced_prediction(
    inverted_path, sahi_model,
    slice_height=128, slice_width=128,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0
)

print(f"Running pass 3 (Brightened) on {bright_path}...")
res3 = get_sliced_prediction(
    bright_path, sahi_model,
    slice_height=128, slice_width=128,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0
)

os.unlink(inverted_path)
os.unlink(bright_path)

all_preds = []
for p in res1.object_prediction_list + res2.object_prediction_list + res3.object_prediction_list:
    x1, y1, x2, y2 = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
    score = p.score.value
    cat = p.category.name.lower()
    # explicitly only keep vehicles
    if any(v in cat for v in ["car", "van", "truck", "bus"]):
        if score >= 0.35: # Dial back from 0.42 to 0.35 to restore valid cars
            all_preds.append((x1, y1, x2, y2, score, cat))

# Lower IoU threshold to 0.20 to aggressively merge overlapping TTA boxes
final_boxes = nms(all_preds, iou_thresh=0.20)

all_entities_img = original_img.copy()
just_cars_img = original_img.copy()
draw_all = ImageDraw.Draw(all_entities_img)
draw_cars = ImageDraw.Draw(just_cars_img)

all_count = len(final_boxes)
car_count = 0

found_classes = set()

for (x1, y1, x2, y2, score, cat_name) in final_boxes:
    found_classes.add(cat_name)
    # All vehicles
    draw_all.rectangle([x1, y1, x2, y2], outline="lime", width=2)
    draw_all.text((x1, max(0, y1-10)), f"{cat_name} {score:.2f}", fill="lime")
    
    # Just cars
    if cat_name == 'car':
        draw_cars.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
        car_count += 1

print(f"Classes present in image: {found_classes}")

print(f"Total vehicles detected post-NMS: {all_count}")
print(f"Total 'cars' detected post-NMS: {car_count}")

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(original_img)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(all_entities_img)
axs[1].set_title(f"All Detected (2 Passes + NMS, n={all_count})")
axs[1].axis('off')

axs[2].imshow(just_cars_img)
axs[2].set_title(f"Just Cars (n={car_count})")
axs[2].axis('off')

plt.tight_layout()
out_file = "visual_comparison.png"
plt.savefig(out_file, dpi=150)
print(f"Saved side-by-side comparison to {out_file}")

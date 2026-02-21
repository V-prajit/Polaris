import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
import shutil
import yaml

try:
    import datasets
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "pyarrow", "fsspec"])
    import datasets

def get_yolo_format(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return 0, cx, cy, w, h

def process_apklot():
    apklot_dir = "data/APKLOT/1. Satellite/Dataset"
    xml_files = glob.glob(f"{apklot_dir}/*/PASCAL_format/Annotations/*.xml")
    
    samples = []
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        img_name = root.find('filename').text
        xml_dir = os.path.dirname(xml_file)
        jpeg_dir = os.path.join(os.path.dirname(xml_dir), "JPEGImages")
        img_path = os.path.join(jpeg_dir, img_name)
        
        if not os.path.exists(img_path):
            stem = os.path.splitext(os.path.basename(xml_file))[0]
            img_path = os.path.join(jpeg_dir, stem + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(jpeg_dir, stem + ".png")
        
        if not os.path.exists(img_path):
            continue
            
        size = root.find('size')
        if size is None:
            img = cv2.imread(img_path)
            if img is None: continue
            img_h, img_w = img.shape[:2]
        else:
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
        bboxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
            
        yolo_labels = []
        for box in bboxes:
            c, rx, ry, rw, rh = get_yolo_format(box, img_w, img_h)
            yolo_labels.append(f"{c} {rx:.6f} {ry:.6f} {rw:.6f} {rh:.6f}")
            
        samples.append({
            'image_path': img_path,
            'labels': yolo_labels
        })
        
    return samples

def process_parkseg():
    print("Loading ParkSeg12k from HuggingFace...")
    ds = datasets.load_dataset('UTEL-UIUC/parkseg12k', streaming=True)
    
    samples = []
    
    for split in ['train', 'test']:
        if split not in ds:
            continue
        for i, item in enumerate(ds[split]):
            image = item['rgb']
            mask = item['mask']
            
            mask_np = np.array(mask)
            if len(mask_np.shape) > 2:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            
            _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_w, img_h = image.size
            yolo_labels = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    c, rx, ry, rw, rh = get_yolo_format([x, y, x+w, y+h], img_w, img_h)
                    yolo_labels.append(f"{c} {rx:.6f} {ry:.6f} {rw:.6f} {rh:.6f}")
                    
            if len(yolo_labels) > 0:
                samples.append({
                    'pil_image': image,
                    'labels': yolo_labels
                })
                
            if len(samples) % 1000 == 0:
                print(f"Processed {len(samples)} ParkSeg12k images...")
                
    return samples

def main():
    out_dir = "data/yolo_dataset"
    os.makedirs(os.path.join(out_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val", "labels"), exist_ok=True)
    
    print("Processing APKLOT...")
    apklot_samples = process_apklot()
    print(f"Found {len(apklot_samples)} APKLOT images")
    
    print("Processing ParkSeg12k...")
    parkseg_samples = process_parkseg()
    print(f"Found {len(parkseg_samples)} ParkSeg12k images")
    
    all_samples = apklot_samples + parkseg_samples
    random.shuffle(all_samples)
    
    val_size = int(len(all_samples) * 0.2)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    
    def save_samples(samples, split):
        print(f"Saving {split} samples...")
        for i, s in enumerate(samples):
            img_name = f"{split}_{i}.jpg"
            img_path = os.path.join(out_dir, split, "images", img_name)
            txt_path = os.path.join(out_dir, split, "labels", img_name.replace(".jpg", ".txt"))
            
            if 'pil_image' in s:
                s['pil_image'].convert("RGB").save(img_path)
            else:
                shutil.copy(s['image_path'], img_path)
                
            with open(txt_path, 'w') as f:
                f.write("\n".join(s['labels']))
                
    save_samples(train_samples, "train")
    save_samples(val_samples, "val")
    
    yaml_path = os.path.join(out_dir, "data.yaml")
    yaml_config = {
        'path': os.path.abspath(out_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'parking_region'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
        
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()

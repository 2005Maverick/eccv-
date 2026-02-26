"""
=================================================================
  ECCV v3 — Kaggle GPU Annotation Notebook
  
  This script is designed to run in a Kaggle environment with GPU.
  It takes a directory of images and produces annotations.jsonl.
  
  Features:
  - YOLOv8 (Ultralytics) for object detection
  - MiDaS for depth estimation
  - EasyOCR for text extraction
  - Texture analysis (statistical features)
=================================================================
"""

import os
import json
import time
import hashlib
import argparse
from datetime import timedelta
from io import BytesIO
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# Device Setup
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  >> DEVICE: {DEVICE}")

# ============================================================
# Models Initialization
# ============================================================

# PyTorch 2.6+ defaults weights_only=True which breaks ultralytics
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": kw.get("weights_only", False)})

# 1. YOLOv8
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt").to(DEVICE)
    print("  >> YOLOv8: LOADED")
except ImportError:
    print("  >> YOLOv8: FAILED (pip install ultralytics)")
    yolo_model = None

# 2. MiDaS Depth
try:
    midas_type = "DPT_Large"  # Highest quality
    midas = torch.hub.load("intel-isl/MiDaS", midas_type)
    midas.to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if midas_type == "DPT_Large" else midas_transforms.small_transform
    print(f"  >> MiDaS ({midas_type}): LOADED")
except Exception as e:
    print(f"  >> MiDaS: FAILED ({e})")
    midas = None

# 3. EasyOCR
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))
    print("  >> EasyOCR: LOADED")
except ImportError:
    print("  >> EasyOCR: FAILED (pip install easyocr)")
    reader = None

# ============================================================
# Processors
# ============================================================

def process_yolo(img_path):
    if not yolo_model: return {"detections": []}
    results = yolo_model(img_path, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            box = boxes[i].xyxy[0].cpu().tolist()
            cls = int(boxes[i].cls[0])
            conf = float(boxes[i].conf[0])
            label = yolo_model.names[cls]
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(x, 1) for x in box]
            })
    return {"detections": detections}

def process_depth(img_path):
    if not midas: return {}
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    
    # Simple depth stats
    h, w = output.shape
    left_part = output[:, :w//2]
    right_part = output[:, w//2:]
    
    return {
        "mean_depth": float(np.mean(output)),
        "std_depth": float(np.std(output)),
        "left_depth_mean": float(np.mean(left_part)),
        "right_depth_mean": float(np.mean(right_part)),
    }

def process_ocr(img_path):
    if not reader: return {"text_instances": []}
    results = reader.readtext(img_path)
    instances = []
    for (bbox, text, prob) in results:
        instances.append({
            "text": text,
            "confidence": round(float(prob), 3),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox]
        })
    return {"text_instances": instances}

def process_texture(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return {}
    
    # Statistical texture features
    mean = np.mean(img)
    std = np.std(img)
    
    # Edge density (Sobel)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # Laplacian variance (blurriness metric)
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    return {
        "brightness_mean": float(mean),
        "brightness_std": float(std),
        "edge_density": float(edge_density),
        "blurriness": float(lap_var)
    }

# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/kaggle/input/eccv-v3-images/images")
    parser.add_argument("--output", type=str, default="annotations.jsonl")
    args = parser.parse_args()
    
    if not os.path.exists(args.img_dir):
        print(f"  ERROR: Image directory not found: {args.img_dir}")
        return

    images = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    
    print(f"  Total images to process: {len(images):,}")
    
    start_time = time.time()
    processed = 0
    
    with open(args.output, "w", encoding="utf-8") as f:
        for img_name in tqdm(images, desc="Annotating"):
            image_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(args.img_dir, img_name)
            
            try:
                ann = {
                    "image_id": image_id,
                    "detections": process_yolo(img_path),
                    "depth": process_depth(img_path),
                    "ocr": process_ocr(img_path),
                    "texture": process_texture(img_path),
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as e:
                print(f"  [ERROR] {image_id}: {e}")
                
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {processed:,}")
    print(f"  Time:      {str(timedelta(seconds=int(total_time)))}")
    print(f"  Rate:      {processed/total_time:.2f} img/sec")
    print(f"  Output:    {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    from datetime import datetime
    main()

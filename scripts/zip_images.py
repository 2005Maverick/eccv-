"""Zip the collected images and manifest for Kaggle upload."""
import os, zipfile, json
from datetime import datetime

DATASET_DIR = r"C:\Users\Dell\Desktop\ECCV DATASET\dataset_final"
IMG_DIR = os.path.join(DATASET_DIR, "images")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")
OUT_FILE = os.path.join(DATASET_DIR, f"eccv_v3_source_images_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")

# 1. Load manifest to find which images to include
if not os.path.exists(MANIFEST_PATH):
    print("Error: manifest.jsonl not found.")
    exit(1)

image_ids = []
for line in open(MANIFEST_PATH, "r", encoding="utf-8"):
    rec = json.loads(line)
    image_ids.append(rec["image_id"])

print(f"Found {len(image_ids)} images in manifest.")

# 2. Zip
with zipfile.ZipFile(OUT_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add manifest
    zipf.write(MANIFEST_PATH, "manifest.jsonl")
    
    # Add images
    count = 0
    for img_id in image_ids:
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(IMG_DIR, f"{img_id}{ext}")
            if os.path.exists(path):
                zipf.write(path, os.path.join("images", f"{img_id}{ext}"))
                count += 1
                if count % 100 == 0:
                    print(f"  Zipped {count}/{len(image_ids)} images...")
                break

print(f"\nDone! Zipped {count} images.")
print(f"File: {OUT_FILE}")
print(f"Size: {os.path.getsize(OUT_FILE) / 1024 / 1024:.1f} MB")

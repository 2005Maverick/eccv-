"""Rebuild manifest.jsonl from images on disk when persistence failed."""
import os, json, hashlib
from PIL import Image

DATASET_DIR = r"C:\Users\Dell\Desktop\ECCV DATASET\dataset_final"
IMG_DIR = os.path.join(DATASET_DIR, "images")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.jsonl")

# 1. Load existing manifest
existing = {}
if os.path.exists(MANIFEST_PATH):
    for line in open(MANIFEST_PATH, "r", encoding="utf-8"):
        rec = json.loads(line)
        existing[rec["image_id"]] = rec

print(f"Loaded {len(existing)} records from manifest.")

# 2. Scan disk
files = [f for f in os.listdir(IMG_DIR) if f.startswith("vlm_") and f.endswith(".jpg")]
files.sort()

new_merged = 0
for f in files:
    img_id = os.path.splitext(f)[0]
    if img_id not in existing:
        path = os.path.join(IMG_DIR, f)
        # Create skeleton record
        try:
            img = Image.open(path)
            # phash
            try:
                import imagehash
                phash = str(imagehash.phash(img))
            except:
                phash = hashlib.md5(open(path, "rb").read(2048)).hexdigest()[:16]
            
            existing[img_id] = {
                "image_id": img_id,
                "url": "reconstructed_from_disk",
                "local_path": path,
                "source": "reconstructed",
                "license": "unknown",
                "phash": phash,
                "metadata": {
                    "width": img.size[0],
                    "height": img.size[1],
                    "title": f
                }
            }
            new_merged += 1
        except:
            continue

print(f"Added {new_merged} records from disk.")

# 3. Save
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    # Sort by ID
    for img_id in sorted(existing.keys()):
        f.write(json.dumps(existing[img_id], ensure_ascii=False) + "\n")

print(f"Saved total {len(existing)} records to {MANIFEST_PATH}.")

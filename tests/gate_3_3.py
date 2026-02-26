"""Gate 3.3 — Annotation pipeline output is complete and cacheable"""
import sys
import os
import json
import time
sys.path.insert(0, ".")

import numpy as np
from PIL import Image

# Create 5 test fixture images
os.makedirs("tests/fixtures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

for i in range(1, 6):
    img_path = f"tests/fixtures/test_{i:03d}.jpg"
    if not os.path.exists(img_path):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img.save(img_path)

# Clear any existing annotations for test
test_output = "data/processed/annotations.jsonl"
if os.path.exists(test_output):
    # Read existing, remove test entries, rewrite
    with open(test_output, "r") as f:
        lines = [l for l in f if '"test_' not in l]
    with open(test_output, "w") as f:
        f.writelines(lines)

from src.processing.annotation_pipeline import AnnotationPipeline

pipeline = AnnotationPipeline()

# Run on 5 fixture images
pipeline.run(
    image_ids=["test_001", "test_002", "test_003", "test_004", "test_005"],
    images_dir="tests/fixtures/"
)

required_keys = {'image_id', 'detections', 'depth', 'shadow', 'ocr', 'texture', 'temporal'}

with open("data/processed/annotations.jsonl") as f:
    records = [json.loads(l) for l in f if '"test_' in l]

assert len(records) == 5, f"Expected 5 records, got {len(records)}"

for rec in records:
    missing = required_keys - set(rec.keys())
    assert not missing, f"Missing keys {missing} in record {rec.get('image_id')}"

# Run again — should use cache, not reprocess
t0 = time.time()
pipeline.run(
    image_ids=["test_001", "test_002", "test_003", "test_004", "test_005"],
    images_dir="tests/fixtures/"
)
elapsed = time.time() - t0
assert elapsed < 2.0, f"Cache not working — reprocessing took {elapsed:.1f}s"

print("GATE 3.3 PASSED")

"""Gate 2.3 — Checkpoint-resume skips already-downloaded images"""
import sys
import os
import json

sys.path.insert(0, ".")

from src.collection.pipeline import CollectionPipeline
from unittest.mock import patch

# Ensure data directories exist
os.makedirs("data/raw", exist_ok=True)

pipeline = CollectionPipeline(config_path="configs/pipeline.yaml")

# Pre-write 3 image IDs to manifest as if already downloaded
with open("data/raw/manifest.jsonl", "w") as f:
    for i in range(3):
        f.write(json.dumps({
            "image_id": f"img_{i:04d}",
            "source": "wikimedia",
            "local_path": f"data/raw/images/img_{i:04d}.jpg",
            "url": "http://example.com",
            "license": "CC",
            "metadata": {}
        }) + "\n")

# Reload checkpoint
pipeline._load_checkpoint()

with patch.object(pipeline, '_download_image', return_value=True) as mock_dl:
    pipeline.run(target_n=3, resume=True)
    assert mock_dl.call_count == 0, f"Expected 0 downloads, got {mock_dl.call_count}"

print("GATE 2.3 PASSED")

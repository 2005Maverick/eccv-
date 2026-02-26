"""
Open Images v7 collector.
Uses the Open Images CSV manifest to collect bbox-annotated images
with stratified sampling across object categories.
"""

import os
import csv
import logging
import random
from typing import List, Optional, Dict
from collections import defaultdict

import requests
from PIL import Image as PILImage
from io import BytesIO

from src.collection.base_collector import BaseCollector, ImageRecord

logger = logging.getLogger(__name__)

# Default path to the Open Images metadata CSV
DEFAULT_MANIFEST_PATH = "data/raw/yfcc_metadata.csv"


class OpenImagesCollector(BaseCollector):
    """
    Collects images from Open Images v7 using a CSV manifest.

    Filters for images with bounding box annotations, ensures minimum
    50 images per object class, and returns stratified samples.
    """

    def __init__(
        self,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        output_dir: str = "data/raw/images",
        min_per_class: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source_name = "openimages"
        self.manifest_path = manifest_path
        self.output_dir = os.path.join(output_dir, self.source_name)
        self.min_per_class = min_per_class
        os.makedirs(self.output_dir, exist_ok=True)

        self._entries_by_class: Dict[str, List[dict]] = defaultdict(list)
        self._loaded = False
        self._used_ids = set()

    def _load_manifest(self):
        """Load and index the Open Images CSV manifest by object class."""
        if self._loaded:
            return

        if not os.path.exists(self.manifest_path):
            logger.warning(
                f"Open Images manifest not found at {self.manifest_path}. "
                f"Skipping this source."
            )
            self._loaded = True
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter: must have bounding box
                    has_bbox = row.get("has_bbox", "").lower() in ("true", "1", "yes")
                    if not has_bbox:
                        continue

                    obj_class = row.get("object_class", row.get("label", "unknown"))
                    self._entries_by_class[obj_class].append(row)

            # Filter out classes with fewer than min_per_class images
            self._entries_by_class = {
                cls: entries
                for cls, entries in self._entries_by_class.items()
                if len(entries) >= self.min_per_class
            }

            total = sum(len(v) for v in self._entries_by_class.values())
            logger.info(
                f"Open Images manifest loaded: {len(self._entries_by_class)} classes, "
                f"{total} images with bboxes"
            )
        except Exception as e:
            logger.error(f"Failed to load Open Images manifest: {e}")

        self._loaded = True

    def _stratified_sample(self, n: int) -> List[dict]:
        """
        Sample n entries stratified across object categories.

        Args:
            n: Total number of entries to sample.

        Returns:
            List of CSV row dicts, stratified by class.
        """
        self._load_manifest()

        if not self._entries_by_class:
            return []

        classes = list(self._entries_by_class.keys())
        per_class = max(1, n // len(classes))
        samples = []

        for cls in classes:
            available = [
                e for e in self._entries_by_class[cls]
                if e.get("image_id", e.get("ImageID", "")) not in self._used_ids
            ]
            class_sample = random.sample(available, min(per_class, len(available)))
            samples.extend(class_sample)

            if len(samples) >= n:
                break

        return samples[:n]

    def _download_image(self, url: str, image_id: str) -> Optional[str]:
        """Download an image and save locally."""
        local_path = os.path.join(self.output_dir, f"{image_id}.jpg")

        if os.path.exists(local_path):
            return local_path

        try:
            response = self._retry_with_backoff(
                requests.get, url, timeout=60, stream=True
            )
            response.raise_for_status()

            img = PILImage.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(local_path, "JPEG", quality=95)
            return local_path
        except Exception as e:
            logger.warning(f"Failed to download Open Images {url}: {e}")
            return None

    def fetch_batch(self, n: int) -> List[ImageRecord]:
        """
        Fetch a stratified batch of images from Open Images.

        Args:
            n: Number of images to fetch.

        Returns:
            List of ImageRecord objects.
        """
        samples = self._stratified_sample(n)
        records = []

        for entry in samples:
            image_id_raw = entry.get("image_id", entry.get("ImageID", ""))
            url = entry.get("url", entry.get("OriginalURL", ""))

            if not url:
                continue

            image_id = self._generate_image_id(url)
            local_path = self._download_image(url, image_id)

            if local_path is None:
                continue

            self._used_ids.add(image_id_raw)

            obj_class = entry.get("object_class", entry.get("label", "unknown"))
            record = ImageRecord(
                image_id=image_id,
                url=url,
                local_path=local_path,
                source="openimages",
                metadata={
                    "object_class": obj_class,
                    "has_bbox": True,
                    "original_id": image_id_raw,
                },
                license=entry.get("license", "CC BY 4.0"),
            )
            records.append(record)

        return records

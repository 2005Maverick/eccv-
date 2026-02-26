"""
YFCC100M dataset collector.
Uses YFCC100M metadata CSV to collect outdoor, GPS-tagged, CC-licensed images.
"""

import os
import csv
import logging
import random
from typing import List, Optional
from datetime import datetime

import requests
from PIL import Image as PILImage
from io import BytesIO

from src.collection.base_collector import BaseCollector, ImageRecord

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = "data/raw/yfcc_metadata.csv"


class YFCCCollector(BaseCollector):
    """
    Collects images from YFCC100M metadata CSV.

    Filters for:
    - has_gps=True
    - is_outdoor=True
    - CC license
    - year < 2020
    Metadata includes: timestamp, lat, lon, camera_model.
    """

    def __init__(
        self,
        metadata_path: str = DEFAULT_METADATA_PATH,
        output_dir: str = "data/raw/images",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source_name = "yfcc"
        self.metadata_path = metadata_path
        self.output_dir = os.path.join(output_dir, self.source_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self._entries: List[dict] = []
        self._loaded = False
        self._current_idx = 0

    def _load_metadata(self):
        """Load and filter YFCC metadata CSV."""
        if self._loaded:
            return

        if not os.path.exists(self.metadata_path):
            logger.warning(
                f"YFCC metadata CSV not found at {self.metadata_path}. "
                f"Skipping this source."
            )
            self._loaded = True
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter: must have GPS
                    has_gps = row.get("has_gps", "").lower() in ("true", "1", "yes")
                    lat = row.get("lat", row.get("latitude", ""))
                    lon = row.get("lon", row.get("longitude", ""))

                    if not has_gps and not (lat and lon):
                        continue

                    # Filter: outdoor
                    is_outdoor = row.get("is_outdoor", "").lower() in ("true", "1", "yes")
                    if not is_outdoor:
                        continue

                    # Filter: CC license
                    license_val = row.get("license", "").lower()
                    if not any(cc in license_val for cc in ["cc", "creative commons", "public"]):
                        continue

                    # Filter: year < 2020
                    timestamp = row.get("timestamp", row.get("date_taken", ""))
                    try:
                        year = int(timestamp[:4]) if timestamp and len(timestamp) >= 4 else 9999
                    except ValueError:
                        year = 9999

                    if year >= 2020:
                        continue

                    self._entries.append(row)

            # Shuffle for variety
            random.seed(42)
            random.shuffle(self._entries)

            logger.info(f"YFCC metadata loaded: {len(self._entries)} qualifying images")
        except Exception as e:
            logger.error(f"Failed to load YFCC metadata: {e}")

        self._loaded = True

    def _download_image(self, url: str, image_id: str) -> Optional[str]:
        """Download and save an image."""
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
            logger.warning(f"Failed to download YFCC image {url}: {e}")
            return None

    def fetch_batch(self, n: int) -> List[ImageRecord]:
        """
        Fetch a batch of YFCC images.

        Args:
            n: Number of images to fetch.

        Returns:
            List of ImageRecord objects.
        """
        self._load_metadata()
        records = []

        while len(records) < n and self._current_idx < len(self._entries):
            entry = self._entries[self._current_idx]
            self._current_idx += 1

            url = entry.get("url", entry.get("download_url", ""))
            if not url:
                continue

            image_id = self._generate_image_id(url)
            local_path = self._download_image(url, image_id)

            if local_path is None:
                continue

            # Extract metadata fields
            lat = entry.get("lat", entry.get("latitude", ""))
            lon = entry.get("lon", entry.get("longitude", ""))
            timestamp = entry.get("timestamp", entry.get("date_taken", ""))
            camera_model = entry.get("camera_model", entry.get("camera", "unknown"))

            record = ImageRecord(
                image_id=image_id,
                url=url,
                local_path=local_path,
                source="yfcc",
                metadata={
                    "timestamp": timestamp,
                    "lat": float(lat) if lat else None,
                    "lon": float(lon) if lon else None,
                    "camera_model": camera_model,
                },
                license=entry.get("license", "CC"),
            )
            records.append(record)

        return records

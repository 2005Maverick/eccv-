"""
Base collector module with ImageRecord dataclass and abstract BaseCollector.
Provides retry logic, rate limiting, and pHash deduplication via SQLite.
"""

import os
import time
import sqlite3
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Represents a single collected image with metadata."""
    image_id: str
    url: str
    local_path: str
    source: str
    metadata: Dict[str, Any]
    license: str


class DuplicateDetector:
    """Perceptual hash deduplication using SQLite storage."""

    def __init__(self, db_path: str = "data/raw/seen_hashes.db", threshold: int = 10):
        """
        Initialize the duplicate detector.

        Args:
            db_path: Path to SQLite database for storing perceptual hashes.
            threshold: Hamming distance threshold. Images with distance < threshold
                       are considered duplicates and will be skipped.
        """
        self.db_path = db_path
        self.threshold = threshold
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Create the hash table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_hashes (
                image_id TEXT PRIMARY KEY,
                phash TEXT NOT NULL,
                source TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def compute_phash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash for an image file."""
        try:
            img = Image.open(image_path)
            h = imagehash.phash(img)
            return str(h)
        except Exception as e:
            logger.warning(f"Failed to compute pHash for {image_path}: {e}")
            return None

    def is_duplicate(self, image_path: str, image_id: str) -> bool:
        """
        Check if an image is a near-duplicate of any previously seen image.

        Args:
            image_path: Path to the image file to check.
            image_id: Unique identifier for this image.

        Returns:
            True if the image is a duplicate (hamming distance < threshold).
        """
        new_hash = self.compute_phash(image_path)
        if new_hash is None:
            return False  # Can't compute hash, allow it through

        new_hash_obj = imagehash.hex_to_hash(new_hash)

        # Check against all stored hashes
        cursor = self.conn.execute("SELECT image_id, phash FROM seen_hashes")
        for existing_id, existing_hash_str in cursor:
            existing_hash = imagehash.hex_to_hash(existing_hash_str)
            distance = new_hash_obj - existing_hash
            if distance < self.threshold:
                logger.debug(
                    f"Duplicate detected: {image_id} matches {existing_id} "
                    f"(hamming distance={distance})"
                )
                return True

        return False

    def register(self, image_id: str, image_path: str, source: str):
        """Register an image's hash in the database."""
        phash = self.compute_phash(image_path)
        if phash is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO seen_hashes (image_id, phash, source, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (image_id, phash, source, time.time())
            )
            self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()


class BaseCollector(ABC):
    """
    Abstract base class for image collectors.

    Provides built-in retry logic with exponential backoff, configurable
    rate limiting, and perceptual hash deduplication.
    """

    def __init__(
        self,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        dedup_detector: Optional[DuplicateDetector] = None
    ):
        """
        Initialize the base collector.

        Args:
            requests_per_second: Rate limit for API requests.
            max_retries: Maximum number of retry attempts on failure.
            dedup_detector: Shared DuplicateDetector instance for cross-source dedup.
        """
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.dedup_detector = dedup_detector
        self._last_request_time = 0.0
        self.source_name = self.__class__.__name__.replace("Collector", "").lower()

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        if self.requests_per_second <= 0:
            return
        min_interval = 1.0 / self.requests_per_second
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with retry logic and exponential backoff.

        Retries: 3 attempts with delays of 1s, 2s, 4s.
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed for "
                        f"{self.source_name}: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed for "
                        f"{self.source_name}: {e}"
                    )
                    raise

    def _generate_image_id(self, url: str) -> str:
        """Generate a deterministic image ID from URL."""
        return f"{self.source_name}_{hashlib.md5(url.encode()).hexdigest()[:12]}"

    @abstractmethod
    def fetch_batch(self, n: int) -> List[ImageRecord]:
        """
        Fetch a batch of images from this source.

        Args:
            n: Number of images to fetch in this batch.

        Returns:
            List of ImageRecord objects for successfully fetched images.
        """
        pass

    def collect(self, target_n: int, output_dir: str = "data/raw/images",
                batch_size: int = 50) -> List[ImageRecord]:
        """
        Collect images up to target count, handling dedup and saving.

        Args:
            target_n: Target number of images to collect from this source.
            output_dir: Base directory for saving images.
            batch_size: Number of images to fetch per batch.

        Returns:
            List of all successfully collected ImageRecord objects.
        """
        collected = []
        source_dir = os.path.join(output_dir, self.source_name)
        os.makedirs(source_dir, exist_ok=True)

        while len(collected) < target_n:
            remaining = target_n - len(collected)
            batch_n = min(batch_size, remaining)

            try:
                batch = self.fetch_batch(batch_n)
            except Exception as e:
                logger.error(f"Batch fetch failed for {self.source_name}: {e}")
                break

            if not batch:
                logger.warning(f"No more images available from {self.source_name}")
                break

            for record in batch:
                # Check deduplication
                if self.dedup_detector and os.path.exists(record.local_path):
                    if self.dedup_detector.is_duplicate(record.local_path, record.image_id):
                        logger.debug(f"Skipping duplicate: {record.image_id}")
                        continue
                    self.dedup_detector.register(
                        record.image_id, record.local_path, self.source_name
                    )

                collected.append(record)
                if len(collected) >= target_n:
                    break

        logger.info(f"{self.source_name}: Collected {len(collected)} images")
        return collected

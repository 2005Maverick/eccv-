"""
Base processor with SQLite cache and error isolation.
All annotation processors inherit from ProcessorBase.
"""

import os
import json
import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProcessorBase(ABC):
    """
    Abstract base class for image annotation processors.

    Provides:
    - SQLite-based result caching (keyed by image_id + processor_name)
    - Error isolation: never crash the pipeline on a single bad image
    """

    def __init__(self, cache_db_path: str = "data/processed/annotation_cache.db"):
        """
        Initialize the processor with cache.

        Args:
            cache_db_path: Path to SQLite cache database.
        """
        self.processor_name = self.__class__.__name__
        self.cache_db_path = cache_db_path
        os.makedirs(os.path.dirname(cache_db_path), exist_ok=True)
        self.conn = sqlite3.connect(cache_db_path)
        self._init_cache()

    def _init_cache(self):
        """Create cache table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS annotation_cache (
                image_id TEXT,
                processor_name TEXT,
                result_json TEXT,
                PRIMARY KEY (image_id, processor_name)
            )
        """)
        self.conn.commit()

    def get_cached(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for an image.

        Args:
            image_id: Unique image identifier.

        Returns:
            Cached result dict, or None if not cached.
        """
        cursor = self.conn.execute(
            "SELECT result_json FROM annotation_cache WHERE image_id = ? AND processor_name = ?",
            (image_id, self.processor_name)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set_cached(self, image_id: str, result: Dict[str, Any]):
        """
        Store result in cache.

        Args:
            image_id: Unique image identifier.
            result: Result dict to cache.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO annotation_cache (image_id, processor_name, result_json) "
            "VALUES (?, ?, ?)",
            (image_id, self.processor_name, json.dumps(result))
        )
        self.conn.commit()

    @abstractmethod
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image and return annotation dict.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict of annotation results specific to this processor.
        """
        pass

    def safe_process(self, image_path: str, image_id: str) -> Dict[str, Any]:
        """
        Process with cache check and error isolation.

        Args:
            image_path: Path to the image file.
            image_id: Unique image identifier for caching.

        Returns:
            Result dict. On error, returns {"error": str, "skipped": True}.
        """
        # Check cache first
        cached = self.get_cached(image_id)
        if cached is not None:
            return cached

        # Process with error isolation
        try:
            result = self.process(image_path)
            self.set_cached(image_id, result)
            return result
        except Exception as e:
            error_result = {"error": str(e), "skipped": True}
            logger.warning(
                f"{self.processor_name} failed on {image_id}: {e}"
            )
            self.set_cached(image_id, error_result)
            return error_result

    def close(self):
        """Close the database connection."""
        self.conn.close()

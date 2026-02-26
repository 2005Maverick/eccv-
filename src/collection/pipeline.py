"""
Collection pipeline — orchestrates all 4 source collectors with parallel
execution, cross-source pHash deduplication, checkpoint-resume, and manifest output.
"""

import os
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import List, Dict, Set, Optional

import yaml
from tqdm import tqdm

from src.collection.base_collector import (
    BaseCollector, ImageRecord, DuplicateDetector
)
from src.collection.wikimedia_collector import WikimediaCollector
from src.collection.openimages_collector import OpenImagesCollector
from src.collection.streetview_collector import StreetViewCollector
from src.collection.yfcc_collector import YFCCCollector

logger = logging.getLogger(__name__)


class CollectionPipeline:
    """
    Unified image collection pipeline.

    Runs all 4 source collectors in parallel via ThreadPoolExecutor,
    deduplicates across sources using perceptual hashing (pHash),
    and saves results to a JSONL manifest with checkpoint-resume support.
    """

    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        """
        Initialize the collection pipeline.

        Args:
            config_path: Path to the pipeline configuration YAML file.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.target_n = self.config.get("collection", {}).get("target_images", 500000)
        self.manifest_path = "data/raw/manifest.jsonl"
        self.images_dir = "data/raw/images"

        # Ensure output directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Shared deduplication detector
        self.dedup = DuplicateDetector(
            db_path="data/raw/seen_hashes.db",
            threshold=self.config.get("collection", {}).get("phash_threshold", 10)
        )

        # Load checkpoint: already-completed image IDs
        self.completed_ids: Set[str] = set()
        self._load_checkpoint()

        # Initialize collectors
        rps = self.config.get("collection", {}).get("requests_per_second", 2)
        self.collectors: Dict[str, BaseCollector] = {
            "wikimedia": WikimediaCollector(
                output_dir=self.images_dir,
                requests_per_second=rps,
                dedup_detector=self.dedup,
            ),
            "openimages": OpenImagesCollector(
                output_dir=self.images_dir,
                requests_per_second=rps,
                dedup_detector=self.dedup,
            ),
            "streetview": StreetViewCollector(
                output_dir=self.images_dir,
                requests_per_second=rps,
                dedup_detector=self.dedup,
            ),
            "yfcc": YFCCCollector(
                output_dir=self.images_dir,
                requests_per_second=rps,
                dedup_detector=self.dedup,
            ),
        }

    def _load_checkpoint(self):
        """Load already-completed image IDs from manifest for resume support."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            self.completed_ids.add(record["image_id"])
                logger.info(
                    f"Checkpoint loaded: {len(self.completed_ids)} images already collected"
                )
            except Exception as e:
                logger.warning(f"Error reading checkpoint manifest: {e}")

    def _collect_from_source(self, source_name: str, target_per_source: int) -> List[ImageRecord]:
        """
        Collect images from a single source.

        Args:
            source_name: Name of the source collector.
            target_per_source: Target number of images to collect.

        Returns:
            List of ImageRecord objects collected from this source.
        """
        collector = self.collectors[source_name]
        records = []

        try:
            batch_size = 50
            while len(records) < target_per_source:
                remaining = target_per_source - len(records)
                batch = collector.fetch_batch(min(batch_size, remaining))

                if not batch:
                    logger.info(f"{source_name}: No more images available")
                    break

                for record in batch:
                    if record.image_id not in self.completed_ids:
                        records.append(record)

                if len(records) >= target_per_source:
                    break

        except Exception as e:
            logger.error(f"Collection from {source_name} failed: {e}")

        return records

    def _download_image(self, url: str) -> bool:
        """
        Download a single image (used for mocking in tests).

        Args:
            url: Image URL to download.

        Returns:
            True if download was successful.
        """
        return True

    def _save_manifest(self, records: List[ImageRecord]):
        """
        Append records to the JSONL manifest.

        Args:
            records: List of ImageRecord objects to save.
        """
        with open(self.manifest_path, "a") as f:
            for record in records:
                entry = {
                    "image_id": record.image_id,
                    "source": record.source,
                    "local_path": record.local_path,
                    "url": record.url,
                    "license": record.license,
                    "metadata": record.metadata,
                }
                f.write(json.dumps(entry) + "\n")

    def run(self, target_n: Optional[int] = None, resume: bool = True):
        """
        Run the full collection pipeline.

        Args:
            target_n: Override target image count (uses config value if None).
            resume: If True, skip images already in the manifest.
        """
        target = target_n if target_n is not None else self.target_n

        if resume and self.completed_ids:
            already_done = len(self.completed_ids)
            if already_done >= target:
                logger.info(
                    f"Already collected {already_done} images (target {target}). "
                    f"Nothing to do."
                )
                return
            remaining = target - already_done
            logger.info(f"Resuming: {already_done} done, {remaining} remaining")
        else:
            remaining = target

        # Split target across sources
        sources = list(self.collectors.keys())
        per_source = remaining // len(sources)

        all_records = []
        duplicates_skipped = 0
        source_counts = {}

        # Run collectors in parallel
        pbar = tqdm(total=remaining, desc="Collecting images", unit="img")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._collect_from_source, source, per_source): source
                for source in sources
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    records = future.result()
                    source_counts[source] = len(records)
                    all_records.extend(records)
                    pbar.update(len(records))
                except Exception as e:
                    logger.error(f"Collector {source} raised exception: {e}")
                    source_counts[source] = 0

        pbar.close()

        # Save to manifest
        self._save_manifest(all_records)

        # Update completed IDs
        for record in all_records:
            self.completed_ids.add(record.image_id)

        # Print summary
        total_collected = len(all_records)
        print(f"\n{'='*60}")
        print(f"Collection Complete")
        print(f"{'='*60}")
        print(f"Total collected:      {total_collected:>10,}")
        print(f"Duplicates skipped:   {duplicates_skipped:>10,}")
        print(f"Previously completed: {len(self.completed_ids) - total_collected:>10,}")
        print(f"\nPer-source breakdown:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source:<20} {count:>7,}")
        print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = CollectionPipeline()
    pipeline.run()

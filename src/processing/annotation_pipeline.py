"""
Annotation pipeline that runs all 7 processors on each image,
merges results, and outputs to annotations.jsonl.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from src.processing.base_processor import ProcessorBase
from src.processing.object_detector import ObjectDetector
from src.processing.depth_estimator import DepthEstimator
from src.processing.shadow_detector import ShadowDetector
from src.processing.ocr_extractor import OCRExtractor
from src.processing.texture_analyzer import TextureAnalyzer
from src.processing.temporal_extractor import TemporalExtractor

logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """
    Unified annotation pipeline.

    Runs all 7 processors on each image, merges results into a single record,
    and outputs to data/processed/annotations.jsonl.
    Supports caching to skip already-annotated images.
    """

    def __init__(self, cache_db_path: str = "data/processed/annotation_cache.db"):
        """
        Initialize all processors.

        Args:
            cache_db_path: Path to the SQLite cache database.
        """
        self.cache_db_path = cache_db_path
        os.makedirs("data/processed", exist_ok=True)

        # Initialize processors with shared cache
        self.object_detector = ObjectDetector(cache_db_path=cache_db_path)
        self.depth_estimator = DepthEstimator(cache_db_path=cache_db_path)
        self.shadow_detector = ShadowDetector(cache_db_path=cache_db_path)
        self.ocr_extractor = OCRExtractor(cache_db_path=cache_db_path)
        self.texture_analyzer = TextureAnalyzer(cache_db_path=cache_db_path)
        self.temporal_extractor = TemporalExtractor(cache_db_path=cache_db_path)

        self.output_path = "data/processed/annotations.jsonl"

    def _get_existing_annotations(self) -> set:
        """Get set of image_ids already in annotations.jsonl."""
        existing = set()
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            existing.add(record.get("image_id", ""))
                        except json.JSONDecodeError:
                            continue
        return existing

    def annotate_image(self, image_id: str, image_path: str,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run all processors on a single image and merge results.

        Args:
            image_id: Unique image identifier.
            image_path: Path to the image file.
            metadata: Optional metadata dict (may contain lat, lon, timestamp).

        Returns:
            Merged annotation dict with keys:
            image_id, detections, depth, shadow, ocr, texture, temporal.
        """
        metadata = metadata or {}

        # Object detection (needed by temporal extractor)
        detections = self.object_detector.safe_process(image_path, image_id)

        # Depth estimation
        depth = self.depth_estimator.safe_process(image_path, image_id)

        # Shadow detection (needs GPS + timestamp)
        lat = metadata.get("lat")
        lon = metadata.get("lon")
        timestamp = metadata.get("timestamp")

        try:
            shadow = self.shadow_detector.process(
                image_path, lat=lat, lon=lon, timestamp=timestamp
            )
        except Exception as e:
            shadow = {"error": str(e), "skipped": True}

        # OCR
        ocr = self.ocr_extractor.safe_process(image_path, image_id)

        # Texture
        texture = self.texture_analyzer.safe_process(image_path, image_id)

        # Temporal (uses YOLO detections for construction score)
        try:
            temporal = self.temporal_extractor.process(
                image_path,
                detections=detections.get("detections", []),
                lat=lat
            )
        except Exception as e:
            temporal = {"error": str(e), "skipped": True}

        return {
            "image_id": image_id,
            "detections": detections,
            "depth": depth,
            "shadow": shadow,
            "ocr": ocr,
            "texture": texture,
            "temporal": temporal,
        }

    def run(self, image_ids: Optional[List[str]] = None,
            images_dir: str = "data/raw/images",
            manifest_path: str = "data/raw/manifest.jsonl"):
        """
        Run annotation pipeline on all images.

        Args:
            image_ids: Optional list of specific image IDs to process.
                       If None, reads from manifest.
            images_dir: Base directory containing images.
            manifest_path: Path to the image manifest JSONL.
        """
        # Determine which images to process
        if image_ids is not None:
            # Use provided IDs — look for images in images_dir
            image_list = []
            for img_id in image_ids:
                # Try common extensions
                for ext in [".jpg", ".jpeg", ".png"]:
                    path = os.path.join(images_dir, f"{img_id}{ext}")
                    if os.path.exists(path):
                        image_list.append({"image_id": img_id, "local_path": path, "metadata": {}})
                        break
                else:
                    # Try subdirectories
                    for subdir in os.listdir(images_dir) if os.path.isdir(images_dir) else []:
                        subpath = os.path.join(images_dir, subdir)
                        for ext in [".jpg", ".jpeg", ".png"]:
                            path = os.path.join(subpath, f"{img_id}{ext}")
                            if os.path.exists(path):
                                image_list.append({"image_id": img_id, "local_path": path, "metadata": {}})
                                break
        else:
            # Read from manifest
            image_list = []
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            image_list.append(json.loads(line))

        # Skip already-annotated
        existing = self._get_existing_annotations()
        to_process = [img for img in image_list if img.get("image_id") not in existing]

        if not to_process:
            logger.info("All images already annotated (cache hit)")
            return

        # Process
        processed = 0
        skipped = len(existing)
        errored = 0

        with open(self.output_path, "a") as f:
            for img_info in tqdm(to_process, desc="Annotating images", unit="img"):
                image_id = img_info["image_id"]
                image_path = img_info.get("local_path", "")
                metadata = img_info.get("metadata", {})

                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    errored += 1
                    # Still write a record with all null fields
                    record = {
                        "image_id": image_id,
                        "detections": {"error": "Image not found", "skipped": True},
                        "depth": {"error": "Image not found", "skipped": True},
                        "shadow": {"error": "Image not found", "skipped": True},
                        "ocr": {"error": "Image not found", "skipped": True},
                        "texture": {"error": "Image not found", "skipped": True},
                        "temporal": {"error": "Image not found", "skipped": True},
                    }
                    f.write(json.dumps(record) + "\n")
                    continue

                try:
                    record = self.annotate_image(image_id, image_path, metadata)
                    f.write(json.dumps(record) + "\n")
                    processed += 1
                except Exception as e:
                    logger.error(f"Fatal error annotating {image_id}: {e}")
                    errored += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"Annotation Pipeline Complete")
        print(f"{'='*60}")
        print(f"Total processed: {processed:>10,}")
        print(f"Skipped (cached): {skipped:>9,}")
        print(f"Errored:          {errored:>9,}")
        print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = AnnotationPipeline()
    pipeline.run()

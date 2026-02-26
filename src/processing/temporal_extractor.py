"""
Temporal metadata extractor from EXIF data.
Used by: temporal_reasoning bias.
"""

import os
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)

# YOLO labels associated with construction activity
CONSTRUCTION_LABELS = {
    "truck", "crane", "bulldozer", "excavator", "scaffolding",
    "construction", "forklift", "tractor", "dump truck",
    "concrete mixer", "backhoe", "loader",
}


def _classify_time_of_day(hour: int) -> str:
    """Classify hour into time-of-day category."""
    if 5 <= hour < 7:
        return "dawn"
    elif 7 <= hour < 12:
        return "morning"
    elif 12 <= hour < 14:
        return "noon"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "dusk"
    else:
        return "night"


def _classify_season(month: int, lat: Optional[float] = None) -> str:
    """
    Classify month into season.
    Adjusts for southern hemisphere if latitude is negative.
    """
    # Standard northern hemisphere mapping
    if month in (3, 4, 5):
        season = "spring"
    elif month in (6, 7, 8):
        season = "summer"
    elif month in (9, 10, 11):
        season = "autumn"
    else:
        season = "winter"

    # Flip for southern hemisphere
    if lat is not None and lat < 0:
        flip = {"spring": "autumn", "summer": "winter",
                "autumn": "spring", "winter": "summer"}
        season = flip.get(season, season)

    return season


class TemporalExtractor(ProcessorBase):
    """
    Temporal metadata extractor.

    Parses EXIF DateTimeOriginal to extract:
    - timestamp (ISO format)
    - time_of_day (dawn/morning/noon/afternoon/dusk/night)
    - season (spring/summer/autumn/winter)
    - construction_score (fraction of YOLO detections that are
      machinery/scaffolding/construction labels)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _extract_exif_datetime(self, image_path: str) -> Optional[datetime]:
        """Extract DateTimeOriginal from EXIF data."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(image_path)
            exif_data = img._getexif()

            if exif_data is None:
                return None

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name == "DateTimeOriginal":
                    # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                    dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    return dt

            return None
        except Exception as e:
            logger.debug(f"EXIF extraction failed for {image_path}: {e}")
            return None

    def _compute_construction_score(self, detections: list) -> float:
        """
        Compute fraction of detections that are construction-related.

        Args:
            detections: List of detection dicts with 'label' key.

        Returns:
            Float between 0 and 1.
        """
        if not detections:
            return 0.0

        construction_count = sum(
            1 for det in detections
            if det.get("label", "").lower() in CONSTRUCTION_LABELS
        )
        return round(construction_count / len(detections), 4)

    def process(self, image_path: str, detections: Optional[list] = None,
                lat: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract temporal metadata from an image.

        Args:
            image_path: Path to the image file.
            detections: Optional YOLO detection results for construction score.
            lat: Optional latitude for hemisphere-aware season detection.

        Returns:
            Dict with: timestamp, time_of_day, season, construction_score.
        """
        dt = self._extract_exif_datetime(image_path)

        if dt is not None:
            timestamp = dt.isoformat()
            time_of_day = _classify_time_of_day(dt.hour)
            season = _classify_season(dt.month, lat)
        else:
            timestamp = None
            time_of_day = None
            season = None

        construction_score = self._compute_construction_score(detections or [])

        return {
            "timestamp": timestamp,
            "time_of_day": time_of_day,
            "season": season,
            "construction_score": construction_score,
        }

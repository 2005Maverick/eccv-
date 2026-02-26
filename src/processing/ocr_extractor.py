"""
OCR extractor using Tesseract via pytesseract.
Used by: text_in_image bias.
"""

import re
import logging
from typing import Dict, Any, List

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)

# Patterns indicating location-relevant text (streets, roads, places)
LOCATION_PATTERNS = [
    r'\b(?:st|street|rd|road|ave|avenue|blvd|boulevard|lane|ln|dr|drive|way|place|pl)\b',
    r'\b(?:north|south|east|west|n|s|e|w)\s*\d+',
    r'\b\d+\s+\w+\s+(?:st|street|rd|road|ave|avenue)\b',
    r'\b(?:exit|route|highway|hwy|interstate|i-)\s*\d+',
    r'\b\d{5}(?:-\d{4})?\b',  # ZIP codes
    r'\b(?:city|town|village|county|district)\b',
]


class OCRExtractor(ProcessorBase):
    """
    Tesseract-based OCR text extraction.

    Returns text blocks with bounding boxes, confidence scores,
    and location relevance detection.
    """

    def __init__(self, confidence_threshold: int = 60, min_text_length: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.min_text_length = min_text_length
        self._location_re = [re.compile(p, re.IGNORECASE) for p in LOCATION_PATTERNS]

    def _is_location_relevant(self, text: str) -> bool:
        """Check if text matches any location-related patterns."""
        for pattern in self._location_re:
            if pattern.search(text):
                return True
        return False

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with:
            - has_text: bool
            - text_blocks: List of {text, bbox, confidence}
            - text_is_location_relevant: bool
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"OCR failed on {image_path}: {e}")
            return {
                "has_text": False,
                "text_blocks": [],
                "text_is_location_relevant": False,
            }

        text_blocks: List[Dict[str, Any]] = []
        all_text_parts = []

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

            # Filter: confidence and text length
            if conf < self.confidence_threshold:
                continue
            if len(text) <= self.min_text_length:
                continue

            bbox = [
                data['left'][i],
                data['top'][i],
                data['left'][i] + data['width'][i],
                data['top'][i] + data['height'][i],
            ]

            text_blocks.append({
                "text": text,
                "bbox": bbox,
                "confidence": conf,
            })
            all_text_parts.append(text)

        # Check location relevance
        combined_text = " ".join(all_text_parts)
        text_is_location_relevant = self._is_location_relevant(combined_text)

        return {
            "has_text": len(text_blocks) > 0,
            "text_blocks": text_blocks,
            "text_is_location_relevant": text_is_location_relevant,
        }

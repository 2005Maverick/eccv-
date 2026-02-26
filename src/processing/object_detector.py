"""
Object detector using YOLOv8x via ultralytics.
Used by: counting, spurious_correlation, scale_invariance biases.
"""

import logging
from typing import Dict, Any, List

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)


class ObjectDetector(ProcessorBase):
    """
    YOLOv8x object detection processor.

    Returns detections with confidence > 0.7 including:
    label, bbox, confidence, and area_fraction.
    Supports batch inference (batch_size=32).
    """

    def __init__(self, confidence_threshold: float = 0.7, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy-load YOLOv8x model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8x.pt")
                logger.info("YOLOv8x model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8x: {e}")
                raise

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Run object detection on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with 'detections' key containing list of detection dicts,
            each with: label, bbox, confidence, area_fraction.
        """
        self._load_model()

        results = self._model(image_path, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            img_h, img_w = result.orig_shape
            img_area = img_w * img_h

            for box in result.boxes:
                conf = float(box.conf[0])

                # Filter: confidence > threshold
                if conf <= self.confidence_threshold:
                    continue

                # Extract bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_area = (x2 - x1) * (y2 - y1)
                area_fraction = bbox_area / img_area if img_area > 0 else 0.0

                # Get class label
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "label": label,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 4),
                    "area_fraction": round(area_fraction, 4),
                })

        return {"detections": detections}

    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run batch object detection.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of detection result dicts.
        """
        self._load_model()
        all_results = []

        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            results = self._model(batch, verbose=False)

            for result in results:
                detections = []
                img_h, img_w = result.orig_shape
                img_area = img_w * img_h

                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf <= self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_fraction = bbox_area / img_area if img_area > 0 else 0.0

                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, f"class_{cls_id}")

                    detections.append({
                        "label": label,
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(conf, 4),
                        "area_fraction": round(area_fraction, 4),
                    })

                all_results.append({"detections": detections})

        return all_results

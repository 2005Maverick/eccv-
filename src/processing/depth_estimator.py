"""
Depth estimator using MiDaS DPT-Large via transformers.
Used by: spatial_relations, scale_invariance biases.
"""

import logging
from typing import Dict, Any

import numpy as np

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)


class DepthEstimator(ProcessorBase):
    """
    MiDaS DPT-Large depth estimation processor.

    Returns average depth in each image quadrant (left, right, top, bottom),
    normalized to [0, 1].
    """

    def __init__(self, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load MiDaS DPT-Large model."""
        if self._model is None:
            try:
                from transformers import DPTForDepthEstimation, DPTImageProcessor
                import torch

                self._processor = DPTImageProcessor.from_pretrained(
                    "Intel/dpt-large"
                )
                self._model = DPTForDepthEstimation.from_pretrained(
                    "Intel/dpt-large"
                )
                self._model.eval()

                # Move to GPU if available
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)

                logger.info(f"MiDaS DPT-Large loaded on {self._device}")
            except Exception as e:
                logger.error(f"Failed to load MiDaS: {e}")
                raise

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Estimate depth map for an image and compute quadrant means.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with: left_depth_mean, right_depth_mean,
                       top_depth_mean, bottom_depth_mean
            All values normalized to [0, 1].
        """
        self._load_model()

        import torch
        from PIL import Image

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            prediction = outputs.predicted_depth

        # Interpolate to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy and normalize to [0, 1]
        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        # Compute quadrant means
        h, w = depth_map.shape
        mid_h, mid_w = h // 2, w // 2

        return {
            "left_depth_mean": round(float(depth_map[:, :mid_w].mean()), 4),
            "right_depth_mean": round(float(depth_map[:, mid_w:].mean()), 4),
            "top_depth_mean": round(float(depth_map[:mid_h, :].mean()), 4),
            "bottom_depth_mean": round(float(depth_map[mid_h:, :].mean()), 4),
        }

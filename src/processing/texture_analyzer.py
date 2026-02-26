"""
Texture analyzer using Canny edge detection and Gabor filters.
Used by: texture bias.
"""

import logging
from typing import Dict, Any

import numpy as np
import cv2

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)


class TextureAnalyzer(ProcessorBase):
    """
    Texture analysis processor.

    Returns:
    - edge_density: Canny edge pixel fraction
    - dominant_texture_freq: Peak Gabor filter response frequency
    - silhouette_extractable: True if edge density > 0.05 and largest contour > 10%
    """

    def __init__(self, edge_threshold: float = 0.05,
                 contour_threshold: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.edge_threshold = edge_threshold
        self.contour_threshold = contour_threshold

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Compute Canny edge pixel fraction."""
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = gray.shape[0] * gray.shape[1]
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0

    def _compute_gabor_freq(self, gray: np.ndarray) -> float:
        """
        Compute dominant texture frequency using Gabor filters.

        Tests multiple frequencies and returns the one with the highest response.
        """
        frequencies = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        max_response = 0.0
        dominant_freq = 0.1

        for freq in frequencies:
            # Create Gabor kernel
            kernel_size = 31
            sigma = 4.0
            theta = 0  # We'll average over orientations
            lambd = 1.0 / freq if freq > 0 else 10.0

            # Average over 4 orientations
            total_response = 0.0
            for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma, angle, lambd, 0.5, 0
                )
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                total_response += np.mean(np.abs(filtered))

            avg_response = total_response / 4.0
            if avg_response > max_response:
                max_response = avg_response
                dominant_freq = freq

        return dominant_freq

    def _check_silhouette(self, gray: np.ndarray, edge_density: float) -> bool:
        """
        Check if a clear silhouette can be extracted.

        True if edge_density > threshold AND largest contour > 10% of image.
        """
        if edge_density <= self.edge_threshold:
            return False

        # Find contours from edges
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        # Check largest contour area
        largest = max(contours, key=cv2.contourArea)
        img_area = gray.shape[0] * gray.shape[1]
        contour_fraction = cv2.contourArea(largest) / img_area if img_area > 0 else 0.0

        return contour_fraction > self.contour_threshold

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze texture properties of an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with: edge_density, dominant_texture_freq, silhouette_extractable.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                "edge_density": 0.0,
                "dominant_texture_freq": 0.0,
                "silhouette_extractable": False,
                "error": "Could not read image",
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edge_density = self._compute_edge_density(gray)
        dominant_freq = self._compute_gabor_freq(gray)
        silhouette = self._check_silhouette(gray, edge_density)

        return {
            "edge_density": round(edge_density, 4),
            "dominant_texture_freq": round(dominant_freq, 4),
            "silhouette_extractable": silhouette,
        }

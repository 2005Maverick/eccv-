"""
Shadow detector using OpenCV for shadow detection and Astropy for
solar geometry validation.
Used by: physical_plausibility bias.
"""

import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import cv2

from src.processing.base_processor import ProcessorBase

logger = logging.getLogger(__name__)


class ShadowDetector(ProcessorBase):
    """
    Shadow detection and physical plausibility verification.

    Step A: Detect shadows via OpenCV HSV thresholding + contour analysis
    Step B: Compute expected shadow angle via Astropy solar geometry
    Step C: Compare detected vs expected angles for plausibility
    """

    def __init__(self, angle_tolerance: float = 25.0, **kwargs):
        super().__init__(**kwargs)
        self.angle_tolerance = angle_tolerance

    def detect_shadow_angle(self, image_path: str) -> Optional[float]:
        """
        Detect shadow angle from an image using HSV thresholding.

        Step A: Shadow detection via OpenCV.
        - HSV thresholding: S < 50, V < 80
        - Contour analysis to find dominant shadow direction.

        Args:
            image_path: Path to the image file.

        Returns:
            Detected shadow angle in degrees from vertical, or None if no shadow found.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Shadow mask: low saturation, low value
        shadow_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 50, 80])
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Use the largest contour as the primary shadow
        largest_contour = max(contours, key=cv2.contourArea)

        # Minimum area check
        img_area = img.shape[0] * img.shape[1]
        if cv2.contourArea(largest_contour) < img_area * 0.01:
            return None

        # Fit an ellipse or line to determine shadow direction
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            angle = ellipse[2]  # Angle of the major axis
        else:
            # Fallback: fit a line
            [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = math.degrees(math.atan2(float(vy), float(vx)))

        return round(angle, 2)

    def compute_expected_shadow(
        self, lat: float, lon: float, timestamp: str
    ) -> Dict[str, Any]:
        """
        Compute expected shadow angle using Astropy solar geometry.

        Step B: Given GPS coordinates and timestamp, compute:
        - Sun azimuth and elevation
        - Expected shadow angle (sun_azimuth + 180°)

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            timestamp: ISO format timestamp string.

        Returns:
            Dict with shadow_angle_expected, sun_elevation, sun_azimuth.
        """
        try:
            from astropy.coordinates import EarthLocation, AltAz, get_sun
            from astropy.time import Time
            import astropy.units as u

            # Parse timestamp
            obs_time = Time(timestamp, format="isot", scale="utc")

            # Create location
            location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)

            # Compute solar position
            altaz_frame = AltAz(obstime=obs_time, location=location)
            sun = get_sun(obs_time).transform_to(altaz_frame)

            sun_azimuth = float(sun.az.deg)
            sun_elevation = float(sun.alt.deg)

            # Shadow angle = sun azimuth + 180° (shadow opposite sun)
            shadow_angle_expected = (sun_azimuth + 180.0) % 360.0

            return {
                "shadow_angle_expected": round(shadow_angle_expected, 2),
                "sun_elevation": round(sun_elevation, 2),
                "sun_azimuth": round(sun_azimuth, 2),
            }
        except Exception as e:
            logger.warning(f"Astropy shadow computation failed: {e}")
            return {
                "shadow_angle_expected": None,
                "sun_elevation": None,
                "sun_azimuth": None,
            }

    def process(self, image_path: str, lat: Optional[float] = None,
                lon: Optional[float] = None,
                timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Full shadow analysis pipeline.

        Args:
            image_path: Path to the image.
            lat: Latitude (from EXIF or metadata). None if unavailable.
            lon: Longitude (from EXIF or metadata). None if unavailable.
            timestamp: Timestamp string. None if unavailable.

        Returns:
            Dict with shadow detection results and plausibility check.
        """
        # Step A: Detect shadow angle
        shadow_angle_detected = self.detect_shadow_angle(image_path)

        # Step B: Expected shadow angle (if GPS + timestamp available)
        if lat is not None and lon is not None and timestamp is not None:
            expected = self.compute_expected_shadow(lat, lon, timestamp)
        else:
            expected = {
                "shadow_angle_expected": None,
                "sun_elevation": None,
                "sun_azimuth": None,
            }

        # Step C: Plausibility check
        shadow_angle_expected = expected.get("shadow_angle_expected")
        if shadow_angle_detected is not None and shadow_angle_expected is not None:
            angle_delta = abs(shadow_angle_detected - shadow_angle_expected)
            # Handle wraparound (e.g., 350° vs 10°)
            if angle_delta > 180:
                angle_delta = 360 - angle_delta
            is_physically_plausible = angle_delta < self.angle_tolerance
        else:
            angle_delta = None
            is_physically_plausible = None

        result = {
            "shadow_angle_detected": shadow_angle_detected,
            "shadow_angle_expected": shadow_angle_expected,
            "sun_elevation": expected.get("sun_elevation"),
            "sun_azimuth": expected.get("sun_azimuth"),
            "angle_delta": round(angle_delta, 2) if angle_delta is not None else None,
            "is_physically_plausible": is_physically_plausible,
        }

        # Fallback flag when GPS/timestamp missing
        if lat is None or lon is None or timestamp is None:
            result["skippable"] = True

        return result

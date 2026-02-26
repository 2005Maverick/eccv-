"""
Google Street View Static API collector.
Captures street-level imagery across a global grid of 100 major cities.
Gracefully falls back if API key is missing.
"""

import os
import logging
import itertools
from typing import List, Optional

import requests
from dotenv import load_dotenv

from src.collection.base_collector import BaseCollector, ImageRecord

logger = logging.getLogger(__name__)

# 100 major cities with GPS coordinates for global coverage
CITY_GRID = [
    # North America
    ("New York", 40.7128, -74.0060), ("Los Angeles", 33.9425, -118.2551),
    ("Chicago", 41.8781, -87.6298), ("Houston", 29.7604, -95.3698),
    ("Toronto", 43.6532, -79.3832), ("Mexico City", 19.4326, -99.1332),
    ("San Francisco", 37.7749, -122.4194), ("Miami", 25.7617, -80.1918),
    ("Vancouver", 49.2827, -123.1207), ("Montreal", 45.5017, -73.5673),
    # Europe
    ("London", 51.5074, -0.1278), ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050), ("Madrid", 40.4168, -3.7038),
    ("Rome", 41.9028, 12.4964), ("Amsterdam", 52.3676, 4.9041),
    ("Vienna", 48.2082, 16.3738), ("Prague", 50.0755, 14.4378),
    ("Stockholm", 59.3293, 18.0686), ("Lisbon", 38.7223, -9.1393),
    # Asia
    ("Tokyo", 35.6762, 139.6503), ("Seoul", 37.5665, 126.9780),
    ("Singapore", 1.3521, 103.8198), ("Hong Kong", 22.3193, 114.1694),
    ("Bangkok", 13.7563, 100.5018), ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.7041, 77.1025), ("Shanghai", 31.2304, 121.4737),
    ("Beijing", 39.9042, 116.4074), ("Taipei", 25.0330, 121.5654),
    ("Osaka", 34.6937, 135.5023), ("Jakarta", 6.2088, 106.8456),
    ("Kuala Lumpur", 3.1390, 101.6869), ("Manila", 14.5995, 120.9842),
    ("Hanoi", 21.0278, 105.8342),
    # Middle East & Africa
    ("Dubai", 25.2048, 55.2708), ("Istanbul", 41.0082, 28.9784),
    ("Cairo", 30.0444, 31.2357), ("Tel Aviv", 32.0853, 34.7818),
    ("Cape Town", -33.9249, 18.4241), ("Nairobi", -1.2921, 36.8219),
    ("Lagos", 6.5244, 3.3792), ("Casablanca", 33.5731, -7.5898),
    ("Riyadh", 24.7136, 46.6753), ("Doha", 25.2854, 51.5310),
    # South America
    ("São Paulo", -23.5505, -46.6333), ("Buenos Aires", -34.6037, -58.3816),
    ("Rio de Janeiro", -22.9068, -43.1729), ("Lima", -12.0464, -77.0428),
    ("Bogotá", 4.7110, -74.0721), ("Santiago", -33.4489, -70.6693),
    ("Medellín", 6.2476, -75.5658), ("Quito", -0.1807, -78.4678),
    # Oceania
    ("Sydney", -33.8688, 151.2093), ("Melbourne", -37.8136, 144.9631),
    ("Auckland", -36.8485, 174.7633), ("Perth", -31.9505, 115.8605),
    # More Europe
    ("Barcelona", 41.3851, 2.1734), ("Munich", 48.1351, 11.5820),
    ("Milan", 45.4642, 9.1900), ("Athens", 37.9838, 23.7275),
    ("Dublin", 53.3498, -6.2603), ("Brussels", 50.8503, 4.3517),
    ("Copenhagen", 55.6761, 12.5683), ("Oslo", 59.9139, 10.7522),
    ("Helsinki", 60.1699, 24.9384), ("Warsaw", 52.2297, 21.0122),
    ("Budapest", 47.4979, 19.0402), ("Bucharest", 44.4268, 26.1025),
    # More Asia
    ("Bangalore", 12.9716, 77.5946), ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639), ("Dhaka", 23.8103, 90.4125),
    ("Karachi", 24.8607, 67.0011), ("Lahore", 31.5204, 74.3587),
    ("Chengdu", 30.5728, 104.0668), ("Shenzhen", 22.5431, 114.0579),
    ("Guangzhou", 23.1291, 113.2644), ("Wuhan", 30.5928, 114.3055),
    # More Africa
    ("Accra", 5.6037, -0.1870), ("Addis Ababa", 9.0245, 38.7468),
    ("Dar es Salaam", -6.7924, 39.2083), ("Kinshasa", -4.4419, 15.2663),
    ("Lusaka", -15.3875, 28.3228),
    # More Americas
    ("Havana", 23.1136, -82.3666), ("Panama City", 8.9824, -79.5199),
    ("Montevideo", -34.9011, -56.1645), ("Caracas", 10.4806, -66.9036),
    ("Guatemala City", 14.6349, -90.5069), ("San José", 9.9281, -84.0907),
    ("Portland", 45.5152, -122.6784), ("Denver", 39.7392, -104.9903),
    ("Atlanta", 33.7490, -84.3880), ("Boston", 42.3601, -71.0589),
    ("Seattle", 47.6062, -122.3321), ("Washington DC", 38.9072, -77.0369),
    ("Philadelphia", 39.9526, -75.1652),
]

# 4 headings for spatial pair challenges
HEADINGS = [0, 90, 180, 270]

# Number of GPS jitter points per city
POINTS_PER_CITY = 50

# GPS jitter range (degrees) for sampling points within a city
GPS_JITTER = 0.02


class StreetViewCollector(BaseCollector):
    """
    Collects street-level images from Google Street View Static API.

    Generates a grid of GPS points across 100 major cities, capturing
    4 heading directions per point for spatial pair challenges.
    Gracefully skips if API key is not configured.
    """

    def __init__(self, output_dir: str = "data/raw/images", **kwargs):
        super().__init__(**kwargs)
        self.source_name = "streetview"
        self.output_dir = os.path.join(output_dir, self.source_name)
        os.makedirs(self.output_dir, exist_ok=True)

        load_dotenv()
        self.api_key = os.getenv("STREET_VIEW_API_KEY", "")
        self._points_iter = None
        self._initialized = False

        if not self.api_key or self.api_key == "your_street_view_api_key_here":
            logger.warning(
                "STREET_VIEW_API_KEY not set. Street View collector will be skipped."
            )
            self.api_key = ""

    def _generate_gps_points(self):
        """Generate GPS sampling points with heading combinations."""
        import random as rng
        rng.seed(42)  # Reproducible grid

        for city_name, lat, lon in CITY_GRID:
            for point_idx in range(POINTS_PER_CITY):
                jittered_lat = lat + rng.uniform(-GPS_JITTER, GPS_JITTER)
                jittered_lon = lon + rng.uniform(-GPS_JITTER, GPS_JITTER)

                for heading in HEADINGS:
                    yield {
                        "city": city_name,
                        "lat": jittered_lat,
                        "lon": jittered_lon,
                        "heading": heading,
                        "point_idx": point_idx,
                    }

    def _init_iterator(self):
        """Lazily initialize the GPS points iterator."""
        if not self._initialized:
            self._points_iter = self._generate_gps_points()
            self._initialized = True

    def _download_streetview(self, lat: float, lon: float, heading: int,
                              image_id: str) -> Optional[str]:
        """
        Download a Street View image for given coordinates and heading.

        Args:
            lat: Latitude.
            lon: Longitude.
            heading: Camera heading (0-360 degrees).
            image_id: Unique identifier for this image.

        Returns:
            Local file path if successful, None otherwise.
        """
        local_path = os.path.join(self.output_dir, f"{image_id}.jpg")

        if os.path.exists(local_path):
            return local_path

        url = (
            f"https://maps.googleapis.com/maps/api/streetview"
            f"?size=640x640&location={lat},{lon}&heading={heading}"
            f"&fov=90&pitch=0&key={self.api_key}"
        )

        try:
            response = self._retry_with_backoff(requests.get, url, timeout=30)
            response.raise_for_status()

            # Check if we got an actual image (not an error page)
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                logger.debug(f"No Street View image at ({lat}, {lon})")
                return None

            with open(local_path, "wb") as f:
                f.write(response.content)

            return local_path
        except Exception as e:
            logger.warning(f"Street View download failed for ({lat}, {lon}): {e}")
            return None

    def fetch_batch(self, n: int) -> List[ImageRecord]:
        """
        Fetch a batch of Street View images.

        Args:
            n: Number of images to fetch.

        Returns:
            List of ImageRecord objects. Empty if API key is missing.
        """
        if not self.api_key:
            return []

        self._init_iterator()
        records = []

        for point in self._points_iter:
            if len(records) >= n:
                break

            lat = point["lat"]
            lon = point["lon"]
            heading = point["heading"]
            city = point["city"]

            image_id = self._generate_image_id(
                f"sv_{lat:.4f}_{lon:.4f}_{heading}"
            )

            local_path = self._download_streetview(lat, lon, heading, image_id)
            if local_path is None:
                continue

            record = ImageRecord(
                image_id=image_id,
                url=f"streetview://{lat},{lon},{heading}",
                local_path=local_path,
                source="streetview",
                metadata={
                    "city": city,
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "point_idx": point["point_idx"],
                },
                license="Google Street View ToS",
            )
            records.append(record)

        return records

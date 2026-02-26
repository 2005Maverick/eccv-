"""
Wikimedia Commons image collector.
Uses the Wikimedia Commons API to fetch CC-licensed images.
No API key required.
"""

import os
import logging
from typing import List, Optional

import requests
from PIL import Image as PILImage
from io import BytesIO

from src.collection.base_collector import BaseCollector, ImageRecord

logger = logging.getLogger(__name__)

# Categories to search for diverse image content
CATEGORIES = [
    "outdoor scenes",
    "urban streets",
    "animals",
    "everyday objects",
]

API_URL = "https://commons.wikimedia.org/w/api.php"


class WikimediaCollector(BaseCollector):
    """
    Collects CC-licensed images from Wikimedia Commons.

    Filters for minimum resolution of 512×512 and only CC-licensed content.
    Searches across predefined categories for content diversity.
    """

    def __init__(self, output_dir: str = "data/raw/images", **kwargs):
        super().__init__(**kwargs)
        self.source_name = "wikimedia"
        self.output_dir = os.path.join(output_dir, self.source_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self._category_index = 0
        self._continue_tokens = {cat: None for cat in CATEGORIES}
        self._exhausted_categories = set()

    def _search_category(self, category: str, limit: int = 50) -> List[dict]:
        """
        Search Wikimedia Commons for images in a category.

        Args:
            category: Search category string.
            limit: Max results per query.

        Returns:
            List of image metadata dicts from the API.
        """
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,  # File namespace
            "gsrsearch": f"filetype:bitmap {category}",
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
        }

        # Add continue token if we have one
        cont_token = self._continue_tokens.get(category)
        if cont_token:
            params["gsroffset"] = cont_token

        try:
            response = self._retry_with_backoff(
                requests.get, API_URL, params=params, timeout=30
            )
            data = response.json()
        except Exception as e:
            logger.error(f"Wikimedia API error for '{category}': {e}")
            return []

        # Update continue token
        if "continue" in data:
            self._continue_tokens[category] = data["continue"].get("gsroffset")
        else:
            self._exhausted_categories.add(category)

        pages = data.get("query", {}).get("pages", {})
        results = []

        for page_id, page in pages.items():
            imageinfo = page.get("imageinfo", [{}])[0]

            # Filter: min resolution 512×512
            width = imageinfo.get("width", 0)
            height = imageinfo.get("height", 0)
            if width < 512 or height < 512:
                continue

            # Filter: must be JPEG or PNG
            mime = imageinfo.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue

            # Extract license info
            extmeta = imageinfo.get("extmetadata", {})
            license_short = extmeta.get("LicenseShortName", {}).get("value", "")

            # Filter: CC-licensed only
            if not any(cc in license_short.lower() for cc in ["cc", "public domain", "pd"]):
                continue

            url = imageinfo.get("url", "")
            if not url:
                continue

            results.append({
                "url": url,
                "width": width,
                "height": height,
                "license": license_short,
                "title": page.get("title", ""),
                "page_id": page_id,
                "category": category,
            })

        return results

    def _download_image(self, url: str, image_id: str) -> Optional[str]:
        """
        Download an image and save it locally.

        Args:
            url: Image URL to download.
            image_id: Unique image identifier for filename.

        Returns:
            Local file path if successful, None otherwise.
        """
        local_path = os.path.join(self.output_dir, f"{image_id}.jpg")

        if os.path.exists(local_path):
            return local_path

        try:
            response = self._retry_with_backoff(
                requests.get, url, timeout=60, stream=True
            )
            response.raise_for_status()

            # Verify it's a valid image
            img_data = response.content
            img = PILImage.open(BytesIO(img_data))

            # Convert to RGB JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(local_path, "JPEG", quality=95)
            return local_path

        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None

    def fetch_batch(self, n: int) -> List[ImageRecord]:
        """
        Fetch a batch of images from Wikimedia Commons.

        Args:
            n: Number of images to fetch.

        Returns:
            List of ImageRecord objects for successfully downloaded images.
        """
        records = []
        attempts = 0
        max_attempts = n * 3  # Allow some failures

        while len(records) < n and attempts < max_attempts:
            # Rotate through categories
            available_cats = [c for c in CATEGORIES if c not in self._exhausted_categories]
            if not available_cats:
                logger.warning("All Wikimedia categories exhausted")
                break

            category = available_cats[self._category_index % len(available_cats)]
            self._category_index += 1

            search_results = self._search_category(category, limit=min(n * 2, 50))

            for result in search_results:
                if len(records) >= n:
                    break

                attempts += 1
                image_id = self._generate_image_id(result["url"])
                local_path = self._download_image(result["url"], image_id)

                if local_path is None:
                    continue

                record = ImageRecord(
                    image_id=image_id,
                    url=result["url"],
                    local_path=local_path,
                    source="wikimedia",
                    metadata={
                        "title": result["title"],
                        "width": result["width"],
                        "height": result["height"],
                        "category": result["category"],
                        "page_id": result["page_id"],
                    },
                    license=result["license"],
                )
                records.append(record)

        return records

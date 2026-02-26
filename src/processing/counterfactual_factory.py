"""
Counterfactual Image Factory — ECCV v3.

Creates minimal-edit image variants where exactly one visual property changes.
Uses CV2/PIL only (no GPU/diffusion models needed).

Transforms:
1. background_swap   — foreground preserved, background replaced
2. color_shift       — object color changed via HSV manipulation
3. object_removal    — one YOLO-detected object inpainted out
4. texture_strip     — texture removed, shape preserved (edge-only)
5. mirror_edit       — horizontal flip + subtle detail change
"""

import os
import json
import uuid
import logging
import random
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)


class CounterfactualFactory:
    """
    Produces counterfactual image variants for causal-contrastive challenges.

    For each source image + annotation, generates one or more controlled
    variants where exactly one visual property changes. All transforms
    are deterministic given a seed, ensuring reproducibility.
    """

    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.manifest: List[Dict] = []

    def process_image(self, img_path: str, annotation: Dict,
                      transforms: List[str] = None) -> List[Dict]:
        """
        Generate counterfactual variants for a single image.

        Args:
            img_path: Path to the source image.
            annotation: Annotation dict with detections, depth, etc.
            transforms: List of transform types to apply. If None, applies all viable ones.

        Returns:
            List of counterfactual manifest entries.
        """
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return []

        image_id = annotation.get("image_id", os.path.splitext(os.path.basename(img_path))[0])
        det_info = annotation.get("detections", {})
        detections = det_info.get("detections", []) if isinstance(det_info, dict) else []

        all_transforms = transforms or ["background_swap", "color_shift",
                                         "object_removal", "texture_strip", "mirror_edit"]
        results = []

        for t in all_transforms:
            try:
                if t == "background_swap":
                    r = self._background_swap(img, image_id, detections)
                elif t == "color_shift":
                    r = self._color_shift(img, image_id, detections)
                elif t == "object_removal":
                    r = self._object_removal(img, image_id, detections)
                elif t == "texture_strip":
                    r = self._texture_strip(img, image_id)
                elif t == "mirror_edit":
                    r = self._mirror_edit(img, image_id)
                else:
                    continue

                if r:
                    results.append(r)
                    self.manifest.append(r)
            except Exception as e:
                logger.warning(f"Transform {t} failed for {image_id}: {e}")

        return results

    def _save_image(self, img: np.ndarray, cf_id: str) -> str:
        """Save a counterfactual image and return its path."""
        filename = f"{cf_id}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def _make_cf_id(self, image_id: str, transform: str) -> str:
        """Generate a unique counterfactual ID."""
        short_uuid = uuid.uuid4().hex[:6]
        return f"{image_id}_cf_{transform}_{short_uuid}"

    # =========================================================================
    # Transform 1: Background Swap
    # =========================================================================
    def _background_swap(self, img: np.ndarray, image_id: str,
                         detections: List[Dict]) -> Optional[Dict]:
        """
        Replace background while keeping foreground objects.
        Uses GrabCut segmentation when detections available,
        otherwise uses brightness-based separation.
        """
        h, w = img.shape[:2]

        # Create foreground mask using GrabCut
        mask = np.zeros((h, w), np.uint8)

        if detections:
            # Use YOLO bboxes as GrabCut initialization
            for det in detections[:5]:  # Limit to 5 objects
                bbox = det.get("bbox", [0, 0, w, h])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD  # Probable foreground

            # Set borders as definite background
            mask[:5, :] = cv2.GC_BGD
            mask[-5:, :] = cv2.GC_BGD
            mask[:, :5] = cv2.GC_BGD
            mask[:, -5:] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(img, mask, None, bgd_model, fgd_model, 3,
                           cv2.GC_INIT_WITH_MASK)
            except Exception:
                # Fallback: use bbox directly
                mask = np.zeros((h, w), np.uint8)
                for det in detections[:5]:
                    bbox = det.get("bbox", [0, 0, w, h])
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        else:
            # No detections: use center region as foreground
            margin_h, margin_w = h // 4, w // 4
            mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

        # Generate replacement backgrounds
        bg_type = self.rng.choice(["gradient", "blur", "solid", "noise"])

        if bg_type == "gradient":
            c1 = self.np_rng.randint(40, 180, 3).astype(np.uint8)
            c2 = self.np_rng.randint(40, 180, 3).astype(np.uint8)
            bg = np.zeros_like(img)
            for i in range(h):
                alpha = i / max(h - 1, 1)
                bg[i, :] = (c1 * (1 - alpha) + c2 * alpha).astype(np.uint8)
        elif bg_type == "blur":
            bg = cv2.GaussianBlur(img, (51, 51), 30)
        elif bg_type == "solid":
            color = self.np_rng.randint(30, 200, 3).astype(np.uint8)
            bg = np.full_like(img, color)
        else:  # noise
            bg = self.np_rng.randint(0, 255, img.shape).astype(np.uint8)

        # Composite: foreground from original, background from new
        fg_mask_3c = fg_mask[:, :, np.newaxis]
        result = img * fg_mask_3c + bg * (1 - fg_mask_3c)
        result = result.astype(np.uint8)

        cf_id = self._make_cf_id(image_id, "bgswap")
        self._save_image(result, cf_id)

        return {
            "original_id": image_id,
            "counterfactual_id": cf_id,
            "transform_type": "background_swap",
            "what_changed": f"Background replaced with {bg_type}",
            "what_preserved": "Foreground objects, their positions, and attributes",
            "bg_type": bg_type,
        }

    # =========================================================================
    # Transform 2: Color Shift
    # =========================================================================
    def _color_shift(self, img: np.ndarray, image_id: str,
                     detections: List[Dict]) -> Optional[Dict]:
        """
        Shift color of the dominant detected object via HSV manipulation.
        """
        if not detections:
            return None

        # Pick a detection with a reasonable bbox
        det = self.rng.choice(detections[:5])
        bbox = det.get("bbox", None)
        if not bbox:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        result = img.copy()
        roi = result[y1:y2, x1:x2]

        # Convert to HSV and shift hue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_shift = self.rng.randint(30, 150)  # Noticeable but not extreme
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv = hsv.astype(np.uint8)
        result[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cf_id = self._make_cf_id(image_id, "colorshift")
        self._save_image(result, cf_id)

        return {
            "original_id": image_id,
            "counterfactual_id": cf_id,
            "transform_type": "color_shift",
            "what_changed": f"Color of {det.get('label', 'object')} shifted by {hue_shift} degrees",
            "what_preserved": "Object identity, position, background, other objects",
            "target_object": det.get("label", "object"),
            "hue_shift": hue_shift,
        }

    # =========================================================================
    # Transform 3: Object Removal
    # =========================================================================
    def _object_removal(self, img: np.ndarray, image_id: str,
                        detections: List[Dict]) -> Optional[Dict]:
        """
        Remove one detected object using cv2.inpaint.
        """
        if not detections:
            return None

        det = self.rng.choice(detections[:5])
        bbox = det.get("bbox", None)
        if not bbox:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        # Create inpainting mask
        mask = np.zeros((h, w), np.uint8)
        # Slight padding around bbox for cleaner inpaint
        pad = 5
        mask[max(0, y1 - pad):min(h, y2 + pad),
             max(0, x1 - pad):min(w, x2 + pad)] = 255

        result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        cf_id = self._make_cf_id(image_id, "objremove")
        self._save_image(result, cf_id)

        return {
            "original_id": image_id,
            "counterfactual_id": cf_id,
            "transform_type": "object_removal",
            "what_changed": f"Removed {det.get('label', 'object')} at bbox [{x1},{y1},{x2},{y2}]",
            "what_preserved": "Background, other objects, scene layout",
            "removed_object": det.get("label", "object"),
            "removed_bbox": [x1, y1, x2, y2],
        }

    # =========================================================================
    # Transform 4: Texture Strip
    # =========================================================================
    def _texture_strip(self, img: np.ndarray, image_id: str) -> Optional[Dict]:
        """
        Remove texture, keep only edges (silhouette version).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Create a clean edge image: white edges on black
        result = np.zeros_like(img)
        result[edges > 0] = [255, 255, 255]

        # Optional: dilate edges slightly for visibility
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        result[edges_dilated > 0] = [255, 255, 255]

        cf_id = self._make_cf_id(image_id, "texstrip")
        self._save_image(result, cf_id)

        return {
            "original_id": image_id,
            "counterfactual_id": cf_id,
            "transform_type": "texture_strip",
            "what_changed": "All texture removed, only edge contours remain",
            "what_preserved": "Object shapes, spatial layout, relative positions",
        }

    # =========================================================================
    # Transform 5: Mirror + Subtle Edit
    # =========================================================================
    def _mirror_edit(self, img: np.ndarray, image_id: str) -> Optional[Dict]:
        """
        Horizontal flip + a subtle color patch change.
        Tests attention to fine spatial and color details.
        """
        # Horizontal flip
        result = cv2.flip(img, 1)

        # Add a subtle color patch in a random region
        h, w = result.shape[:2]
        patch_size = self.rng.randint(20, 50)
        px = self.rng.randint(0, w - patch_size)
        py = self.rng.randint(0, h - patch_size)

        # Shift the hue of the patch region
        patch = result[py:py + patch_size, px:px + patch_size]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv_patch[:, :, 0] = (hsv_patch[:, :, 0] + 90) % 180
        hsv_patch[:, :, 1] = np.clip(hsv_patch[:, :, 1] + 40, 0, 255)
        result[py:py + patch_size, px:px + patch_size] = cv2.cvtColor(
            hsv_patch.astype(np.uint8), cv2.COLOR_HSV2BGR
        )

        cf_id = self._make_cf_id(image_id, "mirror")
        self._save_image(result, cf_id)

        return {
            "original_id": image_id,
            "counterfactual_id": cf_id,
            "transform_type": "mirror_edit",
            "what_changed": "Image horizontally flipped + color patch at "
                           f"({px},{py}) size {patch_size}",
            "what_preserved": "Object types, scene content (mirrored)",
            "flip_type": "horizontal",
            "patch_location": [px, py, patch_size, patch_size],
        }

    # =========================================================================
    # Batch Processing
    # =========================================================================
    def process_all(self, image_dir: str, annotations: List[Dict],
                    transforms: List[str] = None) -> List[Dict]:
        """
        Process all annotated images and generate counterfactuals.

        Args:
            image_dir: Directory containing source images.
            annotations: List of annotation dicts.
            transforms: Optional subset of transforms to apply.

        Returns:
            List of all counterfactual manifest entries.
        """
        all_results = []

        for ann in annotations:
            img_id = ann.get("image_id", "")
            # Try common extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = os.path.join(image_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                logger.warning(f"Image not found for {img_id}")
                continue

            results = self.process_image(img_path, ann, transforms)
            all_results.extend(results)

        return all_results

    def save_manifest(self, path: str):
        """Save the counterfactual manifest to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.manifest:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Saved {len(self.manifest)} counterfactual entries to {path}")

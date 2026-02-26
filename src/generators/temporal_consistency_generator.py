"""
Temporal consistency bias generator — ECCV v3.

Tests multi-frame temporal reasoning: "What changed between frames?"
Creates 2-3 frame sequences from a single image via programmatic edits
and asks what changed over time.

ECCV-level: Diverse templates (30+), always 3 distractors.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)


# ============================================================
# Diverse Question Template Pools
# ============================================================

MOVED_TEMPLATES = [
    "Comparing Frame 1 (Image A) and Frame 2 (Image B), which direction did the {label} move? Answer Left, Right, or Did not move.",
    "Between Frame 1 and Frame 2, the {label} has shifted position. In which direction did it move — left, right, or not at all?",
    "Looking at both frames, has the {label} moved? If so, did it move left or right?",
    "Frame A and Frame B are taken moments apart. Which direction did the {label} travel?",
    "The {label} changed position from Frame 1 to Frame 2. Was the movement to the left, to the right, or is there no movement?",
    "Examine the two frames: did the {label} slide left, right, or stay in place?",
]

COUNT_CHANGED_TEMPLATES = [
    "Comparing Frame 1 (Image A) and Frame 2 (Image B), did the number of {label}s change? Answer Yes or No.",
    "Between Frame 1 and Frame 2, has the count of {label}s changed? Answer Yes or No.",
    "Looking at both frames, is there a different number of {label}s in Frame 2 compared to Frame 1?",
    "Did any {label} appear or disappear between these two frames? Answer Yes or No.",
    "Has the quantity of {label}s in the scene changed from Frame 1 to Frame 2?",
    "Are there more or fewer {label}s in Frame 2 than in Frame 1, or is the count the same?",
]

LIGHTING_TEMPLATES = [
    "Comparing Frame 1 (Image A) and Frame 2 (Image B), did the lighting change? Answer Yes or No.",
    "Between these two frames, has the scene's brightness or lighting shifted? Answer Yes or No.",
    "Does the illumination in Frame 2 differ from Frame 1? Answer Yes or No.",
    "Has the overall lighting intensity changed between the two frames? Answer Yes or No.",
    "Looking at both frames, did the lighting conditions stay the same or change?",
    "Is the scene lit differently in Frame 2 compared to Frame 1? Answer Yes or No.",
]

LIGHTING_DIR_TEMPLATES = [
    "Comparing the two frames, did the scene become brighter or darker? Answer Brighter, Darker, or No change.",
    "How did the lighting change from Frame 1 to Frame 2 — did it get brighter, darker, or stay the same?",
    "Between these frames, was there a brightness increase, decrease, or no change?",
    "Did the illumination shift toward brighter, darker, or remain unchanged between the two frames?",
]

NOTHING_TEMPLATES = [
    "Comparing Frame 1 (Image A) and Frame 2 (Image B), did anything change between the two frames? Answer Yes or No.",
    "Are there any differences between Frame 1 and Frame 2? Answer Yes or No.",
    "Looking at both frames carefully, has anything changed? Answer Yes or No.",
    "Do Frame 1 and Frame 2 show identical scenes, or has something been altered?",
    "Is Frame 2 an exact copy of Frame 1, or can you spot any differences?",
]


# ============================================================
# Image Manipulation Utilities
# ============================================================

def create_object_moved(img_path: str, bbox: List, shift_px: int,
                        output_path: str) -> Optional[Dict]:
    """Simulate an object moving by shifting it within the image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None

        obj_region = img[y1:y2, x1:x2].copy()
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

        nx1 = max(0, min(w - bw, x1 + shift_px))
        nx2 = nx1 + bw
        result[y1:y2, nx1:nx2] = obj_region

        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        direction = "right" if shift_px > 0 else "left"
        return {"shift_px": shift_px, "direction": direction,
                "old_bbox": [x1, y1, x2, y2],
                "new_bbox": [nx1, y1, nx2, y2]}
    except Exception as e:
        logger.warning(f"Object move failed: {e}")
        return None


def create_lighting_change(img_path: str, brightness_delta: int,
                           output_path: str) -> bool:
    """Simulate time passing via brightness/warmth change."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_delta, 0, 255)

        if brightness_delta < 0:
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.int16)
            bgr[:, :, 2] = np.clip(bgr[:, :, 2] + 15, 0, 255)
            bgr[:, :, 0] = np.clip(bgr[:, :, 0] - 10, 0, 255)
            result = bgr.astype(np.uint8)
        else:
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception as e:
        logger.warning(f"Lighting change failed: {e}")
        return False


class TemporalConsistencyGenerator(ChallengeGenerator):
    """
    Temporal consistency generator — ECCV v3.

    Sub-types:
    1. object_moved: Did the object move between frames? Which direction?
    2. count_changed: Did the number of objects change?
    3. lighting_changed: Did the lighting change? (Y/N)
    4. lighting_direction: Did it get brighter or darker?
    5. nothing_changed: Control — frames are identical

    30+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="temporal_consistency",
            ground_truth_method="programmatic_edit"
        )

    def _get_label(self, det):
        label = det.get("label", "object")
        return label if label != "object" else "item"

    def generate_challenge(self, annotations: List[Dict],
                           image_dir: str = None,
                           cf_dir: str = None) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        det_info = ann.get("detections", {})
        detections = det_info.get("detections", []) if isinstance(det_info, dict) else []
        image_id = ann.get("image_id", "unknown")

        sub = random.choice(["object_moved", "count_changed",
                             "lighting_changed", "lighting_direction",
                             "nothing_changed"])

        if sub == "object_moved" and detections:
            return self._gen_object_moved(ann, detections, image_id, image_dir, cf_dir)
        elif sub == "count_changed" and detections:
            return self._gen_count_changed(ann, detections, image_id, image_dir, cf_dir)
        elif sub == "lighting_changed":
            return self._gen_lighting_changed(ann, image_id, image_dir, cf_dir)
        elif sub == "lighting_direction":
            return self._gen_lighting_direction(ann, image_id, image_dir, cf_dir)
        else:
            return self._gen_nothing_changed(ann, image_id)

    def _find_image(self, image_dir, image_id):
        if not image_dir:
            return None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = os.path.join(image_dir, image_id + ext)
            if os.path.exists(p):
                return p
        return None

    def _gen_object_moved(self, ann, detections, image_id, image_dir, cf_dir):
        det = random.choice(detections[:5])
        bbox = det.get("bbox", None)
        if not bbox:
            return None

        label = self._get_label(det)
        shift = random.choice([-80, -60, -40, 40, 60, 80])
        direction = "right" if shift > 0 else "left"

        cf_id = None
        if image_dir and cf_dir:
            src = self._find_image(image_dir, image_id)
            if src:
                cf_id = f"{image_id}_moved_{random.randint(1000, 9999)}"
                dst = os.path.join(cf_dir, cf_id + ".jpg")
                result = create_object_moved(src, bbox, shift, dst)
                if not result:
                    cf_id = None

        if not cf_id:
            return None

        template = random.choice(MOVED_TEMPLATES).format(label=label)

        return self._create_challenge(
            annotations=[ann], # Approximating image_a/b but keeping logic for now
            difficulty="medium" if abs(shift) >= 60 else "hard",
            sub_type="object_moved",
            question_template=template,
            correct_answer=direction.capitalize(),
            distractors=[("Left" if direction == "right" else "Right"),
                                "Did not move",
                                "The object disappeared entirely"],
            metadata={
                "object": det.get("label", "unknown"),
                "shift_px": shift,
                "direction": direction,
            },
        )

    def _gen_count_changed(self, ann, detections, image_id, image_dir, cf_dir):
        if len(detections) < 2:
            return None

        det = random.choice(detections[:5])
        label = self._get_label(det)
        bbox = det.get("bbox", None)
        if not bbox:
            return None

        label_count = sum(1 for d in detections if self._get_label(d) == label)

        cf_id = None
        if image_dir and cf_dir:
            src = self._find_image(image_dir, image_id)
            if src:
                cf_id = f"{image_id}_removed_{random.randint(1000, 9999)}"
                dst = os.path.join(cf_dir, cf_id + ".jpg")
                img = cv2.imread(src)
                if img is not None:
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    mask = np.zeros((h, w), np.uint8)
                    mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
                    result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
                    cv2.imwrite(dst, result, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if not cf_id:
            return None

        template = random.choice(COUNT_CHANGED_TEMPLATES).format(label=label)

        return self._create_challenge(
            annotations=[ann],
            difficulty="medium",
            sub_type="count_changed",
            question_template=template,
            correct_answer="Yes",
            distractors=["No",
                                "The count appears the same at first glance",
                                "Cannot determine from these frames"],
            metadata={
                "object": det.get("label", "unknown"),
                "original_count": label_count,
                "new_count": label_count - 1,
                "removed_bbox": [int(v) for v in bbox],
            },
        )

    def _gen_lighting_changed(self, ann, image_id, image_dir, cf_dir):
        delta = random.choice([-50, -30, 30, 50])

        cf_id = None
        if image_dir and cf_dir:
            src = self._find_image(image_dir, image_id)
            if src:
                cf_id = f"{image_id}_light_{random.randint(1000, 9999)}"
                dst = os.path.join(cf_dir, cf_id + ".jpg")
                if not create_lighting_change(src, delta, dst):
                    cf_id = None

        if not cf_id:
            return None

        template = random.choice(LIGHTING_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy" if abs(delta) >= 40 else "medium",
            sub_type="lighting_changed",
            question_template=template,
            correct_answer="Yes",
            distractors=["No",
                                "The lighting is the same in both frames",
                                "Cannot determine the difference clearly"],
            metadata={
                "brightness_delta": delta,
                "became": "darker" if delta < 0 else "brighter",
            },
        )

    def _gen_lighting_direction(self, ann, image_id, image_dir, cf_dir):
        """Did the scene get brighter or darker?"""
        delta = random.choice([-50, -30, 30, 50])
        correct = "Brighter" if delta > 0 else "Darker"

        cf_id = None
        if image_dir and cf_dir:
            src = self._find_image(image_dir, image_id)
            if src:
                cf_id = f"{image_id}_lightdir_{random.randint(1000, 9999)}"
                dst = os.path.join(cf_dir, cf_id + ".jpg")
                if not create_lighting_change(src, delta, dst):
                    cf_id = None

        if not cf_id:
            return None

        template = random.choice(LIGHTING_DIR_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy" if abs(delta) >= 40 else "medium",
            sub_type="lighting_direction",
            question_template=template,
            correct_answer=correct,
            distractors=["Brighter" if correct == "Darker" else "Darker",
                                "No change",
                                "The color temperature changed but not brightness"],
            metadata={
                "brightness_delta": delta,
                "direction": correct.lower(),
            },
        )

    def _gen_nothing_changed(self, ann, image_id):
        template = random.choice(NOTHING_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy",
            sub_type="nothing_changed",
            question_template=template,
            correct_answer="No",
            distractors=["Yes",
                                "There are subtle differences",
                                "The frames show different scenes"],
            metadata={
                "is_control": True,
            },
        )

"""
Occlusion gradient bias generator — ECCV v3.

Systematically hides increasing percentages (10%-90%) of detected objects
and asks whether they can still be identified. Produces a continuum of
difficulty rather than binary pass/fail.

ECCV-level: Diverse templates (25+), always 3 distractors.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

OCCLUSION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
OCCLUSION_NAMES = {
    0.1: "10%", 0.3: "30%", 0.5: "50%", 0.7: "70%", 0.9: "90%"
}

# ============================================================
# Diverse Question Template Pools
# ============================================================

IDENTIFY_TEMPLATES = [
    "An object in this image is {pct} occluded (partially hidden). What is the object?",
    "{pct} of an object has been covered. Can you identify what the partially hidden object is?",
    "Part of an object ({pct}) is blocked. What type of object is being occluded?",
    "With {pct} of it concealed, what is the partially visible object in this image?",
    "An object is {pct} hidden behind an obstruction. Identify the object.",
    "Approximately {pct} of an object is not visible. What is the occluded object?",
    "Despite being {pct} covered, what object can you still identify?",
    "A significant portion ({pct}) of an object is blocked. What is it?",
]

THRESHOLD_TEMPLATES = [
    "Part of the {label} in this image is hidden ({pct} occluded). Is the {label} still clearly recognizable? Answer Yes or No.",
    "With {pct} of the {label} blocked, can you still confidently identify it? Answer Yes or No.",
    "Is the {label} recognizable when {pct} of it is occluded? Answer Yes or No.",
    "Even with {pct} occlusion, can the {label} be identified? Answer Yes or No.",
    "At {pct} occlusion, does the {label} remain identifiable? Answer Yes or No.",
    "Can you still recognize this as a {label} when {pct} of it is hidden? Answer Yes or No.",
]

COMPARISON_TEMPLATES = [
    "Both images show the same {label} with different amounts hidden. Which image shows more of the {label}? Answer A or B.",
    "Two versions of the same {label} have different levels of occlusion. In which image is more of the {label} visible? Answer A or B.",
    "Comparing these two images, which one reveals more of the {label}? Answer A or B.",
    "The {label} appears in both images but with different coverage. Where is it more visible — A or B?",
    "Which image contains a less occluded view of the {label}? Answer A or B.",
]

STYLE_IDENTIFY_TEMPLATES = [
    "An object is partially hidden by a {style} overlay. What type of object is beneath the occlusion?",
    "A {style} mask covers part of this image. What is the partially concealed object?",
    "Part of the image has been obscured with a {style} pattern. Identify the hidden object.",
    "A {style} occlusion covers {pct} of an object. What is the object?",
]

STYLE_NAMES = {"gray": "gray box", "noise": "random noise", "blur": "blurred region"}


def create_occluded_image(img_path: str, bbox: List, occlusion_pct: float,
                          output_path: str,
                          style: str = "gray") -> bool:
    """Create occlusion version of an image around a specific bbox."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 10 or bh < 10:
            return False

        result = img.copy()
        occlude_w = int(bw * occlusion_pct)
        ox1, oy1, ox2, oy2 = x2 - occlude_w, y1, x2, y2

        if style == "gray":
            result[oy1:oy2, ox1:ox2] = 128
        elif style == "noise":
            noise = np.random.randint(0, 255, (oy2 - oy1, ox2 - ox1, 3), dtype=np.uint8)
            result[oy1:oy2, ox1:ox2] = noise
        elif style == "blur":
            region = result[oy1:oy2, ox1:ox2]
            blurred = cv2.GaussianBlur(region, (51, 51), 25)
            result[oy1:oy2, ox1:ox2] = blurred

        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception as e:
        logger.warning(f"Occlusion creation failed: {e}")
        return False


class OcclusionGradientGenerator(ChallengeGenerator):
    """
    Occlusion gradient bias generator — ECCV v3.

    Sub-types:
    1. identify_occluded: "What is the partially hidden object?"
    2. threshold_test: "Is the object still recognizable?"
    3. occlusion_comparison: "Which image shows more of the object?"
    4. style_identify: "What object is under the {style} overlay?"

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="occlusion_gradient",
            ground_truth_method="yolo_occlusion"
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

        if not detections:
            return None

        det = random.choice(detections[:5])
        true_label = self._get_label(det)
        bbox = det.get("bbox", None)
        if not bbox:
            return None

        sub = random.choice(["identify_occluded", "identify_occluded",
                             "threshold_test", "occlusion_comparison",
                             "style_identify"])

        if sub == "identify_occluded":
            return self._gen_identify(ann, det, true_label, bbox, image_dir, cf_dir)
        elif sub == "threshold_test":
            return self._gen_threshold(ann, det, true_label, bbox, image_dir, cf_dir)
        elif sub == "style_identify":
            return self._gen_style_identify(ann, det, true_label, bbox, image_dir, cf_dir)
        else:
            return self._gen_comparison(ann, det, true_label, bbox, image_dir, cf_dir)

    def _get_wrong_labels(self, true_label):
        from src.generators.typography_conflict_generator import LABEL_CONFLICTS, DEFAULT_CONFLICTS
        return LABEL_CONFLICTS.get(true_label, DEFAULT_CONFLICTS)[:3]

    def _gen_identify(self, ann, det, true_label, bbox, image_dir, cf_dir):
        level = random.choice(OCCLUSION_LEVELS)
        pct_name = OCCLUSION_NAMES[level]

        cf_id = None
        if image_dir and cf_dir:
            img_id = ann.get("image_id", "unknown")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                src = os.path.join(image_dir, img_id + ext)
                if os.path.exists(src):
                    cf_id = f"{img_id}_occ{int(level*100)}_{random.randint(1000,9999)}"
                    dst = os.path.join(cf_dir, cf_id + ".jpg")
                    style = random.choice(["gray", "noise", "blur"])
                    create_occluded_image(src, bbox, level, dst, style)
                    break

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        difficulty = "easy" if level <= 0.3 else ("medium" if level <= 0.5 else "hard")
        wrong_labels = self._get_wrong_labels(true_label)
        template = random.choice(IDENTIFY_TEMPLATES).format(pct=pct_name)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="identify_occluded",
            question_template=template,
            correct_answer=true_label,
            distractors=wrong_labels[:3],
            metadata={
                "occlusion_level": level,
                "occlusion_pct": pct_name,
                "object": det.get("label", "unknown"),
                "has_overlay": cf_id is not None,
            },
        )

    def _gen_threshold(self, ann, det, true_label, bbox, image_dir, cf_dir):
        level = random.choice(OCCLUSION_LEVELS)
        pct_name = OCCLUSION_NAMES[level]
        answer = "Yes" if level <= 0.5 else "No"

        cf_id = None
        if image_dir and cf_dir:
            img_id = ann.get("image_id", "unknown")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                src = os.path.join(image_dir, img_id + ext)
                if os.path.exists(src):
                    cf_id = f"{img_id}_occthresh{int(level*100)}_{random.randint(1000,9999)}"
                    dst = os.path.join(cf_dir, cf_id + ".jpg")
                    create_occluded_image(src, bbox, level, dst, "gray")
                    break

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        difficulty = "easy" if level in [0.1, 0.9] else ("medium" if level in [0.3, 0.7] else "hard")

        template = random.choice(THRESHOLD_TEMPLATES).format(
            label=true_label, pct=pct_name)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="threshold_test",
            question_template=template,
            correct_answer=answer,
            distractors=["No" if answer == "Yes" else "Yes",
                                "Partially recognizable but not confidently",
                                "The occlusion makes identification impossible"],
            metadata={
                "occlusion_level": level,
                "object": det.get("label", "unknown"),
            },
        )

    def _gen_comparison(self, ann, det, true_label, bbox, image_dir, cf_dir):
        levels = sorted(random.sample(OCCLUSION_LEVELS, 2))
        less_occ, more_occ = levels

        cf_id_a, cf_id_b = None, None
        if image_dir and cf_dir:
            img_id = ann.get("image_id", "unknown")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                src = os.path.join(image_dir, img_id + ext)
                if os.path.exists(src):
                    cf_id_a = f"{img_id}_occ{int(less_occ*100)}_cmp_{random.randint(1000,9999)}"
                    cf_id_b = f"{img_id}_occ{int(more_occ*100)}_cmp_{random.randint(1000,9999)}"
                    create_occluded_image(src, bbox, less_occ,
                                         os.path.join(cf_dir, cf_id_a + ".jpg"), "gray")
                    create_occluded_image(src, bbox, more_occ,
                                         os.path.join(cf_dir, cf_id_b + ".jpg"), "gray")
                    break

        image_id = ann.get("image_id", "unknown")
        difficulty = "easy" if abs(more_occ - less_occ) >= 0.4 else ("medium" if abs(more_occ - less_occ) >= 0.2 else "hard")

        template = random.choice(COMPARISON_TEMPLATES).format(label=true_label)

        return self._create_challenge(
            annotations=[ann], # Approximating since we only have one original annotation
            difficulty=difficulty,
            sub_type="occlusion_comparison",
            question_template=template,
            correct_answer="A",
            distractors=["B",
                                "Both show the same amount",
                                "Neither image shows the object clearly"],
            metadata={
                "level_a": less_occ, "level_b": more_occ,
                "object": det.get("label", "unknown"),
            },
        )

    def _gen_style_identify(self, ann, det, true_label, bbox, image_dir, cf_dir):
        """Identify occluded object, with emphasis on occlusion style."""
        level = random.choice([0.3, 0.5, 0.7])
        pct_name = OCCLUSION_NAMES[level]
        style = random.choice(["gray", "noise", "blur"])
        style_name = STYLE_NAMES[style]

        cf_id = None
        if image_dir and cf_dir:
            img_id = ann.get("image_id", "unknown")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                src = os.path.join(image_dir, img_id + ext)
                if os.path.exists(src):
                    cf_id = f"{img_id}_occ_style_{random.randint(1000,9999)}"
                    dst = os.path.join(cf_dir, cf_id + ".jpg")
                    create_occluded_image(src, bbox, level, dst, style)
                    break

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id
        difficulty = "medium" if level <= 0.5 else "hard"

        wrong_labels = self._get_wrong_labels(true_label)
        template = random.choice(STYLE_IDENTIFY_TEMPLATES).format(
            style=style_name, pct=pct_name)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="style_identify",
            question_template=template,
            correct_answer=true_label,
            distractors=wrong_labels[:3],
            metadata={
                "occlusion_level": level,
                "occlusion_style": style,
                "object": det.get("label", "unknown"),
                "has_overlay": cf_id is not None,
            },
        )

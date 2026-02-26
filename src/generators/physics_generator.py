"""
Physical plausibility bias challenge generator.
Tests VLM understanding of lighting physics using brightness analysis.

ECCV-level: Diverse templates (25+), always 3 distractors,
uses brightness asymmetry and gradient analysis.
"""

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

LIGHT_DIRECTION_TEMPLATES = [
    "Which direction does the primary light source appear to come from? Answer left, right, top, or bottom.",
    "Based on the brightness gradient, from which side is the scene illuminated? Answer left, right, top, or bottom.",
    "Analyze the lighting in this image. Where is the main light source positioned? Answer left, right, top, or bottom.",
    "Looking at the brightness distribution, which direction does the dominant light originate from? Answer left, right, top, or bottom.",
    "From the illumination pattern, estimate the light source direction. Options: left, right, top, or bottom.",
    "Which side of the image receives the most direct light? Answer left, right, top, or bottom.",
    "Based on the highlights and shadows, where is the primary light source? Answer left, right, top, or bottom.",
    "The brightness gradient suggests the light comes from which direction? Answer left, right, top, or bottom.",
]

BRIGHTNESS_SIDE_TEMPLATES = [
    "Which side of this image is brighter, the left or the right? Answer Left or Right.",
    "Comparing the left half and right half, which has higher average brightness? Answer Left or Right.",
    "Looking at the horizontal brightness distribution, which side is more illuminated — Left or Right?",
    "Is the left or right portion of this image brighter overall? Answer Left or Right.",
    "Which half receives more light — the left side or the right side? Answer Left or Right.",
    "Examining the brightness asymmetry, which side appears lighter — Left or Right?",
]

CONSISTENCY_TEMPLATES = [
    "Does the lighting in this image appear physically consistent? Answer Yes or No.",
    "Are the lighting conditions across this image consistent with a single light source? Answer Yes or No.",
    "Does the illumination in this photograph appear natural and physically plausible? Answer Yes or No.",
    "Looking at the lighting patterns, does this image appear to have consistent, realistic illumination? Answer Yes or No.",
    "Is the brightness distribution in this image consistent with natural lighting? Answer Yes or No.",
    "Do the highlights and shadows in this scene suggest a physically coherent light setup? Answer Yes or No.",
]

TOP_BOTTOM_TEMPLATES = [
    "Which half of the image is brighter — the top or the bottom? Answer Top or Bottom.",
    "Comparing vertical brightness, is the upper or lower portion more illuminated? Answer Top or Bottom.",
    "Is the top half or bottom half of this image brighter overall? Answer Top or Bottom.",
    "Looking at the vertical light distribution, which half is brighter? Answer Top or Bottom.",
]

QUADRANT_BRIGHTNESS_TEMPLATES = [
    "Which quadrant of the image is the brightest? Answer top-left, top-right, bottom-left, or bottom-right.",
    "Identify the brightest region of this image from: top-left, top-right, bottom-left, or bottom-right.",
    "Looking at the four quadrants, which one receives the most light? Answer top-left, top-right, bottom-left, or bottom-right.",
    "Which corner region of this image has the highest brightness? Answer top-left, top-right, bottom-left, or bottom-right.",
]


def analyze_lighting(img_path: str) -> dict:
    """Analyze lighting direction and consistency from image."""
    img = cv2.imread(img_path)
    if img is None:
        return {"valid": False}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    h, w = gray.shape

    tl = float(np.mean(gray[:h//2, :w//2]))
    tr = float(np.mean(gray[:h//2, w//2:]))
    bl = float(np.mean(gray[h//2:, :w//2]))
    br = float(np.mean(gray[h//2:, w//2:]))

    left_mean = float(np.mean(gray[:, :w//2]))
    right_mean = float(np.mean(gray[:, w//2:]))
    top_mean = float(np.mean(gray[:h//2, :]))
    bottom_mean = float(np.mean(gray[h//2:, :]))

    horiz_diff = right_mean - left_mean
    vert_diff = top_mean - bottom_mean
    asymmetry = (abs(horiz_diff) + abs(vert_diff)) / 255.0

    if abs(horiz_diff) > abs(vert_diff):
        light_dir = "right" if horiz_diff > 0 else "left"
    else:
        light_dir = "top" if vert_diff > 0 else "bottom"

    return {
        "valid": True,
        "quadrants": [tl, tr, bl, br],
        "left_mean": left_mean / 255.0,
        "right_mean": right_mean / 255.0,
        "top_mean": top_mean / 255.0,
        "bottom_mean": bottom_mean / 255.0,
        "asymmetry": round(asymmetry, 4),
        "light_direction": light_dir,
        "horiz_diff": round(horiz_diff / 255.0, 4),
        "vert_diff": round(vert_diff / 255.0, 4),
    }


class PhysicsGenerator(ChallengeGenerator):
    """
    Physical plausibility bias generator — ECCV quality.

    Sub-types:
    1. Light direction: which side is the light coming from?
    2. Brightness side: left vs right brightness
    3. Consistency: is lighting physically consistent?
    4. Top/bottom brightness
    5. Quadrant brightness

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="physical_plausibility",
            ground_truth_method="brightness_gradient"
        )

    def generate_challenge(self, annotations: List[Dict],
                           image_dir: str = None) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        lighting = self._get_lighting(ann, image_dir)
        if not lighting.get("valid"):
            return None

        if lighting["asymmetry"] < 0.02:
            return None

        sub = random.choice(["light_direction", "brightness_side",
                              "consistency", "top_bottom", "quadrant_brightness"])

        if sub == "light_direction":
            return self._gen_light_direction(ann, lighting)
        elif sub == "brightness_side":
            return self._gen_brightness_side(ann, lighting)
        elif sub == "consistency":
            return self._gen_consistency(ann, lighting)
        elif sub == "top_bottom":
            return self._gen_top_bottom(ann, lighting)
        else:
            return self._gen_quadrant_brightness(ann, lighting)

    def _get_lighting(self, ann, image_dir):
        """Compute lighting from image or use depth proxy."""
        if image_dir:
            import os
            img_id = ann.get("image_id", "")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                path = os.path.join(image_dir, img_id + ext)
                if os.path.exists(path):
                    return analyze_lighting(path)

        depth = ann.get("depth", {})
        left = depth.get("left_depth_mean", 0.5)
        right = depth.get("right_depth_mean", 0.5)
        top = depth.get("top_depth_mean", 0.5)
        bottom = depth.get("bottom_depth_mean", 0.5)

        horiz_diff = right - left
        vert_diff = top - bottom
        asymmetry = abs(horiz_diff) + abs(vert_diff)

        if abs(horiz_diff) > abs(vert_diff):
            light_dir = "right" if horiz_diff > 0 else "left"
        else:
            light_dir = "top" if vert_diff > 0 else "bottom"

        # Compute quadrant values
        tl = (top + left) / 2
        tr = (top + right) / 2
        bl = (bottom + left) / 2
        br = (bottom + right) / 2

        return {
            "valid": True,
            "left_mean": left, "right_mean": right,
            "top_mean": top, "bottom_mean": bottom,
            "quadrants": [tl, tr, bl, br],
            "asymmetry": round(asymmetry, 4),
            "light_direction": light_dir,
            "horiz_diff": round(horiz_diff, 4),
            "vert_diff": round(vert_diff, 4),
        }

    def _gen_light_direction(self, ann, lighting):
        correct = lighting["light_direction"]
        all_dirs = ["left", "right", "top", "bottom"]
        distractors = [d for d in all_dirs if d != correct]

        asymmetry = lighting["asymmetry"]
        difficulty = "easy" if asymmetry > 0.15 else ("medium" if asymmetry > 0.06 else "hard")

        template = random.choice(LIGHT_DIRECTION_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="light_direction",
            question_template=template,
            correct_answer=correct,
            distractors=distractors[:3],
            metadata={
                "asymmetry": asymmetry,
            },
        )

    def _gen_brightness_side(self, ann, lighting):
        left_m = lighting["left_mean"]
        right_m = lighting["right_mean"]

        if left_m == right_m:
            return None

        correct = "Left" if left_m > right_m else "Right"
        diff = abs(left_m - right_m)
        difficulty = "easy" if diff > 0.12 else ("medium" if diff > 0.05 else "hard")

        template = random.choice(BRIGHTNESS_SIDE_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="brightness_side",
            question_template=template,
            correct_answer=correct,
            distractors=["Left" if correct == "Right" else "Right",
                          "They are both equally bright",
                          "Cannot determine due to shadows"],
            metadata={
                "asymmetry": round(diff, 4),
                "left_b": round(left_m, 4),
                "right_b": round(right_m, 4),
            },
        )

    def _gen_consistency(self, ann, lighting):
        asymmetry = lighting["asymmetry"]
        difficulty = "easy" if asymmetry > 0.15 else ("medium" if asymmetry > 0.06 else "hard")

        template = random.choice(CONSISTENCY_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="consistency",
            question_template=template,
            correct_answer="Yes",
            distractors=["No",
                                "The lighting appears artificial",
                                "Multiple conflicting light sources detected"],
            metadata={
                "light_direction": lighting["light_direction"],
                "asymmetry": asymmetry,
            },
        )

    def _gen_top_bottom(self, ann, lighting):
        top_m = lighting["top_mean"]
        bottom_m = lighting["bottom_mean"]

        if top_m == bottom_m:
            return None

        correct = "Top" if top_m > bottom_m else "Bottom"
        diff = abs(top_m - bottom_m)
        difficulty = "easy" if diff > 0.12 else ("medium" if diff > 0.05 else "hard")

        template = random.choice(TOP_BOTTOM_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="top_bottom_brightness",
            question_template=template,
            correct_answer=correct,
            distractors=["Top" if correct == "Bottom" else "Bottom",
                          "They are both equally bright",
                          "The brightness is concentrated in the center"],
            metadata={
                "top_b": round(top_m, 4),
                "bottom_b": round(bottom_m, 4),
            },
        )

    def _gen_quadrant_brightness(self, ann, lighting):
        quads = lighting.get("quadrants", [0.5, 0.5, 0.5, 0.5])
        names = ["top-left", "top-right", "bottom-left", "bottom-right"]

        brightest_idx = quads.index(max(quads))
        correct = names[brightest_idx]
        all_quads = ["top-left", "top-right", "bottom-left", "bottom-right"]

        spread = max(quads) - min(quads)
        difficulty = "easy" if spread > 30 else ("medium" if spread > 10 else "hard")

        template = random.choice(QUADRANT_BRIGHTNESS_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="quadrant_brightness",
            question_template=template,
            correct_answer=correct,
            distractors=[d for d in all_quads if d != correct][:3],
            metadata={
                "quadrants": {n: round(q, 4) for n, q in zip(names, quads)},
            },
        )

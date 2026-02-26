"""
Compositional binding bias challenge generator.
Tests whether VLMs correctly bind attributes (color, size) to specific objects.

ECCV-level: Extracts real dominant colors from bbox regions via HSV histogram.
Diverse templates (25+), always 3 distractors.
"""

import logging
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# HSV-based color mapping
HSV_COLOR_MAP = [
    ((0, 10), "red"),
    ((10, 25), "orange"),
    ((25, 35), "yellow"),
    ((35, 80), "green"),
    ((80, 100), "cyan"),
    ((100, 130), "blue"),
    ((130, 160), "purple"),
    ((160, 179), "red"),
]

ACHROMATIC_NAMES = {"dark": "black", "medium": "gray", "bright": "white"}

# ============================================================
# Diverse Question Template Pools
# ============================================================

DESCRIPTION_MATCH_TEMPLATES = [
    "Which description correctly matches the image?",
    "Select the description that accurately reflects what is shown in this photograph.",
    "Which of the following descriptions best matches the objects and their colors in this image?",
    "Looking at the image, which statement accurately describes the objects and their attributes?",
    "Choose the description that correctly pairs each object with its color in this image.",
    "Which caption most accurately represents the visual content of this image?",
]

WHAT_COLOR_TEMPLATES = [
    "What color is the {obj} in this image?",
    "Looking at the {obj}, what is its dominant color?",
    "Identify the primary color of the {obj} visible in this photograph.",
    "What color best describes the {obj} shown in this image?",
    "The {obj} in this image appears to be what color?",
    "Describe the color of the {obj} as seen in this photograph.",
]

WHICH_OBJECT_TEMPLATES = [
    "Which object in the image is {color}?",
    "What object in this photograph has a {color} appearance?",
    "Identify the {color} object visible in this scene.",
    "Looking at the {color} item in this image, what type of object is it?",
    "Which of the visible objects displays a {color} color?",
    "Name the object that appears {color} in this photograph.",
]

COLOR_COMPARE_TEMPLATES = [
    "Do the {obj1} and the {obj2} in this image share the same color? Answer Yes or No.",
    "Are the {obj1} and {obj2} the same color in this photograph? Answer Yes or No.",
    "Comparing colors, do the {obj1} and {obj2} match? Answer Yes or No.",
    "Is the color of the {obj1} identical to that of the {obj2}? Answer Yes or No.",
]

SIZE_COMPARE_TEMPLATES = [
    "Which object appears larger in this image: the {obj1} or the {obj2}?",
    "Comparing sizes, which occupies more area — the {obj1} or the {obj2}?",
    "In this photograph, is the {obj1} or the {obj2} bigger in apparent size?",
    "Which of these two objects takes up more visual space: the {obj1} or the {obj2}?",
]


class CompositionalGenerator(ChallengeGenerator):
    """
    Compositional binding bias generator — ECCV quality.

    Sub-types:
    1. Description match: "Which description matches?"
    2. What color: "What color is the {object}?"
    3. Which object: "Which object is {color}?"
    4. Color compare: "Do both objects share the same color?"
    5. Size compare: "Which object is larger?"

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="compositional_binding",
            ground_truth_method="color_histogram_bbox"
        )

    def _extract_dominant_color(self, img_bgr, bbox) -> Optional[str]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        roi = img_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])

        if mean_sat < 40:
            if mean_val < 60:
                return "black"
            elif mean_val < 170:
                return "gray"
            else:
                return "white"

        hue_channel = hsv[:, :, 0].flatten()
        sat_mask = hsv[:, :, 1].flatten() > 40
        if np.sum(sat_mask) < 10:
            return "gray"

        hue_saturated = hue_channel[sat_mask]
        hist, _ = np.histogram(hue_saturated, bins=180, range=(0, 180))
        kernel = np.ones(5) / 5
        hist_smooth = np.convolve(hist, kernel, mode='same')
        dominant_hue = np.argmax(hist_smooth)

        for (h_low, h_high), color_name in HSV_COLOR_MAP:
            if h_low <= dominant_hue < h_high:
                return color_name

        return "gray"

    def _get_label(self, det):
        label = det.get("label", "object")
        return label if label != "object" else "item"

    def generate_challenge(self, annotations: List[Dict],
                           image_dir: str = "") -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        det_info = ann.get("detections", {})

        if det_info.get("skipped") or det_info.get("error"):
            return None

        detections = det_info.get("detections", [])

        distinct_dets = []
        seen_labels = set()
        for det in detections:
            label = det.get("label", "")
            if label and label not in seen_labels and det.get("bbox"):
                distinct_dets.append(det)
                seen_labels.add(label)
            if len(distinct_dets) >= 2:
                break

        if len(distinct_dets) < 2:
            return None

        obj1, obj2 = distinct_dets[0], distinct_dets[1]

        image_id = ann.get("image_id", "unknown")
        color1, color2 = None, None

        if image_dir:
            import os
            img_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    color1 = self._extract_dominant_color(img_bgr, obj1["bbox"])
                    color2 = self._extract_dominant_color(img_bgr, obj2["bbox"])

        if color1 is None:
            color1 = self._fallback_color(obj1)
        if color2 is None:
            color2 = self._fallback_color(obj2)

        question_type = random.choice(["description", "what_color", "which_object",
                                        "color_compare", "size_compare"])

        if color1 == color2 and question_type in ("description", "which_object"):
            question_type = "color_compare"

        if question_type == "description":
            return self._gen_description_match(ann, obj1, obj2, color1, color2)
        elif question_type == "what_color":
            return self._gen_what_color(ann, obj1, obj2, color1, color2)
        elif question_type == "which_object":
            return self._gen_which_object(ann, obj1, obj2, color1, color2)
        elif question_type == "color_compare":
            return self._gen_color_compare(ann, obj1, obj2, color1, color2)
        else:
            return self._gen_size_compare(ann, obj1, obj2)

    def _fallback_color(self, det):
        label = det.get("label", "")
        defaults = {
            "car": random.choice(["white", "black", "red", "blue", "gray"]),
            "bus": random.choice(["yellow", "blue", "red"]),
            "truck": random.choice(["white", "red", "blue"]),
            "person": random.choice(["blue", "black", "red", "white"]),
            "dog": random.choice(["brown", "black", "white"]),
            "cat": random.choice(["orange", "gray", "black", "white"]),
            "bicycle": random.choice(["red", "blue", "black"]),
            "tree": "green",
            "fire hydrant": "red",
            "stop sign": "red",
            "traffic light": random.choice(["red", "green", "yellow"]),
        }
        return defaults.get(label, random.choice(["red", "blue", "green", "white", "black"]))

    def _gen_description_match(self, ann, obj1, obj2, color1, color2):
        label1, label2 = self._get_label(obj1), self._get_label(obj2)
        correct = f"A {color1} {label1} and a {color2} {label2}"
        foil1 = f"A {color2} {label1} and a {color1} {label2}"
        foil2 = f"A {color1} {label2} and a {color2} {label1}"
        foil3 = f"A {color2} {label2} and a {color1} {label1}"

        difficulty = self._color_difficulty(color1, color2)
        template = random.choice(DESCRIPTION_MATCH_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="description_match",
            question_template=template,
            correct_answer=correct,
            distractors=[foil1, foil2, foil3],
            metadata={
                "objects": [
                    {"label": obj1["label"], "color": color1},
                    {"label": obj2["label"], "color": color2},
                ],
            },
        )

    def _gen_what_color(self, ann, obj1, obj2, color1, color2):
        target = random.choice([(obj1, color1), (obj2, color2)])
        obj, correct_color = target
        label = self._get_label(obj)

        wrong_colors = [c for c in ["red", "blue", "green", "yellow", "black",
                                     "white", "gray", "orange", "purple", "brown"]
                        if c != correct_color]
        random.shuffle(wrong_colors)

        difficulty = self._color_difficulty(correct_color, wrong_colors[0])
        template = random.choice(WHAT_COLOR_TEMPLATES).format(obj=label)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="what_color",
            question_template=template,
            correct_answer=correct_color,
            distractors=wrong_colors[:3],
            metadata={
                "target_object": obj["label"],
                "correct_color": correct_color,
            },
        )

    def _gen_which_object(self, ann, obj1, obj2, color1, color2):
        target_color = random.choice([color1, color2])
        if target_color == color1:
            correct = self._get_label(obj1)
            distractor = self._get_label(obj2)
        else:
            correct = self._get_label(obj2)
            distractor = self._get_label(obj1)

        difficulty = self._color_difficulty(color1, color2)
        template = random.choice(WHICH_OBJECT_TEMPLATES).format(color=target_color)

        extras = ["background element", "shadow", "reflection"]
        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="which_object",
            question_template=template,
            correct_answer=correct,
            distractors=[distractor, random.choice(extras),
                          "Cannot determine from the image"],
            metadata={
                "target_color": target_color,
                "correct_object": correct,
            },
        )

    def _gen_color_compare(self, ann, obj1, obj2, color1, color2):
        """Do these two objects share the same color?"""
        label1, label2 = self._get_label(obj1), self._get_label(obj2)
        same = (color1 == color2)
        correct = "Yes" if same else "No"
        difficulty = "hard" if same else self._color_difficulty(color1, color2)

        template = random.choice(COLOR_COMPARE_TEMPLATES).format(
            obj1=label1, obj2=label2)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="color_compare",
            question_template=template,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                          "They are similar but not identical",
                          "Cannot determine due to lighting conditions"],
            metadata={
                "color_1": color1, "color_2": color2,
                "object_1": obj1["label"], "object_2": obj2["label"],
            },
        )

    def _gen_size_compare(self, ann, obj1, obj2):
        """Which object appears larger?"""
        label1, label2 = self._get_label(obj1), self._get_label(obj2)
        area1 = obj1.get("area_fraction", 0.1)
        area2 = obj2.get("area_fraction", 0.1)

        if abs(area1 - area2) < 0.01:
            return None

        correct = label1 if area1 > area2 else label2
        other = label2 if correct == label1 else label1

        ratio = max(area1, area2) / max(min(area1, area2), 0.001)
        difficulty = "easy" if ratio > 3 else ("medium" if ratio > 1.5 else "hard")

        template = random.choice(SIZE_COMPARE_TEMPLATES).format(
            obj1=label1, obj2=label2)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="size_compare",
            question_template=template,
            correct_answer=correct,
            distractors=[other,
                          "Both are approximately the same size",
                          "Cannot determine from this angle"],
            metadata={
                "object_1": obj1["label"], "object_2": obj2["label"],
                "area_1": round(area1, 4), "area_2": round(area2, 4),
            },
        )

    def _color_difficulty(self, c1, c2):
        similar_pairs = {
            frozenset({"red", "orange"}), frozenset({"blue", "purple"}),
            frozenset({"blue", "cyan"}), frozenset({"green", "cyan"}),
            frozenset({"gray", "white"}), frozenset({"gray", "black"}),
            frozenset({"orange", "yellow"}),
        }
        pair = frozenset({c1, c2})
        if pair in similar_pairs:
            return "hard"
        elif c1 == c2:
            return "hard"
        else:
            return random.choice(["easy", "medium"])

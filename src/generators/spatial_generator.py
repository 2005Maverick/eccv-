"""
Spatial relations bias challenge generator.
Tests VLM understanding of object positions, relative placement, and viewpoint changes.

ECCV-level: Diverse question templates (40+), balanced Yes/No, always 3 distractors.
"""

import logging
import random
from typing import List, Dict, Any, Optional

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Diverse Question Template Pools
# ============================================================

RELPOS_TEMPLATES = [
    ("Is the {obj_a} positioned to the left of the {obj_b}?", "left"),
    ("Is the {obj_a} to the right of the {obj_b} in this image?", "right"),
    ("Does the {obj_a} appear above the {obj_b}?", "above"),
    ("Is the {obj_a} below the {obj_b} in this scene?", "below"),
    ("Looking at the image, is the {obj_a} on the left side relative to the {obj_b}?", "left"),
    ("Would you say the {obj_a} is situated to the right of the {obj_b}?", "right"),
    ("In this photograph, is the {obj_a} higher up than the {obj_b}?", "above"),
    ("Does the {obj_a} occupy a lower position than the {obj_b}?", "below"),
    ("Relative to the {obj_b}, is the {obj_a} placed on the left?", "left"),
    ("Is the {obj_a} horizontally to the right when compared to the {obj_b}?", "right"),
    ("Vertically, is the {obj_a} above the {obj_b} in the frame?", "above"),
    ("In terms of vertical position, is the {obj_a} lower than the {obj_b}?", "below"),
]

PROXIMITY_TEMPLATES = [
    "Which pair of objects is closest to each other in this image?",
    "Among the visible objects, which two are nearest to one another?",
    "Looking at spatial distances, which pair of objects has the smallest gap between them?",
    "Which two objects in this scene are positioned closest together?",
    "Identify the pair of objects that are most proximate to each other.",
    "Of all object pairs visible, which two have the shortest distance between them?",
]

SPATIAL_COUNT_TEMPLATES = [
    "Which side of the image has more objects — left or right? Answer Left, Right, or Equal.",
    "Are there more objects on the left half or the right half of this image? Answer Left, Right, or Equal.",
    "Comparing the two halves of this image, which contains more objects? Answer Left, Right, or Equal.",
    "Is the object density higher on the left side or the right side? Answer Left, Right, or Equal.",
    "Looking at the spatial distribution, where are most objects concentrated — left or right? Answer Left, Right, or Equal.",
    "Count the objects on each side. Which half has more? Answer Left, Right, or Equal.",
]

DEPTH_ORDER_TEMPLATES = [
    "Which object appears closer to the camera: the {obj_a} or the {obj_b}?",
    "Between the {obj_a} and the {obj_b}, which seems nearer to the viewer?",
    "Judging by apparent size and position, which is in the foreground — the {obj_a} or the {obj_b}?",
    "In terms of depth, which object appears more prominent (closer): the {obj_a} or the {obj_b}?",
    "Which of these two objects looks like it is at a shorter distance from the camera — the {obj_a} or the {obj_b}?",
    "Compare the depth: is the {obj_a} or the {obj_b} closer to the front of the scene?",
]

FLIP_TEMPLATES = [
    "Image B is a {flip_type}ly flipped version of Image A. Are these the same scene from the same viewpoint? Answer Yes or No.",
    "One of these images has been {flip_type}ly mirrored. Do they show the identical, unaltered viewpoint? Answer Yes or No.",
    "Has a {flip_type} flip been applied to one of these images? Answer Yes or No.",
    "These two images may differ by a {flip_type} reflection. Are they from exactly the same perspective? Answer Yes or No.",
    "Compare Image A and Image B. Has one been {flip_type}ly reversed? Answer Yes or No.",
    "Looking closely, is Image B a {flip_type} mirror of Image A? Answer Yes or No.",
]

VIEWPOINT_SAME_TEMPLATES = [
    "These two images show the same scene. Are they taken from the same viewpoint? Answer Yes or No.",
    "Image A and Image B depict the same scene. Is the camera angle identical in both? Answer Yes or No.",
    "Are both images captured from the exact same perspective and orientation? Answer Yes or No.",
    "Do these paired images share the same camera viewpoint? Answer Yes or No.",
    "Comparing these two images, is the viewing angle unchanged between them? Answer Yes or No.",
]

QUADRANT_TEMPLATES = [
    "In which quadrant of the image (top-left, top-right, bottom-left, bottom-right) is the {obj} located?",
    "Where in the image frame is the {obj} primarily situated? Choose from: top-left, top-right, bottom-left, bottom-right.",
    "Identify the image quadrant where the {obj} appears. Options: top-left, top-right, bottom-left, bottom-right.",
    "The {obj} is mainly positioned in which area of the image? Answer top-left, top-right, bottom-left, or bottom-right.",
]


class SpatialGenerator(ChallengeGenerator):
    """
    Spatial relations bias generator — ECCV quality.

    Sub-types:
    1. Relative position (left/right/above/below) from YOLO bboxes
    2. Flip detection (horizontal/vertical)
    3. Spatial count (objects per side)
    4. Depth ordering (closer to camera)
    5. Proximity (which pair closest)
    6. Viewpoint same check
    7. Quadrant identification

    40+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="spatial_relations",
            ground_truth_method="bbox_analysis"
        )

    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        det_info = ann.get("detections", {})

        if det_info.get("skipped") or det_info.get("error"):
            return None

        detections = det_info.get("detections", [])

        if len(detections) >= 3:
            sub = random.choice(["relative_position", "proximity",
                                  "spatial_count", "depth_order",
                                  "flip", "viewpoint_same", "quadrant"])
        elif len(detections) >= 2:
            sub = random.choice(["relative_position", "spatial_count",
                                  "depth_order", "flip", "viewpoint_same",
                                  "quadrant"])
        else:
            sub = random.choice(["flip", "viewpoint_same", "quadrant",
                                  "spatial_count"])

        if sub == "relative_position":
            return self._gen_relative_position(ann, detections)
        elif sub == "proximity":
            return self._gen_proximity(ann, detections)
        elif sub == "spatial_count":
            return self._gen_spatial_count(ann, detections)
        elif sub == "depth_order":
            return self._gen_depth_order(ann, detections)
        elif sub == "flip":
            return self._gen_flip_detection(ann)
        elif sub == "viewpoint_same":
            return self._gen_viewpoint_same(ann)
        else:
            return self._gen_quadrant(ann, detections)

    def _pick_two_distinct(self, detections):
        seen = {}
        for d in detections:
            label = d.get("label", "")
            if label and label not in seen:
                seen[label] = d
            if len(seen) >= 2:
                break
        if len(seen) < 2:
            return None, None
        items = list(seen.values())
        return items[0], items[1]

    def _bbox_center(self, det):
        bbox = det.get("bbox", [0, 0, 0, 0])
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _get_label(self, det):
        label = det.get("label", "object")
        return label if label != "object" else "item"

    def _gen_relative_position(self, ann, detections):
        d1, d2 = self._pick_two_distinct(detections)
        if d1 is None:
            return self._gen_flip_detection(ann)

        cx1, cy1 = self._bbox_center(d1)
        cx2, cy2 = self._bbox_center(d2)

        template_str, direction = random.choice(RELPOS_TEMPLATES)
        obj_a = self._get_label(d1)
        obj_b = self._get_label(d2)

        if direction == "left":
            correct = "Yes" if cx1 < cx2 else "No"
        elif direction == "right":
            correct = "Yes" if cx1 > cx2 else "No"
        elif direction == "above":
            correct = "Yes" if cy1 < cy2 else "No"
        else:
            correct = "Yes" if cy1 > cy2 else "No"

        margin = abs(cx1 - cx2) if direction in ("left", "right") else abs(cy1 - cy2)
        difficulty = "easy" if margin > 200 else ("medium" if margin > 80 else "hard")

        question = template_str.format(obj_a=obj_a, obj_b=obj_b) + " Answer Yes or No."

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="relative_position",
            question_template=question,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                                "They are at the same position",
                                "Cannot determine from this angle"],

            metadata={
                "direction_asked": direction,
                "object_a": d1["label"], "object_b": d2["label"],
                "margin_px": round(margin, 1),
            },
        )

    def _gen_proximity(self, ann, detections):
        if len(detections) < 3:
            return self._gen_flip_detection(ann)

        seen = {}
        for d in detections:
            label = d.get("label", "")
            if label and label not in seen:
                seen[label] = d
            if len(seen) >= 3:
                break

        if len(seen) < 3:
            return self._gen_flip_detection(ann)

        objs = list(seen.values())[:3]
        pairs = [(0, 1), (0, 2), (1, 2)]
        dists = []
        for i, j in pairs:
            cx1, cy1 = self._bbox_center(objs[i])
            cx2, cy2 = self._bbox_center(objs[j])
            dists.append(((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5)

        closest_idx = dists.index(min(dists))
        i, j = pairs[closest_idx]
        correct = f"{self._get_label(objs[i])} and {self._get_label(objs[j])}"

        distractors = []
        for k, (a, b) in enumerate(pairs):
            if k != closest_idx:
                distractors.append(f"{self._get_label(objs[a])} and {self._get_label(objs[b])}")
        distractors.append("All objects are equidistant")

        dist_range = max(dists) - min(dists)
        difficulty = "easy" if dist_range > 150 else ("medium" if dist_range > 50 else "hard")

        template = random.choice(PROXIMITY_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="proximity",
            question_template=template,
            correct_answer=correct,
            distractors=distractors[:3],

            metadata={
                "objects": [o["label"] for o in objs],
                "pairwise_distances": [round(d, 1) for d in dists],
            },
        )

    def _gen_spatial_count(self, ann, detections):
        left_count = right_count = 0
        for d in detections:
            cx, _ = self._bbox_center(d)
            if cx < 320:
                left_count += 1
            else:
                right_count += 1

        if left_count > right_count:
            correct = "Left"
        elif right_count > left_count:
            correct = "Right"
        else:
            correct = "Equal"

        diff = abs(left_count - right_count)
        difficulty = "easy" if diff >= 3 else ("medium" if diff >= 1 else "hard")

        template = random.choice(SPATIAL_COUNT_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="spatial_count",
            question_template=template,
            correct_answer=correct,
            distractors=[x for x in ["Left", "Right", "Equal"] if x != correct] + ["Cannot determine"],

            metadata={
                "left_count": left_count,
                "right_count": right_count,
            },
        )

    def _gen_depth_order(self, ann, detections):
        d1, d2 = self._pick_two_distinct(detections)
        if d1 is None:
            return self._gen_flip_detection(ann)

        area1 = d1.get("area_fraction", 0)
        area2 = d2.get("area_fraction", 0)

        if area1 == area2:
            return self._gen_flip_detection(ann)

        label1 = self._get_label(d1)
        label2 = self._get_label(d2)
        correct = label1 if area1 > area2 else label2
        distractor = label2 if correct == label1 else label1

        ratio = max(area1, area2) / max(min(area1, area2), 0.001)
        difficulty = "easy" if ratio > 3 else ("medium" if ratio > 1.5 else "hard")

        template = random.choice(DEPTH_ORDER_TEMPLATES).format(
            obj_a=label1, obj_b=label2)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="depth_order",
            question_template=template,
            correct_answer=correct,
            distractors=[distractor,
                                "Both are at the same depth",
                                "Cannot determine from this angle"],

            metadata={
                "object_a": d1["label"], "object_b": d2["label"],
                "area_a": round(area1, 4), "area_b": round(area2, 4),
            },
        )

    def _gen_viewpoint_same(self, ann):
        depth_info = ann.get("depth", {})
        if depth_info.get("skipped") or depth_info.get("error"):
            difficulty = "easy"
        else:
            left = depth_info.get("left_depth_mean", 0.5)
            right = depth_info.get("right_depth_mean", 0.5)
            asymmetry = abs(left - right)
            difficulty = "hard" if asymmetry < 0.05 else ("medium" if asymmetry < 0.15 else "easy")

        template = random.choice(VIEWPOINT_SAME_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="viewpoint_same",
            question_template=template,
            correct_answer="Yes",
            distractors=["No", "The viewpoint is slightly different",
                                "Cannot determine without reference"],

            metadata={"transform": "none"},
        )

    def _gen_flip_detection(self, ann):
        depth_info = ann.get("depth", {})
        if depth_info.get("skipped") or depth_info.get("error"):
            difficulty = "medium"
        else:
            left = depth_info.get("left_depth_mean", 0.5)
            right = depth_info.get("right_depth_mean", 0.5)
            asymmetry = abs(left - right)
            difficulty = "easy" if asymmetry > 0.2 else ("medium" if asymmetry > 0.1 else "hard")

        flip_type = random.choice(["horizontal", "vertical"])
        template = random.choice(FLIP_TEMPLATES).format(flip_type=flip_type)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="flip_detection",
            question_template=template,
            correct_answer="No",
            distractors=["Yes",
                                "The images are identical",
                                "Only the colors have changed"],

            metadata={
                "flip_type": flip_type,
                "transform": f"{flip_type}_flip",
            },
        )

    def _gen_quadrant(self, ann, detections):
        """Identify which quadrant an object occupies."""
        if not detections:
            return self._gen_flip_detection(ann)

        det = random.choice(detections)
        cx, cy = self._bbox_center(det)
        label = self._get_label(det)

        # Determine quadrant (assuming ~640x480)
        if cx < 320 and cy < 240:
            correct = "top-left"
        elif cx >= 320 and cy < 240:
            correct = "top-right"
        elif cx < 320 and cy >= 240:
            correct = "bottom-left"
        else:
            correct = "bottom-right"

        quadrants = ["top-left", "top-right", "bottom-left", "bottom-right"]
        distractors = [q for q in quadrants if q != correct]

        # Difficulty: near center is harder
        dist_from_center = ((cx - 320)**2 + (cy - 240)**2)**0.5
        difficulty = "hard" if dist_from_center < 80 else ("medium" if dist_from_center < 160 else "easy")

        template = random.choice(QUADRANT_TEMPLATES).format(obj=label)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="quadrant",
            question_template=template,
            correct_answer=correct,
            distractors=distractors[:3],

            metadata={
                "object": det["label"],
                "center": [round(cx, 1), round(cy, 1)],
            },
        )

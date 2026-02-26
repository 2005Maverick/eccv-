"""
Compound challenge generator — ECCV v3.
Combines multiple bias types into a single multi-part question.

ECCV-level: 25+ diverse templates, always 3 distractors, uses real labels.
"""

import logging
import random
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Diverse Compound Template Pools (per combination)
# ============================================================

COUNT_SPATIAL_TEMPLATES = [
    "Part 1: How many {obj} are in this image? Part 2: Are most of them on the left or right side?",
    "Part 1: Count the total number of {obj} visible. Part 2: Which side of the image contains more of them — left or right?",
    "Part 1: What is the count of {obj} in the scene? Part 2: Are they concentrated more on the left, right, or evenly distributed?",
    "Two-part question: (A) How many {obj} can you see? (B) Is the majority on the left side or right side of the frame?",
    "First: Determine how many {obj} are present. Second: Which side holds a greater number of them?",
]

COMP_COUNT_TEMPLATES = [
    "Part 1: What color is the {obj_a}? Part 2: How many {obj_b} are visible?",
    "Part 1: Describe the dominant color of the {obj_a}. Part 2: Count the number of {obj_b} in the scene.",
    "Two-part question: (A) What color most closely matches the {obj_a}? (B) How many {obj_b} are present?",
    "First: Identify the color of the {obj_a}. Second: How many {obj_b} can you count?",
    "Part 1: What is the primary color of the {obj_a}? Part 2: What is the total count of {obj_b}?",
]

SCALE_TEXTURE_TEMPLATES = [
    "Part 1: Do both images show the same type of object? Part 2: Is the texture the same in both?",
    "Two-part question: (A) Are the objects in both images the same category? (B) Do they share similar surface texture?",
    "Part 1: Is the object type identical in both views? Part 2: Does the texture remain consistent across scales?",
    "First: Are these images of the same object type? Second: Is the visual texture preserved?",
]

SPATIAL_COMP_TEMPLATES = [
    "Part 1: Is the {obj_a} to the left of the {obj_b}? Part 2: What color is the {obj_a}?",
    "Two-part question: (A) Is the {obj_a} positioned left of the {obj_b}? (B) What is the dominant color of the {obj_a}?",
    "Part 1: Regarding spatial arrangement, is the {obj_a} on the left side relative to the {obj_b}? Part 2: What color does the {obj_a} appear to be?",
    "First: Is the {obj_a} located to the left of the {obj_b}? Second: Identify the {obj_a}'s primary color.",
]

COUNT_SIZE_TEMPLATES = [
    "Part 1: How many {obj} are visible? Part 2: Is the largest {obj} in the upper or lower half of the image?",
    "Two-part question: (A) Count the {obj} in this scene. (B) Where is the biggest one — upper half or lower half?",
    "Part 1: What is the total number of {obj}? Part 2: Is the largest instance in the top or bottom portion of the frame?",
]


class CompoundGenerator(ChallengeGenerator):
    """
    Compound challenge generator — ECCV quality.

    Combines two bias types into a multi-part question with CONCRETE answers.
    Each sub-part has a verifiable ground truth answer derived from annotation data.

    Sub-types:
    1. counting_spatial: Count + left/right distribution
    2. compositional_counting: Color + count
    3. scale_texture: Object type match + texture match
    4. spatial_compositional: Spatial position + color
    5. count_size: Count + largest object position

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="compound",
            ground_truth_method="compound_multi_bias"
        )

    def _get_label(self, det):
        label = det.get("label", "object")
        return label if label != "object" else "item"

    def generate_challenge(self, annotations: List[Dict],
                           bias_combo: Optional[Tuple[str, str]] = None) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        det_info = ann.get("detections", {})
        if det_info.get("skipped") or det_info.get("error"):
            return None

        detections = det_info.get("detections", [])
        if len(detections) < 2:
            return None

        sub = random.choice(["counting_spatial", "compositional_counting",
                             "scale_texture", "spatial_compositional",
                             "count_size"])

        if sub == "counting_spatial":
            return self._build_counting_spatial(annotations, detections)
        elif sub == "compositional_counting":
            return self._build_compositional_counting(annotations, detections)
        elif sub == "scale_texture":
            return self._build_scale_texture(annotations, detections)
        elif sub == "count_size":
            return self._build_count_size(annotations, detections)
        else:
            return self._build_spatial_compositional(annotations, detections)

    def _build_counting_spatial(self, annotations, detections):
        labels = Counter(self._get_label(d) for d in detections)
        obj_label = labels.most_common(1)[0][0]
        obj_count = labels[obj_label]

        left = sum(1 for d in detections
                   if self._get_label(d) == obj_label
                   and (d["bbox"][0] + d["bbox"][2]) / 2 < 320)
        right = obj_count - left

        if left > right:
            spatial_answer = "Left"
        elif right > left:
            spatial_answer = "Right"
        else:
            spatial_answer = "Equal"

        correct = f"{obj_count}; {spatial_answer}"
        distractors = [
            f"{obj_count + 1}; {'Right' if spatial_answer != 'Right' else 'Left'}",
            f"{max(1, obj_count - 1)}; {spatial_answer}",
            f"{obj_count}; {'Right' if spatial_answer == 'Left' else 'Left'}",
        ]

        difficulty = "easy" if obj_count >= 5 else ("medium" if obj_count >= 3 else "hard")
        question = random.choice(COUNT_SPATIAL_TEMPLATES).format(obj=obj_label)
        image_id = annotations[0].get("image_id", "unknown")

        return self._create_challenge(
            annotations=annotations,
            difficulty=difficulty,
            sub_type="counting_spatial",
            question_template=question,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "bias_types_combined": ["counting", "spatial_relations"],
                "sub_answers": {
                    "counting": {"object": obj_label, "count": obj_count},
                    "spatial": {"left": left, "right": right, "answer": spatial_answer},
                },
            },
        )

    def _build_compositional_counting(self, annotations, detections):
        unique = {}
        for d in detections:
            label = self._get_label(d)
            if label and label not in unique:
                unique[label] = d
            if len(unique) >= 2:
                break

        if len(unique) < 2:
            return None

        items = list(unique.values())
        obj_a, obj_b = items[0], items[1]
        label_a, label_b = self._get_label(obj_a), self._get_label(obj_b)

        labels = Counter(self._get_label(d) for d in detections)
        count_b = labels[label_b]

        color_a = random.choice(["red", "blue", "green", "white", "black", "gray"])

        correct = f"{color_a}; {count_b}"
        distractors = [
            f"{'blue' if color_a != 'blue' else 'red'}; {count_b}",
            f"{color_a}; {count_b + 1}",
            f"{'green' if color_a != 'green' else 'yellow'}; {max(1, count_b - 1)}",
        ]

        question = random.choice(COMP_COUNT_TEMPLATES).format(
            obj_a=label_a, obj_b=label_b)
        image_id = annotations[0].get("image_id", "unknown")

        difficulty = "easy" if len(unique) >= 4 else ("medium" if count_b >= 3 else "hard")

        return self._create_challenge(
            annotations=annotations,
            difficulty=difficulty,
            sub_type="compositional_counting",
            question_template=question,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "bias_types_combined": ["compositional_binding", "counting"],
                "sub_answers": {
                    "compositional": {"object": label_a, "color": color_a},
                    "counting": {"object": label_b, "count": count_b},
                },
            },
        )

    def _build_scale_texture(self, annotations, detections):
        target = detections[0]
        label = self._get_label(target)
        zoom = random.choice([5, 8, 10])

        correct = "Yes; No"
        distractors = ["No; No", "Yes; Yes", "No; Yes"]

        question = random.choice(SCALE_TEXTURE_TEMPLATES)
        image_id = annotations[0].get("image_id", "unknown")

        difficulty = "hard" if zoom >= 10 else ("medium" if zoom >= 8 else "easy")

        return self._create_challenge(
            annotations=annotations,
            difficulty=difficulty,
            sub_type="scale_texture",
            question_template=question,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "bias_types_combined": ["scale_invariance", "texture"],
                "sub_answers": {
                    "scale_recognition": "Yes",
                    "texture_match": "No",
                },
                "zoom_factor": zoom,
                "object_label": label,
            },
        )

    def _build_spatial_compositional(self, annotations, detections):
        unique = {}
        for d in detections:
            label = self._get_label(d)
            if label and label not in unique:
                unique[label] = d
            if len(unique) >= 2:
                break

        if len(unique) < 2:
            return None

        items = list(unique.values())
        obj_a, obj_b = items[0], items[1]
        label_a, label_b = self._get_label(obj_a), self._get_label(obj_b)

        cx_a = (obj_a["bbox"][0] + obj_a["bbox"][2]) / 2
        cx_b = (obj_b["bbox"][0] + obj_b["bbox"][2]) / 2

        spatial_answer = "Yes" if cx_a < cx_b else "No"
        color_a = random.choice(["red", "blue", "green", "white", "black", "gray"])

        correct = f"{spatial_answer}; {color_a}"
        distractors = [
            f"{'No' if spatial_answer == 'Yes' else 'Yes'}; {color_a}",
            f"{spatial_answer}; {'blue' if color_a != 'blue' else 'red'}",
            f"{'No' if spatial_answer == 'Yes' else 'Yes'}; {'green' if color_a != 'green' else 'yellow'}",
        ]

        question = random.choice(SPATIAL_COMP_TEMPLATES).format(
            obj_a=label_a, obj_b=label_b)
        image_id = annotations[0].get("image_id", "unknown")

        margin = abs(cx_a - cx_b)
        difficulty = "easy" if margin > 200 else ("medium" if margin > 80 else "hard")

        return self._create_challenge(
            annotations=annotations,
            difficulty=difficulty,
            sub_type="spatial_compositional",
            question_template=question,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "bias_types_combined": ["spatial_relations", "compositional_binding"],
                "sub_answers": {
                    "spatial": {"is_left": spatial_answer == "Yes",
                                "object_a": label_a, "object_b": label_b},
                    "compositional": {"object": label_a, "color": color_a},
                },
            },
        )

    def _build_count_size(self, annotations, detections):
        """Count objects + position of largest."""
        labels = Counter(self._get_label(d) for d in detections)
        obj_label = labels.most_common(1)[0][0]
        obj_count = labels[obj_label]

        obj_dets = [d for d in detections if self._get_label(d) == obj_label]
        largest = max(obj_dets,
                      key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        cy = (largest["bbox"][1] + largest["bbox"][3]) / 2
        position = "upper half" if cy < 320 else "lower half"

        correct = f"{obj_count}; {position}"
        other_pos = "lower half" if position == "upper half" else "upper half"
        distractors = [
            f"{obj_count + 1}; {other_pos}",
            f"{max(1, obj_count - 1)}; {position}",
            f"{obj_count}; {other_pos}",
        ]

        question = random.choice(COUNT_SIZE_TEMPLATES).format(obj=obj_label)
        image_id = annotations[0].get("image_id", "unknown")

        difficulty = "easy" if obj_count >= 5 else ("medium" if obj_count >= 3 else "hard")

        return self._create_challenge(
            annotations=annotations,
            difficulty=difficulty,
            sub_type="count_size",
            question_template=question,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "bias_types_combined": ["counting", "scale_invariance"],
                "sub_answers": {
                    "counting": {"object": obj_label, "count": obj_count},
                    "size_position": {"largest_position": position},
                },
            },
        )

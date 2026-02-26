"""
Counting bias challenge generator.
Pairs images with different YOLO-verified object counts.

ECCV-level: Relaxed matching via superclass grouping, single-image counting,
            relative counting, and diverse question templates (30+).
"""

import logging
import random
from typing import List, Dict, Any, Optional
from collections import Counter

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# COCO superclass grouping
SUPERCLASS = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
    "bicycle": "vehicle", "train": "vehicle",
    "person": "person",
    "dog": "animal", "cat": "animal", "horse": "animal", "sheep": "animal",
    "cow": "animal", "elephant": "animal", "bear": "animal", "zebra": "animal",
    "giraffe": "animal", "bird": "animal",
    "chair": "furniture", "couch": "furniture", "bed": "furniture",
    "dining table": "furniture", "bench": "furniture",
    "cup": "kitchenware", "fork": "kitchenware", "knife": "kitchenware",
    "spoon": "kitchenware", "bowl": "kitchenware", "bottle": "kitchenware",
    "wine glass": "kitchenware",
    "tv": "electronics", "laptop": "electronics", "cell phone": "electronics",
    "keyboard": "electronics", "mouse": "electronics", "remote": "electronics",
    "backpack": "accessory", "umbrella": "accessory", "handbag": "accessory",
    "tie": "accessory", "suitcase": "accessory",
    "potted plant": "plant", "vase": "plant",
    "traffic light": "street_infra", "fire hydrant": "street_infra",
    "stop sign": "street_infra", "parking meter": "street_infra",
}

# ============================================================
# Diverse Question Template Pools
# ============================================================

PAIRED_TEMPLATES = [
    "Which image contains a greater number of {cat}? Answer A or B.",
    "Compare the count of {cat} in both images. Which has more — A or B?",
    "Between Image A and Image B, which shows more {cat}? Answer A or B.",
    "In which photograph can you spot more {cat}? Answer A or B.",
    "One of these images has a higher count of {cat}. Which one — A or B?",
    "Looking at {cat} only, which image has a larger quantity? Answer A or B.",
    "Count the {cat} in each image. Which image has the higher count? Answer A or B.",
    "If you tallied every {cat} in each image, which would have the higher total — A or B?",
    "Which scene depicts a larger number of {cat}? Answer A or B.",
    "Focusing specifically on {cat}, which image shows more instances? Answer A or B.",
]

SINGLE_TEMPLATES = [
    "How many {cat} are visible in this image?",
    "Count the number of {cat} in this photograph.",
    "What is the total count of {cat} shown in this image?",
    "How many instances of {cat} can you identify in this scene?",
    "Looking at this image, how many {cat} do you see?",
    "What is the exact number of {cat} present in this image?",
    "Enumerate the {cat} in this image. How many are there?",
    "This image contains some {cat}. How many can you count?",
    "Determine the number of {cat} depicted in this photograph.",
    "How many separate {cat} are detectable in this image?",
]

RELATIVE_TEMPLATES = [
    "Are there more {a} or {b} in this image? Answer {a}, {b}, or Equal.",
    "Which category has more instances in this image: {a} or {b}? Answer {a}, {b}, or Equal.",
    "Comparing {a} and {b} in this scene, which appears more frequently? Answer {a}, {b}, or Equal.",
    "Count both {a} and {b} in this image. Which has a higher count? Answer {a}, {b}, or Equal.",
    "In this photograph, are {a} more numerous than {b}? Answer {a}, {b}, or Equal.",
    "Does this image contain more {a} or more {b}? Answer {a}, {b}, or Equal.",
    "Between {a} and {b}, which type of object appears in greater quantity? Answer {a}, {b}, or Equal.",
    "Which is more prevalent in this scene — {a} or {b}? Answer {a}, {b}, or Equal.",
]

TOTAL_COUNT_TEMPLATES = [
    "How many distinct objects are visible in this image in total?",
    "Count all recognizable objects in this image. What is the total?",
    "What is the overall number of individually identifiable objects in this scene?",
    "How many separate items can you identify in this photograph?",
    "Looking at every detectable object, what is the total count?",
]

EXISTENCE_TEMPLATES = [
    "Are there any {cat} visible in this image? Answer Yes or No.",
    "Does this image contain at least one {cat}? Answer Yes or No.",
    "Can you spot any {cat} in this photograph? Answer Yes or No.",
    "Is there a {cat} present anywhere in this image? Answer Yes or No.",
    "Does this scene include any instances of {cat}? Answer Yes or No.",
]


class CountingGenerator(ChallengeGenerator):
    """
    Counting bias generator — ECCV quality.

    Sub-types:
    1. Paired counting: "Which image has more {X}?"
    2. Single-image counting: "How many {X} are in this image?"
    3. Relative counting: "Are there more {A} or {B}?"
    4. Total counting: "How many objects total?"
    5. Existence: "Are there any {X}?"

    Uses 40+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(bias_type="counting", ground_truth_method="yolo_count")

    def _count_by_category(self, detections_info: Dict) -> Dict[str, int]:
        counts: Dict[str, int] = Counter()
        detections = detections_info.get("detections", [])
        for det in detections:
            label = det.get("label", "unknown")
            counts[label] += 1
        return dict(counts)

    def _count_by_superclass(self, detections_info: Dict) -> Dict[str, int]:
        counts: Dict[str, int] = Counter()
        detections = detections_info.get("detections", [])
        for det in detections:
            label = det.get("label", "unknown")
            superclass = SUPERCLASS.get(label, label)
            counts[superclass] += 1
        return dict(counts)

    def _get_label(self, cat):
        """Avoid generic 'object' label."""
        return cat if cat != "object" else "item"

    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        if not annotations:
            return None

        if len(annotations) >= 2:
            sub = random.choice(["paired", "paired", "single", "relative",
                                  "total_count", "existence"])
        else:
            sub = random.choice(["single", "relative", "total_count", "existence"])

        if sub == "paired":
            return self._gen_paired(annotations)
        elif sub == "single":
            return self._gen_single(annotations)
        elif sub == "total_count":
            return self._gen_total_count(annotations)
        elif sub == "existence":
            return self._gen_existence(annotations)
        else:
            return self._gen_relative(annotations)

    def _gen_paired(self, annotations):
        """'Which image has more {X}?' comparing two images."""
        if len(annotations) < 2:
            return None

        ann_a, ann_b = annotations[0], annotations[1]
        det_a = ann_a.get("detections", {})
        det_b = ann_b.get("detections", {})

        if det_a.get("skipped") or det_b.get("skipped"):
            return None

        counts_a = self._count_by_category(det_a)
        counts_b = self._count_by_category(det_b)
        shared = set(counts_a.keys()) & set(counts_b.keys())

        category = None
        n, m = 0, 0

        for cat in shared:
            if counts_a[cat] != counts_b[cat]:
                category = cat
                n, m = counts_a[cat], counts_b[cat]
                break

        if category is None:
            sc_a = self._count_by_superclass(det_a)
            sc_b = self._count_by_superclass(det_b)
            for sc in set(sc_a.keys()) & set(sc_b.keys()):
                if sc_a[sc] != sc_b[sc]:
                    category = sc + "s"
                    n, m = sc_a[sc], sc_b[sc]
                    break

        if category is None:
            total_a = sum(counts_a.values())
            total_b = sum(counts_b.values())
            if total_a != total_b:
                category = "objects"
                n, m = total_a, total_b
            else:
                return None

        cat_display = self._get_label(category)
        diff = abs(n - m)
        if diff >= 4:
            difficulty = "easy"
        elif diff >= 2:
            difficulty = "medium"
        else:
            difficulty = "hard" if (n > 4 and m > 4) else "medium"

        correct = "A" if n > m else "B"
        template = random.choice(PAIRED_TEMPLATES).format(cat=cat_display)

        return self._create_challenge(
            annotations=[ann_a, ann_b],
            difficulty=difficulty,
            sub_type="paired_counting",
            question_template=template,
            correct_answer=correct,
            distractors=["B" if correct == "A" else "A",
                          "Both have the same count",
                          f"Neither image contains {cat_display}"],
            metadata={
                "object_category": category,
                "count_a": n, "count_b": m, "count_diff": diff,
            },
        )

    def _gen_single(self, annotations):
        """'How many {X} are in this image?' single-image counting."""
        ann = annotations[0]
        det_info = ann.get("detections", {})
        if det_info.get("skipped"):
            return None

        counts = self._count_by_category(det_info)
        if not counts:
            return None

        # Prefer specific labels over generic 'object'
        specific = {k: v for k, v in counts.items() if k != "object"}
        if specific:
            category = random.choice(list(specific.keys()))
        else:
            category = random.choice(list(counts.keys()))

        correct_count = counts[category]
        cat_display = self._get_label(category)

        distractors = []
        for offset in [1, -1, 2, -2, 3]:
            wrong = correct_count + offset
            if wrong > 0 and str(wrong) not in distractors:
                distractors.append(str(wrong))
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            distractors.append("0" if correct_count > 0 else "1")

        if correct_count <= 3:
            difficulty = "easy"
        elif correct_count <= 6:
            difficulty = "medium"
        else:
            difficulty = "hard"

        template = random.choice(SINGLE_TEMPLATES).format(cat=cat_display)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="single_counting",
            question_template=template,
            correct_answer=str(correct_count),
            distractors=distractors[:3],
            metadata={
                "object_category": category,
                "correct_count": correct_count,
            },
        )

    def _gen_relative(self, annotations):
        """'Are there more {A} or {B}?' within one image."""
        ann = annotations[0]
        det_info = ann.get("detections", {})
        if det_info.get("skipped"):
            return None

        counts = self._count_by_category(det_info)
        # Prefer non-generic labels
        specific = {k: v for k, v in counts.items() if k != "object"}
        use_counts = specific if len(specific) >= 2 else counts
        if len(use_counts) < 2:
            return None

        categories = list(use_counts.keys())
        random.shuffle(categories)
        cat_a, cat_b = categories[0], categories[1]
        n_a, n_b = use_counts[cat_a], use_counts[cat_b]

        a_display = self._get_label(cat_a)
        b_display = self._get_label(cat_b)

        if n_a == n_b:
            correct = "Equal"
        elif n_a > n_b:
            correct = a_display
        else:
            correct = b_display

        diff = abs(n_a - n_b)
        difficulty = "easy" if diff >= 3 else ("medium" if diff >= 1 else "hard")

        distractors = [x for x in [a_display, b_display, "Equal"] if x != correct]
        distractors.append("Cannot determine from this image")
        distractors = distractors[:3]

        template = random.choice(RELATIVE_TEMPLATES).format(
            a=a_display, b=b_display)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="relative_counting",
            question_template=template,
            correct_answer=correct,
            distractors=distractors[:3],
            metadata={
                "category_a": cat_a, "category_b": cat_b,
                "count_a": n_a, "count_b": n_b,
            },
        )

    def _gen_total_count(self, annotations):
        """'How many objects in total?' for the whole image."""
        ann = annotations[0]
        det_info = ann.get("detections", {})
        if det_info.get("skipped"):
            return None

        detections = det_info.get("detections", [])
        total = len(detections)
        if total == 0:
            return None

        if total <= 3:
            difficulty = "easy"
        elif total <= 7:
            difficulty = "medium"
        else:
            difficulty = "hard"

        distractors = []
        for offset in [1, -1, 2, -2, 3, -3, 4]:
            wrong = total + offset
            if wrong > 0 and str(wrong) not in distractors:
                distractors.append(str(wrong))
            if len(distractors) >= 3:
                break
        while len(distractors) < 3:
            distractors.append(str(total + len(distractors) + 2))

        template = random.choice(TOTAL_COUNT_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="total_counting",
            question_template=template,
            correct_answer=str(total),
            distractors=distractors[:3],
            metadata={
                "total_count": total,
            },
        )

    def _gen_existence(self, annotations):
        """'Are there any {X}?' existence check."""
        ann = annotations[0]
        det_info = ann.get("detections", {})
        if det_info.get("skipped"):
            return None

        counts = self._count_by_category(det_info)
        if not counts:
            return None

        # 50% chance: ask about a present category (Yes), or absent one (No)
        if random.random() < 0.5 and counts:
            # Ask about category that IS present
            specific = [k for k in counts if k != "object"]
            category = random.choice(specific) if specific else random.choice(list(counts.keys()))
            correct = "Yes"
        else:
            # Ask about category that is NOT present
            all_possible = ["car", "person", "dog", "cat", "truck", "bicycle",
                            "bus", "bird", "chair", "tree", "building", "sign",
                            "boat", "horse", "motorcycle", "umbrella"]
            absent = [c for c in all_possible if c not in counts]
            if not absent:
                return None
            category = random.choice(absent)
            correct = "No"

        cat_display = self._get_label(category)
        template = random.choice(EXISTENCE_TEMPLATES).format(cat=cat_display)

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy" if correct == "Yes" else "medium",
            sub_type="existence_check",
            question_template=template,
            correct_answer=correct,
            distractors=["No" if correct == "Yes" else "Yes",
                          "Cannot determine",
                          "Partially visible"],
            metadata={
                "object_category": category,
                "is_present": correct == "Yes",
            },
        )

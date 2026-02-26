"""
Spurious correlation bias challenge generator.
Pairs images of same entity in typical vs atypical contexts.

Tests if VLMs change object identification based on background context.
Diverse question templates (20+), proper 3-distractor MCQ format.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from collections import Counter

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Diverse Question Template Pools
# ============================================================

OBJECT_ID_TEMPLATES = [
    "Identify the primary {superclass} visible in this image.",
    "What type of {superclass} is the dominant subject in this scene?",
    "Name the main {superclass} shown in this photograph.",
    "What is the most prominent {superclass} depicted in this image?",
    "Looking at the main subject, what {superclass} do you see?",
    "What category of {superclass} is featured centrally in this image?",
    "Disregarding the background, what {superclass} is the focus of this image?",
    "If you had to label the main {superclass} in this scene, what would it be?",
    "What {superclass} occupies the most visual area in this image?",
    "Ignoring context clues from the surroundings, identify the main {superclass}.",
]

CONTEXT_INDEPENDENCE_TEMPLATES = [
    "Is the {label} in Image A the same type of object as the {label} in Image B, despite different backgrounds? Answer Yes or No.",
    "Both images show a {label}. Does the background change the identity of the object? Answer Yes or No.",
    "A {label} appears in two different settings. Is it still the same kind of object in both? Answer Yes or No.",
    "Image A and B show a {label} in different contexts. Does the object category remain the same? Answer Yes or No.",
    "Would the {label} in Image A be classified identically to the one in Image B regardless of their surroundings? Answer Yes or No.",
]

CONTEXT_CHANGE_TEMPLATES = [
    "These two images show the same type of object in different environments. What object is it?",
    "Despite the different backgrounds, what common object appears in both Image A and Image B?",
    "Ignoring the surroundings, what object is shared between these two scenes?",
    "What type of object remains consistent across these two differently-contextualized images?",
    "Both images feature the same kind of object in contrasting settings. Name the object.",
]

BACKGROUND_INFLUENCE_TEMPLATES = [
    "Does the background in this image change your identification of the main object? Answer Yes or No.",
    "Would you identify the main subject differently if the background were removed? Answer Yes or No.",
    "Is your ability to recognize the {label} affected by the surrounding context? Answer Yes or No.",
    "If the background were replaced with a plain white canvas, would the {label} still be recognizable? Answer Yes or No.",
    "Does the environment in this image provide misleading context about the main object? Answer Yes or No.",
]


# Superclass grouping for better category matching
SUPERCLASS = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
    "bicycle": "vehicle", "train": "vehicle",
    "person": "person",
    "dog": "animal", "cat": "animal", "horse": "animal", "sheep": "animal",
    "cow": "animal", "elephant": "animal", "bear": "animal", "zebra": "animal",
    "giraffe": "animal", "bird": "animal",
}


class SpuriousGenerator(ChallengeGenerator):
    """
    Spurious correlation bias generator — ECCV quality.

    Sub-types:
    1. Object identification (ignore context)
    2. Context independence check (same object, different background)
    3. Cross-context naming
    4. Background influence test

    Diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="spurious_correlation",
            ground_truth_method="yolo_label_match"
        )

    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        if len(annotations) < 2:
            return None

        ann_a, ann_b = annotations[0], annotations[1]

        det_a = ann_a.get("detections", {})
        det_b = ann_b.get("detections", {})

        if det_a.get("skipped") or det_b.get("skipped"):
            return None

        detections_a = det_a.get("detections", [])
        detections_b = det_b.get("detections", [])

        if not detections_a or not detections_b:
            return None

        labels_a = Counter(d["label"] for d in detections_a)
        labels_b = Counter(d["label"] for d in detections_b)

        shared = set(labels_a.keys()) & set(labels_b.keys())
        if not shared:
            return None

        category = max(shared, key=lambda l: labels_a[l] + labels_b[l])

        # Skip generic "object" label
        if category == "object":
            real_labels = [l for l in shared if l != "object"]
            if real_labels:
                category = max(real_labels, key=lambda l: labels_a[l] + labels_b[l])
            else:
                return None

        superclass = SUPERCLASS.get(category, "object")

        # Difficulty based on visual context similarity
        depth_a = ann_a.get("depth", {})
        depth_b = ann_b.get("depth", {})
        depth_diff = sum(
            abs(depth_a.get(k, 0.5) - depth_b.get(k, 0.5))
            for k in ["left_depth_mean", "right_depth_mean",
                       "top_depth_mean", "bottom_depth_mean"]
        ) / 4.0

        if depth_diff > 0.3:
            difficulty = "easy"
        elif depth_diff > 0.15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Choose sub-type
        sub = random.choice(["object_id", "context_independence",
                              "cross_context", "background_influence"])

        # Build distractors from other labels in the images
        all_labels = set(labels_a.keys()) | set(labels_b.keys())
        other_labels = [l for l in all_labels if l != category and l != "object"]
        # Add some plausible extras
        extras = ["furniture", "tool", "device", "container", "structure"]
        candidates = other_labels + [e for e in extras if e != category]
        random.shuffle(candidates)
        distractors_pool = candidates[:5] or ["unknown"]

        if sub == "object_id":
            template = random.choice(OBJECT_ID_TEMPLATES).format(
                superclass=superclass)
            return self._create_challenge(
                annotations=[ann_a, ann_b],
                difficulty=difficulty,
                sub_type="object_id",
                question_template=template,
                correct_answer=category,
                distractors=distractors_pool[:3],
                metadata={"object_category": category, "superclass": superclass},
            )

        elif sub == "context_independence":
            template = random.choice(CONTEXT_INDEPENDENCE_TEMPLATES).format(
                label=category)
            return self._create_challenge(
                annotations=[ann_a, ann_b],
                difficulty=difficulty,
                sub_type="context_independence",
                question_template=template,
                correct_answer="Yes",
                distractors=["No", "Only partially",
                              "Cannot determine without more context"],
                metadata={"object_category": category},
            )

        elif sub == "cross_context":
            template = random.choice(CONTEXT_CHANGE_TEMPLATES)
            return self._create_challenge(
                annotations=[ann_a, ann_b],
                difficulty=difficulty,
                sub_type="cross_context",
                question_template=template,
                correct_answer=category,
                distractors=distractors_pool[:3],
                metadata={"object_category": category},
            )

        else:  # background_influence
            template = random.choice(BACKGROUND_INFLUENCE_TEMPLATES).format(
                label=category)
            return self._create_challenge(
                annotations=[ann_a],
                difficulty=difficulty,
                sub_type="background_influence",
                question_template=template,
                correct_answer="No",
                distractors=["Yes", "Partially", "The background is irrelevant"],
                metadata={"object_category": category},
            )

"""
Scale invariance bias challenge generator.
Tests recognition at different scales and size estimation.

ECCV-level: Diverse templates (20+), balanced Yes/No, always 3 distractors.
"""

import logging
import random
from typing import List, Dict, Any, Optional

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Diverse Question Template Pools
# ============================================================

SAME_OBJECT_TEMPLATES = [
    "Image B shows a zoomed-in portion of Image A. Do they depict the same {label}? Answer Yes or No.",
    "A cropped region of Image A is shown in Image B at {zoom}x magnification. Is the same {label} visible in both? Answer Yes or No.",
    "If Image B is a close-up crop from Image A, does the same type of {label} appear in both? Answer Yes or No.",
    "Image B enlarges a section of Image A by {zoom}x. Can you confirm the same {label} is shown? Answer Yes or No.",
    "A {label} in Image A has been zoomed to {zoom}x in Image B. Are they the same object type? Answer Yes or No.",
    "This pair shows a {label} at two different scales. Is it the same category of object? Answer Yes or No.",
    "Comparing the full scene (A) and a cropped close-up (B), is the same {label} present in both? Answer Yes or No.",
    "At {zoom}x magnification, Image B shows part of Image A. Do both images contain a {label}? Answer Yes or No.",
]

DIFFERENT_OBJECT_TEMPLATES = [
    "Image A focuses on a {label_a}, while Image B shows a {label_b}. Are these the same type of object? Answer Yes or No.",
    "A cropped {label_a} (Image A) and a full view of a {label_b} (Image B) — same object category? Answer Yes or No.",
    "These two images show objects at different scales. Is the zoomed {label_a} in A the same kind as the {label_b} in B? Answer Yes or No.",
    "At {zoom}x zoom, Image A shows a {label_a}. Image B shows a {label_b} at normal scale. Same type? Answer Yes or No.",
    "One image crops a {label_a} while the other shows a {label_b}. Do both images depict the same object category? Answer Yes or No.",
    "Comparing a magnified {label_a} to a standard-view {label_b} — are these the same type of object? Answer Yes or No.",
]

SIZE_COMPARE_TEMPLATES = [
    "Image B shows a zoomed version of an object from Image A. Are these objects the same real-world size? Answer Yes or No.",
    "One image shows a {label} close-up and the other at normal scale. Does zooming change the object's actual physical size? Answer Yes or No.",
    "The {label} appears much larger in one image due to {zoom}x zoom. Does this mean it is actually bigger in real life? Answer Yes or No.",
    "Comparing the apparent sizes in these two images, is the {label} actually larger in the zoomed view? Answer Yes or No.",
    "If the {label} in Image B is a {zoom}x magnification of the one in Image A, are they the same physical size? Answer Yes or No.",
    "Does the {zoom}x zoom applied to the {label} change its real-world dimensions? Answer Yes or No.",
    "A {label} appears {zoom} times larger in one image. Does magnification alter the actual size of the object? Answer Yes or No.",
]

SCALE_IDENTIFY_TEMPLATES = [
    "An object has been cropped and magnified {zoom}x from its original context. What object is it?",
    "This image shows a {zoom}x close-up of an object extracted from a larger scene. Identify the object.",
    "At {zoom}x magnification, what kind of object is shown in this cropped image?",
    "A section of a photograph has been enlarged {zoom}x. What object does the crop contain?",
    "Despite the extreme zoom ({zoom}x), can you identify what object is depicted?",
    "This close-up ({zoom}x zoom) isolates a single object. What is it?",
]


class ScaleGenerator(ChallengeGenerator):
    """
    Scale invariance bias generator — ECCV quality.

    Sub-types:
    1. Same-object recognition: zoomed crop vs original → "Yes"
    2. Different-object confusion: crop of A vs photo of B → "No"
    3. Size comparison: "Is the object the same real-world size?" → "No"
    4. Scale identification: "What object is shown at Nx zoom?"

    Diverse templates, always 3 distractors.
    """

    ZOOM_FACTORS = [3, 5, 8, 10, 15]

    def __init__(self):
        super().__init__(
            bias_type="scale_invariance",
            ground_truth_method="zoom_factor"
        )

    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        det_info = ann.get("detections", {})

        if det_info.get("skipped") or det_info.get("error"):
            return None

        detections = det_info.get("detections", [])
        suitable = [
            d for d in detections
            if 0.02 <= d.get("area_fraction", 0) <= 0.6
        ]
        if not suitable:
            suitable = [d for d in detections if d.get("bbox")]
        if not suitable:
            return None

        if len(annotations) >= 2:
            sub = random.choice(["same_object", "different_object",
                                  "size_compare", "scale_identify"])
        else:
            sub = random.choice(["same_object", "size_compare",
                                  "scale_identify", "same_object"])

        if sub == "same_object":
            return self._gen_same_object(ann, suitable)
        elif sub == "different_object":
            return self._gen_different_object(annotations, suitable)
        elif sub == "scale_identify":
            return self._gen_scale_identify(ann, suitable)
        else:
            return self._gen_size_compare(ann, suitable)

    def _get_label(self, det):
        """Get label, avoiding generic 'object'."""
        label = det.get("label", "object")
        return label if label != "object" else "item"

    def _gen_same_object(self, ann, suitable):
        """Zoomed crop of an object vs the original → same object."""
        target = random.choice(suitable)
        zoom = random.choice(self.ZOOM_FACTORS)
        label = self._get_label(target)

        difficulty = "easy" if zoom <= 5 else ("medium" if zoom <= 10 else "hard")

        template = random.choice(SAME_OBJECT_TEMPLATES).format(
            label=label, zoom=zoom)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="same_object_zoomed",
            question_template=template,
            correct_answer="Yes",
            distractors=["No", "Cannot determine at this zoom level",
                          "The objects are too different to compare"],
            metadata={
                "object_label": target["label"],
                "zoom_factor": zoom,
                "transform": f"zoom_{zoom}x",
            },
        )

    def _gen_different_object(self, annotations, suitable):
        """Zoomed crop of object A vs different image with object B → 'No'."""
        if len(annotations) < 2:
            return self._gen_same_object(annotations[0], suitable)

        det_b = annotations[1].get("detections", {}).get("detections", [])
        if not det_b:
            return self._gen_same_object(annotations[0], suitable)

        target_a = random.choice(suitable)
        target_b = random.choice(det_b)

        if target_a.get("label") == target_b.get("label"):
            return self._gen_same_object(annotations[0], suitable)

        zoom = random.choice(self.ZOOM_FACTORS)
        label_a = self._get_label(target_a)
        label_b = self._get_label(target_b)

        difficulty = "easy" if zoom <= 5 else ("medium" if zoom <= 10 else "hard")

        template = random.choice(DIFFERENT_OBJECT_TEMPLATES).format(
            label_a=label_a, label_b=label_b, zoom=zoom)

        return self._create_challenge(
            annotations=[annotations[0], annotations[1]],
            difficulty=difficulty,
            sub_type="different_object_zoomed",
            question_template=template,
            correct_answer="No",
            distractors=["Yes", "They could be the same category",
                          "Not enough information to determine"],
            metadata={
                "object_a_label": target_a["label"],
                "object_b_label": target_b["label"],
                "zoom_factor": zoom,
            },
        )

    def _gen_size_compare(self, ann, suitable):
        """Are these objects the same real-world size? Always No for zoomed."""
        target = random.choice(suitable)
        zoom = random.choice(self.ZOOM_FACTORS)
        label = self._get_label(target)

        difficulty = "easy" if zoom >= 10 else ("medium" if zoom >= 5 else "hard")

        template = random.choice(SIZE_COMPARE_TEMPLATES).format(
            label=label, zoom=zoom)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="size_comparison",
            question_template=template,
            correct_answer="No",
            distractors=["Yes", "The zoom makes it appear larger but it might be",
                          "Size cannot be determined from these images"],
            metadata={
                "object_label": target["label"],
                "zoom_factor": zoom,
            },
        )

    def _gen_scale_identify(self, ann, suitable):
        """Identify an object shown at extreme zoom."""
        target = random.choice(suitable)
        zoom = random.choice(self.ZOOM_FACTORS)
        label = self._get_label(target)

        difficulty = "easy" if zoom <= 5 else ("medium" if zoom <= 10 else "hard")

        template = random.choice(SCALE_IDENTIFY_TEMPLATES).format(zoom=zoom)

        # Build distractors from other detections + plausible extras
        all_labels = list(set(d["label"] for d in suitable if d["label"] != target["label"]))
        extras = ["furniture", "appliance", "tool", "container"]
        pool = [l for l in all_labels if l != "object"] + [e for e in extras if e != label]
        random.shuffle(pool)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="scale_identify",
            question_template=template,
            correct_answer=label,
            distractors=pool[:3] or ["unknown item", "blurred artifact", "background element"],
            metadata={
                "object_label": target["label"],
                "zoom_factor": zoom,
            },
        )

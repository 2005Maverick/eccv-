"""
Texture bias challenge generator.
Tests VLM reliance on texture cues for object recognition.

ECCV-level: Balanced Yes/No, diverse templates (30+), always 3 distractors.
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

SAME_OBJECT_TEMPLATES = [
    "Image A shows a photograph and Image B shows its silhouette. Do both depict the same type of {obj}? Answer Yes or No.",
    "A photo of a {obj} is paired with its edge-detected outline. Are they the same object? Answer Yes or No.",
    "One image is a full-color photo, the other is an outline/silhouette derived from it. Do they show the same {obj}? Answer Yes or No.",
    "Compare the textured image with its silhouette version. Is the same {obj} depicted in both? Answer Yes or No.",
    "The silhouette was extracted from the photograph. Does it still represent the same {obj}? Answer Yes or No.",
    "After stripping all color and texture, is the resulting silhouette still identifiable as a {obj}? Answer Yes or No.",
]

CROSS_OBJECT_TEMPLATES = [
    "The silhouette is from one image, while the photograph is from a different scene. Do they show the same object type? Answer Yes or No.",
    "Image A's silhouette and Image B's photograph come from different sources. Are they the same kind of object? Answer Yes or No.",
    "This silhouette was extracted from a different image than the photo shown. Do they depict the same object? Answer Yes or No.",
    "A silhouette from one scene is paired with a photo from another. Is the object category the same in both? Answer Yes or No.",
    "These images come from different photographs: one silhouette, one full-color. Do they represent the same type of object? Answer Yes or No.",
]

SHAPE_RECOGNITION_TEMPLATES = [
    "Can you identify the {obj} from its silhouette alone, without any color or texture cues? Answer Yes or No.",
    "Looking only at the shape outline, is this {obj} still recognizable? Answer Yes or No.",
    "Without texture or color information, is the silhouette of this {obj} distinctive enough to identify? Answer Yes or No.",
    "If all texture were removed, would the contour alone reveal this is a {obj}? Answer Yes or No.",
    "Based purely on the edge outline, can you determine that this is a {obj}? Answer Yes or No.",
    "Does the shape of this {obj}'s silhouette provide enough information for identification? Answer Yes or No.",
    "Stripped of all surface detail, is the {obj}'s outline unique enough to recognize? Answer Yes or No.",
]

TEXTURE_IMPORTANCE_TEMPLATES = [
    "Is texture more important than shape for recognizing the {obj} in this image? Answer Yes or No.",
    "Would removing all texture (keeping only edges) make the {obj} significantly harder to identify? Answer Yes or No.",
    "For this particular {obj}, does surface texture provide more diagnostic information than overall shape? Answer Yes or No.",
    "Is the {obj}'s identity more dependent on its surface patterns than its contour? Answer Yes or No.",
    "When identifying this {obj}, is the texture (color, patterns, markings) more informative than the silhouette? Answer Yes or No.",
    "If you could only use one cue — shape or texture — to identify this {obj}, would texture be more useful? Answer Yes or No.",
]

SHAPE_VS_TEXTURE_TEMPLATES = [
    "This image has been converted to a silhouette. What type of object does the shape represent?",
    "Looking only at the outline with no color or texture, identify the object shown.",
    "What object is depicted by this contour/edge map?",
    "From the silhouette alone, what kind of {category} is this?",
    "The texture has been completely removed. What object does this shape suggest?",
]


class TextureGenerator(ChallengeGenerator):
    """
    Texture bias generator — ECCV quality with balanced Yes/No.

    Sub-types:
    1. Same-object silhouette: photo vs its silhouette -> "Yes"
    2. Cross-object silhouette: silhouette of A, photo of B -> "No"
    3. Shape recognition (Yes/No): identifiable from silhouette?
    4. Texture importance: is texture more important than shape?
    5. Shape identification: identify object from silhouette alone

    30+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(bias_type="texture", ground_truth_method="silhouette_match")

    def _get_main_obj(self, ann):
        """Get main object label, avoiding generic 'object'."""
        det_info = ann.get("detections", {})
        detections = det_info.get("detections", [])
        for d in detections:
            label = d.get("label", "object")
            if label != "object":
                return label
        return "item" if not detections else detections[0].get("label", "item")

    def generate_challenge(self, annotations: List[Dict],
                           all_annotations: List[Dict] = None) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        texture_info = ann.get("texture", {})

        if texture_info.get("skipped") or texture_info.get("error"):
            return None

        edge_density = texture_info.get("edge_density", 0.0)
        if edge_density < 0.01:
            return None

        sub_type = random.choice([
            "same_object", "cross_object", "shape_yes", "shape_no",
            "texture_importance", "shape_identify",
        ])

        if sub_type == "same_object":
            return self._gen_same_object(ann, edge_density)
        elif sub_type == "cross_object":
            return self._gen_cross_object_single(ann, edge_density, all_annotations)
        elif sub_type == "shape_yes":
            return self._gen_shape_recognition(ann, edge_density, answer_yes=True)
        elif sub_type == "shape_no":
            return self._gen_shape_recognition(ann, edge_density, answer_yes=False)
        elif sub_type == "shape_identify":
            return self._gen_shape_identify(ann, edge_density)
        else:
            return self._gen_texture_importance(ann, edge_density)

    def _gen_same_object(self, ann, edge_density):
        """Photo vs its own silhouette -> same object = 'Yes'."""
        difficulty = "easy" if edge_density > 0.15 else ("medium" if edge_density > 0.05 else "hard")
        main_obj = self._get_main_obj(ann)
        template = random.choice(SAME_OBJECT_TEMPLATES).format(obj=main_obj)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="same_object_silhouette",
            question_template=template,
            correct_answer="Yes",
            distractors=["No", "The silhouette is too abstract to match",
                          "Cannot determine without color"],
            metadata={
                "edge_density": edge_density,
                "transform": "silhouette",
            },
        )

    def _gen_cross_object_single(self, ann, edge_density, all_annotations=None):
        """Silhouette of this image vs a DIFFERENT image -> 'No'."""
        difficulty = "easy" if edge_density > 0.15 else ("medium" if edge_density > 0.05 else "hard")
        image_id = ann.get("image_id", "unknown")

        other_id = image_id
        if all_annotations:
            others = [a for a in all_annotations if a.get("image_id") != image_id]
            if others:
                other_id = random.choice(others).get("image_id", image_id)

        if other_id == image_id:
            return self._gen_shape_recognition(ann, edge_density, answer_yes=False)

        template = random.choice(CROSS_OBJECT_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="cross_object_silhouette",
            question_template=template,
            correct_answer="No",
            distractors=["Yes",
                                "They appear similar but are different",
                                "Cannot compare without seeing both in full color"],
            metadata={
                "edge_density": edge_density,
                "transform": "silhouette_cross",
            },
        )

    def _gen_shape_recognition(self, ann, edge_density, answer_yes=True):
        """Can you identify the object from its silhouette alone?"""
        main_obj = self._get_main_obj(ann)

        if answer_yes:
            correct = "Yes"
            difficulty = "easy" if edge_density > 0.15 else ("medium" if edge_density > 0.05 else "hard")
        else:
            correct = "No"
            difficulty = "easy" if edge_density < 0.03 else ("medium" if edge_density < 0.08 else "hard")

        template = random.choice(SHAPE_RECOGNITION_TEMPLATES).format(obj=main_obj)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="shape_recognition",
            question_template=template,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                          "Only with additional context",
                          "The shape is somewhat recognizable"],
            metadata={
                "edge_density": edge_density,
                "main_object": main_obj,
                "answer_rationale": "high_edge_density" if answer_yes else "low_edge_density",
            },
        )

    def _gen_texture_importance(self, ann, edge_density):
        """Is texture more important than shape for recognizing this object?"""
        main_obj = self._get_main_obj(ann)

        texture_objects = {"cat", "dog", "bird", "zebra", "giraffe", "bear",
                          "horse", "cow", "sheep", "elephant"}
        shape_objects = {"car", "truck", "bus", "bicycle", "motorcycle",
                        "airplane", "boat", "chair", "bench", "clock",
                        "umbrella", "bottle", "cup"}

        if main_obj in texture_objects:
            correct = "Yes"
            difficulty = "easy"
        elif main_obj in shape_objects:
            correct = "No"
            difficulty = "easy"
        elif edge_density > 0.10:
            correct = "No"
            difficulty = "medium"
        else:
            correct = "Yes"
            difficulty = "medium"

        template = random.choice(TEXTURE_IMPORTANCE_TEMPLATES).format(obj=main_obj)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="texture_importance",
            question_template=template,
            correct_answer=correct,
            distractors=["No" if correct == "Yes" else "Yes",
                          "Both cues are equally vital",
                          "Depends on the specific category of object"],
            metadata={
                "object": main_obj,
                "edge_density": edge_density,
                "is_texture_dominant": correct == "Yes",
            },
        )

    def _gen_shape_identify(self, ann, edge_density):
        """Identify the object from its silhouette alone."""
        main_obj = self._get_main_obj(ann)
        if main_obj in ("item", "object"):
            return self._gen_texture_importance(ann, edge_density)

        difficulty = "easy" if edge_density > 0.15 else ("medium" if edge_density > 0.08 else "hard")

        det_info = ann.get("detections", {})
        detections = det_info.get("detections", [])
        other_labels = list(set(d["label"] for d in detections if d["label"] != main_obj and d["label"] != "object"))
        extras = ["building", "appliance", "tool", "instrument", "furniture"]
        pool = other_labels + [e for e in extras if e != main_obj]
        random.shuffle(pool)

        template = random.choice(SHAPE_VS_TEXTURE_TEMPLATES).format(
            obj=main_obj, category="object")

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="shape_identify",
            question_template=template,
            correct_answer=main_obj,
            distractors=pool[:3] or ["unknown shape", "abstract form", "unidentifiable"],
            metadata={
                "object": main_obj,
                "edge_density": edge_density,
            },
        )

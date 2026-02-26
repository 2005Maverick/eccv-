"""
Typography-visual conflict bias generator — ECCV v3.
Tests whether VLMs trust text labels or visual content.

Creates challenges where overlaid text contradicts the visual content.
Uses counterfactual images with programmatically added text labels.

ECCV-level: Diverse templates (30+), always 3 distractors.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# Common YOLO labels and plausible wrong alternatives
LABEL_CONFLICTS = {
    "car": ["truck", "bus", "bicycle", "boat"],
    "person": ["mannequin", "statue", "robot", "doll"],
    "dog": ["cat", "wolf", "fox", "bear"],
    "cat": ["dog", "rabbit", "raccoon", "squirrel"],
    "bird": ["bat", "butterfly", "airplane", "drone"],
    "truck": ["car", "van", "tank", "tractor"],
    "bus": ["train", "truck", "van", "tram"],
    "bicycle": ["motorcycle", "scooter", "skateboard", "unicycle"],
    "horse": ["cow", "donkey", "deer", "goat"],
    "cow": ["horse", "bull", "buffalo", "sheep"],
    "chair": ["stool", "bench", "couch", "throne"],
    "bottle": ["can", "jar", "flask", "pitcher"],
    "tree": ["pole", "tower", "antenna", "pillar"],
    "sign": ["billboard", "poster", "banner", "screen"],
    "building": ["mountain", "cliff", "wall", "monument"],
    "animal": ["toy", "stuffed_animal", "robot", "statue"],
    "vehicle": ["container", "machine", "engine", "structure"],
    "object": ["artifact", "sculpture", "tool", "device"],
}

DEFAULT_CONFLICTS = ["unknown", "unidentified", "different_object", "not_this"]

# ============================================================
# Diverse Question Template Pools
# ============================================================

WRONG_LABEL_TEMPLATES = [
    'The text label says "{wrong}". What object is actually shown in the image?',
    'A label reading "{wrong}" is overlaid on this image. What is the real object depicted?',
    'Someone placed a text label "{wrong}" on this image. What does the image actually show?',
    'The overlaid text reads "{wrong}" — but what is the true identity of the object?',
    'Ignore the text label "{wrong}". What object is truly visible in the image?',
    'The image has a text overlay saying "{wrong}". Does this match the actual object? If not, what is it?',
    'A misleading label "{wrong}" has been added. What is the real object in the image?',
]

CORRECT_LABEL_TEMPLATES = [
    'The text label says "{label}". Does the text label correctly identify the object? Answer Yes or No.',
    'An overlay reads "{label}". Is this label accurate for the object shown? Answer Yes or No.',
    'The word "{label}" is written on this image. Does it correctly describe what you see? Answer Yes or No.',
    'A text label identifies this as a "{label}". Is this identification correct? Answer Yes or No.',
    'The overlaid text claims this is a "{label}". Do you agree with this label? Answer Yes or No.',
]

MATCH_TEMPLATES = [
    "Does the text label on this image correctly identify the main object? Answer Yes or No.",
    "Is the overlaid text an accurate description of the object shown? Answer Yes or No.",
    "Does the written label match what you actually see in the image? Answer Yes or No.",
    "Is the text overlay a truthful identification of the depicted object? Answer Yes or No.",
    "Comparing the text label to the visual content, do they agree? Answer Yes or No.",
]

TRUST_TEMPLATES = [
    'The image shows a {true_label}, but the text says "{wrong}". Should you trust the visual content or the text label?',
    'There is a conflict: the object appears to be a {true_label}, but text says "{wrong}". Which is more reliable — the visual or the text?',
    'Visual analysis suggests this is a {true_label}; the text claims it is a {wrong}. Which should be trusted?',
    'A {true_label} is visible, yet the label reads "{wrong}". What takes priority — the image content or the text?',
]


def create_typography_overlay(img_path: str, label_text: str,
                              output_path: str,
                              font_size: int = 36,
                              position: str = "center") -> bool:
    """Overlay a text label on an image using PIL."""
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (IOError, OSError):
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        w, h = img.size

        if position == "center":
            x = (w - text_w) // 2
            y = (h - text_h) // 2
        elif position == "top":
            x = (w - text_w) // 2
            y = 20
        else:
            x = (w - text_w) // 2
            y = h - text_h - 20

        padding = 8
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        img.save(output_path, quality=92)
        return True
    except Exception as e:
        logger.warning(f"Typography overlay failed: {e}")
        return False


class TypographyConflictGenerator(ChallengeGenerator):
    """
    Typography-visual conflict bias generator — ECCV v3.

    Sub-types:
    1. wrong_label: Text says "cat" but image shows a dog → identify real object
    2. correct_label: Text correctly identifies the object (control)
    3. label_match: "Does the text label match the object?"
    4. trust_conflict: Text vs visual — which should you trust?

    30+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="typography_conflict",
            ground_truth_method="yolo_vs_text"
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

        sub = random.choice(["wrong_label", "wrong_label",
                             "correct_label", "label_match",
                             "trust_conflict"])

        if sub == "wrong_label":
            return self._gen_wrong_label(ann, det, true_label, image_dir, cf_dir)
        elif sub == "correct_label":
            return self._gen_correct_label(ann, det, true_label, image_dir, cf_dir)
        elif sub == "trust_conflict":
            return self._gen_trust_conflict(ann, det, true_label, image_dir, cf_dir)
        else:
            return self._gen_label_match(ann, det, true_label, image_dir, cf_dir)

    def _get_wrong_label(self, true_label: str) -> str:
        conflicts = LABEL_CONFLICTS.get(true_label, DEFAULT_CONFLICTS)
        return random.choice(conflicts)

    def _make_overlay(self, ann, label_text, image_dir, cf_dir, suffix, position="bottom"):
        """Helper to create overlay image. Returns cf_id or None."""
        if not (image_dir and cf_dir):
            return None
        img_id = ann.get("image_id", "unknown")
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            src = os.path.join(image_dir, img_id + ext)
            if os.path.exists(src):
                cf_id = f"{img_id}_typo_{suffix}_{random.randint(1000, 9999)}"
                dst = os.path.join(cf_dir, cf_id + ".jpg")
                if create_typography_overlay(src, label_text, dst, position=position):
                    return cf_id
        return None

    def _gen_wrong_label(self, ann, det, true_label, image_dir, cf_dir):
        wrong = self._get_wrong_label(true_label)
        cf_id = self._make_overlay(ann, f"This is a {wrong}", image_dir, cf_dir, "wrong")

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        if wrong in LABEL_CONFLICTS.get(true_label, [])[:1]:
            confound = 0.8
        else:
            confound = 0.4

        wrong_labels = LABEL_CONFLICTS.get(true_label, DEFAULT_CONFLICTS)
        other_wrong = [w for w in wrong_labels if w != wrong][:1]

        template = random.choice(WRONG_LABEL_TEMPLATES).format(wrong=wrong)

        return self._create_challenge(
            annotations=[ann],
            difficulty="hard" if confound > 0.6 else "medium",
            sub_type="wrong_label",
            question_template=template,
            correct_answer=true_label,
            distractors=[wrong,
                                other_wrong[0] if other_wrong else "Cannot determine",
                                "Both labels could be correct"],
            metadata={
                "true_label": det.get("label", "unknown"),
                "text_label": wrong,
                "has_overlay": cf_id is not None,
            },
        )

    def _gen_correct_label(self, ann, det, true_label, image_dir, cf_dir):
        cf_id = self._make_overlay(ann, f"This is a {true_label}", image_dir, cf_dir, "correct")

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        template = random.choice(CORRECT_LABEL_TEMPLATES).format(label=true_label)

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy",
            sub_type="correct_label",
            question_template=template,
            correct_answer="Yes",
            distractors=["No",
                                "The label is partially correct",
                                "Cannot determine from the image"],
            metadata={
                "true_label": det.get("label", "unknown"),
                "text_label": true_label,
                "has_overlay": cf_id is not None,
                "causal_pair": {
                    "original_id": image_id,
                    "counterfactual_id": cf_id or image_id,
                    "transform_type": "typography_overlay",
                    "what_changed": f'Added correct text label "{true_label}"',
                    "what_preserved": "Visual content, object identity",
                },
                "confound_strength": 0.1,
            },
        )

    def _gen_label_match(self, ann, det, true_label, image_dir, cf_dir):
        wrong = self._get_wrong_label(true_label)
        cf_id = self._make_overlay(ann, f"LABEL: {wrong}", image_dir, cf_dir, "match", "top")

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        template = random.choice(MATCH_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty="medium",
            sub_type="label_match",
            question_template=template,
            correct_answer="No",
            distractors=["Yes",
                                "The label is close but not exact",
                                "There is no text label visible"],
            metadata={
                "true_label": det.get("label", "unknown"),
                "text_label": wrong,
                "has_overlay": cf_id is not None,
                "causal_pair": {
                    "original_id": image_id,
                    "counterfactual_id": cf_id or image_id,
                    "transform_type": "typography_overlay",
                    "what_changed": f'Added wrong text label "{wrong}"',
                    "what_preserved": "Visual content, object identity",
                },
                "confound_strength": 0.6,
            },
        )

    def _gen_trust_conflict(self, ann, det, true_label, image_dir, cf_dir):
        """Explicit conflict: text says X, image shows Y — which to trust?"""
        wrong = self._get_wrong_label(true_label)
        cf_id = self._make_overlay(ann, f"This is a {wrong}", image_dir, cf_dir, "trust")

        image_id = ann.get("image_id", "unknown")
        display_id = cf_id or image_id

        template = random.choice(TRUST_TEMPLATES).format(
            true_label=true_label, wrong=wrong)

        return self._create_challenge(
            annotations=[ann],
            difficulty="hard",
            sub_type="trust_conflict",
            question_template=template,
            correct_answer="The visual content (it is a " + true_label + ")",
            distractors=["The text label (it is a " + wrong + ")",
                                "Both are equally reliable",
                                "Neither — additional context is needed"],
            metadata={
                "true_label": det.get("label", "unknown"),
                "text_label": wrong,
                "has_overlay": cf_id is not None,
            },
        )

"""
Temporal reasoning bias challenge generator.
Uses image brightness/color temperature to estimate time-of-day
and creates ordering or comparison challenges.

ECCV-level: Works without EXIF data by analyzing pixel statistics.
Diverse question templates (20+), proper 3-distractor MCQ format.
"""

import logging
import random
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

TIME_ORDER = ["night", "dawn", "morning", "noon", "afternoon", "dusk"]


def estimate_time_of_day(img_path: str) -> dict:
    """Estimate time of day from image brightness and color temperature."""
    img = cv2.imread(img_path)
    if img is None:
        return {"time_of_day": None, "brightness": 0, "warmth": 0}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2])) / 255.0

    b_mean = float(np.mean(img[:, :, 0]))
    r_mean = float(np.mean(img[:, :, 2]))
    warmth = (r_mean - b_mean) / 255.0

    h, w = img.shape[:2]
    sky = img[:h // 5, :, :]
    sky_brightness = float(np.mean(cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY))) / 255.0

    if brightness < 0.18:
        tod = "night"
    elif brightness < 0.30 and warmth > 0.05:
        tod = "dawn"
    elif brightness < 0.35:
        tod = "dawn" if warmth > 0 else "night"
    elif brightness > 0.60 and warmth < 0.02:
        tod = "noon"
    elif brightness > 0.50:
        tod = "morning" if warmth < 0.05 else "afternoon"
    elif warmth > 0.08:
        tod = "dusk"
    else:
        tod = "afternoon"

    return {
        "time_of_day": tod,
        "brightness": round(brightness, 4),
        "warmth": round(warmth, 4),
        "sky_brightness": round(sky_brightness, 4),
    }


# ============================================================
# Diverse Question Template Pools
# ============================================================

BRIGHTNESS_COMPARE_TEMPLATES = [
    "Which image appears to be taken at a brighter time of day? Answer A or B.",
    "One of these images was captured under stronger ambient light. Which one — A or B?",
    "Compare the overall illumination in these two images. Which is brighter? Answer A or B.",
    "Between Image A and Image B, which scene shows more natural daylight? Answer A or B.",
    "Looking at the average luminosity, which image appears more well-lit? Answer A or B.",
    "Which photograph seems to have been taken during a brighter period of the day — A or B?",
    "If you rank these images by ambient brightness, which ranks higher? Answer A or B.",
    "Judging by the lighting conditions alone, which image was taken in a brighter environment — A or B?",
    "Which image shows a scene that is better illuminated by natural light? Answer A or B.",
    "Estimate the relative brightness: which of these two images has higher overall luminance? Answer A or B.",
    "Which image appears to be from a time of day with more sunlight? Answer A or B.",
    "Compare the exposure levels of these two images. Which appears visually brighter? Answer A or B.",
]

TIME_CLASSIFY_TEMPLATES = [
    "What time of day does this image appear to show? Choose from: {options}.",
    "Based on the lighting and color temperature, what period of the day is depicted? Choose from: {options}.",
    "Estimate the likely time of day when this photo was taken. Options: {options}.",
    "Judging by the sky brightness and color warmth, when was this image most likely captured? Choose from: {options}.",
    "The lighting in this scene suggests what approximate time of day? Choose from: {options}.",
    "Looking at the ambient light and shadows, what time of day is this scene from? Options: {options}.",
    "Based on visual cues such as brightness and color tone, identify the time of day. Options: {options}.",
    "This photograph's lighting suggests it was taken during which part of the day? Choose from: {options}.",
    "Analyze the natural lighting in this image. What time of day does it appear to be? Options: {options}.",
    "From the luminance and color balance, estimate when this image was captured. Choose from: {options}.",
    "What approximate period of the day does the lighting in this image indicate? Options: {options}.",
    "Given the overall brightness and warm/cool tones, classify the time of day. Choose from: {options}.",
]

BRIGHTNESS_RANK_TEMPLATES = [
    "Which image appears darker overall — A or B?",
    "One of these images has notably lower brightness. Which one? Answer A or B.",
    "Comparing these two scenes, which has less ambient light? Answer A or B.",
    "Which photograph was taken under dimmer lighting conditions — A or B?",
    "Between A and B, which image looks like it was captured in lower light? Answer A or B.",
    "Which image appears to come from a time of day with less sunlight — A or B?",
]

WARMTH_COMPARE_TEMPLATES = [
    "Which image has a warmer color tone (more orange/golden)? Answer A or B.",
    "Comparing the color temperature, which image appears warmer — A or B?",
    "One image has a distinctly golden/warm cast while the other is cooler. Which is warmer? Answer A or B.",
    "Which photograph shows a color palette more suggestive of golden hour? Answer A or B.",
    "Between Image A and Image B, which has a stronger warm-toned color cast? Answer A or B.",
    "Looking at the red-to-blue ratio, which image appears to have warmer lighting — A or B?",
]

SKY_BRIGHTNESS_TEMPLATES = [
    "Looking at the sky region, which image has a brighter sky — A or B?",
    "Compare the upper portions of these images. Which has a more luminous sky area? Answer A or B.",
    "Which image's sky region appears brighter, suggesting more daylight? Answer A or B.",
    "Between A and B, which has a brighter upper canopy or sky zone? Answer A or B.",
]


class TemporalGenerator(ChallengeGenerator):
    """
    Temporal reasoning bias generator — ECCV quality.

    Sub-types:
    1. Brightness comparison: which image was taken at a brighter time of day?
    2. Time-of-day classification: what time of day does this image show?
    3. Darkness comparison: which image is darker?
    4. Warmth comparison: which image has warmer color tones?
    5. Sky brightness: which image has a brighter sky?

    Uses 40+ diverse question templates. Always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="temporal_reasoning",
            ground_truth_method="brightness_analysis"
        )

    def generate_challenge(self, annotations: List[Dict],
                           image_dir: str = None) -> Optional[Challenge]:
        if not annotations:
            return None

        if len(annotations) >= 2:
            sub = random.choice([
                "brightness_compare", "brightness_compare",
                "darkness_compare", "warmth_compare",
                "sky_brightness", "time_classify",
            ])
        else:
            sub = "time_classify"

        if sub == "brightness_compare":
            return self._gen_brightness_compare(annotations[:2], image_dir)
        elif sub == "darkness_compare":
            return self._gen_darkness_compare(annotations[:2], image_dir)
        elif sub == "warmth_compare":
            return self._gen_warmth_compare(annotations[:2], image_dir)
        elif sub == "sky_brightness":
            return self._gen_sky_brightness(annotations[:2], image_dir)
        else:
            return self._gen_time_classify(annotations[0], image_dir)

    def _get_time_info(self, ann, image_dir):
        """Get time-of-day info, computing from image if needed."""
        temporal = ann.get("temporal", {})
        if temporal.get("time_of_day") and temporal["time_of_day"] != "unknown":
            return temporal

        if image_dir:
            import os
            img_id = ann.get("image_id", "")
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                path = os.path.join(image_dir, img_id + ext)
                if os.path.exists(path):
                    return estimate_time_of_day(path)

        depth = ann.get("depth", {})
        avg_bright = sum(depth.get(k, 0.5) for k in
                        ["left_depth_mean", "right_depth_mean",
                         "top_depth_mean", "bottom_depth_mean"]) / 4.0
        if avg_bright < 0.2:
            tod = "night"
        elif avg_bright < 0.35:
            tod = "dawn"
        elif avg_bright > 0.6:
            tod = "noon"
        else:
            tod = "afternoon"
        return {"time_of_day": tod, "brightness": avg_bright, "warmth": 0, "sky_brightness": avg_bright}

    def _gen_brightness_compare(self, anns, image_dir):
        """Compare two images by brightness."""
        info_a = self._get_time_info(anns[0], image_dir)
        info_b = self._get_time_info(anns[1], image_dir)

        bright_a = info_a.get("brightness", 0.5)
        bright_b = info_b.get("brightness", 0.5)

        if abs(bright_a - bright_b) < 0.05:
            return None

        correct = "A" if bright_a > bright_b else "B"
        diff = abs(bright_a - bright_b)
        difficulty = "easy" if diff > 0.25 else ("medium" if diff > 0.10 else "hard")

        template = random.choice(BRIGHTNESS_COMPARE_TEMPLATES)

        return self._create_challenge(
            annotations=[anns[0], anns[1]],
            difficulty=difficulty,
            sub_type="brightness_compare",
            question_template=template,
            correct_answer=correct,
            distractors=["B" if correct == "A" else "A",
                                "Both are equally bright",
                                "Cannot determine from the images"],
            metadata={
                "brightness_a": bright_a,
                "brightness_b": bright_b,
                "brightness_diff": round(diff, 4),
            },
        )

    def _gen_darkness_compare(self, anns, image_dir):
        """Compare two images: which is darker?"""
        info_a = self._get_time_info(anns[0], image_dir)
        info_b = self._get_time_info(anns[1], image_dir)

        bright_a = info_a.get("brightness", 0.5)
        bright_b = info_b.get("brightness", 0.5)

        if abs(bright_a - bright_b) < 0.05:
            return None

        # Darker = lower brightness
        correct = "A" if bright_a < bright_b else "B"
        diff = abs(bright_a - bright_b)
        difficulty = "easy" if diff > 0.25 else ("medium" if diff > 0.10 else "hard")

        template = random.choice(BRIGHTNESS_RANK_TEMPLATES)

        return self._create_challenge(
            annotations=[anns[0], anns[1]],
            difficulty=difficulty,
            sub_type="darkness_compare",
            question_template=template,
            correct_answer=correct,
            distractors=["B" if correct == "A" else "A",
                                "Both are equally dark",
                                "Cannot determine from the images"],
            metadata={
                "darkness_a": round(1.0 - bright_a, 4),
                "darkness_b": round(1.0 - bright_b, 4),
            },
        )

    def _gen_warmth_compare(self, anns, image_dir):
        """Compare color temperature between two images."""
        info_a = self._get_time_info(anns[0], image_dir)
        info_b = self._get_time_info(anns[1], image_dir)

        warmth_a = info_a.get("warmth", 0)
        warmth_b = info_b.get("warmth", 0)

        if abs(warmth_a - warmth_b) < 0.02:
            return None

        correct = "A" if warmth_a > warmth_b else "B"
        diff = abs(warmth_a - warmth_b)
        difficulty = "easy" if diff > 0.10 else ("medium" if diff > 0.04 else "hard")

        template = random.choice(WARMTH_COMPARE_TEMPLATES)

        return self._create_challenge(
            annotations=[anns[0], anns[1]],
            difficulty=difficulty,
            sub_type="warmth_compare",
            question_template=template,
            correct_answer=correct,
            distractors=["B" if correct == "A" else "A",
                                "Both have the same color temperature",
                                "Lighting makes comparison impossible"],
            metadata={
                "warmth_a": round(warmth_a, 4),
                "warmth_b": round(warmth_b, 4),
                "warmth_diff": round(diff, 4),
            },
        )

    def _gen_sky_brightness(self, anns, image_dir):
        """Compare sky brightness between two images."""
        info_a = self._get_time_info(anns[0], image_dir)
        info_b = self._get_time_info(anns[1], image_dir)

        sky_a = info_a.get("sky_brightness", info_a.get("brightness", 0.5))
        sky_b = info_b.get("sky_brightness", info_b.get("brightness", 0.5))

        if abs(sky_a - sky_b) < 0.05:
            return None

        correct = "A" if sky_a > sky_b else "B"
        diff = abs(sky_a - sky_b)
        difficulty = "easy" if diff > 0.20 else ("medium" if diff > 0.08 else "hard")

        template = random.choice(SKY_BRIGHTNESS_TEMPLATES)

        return self._create_challenge(
            annotations=[anns[0], anns[1]],
            difficulty=difficulty,
            sub_type="sky_brightness",
            question_template=template,
            correct_answer=correct,
            distractors=["B" if correct == "A" else "A",
                                "The sky is not visible in both",
                                "Both skies have identical brightness"],
            metadata={
                "sky_a": round(sky_a, 4),
                "sky_b": round(sky_b, 4),
            },
        )

    def _gen_time_classify(self, ann, image_dir):
        """Classify a single image's time of day."""
        info = self._get_time_info(ann, image_dir)
        tod = info.get("time_of_day", "unknown")
        if tod == "unknown" or tod is None:
            return None

        if tod in ("night", "noon"):
            difficulty = "easy"
        elif tod in ("dawn", "dusk"):
            difficulty = "hard"
        else:
            difficulty = "medium"

        all_times = ["night", "dawn", "morning", "noon", "afternoon", "dusk"]
        distractors = [t for t in all_times if t != tod]
        random.shuffle(distractors)

        options = ", ".join(all_times)
        template = random.choice(TIME_CLASSIFY_TEMPLATES).format(options=options)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="time_classify",
            question_template=template,
            correct_answer=tod.capitalize(),
            distractors=distractors[:3],
            metadata={
                "time_of_day": tod,
                "brightness": round(info.get("brightness", 0), 4),
                "warmth": round(info.get("warmth", 0), 4),
            },
        )

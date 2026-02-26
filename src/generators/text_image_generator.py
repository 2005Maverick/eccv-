"""
Text-in-image bias challenge generator.
Tests VLM ability to read and reason about visible text.

ECCV-level: Diverse templates (25+), always 3 distractors, multiple sub-types.
"""

import logging
import random
from typing import List, Dict, Any, Optional

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Diverse Question Template Pools
# ============================================================

TEXT_PRESENCE_YES_TEMPLATES = [
    "Does this image contain any readable text? Answer Yes or No.",
    "Is there visible text anywhere in this photograph? Answer Yes or No.",
    "Can you detect any written words or characters in this image? Answer Yes or No.",
    "Are there any text elements (signs, labels, logos) visible in this image? Answer Yes or No.",
    "Does this scene include any readable written content? Answer Yes or No.",
    "Is any typographic content visible in this photograph? Answer Yes or No.",
]

TEXT_PRESENCE_NO_TEMPLATES = [
    "Does this image contain any readable text? Answer Yes or No.",
    "Is there any visible text or writing in this photograph? Answer Yes or No.",
    "Can you identify any written words in this image? Answer Yes or No.",
    "Are there any text elements present in this scene? Answer Yes or No.",
    "Does this photograph include any readable characters or words? Answer Yes or No.",
]

TEXT_READ_TEMPLATES = [
    "What text is visible in this image?",
    "Read the text content shown in this photograph.",
    "What words or characters can you see written in this image?",
    "Identify the visible text in this scene.",
    "What does the text in this image say?",
    "Transcribe the readable text from this photograph.",
    "What written content is displayed in this image?",
    "Read aloud the text visible in this picture.",
]

TEXT_COMPARE_TEMPLATES = [
    "Do both images contain the same text? Answer Yes or No.",
    "Is the text in Image A identical to the text in Image B? Answer Yes or No.",
    "Comparing the visible text in both images, do they match? Answer Yes or No.",
    "Are the written contents in these two images the same? Answer Yes or No.",
    "Do these two photographs display identical text? Answer Yes or No.",
]

TEXT_PRESENCE_PAIR_TEMPLATES = [
    "Which image contains readable text? Answer A, B, or Neither.",
    "In which of these images can you find visible text? Answer A, B, or Neither.",
    "One of these images has text in it. Which one — A, B, or Neither?",
    "Identify which image contains written words. Answer A, B, or Neither.",
    "Between Image A and Image B, which has visible text content? Answer A, B, or Neither.",
]

TEXT_COUNT_TEMPLATES = [
    "How many separate text regions or blocks are visible in this image?",
    "Count the distinct text elements (signs, labels, etc.) in this photograph.",
    "How many individual pieces of text can you find in this image?",
]

TEXT_LENGTH_TEMPLATES = [
    "Is the text in this image a single word or multiple words? Answer Single or Multiple.",
    "Does the text in this image contain more than one word? Answer Yes or No.",
    "Is the visible text in this image short (1-2 words) or long (3+ words)? Answer Short or Long.",
]


class TextImageGenerator(ChallengeGenerator):
    """
    Text-in-image bias generator — ECCV quality.

    Sub-types:
    1. Text presence (Yes): does this image contain readable text?
    2. Text presence (No): does this image contain text? (for images without text)
    3. Text reading: what text is visible?
    4. Text comparison: do both images have the same text?
    5. Text presence pair: which image has text?
    6. Text count: how many text blocks?
    7. Text length: single word or multiple?

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="text_in_image",
            ground_truth_method="ocr_text"
        )

    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        if not annotations:
            return None

        ann_a = annotations[0]
        ocr_a = ann_a.get("ocr", {})

        if len(annotations) >= 2:
            ann_b = annotations[1]
            ocr_b = ann_b.get("ocr", {})

            if ocr_a.get("has_text") and ocr_b.get("has_text"):
                return self._gen_text_compare(ann_a, ann_b, ocr_a, ocr_b)

            if ocr_a.get("has_text") and not ocr_b.get("has_text"):
                return self._gen_text_presence_pair(ann_a, ann_b, has_text_is_a=True)
            if ocr_b.get("has_text") and not ocr_a.get("has_text"):
                return self._gen_text_presence_pair(ann_b, ann_a, has_text_is_a=True)

        if ocr_a.get("has_text"):
            sub = random.choice(["text_read", "text_presence",
                                  "text_count", "text_length"])
            if sub == "text_read":
                return self._gen_text_read(ann_a, ocr_a)
            elif sub == "text_count":
                return self._gen_text_count(ann_a, ocr_a)
            elif sub == "text_length":
                return self._gen_text_length(ann_a, ocr_a)
            else:
                return self._gen_text_presence_single(ann_a, ocr_a)

        return self._gen_no_text(ann_a)

    def _extract_text(self, ocr):
        blocks = ocr.get("text_blocks", [])
        if not blocks:
            return ""
        texts = []
        for b in blocks:
            if isinstance(b, dict):
                texts.append(b.get("text", ""))
            elif isinstance(b, str):
                texts.append(b)
        return " ".join(t.strip() for t in texts if t.strip())

    def _gen_text_compare(self, ann_a, ann_b, ocr_a, ocr_b):
        text_a = self._extract_text(ocr_a)
        text_b = self._extract_text(ocr_b)
        if not text_a or not text_b:
            return None

        same = text_a.strip().lower() == text_b.strip().lower()
        correct = "Yes" if same else "No"

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        overlap = words_a & words_b
        total = words_a | words_b
        similarity = len(overlap) / max(len(total), 1)

        if same:
            difficulty = "easy"
        elif similarity < 0.1:
            difficulty = "easy"
        elif similarity < 0.3:
            difficulty = "medium"
        else:
            difficulty = "hard"

        template = random.choice(TEXT_COMPARE_TEMPLATES)

        return self._create_challenge(
            annotations=[ann_a, ann_b],
            difficulty=difficulty,
            sub_type="text_comparison",
            question_template=template,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                                "The text is only partially different",
                                "One image is too blurred to read"],
            metadata={
                "text_a": text_a[:100],
                "text_b": text_b[:100],
                "text_similarity": round(similarity, 3),
            },
        )

    def _gen_text_presence_pair(self, ann_with, ann_without, has_text_is_a=True):
        correct = "A" if has_text_is_a else "B"
        # id_a = ann_with.get("image_id", "unknown") if has_text_is_a else ann_without.get("image_id", "unknown")
        # id_b = ann_without.get("image_id", "unknown") if has_text_is_a else ann_with.get("image_id", "unknown")

        template = random.choice(TEXT_PRESENCE_PAIR_TEMPLATES)

        distractor_a = "A" if correct == "B" else "B"
        distractors = [distractor_a, "Neither", "Both images contain text"]

        return self._create_challenge(
            annotations=[ann_with, ann_without] if has_text_is_a else [ann_without, ann_with],
            difficulty="easy",
            sub_type="text_presence_pair",
            question_template=template,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "has_text_id": ann_with.get("image_id", "unknown"),
            },
        )

    def _gen_text_read(self, ann, ocr):
        text = self._extract_text(ocr)
        if not text or len(text) < 2:
            return self._gen_text_presence_single(ann, ocr)

        text_clean = text[:60].strip()
        words = text_clean.split()
        if len(words) > 8:
            text_clean = " ".join(words[:8])

        distractors = []
        if len(words) >= 2:
            shuffled = words.copy()
            random.shuffle(shuffled)
            distractors.append(" ".join(shuffled[:min(len(shuffled), 5)]))
        distractors.append("No text visible")
        distractors.append("Cannot determine")

        difficulty = "easy" if len(text_clean) > 10 else "medium"
        template = random.choice(TEXT_READ_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="text_read",
            question_template=template,
            correct_answer=text_clean,
            distractors=distractors[:3],
            metadata={
                "full_text": text[:200],
                "word_count": len(words),
            },
        )

    def _gen_text_presence_single(self, ann, ocr):
        n_blocks = len(ocr.get("text_blocks", []))
        difficulty = "easy" if n_blocks > 3 else ("medium" if n_blocks > 1 else "hard")

        template = random.choice(TEXT_PRESENCE_YES_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="text_presence",
            question_template=template,
            correct_answer="Yes",
            distractors=["No",
                                "There are symbols but not readable text",
                                "The image quality is too low to determine"],
            metadata={
                "text_sample": self._extract_text(ocr)[:100],
                "block_count": n_blocks,
            },
        )

    def _gen_no_text(self, ann):
        template = random.choice(TEXT_PRESENCE_NO_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty="medium",
            sub_type="no_text",
            question_template=template,
            correct_answer="No",
            distractors=["Yes",
                                "There might be text that is not clearly visible",
                                "Text is present but illegible"],
            metadata={"sub_type": "no_text"}, # This metadata entry is redundant but kept for consistency if needed elsewhere
        )

    def _gen_text_count(self, ann, ocr):
        """How many text blocks are there?"""
        blocks = ocr.get("text_blocks", [])
        count = len(blocks)
        if count == 0:
            return self._gen_no_text(ann)

        difficulty = "easy" if count <= 2 else ("medium" if count <= 5 else "hard")

        distractors = []
        for offset in [1, -1, 2]:
            wrong = count + offset
            if wrong > 0 and str(wrong) not in distractors:
                distractors.append(str(wrong))
        if len(distractors) < 3:
            distractors.append("0")

        template = random.choice(TEXT_COUNT_TEMPLATES)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="text_count",
            question_template=template,
            correct_answer=str(count),
            distractors=distractors[:3],
            metadata={"block_count": count},
        )

    def _gen_text_length(self, ann, ocr):
        """Is the text short or long?"""
        text = self._extract_text(ocr)
        words = text.split()

        if len(words) <= 2:
            correct = "Short"
        else:
            correct = "Long"

        difficulty = "easy" if len(words) == 1 or len(words) >= 5 else "medium"

        template = random.choice(TEXT_LENGTH_TEMPLATES)

        # Adapt answer format to template
        if "Single or Multiple" in template:
            correct = "Single" if len(words) == 1 else "Multiple"
            distractors = ["Single" if correct == "Multiple" else "Multiple",
                          "Cannot determine from the image",
                          "The text is partially cut off"]
        elif "Yes or No" in template:
            correct = "Yes" if len(words) > 1 else "No"
            distractors = ["Yes" if correct == "No" else "No",
                          "Cannot determine",
                          "The text is fragmented"]
        else:
            distractors = ["Short" if correct == "Long" else "Long",
                          "Cannot determine from the image",
                          "The text length varies across regions"]

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="text_length",
            question_template=template,
            correct_answer=correct,
            distractors=distractors[:3],
            metadata={
                "word_count": len(words),
                "text_sample": text[:100],
            },
        )

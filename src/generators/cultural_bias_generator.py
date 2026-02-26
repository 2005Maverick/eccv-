"""
Cultural visual bias generator — ECCV v3.

Tests whether VLMs exhibit cultural bias in visual interpretation.
Connects to the multilingual angle: the correct answer may depend
on the cultural context of the language being used.

ECCV-level: Diverse templates (25+), always 3 distractors.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional

from src.generators.base_generator import ChallengeGenerator, Challenge

logger = logging.getLogger(__name__)

# ============================================================
# Cultural Knowledge Base
# ============================================================

CULTURAL_ITEMS = [
    {
        "id": "hand_thumbsup",
        "visual_cue": "thumbs-up gesture",
        "interpretations": {
            "en": {"meaning": "approval or good", "sentiment": "positive"},
            "ar": {"meaning": "offensive gesture", "sentiment": "negative"},
            "hi": {"meaning": "approval", "sentiment": "positive"},
            "zh": {"meaning": "number one or good", "sentiment": "positive"},
            "es": {"meaning": "approval", "sentiment": "positive"},
        },
        "category": "gesture",
    },
    {
        "id": "color_white_clothing",
        "visual_cue": "person wearing all white clothes",
        "interpretations": {
            "en": {"meaning": "wedding or purity", "sentiment": "positive"},
            "hi": {"meaning": "mourning or funeral", "sentiment": "negative"},
            "zh": {"meaning": "mourning or funeral", "sentiment": "negative"},
            "ar": {"meaning": "purity or peace", "sentiment": "positive"},
            "es": {"meaning": "wedding or purity", "sentiment": "positive"},
        },
        "category": "clothing",
    },
    {
        "id": "color_red",
        "visual_cue": "red envelope or red decoration",
        "interpretations": {
            "en": {"meaning": "warning or danger", "sentiment": "negative"},
            "zh": {"meaning": "luck and prosperity", "sentiment": "positive"},
            "hi": {"meaning": "auspicious or festive", "sentiment": "positive"},
            "ar": {"meaning": "warning or danger", "sentiment": "negative"},
            "es": {"meaning": "passion or warning", "sentiment": "mixed"},
        },
        "category": "color_symbolism",
    },
    {
        "id": "owl_symbolism",
        "visual_cue": "owl perched on a branch",
        "interpretations": {
            "en": {"meaning": "wisdom and knowledge", "sentiment": "positive"},
            "hi": {"meaning": "bad omen or death", "sentiment": "negative"},
            "ar": {"meaning": "bad omen", "sentiment": "negative"},
            "zh": {"meaning": "death or bad luck", "sentiment": "negative"},
            "es": {"meaning": "wisdom", "sentiment": "positive"},
        },
        "category": "animal_symbolism",
    },
    {
        "id": "number_four",
        "visual_cue": "number 4 displayed prominently",
        "interpretations": {
            "en": {"meaning": "just a number", "sentiment": "neutral"},
            "zh": {"meaning": "unlucky (sounds like death)", "sentiment": "negative"},
            "hi": {"meaning": "just a number", "sentiment": "neutral"},
            "ar": {"meaning": "just a number", "sentiment": "neutral"},
            "es": {"meaning": "just a number", "sentiment": "neutral"},
        },
        "category": "number_symbolism",
    },
    {
        "id": "left_hand_food",
        "visual_cue": "person eating with left hand",
        "interpretations": {
            "en": {"meaning": "normal", "sentiment": "neutral"},
            "ar": {"meaning": "impolite or taboo", "sentiment": "negative"},
            "hi": {"meaning": "impolite or taboo", "sentiment": "negative"},
            "zh": {"meaning": "normal", "sentiment": "neutral"},
            "es": {"meaning": "normal", "sentiment": "neutral"},
        },
        "category": "social_norm",
    },
    {
        "id": "black_cat",
        "visual_cue": "black cat crossing a path",
        "interpretations": {
            "en": {"meaning": "bad luck", "sentiment": "negative"},
            "hi": {"meaning": "bad luck", "sentiment": "negative"},
            "zh": {"meaning": "good luck in some regions", "sentiment": "positive"},
            "ar": {"meaning": "bad omen", "sentiment": "negative"},
            "es": {"meaning": "bad luck", "sentiment": "negative"},
        },
        "category": "superstition",
    },
    {
        "id": "lotus_flower",
        "visual_cue": "lotus flower in water",
        "interpretations": {
            "en": {"meaning": "beauty or exotic flower", "sentiment": "positive"},
            "hi": {"meaning": "purity and divine creation", "sentiment": "very_positive"},
            "zh": {"meaning": "purity rising from mud", "sentiment": "very_positive"},
            "ar": {"meaning": "beauty", "sentiment": "positive"},
            "es": {"meaning": "beauty or exotic flower", "sentiment": "positive"},
        },
        "category": "flower_symbolism",
    },
]

LANG_NAMES = {
    "en": "English/Western",
    "ar": "Arabic/Middle Eastern",
    "hi": "Hindi/South Asian",
    "zh": "Chinese/East Asian",
    "es": "Spanish/Latin",
}

# ============================================================
# Diverse Question Template Pools
# ============================================================

INTERPRETATION_TEMPLATES = [
    "What does this {visual_cue} typically mean in {lang} culture?",
    "How would someone from a {lang} background interpret this {visual_cue}?",
    "In {lang} cultural context, what is the significance of this {visual_cue}?",
    "What meaning does {lang} culture assign to a {visual_cue}?",
    "From a {lang} perspective, what does this {visual_cue} symbolize?",
    "In {lang} tradition, how is a {visual_cue} typically understood?",
    "What cultural interpretation does {lang} culture give to this {visual_cue}?",
]

CONTRAST_TEMPLATES = [
    'Does the visual concept "{visual_cue}" have a different cultural meaning in {l1} vs {l2} culture? Answer Yes or No.',
    'Would a person from {l1} and a person from {l2} interpret the "{visual_cue}" differently? Answer Yes or No.',
    'Is the cultural significance of a {visual_cue} different between {l1} and {l2} traditions? Answer Yes or No.',
    'Do {l1} and {l2} cultures assign different meanings to a {visual_cue}? Answer Yes or No.',
    'Would the interpretation of this {visual_cue} vary between {l1} and {l2} cultural contexts? Answer Yes or No.',
]

SENTIMENT_TEMPLATES = [
    'In {lang} culture, is the visual concept of "{visual_cue}" perceived as positive, negative, or neutral?',
    'How would {lang} culture view a {visual_cue} — positively, negatively, or neutrally?',
    'What sentiment does {lang} culture associate with a {visual_cue}? Answer positive, negative, or neutral.',
    'From a {lang} cultural perspective, is a {visual_cue} considered positive, negative, or neutral?',
    'Would a {visual_cue} evoke a positive, negative, or neutral reaction in {lang} culture?',
]

UNIVERSAL_TEMPLATES = [
    'Is the meaning of a {visual_cue} universal across all cultures? Answer Yes or No.',
    'Would every culture interpret a {visual_cue} the same way? Answer Yes or No.',
    'Does the {visual_cue} carry the same symbolic meaning in all world cultures? Answer Yes or No.',
]


class CulturalBiasGenerator(ChallengeGenerator):
    """
    Cultural visual bias generator — ECCV v3.

    Sub-types:
    1. cultural_interpretation: What does this visual mean in culture X?
    2. cross_cultural_contrast: Same image, different cultures → different answers?
    3. sentiment_prediction: Is this visual positive/negative in culture X?
    4. universality: Is this symbol universal?

    25+ diverse templates, always 3 distractors.
    """

    def __init__(self):
        super().__init__(
            bias_type="cultural_visual_bias",
            ground_truth_method="cultural_knowledge_base"
        )

    def generate_challenge(self, annotations: List[Dict],
                           image_dir: str = None,
                           **kwargs) -> Optional[Challenge]:
        if not annotations:
            return None

        ann = annotations[0]
        image_id = ann.get("image_id", "unknown")

        sub = random.choice(["cultural_interpretation", "cultural_interpretation",
                             "cross_cultural_contrast", "sentiment_prediction",
                             "universality"])

        item = random.choice(CULTURAL_ITEMS)

        if sub == "cultural_interpretation":
            return self._gen_interpretation(ann, item, image_id)
        elif sub == "cross_cultural_contrast":
            return self._gen_contrast(ann, item, image_id)
        elif sub == "universality":
            return self._gen_universality(ann, item, image_id)
        else:
            return self._gen_sentiment(ann, item, image_id)

    def _gen_interpretation(self, ann, item, image_id):
        lang = random.choice(list(item["interpretations"].keys()))
        interp = item["interpretations"][lang]
        lang_name = LANG_NAMES.get(lang, lang)
        correct = interp["meaning"]

        other_meanings = [
            item["interpretations"][l]["meaning"]
            for l in item["interpretations"]
            if l != lang and item["interpretations"][l]["meaning"] != correct
        ]
        distractors = list(set(other_meanings))
        if len(distractors) < 3:
            extras = ["No specific meaning", "Universal symbol", "Decoration only"]
            distractors += [e for e in extras if e != correct]
        distractors = distractors[:3]

        en_meaning = item["interpretations"].get("en", {}).get("meaning", "")
        is_different = (correct != en_meaning)
        difficulty = "hard" if is_different else "easy"

        template = random.choice(INTERPRETATION_TEMPLATES).format(
            visual_cue=item["visual_cue"], lang=lang_name)

        return self._create_challenge(
            annotations=[ann],
            difficulty=difficulty,
            sub_type="cultural_interpretation",
            question_template=template,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "cultural_item": item["id"],
                "visual_cue": item["visual_cue"],
                "target_language": lang,
                "target_culture": lang_name,
                "category": item["category"],
                "language_conditional": True,
            },
        )

    def _gen_contrast(self, ann, item, image_id):
        langs = list(item["interpretations"].keys())
        if len(langs) < 2:
            return None

        l1, l2 = random.sample(langs, 2)
        m1 = item["interpretations"][l1]["meaning"]
        m2 = item["interpretations"][l2]["meaning"]
        same = (m1 == m2)
        correct = "No" if same else "Yes"

        template = random.choice(CONTRAST_TEMPLATES).format(
            visual_cue=item["visual_cue"],
            l1=LANG_NAMES[l1], l2=LANG_NAMES[l2])

        return self._create_challenge(
            annotations=[ann],
            difficulty="medium",
            sub_type="cross_cultural_contrast",
            question_template=template,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                                "All cultures share the same interpretation",
                                "Cultural context does not affect visual meaning"],
            metadata={
                "cultural_item": item["id"],
                "culture_a": l1, "culture_b": l2,
                "meaning_a": m1, "meaning_b": m2,
                "meanings_differ": not same,
            },
        )

    def _gen_sentiment(self, ann, item, image_id):
        lang = random.choice(list(item["interpretations"].keys()))
        interp = item["interpretations"][lang]
        sentiment = interp["sentiment"]
        lang_name = LANG_NAMES.get(lang, lang)

        all_sentiments = ["Positive", "Negative", "Neutral", "Mixed"]

        if sentiment in ["positive", "very_positive"]:
            correct = "Positive"
        elif sentiment == "negative":
            correct = "Negative"
        elif sentiment == "mixed":
            correct = "Mixed"
        else:
            correct = "Neutral"

        distractors = [s for s in all_sentiments if s != correct][:3]

        template = random.choice(SENTIMENT_TEMPLATES).format(
            lang=lang_name, visual_cue=item["visual_cue"])

        return self._create_challenge(
            annotations=[ann],
            difficulty="medium",
            sub_type="sentiment_prediction",
            question_template=template,
            correct_answer=correct,
            distractors=distractors,
            metadata={
                "cultural_item": item["id"],
                "target_language": lang,
                "target_culture": lang_name,
                "sentiment": sentiment,
            },
        )

    def _gen_universality(self, ann, item, image_id):
        """Is this symbol's meaning universal across cultures?"""
        meanings = set(item["interpretations"][l]["meaning"]
                       for l in item["interpretations"])
        is_universal = len(meanings) == 1
        correct = "Yes" if is_universal else "No"

        template = random.choice(UNIVERSAL_TEMPLATES).format(
            visual_cue=item["visual_cue"])

        return self._create_challenge(
            annotations=[ann],
            difficulty="easy" if not is_universal else "hard",
            sub_type="universality",
            question_template=template,
            correct_answer=correct,
            distractors=["Yes" if correct == "No" else "No",
                                "It depends on the specific context",
                                "Most cultures agree but some differ"],
            metadata={
                "cultural_item": item["id"],
                "distinct_meanings": len(meanings),
                "is_universal": is_universal,
            },
        )

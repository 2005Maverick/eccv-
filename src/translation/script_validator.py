"""
Script validator for translated text.
Ensures each translated output belongs to the correct Unicode script.
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Unicode script ranges (simplified)
SCRIPT_RANGES: Dict[str, list] = {
    "Latin": [
        (0x0041, 0x005A),  # A-Z
        (0x0061, 0x007A),  # a-z
        (0x00C0, 0x024F),  # Latin Extended
    ],
    "Devanagari": [
        (0x0900, 0x097F),  # Devanagari
        (0xA8E0, 0xA8FF),  # Devanagari Extended
    ],
    "Arabic": [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0xFB50, 0xFDFF),  # Arabic PF-A
        (0xFE70, 0xFEFF),  # Arabic PF-B
    ],
    "CJK": [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x2E80, 0x2EFF),  # CJK Radicals
        (0x3000, 0x303F),  # CJK Symbols
    ],
}

# Language to expected script mapping
LANG_SCRIPT_MAP: Dict[str, str] = {
    "en": "Latin",
    "es": "Latin",
    "zh": "CJK",
    "hi": "Devanagari",
    "ar": "Arabic",
}


def _char_in_script(char: str, script: str) -> bool:
    """Check if a character belongs to a given script."""
    cp = ord(char)
    for start, end in SCRIPT_RANGES.get(script, []):
        if start <= cp <= end:
            return True
    return False


def validate_script(text: str, lang_code: str, threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Validate that translated text is in the correct script.

    Args:
        text: Translated text to validate.
        lang_code: ISO 639-1 language code.
        threshold: Minimum fraction of script-matching characters (default 0.7).

    Returns:
        (is_valid, script_fraction): bool and fraction of character that match.
    """
    expected_script = LANG_SCRIPT_MAP.get(lang_code)
    if expected_script is None:
        logger.warning(f"Unknown language code: {lang_code}")
        return True, 1.0  # Unknown language, assume valid

    # Filter to alphabetic characters only (ignore numbers, punctuation, spaces)
    alpha_chars = [c for c in text if c.isalpha()]

    if not alpha_chars:
        return True, 1.0  # No alphabetic characters, consider valid

    # Count characters in the expected script
    matching = sum(1 for c in alpha_chars if _char_in_script(c, expected_script))
    fraction = matching / len(alpha_chars)

    is_valid = fraction >= threshold

    if not is_valid:
        logger.warning(
            f"Script validation failed for '{lang_code}': "
            f"{fraction:.1%} {expected_script} (threshold {threshold:.0%})"
        )

    return is_valid, round(fraction, 4)


def validate_variable_slots(template: str, translated: str) -> bool:
    """
    Validate that all {variable} slots survived translation.

    Args:
        template: Original English template.
        translated: Translated template.

    Returns:
        True if all variable slots are preserved.
    """
    original_vars = set(re.findall(r'\{(\w+)\}', template))
    translated_vars = set(re.findall(r'\{(\w+)\}', translated))

    if original_vars != translated_vars:
        missing = original_vars - translated_vars
        extra = translated_vars - original_vars
        if missing:
            logger.warning(f"Variable slots lost in translation: {missing}")
        if extra:
            logger.warning(f"Unexpected variable slots in translation: {extra}")
        return False

    return True

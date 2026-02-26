"""
Translator using Helsinki-NLP/opus-mt models via HuggingFace transformers.
Template-first approach: translate templates → fill variables → validate scripts.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Variable placeholder pattern
VAR_PATTERN = re.compile(r'\{(\w+)\}')

# Sentinel markers to protect variable slots during translation
SENTINEL_PREFIX = "VARSLOT"


class Translator:
    """
    Helsinki-NLP/opus-mt translator with variable slot preservation.

    Template-first approach:
    1. Replace {variable} with VARSLOT_N sentinel tokens
    2. Translate the template text
    3. Restore sentinels back to {variable} in target language
    4. At runtime: fill variables with actual values

    This ensures variables are never translated or corrupted.
    """

    def __init__(self, model_cache_dir: str = "models/"):
        self.model_cache_dir = model_cache_dir
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

    def _load_model(self, model_id: str):
        """Lazy-load translation model."""
        if model_id in self._models:
            return

        try:
            from transformers import MarianTokenizer, MarianMTModel

            logger.info(f"Loading translation model: {model_id}")
            self._tokenizers[model_id] = MarianTokenizer.from_pretrained(
                model_id, cache_dir=self.model_cache_dir
            )
            self._models[model_id] = MarianMTModel.from_pretrained(
                model_id, cache_dir=self.model_cache_dir
            )
            logger.info(f"Loaded {model_id}")
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            raise

    def _protect_variables(self, text: str) -> tuple:
        """
        Replace {variable} slots with sentinel tokens.

        Returns:
            (protected_text, variable_map) where variable_map maps
            sentinel tokens to original variable names.
        """
        variable_map = {}
        counter = 0

        def replace_fn(match):
            nonlocal counter
            var_name = match.group(1)
            sentinel = f"{SENTINEL_PREFIX}{counter}"
            variable_map[sentinel] = var_name
            counter += 1
            return sentinel

        protected = VAR_PATTERN.sub(replace_fn, text)
        return protected, variable_map

    def _restore_variables(self, text: str, variable_map: Dict[str, str]) -> str:
        """Restore sentinel tokens back to {variable} format."""
        restored = text
        for sentinel, var_name in variable_map.items():
            restored = restored.replace(sentinel, f"{{{var_name}}}")
        return restored

    def translate_text(self, text: str, model_id: str) -> str:
        """
        Translate a single text string.

        Args:
            text: Source text (English).
            model_id: Helsinki-NLP model identifier.

        Returns:
            Translated text.
        """
        self._load_model(model_id)

        tokenizer = self._tokenizers[model_id]
        model = self._models[model_id]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return result

    def translate_template(self, template: str, model_id: str) -> str:
        """
        Translate a template while preserving {variable} slots.

        Args:
            template: English template with {variable} placeholders.
            model_id: Helsinki-NLP model identifier.

        Returns:
            Translated template with {variable} slots preserved.
        """
        # Step 1: Protect variables
        protected, variable_map = self._protect_variables(template)

        # Step 2: Translate
        translated = self.translate_text(protected, model_id)

        # Step 3: Restore variables
        result = self._restore_variables(translated, variable_map)

        return result

    def translate_batch(self, texts: List[str], model_id: str,
                        batch_size: int = 32) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts.
            model_id: Helsinki-NLP model identifier.
            batch_size: Batch size for inference.

        Returns:
            List of translated texts.
        """
        self._load_model(model_id)

        tokenizer = self._tokenizers[model_id]
        model = self._models[model_id]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt",
                              padding=True, truncation=True)
            translated = model.generate(**inputs)

            for t in translated:
                result = tokenizer.decode(t, skip_special_tokens=True)
                results.append(result)

        return results

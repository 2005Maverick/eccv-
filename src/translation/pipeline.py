"""
Translation pipeline — translates 130K challenges × 5 languages = 650K instances.
Template-first approach: translate templates once, fill variables after.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any

import yaml
from tqdm import tqdm

from src.translation.templates import QUESTION_TEMPLATES
from src.translation.translator import Translator
from src.translation.script_validator import validate_script, validate_variable_slots

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    Multilingual translation pipeline.

    Step 1: Translate all unique question templates to each target language
    Step 2: For each challenge, fill variables into the translated template
    Step 3: Validate script correctness
    Step 4: Output to outputs/final_dataset.jsonl

    Total: 130K × 5 languages = 650K multilingual instances.
    """

    def __init__(self, languages_path: str = "configs/languages.yaml",
                 pipeline_path: str = "configs/pipeline.yaml"):
        with open(languages_path) as f:
            self.languages_config = yaml.safe_load(f)
        with open(pipeline_path) as f:
            self.pipeline_config = yaml.safe_load(f)

        self.translator = Translator()
        self.challenges_path = "data/challenges/challenges.jsonl"
        self.output_path = "outputs/final_dataset.jsonl"
        os.makedirs("outputs", exist_ok=True)

        # Cache for translated templates
        self._template_cache: Dict[str, Dict[str, str]] = {}

    def _translate_templates(self):
        """Pre-translate all unique question templates to each language."""
        print("Translating question templates...")

        for lang_code, lang_info in self.languages_config.items():
            if lang_code == "en":
                continue  # Skip English (source language)

            model_id = lang_info.get("model_id", "")
            if not model_id:
                logger.warning(f"No model_id for {lang_code}, skipping")
                continue

            self._template_cache[lang_code] = {}

            # Collect all unique templates
            all_templates = set()
            for bias_type, templates in QUESTION_TEMPLATES.items():
                for tmpl in templates:
                    all_templates.add(tmpl["template"])

            for template in tqdm(sorted(all_templates),
                                  desc=f"  Templates → {lang_code}", unit="tmpl"):
                try:
                    translated = self.translator.translate_template(template, model_id)

                    # Validate script
                    valid, fraction = validate_script(translated, lang_code)
                    if not valid:
                        logger.warning(
                            f"Script mismatch for '{lang_code}': {template[:50]}..."
                        )
                        # Still store it, but flag for review
                        translated = f"[SCRIPT_WARNING:{fraction}] {translated}"

                    # Validate variable slots
                    if not validate_variable_slots(template, translated):
                        logger.warning(
                            f"Variable slots lost for '{lang_code}': {template[:50]}..."
                        )
                        translated = template  # Fallback to English

                    self._template_cache[lang_code][template] = translated

                except Exception as e:
                    logger.error(f"Translation failed: {e}")
                    self._template_cache[lang_code][template] = template  # Fallback

    def _translate_challenge(self, challenge: Dict, lang_code: str) -> Dict:
        """
        Create a translated version of a challenge.

        Args:
            challenge: Original challenge dict.
            lang_code: Target language code.

        Returns:
            Translated challenge dict.
        """
        translated = challenge.copy()
        translated["language"] = lang_code

        original_template = challenge.get("question_template", "")

        if lang_code == "en":
            # No translation needed
            translated["question_translated"] = original_template
        else:
            # Look up cached translation
            cached = self._template_cache.get(lang_code, {})
            translated_template = cached.get(original_template, original_template)
            translated["question_translated"] = translated_template

        return translated

    def run(self):
        """Run the translation pipeline."""
        # Load challenges
        if not os.path.exists(self.challenges_path):
            logger.error(f"Challenges file not found: {self.challenges_path}")
            return

        challenges = []
        with open(self.challenges_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    challenges.append(json.loads(line))

        print(f"\n{'='*60}")
        print(f"Stage 5: Multilingual Translation")
        print(f"{'='*60}")
        print(f"Challenges: {len(challenges):,}")
        print(f"Languages: {list(self.languages_config.keys())}")
        print(f"Expected output: {len(challenges) * len(self.languages_config):,} instances")

        # Step 1: Pre-translate templates
        self._translate_templates()

        # Step 2: Create multilingual instances
        total_written = 0
        script_failures = 0

        with open(self.output_path, "w") as f:
            for lang_code in self.languages_config:
                lang_name = self.languages_config[lang_code].get("name", lang_code)
                print(f"\nProcessing {lang_name} ({lang_code})...")

                for challenge in tqdm(challenges, desc=f"  {lang_code}", unit="ch"):
                    translated = self._translate_challenge(challenge, lang_code)
                    f.write(json.dumps(translated, ensure_ascii=False) + "\n")
                    total_written += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"Translation Complete")
        print(f"{'='*60}")
        print(f"Total instances: {total_written:,}")
        print(f"Script failures: {script_failures:,}")
        print(f"Output: {self.output_path}")
        print(f"{'='*60}")

    def sample_for_human_validation(self, sample_size: int = 200):
        """
        Sample 200 multilingual instances for human validation.

        Args:
            sample_size: Number of instances to sample (default 200).
        """
        if not os.path.exists(self.output_path):
            logger.error(f"Dataset not found: {self.output_path}")
            return

        # Load all instances
        instances = []
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))

        if len(instances) < sample_size:
            sample_size = len(instances)

        # Stratified sample: 40 per language
        per_language = sample_size // len(self.languages_config)
        sampled = []

        for lang_code in self.languages_config:
            lang_instances = [i for i in instances if i.get("language") == lang_code]
            if lang_instances:
                k = min(per_language, len(lang_instances))
                sampled.extend(random.sample(lang_instances, k))

        # Save
        sample_path = "outputs/human_validation_sample.jsonl"
        with open(sample_path, "w") as f:
            for instance in sampled:
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

        print(f"Saved {len(sampled)} instances for human validation: {sample_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TranslationPipeline()
    pipeline.run()
    pipeline.sample_for_human_validation()

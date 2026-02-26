"""
Generation pipeline — orchestrates all bias generators to produce 130K challenges.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

import yaml
from tqdm import tqdm

from src.generators.base_generator import Challenge
from src.generators.texture_generator import TextureGenerator
from src.generators.counting_generator import CountingGenerator
from src.generators.spatial_generator import SpatialGenerator
from src.generators.physics_generator import PhysicsGenerator
from src.generators.temporal_generator import TemporalGenerator
from src.generators.spurious_generator import SpuriousGenerator
from src.generators.compositional_generator import CompositionalGenerator
from src.generators.text_image_generator import TextImageGenerator
from src.generators.scale_generator import ScaleGenerator
from src.generators.compound_generator import CompoundGenerator

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Challenge generation pipeline.

    Runs each generator to hit per-bias target counts from configs/biases.yaml.
    Compound generator runs LAST (depends on single-bias images being validated).
    Total target: 90K single-bias + 40K compound = 130K challenges.
    Rejects any challenge where confound_check_passed = False.
    """

    def __init__(self, config_path: str = "configs/pipeline.yaml",
                 biases_path: str = "configs/biases.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        with open(biases_path) as f:
            self.biases_config = yaml.safe_load(f)

        random.seed(self.config.get("random_seed", 42))

        self.annotations_path = "data/processed/annotations.jsonl"
        self.output_path = "data/challenges/challenges.jsonl"
        os.makedirs("data/challenges", exist_ok=True)

        # Initialize generators
        self.generators = {
            "texture": TextureGenerator(),
            "counting": CountingGenerator(),
            "spatial_relations": SpatialGenerator(),
            "physical_plausibility": PhysicsGenerator(),
            "temporal_reasoning": TemporalGenerator(),
            "spurious_correlation": SpuriousGenerator(),
            "compositional_binding": CompositionalGenerator(),
            "text_in_image": TextImageGenerator(),
            "scale_invariance": ScaleGenerator(),
        }
        self.compound_generator = CompoundGenerator()

    def _load_annotations(self) -> List[Dict]:
        """Load all annotations from JSONL."""
        annotations = []
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        annotations.append(json.loads(line))
        logger.info(f"Loaded {len(annotations)} annotations")
        return annotations

    def _generate_single_bias(self, bias_type: str, generator,
                               annotations: List[Dict],
                               target_count: int) -> List[Dict]:
        """Generate challenges for a single bias type."""
        challenges = []
        rejected = 0

        # Shuffle annotations for variety
        shuffled = annotations.copy()
        random.shuffle(shuffled)

        pbar = tqdm(total=target_count, desc=f"  {bias_type}", unit="ch", leave=False)

        i = 0
        while len(challenges) < target_count and i < len(shuffled):
            # Different generators need different numbers of annotations
            if bias_type in ("counting", "spurious_correlation", "text_in_image"):
                # Need pairs
                if i + 1 >= len(shuffled):
                    break
                ann_batch = [shuffled[i], shuffled[i + 1]]
                i += 2
            elif bias_type == "temporal_reasoning":
                # Need triplets
                if i + 2 >= len(shuffled):
                    break
                ann_batch = [shuffled[i], shuffled[i + 1], shuffled[i + 2]]
                i += 3
            else:
                # Single image
                ann_batch = [shuffled[i]]
                i += 1

            try:
                challenge = generator.generate_challenge(ann_batch)
                if challenge is not None:
                    if challenge.confound_check_passed:
                        challenges.append(challenge.to_dict())
                        pbar.update(1)
                    else:
                        rejected += 1
                        logger.debug(f"Rejected confounded challenge for {bias_type}")
            except Exception as e:
                logger.debug(f"Generator error for {bias_type}: {e}")
                continue

        pbar.close()

        if rejected > 0:
            logger.info(f"  {bias_type}: rejected {rejected} confounded challenges")

        return challenges

    def run(self):
        """Run the full generation pipeline."""
        annotations = self._load_annotations()

        if not annotations:
            logger.error("No annotations found. Run Stage 3 first.")
            return

        all_challenges = []
        stats = {"generated": Counter(), "rejected": Counter()}

        print(f"\n{'='*60}")
        print(f"Stage 4: Adversarial Challenge Generation")
        print(f"{'='*60}")
        print(f"Annotations available: {len(annotations):,}")

        # Generate single-bias challenges first
        for bias_type, generator in self.generators.items():
            target = self.biases_config.get(bias_type, {}).get("target_challenge_count", 10000)
            print(f"\nGenerating {bias_type} (target: {target:,})...")

            challenges = self._generate_single_bias(
                bias_type, generator, annotations, target
            )
            all_challenges.extend(challenges)
            stats["generated"][bias_type] = len(challenges)

        # Compound generator runs LAST
        compound_target = self.config.get("compound_challenge_count", 40000)
        print(f"\nGenerating compound challenges (target: {compound_target:,})...")

        compound_challenges = []
        shuffled = annotations.copy()
        random.shuffle(shuffled)

        for i in tqdm(range(0, min(compound_target * 2, len(shuffled) - 1), 2),
                      desc="  compound", unit="ch"):
            if len(compound_challenges) >= compound_target:
                break

            ann_pair = [shuffled[i], shuffled[i + 1]]
            try:
                challenge = self.compound_generator.generate_challenge(ann_pair)
                if challenge is not None:
                    compound_challenges.append(challenge.to_dict())
            except Exception as e:
                continue

        all_challenges.extend(compound_challenges)
        stats["generated"]["compound"] = len(compound_challenges)

        # Save to JSONL
        with open(self.output_path, "w") as f:
            for challenge in all_challenges:
                f.write(json.dumps(challenge) + "\n")

        # Print summary
        total = len(all_challenges)
        print(f"\n{'='*60}")
        print(f"Generation Complete")
        print(f"{'='*60}")
        print(f"Total challenges: {total:,}")
        print(f"\nPer-bias breakdown:")
        for bias, count in sorted(stats["generated"].items()):
            frac = count / total if total > 0 else 0
            print(f"  {bias:<28} {count:>7,}  ({frac:.1%})")

        # Difficulty distribution
        diff_counts = Counter(c["difficulty"] for c in all_challenges)
        print(f"\nDifficulty distribution:")
        for diff, count in sorted(diff_counts.items()):
            frac = count / total if total > 0 else 0
            print(f"  {diff:<10} {count:>7,}  ({frac:.1%})")
        print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = GenerationPipeline()
    pipeline.run()

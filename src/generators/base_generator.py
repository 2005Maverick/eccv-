"""
Base challenge generator with Challenge dataclass and single-factor isolation validation.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Challenge:
    """Represents a single adversarial challenge."""
    challenge_id: str
    bias_type: str
    sub_type: str                # e.g. "relative_position", "existential_counting"
    difficulty: str              # easy / medium / hard
    image_a_id: str
    image_b_id: str              # same as image_a_id for single-image transforms
    question_template: str       # uses {variable} slots
    correct_answer: str
    distractor_answers: List[str]
    ground_truth_method: str     # e.g. "yolo_count", "astropy_shadow", "ocr_text"
    confound_check_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    # v3 causal-contrastive fields
    causal_pair: Dict[str, Any] = field(default_factory=dict)
    # {original_id, counterfactual_id, transform_type, what_changed, what_preserved}
    confound_strength: float = 0.0  # 0.0 (no confound) to 1.0 (strong confound)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)


class ChallengeGenerator(ABC):
    """
    Abstract base class for adversarial challenge generators.

    Each generator produces challenges for a specific bias type.
    CRITICAL: Each challenge must vary EXACTLY ONE factor between
    Image A and Image B. Challenges that vary multiple factors are
    confounded and scientifically invalid — they must be rejected.
    """

    def __init__(self, bias_type: str, ground_truth_method: str):
        self.bias_type = bias_type
        self.ground_truth_method = ground_truth_method

    def _generate_id(self) -> str:
        """Generate a unique challenge ID."""
        return f"{self.bias_type}_{uuid.uuid4().hex[:12]}"

    @abstractmethod
    def generate_challenge(self, annotations: List[Dict]) -> Optional[Challenge]:
        """
        Generate a single challenge from annotation data.

        Args:
            annotations: List of image annotation dicts.

        Returns:
            Challenge object if valid, None if rejected.
        """
        pass

    def validate_single_factor_isolation(self, challenge: Challenge,
                                          annotations: List[Dict]) -> bool:
        """
        Validate that only the bias-relevant attribute differs between images.

        This is the scientific boundary condition. Returns False if the
        challenge is confounded (multiple factors differ).

        Args:
            challenge: The generated challenge.
            annotations: The annotations used to generate it.

        Returns:
            True if single-factor isolation holds, False if confounded.
        """
        if len(annotations) < 2:
            return True  # Single-image transform, always valid

        ann_a = annotations[0]
        ann_b = annotations[1]

        # Define which attributes are bias-relevant for each type
        bias_relevant_keys = {
            "texture": {"texture"},
            "counting": {"detections"},
            "spatial_relations": {"depth"},
            "physical_plausibility": {"shadow"},
            "temporal_reasoning": {"temporal"},
            "spurious_correlation": {"detections"},
            "compositional_binding": {"detections"},
            "text_in_image": {"ocr"},
            "scale_invariance": {"detections"},
        }

        relevant = bias_relevant_keys.get(self.bias_type, set())
        non_relevant_keys = {"detections", "depth", "shadow", "ocr", "texture", "temporal"} - relevant

        # Check that non-relevant attributes are similar
        for key in non_relevant_keys:
            val_a = ann_a.get(key, {})
            val_b = ann_b.get(key, {})

            # For counting bias: check that the object category is the same
            if self.bias_type == "counting" and key == "detections":
                continue  # Detections ARE the bias-relevant attribute

            # Simple similarity check: if both are dicts with 'detections' lists,
            # compare the labels (not counts)
            if key == "detections" and isinstance(val_a, dict) and isinstance(val_b, dict):
                labels_a = set(d.get("label", "") for d in val_a.get("detections", []))
                labels_b = set(d.get("label", "") for d in val_b.get("detections", []))
                # If the set of detected object types differs significantly, it's confounded
                if labels_a and labels_b:
                    overlap = labels_a & labels_b
                    if len(overlap) < min(len(labels_a), len(labels_b)) * 0.5:
                        logger.debug(
                            f"Confound detected: different object types in {key} "
                            f"({labels_a} vs {labels_b})"
                        )
                        return False

        return True

    def _create_challenge(self, annotations: List[Dict],
                           difficulty: str, sub_type: str,
                           question_template: str,
                           correct_answer: str, distractors: List[str],
                           metadata: Optional[Dict] = None) -> Optional[Challenge]:
        """
        Helper to create a challenge with automatic validation.

        Args:
            annotations: Annotation dicts used.
            difficulty: easy/medium/hard.
            sub_type: Specific sub-category of the bias.
            question_template: Question with {variable} slots.
            correct_answer: The correct answer.
            distractors: List of distractor answers.
            metadata: Additional metadata.

        Returns:
            Challenge if valid, None if confounded.
        """
        image_a_id = annotations[0].get("image_id", "unknown_a")
        image_b_id = annotations[1].get("image_id", image_a_id) if len(annotations) > 1 else image_a_id

        challenge = Challenge(
            challenge_id=self._generate_id(),
            bias_type=self.bias_type,
            sub_type=sub_type,
            difficulty=difficulty,
            image_a_id=image_a_id,
            image_b_id=image_b_id,
            question_template=question_template,
            correct_answer=correct_answer,
            distractor_answers=distractors,
            ground_truth_method=self.ground_truth_method,
            confound_check_passed=True,
            metadata=metadata or {},
        )

        # Validate single-factor isolation
        if not self.validate_single_factor_isolation(challenge, annotations):
            challenge.confound_check_passed = False
            logger.info(
                f"Rejected confounded challenge {challenge.challenge_id} "
                f"({self.bias_type})"
            )
            return None

        return challenge

"""Gate 4.1 — Confounded challenge is rejected, not accepted"""
import sys
sys.path.insert(0, ".")

from src.generators.spatial_generator import SpatialGenerator

gen = SpatialGenerator()

# Mock annotation pair where BOTH spatial orientation AND object count differ
# (would be confounded — testing two things at once)
confounded_annotations = [
    {"image_id": "img_a", "detections": {"detections": [{"label": "cat"}] * 3},
     "depth": {"left_depth_mean": 0.3}},
    {"image_id": "img_b", "detections": {"detections": [{"label": "cat"}] * 7},
     # count also differs
     "depth": {"left_depth_mean": 0.7}},
]

result = gen.generate_challenge(confounded_annotations)

if result is not None:
    assert result.confound_check_passed == False, \
        "Confounded challenge must have confound_check_passed=False"

print("GATE 4.1 PASSED")

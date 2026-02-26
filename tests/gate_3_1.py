"""Gate 3.1 — Shadow plausibility is deterministically correct"""
import sys
sys.path.insert(0, ".")

from src.processing.shadow_detector import ShadowDetector

detector = ShadowDetector()

# Known case: NYC (40.71°N, 74.01°W), summer solstice noon
# Sun is due south, elevation ~72° → expected shadow points north (~180° azimuth)
result = detector.compute_expected_shadow(
    lat=40.7128, lon=-74.0060,
    timestamp="2024-06-21T12:00:00"
)

assert result['shadow_angle_expected'] is not None
assert 160 < result['shadow_angle_expected'] < 200, \
    f"Expected shadow ~180° (north), got {result['shadow_angle_expected']}"
assert result['sun_elevation'] > 60, "Sun should be high at NYC noon in June"

print("GATE 3.1 PASSED")

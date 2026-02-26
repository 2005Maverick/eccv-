"""Gate 1.1 — Bias taxonomy completeness"""
import yaml

with open("configs/biases.yaml") as f:
    biases = yaml.safe_load(f)

required_biases = {
    'texture', 'counting', 'spatial_relations', 'physical_plausibility',
    'temporal_reasoning', 'spurious_correlation', 'compositional_binding',
    'text_in_image', 'scale_invariance'
}

assert set(biases.keys()) == required_biases, \
    f"Missing or extra biases: {set(biases.keys()) ^ required_biases}"

for name, meta in biases.items():
    splits = meta['difficulty_split']
    assert abs(sum(splits.values()) - 1.0) < 0.01, f"{name} splits don't sum to 1.0"
    assert 'ground_truth_method' in meta, f"{name} missing ground_truth_method"
    assert 'target_challenge_count' in meta, f"{name} missing target_challenge_count"

total = sum(b['target_challenge_count'] for b in biases.values())
assert 85000 <= total <= 95000, f"Single-bias targets should sum ~90K, got {total}"

print("GATE 1.1 PASSED")

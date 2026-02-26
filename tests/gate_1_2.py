"""Gate 1.2 — Pipeline config has all required keys"""
import yaml

with open("configs/pipeline.yaml") as f:
    cfg = yaml.safe_load(f)

required_keys = {
    'target_dataset_size', 'compound_challenge_count', 'languages',
    'random_seed', 'image_sources', 'difficulty_distribution'
}

missing = required_keys - set(cfg.keys())
assert not missing, f"Missing pipeline config keys: {missing}"
assert cfg['target_dataset_size'] == 130000
assert cfg['compound_challenge_count'] == 40000
assert set(cfg['languages']) == {'en', 'es', 'zh', 'hi', 'ar'}
assert cfg['random_seed'] == 42

print("GATE 1.2 PASSED")

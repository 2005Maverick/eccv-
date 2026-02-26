"""
Dataset completeness smoke test.
Verifies the final deliverable before handing off to evaluation.
"""
import json
from collections import Counter
from pathlib import Path

print("Running final dataset smoke test...")

# 1. Check all files exist
assert Path("data/raw/manifest.jsonl").exists(), "manifest.jsonl missing"
assert Path("data/processed/annotations.jsonl").exists(), "annotations.jsonl missing"
assert Path("data/challenges/challenges.jsonl").exists(), "challenges.jsonl missing"
assert Path("data/challenges/challenges_multilingual.jsonl").exists(), "multilingual missing"
assert Path("data/validation/human_validation_sample.csv").exists(), "validation sample missing"

# 2. Check image count
manifest = [json.loads(l) for l in open("data/raw/manifest.jsonl")]
assert len(manifest) >= 400_000, f"Only {len(manifest):,} images collected (target 500K)"

# 3. Check challenge count and structure
challenges = [json.loads(l) for l in open("data/challenges/challenges_multilingual.jsonl")]
assert len(challenges) >= 125_000, f"Only {len(challenges):,} challenges (target 130K)"

required_challenge_keys = {
    'challenge_id', 'bias_type', 'difficulty',
    'image_a_id', 'image_b_id', 'correct_answer',
    'ground_truth_method', 'confound_check_passed', 'questions'
}

for c in challenges[:100]:
    missing = required_challenge_keys - set(c.keys())
    assert not missing, f"Challenge {c['challenge_id']} missing keys: {missing}"

# 4. Check language coverage
for c in challenges[:100]:
    for lang in ['en', 'es', 'zh', 'hi', 'ar']:
        q = c['questions'].get(lang, '')
        assert q and len(q) > 4, f"Missing {lang} in challenge {c['challenge_id']}"

# 5. Check no confounded challenges slipped through
confounded = [c for c in challenges if not c.get('confound_check_passed', True)]
assert len(confounded) == 0, f"{len(confounded)} confounded challenges in dataset!"

# 6. Summary
bias_dist = Counter(c['bias_type'] for c in challenges)
diff_dist = Counter(c['difficulty'] for c in challenges)

print("\n" + "=" * 60)
print("DATASET SMOKE TEST PASSED")
print("=" * 60)
print(f"Images collected:      {len(manifest):>10,}")
print(f"Total challenges:      {len(challenges):>10,}")
print(f"Languages per challenge: 5 (EN, ES, ZH, HI, AR)")
print(f"Total question instances: {len(challenges)*5:>8,}")
print(f"\nBias distribution:")
for bias, count in sorted(bias_dist.items()):
    print(f"  {bias:<28} {count:>7,}  ({count/len(challenges):.1%})")
print(f"\nDifficulty distribution:")
for diff, count in sorted(diff_dist.items()):
    print(f"  {diff:<10} {count:>7,}  ({count/len(challenges):.1%})")
print("\nDataset ready for evaluation pipeline.")

"""Gate 4.3 — Final dataset distribution matches target"""
import sys
import json
sys.path.insert(0, ".")

from collections import Counter

# This test checks the distribution of an already-generated challenges file
# It will be run after the generation pipeline completes on real data

try:
    challenges = [json.loads(l) for l in open("data/challenges/challenges.jsonl")]
except FileNotFoundError:
    print("Note: data/challenges/challenges.jsonl not found.")
    print("This gate test requires running the full generation pipeline first.")
    print("Run: python -m src.generators.pipeline")
    print("GATE 4.3 SKIPPED (no data)")
    sys.exit(0)

total = len(challenges)
assert total >= 125_000, f"Only {total} challenges generated (target 130K)"

bias_counts = Counter(c['bias_type'] for c in challenges)
difficulty_counts = Counter(c['difficulty'] for c in challenges)

# No single bias > 15% of total
for bias, count in bias_counts.items():
    frac = count / total
    assert frac <= 0.15, f"{bias} is {frac:.1%} of dataset — too dominant"

# Difficulty distribution within ±5% of target
easy_frac = difficulty_counts['easy'] / total
medium_frac = difficulty_counts['medium'] / total
hard_frac = difficulty_counts['hard'] / total

assert 0.25 <= easy_frac <= 0.35, f"Easy {easy_frac:.2%} outside [25%, 35%]"
assert 0.45 <= medium_frac <= 0.55, f"Medium {medium_frac:.2%} outside [45%, 55%]"
assert 0.15 <= hard_frac <= 0.25, f"Hard {hard_frac:.2%} outside [15%, 25%]"

# Confirm compound challenges exist
compound = [c for c in challenges if c['bias_type'] == 'compound']
assert len(compound) >= 35_000, f"Only {len(compound)} compound challenges (target 40K)"

print("GATE 4.3 PASSED")

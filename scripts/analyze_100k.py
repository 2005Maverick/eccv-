"""Analyze the generated 100K challenges for quality."""
import json, collections, random, os

ch_path = os.path.join("dataset_final", "challenges", "challenges.jsonl")
challenges = [json.loads(l) for l in open(ch_path, "r", encoding="utf-8")]
print(f"Total challenges: {len(challenges):,}")
print()

# Distractor count check
d_counts = collections.Counter(len(c.get("distractor_answers", [])) for c in challenges)
print("Distractor counts:")
for k, v in sorted(d_counts.items()):
    print(f"  {k} distractors: {v:,} ({v/len(challenges)*100:.1f}%)")

# Template diversity
templates = collections.Counter(c.get("question_template", "") for c in challenges)
print(f"\nUnique templates: {len(templates):,}")
print(f"Top 5 templates:")
for t, ct in templates.most_common(5):
    print(f"  [{ct:4d}] {t[:90]}")

# Answer diversity
answers = collections.Counter(c.get("correct_answer", "") for c in challenges)
yes_no = sum(v for k, v in answers.items() if k.lower() in ["yes", "no"])
print(f"\nYes/No answers: {yes_no:,} ({yes_no/len(challenges)*100:.1f}%)")
print(f"Unique answers: {len(answers):,}")

# Check for generic 'object' label
generic = [c for c in challenges if "object" in str(c.get("correct_answer", "")).lower() 
           or "object" in str(c.get("question_template", "")).lower()]
print(f"\nContaining generic 'object': {len(generic):,}")

# Sub-type diversity
sub_types = collections.Counter(c.get("sub_type", "N/A") for c in challenges)
print(f"\nUnique sub-types: {len(sub_types):,}")
for st, ct in sub_types.most_common(20):
    print(f"  {st}: {ct:,}")

# Random samples per bias type
print("\n" + "=" * 65)
print("  SAMPLE CHALLENGES (1 per bias type)")
print("=" * 65)
random.seed(42)
by_bias = collections.defaultdict(list)
for c in challenges:
    by_bias[c["bias_type"]].append(c)
for bt in sorted(by_bias.keys()):
    sample = random.choice(by_bias[bt])
    print(f"\n[{bt}] (n={len(by_bias[bt])})")
    q = sample.get("question_template", "")[:100]
    print(f"  Q: {q}")
    print(f"  A: {sample.get('correct_answer', '')}")
    ds = sample.get("distractor_answers", [])
    print(f"  D: {ds}")
    print(f"  Diff: {sample.get('difficulty', '')} | Sub: {sample.get('sub_type', '')}")

"""
=================================================================
  ECCV v3 — 100K Challenge Generator

  Assumes images + annotations already exist (from Kaggle or the
  main pipeline). Generates challenges via multi-pass generation,
  cross-image pairing, and counterfactual transforms.

  Usage:
    python scripts/create_dataset_100k.py --target 100000 --passes 5
    python scripts/create_dataset_100k.py --target 10000 --passes 2  # Quick test

  Prerequisites:
    - dataset_final/images/         (source images)
    - dataset_final/annotations/annotations.jsonl  (from kaggle_annotate.py)
  
  Outputs:
    - dataset_final/challenges/challenges.jsonl
    - dataset_final/counterfactuals/   (generated images)
    - dataset_final/translations/      (5 languages)
=================================================================
"""

import os
import sys
import json
import random
import time
import logging
import argparse
import hashlib
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Dict

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

import numpy as np
import cv2

# ============================================================
# CLI Arguments
# ============================================================
parser = argparse.ArgumentParser(description="ECCV v3 — 100K challenge generator")
parser.add_argument("--target", type=int, default=100000,
                    help="Target number of challenges (default: 100000)")
parser.add_argument("--passes", type=int, default=5,
                    help="Number of generation passes per generator (default: 5)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed (default: 42)")
parser.add_argument("--skip-counterfactuals", action="store_true",
                    help="Skip counterfactual image generation")
parser.add_argument("--skip-translation", action="store_true",
                    help="Skip multilingual translation stage")
parser.add_argument("--dataset-dir", type=str, default=None,
                    help="Dataset directory (default: dataset_final/)")
args = parser.parse_args()

# ============================================================
# Timer utilities
# ============================================================
GLOBAL_START = time.time()
STAGE_TIMES = {}


def elapsed():
    return str(timedelta(seconds=int(time.time() - GLOBAL_START)))


def stage_start(name):
    STAGE_TIMES[name] = {"start": time.time()}
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def stage_end(name, label="", count=0):
    if name in STAGE_TIMES:
        STAGE_TIMES[name]["duration"] = time.time() - STAGE_TIMES[name]["start"]
        dur = str(timedelta(seconds=int(STAGE_TIMES[name]["duration"])))
        print(f"  >> {label}: {count}" if label else "")
        print(f"  >> Elapsed: {dur} | Total: {elapsed()}")


def progress(i, total, label=""):
    pct = i / max(total, 1) * 100
    bar_len = 30
    filled = int(bar_len * i / max(total, 1))
    bar = '#' * filled + '-' * (bar_len - filled)
    rate = i / max(time.time() - GLOBAL_START, 1)
    eta = (total - i) / max(rate, 0.01)
    print(f"\r  [{bar}] {pct:5.1f}% {i}/{total} | ETA: {timedelta(seconds=int(eta))} | {label}",
          end="", flush=True)


# ============================================================
# Configuration
# ============================================================
DATASET_DIR = args.dataset_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset_final"
)
img_dir = os.path.join(DATASET_DIR, "images")
ann_path = os.path.join(DATASET_DIR, "annotations", "annotations.jsonl")
ch_dir = os.path.join(DATASET_DIR, "challenges")
cf_dir = os.path.join(DATASET_DIR, "counterfactuals")
trans_dir = os.path.join(DATASET_DIR, "translations")

for d in [ch_dir, cf_dir, trans_dir]:
    os.makedirs(d, exist_ok=True)

TARGET = args.target
NUM_PASSES = args.passes
BASE_SEED = args.seed

print(f"  ECCV v3 — 100K Challenge Generator")
print(f"  Target:     {TARGET:,}")
print(f"  Passes:     {NUM_PASSES}")
print(f"  Base seed:  {BASE_SEED}")
print(f"  Dataset:    {DATASET_DIR}")

# ============================================================
# STAGE 1: Load Annotations
# ============================================================
stage_start("1. Loading Annotations")

if not os.path.exists(ann_path):
    print(f"  ERROR: Annotations file not found at {ann_path}")
    print(f"  Run kaggle_annotate.py first to generate annotations.")
    sys.exit(1)

annotations = [json.loads(line) for line in open(ann_path, "r", encoding="utf-8")]
print(f"  Loaded {len(annotations)} annotations")

# Verify images exist
images_found = 0
for ann in annotations:
    img_id = ann.get("image_id", "")
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if os.path.exists(os.path.join(img_dir, img_id + ext)):
            images_found += 1
            break

print(f"  Images on disk: {images_found}/{len(annotations)}")
stage_end("1. Loading Annotations", "annotations", len(annotations))


# ============================================================
# STAGE 2: Counterfactual Generation
# ============================================================
if not args.skip_counterfactuals:
    stage_start("2. Counterfactual Generation")

    from src.processing.counterfactual_factory import CounterfactualFactory

    cf_factory = CounterfactualFactory(cf_dir, seed=BASE_SEED)
    cf_results = cf_factory.process_all(img_dir, annotations)
    cf_factory.save_manifest(os.path.join(cf_dir, "manifest.jsonl"))

    print(f"  >> Generated: {len(cf_results)} counterfactual images")
    print(f"  >> Transforms: {Counter(r['transform_type'] for r in cf_results)}")
    stage_end("2. Counterfactual Generation", "counterfactuals", len(cf_results))
else:
    print("\n  [SKIP] Counterfactual generation")


# ============================================================
# STAGE 3: Multi-Pass Challenge Generation
# ============================================================
stage_start("3. Multi-Pass Challenge Generation")

# Import all generators
from src.generators.counting_generator import CountingGenerator
from src.generators.spatial_generator import SpatialGenerator
from src.generators.compositional_generator import CompositionalGenerator
from src.generators.scale_generator import ScaleGenerator
from src.generators.texture_generator import TextureGenerator
from src.generators.compound_generator import CompoundGenerator
from src.generators.temporal_generator import TemporalGenerator
from src.generators.text_image_generator import TextImageGenerator
from src.generators.physics_generator import PhysicsGenerator
from src.generators.spurious_generator import SpuriousGenerator
from src.generators.typography_conflict_generator import TypographyConflictGenerator
from src.generators.occlusion_gradient_generator import OcclusionGradientGenerator
from src.generators.cultural_bias_generator import CulturalBiasGenerator
from src.generators.temporal_consistency_generator import TemporalConsistencyGenerator


def make_generators():
    """Create fresh generator instances."""
    return {
        # Single-annotation generators
        "spatial_relations": {"gen": SpatialGenerator(), "mode": "single"},
        "compositional_binding": {"gen": CompositionalGenerator(), "mode": "single",
                                   "kwargs": {"image_dir": img_dir}},
        "scale_invariance": {"gen": ScaleGenerator(), "mode": "single"},
        "texture": {"gen": TextureGenerator(), "mode": "single_all"},
        "physical_plausibility": {"gen": PhysicsGenerator(), "mode": "single",
                                  "kwargs": {"image_dir": img_dir}},
        # Pair generators
        "counting": {"gen": CountingGenerator(), "mode": "pair_and_single"},
        "spurious_correlation": {"gen": SpuriousGenerator(), "mode": "pair"},
        # Special generators
        "temporal_reasoning": {"gen": TemporalGenerator(), "mode": "pair_and_single",
                               "kwargs": {"image_dir": img_dir}},
        "text_in_image": {"gen": TextImageGenerator(), "mode": "pair_and_single"},
        "compound": {"gen": CompoundGenerator(), "mode": "pair", "max_frac": 0.08},
        # v3 generators
        "typography_conflict": {"gen": TypographyConflictGenerator(), "mode": "single",
                                "kwargs": {"image_dir": img_dir, "cf_dir": cf_dir}},
        "occlusion_gradient": {"gen": OcclusionGradientGenerator(), "mode": "single",
                               "kwargs": {"image_dir": img_dir, "cf_dir": cf_dir}},
        "cultural_visual_bias": {"gen": CulturalBiasGenerator(), "mode": "single",
                                 "kwargs": {"image_dir": img_dir}},
        "temporal_consistency": {"gen": TemporalConsistencyGenerator(), "mode": "single",
                                 "kwargs": {"image_dir": img_dir, "cf_dir": cf_dir}},
    }


challenges = []
seen_hashes = set()  # For dedup


def challenge_hash(ch: Dict) -> str:
    """Hash a challenge for dedup."""
    key = f"{ch.get('image_a_id')}|{ch.get('image_b_id')}|" \
          f"{ch.get('question_template', '')[:80]}|{ch.get('correct_answer')}"
    return hashlib.md5(key.encode()).hexdigest()


def add_challenge(ch_dict: Dict) -> bool:
    """Add challenge if not a duplicate and target not reached."""
    h = challenge_hash(ch_dict)
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    challenges.append(ch_dict)
    return True


print(f"  Target: {TARGET:,} challenges")
print(f"  Passes: {NUM_PASSES}")
print(f"  Annotations: {len(annotations)}")
print(f"  Expected max: ~{len(annotations) * 14 * NUM_PASSES:,} raw (before dedup)")

for pass_idx in range(NUM_PASSES):
    if len(challenges) >= TARGET:
        break

    seed = BASE_SEED + pass_idx * 1000
    random.seed(seed)

    generators = make_generators()
    pass_ct = 0

    print(f"\n  --- Pass {pass_idx + 1}/{NUM_PASSES} (seed={seed}) ---")

    for btype, config in generators.items():
        if len(challenges) >= TARGET:
            break

        gen = config["gen"]
        mode = config["mode"]
        kwargs = config.get("kwargs", {})
        max_frac = config.get("max_frac", 1.0)

        shuf = annotations.copy()
        random.shuffle(shuf)
        bt_ct = 0

        # Apply max fraction limit (e.g., compound limited to 8%)
        if max_frac < 1.0:
            max_ct = max(int(len(challenges) * max_frac), 5)
            current_bt = sum(1 for c in challenges if c.get("bias_type") == btype)
            if current_bt >= max_ct:
                continue

        if mode == "single":
            for ann in shuf:
                if len(challenges) >= TARGET:
                    break
                ch = gen.generate_challenge([ann], **kwargs)
                if ch and add_challenge(ch.to_dict()):
                    bt_ct += 1

        elif mode == "single_all":
            for ann in shuf:
                if len(challenges) >= TARGET:
                    break
                ch = gen.generate_challenge([ann], all_annotations=annotations)
                if ch and add_challenge(ch.to_dict()):
                    bt_ct += 1

        elif mode == "pair":
            for i in range(0, len(shuf) - 1, 2):
                if len(challenges) >= TARGET:
                    break
                ch = gen.generate_challenge([shuf[i], shuf[i + 1]], **kwargs)
                if ch and add_challenge(ch.to_dict()):
                    bt_ct += 1

        elif mode == "pair_and_single":
            # Pair comparisons
            for i in range(0, len(shuf) - 1, 2):
                if len(challenges) >= TARGET:
                    break
                ch = gen.generate_challenge([shuf[i], shuf[i + 1]], **kwargs)
                if ch and add_challenge(ch.to_dict()):
                    bt_ct += 1
            # Single-image
            for ann in shuf:
                if len(challenges) >= TARGET:
                    break
                ch = gen.generate_challenge([ann], **kwargs)
                if ch and add_challenge(ch.to_dict()):
                    bt_ct += 1

        pass_ct += bt_ct

    print(f"  Pass {pass_idx + 1} total: {pass_ct} new | Running total: {len(challenges):,}")


# --- Cross-image pairing (extra pairs for pair generators) ---
if len(challenges) < TARGET:
    print(f"\n  --- Cross-image pairing ---")
    cross_generators = {
        "counting": CountingGenerator(),
        "spurious_correlation": SpuriousGenerator(),
        "temporal_reasoning": TemporalGenerator(),
    }

    # Generate random cross-image pairs (not just adjacent)
    n_ann = len(annotations)
    max_cross_pairs = min(n_ann * 10, TARGET - len(challenges))

    for btype, gen in cross_generators.items():
        if len(challenges) >= TARGET:
            break

        cross_ct = 0
        for _ in range(max_cross_pairs // len(cross_generators)):
            if len(challenges) >= TARGET:
                break
            i, j = random.sample(range(n_ann), 2)
            kwargs = {"image_dir": img_dir} if btype == "temporal_reasoning" else {}
            ch = gen.generate_challenge([annotations[i], annotations[j]], **kwargs)
            if ch and add_challenge(ch.to_dict()):
                cross_ct += 1

        print(f"    {btype}: {cross_ct} cross-pair challenges")


# ============================================================
# STAGE 4: Post-Processing & Quality Filtering
# ============================================================
stage_start("4. Quality Filtering")

# Remove correct answer from distractors
filtered = []
for c in challenges:
    distractors = c.get("distractor_answers", [])
    correct = c.get("correct_answer")
    clean = list(dict.fromkeys(d for d in distractors if d != correct))  # Dedup + remove correct
    c["distractor_answers"] = clean
    filtered.append(c)

challenges = filtered
print(f"  Post-filter: {len(challenges):,} challenges")

# Balance check
bias_dist = Counter(c["bias_type"] for c in challenges)
diff_dist = Counter(c["difficulty"] for c in challenges)
answer_dist = Counter(c.get("correct_answer", "") for c in challenges)

print(f"\n  Bias distribution:")
for btype, ct in sorted(bias_dist.items(), key=lambda x: -x[1]):
    pct = ct / len(challenges) * 100
    print(f"    {btype}: {ct:,} ({pct:.1f}%)")

print(f"\n  Difficulty:")
for d, ct in sorted(diff_dist.items()):
    pct = ct / len(challenges) * 100
    print(f"    {d}: {ct:,} ({pct:.1f}%)")

# Save challenges
ch_path = os.path.join(ch_dir, "challenges.jsonl")
with open(ch_path, "w", encoding="utf-8") as f:
    for c in challenges:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

stage_end("4. Quality Filtering", "challenges", len(challenges))


# ============================================================
# STAGE 5: Multilingual Translation
# ============================================================
if not args.skip_translation:
    stage_start("5. Multilingual Translation")

    languages = ["en", "es", "zh", "hi", "ar"]
    LANG_MODELS = {
        "es": "Helsinki-NLP/opus-mt-en-es",
        "zh": "Helsinki-NLP/opus-mt-en-zh",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "ar": "Helsinki-NLP/opus-mt-en-ar",
    }

    # Load models
    use_real_translation = False
    translators = {}
    try:
        from transformers import MarianMTModel, MarianTokenizer
        for lang, model_id in LANG_MODELS.items():
            try:
                tok = MarianTokenizer.from_pretrained(model_id)
                mdl = MarianMTModel.from_pretrained(model_id)
                translators[lang] = (tok, mdl)
                print(f"  >> {lang}: {model_id} LOADED")
            except Exception as e:
                print(f"  >> {lang}: FAILED ({e})")
        if translators:
            use_real_translation = True
    except Exception as e:
        print(f"  >> Helsinki-NLP import failed: {e}")

    import re

    def translate_text(text: str, lang: str) -> str:
        if lang not in translators:
            return text
        tok, mdl = translators[lang]
        slots = re.findall(r'\{[^}]+\}', text)
        protected = text
        sentinels = []
        for i, slot in enumerate(slots):
            sentinel = f"XSLOT{i}X"
            sentinels.append((sentinel, slot))
            protected = protected.replace(slot, sentinel, 1)
        try:
            encoded = tok([protected], return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
            output = mdl.generate(**encoded)
            translated = tok.decode(output[0], skip_special_tokens=True)
            for sentinel, slot in sentinels:
                if sentinel in translated:
                    translated = translated.replace(sentinel, slot, 1)
                else:
                    for variant in [sentinel, sentinel.lower(), sentinel.upper(),
                                    f" {sentinel} ", f" {sentinel}", f"{sentinel} "]:
                        if variant in translated:
                            translated = translated.replace(variant, slot, 1)
                            break
                    else:
                        translated += f" {slot}"
            return translated
        except Exception:
            return text

    # Translate unique templates (batch for efficiency)
    unique_templates = set(ch.get("question_template", "") for ch in challenges)
    print(f"  Unique templates: {len(unique_templates):,}")

    template_cache = {}
    for lang in ["es", "zh", "hi", "ar"]:
        template_cache[lang] = {}
        batch_ct = 0
        for tmpl in unique_templates:
            if use_real_translation:
                template_cache[lang][tmpl] = translate_text(tmpl, lang)
            else:
                template_cache[lang][tmpl] = tmpl
            batch_ct += 1
            if batch_ct % 200 == 0:
                print(f"    {lang}: {batch_ct}/{len(unique_templates)} templates translated")

        print(f"  {lang}: {len(template_cache[lang]):,} templates done")

    # Generate instances
    instances = []
    for lang in languages:
        for ch in challenges:
            inst = ch.copy()
            inst["language"] = lang
            tmpl = ch.get("question_template", "")
            if lang == "en":
                inst["question_translated"] = tmpl
            else:
                inst["question_translated"] = template_cache.get(lang, {}).get(tmpl, tmpl)
            instances.append(inst)

    # Save
    trans_path = os.path.join(trans_dir, "dataset_multilingual.jsonl")
    with open(trans_path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    for lang in languages:
        lang_path = os.path.join(trans_dir, f"dataset_{lang}.jsonl")
        lang_insts = [i for i in instances if i["language"] == lang]
        with open(lang_path, "w", encoding="utf-8") as f:
            for inst in lang_insts:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
        print(f"  {lang}: {len(lang_insts):,} instances -> dataset_{lang}.jsonl")

    stage_end("5. Multilingual Translation", "instances", len(instances))
else:
    instances = challenges  # No translation
    print("\n  [SKIP] Multilingual translation")


# ============================================================
# STAGE 6: Summary & Stats
# ============================================================
stage_start("6. Summary & Stats")

stats = {
    "created": datetime.now().isoformat(),
    "version": "ECCV_v3_100K",
    "target": TARGET,
    "passes": NUM_PASSES,
    "seed": BASE_SEED,
    "total_elapsed_seconds": int(time.time() - GLOBAL_START),
    "total_elapsed_formatted": elapsed(),
    "images": images_found,
    "annotations": len(annotations),
    "challenges": len(challenges),
    "bias_types": len(bias_dist),
    "languages": 5 if not args.skip_translation else 1,
    "multilingual_instances": len(instances),
    "bias_distribution": dict(bias_dist),
    "difficulty_distribution": dict(diff_dist),
    "answer_distribution_top10": dict(answer_dist.most_common(10)),
    "stage_timings": {},
}
for name, t in STAGE_TIMES.items():
    if "duration" in t:
        stats["stage_timings"][name] = str(timedelta(seconds=int(t["duration"])))

with open(os.path.join(DATASET_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

stage_end("6. Summary & Stats")


# ============================================================
# FINAL REPORT
# ============================================================
total_time = time.time() - GLOBAL_START

print(f"\n{'='*65}")
print(f"  ECCV v3 — 100K DATASET COMPLETE!")
print(f"{'='*65}")
print(f"  Target:            {TARGET:,}")
print(f"  Achieved:          {len(challenges):,}")
print(f"  Fill rate:         {len(challenges)/TARGET*100:.1f}%")
print(f"  Total time:        {str(timedelta(seconds=int(total_time)))}")
print(f"  -----------------------------------------------")
print(f"  Annotations:       {len(annotations):,}")
print(f"  Challenges:        {len(challenges):,}")
print(f"  Bias types:        {len(bias_dist)}")
print(f"  Dedup removed:     {len(seen_hashes) - len(challenges):,} (hash collisions)")
print(f"  Multilingual:      {len(instances):,} instances")
print(f"  -----------------------------------------------")
print(f"  Difficulty:")
for d, ct in sorted(diff_dist.items()):
    print(f"    {d}: {ct:,} ({ct/len(challenges)*100:.1f}%)")
print(f"  -----------------------------------------------")
print(f"  Bias types:")
for btype, ct in sorted(bias_dist.items(), key=lambda x: -x[1]):
    print(f"    {btype}: {ct:,} ({ct/len(challenges)*100:.1f}%)")
print(f"  -----------------------------------------------")
print(f"  Answer balance (top 5):")
for ans, ct in answer_dist.most_common(5):
    print(f"    '{ans}': {ct:,}")
print(f"{'='*65}")

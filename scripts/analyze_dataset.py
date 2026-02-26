#!/usr/bin/env python3
"""Comprehensive analysis of the ECCV v2 dataset. Writes report to file."""
import json, collections, os, statistics, sys

OUT = "dataset_final"
REPORT = os.path.join(OUT, "analysis_report.txt")

ch_path = os.path.join(OUT, "challenges", "challenges.jsonl")
ann_path = os.path.join(OUT, "annotations", "annotations.jsonl")
img_dir = os.path.join(OUT, "images")

challenges = [json.loads(l) for l in open(ch_path, "r", encoding="utf-8")]
annotations = []
if os.path.exists(ann_path):
    annotations = [json.loads(l) for l in open(ann_path, "r", encoding="utf-8")]

lines = []
def p(s=""):
    lines.append(s)

p("=" * 70)
p("  ECCV v2 DATASET - COMPREHENSIVE ANALYSIS")
p("=" * 70)

p("")
p("Total challenges: %d" % len(challenges))
p("Total annotations: %d" % len(annotations))
n_images = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
p("Total images on disk: %d" % n_images)

# Bias distribution
p("")
p("-" * 50)
p("BIAS TYPE DISTRIBUTION")
p("-" * 50)
bc = collections.Counter(c["bias_type"] for c in challenges)
for k, v in bc.most_common():
    pct = v / len(challenges) * 100
    bar = "#" * int(v / 3)
    p("  %-30s %4d (%5.1f%%) %s" % (k, v, pct, bar))

# Difficulty distribution
p("")
p("-" * 50)
p("DIFFICULTY DISTRIBUTION")
p("-" * 50)
dc = collections.Counter(c["difficulty"] for c in challenges)
for k in ["easy", "medium", "hard"]:
    v = dc.get(k, 0)
    pct = v / len(challenges) * 100
    bar = "#" * int(v / 3)
    p("  %-10s %4d (%5.1f%%) %s" % (k, v, pct, bar))

# Difficulty x Bias matrix
p("")
p("-" * 50)
p("DIFFICULTY x BIAS MATRIX")
p("-" * 50)
p("  %-30s %5s %5s %5s %5s" % ("Bias Type", "easy", "med", "hard", "tot"))
p("  " + "-" * 50)
for bt, _ in bc.most_common():
    d = collections.Counter(c["difficulty"] for c in challenges if c["bias_type"] == bt)
    t = sum(d.values())
    p("  %-30s %5d %5d %5d %5d" % (bt, d.get("easy", 0), d.get("medium", 0), d.get("hard", 0), t))

# Sub-type distribution
p("")
p("-" * 50)
p("SUB-TYPE DISTRIBUTION")
p("-" * 50)
sub = collections.Counter()
for c in challenges:
    st = c.get("metadata", {}).get("sub_type", "N/A")
    sub[(c["bias_type"], st)] += 1
for (bt, st), cnt in sorted(sub.items(), key=lambda x: (-x[1])):
    p("  %-30s / %-25s %4d" % (bt, st, cnt))

# Answer distribution
p("")
p("-" * 50)
p("ANSWER DISTRIBUTION (top 20)")
p("-" * 50)
ans = collections.Counter(str(c["correct_answer"]) for c in challenges)
for a, cnt in ans.most_common(20):
    pct = cnt / len(challenges) * 100
    p("  %-40s %4d (%5.1f%%)" % (a, cnt, pct))

# Yes/No balance per bias
p("")
p("-" * 50)
p("YES/NO BALANCE PER BIAS")
p("-" * 50)
for bt, _ in bc.most_common():
    bt_ch = [c for c in challenges if c["bias_type"] == bt]
    yes_n = sum(1 for c in bt_ch if c["correct_answer"] == "Yes")
    no_n = sum(1 for c in bt_ch if c["correct_answer"] == "No")
    other = len(bt_ch) - yes_n - no_n
    if yes_n + no_n > 0:
        ratio = yes_n / (yes_n + no_n) * 100
        p("  %-30s  Yes=%3d  No=%3d  Other=%3d  (Yes%%=%d%%)" % (bt, yes_n, no_n, other, ratio))
    else:
        p("  %-30s  Yes=%3d  No=%3d  Other=%3d  (N/A)" % (bt, yes_n, no_n, other))

# Question template diversity
p("")
p("-" * 50)
p("QUESTION TEMPLATE DIVERSITY")
p("-" * 50)
tmpl = collections.Counter(c["question_template"] for c in challenges)
p("  Unique templates: %d" % len(tmpl))
p("  Templates used once: %d" % sum(1 for v in tmpl.values() if v == 1))
p("  Top 10:")
for t, cnt in tmpl.most_common(10):
    p("    [%3d] %s" % (cnt, t[:75]))

# Ground truth methods
p("")
p("-" * 50)
p("GROUND TRUTH METHODS")
p("-" * 50)
gt = collections.Counter(c["ground_truth_method"] for c in challenges)
for g, cnt in gt.most_common():
    p("  %-30s %4d" % (g, cnt))

# Confound check
p("")
p("-" * 50)
p("CONFOUND CHECK STATUS")
p("-" * 50)
passed = sum(1 for c in challenges if c.get("confound_check_passed", False))
p("  Passed: %d/%d (%.1f%%)" % (passed, len(challenges), passed / len(challenges) * 100))

# Image usage
p("")
p("-" * 50)
p("IMAGE USAGE ANALYSIS")
p("-" * 50)
img_use = collections.Counter()
for c in challenges:
    img_use[c["image_a_id"]] += 1
    img_use[c["image_b_id"]] += 1
unique = len(img_use)
vals = list(img_use.values())
p("  Unique images referenced: %d" % unique)
p("  Avg uses per image:  %.1f" % statistics.mean(vals))
p("  Median uses:         %.1f" % statistics.median(vals))
p("  Max uses:            %d (image: %s)" % (max(vals), img_use.most_common(1)[0][0]))
p("  Min uses:            %d" % min(vals))

# Distractor quality
p("")
p("-" * 50)
p("DISTRACTOR QUALITY")
p("-" * 50)
dist_counts = collections.Counter(len(c.get("distractor_answers", [])) for c in challenges)
for n, cnt in sorted(dist_counts.items()):
    p("  %d distractors: %d challenges" % (n, cnt))
dup_d = sum(1 for c in challenges if len(c.get("distractor_answers",[])) != len(set(c.get("distractor_answers",[]))))
ans_in_d = sum(1 for c in challenges if c["correct_answer"] in c.get("distractor_answers", []))
p("  Duplicate distractors: %d" % dup_d)
p("  Correct answer in distractors: %d" % ans_in_d)

# Translation files
p("")
p("-" * 50)
p("TRANSLATION FILES")
p("-" * 50)
trans_dir = os.path.join(OUT, "translations")
if os.path.isdir(trans_dir):
    for f in sorted(os.listdir(trans_dir)):
        fp = os.path.join(trans_dir, f)
        n_lines = sum(1 for _ in open(fp, "r", encoding="utf-8"))
        sz = os.path.getsize(fp)
        p("  %-25s  %5d lines  %.1f KB" % (f, n_lines, sz / 1024))

# Annotation quality
p("")
p("-" * 50)
p("ANNOTATION QUALITY")
p("-" * 50)
if annotations:
    det_counts = []
    ocr_found = 0
    texture_found = 0
    labels = collections.Counter()
    for a in annotations:
        dets_raw = a.get("detections", [])
        if isinstance(dets_raw, dict):
            dets = dets_raw.get("detections", [])
        elif isinstance(dets_raw, list):
            dets = dets_raw
        else:
            dets = []
        det_counts.append(len(dets))
        for d in dets:
            if isinstance(d, dict):
                labels[d.get("label", "unknown")] += 1
        # OCR can be in ocr_text or in a nested ocr dict
        ocr_val = a.get("ocr_text") or a.get("ocr", {})
        if ocr_val:
            ocr_found += 1
        # Texture can be nested
        tex = a.get("texture", {})
        if isinstance(tex, dict) and tex.get("edge_density", 0) > 0:
            texture_found += 1
        elif isinstance(a.get("edge_density", 0), (int, float)) and a.get("edge_density", 0) > 0:
            texture_found += 1
    p("  Total annotations:   %d" % len(annotations))
    if det_counts:
        p("  Avg detections/img:  %.1f" % statistics.mean(det_counts))
        p("  Max detections/img:  %d" % max(det_counts))
    p("  Images with OCR text: %d" % ocr_found)
    p("  Images with texture:  %d" % texture_found)
    p("  Detection labels (%d unique):" % len(labels))
    for lbl, cnt in labels.most_common(15):
        p("    %-20s %4d" % (lbl, cnt))

# Quality issues
p("")
p("-" * 50)
p("POTENTIAL QUALITY ISSUES")
p("-" * 50)
issues = []
empty_ans = sum(1 for c in challenges if not c.get("correct_answer"))
if empty_ans:
    issues.append("Empty correct answers: %d" % empty_ans)
varies_ans = sum(1 for c in challenges if str(c.get("correct_answer", "")).lower() == "varies")
if varies_ans:
    issues.append("'varies' answers (non-concrete): %d" % varies_ans)
same_img = sum(1 for c in challenges if c["image_a_id"] == c["image_b_id"])
p("  Same image A and B: %d/%d (%d%%)" % (same_img, len(challenges), same_img * 100 // len(challenges)))
expected = {"spatial_relations", "compositional_binding", "counting", "scale_invariance",
            "texture", "compound", "temporal_reasoning", "text_in_image",
            "physical_plausibility", "spurious_correlation",
            "typography_conflict", "occlusion_gradient",
            "cultural_visual_bias", "temporal_consistency"}
missing = expected - set(bc.keys())
if missing:
    issues.append("Missing bias types: %s" % str(missing))
for bt, cnt in bc.items():
    if cnt < 10:
        issues.append("Low count for '%s': only %d challenges" % (bt, cnt))
if issues:
    for i in issues:
        p("  WARNING: %s" % i)
else:
    p("  No critical issues found.")

p("")
p("=" * 70)
p("  ANALYSIS COMPLETE - %d challenges analyzed" % len(challenges))
p("=" * 70)

# Write report
report_text = "\n".join(lines)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_text)

print("Report written to: %s" % REPORT)
print("Lines: %d" % len(lines))

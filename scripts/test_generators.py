"""Quick generator quality test - all 14 generators."""
import sys, os, random
sys.path.insert(0, ".")
random.seed(42)

FAKE_ANN = {
    "image_id": "test_img_001",
    "detections": {
        "detections": [
            {"label": "car", "bbox": [100, 150, 300, 350], "confidence": 0.92},
            {"label": "person", "bbox": [400, 100, 500, 400], "confidence": 0.88},
            {"label": "dog", "bbox": [50, 300, 200, 450], "confidence": 0.75},
            {"label": "bicycle", "bbox": [500, 200, 600, 350], "confidence": 0.70},
            {"label": "car", "bbox": [320, 160, 450, 310], "confidence": 0.65},
        ],
    },
    "ocr": {
        "text_blocks": [
            {"text": "STOP", "bbox": [10, 10, 80, 40]},
            {"text": "SPEED LIMIT 30", "bbox": [100, 10, 250, 40]},
        ]
    },
    "time_of_day": "afternoon",
    "weather": "clear",
    "depth": {
        "left_depth_mean": 0.4, "right_depth_mean": 0.6,
        "top_depth_mean": 0.3, "bottom_depth_mean": 0.7,
    },
    "texture": {"edge_density": 0.12},
}

ann_list = [FAKE_ANN, FAKE_ANN]
out = open(os.path.join("scripts", "result.txt"), "w", encoding="utf-8")

def log(msg):
    out.write(msg + "\n")
    out.flush()

generators = [
    ("temporal_generator",           "TemporalGenerator"),
    ("spurious_generator",           "SpuriousGenerator"),
    ("scale_generator",              "ScaleGenerator"),
    ("counting_generator",           "CountingGenerator"),
    ("spatial_generator",            "SpatialGenerator"),
    ("texture_generator",            "TextureGenerator"),
    ("physics_generator",            "PhysicsGenerator"),
    ("text_image_generator",         "TextImageGenerator"),
    ("compositional_generator",      "CompositionalGenerator"),
    ("occlusion_gradient_generator", "OcclusionGradientGenerator"),
    ("cultural_bias_generator",      "CulturalBiasGenerator"),
    ("compound_generator",           "CompoundGenerator"),
    ("temporal_consistency_generator","TemporalConsistencyGenerator"),
    ("typography_conflict_generator","TypographyConflictGenerator"),
]

log("=" * 65)
log("  GENERATOR QUALITY VERIFICATION")
log("=" * 65)

all_pass = True
for mod_name, cls_name in generators:
    try:
        mod = __import__("src.generators." + mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
    except Exception as e:
        log("[FAIL] %-40s IMPORT ERROR: %s" % (cls_name, e))
        all_pass = False
        continue

    gen = cls()
    challenges = []
    for _ in range(30):
        try:
            c = gen.generate_challenge(ann_list)
            if c:
                challenges.append(c)
        except Exception as e:
            log("[FAIL] %-40s EXCEPTION: %s" % (cls_name, e))
            all_pass = False
            break
    else:
        if not challenges:
            log("[FAIL] %-40s No challenges generated" % cls_name)
            all_pass = False
            continue

        n = len(challenges)
        n_unique = len(set(c.question_template for c in challenges))
        dist_counts = [len(c.distractor_answers or []) for c in challenges]
        min_dist = min(dist_counts)
        avg_dist = sum(dist_counts) / n

        issues = []
        if min_dist < 3:
            issues.append("min distractors=%d" % min_dist)
        
        # Check correct answer in distractors
        ans_in_dist = sum(1 for c in challenges if c.correct_answer in (c.distractor_answers or []))
        if ans_in_dist:
            issues.append("correct_in_distractors=%d" % ans_in_dist)

        if issues:
            log("[FAIL] %-40s %d challenges, %d templates, ISSUES: %s" % (
                cls_name, n, n_unique, "; ".join(issues)))
            all_pass = False
        else:
            log("[OK]   %-40s %2d challenges, %2d unique templates, avg %.1f distractors" % (
                cls_name, n, n_unique, avg_dist))

log("")
log("=" * 65)
if all_pass:
    log("  ALL 14 GENERATORS PASSED")
else:
    log("  SOME GENERATORS HAVE ISSUES")
log("=" * 65)
out.close()

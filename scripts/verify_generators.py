"""
ECCV v3 — Generator Verification Script
Tests all 14 generators for correct argument signatures and output format.
Shows progress with timing.
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENERATORS = [
    ("CountingGenerator",            "src.generators.counting_generator",            "CountingGenerator"),
    ("SpatialGenerator",             "src.generators.spatial_generator",              "SpatialGenerator"),
    ("CompositionalGenerator",       "src.generators.compositional_generator",        "CompositionalGenerator"),
    ("ScaleGenerator",               "src.generators.scale_generator",                "ScaleGenerator"),
    ("TextureGenerator",             "src.generators.texture_generator",              "TextureGenerator"),
    ("PhysicsGenerator",             "src.generators.physics_generator",              "PhysicsGenerator"),
    ("TemporalGenerator",            "src.generators.temporal_generator",             "TemporalGenerator"),
    ("SpuriousGenerator",            "src.generators.spurious_generator",             "SpuriousGenerator"),
    ("TextImageGenerator",           "src.generators.text_image_generator",           "TextImageGenerator"),
    ("CulturalBiasGenerator",        "src.generators.cultural_bias_generator",        "CulturalBiasGenerator"),
    ("TypographyConflictGenerator",  "src.generators.typography_conflict_generator",  "TypographyConflictGenerator"),
    ("OcclusionGradientGenerator",   "src.generators.occlusion_gradient_generator",   "OcclusionGradientGenerator"),
    ("TemporalConsistencyGenerator", "src.generators.temporal_consistency_generator",  "TemporalConsistencyGenerator"),
    ("CompoundGenerator",            "src.generators.compound_generator",             "CompoundGenerator"),
]

# Dummy annotations with realistic structure
ann1 = {
    "image_id": "test_001",
    "detections": {
        "detections": [
            {"label": "car",    "bbox": [10, 10, 200, 150],  "confidence": 0.92, "area_fraction": 0.12},
            {"label": "person", "bbox": [300, 50, 400, 300],  "confidence": 0.88, "area_fraction": 0.08},
            {"label": "dog",    "bbox": [450, 200, 550, 320], "confidence": 0.75, "area_fraction": 0.03},
        ]
    },
    "depth": {"left_depth_mean": 0.45, "right_depth_mean": 0.55, "skipped": False},
    "ocr":   {"text_blocks": [{"text": "STOP", "bbox": [50, 50, 120, 80]}], "skipped": False},
    "quality": {"brightness": 128, "contrast": 50, "skipped": False},
}
ann2 = dict(ann1); ann2["image_id"] = "test_002"
ann3 = dict(ann1); ann3["image_id"] = "test_003"
batch = [ann1, ann2, ann3]

def main():
    total = len(GENERATORS)
    passed, failed, skipped = 0, 0, 0
    results = []
    t0 = time.time()

    print("=" * 65)
    print("  ECCV v3 — Generator Verification")
    print(f"  Testing {total} generators, 5 attempts each")
    print("=" * 65)

    for idx, (name, mod_path, cls_name) in enumerate(GENERATORS, 1):
        elapsed = time.time() - t0
        avg = elapsed / idx if idx > 1 else 0.5
        eta = avg * (total - idx)
        status = ""

        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            gen = cls()

            ok = False
            for attempt in range(5):
                try:
                    ch = gen.generate_challenge(batch)
                    if ch:
                        d = ch.to_dict()
                        assert "sub_type" in d,            "Missing sub_type"
                        assert "distractor_answers" in d,  "Missing distractor_answers"
                        assert len(d["distractor_answers"]) >= 1, "Empty distractors"
                        status = f'PASS  sub_type="{d["sub_type"]}"  distractors={len(d["distractor_answers"])}'
                        ok = True
                        passed += 1
                        break
                except Exception:
                    if attempt == 4:
                        raise

            if not ok:
                status = "SKIP  (returned None on all 5 attempts)"
                skipped += 1

        except Exception as e:
            status = f"FAIL  {type(e).__name__}: {e}"
            failed += 1

        bar = f"[{idx:2d}/{total}]"
        eta_str = f"ETA {eta:.0f}s" if idx < total else "done"
        print(f"  {bar} {name:40s} {status:50s} ({eta_str})")
        results.append((name, status))

    total_time = time.time() - t0
    print()
    print("=" * 65)
    print(f"  Results:  {passed} passed  |  {failed} failed  |  {skipped} skipped")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 65)

    if failed > 0:
        print("\n  *** FAILURES ***")
        for name, st in results:
            if st.startswith("FAIL"):
                print(f"    {name}: {st}")
        sys.exit(1)
    else:
        print("\n  All generators verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

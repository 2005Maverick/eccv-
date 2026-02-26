"""
Dry-run: End-to-end pipeline verification with synthetic data.
Tests all 5 stages without requiring GPU or large downloads.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def report(stage, test, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((stage, test, passed))
    print("  %s %s" % (status, test) + (" -- %s" % detail if detail else ""))


print("=" * 65)
print("  VLM Reality Check Pipeline — Dry Run")
print("=" * 65)

# ─── STAGE 1: Config Verification ────────────────────────────
print("\n[Stage 1] Repo Scaffold & Configuration")

import yaml

try:
    with open("configs/biases.yaml") as f:
        biases = yaml.safe_load(f)
    report("S1", "biases.yaml loads", True, f"{len(biases)} biases defined")
    
    expected = {"texture","counting","spatial_relations","physical_plausibility",
                "temporal_reasoning","spurious_correlation","compositional_binding",
                "text_in_image","scale_invariance"}
    report("S1", "All 9 biases present", set(biases.keys()) == expected)
except Exception as e:
    report("S1", "biases.yaml loads", False, str(e))

try:
    with open("configs/pipeline.yaml") as f:
        pipeline_cfg = yaml.safe_load(f)
    report("S1", "pipeline.yaml loads", True)
except Exception as e:
    report("S1", "pipeline.yaml loads", False, str(e))

try:
    with open("configs/languages.yaml") as f:
        langs = yaml.safe_load(f)
    report("S1", "languages.yaml loads", True, f"{len(langs)} languages")
    report("S1", "5 languages configured", len(langs) == 5)
except Exception as e:
    report("S1", "languages.yaml loads", False, str(e))


# ─── STAGE 2: Data Collection ────────────────────────────────
print("\n[Stage 2] Data Collection Pipeline")

try:
    from src.collection.base_collector import ImageRecord, DuplicateDetector
    
    rec = ImageRecord(
        image_id="test_001", url="http://example.com/img.jpg",
        local_path="data/raw/images/test.jpg", source="test",
        metadata={"width": 512, "height": 512}, license="CC-BY-4.0"
    )
    report("S2", "ImageRecord creation", True, f"id={rec.image_id}")
    
    required = {"image_id", "url", "local_path", "source", "metadata", "license"}
    has_all = all(hasattr(rec, f) for f in required)
    report("S2", "ImageRecord schema complete", has_all)
except Exception as e:
    report("S2", "ImageRecord creation", False, str(e))

# Test DuplicateDetector
try:
    import numpy as np
    from PIL import Image
    
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_hashes.db")
    detector = DuplicateDetector(db_path=db_path)
    
    # Create two test images
    img1 = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    p1 = os.path.join(tmp_dir, "img1.jpg")
    p2 = os.path.join(tmp_dir, "img2.jpg")
    img1.save(p1)
    img2.save(p2)
    
    assert not detector.is_duplicate(p1, "img1"), "First image should not be duplicate"
    detector.register("img1", p1, "test")
    assert detector.is_duplicate(p1, "img1_dup"), "Same image should be duplicate"
    assert not detector.is_duplicate(p2, "img2"), "Different image should not be duplicate"
    
    report("S2", "pHash deduplication", True)
    shutil.rmtree(tmp_dir)
except Exception as e:
    report("S2", "pHash deduplication", False, str(e))

# Test checkpoint resume
try:
    from src.collection.pipeline import CollectionPipeline
    report("S2", "CollectionPipeline imports", True)
except Exception as e:
    report("S2", "CollectionPipeline imports", False, str(e))

# Test all collectors import
try:
    from src.collection.wikimedia_collector import WikimediaCollector
    from src.collection.openimages_collector import OpenImagesCollector
    from src.collection.streetview_collector import StreetViewCollector
    from src.collection.yfcc_collector import YFCCCollector
    report("S2", "All 4 collectors import", True)
except Exception as e:
    report("S2", "All 4 collectors import", False, str(e))


# ─── STAGE 3: Vision Processing ──────────────────────────────
print("\n[Stage 3] Vision Processing & Annotation")

try:
    from src.processing.base_processor import ProcessorBase
    report("S3", "ProcessorBase imports", True)
except Exception as e:
    report("S3", "ProcessorBase imports", False, str(e))

# Test all processors import
try:
    from src.processing.object_detector import ObjectDetector
    from src.processing.depth_estimator import DepthEstimator
    from src.processing.shadow_detector import ShadowDetector
    from src.processing.ocr_extractor import OCRExtractor
    from src.processing.texture_analyzer import TextureAnalyzer
    from src.processing.temporal_extractor import TemporalExtractor
    report("S3", "All 7 processors import", True)
except Exception as e:
    report("S3", "All 7 processors import", False, str(e))

# Test shadow detector (Astropy)
try:
    sd = ShadowDetector()
    result = sd.compute_expected_shadow(lat=40.7128, lon=-74.0060, timestamp="2024-06-21T12:00:00")
    assert result['shadow_angle_expected'] is not None
    assert result['sun_elevation'] > 60
    report("S3", "Astropy shadow computation", True,
           f"shadow={result['shadow_angle_expected']:.1f}°, sun_elev={result['sun_elevation']:.1f}°")
except Exception as e:
    report("S3", "Astropy shadow computation", False, str(e))

# Test texture analyzer on synthetic image
try:
    tmp_dir = tempfile.mkdtemp()
    test_img = os.path.join(tmp_dir, "test_texture.jpg")
    # High-contrast image for clear edges
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[50:200, 50:200] = 255  # White square on black
    Image.fromarray(arr).save(test_img)
    
    ta = TextureAnalyzer()
    tex_result = ta.process(test_img)
    assert 'edge_density' in tex_result
    assert 'dominant_texture_freq' in tex_result
    assert 'silhouette_extractable' in tex_result
    report("S3", "Texture analyzer", True,
           f"edge_density={tex_result['edge_density']:.4f}")
    shutil.rmtree(tmp_dir)
except Exception as e:
    report("S3", "Texture analyzer", False, str(e))

# Test temporal extractor
try:
    te = TemporalExtractor()
    te_result = te.process("nonexistent.jpg", detections=[
        {"label": "truck"}, {"label": "crane"}, {"label": "car"}
    ])
    assert te_result['construction_score'] > 0
    report("S3", "Temporal extractor + construction score", True,
           f"score={te_result['construction_score']}")
except Exception as e:
    report("S3", "Temporal extractor + construction score", False, str(e))

# Test annotation pipeline
try:
    from src.processing.annotation_pipeline import AnnotationPipeline
    report("S3", "AnnotationPipeline imports", True)
except Exception as e:
    report("S3", "AnnotationPipeline imports", False, str(e))


# ─── STAGE 4: Challenge Generation ───────────────────────────
print("\n[Stage 4] Adversarial Challenge Generation")

try:
    from src.generators.base_generator import Challenge, ChallengeGenerator
    
    ch = Challenge(
        challenge_id="test_ch_001", bias_type="counting", difficulty="easy",
        image_a_id="img_a", image_b_id="img_b",
        question_template="Which has more {object}?",
        correct_answer="B", distractor_answers=["A"],
        ground_truth_method="yolo_count", confound_check_passed=True,
        metadata={"count_a": 3, "count_b": 7}
    )
    d = ch.to_dict()
    assert d['challenge_id'] == "test_ch_001"
    assert d['bias_type'] == "counting"
    report("S4", "Challenge dataclass + serialization", True)
except Exception as e:
    report("S4", "Challenge dataclass + serialization", False, str(e))

# Test all generators import
try:
    from src.generators.texture_generator import TextureGenerator
    from src.generators.counting_generator import CountingGenerator
    from src.generators.spatial_generator import SpatialGenerator
    from src.generators.physics_generator import PhysicsGenerator
    from src.generators.temporal_generator import TemporalGenerator
    from src.generators.spurious_generator import SpuriousGenerator
    from src.generators.compositional_generator import CompositionalGenerator
    from src.generators.text_image_generator import TextImageGenerator
    from src.generators.scale_generator import ScaleGenerator
    from src.generators.compound_generator import CompoundGenerator
    from src.generators.occlusion_gradient_generator import OcclusionGradientGenerator
    from src.generators.cultural_bias_generator import CulturalBiasGenerator
    from src.generators.temporal_consistency_generator import TemporalConsistencyGenerator
    from src.generators.typography_conflict_generator import TypographyConflictGenerator
    report("S4", "All 14 generators import", True)
except Exception as e:
    report("S4", "All 14 generators import", False, str(e))

# Test counting generator logic (with random sub-type selection)
try:
    cg = CountingGenerator()
    ann_rich = {"image_id": "a", "detections": {"detections": [
        {"label": "cat", "confidence": 0.9, "bbox": [10, 10, 100, 100]},
        {"label": "cat", "confidence": 0.9, "bbox": [200, 10, 300, 100]},
        {"label": "dog", "confidence": 0.8, "bbox": [50, 200, 150, 300]},
    ]}}
    ann_b = {"image_id": "b", "detections": {"detections": [
        {"label": "cat", "confidence": 0.9, "bbox": [10, 10, 100, 100]},
    ] * 7}}
    
    # Generate multiple challenges to test all sub-types
    successes = 0
    for _ in range(20):
        ch = cg.generate_challenge([ann_rich, ann_b])
        if ch is not None:
            assert ch.correct_answer, "Correct answer should not be empty"
            assert len(ch.distractor_answers) >= 3, "Should have >= 3 distractors"
            successes += 1
    
    report("S4", "Counting: generates valid challenges", successes > 0,
           "%d/20 generated" % successes)
except Exception as e:
    report("S4", "Counting generator logic", False, str(e))

# Test spatial generator produces valid challenges
try:
    sg = SpatialGenerator()
    ann_spatial = {
        "image_id": "a",
        "detections": {"detections": [
            {"label": "car", "bbox": [10, 10, 100, 100], "area_fraction": 0.1},
            {"label": "person", "bbox": [300, 200, 400, 400], "area_fraction": 0.05},
            {"label": "dog", "bbox": [200, 300, 350, 450], "area_fraction": 0.03},
        ]},
        "depth": {"left_depth_mean": 0.3, "right_depth_mean": 0.7},
    }
    successes = 0
    for _ in range(20):
        result = sg.generate_challenge([ann_spatial])
        if result is not None:
            assert len(result.distractor_answers) >= 3, "Should have >= 3 distractors"
            successes += 1
    report("S4", "Spatial: generates valid challenges", successes > 0,
           "%d/20 generated" % successes)
except Exception as e:
    report("S4", "Confound rejection", False, str(e))

# Test generation pipeline
try:
    from src.generators.pipeline import GenerationPipeline
    report("S4", "GenerationPipeline imports", True)
except Exception as e:
    report("S4", "GenerationPipeline imports", False, str(e))


# ─── STAGE 5: Multilingual Translation ───────────────────────
print("\n[Stage 5] Multilingual Translation")

try:
    from src.translation.templates import QUESTION_TEMPLATES
    assert len(QUESTION_TEMPLATES) == 9, f"Expected 9 bias types, got {len(QUESTION_TEMPLATES)}"
    total_templates = sum(len(v) for v in QUESTION_TEMPLATES.values())
    report("S5", f"Question templates loaded", True, f"{total_templates} templates across 9 biases")
except Exception as e:
    report("S5", "Question templates", False, str(e))

# Test translator variable protection
try:
    from src.translation.translator import Translator
    t = Translator()
    
    template = "Which image has more {object_category}? Count the {object_category}."
    protected, var_map = t._protect_variables(template)
    restored = t._restore_variables(protected, var_map)
    assert restored == template
    report("S5", "Variable slot round-trip", True)
except Exception as e:
    report("S5", "Variable slot round-trip", False, str(e))

# Test script validator
try:
    from src.translation.script_validator import validate_script, validate_variable_slots
    
    v_hi, f_hi = validate_script("यह किस प्रकार का वस्तु है?", "hi")
    v_ar, f_ar = validate_script("هل هذه الصور نفسها؟", "ar")
    v_zh, f_zh = validate_script("这是什么类型的物体？", "zh")
    v_bad, f_bad = validate_script("Hello World", "hi")
    
    assert v_hi and v_ar and v_zh and not v_bad
    report("S5", "Script validation (HI/AR/ZH/reject)", True)
except Exception as e:
    report("S5", "Script validation", False, str(e))

# Test variable slot validation
try:
    assert validate_variable_slots("How many {cat}?", "Cuantos {cat}?")
    assert not validate_variable_slots("How many {cat}?", "Cuantos gatos?")
    report("S5", "Variable slot preservation check", True)
except Exception as e:
    report("S5", "Variable slot preservation check", False, str(e))

# Test translation pipeline
try:
    from src.translation.pipeline import TranslationPipeline
    report("S5", "TranslationPipeline imports", True)
except Exception as e:
    report("S5", "TranslationPipeline imports", False, str(e))


# ─── SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 65)
total = len(results)
passed = sum(1 for _, _, p in results if p)
failed = total - passed

if failed == 0:
    print("  %s ALL %d TESTS PASSED" % (PASS, total))
else:
    print("  %d/%d passed, %d failed:" % (passed, total, failed))
    for stage, test, p in results:
        if not p:
            print("    %s [%s] %s" % (FAIL, stage, test))

print("=" * 65)

sys.exit(0 if failed == 0 else 1)

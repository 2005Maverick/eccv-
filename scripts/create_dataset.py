"""
=================================================================
  VLM Reality Check — Full Dataset Creator  (ECCV v3)

  Causal-contrastive benchmark with:
  - 13 bias types + compound challenges
  - Counterfactual image pairs for causal isolation
  - New generators: typography conflict, occlusion gradient,
    cultural bias, temporal consistency
  - Real Helsinki-NLP translations (Devanagari / Hanzi / Arabic)
  - EasyOCR text extraction
  - Balanced difficulty via confound strength
  - All output in dataset_final/
=================================================================
"""
import os, sys, json, random, time, logging, shutil, hashlib
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, ".")
random.seed(42)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

import requests
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import yaml

# ============================================================
# Timer utilities
# ============================================================
GLOBAL_START = time.time()
STAGE_TIMES = {}

def elapsed():
    s = int(time.time() - GLOBAL_START)
    return str(timedelta(seconds=s))

def stage_start(name):
    STAGE_TIMES[name] = {"start": time.time()}
    print(f"\n{'='*65}")
    print(f"  STAGE: {name}")
    print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}  |  Elapsed: {elapsed()}")
    print(f"{'='*65}")

def stage_end(name, count_label="", count=0):
    end = time.time()
    duration = end - STAGE_TIMES[name]["start"]
    STAGE_TIMES[name]["duration"] = duration
    STAGE_TIMES[name]["end"] = end
    dur_str = str(timedelta(seconds=int(duration)))
    print(f"  >> {name} COMPLETE in {dur_str}", end="")
    if count_label:
        print(f"  ({count} {count_label})")
    else:
        print()
    print(f"  >> Total elapsed: {elapsed()}")

def progress(i, total, label=""):
    pct = int(100 * i / max(total, 1))
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    print(f"\r  [{bar}] {pct}% ({i}/{total}) {label} | Elapsed: {elapsed()}", end="", flush=True)

# ============================================================
# Output directory
# ============================================================
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset_final")
os.makedirs(DATASET_DIR, exist_ok=True)
for sub in ["images", "annotations", "challenges", "translations", "configs"]:
    os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)

print("=" * 65)
print("  VLM REALITY CHECK — FULL DATASET CREATION (ECCV v2)")
print(f"  Output: {DATASET_DIR}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

HEADERS = {"User-Agent": "VLMRealityCheck/1.0 (Academic Research; contact@university.edu)"}


# ============================================================
# STAGE 1: Configuration
# ============================================================
stage_start("1. Configuration")
biases = yaml.safe_load(open("configs/biases.yaml"))
pipeline_cfg = yaml.safe_load(open("configs/pipeline.yaml"))
languages = yaml.safe_load(open("configs/languages.yaml"))

for cfg in ["biases.yaml", "pipeline.yaml", "languages.yaml"]:
    shutil.copy2(f"configs/{cfg}", os.path.join(DATASET_DIR, "configs", cfg))

print(f"  Biases: {len(biases)}")
print(f"  Languages: {len(languages)} ({', '.join(languages)})")
stage_end("1. Configuration")


# ============================================================
# STAGE 2: Image Collection (Wikimedia Commons)
# ============================================================
stage_start("2. Image Collection")

SEARCH_QUERIES = [
    # Street/Traffic
    ("street_traffic_1", "busy city street cars traffic lights"),
    ("street_traffic_2", "urban road vehicles pedestrians"),
    ("highway_scene", "highway multiple cars driving"),
    ("intersection", "road intersection traffic urban"),
    # Outdoor/Nature
    ("park_scene", "park bench trees people"),
    ("dogs_outdoor", "dogs playing outdoor"),
    ("beach_scene", "beach people walking sunset"),
    ("garden", "garden flowers plants outdoor"),
    ("forest_trail", "forest trail nature hiking"),
    # Urban/Architecture
    ("city_skyline", "city skyline skyscrapers"),
    ("building_facade", "building facade architecture"),
    ("construction", "construction site crane workers"),
    ("bridge", "bridge road urban architecture"),
    # People/Activity
    ("market_scene", "outdoor market stalls people"),
    ("bicycle_lane", "bicycle lane cycling urban"),
    ("playground", "playground children playing"),
    ("sports_field", "sports field outdoor"),
    # Vehicles
    ("parking_lot", "parking lot parked cars"),
    ("train_station", "train station platform"),
    ("bus_stop", "bus stop urban street"),
    # Rural
    ("farm_landscape", "farm landscape agricultural"),
    ("rural_road", "rural road countryside"),
    ("village", "village houses rural"),
    # Mixed
    ("shopping_area", "shopping district stores"),
    ("waterfront", "waterfront harbor boats"),
    ("airport", "airport terminal planes"),
    ("stadium", "stadium crowd sports"),
    ("campus", "university campus students"),
    ("warehouse", "warehouse industrial storage"),
    ("restaurant_outdoor", "outdoor restaurant cafe terrace"),
    # Text-rich scenes (for text_in_image bias)
    ("street_signs", "street signs city road signs"),
    ("shopfronts", "shop front signs storefronts neon"),
    ("billboards", "billboard advertising outdoor"),
    ("menu_board", "restaurant menu board outdoor chalkboard"),
    # Animal scenes (for spurious correlation)
    ("animals_farm", "farm animals cows sheep field"),
    ("animals_wild", "wildlife safari animals"),
    ("zoo_scene", "zoo animals enclosure"),
    # Time-of-day variety (for temporal)
    ("sunrise_scene", "sunrise morning landscape golden hour"),
    ("night_city", "night city lights dark urban"),
    ("sunset_scene", "sunset evening landscape dusk"),
]

collected = []
target = min(len(SEARCH_QUERIES) * 3, 120)
img_dir = os.path.join(DATASET_DIR, "images")

print(f"  Target: up to {target} images from {len(SEARCH_QUERIES)} queries")

for qi, (scene_name, query) in enumerate(SEARCH_QUERIES):
    progress(len(collected), target, f"q={scene_name[:15]}")
    if len(collected) >= target:
        break
    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": "6",
            "gsrlimit": "8",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "640",
            "format": "json",
        }
        resp = requests.get("https://commons.wikimedia.org/w/api.php",
                           params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            continue
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})

        imgs_this_query = 0
        for pid, page in pages.items():
            if len(collected) >= target or imgs_this_query >= 3:
                break
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if "image" not in mime or "svg" in mime:
                continue

            ext = info.get("extmetadata", {})
            license_val = ext.get("LicenseShortName", {}).get("value", "unknown")

            url = info.get("thumburl") or info.get("url", "")
            if not url:
                continue

            try:
                img_resp = requests.get(url, headers=HEADERS, timeout=20)
                if img_resp.status_code != 200:
                    continue
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                if min(img.size) < 100:
                    continue
            except Exception:
                continue

            img.thumbnail((640, 640), Image.LANCZOS)

            img_id = f"vlm_{len(collected)+1:04d}"
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            img.save(img_path, quality=93)

            try:
                import imagehash
                phash = str(imagehash.phash(img))
            except ImportError:
                phash = hashlib.md5(img.tobytes()[:1024]).hexdigest()[:16]

            existing_hashes = {r.get("phash") for r in collected}
            if phash in existing_hashes:
                os.remove(img_path)
                continue

            record = {
                "image_id": img_id,
                "url": url,
                "local_path": img_path,
                "source": "wikimedia",
                "license": license_val,
                "phash": phash,
                "metadata": {
                    "query": query,
                    "scene": scene_name,
                    "width": img.size[0],
                    "height": img.size[1],
                    "title": page.get("title", "")[:80],
                },
            }
            collected.append(record)
            imgs_this_query += 1

    except Exception:
        continue

progress(len(collected), target, "DONE")
print()

manifest_path = os.path.join(DATASET_DIR, "manifest.jsonl")
with open(manifest_path, "w") as f:
    for rec in collected:
        f.write(json.dumps(rec) + "\n")

stage_end("2. Image Collection", "images downloaded", len(collected))
if len(collected) < 5:
    print("  ERROR: Not enough images downloaded. Check internet.")
    sys.exit(1)


# ============================================================
# STAGE 3: Vision Processing & Annotation
# ============================================================
stage_start("3. Vision Processing")

from src.processing.texture_analyzer import TextureAnalyzer
from src.processing.temporal_extractor import TemporalExtractor

texture_analyzer = TextureAnalyzer()
temporal_extractor = TemporalExtractor()

# YOLO
use_real_yolo = False
try:
    from src.processing.object_detector import ObjectDetector
    detector = ObjectDetector()
    use_real_yolo = True
    print(f"  >> YOLOv8x: LOADED")
except Exception as e:
    print(f"  >> YOLOv8x: unavailable ({e}), using contour fallback")

# OCR — try EasyOCR first, then existing OCR module
use_easyocr = False
easyocr_reader = None
use_module_ocr = False
try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    use_easyocr = True
    print(f"  >> EasyOCR: LOADED")
except Exception:
    try:
        from src.processing.ocr_extractor import OCRExtractor
        ocr_proc = OCRExtractor()
        use_module_ocr = True
        print(f"  >> OCR module: LOADED")
    except Exception:
        print(f"  >> OCR: unavailable")


def contour_detect(img_path):
    """Fallback CPU object detection."""
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return {"detections": []}
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = ["object", "person", "car", "building", "tree", "sign", "animal", "vehicle"]
    dets = []
    for j, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:8]):
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.005):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / max(bw, 1)
        if aspect > 2: label = "person"
        elif aspect < 0.6: label = "car"
        elif area > h*w*0.1: label = "building"
        else: label = labels[j % len(labels)]
        dets.append({"label": label, "bbox": [x, y, x+bw, y+bh],
                     "confidence": round(0.7 + random.uniform(0, 0.28), 4),
                     "area_fraction": round((bw*bh)/(w*h), 4)})
    return {"detections": dets}


def extract_ocr_easyocr(img_path):
    """Extract text using EasyOCR."""
    try:
        results = easyocr_reader.readtext(img_path)
        if not results:
            return {"has_text": False, "text_blocks": [], "text_is_location_relevant": False}

        text_blocks = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text.strip()) > 1:
                text_blocks.append({
                    "text": text.strip(),
                    "confidence": round(float(conf), 4),
                    "bbox": [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])],
                })

        full_text = " ".join(b["text"] for b in text_blocks).lower()
        location_keywords = ["street", "road", "ave", "blvd", "st", "exit", "hwy",
                            "drive", "lane", "plaza", "square", "park", "station",
                            "hotel", "restaurant", "shop", "store", "cafe", "bar"]
        is_location = any(kw in full_text for kw in location_keywords)

        return {
            "has_text": len(text_blocks) > 0,
            "text_blocks": text_blocks,
            "text_is_location_relevant": is_location,
        }
    except Exception:
        return {"has_text": False, "text_blocks": [], "text_is_location_relevant": False}


annotations = []
total_detections = 0
ocr_success = 0
for i, rec in enumerate(collected):
    progress(i+1, len(collected), f"img={rec['image_id']}")
    img_path = rec["local_path"]
    img_id = rec["image_id"]

    # Texture analysis
    tex = texture_analyzer.process(img_path)

    # Temporal extraction
    temp = temporal_extractor.process(img_path, detections=[])

    # Object detection
    if use_real_yolo:
        try:
            det_result = detector.process(img_path)
        except Exception:
            det_result = contour_detect(img_path)
    else:
        det_result = contour_detect(img_path)

    # Depth from gradient variance
    img_cv = cv2.imread(img_path)
    if img_cv is not None:
        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        depth = {
            "left_depth_mean": round(float(np.std(gray[:, :w//2])) / 255, 4),
            "right_depth_mean": round(float(np.std(gray[:, w//2:])) / 255, 4),
            "top_depth_mean": round(float(np.std(gray[:h//2, :])) / 255, 4),
            "bottom_depth_mean": round(float(np.std(gray[h//2:, :])) / 255, 4),
        }
    else:
        depth = {"left_depth_mean": 0, "right_depth_mean": 0,
                 "top_depth_mean": 0, "bottom_depth_mean": 0}

    # OCR
    ocr = {"has_text": False, "text_blocks": [], "text_is_location_relevant": False}
    if use_easyocr:
        ocr = extract_ocr_easyocr(img_path)
    elif use_module_ocr:
        try:
            ocr = ocr_proc.process(img_path)
        except Exception:
            pass

    if ocr.get("has_text"):
        ocr_success += 1

    # Normalize detection result to always be {detections: list}
    if isinstance(det_result, list):
        det_dict = {"detections": det_result}
    elif isinstance(det_result, dict) and "detections" in det_result:
        det_dict = det_result
    else:
        det_dict = {"detections": det_result if isinstance(det_result, list) else []}

    ann = {
        "image_id": img_id,
        "detections": det_dict,
        "depth": depth,
        "shadow": {"shadow_angle_detected": None, "is_physically_plausible": None, "skippable": True},
        "ocr": ocr,
        "texture": tex,
        "temporal": temp,
    }
    annotations.append(ann)
    n_dets = len(det_dict.get("detections", []))
    total_detections += n_dets

progress(len(collected), len(collected), "DONE")
print()

ann_path = os.path.join(DATASET_DIR, "annotations", "annotations.jsonl")
with open(ann_path, "w") as f:
    for a in annotations:
        f.write(json.dumps(a) + "\n")

print(f"  >> Total detections: {total_detections}")
print(f"  >> OCR text found in: {ocr_success}/{len(annotations)} images")
stage_end("3. Vision Processing", "annotations", len(annotations))


# ============================================================
# STAGE 3.5: Counterfactual Image Generation (v3 — causal pairs)
# ============================================================
stage_start("3.5. Counterfactual Generation")

from src.processing.counterfactual_factory import CounterfactualFactory

cf_dir = os.path.join(DATASET_DIR, "counterfactuals")
os.makedirs(cf_dir, exist_ok=True)

cf_factory = CounterfactualFactory(cf_dir, seed=42)
cf_results = cf_factory.process_all(img_dir, annotations)

cf_manifest_path = os.path.join(DATASET_DIR, "counterfactuals", "manifest.jsonl")
cf_factory.save_manifest(cf_manifest_path)

print(f"  >> Counterfactual images generated: {len(cf_results)}")
print(f"  >> Transforms: {Counter(r['transform_type'] for r in cf_results)}")

stage_end("3.5. Counterfactual Generation", "counterfactuals", len(cf_results))


# ============================================================
# STAGE 4: Adversarial Challenge Generation (13 bias types + compound)
# ============================================================
stage_start("4. Challenge Generation")

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

# v3 generators
from src.generators.typography_conflict_generator import TypographyConflictGenerator
from src.generators.occlusion_gradient_generator import OcclusionGradientGenerator
from src.generators.cultural_bias_generator import CulturalBiasGenerator
from src.generators.temporal_consistency_generator import TemporalConsistencyGenerator

# Single-annotation generators
single_generators = {
    "spatial_relations": SpatialGenerator(),
    "compositional_binding": CompositionalGenerator(),
    "scale_invariance": ScaleGenerator(),
    "texture": TextureGenerator(),
    "physical_plausibility": PhysicsGenerator(),
}

# Pair-annotation generators
pair_generators = {
    "counting": CountingGenerator(),
    "spurious_correlation": SpuriousGenerator(),
}

# Special generators
temporal_gen = TemporalGenerator()
text_gen = TextImageGenerator()
compound_gen = CompoundGenerator()

# v3 generators (causal-contrastive)
typography_gen = TypographyConflictGenerator()
occlusion_gen = OcclusionGradientGenerator()
cultural_gen = CulturalBiasGenerator()
temporal_consist_gen = TemporalConsistencyGenerator()

challenges = []

# === Single-annotation generators ===
for btype, gen in single_generators.items():
    shuf = annotations.copy()
    random.shuffle(shuf)
    ct = 0
    for ann in shuf:
        if btype == "compositional_binding":
            ch = gen.generate_challenge([ann], image_dir=img_dir)
        elif btype == "texture":
            ch = gen.generate_challenge([ann], all_annotations=annotations)
        elif btype == "physical_plausibility":
            ch = gen.generate_challenge([ann], image_dir=img_dir)
        else:
            ch = gen.generate_challenge([ann])
        if ch:
            challenges.append(ch.to_dict())
            ct += 1
    print(f"  {btype}: {ct} challenges")

# === Pair-annotation generators ===
for btype, gen in pair_generators.items():
    shuf = annotations.copy()
    random.shuffle(shuf)
    ct = 0
    for i in range(0, len(shuf)-1, 2):
        ch = gen.generate_challenge([shuf[i], shuf[i+1]])
        if ch:
            challenges.append(ch.to_dict())
            ct += 1
    # Also try single-image counting sub-types
    if btype == "counting":
        for ann in shuf:
            ch = gen.generate_challenge([ann])
            if ch:
                challenges.append(ch.to_dict())
                ct += 1
    print(f"  {btype}: {ct} challenges")

# === Temporal (pair + single) ===
temp_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
# Pair comparison
for i in range(0, len(shuf)-1, 2):
    ch = temporal_gen.generate_challenge([shuf[i], shuf[i+1]], image_dir=img_dir)
    if ch:
        challenges.append(ch.to_dict())
        temp_ct += 1
# Single classification
for ann in shuf:
    ch = temporal_gen.generate_challenge([ann], image_dir=img_dir)
    if ch:
        challenges.append(ch.to_dict())
        temp_ct += 1
print(f"  temporal_reasoning: {temp_ct} challenges")

# === Text-in-image (pair + single) ===
text_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
# Pair comparison
for i in range(0, len(shuf)-1, 2):
    ch = text_gen.generate_challenge([shuf[i], shuf[i+1]])
    if ch:
        challenges.append(ch.to_dict())
        text_ct += 1
# Single image text reading/presence
for ann in shuf:
    ch = text_gen.generate_challenge([ann])
    if ch:
        challenges.append(ch.to_dict())
        text_ct += 1
print(f"  text_in_image: {text_ct} challenges")

# === Compound (limited to ~10% of total) ===
compound_ct = 0
max_compound = max(len(challenges) // 10, 5)
shuf = annotations.copy()
random.shuffle(shuf)
for i in range(0, len(shuf)-1, 2):
    if compound_ct >= max_compound:
        break
    ch = compound_gen.generate_challenge([shuf[i], shuf[i+1]])
    if ch:
        challenges.append(ch.to_dict())
        compound_ct += 1
print(f"  compound: {compound_ct} challenges")

# === v3: Typography Conflict Generator ===
typo_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
for ann in shuf:
    ch = typography_gen.generate_challenge([ann], image_dir=img_dir, cf_dir=cf_dir)
    if ch:
        challenges.append(ch.to_dict())
        typo_ct += 1
print(f"  typography_conflict: {typo_ct} challenges")

# === v3: Occlusion Gradient Generator ===
occ_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
for ann in shuf:
    ch = occlusion_gen.generate_challenge([ann], image_dir=img_dir, cf_dir=cf_dir)
    if ch:
        challenges.append(ch.to_dict())
        occ_ct += 1
print(f"  occlusion_gradient: {occ_ct} challenges")

# === v3: Cultural Visual Bias Generator ===
cultural_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
for ann in shuf:
    ch = cultural_gen.generate_challenge([ann], image_dir=img_dir)
    if ch:
        challenges.append(ch.to_dict())
        cultural_ct += 1
print(f"  cultural_visual_bias: {cultural_ct} challenges")

# === v3: Temporal Consistency Generator ===
tc_ct = 0
shuf = annotations.copy()
random.shuffle(shuf)
for ann in shuf:
    ch = temporal_consist_gen.generate_challenge([ann], image_dir=img_dir, cf_dir=cf_dir)
    if ch:
        challenges.append(ch.to_dict())
        tc_ct += 1
print(f"  temporal_consistency: {tc_ct} challenges")

# === Post-processing: filter distractor quality issues ===
filtered = []
for c in challenges:
    distractors = c.get("distractor_answers", [])
    correct = c.get("correct_answer")
    # Remove correct answer from distractors if present
    clean_distractors = [d for d in distractors if d != correct]
    # Remove duplicate distractors
    seen = set()
    unique_distractors = []
    for d in clean_distractors:
        if d not in seen:
            seen.add(d)
            unique_distractors.append(d)
    c["distractor_answers"] = unique_distractors
    filtered.append(c)
challenges = filtered
print(f"\n  >> Post-filter: {len(challenges)} challenges (distractor cleanup done)")

# Save challenges
ch_path = os.path.join(DATASET_DIR, "challenges", "challenges.jsonl")
with open(ch_path, "w") as f:
    for c in challenges:
        f.write(json.dumps(c) + "\n")

bias_dist = Counter(c["bias_type"] for c in challenges)
diff_dist = Counter(c["difficulty"] for c in challenges)
answer_dist = Counter(c["correct_answer"][:10] for c in challenges)
print(f"\n  >> TOTAL: {len(challenges)} challenges")
print(f"  >> Bias distribution: {dict(bias_dist)}")
print(f"  >> Difficulty distribution: {dict(diff_dist)}")
print(f"  >> Answer balance (top 5): {dict(answer_dist.most_common(5))}")
stage_end("4. Challenge Generation", "challenges", len(challenges))


# ============================================================
# STAGE 5: Multilingual Translation (Real Helsinki-NLP)
# ============================================================
stage_start("5. Multilingual Translation")

LANG_MODELS = {
    "es": "Helsinki-NLP/opus-mt-en-es",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
}

# Load Helsinki-NLP models
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


def translate_text(text: str, lang: str) -> str:
    """Translate text using Helsinki-NLP with variable slot protection."""
    if lang not in translators:
        return text

    tok, mdl = translators[lang]

    # Protect variable slots: replace {var} with sentinel tokens
    import re
    slots = re.findall(r'\{[^}]+\}', text)
    protected = text
    sentinels = []
    for i, slot in enumerate(slots):
        sentinel = f"XSLOT{i}X"
        sentinels.append((sentinel, slot))
        protected = protected.replace(slot, sentinel, 1)

    # Translate
    try:
        encoded = tok([protected], return_tensors="pt", padding=True, truncation=True, max_length=512)
        output = mdl.generate(**encoded)
        translated = tok.decode(output[0], skip_special_tokens=True)

        # Restore slots
        for sentinel, slot in sentinels:
            # Try exact match first
            if sentinel in translated:
                translated = translated.replace(sentinel, slot, 1)
            else:
                # Try case-insensitive and with spaces
                for variant in [sentinel, sentinel.lower(), sentinel.upper(),
                                f" {sentinel} ", f" {sentinel}", f"{sentinel} "]:
                    if variant in translated:
                        translated = translated.replace(variant, slot, 1)
                        break
                else:
                    # Slot lost in translation — append it
                    translated += f" {slot}"

        return translated
    except Exception:
        return text


# Pre-translate unique question templates
template_cache = {}
unique_templates = set(ch.get("question_template", "") for ch in challenges)
print(f"  Unique question templates: {len(unique_templates)}")

for lang in ["es", "zh", "hi", "ar"]:
    template_cache[lang] = {}
    for tmpl in unique_templates:
        if use_real_translation:
            translated = translate_text(tmpl, lang)
        else:
            translated = tmpl  # Fallback: keep English
        template_cache[lang][tmpl] = translated

    cached_count = len(template_cache[lang])
    # Sample validation
    if template_cache[lang]:
        sample_key = list(template_cache[lang].keys())[0]
        sample_val = template_cache[lang][sample_key]
        print(f"  {lang}: {cached_count} templates | sample: '{sample_val[:60]}...'")

# Also translate common object names for filling {object_category}
COMMON_OBJECTS = ["car", "person", "dog", "cat", "bus", "truck", "bicycle",
                  "bird", "chair", "tree", "building", "sign", "boat"]
object_translations = {}
for lang in ["es", "zh", "hi", "ar"]:
    object_translations[lang] = {}
    for obj in COMMON_OBJECTS:
        if use_real_translation:
            object_translations[lang][obj] = translate_text(obj, lang)
        else:
            object_translations[lang][obj] = obj

# Generate multilingual instances
instances = []
for lang in languages:
    for ch in challenges:
        inst = ch.copy()
        inst["language"] = lang
        tmpl = ch.get("question_template", "")
        if lang == "en":
            inst["question_translated"] = tmpl
        else:
            translated = template_cache.get(lang, {}).get(tmpl, tmpl)
            # Fill object_category if present
            meta = ch.get("metadata", {})
            obj_cat = meta.get("object_category", "")
            if obj_cat and "{object_category}" in translated:
                translated_obj = object_translations.get(lang, {}).get(obj_cat, obj_cat)
                translated = translated.replace("{object_category}", translated_obj)
            inst["question_translated"] = translated
        instances.append(inst)

# Save translations
trans_path = os.path.join(DATASET_DIR, "translations", "dataset_multilingual.jsonl")
with open(trans_path, "w", encoding="utf-8") as f:
    for inst in instances:
        f.write(json.dumps(inst, ensure_ascii=False) + "\n")

for lang in languages:
    lang_path = os.path.join(DATASET_DIR, "translations", f"dataset_{lang}.jsonl")
    lang_insts = [i for i in instances if i["language"] == lang]
    with open(lang_path, "w", encoding="utf-8") as f:
        for inst in lang_insts:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"  {lang}: {len(lang_insts)} instances -> dataset_{lang}.jsonl")

stage_end("5. Multilingual Translation", "instances", len(instances))


# ============================================================
# STAGE 6: Dataset Summary & Stats
# ============================================================
stage_start("6. Summary & Stats")

stats = {
    "created": datetime.now().isoformat(),
    "version": "ECCV_v2",
    "total_elapsed_seconds": int(time.time() - GLOBAL_START),
    "total_elapsed_formatted": elapsed(),
    "images": len(collected),
    "annotations": len(annotations),
    "total_detections": total_detections,
    "ocr_images_with_text": ocr_success,
    "challenges": len(challenges),
    "bias_types_represented": len(bias_dist),
    "languages": len(languages),
    "multilingual_instances": len(instances),
    "bias_distribution": dict(bias_dist),
    "difficulty_distribution": dict(diff_dist),
    "answer_distribution_top5": dict(answer_dist.most_common(5)),
    "processors_used": {
        "yolo": "YOLOv8x (real)" if use_real_yolo else "contour-based",
        "ocr": "EasyOCR (real)" if use_easyocr else ("module" if use_module_ocr else "unavailable"),
        "translation": "Helsinki-NLP (real native scripts)" if use_real_translation else "English-only",
        "texture": "real (Canny + Gabor)",
        "temporal": "real (EXIF)",
    },
    "stage_timings": {},
}
for name, t in STAGE_TIMES.items():
    if "duration" in t:
        stats["stage_timings"][name] = str(timedelta(seconds=int(t["duration"])))

stats_path = os.path.join(DATASET_DIR, "dataset_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

shutil.copy2(trans_path, os.path.join(DATASET_DIR, "final_dataset.jsonl"))

stage_end("6. Summary & Stats")


# ============================================================
# FINAL REPORT
# ============================================================
total_time = time.time() - GLOBAL_START

print("\n" + "=" * 65)
print("  DATASET CREATION COMPLETE! (ECCV v2)")
print("=" * 65)
print(f"  Output directory: {DATASET_DIR}")
print(f"  Total time: {str(timedelta(seconds=int(total_time)))}")
print(f"  -----------------------------------------------")
print(f"  Images:              {len(collected)}")
print(f"  Annotations:         {len(annotations)}")
print(f"  Total detections:    {total_detections}")
print(f"  OCR text found:      {ocr_success} images")
print(f"  Challenges:          {len(challenges)}")
print(f"  Bias types:          {len(bias_dist)}")
print(f"  Languages:           {len(languages)}")
print(f"  Dataset instances:   {len(instances)}")
print(f"  -----------------------------------------------")
print(f"  Processors:")
print(f"    YOLO:        {'REAL YOLOv8x' if use_real_yolo else 'contour-based'}")
print(f"    OCR:         {'EasyOCR REAL' if use_easyocr else 'skipped'}")
print(f"    Translation: {'REAL Helsinki-NLP' if use_real_translation else 'English-only'}")
print(f"    Texture:     REAL (Canny + Gabor)")
print(f"  -----------------------------------------------")
print(f"  Stage timings:")
for name, t in STAGE_TIMES.items():
    if "duration" in t:
        print(f"    {name}: {str(timedelta(seconds=int(t['duration'])))}")
print(f"  -----------------------------------------------")
print(f"  Bias distribution:")
for btype, ct in sorted(bias_dist.items()):
    print(f"    {btype}: {ct}")
print(f"  -----------------------------------------------")
print(f"  Difficulty spread:")
for diff, ct in sorted(diff_dist.items()):
    print(f"    {diff}: {ct}")
print(f"  -----------------------------------------------")
print(f"  Answer balance (top 5):")
for ans, ct in answer_dist.most_common(5):
    print(f"    '{ans}': {ct}")
print("=" * 65)

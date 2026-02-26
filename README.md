# VLM Reality Check

> A large-scale adversarial diagnostic benchmark measuring **9 systematic biases** in Vision-Language Models across **130K image-pair challenges** and **5 languages**.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VLM REALITY CHECK PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 1: Repo Scaffold & Config                                        │
│  ├── configs/biases.yaml (9 bias types)                                 │
│  ├── configs/pipeline.yaml (global settings)                            │
│  └── configs/languages.yaml (EN, ES, ZH, HI, AR)                       │
│       │                                                                 │
│       ▼ [GATE 1: taxonomy + config + Makefile]                          │
│                                                                         │
│  STAGE 2: Data Collection Pipeline                                      │
│  ├── Wikimedia Commons (CC-licensed, ≥512×512)                          │
│  ├── Open Images v7 (bbox stratified)                                   │
│  ├── Google Street View (100 cities × 50 GPS × 4 headings)             │
│  └── YFCC100M (GPS + outdoor + CC)                                     │
│       │  Target: 500K images, pHash deduplication                       │
│       ▼ [GATE 2: schema + dedup + checkpoint]                           │
│                                                                         │
│  STAGE 3: Vision Processing & Annotation                                │
│  ├── YOLOv8x object detection                                          │
│  ├── MiDaS DPT-Large depth estimation                                  │
│  ├── Astropy + OpenCV shadow detection                                  │
│  ├── Tesseract OCR extraction                                           │
│  ├── Canny + Gabor texture analysis                                     │
│  └── EXIF temporal extraction                                           │
│       │  Output: annotations.jsonl                                      │
│       ▼ [GATE 3: shadow + detector + cache]                             │
│                                                                         │
│  STAGE 4: Adversarial Challenge Generator (9 Biases)                    │
│  ├── texture, counting, spatial, physics, temporal                      │
│  ├── spurious, compositional, text-in-image, scale                      │
│  └── compound (combines 2 single-bias)                                  │
│       │  Target: 90K single + 40K compound = 130K                       │
│       ▼ [GATE 4: confound + counting + distribution]                    │
│                                                                         │
│  STAGE 5: Multilingual Translation                                      │
│  ├── Template-first translation (slot masking)                          │
│  ├── Helsinki-NLP/opus-mt models                                        │
│  ├── Script validation (CJK, Devanagari, Arabic)                        │
│  └── Human validation sampling                                          │
│       │  Target: 130K × 5 = 650K instances                              │
│       ▼ [GATE 5: slots + script + completeness]                         │
│                                                                         │
│  ✅ DATASET COMPLETE                                                     │
│  └── data/challenges/challenges_multilingual.jsonl                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 9 Bias Types

| Bias | Ground Truth | Challenge Design |
|------|-------------|-----------------|
| Texture | Silhouette match | Photo vs silhouette of same object |
| Counting | YOLO count | Compare images with different object counts |
| Spatial Relations | Flip detection | Original vs horizontally/vertically flipped |
| Physical Plausibility | Astropy shadow | Real vs manipulated shadow angles |
| Temporal Reasoning | EXIF timestamp | Order images chronologically |
| Spurious Correlation | YOLO label | Same object, typical vs atypical context |
| Compositional Binding | Color histogram | Correct vs attribute-swapped descriptions |
| Text-in-Image | OCR text | Location matching from visible text |
| Scale Invariance | Zoom factor | Same object at different scales |

## Quickstart

```bash
# 1. Clone and setup
git clone <repo-url>
cd vlm-reality-check
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# 2. Run the full pipeline
make all

# 3. Or run stages individually
make collect      # Stage 2: Download 500K images
make process      # Stage 3: Annotate with 7 processors
make generate     # Stage 4: Generate 130K challenges
make translate    # Stage 5: Translate to 5 languages
make validate_dataset  # Final smoke test
```

## Project Structure

```
vlm-reality-check/
├── configs/
│   ├── biases.yaml          # 9 bias type definitions
│   ├── pipeline.yaml        # Global pipeline settings
│   └── languages.yaml       # Translation model IDs
├── src/
│   ├── __init__.py
│   ├── collection/          # Stage 2: Image collectors
│   ├── processing/          # Stage 3: Vision processors
│   ├── generators/          # Stage 4: Challenge generators
│   └── translation/         # Stage 5: Translation pipeline
├── data/
│   ├── raw/                 # Collected images + manifest
│   ├── processed/           # Annotations
│   ├── challenges/          # Generated challenges
│   └── validation/          # Human validation samples
├── tests/                   # Gate tests
├── scripts/                 # Utility scripts
├── outputs/                 # Pipeline outputs
├── Makefile
├── requirements.txt
└── README.md
```

## Gate Tests

Each stage has built-in gate tests. All must pass before proceeding:

```bash
python tests/gate_1_1.py  # Bias taxonomy completeness
python tests/gate_1_2.py  # Pipeline config validation
# ... (15 gate tests total)
python scripts/smoke_test.py  # Final integration test
```

## License

Research use. See individual image source licenses in `data/raw/manifest.jsonl`.

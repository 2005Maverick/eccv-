# VLM Reality Check — Pipeline Makefile

.PHONY: collect process generate translate validate_dataset all clean

# Stage 2: Collect images from all sources
collect:
	python -m src.collection.pipeline

# Stage 3: Process and annotate all collected images
process:
	python -m src.processing.annotation_pipeline

# Stage 4: Generate adversarial challenges
generate:
	python -m src.generators.pipeline

# Stage 5: Translate challenges to all target languages
translate:
	python -m src.translation.translation_pipeline

# Final: Validate the complete dataset
validate_dataset:
	python scripts/smoke_test.py

# Run all stages sequentially
all: collect process generate translate validate_dataset

# Clean generated data (preserves raw images)
clean:
	rm -f data/processed/annotations.jsonl
	rm -f data/processed/annotation_cache.db
	rm -f data/challenges/challenges.jsonl
	rm -f data/challenges/challenges_multilingual.jsonl
	rm -f data/validation/human_validation_sample.csv

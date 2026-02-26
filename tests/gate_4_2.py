"""Gate 4.2 — Counting generator assigns difficulty and answer correctly"""
import sys
sys.path.insert(0, ".")

from src.generators.counting_generator import CountingGenerator

gen = CountingGenerator()

# 3 cats vs 7 cats → easy (|diff|=4), answer=B
ann_3 = {"image_id": "img_a", "detections": {"detections": [{"label": "cat", "confidence": 0.9}] * 3}}
ann_7 = {"image_id": "img_b", "detections": {"detections": [{"label": "cat", "confidence": 0.9}] * 7}}

challenge = gen.generate_challenge([ann_3, ann_7])
assert challenge is not None, "Challenge should not be None"
assert challenge.correct_answer == "B", f"B has more cats, got {challenge.correct_answer}"
assert challenge.difficulty == "easy", f"|3-7|=4 should be easy, got {challenge.difficulty}"

# 5 cats vs 6 cats → hard (both >4, subitizing zone)
ann_5 = {"image_id": "img_c", "detections": {"detections": [{"label": "cat", "confidence": 0.9}] * 5}}
ann_6 = {"image_id": "img_d", "detections": {"detections": [{"label": "cat", "confidence": 0.9}] * 6}}

challenge_hard = gen.generate_challenge([ann_5, ann_6])
assert challenge_hard is not None, "5 vs 6 challenge should not be None"
assert challenge_hard.difficulty == "hard", f"|5-6|=1 and both >4 should be hard, got {challenge_hard.difficulty}"

print("GATE 4.2 PASSED")

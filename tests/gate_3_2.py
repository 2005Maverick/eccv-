"""Gate 3.2 — Object detector returns correct schema"""
import sys
import os
sys.path.insert(0, ".")

from src.processing.object_detector import ObjectDetector

detector = ObjectDetector()

# Use a test fixture image
test_img = "tests/fixtures/sample.jpg"

if not os.path.exists(test_img):
    # Create a simple test image if fixture doesn't exist
    import numpy as np
    from PIL import Image
    os.makedirs("tests/fixtures", exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    img.save(test_img)
    print("Note: Created random test image. Schema validation still applies.")

result = detector.process(test_img)

assert 'detections' in result, "Missing 'detections' key"
assert isinstance(result['detections'], list), "'detections' should be a list"

for det in result['detections']:
    assert 'label' in det and 'confidence' in det and 'bbox' in det and 'area_fraction' in det, \
        f"Detection missing required fields: {det}"
    assert det['confidence'] > 0.7, "Low-confidence detections should be filtered"
    assert 0 < det['area_fraction'] <= 1.0, f"area_fraction out of range: {det['area_fraction']}"

print("GATE 3.2 PASSED")

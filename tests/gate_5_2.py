"""Gate 5.2 — Script validator correctly accepts/rejects scripts"""
import sys
sys.path.insert(0, ".")

from src.translation.script_validator import validate_script, validate_variable_slots

# Hindi text (Devanagari 70%+)
hindi_text = "यह किस प्रकार का वस्तु है?"
valid_hi, frac_hi = validate_script(hindi_text, "hi")
assert valid_hi, f"Hindi text should be valid Devanagari (got {frac_hi})"

# Arabic text
arabic_text = "هل هذه الصور نفسها؟"
valid_ar, frac_ar = validate_script(arabic_text, "ar")
assert valid_ar, f"Arabic text should be valid Arabic (got {frac_ar})"

# Chinese text
chinese_text = "这是什么类型的物体？"
valid_zh, frac_zh = validate_script(chinese_text, "zh")
assert valid_zh, f"Chinese text should be valid CJK (got {frac_zh})"

# English text pretending to be Hindi should fail
english_as_hindi = "How many cats are in this image?"
valid_bad, frac_bad = validate_script(english_as_hindi, "hi")
assert not valid_bad, f"English text should NOT pass as Devanagari (got {frac_bad})"

# Variable slot validation
assert validate_variable_slots(
    "How many {object_category}?",
    "Cuantos {object_category}?"
), "Variable slots should match"

assert not validate_variable_slots(
    "How many {object_category}?",
    "Cuantos objetos?"
), "Missing variable slot should fail"

print("GATE 5.2 PASSED")

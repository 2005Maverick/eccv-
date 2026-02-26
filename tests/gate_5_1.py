"""Gate 5.1 — Variable slots survive translation round-trip"""
import sys
sys.path.insert(0, ".")

from src.translation.translator import Translator

translator = Translator()

# Test that variable slots are preserved through protection/restoration
template = "Which image has more {object_category}? Answer A or B."

# Test internal protection/restoration mechanism
protected, var_map = translator._protect_variables(template)

# Verify sentinel is in protected text
assert "VARSLOT0" in protected, f"Sentinel not found: {protected}"

# Verify original variable is NOT in protected text
assert "{object_category}" not in protected, "Variable should be replaced with sentinel"

# Verify restoration works
restored = translator._restore_variables(protected, var_map)
assert "{object_category}" in restored, f"Variable not restored: {restored}"
assert restored == template, f"Round-trip failed: {restored} != {template}"

# Test multiple variables
template2 = "The {color} {animal} is near the {location}."
protected2, var_map2 = translator._protect_variables(template2)
restored2 = translator._restore_variables(protected2, var_map2)
assert restored2 == template2, f"Multi-var round-trip failed: {restored2}"

print("GATE 5.1 PASSED")

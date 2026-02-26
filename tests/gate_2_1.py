"""Gate 2.1 — ImageRecord schema is correct"""
import sys
sys.path.insert(0, ".")

from src.collection.base_collector import ImageRecord
import dataclasses

fields = {f.name for f in dataclasses.fields(ImageRecord)}
required = {'image_id', 'url', 'local_path', 'source', 'metadata', 'license'}
assert required.issubset(fields), f"Missing fields: {required - fields}"

print("GATE 2.1 PASSED")

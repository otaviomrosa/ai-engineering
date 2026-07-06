import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from day01.model_metadata import ModelMetadata
from day02.model_registry import ModelRegistry

d1 = {
    "model_name": "resnet50",
    "version": "v1.2.0",
    "framework": "pytorch",
    "inference_latency_ms": 50.3,
    "is_production": True
}

d2 = {
    "model_name": "u-net",
    "version": "v1.20",
    "framework": "tensorflow",
    "inference_latency_ms": 50.3,
    "is_production": False
}

m1 = ModelMetadata.from_dict(d1)
m2 = ModelMetadata.from_dict(d2)

print(ModelMetadata.is_valid_version(m1.version))
print(ModelMetadata.is_valid_version(m2.version))

registry = ModelRegistry()

registry.register(m1)
registry.register(m2)

try:
    m2.latency = -5
except ValueError as e:
    print(f"Caught expected error: {e}")

registry.promote("u-net", "v1.20")

print(m2)

print(m2.predict(2))

print(m1.describe())

try:
    m2.newattr = "This is totally gonna work"
except AttributeError as e:
    print(f"Caught expected error: {e}")
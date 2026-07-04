import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from day01.model_metadata import ModelMetadata

class ModelRegistry():
    def __init__(self):
        self.models = set()

    def __len__(self):
        return len(self.models)
    
    def __repr__(self):
        names = ", ".join(m.model_name for m in self.models)
        return (f"ModelRegistry(count={len(self)}, models=[{names}])")

    def register(self, model: ModelMetadata) -> None:
        if not isinstance(model, ModelMetadata):
            raise TypeError("The model must be an instance of the ModelMetadata class")
        if model in self.models:
            raise ValueError("A model with this name and version already exists.")
        self.models.add(model)
        
    def get(self, model_name: str, version: str) -> ModelMetadata:
        for m in self.models:
            if m.model_name == model_name and m.version == version:
                return m
        raise KeyError("The model was not found in the registry.")
    
    def promote(self, model_name: str, version: str) -> None:
        m = self.get(model_name, version)
        if m.is_production:
            raise ValueError("This model is already promoted.")
        m.is_production = True

if __name__ == "__main__":
    
    registry = ModelRegistry()

    m1 = ModelMetadata("resnet50", "v1.2.0", "pytorch", 50.3, True)
    m2 = ModelMetadata("resnet50", "v1.3.0", "tensorflow", 12.5, False)
    m3 = ModelMetadata("bert-base", "v2.0.1", "onnx", 400.3, False)
    dm3 = ModelMetadata("bert-base", "v2.0.1", "tensorflow", 403.2, True)

    registry.register(m1)
    registry.register(m2)
    registry.register(m3)

    try:
        registry.register(dm3) # Should raise ValueError
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print(registry.get("resnet50", "v1.3.0"))

    registry.promote("resnet50", "v1.3.0")

    print(str(registry.get("resnet50", "v1.3.0")))

    try:
        registry.get("u-net", "v11.0.2")
    except KeyError as e:
        print(f"Caught expected error: {e}")

    print(len(registry))

    print(repr(registry))
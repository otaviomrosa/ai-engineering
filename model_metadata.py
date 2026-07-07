import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from day04.base_model import BaseModel

class ModelMetadata(BaseModel):

    SUPPORTED_FRAMEWORKS = frozenset({"onnx", "pytorch", "tensorflow"})
    
    def __init__(self, name: str, version: str, framework: str, latency: float, is_production: bool = False):
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"The framework {framework} is not supported. "
                f"Supported frameworks are {', '.join(self.SUPPORTED_FRAMEWORKS)}"
            )
        
        self.model_name = name
        self.version = version
        self.framework = framework
        self.is_production = is_production
        self.latency = latency

    __slots__ = ("model_name", "version", "framework", "_is_production", "_latency")

    @property
    def latency(self) -> float:
        return self._latency

    @latency.setter
    def latency(self, value: float) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Latency should be a positive number.")
        self._latency = value

    @property
    def is_production(self) -> bool:
        return self._is_production
    
    @is_production.setter
    def is_production(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")
        self._is_production = value

    def __repr__(self):
        return (f"ModelMetadata('{self.model_name}', '{self.version}', '{self.framework}', {self.latency}, {self.is_production})")
    
    def __str__(self):
        return ("[PROD] " if self.is_production is True else "[STAGING] ") + self.model_name + " " + self.version + " | " + self.framework + " | " + str(self.latency) + "ms"

    def __eq__(self, other):
        return self.model_name == other.model_name and self.version == other.version
    
    def __hash__(self):
        return hash((self.model_name, self.version))
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        try:
            return cls(
                data["model_name"],
                data["version"],
                data["framework"],
                data["inference_latency_ms"],
                data["is_production"]
            )
        except KeyError as e:
            raise KeyError(f"Missing required field: {e}") from e
    
    @staticmethod
    def is_valid_version(version:str) -> bool:
        return bool(re.fullmatch(r"^v\d+\.\d+\.\d+$", version))
    
    def predict(self, input_data) -> dict:
        result = f"self.inference({input_data})"
        return {"model": self.model_name, "version": self.version, "result": result, "latency_ms": self.latency}

    def validate_input(self, input_data) -> bool:
        return True if input_data else False

    def get_model_info(self) -> dict:
        return {"name": self.model_name, "version": self.version, "framework": self.framework}

if __name__ == "__main__":

    m1 = ModelMetadata("resnet50", "v1.2.0", "pytorch", 50.3, True)
    m2 = ModelMetadata("resnet50", "v1.2.0", "tensorflow", 12.5, False)
    m3 = ModelMetadata("bert-base", "v2.0.1", "onnx", 400.3, False)
    m4 = ModelMetadata("bert-base", "v2.0.1", "pytorch", 1023.1, True)
    
    model_suite = set()
    model_suite.add(m1)
    model_suite.add(m2)
    model_suite.add(m3)
    model_suite.add(m4)

    print(model_suite)
    print(m1)
    print(m4)

    try:
        m5 = ModelMetadata("resnet50", "v1.2.0", "jax", 29.2, True)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print(m3.__dict__)
    except AttributeError as e:
        print(f"Caught expected errorr: {e}")

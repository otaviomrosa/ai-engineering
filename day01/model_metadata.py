import re

class ModelMetadata():

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

if __name__ == "__main__":

    exp1 = ModelMetadata("resnet50", "v1.2.0", "pytorch", 50.3, True)
    exp2 = ModelMetadata("resnet50", "v1.2.0", "tensorflow", 12.5, False)
    exp3 = ModelMetadata("bert-base", "v2.0.1", "onnx", 400.3, False)
    exp4 = ModelMetadata("bert-base", "v2.0.1", "pytorch", 1023.1, True)
    
    experiment_suite = set()
    experiment_suite.add(exp1)
    experiment_suite.add(exp2)
    experiment_suite.add(exp3)
    experiment_suite.add(exp4)

    print(experiment_suite)
    print(exp1)
    print(exp4)

    try:
        exp5 = ModelMetadata("resnet50", "v1.2.0", "jax", 29.2, True)
    except ValueError as e:
        print(f"Caught expected error: {e}")

from abc import ABC, abstractmethod

class BaseModel(ABC):

    __slots__ = ()

    @abstractmethod
    def predict(self, input_data) -> dict:
        """Run inference and return a result dict"""

    @abstractmethod
    def validate_input(self, input_data) -> bool:
        """Return True if input_data is valid for this model"""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return a dict with model metadata"""

    def describe(self) -> str:
        info = self.get_model_info()
        return f"{info['name']} {info['version']} ({info['framework']})"
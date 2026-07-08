import pytest
from model_registry import ModelMetadata

class TestMetadata():
    def test_valid_construction(self):
        model = ModelMetadata("resnet50", "v1.30.2", "pytorch", 240, False)
        assert model.model_name == "resnet50"
        assert model.version == "v1.30.2"
        assert model.framework == "pytorch"
        assert model.latency == 240
        assert model.is_production == False

    def test_invalid_framework(self):
        with pytest.raises(ValueError):
            model = ModelMetadata("resnet50", "v1.30.2", "scale", 123, True)

    def test_negative_latency(self):
        with pytest.raises(ValueError):
            model = ModelMetadata("resnet60", "v1.2.3", "pytorch", -431, False)

    def test_invalid_is_production(self):
        with pytest.raises(TypeError):
            model = ModelMetadata("resnet20", "v3.1.1", "tensorflow", 152, "True")

    def test_equality(self):
        m1 = ModelMetadata("resnet50", "v1.2.0", "pytorch", 50.3, True)
        m2 = ModelMetadata("resnet50", "v1.2.0", "tensorflow", 12.5, False)
        m3 = ModelMetadata("bert-base", "v2.0.1", "onnx", 400.3, False)
        m4 = ModelMetadata("bert-base", "v2.0.2", "pytorch", 1023.1, True)
        assert m1 == m2
        assert m3 != m4

    def test_hash_deduplication(self):
        m1 = ModelMetadata("resnet50", "v1.2.0", "pytorch", 50.3, True)
        m2 = ModelMetadata("resnet50", "v1.2.0", "tensorflow", 12.5, False)
        m3 = ModelMetadata("bert-base", "v2.0.2", "onnx", 400.3, False)
        m4 = ModelMetadata("bert-base", "v2.0.2", "pytorch", 1023.1, True)

        assert len({m1, m2, m3, m4}) == 2

    def test_from_dict(self):
        modeldict = {"model_name":"resnet50", "version": "v1.2.0", "framework": "pytorch", "inference_latency_ms": 50.3, "is_production": True}
        model = ModelMetadata.from_dict(modeldict)
        assert model.model_name == "resnet50"
        assert model.version == "v1.2.0"
        assert model.framework == "pytorch"
        assert model.latency == 50.3
        assert model.is_production == True

    def test_version_validation(self):
        assert ModelMetadata.is_valid_version("v1.2.3") == True
        assert ModelMetadata.is_valid_version("1.2.3") == False
        assert ModelMetadata.is_valid_version("v134134124.12342141242.5342523543") == True
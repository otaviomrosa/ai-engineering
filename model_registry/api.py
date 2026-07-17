from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel
from .registry import ModelRegistry
from .metadata import ModelMetadata

app = FastAPI(title="Model Registry API", version = "0.1.0")

registry = ModelRegistry()

class ModelRegistrationRequest(PydanticBaseModel):
    model_name: str
    version: str
    framework: str
    inference_latency_ms: float
    is_production: bool = False

class ModelResponse(PydanticBaseModel):
    model_name:str
    version: str
    framework: str
    inference_latency_ms: float
    is_production: bool

class InferenceRequest(PydanticBaseModel):
    input_data: dict


class InferenceResponse(PydanticBaseModel):
    model_name: str
    version: str
    result: dict
    latency_ms: float

def metadata_to_response(model: ModelMetadata) -> ModelResponse:
    return ModelResponse(
        model_name = model.model_name,
        version = model.version,
        framework = model.framework,
        inference_latency_ms = model.latency,
        is_production = model.is_production
    )

@app.post("/models", response_model = ModelResponse)
async def register_model(model: ModelRegistrationRequest):
    try:
        m = ModelMetadata.from_dict(model.model_dump())
        registry.register(m)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return metadata_to_response(m)

@app.get("/models", response_model = list[ModelResponse])
async def list_models():
    return list(metadata_to_response(m) for m in registry.models)

@app.get("/models/{model_name}/{version}")
async def get_model(model_name, version):
    return metadata_to_response(registry.get(model_name, version))

@app.post("/models/{model_name}/{version}/promote")
async def promote_model(model_name, version):
    try:
        model = registry.get(model_name, version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if model.is_production:
        raise HTTPException(status_code=400, detail="Model is already production")
    registry.promote(model_name, version)
    return metadata_to_response(model)

@app.post("/models/{model_name}/{version}/infer", response_model = InferenceResponse)
async def run_async_inference(model_name, version, data: InferenceRequest):
    try:
        model = registry.get(model_name, version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    inference_result =  await model.async_predict(data.input_data)
    return InferenceResponse(model_name = model.model_name, version = model.version, result = inference_result, latency_ms = model.latency)
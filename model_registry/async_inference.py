import asyncio, time
from .metadata import ModelMetadata

async def simulate_inference(model: ModelMetadata, input_data: dict):
    await asyncio.sleep(model.latency/1000)
    result = model.predict(input_data)
    return result

async def run_batch_inference(tuples: list[tuple[ModelMetadata, dict]]):
    inferences = [simulate_inference(*t) for t in tuples]
    results = await asyncio.gather(*inferences)
    return results

async def benchmark():
    models = [
        ModelMetadata("resnet50", "v1.2.0", "pytorch", 20, True),
        ModelMetadata("resnet50", "v1.2.0", "tensorflow", 50, False),
        ModelMetadata("bert-base", "v2.0.1", "onnx", 80, False),
        ModelMetadata("bert-base", "v2.0.1", "pytorch", 150, True),
    ]

    mock_input_data = {"data": "label", "data2": "label2"}

    sequential_results = []

    start = time.perf_counter()
    for model in models:
        sequential_results.append(await simulate_inference(model, mock_input_data))
    
    elapsed = time.perf_counter() - start
    print(f"Sequential inference ran in {elapsed:0.2f} seconds")

    start = time.perf_counter()

    inference_pairs = [(m, mock_input_data) for m in models]

    batch_results = await run_batch_inference(inference_pairs)

    elapsed = time.perf_counter() - start
    print(f"Batch inference ran in {elapsed: 0.2f} seconds")

if __name__ == "__main__":
    asyncio.run(benchmark())

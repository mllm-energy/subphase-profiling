import torch
import numpy as np
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

torch.set_grad_enabled(False)

model_name = "OpenGVLab/InternVL3-8B"
print(f"Loading {model_name}...")

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.80,
    max_num_seqs=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
)

# Five distinct (image, prompt) pairs so multimodal / prefix caches never hit on the profiled request.
base = Image.open("eer.jpg").convert("RGB").resize((448, 448))
base_arr = np.asarray(base, dtype=np.uint8)


def make_variant(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, size=base_arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


PROMPTS = [
    "<image>\nWhat building is this? Describe it in detail. Taken from Austin, Texas.",
    "<image>\nIdentify the architectural style of this structure and its likely purpose.",
    "<image>\nList the prominent features visible in this photograph, top to bottom.",
    "<image>\nWhat materials appear to be used in the construction shown here?",
    "<image>\nDescribe the lighting, weather, and time of day captured in this image.",
]
images = [make_variant(seed=i) for i in range(len(PROMPTS))]


def request(i: int):
    return {"prompt": PROMPTS[i], "multi_modal_data": {"image": images[i]}}


sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

print("--- WARMUP ---")
for i in range(4):
    llm.generate(request(i), sampling_params)
    torch.cuda.synchronize()
    print(f"  {i + 1}/4")

print("--- PROFILE (request 4) ---")
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")
t0 = perf_counter()
llm.generate(request(4), sampling_params)
torch.cuda.synchronize()
print(f"done in {perf_counter() - t0:.4f}s")
torch.cuda.nvtx.range_pop()

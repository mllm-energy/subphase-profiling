"""Batch sweep variant. BATCH_SIZE comes from env var BATCH_SIZE (default 1).
Issues N concurrent requests so vLLM batches them into the same forward passes."""
import os
import torch
import numpy as np
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

torch.set_grad_enabled(False)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
print(f"Batch size: {BATCH_SIZE}")

llm = LLM(
    model="OpenGVLab/InternVL3-8B",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.80,
    max_num_seqs=BATCH_SIZE,
    enforce_eager=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
)

PROMPTS = [
    "<image>\nWhat building is this? Describe it in detail. Taken from Austin, Texas.",
    "<image>\nIdentify the architectural style of this structure and its likely purpose.",
    "<image>\nList the prominent features visible in this photograph, top to bottom.",
    "<image>\nWhat materials appear to be used in the construction shown here?",
    "<image>\nDescribe the lighting, weather, and time of day captured in this image.",
    "<image>\nGuess the city and decade this photograph might have been taken in.",
    "<image>\nWhat are the dominant colors and how do they affect the mood?",
    "<image>\nDescribe the spatial composition and any focal points.",
    "<image>\nWhat textures and surfaces are most prominent in this image?",
    "<image>\nWhat would you guess about the photographer's vantage point?",
    "<image>\nHow does the lighting interact with the structure's geometry?",
    "<image>\nWhat is the relationship between foreground and background here?",
    "<image>\nDescribe the level of detail captured in the upper third of the image.",
    "<image>\nIs this image likely posed or candid? Justify your answer.",
    "<image>\nWhat ambient sounds might accompany this scene?",
    "<image>\nWhat would change if this photograph were taken at sunset?",
]
PROMPTS = PROMPTS[:max(BATCH_SIZE, 1) * 4 + 16]  # ensure enough variants

base = Image.open("eer.jpg").convert("RGB").resize((448, 448))
base_arr = np.asarray(base, dtype=np.uint8)

def make_variant(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, size=base_arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))

# Pre-build enough images for warmup + profiling at given batch size
TOTAL_VARIANTS = (BATCH_SIZE * 5) + 4   # 4 warmup batches of size B + 1 profiled batch + slack
images = [make_variant(seed=i) for i in range(TOTAL_VARIANTS)]
prompts_pool = [PROMPTS[i % len(PROMPTS)] for i in range(TOTAL_VARIANTS)]

def make_batch(start_idx: int):
    return [
        {"prompt": prompts_pool[start_idx + j],
         "multi_modal_data": {"image": images[start_idx + j]}}
        for j in range(BATCH_SIZE)
    ]

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

print(f"--- STAGE: WARMUP START (batch={BATCH_SIZE}) ---")
idx = 0
for w in range(4):
    batch = make_batch(idx); idx += BATCH_SIZE
    llm.generate(batch, sampling_params)
    torch.cuda.synchronize()
    print(f"Warmup batch {w+1}/4 complete.")

print(f"--- STAGE: PROFILING START (batch={BATCH_SIZE}) ---")
batch = make_batch(idx)
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")
start_time = perf_counter()
llm.generate(batch, sampling_params)
torch.cuda.synchronize()
elapsed = perf_counter() - start_time
torch.cuda.nvtx.range_pop()

print(f"Profiled Batch (B={BATCH_SIZE}) finished in {elapsed:.4f} seconds")
print("--- STAGE: PROFILING END ---")

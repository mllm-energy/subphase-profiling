import torch
import numpy as np
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

torch.set_grad_enabled(False)

# ==========================================
# 1. MODEL SETUP — caches disabled so every
#    request actually runs Prefill + Vision Encoder
# ==========================================
model_name = "llava-hf/llava-1.5-7b-hf"
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

# ==========================================
# 2. INPUT SETUP — 5 distinct (image, prompt) pairs.
#    LLaVA-1.5 prompt format: "USER: <image>\n{question}\nASSISTANT:"
# ==========================================
base = Image.open("eer.jpg").convert("RGB").resize((336, 336))
base_arr = np.asarray(base, dtype=np.uint8)

def make_variant(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, size=base_arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))

QUESTIONS = [
    "What building is this? Describe it in detail. Taken from Austin, Texas.",
    "Identify the architectural style of this structure and its likely purpose.",
    "List the prominent features visible in this photograph, top to bottom.",
    "What materials appear to be used in the construction shown here?",
    "Describe the lighting, weather, and time of day captured in this image.",
]
PROMPTS = [f"USER: <image>\n{q}\nASSISTANT:" for q in QUESTIONS]
images = [make_variant(seed=i) for i in range(len(PROMPTS))]

def request(i: int):
    return {"prompt": PROMPTS[i], "multi_modal_data": {"image": images[i]}}

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

# ==========================================
# 3. WARMUP PHASE — 4 distinct inputs, no profiler attached
# ==========================================
print("--- STAGE: WARMUP START ---")
for i in range(4):
    llm.generate(request(i), sampling_params)
    torch.cuda.synchronize()
    print(f"Warmup request {i+1}/4 complete.")

# ==========================================
# 4. PROFILING PHASE — input #4 is unseen by both caches
# ==========================================
print("--- STAGE: PROFILING START ---")
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")
start_time = perf_counter()
llm.generate(request(4), sampling_params)
torch.cuda.synchronize()
elapsed = perf_counter() - start_time
torch.cuda.nvtx.range_pop()

print(f"Profiled Request finished in {elapsed:.4f} seconds")
print("--- STAGE: PROFILING END ---")
print("Done! Check your NCU output.")

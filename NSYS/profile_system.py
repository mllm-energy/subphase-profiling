import torch
import time
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "eer.jpg")

raw_image = Image.open(image_path).convert("RGB")

# Turn off PyTorch gradient tracking globally
torch.set_grad_enabled(False)

# ==========================================
# 1. MODEL SETUP
# ==========================================
model_name = "OpenGVLab/InternVL3-8B"
print(f"Loading {model_name}...")

llm = LLM(
    model=model_name, 
    trust_remote_code=True, 
    max_model_len=4096,           # Back to normal
    gpu_memory_utilization=0.90,  # Safe, stable ceiling
    max_num_seqs=1,
    enforce_eager=True,
)

# ==========================================
# 2. DEFINE THE GRANULAR TEST SUITE
# ==========================================

test_cases = [
    {
        "name": "TEST_1_LOW_RES_10_TOKENS",
        "prompt": "<image>\nWhat building is this?",
        "image": raw_image.resize((448, 896)), # 1 Patch
        "max_tokens": 10
    },
    # {
    #     "name": "TEST_2_HIGH_RES_10_TOKENS",
    #     "prompt": "<image>\nWhat building is this?",
    #     "image": raw_image.resize((1344, 1344)), # 9 Patches
    #     "max_tokens": 10
    # },
    # {
    #     "name": "TEST_3_LOW_RES_32_TOKENS",
    #     "prompt": "<image>\nDescribe the architectural style of this building.",
    #     "image": raw_image.resize((448, 448)), # 1 Patch
    #     "max_tokens": 32
    # },
    # {
    #     "name": "TEST_4_HIGH_RES_32_TOKENS",
    #     "prompt": "<image>\nDescribe the architectural style of this building.",
    #     "image": raw_image.resize((1344, 1344)), # 9 Patches
    #     "max_tokens": 32
    # },
    # {
    #     "name": "TEST_5_LOW_RES_64_TOKENS",
    #     "prompt": "<image>\nWrite a detailed paragraph about this building and its surroundings.",
    #     "image": raw_image.resize((448, 448)), # 1 Patch
    #     "max_tokens": 64
    # },
    # {
    #     "name": "TEST_6_HIGH_RES_64_TOKENS",
    #     "prompt": "<image>\nWrite a detailed paragraph about this building and its surroundings.",
    #     "image": raw_image.resize((1344, 1344)), # 9 Patches
    #     "max_tokens": 64
    # }
]

# ==========================================
# 3. WARMUP PHASE
# ==========================================
print("--- STAGE: WARMUP START ---")
# Warmup using the heaviest case (Test 6) to ensure max VRAM allocation
warmup_inputs = {"prompt": test_cases[0]["prompt"], "multi_modal_data": {"image": test_cases[0]["image"]}}
warmup_params = SamplingParams(temperature=0.0, max_tokens=10)

for i in range(2):
    llm.generate(warmup_inputs, warmup_params)
    torch.cuda.synchronize()
    print(f"Warmup request {i+1}/2 complete.")

# ==========================================
# 4. PROFILING PHASE
# ==========================================
print("--- STAGE: PROFILING START ---")

for case in test_cases:
    print(f"\nRunning {case['name']}...")
    
    inputs = {"prompt": case["prompt"], "multi_modal_data": {"image": case["image"]}}
    sampling_params = SamplingParams(temperature=0.0, max_tokens=case["max_tokens"])
    
    torch.cuda.synchronize()
    
    # Push NVTX marker for the specific test case
    torch.cuda.nvtx.range_push(case["name"])
    start_time = perf_counter()
    
    llm.generate([inputs], sampling_params)
    
    torch.cuda.synchronize() 
    elapsed = perf_counter() - start_time
    torch.cuda.nvtx.range_pop()
    
    print(f"[{case['name']}] finished in {elapsed:.4f} seconds")

print("\n--- STAGE: PROFILING END ---")
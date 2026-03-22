import torch
import time
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

# Turn off PyTorch gradient tracking globally
torch.set_grad_enabled(False)

# ==========================================
# 1. MODEL SETUP
# ==========================================
model_name = "OpenGVLab/InternVL3-8B"
print(f"Loading {model_name}...")

# The LLM() call handles VRAM allocation internally, acting as the primary warmup
llm = LLM(
    model=model_name, 
    trust_remote_code=True, 
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    max_num_seqs=1,
    enforce_eager=True, # REQUIRED for accurate NCU kernel breakdown
)

# ==========================================
# 2. DEFINE THE TEST SUITE (Single Test)
# ==========================================
try:
    raw_image = Image.open("eer.jpg").convert("RGB")
except FileNotFoundError:
    print("WARNING: eer.jpg not found. Creating a blank test image so the script doesn't crash.")
    raw_image = Image.new("RGB", (448, 448))

# Only keeping Test 1 (Low Res / 10 Tokens)
test_cases = [
    {
        "name": "Test", 
        "prompt": "<image>\nWhat building is this?", 
        "image": raw_image.resize((448, 448)), 
        "max_tokens": 10
    }
]

# ==========================================
# 3. PROFILING PHASE (Warmup completely removed)
# ==========================================
print("--- STAGE: PROFILING START ---")

for case in test_cases:
    print(f"\n[NCU TRIGGER] Starting {case['name']}...")
    
    inputs = {"prompt": case["prompt"], "multi_modal_data": {"image": case["image"]}}
    sampling_params = SamplingParams(temperature=0.0, max_tokens=case["max_tokens"])
    
    torch.cuda.synchronize()
    
    # CRITICAL: This is the marker NCU uses to decide when to "work"
    torch.cuda.nvtx.range_push(case["name"])
    
    llm.generate(inputs, sampling_params)
    
    torch.cuda.synchronize() 
    torch.cuda.nvtx.range_pop()
    print(f"Finished {case['name']}")

print("\n--- STAGE: PROFILING END ---")
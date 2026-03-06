import torch
import time
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

# Turn off PyTorch gradient tracking globally to save memory
torch.set_grad_enabled(False)

# ==========================================
# 1. MODEL SETUP & STRICT VRAM LIMITS
# ==========================================
model_name = "OpenGVLab/InternVL3-8B"
print(f"Loading {model_name}...")

llm = LLM(
    model=model_name, 
    trust_remote_code=True, 
    max_model_len=2048,           
    gpu_memory_utilization=0.80,  
    max_num_seqs=1,
    enforce_eager=True,
)

# ==========================================
# 2. INPUT SETUP
# ==========================================
prompt = "<image>\nWhat building is this? Please describe it in detail. It was taken from Austin, Texas."
image = Image.open("eer.jpg").convert("RGB").resize((448, 448))
inputs = {
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

# ==========================================
# 3. WARMUP PHASE
# ==========================================
print("--- STAGE: WARMUP START ---")
for i in range(3):
    llm.generate(inputs, sampling_params)
    torch.cuda.synchronize()
    print(f"Warmup request {i+1}/3 complete.")

# ==========================================
# 4. PROFILING PHASE (CLOSED-LOOP)
# ==========================================
print("--- STAGE: PROFILING START ---")

# Start the NVTX Marker.
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")

start_time = perf_counter()
llm.generate(inputs, sampling_params)
torch.cuda.synchronize() 
elapsed = perf_counter() - start_time

print(f"Profiled Request finished in {elapsed:.4f} seconds")

# End the NVTX Marker
torch.cuda.nvtx.range_pop()

print("--- STAGE: PROFILING END ---")
print("Done! Check your NCU output.")
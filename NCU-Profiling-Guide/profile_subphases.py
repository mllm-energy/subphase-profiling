import torch
import time
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
)

# ==========================================
# 2. INPUT SETUP
# ==========================================
prompt = "<image>\nPlease describe this image in detail."
image = Image.new('RGB', (224, 224), color='blue')
sampling_params = SamplingParams(temperature=0.0, max_tokens=15)

inputs = {
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}

# ==========================================
# 3. WARMUP PHASE
# ==========================================
print("--- STAGE: WARMUP START ---")
for i in range(2):
    llm.generate(inputs, sampling_params)
    print(f"Warmup request {i+1}/2 complete.")

# Force PyTorch to empty the "scratchpad" memory before profiling
torch.cuda.empty_cache()
torch.cuda.synchronize()
time.sleep(2) 

# ==========================================
# 4. PROFILING PHASE (CLOSED-LOOP)
# ==========================================
print("--- STAGE: PROFILING START ---")

# Start the NVTX Marker.
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")

for i in range(3):
    start_time = time.time()
    
    llm.generate(inputs, sampling_params)
    torch.cuda.synchronize() 
    
    elapsed = time.time() - start_time
    print(f"Profiled Request {i+1}/3 finished in {elapsed:.4f} seconds")

# End the NVTX Marker
torch.cuda.nvtx.range_pop()

print("--- STAGE: PROFILING END ---")
print("Done! Check your NCU output.")
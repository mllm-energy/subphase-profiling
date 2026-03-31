# Profiling the Subphases

Guide for profiling vLLM sub-phases with NVIDIA Nsight Compute (NCU) using NVTX markers.

## Prerequisites

- NVIDIA GPU with drivers installed
- NVIDIA Nsight Compute (NCU) installed
- Python 3

## Setup

### 1. Clone the VLLM repository

Clone the official vLLM source to allow for code modifications:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### 2. Virtual Environment Setup

From inside the cloned `vllm` directory, we use a dedicated venv to prevent dependency conflicts between vLLM and the system.

Run:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install --upgrade pip`
- `curl -LsSf https://astral.sh/uv/install.sh | sh`

Restart the terminal *or* run `export PATH="$HOME/.local/bin:$PATH"` so that `uv` is on your PATH.
Check if `uv` installed by `uv --version`.

### 3. Library Installation

From inside the `vllm` directory with the venv activated, install the specific libraries required to run InternVL3 within the vLLM engine:

- `VLLM_USE_PRECOMPILED=1 uv pip install --editable .` (single command with the env var) (NOTE: THIS MIGHT NOT WORK, IF IT DOESN'T WORK TRY TO DOWNGRADE  TO AN OLD COMMIT OF VLLM USING THIS COMMAND -> export VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~2))
- `pip install torch torchvision torchaudio timm einops pillow`

### 4. Model Code Modifications

To enable precise profiling of sub-phases, we modified the vLLM source code to include NVTX markers.
This allows the profiler to see where the model forward pass begins and ends.

Check internvl.py and look for NVTX using ctrl-f, look at places I add the code and then look at the current internvl.py file from the version of the repo you just copied to see where to add NVTX ranges. You cannot copy paste the file in the repo because VLLM source code is continuously changing.

Adds NVTX markers for Vision Encoder vs LLM prefill vs decode

### 5. Profiling Script

I created a specialized script which the profiler uses to run the closed-loop request cycle, and generate outputs.

The script profile_suite.py is in NCU-Profiling-Guide/ in this repo. Place it in the directory which contains all the files you need to be called with the command. This script should work if it doesn't try prompting AI to see the issue based on terminal output.

### 6. Profiling Commands

Run these sequentially. This profiles the model, exports the specific GPU trace to CSV, and renames it for the Python script.

# NSYS Profiling

```bash
/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx \
    --output=mllm_profile \
    --force-overwrite=true \
    python3 profile_suite.py
```

# Export NSYS to CSV
```bash
/usr/local/cuda/bin/nsys stats \
    --report cuda_gpu_trace \
    --format csv \
    --output mllm_nsys_gpu_trace \
    mllm_profile.nsys-rep
```

# NCU Profiling for all kernels

Nsight Compute extracts deep hardware metrics (SM throughput, DRAM throughput, active cycles, and grid size). Because it replays kernels, we profile the Vision Encoder, LLM Prefill, and LLM Decode phases individually using the --nvtx-include flag.

Note: ncu strictly requires sudo privileges to access the physical hardware performance counters on the GPU.

# 1. Profile Vision Encoder
```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_Vision_Encoder/" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_vision_encoder_test1 python3 profile_suite.py
```

# 2. Profile LLM Prefill
```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_LLM_Prefill/" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_llm_prefill_test1 python3 profile_suite.py
```

# 3. Profile LLM Decode
```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_LLM_Decode/" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_decode_test1 python3 profile_suite.py
```

# Export NCU Data to CSV

```bash
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_vision_encoder_test1.ncu-rep > ncu_vision_encoder_test1.csv
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_llm_prefill_test1.ncu-rep > ncu_llm_prefill_test1.csv
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_decode_test1.ncu-rep > ncu_decode_test1.csv
```

# NCU Profiling for specific kernels for faster runtimes, this is the data I have currently, includes the most heavyweight kernels running which is the majority (might try to reprofile with all for completeness if possible by presentation day)

# 1. Profile Vision Encoder

```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_Vision_Encoder/" \
    -k "regex:gemm|flash|rms_norm|rotary" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_vision_encoder_test1 python3 profile_suite.py
```

# 2. Profile LLM Prefill

```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_LLM_Prefill/" \
    -k "regex:gemm|flash|rms_norm|rotary" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_llm_prefill_test1 python3 profile_suite.py
```

# 3. Profile LLM Decode
```bash
sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "InternVL_LLM_Decode/" \
    -k "regex:gemm|flash|rms_norm|rotary" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
    -o ncu_decode_test1 python3 profile_suite.py
```

# Export NCU Data to CSV

```bash
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_vision_encoder_test1.ncu-rep > ncu_vision_encoder_test1.csv
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_llm_prefill_test1.ncu-rep > ncu_llm_prefill_test1.csv
sudo /usr/local/cuda/bin/ncu --export csv --page details -i ncu_decode_test1.ncu-rep > ncu_decode_test1.csv
```
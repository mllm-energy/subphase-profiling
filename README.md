# Profiling the Subphases

Guide for profiling vLLM sub-phases with NVIDIA Nsight Compute (NCU) using NVTX markers.

## Prerequisites

- NVIDIA GPU with drivers installed
- NVIDIA Nsight Compute (NCU) installed
- NVIDIA Nsight Systems (nsys) installed (for timeline profiling)
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

- `VLLM_USE_PRECOMPILED=1 uv pip install --editable .` (single command with the env var)
- `pip install torch torchvision torchaudio timm einops pillow`

### 4. Model Code Modifications

Copy `NCU-Profiling-Guide/internvl.py` into your vLLM clone. This repo's vllm already has LLM NVTX markers in `gpu_model_runner.py`; if using an external vLLM clone, copy `vllm/vllm/v1/worker/gpu_model_runner.py` or apply the same NVTX block around `_model_forward`.

### 5. Profiling Script

I created a specialized script which the profiler uses to run the closed-loop request cycle, and generate outputs.

The script `profile_subphases.py` is in `NCU-Profiling-Guide/` in this repo. Place it in your project directory (e.g. `subphase-profiling`) and run the NCU command from that directory. Make sure to keep it in a separate folder from the `vllm` directory.

### 6. Command for profiling

From the directory containing `profile_subphases.py`, run the NCU profiler. I used sudo to bypass Linux security restrictions that would cause crashes. This produces `InternVL_Hardware_Profile.ncu-rep`.

Here is the script:

```bash
sudo $(which ncu) --target-processes all \
    --replay-mode kernel \
    --nvtx \
    --nvtx-range "CLOSED_LOOP_INFERENCE" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o InternVL_Hardware_Profile \
    python3 profile_subphases.py
```

### 7. Exporting to CSV

The command above produced a binary `.ncu-rep` file.
I then used the NCU export tool (in NVIDIA Nsight Compute: File → Export to CSV, or the `ncu --export` CLI) to turn that into the `existing_data.csv` we used for analysis.
Make sure to download the NVIDIA Nsight Compute software so that you can get the CSV as well.

Since our main goal is to find each subphase separately we need to add the NVTX hooks in different spots than I did.

| Subphase       | NVTX range                  | Location                    |
|----------------|-----------------------------|-----------------------------|
| Vision encoder | `InternVL_Vision_Encoder/`  | internvl.py extract_feature |
| MLP connector  | `InternVL_MLP_Connector/`   | internvl.py extract_feature |
| LLM prefill    | `InternVL_LLM_Prefill/`     | gpu_model_runner.py         |
| LLM decode     | `InternVL_LLM_Decode/`      | gpu_model_runner.py         |

### Subphase profiling with NCU

Use `--nvtx-include` (not `--nvtx-range`) to profile a single subphase. Example for the vision encoder:

```bash
cd NCU-Profiling-Guide && nohup sudo $(which ncu) --target-processes all \
    --replay-mode kernel --nvtx --nvtx-include "InternVL_Vision_Encoder/" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o InternVL_Vision_Encoder --csv \
    $(which python3) profile_subphases.py > vision_encoder_profile.csv 2>&1 &
```

Replace `InternVL_Vision_Encoder/` with `InternVL_MLP_Connector/`, `InternVL_LLM_Prefill/`, or `InternVL_LLM_Decode/` for the other subphases.

**Note:** `profile_subphases.py` uses `enforce_eager=True` so NVTX markers are visible to NCU (CUDA graph replay hides kernels from NVTX filtering).

### Nsight Systems Timeline Profiling

Nsight Systems (nsys) gives a **timeline** showing how the four subphases execute over time (wall-clock, sequencing, GPU idle gaps). NCU gives per-kernel metrics; nsys shows when each subphase runs.

From `NCU-Profiling-Guide/` with your venv activated:

```bash
sudo $(which nsys) profile \
    --trace=cuda,nvtx \
    --nvtx-capture=range@CLOSED_LOOP_INFERENCE \
    --output=InternVL_Timeline \
    ../.venv/bin/python3 profile_subphases.py
```

Use your venv Python (e.g. `../.venv/bin/python3` when the venv is in the repo root). Add `--force-overwrite true` if `InternVL_Timeline.nsys-rep` already exists.

- `--trace=cuda,nvtx` — capture CUDA API calls, kernel launches, and NVTX ranges
- `--nvtx-capture=range@CLOSED_LOOP_INFERENCE` — only collect detailed GPU data during the profiling phase (skips warmup, keeps file small)

**Viewing results:** Open `InternVL_Timeline.nsys-rep` in the Nsight Systems GUI (download from NVIDIA). The timeline shows NVTX rows with `CLOSED_LOOP_INFERENCE` as the outer range and `InternVL_Vision_Encoder`, `InternVL_MLP_Connector`, `InternVL_LLM_Prefill`, `InternVL_LLM_Decode` nested inside.

**Connecting nsys + NCU:** Kernel names link the two. In nsys you see which kernels run in which subphase and how long they take; in NCU you see detailed hardware metrics for those same kernels.

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

- `VLLM_USE_PRECOMPILED=1 uv pip install --editable .` (single command with the env var)
- `pip install torch torchvision torchaudio timm einops pillow`

### 4. Model Code Modifications

To enable precise profiling of sub-phases, we modified the vLLM source code to include NVTX markers.
This allows the profiler to "see" where the model forward pass begins and ends.

Copy `NCU-Profiling-Guide/gpu_model_runner.py` from this repo over the existing `gpu_model_runner.py` in your vLLM clone. The path in the vLLM repo is:

`vllm/vllm/gpu_model_runner.py`

These are the modifications:

```diff
+   torch.cuda.nvtx.range_push("vLLM_Inference_Step")

    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )

+   torch.cuda.nvtx.range_pop()
```

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

Since our main goal is to find each subphase separately we need to add the NVTX hooks in different spots than I did

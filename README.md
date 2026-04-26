# Energy-efficient MLLM serving — measurements

Paper artifact: **code** (drivers + vLLM drop-ins), **data** (sweeps), and optional **analysis** (small summary CSVs).

Per-phase utilization, latency, and energy across SM clock and batch for **InternVL3-8B** and **LLaVA-1.5-7B** on NVIDIA A100.

## Headline result

Combined batching (**b=16**) and SM clock reduction (**210 MHz**) give a measured **27.2× per-output-token decode energy reduction on InternVL3-8B**, with a parallel LLaVA-1.5-7B sweep.

## Repository layout

```
./
├── README.md                 ← overview (this file)
├── code/README.md            ← how to install patches and run drivers (start here to reproduce)
├── code/internvl3/           ← InternVL3-8B: run.sh, profiling/, shell/, vllm_patches/
├── code/llava/               ← LLaVA-1.5-7B: same structure
├── data/                     ← raw sweep outputs (see data/DATA_DICTIONARY.md)
└── analysis/                 ← optional cross-sweep summary CSVs + short README
```

NVTX names use an **`InternVL_`** or **`LLaVA_`** prefix in traces; published tables often strip that to **`Prefill`**, **`Decode`**, **`Vision_Encoder`**, **`MLP_Connector`**.

## Reproduce measurements

All steps live in **[`code/README.md`](code/README.md)**:

1. Install **vLLM v0.19.1**.
2. Copy the **`*_patched.py`** files from **`code/internvl3/vllm_patches/`** or **`code/llava/vllm_patches/`** into your `site-packages` vLLM tree.
3. From **`code/internvl3`** or **`code/llava`**, run **`./run.sh`** (see subcommands and env vars there).

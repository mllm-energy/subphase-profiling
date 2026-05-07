# Energy-efficient MLLM serving — measurements

Paper artifact: **code** (drivers + vLLM drop-ins), **data** (sweeps), and optional **analysis** (small summary CSVs).

Per-phase utilization, latency, and energy across SM clock and batch for **InternVL3-8B** and **LLaVA-1.5-7B** on NVIDIA A100, under two contrasting multimodal workloads:

| Workload | InternVL3 image | InternVL3 input | LLaVA image | LLaVA input | Output tokens |
|---|---|---|---|---|---|
| **inputheavy** (VQA-style, prefill-dominated) | 1344×1344 | 128 text tok | 336×336 (fixed) | 128 text tok | 16 |
| **outputheavy** (generation-style, decode-dominated) | 448×448 | 128 text tok | 336×336 (fixed) | 128 text tok | 256 |

These match the spec used in the companion `mllm_sm_vars` SM-masking work, so the two studies are directly comparable.

## Repository layout

```
./
├── README.md                 ← overview (this file)
├── code/README.md            ← how to install patches and run drivers (start here to reproduce)
├── code/internvl3/           ← InternVL3-8B: run.sh, profiling/, shell/, vllm_patches/
├── code/llava/               ← LLaVA-1.5-7B: same structure
└── data/                     ← raw sweep outputs
    ├── internvl3_inputheavy/
    └── llava_inputheavy/
```

NVTX names use an **`InternVL_`** or **`LLaVA_`** prefix in traces; published tables often strip that to **`Prefill`**, **`Decode`**, **`Vision_Encoder`**, **`MLP_Connector`**.

## Reproduce measurements

All steps live in **[`code/README.md`](code/README.md)**:

1. Install **vLLM v0.19.1** (clean install, no other modifications).
2. Copy the **`*_patched.py`** files from **`code/internvl3/vllm_patches/`** or **`code/llava/vllm_patches/`** into your `site-packages` vLLM tree.
3. From **`code/internvl3`** or **`code/llava`**, run **`./run.sh inputheavy`** or **`./run.sh outputheavy`** (see subcommands and env vars there).

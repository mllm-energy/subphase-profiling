# Code — profiling drivers

Everything under `code/` reproduces **vLLM v0.19.1** inference subphase measurements on NVIDIA GPUs: **nsight systems** (timelines), **nsight compute** (SM/DRAM %), and **nvidia-smi** power sampling aligned to NVTX ranges.

Two self-contained trees (different models):

| Directory | Model | Entrypoint |
|-----------|--------|-----------------|
| `internvl3/` | InternVL3-8B | `./run.sh inputheavy` or `./run.sh outputheavy` |
| `llava/` | LLaVA-1.5-7B | `./run.sh inputheavy` or `./run.sh outputheavy` |

In each tree, **`run.sh` changes the working directory to that folder** so `eer.jpg` and `profiling/*.py` paths stay correct. Do not run the Python workloads from the repo root without adjusting paths.

---

## Workloads

Both trees expose the same two workloads, mirroring the `mllm_sm_vars` SM-masking spec:

| Workload | InternVL3 image | LLaVA image | Text input | Output tokens | Profile script |
|---|---|---|---|---|---|
| `inputheavy` | 1344×1344 (multi-tile) | 336×336 (fixed) | 128 tok | 16 tok | `profile_subphases_inputheavy.py` / `profile_llava_inputheavy.py` |
| `outputheavy` | 448×448 (single tile) | 336×336 (fixed) | 128 tok | 256 tok | `profile_subphases_outputheavy.py` / `profile_llava_outputheavy.py` |

Both profiles honour the **`BATCH_SIZE`** environment variable (default 1) — set `BATCH_SIZE=N` to issue N concurrent requests per profiled forward. The driver scripts use this for the batch sweep across `b ∈ {1, 2, 4, 8, 16}`.

LLaVA-1.5 has a fixed 336×336 image size (no dynamic tiling), so `inputheavy` vs `outputheavy` for LLaVA differs only in output-token count.

---

## Prerequisites

- **vLLM v0.19.1** — install first into a clean venv, then **overwrite** two modules with the `*_patched.py` files from the matching tree (commands below).
- **Nsight Compute** — e.g. `/usr/local/cuda/bin/ncu`
- **Nsight Systems** — e.g. `/usr/local/cuda/bin/nsys`
- **Python 3.10+** and whatever PyTorch vLLM pulls in
- **`zip`** for packaging
- **sudo** — GPU clock locking (`nvidia-smi -lgc`) and usually NCU profiling

A clean venv is critical — if your `vllm` install has been modified for any other project (e.g. SM-masking work), the patches in `vllm_patches/` will collide with those mods.

```bash
python3 -m venv /home/cc/subphase-profiling/.venv
source /home/cc/subphase-profiling/.venv/bin/activate
pip install --upgrade pip
pip install vllm==0.19.1
pip install timm einops pillow
```

---

## Shared layout (both trees)

| Path | Purpose |
|------|---------|
| `run.sh` | Bash driver (subcommands below) |
| `profiling/` | Python workloads + `align_power_nvtx.py` (integrates power CSV with NVTX windows from the nsys report) |
| `shell/` | `power_sample.sh`, `ncu_phases.sh` (NVTX include strings for NCU) |
| `vllm_patches/` | Vanilla upstream copies (`*.py`) + **`*_patched.py`** drop-ins for your install |
| `eer.jpg` | Test image at tree root |

**`align_power_nvtx.py`** exists in both trees; the LLaVA copy respects **`NVTX_PHASES`** (comma-separated phase names). `llava/run.sh` sets that from **`LLAVA_MODEL_PREFIX`**. InternVL uses fixed `InternVL_*` labels in the patched modules.

---

## Methodology gotchas (read once)

1. **Engine subprocess** — vLLM v1 can hide kernels from NCU. Export **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** (the drivers already do).
2. **NCU NVTX filters** — use a **trailing slash**, e.g. `--nvtx-include "CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/"`, or you may record zero kernels.
3. **Root / profiling permissions** — on many A100 setups NCU needs **`sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 ... ncu`** (LLaVA's NCU phases use this wrapper).
4. **Caches** — workloads disable prefix and multimodal processor caches and vary inputs so the profiled step is not a cache hit.
5. **Replay** — NCU **`--replay-mode application`** is used for speed; workloads keep **`temperature=0`** and fixed seeds so replay stays valid.

---

## InternVL3 (`internvl3/`)

### Install instrumentation into site-packages

```bash
cd internvl3
VLLM=$(python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')
cp vllm_patches/gpu_model_runner_patched.py "$VLLM/v1/worker/gpu_model_runner.py"
cp vllm_patches/internvl_patched.py           "$VLLM/model_executor/models/internvl.py"
```

Unpatched `gpu_model_runner.py` / `internvl.py` in the same folder are reference snapshots for diffing against **your** wheel if line numbers drift.

### `run.sh` commands

| Command | Meaning |
|---------|---------|
| `inputheavy` | Full pipeline: freq sweep (default + 5 locked clocks) + batch sweep + `combined_energy.csv` + zip — **input-heavy** workload |
| `outputheavy` | Same but **output-heavy** workload |
| `batch <workload>` | Batch sweep only (under `<RESULTS>/batch_sweep`) |
| `ncu-sweep <workload>` | Four NCU phases only |
| `one <workload> <label> <outdir>` | One config: 3× nsys+power + NCU per phase |
| `help` | Usage |

**Environment (optional):** `INTERNVL_RESULTS_INPUTHEAVY`, `INTERNVL_RESULTS_OUTPUTHEAVY`, `INTERNVL_ZIP_INPUTHEAVY`, `INTERNVL_ZIP_OUTPUTHEAVY`, `INTERNVL_SKIP_ZIP=1`.

### Examples

```bash
cd internvl3
./run.sh help

# Full sweep, input-heavy workload (overnight):
sudo -E nohup ./run.sh inputheavy > ~/results_inputheavy/sweep_log.txt 2>&1 & disown

# One-config smoke test:
./run.sh one inputheavy default /tmp/smoke

# Manual NCU example:
sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH=$PATH HOME=$HOME \
  /usr/local/cuda/bin/ncu --nvtx \
  --nvtx-include "CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/" \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --target-processes all --replay-mode application --launch-count 200 \
  -o ./out/Prefill \
  python3 profiling/profile_subphases_inputheavy.py
```

---

## LLaVA (`llava/`)

### Install instrumentation

Upstream revision is recorded in **`vllm_patches/SOURCE.txt`** (if present).

```bash
cd llava
VLLM=$(python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')
cp vllm_patches/gpu_model_runner_patched.py "$VLLM/v1/worker/gpu_model_runner.py"
cp vllm_patches/llava_patched.py           "$VLLM/model_executor/models/llava.py"
```

Sanity check: `python3 -c "import vllm; from vllm.model_executor.models import llava"`

### `run.sh` commands

| Command | Meaning |
|---------|---------|
| `inputheavy` | Full pipeline: default clock + 5-freq sweep + batch sweep + `combined_energy.csv` + zip |
| `outputheavy` | Same but output-heavy workload |
| `batch <workload>` | Batch sweep only |
| `ncu-sweep <workload>` | Four NCU phases using the chosen workload |
| `one <workload> <label> <outdir>` | 3× nsys reps + aggregate + four NCU phases |
| `help` | Usage |

**Environment:** `LLAVA_RESULTS_INPUTHEAVY`, `LLAVA_RESULTS_OUTPUTHEAVY`, `LLAVA_ZIP_INPUTHEAVY`, `LLAVA_ZIP_OUTPUTHEAVY`, `LLAVA_SKIP_ZIP=1`, `LLAVA_MODEL_PREFIX` (default `LLaVA`; must match NVTX strings in the patched files).

```bash
cd llava
sudo -E nohup ./run.sh inputheavy > /path/to/results_llava_inputheavy/sweep_log.txt 2>&1 & disown
```

**Gotchas:** install **`zip`** before the run finishes; both clock locking and NCU typically need sudo (the script wraps NCU accordingly).

---

## Result directory shape

Under your chosen results root (e.g. `INTERNVL_RESULTS_INPUTHEAVY` or `LLAVA_RESULTS_INPUTHEAVY`):

```
freq_default/, freq_<MHz>/
  rep_1/ rep_2/ rep_3/
    timeline.nsys-rep, power_samples.csv, phase_energy.csv
  phase_energy_agg.csv
  Prefill.ncu-rep, Decode.ncu-rep, ...
batch_sweep/b1/ b2/ b4/ b8/ b16/
  rep_*/ ... phase_energy_agg.csv
combined_energy.csv
```

---

## Hardware used in the paper

NVIDIA A100-PCIE-40GB, driver **535.288.01**, CUDA **12.2** (Chameleon Cloud bare metal). Other GPUs/drivers may work but were not part of the original artifact.

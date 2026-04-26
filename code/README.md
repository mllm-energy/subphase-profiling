# Code — profiling drivers

Everything under `code/` reproduces **vLLM v0.19.1** inference subphase measurements on NVIDIA GPUs: **nsight systems** (timelines), **nsight compute** (SM/DRAM %), and **nvidia-smi** power sampling aligned to NVTX ranges.

Two self-contained trees (different models, different machines in the original study):

| Directory | Model | Your entrypoint |
|-----------|--------|-----------------|
| `internvl3/` | InternVL3-8B | `./run.sh` |
| `llava/` | LLaVA-1.5-7B | `./run.sh` |

In each tree, **`run.sh` changes the working directory to that folder** so `eer.jpg` and `profiling/*.py` paths stay correct. Do not run the Python workloads from the repo root without adjusting paths.

---

## Prerequisites

- **vLLM v0.19.1** — install first, then **overwrite** two modules with the `*_patched.py` files from the matching tree (commands below).
- **Nsight Compute** — e.g. `/usr/local/cuda/bin/ncu`
- **Nsight Systems** — e.g. `/usr/local/cuda/bin/nsys`
- **Python 3.10+** and whatever PyTorch vLLM pulls in
- **`zip`** for packaging (overnight drivers)
- **sudo** — GPU clock locking (`nvidia-smi -lgc`) and usually NCU profiling

---

## Shared layout (both trees)

| Path | Purpose |
|------|---------|
| `run.sh` | Bash driver (subcommands below) |
| `profiling/` | Python workloads + `align_power_nvtx.py` (integrates power CSV with NVTX windows from the nsys report) |
| `shell/` | `power_sample.sh`, `ncu_phases.sh` (NVTX include strings for NCU) |
| `vllm_patches/` | Vanilla upstream copies (`*.py`) + **`*_patched.py`** drop-ins for your install |
| `eer.jpg` | Test image at tree root (not always committed; place your own if missing) |

**`align_power_nvtx.py`** exists in both trees; the LLaVA copy respects **`NVTX_PHASES`** (comma-separated phase names). `llava/run.sh` sets that from **`LLAVA_MODEL_PREFIX`**. InternVL uses fixed `InternVL_*` labels in the patched modules.

---

## Methodology gotchas (read once)

1. **Engine subprocess** — vLLM v1 can hide kernels from NCU. Export **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** (the drivers already do).
2. **NCU NVTX filters** — use a **trailing slash**, e.g. `--nvtx-include "CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/"`, or you may record zero kernels.
3. **Root / profiling permissions** — on many A100 setups NCU needs **`sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 ... ncu`** (LLaVA’s NCU phases use this wrapper).
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

### Workloads (`profiling/`)

- **`profile_subphases.py`** — short prompt, one 448×448 tile  
- **`profile_subphases_realistic.py`** — long prompt, ten 1344×1344 tiles  
- **`profile_subphases_batch.py`** — batch size from **`BATCH_SIZE`** env var  

All wrap the timed `generate()` in **`CLOSED_LOOP_INFERENCE`** and use five distinct (image, prompt) pairs.

### `run.sh` commands

| Command | Meaning |
|---------|---------|
| `realistic` | Default full sweep: frequencies + batch + `combined_energy.csv` + optional realism check + zip |
| `small` | Smaller workload overnight + zip |
| `batch` | Batch sweep only |
| `ncu-sweep` | Four NCU phases, small profile |
| `one-small <label> <outdir>` | One config, small profile |
| `one <label> <outdir> [profile.py]` | One config, realistic profile (default), 3× nsys reps + NCU |
| `help` | Usage |

**Environment (optional):** `INTERNVL_RESULTS_SMALL`, `INTERNVL_RESULTS_REALISTIC`, `INTERNVL_ZIP_SMALL`, `INTERNVL_ZIP_REALISTIC`, `INTERNVL_SKIP_ZIP=1`, `INTERNVL_SKIP_REALISM=1`.

### Examples

```bash
cd internvl3
./run.sh help

sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH=$PATH HOME=$HOME \
  /usr/local/cuda/bin/ncu --nvtx \
  --nvtx-include "CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/" \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --target-processes all --replay-mode application --launch-count 200 \
  -o ./out/Prefill \
  python3 profiling/profile_subphases.py

sudo -E nohup ./run.sh > ~/results2/sweep_log.txt 2>&1 & disown
```

---

## LLaVA (`llava/`)

### Install instrumentation

Upstream revision is recorded in **`vllm_patches/SOURCE.txt`**.

```bash
cd llava
VLLM=$(python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')
cp vllm_patches/gpu_model_runner_patched.py "$VLLM/v1/worker/gpu_model_runner.py"
cp vllm_patches/llava_patched.py           "$VLLM/model_executor/models/llava.py"
```

Sanity check: `python3 -c "import vllm; from vllm.model_executor.models import llava"`

### Workloads (`profiling/`)

`profile_llava.py` (short), `profile_llava_realistic.py` (paper-like long prompt), `profile_llava_batch.py` ( **`BATCH_SIZE`** ). Same cache-defeat pattern as InternVL.

### `run.sh` commands

| Command | Meaning |
|---------|---------|
| `realistic` | Default: default clock + frequency sweep + batch + `combined_energy.csv` + zip |
| `batch` | Batch sweep only (optional out root: `./run.sh batch /path`) |
| `ncu-sweep` | Four NCU phases using the **small** profile |
| `one <label> <outdir> [profile.py]` | 3× nsys reps + aggregate + four NCU phases |
| `help` | Usage |

**Environment:** `LLAVA_RESULTS`, `LLAVA_ZIP`, `LLAVA_SKIP_ZIP=1`, `LLAVA_MODEL_PREFIX` (default `LLaVA`; must match NVTX strings in the patched files), `LLAVA_NCU_SWEEP_OUT`.

```bash
cd llava
sudo -E nohup ./run.sh > /path/to/results_llava/sweep_log.txt 2>&1 & disown
```

**Gotchas:** install **`zip`** before the run finishes; both clock locking and NCU typically need sudo (the script wraps NCU accordingly).

---

## Result directory shape (v2-style)

Under your chosen results root (e.g. `INTERNVL_RESULTS_REALISTIC` or `LLAVA_RESULTS`):

```
freq_default/, freq_<MHz>/
  rep_1/ rep_2/ rep_3/
    timeline.nsys-rep, power_samples.csv, phase_energy.csv
  phase_energy_agg.csv
  Prefill.ncu-rep, Decode.ncu-rep, ...
batch_sweep/b1/ … b16/
  rep_*/ … phase_energy_agg.csv
combined_energy.csv
```

InternVL **`realistic`** may also write **`realism_check.txt`** (small vs realistic comparison).

---

## Hardware used in the paper

NVIDIA A100-PCIE-40GB, driver **535.288.01**, CUDA **12.2** (Chameleon Cloud bare metal). Other GPUs/drivers may work but were not part of the original artifact.

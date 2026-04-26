# Handoff: cross-model replication on second A100

You are a Claude Code session running on a second A100 (Chameleon Cloud bare metal),
in parallel with another session that's running the primary frequency sweep on the
first A100. Your job: replicate the methodology on a different MLLM so the paper's
findings aren't InternVL3-specific.

## Project context

**Paper**: "Energy-efficient Multi-Modal LLM Serving" — characterization study showing
that MLLM inference subphases (Vision Encoder, Prefill, Decode, MLP Connector) have
heterogeneous resource demands, the GPU is underutilized in every phase, and per-phase
frequency multiplexing could save substantial energy.

**Hypothesis**: "Sub-phases during inference serving of MLLMs do not exhaust GPU resources."

**Headline result so far** (InternVL3-8B, A100, b=1):
- Decode owns 88% of wall time but uses 13% mean SM utilization
- Locking SM clock to 210 MHz costs +14% latency on Decode but saves 40% energy
- Prefill, Decode, Vision_Encoder all show 30–50% energy savings at lower clocks

A reviewer's first question: "is this just an InternVL3 quirk?" Your job is to
collect the same data on a second model so we can answer "no."

## Target model: LLaVA-1.5-7B (or Qwen2-VL-7B as backup)

Why LLaVA-1.5-7B:
- Different vision encoder (CLIP ViT-L/14) vs InternVL's InternViT-300M
- Different MLP connector (2-layer MLP vs single GEMM)
- Same approximate size (7B vs 8B LLM)
- Native vLLM support, no extra config

Why Qwen2-VL-7B as backup:
- Even more architecturally distant (Qwen language tower, different vision approach)
- Use this if LLaVA's NVTX patching is hard

## Methodology recap (DO NOT SKIP — these are non-obvious gotchas the first session learned)

1. **vLLM v1 spawns kernels in a child subprocess (EngineCore).** ncu can't see them
   by default. Fix: set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so everything runs in the
   main process.

2. **ncu's NVTX filter syntax is footgun-y.** Bare `--nvtx-include "Name"` matches
   only at start/end of range and silently profiles zero kernels. Use the trailing
   slash form: `--nvtx-include "Name/"` to match kernels INSIDE the range. Use nested
   form `--nvtx-include "Outer/Inner/"` to scope to inside-an-outer-range only.

3. **GPU performance counters require root.** A100 has `RmProfilingAdminOnly=1`
   (check via `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`).
   Run ncu via `sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH=$PATH HOME=$HOME
   /usr/local/cuda/bin/ncu ...`

4. **vLLM's prefix cache and MM encoder cache make the profiled run a no-op.** Same
   prompt + same image across warmup and profiling means the profiled run is a cache
   hit and prefill/vision encoder skip almost entirely. Fix:
   - `enable_prefix_caching=False`
   - `mm_processor_cache_gb=0`
   - 5 distinct (image, prompt) pairs (numpy noise on the image bytes defeats hash matching)

5. **`--replay-mode application` is much faster than per-kernel replay** for ncu, and
   it works as long as the run is deterministic (`temperature=0.0`, fixed numpy seeds).

6. **Power sampling**: `nvidia-smi --query-gpu=power.draw,clocks.gr,clocks.mem
   --format=csv,noheader,nounits -lms 50`, run as a background process during nsys
   profiling, then align by timestamp to NVTX windows from `timeline.sqlite`.

## Code on the first A100 (copy this over)

The first session has working scripts at `/home/cc/subphase-profiling/NCU-Profiling-Guide/`.
Files of interest:
- `profile_subphases.py` — base profile script (4 warmup + 1 profiled, NVTX-marked)
- `profile_subphases_realistic.py` — same but with longer prompt + multi-tile image
- `profile_subphases_batch.py` — same but parameterized by `BATCH_SIZE` env var
- `power_sample.sh` — background nvidia-smi sampler
- `align_power_nvtx.py` — integrate power(t) over NVTX windows → joules per phase
- `aggregate_reps.py` — combine N reps to mean+std
- `run_one_freq_v2.sh` — 3 reps + ncu sweep at one freq
- `run_overnight_v2.sh` — top-level driver (P0 + freq sweep + batch sweep + zip)

Get the directory across:
```
# from your laptop:
scp -r cc@<first-a100-ip>:~/subphase-profiling/NCU-Profiling-Guide/ \
    cc@<second-a100-ip>:~/subphase-profiling/
```

Or git: have the first session push to a temporary repo, second session clones.

## Required NVTX patches (translate from InternVL to LLaVA)

The first session patched two vLLM files. You need analogous patches:

**A. `vllm/v1/worker/gpu_model_runner.py`** — already model-agnostic, copy as-is.
Adds NVTX ranges around `_model_forward` distinguishing Prefill vs Decode via
`self._is_uniform_decode()`. The patch lives around line 4040–4047 in the current
vLLM v0.19.1.

**B. Model-specific vision encoder patch.** For InternVL3 the patch was in
`vllm/model_executor/models/internvl.py:extract_feature()` adding NVTX around
`self.vision_model(...)` and `self.mlp1(...)`.

For LLaVA-1.5, the equivalent location is in
`vllm/model_executor/models/llava.py`. Look for the method that calls the vision
tower (likely named `_image_pixels_to_features` or `_process_image_input`) and the
MLP projector (likely `multi_modal_projector` or similar). Add:
```python
torch.cuda.nvtx.range_push("LLaVA_Vision_Encoder")
... vision tower call ...
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("LLaVA_MLP_Connector")
... projector call ...
torch.cuda.nvtx.range_pop()
```

Then update the NVTX names in the runtime patch in `gpu_model_runner.py` from
`InternVL_LLM_Prefill`/`InternVL_LLM_Decode` to `LLaVA_LLM_Prefill`/`LLaVA_LLM_Decode`
(or use generic names like `LLM_Prefill`/`LLM_Decode` so the same patch works for
both models).

## What to run

Same workflow as the first session, but on LLaVA. Suggested execution:

1. **Apply NVTX patches** (10–20 min)
2. **Sanity test**: `VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 profile_llava.py`
   to confirm the model loads and NVTX ranges appear in nsys.
3. **Single nsys+power run** to confirm phase durations are non-zero
4. **One ncu phase** (e.g., Prefill) at default clock to confirm kernels are profiled
5. **Full overnight sweep**: P0 + 5 frequencies + batch sweep, save to `/home/cc/results_llava/`
6. **Zip and prepare for download**

## Useful snippets

**Check ncu permissions:**
```
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
sudo nvidia-smi -pm 1
```

**One-shot test command pattern (after patches applied):**
```
sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH=$PATH HOME=$HOME \
  /usr/local/cuda/bin/ncu --nvtx --nvtx-include "CLOSED_LOOP_INFERENCE/LLaVA_LLM_Prefill/" \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --target-processes all --replay-mode application --launch-count 50 \
  -o /home/cc/results_llava/test_prefill \
  python3 profile_llava.py
```

## Output directory contract

Match the first session's structure so cross-model joins are trivial:
```
/home/cc/results_llava/
  freq_default/, freq_{210,510,810,1110,1410}MHz/
    rep_1/, rep_2/, rep_3/  (timeline.nsys-rep, power_samples.csv, phase_energy.csv)
    phase_energy_agg.csv
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.ncu-rep
  batch_sweep/b{1,4,8,16}/
  combined_energy.csv
  sweep_log.txt
```

## When done

Zip the results: `cd /home/cc && zip -r results_llava.zip results_llava/ -x "*.sqlite"`
Tell the user the scp command. Both sweeps' zips will be merged for the paper.

## Don't waste time on

- Trying to use `nvidia-smi -lmc` for memory clock — A100 PCIe usually doesn't expose this
- Fancy tokenizer tricks — short prompts are fine; the methodology is invariant
- Large batch sizes (>16) without checking VRAM headroom first
- Re-deriving the cache-defeat strategy — the recipe in `profile_subphases.py` works

## Quick context dump

The first session's full conversation included diagnosing 3 stacked bugs (subprocess
profiling, NVTX filter syntax, perf-counter permissions), then validating cache defeat
via direct shape inspection of `pixel_values_flat`, then doing a 5-frequency sweep
with --replay-mode application that finished in ~3 hours and showed:

- Decode SM 13%, DRAM 22% (memory-bound)
- Prefill SM 41%, DRAM 19% (compute-bound on matmul)
- Vision_Encoder SM 36%, DRAM 12% (mixed)
- 40% Decode energy savings at 210 MHz vs boost, +14% latency hit

Currently a second sweep at "realistic" workload (10 tiles + 535-token prompt) is
running on the first A100 to confirm findings hold at production scales. Expected to
produce nearly identical numbers because at b=1 the GPU absorbs even 10x larger
inputs without runtime growth — itself a paper-relevant finding.

Your job is parallel and doesn't depend on those results. Get LLaVA on the same axes.


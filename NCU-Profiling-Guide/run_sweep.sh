#!/usr/bin/env bash
# Sequential ncu sweep across the four InternVL3-8B subphases.
# Run as:  sudo -E ./run_sweep.sh   (sudo needed for GPU perf counters)
set -u
cd "$(dirname "$0")"

OUT=/home/cc/results
mkdir -p "$OUT"
NCU=/usr/local/cuda/bin/ncu
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
COMMON=(--metrics "$METRICS" --target-processes all --replay-mode application)

export VLLM_ENABLE_V1_MULTIPROCESSING=0

# (phase_name, nvtx_filter, launch_count) — counts cap kernel replays so the sweep finishes overnight.
# Decode runs 64 steps with the same kernel set, so 200 launches covers every unique shape with margin.
PHASES=(
  "Prefill           CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/        200"
  "Decode            CLOSED_LOOP_INFERENCE/InternVL_LLM_Decode/         200"
  "Vision_Encoder    CLOSED_LOOP_INFERENCE/InternVL_Vision_Encoder/     200"
  "MLP_Connector     CLOSED_LOOP_INFERENCE/InternVL_MLP_Connector/      50"
)

for row in "${PHASES[@]}"; do
  read -r name filter count <<< "$row"
  echo
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] Phase: $name (launch-count=$count)"
  echo "  filter: $filter"
  echo "=========================================="
  log="$OUT/${name}_log.txt"
  rep="$OUT/${name}"

  "$NCU" --nvtx --nvtx-include "$filter" \
         "${COMMON[@]}" \
         --launch-count "$count" \
         -o "$rep" \
         python3 profile_subphases.py \
         > "$log" 2>&1

  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] FAILED ($status). Last log lines:"
    tail -20 "$log"
    echo "Continuing to next phase..."
    continue
  fi

  # Quick sanity check from the report
  kernels=$("$NCU" --import "$rep.ncu-rep" --csv 2>/dev/null | tail -n +2 | wc -l)
  echo "[$(date +%H:%M:%S)] $name done. $kernels metric rows in $rep.ncu-rep"
done

echo
echo "Sweep complete. Reports in $OUT/"
ls -la "$OUT"/*.ncu-rep

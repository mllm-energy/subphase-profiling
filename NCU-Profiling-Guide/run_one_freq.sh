#!/usr/bin/env bash
# Run nsys+power session, then ncu 4-phase sweep, at the currently-locked clock.
# Usage:  ./run_one_freq.sh <freq_mhz> <out_dir>
# Caller is responsible for sudo nvidia-smi -lgc <freq> beforehand.
set -u
FREQ="${1:?need freq mhz}"
OUTDIR="${2:?need out dir}"
mkdir -p "$OUTDIR"

cd "$(dirname "$0")"
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
SCRIPT_DIR="$(pwd)"
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "[$(date +%H:%M:%S)] === freq=${FREQ}MHz, out=$OUTDIR ==="
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

# ---- Step A: nsys + power sampling (real timing → energy) ----
echo "[$(date +%H:%M:%S)]  A) nsys + power sampling"
PWR_CSV="$OUTDIR/power_samples.csv"
bash "$SCRIPT_DIR/power_sample.sh" "$PWR_CSV" &
PWR_PID=$!
sleep 0.3

"$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
        -o "$OUTDIR/timeline" \
        python3 "$SCRIPT_DIR/profile_subphases.py" \
        > "$OUTDIR/nsys_log.txt" 2>&1

kill "$PWR_PID" 2>/dev/null || true
wait "$PWR_PID" 2>/dev/null || true

# Align power to NVTX windows -> phase_energy.csv
python3 "$SCRIPT_DIR/align_power_nvtx.py" \
        "$OUTDIR/timeline.nsys-rep" "$PWR_CSV" "$OUTDIR/phase_energy.csv" \
        > "$OUTDIR/align_log.txt" 2>&1 || echo "  align failed (see align_log.txt)"

# ---- Step B: ncu 4-phase utilization sweep ----
echo "[$(date +%H:%M:%S)]  B) ncu 4-phase sweep"
PHASES=(
  "Prefill         CLOSED_LOOP_INFERENCE/InternVL_LLM_Prefill/        200"
  "Decode          CLOSED_LOOP_INFERENCE/InternVL_LLM_Decode/         200"
  "Vision_Encoder  CLOSED_LOOP_INFERENCE/InternVL_Vision_Encoder/     200"
  "MLP_Connector   CLOSED_LOOP_INFERENCE/InternVL_MLP_Connector/      50"
)
for row in "${PHASES[@]}"; do
  read -r name filter count <<< "$row"
  log="$OUTDIR/${name}_log.txt"
  rep="$OUTDIR/${name}"
  echo "[$(date +%H:%M:%S)]    ncu phase: $name (lc=$count)"
  "$NCU" --nvtx --nvtx-include "$filter" \
         --metrics "$METRICS" --target-processes all \
         --replay-mode application --launch-count "$count" \
         -o "$rep" python3 "$SCRIPT_DIR/profile_subphases.py" \
         > "$log" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)]    FAILED: $name (see $log)"
  fi
done

echo "[$(date +%H:%M:%S)] === freq=${FREQ}MHz done ==="

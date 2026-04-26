#!/usr/bin/env bash
# v2: 3 reps of nsys+power (for noise estimation), then 1 ncu 4-phase sweep.
# Usage:  ./run_one_freq_v2.sh <freq_label> <out_dir> <profile_script> [model_prefix]
#   model_prefix defaults to "InternVL"; set "LLaVA" for the LLaVA sweep.
#   align_power_nvtx.py reads NVTX_PHASES env var (export it from the driver).
set -u
FREQ_LABEL="${1:?need freq label}"
OUTDIR="${2:?need out dir}"
PROFILE="${3:?need profile script path}"
MODEL_PREFIX="${4:-InternVL}"
mkdir -p "$OUTDIR"

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "[$(date +%H:%M:%S)] === freq=$FREQ_LABEL, profile=$(basename $PROFILE), out=$OUTDIR ==="
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

# ---- Step A: 3 reps of nsys + power sampling ----
N_REPS=3
for rep in $(seq 1 $N_REPS); do
  REP_DIR="$OUTDIR/rep_$rep"
  mkdir -p "$REP_DIR"
  echo "[$(date +%H:%M:%S)]   A.$rep) nsys + power (rep $rep/$N_REPS)"
  PWR_CSV="$REP_DIR/power_samples.csv"
  bash "$SCRIPT_DIR/power_sample.sh" "$PWR_CSV" &
  PWR_PID=$!
  sleep 0.3
  "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
          -o "$REP_DIR/timeline" \
          python3 "$PROFILE" \
          > "$REP_DIR/nsys_log.txt" 2>&1
  kill "$PWR_PID" 2>/dev/null || true
  wait "$PWR_PID" 2>/dev/null || true
  python3 "$SCRIPT_DIR/align_power_nvtx.py" \
          "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
          > "$REP_DIR/align_log.txt" 2>&1 || echo "    align failed (rep $rep)"
done

# Aggregate reps -> phase_energy_agg.csv with mean+std
python3 "$SCRIPT_DIR/aggregate_reps.py" "$OUTDIR" "$OUTDIR/phase_energy_agg.csv" \
        > "$OUTDIR/aggregate_log.txt" 2>&1 || echo "  aggregation failed"

# ---- Step B: ncu 4-phase utilization sweep (single run; replay handles statistics) ----
echo "[$(date +%H:%M:%S)]   B) ncu 4-phase sweep"
PHASES=(
  "Prefill         CLOSED_LOOP_INFERENCE/${MODEL_PREFIX}_LLM_Prefill/        200"
  "Decode          CLOSED_LOOP_INFERENCE/${MODEL_PREFIX}_LLM_Decode/         200"
  "Vision_Encoder  CLOSED_LOOP_INFERENCE/${MODEL_PREFIX}_Vision_Encoder/     200"
  "MLP_Connector   CLOSED_LOOP_INFERENCE/${MODEL_PREFIX}_MLP_Connector/      50"
)
for row in "${PHASES[@]}"; do
  read -r name filter count <<< "$row"
  log="$OUTDIR/${name}_log.txt"
  rep="$OUTDIR/${name}"
  echo "[$(date +%H:%M:%S)]      ncu phase: $name (lc=$count)"
  sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH="$PATH" HOME="$HOME" \
    "$NCU" --nvtx --nvtx-include "$filter" \
         --metrics "$METRICS" --target-processes all \
         --replay-mode application --launch-count "$count" \
         -o "$rep" python3 "$PROFILE" \
         > "$log" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)]      FAILED: $name"
  fi
done

echo "[$(date +%H:%M:%S)] === freq=$FREQ_LABEL done ==="

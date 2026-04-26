#!/usr/bin/env bash
# Batch sweep at default clock. nsys+power only (no ncu). 3 reps per batch size.
# Usage: ./run_batch_sweep.sh [outroot] [profile_script]
#   outroot defaults to /home/cc/results2/batch_sweep
#   profile_script defaults to profile_subphases_batch.py
#   align_power_nvtx.py reads NVTX_PHASES env var (export it from the driver).
set -u
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
NSYS=/usr/local/cuda/bin/nsys
OUTROOT="${1:-/home/cc/results2/batch_sweep}"
PROFILE_BATCH="${2:-$SCRIPT_DIR/profile_subphases_batch.py}"
mkdir -p "$OUTROOT"

export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "[$(date +%H:%M:%S)] === BATCH SWEEP START (default clock) ==="
nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader

for B in 1 4 8 16; do
  BDIR="$OUTROOT/b$B"
  mkdir -p "$BDIR"
  echo "[$(date +%H:%M:%S)] --- batch=$B ---"
  for rep in 1 2 3; do
    REP_DIR="$BDIR/rep_$rep"
    mkdir -p "$REP_DIR"
    echo "[$(date +%H:%M:%S)]   rep $rep/3"
    PWR_CSV="$REP_DIR/power_samples.csv"
    bash "$SCRIPT_DIR/power_sample.sh" "$PWR_CSV" &
    PWR_PID=$!
    sleep 0.3
    BATCH_SIZE=$B "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
        -o "$REP_DIR/timeline" \
        python3 "$PROFILE_BATCH" \
        > "$REP_DIR/nsys_log.txt" 2>&1 \
      || echo "[$(date +%H:%M:%S)]   FAILED rep $rep at B=$B (continuing)"
    kill "$PWR_PID" 2>/dev/null || true
    wait "$PWR_PID" 2>/dev/null || true
    python3 "$SCRIPT_DIR/align_power_nvtx.py" \
            "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
            > "$REP_DIR/align_log.txt" 2>&1 || true
  done
  python3 "$SCRIPT_DIR/aggregate_reps.py" "$BDIR" "$BDIR/phase_energy_agg.csv" \
          > "$BDIR/aggregate_log.txt" 2>&1 || true
done
echo "[$(date +%H:%M:%S)] === BATCH SWEEP DONE ==="

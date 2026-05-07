#!/usr/bin/env bash
# InternVL3-8B per-subphase batch sweep (nsys + power + NCU).
# No frequency sweep — runs at default GPU clock.
# Per (workload, batch) point: 3 nsys+power reps, 4 NCU phase profiles,
# then CSV exports (nsys cuda_gpu_trace + NCU details) so the data is
# ready to plot without opening any GUI.
#
# Two workloads (mirror mllm_sm_vars):
#   inputheavy  -> 1344x1344 image, 128 input tokens, 16  output tokens
#   outputheavy -> 448x448  image, 128 input tokens, 256 output tokens
#
# Usage:
#   ./run.sh [command] [args...]
#
#   inputheavy           batch sweep (1,2,4,8,16) + nsys + power + NCU + CSV exports
#   outputheavy          same but output-heavy
#   batch <wl>           batch sweep nsys+power only (no NCU)
#   ncu-sweep <wl>       4 NCU phases at default batch=1 only (smoke / quick check)
#   one <wl> <label> <outdir>   one-config full pipeline (3 reps + 4 NCU + CSV)
#   help
#
# Env: INTERNVL_RESULTS_INPUTHEAVY  (default /home/cc/results_inputheavy)
#      INTERNVL_RESULTS_OUTPUTHEAVY (default /home/cc/results_outputheavy)
#      INTERNVL_ZIP_INPUTHEAVY, INTERNVL_ZIP_OUTPUTHEAVY
#      INTERNVL_BATCHES (default "1 2 4 8 16")
#      INTERNVL_SKIP_ZIP=1
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHELL_DIR="$SCRIPT_DIR/shell"
PROFILING_DIR="$SCRIPT_DIR/profiling"
cd "$SCRIPT_DIR"
# shellcheck source=shell/ncu_phases.sh
source "$SHELL_DIR/ncu_phases.sh"

NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

RESULTS_IH="${INTERNVL_RESULTS_INPUTHEAVY:-/home/cc/results_inputheavy}"
RESULTS_OH="${INTERNVL_RESULTS_OUTPUTHEAVY:-/home/cc/results_outputheavy}"
ZIP_IH="${INTERNVL_ZIP_INPUTHEAVY:-/home/cc/results_inputheavy.zip}"
ZIP_OH="${INTERNVL_ZIP_OUTPUTHEAVY:-/home/cc/results_outputheavy.zip}"
PROFILE_IH="$PROFILING_DIR/profile_subphases_inputheavy.py"
PROFILE_OH="$PROFILING_DIR/profile_subphases_outputheavy.py"
BATCHES="${INTERNVL_BATCHES:-1 2 4 8 16}"

resolve_workload() {
  case "$1" in
    inputheavy)  echo "$RESULTS_IH" "$ZIP_IH" "$PROFILE_IH" ;;
    outputheavy) echo "$RESULTS_OH" "$ZIP_OH" "$PROFILE_OH" ;;
    *) echo "unknown workload: $1 (expected inputheavy|outputheavy)" >&2; return 1 ;;
  esac
}

# Mean+std of rep_*/phase_energy.csv -> phase_energy_agg.csv
aggregate_phase_energy_reps() {
  _AR_PARENT="$1" _AR_OUT="$2" python3 <<'PY'
import csv, glob, os
from statistics import mean, stdev

freq_dir = os.environ["_AR_PARENT"]
out_csv = os.environ["_AR_OUT"]
per_phase = {}
n_reps = 0
for rep_dir in sorted(glob.glob(os.path.join(freq_dir, "rep_*"))):
    csv_path = os.path.join(rep_dir, "phase_energy.csv")
    if not os.path.exists(csv_path):
        continue
    n_reps += 1
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ph = row["phase"]
            per_phase.setdefault(
                ph,
                {"energy_j": [], "duration_ms": [], "avg_power_w": [], "instances": int(row["instances"])},
            )
            per_phase[ph]["energy_j"].append(float(row["total_energy_j"]))
            per_phase[ph]["duration_ms"].append(float(row["total_duration_ms"]))
            per_phase[ph]["avg_power_w"].append(float(row["avg_power_w"]))

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase","n_reps","instances","energy_mean_j","energy_std_j",
                "duration_mean_ms","duration_std_ms","power_mean_w","power_std_w"])
    for ph, d in per_phase.items():
        es, ds, ps = d["energy_j"], d["duration_ms"], d["avg_power_w"]
        w.writerow([
            ph, len(es), d["instances"],
            round(mean(es), 4), round(stdev(es), 4) if len(es) > 1 else 0.0,
            round(mean(ds), 4), round(stdev(ds), 4) if len(ds) > 1 else 0.0,
            round(mean(ps), 3), round(stdev(ps), 3) if len(ps) > 1 else 0.0,
        ])
print(f"wrote {out_csv} from {n_reps} reps")
PY
}

# nsys -> CSV (cuda_gpu_trace report) so we can read kernel-level timing.
nsys_export_csv() {
  local NSYSREP="$1" OUT_PREFIX="$2"
  # --force-export=true bypasses nsys's "cache is stale" refusal that
  # otherwise fails the export when align_power_nvtx.py has already touched
  # the .sqlite for this nsys-rep (which it does to query NVTX events).
  "$NSYS" stats --report cuda_gpu_trace --format csv --force-export=true \
    --output "$OUT_PREFIX" "$NSYSREP" >/dev/null 2>&1 || \
    echo "    nsys stats export failed for $NSYSREP"
}

# NCU -> CSV (details page) so kernel SM/DRAM throughput is plotter-readable.
ncu_export_csv() {
  local NCUREP="$1" OUT_CSV="$2"
  "$NCU" --import "$NCUREP" --csv --page details > "$OUT_CSV" 2>/dev/null || \
    echo "    ncu csv export failed for $NCUREP"
}

# 3 reps of nsys+power, then 4 NCU phase profiles, then CSV exports.
# Used for both batch sweep and one-off "one" command.
one_config() {
  local OUTDIR="$1" PROFILE="$2" BATCH="${3:-1}"
  mkdir -p "$OUTDIR"
  echo "[$(date +%H:%M:%S)] === config out=$OUTDIR profile=$(basename "$PROFILE") batch=$BATCH ==="
  nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

  local N_REPS=3 rep
  for rep in $(seq 1 "$N_REPS"); do
    local REP_DIR="$OUTDIR/rep_$rep"
    mkdir -p "$REP_DIR"
    local PWR_CSV="$REP_DIR/power_samples.csv"
    setsid bash "$SHELL_DIR/power_sample.sh" "$PWR_CSV" &
    local PWR_PID=$!
    sleep 0.3
    BATCH_SIZE=$BATCH "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
      -o "$REP_DIR/timeline" \
      python3 "$PROFILE" \
      > "$REP_DIR/nsys_log.txt" 2>&1
    # Negative PID = whole process group. Kills the bash wrapper AND the
    # nvidia-smi child it spawned via setsid; otherwise nvidia-smi orphans
    # itself to PID 1 and keeps appending to power_samples.csv for hours.
    kill -- -"$PWR_PID" 2>/dev/null || kill "$PWR_PID" 2>/dev/null || true
    wait "$PWR_PID" 2>/dev/null || true
    python3 "$PROFILING_DIR/align_power_nvtx.py" \
      "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
      > "$REP_DIR/align_log.txt" 2>&1 || echo "    align failed rep=$rep"
    nsys_export_csv "$REP_DIR/timeline.nsys-rep" "$REP_DIR/timeline"
  done
  aggregate_phase_energy_reps "$OUTDIR" "$OUTDIR/phase_energy_agg.csv" \
    > "$OUTDIR/aggregate_log.txt" 2>&1 || echo "  aggregate failed"

  for row in "${INTERNVL_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    local log="$OUTDIR/${name}_log.txt" ncurep="$OUTDIR/${name}"
    echo "[$(date +%H:%M:%S)]    ncu: $name"
    # Per-phase launch caps (see shell/ncu_phases.sh): high enough to capture
    # every unique kernel + plenty of replicates for stable per-kernel stats,
    # without exploding decode at large output_tokens.
    BATCH_SIZE=$BATCH "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all \
      --replay-mode application --launch-count "$count" \
      -f -o "$ncurep" python3 "$PROFILE" \
      > "$log" 2>&1 || { echo "    FAILED $name"; tail -10 "$log"; continue; }
    ncu_export_csv "$ncurep.ncu-rep" "$OUTDIR/${name}.csv"
  done
}

cmd_batch_full() {
  local WL="$1"
  local RESULTS ZIP_OUT PROFILE
  read -r RESULTS ZIP_OUT PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  mkdir -p "$RESULTS"

  echo "[$(date +%H:%M:%S)] === batch-full ($WL) -> $RESULTS ==="
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  nvidia-smi -pm 1 >/dev/null 2>&1 || true

  local B
  for B in $BATCHES; do
    one_config "$RESULTS/b$B" "$PROFILE" "$B"
  done

  python3 "$SCRIPT_DIR/profiling/aggregate_results.py" "$RESULTS" "$WL" \
    > "$RESULTS/aggregate_results_log.txt" 2>&1 || echo "  aggregate failed"

  if [[ "${INTERNVL_SKIP_ZIP:-}" != "1" ]]; then
    zip_results "$RESULTS" "$ZIP_OUT" "README_DOWNLOAD_${WL}.txt"
  fi
  echo "[$(date +%H:%M:%S)] === done ($WL) ==="
}

# nsys-only batch sweep (no NCU). Faster. Useful for energy-only studies.
cmd_batch_nsys_only() {
  local WL="$1"
  local RESULTS _ PROFILE
  read -r RESULTS _ PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  mkdir -p "$RESULTS/batch_sweep"
  echo "[$(date +%H:%M:%S)] batch (nsys-only) ($WL) -> $RESULTS/batch_sweep"
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader

  local B rep BDIR REP_DIR PWR_CSV PWR_PID
  for B in $BATCHES; do
    BDIR="$RESULTS/batch_sweep/b$B"
    mkdir -p "$BDIR"
    for rep in 1 2 3; do
      REP_DIR="$BDIR/rep_$rep"
      mkdir -p "$REP_DIR"
      PWR_CSV="$REP_DIR/power_samples.csv"
      bash "$SHELL_DIR/power_sample.sh" "$PWR_CSV" &
      PWR_PID=$!
      sleep 0.3
      BATCH_SIZE=$B "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
        -o "$REP_DIR/timeline" \
        python3 "$PROFILE" \
        > "$REP_DIR/nsys_log.txt" 2>&1
      kill "$PWR_PID" 2>/dev/null || true
      wait "$PWR_PID" 2>/dev/null || true
      python3 "$PROFILING_DIR/align_power_nvtx.py" \
        "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
        > "$REP_DIR/align_log.txt" 2>&1 || true
      nsys_export_csv "$REP_DIR/timeline.nsys-rep" "$REP_DIR/timeline"
    done
    aggregate_phase_energy_reps "$BDIR" "$BDIR/phase_energy_agg.csv" \
      > "$BDIR/aggregate_log.txt" 2>&1 || true
  done
}

cmd_ncu_sweep() {
  local WL="$1"
  read -r RESULTS _ PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  local OUT="$RESULTS/ncu_sweep_b1"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] ncu-sweep batch=1 ($WL) -> $OUT"
  for row in "${INTERNVL_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    echo "[$(date +%H:%M:%S)] Phase: $name"
    local log="$OUT/${name}_log.txt" ncurep="$OUT/${name}"
    BATCH_SIZE=1 "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all --replay-mode application \
      --launch-count "$count" -f -o "$ncurep" \
      python3 "$PROFILE" > "$log" 2>&1 || { tail -20 "$log"; continue; }
    ncu_export_csv "$ncurep.ncu-rep" "$OUT/${name}.csv"
  done
  ls -la "$OUT"/*.ncu-rep 2>/dev/null || true
}

zip_results() {
  local RESULTS="$1" ZIP_OUT="$2" README_NAME="$3"
  local parent name
  parent="$(dirname "$RESULTS")"
  name="$(basename "$RESULTS")"
  rm -f "$ZIP_OUT"
  ( cd "$parent" && zip -r "$ZIP_OUT" "$name/" -x "${name}/*.sqlite" >/dev/null 2>&1 )
  echo "zip: $(ls -lh "$ZIP_OUT" | awk '{print $5, $NF}')"
  cat > "${parent}/${README_NAME}" <<EOF
Packaged $(date). scp user@host:${ZIP_OUT} ~/Downloads/
EOF
}

usage() {
  cat <<'EOF'
Usage: ./run.sh [command] [args...]
  inputheavy           batch sweep (1,2,4,8,16) + nsys+power+NCU + CSV exports + aggregate
  outputheavy          same, output-heavy
  batch <workload>     nsys-only batch sweep (no NCU)
  ncu-sweep <workload> NCU phases at batch=1 only
  one <workload> <label> <outdir>   one-batch (B=1) full pipeline at <outdir>
  help

Workloads (mirror mllm_sm_vars):
  inputheavy   1344x1344 image, 128 in, 16  out
  outputheavy  448x448  image, 128 in, 256 out

Env: INTERNVL_RESULTS_INPUTHEAVY, INTERNVL_RESULTS_OUTPUTHEAVY,
     INTERNVL_ZIP_INPUTHEAVY, INTERNVL_ZIP_OUTPUTHEAVY,
     INTERNVL_BATCHES (default "1 2 4 8 16"),
     INTERNVL_SKIP_ZIP=1
EOF
}

main() {
  local cmd="${1:-help}"
  case "$cmd" in
    help|-h|--help) usage; exit 0 ;;
    inputheavy)  cmd_batch_full inputheavy ;;
    outputheavy) cmd_batch_full outputheavy ;;
    batch)
      shift
      [[ $# -ge 1 ]] || { echo "usage: $0 batch <workload>"; exit 1; }
      cmd_batch_nsys_only "$1"
      ;;
    ncu-sweep)
      shift
      [[ $# -ge 1 ]] || { echo "usage: $0 ncu-sweep <workload>"; exit 1; }
      cmd_ncu_sweep "$1"
      ;;
    one)
      shift
      [[ $# -ge 3 ]] || { echo "usage: $0 one <workload> <label> <outdir>"; exit 1; }
      local WL="$1" LABEL="$2" OUTDIR="$3"
      local _R _Z PROFILE
      read -r _R _Z PROFILE <<< "$(resolve_workload "$WL")" || exit 1
      one_config "$OUTDIR/$LABEL" "$PROFILE" 1
      ;;
    *)
      echo "unknown command: $cmd" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"

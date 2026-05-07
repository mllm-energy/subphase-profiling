#!/usr/bin/env bash
# LLaVA-1.5-7B measurement drivers (nsys + power + NCU).
# Two workloads with identical structure (LLaVA-1.5 has fixed 336x336 image
# size, so the workload axis here is purely output-token count):
#   inputheavy  -> 128 input tokens, 16  output tokens
#   outputheavy -> 128 input tokens, 256 output tokens
#
# Usage:
#   ./run.sh [command] [args...]
#
#   inputheavy           freq sweep + batch sweep + combined_energy + zip
#   outputheavy          same but output-heavy
#   batch <workload>     batch sweep only for one workload
#   ncu-sweep <workload> four NCU phases only for one workload
#   one <wl> <label> <outdir>   3x nsys+power + NCU at one config
#   help
#
# Env: LLAVA_RESULTS_INPUTHEAVY  (default /home/cc/results_llava_inputheavy)
#      LLAVA_RESULTS_OUTPUTHEAVY (default /home/cc/results_llava_outputheavy)
#      LLAVA_ZIP_INPUTHEAVY, LLAVA_ZIP_OUTPUTHEAVY
#      LLAVA_MODEL_PREFIX (default LLaVA; must match NVTX strings in patched files)
#      LLAVA_SKIP_ZIP=1
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHELL_DIR="$SCRIPT_DIR/shell"
PROFILING_DIR="$SCRIPT_DIR/profiling"
cd "$SCRIPT_DIR"
# shellcheck source=shell/ncu_phases.sh
source "$SHELL_DIR/ncu_phases.sh"

P="$LLAVA_MODEL_PREFIX"
export NVTX_PHASES="CLOSED_LOOP_INFERENCE,${P}_LLM_Prefill,${P}_LLM_Decode,${P}_Vision_Encoder,${P}_MLP_Connector"

NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
FREQS=(210 510 810 1110 1410)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

RESULTS_IH="${LLAVA_RESULTS_INPUTHEAVY:-/home/cc/results_llava_inputheavy}"
RESULTS_OH="${LLAVA_RESULTS_OUTPUTHEAVY:-/home/cc/results_llava_outputheavy}"
ZIP_IH="${LLAVA_ZIP_INPUTHEAVY:-/home/cc/results_llava_inputheavy.zip}"
ZIP_OH="${LLAVA_ZIP_OUTPUTHEAVY:-/home/cc/results_llava_outputheavy.zip}"
PROFILE_IH="$PROFILING_DIR/profile_llava_inputheavy.py"
PROFILE_OH="$PROFILING_DIR/profile_llava_outputheavy.py"

resolve_workload() {
  case "$1" in
    inputheavy)  echo "$RESULTS_IH" "$ZIP_IH" "$PROFILE_IH" ;;
    outputheavy) echo "$RESULTS_OH" "$ZIP_OH" "$PROFILE_OH" ;;
    *) echo "unknown workload: $1 (expected inputheavy|outputheavy)" >&2; return 1 ;;
  esac
}

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
    w.writerow(
        [
            "phase",
            "n_reps",
            "instances",
            "energy_mean_j",
            "energy_std_j",
            "duration_mean_ms",
            "duration_std_ms",
            "power_mean_w",
            "power_std_w",
        ]
    )
    for ph, d in per_phase.items():
        es, ds, ps = d["energy_j"], d["duration_ms"], d["avg_power_w"]
        w.writerow(
            [
                ph,
                len(es),
                d["instances"],
                round(mean(es), 4),
                round(stdev(es), 4) if len(es) > 1 else 0.0,
                round(mean(ds), 4),
                round(stdev(ds), 4) if len(ds) > 1 else 0.0,
                round(mean(ps), 3),
                round(stdev(ps), 3) if len(ps) > 1 else 0.0,
            ]
        )

print(f"wrote {out_csv} from {n_reps} reps")
PY
}

cleanup_clocks() {
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true
}

one_freq() {
  local FREQ_LABEL="$1" OUTDIR="$2" PROFILE="$3"
  mkdir -p "$OUTDIR"
  echo "[$(date +%H:%M:%S)] === freq=$FREQ_LABEL profile=$(basename "$PROFILE") out=$OUTDIR ==="
  nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

  local N_REPS=3 rep
  for rep in $(seq 1 "$N_REPS"); do
    local REP_DIR="$OUTDIR/rep_$rep"
    mkdir -p "$REP_DIR"
    echo "[$(date +%H:%M:%S)]   nsys+power rep $rep/$N_REPS"
    local PWR_CSV="$REP_DIR/power_samples.csv"
    bash "$SHELL_DIR/power_sample.sh" "$PWR_CSV" &
    local PWR_PID=$!
    sleep 0.3
    "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
      -o "$REP_DIR/timeline" \
      python3 "$PROFILE" \
      > "$REP_DIR/nsys_log.txt" 2>&1
    kill "$PWR_PID" 2>/dev/null || true
    wait "$PWR_PID" 2>/dev/null || true
    python3 "$PROFILING_DIR/align_power_nvtx.py" \
      "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
      > "$REP_DIR/align_log.txt" 2>&1 || echo "    align failed (rep $rep)"
  done

  aggregate_phase_energy_reps "$OUTDIR" "$OUTDIR/phase_energy_agg.csv" \
    > "$OUTDIR/aggregate_log.txt" 2>&1 || echo "  aggregation failed"

  echo "[$(date +%H:%M:%S)]   ncu 4-phase sweep"
  local row name filter count log ncurep
  for row in "${LLAVA_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    log="$OUTDIR/${name}_log.txt"
    ncurep="$OUTDIR/${name}"
    echo "[$(date +%H:%M:%S)]      ncu phase: $name (lc=$count)"
    sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH="$PATH" HOME="$HOME" \
      "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all \
      --replay-mode application --launch-count "$count" \
      -o "$ncurep" python3 "$PROFILE" \
      > "$log" 2>&1 \
      || echo "[$(date +%H:%M:%S)]      FAILED: $name"
  done
  echo "[$(date +%H:%M:%S)] === freq=$FREQ_LABEL done ==="
}

cmd_ncu_sweep() {
  local WL="$1"
  read -r RESULTS _ PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  local OUT="$RESULTS/ncu_sweep"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] ncu-sweep ($WL) -> $OUT"
  local row name filter count log ncurep
  for row in "${LLAVA_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    echo "[$(date +%H:%M:%S)] Phase: $name"
    log="$OUT/${name}_log.txt"
    ncurep="$OUT/${name}"
    sudo -E env VLLM_ENABLE_V1_MULTIPROCESSING=0 PATH="$PATH" HOME="$HOME" \
      "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all \
      --replay-mode application --launch-count "$count" \
      -o "$ncurep" python3 "$PROFILE" \
      > "$log" 2>&1 || tail -20 "$log"
  done
  ls -la "$OUT"/*.ncu-rep 2>/dev/null || true
}

cmd_batch_sweep() {
  local WL="$1"
  read -r RESULTS _ PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  local OUTROOT="$RESULTS/batch_sweep"
  mkdir -p "$OUTROOT"
  echo "[$(date +%H:%M:%S)] === BATCH SWEEP START ($WL) -> $OUTROOT ==="
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader

  local B rep BDIR REP_DIR PWR_CSV PWR_PID
  for B in 1 2 4 8 16; do
    BDIR="$OUTROOT/b$B"
    mkdir -p "$BDIR"
    echo "[$(date +%H:%M:%S)] --- batch=$B ---"
    for rep in 1 2 3; do
      REP_DIR="$BDIR/rep_$rep"
      mkdir -p "$REP_DIR"
      echo "[$(date +%H:%M:%S)]   rep $rep/3"
      PWR_CSV="$REP_DIR/power_samples.csv"
      bash "$SHELL_DIR/power_sample.sh" "$PWR_CSV" &
      PWR_PID=$!
      sleep 0.3
      BATCH_SIZE=$B "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
        -o "$REP_DIR/timeline" \
        python3 "$PROFILE" \
        > "$REP_DIR/nsys_log.txt" 2>&1 \
        || echo "[$(date +%H:%M:%S)]   FAILED rep $rep at B=$B (continuing)"
      kill "$PWR_PID" 2>/dev/null || true
      wait "$PWR_PID" 2>/dev/null || true
      python3 "$PROFILING_DIR/align_power_nvtx.py" \
        "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
        > "$REP_DIR/align_log.txt" 2>&1 || true
    done
    aggregate_phase_energy_reps "$BDIR" "$BDIR/phase_energy_agg.csv" \
      > "$BDIR/aggregate_log.txt" 2>&1 || true
  done
  echo "[$(date +%H:%M:%S)] === BATCH SWEEP DONE ==="
}

combine_freq_energy_csv() {
  _CF_PARENT="$1" _CF_OUT="$2" python3 <<'PY'
import csv, glob, os, re, sys
parent = os.environ["_CF_PARENT"]
out_path = os.environ["_CF_OUT"]
rows = []
for d in sorted(glob.glob(os.path.join(parent, "freq_*"))):
    agg = os.path.join(d, "phase_energy_agg.csv")
    plain = os.path.join(d, "phase_energy.csv")
    path = agg if os.path.exists(agg) else plain if os.path.exists(plain) else None
    if not path:
        continue
    label = os.path.basename(d).replace("freq_", "")
    m = re.match(r"(\d+)MHz", label)
    freq = int(m.group(1)) if m else -1
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"freq_mhz": freq, **r})
if not rows:
    print("error: no freq_*/phase_energy*.csv under", parent, file=sys.stderr)
    sys.exit(2)
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_path} ({len(rows)} rows)")
PY
}

zip_results() {
  local RESULTS_DIR="$1" ZIP_PATH="$2" README_NAME="$3"
  local parent name
  parent="$(dirname "$RESULTS_DIR")"
  name="$(basename "$RESULTS_DIR")"
  rm -f "$ZIP_PATH"
  ( cd "$parent" && zip -r "$ZIP_PATH" "$name/" -x "${name}/*.sqlite" >/dev/null 2>&1 )
  echo "zip: $(ls -lh "$ZIP_PATH" | awk '{print $5, $NF}')"
  cat > "${parent}/${README_NAME}" <<EOF
Packaged $(date). scp user@host:${ZIP_PATH} ~/Downloads/
EOF
}

cmd_overnight() {
  local WL="$1"
  local RESULTS ZIP_OUT PROFILE
  read -r RESULTS ZIP_OUT PROFILE <<< "$(resolve_workload "$WL")" || exit 1
  mkdir -p "$RESULTS"
  trap 'echo "[$(date +%H:%M:%S)] cleanup: resetting GPU clocks"; cleanup_clocks' EXIT

  echo "[$(date +%H:%M:%S)] === LLaVA OVERNIGHT START ($WL) ==="
  echo "host: $(hostname)  user: $(whoami)"
  echo "model_prefix: $P  outroot: $RESULTS"
  echo "NVTX_PHASES=$NVTX_PHASES"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true

  echo "[$(date +%H:%M:%S)] === P0: baseline at default clock ==="
  one_freq "default" "$RESULTS/freq_default" "$PROFILE" || true

  local f
  for f in "${FREQS[@]}"; do
    echo
    echo "[$(date +%H:%M:%S)] === P1: locking SM clock to ${f} MHz ==="
    if ! sudo nvidia-smi -lgc "${f},${f}" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)]   ! lock to ${f}MHz failed, skipping"
      continue
    fi
    sleep 2
    actual=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)
    echo "[$(date +%H:%M:%S)]   locked. actual clocks.gr=${actual}MHz"
    one_freq "$f" "$RESULTS/freq_${f}MHz" "$PROFILE" || true
    cleanup_clocks
    sleep 2
  done

  echo
  echo "[$(date +%H:%M:%S)] === BATCH SWEEP ==="
  cmd_batch_sweep "$WL"

  echo
  echo "[$(date +%H:%M:%S)] === aggregating combined_energy.csv ==="
  combine_freq_energy_csv "$RESULTS" "$RESULTS/combined_energy.csv"

  if [[ "${LLAVA_SKIP_ZIP:-}" != "1" ]]; then
    echo
    echo "[$(date +%H:%M:%S)] === packaging zip ==="
    zip_results "$RESULTS" "$ZIP_OUT" "README_DOWNLOAD_LLAVA_${WL}.txt"
  fi

  echo "[$(date +%H:%M:%S)] === LLaVA OVERNIGHT DONE ($WL) ==="
  cleanup_clocks
}

usage() {
  cat <<'EOF'
Usage: ./run.sh [command] [args...]
  inputheavy           full pipeline: freq sweep + batch + combine + zip
  outputheavy          full pipeline: freq sweep + batch + combine + zip
  batch <workload>     batch sweep only
  ncu-sweep <workload> four NCU phases only
  one <workload> <label> <outdir>   one-config nsys+power+NCU
  help

Workloads (LLaVA-1.5 has fixed 336x336 image; axis is output-token count):
  inputheavy   128 in, 16  out
  outputheavy  128 in, 256 out

Env: LLAVA_RESULTS_INPUTHEAVY, LLAVA_RESULTS_OUTPUTHEAVY,
     LLAVA_ZIP_INPUTHEAVY, LLAVA_ZIP_OUTPUTHEAVY,
     LLAVA_MODEL_PREFIX (default LLaVA), LLAVA_SKIP_ZIP=1
EOF
}

main() {
  local cmd="${1:-help}"
  case "$cmd" in
    help|-h|--help) usage; exit 0 ;;
    inputheavy)  cmd_overnight inputheavy ;;
    outputheavy) cmd_overnight outputheavy ;;
    batch)
      shift
      [[ $# -ge 1 ]] || { echo "usage: $0 batch <workload>"; exit 1; }
      cmd_batch_sweep "$1"
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
      one_freq "$LABEL" "$OUTDIR" "$PROFILE"
      ;;
    *)
      echo "unknown command: $cmd" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"

#!/usr/bin/env bash
# InternVL3-8B measurement drivers (nsys + power + NCU).
# Default (no args): full realistic overnight pipeline (old run_overnight_v2).
#
# Usage:
#   ./run.sh [command] [args...]
#
#   realistic          freq sweep + batch + combined_energy + realism_check + zip
#   small              v1 small-workload overnight + zip
#   batch              batch sweep only (under INTERNVL_RESULTS_REALISTIC/batch_sweep)
#   ncu-sweep          four NCU phases only (small profile; writes under INTERNVL_RESULTS_SMALL)
#   one-small <lbl> <outdir>
#   one <lbl> <outdir> [profile.py]   3× nsys+power + NCU; default profile = profile_subphases_realistic.py
#   help
#
# Env: INTERNVL_RESULTS_SMALL, INTERNVL_RESULTS_REALISTIC, INTERNVL_ZIP_*,
#      INTERNVL_SKIP_ZIP=1, INTERNVL_SKIP_REALISM=1 — see code/README.md
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
FREQS=(210 510 810 1110 1410)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

RESULTS_SMALL="${INTERNVL_RESULTS_SMALL:-/home/cc/results}"
RESULTS_REAL="${INTERNVL_RESULTS_REALISTIC:-/home/cc/results2}"
ZIP_SMALL="${INTERNVL_ZIP_SMALL:-/home/cc/results_overnight.zip}"
ZIP_REAL="${INTERNVL_ZIP_REALISTIC:-/home/cc/results2_overnight.zip}"
PROFILE_REALISTIC="$PROFILING_DIR/profile_subphases_realistic.py"

# Mean+std of rep_*/phase_energy.csv → phase_energy_agg.csv
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
  nvidia-smi -rgc >/dev/null 2>&1 || true
}

one_freq_small() {
  local FREQ="$1" OUTDIR="$2"
  mkdir -p "$OUTDIR"
  echo "[$(date +%H:%M:%S)] === one-small freq=$FREQ out=$OUTDIR ==="
  nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

  local PWR_CSV="$OUTDIR/power_samples.csv"
  bash "$SHELL_DIR/power_sample.sh" "$PWR_CSV" &
  local PWR_PID=$!
  sleep 0.3
  "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
    -o "$OUTDIR/timeline" \
    python3 "$PROFILING_DIR/profile_subphases.py" \
    > "$OUTDIR/nsys_log.txt" 2>&1
  kill "$PWR_PID" 2>/dev/null || true
  wait "$PWR_PID" 2>/dev/null || true
  python3 "$PROFILING_DIR/align_power_nvtx.py" \
    "$OUTDIR/timeline.nsys-rep" "$PWR_CSV" "$OUTDIR/phase_energy.csv" \
    > "$OUTDIR/align_log.txt" 2>&1 || echo "  align failed"

  for row in "${INTERNVL_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    local log="$OUTDIR/${name}_log.txt" rep="$OUTDIR/${name}"
    echo "[$(date +%H:%M:%S)]    ncu: $name"
    "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all \
      --replay-mode application --launch-count "$count" \
      -o "$rep" python3 "$PROFILING_DIR/profile_subphases.py" \
      > "$log" 2>&1 || echo "    FAILED $name"
  done
}

one_freq_v2() {
  local FREQ_LABEL="$1" OUTDIR="$2" PROFILE="${3:-$PROFILE_REALISTIC}"
  mkdir -p "$OUTDIR"
  echo "[$(date +%H:%M:%S)] === one freq=$FREQ_LABEL profile=$(basename "$PROFILE") out=$OUTDIR ==="
  nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

  local N_REPS=3 rep
  for rep in $(seq 1 "$N_REPS"); do
    local REP_DIR="$OUTDIR/rep_$rep"
    mkdir -p "$REP_DIR"
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
      > "$REP_DIR/align_log.txt" 2>&1 || echo "    align failed rep=$rep"
  done
  aggregate_phase_energy_reps "$OUTDIR" "$OUTDIR/phase_energy_agg.csv" \
    > "$OUTDIR/aggregate_log.txt" 2>&1 || echo "  aggregate failed"

  for row in "${INTERNVL_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    local log="$OUTDIR/${name}_log.txt" rep="$OUTDIR/${name}"
    echo "[$(date +%H:%M:%S)]    ncu: $name"
    "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all \
      --replay-mode application --launch-count "$count" \
      -o "$rep" python3 "$PROFILE" \
      > "$log" 2>&1 || echo "    FAILED $name"
  done
}

cmd_ncu_sweep() {
  local OUT="$RESULTS_SMALL"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] ncu-sweep → $OUT"
  for row in "${INTERNVL_NCU_PHASE_ROWS[@]}"; do
    read -r name filter count <<< "$row"
    echo "[$(date +%H:%M:%S)] Phase: $name"
    local log="$OUT/${name}_log.txt" rep="$OUT/${name}"
    "$NCU" --nvtx --nvtx-include "$filter" \
      --metrics "$METRICS" --target-processes all --replay-mode application \
      --launch-count "$count" -o "$rep" \
      python3 "$PROFILING_DIR/profile_subphases.py" > "$log" 2>&1 || tail -20 "$log"
  done
  ls -la "$OUT"/*.ncu-rep 2>/dev/null || true
}

cmd_batch() {
  local OUTROOT="$RESULTS_REAL/batch_sweep"
  mkdir -p "$OUTROOT"
  echo "[$(date +%H:%M:%S)] batch sweep → $OUTROOT"
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader
  local B rep BDIR REP_DIR PWR_CSV PWR_PID
  for B in 1 4 8 16; do
    BDIR="$OUTROOT/b$B"
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
        python3 "$PROFILING_DIR/profile_subphases_batch.py" \
        > "$REP_DIR/nsys_log.txt" 2>&1
      kill "$PWR_PID" 2>/dev/null || true
      wait "$PWR_PID" 2>/dev/null || true
      python3 "$PROFILING_DIR/align_power_nvtx.py" \
        "$REP_DIR/timeline.nsys-rep" "$PWR_CSV" "$REP_DIR/phase_energy.csv" \
        > "$REP_DIR/align_log.txt" 2>&1 || true
    done
    aggregate_phase_energy_reps "$BDIR" "$BDIR/phase_energy_agg.csv" \
      > "$BDIR/aggregate_log.txt" 2>&1 || true
  done
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

realism_check_txt() {
  export _RC_OLD="${INTERNVL_RESULTS_SMALL:-/home/cc/results}"
  export _RC_NEW="${INTERNVL_RESULTS_REALISTIC:-/home/cc/results2}"
  python3 <<'PY'
import csv, os

def load_energy(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["phase"]] = r
    return out

old = load_energy(os.path.join(os.environ["_RC_OLD"], "freq_default", "phase_energy.csv"))
new = load_energy(os.path.join(os.environ["_RC_NEW"], "freq_default", "phase_energy_agg.csv"))
phases = [
    "CLOSED_LOOP_INFERENCE",
    "InternVL_LLM_Prefill",
    "InternVL_LLM_Decode",
    "InternVL_Vision_Encoder",
    "InternVL_MLP_Connector",
]
lines = [
    "=" * 90,
    f"{'REALISM CHECK: small vs realistic (default clock)':^90}",
    "=" * 90,
    "",
    "Small: 70-token prompt, 1 tile (eer.jpg @ 448×448)",
    "Realistic: long prompt, 10 tiles (eer.jpg @ 1344×1344)",
    "",
    f"{'Phase':<28} {'old ms':>10} {'new ms':>14} {'old J':>8} {'new J':>14} {'dur ratio':>11}",
    "-" * 90,
]
for p in phases:
    o, n = old.get(p), new.get(p)
    if not o or not n:
        lines.append(f"{p:<28} (missing)")
        continue
    od, oe = float(o["total_duration_ms"]), float(o["total_energy_j"])
    nd, ne = float(n["duration_mean_ms"]), float(n["energy_mean_j"])
    nd_std = float(n.get("duration_std_ms", 0))
    ne_std = float(n.get("energy_std_j", 0))
    ratio = nd / od if od else 0
    lines.append(
        f"{p:<28} {od:>10.2f} {nd:>9.2f}±{nd_std:<3.1f} {oe:>8.2f} "
        f"{ne:>9.2f}±{ne_std:<3.1f} {ratio:>10.2f}x"
    )
lines.extend(["", "Interpretation: ratios ≈1 → GPU absorbed more work without linear time growth;", ""])
txt = "\n".join(lines)
new_dir = os.environ["_RC_NEW"]
os.makedirs(new_dir, exist_ok=True)
out_path = os.path.join(new_dir, "realism_check.txt")
with open(out_path, "w") as f:
    f.write(txt + "\n")
print(txt)
print(f"\nwrote {out_path}")
PY
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

cmd_overnight_small() {
  local RESULTS="$RESULTS_SMALL"
  local ZIP_OUT="$ZIP_SMALL"
  trap 'echo "[$(date +%H:%M:%S)] cleanup: nvidia-smi -rgc"; cleanup_clocks' EXIT

  echo "[$(date +%H:%M:%S)] overnight small RESULTS=$RESULTS"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.gr,clocks.max.mem --format=csv,noheader
  nvidia-smi -pm 1 >/dev/null 2>&1 || true

  one_freq_small "default" "$RESULTS/freq_default" || true
  local f
  for f in "${FREQS[@]}"; do
    echo "[$(date +%H:%M:%S)] lock ${f} MHz"
    nvidia-smi -lgc "${f},${f}" >/dev/null 2>&1 || { echo "skip $f"; continue; }
    sleep 2
    one_freq_small "$f" "$RESULTS/freq_${f}MHz" || true
    cleanup_clocks
    sleep 2
  done

  combine_freq_energy_csv "$RESULTS" "$RESULTS/combined_energy.csv"
  if [[ "${INTERNVL_SKIP_ZIP:-}" != "1" ]]; then
    zip_results "$RESULTS" "$ZIP_OUT" "README_DOWNLOAD.txt"
  fi
  echo "[$(date +%H:%M:%S)] done (small)"
}

cmd_overnight_realistic() {
  local RESULTS="$RESULTS_REAL"
  local ZIP_OUT="$ZIP_REAL"
  mkdir -p "$RESULTS"
  trap 'echo "[$(date +%H:%M:%S)] cleanup: nvidia-smi -rgc"; cleanup_clocks' EXIT

  echo "[$(date +%H:%M:%S)] overnight realistic RESULTS=$RESULTS"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  nvidia-smi -pm 1 >/dev/null 2>&1 || true

  one_freq_v2 "default" "$RESULTS/freq_default" "$PROFILE_REALISTIC" || true
  local f
  for f in "${FREQS[@]}"; do
    echo "[$(date +%H:%M:%S)] lock ${f} MHz"
    nvidia-smi -lgc "${f},${f}" >/dev/null 2>&1 || { echo "skip $f"; continue; }
    sleep 2
    one_freq_v2 "$f" "$RESULTS/freq_${f}MHz" "$PROFILE_REALISTIC" || true
    cleanup_clocks
    sleep 2
  done

  cmd_batch || true
  combine_freq_energy_csv "$RESULTS" "$RESULTS/combined_energy.csv"
  if [[ "${INTERNVL_SKIP_REALISM:-}" != "1" ]]; then
    realism_check_txt || true
  fi

  if [[ "${INTERNVL_SKIP_ZIP:-}" != "1" ]]; then
    zip_results "$RESULTS" "$ZIP_OUT" "README_DOWNLOAD2.txt"
  fi
  echo "[$(date +%H:%M:%S)] done (realistic)"
}

usage() {
  cat <<'EOF'
Usage: ./run.sh [command]
  realistic    full v2 pipeline (default): freq sweep + batch + combine + realism + zip
  small        v1 small-workload overnight + zip
  batch        batch sweep only
  ncu-sweep    four NCU phases only (small profile)
  one-small <label> <outdir>
  one <label> <outdir> [profile.py]
  help
Env: INTERNVL_RESULTS_SMALL, INTERNVL_RESULTS_REALISTIC, INTERNVL_SKIP_ZIP=1,
     INTERNVL_SKIP_REALISM=1 (skip realism_check.txt), ...
EOF
}

main() {
  local cmd="${1:-realistic}"
  case "$cmd" in
    help|-h|--help) usage; exit 0 ;;
    realistic)
      cmd_overnight_realistic
      ;;
    small)
      cmd_overnight_small
      ;;
    batch)
      cmd_batch
      ;;
    ncu-sweep)
      cmd_ncu_sweep
      ;;
    one-small)
      shift
      [[ $# -ge 2 ]] || { echo "usage: $0 one-small <label> <outdir>"; exit 1; }
      one_freq_small "$1" "$2"
      ;;
    one)
      shift
      [[ $# -ge 2 ]] || { echo "usage: $0 one <label> <outdir> [profile.py]"; exit 1; }
      one_freq_v2 "$1" "$2" "${3:-$PROFILE_REALISTIC}"
      ;;
    *)
      echo "unknown command: $cmd" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"

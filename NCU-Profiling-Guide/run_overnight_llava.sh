#!/usr/bin/env bash
# LLaVA-1.5-7B overnight: realistic-workload freq sweep + batch sweep + zip.
# Output: /home/cc/results_llava/  (cross-model joinable with results2/ from the InternVL run)
set -u
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
RESULTS=/home/cc/results_llava
mkdir -p "$RESULTS"

PROFILE_SCRIPT="$SCRIPT_DIR/profile_llava_realistic.py"
BATCH_SCRIPT="$SCRIPT_DIR/profile_llava_batch.py"
MODEL_PREFIX="LLaVA"
FREQS=(210 510 810 1110 1410)

# Phase names for align_power_nvtx.py — both InternVL and LLaVA use the
# same `CLOSED_LOOP_INFERENCE` outer wrapper; the inner four take the model prefix.
export NVTX_PHASES="CLOSED_LOOP_INFERENCE,${MODEL_PREFIX}_LLM_Prefill,${MODEL_PREFIX}_LLM_Decode,${MODEL_PREFIX}_Vision_Encoder,${MODEL_PREFIX}_MLP_Connector"

cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: resetting GPU clocks"
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[$(date +%H:%M:%S)] === LLaVA OVERNIGHT START (realistic workload + batch sweep) ==="
echo "host: $(hostname)  user: $(whoami)"
echo "model_prefix: $MODEL_PREFIX  outroot: $RESULTS"
echo "NVTX_PHASES=$NVTX_PHASES"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true

# ---- P0 baseline (default clock) ----
echo "[$(date +%H:%M:%S)] === P0: baseline at default clock ==="
bash "$SCRIPT_DIR/run_one_freq_v2.sh" "default" "$RESULTS/freq_default" "$PROFILE_SCRIPT" "$MODEL_PREFIX" || true

# ---- P1 frequency sweep ----
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
  bash "$SCRIPT_DIR/run_one_freq_v2.sh" "$f" "$RESULTS/freq_${f}MHz" "$PROFILE_SCRIPT" "$MODEL_PREFIX" || true
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true
  sleep 2
done

# ---- Batch sweep (default clock) ----
echo
echo "[$(date +%H:%M:%S)] === BATCH SWEEP ==="
bash "$SCRIPT_DIR/run_batch_sweep.sh" "$RESULTS/batch_sweep" "$BATCH_SCRIPT" || true

# ---- Aggregate combined energy CSV across freqs ----
echo
echo "[$(date +%H:%M:%S)] === aggregating combined_energy.csv ==="
RESULTS="$RESULTS" python3 - <<'PY'
import csv, glob, os, re
results = os.environ["RESULTS"]
out = os.path.join(results, "combined_energy.csv")
rows = []
for d in sorted(glob.glob(os.path.join(results, "freq_*"))):
    fcsv = os.path.join(d, "phase_energy_agg.csv")
    if not os.path.exists(fcsv): continue
    label = os.path.basename(d).replace("freq_", "")
    m = re.match(r"(\d+)MHz", label)
    freq = int(m.group(1)) if m else -1
    with open(fcsv) as f:
        for r in csv.DictReader(f):
            rows.append({"freq_mhz": freq, **r})
if rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
PY

# ---- Zip ----
echo
echo "[$(date +%H:%M:%S)] === packaging zip ==="
cd /home/cc
ZIP=/home/cc/results_llava_overnight.zip
rm -f "$ZIP"
zip -r "$ZIP" results_llava/ -x "results_llava/*.sqlite" >/dev/null 2>&1
echo "zip: $(ls -lh $ZIP | awk '{print $5, $NF}')"

cat > /home/cc/README_DOWNLOAD_LLAVA.txt <<EOF
LLaVA-1.5-7B sweep finished at $(date).
On your laptop:
  scp cc@<chameleon-ip>:~/results_llava_overnight.zip ~/Downloads/

Contents:
  results_llava/freq_default/, freq_{210,510,810,1110,1410}MHz/   per-freq runs
    rep_1/ rep_2/ rep_3/             nsys+power per rep
      timeline.nsys-rep, power_samples.csv, phase_energy.csv
    phase_energy_agg.csv             mean+std across the 3 reps
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.ncu-rep
  results_llava/batch_sweep/b{1,4,8,16}/  batch sweep at default clock (no ncu)
    rep_1/ rep_2/ rep_3/, phase_energy_agg.csv
  results_llava/combined_energy.csv       all freqs joined
  results_llava/sweep_log.txt             full driver log
EOF

echo "[$(date +%H:%M:%S)] === LLaVA OVERNIGHT DONE ==="
nvidia-smi -rgc >/dev/null 2>&1 || true

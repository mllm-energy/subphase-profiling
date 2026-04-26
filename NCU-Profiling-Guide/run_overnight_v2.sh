#!/usr/bin/env bash
# Overnight v2: realistic workload freq sweep + batch sweep + zip.
set -u
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
RESULTS=/home/cc/results2
mkdir -p "$RESULTS"

PROFILE_SCRIPT="$SCRIPT_DIR/profile_subphases_realistic.py"
FREQS=(210 510 810 1110 1410)

cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: resetting GPU clocks"
  nvidia-smi -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[$(date +%H:%M:%S)] === OVERNIGHT V2 START (realistic workload + batch sweep) ==="
echo "host: $(hostname)  user: $(whoami)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi -pm 1 >/dev/null 2>&1 || true

# ---- P0 baseline (default clock) ----
echo "[$(date +%H:%M:%S)] === P0: baseline at default clock ==="
bash "$SCRIPT_DIR/run_one_freq_v2.sh" "default" "$RESULTS/freq_default" "$PROFILE_SCRIPT" || true

# ---- P1 frequency sweep ----
for f in "${FREQS[@]}"; do
  echo
  echo "[$(date +%H:%M:%S)] === P1: locking SM clock to ${f} MHz ==="
  if ! nvidia-smi -lgc "${f},${f}" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)]   ! lock to ${f}MHz failed, skipping"
    continue
  fi
  sleep 2
  actual=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)
  echo "[$(date +%H:%M:%S)]   locked. actual clocks.gr=${actual}MHz"
  bash "$SCRIPT_DIR/run_one_freq_v2.sh" "$f" "$RESULTS/freq_${f}MHz" "$PROFILE_SCRIPT" || true
  nvidia-smi -rgc >/dev/null 2>&1 || true
  sleep 2
done

# ---- Batch sweep (default clock) ----
echo
echo "[$(date +%H:%M:%S)] === BATCH SWEEP ==="
bash "$SCRIPT_DIR/run_batch_sweep.sh" || true

# ---- Aggregate combined energy CSV across freqs ----
echo
echo "[$(date +%H:%M:%S)] === aggregating combined_energy.csv ==="
python3 - <<'PY'
import csv, glob, os, re
out = "/home/cc/results2/combined_energy.csv"
rows = []
for d in sorted(glob.glob("/home/cc/results2/freq_*")):
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

# ---- Realism check (compare small vs realistic) ----
echo
echo "[$(date +%H:%M:%S)] === realism_check.txt ==="
python3 "$SCRIPT_DIR/realism_check.py" || true

# ---- Zip ----
echo
echo "[$(date +%H:%M:%S)] === packaging zip ==="
cd /home/cc
ZIP=/home/cc/results2_overnight.zip
rm -f "$ZIP"
zip -r "$ZIP" results2/ -x "results2/*.sqlite" >/dev/null 2>&1
echo "zip: $(ls -lh $ZIP | awk '{print $5, $NF}')"

cat > /home/cc/README_DOWNLOAD2.txt <<EOF
Realistic-workload sweep finished at $(date).
On your laptop:
  scp cc@<chameleon-ip>:~/results2_overnight.zip ~/Downloads/

Contents:
  results2/freq_default/, freq_{210,510,810,1110,1410}MHz/   per-freq runs
    rep_1/ rep_2/ rep_3/             nsys+power per rep
      timeline.nsys-rep, power_samples.csv, phase_energy.csv
    phase_energy_agg.csv             mean+std across the 3 reps
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.ncu-rep
  results2/batch_sweep/b{1,4,8,16}/  batch sweep at default clock (no ncu)
    rep_1/ rep_2/ rep_3/, phase_energy_agg.csv
  results2/combined_energy.csv       all freqs joined
  results2/realism_check.txt         small vs realistic comparison
  results2/sweep_log.txt             full driver log
EOF

echo "[$(date +%H:%M:%S)] === OVERNIGHT V2 DONE ==="
nvidia-smi -rgc >/dev/null 2>&1 || true

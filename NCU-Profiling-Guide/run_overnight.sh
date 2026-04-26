#!/usr/bin/env bash
# Overnight driver: P0 baseline + P1 frequency sweep + final zip.
# Run as:   sudo -E nohup ./run_overnight.sh > ~/results/sweep_log.txt 2>&1 &
set -u

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
RESULTS=/home/cc/results

FREQS=(210 510 810 1110 1410)

cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: resetting GPU clocks"
  nvidia-smi -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[$(date +%H:%M:%S)] === OVERNIGHT SWEEP START ==="
echo "host: $(hostname)  user: $(whoami)  pwd: $(pwd)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.gr,clocks.max.mem --format=csv,noheader

echo "Persistence mode on (improves clock-lock reliability)"
nvidia-smi -pm 1 >/dev/null 2>&1 || true

# ---- P0: baseline at default (unlocked) clock ----
echo "[$(date +%H:%M:%S)] === P0: baseline at default boost clock ==="
bash "$SCRIPT_DIR/run_one_freq.sh" "default" "$RESULTS/freq_default" || true

# ---- P1: locked-clock sweep ----
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
  bash "$SCRIPT_DIR/run_one_freq.sh" "$f" "$RESULTS/freq_${f}MHz" || true
  nvidia-smi -rgc >/dev/null 2>&1 || true
  sleep 2
done

# ---- Aggregate energy CSVs across frequencies ----
echo
echo "[$(date +%H:%M:%S)] === aggregating combined_energy.csv ==="
python3 - <<'PY'
import csv, glob, os, re
out = "/home/cc/results/combined_energy.csv"
rows = []
for d in sorted(glob.glob("/home/cc/results/freq_*")):
    fcsv = os.path.join(d, "phase_energy.csv")
    if not os.path.exists(fcsv): continue
    label = os.path.basename(d).replace("freq_", "")
    m = re.match(r"(\d+)MHz", label)
    freq = int(m.group(1)) if m else -1  # -1 = default/unlocked
    with open(fcsv) as f:
        for r in csv.DictReader(f):
            r2 = {"freq_mhz": freq, **r}
            rows.append(r2)
if rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
else:
    print("no phase_energy.csv files found")
PY

# ---- Zip everything ----
echo
echo "[$(date +%H:%M:%S)] === packaging zip ==="
cd /home/cc
TS=$(date +%Y%m%d_%H%M%S)
ZIP=/home/cc/results_overnight.zip
rm -f "$ZIP"
zip -r "$ZIP" results/ -x "results/*.sqlite" >/dev/null
echo "zip: $(ls -lh $ZIP | awk '{print $5, $NF}')"

cat > /home/cc/README_DOWNLOAD.txt <<EOF
Overnight sweep finished at $(date).
On your laptop, run:
  scp cc@<chameleon-ip>:~/results_overnight.zip ~/Downloads/

Contents (skipping .sqlite to keep zip small):
  results/freq_default/             baseline (unlocked clock)
  results/freq_{210,510,810,1110,1410}MHz/   per-frequency runs
    timeline.nsys-rep         (open in Nsight Systems)
    power_samples.csv         (50ms power/clock samples)
    phase_energy.csv          (joules per NVTX phase)
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.ncu-rep
    *_log.txt                 (per-stage logs)
  results/combined_energy.csv     joined latency-vs-energy frontier
  results/csv/                    per-phase utilization tables (existing)
  results/sweep_log.txt           full driver log
EOF

echo "[$(date +%H:%M:%S)] === OVERNIGHT SWEEP DONE ==="
nvidia-smi -rgc >/dev/null 2>&1 || true

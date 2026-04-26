#!/usr/bin/env bash
# Background nvidia-smi power/clock sampler.
# Usage: ./power_sample.sh <output.csv>
# Stop with: kill $(cat <output.csv>.pid)
set -u
OUT="${1:?need output csv path}"
PIDFILE="${OUT}.pid"
echo $$ > "$PIDFILE"
echo "timestamp_ns,power_w,clock_gr_mhz,clock_mem_mhz,temp_c,gpu_util_pct,mem_util_pct" > "$OUT"
nvidia-smi --query-gpu=power.draw,clocks.gr,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory \
           --format=csv,noheader,nounits -lms 50 2>/dev/null | \
while IFS= read -r line; do
  ts=$(date +%s%N)
  echo "${ts},${line}" >> "$OUT"
done

#!/usr/bin/env bash
# Background nvidia-smi power/clock sampler.
# Usage: ./power_sample.sh <output.csv>
# Stop with: kill <pid_of_this_script>     (children are killed too via trap)
set -u
OUT="${1:?need output csv path}"
PIDFILE="${OUT}.pid"
echo $$ > "$PIDFILE"

# Run nvidia-smi in its own process group + kill it on parent termination.
# Without this, killing only the bash wrapper leaves nvidia-smi orphaned and it
# keeps appending to OUT for the rest of the parent run -> contaminated samples.
NVSMI_PID=""
cleanup() {
  if [[ -n "$NVSMI_PID" ]]; then
    kill -- -"$NVSMI_PID" 2>/dev/null || kill "$NVSMI_PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  exit 0
}
trap cleanup INT TERM EXIT

echo "timestamp_ns,power_w,clock_gr_mhz,clock_mem_mhz,temp_c,gpu_util_pct,mem_util_pct" > "$OUT"

# Start nvidia-smi in background as a new process group leader (setsid),
# pipe its CSV output to a fifo we read in this script. That way we own the
# child's PID and can kill its whole pgid in cleanup().
FIFO="$(mktemp -u)"
mkfifo "$FIFO"

setsid nvidia-smi --query-gpu=power.draw,clocks.gr,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory \
       --format=csv,noheader,nounits -lms 50 > "$FIFO" 2>/dev/null &
NVSMI_PID=$!

while IFS= read -r line; do
  ts=$(date +%s%N)
  printf "%s,%s\n" "$ts" "$line" >> "$OUT"
done < "$FIFO"

rm -f "$FIFO"

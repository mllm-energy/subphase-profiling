#!/usr/bin/env python3
"""Integrate per-phase joules from a power CSV using NVTX windows from an nsys sqlite."""
import csv, sys, sqlite3, subprocess, os
from collections import defaultdict

if len(sys.argv) != 4:
    print("usage: align_power_nvtx.py <timeline.nsys-rep> <power.csv> <out.csv>", file=sys.stderr)
    sys.exit(1)
nsys_rep, power_csv, out_csv = sys.argv[1:4]

# Convert nsys-rep -> sqlite if needed
sqlite_path = nsys_rep.replace('.nsys-rep', '.sqlite')
if not os.path.exists(sqlite_path):
    subprocess.run(['nsys', 'export', '--type=sqlite', '-o', sqlite_path, nsys_rep], check=True)

# Read NVTX events from sqlite (start/end in ns since nsys profile start)
con = sqlite3.connect(sqlite_path)
cur = con.cursor()
# Profile-start anchor in wall-clock ns (TARGET_INFO_SESSION_START_TIME)
cur.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME")
row = cur.fetchone()
session_start_ns = row[0] if row else 0
if session_start_ns == 0:
    print("warning: no session start anchor; assuming nsys timestamps already wall-clock", file=sys.stderr)

CLOSED_LOOP = 'CLOSED_LOOP_INFERENCE'
INNER_PHASES = ['InternVL_LLM_Prefill', 'InternVL_LLM_Decode',
                'InternVL_Vision_Encoder', 'InternVL_MLP_Connector']
phases = [CLOSED_LOOP] + INNER_PHASES
nvtx_windows = defaultdict(list)

# Collect CLOSED_LOOP_INFERENCE windows first; inner phases are filtered to
# events nested inside one of these so warmup-call NVTX events do not pollute
# the per-phase totals.
cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE text=?", (CLOSED_LOOP,))
for s, e in cur.fetchall():
    nvtx_windows[CLOSED_LOOP].append((session_start_ns + s, session_start_ns + e))

closed_loop_windows = nvtx_windows[CLOSED_LOOP]


def _is_nested(s_ns, e_ns):
    for cs, ce in closed_loop_windows:
        if cs <= s_ns and e_ns <= ce:
            return True
    return False


for ph in INNER_PHASES:
    cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE text=?", (ph,))
    for s, e in cur.fetchall():
        ws, we = session_start_ns + s, session_start_ns + e
        # If no CLOSED_LOOP windows exist (e.g., script ran without the
        # outer push/pop), keep all events to avoid silently emptying output.
        if not closed_loop_windows or _is_nested(ws, we):
            nvtx_windows[ph].append((ws, we))

# Load power samples
samples = []  # (ts_ns, power_w)
with open(power_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            samples.append((int(row['timestamp_ns']), float(row['power_w'])))
        except (ValueError, KeyError):
            continue
samples.sort()
if not samples:
    print("error: no power samples", file=sys.stderr); sys.exit(2)

# Trapezoidal integration of power(t) over each NVTX window.
# Walks adjacent sample pairs; for each pair, clips the segment to [t0, t1]
# and integrates linearly-interpolated power over the clipped span.
def integrate(t0, t1):
    if t1 <= t0: return 0.0, 0
    energy_j = 0.0
    n = 0
    for i in range(len(samples) - 1):
        ts_a, p_a = samples[i]
        ts_b, p_b = samples[i + 1]
        if ts_b < t0: continue
        if ts_a > t1: break
        a = max(ts_a, t0)
        b = min(ts_b, t1)
        if b <= a: continue
        if ts_b == ts_a:
            avg_p = (p_a + p_b) / 2
        else:
            span = ts_b - ts_a
            p_at_a = p_a + (p_b - p_a) * (a - ts_a) / span
            p_at_b = p_a + (p_b - p_a) * (b - ts_a) / span
            avg_p = (p_at_a + p_at_b) / 2
        energy_j += avg_p * (b - a) / 1e9
        n += 1
    return energy_j, n

with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['phase', 'instances', 'total_duration_ms', 'total_energy_j',
                'avg_power_w', 'energy_per_instance_j', 'samples_in_phase'])
    for ph in phases:
        wins = nvtx_windows.get(ph, [])
        total_e = 0.0; total_dur_ns = 0; total_samples = 0
        for s, e in wins:
            ej, n = integrate(s, e)
            total_e += ej; total_dur_ns += (e - s); total_samples += n
        if not wins: continue
        dur_ms = total_dur_ns / 1e6
        avg_p = (total_e / (total_dur_ns / 1e9)) if total_dur_ns > 0 else 0.0
        w.writerow([ph, len(wins), round(dur_ms, 3),
                    round(total_e, 3), round(avg_p, 2),
                    round(total_e / len(wins), 4) if wins else 0,
                    total_samples])

print(f"wrote {out_csv}")

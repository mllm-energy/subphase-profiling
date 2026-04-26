#!/usr/bin/env python3
"""Aggregate phase_energy.csv files across N rep_*/ subdirs to mean+std."""
import csv, sys, os, glob
from statistics import mean, stdev

if len(sys.argv) != 3:
    print("usage: aggregate_reps.py <freq_dir> <out.csv>", file=sys.stderr); sys.exit(1)
freq_dir, out_csv = sys.argv[1:3]

per_phase = {}
n_reps = 0
for rep_dir in sorted(glob.glob(os.path.join(freq_dir, "rep_*"))):
    csv_path = os.path.join(rep_dir, "phase_energy.csv")
    if not os.path.exists(csv_path): continue
    n_reps += 1
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ph = row['phase']
            per_phase.setdefault(ph, {'energy_j': [], 'duration_ms': [], 'avg_power_w': [], 'instances': int(row['instances'])})
            per_phase[ph]['energy_j'].append(float(row['total_energy_j']))
            per_phase[ph]['duration_ms'].append(float(row['total_duration_ms']))
            per_phase[ph]['avg_power_w'].append(float(row['avg_power_w']))

with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['phase', 'n_reps', 'instances',
                'energy_mean_j', 'energy_std_j',
                'duration_mean_ms', 'duration_std_ms',
                'power_mean_w', 'power_std_w'])
    for ph, d in per_phase.items():
        es = d['energy_j']; ds = d['duration_ms']; ps = d['avg_power_w']
        w.writerow([ph, len(es), d['instances'],
                    round(mean(es), 4), round(stdev(es), 4) if len(es) > 1 else 0.0,
                    round(mean(ds), 4), round(stdev(ds), 4) if len(ds) > 1 else 0.0,
                    round(mean(ps), 3), round(stdev(ps), 3) if len(ps) > 1 else 0.0])

print(f"wrote {out_csv} from {n_reps} reps")

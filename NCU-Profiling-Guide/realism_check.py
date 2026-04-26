#!/usr/bin/env python3
"""Compare small vs realistic workload phase durations and energies."""
import csv, sys, os

OLD_DIR = "/home/cc/results"
NEW_DIR = "/home/cc/results2"

def load_energy(path):
    out = {}
    if not os.path.exists(path): return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['phase']] = r
    return out

old = load_energy(os.path.join(OLD_DIR, "freq_default", "phase_energy.csv"))

# New uses _agg with mean across reps
new = load_energy(os.path.join(NEW_DIR, "freq_default", "phase_energy_agg.csv"))

phases = ['CLOSED_LOOP_INFERENCE', 'InternVL_LLM_Prefill', 'InternVL_LLM_Decode',
          'InternVL_Vision_Encoder', 'InternVL_MLP_Connector']

lines = []
lines.append("=" * 90)
lines.append(f"{'REALISM CHECK: small workload vs realistic workload (default clock)':^90}")
lines.append("=" * 90)
lines.append("")
lines.append(f"Small workload   : 70-token prompt, 1 vision tile (eer.jpg @ 448x448)")
lines.append(f"Realistic        : 535-token prompt, 10 vision tiles (eer.jpg @ 1344x1344)")
lines.append("")
lines.append(f"{'Phase':<28} {'old dur ms':>10} {'new dur ms':>14} {'old J':>8} {'new J':>14} {'dur ratio':>11}")
lines.append("-" * 90)
for p in phases:
    o = old.get(p)
    n = new.get(p)
    if not o or not n:
        lines.append(f"{p:<28} (missing)")
        continue
    od = float(o['total_duration_ms']); oe = float(o['total_energy_j'])
    nd = float(n['duration_mean_ms']); ne = float(n['energy_mean_j'])
    nd_std = float(n.get('duration_std_ms', 0)); ne_std = float(n.get('energy_std_j', 0))
    ratio = nd / od if od else 0
    lines.append(
        f"{p:<28} {od:>10.2f} {nd:>9.2f}±{nd_std:<3.1f} {oe:>8.2f} {ne:>9.2f}±{ne_std:<3.1f} {ratio:>10.2f}x"
    )

lines.append("")
lines.append("Interpretation:")
lines.append("  - Ratios near 1.0 indicate the GPU absorbed the larger workload at b=1")
lines.append("    without proportional runtime growth — i.e., even more underutilized")
lines.append("    headroom than the small-workload sweep showed.")
lines.append("  - Larger ratios (>>1) would indicate the workload is now saturating the GPU")
lines.append("    differently (i.e., the original results don't generalize).")
lines.append("")

txt = "\n".join(lines)
out_path = os.path.join(NEW_DIR, "realism_check.txt")
with open(out_path, 'w') as f: f.write(txt + "\n")
print(txt)
print(f"\nwrote {out_path}")

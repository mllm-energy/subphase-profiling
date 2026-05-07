#!/usr/bin/env python3
"""Walk a batch-sweep results tree and emit consolidated CSVs for plotting.

Inputs (per batch dir under <results_root>):
  b<N>/
    rep_<i>/timeline_cuda_gpu_trace.csv      (nsys export, per-kernel timing)
    rep_<i>/phase_energy.csv                  (per-phase duration / energy / power)
    phase_energy_agg.csv                       (mean+std across reps)
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.csv  (NCU CSV export)
    {Prefill,Decode,Vision_Encoder,MLP_Connector}.ncu-rep (binary, ignored)

Outputs (at <results_root>):
  kernel_traces.csv      — per (workload, batch, subphase, kernel_name, kernel_id) row
                           with SM%, DRAM% from NCU. This is the time-vs-throughput
                           trace data the plotter needs.
  phase_durations.csv    — per (workload, batch, subphase) row with mean+std
                           duration/energy/power from phase_energy_agg.csv
  end_to_end.csv         — per (workload, batch) row with mean+std end-to-end
                           latency (CLOSED_LOOP_INFERENCE NVTX range)

Usage:
  python3 aggregate_results.py <results_root> <workload>
"""
import csv
import glob
import os
import re
import sys
from statistics import mean, stdev

SUBPHASES = ["Prefill", "Decode", "Vision_Encoder", "MLP_Connector"]
NVTX_FOR_PHASE = {
    "Prefill": "InternVL_LLM_Prefill",
    "Decode": "InternVL_LLM_Decode",
    "Vision_Encoder": "InternVL_Vision_Encoder",
    "MLP_Connector": "InternVL_MLP_Connector",
}


def _mean_std(vals):
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(stdev(vals))


def parse_ncu_csv(path):
    """Parse an NCU --csv --page details export.

    NCU's CSV format has a multi-line header (info section) before the data
    table; the data table itself has columns including 'ID', 'Kernel Name',
    and the requested metrics. Returns a list of dicts with normalized keys.
    """
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Find the first line that looks like a header (contains 'ID' and 'Kernel
    # Name' separated by commas). Some NCU versions emit multiple sections.
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        cols = [c.strip().strip('"') for c in line.split(",")]
        if "ID" in cols and any("Kernel" in c for c in cols):
            header_idx = i
            break
    if header_idx is None:
        return []

    rows = []
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        # Strip BOM and surrounding whitespace from keys/values
        clean = {(k.strip().strip('"') if k else k): (v.strip().strip('"') if v else v) for k, v in row.items()}
        rows.append(clean)
    return rows


def collect_kernel_traces(results_root, workload):
    """Parse NCU --csv --page details exports. Each launch produces ONE row
    per requested metric (long format), so we pivot to one row per launch
    with metrics as columns."""
    out = []
    batch_dirs = sorted(glob.glob(os.path.join(results_root, "b*")))
    for bdir in batch_dirs:
        m = re.match(r"b(\d+)$", os.path.basename(bdir))
        if not m:
            continue
        batch = int(m.group(1))
        for sub in SUBPHASES:
            csv_path = os.path.join(bdir, f"{sub}.csv")
            rows = parse_ncu_csv(csv_path)
            # (id, kernel_name) -> {"sm_throughput_pct": .., "dram_throughput_pct": ..}
            by_kernel = {}
            for r in rows:
                kid = r.get("ID", "")
                kname = r.get("Kernel Name", "")
                metric_name = r.get("Metric Name", "")
                metric_value = r.get("Metric Value", "")
                if not kname or not metric_name:
                    continue
                key = (kid, kname)
                slot = by_kernel.setdefault(
                    key,
                    {"block_size": r.get("Block Size", ""),
                     "grid_size": r.get("Grid Size", ""),
                     "sm_throughput_pct": None,
                     "dram_throughput_pct": None},
                )
                try:
                    val = float(str(metric_value).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue
                ml = metric_name.lower()
                if "sm__throughput" in ml and "pct_of_peak" in ml:
                    slot["sm_throughput_pct"] = val
                elif "dram__throughput" in ml and "pct_of_peak" in ml:
                    slot["dram_throughput_pct"] = val
            for (kid, kname), d in by_kernel.items():
                out.append({
                    "workload": workload,
                    "batch_size": batch,
                    "subphase": sub,
                    "kernel_id": kid,
                    "kernel_name": kname,
                    "block_size": d["block_size"],
                    "grid_size": d["grid_size"],
                    "sm_throughput_pct": "" if d["sm_throughput_pct"] is None else d["sm_throughput_pct"],
                    "dram_throughput_pct": "" if d["dram_throughput_pct"] is None else d["dram_throughput_pct"],
                })
    return out


def collect_phase_durations(results_root, workload):
    """Per-phase mean+std duration/energy/power from phase_energy_agg.csv at each batch."""
    out = []
    for bdir in sorted(glob.glob(os.path.join(results_root, "b*"))):
        m = re.match(r"b(\d+)$", os.path.basename(bdir))
        if not m:
            continue
        batch = int(m.group(1))
        agg = os.path.join(bdir, "phase_energy_agg.csv")
        if not os.path.exists(agg):
            continue
        with open(agg) as f:
            for r in csv.DictReader(f):
                out.append({
                    "workload": workload,
                    "batch_size": batch,
                    "phase": r["phase"],
                    "n_reps": r.get("n_reps", ""),
                    "duration_mean_ms": r.get("duration_mean_ms", ""),
                    "duration_std_ms": r.get("duration_std_ms", ""),
                    "energy_mean_j": r.get("energy_mean_j", ""),
                    "energy_std_j": r.get("energy_std_j", ""),
                    "power_mean_w": r.get("power_mean_w", ""),
                    "power_std_w": r.get("power_std_w", ""),
                })
    return out


def collect_end_to_end(results_root, workload):
    """End-to-end latency = CLOSED_LOOP_INFERENCE NVTX range duration, averaged across reps.
    Also reports the per-rep sum-of-inner-phases and the overhead gap (e2e - phases),
    which represents host-side / orchestration time outside the NVTX brackets.
    """
    inner_phases = {"InternVL_LLM_Prefill", "InternVL_LLM_Decode",
                    "InternVL_Vision_Encoder", "InternVL_MLP_Connector"}
    out = []
    for bdir in sorted(glob.glob(os.path.join(results_root, "b*"))):
        m = re.match(r"b(\d+)$", os.path.basename(bdir))
        if not m:
            continue
        batch = int(m.group(1))
        e2e_durations = []
        e2e_energies = []
        phase_sums = []
        overheads = []
        for rep_dir in sorted(glob.glob(os.path.join(bdir, "rep_*"))):
            pe = os.path.join(rep_dir, "phase_energy.csv")
            if not os.path.exists(pe):
                continue
            e2e_dur = None
            e2e_e = None
            inner_sum = 0.0
            with open(pe) as f:
                for r in csv.DictReader(f):
                    ph = r.get("phase", "")
                    try:
                        d = float(r["total_duration_ms"])
                    except (KeyError, ValueError):
                        d = 0.0
                    if ph == "CLOSED_LOOP_INFERENCE":
                        e2e_dur = d
                        try:
                            e2e_e = float(r["total_energy_j"])
                        except (KeyError, ValueError):
                            e2e_e = 0.0
                    elif ph in inner_phases:
                        inner_sum += d
            if e2e_dur is not None:
                e2e_durations.append(e2e_dur)
                e2e_energies.append(e2e_e if e2e_e is not None else 0.0)
                phase_sums.append(inner_sum)
                overheads.append(e2e_dur - inner_sum)
        dm, ds = _mean_std(e2e_durations)
        em, es = _mean_std(e2e_energies)
        psm, pss = _mean_std(phase_sums)
        ovm, ovs = _mean_std(overheads)
        out.append({
            "workload": workload,
            "batch_size": batch,
            "n_reps": len(e2e_durations),
            "duration_mean_ms": round(dm, 4),
            "duration_std_ms": round(ds, 4),
            "phase_sum_mean_ms": round(psm, 4),
            "phase_sum_std_ms": round(pss, 4),
            "overhead_mean_ms": round(ovm, 4),
            "overhead_std_ms": round(ovs, 4),
            "energy_mean_j": round(em, 4),
            "energy_std_j": round(es, 4),
        })
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    if len(sys.argv) != 3:
        print("usage: aggregate_results.py <results_root> <workload>", file=sys.stderr)
        sys.exit(2)
    root, workload = sys.argv[1], sys.argv[2]
    if not os.path.isdir(root):
        print(f"results dir not found: {root}", file=sys.stderr)
        sys.exit(1)
    print(f"aggregating {root} (workload={workload})")

    kt = collect_kernel_traces(root, workload)
    write_csv(
        os.path.join(root, "kernel_traces.csv"),
        kt,
        ["workload", "batch_size", "subphase", "kernel_id", "kernel_name",
         "block_size", "grid_size", "sm_throughput_pct", "dram_throughput_pct"],
    )
    pd = collect_phase_durations(root, workload)
    write_csv(
        os.path.join(root, "phase_durations.csv"),
        pd,
        ["workload", "batch_size", "phase", "n_reps",
         "duration_mean_ms", "duration_std_ms",
         "energy_mean_j", "energy_std_j",
         "power_mean_w", "power_std_w"],
    )
    e2e = collect_end_to_end(root, workload)
    write_csv(
        os.path.join(root, "end_to_end.csv"),
        e2e,
        ["workload", "batch_size", "n_reps",
         "duration_mean_ms", "duration_std_ms",
         "phase_sum_mean_ms", "phase_sum_std_ms",
         "overhead_mean_ms", "overhead_std_ms",
         "energy_mean_j", "energy_std_j"],
    )
    print("done")


if __name__ == "__main__":
    main()

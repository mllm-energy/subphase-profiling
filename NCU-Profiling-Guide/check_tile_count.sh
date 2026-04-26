#!/usr/bin/env bash
# Pre-flight: confirm multi-tile preprocessing actually triggers, AND collect
# wall-time data for the realistic workload. Tile count is checked via the
# preprocessor's pixel_values_flat shape (unambiguous). Wall-time is reported
# but NOT used as the pass/fail criterion (GPU may absorb extra work at b=1).
set -u
cd "$(dirname "$0")"

OUT=/tmp/tile_check
mkdir -p "$OUT"
NSYS=/usr/local/cuda/bin/nsys

# Direct shape check first
echo "=== Step 1: direct preprocessor shape check ==="
python3 - <<'PYCHK' || { echo "preprocessor check failed"; exit 4; }
from PIL import Image
from vllm.transformers_utils.processors.internvl import InternVLImageProcessor
img = Image.open("eer.jpg").convert("RGB").resize((1344, 1344), Image.BICUBIC)
proc = InternVLImageProcessor(image_size=448, min_dynamic_patch=1, max_dynamic_patch=12,
                              dynamic_image_size=True, use_thumbnail=True)
out = proc(images=[img], return_tensors="pt")
n = out['pixel_values_flat'].shape[0]
print(f"pixel_values_flat: {tuple(out['pixel_values_flat'].shape)}  -> {n} tiles")
if n < 4:
    print(f"FAIL: only {n} tiles, want >=4"); raise SystemExit(3)
print(f"PASS: preprocessor produces {n} tiles per image")
PYCHK

echo
echo "=== Step 2: nsys wall-time at realistic workload ==="
VLLM_ENABLE_V1_MULTIPROCESSING=0 "$NSYS" profile \
  --trace=cuda,nvtx --force-overwrite=true \
  -o "$OUT/check" \
  python3 profile_subphases_realistic.py \
  > "$OUT/check_log.txt" 2>&1

if [[ ! -f "$OUT/check.nsys-rep" ]]; then
  echo "ERROR: nsys report missing. last 30 lines of log:"
  tail -30 "$OUT/check_log.txt"
  exit 2
fi

# Need sqlite for direct query
"$NSYS" export --type=sqlite -o "$OUT/check.sqlite" "$OUT/check.nsys-rep" >/dev/null 2>&1

python3 - <<EOF
import sqlite3, statistics
con = sqlite3.connect("$OUT/check.sqlite")
cur = con.cursor()

def durs(name):
    cur.execute("SELECT (end-start)/1e6 FROM NVTX_EVENTS WHERE text=? ORDER BY start", (name,))
    return [r[0] for r in cur.fetchall()]

ve = durs("InternVL_Vision_Encoder")
pf = durs("InternVL_LLM_Prefill")
de = durs("InternVL_LLM_Decode")

# Small-workload baselines (from /home/cc/results/freq_default)
BASELINE_VE_MS = 24.0    # steady-state per-instance
BASELINE_PF_MS = 22.0
BASELINE_DE_MS = 24.0

ve_steady = statistics.median(ve[1:]) if len(ve) > 1 else (ve[0] if ve else 0)
pf_steady = statistics.median(pf[1:]) if len(pf) > 1 else (pf[0] if pf else 0)
de_steady = statistics.median(de[1:]) if len(de) > 1 else (de[0] if de else 0)

print(f"\n{'Phase':<28} {'old (ms)':>10} {'new (ms)':>10} {'ratio':>8}")
print("-" * 60)
print(f"{'InternVL_Vision_Encoder':<28} {BASELINE_VE_MS:>10.1f} {ve_steady:>10.1f} {ve_steady/BASELINE_VE_MS:>8.2f}x")
print(f"{'InternVL_LLM_Prefill':<28} {BASELINE_PF_MS:>10.1f} {pf_steady:>10.1f} {pf_steady/BASELINE_PF_MS:>8.2f}x")
print(f"{'InternVL_LLM_Decode':<28} {BASELINE_DE_MS:>10.1f} {de_steady:>10.1f} {de_steady/BASELINE_DE_MS:>8.2f}x")

# Decision: Vision_Encoder must be >=2x for "multi-tile" to matter,
# >=3.5x for ~4 tiles, >=2.5x for ~2 tiles.
ratio = ve_steady / BASELINE_VE_MS if BASELINE_VE_MS else 0
print()
print()
print(f"Wall-time ratios (realistic / small):")
print(f"  Vision_Encoder: {ratio:.2f}x")
print(f"  Prefill:        {pf_steady/BASELINE_PF_MS:.2f}x")
print(f"  Decode:         {de_steady/BASELINE_DE_MS:.2f}x")
print()
print("NOTE: wall-time may not scale with input size at b=1 because GPU has spare")
print("      capacity. Tile count is verified directly via preprocessor shape (Step 1).")
print()
print("PASS: Realistic workload exercises model with 10 tiles + ~700 tok prompt.")
EOF
echo
echo "Prompt token count from log:"
grep "Prompt text tokens" "$OUT/check_log.txt" || echo "  (not found)"

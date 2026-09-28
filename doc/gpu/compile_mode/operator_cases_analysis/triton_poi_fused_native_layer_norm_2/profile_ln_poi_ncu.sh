#!/usr/bin/env bash
# Nsight Compute L2 contrast for triton_poi_fused_native_layer_norm_2.
# Requires an ncu build that lists this GPU chip (GB206 / RTX 5060 Ti).
# Current host ncu 2025.1.1 does NOT — see ablation_timing_report.md §7.
set -euo pipefail

cd "$(dirname "$0")"
source /venv/main/bin/activate
mkdir -p ncu

NCU_BIN="${NCU_BIN:-ncu}"
SIZES=(${SIZES:-24 32 64})

for mb in "${SIZES[@]}"; do
  echo "===== ncu TARGET_IO_MB=$mb ====="
  "$NCU_BIN" --force-overwrite \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name regex:triton_poi_fused_native_layer_norm_2 \
    --launch-skip 5 --launch-count 3 \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section SpeedOfLight \
    --export "ncu/ln_poi_io${mb}" \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=8 CUPTI_RANGE=0 TARGET_IO_MB="$mb" \
    python profile_ln_poi.py
done

echo "=== L2 / DRAM metric peek ==="
for mb in "${SIZES[@]}"; do
  echo "--- io${mb} ---"
  "$NCU_BIN" --import "ncu/ln_poi_io${mb}.ncu-rep" --page raw 2>/dev/null \
    | grep -iE 'lts__t_sector_hit|dram__bytes|l2 hit|dram__' || true
done
echo DONE

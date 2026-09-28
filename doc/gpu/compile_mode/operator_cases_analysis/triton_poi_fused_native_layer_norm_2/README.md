# `triton_poi_fused_native_layer_norm_2` — operator case

Standalone Triton recreation of the Inductor GRAPH5 ViT L0 LayerNorm **pointwise
epilogue** (`model__4_inference_4.4/output_code.py`), with matching:

| Piece | Value |
|-------|--------|
| Shapes | `x/out [1,1024,768] bf16`, `mean/var [1,1024,1] f32`, `γ/β [768] bf16` |
| Layout | feature-major load `y + Y·x`; contiguous store `x + X·y` |
| Launch | Grid2D `(ceil(X/XBLOCK), ceil(Y/YBLOCK))` = **(12, 16)** @ `XBLOCK=64`, `YBLOCK=64`, `num_warps=16` |
| Math | `(x−mean)·rsqrt(var/X+ε)·γ+β` (var = sum of squared diffs) |

## Run unit test

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/triton_poi_fused_native_layer_norm_2
source /venv/main/bin/activate
SMOKE_DEVICE=cuda python test_triton_poi_fused_native_layer_norm_2.py
```

## Nsight size sweep (algo IO 8→128 MB)

```bash
export PATH=/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
mkdir -p nsys
# baseline
nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-event-trace=false \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none \
  --output=nsys/ln_poi_baseline \
  env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 \
  python profile_ln_poi.py
# sized (+ L2 zoom 29–35)
for mb in 8 16 24 29 30 31 32 33 34 35 48 64 96 128; do
  nsys profile --force-overwrite=true \
    --trace=cuda,nvtx,osrt --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output=nsys/ln_poi_io${mb} \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 TARGET_IO_MB=$mb \
    python profile_ln_poi.py
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    --output=nsys/ln_poi_io${mb} nsys/ln_poi_io${mb}.nsys-rep
done
```

Report: [`ablation_timing_report.md`](ablation_timing_report.md).

Unit test writes `ln_poi_baseline_report.json`.

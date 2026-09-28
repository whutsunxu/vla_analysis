# Nsight profiles — Pre upsample ablations

GPU: RTX 5060 Ti · Nsight Systems **2025.1.3** · warmup 5 + **10** measured iters  
Harness: `../profile_prepare_images_upsample.py` (no correctness checks)  
Capture: `--capture-range=cudaProfilerApi` (steady window only)

| File | Variant | Ops |
|------|---------|-----|
| `pre_upsample_full.nsys-rep` | `full` | interpolate 256→512 + `*2-1` + cam/patch masks |
| `pre_upsample_no_scale.nsys-rep` | `no_scale` | interpolate only + masks (drop `*2-1`) |
| `pre_upsample_no_upsample.nsys-rep` | `no_upsample` | `*2-1` on `[1,3,256,256]` + masks (drop interpolate) |

Companion `*_cuda_gpu_kern_sum.csv` next to each `.nsys-rep`.

### Host wall (sync’d, 10 iters)

| Variant | mean | median | min | max |
|---------|-----:|-------:|----:|----:|
| full | 0.274 ms | 0.254 ms | 0.241 | 0.495 |
| no_scale | 0.183 ms | 0.161 ms | 0.157 | 0.351 |
| no_upsample | 0.265 ms | 0.244 ms | 0.234 | 0.377 |

### GPU kernel avg (per launch, from kern_sum; ×10 iters)

| Variant | upsample | mul | add | fill_bool (×2 masks) |
|---------|---------:|----:|----:|---------------------:|
| full | ~34.5 µs | ~4.9 µs | ~5.0 µs | ~1.2 µs |
| no_scale | ~34.5 µs | — | — | ~1.2 µs |
| no_upsample | — | ~2.0 µs | ~2.0 µs | ~1.2 µs |

Re-run:

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/pre-process-upsample
source /venv/main/bin/activate
export PATH=/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
for v in full no_scale no_upsample; do
  nsys profile --force-overwrite=true \
    --trace=cuda,nvtx,osrt --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output=nsys/pre_upsample_${v} \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 PROFILE_VARIANT=$v \
    python profile_prepare_images_upsample.py
done
```

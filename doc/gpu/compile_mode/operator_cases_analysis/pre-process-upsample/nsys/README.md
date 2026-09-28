# Nsight profiles — Pre upsample ablations

GPU: RTX 5060 Ti · Nsight Systems **2025.1.3** · warmup 5 + **10** measured iters  
Harness: `../profile_prepare_images_upsample.py` (no correctness checks)  
Capture: `--capture-range=cudaProfilerApi` (steady window only)

| File | Variant | Ops |
|------|---------|-----|
| `pre_upsample_full.nsys-rep` | `full` | interpolate 256→512 + `*2-1` + masks |
| `pre_upsample_no_scale.nsys-rep` | `no_scale` | interpolate only + masks |
| `pre_upsample_no_upsample.nsys-rep` | `no_upsample` | `*2-1` on **native 512²** + masks (no interpolate) |

Companion `*_cuda_gpu_kern_sum.csv` next to each `.nsys-rep`.  
Full write-up: [`../ablation_timing_report.md`](../ablation_timing_report.md).

### Host wall (sync’d, 10 iters)

| Variant | mean | median | min | max |
|---------|-----:|-------:|----:|----:|
| full | 0.274 ms | 0.254 ms | 0.241 | 0.495 |
| no_scale | 0.183 ms | 0.161 ms | 0.157 | 0.351 |
| no_upsample @512² | 0.338 ms | 0.310 ms | 0.308 | 0.555 |

### GPU kernel avg (per launch; ×10 iters)

| Variant | upsample | mul | add | fill_bool |
|---------|---------:|----:|----:|----------:|
| full | 34.49 µs | 4.94 µs | 5.03 µs | 1.16 µs |
| no_scale | 34.48 µs | — | — | 1.15 µs |
| no_upsample @512² | — | 4.93 µs | 4.92 µs | 1.16 µs |

### Buffer-size sweep (`no_upsample`, masks off)

| File | Target algo IO | Shape |
|------|---------------:|------|
| `pre_upsample_no_upsample_io12.nsys-rep` | 12 MB | `[1,3,707,707]` |
| `pre_upsample_no_upsample_io24.nsys-rep` | 24 MB | `[1,3,1000,1000]` |
| `pre_upsample_no_upsample_io30.nsys-rep` | 30 MB | `[1,3,1118,1118]` |
| `pre_upsample_no_upsample_io32.nsys-rep` | 32 MB | `[1,3,1155,1155]` |
| `pre_upsample_no_upsample_io36.nsys-rep` | 36 MB | `[1,3,1225,1225]` |

See [`../ablation_timing_report.md`](../ablation_timing_report.md) §6 — `mul` BD util falls from ~300% → **98.5%** as algo IO crosses L2 (32 MB).

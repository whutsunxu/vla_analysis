# Pre · prepare_images upsample — ablation timing report

**Date:** 2026-09-28 (rev: `no_upsample` @ **512²** + roofline columns)  
**GPU:** NVIDIA GeForce RTX 5060 Ti (DRAM peak **448 GB/s**, FP32 CUDA peak **23.7 TFLOP/s** — same §0.1 peaks as eager / Inductor lists)  
**Nsight:** Systems 2025.1.3 · CUPTI range = measured window only  
**Harness:** `profile_prepare_images_upsample.py`  
**Protocol:** warmup **5** (outside CUPTI) + measured **10** sync’d iters (inside `cudaProfilerApi`)

Artifacts: `nsys/pre_upsample_{full,no_scale,no_upsample}.nsys-rep` + `*_cuda_gpu_kern_sum.csv`.

Roofline / IO rules match `SmolVLA_op_list_inductor_nsight.md` §Pre and eager `smolVLA_kerne_list_gpu_backend.md` §0.3 (bilinear = **7** Basic/out; elementwise = **1** FLOP/out; IO = Σ bytes(inputs)+Σ bytes(output); BD util ⚠ when algo-IO ≫ DRAM).

---

## 0. Experiment matrix

| Variant | Ops under test | Input → output |
|---------|----------------|----------------|
| **full** | `F.interpolate` 256→512 + `×2−1` + cam/patch masks | `[1,3,256,256]→[1,3,512,512]` |
| **no_scale** | interpolate + masks only (drop `×2−1`) | `[1,3,256,256]→[1,3,512,512]` |
| **no_upsample** | `×2−1` + masks only (drop interpolate); **input already 512²** | `[1,3,512,512]→[1,3,512,512]` |

Masks always: `cam_mask [1] bool` + `patch_mask [1,32,32] bool` → **2×** `FillFunctor<bool>` per iter. Fill row uses the Pre-table shape `[1,32,32] bool` (patch); cam fill shares the same kernel family (launch-floor dominated).

---

## 1. Average time across each 10-iter batch

### 1.1 Host wall (CUDA synchronize around each iter)

| Iter | full | no_scale | no_upsample (512²) |
|---:|---:|---:|---:|
| 0 | 0.4947 | 0.3511 | 0.5547 |
| 1 | 0.2543 | 0.1840 | 0.3320 |
| 2 | 0.2597 | 0.1609 | 0.3097 |
| 3 | 0.2441 | 0.1573 | 0.3095 |
| 4 | 0.2420 | 0.1576 | 0.3086 |
| 5 | 0.2533 | 0.1573 | 0.3089 |
| 6 | 0.2581 | 0.1764 | 0.3281 |
| 7 | 0.2407 | 0.1574 | 0.3079 |
| 8 | 0.2407 | 0.1723 | 0.3077 |
| 9 | 0.2564 | 0.1572 | 0.3164 |
| **mean (n=10)** | **0.2744** | **0.1832** | **0.3384** |
| median | 0.2543 | 0.1609 | 0.3097 |
| min | 0.2407 | 0.1572 | 0.3077 |
| max | 0.4947 | 0.3511 | 0.5547 |
| **mean excl. iter0** | **0.2500** | **0.1645** | **0.3143** |

Host wall ≫ GPU busy (below): dominated by launch / driver / sync overhead for these tiny kernels.

### 1.2 GPU busy time per iter (CUPTI Σ kernel duration / 10)

| Variant | GPU busy / iter | Breakdown |
|---------|----------------:|-----------|
| **full** | **46.773 µs** | upsample + mul + add + 2×fill |
| **no_scale** | **36.786 µs** | upsample + 2×fill |
| **no_upsample** | **12.154 µs** | mul + add + 2×fill @ **512²** |

---

## 2. Per-kernel roofline tables (same columns as Inductor §Pre)

**GPU time per launch** = `cuda_gpu_kern_sum` Avg over the 10-iter CUPTI window.  
**IO / GFLOPs / AI / Theoretical bottleneck** from shapes + heuristics (bilinear = 7 Basic/out).  
**BD / GFLOPs/sec / util** from IO÷time and GFLOPs÷time vs 448 GB/s and 23.7 TFLOP/s.

### 2.1 Variant `full` — interpolate + `×2−1` + masks

| Order | kernel / op | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning |
|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| 1 | `upsample_bilinear2d` | `[1,3,256,256] float32` | `[1,3,512,512] float32` | 3.93e-03 | 114.0 | 25.5% | 5.51e-03 (FP32) | 159.6 | 0.674% | 1.40 | bd-bounded | **34.48 µs** | ATen bilinear upsample 256→512 |
| 2 | `vectorized_elementwise_mul` | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1272.5 | 284.0% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 159.1 | 0.671% | 0.125 | bd-bounded | **4.94 µs** | Image scale `×2` |
| 3 | `vectorized_elementwise_add` | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1249.8 | 279.0% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 156.2 | 0.659% | 0.125 | bd-bounded | **5.03 µs** | Image bias `−1` |
| 4 | `vectorized_elementwise_fill_bool` | — | `[1,32,32] bool` | 1.02e-06 | 0.887 | 0.198% | 0 | 0 | — | — | bd-bounded | **1.16 µs** | Fill mask `ones→bool` (avg of 2×/iter) |

### 2.2 Variant `no_scale` — interpolate + masks only

| Order | kernel / op | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning |
|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| 1 | `upsample_bilinear2d` | `[1,3,256,256] float32` | `[1,3,512,512] float32` | 3.93e-03 | 114.0 | 25.5% | 5.51e-03 (FP32) | 159.7 | 0.674% | 1.40 | bd-bounded | **34.48 µs** | ATen bilinear upsample 256→512 |
| 2 | `vectorized_elementwise_fill_bool` | — | `[1,32,32] bool` | 1.02e-06 | 0.888 | 0.198% | 0 | 0 | — | — | bd-bounded | **1.15 µs** | Fill mask `ones→bool` (avg of 2×/iter) |

### 2.3 Variant `no_upsample` — `×2−1` @ **512²** + masks

| Order | kernel / op | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning |
|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| 1 | `vectorized_elementwise_mul` | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1277.5 | 285.1% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 159.7 | 0.674% | 0.125 | bd-bounded | **4.92 µs** | Image scale `×2` |
| 2 | `vectorized_elementwise_add` | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1279.1 | 285.5% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 159.9 | 0.675% | 0.125 | bd-bounded | **4.92 µs** | Image bias `−1` |
| 3 | `vectorized_elementwise_fill_bool` | — | `[1,32,32] bool` | 1.02e-06 | 0.887 | 0.198% | 0 | 0 | — | — | bd-bounded | **1.16 µs** | Fill mask `ones→bool` (avg of 2×/iter) |

### 2.4 Cross-variant kernel metrics (same IO model)

| Kernel | Variant | GPU time | BD /GB/s | BD util | GFLOPs/sec | FLOPs util | AI |
|--------|---------|---------:|---------:|--------:|-----------:|-----------:|---:|
| upsample | full | 34.48 µs | 114.0 | 25.5% | 159.6 | 0.674% | 1.40 |
| upsample | no_scale | 34.48 µs | 114.0 | 25.5% | 159.7 | 0.674% | 1.40 |
| mul | full | 4.94 µs | 1272.5 | 284% ⚠ | 159.1 | 0.671% | 0.125 |
| mul | no_upsample | 4.92 µs | 1277.5 | 285% ⚠ | 159.7 | 0.674% | 0.125 |
| add | full | 5.03 µs | 1249.8 | 279% ⚠ | 156.2 | 0.659% | 0.125 |
| add | no_upsample | 4.92 µs | 1279.1 | 286% ⚠ | 159.9 | 0.675% | 0.125 |
| fill | full / no_scale / no_upsample | ~1.15–1.16 µs | ~0.89 | ~0.20% | 0 | — | — |

Launch counts in the 10-iter window:

| Kernel | full | no_scale | no_upsample |
|--------|-----:|---------:|------------:|
| upsample | 10 | 10 | 0 |
| mul | 10 | 0 | 10 |
| add | 10 | 0 | 10 |
| fill_bool | 20 | 20 | 20 |

---

## 3. Timing deltas (GPU busy / host)

### 3.1 Pairwise GPU busy / iter

| Comparison | Δ GPU busy / iter | Interpretation |
|------------|------------------:|----------------|
| **full − no_scale** | **+9.987 µs** | Cost of `×2−1` at 512² (mul+add ≈ **9.98 µs**) |
| **full − no_upsample** | **+34.619 µs** | ≈ upsample alone (**34.49 µs**); scale+masks cancel |
| **no_scale − no_upsample** | **+24.632 µs** | Upsample (~34.5) vs scale@512² (~9.8) + same masks |

Additive check:  
`no_scale.busy + no_upsample.busy − masks_once ≈ 36.79 + 12.15 − 2.31 = 46.63 µs` ≈ **full 46.77 µs** (within ~0.1 µs).

### 3.2 Host-wall deltas (10-iter mean)

| Comparison | Δ host mean |
|------------|------------:|
| full − no_scale | **+91.2 µs** |
| no_upsample − full | **+64.0 µs** |
| no_upsample − no_scale | **+155.2 µs** |

### 3.3 Stability (per-launch StdDev)

| Kernel | full σ | no_scale σ | no_upsample σ |
|--------|-------:|-----------:|--------------:|
| upsample | 0.373 µs | 0.357 µs | — |
| mul | 0.071 µs | — | 0.024 µs |
| add | 0.052 µs | — | 0.034 µs |
| fill_bool | 0.067 µs | 0.068 µs | 0.072 µs |

---

## 4. Findings

1. **All Pre ops are `bd-bounded`** under the Inductor/eager roofline (AI 0.125–1.40 ≪ ridge ~52.9). Upsample achieves only **~25%** DRAM util; mul/add show **⚠ BD util ~280%** → algorithmic IO ≫ DRAM (L2/cache hits), same pattern as Inductor §Pre.
2. **Upsample dominates** (~74% of `full` GPU busy; ~160 GFLOP/s, **0.67%** of FP32 peak).
3. **Scale `×2−1` @512² ≈ 10 µs/iter**; mul/add metrics match between `full` and `no_upsample` (BD ~1270 GB/s algo, GFLOPs/sec ~160).
4. **Masks negligible** (~1.16 µs, BD util ~0.2%).
5. **Additive decomposition holds** at matched 512²: `upsample + scale + masks ≈ full`.
6. Micro-bench GPU times are **~1.2–1.35×** Inductor §Pre (in-model warm chunk): same kernels/grids, colder clocks / looser launch cadence — use §Pre for in-model absolute µs; this report for ablation + roofline.

---

## 5. Relation to Inductor §Pre (same IO model, in-model times)

| Kernel | §Pre GPU time | §Pre BD | This `full` GPU time | This `full` BD |
|--------|--------------:|--------:|---------------------:|---------------:|
| upsample | **29.38 µs** | 133.8 GB/s (29.9%) | **34.48 µs** | 114.0 GB/s (25.5%) |
| mul | **3.65 µs** | 1723 GB/s (⚠) | **4.94 µs** | 1273 GB/s (⚠) |
| add | **3.71 µs** | 1695 GB/s (⚠) | **5.03 µs** | 1250 GB/s (⚠) |
| fill | **0.80 µs** | 1.28 GB/s | **1.16 µs** | 0.89 GB/s |

IO Volume / GFLOPs / AI / bottleneck labels are **identical** by construction (same shapes + heuristics); only measured time → BD / GFLOPs/sec / util differ.

---

## 6. Buffer-size sweep (`no_upsample` only) — why BD util drops toward ≤100%

**Goal:** Grow algo IO (`bytes_in + bytes_out` for one `mul`/`add`) through **12 / 24 / 30 / 32 / 36 MB** and watch BD util as the working set approaches **L2 = 32 MB** (RTX 5060 Ti).

**Setup:** `PROFILE_VARIANT=no_upsample TARGET_IO_MB=<N> INCLUDE_MASKS=0`  
Shape `[1,3,H,W] f32` with `2·3·H·W·4 ≈ TARGET_IO_MB` decimal MB. Scale only (`×2−1`); no interpolate, no masks. Warmup 5 + 10 CUPTI iters.  
Artifacts: `nsys/pre_upsample_no_upsample_io{12,24,30,32,36}.nsys-rep`.

Baseline row = prior `no_upsample` @512² (algo IO ≈ **6.29 MB**).

### 6.1 Shapes and host wall

| Target IO | Shape | Buffer (one tensor) | Algo IO (in+out) | Host mean (10 iters) |
|----------:|------:|--------------------:|-----------------:|---------------------:|
| ~6.29 MB (baseline) | `[1,3,512,512]` | 3.15 MB | 6.29 MB | 0.338 ms |
| **12 MB** | `[1,3,707,707]` | 6.00 MB | 12.00 MB | 0.165 ms |
| **24 MB** | `[1,3,1000,1000]` | 12.00 MB | 24.00 MB | 0.171 ms |
| **30 MB** | `[1,3,1118,1118]` | 15.00 MB | 30.00 MB | 0.176 ms |
| **32 MB** | `[1,3,1155,1155]` | 16.01 MB | 32.02 MB | 0.192 ms |
| **36 MB** | `[1,3,1225,1225]` | 18.01 MB | 36.02 MB | 0.198 ms |

### 6.2 Roofline vs size (`mul` / `add`; AI = 0.125; all `bd-bounded`)

Same IO model as §2: `IO = 2 · numel · 4`, `FLOPs = numel` (FP32).

| Algo IO | `mul` time | `mul` BD | `mul` BD util | `mul` GFLOPs/sec | `add` time | `add` BD | `add` BD util |
|--------:|-----------:|---------:|--------------:|-----------------:|-----------:|---------:|--------------:|
| 6.29 MB | **4.92 µs** | 1278 | **285%** ⚠ | 160 | **4.92 µs** | 1279 | **286%** ⚠ |
| 12.0 MB | **9.04 µs** | 1327 | **296%** ⚠ | 166 | **9.00 µs** | 1333 | **298%** ⚠ |
| 24.0 MB | **35.82 µs** | 670 | **150%** ⚠ | 83.7 | **19.88 µs** | 1207 | **270%** ⚠ |
| 30.0 MB | **60.29 µs** | 498 | **111%** ⚠ | 62.2 | **36.37 µs** | 825 | **184%** ⚠ |
| 32.0 MB | **67.90 µs** | 472 | **105%** ⚠ | 58.9 | **40.74 µs** | 786 | **175%** ⚠ |
| 36.0 MB | **81.62 µs** | 441 | **98.5%** | 55.2 | **51.88 µs** | 694 | **155%** ⚠ |

### 6.3 Interpretation

1. **Small IO (6–12 MB):** both mul and add show BD util **~285–300%**. Working set ≪ L2 (32 MB) → almost all traffic is on-chip; algo-IO≠DRAM.
2. **Crossing toward L2 (24→36 MB):** **`mul` BD util falls monotonically** 150% → 111% → 105% → **98.5%**. Once algo IO ≳ L2, the DRAM byte model becomes realistic and util settles near / below 100%.
3. **`add` stays “faster” / higher util than `mul` at large sizes:** `x*2-1` is two launches — mul writes a temp that often remains **hot in L2** for the following add, so add’s measured time understates DRAM need vs the same algo IO. Mul typically pays the cold read from the input buffer.
4. **This is direct evidence for the ⚠ tag:** BD util >100% is an artifact of the DRAM byte model on L2-resident working sets, not super-DRAM hardware. Growing the buffer until it stresses L2 removes the paradox for `mul`.
5. **Pre @512² (6.29 MB IO)** sits deep in the L2-resident regime — same reason Inductor §Pre mul/add report ⚠ ~380% util.

```text
BD util(mul) vs algo IO (sketch)

  300% | **  **
       |          *
  150% |                *
  100% |                    *  *   ← ~L2 (32 MB)
       +----12---24---30--32--36  MB algo IO
```

---

## 7. How to reproduce

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/pre-process-upsample
source /venv/main/bin/activate
export PATH=/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH

# Original three-way ablation
for v in full no_scale no_upsample; do
  nsys profile --force-overwrite=true \
    --trace=cuda,nvtx,osrt --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output=nsys/pre_upsample_${v} \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 PROFILE_VARIANT=$v \
    python profile_prepare_images_upsample.py
done

# Buffer-size sweep (no_upsample, masks off)
for mb in 12 24 30 32 36; do
  nsys profile --force-overwrite=true \
    --trace=cuda,nvtx,osrt --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output=nsys/pre_upsample_no_upsample_io${mb} \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 INCLUDE_MASKS=0 \
        PROFILE_VARIANT=no_upsample TARGET_IO_MB=$mb \
    python profile_prepare_images_upsample.py
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    --output=nsys/pre_upsample_no_upsample_io${mb} \
    nsys/pre_upsample_no_upsample_io${mb}.nsys-rep
done
```

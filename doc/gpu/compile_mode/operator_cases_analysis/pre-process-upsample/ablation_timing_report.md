# Pre · prepare_images upsample — ablation timing report

**Date:** 2026-09-28 (rev: `no_upsample` @ **512²**)  
**GPU:** NVIDIA GeForce RTX 5060 Ti  
**Nsight:** Systems 2025.1.3 · CUPTI range = measured window only  
**Harness:** `profile_prepare_images_upsample.py`  
**Protocol:** warmup **5** (outside CUPTI) + measured **10** sync’d iters (inside `cudaProfilerApi`)

Artifacts: `nsys/pre_upsample_{full,no_scale,no_upsample}.nsys-rep` + `*_cuda_gpu_kern_sum.csv`.

---

## 0. Experiment matrix

| Variant | Ops under test | Input → output |
|---------|----------------|----------------|
| **full** | `F.interpolate` 256→512 + `×2−1` + cam/patch masks | `[1,3,256,256]→[1,3,512,512]` |
| **no_scale** | interpolate + masks only (drop `×2−1`) | `[1,3,256,256]→[1,3,512,512]` |
| **no_upsample** | `×2−1` + masks only (drop interpolate); **input already 512²** | `[1,3,512,512]→[1,3,512,512]` |

Masks always: `cam_mask [1] bool` + `patch_mask [1,32,32] bool` → **2×** `FillFunctor<bool>` per iter.

`no_upsample` uses native **512²** so mul/add are size-matched to `full` (apples-to-apples scale cost).

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

## 2. Per-kernel comparison across the 3 experiments

Source: `nsys stats --report cuda_gpu_kern_sum` (Avg = mean over launches in the CUPTI window).

### 2.1 Average GPU time per launch (µs)

| Kernel (ATen role) | full | no_scale | no_upsample @512² | Notes |
|--------------------|-----:|---------:|------------------:|-------|
| `upsample_bilinear2d` | **34.485** | **34.479** | — | Identical when present (Δ −0.006 µs) |
| `mul` (`×2`) | **4.944** | — | **4.925** | Size-matched; Δ **−0.019 µs** (~same) |
| `add` (`−1`) | **5.034** | — | **4.919** | Size-matched; Δ **−0.115 µs** (~same) |
| `fill_bool` (mask) | **1.155** | **1.154** | **1.155** | Invariant across variants |

Launch counts in the 10-iter window:

| Kernel | full | no_scale | no_upsample |
|--------|-----:|---------:|------------:|
| upsample | 10 | 10 | 0 |
| mul | 10 | 0 | 10 |
| add | 10 | 0 | 10 |
| fill_bool | 20 | 20 | 20 |

### 2.2 Pairwise differences (GPU busy / iter)

| Comparison | Δ GPU busy / iter | Interpretation |
|------------|------------------:|----------------|
| **full − no_scale** | **+9.987 µs** | Cost of `×2−1` at 512² (mul+add ≈ 4.94+5.03 = **9.98 µs**) |
| **full − no_upsample** | **+34.619 µs** | ≈ upsample alone (**34.49 µs**); scale+masks cancel |
| **no_scale − no_upsample** | **+24.632 µs** | Upsample (~34.5) vs scale@512² (~9.8) + same masks |

Additive check:  
`no_scale.busy + no_upsample.busy − masks_once ≈ 36.79 + 12.15 − 2.31 = 46.63 µs` ≈ **full 46.77 µs** (within ~0.1 µs).  
→ costs are **additive** when sizes match.

Host-wall deltas (10-iter mean):

| Comparison | Δ host mean |
|------------|------------:|
| full − no_scale | **+91.2 µs** |
| no_upsample − full | **+64.0 µs** |
| no_upsample − no_scale | **+155.2 µs** |

Host still not a clean proxy for GPU op cost; CUPTI busy ranks ops correctly: **upsample ≫ scale@512² ≫ masks**.

### 2.3 Stability (per-launch StdDev)

| Kernel | full σ | no_scale σ | no_upsample σ |
|--------|-------:|-----------:|--------------:|
| upsample | 0.373 µs | 0.357 µs | — |
| mul | 0.071 µs | — | 0.024 µs |
| add | 0.052 µs | — | 0.034 µs |
| fill_bool | 0.067 µs | 0.068 µs | 0.072 µs |

---

## 3. Findings

1. **Upsample dominates Pre GPU time** (~74% of `full` busy: 34.5 / 46.8 µs).
2. **Scale `×2−1` at 512² is ~10 µs/iter** and matches between `full` and `no_upsample` (mul/add within ~0.1 µs).
3. **Masks ~2.3 µs/iter**, unchanged across ablations.
4. **Additive decomposition holds** when `no_upsample` uses 512²:  
   `upsample + scale + masks ≈ full`.
5. Earlier `no_upsample` @256² understated scale cost (~4 µs vs ~10 µs); **512² is the correct control** for comparing against `full`.

---

## 4. Relation to Inductor Pre table

Warm compile chunk Pre (cam0) in `SmolVLA_op_list_inductor_nsight.md`: upsample ~29 µs, mul/add ~3.7 µs, fill ~0.8 µs. This micro-bench is slightly higher (34.5 / ~4.9 / ~1.2 µs) on the same GPU class — same op order, different capture context. Ranking unchanged.

---

## 5. How to reproduce

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
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    --output=nsys/pre_upsample_${v} nsys/pre_upsample_${v}.nsys-rep
done
```

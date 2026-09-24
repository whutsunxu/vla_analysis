# SmolVLA GPU kernel launch list (Nsight)

Companion to `SmolVLA_Operator_gpu_List.md` (§3 Stage‑2 prefill). Source: `doc/nsight/smolvla_nsys.nsys-rep` → `CUPTI_ACTIVITY_KIND_KERNEL`. Stage‑2 window **564.90→576.32 ms**.

## 1. Prefill first layer — kernel launches in order

Table below is prefill **layer 1** of `L_p=16`: first **Q** GEMM through the post‑MLP RMSNorm that feeds layer‑2 Q (next Q at **565.6817 ms**). **77** launches; Q / attn‑out / MLP‑down are **separate** launches of the same CUTLASS name. `dur` = this launch’s GPU time (not an average). Cross-layer sameness and duration deviation: **§2–3**.

| # | t (ms) | dur (µs) | Kernel |
|--:|-------:|---------:|--------|
| 1 | 564.9483 | 14.50 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8` grid=(32,1,1) — **Q Linear** |
| 2 | 564.9649 | 11.10 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8` grid=(64,2,1) — **K Linear** |
| 3 | 564.9782 | 11.36 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8` grid=(64,2,1) — **V Linear** |
| 4 | 565.0042 | 3.23 | `direct_copy_kernel` grid=(452,1,1) |
| 5 | 565.0135 | 0.70 | `arange` grid=(1,1,1) |
| 6 | 565.0237 | 0.90 | `AUnaryFunctor` Mul grid=(1,1,1) |
| 7 | 565.0339 | 0.74 | `FillFunctor` grid=(1,1,1) |
| 8 | 565.0400 | 1.50 | `elementwise` grid=(1,1,1) |
| 9 | 565.0495 | 1.25 | `direct_copy_kernel` grid=(1,1,1) |
| 10 | 565.0597 | 1.15 | `BinaryFunctor` grid=(31,1,1) |
| 11 | 565.0679 | 1.54 | `sin_kernel` grid=(8,1,1) — RoPE |
| 12 | 565.0741 | 1.54 | `cos_kernel` grid=(8,1,1) — RoPE |
| 13 | 565.0942 | 1.86 | `BinaryFunctor` grid=(452,1,1) |
| 14 | 565.0970 | 1.76 | `BinaryFunctor` grid=(452,1,1) |
| 15 | 565.1008 | 1.70 | `CUDAFunctor_add` grid=(113,1,1) |
| 16 | 565.1088 | 1.70 | `direct_copy_kernel` grid=(452,1,1) |
| 17 | 565.1148 | 1.86 | `BinaryFunctor` grid=(452,1,1) |
| 18 | 565.1197 | 1.79 | `BinaryFunctor` grid=(452,1,1) |
| 19 | 565.1245 | 1.60 | `CUDAFunctor_add` grid=(113,1,1) |
| 20 | 565.1300 | 1.70 | `direct_copy_kernel` grid=(452,1,1) |
| 21 | 565.1366 | 1.41 | `bfloat16_copy_kernel` grid=(226,1,1) |
| 22 | 565.1469 | 1.89 | `direct_copy_kernel` grid=(151,1,1) |
| 23 | 565.1534 | 0.70 | `arange` grid=(1,1,1) |
| 24 | 565.1589 | 0.90 | `AUnaryFunctor` Mul grid=(1,1,1) |
| 25 | 565.1665 | 0.74 | `FillFunctor` grid=(1,1,1) |
| 26 | 565.1700 | 1.44 | `elementwise` grid=(1,1,1) |
| 27 | 565.1774 | 1.25 | `direct_copy_kernel` grid=(1,1,1) |
| 28 | 565.1858 | 1.09 | `BinaryFunctor` grid=(31,1,1) |
| 29 | 565.1912 | 1.44 | `sin_kernel` grid=(8,1,1) — RoPE |
| 30 | 565.1968 | 1.41 | `cos_kernel` grid=(8,1,1) — RoPE |
| 31 | 565.2063 | 1.22 | `BinaryFunctor` grid=(151,1,1) |
| 32 | 565.2114 | 1.18 | `BinaryFunctor` grid=(151,1,1) |
| 33 | 565.2159 | 1.28 | `CUDAFunctor_add` grid=(38,1,1) |
| 34 | 565.2216 | 1.15 | `direct_copy_kernel` grid=(151,1,1) |
| 35 | 565.2271 | 1.18 | `BinaryFunctor` grid=(151,1,1) |
| 36 | 565.2322 | 1.12 | `BinaryFunctor` grid=(151,1,1) |
| 37 | 565.2368 | 1.22 | `CUDAFunctor_add` grid=(38,1,1) |
| 38 | 565.2424 | 1.09 | `direct_copy_kernel` grid=(151,1,1) |
| 39 | 565.2483 | 0.96 | `bfloat16_copy_kernel` grid=(76,1,1) |
| 40 | 565.2704 | 3.01 | `CatArrayBatchedCopy` grid=(72,2,1) |
| 41 | 565.2770 | 3.01 | `CatArrayBatchedCopy` grid=(72,2,1) |
| 42 | 565.2931 | 2.82 | `direct_copy_kernel` grid=(452,1,1) |
| 43 | 565.3025 | 2.75 | `direct_copy_kernel` grid=(452,1,1) |
| 44 | 565.3084 | 3.20 | `direct_copy_kernel` grid=(452,1,1) |
| 45 | 565.3141 | 3.23 | `direct_copy_kernel` grid=(452,1,1) |
| 46 | 565.3331 | 21.06 | `magma_sgemmEx_kernel` grid=(4,4,15) — **QKᵀ** |
| 47 | 565.3550 | 4.13 | `AUnaryFunctor` Mul grid=(851,1,1) — **scale ÷8** |
| 48 | 565.3612 | 0.74 | `FillFunctor` grid=(1,1,1) |
| 49 | 565.3632 | 7.26 | `where_kernel` grid=(3404,1,1) — **attn mask where** |
| 50 | 565.3724 | 6.30 | `softmax_warp_forward<(int)8>` grid=(904,1,1) — **Softmax** |
| 51 | 565.3796 | 2.85 | `bfloat16_copy_kernel` grid=(851,1,1) |
| 52 | 565.3927 | 9.60 | `cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nn_align1` grid=(4,1,15) — **A·V** |
| 53 | 565.4032 | 2.75 | `direct_copy_kernel` grid=(452,1,1) |
| 54 | 565.4279 | 14.50 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8` grid=(32,1,1) — **attn-out Linear** |
| 55 | 565.4441 | 4.03 | `CUDAFunctor_add<float>` grid=(452,1,1) — residual add (**L1 only**; see §2) |
| 56 | 565.4538 | 3.26 | `direct_copy_kernel` grid=(452,1,1) |
| 57 | 565.4606 | 1.60 | `pow_tensor_scalar` grid=(226,1,1) — RMSNorm |
| 58 | 565.4706 | 2.59 | `reduce_kernel` MeanOps grid=(16,1,1) — RMSNorm |
| 59 | 565.4788 | 0.96 | `CUDAFunctorOnSelf_add` grid=(1,1,1) |
| 60 | 565.4884 | 0.96 | `rsqrt_kernel` grid=(1,1,1) |
| 61 | 565.4949 | 2.50 | `BinaryFunctor` Mul float grid=(904,1,1) |
| 62 | 565.5016 | 1.44 | `bfloat16_copy_kernel` grid=(226,1,1) |
| 63 | 565.5073 | 2.82 | `BinaryFunctor` bf16 grid=(452,1,1) |
| 64 | 565.5239 | 40.67 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8` grid=(32,3,1) — **MLP gate** |
| 65 | 565.5670 | 2.78 | `silu_kernel` grid=(603,1,1) — **SiLU** |
| 66 | 565.5711 | 40.96 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8` grid=(32,3,1) — **MLP up** |
| 67 | 565.6141 | 2.34 | `BinaryFunctor` bf16 grid=(603,1,1) — gate×up |
| 68 | 565.6182 | 33.89 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8` grid=(32,1,1) — **MLP down** |
| 69 | 565.6530 | 1.44 | `CUDAFunctor_add` grid=(226,1,1) — residual add |
| 70 | 565.6551 | 3.20 | `direct_copy_kernel` grid=(452,1,1) |
| 71 | 565.6592 | 1.63 | `pow_tensor_scalar` grid=(226,1,1) — next-layer RMSNorm |
| 72 | 565.6632 | 2.62 | `reduce_kernel` MeanOps grid=(16,1,1) |
| 73 | 565.6673 | 0.93 | `CUDAFunctorOnSelf_add` grid=(1,1,1) |
| 74 | 565.6694 | 0.99 | `rsqrt_kernel` grid=(1,1,1) |
| 75 | 565.6714 | 2.43 | `BinaryFunctor` Mul float grid=(904,1,1) |
| 76 | 565.6755 | 1.34 | `bfloat16_copy_kernel` grid=(226,1,1) |
| 77 | 565.6776 | 2.75 | `BinaryFunctor` bf16 grid=(452,1,1) |

Three tiny kernels sit *before* layer‑1 Q (564.903–564.922 ms: Binary Mul ×2 + bf16 copy) — window-edge / Stage‑1 bleed, omitted above.

## 2. Op-list difference across the 16 prefill layers

All **16** layers in **564.90→576.32 ms** have **exactly 77** launches in the same Q→…→post‑MLP‑RMSNorm span. Compared by (kernel family + grid):

| Slot | Layers 2–16 (majority, 15×) | Layer 1 only |
|---:|---|---|
| **#55** (first residual after attn-out) | `CUDAFunctor_add<c10::BFloat16>` grid=**(226,1,1)** — **~1.47 µs** mean | `CUDAFunctor_add<float>` + `LoadWithCast` grid=**(452,1,1)** — **4.03 µs** |

No other slot differs: the other **76** kernels match name+grid across all 16 layers (including Q / K / V / `QKᵀ` / where / Softmax / `A·V` / attn-out / gate / SiLU / up / down / second residual).

## 3. Duration deviation across the 16 layers

Per-layer sum of the 77 launch durations: **mean 328.93 µs**, σ **0.75 µs**, range **[328.22, 331.46]** (L1 is the high outlier mainly from #55).

Major ops (same kernel+grid on every layer) — duration over 16 layers:

| # | Op | mean (µs) | σ (µs) | min–max | Δ (max−min) | CV |
|--:|---|---:|---:|---|---:|---:|
| 1 | Q Linear | 14.39 | 0.11 | 14.24–14.66 | 0.42 | 0.8% |
| 2 | K Linear | 11.33 | 0.10 | 11.10–11.49 | 0.38 | 0.9% |
| 3 | V Linear | 11.36 | 0.17 | 11.10–11.71 | 0.61 | 1.5% |
| 46 | `QKᵀ` magma | 20.94 | 0.14 | 20.67–21.18 | 0.51 | 0.7% |
| 50 | Softmax | 6.41 | 0.16 | 6.14–6.66 | 0.51 | 2.4% |
| 52 | `A·V` | 9.77 | 0.14 | 9.50–10.02 | 0.51 | 1.5% |
| 54 | attn-out Linear | 14.58 | 0.14 | 14.27–14.82 | 0.54 | 0.9% |
| 64 | MLP gate | 40.49 | 0.12 | 40.26–40.67 | 0.42 | 0.3% |
| 66 | MLP up | 40.57 | 0.21 | 40.19–40.96 | 0.77 | 0.5% |
| 68 | MLP down | 33.78 | 0.15 | 33.57–34.11 | 0.54 | 0.5% |

All other same-kernel slots: **Δ ≤ ~0.5 µs** and **CV ≲ 6%** (tiny elementwise/copy). The only large cross-layer duration swing is **#55**, and that is the **kernel/dtype change** above (L1 FP32 cast-add vs L2–16 BF16 add), not run-to-run noise on one kernel.

**Verdict:** Across Stage‑2 prefill, the kernel list is **the same for 76/77 slots**; durations of the shared GEMM/attn kernels are **tight** (CV ≤ 2.4%, Δ ≤ 0.77 µs). Sole op-list difference = **layer‑1 post‑attn residual** FP32 path.

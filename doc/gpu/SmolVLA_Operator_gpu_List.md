# SmolVLA Operator List (GPU / Nsight-verified)

This document mirrors `SmolVLA_Operator_List.md` for the same `lerobot/smolvla_base` inference path, but replaces algorithmic FLOP accounting with **what actually runs on the GPU**, verified against:

| Artifact | Role |
|---|---|
| `doc/nsight/smolvla_nsys.nsys-rep` | Nsight Systems timeline (open with `nsys-ui`) |
| `doc/nsight/smolvla_nsys_cuda_gpu_kern_sum.csv` | Aggregated CUDA kernel times |
| `doc/nsight/smolvla_nsys_kernel_summary.json` | Classified summary |
| `src/smolvla_test_infer.py` / `src/smolvla_nsight_target.py` | Stage 0–4 chunk-fill workload |
| `doc/smolVLA_profiling.md` | Stage wall-clock (CUDA-synchronized) |

Capture: **Nsight Systems 2025.1.3**, RTX 5060 Ti, `torch 2.11.0+cu128`, workload = model load + **warmup×2 + 1 measured** Stage 0–4 chunk fill.  
Therefore for inference kernels whose launch counts divide cleanly by 3:

```text
launches_per_chunk ≈ Instances / 3
time_per_chunk_ms  ≈ Total_ms / 3
```

Session totals include that 3× inference traffic (plus a few load-only kernels). Stage wall clocks from `smolVLA_profiling.md` (5-run mean, warmup excluded) are quoted alongside for latency context.

---

## 0. Notation, fixed dimensions, and GPU datatype rules

Same symbols as `SmolVLA_Operator_List.md` (`B`, `C=3`, `S_p=241`, `N=50`, `M=10`, `L_v=12`, `L_p=16`, `L_e=16`, …).

### 0.1 What the Nsight file shows vs the paper-style operator list

| Algorithmic view (`SmolVLA_Operator_List.md`) | GPU reality (this file / CSV) |
|---|---|
| Separate `QKᵀ`, scale, mask, softmax, `A·V` in ViT | **Fused** `pytorch_flash::flash_fwd_kernel` (one kernel) |
| Separate Linear then activation | Often **CUTLASS GEMM + epilogue** (`*_gemm_relu_*` family) or separate GELU/SiLU kernels |
| Many `float32` math ops in Stage 1 sinusoid | `sin` / `cos` / `pow` / `rsqrt` **elementwise CUDA kernels**; some paths still use **FP64** intermediates then cast |
| Vision / VLM weights `bfloat16` | Confirmed: CUTLASS `bf16` tensorop/WMMA, `implicit_convolve_sgemm<__nv_bfloat16,…>` |
| Text/expert attention softmax in FP32 | Confirmed: `softmax_warp_forward<float,…>` (not Flash) |
| Cast / copy “0 FLOPs” | **Large GPU cost**: `direct_copy` / `bfloat16_copy` with `LoadWithCast` / `StoreWithCast` |

### 0.2 Session-level kernel mix (from CSV)

| Category (name heuristic) | Session ms | Share | Launches | Unique kernels |
|---|---:|---:|---:|---:|
| matmul / GEMM / Flash / conv | 110.98 | 64.5% | 6768 | 24 |
| elementwise (incl. copy/cast) | 52.09 | 30.3% | 31689 | 50 |
| reduction | 3.28 | 1.9% | 1404 | 5 |
| other | 2.39 | 1.4% | 882 | 14 |
| LayerNorm | 1.82 | 1.1% | 225 | 1 |
| Softmax (standalone) | 1.49 | 0.9% | 528 | 2 |
| gather / embedding | 0.05 | 0.03% | 12 | 1 |
| **Total** | **172.10** | 100% | **41508** | **97** |

### 0.3 Backend inventory (verified by kernel names)

| Backend | Evidence in CSV | Typical role |
|---|---|---|
| **CUTLASS** (OSS, via PyTorch ATen) | `cutlass::Kernel2<cutlass_80_tensorop_bf16_…>`, `…_wmma_…`, `…_simt_sgemm_…` | Dominant bf16 / some fp32 Linear & MatMul |
| **FlashAttention** (`pytorch_flash`) | `pytorch_flash::flash_fwd_kernel<… cutlass::bfloat16_t …>` | **ViT self-attention only** (fused) |
| **cuBLAS / cuBLASLt** | `magma_sgemmEx_kernel` + `cublasLtEpilogue_t`, `cublasLt::splitKreduce_kernel`, `sgemm_largek_lds64`, `gemvx` | FP32 Linears / split-K epilogues |
| **cuDNN-style conv** | `implicit_convolve_sgemm<__nv_bfloat16,…>` | ViT patch `Conv2d` |
| **ATen CUDA elementwise** | `vectorized_elementwise_kernel`, `unrolled_elementwise_kernel`, … | cast, add/mul/div, GELU, SiLU, sin/cos, where |
| **ATen norm / softmax** | `vectorized_layer_norm_kernel`, `softmax_warp_forward` | ViT LN; VLM+expert softmax |

SmolVLA / LeRobot do **not** call CUTLASS APIs directly; PyTorch dispatches them.

### 0.4 Table columns in this document

| Column | Meaning |
|---|---|
| Operator | Logical op — **Linears listed separately** even when they share a CUTLASS/cuBLASLt kernel name |
| Input → output (shape, dtype) | Activation shapes (`B=1` on robot) and runtime dtypes, in one cell |
| **GPU time per launch** | CSV **Avg** for one Repeat (e.g. `avg 199.4 µs`). Do **not** pre-multiply by `C` / Repeat. *[med]* = shared tile Avg (§11) |
| **FLOPs** | Unified count **per launch** (`add`=`mul`=`div`=`exp`=`sin`/`cos`/`sqrt`=1). MatMul `(2K−1)MN`; Conv `H·W·C_out·(2k²C_in−1)`; GELU 13/elem; SiLU 5/elem; Softmax `5K−2`/row; LN `7D+3`/token; RMSNorm ≈`4D+3`/token. Cast/copy = 0. Chunk total ≈ FLOPs × Repeat |
| **TFLOPS/s** | `FLOPs_launch / time_launch` |
| **FLOPs util** | **Calculated** `TFLOPS/s ÷ peak` (**not capped**; may exceed 100% if peak/time attribution is optimistic). Peak = **94.8** TFLOP/s (BF16 dense TC) for BF16 **Conv / MatMul / Linear**; **23.7** TFLOP/s (CUDA) for FP32 GEMM and non-GEMM ops. `†` = Flash vs TC peak (approximate) |
| **IO Volume/MB** | Bytes **per launch** (`|X|+|W|+|Y|` or elem R+W) ÷ 1e6 from tensor sizes. Chunk total ≈ value × Repeat. This is **traffic accounting**, not a measured DRAM counter |
| **BD GB/s** | **Calculated** `IO_bytes / time_launch`. Can exceed DRAM peak when traffic is L2-hit, time is *[med]*-shared, or IO is over-counted |
| **BD util** | **Calculated** `BD GB/s ÷ 448` (**not capped**). Values >100% mean the simple IO/time model is not a true DRAM util |
| **Arithmetic Intensity** | `FLOPs ÷ (IO Volume/MB × 10⁶)` when IO is filled |
| **Bound** | Roofline vs §0.6 when AI is filled. Ridge = **TC 211.6** for BF16 Conv/MatMul/Linear; **CUDA 52.9** otherwise. `†` = Flash vs TC (approximate) |
| Repeat / GEMM launches | How often the op runs per chunk; discrete GEMM launches /chunk (metadata — multiply into time/FLOPs/IO for chunk totals) |
| GPU kernel(s) / Fusion | From `smolvla_nsys_cuda_gpu_kern_sum.csv` (metadata) |

**Timing rule:** Nsight merges equal-shape launches into one CSV row; per-op launch time is still that row’s **Avg**. Chunk time = Avg × Repeat (e.g. ViT Q chunk = 36 × 31.70 µs).

### 0.5 Stage wall-clock reference (not kernel sum)

From `doc/smolVLA_profiling.md` (CUDA sync, 5-run mean):

| Stage | Mean wall (ms) | Share of ~91 ms wall |
|---|---:|---:|
| 0 prefix embed | 23.9 | 26% |
| 2 VLM prefill | 7.2 | 8% |
| 1+3 Euler loop | 59.1 | 65% |
| 4 crop/queue/post | ~0.05 | ~0% |
| **Wall total** | **91.3** | 100% |

Kernel-session sum (172 ms) ≠ one-chunk wall (91 ms) because the Nsight session includes **2 warmups + 1 measure** (~3×) plus load-time traffic.

### 0.6 Platform metrics (RTX 5060 Ti — theoretical peaks)

Capture GPU: **NVIDIA GeForce RTX 5060 Ti** (Blackwell **GB206**, SM **12.0**). Reference boost **2572 MHz**.

| Resource | Spec |
|---|---|
| SMs / CUDA cores / Tensor Cores | **36** / **4608** / **144** (5th gen) |
| Memory | GDDR7, 128-bit, 28 Gbps → **448 GB/s** |
| VRAM (this host) | ~16.6 GB |

| Compute path | Precision | Peak throughput | Physical intensity (FLOPs/byte) |
|---|---|---|---:|
| Vector / CUDA cores | FP32 | **23.7 TFLOP/s** | **52.9** |
| Vector / CUDA cores | BF16 | **~23.7 TFLOP/s** * | **~52.9** |
| Matrix / Tensor Cores | FP32 | No native FP32 Tensor Core mode | — |
| Matrix / Tensor Cores | TF32 | **~23.7 TFLOP/s** (GeForce-style quote) | **~52.9** |
| Matrix / Tensor Cores | BF16, FP32 accumulate | **~94.8 TFLOP/s** (= 4× FP32 CUDA) | **~211.6** |
| Matrix / Tensor Cores | BF16, FP32 accumulate + 2:4 sparsity | **~189.6 TFLOP/s** | **~423.2** |
| DRAM | GDDR7 | **448 GB/s** | — (bandwidth only) |

Physical intensity (ridge point) = `Peak FLOP/s ÷ Peak DRAM BW` = `TFLOP/s ÷ 0.448` with DRAM = **448 GB/s**. Stage **Bound** column: compare each op’s **Arithmetic Intensity** to the ridge for its path (TC **211.6** for Conv/MatMul/Linear; CUDA **52.9** otherwise; Flash uses TC temporarily — see §0.4 `†`). AI ≥ ridge → **calc-bound**; else **bw-bound**.

\* Some docs list packed CUDA BF16 as 2× FP32 (~47 TFLOP/s → physical intensity ~**104.9**); consumer sheets often quote ≈ FP32.

**Roofline use with this doc:** compare BF16 CUTLASS/Flash GEMMs to **94.8 TFLOP/s** dense TC (ridge **~211.6** FLOPs/byte; PyTorch path here does **not** use 2:4 sparsity). Compare FP32 SIMT/cuBLASLt GEMMs to **23.7 TFLOP/s** CUDA (ridge **52.9**). Compare copy/cast and low–arithmetic-intensity elementwise to **448 GB/s**. Peaks assume sustained boost; boards may clock lower under power/thermal limits. NVIDIA “AI TOPS” marketing (e.g. 759) is a different sparse low-precision metric — not the BF16 dense peak above.

---

## 1. Stage 0 — prefix embedding (GPU)

**Function:** three cameras + language + state → prefix `P` `(B,241,960)`.

### 1.1 Image resize and normalization

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Bilinear upsample | `(B,3,256,256) → (B,3,512,512)`, `float32 → float32` | **avg 27.8 µs** | 5.51M | 0.20 | 0.84% | 3.93 | 142.1 | 31.7% | 1.4 | bw-bound | `C/chunk` | `upsample_bilinear2d_out_frame<float,float>` | spatial resize | 0 |
| 2 | Scale/bias `2x-1` | `(B,3,512,512) →` same, `float32 → float32` | **avg 1.17 µs** *[med]* | 1.57M | 1.35 | 5.7% | 9.44 | 8089.0 | 1805.6% | 0.2 | bw-bound | `C/chunk` | ATen mul/add | *[med]* shared time — no DRAM BD | 0 |
| 3 | Cast image → vision dtype | `(B,3,512,512) →` same, **`float32 → bfloat16`** | **avg 1.96 µs** *[med]* | 0 | — | — | 4.72 | 2399.3 | 535.6% | 0 | bw-bound | `C/chunk` | `direct_copy` + cast | *[med]*; cast traffic uncertain | 0 |

**Verified:** upsample Inst `9 = C×3`.

### 1.2 ViT patch embedding

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Patch `Conv2d` `3→768`, k/s=16 | `(B,3,512,512) → (B,768,32,32)` → `(B,1024,768)`, `bfloat16 → bfloat16` | **avg 199.4 µs** | 1.21G | 6.06 | 6.4% | 4.33 | 21.7 | 4.8% | 279.1 | calc-bound | `C/chunk` | `implicit_convolve_sgemm<__nv_bfloat16,…>` | BF16 implicit-GEMM conv | **3** |
| 2 | Position-ID build | mask → `(B,1024)`, `bool → int64` | tiny | — | — | — | — | — | — | — | — | `C/chunk` | arange / bucketize | mostly small | 0 |
| 3 | Position-embedding gather | `(B,1024) → (B,1024,768)`, `int64 → bfloat16` | **avg 4.24 µs** *[med]* | 0 | — | — | 1.57 | 363.0 | 81.0% | 0 | bw-bound | `C/chunk` | `vectorized_gather_kernel` | — | 0 |
| 4 | Patch + position add | two `(B,1024,768) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 20.89 µs** | 786.7k | 0.04 | 0.17% | 4.72 | 224.7 | 50.2% | 0.2 | bw-bound | `C/chunk` | BF16 add | — | 0 |

### 1.3 ViT encoder: 12 blocks × 3 cameras

Every Linear below is a **separate operator / GEMM call site**. Attention score path is **fused** (Flash).
`Repeat = 36/chunk` means `C×L_v = 3×12` unless noted.

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | `LayerNorm1` (pre-attn) | `(B,1024,768) →` same, act BF16; stats **FP32** | **avg 8.07 µs** | 5.51M | 0.68 | 2.9% | 3.15 | 389.9 | 87.0% | 1.7 | bw-bound | `36/chunk` | `vectorized_layer_norm_kernel` | fused | 0 |
| 2 | **Q projection** Linear `768→768` | `(B,1024,768) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 31.70 µs** | 1.21G | 38.09 | 40.2% | 4.33 | 136.5 | 30.5% | 279.1 | calc-bound | `36/chunk` | `cutlass …64x64_32x6` | BF16 TC; shared n1=144 | **36** |
| 3 | **K projection** Linear `768→768` | `(B,1024,768) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 31.70 µs** | 1.21G | 38.09 | 40.2% | 4.33 | 136.5 | 30.5% | 279.1 | calc-bound | `36/chunk` | `cutlass …64x64_32x6` | BF16 TC; shared n1=144 | **36** |
| 4 | **V projection** Linear `768→768` | `(B,1024,768) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 31.70 µs** | 1.21G | 38.09 | 40.2% | 4.33 | 136.5 | 30.5% | 279.1 | calc-bound | `36/chunk` | `cutlass …64x64_32x6` | BF16 TC; shared n1=144 | **36** |
| 5 | Reshape + transpose to heads | `(B,1024,768) → (B,12,1024,64)`, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `3×36/chunk` | view | layout | 0 |
| 6 | **Fused self-attention** | Q,K,V → context `(B,12,1024,64)`, `bfloat16` | **avg 82.35 µs** | 3.28G | 39.87 | 42.1%† | 6.29 | 76.4 | 17.1% | 521.9 | calc-bound† | `36/chunk` | **`pytorch_flash::flash_fwd_kernel`** | fused QK/softmax/AV | 0 |
| 7 | Transpose + reshape concat heads | `(B,12,1024,64) → (B,1024,768)`, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `36/chunk` | view | layout | 0 |
| 8 | **Attn out Linear** `768→768` | `(B,1024,768) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 31.69 µs** | 1.21G | 38.09 | 40.2% | 4.33 | 136.5 | 30.5% | 279.1 | calc-bound | `36/chunk` | same as Q | identical shape | **36** |
| 9 | First residual add | two `(B,1024,768)`, `bfloat16` | **avg 20.89 µs** | 786.4k | 0.04 | 0.17% | 4.72 | 225.9 | 50.4% | 0.2 | bw-bound | `36/chunk` | BF16 add | — | 0 |
| 10 | `LayerNorm2` (pre-MLP) | `(B,1024,768) →` same, act BF16; stats FP32 | **avg 8.07 µs** | 5.51M | 0.68 | 0.72% | 3.15 | 389.9 | 87.0% | 1.7 | bw-bound | `36/chunk` | same LN | fused | 0 |
| 11 | **MLP up Linear** `768→3072` | `(B,1024,768) → (B,1024,3072)`, `bfloat16 → bfloat16` | **avg 119.80 µs** | 4.83G | 40.30 | 42.5% | 12.58 | 105.0 | 23.4% | 383.8 | calc-bound | `36/chunk` | `cutlass …256x128` | BF16 TC | **36** |
| 12 | GELU | `(B,1024,3072) →` same, `bfloat16` | **avg 9.77 µs** | 40.83M | 4.18 | 17.6% | 12.58 | 1286.9 | 287.3% | 3.2 | bw-bound | `36/chunk` | `GeluCUDAKernelImpl` | **MLP epilogue** on `fc1` output (L2-hot) | 0 |
| 13 | **MLP down Linear** `3072→768` | `(B,1024,3072) → (B,1024,768)`, `bfloat16 → bfloat16` | **avg 151.99 µs** | 4.83G | 31.78 | 33.5% | 12.58 | 82.8 | 18.5% | 383.9 | calc-bound | `36/chunk` | `cutlass …256x64` | BF16 TC | **36** |
| 14 | Second residual add | two `(B,1024,768)`, `bfloat16` | **avg 20.89 µs** | 786.4k | 0.04 | 0.17% | 4.72 | 225.9 | 50.4% | 0.2 | bw-bound | `36/chunk` | BF16 add | — | 0 |

After 12 blocks, **`post_layernorm`** once per camera — included in LN Inst `75/chunk = 36+36+3`.

**ViT Linear time /chunk:** `(4×31.70 + 119.80 + 151.99) µs × 36 = **14.35 ms**` (+ Flash 82.35×36 + GELU 9.77×36). Count checks: Flash/GELU `36/chunk`; LN `75/chunk`.

### 1.4 PixelShuffle and connector

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | PixelShuffle rearrange | `(B,1024,768) → (B,64,12288)`, `bfloat16` | ≈0 arith | — | — | — | — | — | — | — | — | `C/chunk` | view | layout | 0 |
| 2 | **Connector Linear** `12288→960` | `(B,64,12288) → (B,64,960)`, `bfloat16 → bfloat16` | **avg 63.16 µs** | 1.51G | 23.97 | 25.3% | 25.29 | 401.4 | 89.6% | 59.7 | bw-bound | `C/chunk` | `…s16816gemm_bf16_256x64` | BF16 TC | **3** |
| 3 | Multiply `√960` | `(B,64,960) →` same, `bfloat16` | **avg 0.91 µs** *[med]* | 61.4k | 0.07 | 0.30% | 0.247 | 273.1 | 61.0% | 0.2 | bw-bound | `C/chunk` | mul | — | 0 |
| 4 | Expand camera mask | `(B,) → (B,64)`, `bool` | tiny | — | — | — | — | — | — | — | — | `C/chunk` | expand | — | 0 |

### 1.5 Language-token embedding

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Tokenize / pad (CPU) | strings → `(B,48)`, `int64` | not in CUDA CSV | — | — | — | — | — | — | — | — | `1/chunk` | — | CPU | 0 |
| 2 | Embedding gather | `(B,48) → (B,48,960)`, `int64 → bfloat16` | tiny | — | — | — | — | — | — | — | — | `1/chunk` | gather | — | 0 |
| 3 | Multiply `√960` | `(B,48,960) →` same, `bfloat16` | **avg 0.91 µs** *[med]* | 46.1k | 0.05 | 0.21% | 0.18 | 204.8 | 45.7% | 0.2 | bw-bound | `1/chunk` | mul | — | 0 |

### 1.6 Robot-state embedding

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Pad state | `(B,6) → (B,32)`, `float32` | tiny | — | — | — | — | — | — | — | — | `1/chunk` | copy | — | 0 |
| 2 | **State Linear** `32→960` | `(B,32) → (B,960)`, `float32 → float32` | **avg 2.23 µs** | 61.4k | 0.03 | 0.13% | 0.13 | 58.6 | 13.1% | 0.5 | bw-bound | `1/chunk` | **`gemvx`** | FP32; Inst/3=1 | **1** |

### 1.7 Prefix assembly

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Concat tokens → `P` | → `(B,241,960)`, mixed → **FP32** | **not uniquely timed** | — | — | — | — | — | — | — | — | `1/chunk` | FP32 Cat / copies | upcast | 0 |
| 2 | Concat validity masks | → `(B,241)`, `bool` | tiny | — | — | — | — | — | — | — | — | `1/chunk` | cat | — | 0 |
| 3 | Attention-group mask | → `(B,241)`, `bool` | tiny | — | — | — | — | — | — | — | — | `1/chunk` | fill | — | 0 |

### 1.8 Stage-0 GPU reading

- Wall Stage 0 ≈ **23.9 ms**.
- ViT Q/K/V/out: **1.141 ms/chunk each** (`…64x64…` avg 31.70 µs × 36).
- ViT MLP up / down: **4.313 / 5.472 ms/chunk** (`256x128` / `256x64`).
- ViT residual adds: **0.752 ms/chunk each** (`CUDAFunctor_add<BF16>` avg 20.89 µs × 36; n1=78 = 36+36+3+3).
- Connector: **0.189 ms/chunk** (`…s16816gemm_bf16_256x64` no-relu, Inst/3=3).
- State Linear: **2.2 µs/chunk** (`gemvx`, Inst/3=1).
- Times marked *[med]* share a CSV row with other equal-tile ops; still `Avg × this_op_launches`.

---

## 2. Stage 1 — action-suffix embedding (GPU)

**Function:** `x_t` + flow time `t` → `U_t` `(B,50,720)`, **M=10** times per chunk.

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | **Action-in Linear** `32→720` | `(B,50,32) → (B,50,720)`, `float32 → float32` | **avg 3.10 µs** | 2.27M | 0.73 | 3.1% | 0.243 | 78.2 | 17.5% | 9.4 | bw-bound | `M/chunk` | `simt_sgemm_32x128` | FP32 unique n1=10 | **10** |
| 2 | Freq / period construction | scalars → `(360,)`, FP64 | **avg 6.04 µs** *[med]* | 1.1k | 0.0002 | 0.00% | 0.0060 | 1.0 | 0.22% | 0.2 | bw-bound | `M/chunk` | `pow`/linspace | FP64 | 0 |
| 3 | `sin` / `cos` of `t`-scaled freqs | `(B,360)` each, FP64 | **avg 3.36 µs** | 720 | 0.0001 | 0.00% | 0.012 | 1.7 | 0.38% | 0.1 | bw-bound | `2×M/chunk` | sin/cos`<double>` | — | 0 |
| 4 | Cast time embedding | `(B,720) →` same, **`float64 → float32`** | **avg 1.96 µs** *[med]* | 0 | — | — | 0.0090 | 4.3 | 0.96% | 0 | bw-bound | `M/chunk` | copy-cast | — | 0 |
| 5 | Broadcast time over horizon | `(B,720) → (B,50,720)`, `float32` | tiny | — | — | — | — | — | — | — | — | `M/chunk` | expand | — | 0 |
| 6 | Concat action ‖ time | → `(B,50,1440)`, `float32` | **avg 2.17 µs** *[med]* | 0 | — | — | 0.288 | 130.9 | 29.2% | 0 | bw-bound | `M/chunk` | Cat OpaqueType4 | — | 0 |
| 7 | **Fusion Linear mid** `1440→720` | `(B,50,1440) → (B,50,720)`, `float32 → float32` | **avg 39.14 µs** | 104.00M | 2.65 | 11.2% | 4.58 | 117.1 | 26.1% | 22.6 | bw-bound | `M/chunk` | `sgemm_largek_lds64` | FP32 | **10** |
| 8 | SiLU | `(B,50,720) →` same, `float32` | **avg 1.18 µs** | 180.0k | 0.15 | 0.63% | 0.288 | 240.0 | 53.6% | 0.6 | bw-bound | `M/chunk` | `silu_kernel<float>` | — | 0 |
| 9 | **Fusion Linear out** `720→720` | `(B,50,720) → (B,50,720)`, `float32 → float32` | **avg 15.27 µs** *[med]* | 51.80M | 3.39 | 14.3% | 2.36 | 154.4 | 34.5% | 21.9 | bw-bound | `M/chunk` | `simt_sgemm_64x64_tn` | n1=180 group | **10** |
| 10 | Suffix pad / attn-group masks | → `(B,50)`, bool | tiny | — | — | — | — | — | — | — | — | `M/chunk` | fill | — | 0 |

**Stage-1 GEMM call budget /chunk:** **30** FP32 Linears (`10+10+10`).

| Op | Kernel (CSV) | Inst/3 | Avg µs | Time /chunk |
|---|---|---:|---:|---:|
| Action-in `32→720` | `simt_sgemm_32x128` | 10 | 3.10 | **0.031 ms** |
| Fusion mid `1440→720` | `sgemm_largek_lds64` | 10 | 39.14 | **0.391 ms** |
| Fusion out `720→720` | `simt_sgemm_64x64_tn` | 10 of 180 | 15.27 | **0.153 ms** *[med]* |
| SiLU | `silu_kernel<float>` | 10 | 1.18 | **0.012 ms** |
| sin+cos (time emb, FP64) | `sin`/`cos`<double> | 10 each | 3.36 | **0.067 ms** |

FP32 `simt_sgemm_64x64_tn` n1=180 = fusion-out(10) + velocity(10) + cross-K(80) + cross-V(80). Stage-1 FP32 sin/cos are the **double** kernels (Inst/3=10); the large FP32 sin/cos pool (272) is RoPE in Stages 2–3.

---

## 3. Stage 2 — prefix prefill / KV cache (GPU)

**Function:** 16-layer VLM over `P`; build prefix KV.

### 3.1 Prefix masks and positions

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Build 2-D attn mask | `(B,241) → (B,241,241)`, bool | tiny | — | — | — | — | — | — | — | — | `1/chunk` | compare/scan | — | 0 |
| 2 | Position IDs | `(B,241) → (B,241)`, bool→int64 | tiny | — | — | — | — | — | — | — | — | `1/chunk` | cumsum | — | 0 |

### 3.2 One VLM prefill layer (eager attention — **not** Flash)

Every row runs `L_p = 16/chunk` unless noted. Softmax is a **standalone FP32 CUDA kernel** (`softmax_warp_forward<float>`, Inst/3=96 with odd). **FLOPs util / BD util are calculated** from FLOPs·IO·time (may exceed 100%). Installed path: **Q/K → FP32 before `QKᵀ`** (`magma_sgemmEx<float>`), scores FP32 through scale/`where`/softmax, **A → BF16** for `A·V` (`cutlass_75`). Differs from FLOP-doc bf16-`QKᵀ` assumption.

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Input RMSNorm | `(B,241,960) →` same, stats FP32 | **avg 4.13 µs** | 926.2k | 0.22 | 0.93% | 0.926 | 224.3 | 50.1% | 1.0 | bw-bound | `16/chunk` | rsqrt+mean/pow | — | 0 |
| 2 | **Q projection** Linear `960→960` | `(B,241,960) → (B,241,960)`, `bfloat16 → bfloat16` | **avg 20.94 µs** *[med]* | 443.75M | 21.21 | 22.4% | 2.77 | 132.2 | 29.5% | 160.4 | bw-bound | `16/chunk` | CUTLASS bf16 | n1=48 group | **16** |
| 3 | **K projection** Linear `960→320` | `(B,241,960) → (B,241,320)`, `bfloat16 → bfloat16` | **avg 11.36 µs** *[med]* | 148.12M | 13.01 | 13.7% | 1.23 | 108.3 | 24.2% | 120.2 | bw-bound | `16/chunk` | CUTLASS bf16 | n1=32=K+V | **16** |
| 4 | **V projection** Linear `960→320` | `(B,241,960) → (B,241,320)`, `bfloat16 → bfloat16` | **avg 11.38 µs** *[med]* | 148.12M | 13.01 | 13.7% | 1.23 | 108.3 | 24.2% | 120.2 | bw-bound | `16/chunk` | CUTLASS bf16 | same as K | **16** |
| 5 | Reshape heads | Q15 / K5 V5, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `3×16/chunk` | view | — | 0 |
| 6 | RoPE angles + sin/cos | → `(B,241,1,32)` per Q,K, FP32 | **avg 1.51 µs** | 962 | 0.0003 | 0.00% | 0.0078 | 2.5 | 0.56% | 0.1 | bw-bound | `sin×32+cos×32` | sin/cos/pow | n1=272 | 0 |
| 7 | RoPE apply on Q,K | head tensors, FP32→BF16 | **avg 1.84 µs** *[med]* | 28.9k | 0.02 | 0.08% | 0.463 | 251.0 | 56.0% | 0.1 | bw-bound | `2×16/chunk` | mul/add | epilogue on RoPE temps | 0 |
| 8 | Cache append K,V | → `(B,5,241,64)`, `bfloat16` | **avg 1.00 µs** *[med]* | 0 | — | — | 0.309 | 308.5 | 68.9% | 0 | bw-bound | `16/chunk` | copy | DRAM write to cache | 0 |
| 9 | Expand K,V 5→15 heads | `(B,241,5,64) → (B,241,15,64)`, `bfloat16 → bfloat16` | **avg 1.00 µs** *[med]* | 0 | — | — | 0.463 | 462.7 | 103.3% | 0 | bw-bound | `2×16/chunk` | expand | often view; no separate DRAM IO | 0 |
| 10 | Transpose Q,K | `(B,241,15,64) → (B,15,241,64)`, `bfloat16 → bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `2×16/chunk` | view | — | 0 |
| 11 | **Attn MatMul `QKᵀ`** | `(B,15,241,64)×(B,15,64,241) → (B,15,241,241)`, **`float32 → float32`** | **avg 10.61 µs** *[med]* | 110.62M | 10.41 | 43.9% | 5.34 | 503 | 112.3% | 20.7 | bw-bound | `16/chunk` | `magma_sgemmEx<float>` | Q,K cast FP32; Avg mixes 3 shapes — BD not a clean DRAM rate | **16** |
| 12 | Scale `÷8` | `(B,15,241,241) →` same, `float32 → float32` | **avg 1.17 µs** *[med]* | 871.2k | 0.73 | 3.1% | 6.97 | 5869.2 | 1310.1% | 0.1 | bw-bound | `16/chunk` | mul | **score epilogue** (L2 after `QKᵀ`); no DRAM IO | 0 |
| 13 | `where(mask, score, -∞)` | mask `bool` + scores, `bool/float32 → float32` | **avg 2.99 µs** *[med]* | 871.2k | 0.29 | 1.2% | 6.97 | 2323.2 | 518.6% | 0.1 | bw-bound | `16/chunk` | `where_kernel` | **score epilogue**; no DRAM IO | 0 |
| 14 | Softmax over keys | `(B,15,241,241) →` same, `float32 → float32` | **avg 2.86 µs** | 4.35M | 1.51 | 6.4% | 6.97 | 2424.3 | 541.1% | 0.6 | bw-bound | `16/chunk` | `softmax_warp_forward<float,(int)8>` | standalone FP32 kernel (Inst matches eager `QKᵀ`) | 0 |
| 15 | Cast attn probs for `A·V` | `(B,15,241,241) →` same, **`float32 → bfloat16`** | **avg 1.96 µs** *[med]* | 0 | — | — | 5.23 | 2668.4 | 595.6% | 0 | bw-bound | `16/chunk` | copy-cast | before BF16 `A·V` | 0 |
| 16 | **Attn MatMul `A·V`** | `(B,15,241,241)×(B,15,241,64) → (B,15,241,64)`, `bfloat16 → bfloat16` | **avg 7.44 µs** *[med]* | 111.25M | 14.96 | 15.8% | 2.67 | 358.7 | 80.1% | 41.7 | bw-bound | `16/chunk` | `cutlass_75` bf16 | n1=96 VLM+even | **16** |
| 17 | Concat heads | `(B,15,241,64) → (B,241,960)`, `bfloat16 → bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `16/chunk` | view | — | 0 |
| 18 | **Attn out Linear** `960→960` | `(B,241,960) → (B,241,960)`, `bfloat16 → bfloat16` | **avg 20.94 µs** *[med]* | 443.75M | 21.21 | 22.4% | 2.77 | 132.2 | 29.5% | 160.4 | bw-bound | `16/chunk` | CUTLASS | n1=48 | **16** |
| 19 | First residual add | two `(B,241,960) → (B,241,960)`, `bfloat16 → bfloat16` | **avg 0.92 µs** *[med]* | 231.2k | 0.25 | 1.1% | 1.39 | 1480.7 | 330.5% | 0.2 | bw-bound | `16/chunk` | BF16 add | epilogue / *[med]*; no DRAM BD | 0 |
| 20 | Post-attn RMSNorm | `(B,241,960) →` same, stats `float32`; act `bfloat16 → bfloat16` | **avg 4.13 µs** | 926.2k | 0.22 | 0.93% | 0.926 | 224.3 | 50.1% | 1.0 | bw-bound | `16/chunk` | rsqrt/mean | — | 0 |
| 21 | **MLP gate Linear** `960→2560` | `(B,241,960) → (B,241,2560)`, `bfloat16 → bfloat16` | **avg 15.51 µs** *[med]* | 1.18G | 76.38 | 80.6% | 6.61 | 426.6 | 95.2% | 179.1 | bw-bound | `16/chunk` | CUTLASS | n1=192 | **16** |
| 22 | **MLP up Linear** `960→2560` | `(B,241,960) → (B,241,2560)`, `bfloat16 → bfloat16` | **avg 15.50 µs** *[med]* | 1.18G | 76.38 | 80.6% | 6.61 | 426.6 | 95.2% | 179.1 | bw-bound | `16/chunk` | CUTLASS | same | **16** |
| 23 | SiLU on gate | `(B,241,2560) →` same, `bfloat16 → bfloat16` | **avg 1.38 µs** | 3.08M | 2.24 | 9.5% | 2.47 | 1794.8 | 400.6% | 1.2 | bw-bound | `16/chunk` | `silu_kernel<bf16>` | **MLP epilogue** on gate GEMM output | 0 |
| 24 | Gate ⊗ up | `2×(B,241,2560) → (B,241,2560)`, `bfloat16 → bfloat16` | **avg 1.84 µs** *[med]* | 616.9k | 0.34 | 1.4% | 3.7 | 2042.4 | 455.9% | 0.2 | bw-bound | `16/chunk` | mul | **MLP epilogue** | 0 |
| 25 | **MLP down Linear** `2560→960` | `(B,241,2560) → (B,241,960)`, `bfloat16 → bfloat16` | **avg 20.94 µs** *[med]* | 1.18G | 56.57 | 59.7% | 6.61 | 315.8 | 70.5% | 179.1 | bw-bound | `16/chunk` | CUTLASS | n1=48 | **16** |
| 26 | Second residual add | two `(B,241,960) → (B,241,960)`, `bfloat16 → bfloat16` | **avg 0.92 µs** *[med]* | 231.2k | 0.25 | 1.1% | 1.39 | 1480.7 | 330.5% | 0.2 | bw-bound | `16/chunk` | BF16 add | epilogue / *[med]* | 0 |

**Stage-2 Linear+attn GEMM budget /chunk:** 144. Attributed GEMM time ≈ **2.15 ms**. Installed `QKᵀ` is **FP32 magma**. Softmax/where share 176/chunk with Stage 3. Wall Stage 2 ≈ **7.2 ms**.

### 3.3 Prefill first layer — kernel launches in order (Nsight)

Moved to **`smolVLA_kernel_gpu_list.md`**: layer‑1’s **77** CUDA launches in order (Stage‑2 window **564.90→576.32 ms**), plus cross-layer op-list / duration comparison for all **16** prefill layers.

---

## 4. Stage 3 — expert decode + Euler (GPU)

**Function:** `M=10` Euler steps; each rebuilds suffix (§2) then 16 expert layers (8 even self-attn + 8 odd cross-attn).

### 4.1 Even expert layers — self-attention (`80/chunk = 8×10`)

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Expert input RMSNorm | `(B,50,720)`, stats FP32 | **avg 4.13 µs** | 144.1k | 0.03 | 0.13% | 0.144 | 34.9 | 7.8% | 1.0 | bw-bound | `80/chunk` | rsqrt/mean | — | 0 |
| 2 | **Q projection** Linear `720→960` | `(B,50,720) → (B,50,960)`, `bfloat16` | **avg 6.67 µs** *[med]* | 69.12M | 10.35 | 10.9% | 1.55 | 232.3 | 51.9% | 44.6 | bw-bound | `80/chunk` | CUTLASS | n1=480 | **80** |
| 3 | **K projection** Linear `720→320` | `(B,50,720) → (B,50,320)`, `bfloat16` | **avg 6.67 µs** *[med]* | 23.00M | 3.45 | 3.6% | 0.565 | 84.6 | 18.9% | 40.8 | bw-bound | `80/chunk` | CUTLASS | n1=480 | **80** |
| 4 | **V projection** Linear `720→320` | `(B,50,720) → (B,50,320)`, `bfloat16` | **avg 6.67 µs** *[med]* | 23.00M | 3.45 | 3.6% | 0.565 | 84.6 | 18.9% | 40.8 | bw-bound | `80/chunk` | CUTLASS | n1=480 | **80** |
| 5 | Reshape Q15 / K5 / V5 | heads, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `3×80/chunk` | view | — | 0 |
| 6 | RoPE on Q,K | same shapes, FP32→BF16 | **avg 1.51 µs** | 40 | 0.0000 | 0.00% | 0.0003 | 0.1 | 0.02% | 0.1 | bw-bound | `sin×160+cos×160` | sin/cos+mul/add | — | 0 |
| 7 | Cache append suffix K,V | → `(B,5,291,64)`, `bfloat16` | **avg 1.00 µs** *[med]* | 0 | — | — | 0.372 | 372.5 | 83.1% | 0 | bw-bound | `80/chunk` | copy | — | 0 |
| 8 | Expand K,V 5→15 | len 291, `bfloat16` | **avg 1.00 µs** *[med]* | 0 | — | — | 0.096 | 96.0 | 21.4% | 0 | bw-bound | `2×80/chunk` | expand | — | 0 |
| 9 | Cast/transpose Q,K | → FP32 scores path, BF16→FP32 | **avg 1.96 µs** *[med]* | 0 | — | — | 0.982 | 500.4 | 111.7% | 0 | bw-bound | `2×80/chunk` | copy-cast | cast into score path; no separate DRAM BD | 0 |
| 10 | **Attn MatMul `QKᵀ`** (+ scale) | → `(B,15,50,291)`, `float32` | **avg 10.61 µs** *[med]* | 27.75M | 2.61 | 11.0% | 2.18 | 205.6 | 45.9% | 12.7 | bw-bound | `80/chunk` | `magma_sgemmEx` | *[med]* mixed shapes | **80** |
| 11 | `where` mask | scores, FP32 | **avg 2.99 µs** *[med]* | 218.2k | 0.07 | 0.30% | 1.75 | 584.4 | 130.4% | 0.1 | bw-bound | `80/chunk` | `where_kernel` | **score epilogue** | 0 |
| 12 | Softmax | `(B,15,50,291)`, FP32 | **avg 2.79 µs** | 1.09M | 0.39 | 1.6% | 1.75 | 626.4 | 139.8% | 0.6 | bw-bound | `80/chunk` | `softmax_warp …(int)9` | **score epilogue** | 0 |
| 13 | **Attn MatMul `A·V`** | → `(B,15,50,64)`, `bfloat16` | **avg 7.44 µs** *[med]* | 27.88M | 3.75 | 4.0% | 1.09 | 146.7 | 32.7% | 25.6 | bw-bound | `80/chunk` | `cutlass_75` | n1=96 | **80** |
| 14 | Concat heads | → `(B,50,960)`, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `80/chunk` | view | — | 0 |

### 4.2 Odd expert layers — cross-attention to prefix (`80/chunk`)

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Expert input RMSNorm | `(B,50,720)`, stats FP32 | **avg 4.13 µs** | 144.1k | 0.03 | 0.13% | 0.144 | 34.9 | 7.8% | 1.0 | bw-bound | `80/chunk` | rsqrt/mean | — | 0 |
| 2 | **Q projection** Linear `720→960` | `(B,50,720) → (B,50,960)`, `bfloat16` | **avg 6.67 µs** *[med]* | 69.12M | 10.35 | 10.9% | 1.55 | 232.3 | 51.9% | 44.6 | bw-bound | `80/chunk` | CUTLASS | n1=480 | **80** |
| 3 | Read prefix K5/V5 | `(B,5,241,64) → (B,241,320)`, `bfloat16` | ≈0 | — | — | — | — | — | — | — | — | `2×80/chunk` | view | — | 0 |
| 4 | **Cross-K Linear** `320→320` | `(B,241,320) → (B,241,320)`, **FP32** | **avg 15.27 µs** *[med]* | 49.25M | 3.23 | 13.6% | 1.03 | 67.2 | 15.0% | 48.0 | bw-bound | `80/chunk` | `simt_sgemm_64x64_tn` | n1=180 | **80** |
| 5 | **Cross-V Linear** `320→320` | `(B,241,320) → (B,241,320)`, **FP32** | **avg 15.28 µs** *[med]* | 49.25M | 3.23 | 13.6% | 1.03 | 67.2 | 15.0% | 48.0 | bw-bound | `80/chunk` | same simt_tn | n1=180 | **80** |
| 6 | Reshape K5/V5 | → `(B,241,5,64)`, `float32` | ≈0 | — | — | — | — | — | — | — | — | `2×80/chunk` | view | — | 0 |
| 7 | RoPE on suffix Q only | Q `(B,50,15,64)`, FP32→BF16 | **avg 1.51 µs** | 40 | 0.0000 | 0.00% | 0.0004 | 0.1 | 0.02% | 0.1 | bw-bound | `sin×80+cos×80` | sin/cos | — | 0 |
| 8 | Expand K,V 5→15 | → `(B,241,15,64)`, `float32` | **avg 1.37 µs** *[med]* | 0 | — | — | 0.925 | 676.1 | 150.9% | 0 | bw-bound | `2×80/chunk` | expand | often view | 0 |
| 9 | **Attn MatMul `QKᵀ`** (+ scale) | → `(B,15,50,241)`, `float32` | **avg 10.61 µs** *[med]* | 23.00M | 2.16 | 9.1% | 1.84 | 173.4 | 38.7% | 12.5 | bw-bound | `80/chunk` | `magma_sgemmEx` | *[med]* | **80** |
| 10 | `where` + Softmax | scores `(B,15,50,241)`, FP32 | **avg 5.85 µs** | 1.08M | 0.19 | 0.80% | 2.89 | 494.4 | 110.4% | 0.4 | bw-bound | `80/chunk` each | where + softmax | **score epilogue** (sum of two kernels) | 0 |
| 11 | **Attn MatMul `A·V`** | → `(B,15,50,64)`, `float32` | **avg 9.48 µs** | 23.12M | 2.44 | 10.3% | 1.84 | 194.2 | 43.3% | 12.5 | bw-bound | `80/chunk` | `simt_sgemm_64x64_nn` | FP32 | **80** |
| 12 | Concat heads | → `(B,50,960)`, `float32` | ≈0 | — | — | — | — | — | — | — | — | `80/chunk` | view | — | 0 |

### 4.3 Common expert tail — out Linear + SwiGLU MLP (`160/chunk = 16×10`)

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Cast attn result to out-weight dtype | `(B,50,960)`, often FP32→BF16 | **avg 1.37 µs** *[med]* | 0 | — | — | 0.288 | 210.4 | 47.0% | 0 | bw-bound | `160/chunk` | copy-cast | — | 0 |
| 2 | **Attn out Linear** `960→720` | `(B,50,960) → (B,50,720)`, `bfloat16` | **avg 6.67 µs** *[med]* | 69.06M | 10.36 | 10.9% | 1.55 | 232.5 | 51.9% | 44.6 | bw-bound | `160/chunk` | CUTLASS | n1=480 | **160** |
| 3 | First residual add | two `(B,50,720)`, `bfloat16` | **avg 0.92 µs** *[med]* | 36.0k | 0.04 | 0.17% | 0.216 | 235.1 | 52.5% | 0.2 | bw-bound | `160/chunk` | BF16 add | — | 0 |
| 4 | Post-attn RMSNorm | `(B,50,720)`, stats FP32 | **avg 4.13 µs** | 144.1k | 0.03 | 0.13% | 0.144 | 34.9 | 7.8% | 1.0 | bw-bound | `160/chunk` | rsqrt/mean | — | 0 |
| 5 | **MLP gate Linear** `720→2048` | `(B,50,720) → (B,50,2048)`, `bfloat16` | **avg 12.01 µs** | 147.38M | 12.27 | 12.9% | 3.23 | 268.5 | 59.9% | 45.7 | bw-bound | `160/chunk` | `wmma …64x1` | n1=320 | **160** |
| 6 | **MLP up Linear** `720→2048` | `(B,50,720) → (B,50,2048)`, `bfloat16` | **avg 12.01 µs** | 147.38M | 12.27 | 12.9% | 3.23 | 268.5 | 59.9% | 45.7 | bw-bound | `160/chunk` | same as gate | — | **160** |
| 7 | SiLU on gate | `(B,50,2048)`, `bfloat16` | **avg 1.38 µs** | 512.0k | 0.37 | 1.6% | 0.41 | 296.5 | 66.2% | 1.2 | bw-bound | `160/chunk` | `silu_kernel` | **MLP epilogue** | 0 |
| 8 | Gate ⊗ up | `2×(B,50,2048)`, `bfloat16` | **avg 1.84 µs** *[med]* | 102.4k | 0.06 | 0.25% | 0.614 | 334.4 | 74.6% | 0.2 | bw-bound | `160/chunk` | mul | **MLP epilogue** | 0 |
| 9 | **MLP down Linear** `2048→720` | `(B,50,2048) → (B,50,720)`, `bfloat16` | **avg 15.51 µs** *[med]* | 147.44M | 9.50 | 10.0% | 3.23 | 208.0 | 46.4% | 45.7 | bw-bound | `160/chunk` | CUTLASS | n1=192 | **160** |
| 10 | Second residual add | two `(B,50,720)`, `bfloat16` | **avg 0.92 µs** *[med]* | 36.0k | 0.04 | 0.17% | 0.216 | 235.1 | 52.5% | 0.2 | bw-bound | `160/chunk` | BF16 add | — | 0 |

### 4.4 Velocity head + Euler update (`M/chunk`)

| Order | Operator | Input → output (shape, dtype) | GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Expert final RMSNorm | `(B,50,720)`, stats FP32 | **avg 4.13 µs** | 144.0k | 0.04 | 0.17% | 0.144 | 35.1 | 7.8% | 1.0 | bw-bound | `M/chunk` | rsqrt/mean | — | 0 |
| 2 | Cast expert out | `(B,50,720)`, **BF16→FP32** | **avg 1.96 µs** *[med]* | 0 | — | — | 0.216 | 108.0 | 24.1% | 0 | bw-bound | `M/chunk` | copy-cast | — | 0 |
| 3 | **Action / velocity Linear** `720→32` | `(B,50,720) → (B,50,32)`, `float32` | **avg 15.27 µs** *[med]* | 2.30M | 0.15 | 0.63% | 0.243 | 15.9 | 3.5% | 9.5 | bw-bound | `M/chunk` | `simt_sgemm_64x64_tn` | n1=180 | **10** |
| 4 | Euler `x ← x + dt·v` | `(B,50,32)`, `float32` | tiny | — | — | — | — | — | — | — | — | `M/chunk` | add/mul | — | 0 |
| 5 | `cache.crop(241)` | `(B,5,291,64)→(B,5,241,64)`, `bfloat16` | tiny | — | — | — | — | — | — | — | — | `M/chunk` | slice | — | 0 |

**Stage-3 Linear-only GEMM budget /chunk (excl. attn QK/AV):** **1130**.

---

## 5. Stage 4 — crop / queue / postprocess (GPU)

| Order | Operator | Input → output shape | Input → output dtype | Repeat | GPU kernel(s) | Fusion / dtype note | GEMM launches /chunk | GPU time |
|---:|---|---|---|---:|---|---|---:|---|
| 1 | Crop action dim | `(B,50,32) → (B,50,6)` | `float32` | `1/chunk` | slice/view | — | 0 | ≪1 ms |
| 2 | Queue + pop one step | `(B,50,6) → (B,6)` | `float32` | `1/chunk` | view | — | 0 | tiny |
| 3 | Unnormalize postprocess | `(B,6) → (B,6)` | `float32` | `1/chunk` | mul/add | — | 0 | Stage4 wall ~0.05 ms |

---

## 5b. Per-call GEMM inventory (timed)

| Stage | Operator | Weight / matmul shape (K→N), activation leading dims | Launches /chunk | Dtype | GPU time /chunk |
|---|---|---|---:|---|---:|
| 0 | ViT Q / K / V | `768→768`, act `(B,1024,*)` | 36 each | BF16 | **1.141 ms each** |
| 0 | ViT attn out | `768→768`, `(B,1024,*)` | 36 | BF16 | **1.141 ms** |
| 0 | ViT MLP up `fc1` | `768→3072`, `(B,1024,*)` | 36 | BF16 | **4.313 ms** |
| 0 | ViT MLP down `fc2` | `3072→768`, `(B,1024,*)` | 36 | BF16 | **5.472 ms** |
| 0 | Connector | `12288→960`, `(B,64,*)` | 3 | BF16 | **0.189 ms** |
| 0 | State Linear | `32→960`, `(B,*)` | 1 | FP32 | **0.002 ms** |
| 1 | Action-in | `32→720`, `(B,50,*)` | 10 | FP32 | **0.031 ms** |
| 1 | Fusion mid | `1440→720`, `(B,50,*)` | 10 | FP32 | **0.391 ms** |
| 1 | Fusion out | `720→720`, `(B,50,*)` | 10 | FP32 | **0.153 ms** *[med]* |
| 2 | VLM Q | `960→960`, `(B,241,*)` | 16 | BF16 | **0.335 ms** *[med]* |
| 2 | VLM K / V | `960→320`, `(B,241,*)` | 16 each | BF16 | **0.182 ms each** *[med]* |
| 2 | VLM attn out | `960→960`, `(B,241,*)` | 16 | BF16 | **0.335 ms** *[med]* |
| 2 | VLM MLP gate / up | `960→2560`, `(B,241,*)` | 16 each | BF16 | **0.248 ms each** *[med]* |
| 2 | VLM MLP down | `2560→960`, `(B,241,*)` | 16 | BF16 | **0.335 ms** *[med]* |
| 2 | Prefill `QKᵀ` | scores `(B,15,241,241)` | 16 | **FP32** | **0.170 ms** *[med]* |
| 2 | Prefill `A·V` | `(B,15,241,*)` | 16 | BF16 | **0.119 ms** *[med]* |
| 3e | Expert Q | `720→960`, `(B,50,*)` | 80 | BF16 | **0.534 ms** *[med]* |
| 3e | Expert K / V | `720→320`, `(B,50,*)` | 80 each | BF16 | **0.534 ms each** *[med]* |
| 3e | Even `QKᵀ` | scores `(B,15,50,291)` | 80 | FP32 | **0.849 ms** *[med]* |
| 3e | Even `A·V` | `(B,15,50,*)` | 80 | BF16 | **0.595 ms** *[med]* (`cutlass_75`) |
| 3o | Expert Q | `720→960`, `(B,50,*)` | 80 | BF16 | **0.534 ms** *[med]* |
| 3o | Cross-K / Cross-V | `320→320`, `(B,241,*)` | 80 each | **FP32** | **1.221 ms each** *[med]* |
| 3o | Odd `QKᵀ` | scores `(B,15,50,241)` | 80 | FP32 | **0.849 ms** *[med]* |
| 3o | Odd `A·V` | `(B,15,50,*)` | 80 | FP32 | **0.758 ms** (`simt_nn`) |
| 3 | Attn out | `960→720`, `(B,50,*)` | 160 | BF16 | **1.067 ms** *[med]* |
| 3 | MLP gate / up | `720→2048`, `(B,50,*)` | 160 each | BF16 | **1.922 ms each** |
| 3 | MLP down | `2048→720`, `(B,50,*)` | 160 | BF16 | **2.482 ms** *[med]* |
| 3 | Velocity head | `720→32`, `(B,50,*)` | 10 | FP32 | **0.153 ms** *[med]* |

---

## 6. Compact GPU kernel ledger (CSV top rows)

Primary evidence table — times are **session** totals from `smolvla_nsys_cuda_gpu_kern_sum.csv`.

| Rank | Time % | ms | Inst | Avg µs | Kernel (short) | Logical role |
|---:|---:|---:|---:|---:|---|---|
| 1 | 9.5 | 16.415 | 108 | 152.0 | `cutlass … s16816gemm_relu … 256x64` | BF16 Linear/GEMM (CUTLASS TC) |
| 2 | 8.0 | 13.693 | 432 | 31.7 | `cutlass … s16816gemm_relu … 64x64` | BF16 Linear/GEMM |
| 3 | 7.5 | 12.939 | 108 | 119.8 | `cutlass … s16816gemm_relu … 256x128` | BF16 Linear/GEMM |
| 4 | 6.7 | 11.534 | 960 | 12.0 | `cutlass … wmma … s161616gemm … 32x32_64` | BF16 WMMA GEMM |
| 5 | 5.6 | 9.604 | 1440 | 6.7 | `cutlass … wmma … 32x32_128` | BF16 WMMA GEMM |
| 6 | 5.2 | 8.934 | 576 | 15.5 | `cutlass … s16816gemm_relu … 128x64` | BF16 Linear/GEMM |
| 7 | 5.2 | 8.894 | 108 | 82.4 | **`pytorch_flash::flash_fwd_kernel`** | **Fused ViT attention** |
| 8 | 4.8 | 8.243 | 540 | 15.3 | `cutlass … simt_sgemm … tn` | FP32 SIMT GEMM |
| 9 | 4.6 | 7.973 | 4074 | 2.0 | `direct_copy` + **LoadWithCast/StoreWithCast** | **Dtype cast / copy** |
| 10 | 4.0 | 6.848 | 1107 | 6.2 | `direct_copy` BF16 path | Cast/copy |
| 11 | 3.3 | 5.602 | 528 | 10.6 | `magma_sgemmEx_kernel<float,… cublasLtEpilogue>` | FP32 cuBLASLt GEMM |
| 12–14 | ~8.1 | ~13.9 | many | — | mul / add / copy elementwise | Residuals, scales, broadcasts |
| 15 | 1.8 | 3.016 | 144 | 20.9 | cutlass s16816gemm 128×64_64 | BF16 GEMM |
| 16 | 1.5 | 2.519 | 1089 | 2.3 | `ReduceOp<…MeanOps>` | Norm / mean reductions |
| 21 | 1.1 | 1.816 | 225 | 8.1 | **`vectorized_layer_norm_kernel<BF16,float>`** | ViT LayerNorm |
| 22 | 1.0 | 1.795 | 9 | 199.4 | **`implicit_convolve_sgemm<__nv_bfloat16>`** | Patch Conv2d |
| 34 | 0.6 | 1.055 | 108 | 9.8 | **`GeluCUDAKernelImpl`** | ViT MLP GELU |
| 40–42 | 0.9 | 1.493 | 528 | ~2.8 | **`softmax_warp_forward<float>`** | VLM+expert softmax |
| 49 | 0.1 | 0.250 | 9 | 27.8 | **`upsample_bilinear2d_out_frame<float>`** | Image resize |

Full 97-row dump: `doc/nsight/smolvla_nsys_cuda_gpu_kern_sum.csv`.

---

## 7. Fusions, downcasts, and other transformations (Nsight-confirmed)

### 7.1 Fused operators (algorithmic many → GPU few)

| Fusion | Replaces (FLOP doc) | Evidence |
|---|---|---|
| **FlashAttention forward** | ViT `QKᵀ` + scale + mask + softmax + `A·V` | Single `flash_fwd_kernel`; softmax Inst exclude ViT |
| **CUTLASS GEMM epilogue** | Linear (+ optional bias / relu-family epilogue) | `cutlass_80_tensorop_bf16_*gemm_relu_*` |
| **LayerNorm vectorized** | mean + var + rsqrt + affine | One `vectorized_layer_norm_kernel` |
| **cuBLASLt epilogue / splitK reduce** | GEMM + bias/scale reduce | `cublasLt::splitKreduce_kernel`, `cublasLtEpilogue_t` |

**Not fused (still separate kernels):** GELU, SiLU, sin/cos, where-mask, many residual adds, dtype casts.

### 7.2 Downcasts / upcasts / promotions

| Transform | Where | Kernel evidence |
|---|---|---|
| **FP32 → BF16** | images into ViT; many activations into bf16 Linears | `direct_copy` + `LoadWithCast` / `StoreWithCast`; `bfloat16_copy_kernel` |
| **BF16 → FP32** | prefix `cat` with FP32 state token; attention softmax path | `CatArrayBatchedCopy`; softmax `<float>` |
| **FP64 → FP32** | Stage-1 sinusoidal embedding cast | FP64 unary kernels + copy/cast |
| **BF16 compute, FP32 accumulate (implicit)** | Tensor Core GEMM | CUTLASS `bf16` tensorop/WMMA names |
| **BF16 act, FP32 LN stats** | ViT LayerNorm | `vectorized_layer_norm_kernel<c10::BFloat16, float>` |

### 7.3 Other transformations worth tracking

| Transform | GPU signal |
|---|---|
| Bilinear resize | `upsample_bilinear2d_out_frame` |
| Im2col-style patch embed | `implicit_convolve_sgemm` |
| Masked attention fill | `where_kernel` (−inf path) before softmax |
| RoPE / time sinusoids | `sin` + `cos` + `pow` + `rsqrt` |
| Contiguous / layout fixups | high volume of `direct_copy` / `elementwise` copies (~30% of session time!) |

**Bandwidth lesson:** copy/cast elementwise is **~30% of measured GPU kernel time** in this capture — invisible in pure FLOP tables, critical for GPU perf.

---

## 8. Cross-stage GPU comparison

| Stage | Wall ms (profile) | Signature GPU kernels (verified counts) | Main dtype on GPU |
|---|---:|---|---|
| 0 Prefix | 23.9 | upsample×3, conv×3, **Flash×36**, LN×75, GELU×36, CUTLASS bf16 Linears | BF16 (+ FP32 resize/state) |
| 1 Suffix | ~2 (inside Euler) | FP32 GEMM, SiLU, sin/cos/pow | FP32 / transient FP64 |
| 2 Prefill | 7.2 | CUTLASS bf16, **softmax + where** (part of 176/chunk) | BF16 + FP32 attn |
| 3 Expert×M | ~57 | CUTLASS + cuBLASLt FP32 odd paths, softmax/where rest of 176 | Mixed BF16/FP32 |
| 4 Post | ~0.05 | tiny elementwise | FP32 |

Bottleneck reading on this GPU:

1. **BF16 CUTLASS GEMMs** (~half+ of kernel time) — MLP/Linear walls.
2. **Cast/copy traffic** (~30%) — mixed-precision tax.
3. **Flash ViT attn** (~5% session / ~3 ms per chunk) — heavy arithmetically in FLOP doc, cheaper than naive softmax on GPU.
4. **Standalone softmax** (~0.5 ms/chunk) — only VLM+expert.

---

## 9. Code-source map (unchanged modules, GPU dispatch)

| Stage | Python / module | GPU backends observed |
|---|---|---|
| 0 images | `SmolVLAPolicy.prepare_images`, `resize_with_pad` | ATen upsample, cast |
| 0 ViT | `SmolVLMVisionEmbeddings`, encoder layers | cuDNN implicit conv, FlashAttn, CUTLASS, LN, GELU |
| 0 connector / lang / state | `SmolVLMConnector`, embed_language, `state_proj` | CUTLASS / cuBLASLt, gather, cat |
| 1 suffix | `embed_suffix`, `create_sinusoidal_pos_embedding` | ATen trig/pow, FP32 GEMM, SiLU |
| 2–3 | `SmolVLMWithExpertModel.forward`, expert layers | CUTLASS, cuBLASLt, softmax_warp, where |
| 4 | crop + LeRobot postprocessor | tiny ATen ops |

Profiler entry: `src/smolvla_nsight_target.py` → `profile_stage_timings` in `src/smolvla_test_infer.py`.

---

## 10. Method limits / how to refine further

1. **No NVTX stage ranges** → some GEMM rows still group **different weight shapes** that happened to pick the same CUTLASS tile (marked *[med]*). Times use `Avg × this_op_launches`; swap within a group if NVTX shows otherwise.
2. **Nsight Compute (`ncu`)** failed on driver 13.1 / Blackwell → no SOL/roofline per kernel yet.
3. `/3` per-chunk scaling assumes every listed launch is inference-only and equally present in each of the three fills; load-only kernels (rare `n=3`) are called out separately.
4. CUTLASS symbols named `*_relu_*` indicate an **epilogue template family**; do not read them as “ReLU activation after every Linear” without checking the actual epilogue args.
5. Elementwise helpers without a unique Inst match (tiny muls, copy-cast) use the pool’s **CSV Avg × expected launches** for that op — marked *[med]*.

---

## 11. CSV launch → operator attribution (per I/O dims)

Rule used throughout: **`time_op = CSV_Avg_µs × launches_op / chunk`**, even when Nsight merges equal-tile launches into one row.

### 11.1 High-confidence (Inst/3 uniquely matches op or equal-shape group)

| CSV kernel (short) | Inst/3 | Avg µs | Attributed ops (launches each) | Time each /chunk |
|---|---:|---:|---|---:|
| `…s16816gemm_relu…64x64_32x6` | 144 | 31.70 | ViT Q, K, V, out (36 each; all `768→768`) | **1.141 ms** |
| `…s16816gemm_relu…256x128` | 36 | 119.80 | ViT MLP up `768→3072` | **4.313 ms** |
| `…s16816gemm_relu…256x64` | 36 | 151.99 | ViT MLP down `3072→768` | **5.472 ms** |
| `flash_fwd_kernel` | 36 | 82.35 | ViT fused attn | **2.965 ms** |
| `GeluCUDAKernelImpl` | 36 | 9.77 | ViT GELU | **0.352 ms** |
| `vectorized_layer_norm` | 75 | 8.07 | ViT LN (36+36+3) | **0.605 ms total** |
| `implicit_convolve…bf16` | 3 | 199.43 | Patch Conv2d | **0.598 ms** |
| `upsample_bilinear2d` | 3 | 27.81 | Image upsample | **0.083 ms** |
| `…s16816gemm_bf16_256x64` (no relu) | 3 | 63.16 | Connector `12288→960` | **0.189 ms** |
| `gemvx` | 1 | 2.23 | State Linear `32→960` | **0.002 ms** |
| `sgemm_largek_lds64` | 10 | 39.14 | Fusion mid `1440→720` | **0.391 ms** |
| `simt_sgemm_32x128` | 10 | 3.10 | Action-in `32→720` | **0.031 ms** |
| `silu_kernel<float>` | 10 | 1.18 | Stage-1 SiLU | **0.012 ms** |
| `silu_kernel<bf16>` | 176 | 1.38 | VLM SiLU×16 + expert SiLU×160 | 0.022 / 0.221 |
| `wmma …32x32_64x1` | 320 | 12.01 | Expert gate+up `720→2048` (160 each) | **1.922 ms each** |
| `magma_sgemmEx<float>` | 176 | 10.61 | All eager `QKᵀ` (FP32): VLM16+even80+odd80 *[med — mixed shapes]* | 0.170 / 0.849 / 0.849 |
| `where_kernel` | 176 | 2.99 | VLM16 + expert160 *[med — mixed score volumes]* | 0.048 / 0.239(×2) |
| `softmax_warp …(int)9` | 80 | 2.79 | Expert even softmax | **0.223 ms** |
| `softmax_warp …(int)8` | 96 | 2.86 | VLM16 + expert odd80 | 0.046 / 0.229 |
| `CUDAFunctor_add<BF16>` (large) | 78 | 20.89 | ViT residual×2 (36) + patch+pos (3) + **3 unexplained** *[med]* | **0.752 / 0.063** |
| float `sin` / `cos` | 272 each | 1.51 | RoPE: VLM Q+K (32) + even Q+K (160) + odd Q (80) = **272** | Stage2 **0.097** / even **0.483** / odd **0.242** (sin+cos) |
| mean+rsqrt+pow_scalar | 363 each | ~4.14 sum | Every RMSNorm: VLM in/out 32 + **final prefix 1** + expert 320 + velocity-norm 10 = **363** | **~4.14 µs × N** |

### 11.2 Medium-confidence (Inst/3 matches a mixed equal-tile group — same Avg for each member)

| CSV kernel (short) | Inst/3 | Avg µs | Group members (I/O dims → launches) | Time /chunk |
|---|---:|---:|---|---|
| `simt_sgemm_64x64_tn` | 180 | 15.27 | Fusion-out×10 + velocity×10 + cross-K×80 + cross-V×80 | 0.153 / 0.153 / **1.221** / **1.221** |
| `wmma …32x32_64x2` | 32 | 11.36 | VLM K+V `960→320` (16 each) | **0.182 ms each** |
| `…s16816gemm_relu…128x64_64x4` | 48 | 20.94 | VLM Q + out + MLP down (16 each; `960→960` / `2560→960`) | **0.335 ms each** |
| `…s16816gemm_relu…128x64_32x6` | 192 | 15.51 | VLM gate+up×16 + expert MLP down×160 | 0.248 / 0.248 / **2.482** |
| `wmma …32x32_128x2` | 480 | 6.67 | Expert Q(e+o)×160 + K×80 + V×80 + out×160 | 1.067 / 0.534 / 0.534 / 1.067 |
| `simt_sgemm_64x64_nn` | 80 | 9.48 | Expert **odd** `A·V` ×80 (**FP32**) | **0.758 ms** |
| `cutlass_75 …64x64_nn` | 96 | 7.44 | VLM `A·V`×16 + expert **even** `A·V`×80 (**BF16**) | 0.119 / **0.595** |

Within each row, every member uses the **same Avg**; only launch counts differ by I/O case. If NVTX later reassigns members inside a group, keep Avg and move the launch budgets.

### 11.3 Confidence legend

| Tag | Meaning |
|---|---|
| (none) / high | Inst/3 uniquely identifies the op or a single equal-shape family |
| *[med]* | Op is one of several shapes that share one CSV kernel name; time = Avg × this op’s launches |
| *[low]* | Fallback Avg from a large elementwise pool without a clean count match |

---


## 11.4 Verification errata (2026-09-23 re-check vs CSV + Operator_List + profiling)

Cross-checked against `smolvla_nsys_cuda_gpu_kern_sum.csv` (same capture as `smolvla_nsys.nsys-rep`; `nsys` CLI not available in this workspace), `SmolVLA_Operator_List.md`, and `smolVLA_profiling.md`.

| Fixed | Was | Now |
|---|---|---|
| Even vs odd `A·V` kernels | even→`simt_nn`, odd→`cutlass_75` (dtype mismatch) | even+VLM→**`cutlass_75` BF16**; odd→**`simt_nn` FP32** |
| Prefill `QKᵀ` dtype | bf16 / “cast scores” story | **FP32 `magma`** (installed Q/K cast); FLOP-doc bf16-QK is analysis-only |
| RoPE FP32 `sin`/`cos` launches | attributed 176 | **272** = VLM32 + even160 + odd80 |
| Prefix `cat` timing | bf16 OpaqueType2 Avg | **not uniquely timed** (output is FP32) |
| RMSNorm helper count | ≈362 | **363** (+ final prefix RMSNorm) |
| magma / where confidence | high | **[med]** (mixed shapes in one CSV row) |
| Prefill attn dtypes (rows 9–16) | incomplete / FLOP-doc mix | **Installed:** expand/transpose BF16; `QKᵀ` **FP32**; score epilogue FP32; `A·V` **BF16** |
| FLOPs / BD util | capped / cleared epilogue | **Calculated uncapped** `TFLOPS÷peak` and `BD÷448` (may exceed 100%) |

**Still OK without change:** ViT Q/K/V/out/MLP/Flash/GELU/LN/conv/upsample/connector/`gemvx`/fusion-mid/expert gate+up counts & Avgs; softmax `(int)9`↔even(291) vs `(int)8`↔VLM+odd(241); Stage wall clocks vs profiling.md; `prefix_embs`/`suffix_embs`/`v_t` FP32.


## 12. Quick answers

**Does matmul use open-source CUTLASS?**  
**Yes, predominantly** — top time is `cutlass::Kernel2<…>`. Also cuBLASLt (FP32) and FlashAttention (fused attn, CUTLASS types).

**Are there more fused ops / downcasts than the FLOP list?**  
**Yes** — FlashAttention, GEMM epilogues, fused LayerNorm; heavy **FP32↔BF16** copy-cast; Stage-1 **FP64→FP32**; prefix **BF16/FP32→FP32** cat.

**Are operators verified against the profiling file?**  
**Yes** for count-sensitive ops (Flash 36/chunk, LN 75/chunk, GELU 36/chunk, softmax+where 176/chunk, upsample/conv 3/chunk). GEMM **names** are CUTLASS/cuBLASLt; per-op times in mixed-tile groups remain *[med]* until NVTX. Prefill `QKᵀ` is **FP32 magma** on the installed path (not the FLOP-doc bf16-QK assumption).

---

## 13. FLOPs / bandwidth columns (in Stage 0–3 tables)

Stage **0–3** tables lead with **GPU time per launch | FLOPs | TFLOPS/s | FLOPs util | IO Volume/MB | BD GB/s | BD util | Arithmetic Intensity | Bound** (all per launch / one Repeat). Multiply by **Repeat** for per-chunk totals. Counting rules: §0.4.

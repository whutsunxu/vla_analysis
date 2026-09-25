# SmolVLA GPU kernel list — **compile mode** (from `smolvla_nsys_compile.nsys-rep`)

## 0. Source and time resolution

| Item | Value |
|---|---|
| Report | `doc/gpu/compile_mode/nsight/smolvla_nsys_compile.nsys-rep` |
| Extract | Nsight Systems **2025.1.3** `nsys export --type=sqlite` → `CUPTI_ACTIVITY_KIND_KERNEL` |
| Kernel column | **Full** CUPTI `demangledName` (untruncated) |
| Native timestamps | **nanoseconds** (`start`, `end`) |
| Per-launch GPU time | `(end - start)` ns → **µs** (`÷ 1000`), printed to **0.01 µs** |
| Compile path | `torch.compile` **`reduce-overhead`** on `select_action` (`src/smolvla_nsight_target_compile.py`) |
| Companion IR | `SmolVLA_Fused_CompileOp_List_gpu_backend.md` (Inductor fused launches) |
| Eager counterpart | `doc/gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md` |
| Measured chunk | steady `select_action` **#4** (landmark `triton_per_fused_sum_0` @ **19.587961 s** absolute) |

**Promise:** GPU times are CUPTI `(end-start)` from the `.nsys-rep` export. I/O / FLOPs columns are filled when shapes are known from the fused IR / FX list; otherwise tagged `—` / *[inferred]*. `reduce-overhead` uses **CUDA graphs**; this capture used `--cuda-graph-trace=node` so individual kernels inside graphs appear.

### 0.1 Platform metrics (RTX 5060 Ti — theoretical peaks)

Same sheet as eager kerne_list §0.1: **36** SMs, **448 GB/s** GDDR7, BF16 TC **~94.8 TFLOP/s**, FP32 CUDA **23.7 TFLOP/s**.

| Resource | Spec |
|---|---|
| SMs / CUDA cores / Tensor Cores | **36** / **4608** / **144** |
| Memory | GDDR7 → **448 GB/s** |
| BF16 TC peak (dense) | **~94.8 TFLOP/s** |
| FP32 CUDA peak | **23.7 TFLOP/s** |

### 0.2 Absolute timeline (Nsight GUI / CUPTI `start`)

`t0` = first CUPTI kernel = **8.761588 s**. Rel. ms = `(start − t0) / 1e6`.

| Absolute window | What is running |
|---|---|
| **8.762 → ~10.7 s** | Cold `torch.compile` / Inductor / CUDA-graph capture + early warmup (host-heavy; sparse GPU). |
| **19.271447 → 19.405038 s** | `select_action` chunk **#0** — wall **133.59 ms**, busy Σ **79.98 ms**, **4741** launches |
| **19.433590 → 19.484804 s** | `select_action` chunk **#1** — wall **51.21 ms**, busy Σ **43.79 ms**, **4101** launches |
| **19.487493 → 19.535699 s** | `select_action` chunk **#2** — wall **48.21 ms**, busy Σ **43.77 ms**, **4101** launches |
| **19.537857 → 19.585895 s** | `select_action` chunk **#3** — wall **48.04 ms**, busy Σ **43.79 ms**, **4101** launches |
| **19.587961 → 19.635992 s** | `select_action` chunk **#4** — wall **48.03 ms**, busy Σ **43.78 ms**, **4101** launches **← measured** |
| **19.637544 → 19.659117 s** | `select_action` chunk **#5** — wall **21.57 ms**, busy Σ **20.09 ms**, **3610** launches |

**GUI tip:** zoom to **~19.59–19.64 s** for the tabulated steady chunk.

### 0.3 How this differs from eager kerne_list

| Eager | Compile (`reduce-overhead`) |
|---|---|
| Many tiny ATen elementwise / RMSNorm / RoPE launches | Triton `triton_*_fused_*` merges those |
| Softmax / Flash as separate templates | Often FMHA CUTLASS + fused pre/post Triton |
| Stage 0–4 walls from eager landmarks | Same semantic stages, but landmarks shift to fused names |
| ~13.8k launches / ~129 ms wall (eager chunk #3) | This steady chunk: **4101** launches / **48.0 ms** wall / **43.8 ms** busy |

---

## 1. Chunk durations (all `select_action` windows)

| Chunk | Wall-span (ms) | Busy Σ (ms) | Gap (ms) | Gap ratio | # launches | Absolute window (s) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 133.591 | 79.979 | 53.612 | **40.1%** | 4741 | `19.271447→19.405038` |
| 1 | 51.214 | 43.790 | 7.424 | **14.5%** | 4101 | `19.433590→19.484804` |
| 2 | 48.206 | 43.775 | 4.431 | **9.2%** | 4101 | `19.487493→19.535699` |
| 3 | 48.038 | 43.786 | 4.251 | **8.8%** | 4101 | `19.537857→19.585895` |
| 4 ← meas. | 48.031 | 43.780 | 4.250 | **8.8%** | 4101 | `19.587961→19.635992` |
| 5 | 21.573 | 20.094 | 1.479 | **6.9%** | 3610 | `19.637544→19.659117` |

Measured analysis below uses chunk **#4** (wall 48.03 ms ≈ host `select_action` ~50 ms).

## 2. Measured chunk — kernel mix (by busy time)

Busy Σ **43.78 ms** = **43.78 ms**; **4101** launches; **112** distinct names.

| Busy % | Busy Σ (µs) | # launches | Avg µs | Operator / kernel family |
|---:|---:|---:|---:|---|
| 60.8 | 26624.4 | 1381 | 19.28 | `Linear / GEMM BF16 (CUTLASS TC)` |
| 15.7 | 6855.8 | 36 | 190.44 | `Mem-eff / FlashAttention (CUTLASS FMHA)` |
| 4.2 | 1818.5 | 175 | 10.39 | `Attn matmul FP32 (MAGMA sgemmEx)` |
| 4.1 | 1799.2 | 280 | 6.43 | `Linear / GEMM (CUTLASS)` |
| 2.5 | 1081.5 | 32 | 33.80 | `Other GPU kernel` |
| 1.0 | 450.8 | 204 | 2.21 | `cuBLAS / cuBLASLt GEMM or epilogue` |
| 0.8 | 372.1 | 36 | 10.34 | `triton_poi_fused_gelu_view_7` |
| 0.8 | 330.4 | 80 | 4.13 | `triton_per_fused__softmax__to_copy_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_14` |
| 0.7 | 308.5 | 80 | 3.86 | `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_29` |
| 0.7 | 297.3 | 80 | 3.72 | `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_10` |
| 0.6 | 276.0 | 80 | 3.45 | `triton_per_fused__softmax_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_slice_sub_unsqueeze_view_where_31` |
| 0.5 | 228.5 | 160 | 1.43 | `triton_poi_fused__unsafe_view_clone_expand_unsqueeze_view_25` |
| 0.5 | 205.0 | 15 | 13.67 | `triton_red_fused__softmax__to_copy_bitwise_and_exp_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_19` |
| 0.4 | 173.9 | 160 | 1.09 | `triton_poi_fused__unsafe_view_mul_silu_27` |
| 0.4 | 169.1 | 80 | 2.11 | `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_11` |
| 0.3 | 116.6 | 15 | 7.77 | `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_div_mul_pow_sin_slice_split_sub_unsqueeze_view_18` |
| 0.2 | 89.3 | 80 | 1.12 | `triton_poi_fused__to_copy__unsafe_view_transpose_view_24` |
| 0.2 | 86.4 | 3 | 28.81 | `Bilinear upsample` |
| 0.2 | 85.2 | 15 | 5.68 | `triton_per_fused_add_native_layer_norm_view_13` |
| 0.2 | 83.9 | 12 | 6.99 | `triton_per_fused_add_native_layer_norm_view_14` |
| 0.2 | 82.8 | 70 | 1.18 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_42` |
| 0.2 | 78.0 | 70 | 1.11 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_40` |
| 0.2 | 78.0 | 70 | 1.11 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_41` |
| 0.2 | 72.8 | 70 | 1.04 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_34` |
| 0.2 | 71.5 | 15 | 4.76 | `triton_per_fused_add_native_layer_norm_view_12` |
| 0.2 | 70.5 | 80 | 0.88 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 0.2 | 70.4 | 9 | 7.82 | `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_expand_ge_index_new_ones_scalar_tensor_transpose_unsqueeze_view_where_3` |
| 0.2 | 67.7 | 80 | 0.85 | `triton_poi_fused__to_copy__unsafe_view_clone_permute_view_32` |
| 0.1 | 65.5 | 3 | 21.85 | `triton_poi_fused_add_embedding_0` |
| 0.1 | 58.3 | 15 | 3.89 | `triton_per_fused_add_native_layer_norm_view_11` |
| 0.1 | 56.0 | 80 | 0.70 | `triton_poi_fused_clone_permute_view_15` |
| 0.1 | 54.7 | 3 | 18.24 | `triton_red_fused_add_native_layer_norm_view_10` |
| 0.1 | 48.1 | 1 | 48.13 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_63` |
| 0.1 | 48.1 | 1 | 48.13 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_67` |
| 0.1 | 48.1 | 1 | 48.10 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_69` |
| 0.1 | 48.1 | 1 | 48.06 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_4` |
| 0.1 | 48.1 | 1 | 48.06 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_60` |
| 0.1 | 48.1 | 24 | 2.00 | `triton_poi_fused__unsafe_view_cat_clone_expand_slice_transpose_unsqueeze_view_58` |
| 0.1 | 48.0 | 1 | 48.03 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_68` |
| 0.1 | 48.0 | 1 | 48.03 | `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_52` |

## 3. Measured chunk — launching order (first 80)

Calling order from CUPTI `start`. Cross-ref fused IR: `SmolVLA_Fused_CompileOp_List_gpu_backend.md` graph #8 often starts with `triton_per_fused_sum_0` (mask reduces) before RoPE / attn fusions.

| Order | Operator | GPU time | Rel. ms | Absolute s | Kernel |
|---:|---|---:|---:|---:|---|
| 1 | Triton fused `triton_per_fused_sum_0` | **1.02 µs** | 10826.373 | 19.587961 | `triton_per_fused_sum_0` |
| 2 | Triton fused `triton_per_fused_sum_0` | **0.99 µs** | 10826.375 | 19.587963 | `triton_per_fused_sum_0` |
| 3 | Triton fused `triton_per_fused_sum_0` | **0.99 µs** | 10826.376 | 19.587964 | `triton_per_fused_sum_0` |
| 4 | Triton fused `triton_per_fused_sum_0` | **0.96 µs** | 10826.378 | 19.587965 | `triton_per_fused_sum_0` |
| 5 | Triton fused `triton_per_fused_sum_0` | **0.83 µs** | 10826.379 | 19.587967 | `triton_per_fused_sum_0` |
| 6 | Triton fused `triton_per_fused_sum_0` | **0.99 µs** | 10826.380 | 19.587968 | `triton_per_fused_sum_0` |
| 7 | Triton fused `triton_per_fused_sum_0` | **0.90 µs** | 10826.382 | 19.587969 | `triton_per_fused_sum_0` |
| 8 | Triton fused `triton_per_fused_sum_0` | **0.83 µs** | 10826.383 | 19.587971 | `triton_per_fused_sum_0` |
| 9 | Triton fused `triton_per_fused_sum_0` | **0.99 µs** | 10826.384 | 19.587972 | `triton_per_fused_sum_0` |
| 10 | Triton fused `triton_per_fused_sum_0` | **1.02 µs** | 10826.386 | 19.587973 | `triton_per_fused_sum_0` |
| 11 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.09 µs** | 10826.387 | 19.587975 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 12 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.388 | 19.587976 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 13 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.390 | 19.587977 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 14 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.391 | 19.587979 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 15 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.80 µs** | 10826.392 | 19.587980 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 16 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.80 µs** | 10826.394 | 19.587981 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 17 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.395 | 19.587982 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 18 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.396 | 19.587984 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 19 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.397 | 19.587985 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 20 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.399 | 19.587986 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 21 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.83 µs** | 10826.400 | 19.587988 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 22 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.401 | 19.587989 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 23 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.402 | 19.587990 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 24 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.15 µs** | 10826.404 | 19.587992 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 25 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.86 µs** | 10826.406 | 19.587993 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 26 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.80 µs** | 10826.407 | 19.587994 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 27 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.09 µs** | 10826.408 | 19.587996 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 28 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.409 | 19.587997 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 29 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.411 | 19.587999 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 30 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.412 | 19.588000 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 31 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.413 | 19.588001 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 32 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.86 µs** | 10826.415 | 19.588002 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 33 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.416 | 19.588004 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 34 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.417 | 19.588005 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 35 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.12 µs** | 10826.419 | 19.588006 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 36 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.420 | 19.588008 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 37 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.421 | 19.588009 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 38 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.423 | 19.588010 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 39 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.83 µs** | 10826.424 | 19.588012 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 40 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.425 | 19.588013 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 41 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.426 | 19.588014 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 42 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.428 | 19.588015 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 43 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.18 µs** | 10826.429 | 19.588017 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 44 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.74 µs** | 10826.431 | 19.588018 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 45 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.432 | 19.588019 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 46 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.86 µs** | 10826.433 | 19.588021 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 47 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.434 | 19.588022 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 48 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.436 | 19.588023 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 49 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.437 | 19.588025 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 50 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.438 | 19.588026 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 51 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.439 | 19.588027 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 52 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.441 | 19.588028 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 53 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.86 µs** | 10826.442 | 19.588030 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 54 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.443 | 19.588031 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 55 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.444 | 19.588032 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 56 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.446 | 19.588034 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 57 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.447 | 19.588035 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 58 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.448 | 19.588036 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 59 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.12 µs** | 10826.450 | 19.588037 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 60 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.74 µs** | 10826.451 | 19.588039 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 61 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.452 | 19.588040 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 62 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.454 | 19.588041 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 63 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.86 µs** | 10826.455 | 19.588043 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 64 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.456 | 19.588044 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 65 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.457 | 19.588045 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 66 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.459 | 19.588046 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 67 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.460 | 19.588048 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 68 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.461 | 19.588049 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 69 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.463 | 19.588050 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 70 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.464 | 19.588052 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 71 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.465 | 19.588053 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 72 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.96 µs** | 10826.466 | 19.588054 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 73 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.468 | 19.588055 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 74 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.469 | 19.588057 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 75 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **1.09 µs** | 10826.470 | 19.588058 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 76 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.90 µs** | 10826.472 | 19.588059 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 77 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.473 | 19.588060 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 78 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.93 µs** | 10826.474 | 19.588062 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 79 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.80 µs** | 10826.475 | 19.588063 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| 80 | Triton fused `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | **0.77 µs** | 10826.476 | 19.588064 | `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` |
| … | *(4021 more launches in chunk — see sqlite)* | | | | |

## 4. Stage sketch inside measured chunk

Compile fuses eager Stage 0–3 micro-ops; approximate windows by landmarks:

| Region | Landmark | Notes |
|---|---|---|
| prepare_images | first `upsample_bilinear2d` @ order 3611 (×3 in chunk) | Still eager-ish preprocess outside Inductor graphs |
| Vision / connector | `triton_poi_fused_convolution_*`, CUTLASS BF16 GEMM, `triton_*_fused_*layer_norm*`, FMHA | Maps to FX graphs #1–#6 |
| Prefill + expert Euler | `triton_per_fused_sum_0`, RoPE `triton_poi_fused_*cos*sin*`, `magma_sgemmEx`, FMHA, `triton_*_silu*` | Maps to FX graph #8 (bulk) |

### 4.1 First prepare / patch window (40 launches from first upsample)

Orders **3611–3650**; busy Σ **537.79 µs**. Compare eager kerne_list §2 (many elementwise vs fused Triton here).

| Order | Operator | GPU time | Kernel |
|---:|---|---:|---|
| 3611 | Bilinear upsample | **29.38 µs** | `void at::native::<unnamed>::upsample_bilinear2d_out_frame<float, float>(int, T2, T2, bool, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, const T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, const T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>)` |
| 3612 | Mul *[no aten]* | **3.65 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3613 | Add *[no aten]* | **3.71 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3614 | Fill *[no aten]* | **0.80 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<bool>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 3615 | Bilinear upsample | **28.22 µs** | `void at::native::<unnamed>::upsample_bilinear2d_out_frame<float, float>(int, T2, T2, bool, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, const T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, const T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>)` |
| 3616 | Mul *[no aten]* | **3.62 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3617 | Add *[no aten]* | **3.78 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3618 | Fill *[no aten]* | **0.77 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<bool>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 3619 | Bilinear upsample | **28.83 µs** | `void at::native::<unnamed>::upsample_bilinear2d_out_frame<float, float>(int, T2, T2, bool, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, const T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, const T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>, torch::headeronly::detail::GenericPackedTensorAccessor<torch::headeronly::detail::TensorAccessor<c10::ArrayRef<long>, T1, (unsigned long)3, torch::headeronly::DefaultPtrTraits, long>, at::detail::IndexBoundsCheck<(unsigned long)4, long>, T1, (unsigned long)4, torch::headeronly::DefaultPtrTraits, long>)` |
| 3620 | Mul *[no aten]* | **3.58 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3621 | Add *[no aten]* | **3.68 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| 3622 | Fill *[no aten]* | **0.77 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<bool>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 3623 | Fill *[no aten]* | **0.77 µs** | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| 3624 | Other GPU kernel | **11.55 µs** | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<float, (int)2, (int)1, (int)1>, at::native::Copy<float, float>>(T1, T2, T3...)` |
| 3625 | Triton fused `triton_poi_fused__to_copy_0` | **2.50 µs** | `triton_poi_fused__to_copy_0` |
| 3626 | Triton fused `triton_poi_fused__to_copy_0` | **0.64 µs** | `triton_poi_fused__to_copy_0` |
| 3627 | Triton fused `triton_poi_fused_arange_0` | **0.86 µs** | `triton_poi_fused_arange_0` |
| 3628 | Triton fused `triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1` | **1.22 µs** | `triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1` |
| 3629 | Triton fused `triton_poi_fused_add_mul_unsqueeze_2` | **0.77 µs** | `triton_poi_fused_add_mul_unsqueeze_2` |
| 3630 | Triton fused `triton_poi_fused_convolution_3` | **2.72 µs** | `triton_poi_fused_convolution_3` |
| 3631 | Triton fused `triton_poi_fused_convolution_4` | **13.31 µs** | `triton_poi_fused_convolution_4` |
| 3632 | Other GPU kernel | **6.94 µs** | `void nhwcAddPaddingKernel<__nv_bfloat16, __nv_bfloat16, float, (bool)1, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, const T1 *, T2 *, int, int, int, int, T3, T3, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)` |
| 3633 | Other GPU kernel | **5.76 µs** | `void nhwcAddPaddingKernel<__nv_bfloat16, __nv_bfloat16, float, (bool)1, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, const T1 *, T2 *, int, int, int, int, T3, T3, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)` |
| 3634 | Other GPU kernel | **326.59 µs** | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_64x64_32x10_nhwc_align8>(T1::Params)` |
| 3635 | Triton fused `triton_poi_fused_convolution_5` | **2.94 µs** | `triton_poi_fused_convolution_5` |
| 3636 | Triton fused `triton_poi_fused_full_6` | **0.64 µs** | `triton_poi_fused_full_6` |
| 3637 | Reduce *[no aten]* | **1.44 µs** | `void at_cuda_detail::cub::DeviceReduceSingleTileKernel<at_cuda_detail::cub::DeviceReducePolicy<int, unsigned long long, cuda::std::__4::plus<void>>::Policy600, at_cuda_detail::cub::TransformInputIterator<bool, at::native::<unnamed>::NonZeroOp<bool>, const bool *, long>, int *, unsigned long long, cuda::std::__4::plus<void>, int, int, cuda::std::__4::__identity>(T2, T3, T4, T5, T6, T8)` |
| 3638 | Other GPU kernel | **0.70 µs** | `void at_cuda_detail::cub::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, (bool)1>, int *>(T1, int, T2)` |
| 3639 | Other GPU kernel | **1.50 µs** | `void at_cuda_detail::cub::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::device_select_policy_hub<long, bool, int, (bool)0, (bool)0>::Policy900, at_cuda_detail::cub::CountingInputIterator<long, long>, at_cuda_detail::cub::TransformInputIterator<bool, at::native::<unnamed>::NonZeroOp<bool>, const bool *, long>, long *, int *, at_cuda_detail::cub::ScanTileState<int, (bool)1>, at_cuda_detail::cub::NullType, at_cuda_detail::cub::NullType, int, (bool)0, (bool)0>(T2, T3, T4, T5, T6, T7, T8, T9, int, at_cuda_detail::cub::detail::vsmem_t)` |
| 3640 | Other GPU kernel | **1.82 µs** | `void at::native::<unnamed>::write_indices<long>(long *, at::native::<unnamed>::TensorDims<T1>, int, T1, long *, long)` |
| 3641 | Elementwise *[no aten]* | **3.36 µs** | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)8>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)` |
| 3642 | Reduce *[no aten]* | **1.44 µs** | `void at_cuda_detail::cub::DeviceReduceSingleTileKernel<at_cuda_detail::cub::DeviceReducePolicy<int, unsigned long long, cuda::std::__4::plus<void>>::Policy600, at_cuda_detail::cub::TransformInputIterator<bool, at::native::<unnamed>::NonZeroOp<bool>, const bool *, long>, int *, unsigned long long, cuda::std::__4::plus<void>, int, int, cuda::std::__4::__identity>(T2, T3, T4, T5, T6, T8)` |
| 3643 | Other GPU kernel | **0.74 µs** | `void at_cuda_detail::cub::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, (bool)1>, int *>(T1, int, T2)` |
| 3644 | Other GPU kernel | **1.50 µs** | `void at_cuda_detail::cub::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::device_select_policy_hub<long, bool, int, (bool)0, (bool)0>::Policy900, at_cuda_detail::cub::CountingInputIterator<long, long>, at_cuda_detail::cub::TransformInputIterator<bool, at::native::<unnamed>::NonZeroOp<bool>, const bool *, long>, long *, int *, at_cuda_detail::cub::ScanTileState<int, (bool)1>, at_cuda_detail::cub::NullType, at_cuda_detail::cub::NullType, int, (bool)0, (bool)0>(T2, T3, T4, T5, T6, T7, T8, T9, int, at_cuda_detail::cub::detail::vsmem_t)` |
| 3645 | Other GPU kernel | **1.57 µs** | `void at::native::<unnamed>::write_indices<long>(long *, at::native::<unnamed>::TensorDims<T1>, int, T1, long *, long)` |
| 3646 | Elementwise *[no aten]* | **3.36 µs** | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)8>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)` |
| 3647 | Triton fused `triton_poi_fused_add_embedding_0` | **21.86 µs** | `triton_poi_fused_add_embedding_0` |
| 3648 | Triton fused `triton_red_fused_native_layer_norm_0` | **4.32 µs** | `triton_red_fused_native_layer_norm_0` |
| 3649 | Triton fused `triton_per_fused_native_layer_norm_1` | **3.04 µs** | `triton_per_fused_native_layer_norm_1` |
| 3650 | Triton fused `triton_poi_fused_native_layer_norm_2` | **3.14 µs** | `triton_poi_fused_native_layer_norm_2` |

### 4.2 First attention / FMHA neighborhood

Orders **3650–3679** around first FMHA (order 3655). Nearby Triton names show fused mask / RoPE / silu epilogues.

| Order | Operator | GPU time | Kernel |
|---:|---|---:|---|
| 3650 | Triton fused `triton_poi_fused_native_layer_norm_2` | **3.14 µs** | `triton_poi_fused_native_layer_norm_2` |
| 3651 | Linear / GEMM BF16 (CUTLASS TC) | **32.61 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3652 | Linear / GEMM BF16 (CUTLASS TC) | **32.54 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3653 | Linear / GEMM BF16 (CUTLASS TC) | **33.28 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3654 | Triton fused `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_expand_ge_index_new_ones_scalar_tensor_transpose_unsqueeze_view_where_3` | **7.84 µs** | `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_expand_ge_index_new_ones_scalar_tensor_transpose_unsqueeze_view_where_3` |
| 3655 | Mem-eff / FlashAttention (CUTLASS FMHA) | **192.67 µs** | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 3656 | Linear / GEMM BF16 (CUTLASS TC) | **31.46 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3657 | Triton fused `triton_red_fused_add_native_layer_norm_view_4` | **4.80 µs** | `triton_red_fused_add_native_layer_norm_view_4` |
| 3658 | Triton fused `triton_per_fused_add_native_layer_norm_view_5` | **0.80 µs** | `triton_per_fused_add_native_layer_norm_view_5` |
| 3659 | Triton fused `triton_poi_fused_add_native_layer_norm_view_6` | **3.52 µs** | `triton_poi_fused_add_native_layer_norm_view_6` |
| 3660 | Linear / GEMM BF16 (CUTLASS TC) | **120.64 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` |
| 3661 | Triton fused `triton_poi_fused_gelu_view_7` | **10.72 µs** | `triton_poi_fused_gelu_view_7` |
| 3662 | Linear / GEMM BF16 (CUTLASS TC) | **152.06 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params)` |
| 3663 | Triton fused `triton_red_fused_add_native_layer_norm_view_8` | **12.58 µs** | `triton_red_fused_add_native_layer_norm_view_8` |
| 3664 | Linear / GEMM BF16 (CUTLASS TC) | **31.26 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3665 | Linear / GEMM BF16 (CUTLASS TC) | **32.70 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3666 | Linear / GEMM BF16 (CUTLASS TC) | **31.78 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3667 | Mem-eff / FlashAttention (CUTLASS FMHA) | **190.75 µs** | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 3668 | Linear / GEMM BF16 (CUTLASS TC) | **31.36 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3669 | Triton fused `triton_red_fused_add_native_layer_norm_view_9` | **12.77 µs** | `triton_red_fused_add_native_layer_norm_view_9` |
| 3670 | Linear / GEMM BF16 (CUTLASS TC) | **119.87 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` |
| 3671 | Triton fused `triton_poi_fused_gelu_view_7` | **10.62 µs** | `triton_poi_fused_gelu_view_7` |
| 3672 | Linear / GEMM BF16 (CUTLASS TC) | **151.74 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(T1::Params)` |
| 3673 | Triton fused `triton_red_fused_add_native_layer_norm_view_10` | **18.18 µs** | `triton_red_fused_add_native_layer_norm_view_10` |
| 3674 | Linear / GEMM BF16 (CUTLASS TC) | **31.52 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3675 | Linear / GEMM BF16 (CUTLASS TC) | **32.45 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3676 | Linear / GEMM BF16 (CUTLASS TC) | **32.19 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3677 | Mem-eff / FlashAttention (CUTLASS FMHA) | **189.02 µs** | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 3678 | Linear / GEMM BF16 (CUTLASS TC) | **30.94 µs** | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)` |
| 3679 | Triton fused `triton_per_fused_add_native_layer_norm_view_11` | **3.78 µs** | `triton_per_fused_add_native_layer_norm_view_11` |

## 5. Slowest 25 launches (measured chunk)

| Rank | GPU time | Order | Operator | Kernel |
|---:|---:|---:|---|---|
| 1 | **326.59 µs** | 3634 | Other GPU kernel | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_64x64_32x10_nhwc_align8>(T1::Params)` |
| 2 | **322.18 µs** | 3946 | Other GPU kernel | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_64x64_32x10_nhwc_align8>(T1::Params)` |
| 3 | **322.14 µs** | 3790 | Other GPU kernel | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_64x64_32x10_nhwc_align8>(T1::Params)` |
| 4 | **193.02 µs** | 3728 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 5 | **192.86 µs** | 3967 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 6 | **192.67 µs** | 3655 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 7 | **192.03 µs** | 3708 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 8 | **191.81 µs** | 3811 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 9 | **191.07 µs** | 4081 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 10 | **191.04 µs** | 3739 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 11 | **190.88 µs** | 4030 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 12 | **190.75 µs** | 3667 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 13 | **190.72 µs** | 4061 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 14 | **190.62 µs** | 4010 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 15 | **190.59 µs** | 4051 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 16 | **190.56 µs** | 3999 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 17 | **190.53 µs** | 3759 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 18 | **190.50 µs** | 3874 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 19 | **190.46 µs** | 4071 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 20 | **190.43 µs** | 3823 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 21 | **190.43 µs** | 3749 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 22 | **190.43 µs** | 3687 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 23 | **190.37 µs** | 3979 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 24 | **190.34 µs** | 3925 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |
| 25 | **190.30 µs** | 3843 | Mem-eff / FlashAttention (CUTLASS FMHA) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, (bool)1, (int)64, (int)64, (int)64, (bool)1, (bool)1>::Params)` |

## 6. Compile vs eager (same GPU / model)

| Metric | Eager chunk #3 (kerne_list) | Compile steady chunk (this file) |
|---|---:|---:|
| Wall-span | ~128.7 ms | **48.0 ms** |
| Busy Σ | ~57.4 ms | **43.8 ms** |
| Gap ratio | ~55% | **8.8%** |
| # CUPTI launches | ~13836 | **4101** |
| Host `select_action` (smoke) | ~90 ms e2e stages | ~47–50 ms warm |

Fewer launches + fused Triton epilogues cut launch overhead; remaining busy is dominated by CUTLASS BF16 GEMM + FMHA (see §2).

## 7. Skill — rebuild this list

```bash
export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home
export SMOKE_COMPILE_MODE=reduce-overhead
export NSIGHT_WARMUP=3 NSIGHT_REPEATS=3
# Need Nsight Systems ≥2025.1.3 (not the nsight-compute bundled nsys alone)
nsys profile -o doc/gpu/compile_mode/nsight/smolvla_nsys_compile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  python src/smolvla_nsight_target_compile.py
nsys export --type=sqlite -o doc/gpu/compile_mode/nsight/smolvla_nsys_compile.sqlite \
  doc/gpu/compile_mode/nsight/smolvla_nsys_compile.nsys-rep
python src/smolvla_compile_kerne_list_from_nsys.py
```

Note: `--capture-range=cudaProfilerApi` may yield an empty report with CUDA graphs on some hosts; full-session + landmark chunking (this recipe) is more reliable.

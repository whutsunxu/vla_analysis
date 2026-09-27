# SmolVLA Fused Compile Operator List (GPU backend)

This document is the **Inductor post-fusion / codegen** counterpart of `SmolVLA_CompileOp_List_gpu_backend.md` (pre-codegen FX aten nodes).

| Artifact | Role |
|---|---|
| `smolvla_compile_fused_kernels.json` | structured fused launch list |
| `smolvla_compile_chrome_trace.json` | warm-run chrome / Kineto trace |
| `inductor_debug/**/output_code.py` | Inductor codegen source of truth |
| `SmolVLA_Fused_CompileOp_List_gpu_backend.md` | this operator list |

---

## 0. Capture method and scope

### 0.1 What was compiled

| Item | Value |
|---|---|
| Model | `lerobot/smolvla_base` |
| Device | `cuda` — None |
| Torch / CUDA | None / None |
| `torch.compile` mode | **`reduce-overhead`** |
| Compiled callable | `VLAFlowMatching.sample_actions` |
| Env | `TORCH_COMPILE_DEBUG=1`, `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`, `TORCH_LOGS=output_code` |
| Graphs / debug dirs | **8** |

### 0.2 Method

1. Compile `sample_actions` with Inductor debug dumps (`ir_post_fusion.txt`, `output_code.py`, provenance JSON).
2. Walk each graph’s `Runner.call` / `partition_*` body in **source order** to list launches: `aten::*`, `extern_kernels.*`, `triton_*_fused_*.run`.
3. Attach fused aten origins from `inductor_provenance_tracking_node_mappings.json` (`cppCodeToPost`) and I/O shapes/dtypes from `empty_strided_*` / FX header comments.
4. **Pre** (before graph #1): eager `prepare_images` upsample path from nsys CUPTI (not Inductor) — see §Pre.

| FX list (`CompileOp`) | Fused list (this file) |
|---|---|
| Pre-codegen aten / view / transpose | Post-fusion **GPU launches** |
| One row per FX node | One row per Triton / extern / leftover aten |
| Fusion not visible | `_fused_` names + provenance = merged atens |

### 0.3 FX graphs — roles and cross-graph dataflow

Same Dynamo partitions as the FX compile list (8 graphs). See `SmolVLA_CompileOp_List_gpu_backend.md` §0.3 for the full ascii dataflow.

| Graph | Fused launches | Role | Key tensors in → out |
|---:|---:|---|---|
| **Pre** | 4 | Eager `prepare_images` (outside Inductor): bilinear upsample + `2x−1` + mask fill. Table = **first camera** (×3 cams/chunk). | `[1,3,256,256] f32` → `[1,3,512,512] f32` (+ mask) |
| **1** | 1 | Cast camera image f32 → bf16 (ViT dtype). | `image` `[1,3,512,512] f32` → `[…] bf16` |
| **2** | 1 | Build full ViT patch attention mask (`ones` → bool). | `ones` → mask `[1,32,32] bool` |
| **3** | 8 | ViT patch embed (`conv2d` 16×16) + mask-derived position-id bookkeeping. | `pixel_values` + patch weight/bias + mask → patch tokens `[1,1024,768]`, `position_ids`, flat mask |
| **4** | 2 | Add learned position embedding to patch tokens. | `embeddings` + `position_embedding.weight` → `embeddings` `[1,1024,768] bf16` |
| **5** | 14 | ViT encoder stack (12× LayerNorm / SDPA / MLP) → last_hidden_state. | pos-aware embeddings + mask + layer params → `last_hidden_state` `[1,1024,768] bf16` |
| **6** | 2 | Vision→language connector (spatial pack / PixelShuffle-style reshape + Linear). | `last_hidden_state` → `image_hidden_states` `[1,64,960] bf16` |
| **7** | 1 | Language token embedding lookup. | `tokens` `[1,48]` + `embed_tokens.weight` → `embedding` `[1,48,960] bf16` |
| **8** | 46 | Prefix + VLM/expert flow-matching (attn / RoPE / Euler) → action chunk. | prefix acts + masks + weights → `x_t` / actions `[1,50,32] f32` |

### 0.4 Global fused-launch histogram (top 40)

**Total fused launches (sum over graphs):** 3541

| kernel / op | count |
|---|---:|
| `extern_kernels.mm` | 1258 |
| `extern_kernels.bmm` | 350 |
| `triton_poi_fused__unsafe_view_clone_expand_unsqueeze_view_25` | 160 |
| `triton_poi_fused__unsafe_view_mul_silu_27` | 160 |
| `extern_kernels.addmm` | 82 |
| `triton_per_fused_add_cumsum_min_sub_unsqueeze_1` | 80 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_10` | 80 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_11` | 80 |
| `triton_per_fused__softmax__to_copy_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_14` | 80 |
| `triton_poi_fused_clone_permute_view_15` | 80 |
| `triton_poi_fused__to_copy__unsafe_view_transpose_view_24` | 80 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_29` | 80 |
| `triton_per_fused__softmax_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_slice_sub_unsqueeze_view_where_31` | 80 |
| `triton_poi_fused__to_copy__unsafe_view_clone_permute_view_32` | 80 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_34` | 70 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_40` | 70 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_41` | 70 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_42` | 70 |
| `triton_poi_fused_clone_expand_transpose_unsqueeze_54` | 24 |
| `triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_56` | 24 |
| `triton_poi_fused__unsafe_view_cat_clone_expand_slice_transpose_unsqueeze_view_58` | 24 |
| `triton_poi_fused__to_copy__unsafe_view_cat_clone_expand_slice_transpose_unsqueeze_59` | 24 |
| `triton_poi_fused__unsafe_view_cat_clone_expand_slice_transpose_unsqueeze_view_61` | 16 |
| `triton_poi_fused__to_copy__unsafe_view_cat_clone_expand_slice_transpose_unsqueeze_62` | 16 |
| `triton_poi_fused__unsafe_view_cat_slice_transpose_view_64` | 16 |
| `triton_poi_fused__to_copy_cat_slice_transpose_65` | 16 |
| `triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_16` | 15 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_div_mul_pow_sin_slice_split_sub_unsqueeze_view_18` | 15 |
| `triton_red_fused__softmax__to_copy_bitwise_and_exp_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_19` | 15 |
| `triton_poi_fused_clone_permute_view_20` | 15 |
| `triton_poi_fused__unsafe_view_mul_silu_22` | 15 |
| `triton_poi_fused_gelu_view_7` | 12 |
| `triton_per_fused_sum_0` | 10 |
| `triton_per_fused_cumsum_lift_fresh_unsqueeze_2` | 10 |
| `triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_5` | 10 |
| `triton_poi_fused_addmm_silu_view_6` | 10 |
| `triton_per_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_7` | 10 |
| `triton_per_fused__to_copy__unsafe_view_add_addmm_mean_mul_pow_rsqrt_view_26` | 10 |
| `triton_per_fused__to_copy__unsafe_view_add_addmm_mean_mul_pow_rsqrt_view_28` | 10 |
| `triton_per_fused__to_copy__unsafe_view_add_addmm_mean_mul_pow_rsqrt_view_33` | 10 |

---

## Pre. Eager `prepare_images` — upsample / scale / mask (outside Inductor)

**Meaning:** Camera preprocess before compiled `sample_actions` graphs: bilinear upsample 256→512, image scale `2x−1`, bool mask fill. **Not** an Inductor fused graph — still eager ATen kernels (see `smolVLA_kerne_list_gpu_backend.md` §4.1).

**Dataflow (this stage):** `[1,3,256,256] f32` → `[1,3,512,512] f32` (+ `[1,32,32] bool` mask). **Repeat ×3 cameras** / chunk; table = **first camera** only (same convention as G1–G6).

**Launches (calling order):** 4 · triton=0 · extern=0 · aten=4 (*(representative)* **cam0** only; ×3 cams in full chunk)

Ops below are in **CUPTI calling order** (warm chunk #3, first upsample window) — column **Order**. Input/Output from eager Stage‑0 conventions / shapes. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; bilinear = 7 Basic/out). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolVLA_kerne_list_gpu_backend.md` §4.1 orders 3611–3614). **meaning** / **fused aten** are the last two columns (`fused aten` = ATen op; no Inductor fusion here).

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `upsample_bilinear2d` | `aten` | Pre · prepare_images · cam0 | `[1,3,256,256] float32` | `[1,3,512,512] float32` | 3.93e-03 | 133.8 | 29.9% | 5.50e-03 (FP32) | 187.2 | 0.790% | 1.40 | bd-bounded | **29.38 µs** | ATen bilinear upsample 256→512 (eager<br>`prepare_images`) | `upsample_bilinear2d` |
| 2 | `vectorized_elementwise_mul` | `aten` | Pre · prepare_images · cam0 | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1723.3 | 384.7% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 215.3 | 0.909% | 0.125 | bd-bounded | **3.65 µs** | Image scale `×2` (eager elementwise;<br>*[no aten]* name match) | `mul` |
| 3 | `vectorized_elementwise_add` | `aten` | Pre · prepare_images · cam0 | `[1,3,512,512] float32` | `[1,3,512,512] float32` | 6.29e-03 | 1695.4 | 378.4% ⚠L2/cache·algo-IO≠DRAM | 7.86e-04 (FP32) | 211.9 | 0.894% | 0.125 | bd-bounded | **3.71 µs** | Image bias `−1` (eager elementwise;<br>*[no aten]* name match) | `add` |
| 4 | `vectorized_elementwise_fill_bool` | `aten` | Pre · prepare_images · cam0 | — | `[1,32,32] bool` | 1.02e-06 | 1.275 | 0.285% | 0 | 0 | — | — | bd-bounded | **0.80 µs** | Fill ViT patch mask `ones→bool`<br>(eager; *[no aten]* name match) | `fill_` |

---

## 1. Fused Inductor graph #1 — Cast camera image f32 → bf16

**Meaning:** Cast camera image f32 → bf16 (ViT dtype).

**Dataflow (this graph):** `image` `[1,3,512,512] f32` → `[…] bf16`

**Launches (calling order):** 1 · triton=1 · extern=0 · aten=0

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_poi_fused__to_copy_0` | `triton` | G1 · cast | `[1,3,512,512] float32` | `[1,3,512,512] bfloat16` | 4.72e-03 | 1867.1 | 416.8% ⚠L2/cache·algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **2.53 µs** | pointwise Triton fusion: convert_element_type | `convert_element_type` |

---

## 2. Fused Inductor graph #2 — Build full ViT patch attention mask

**Meaning:** Build full ViT patch attention mask (`ones` → bool).

**Dataflow (this graph):** `ones` → mask `[1,32,32] bool`

**Launches (calling order):** 1 · triton=1 · extern=0 · aten=0

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_poi_fused__to_copy_0` | `triton` | G2 · mask | — | `[1,32,32] bool` | 1.02e-06 | 1.678 | 0.4% | 0 | 0 | — | — | bd-bounded | **0.61 µs** | pointwise Triton fusion: full_default | `full_default` |

---

## 3. Fused Inductor graph #3 — ViT patch embed

**Meaning:** ViT patch embed (`conv2d` 16×16) + mask-derived position-id bookkeeping.

**Dataflow (this graph):** `pixel_values` + patch weight/bias + mask → patch tokens `[1,1024,768]`, `position_ids`, flat mask

**Launches (calling order):** 8 · triton=7 · extern=1 · aten=0

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_poi_fused_arange_0` | `triton` | G3 · patch_embed | — | `[31] float32` | 1.24e-07 | 0.144 | 0.0% | 3.10e-08 (FP32) | 0.0359 | 0.000151% | 2.50e-01 | bd-bounded | **0.86 µs** | pointwise Triton fusion: add, mul,<br>convert_element_type, iota | `add`, `mul`, `convert_element_type`, `iota` |
| 2 | `triton_per_fused__to_copy_arange_`<br>`bucketize_clamp_mul_reciprocal_select_`<br>`sum_unsqueeze_1` | `triton` | G3 · patch_embed | `[1,32,32] bool; [31] float32; [1,32] int64` | `[1,32] int64` | 1.66e-06 | 1.365 | 0.3% | 3.20e-08 (FP32) | 0.0263 | 0.000111% | 1.93e-02 | bd-bounded | **1.22 µs** | persistent-reduction Triton fusion: sum_1, select,<br>sum_2, select_1, bucketize, convert_element_type_3,<br>clamp_max, mul_5, unsqueeze,<br>convert_element_type_1, add_1, mul_3, …(+20) | `sum_1`, `select`, `sum_2`, `select_1`, `bucketize`,<br>`convert_element_type_3`, `clamp_max`, `mul_5`,<br>`unsqueeze`, `convert_element_type_1`, `add_1`,<br>`mul_3`, `iota_1`, `unsqueeze_1`, `mul_1`,<br>`reciprocal`, `add`, `mul`, `convert_element_type`,<br>`iota`, `bucketize_1`, `convert_element_type_4`,<br>`clamp_max_1`, `mul_6`, `unsqueeze_2`,<br>`convert_element_type_2`, `add_2`, `mul_4`, `iota_2`,<br>`unsqueeze_3`, `mul_2`, `reciprocal_1` |
| 3 | `triton_poi_fused_add_mul_unsqueeze_2` | `triton` | G3 · patch_embed | `[1,32] int64; [1,32] int64` | `[1,32,32] int64` | 8.70e-06 | 11.82 | 2.6% | 1.02e-06 (FP32) | 1.386 | 0.00585% | 1.18e-01 | bd-bounded | **0.74 µs** | pointwise Triton fusion: add_3, mul_7, unsqueeze_4,<br>unsqueeze_5 | `add_3`, `mul_7`, `unsqueeze_4`, `unsqueeze_5` |
| 4 | `triton_poi_fused_convolution_3` | `triton` | G3 · patch_embed | `[1,3,512,512] bfloat16` | `[1,3,512,512] bfloat16` | 3.15e-03 | 1144.6 | 255.5% ⚠L2/cache·algo-IO≠DRAM | — | — | — | — | bd-bounded | **2.75 µs** | pointwise Triton fusion: convolution | `convolution` |
| 5 | `triton_poi_fused_convolution_4` | `triton` | G3 · patch_embed | `[768,3,16,16] bfloat16` | `[768,3,16,16] bfloat16` | 2.36e-03 | 175.6 | 39.2% | — | — | — | — | bd-bounded | **13.44 µs** | pointwise Triton fusion: convolution | `convolution` |
| 6 | `extern_kernels.convolution` | `extern` | G3 · patch_embed | `[1,3,512,512] bfloat16 × [768,3,16,16] bfloat16` | `[1,768,32,32] bfloat16` | 4.33e-03 | 13.19 | 2.9% | 2D 1.61 (BF16) | 4903.3 | 5.17% | 372 | 2D-calc-bounded | **328.35 µs** | Conv (aten.convolution / conv2d) via cuDNN/cuBLAS | `convolution` |
| 7 | `triton_poi_fused_convolution_5` | `triton` | G3 · patch_embed | `[768] bfloat16` | `[1,768,32,32] bfloat16` | 1.57e-03 | 570.5 | 127.3% ⚠L2/cache·algo-IO≠DRAM | — | — | — | — | bd-bounded | **2.75 µs** | pointwise Triton fusion: convolution | `convolution` |
| 8 | `triton_poi_fused_full_6` | `triton` | G3 · patch_embed | — | `[1,1024] int64` | 8.19e-06 | 12.8 | 2.9% | 0 | 0 | — | — | bd-bounded | **0.64 µs** | pointwise Triton fusion: full_default | `full_default` |

---

## 4. Fused Inductor graph #4 — Add learned position embedding to patch tokens

**Meaning:** Add learned position embedding to patch tokens.

**Dataflow (this graph):** `embeddings` + `position_embedding.weight` → `embeddings` `[1,1024,768] bf16`

**Launches (calling order):** 2 · triton=1 · extern=0 · aten=1

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `aten::index_put_` | `aten` | G4 · pos_embed | `[1,1024] int64; [1,1024] bool; [1024] int64` | `[1,1024] int64` | 2.56e-05 | 7.619 | 1.7% | — | — | — | — | bd-bounded | **3.36 µs** | ATen op `aten::index_put_` (Inductor wrapper /<br>mutation) | `aten::index_put_` |
| 2 | `triton_poi_fused_add_embedding_0` | `triton` | G4 · pos_embed | `[1,1024,768] bfloat16 × [1,1024,768] bfloat16` | `[1,1024,768] bfloat16` | 4.72e-03 | 213.5 | 47.6% | 7.86e-04 (BF16) | 35.55 | 0.15% | 1.67e-01 | bd-bounded | **22.11 µs** | pointwise Triton fusion: add, embedding | `add`, `embedding` |

---

## 5. Fused Inductor graph #5 — ViT encoder stack

**Meaning:** ViT encoder stack (12× LayerNorm / SDPA / MLP) → last_hidden_state.

**Dataflow (this graph):** pos-aware embeddings + mask + layer params → `last_hidden_state` `[1,1024,768] bf16`

**Stage column (G5):** collapsed to one representative **`ViT · L0/12`** (LN / attn / MLP). Pass `--full` to annotate without collapsing.

**Launches (calling order):** 14 · triton=8 · extern=6 · aten=0 (*(representative)* **ViT L0/12** only)

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_red_fused_native_layer_norm_0` | `triton` | ViT · L0/12 · LN | `[1,1024,768] bfloat16` | `[1,1024,1,6] float32; [1,1024,1,6] float32` | 1.62e-03 | 375.0 | 83.7% | 1.57e-03 (FP32) | 363.4 | 1.53% | 9.70e-01 | bd-bounded | **4.32 µs** | reduction Triton fusion: var_mean,<br>convert_element_type, clone | `var_mean`, `convert_element_type`, `clone` |
| 2 | `triton_per_fused_native_layer_norm_1` | `triton` | ViT · L0/12 · LN | `[1,1024,1,6] float32; [1,1024,1,6] float32` | `[1,1024,1] float32; [1,1024,1] float32` | 5.73e-05 | 18.09 | 4.0% | 2.87e-05 (FP32) | 9.059 | 0.0382% | 5.00e-01 | bd-bounded | **3.17 µs** | persistent-reduction Triton fusion: var_mean,<br>convert_element_type, clone | `var_mean`, `convert_element_type`, `clone` |
| 3 | `triton_poi_fused_native_layer_norm_2` | `triton` | ViT · L0/12 · LN | `[1,1024,768] bfloat16 × [1,1024,1] float32`<br>`× [1,1024,1] float32 × [768] bfloat16 × [768] bfloat16` | `[1,1024,768] bfloat16` | 3.16e-03 | 949.5 | 211.9% ⚠L2/cache·algo-IO≠DRAM | 5.51e-03 (FP32) | 1655.6 | 6.99% | 1.74 | bd-bounded | **3.33 µs** | pointwise Triton fusion: convert_element_type_1,<br>add_3, mul_1, mul, sub, convert_element_type,<br>clone, rsqrt, add_2, var_mean | `convert_element_type_1`, `add_3`, `mul_1`, `mul`,<br>`sub`, `convert_element_type`, `clone`, `rsqrt`,<br>`add_2`, `var_mean` |
| 4 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · attn | `[1024,768] bfloat16 × [768,768] bfloat16`<br>`× [768] bfloat16` | `[1024,768] bfloat16` | 4.33e-03 | 131.4 | 29.3% | 2D 1.21 (BF16) + 1D 7.86e-04 | 36735.0 | 38.8% | 279 | 2D-calc-bounded | **32.96 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 5 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · attn | `[1024,768] bfloat16 × [768,768] bfloat16`<br>`× [768] bfloat16` | `[1024,768] bfloat16` | 4.33e-03 | 133.8 | 29.9% | 2D 1.21 (BF16) + 1D 7.86e-04 | 37425.4 | 39.5% | 279 | 2D-calc-bounded | **32.35 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 6 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · attn | `[1024,768] bfloat16 × [768,768] bfloat16`<br>`× [768] bfloat16` | `[1024,768] bfloat16` | 4.33e-03 | 131.1 | 29.3% | 2D 1.21 (BF16) + 1D 7.86e-04 | 36663.8 | 38.7% | 279 | 2D-calc-bounded | **33.02 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 7 | `triton_poi_fused__scaled_dot_product_`<br>`efficient_attention_add_arange_bitwise_`<br>`and_expand_ge_index_new_ones_scalar_`<br>`tensor_transpose_unsqueeze_view_where_3` | `triton` | ViT · L0/12 · attn | `[1,32,32] bool; [1,1,1024,1024] bfloat16`<br>`; [1,1,1024,1024] bfloat16; [1,1,1024,1024] bfloat16` | `[1,1,1024,1024] bfloat16` | 8.39e-03 | 1070.2 | 238.9% ⚠L2/cache·algo-IO≠DRAM | 1.05e-03 (BF16) | 133.9 | 0.565% | 1.25e-01 | bd-bounded | **7.84 µs** | pointwise Triton fusion: _scaled_dot_product_efficient_attention,<br>permute_1, view_3, view_2, permute_3, view_6,<br>view_5, permute_5, view_9, view_8, expand_1, where,<br>…(+65) | `_scaled_dot_product_efficient_attention`,<br>`permute_1`, `view_3`, `view_2`, `permute_3`,<br>`view_6`, `view_5`, `permute_5`, `view_9`, `view_8`,<br>`expand_1`, `where`, `expand`, `bitwise_and_1`,<br>`bitwise_and`, `full_default`, `ge`, `unsqueeze_5`,<br>`unsqueeze_4`, `unsqueeze_3`, `add`, `iota_2`,<br>`index`, `view`, `unsqueeze_2`, `unsqueeze_1`,<br>`unsqueeze`, `iota`, `unsqueeze_8`, `unsqueeze_7`,<br>`unsqueeze_6`, `add_1`, `iota_3`, `full_default_2`,<br>`full_default_1`,<br>`_scaled_dot_product_efficient_attention_1`,<br>`permute_11`, `view_19`, `view_18`, `permute_13`,<br>`view_22`, `view_21`, `permute_15`, `view_25`,<br>`view_24`, `expand_2`, `where_1`, `full_default_4`,<br>`full_default_3`,<br>`_scaled_dot_product_efficient_attention_2`,<br>`permute_21`, `view_35`, `view_34`, `permute_23`,<br>`view_38`, `view_37`, `permute_25`, `view_41`,<br>`view_40`, `expand_3`, `where_2`, `full_default_6`,<br>`full_default_5`,<br>`_scaled_dot_product_efficient_attention_3`,<br>`permute_31`, `view_51`, `view_50`, `permute_33`,<br>`view_54`, `view_53`, `permute_35`, `view_57`,<br>`view_56`, `expand_4`, `where_3`, `full_default_8`,<br>`full_default_7` |
| 8 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · attn | `[1024,768] bfloat16 × [768,768] bfloat16`<br>`× [768] bfloat16` | `[1024,768] bfloat16` | 4.33e-03 | 136.7 | 30.5% | 2D 1.21 (BF16) + 1D 7.86e-04 | 38219.3 | 40.3% | 279 | 2D-calc-bounded | **31.68 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 9 | `triton_red_fused_add_native_layer_norm_view_4` | `triton` | ViT · L0/12 · LN | `[1,1024,768] bfloat16` | `[1,1024,1,6] float32; [1,1024,1,6] float32` | 1.62e-03 | 333.1 | 74.3% | 1.57e-03 (FP32) | 322.8 | 1.36% | 9.70e-01 | bd-bounded | **4.86 µs** | reduction Triton fusion: var_mean_1,<br>convert_element_type_14, clone_1, add_4, view_12 | `var_mean_1`, `convert_element_type_14`, `clone_1`,<br>`add_4`, `view_12` |
| 10 | `triton_per_fused_add_native_layer_norm_view_5` | `triton` | ViT · L0/12 · LN | `[1,1024,1,6] float32; [1,1024,1,6] float32` | `[1,1024,1] float32; [1,1024,1] float32` | 5.73e-05 | 68.87 | 15.4% | 2.87e-05 (FP32) | 34.5 | 0.146% | 5.00e-01 | bd-bounded | **0.83 µs** | persistent-reduction Triton fusion: var_mean_1,<br>convert_element_type_14, clone_1, add_4, view_12 | `var_mean_1`, `convert_element_type_14`, `clone_1`,<br>`add_4`, `view_12` |
| 11 | `triton_poi_fused_add_native_layer_norm_view_6` | `triton` | ViT · L0/12 · LN | `[1,1024,768] bfloat16 × [1,1024,1] float32`<br>`× [1,1024,1] float32 × [768] bfloat16 × [768] bfloat16` | `[1,1024,768] bfloat16` | 3.16e-03 | 897.7 | 200.4% ⚠L2/cache·algo-IO≠DRAM | 5.51e-03 (FP32) | 1565.3 | 6.60% | 1.74 | bd-bounded | **3.52 µs** | pointwise Triton fusion: convert_element_type_15,<br>add_6, mul_3, mul_2, sub_1,<br>convert_element_type_14, clone_1, add_4, view_12,<br>rsqrt_1, add_5, var_mean_1 | `convert_element_type_15`, `add_6`, `mul_3`, `mul_2`,<br>`sub_1`, `convert_element_type_14`, `clone_1`,<br>`add_4`, `view_12`, `rsqrt_1`, `add_5`, `var_mean_1` |
| 12 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · MLP | `[1024,768] bfloat16 × [768,3072] bfloat16`<br>`× [768] bfloat16` | `[1024,3072] bfloat16` | 1.26e-02 | 104.5 | 23.3% | 2D 4.83 (BF16) + 1D 3.15e-03 | 40094.5 | 42.3% | 384 | 2D-calc-bounded | **120.54 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 13 | `triton_poi_fused_gelu_view_7` | `triton` | ViT · L0/12 · MLP | `[1,1024,3072] bfloat16` | `[1,1024,3072] bfloat16` | 1.26e-02 | 1178.9 | 263.1% ⚠L2/cache·algo-IO≠DRAM | 3.15e-03 (BF16) | 294.7 | 1.24% | 2.50e-01 | bd-bounded | **10.69 µs** | pointwise Triton fusion: convert_element_type_20,<br>mul_9, mul_8, convert_element_type_19, view_14,<br>add_8, tanh, mul_7, add_7, mul_6, mul_5, mul_4 | `convert_element_type_20`, `mul_9`, `mul_8`,<br>`convert_element_type_19`, `view_14`, `add_8`,<br>`tanh`, `mul_7`, `add_7`, `mul_6`, `mul_5`, `mul_4` |
| 14 | `extern_kernels.addmm` | `extern` | ViT · L0/12 · MLP | `[1024,3072] bfloat16 × [3072,768] bfloat16`<br>`× [3072] bfloat16` | `[1024,768] bfloat16` | 1.26e-02 | 82.91 | 18.5% | 2D 4.83 (BF16) + 1D 7.86e-04 | 31788.2 | 33.5% | 384 | 2D-calc-bounded | **151.97 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |

---

## 6. Fused Inductor graph #6 — Vision→language connector

**Meaning:** Vision→language connector (spatial pack / PixelShuffle-style reshape + Linear).

**Dataflow (this graph):** `last_hidden_state` → `image_hidden_states` `[1,64,960] bf16`

**Launches (calling order):** 2 · triton=1 · extern=1 · aten=0

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_poi_fused__unsafe_view_clone_`<br>`permute_view_0` | `triton` | G6 · connector | `[1,1024,768] bfloat16` | `[1,8,8,12288] bfloat16` | 3.15e-03 | 1426.6 | 318.4% ⚠L2/cache·algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **2.21 µs** | pointwise Triton fusion: clone_1, permute_1,<br>view_2, clone, permute, view_1, view | `clone_1`, `permute_1`, `view_2`, `clone`, `permute`,<br>`view_1`, `view` |
| 2 | `extern_kernels.mm` | `extern` | G6 · connector | `[64,12288] bfloat16 × [12288,960] bfloat16` | `[64,960] bfloat16` | 2.53e-02 | 400.9 | 89.5% | 2D 1.51 (BF16) | 23928.8 | 25.2% | 59.7 | bd-bounded | **63.10 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm` |

---

## 7. Fused Inductor graph #7 — Language token embedding lookup

**Meaning:** Language token embedding lookup.

**Dataflow (this graph):** `tokens` `[1,48]` + `embed_tokens.weight` → `embedding` `[1,48,960] bf16`

**Launches (calling order):** 1 · triton=1 · extern=0 · aten=0

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_poi_fused_embedding_0` | `triton` | G7 · lang_embed | `[49280,960] bfloat16 × ids [1,48] int64` | `[1,48,960] bfloat16` | 9.47e-02 | 79983.1 | 17853.4% ⚠launch-floor+algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.18 µs** | pointwise Triton fusion: embedding | `embedding` |

---

## 8. Fused Inductor graph #8 — Prefix + VLM/expert flow-matching

**Meaning:** Prefix + VLM/expert flow-matching (attn / RoPE / Euler) → action chunk.

**Dataflow (this graph):** prefix acts + masks + weights → `x_t` / actions `[1,50,32] f32`

**Stage column (G8):** collapsed to one representative **`prefill · L0/15`**, one **`euler0 · expert L0/16`** (plus `prefix` / `euler0 · suffix` / `euler0 · update` / `tail`). Pass `--full` to annotate without collapsing. Full ×10×16 list requires regen from JSON first.

**Launches (calling order):** 46 · triton=34 · extern=12 · aten=0 (*(representative)* **prefill L0** + **euler0 / expert L0** only)

Ops below are in **Inductor codegen calling order** (`Runner.call` / `partition_*`) — column **Order**. Input/Output from buffer allocs / FX meta when available. **IO / GFLOPs / AI / Theoretical bottleneck** from shapes + op heuristics (same roofline rule as eager kerne_list §0.3; `—` when shapes missing). **BD / GFLOPs/sec / util / GPU time** from nsys CUPTI (`smolvla_compile_fused_table_timing.py`). **meaning** / **fused aten** (provenance) are the last two columns.

| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning | fused aten |
|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|
| 1 | `triton_per_fused_sum_0` | `triton` | prefix | `[1,241] bool` | `[1] int64` | 2.49e-07 | 0.243 | 0.1% | 2.41e-07 (FP32) | 0.235 | 0.000993% | 9.68e-01 | bd-bounded | **1.02 µs** | persistent-reduction Triton fusion: sum_17 | `sum_17` |
| 2 | `triton_per_fused_add_cumsum_min_sub_`<br>`unsqueeze_1` | `triton` | prefix | `[1] int64` | `[1,1] int64` | 1.60e-08 | 0.0147 | 0.0% | 1.00e-09 (FP32) | 0.000919 | 3.88e-06% | 6.25e-02 | bd-bounded | **1.09 µs** | persistent-reduction Triton fusion: min_1, sub_51,<br>add_147, unsqueeze_189, view_default_9,<br>mul_tensor_9, iota_default_9 | `min_1`, `sub_51`, `add_147`, `unsqueeze_189`,<br>`view_default_9`, `mul_tensor_9`, `iota_default_9` |
| 3 | `triton_per_fused_cumsum_lift_fresh_`<br>`unsqueeze_2` | `triton` | prefix | `[1,50] float32` | `[1,50] float32` | 4.00e-07 | 0.368 | 0.1% | 5.00e-08 (FP32) | 0.046 | 0.000194% | 1.25e-01 | bd-bounded | **1.09 µs** | persistent-reduction Triton fusion: cumsum_2,<br>unsqueeze_183, lift_fresh_copy_33 | `cumsum_2`, `unsqueeze_183`, `lift_fresh_copy_33` |
| 4 | `triton_per_fused_cumsum_3` | `triton` | prefix | `[1,241] bool` | `[1,241] int64` | 2.17e-06 | 2.119 | 0.5% | 2.41e-07 (FP32) | 0.235 | 0.000993% | 1.11e-01 | bd-bounded | **1.02 µs** | persistent-reduction Triton fusion: cumsum | `cumsum` |
| 5 | `triton_poi_fused__to_copy_cat_cos_`<br>`expand_linspace_mul_pow_reciprocal_sin_`<br>`unsqueeze_view_4` | `triton` | prefix | `[1,241] bool` | `[1,50,720] float32` | 1.44e-04 | 2.996 | 0.7% | 3.60e-05 (FP32) | 0.749 | 0.00316% | 2.50e-01 | bd-bounded | **48.06 µs** | pointwise Triton fusion: cat_1, view_513,<br>expand_97, unsqueeze_182, convert_element_type_532,<br>cat, sin_32, unsqueeze_180, mul_296, mul_295,<br>mul_294, reciprocal, …(+13) | `cat_1`, `view_513`, `expand_97`, `unsqueeze_182`,<br>`convert_element_type_532`, `cat`, `sin_32`,<br>`unsqueeze_180`, `mul_296`, `mul_295`, `mul_294`,<br>`reciprocal`, `mul_293`, `pow_66`, `where_16`, `lt`,<br>`iota_32`, `add_145`, `mul_291`,<br>`convert_element_type_530`, `sub_50`, `mul_292`,<br>`convert_element_type_531`, `sub_49`, `cos_32` |
| 6 | `extern_kernels.addmm` | `extern` | prefix | `[50,32] float32 × [32,720] float32 × [32] float32` | `[50,720] float32` | 2.43e-04 | 98.62 | 22.0% | 2D 2.30e-03 (FP32) + 1D 3.60e-05 | 948.1 | 4.00% | 9.64 | bd-bounded | **2.46 µs** | GEMM + bias (aten.linear / addmm) via cuBLAS | `addmm` |
| 7 | `triton_poi_fused__to_copy_cat_cos_`<br>`expand_linspace_mul_pow_reciprocal_sin_`<br>`unsqueeze_view_5` | `triton` | prefix | `[50,720] float32` | `[1,50,720] float32` | 2.88e-04 | 391.3 | 87.3% | 3.60e-05 (FP32) | 48.91 | 0.206% | 1.25e-01 | bd-bounded | **0.74 µs** | pointwise Triton fusion: cat_1, view_513,<br>expand_97, unsqueeze_182, convert_element_type_532,<br>cat, sin_32, unsqueeze_180, mul_296, mul_295,<br>mul_294, reciprocal, …(+13) | `cat_1`, `view_513`, `expand_97`, `unsqueeze_182`,<br>`convert_element_type_532`, `cat`, `sin_32`,<br>`unsqueeze_180`, `mul_296`, `mul_295`, `mul_294`,<br>`reciprocal`, `mul_293`, `pow_66`, `where_16`, `lt`,<br>`iota_32`, `add_145`, `mul_291`,<br>`convert_element_type_530`, `sub_50`, `mul_292`,<br>`convert_element_type_531`, `sub_49`, `cos_32` |
| 8 | `extern_kernels.mm` | `extern` | prefix | `[50,1440] float32 × [1440,720] float32` | `[50,720] float32` | 4.58e-03 | 301.3 | 67.3% | 2D 1.04e-01 (FP32) | 6842.1 | 28.9% | 22.6 | bd-bounded | **15.20 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 9 | `triton_poi_fused_addmm_silu_view_6` | `triton` | euler0 · suffix | `[1,50,720] float32 × [720] float32` | `[50,720] float32` | 2.91e-04 | 239.3 | 53.4% | 3.60e-05 (FP32) | 29.61 | 0.125% | 1.24e-01 | bd-bounded | **1.22 µs** | pointwise Triton fusion: div_64, view_515,<br>add_tensor_29, add_146, exp_32, neg_16 | `div_64`, `view_515`, `add_tensor_29`, `add_146`,<br>`exp_32`, `neg_16` |
| 10 | `extern_kernels.mm` | `extern` | euler0 · suffix | `[50,720] bfloat16 × [720,720] bfloat16` | `[50,720] float32` | 1.25e-03 | 134.2 | 30.0% | 2D 5.18e-02 (BF16) | 5562.7 | 5.87% | 41.4 | bd-bounded | **9.31 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 11 | `triton_per_fused__to_copy_add_addmm_`<br>`mean_mul_pow_rsqrt_view_7` | `triton` | euler0 · suffix | `[720] float32 × [50,720] float32 × [720] bfloat16` | `[1,50,720] bfloat16` | 2.20e-04 | 156.3 | 34.9% | 2.88e-04 (FP32) | 204.5 | 0.863% | 1.31 | bd-bounded | **1.41 µs** | persistent-reduction Triton fusion: mean_33,<br>pow_67, view_517, add_tensor_28,<br>convert_element_type_533, mul_300, mul_299,<br>rsqrt_33, add_148 | `mean_33`, `pow_67`, `view_517`, `add_tensor_28`,<br>`convert_element_type_533`, `mul_300`, `mul_299`,<br>`rsqrt_33`, `add_148` |
| 12 | `triton_per_fused__to_copy_add_mean_mul_`<br>`pow_rsqrt_8` | `triton` | prefill · L0/15 | `[1,241,960] float32; [960] bfloat16` | `[1,241,960] bfloat16` | 1.39e-03 | 658.1 | 146.9% ⚠L2/cache·algo-IO≠DRAM | 1.85e-03 (FP32) | 875.9 | 3.70% | 1.33 | bd-bounded | **2.11 µs** | persistent-reduction Triton fusion: mean, pow_1,<br>convert_element_type, mul_2, mul_1, rsqrt, add | `mean`, `pow_1`, `convert_element_type`, `mul_2`,<br>`mul_1`, `rsqrt`, `add` |
| 13 | `extern_kernels.mm` | `extern` | prefill · L0/15 | `[241,960] bfloat16 × [960,320] bfloat16` | `[241,320] bfloat16` | 1.23e-03 | 108.3 | 24.2% | 2D 1.48e-01 (BF16) | 13028.2 | 13.7% | 120 | bd-bounded | **11.36 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 14 | `triton_poi_fused__unsafe_view_cat_clone_`<br>`expand_transpose_unsqueeze_view_9` | `triton` | euler0 · suffix | `[241,320] bfloat16; [50,320] bfloat16` | `[1,291,5,3,64] bfloat16` | 7.45e-04 | 541.4 | 120.9% ⚠launch-floor+algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.38 µs** | pointwise Triton fusion: clone_150, expand_101,<br>unsqueeze_199, permute_265, cat_4, permute_4,<br>view_8, view_7, permute_263, view_526, view_525 | `clone_150`, `expand_101`, `unsqueeze_199`,<br>`permute_265`, `cat_4`, `permute_4`, `view_8`,<br>`view_7`, `permute_263`, `view_526`, `view_525` |
| 15 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_cumsum_div_mul_pow_`<br>`sin_slice_split_sub_unsqueeze_view_10` | `triton` | euler0 · suffix | `[50,960] bfloat16` | `[1] int64` | 9.60e-05 | 23.44 | 5.2% | 1.00e-09 (FP32) | 0.000244 | 1.03e-06% | 1.04e-05 | bd-bounded | **4.10 µs** | pointwise Triton fusion: slice_scatter_default_60,<br>copy_64, slice_161, sub_52, mul_303, split_32,<br>convert_element_type_540, view_520, view_519,<br>cos_33, unsqueeze_193, div_65, …(+24) | `slice_scatter_default_60`, `copy_64`, `slice_161`,<br>`sub_52`, `mul_303`, `split_32`,<br>`convert_element_type_540`, `view_520`, `view_519`,<br>`cos_33`, `unsqueeze_193`, `div_65`,<br>`convert_element_type_542`, `unsqueeze_190`,<br>`sub_51`, `add_147`, `unsqueeze_189`,<br>`view_default_9`, `mul_tensor_9`, `iota_default_9`,<br>`unsqueeze_192`, `unsqueeze_191`, `pow_68`,<br>`mul_302`, `convert_element_type_541`, `add_149`,<br>`mul_301`, `iota_33`, `mul_304`, `sin_33`,<br>`slice_scatter_default_61`, `copy_65`, `slice_164`,<br>`add_150`, `mul_305`, `mul_306` |
| 16 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_cumsum_div_mul_pow_`<br>`sin_slice_split_sub_unsqueeze_view_11` | `triton` | euler0 · suffix | `[50,320] bfloat16` | `[1] int64` | 3.20e-05 | 14.49 | 3.2% | 1.00e-09 (FP32) | 0.000453 | 1.91e-06% | 3.12e-05 | bd-bounded | **2.21 µs** | pointwise Triton fusion: slice_scatter_default_62,<br>copy_66, slice_166, sub_53, mul_309, split_33,<br>convert_element_type_544, view_523, view_522,<br>cos_34, unsqueeze_197, div_66, …(+24) | `slice_scatter_default_62`, `copy_66`, `slice_166`,<br>`sub_53`, `mul_309`, `split_33`,<br>`convert_element_type_544`, `view_523`, `view_522`,<br>`cos_34`, `unsqueeze_197`, `div_66`,<br>`convert_element_type_546`, `unsqueeze_194`,<br>`sub_51`, `add_147`, `unsqueeze_189`,<br>`view_default_9`, `mul_tensor_9`, `iota_default_9`,<br>`unsqueeze_196`, `unsqueeze_195`, `pow_69`,<br>`mul_308`, `convert_element_type_545`, `add_151`,<br>`mul_307`, `iota_34`, `mul_310`, `sin_34`,<br>`slice_scatter_default_63`, `copy_67`, `slice_169`,<br>`add_152`, `mul_311`, `mul_312` |
| 17 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_div_mul_pow_sin_`<br>`slice_split_sub_unsqueeze_view_12` | `triton` | prefill · L0/15 | `[241,320] bfloat16; [1,241] int64` | `[1,241,5,64] float32` | 4.65e-04 | 159.7 | 35.6% | 7.71e-05 (FP32) | 26.48 | 0.112% | 1.66e-01 | bd-bounded | **2.91 µs** | pointwise Triton fusion: slice_scatter_default_3,<br>slice_scatter_default_2, copy_2, slice_6, sub_2,<br>mul_11, split_1, convert_element_type_11, view_5,<br>view_4, cos_1, unsqueeze_11, …(+19) | `slice_scatter_default_3`, `slice_scatter_default_2`,<br>`copy_2`, `slice_6`, `sub_2`, `mul_11`, `split_1`,<br>`convert_element_type_11`, `view_5`, `view_4`,<br>`cos_1`, `unsqueeze_11`, `div_1`,<br>`convert_element_type_13`, `unsqueeze_8`, `sub`,<br>`unsqueeze_10`, `unsqueeze_9`, `pow_3`, `mul_10`,<br>`convert_element_type_12`, `add_3`, `mul_9`,<br>`iota_1`, `mul_12`, `sin_1`, `copy_3`, `slice_9`,<br>`add_4`, `mul_13`, `mul_14` |
| 18 | `triton_poi_fused__to_copy__unsafe_view_`<br>`cat_clone_expand_transpose_unsqueeze_13` | `triton` | prefill · L0/15 | `[1,241,5,64] float32` | `[1,291,15,64] float32` | 1.43e-03 | 798.0 | 178.1% ⚠L2/cache·algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.79 µs** | pointwise Triton fusion: convert_element_type_549,<br>view_527, clone_149, expand_100, unsqueeze_198,<br>permute_264, cat_3, permute_3,<br>convert_element_type_14, permute_262,<br>convert_element_type_547 | `convert_element_type_549`, `view_527`, `clone_149`,<br>`expand_100`, `unsqueeze_198`, `permute_264`,<br>`cat_3`, `permute_3`, `convert_element_type_14`,<br>`permute_262`, `convert_element_type_547` |
| 19 | `extern_kernels.bmm` | `extern` | euler0 · suffix | `[15,50,64] float32 × [15,64,291] float32` | `[15,50,291] float32` | 2.18e-03 | 190.8 | 42.6% | — | — | — | — | bd-bounded | **11.42 µs** | Batched matmul (aten.bmm) via cuBLAS | `bmm_32` |
| 20 | `triton_per_fused__softmax__to_copy_`<br>`bitwise_and_cat_exp_expand_le_mul_`<br>`prepare_softmax_online_scalar_tensor_`<br>`sub_unsqueeze_view_where_14` | `triton` | euler0 · suffix | `[1,241] bool; [1,50] float32; [15,50,291] float32` | `[1,15,50,291] bfloat16` | 1.31e-03 | 312.5 | 69.8% | 1.09e-03 (BF16) | 260.0 | 1.10% | 8.33e-01 | bd-bounded | **4.19 µs** | persistent-reduction Triton fusion: prepare_softmax_online_default_159,<br>where_17, unsqueeze_200, cat_2, expand_99,<br>unsqueeze_184, bitwise_and_1, le_1, unsqueeze_185,<br>unsqueeze_186, full_default_17, mul_313, …(+6) | `prepare_softmax_online_default_159`, `where_17`,<br>`unsqueeze_200`, `cat_2`, `expand_99`,<br>`unsqueeze_184`, `bitwise_and_1`, `le_1`,<br>`unsqueeze_185`, `unsqueeze_186`, `full_default_17`,<br>`mul_313`, `view_531`, `full_default_18`,<br>`convert_element_type_550`, `div_67`,<br>`exp_default_159`, `sub_tensor_159` |
| 21 | `triton_poi_fused_clone_permute_view_15` | `triton` | euler0 · suffix | `[15,50,64] bfloat16` | `[1,50,15,64] bfloat16` | 1.92e-04 | 230.8 | 51.5% | 0 | 0 | — | — | bd-bounded | **0.83 µs** | pointwise Triton fusion: clone_151, permute_270,<br>view_536 | `clone_151`, `permute_270`, `view_536` |
| 22 | `triton_poi_fused__unsafe_view_clone_`<br>`expand_transpose_unsqueeze_view_16` | `triton` | prefill · L0/15 | `[241,320] bfloat16` | `[1,241,5,3,64] bfloat16` | 6.17e-04 | 401.7 | 89.7% | 0 | 0 | — | — | bd-bounded | **1.54 µs** | pointwise Triton fusion: clone_6, expand_1,<br>unsqueeze_13, permute_6, permute_4, view_8, view_7 | `clone_6`, `expand_1`, `unsqueeze_13`, `permute_6`,<br>`permute_4`, `view_8`, `view_7` |
| 23 | `triton_poi_fused__to_copy__unsafe_view_`<br>`clone_expand_transpose_unsqueeze_17` | `triton` | prefill · L0/15 | `[1,241,5,64] float32` | `[1,241,15,64] float32` | 1.23e-03 | 768.8 | 171.6% ⚠L2/cache·algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.60 µs** | pointwise Triton fusion: convert_element_type_16,<br>view_9, clone_5, expand, unsqueeze_12, permute_5,<br>permute_3, convert_element_type_14 | `convert_element_type_16`, `view_9`, `clone_5`,<br>`expand`, `unsqueeze_12`, `permute_5`, `permute_3`,<br>`convert_element_type_14` |
| 24 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_div_mul_pow_sin_`<br>`slice_split_sub_unsqueeze_view_18` | `triton` | prefill · L0/15 | `[241,960] bfloat16; [1,241] int64` | `[1,241,15,64] float32` | 1.39e-03 | 171.7 | 38.3% | 2.31e-04 (FP32) | 28.53 | 0.12% | 1.66e-01 | bd-bounded | **8.10 µs** | pointwise Triton fusion: slice_scatter_default_1,<br>slice_scatter_default, copy, slice_1, sub_1, mul_5,<br>split, convert_element_type_7, view_2, view_1, cos,<br>unsqueeze_7, …(+19) | `slice_scatter_default_1`, `slice_scatter_default`,<br>`copy`, `slice_1`, `sub_1`, `mul_5`, `split`,<br>`convert_element_type_7`, `view_2`, `view_1`, `cos`,<br>`unsqueeze_7`, `div`, `convert_element_type_9`,<br>`unsqueeze_4`, `sub`, `unsqueeze_6`, `unsqueeze_5`,<br>`pow_2`, `mul_4`, `convert_element_type_8`, `add_1`,<br>`mul_3`, `iota`, `mul_6`, `sin`, `copy_1`, `slice_4`,<br>`add_2`, `mul_7`, `mul_8` |
| 25 | `extern_kernels.bmm` | `extern` | prefill · L0/15 | `[15,241,64] float32 × [15,64,241] float32` | `[15,241,241] float32` | 5.34e-03 | 243.3 | 54.3% | — | — | — | — | bd-bounded | **21.95 µs** | Batched matmul (aten.bmm) via cuBLAS | `bmm_32` |
| 26 | `triton_red_fused__softmax__to_copy_`<br>`bitwise_and_exp_le_mul_prepare_softmax_`<br>`online_scalar_tensor_sub_unsqueeze_view_`<br>`where_19` | `triton` | prefill · L0/15 | `[1,241] int64; [1,241] bool; [15,241,241] float32`<br>`; [1,15,241,241] float32` | `[1,15,241,241] bfloat16` | 8.71e-03 | 624.3 | 139.3% ⚠L2/cache·algo-IO≠DRAM | 4.36e-03 (BF16) | 312.5 | 1.32% | 5.00e-01 | bd-bounded | **13.95 µs** | reduction Triton fusion: where, unsqueeze_14,<br>bitwise_and, le, unsqueeze, unsqueeze_1, mul,<br>unsqueeze_2, unsqueeze_3, mul_15, view_13,<br>full_default, …(+5) | `where`, `unsqueeze_14`, `bitwise_and`, `le`,<br>`unsqueeze`, `unsqueeze_1`, `mul`, `unsqueeze_2`,<br>`unsqueeze_3`, `mul_15`, `view_13`, `full_default`,<br>`prepare_softmax_online_default_174`,<br>`convert_element_type_17`, `div_2`,<br>`exp_default_174`, `sub_tensor_174` |
| 27 | `triton_poi_fused_clone_permute_view_20` | `triton` | prefill · L0/15 | `[15,241,64] bfloat16` | `[1,241,15,64] bfloat16` | 9.25e-04 | 825.9 | 184.4% ⚠launch-floor+algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.12 µs** | pointwise Triton fusion: clone_7, permute_11,<br>view_18 | `clone_7`, `permute_11`, `view_18` |
| 28 | `triton_per_fused__to_copy__unsafe_view_`<br>`add_mean_mul_pow_rsqrt_21` | `triton` | prefill · L0/15 | `[241,960] bfloat16; [1,241,960] float32`<br>`; [960] bfloat16` | `[1,241,960] bfloat16` | 1.85e-03 | 837.9 | 187.0% ⚠L2/cache·algo-IO≠DRAM | 1.85e-03 (FP32) | 837.9 | 3.54% | 9.99e-01 | bd-bounded | **2.21 µs** | persistent-reduction Triton fusion: mean_1, pow_4,<br>convert_element_type_23, convert_element_type_22,<br>add_5, view_21, mul_17, convert_element_type_24,<br>mul_16, rsqrt_1, add_6 | `mean_1`, `pow_4`, `convert_element_type_23`,<br>`convert_element_type_22`, `add_5`, `view_21`,<br>`mul_17`, `convert_element_type_24`, `mul_16`,<br>`rsqrt_1`, `add_6` |
| 29 | `triton_poi_fused__unsafe_view_mul_silu_22` | `triton` | prefill · L0/15 | `[1,241,2560] bfloat16` | `[241,2560] bfloat16` | 2.47e-03 | 952.9 | 212.7% ⚠L2/cache·algo-IO≠DRAM | 6.17e-04 (BF16) | 238.0 | 1.00% | 2.50e-01 | bd-bounded | **2.59 µs** | pointwise Triton fusion: mul_18,<br>convert_element_type_28, div_3,<br>convert_element_type_27, view_25, add_7, exp_1,<br>neg, view_27 | `mul_18`, `convert_element_type_28`, `div_3`,<br>`convert_element_type_27`, `view_25`, `add_7`,<br>`exp_1`, `neg`, `view_27` |
| 30 | `triton_per_fused__to_copy__unsafe_view_`<br>`add_mean_mul_pow_rsqrt_23` | `triton` | prefill · L0/15 | `[241,960] bfloat16; [241,960] bfloat16`<br>`; [1,241,960] float32; [960] bfloat16` | `[1,241,960] bfloat16` | 2.32e-03 | 941.6 | 210.2% ⚠L2/cache·algo-IO≠DRAM | 1.85e-03 (FP32) | 750.8 | 3.17% | 7.99e-01 | bd-bounded | **2.46 µs** | persistent-reduction Triton fusion: mean_2, pow_5,<br>convert_element_type_33, add_8, view_29,<br>convert_element_type_22, add_5, view_21, mul_20,<br>convert_element_type_34, mul_19, rsqrt_2, …(+1) | `mean_2`, `pow_5`, `convert_element_type_33`,<br>`add_8`, `view_29`, `convert_element_type_22`,<br>`add_5`, `view_21`, `mul_20`,<br>`convert_element_type_34`, `mul_19`, `rsqrt_2`,<br>`add_9` |
| 31 | `triton_poi_fused__to_copy__unsafe_view_`<br>`transpose_view_24` | `triton` | prefill · L0/15 | `[1,241,5,64] float32` | `[1,241,5,64] float32` | 6.17e-04 | 714.1 | 159.4% ⚠launch-floor+algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **0.86 µs** | pointwise Triton fusion: convert_element_type_571,<br>permute_276, permute_20, view_40, view_39 | `convert_element_type_571`, `permute_276`,<br>`permute_20`, `view_40`, `view_39` |
| 32 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[241,320] float32 × [320,320] float32` | `[241,320] float32` | 1.03e-03 | 157.8 | 35.2% | 2D 4.94e-02 (FP32) | 7567.4 | 31.9% | 48.1 | bd-bounded | **6.53 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 33 | `triton_poi_fused__unsafe_view_clone_`<br>`expand_unsqueeze_view_25` | `triton` | euler0 · expert L0/16 | `[241,320] float32` | `[1,241,5,3,64] float32` | 1.23e-03 | 854.2 | 190.7% ⚠launch-floor+algo-IO≠DRAM | 0 | 0 | — | — | bd-bounded | **1.44 µs** | pointwise Triton fusion: clone_154, expand_107,<br>unsqueeze_206, view_560, view_559 | `clone_154`, `expand_107`, `unsqueeze_206`,<br>`view_560`, `view_559` |
| 34 | `triton_per_fused__to_copy__unsafe_view_`<br>`add_addmm_mean_mul_pow_rsqrt_view_26` | `triton` | euler0 · suffix | `[50,720] bfloat16 × [720] float32 × [50,720] float32`<br>`× [720] bfloat16` | `[1,50,720] bfloat16` | 2.92e-04 | 182.5 | 40.7% | 2.88e-04 (FP32) | 180.0 | 0.759% | 9.85e-01 | bd-bounded | **1.60 µs** | persistent-reduction Triton fusion: mean_34,<br>pow_70, convert_element_type_556,<br>convert_element_type_555, add_153, view_539,<br>view_517, add_tensor_28, mul_315,<br>convert_element_type_557, mul_314, rsqrt_34, …(+1) | `mean_34`, `pow_70`, `convert_element_type_556`,<br>`convert_element_type_555`, `add_153`, `view_539`,<br>`view_517`, `add_tensor_28`, `mul_315`,<br>`convert_element_type_557`, `mul_314`, `rsqrt_34`,<br>`add_154` |
| 35 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[50,720] bfloat16 × [720,2048] bfloat16` | `[50,2048] bfloat16` | 3.23e-03 | 221.4 | 49.4% | 2D 1.47e-01 (BF16) | 10074.0 | 10.6% | 45.7 | bd-bounded | **14.59 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 36 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[50,720] bfloat16 × [720,2048] bfloat16` | `[50,2048] bfloat16` | 3.23e-03 | 281.2 | 62.8% | 2D 1.47e-01 (BF16) | 12796.0 | 13.5% | 45.7 | bd-bounded | **11.49 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 37 | `triton_poi_fused__unsafe_view_mul_silu_27` | `triton` | euler0 · expert L0/16 | `[1,50,2048] bfloat16` | `[50,2048] bfloat16` | 4.10e-04 | 376.8 | 84.1% | 1.02e-04 (BF16) | 93.75 | 0.396% | 2.50e-01 | bd-bounded | **1.09 µs** | pointwise Triton fusion: mul_316,<br>convert_element_type_561, div_68,<br>convert_element_type_560, view_543, add_155,<br>exp_34, neg_17, view_545 | `mul_316`, `convert_element_type_561`, `div_68`,<br>`convert_element_type_560`, `view_543`, `add_155`,<br>`exp_34`, `neg_17`, `view_545` |
| 38 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[50,2048] bfloat16 × [2048,720] bfloat16` | `[50,720] bfloat16` | 3.23e-03 | 318.4 | 71.1% | 2D 1.47e-01 (BF16) | 14491.3 | 15.3% | 45.7 | bd-bounded | **10.14 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 39 | `triton_per_fused__to_copy__unsafe_view_`<br>`add_addmm_mean_mul_pow_rsqrt_view_28` | `triton` | euler0 · expert L0/16 | `[50,720] bfloat16 × [50,720] bfloat16`<br>`× [720] float32 × [50,720] float32 × [720] bfloat16` | `[1,50,720] bfloat16` | 3.64e-04 | 316.0 | 70.5% | 2.88e-04 (FP32) | 250.0 | 1.05% | 7.91e-01 | bd-bounded | **1.15 µs** | persistent-reduction Triton fusion: mean_35,<br>pow_71, convert_element_type_566, add_156,<br>view_547, convert_element_type_555, add_153,<br>view_539, view_517, add_tensor_28, mul_318,<br>convert_element_type_567, …(+3) | `mean_35`, `pow_71`, `convert_element_type_566`,<br>`add_156`, `view_547`, `convert_element_type_555`,<br>`add_153`, `view_539`, `view_517`, `add_tensor_28`,<br>`mul_318`, `convert_element_type_567`, `mul_317`,<br>`rsqrt_35`, `add_157` |
| 40 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[50,720] bfloat16 × [720,960] bfloat16` | `[50,960] bfloat16` | 1.55e-03 | 232.9 | 52.0% | 2D 6.91e-02 (BF16) | 10381.6 | 11.0% | 44.6 | bd-bounded | **6.66 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 41 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_cumsum_div_mul_pow_`<br>`sin_slice_split_sub_unsqueeze_view_29` | `triton` | euler0 · expert L0/16 | `[50,960] bfloat16; [1] int64` | `[1,1] int64` | 9.60e-05 | 24.79 | 5.5% | 1.00e-09 (FP32) | 0.000258 | 1.09e-06% | 1.04e-05 | bd-bounded | **3.87 µs** | pointwise Triton fusion: slice_scatter_default_64,<br>copy_68, slice_173, sub_56, mul_321, split_34,<br>convert_element_type_572, view_552, view_551,<br>cos_35, unsqueeze_204, div_69, …(+25) | `slice_scatter_default_64`, `copy_68`, `slice_173`,<br>`sub_56`, `mul_321`, `split_34`,<br>`convert_element_type_572`, `view_552`, `view_551`,<br>`cos_35`, `unsqueeze_204`, `div_69`,<br>`convert_element_type_574`, `unsqueeze_201`,<br>`sub_55`, `sub_51`, `add_147`, `unsqueeze_189`,<br>`view_default_9`, `mul_tensor_9`, `iota_default_9`,<br>`unsqueeze_203`, `unsqueeze_202`, `pow_72`,<br>`mul_320`, `convert_element_type_573`, `add_158`,<br>`mul_319`, `iota_35`, `mul_322`, `sin_35`,<br>`slice_scatter_default_65`, `copy_69`, `slice_176`,<br>`add_159`, `mul_323`, `mul_324` |
| 42 | `triton_poi_fused__to_copy__unsafe_view_`<br>`add_arange_copy_cos_div_mul_pow_sin_`<br>`slice_split_sub_transpose_unsqueeze_view_30` | `triton` | prefill · L0/15 | `[241,320] bfloat16; [1,241] int64`<br>`; [1,241,5,64] bfloat16; [1,241,5,64] float32; —`<br>`; [1,241,5,64] float32; [1,241,5,64] float32`<br>`; [1,241,5,64] float32; [1,241,5,64] float32`<br>`; [1,241,5,64] float32; [1,241,5,64] float32`<br>`; [1,241,5,64] float32` | `[1,241,5,64] float32` | 3.09e-03 | 665.9 | 148.6% ⚠L2/cache·algo-IO≠DRAM | 7.71e-05 (FP32) | 16.62 | 0.0701% | 2.50e-02 | bd-bounded | **4.64 µs** | pointwise Triton fusion: slice_scatter_default_7,<br>slice_scatter_default_6, copy_6, slice_16, sub_5,<br>mul_29, split_3, convert_element_type_45, view_37,<br>view_36, cos_3, unsqueeze_22, …(+41) | `slice_scatter_default_7`, `slice_scatter_default_6`,<br>`copy_6`, `slice_16`, `sub_5`, `mul_29`, `split_3`,<br>`convert_element_type_45`, `view_37`, `view_36`,<br>`cos_3`, `unsqueeze_22`, `div_5`,<br>`convert_element_type_47`, `unsqueeze_19`, `sub`,<br>`unsqueeze_21`, `unsqueeze_20`, `pow_7`, `mul_28`,<br>`convert_element_type_46`, `add_12`, `mul_27`,<br>`iota_3`, `mul_30`, `sin_3`, `copy_7`, `slice_19`,<br>`add_13`, `mul_31`, `mul_32`,<br>`convert_element_type_48`,<br>`convert_element_type_570`, `permute_275`,<br>`permute_19`, `convert_element_type_1032`,<br>`permute_519`, `convert_element_type_1494`,<br>`permute_763`, `convert_element_type_1956`,<br>`permute_1007`, `convert_element_type_2418`,<br>`permute_1251`, `convert_element_type_2880`,<br>`permute_1495`, `convert_element_type_3342`,<br>`permute_1739`, `convert_element_type_3804`,<br>`permute_1983`, `convert_element_type_4266`,<br>`permute_2227`, `convert_element_type_4728`,<br>`permute_2471` |
| 43 | `extern_kernels.mm` | `extern` | euler0 · expert L0/16 | `[241,320] float32 × [320,320] float32` | `[241,320] float32` | 1.03e-03 | 157.8 | 35.2% | 2D 4.94e-02 (FP32) | 7567.4 | 31.9% | 48.1 | bd-bounded | **6.53 µs** | GEMM matmul (aten.mm / linear without fused bias)<br>via cuBLAS | `mm_default_29` |
| 44 | `triton_poi_fused_add_addmm_mul_view_47` | `triton` | euler0 · update | `[1,50,32] float32 × [1,50,32] float32 × [32] float32` | `[50,32] float32` | 1.93e-05 | — | — | 1.60e-06 (FP32) | — | — | 8.28e-02 | bd-bounded | — | pointwise Triton fusion: add_277, mul_541,<br>view_1047, add_tensor_27 | `add_277`, `mul_541`, `view_1047`, `add_tensor_27` |
| 45 | `triton_poi_fused__unsafe_view_cat_slice_`<br>`transpose_view_70` | `triton` | tail | `[50,32] float32` | `[50,32] float32` | 1.28e-05 | — | — | — | — | — | — | bd-bounded | — | pointwise Triton fusion: cat_175, slice_1522,<br>permute_2459, view_5350, view_5349 | `cat_175`, `slice_1522`, `permute_2459`, `view_5350`,<br>`view_5349` |
| 46 | `triton_poi_fused__to_copy_cat_slice_`<br>`transpose_71` | `triton` | tail | `[50,32] float32` | `[50,32] float32` | 1.28e-05 | — | — | 0 | — | — | — | bd-bounded | — | pointwise Triton fusion: cat_174, slice_1521,<br>permute_2458, convert_element_type_4705 | `cat_174`, `slice_1521`, `permute_2458`,<br>`convert_element_type_4705` |

---

## How to rebuild

```bash
export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home
export SMOKE_COMPILE_MODE=reduce-overhead
python src/smolvla_compile_fused_dump.py   # cold compile + debug dumps
python src/smolvla_compile_fused_regen_md.py
python src/smolvla_compile_fused_fill_io.py       # infer missing Input/Output
python src/smolvla_compile_fused_annotate_stage.py  # stage col; collapse G8 to L0/euler0
# python src/smolvla_compile_fused_annotate_stage.py --full  # keep full ×10×16 unrolled G8
python src/smolvla_compile_fused_table_metrics.py  # fill IO/GFLOPs/AI/bottleneck
python src/smolvla_compile_fused_table_timing.py   # BD/util/GFLOPs-sec/GPU time from nsys
python src/smolvla_compile_fused_table_wrap.py     # soft-wrap long name/input/meaning/fused
```

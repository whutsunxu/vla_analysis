# SmolVLA Aten Operator List (GPU)

This document mirrors `SmolVLA_Operator_List.md` for the same `lerobot/smolvla_base` inference path, but replaces algorithmic FLOP accounting with **what actually runs on the GPU**, verified against:

| Artifact | Role |
|---|---|
| `doc/gpu/smolvla_aten_chrono.json` | aten op dispatch record |
| `doc/gpu/smolvla_aten_chrono.log` | aten op launch log |

**Reusable skill (how to rebuild this file for another inference path):** **§7**.

---

## 0. Notation, fixed dimensions, and GPU datatype rules

### 0.1 Chronological ATen capture (`TorchDispatchMode`)

To list **ATen-level** operators in **call order** (with input shapes + dtypes), this repo uses `torch.utils._python_dispatch.TorchDispatchMode` via `src/smolvla_aten_profile.py`, writing:

- `doc/gpu/smolvla_aten_chrono.json` — structured per-stage event lists
- `doc/gpu/smolvla_aten_chrono.log` — human-readable chronological log

**Method (summary):**

1. Load `lerobot/smolvla_base` on CUDA; warmup one Stage 0→4 path.
2. For each stage separately, enter a `TorchDispatchMode` that **appends** one row per dispatcher call (no `key_averages` aggregation).
3. Each row records: sequence index, `aten::*` name, input tensor **shapes**, input **dtypes**.
4. Stage 1 and Stage 3 are captured for **one** Euler step (ops repeat `M=10` times per chunk).
5. Optional noise filter drops pure alloc/meta ops (`empty` / `empty_strided` / `alias` / `detach`) so the log stays readable.

Measured counts on RTX 5060 Ti (filtered): Stage 0 ≈901, Stage 2 ≈1890, Stage 1 ≈27, Stage 3 ≈1693, Stage 4 ≈5 events.

**Why not `TorchFunctionMode` / `profiler.key_averages`?**
`TorchFunctionMode` often misses ops that go through the C++ ATen path (observed: only a few `conv2d`). `key_averages(group_by_input_shape=True)` **aggregates** by `(name, shapes)` and sorts by count — not launch order.

**Potential risks / limits:**

| Risk | What it means |
|---|---|
| **Not every CUDA kernel** | One `aten::` (e.g. fused Flash / cuDNN / CUTLASS epilogue) can map to **many** Nsight kernels; chrono ATen ≠ kernel ledger |
| **Fusion collapses detail** | Softmax / mask / scale inside Flash appear as fewer aten rows than in the paper-style op list |
| **Custom extensions** | Code that never hits the ATen dispatcher is invisible |
| **Intentional skips** | Filtered `empty*` / `alias` / `detach` are absent from the log (re-enable in the script if needed) |
| **Compiled / CUDA-graph paths** | `torch.compile` / graphs may change or bypass normal dispatch |
| **Stage 1/3 = 1 Euler step** | Full-chunk counts ≈ listed × `M` for those stages |
| **No GPU timing here** | This capture is for **name / shape / dtype / order**; times stay in Nsight / `smolVLA_profiling.md` |
| **Overhead** | Dispatch wrapping slows the run; use for operator inventory, not latency measurement |

**Use together:** ATen chrono list = *what PyTorch called*; Nsight = *what the GPU ran*. Cross-check shapes/dtypes here against kernel fingerprints in §3.3 / `smolVLA_kernel_gpu_list.md`.

---

## 1. Stage 0 — prefix embedding (GPU)

**Function:** three cameras + language + state → prefix `P` `(B,241,960)`.

**Source for this section:** chronological ATen log `doc/gpu/smolvla_aten_chrono.log` / `.json` covers **`embed_prefix`** (Stage 0 body, **901** launches, §0.1). **§1.0** documents `prepare_images` (runs immediately before `embed_prefix`); it is **not** in that chrono file yet — shapes/dtypes/kernels from Nsight + prior Stage‑0 tables. From §1.1 on, tables are **fine-grained** `aten::*` in call order. Semantic labels (upsample / “Flash” / “PixelShuffle”) still annotate many→one or renamed mappings.

**How to read the mapping**

| Idea | Detail |
|---|---|
| 1 semantic → N aten | e.g. “position-ID build” ≈ `#7–39`; “PixelShuffle” ≈ `#287–292` view/permute/reshape (no `aten::pixel_shuffle`) |
| 1 aten → 1 semantic | e.g. `aten::scaled_dot_product_attention` ≡ fused self-attn (Nsight: Flash kernel) |
| Preprocess (§1.0) | `prepare_images`: bilinear upsample / `2x−1` / FP32→BF16 **before** chrono `#1`; embed log starts at `(1,3,512,512)` |
| Repeats | Per-camera patch+ViT+connector **×3** (≈`#1–297`, `#298–594`, `#595–891`); each ViT **block** ×`L_v=12`/camera → **36** blocks/chunk |

Nsight kernel names stay in §6; §1.1+ prioritizes **aten identity + shapes + dtypes**.

### 1.0 `prepare_images` — resize / norm / cast (before `embed_prefix`; ×`C=3`)

Called as `policy.prepare_images(batch)` **before** `model.embed_prefix(...)`. Not present in `smolvla_aten_chrono.*` (that capture wrapped only `embed_prefix`). Recorded here from Nsight + Stage‑0 operator inventory so the Stage‑0 story is complete end-to-end.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | **`upsample_bilinear2d`** | `[B,3,256,256] float32` | `[B,3,512,512] float32` | Bilinear upsample 256²→512² (Nsight `upsample_bilinear2d_out_frame`; ×`C=3`) |
| 2 | `mul` / `add` | `[B,3,512,512] float32` | `[B,3,512,512] float32` | Scale/bias `2x − 1` → ~[−1, 1] |
| 3 | `_to_copy` / `to` | `[B,3,512,512] float32` | `[B,3,512,512] bfloat16` | Cast → vision dtype (feeds patch `conv2d`; chrono `#4` sees **bf16**) |

**Repeat:** each of the three cameras. After this block, `embed_prefix` chrono **`#1`** is `aten::to` on `[1,3,512,512] float32` then patch path in §1.1.

### 1.1 Per camera — image enter + patch embed + position (template; ×`C=3`)

First camera: **`#1–45`**. Cameras 2/3 repeat (next starts `#298`, `#595`). *Assumes §1.0 already produced 512² inputs.* Output shapes inferred from the next aten’s inputs / known ViT layout (chrono records inputs only).

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | `to` | `[1,3,512,512] float32` | `[1,3,512,512] float32` | Image on device / dtype path |
| 2 | `ones` | (size args) | `[1,32,32] float32` | Patch-grid helper |
| 3 | `to` | `[1,32,32] float32` | `[1,32,32] float32` | Mask on device |
| 4 | **`conv2d`** | `[1,3,512,512] bf16` × `[768,3,16,16] bf16` + `[768] bf16` | `[1,768,32,32] bf16` | Patch Conv `3→768`, k/s=16 |
| 5 | `flatten` | `[1,768,32,32] bf16` | `[1,768,1024] bf16` | Spatial → token axis |
| 6 | `transpose` | `[1,768,1024] bf16` | `[1,1024,768] bf16` | Tokens `(B,1024,768)` |
| 7 | `arange` | (size args) | 1‑D index range | Position-ID: start range |
| 8 | `full` | (size / fill args) | filled tensor | Position-ID: constant fill |
| 9 | `select` | `[1,32,32] bool` | `[1,32] bool` | Position-ID: pick mask row/plane |
| 10 | `sum` | `[1,32] bool` | `[1] int64` | Position-ID: count valid (H) |
| 11 | `select` | `[1,32,32] bool` | `[1,32] bool` | Position-ID: pick mask col/plane |
| 12 | `sum` | `[1,32] bool` | `[1] int64` | Position-ID: count valid (W) |
| 13 | `reciprocal` | `[1] int64` | `[1] float32` | Position-ID: 1/count (H) |
| 14 | `mul` | `[1] float32` | `[1] float32` | Position-ID: scale (H) |
| 15 | `reciprocal` | `[1] int64` | `[1] float32` | Position-ID: 1/count (W) |
| 16 | `mul` | `[1] float32` | `[1] float32` | Position-ID: scale (W) |
| 17 | `arange` | (size args) | `[32] float32` | Position-ID: H coords |
| 18 | `arange` | (size args) | `[32] float32` | Position-ID: W coords |
| 19 | `unsqueeze` | `[32] float32` | `[1,32]` or `[32,1]` float32 | Position-ID: broadcast prep (H) |
| 20 | `unsqueeze` | `[1] float32` | `[1,1] float32` | Position-ID: scale broadcast (H) |
| 21 | `mul` | `[1,32] float32` × `[1,1] float32` | `[1,32] float32` | Position-ID: scaled H |
| 22 | `unsqueeze` | `[32] float32` | `[1,32]` or `[32,1]` float32 | Position-ID: broadcast prep (W) |
| 23 | `unsqueeze` | `[1] float32` | `[1,1] float32` | Position-ID: scale broadcast (W) |
| 24 | `mul` | `[1,32] float32` × `[1,1] float32` | `[1,32] float32` | Position-ID: scaled W |
| 25 | `clamp` | `[1,32] float32` | `[1,32] float32` | Position-ID: clamp H |
| 26 | `clamp` | `[1,32] float32` | `[1,32] float32` | Position-ID: clamp W |
| 27 | `to` | `[1,32] float32` | `[1,32] bf16` | Position-ID: dtype for bucketize (H) |
| 28 | `to` | `[1,32] float32` | `[1,32] bf16` | Position-ID: dtype for bucketize (W) |
| 29 | `bucketize` | `[1,32] bf16`, boundaries `[31] float32` | `[1,32] int64` | Position-ID: bin H → int |
| 30 | `bucketize` | `[1,32] bf16`, boundaries `[31] float32` | `[1,32] int64` | Position-ID: bin W → int |
| 31 | `unsqueeze` | `[1,32] int64` | `[1,32,1] int64` | Position-ID: H for outer combine |
| 32 | `mul` | `[1,32,1] int64` | `[1,32,1] int64` | Position-ID: × stride (H) |
| 33 | `unsqueeze` | `[1,32] int64` | `[1,1,32] int64` | Position-ID: W for outer combine |
| 34 | `add` | `[1,32,1] int64` + `[1,1,32] int64` | `[1,32,32] int64` | Position-ID: 2‑D ids |
| 35 | `reshape` | `[1,32,32] int64` | `[1,1024] int64` | Position-ID: flat token ids |
| 36 | `view` | `[1,32,32] bool` | `[1,1024] bool` | Position-ID: flat mask |
| 37 | `index` | ids `[1,1024] int64`, mask `[1,1024] bool` | selected ids | Position-ID: gather under mask |
| 38 | `view` | `[1,32,32] bool` | `[1,1024] bool` | Position-ID: mask again |
| 39 | `index_put_` | dest `[1,1024] int64`, mask, src `[1024] int64` | **`[1,1024] int64`** | Position-ID: write final ids |
| 40 | **`embedding`** | table `[1024,768] bf16`, ids `[1,1024] int64` | `[1,1024,768] bf16` | Position-embedding gather |
| 41 | `add` | `[1,1024,768] bf16` + `[1,1024,768] bf16` | `[1,1024,768] bf16` | Patch + position |
| 42 | `view` | `[1,32,32] bool` | `[1,1024] bool` | Validity mask flatten |
| 43 | `to` | `[1,1024] bool` | `[1,1024] bool` | Mask dtype/device |
| 44 | `all` | `[1,1024] bool` | `[] bool` | All-valid check |
| 45 | `is_nonzero` | `[] bool` | Python bool | Control-flow predicate |

**vs prior semantic patch table:** upsample/scale/cast in **§1.0**; pos-ID is **`#7–39`** (one row per aten above).

### 1.2 One ViT encoder block (template; ×`C×L_v = 36`)

Example: first block **`#46–65`**. Rhythm repeats 12×3 cameras (`scaled_dot_product_attention` **36**, `gelu` **36**, `layer_norm` **75** = 36+36+3 post-LN).

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 46 | **`layer_norm`** | `[1,1024,768] bf16`, w/b `[768] bf16` | `[1,1024,768] bf16` | Pre-attn LN |
| 47 | **`linear`** | `[1,1024,768] bf16` × `[768,768] bf16` + bias | `[1,1024,768] bf16` | **Q** `768→768` |
| 48 | `view` | `[1,1024,768] bf16` | `[1,1024,12,64] bf16` | Split heads (Q) |
| 49 | `transpose` | `[1,1024,12,64] bf16` | `[1,12,1024,64] bf16` | Heads-first (Q) |
| 50 | **`linear`** | same as Q | `[1,1024,768] bf16` | **K** |
| 51–52 | `view`, `transpose` | as Q path | `[1,12,1024,64] bf16` | Reshape heads (K) |
| 53 | **`linear`** | same as Q | `[1,1024,768] bf16` | **V** |
| 54–55 | `view`, `transpose` | as Q path | `[1,12,1024,64] bf16` | Reshape heads (V) |
| 56 | **`scaled_dot_product_attention`** | Q,K,V `[1,12,1024,64] bf16` | `[1,12,1024,64] bf16` | Fused self-attn (**Flash**) |
| 57 | `transpose` | `[1,12,1024,64] bf16` | `[1,1024,12,64] bf16` | Undo heads-first |
| 58 | `reshape` | `[1,1024,12,64] bf16` | `[1,1024,768] bf16` | Concat heads |
| 59 | **`linear`** | `[1,1024,768] bf16` × `[768,768] bf16` | `[1,1024,768] bf16` | Attn-out |
| 60 | `add` | two `[1,1024,768] bf16` | `[1,1024,768] bf16` | Residual 1 |
| 61 | **`layer_norm`** | `[1,1024,768] bf16`, w/b `[768]` | `[1,1024,768] bf16` | Pre-MLP LN |
| 62 | **`linear`** | `[1,1024,768] bf16` × `[3072,768] bf16` | `[1,1024,3072] bf16` | MLP up `768→3072` |
| 63 | **`gelu`** | `[1,1024,3072] bf16` | `[1,1024,3072] bf16` | GELU |
| 64 | **`linear`** | `[1,1024,3072] bf16` × `[768,3072] bf16` | `[1,1024,768] bf16` | MLP down `3072→768` |
| 65 | `add` | two `[1,1024,768] bf16` | `[1,1024,768] bf16` | Residual 2 |

Next block starts with another `layer_norm` (e.g. `#66`). After layer 12: post-`layer_norm` then connector.

**vs old §1.3:** same math; name differs (`scaled_dot_product_attention` vs “Flash”); each Linear/LN/GELU is an explicit aten row.

### 1.3 Per camera — rearrange + connector (×`C=3`)

First camera: **`#286–297`**. Same at `#588+`, `#885+`.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 286 | `layer_norm` | `[1,1024,768] bf16`, w/b `[768]` | `[1,1024,768] bf16` | ViT `post_layernorm` |
| 287 | `view` | `[1,1024,768] bf16` | `[1,32,32,768] bf16` | Spatial restore |
| 288 | `view` | `[1,32,32,768] bf16` | `[1,32,8,3072] bf16` * | PixelShuffle prep |
| 289 | `permute` | `[1,32,8,3072] bf16` | `[1,8,32,3072] bf16` | |
| 290 | `reshape` | `[1,8,32,3072] bf16` | `[1,8,8,12288] bf16` * | |
| 291 | `permute` | `[1,8,8,12288] bf16` | `[1,8,8,12288] bf16` | layout tweak |
| 292 | `reshape` | `[1,8,8,12288] bf16` | **`[1,64,12288] bf16`** | Flat connector tokens |
| 293 | **`linear`** | `[1,64,12288] bf16` × `[960,12288] bf16` | `[1,64,960] bf16` | Connector `12288→960` |
| 294–295 | `lift_fresh`, `mul` | `[1,64,960] bf16` × √960 | `[1,64,960] bf16` | Scale |
| 296–297 | `unsqueeze`, `expand` | camera mask bool | `[1,64] bool` | Expand camera mask |

\* Intermediate layouts from chrono input chain (`#288–292`); exact permute axes follow Vision connector PixelShuffle path.

### 1.4 Language + state + prefix assembly (once / chunk)

Chrono tail **`#892–901`**.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 892 | **`embedding`** | table `[49280,960] bf16`, ids `[1,48] int64` | `[1,48,960] bf16` | Language-token embed |
| 893 | `mul` | `[1,48,960] bf16` | `[1,48,960] bf16` | ×`√960` |
| 894 | **`linear`** | `[1,32] float32` × `[960,32] float32` + bias | `[1,960] float32` | State Linear `32→960` (`gemvx`) |
| 895 | `unsqueeze` | `[1,960] float32` | `[1,1,960] float32` | State as length‑1 token |
| 896 | `ones` | (size) | mask for state token | |
| 897 | **`cat`** | 3×`[1,64,960] bf16` + `[1,48,960] bf16` + `[1,1,960] float32` | **`[1,241,960]`** (promoted; see note) | Concat → prefix `P` |
| 898 | `cat` | 3×`[1,64] bool` + `[1,48] bool` + `[1,1] bool` | `[1,241] bool` | Pad / validity mask |
| 899–901 | `lift_fresh`, `unsqueeze`, `expand` | `[241] bool` | `[1,241] bool` | Attn-group mask |

**Note on `#897`:** chrono inputs mix bf16 vision/lang with **fp32** state; runtime `cat` promotes — downstream Stage‑2 prefill uses prefix embeds as recorded in stage tensors (`prefix_embs` often **float32** after assembly).

**vs old §1.5–1.7:** tokenize stays CPU (not in GPU chrono); embed/state/cat match.

### 1.5 Stage-0 reading

- **901** aten launches in `embed_prefix` (filtered); not 1:1 with Nsight kernels (views may have no kernel; Flash is one kernel under SDPA).
- **Match caveats:** (1) §1.0 preprocess is Nsight-backed (not yet in `smolvla_aten_chrono`); (2) “PixelShuffle” ≠ one aten; (3) fused attn = **`aten::scaled_dot_product_attention`**; (4) many helpers were omitted from the old semantic tables; (5) **Output** column is inferred (chrono logs inputs only).
- Wall Stage 0 ≈ **23.9 ms**. Nsight GEMM/Flash Avgs: §6 (Q/K/V/out ~1.141 ms each/chunk; MLP up/down 4.313/5.472; connector 0.189; state `gemvx` ~2.2 µs).

---

## 2. Stage 1 — action-suffix embedding (GPU)

**Function:** `x_t` + flow time `t` → `U_t` `(B,50,720)`, **M=10** times per chunk.

**Source:** chronological ATen log Stage **`1_suffix_embed`** — **27** launches for **one** Euler step (§0.1). Full chunk ≈ ×`M=10`. Tables below are fine-grained `aten::*` in call order (same style as §1). Output shapes inferred (chrono logs inputs only).

### 2.1 One Euler step — full path (`#1–27`; ×`M=10`/chunk)

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | **`linear`** | `[1,50,32] float32 × [720,32] float32 × [720] float32` | `[1,50,720] float32` | **Action-in Linear** `32→720` |
| 2 | `linspace` | `(size/scalar args)` | `[360] float64` | Time-emb: linspace freqs |
| 3 | `pow` | `[360] float64` | `[360] float64` | Time-emb: period powers |
| 4 | `mul` | `[360] float64` | `[360] float64` | Time-emb: scale |
| 5 | `reciprocal` | `[360] float64` | `[360] float64` | Time-emb: reciprocal periods |
| 6 | `mul` | `[360] float64` | `[360] float64` | Time-emb: scale |
| 7 | `mul` | `[360] float64` | `[360] float64` | Time-emb: scale |
| 8 | `mul` | `[360] float64` | `[360] float64` | Time-emb: scale |
| 9 | `unsqueeze` | `[360] float64` | `[1,360] float64` | Time-emb: unsqueeze freqs |
| 10 | `unsqueeze` | `[1] float32` | `[1,1] float32` | Time-emb: unsqueeze `t` |
| 11 | `mul` | `[1,360] float64 × [1,1] float32` | `[1,360] float64` | Time-emb: `t` × freqs |
| 12 | `sin` | `[1,360] float64` | `[1,360] float64` | Time-emb: **sin** (FP64) |
| 13 | `cos` | `[1,360] float64` | `[1,360] float64` | Time-emb: **cos** (FP64) |
| 14 | `cat` | `[1,360] float64 × [1,360] float64` | `[1,720] float64` | Time-emb: cat sin‖cos → 720 |
| 15 | `to` | `[1,720] float64` | `[1,720] float32` | Time-emb: **cast** float64→float32 |
| 16 | `unsqueeze` | `[1,720] float32` | `[1,1,720] float32` | Unsqueeze time emb |
| 17 | `expand_as` | `[1,1,720] float32 × [1,50,720] float32` | `[1,50,720] float32` | Broadcast time over horizon `50` |
| 18 | `cat` | `[1,50,720] float32 × [1,50,720] float32` | `[1,50,1440] float32` | Concat action ‖ time → 1440 |
| 19 | **`linear`** | `[1,50,1440] float32 × [720,1440] float32 × [720] float32` | `[1,50,720] float32` | **Fusion mid Linear** `1440→720` |
| 20 | **`silu`** | `[1,50,720] float32` | `[1,50,720] float32` | **SiLU** |
| 21 | **`linear`** | `[1,50,720] float32 × [720,720] float32 × [720] float32` | `[1,50,720] float32` | **Fusion out Linear** `720→720` |
| 22 | `ones` | `(size/scalar args)` | `[1,50,720] float32 (new)` | Suffix ones helper |
| 23 | `cat` | `[1,50,720] float32` | `[1,50,720] float32` | Pack suffix emb |
| 24 | `cat` | `[1,50] bool` | `[1,50] bool` | Suffix pad/attn mask |
| 25 | `lift_fresh` | `[50] float32` | `[50] float32` | Attn-group mask source |
| 26 | `unsqueeze` | `[50] float32` | `[1,50] float32` | Unsqueeze mask |
| 27 | `expand` | `[1,50] float32` | `[1,50] float32` | Expand attn-group mask |

### 2.2 Stage-1 reading

- **27** aten launches / Euler step (filtered); **×10** ≈ **270**/chunk.
- Time embedding is **FP64** through `sin`/`cos`, then **`to` → float32** before fusion Linears (all FP32).
- Nsight GEMM /chunk (from prior inventory): action-in **0.031 ms**, fusion mid **0.391 ms**, fusion out **0.153 ms** *[med]*; SiLU **0.012 ms**; FP64 sin+cos **0.067 ms**.
- Wall Stage 1 is small vs Stage 0/3; GEMM budget **30** FP32 Linears /chunk (`10+10+10`).

---

## 3. Stage 2 — prefix prefill / KV cache (GPU)

**Function:** 16-layer VLM over prefix `P` `(B,241,960)`; build prefix KV.

**Source:** chrono Stage **`2_prefill`** — **1890** launches /chunk (`L_p=16` inside). Structure: mask/positions **`#1–10`**, then **16×** layer template (**117** aten each, silu spacing), then final RMSNorm **`#1883–1890`**. Eager attention (not Flash): `matmul` + `where` + `softmax`.

### 3.1 Prefix masks and positions (`#1–10`)

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | `cumsum` | `[1,241] bool` | `[1,241] int64` | Position ids via cumsum on pad mask |
| 2 | `unsqueeze` | `[1,241] int64` | `[1,241] int64` | Unsqueeze positions (Q side) |
| 3 | `unsqueeze` | `[1,241] int64` | `[1,1,241] int64` | Unsqueeze positions (K side) |
| 4 | `le` | `[1,1,241] int64 × [1,241,1] int64` | `[1,1,241] bool` | Causal/prefix `le` on positions |
| 5 | `unsqueeze` | `[1,241] bool` | `[1,241] bool` | Unsqueeze pad mask |
| 6 | `unsqueeze` | `[1,241] bool` | `[1,1,241] bool` | Unsqueeze pad mask |
| 7 | `mul` | `[1,1,241] bool × [1,241,1] bool` | `[1,1,241] bool` | Outer pad-mask product |
| 8 | `__and__` | `[1,241,241] bool × [1,241,241] bool` | `[1,241,241] bool` | Combine causal ∧ pad → attn mask `[1,241,241]` |
| 9 | `cumsum` | `[1,241] bool` | `[1,241] int64` | Cumsum (positions path) |
| 10 | `sub` | `[1,241] int64` | `[1,241] int64` | Adjust position ids |

### 3.2 One VLM prefill layer (template; ×`L_p=16`)

Example: first layer **`#11–127`**. Next layers start every **+117** (layer‑2 Q Linear at `#137`, …). Softmax is standalone FP32; **`QKᵀ`** is FP32 `matmul` (Nsight: `magma_sgemmEx`); **`A·V`** is bf16.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 11 | `to` | `[1,241,960] float32` | `[1,241,960] float32` | Cast act → FP32 for RMSNorm |
| 12 | `pow` | `[1,241,960] float32` | `[1,241,960] float32` | RMSNorm: `x²` |
| 13 | `mean` | `[1,241,960] float32` | `[1,241,1] float32` | RMSNorm: mean |
| 14 | `add` | `[1,241,1] float32` | `[1,241,1] float32` | RMSNorm: +eps |
| 15 | `rsqrt` | `[1,241,1] float32` | `[1,241,1] float32` | RMSNorm: rsqrt |
| 16 | `mul` | `[1,241,960] float32 × [1,241,1] float32` | `[1,241,960] float32` | RMSNorm: × rsqrt |
| 17 | `to` | `[1,241,960] float32` | `[1,241,960] float32 (cast)` | Cast normalized |
| 18 | `mul` | `[960] bf16 × [1,241,960] float32` | `[1,241,960] float32` | RMSNorm: × weight |
| 19 | `to` | `[1,241,960] float32` | `[1,241,960] bf16` | Cast → bf16 for projections |
| 20 | **`linear`** | `[1,241,960] bf16 × [960,960] bf16` | `[1,241,960] bf16` | **Q Linear** `960→960` |
| 21 | `view` | `[1,241,960] bf16` | `[1,241,960] bf16` | Reshape Q → heads |
| 22 | **`linear`** | `[1,241,960] bf16 × [320,960] bf16` | `[1,241,320] bf16` | **K Linear** `960→320` |
| 23 | `view` | `[1,241,320] bf16` | `[1,241,960] bf16` | Reshape K → heads |
| 24 | **`linear`** | `[1,241,960] bf16 × [320,960] bf16` | `[1,241,320] bf16` | **V Linear** `960→320` |
| 25 | `view` | `[1,241,320] bf16` | `[1,241,15,64] bf16` | Reshape V → heads |
| 26 | `cat` | `[1,241,15,64] bf16` | `[1,241,15,64] bf16` | Pack Q `[…,15,64]` |
| 27 | `cat` | `[1,241,5,64] bf16` | `[1,241,5,64] bf16` | Pack K `[…,5,64]` |
| 28 | `cat` | `[1,241,5,64] bf16` | `[1,241,5,64] bf16` | Pack V `[…,5,64]` |
| 29 | `to` | `[1,241,15,64] bf16` | `[1,241,15,64] bf16 (cast)` | Cast Q for RoPE |
| 30 | `arange` | `(size/scalar args)` | `[32] float32 (new)` | RoPE(Q): angle build |
| 31 | `mul` | `[32] float32` | `[32] float32` | RoPE(Q): angle build |
| 32 | `pow` | `[32] float32` | `[32] float32` | RoPE(Q): angle build |
| 33 | `unsqueeze` | `[1,241] int64` | `[1,241,1] int64` | RoPE(Q): angle build |
| 34 | `to` | `[1,241,1] int64` | `[1,241,1] int64 (cast)` | RoPE(Q): angle build |
| 35 | `unsqueeze` | `[32] float32` | `[1,32] float32` | RoPE(Q): angle build |
| 36 | `unsqueeze` | `[1,32] float32` | `[1,1,32] float32` | RoPE(Q): angle build |
| 37 | `to` | `[1,1,32] float32` | `[1,1,32] float32 (cast)` | RoPE(Q): angle build |
| 38 | `div` | `[1,241,1] float32 × [1,1,32] float32` | `[1,1,32] float32` | RoPE(Q): angle build |
| 39 | `unsqueeze` | `[1,241,32] float32` | `[1,241,1,32] float32` | RoPE(Q): angle build |
| 40 | `sin` | `[1,241,1,32] float32` | `[1,241,1,32] float32` | RoPE(Q): angle build |
| 41 | `cos` | `[1,241,1,32] float32` | `[1,241,1,32] float32` | RoPE(Q): angle build |
| 42 | `split` | `[1,241,15,64] float32` | `2× half-dim (RoPE)` | RoPE(Q): apply |
| 43 | `empty_like` | `[1,241,15,64] float32` | `[1,241,15,32] float32 (new)` | RoPE(Q): apply |
| 44 | `mul` | `[1,241,15,32] float32 × [1,241,1,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 45 | `mul` | `[1,241,15,32] float32 × [1,241,1,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 46 | `sub` | `[1,241,15,32] float32 × [1,241,15,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 47 | `slice` | `[1,241,15,64] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 48 | `copy_` | `[1,241,15,32] float32 × [1,241,15,32] float32` | `[1,241,15,32] float32 (in-place)` | RoPE(Q): apply |
| 49 | `mul` | `[1,241,15,32] float32 × [1,241,1,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 50 | `mul` | `[1,241,15,32] float32 × [1,241,1,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 51 | `add` | `[1,241,15,32] float32 × [1,241,15,32] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 52 | `slice` | `[1,241,15,64] float32` | `[1,241,15,32] float32` | RoPE(Q): apply |
| 53 | `copy_` | `[1,241,15,32] float32 × [1,241,15,32] float32` | `[1,241,15,32] float32 (in-place)` | RoPE(Q): apply |
| 54 | `to` | `[1,241,15,64] float32` | `[1,241,15,64] float32 (cast)` | RoPE(Q): apply |
| 55 | `to` | `[1,241,5,64] bf16` | `[1,241,5,64] bf16 (cast)` | RoPE(K): angle build |
| 56 | `arange` | `(size/scalar args)` | `[32] float32 (new)` | RoPE(K): angle build |
| 57 | `mul` | `[32] float32` | `[32] float32` | RoPE(K): angle build |
| 58 | `pow` | `[32] float32` | `[32] float32` | RoPE(K): angle build |
| 59 | `unsqueeze` | `[1,241] int64` | `[1,241,1] int64` | RoPE(K): angle build |
| 60 | `to` | `[1,241,1] int64` | `[1,241,1] int64 (cast)` | RoPE(K): angle build |
| 61 | `unsqueeze` | `[32] float32` | `[1,32] float32` | RoPE(K): angle build |
| 62 | `unsqueeze` | `[1,32] float32` | `[1,1,32] float32` | RoPE(K): angle build |
| 63 | `to` | `[1,1,32] float32` | `[1,1,32] float32 (cast)` | RoPE(K): angle build |
| 64 | `div` | `[1,241,1] float32 × [1,1,32] float32` | `[1,1,32] float32` | RoPE(K): angle build |
| 65 | `unsqueeze` | `[1,241,32] float32` | `[1,241,1,32] float32` | RoPE(K): angle build |
| 66 | `sin` | `[1,241,1,32] float32` | `[1,241,1,32] float32` | RoPE(K): angle build |
| 67 | `cos` | `[1,241,1,32] float32` | `[1,241,1,32] float32` | RoPE(K): angle build |
| 68 | `split` | `[1,241,5,64] float32` | `2× half-dim (RoPE)` | RoPE(K): apply |
| 69 | `empty_like` | `[1,241,5,64] float32` | `[1,241,5,32] float32 (new)` | RoPE(K): apply |
| 70 | `mul` | `[1,241,5,32] float32 × [1,241,1,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 71 | `mul` | `[1,241,5,32] float32 × [1,241,1,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 72 | `sub` | `[1,241,5,32] float32 × [1,241,5,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 73 | `slice` | `[1,241,5,64] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 74 | `copy_` | `[1,241,5,32] float32 × [1,241,5,32] float32` | `[1,241,5,32] float32 (in-place)` | RoPE(K): apply |
| 75 | `mul` | `[1,241,5,32] float32 × [1,241,1,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 76 | `mul` | `[1,241,5,32] float32 × [1,241,1,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 77 | `add` | `[1,241,5,32] float32 × [1,241,5,32] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 78 | `slice` | `[1,241,5,64] float32` | `[1,241,5,32] float32` | RoPE(K): apply |
| 79 | `copy_` | `[1,241,5,32] float32 × [1,241,5,32] float32` | `[1,241,5,32] float32 (in-place)` | RoPE(K): apply |
| 80 | `to` | `[1,241,5,64] float32` | `[1,241,5,64] bf16` | RoPE(K): apply |
| 81 | `transpose` | `[1,241,5,64] bf16` | `[1,241,5,64] bf16` | KV cache transpose / cat |
| 82 | `transpose` | `[1,241,5,64] bf16` | `[0] bf16` | KV cache transpose / cat |
| 83 | `lift_fresh` | `[0] bf16` | `[0] bf16` | KV cache transpose / cat |
| 84 | `lift_fresh` | `[0] bf16` | `[0] bf16` | KV cache transpose / cat |
| 85 | `cat` | `[0] bf16 × [1,5,241,64] bf16` | `concat` | KV cache transpose / cat |
| 86 | `cat` | `[0] bf16 × [1,5,241,64] bf16` | `concat` | KV cache transpose / cat |
| 87 | `transpose` | `[1,5,241,64] bf16` | `[1,5,241,64] bf16` | KV cache transpose / cat |
| 88 | `transpose` | `[1,5,241,64] bf16` | `[1,241,5,64] bf16` | KV cache transpose / cat |
| 89 | `unsqueeze` | `[1,241,5,64] bf16` | `[1,241,5,1,64] bf16` | Expand K/V 5→15 heads |
| 90 | `expand` | `[1,241,5,1,64] bf16` | `[1,241,5,3,64] bf16` | Expand K/V 5→15 heads |
| 91 | `reshape` | `[1,241,5,3,64] bf16` | `[1,241,5,64] bf16` | Expand K/V 5→15 heads |
| 92 | `unsqueeze` | `[1,241,5,64] bf16` | `[1,241,5,1,64] bf16` | Expand K/V 5→15 heads |
| 93 | `expand` | `[1,241,5,1,64] bf16` | `[1,241,5,3,64] bf16` | Expand K/V 5→15 heads |
| 94 | `reshape` | `[1,241,5,3,64] bf16` | `[1,241,15,64] bf16` | Expand K/V 5→15 heads |
| 95 | `to` | `[1,241,15,64] bf16` | `[1,241,15,64] bf16` | Cast/transpose Q,K for scores |
| 96 | `to` | `[1,241,15,64] bf16` | `[1,241,15,64] float32` | Cast/transpose Q,K for scores |
| 97 | `transpose` | `[1,241,15,64] float32` | `[1,241,15,64] float32` | Cast/transpose Q,K for scores |
| 98 | `transpose` | `[1,241,15,64] float32` | `[1,15,241,64] float32` | Cast/transpose Q,K for scores |
| 99 | `transpose` | `[1,15,241,64] float32` | `[1,15,241,64] float32` | Cast/transpose Q,K for scores |
| 100 | **`matmul`** | `[1,15,241,64] float32 × [1,15,64,241] float32` | `[1,15,241,241] float32` | **Attn MatMul `QKᵀ`** (FP32) |
| 101 | `mul_` | `[1,15,241,241] float32` | `[1,15,241,241] float32` | Scale scores `mul_` |
| 102 | `to` | `[1,15,241,241] float32` | `[1,15,241,241] float32 (cast)` | Scores dtype path |
| 103 | `unsqueeze` | `[1,241,241] bool` | `[1,1,241,241] bool` | Unsqueeze attn mask |
| 104 | `where` | `[1,1,241,241] bool × [1,15,241,241] float32` | `[1,15,241,241] float32` | `where(mask, score, −∞)` |
| 105 | **`softmax`** | `[1,15,241,241] float32` | `[1,15,241,241] float32` | **Softmax** (FP32) |
| 106 | `to` | `[1,15,241,241] float32` | `[1,15,241,241] float32 (cast)` | Cast attn probs → bf16 |
| 107 | `permute` | `[1,241,15,64] bf16` | `[1,15,241,241] bf16` | Permute V heads-first |
| 108 | **`matmul`** | `[1,15,241,241] bf16 × [1,15,241,64] bf16` | `[1,15,241,64] bf16` | **Attn MatMul `A·V`** (bf16) |
| 109 | `permute` | `[1,15,241,64] bf16` | `[1,241,15,64] bf16` | Permute attn out |
| 110 | `reshape` | `[1,241,15,64] bf16` | `[1,241,960] bf16` | Concat heads → 960 |
| 111 | **`linear`** | `[1,241,960] bf16 × [960,960] bf16` | `[1,241,960] bf16` | **Attn-out Linear** `960→960` |
| 112 | `add_` | `[1,241,960] bf16 × [1,241,960] float32` | `[1,241,960] bf16` | Residual add 1 |
| 113 | `clone` | `[1,241,960] bf16` | `[1,241,960] bf16` | Clone residual stream |
| 114 | `to` | `[1,241,960] bf16` | `[1,241,960] float32` | Cast for post-attn RMSNorm |
| 115 | `pow` | `[1,241,960] float32` | `[1,241,960] float32` | Post-attn RMSNorm: `x²` |
| 116 | `mean` | `[1,241,960] float32` | `[1,241,1] float32` | mean |
| 117 | `add` | `[1,241,1] float32` | `[1,241,1] float32` | +eps |
| 118 | `rsqrt` | `[1,241,1] float32` | `[1,241,1] float32` | rsqrt |
| 119 | `mul` | `[1,241,960] float32 × [1,241,1] float32` | `[1,241,960] float32` | × rsqrt |
| 120 | `to` | `[1,241,960] float32` | `[1,241,960] float32 (cast)` | cast |
| 121 | `mul` | `[960] bf16 × [1,241,960] bf16` | `[1,241,960] bf16` | × weight |
| 122 | **`linear`** | `[1,241,960] bf16 × [2560,960] bf16` | `[1,241,2560] bf16` | **MLP gate Linear** `960→2560` |
| 123 | **`silu`** | `[1,241,2560] bf16` | `[1,241,2560] bf16` | **SiLU** on gate |
| 124 | **`linear`** | `[1,241,960] bf16 × [2560,960] bf16` | `[1,241,2560] bf16` | **MLP up Linear** `960→2560` |
| 125 | `mul` | `[1,241,2560] bf16 × [1,241,2560] bf16` | `[1,241,2560] bf16` | Gate ⊗ up |
| 126 | **`linear`** | `[1,241,2560] bf16 × [960,2560] bf16` | `[1,241,960] bf16` | **MLP down Linear** `2560→960` |
| 127 | `add_` | `[1,241,960] bf16 × [1,241,960] bf16` | `[1,241,960] bf16` | Residual add 2 |

### 3.3 Final VLM RMSNorm (`#1883–1890`)

After layer 16 residual:

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1883 | `to` | `[1,241,960] bf16` | `[1,241,960] float32` | Final VLM RMSNorm: cast |
| 1884 | `pow` | `[1,241,960] float32` | `[1,241,960] float32` | `x²` |
| 1885 | `mean` | `[1,241,960] float32` | `[1,241,1] float32` | mean |
| 1886 | `add` | `[1,241,1] float32` | `[1,241,1] float32` | +eps |
| 1887 | `rsqrt` | `[1,241,1] float32` | `[1,241,1] float32` | rsqrt |
| 1888 | `mul` | `[1,241,960] float32 × [1,241,1] float32` | `[1,241,960] float32` | × rsqrt |
| 1889 | `to` | `[1,241,960] float32` | `[1,241,960] float32 (cast)` | cast |
| 1890 | `mul` | `[960] bf16 × [1,241,960] bf16` | `[1,241,960] bf16` | × weight |

### 3.4 Prefill first layer — CUDA kernel order (Nsight)

Moved to **`smolVLA_kernel_gpu_list.md`**: layer‑1’s **77** CUDA launches in order (Stage‑2 window **564.90→576.32 ms**), plus cross-layer comparison for all **16** prefill layers.

### 3.5 Stage-2 reading

- **1890** aten /chunk; not 1:1 with Nsight kernels.
- Installed path: Q/K → FP32 before `QKᵀ`; scores FP32 through scale/`where`/softmax; **A → bf16** for `A·V`.
- Attributed GEMM time ≈ **2.15 ms**; wall Stage 2 ≈ **7.2 ms**. Linear+attn GEMM budget /chunk: **144**.

---

## 4. Stage 3 — expert decode + Euler (GPU)

**Function:** `M=10` Euler steps; each rebuilds suffix (§2) then **16** expert layers (**8** even self-attn + **8** odd cross-attn), then velocity + Euler + cache crop.

**Source:** chrono Stage **`3_expert_euler`** — **1693** launches for **one** Euler step (§0.1). Full chunk ≈ ×`M=10`. Even self-attn uses KV len **291** (prefix 241 + suffix 50); odd cross-attn scores over prefix **241** only; cross-K/V Linears are **FP32**.

### 4.1 Mask / position prelude (`#1–16`; once / Euler step)

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | `unsqueeze` | `[1,241] bool` | `[1,1,241] bool` | Expand prefix pad mask |
| 2 | `expand` | `[1,1,241] bool` | `[1,50] bool` | Broadcast prefix mask |
| 3 | `cumsum` | `[1,50] float32` | `[1,50] float32` | Suffix position cumsum |
| 4 | `unsqueeze` | `[1,50] float32` | `[1,50] float32` | Unsqueeze suffix pos (Q) |
| 5 | `unsqueeze` | `[1,50] float32` | `[1,1,50] float32` | Unsqueeze suffix pos (K) |
| 6 | `le` | `[1,1,50] float32 × [1,50,1] float32` | `[1,1,50] bool` | Suffix causal `le` |
| 7 | `unsqueeze` | `[1,50] bool` | `[1,50] bool` | Unsqueeze suffix pad |
| 8 | `unsqueeze` | `[1,50] bool` | `[1,1,50] bool` | Unsqueeze suffix pad |
| 9 | `mul` | `[1,1,50] bool × [1,50,1] bool` | `[1,1,50] bool` | Suffix pad outer product |
| 10 | `__and__` | `[1,50,50] bool × [1,50,50] bool` | `[1,50,50] bool` | Suffix causal ∧ pad |
| 11 | `cat` | `[1,50,241] bool × [1,50,50] bool` | `[1,50,291] bool` | Cat prefix‖suffix attn mask → len 291 |
| 12 | `sum` | `[1,241] bool` | `reduced scalar/vec` | Sum prefix pad (offset) |
| 13 | `unsqueeze` | `[1] int64` | `[1,50] int64` | Unsqueeze offset |
| 14 | `cumsum` | `[1,50] bool` | `[1,50] int64` | Suffix cumsum (bool) |
| 15 | `add` | `[1,1] int64 × [1,50] int64` | `[1,50] int64` | Add prefix len to suffix positions |
| 16 | `sub` | `[1,50] int64` | `[1,50] int64` | Adjust suffix position ids |

### 4.2 Even expert layer — self-attention (template; ×`8`/step → `80`/chunk)

Example: first even layer **`#17–131`**. Rhythm: Q/K/V from suffix 720‑d, RoPE on Q+K, cat prefix‖suffix KV → len 291, FP32 `QKᵀ` + bf16 `A·V`, out Linear `960→720`, SwiGLU MLP.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 17 | `to` | `[1,50,720] float32` | `[1,50,720] float32` | Cast → FP32 for RMSNorm |
| 18 | `pow` | `[1,50,720] float32` | `[1,50,720] float32` | RMSNorm: `x²` |
| 19 | `mean` | `[1,50,720] float32` | `[1,50,1] float32` | mean |
| 20 | `add` | `[1,50,1] float32` | `[1,50,1] float32` | +eps |
| 21 | `rsqrt` | `[1,50,1] float32` | `[1,50,1] float32` | rsqrt |
| 22 | `mul` | `[1,50,720] float32 × [1,50,1] float32` | `[1,50,720] float32` | × rsqrt |
| 23 | `to` | `[1,50,720] float32` | `[1,50,720] float32 (cast)` | cast |
| 24 | `mul` | `[720] bf16 × [1,50,720] float32` | `[1,50,720] float32` | × weight |
| 25 | `to` | `[1,50,720] float32` | `[1,50,720] bf16` | Cast → bf16 |
| 26 | **`linear`** | `[1,50,720] bf16 × [960,720] bf16` | `[1,50,960] bf16` | **Q Linear** `720→960` |
| 27 | `view` | `[1,50,960] bf16` | `[1,50,720] bf16` | Reshape Q heads |
| 28 | **`linear`** | `[1,50,720] bf16 × [320,720] bf16` | `[1,50,320] bf16` | **K Linear** `720→320` |
| 29 | `view` | `[1,50,320] bf16` | `[1,50,720] bf16` | Reshape K heads |
| 30 | **`linear`** | `[1,50,720] bf16 × [320,720] bf16` | `[1,50,320] bf16` | **V Linear** `720→320` |
| 31 | `view` | `[1,50,320] bf16` | `[1,50,15,64] bf16` | Reshape V heads |
| 32 | `cat` | `[1,50,15,64] bf16` | `[1,50,15,64] bf16` | Pack Q |
| 33 | `cat` | `[1,50,5,64] bf16` | `[1,50,5,64] bf16` | Pack K |
| 34 | `cat` | `[1,50,5,64] bf16` | `[1,50,5,64] bf16` | Pack V |
| 35 | `to` | `[1,50,15,64] bf16` | `[1,50,15,64] bf16 (cast)` | Cast Q for RoPE |
| 36 | `arange` | `(size/scalar args)` | `[32] float32 (new)` | RoPE(Q) |
| 37 | `mul` | `[32] float32` | `[32] float32` | RoPE(Q) |
| 38 | `pow` | `[32] float32` | `[32] float32` | RoPE(Q) |
| 39 | `unsqueeze` | `[1,50] int64` | `[1,50,1] int64` | RoPE(Q) |
| 40 | `to` | `[1,50,1] int64` | `[1,50,1] int64 (cast)` | RoPE(Q) |
| 41 | `unsqueeze` | `[32] float32` | `[1,32] float32` | RoPE(Q) |
| 42 | `unsqueeze` | `[1,32] float32` | `[1,1,32] float32` | RoPE(Q) |
| 43 | `to` | `[1,1,32] float32` | `[1,1,32] float32 (cast)` | RoPE(Q) |
| 44 | `div` | `[1,50,1] float32 × [1,1,32] float32` | `[1,1,32] float32` | RoPE(Q) |
| 45 | `unsqueeze` | `[1,50,32] float32` | `[1,50,1,32] float32` | RoPE(Q) |
| 46 | `sin` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(Q) |
| 47 | `cos` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(Q) |
| 48 | `split` | `[1,50,15,64] float32` | `2× half-dim (RoPE)` | RoPE(Q) |
| 49 | `empty_like` | `[1,50,15,64] float32` | `[1,50,15,32] float32 (new)` | RoPE(Q) |
| 50 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 51 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 52 | `sub` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 53 | `slice` | `[1,50,15,64] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 54 | `copy_` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32 (in-place)` | RoPE(Q) |
| 55 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 56 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 57 | `add` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 58 | `slice` | `[1,50,15,64] float32` | `[1,50,15,32] float32` | RoPE(Q) |
| 59 | `copy_` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32 (in-place)` | RoPE(Q) |
| 60 | `to` | `[1,50,15,64] float32` | `[1,50,15,64] float32 (cast)` | RoPE(Q) |
| 61 | `to` | `[1,50,5,64] bf16` | `[1,50,5,64] bf16 (cast)` | RoPE(K) |
| 62 | `arange` | `(size/scalar args)` | `[32] float32 (new)` | RoPE(K) |
| 63 | `mul` | `[32] float32` | `[32] float32` | RoPE(K) |
| 64 | `pow` | `[32] float32` | `[32] float32` | RoPE(K) |
| 65 | `unsqueeze` | `[1,50] int64` | `[1,50,1] int64` | RoPE(K) |
| 66 | `to` | `[1,50,1] int64` | `[1,50,1] int64 (cast)` | RoPE(K) |
| 67 | `unsqueeze` | `[32] float32` | `[1,32] float32` | RoPE(K) |
| 68 | `unsqueeze` | `[1,32] float32` | `[1,1,32] float32` | RoPE(K) |
| 69 | `to` | `[1,1,32] float32` | `[1,1,32] float32 (cast)` | RoPE(K) |
| 70 | `div` | `[1,50,1] float32 × [1,1,32] float32` | `[1,1,32] float32` | RoPE(K) |
| 71 | `unsqueeze` | `[1,50,32] float32` | `[1,50,1,32] float32` | RoPE(K) |
| 72 | `sin` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(K) |
| 73 | `cos` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(K) |
| 74 | `split` | `[1,50,5,64] float32` | `2× half-dim (RoPE)` | RoPE(K) |
| 75 | `empty_like` | `[1,50,5,64] float32` | `[1,50,5,32] float32 (new)` | RoPE(K) |
| 76 | `mul` | `[1,50,5,32] float32 × [1,50,1,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 77 | `mul` | `[1,50,5,32] float32 × [1,50,1,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 78 | `sub` | `[1,50,5,32] float32 × [1,50,5,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 79 | `slice` | `[1,50,5,64] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 80 | `copy_` | `[1,50,5,32] float32 × [1,50,5,32] float32` | `[1,50,5,32] float32 (in-place)` | RoPE(K) |
| 81 | `mul` | `[1,50,5,32] float32 × [1,50,1,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 82 | `mul` | `[1,50,5,32] float32 × [1,50,1,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 83 | `add` | `[1,50,5,32] float32 × [1,50,5,32] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 84 | `slice` | `[1,50,5,64] float32` | `[1,50,5,32] float32` | RoPE(K) |
| 85 | `copy_` | `[1,50,5,32] float32 × [1,50,5,32] float32` | `[1,50,5,32] float32 (in-place)` | RoPE(K) |
| 86 | `to` | `[1,50,5,64] float32` | `[1,50,5,64] bf16` | RoPE(K) |
| 87 | `transpose` | `[1,50,5,64] bf16` | `[1,50,5,64] bf16` | Cache cat prefix‖suffix KV (len 291) |
| 88 | `transpose` | `[1,50,5,64] bf16` | `[1,5,241,64] bf16` | Cache cat prefix‖suffix KV (len 291) |
| 89 | `cat` | `[1,5,241,64] bf16 × [1,5,50,64] bf16` | `concat` | Cache cat prefix‖suffix KV (len 291) |
| 90 | `cat` | `[1,5,241,64] bf16 × [1,5,50,64] bf16` | `concat` | Cache cat prefix‖suffix KV (len 291) |
| 91 | `transpose` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache cat prefix‖suffix KV (len 291) |
| 92 | `transpose` | `[1,5,291,64] bf16` | `[1,291,5,64] bf16` | Cache cat prefix‖suffix KV (len 291) |
| 93 | `unsqueeze` | `[1,291,5,64] bf16` | `[1,291,5,1,64] bf16` | Expand K/V 5→15 |
| 94 | `expand` | `[1,291,5,1,64] bf16` | `[1,291,5,3,64] bf16` | Expand K/V 5→15 |
| 95 | `reshape` | `[1,291,5,3,64] bf16` | `[1,291,5,64] bf16` | Expand K/V 5→15 |
| 96 | `unsqueeze` | `[1,291,5,64] bf16` | `[1,291,5,1,64] bf16` | Expand K/V 5→15 |
| 97 | `expand` | `[1,291,5,1,64] bf16` | `[1,291,5,3,64] bf16` | Expand K/V 5→15 |
| 98 | `reshape` | `[1,291,5,3,64] bf16` | `[1,50,15,64] bf16` | Expand K/V 5→15 |
| 99 | `to` | `[1,50,15,64] bf16` | `[1,50,15,64] bf16 (cast)` | Cast/transpose for scores |
| 100 | `to` | `[1,291,15,64] bf16` | `[1,291,15,64] bf16 (cast)` | Cast/transpose for scores |
| 101 | `transpose` | `[1,50,15,64] float32` | `[1,291,15,64] float32` | Cast/transpose for scores |
| 102 | `transpose` | `[1,291,15,64] float32` | `[1,15,291,64] float32` | Cast/transpose for scores |
| 103 | `transpose` | `[1,15,291,64] float32` | `[1,15,50,64] float32` | Cast/transpose for scores |
| 104 | **`matmul`** | `[1,15,50,64] float32 × [1,15,64,291] float32` | `[1,15,50,291] float32` | **Attn MatMul `QKᵀ`** `[…,50,291]` FP32 |
| 105 | `mul_` | `[1,15,50,291] float32` | `[1,15,50,291] float32` | Scale scores |
| 106 | `to` | `[1,15,50,291] float32` | `[1,15,50,291] float32 (cast)` | Scores path |
| 107 | `unsqueeze` | `[1,50,291] bool` | `[1,1,50,291] bool` | Unsqueeze mask |
| 108 | `where` | `[1,1,50,291] bool × [1,15,50,291] float32` | `[1,15,50,291] float32` | `where` mask |
| 109 | **`softmax`** | `[1,15,50,291] float32` | `[1,15,50,291] float32` | **Softmax** |
| 110 | `to` | `[1,15,50,291] float32` | `[1,15,50,291] float32 (cast)` | Cast probs → bf16 |
| 111 | `permute` | `[1,291,15,64] bf16` | `[1,15,50,291] bf16` | Permute V |
| 112 | **`matmul`** | `[1,15,50,291] bf16 × [1,15,291,64] bf16` | `[1,15,50,64] bf16` | **Attn MatMul `A·V`** bf16 |
| 113 | `permute` | `[1,15,50,64] bf16` | `[1,50,15,64] bf16` | Permute out |
| 114 | `reshape` | `[1,50,15,64] bf16` | `[1,50,960] bf16` | Concat heads |
| 115 | **`linear`** | `[1,50,960] bf16 × [720,960] bf16` | `[1,50,720] bf16` | **Attn-out Linear** `960→720` |
| 116 | `add_` | `[1,50,720] bf16 × [1,50,720] float32` | `[1,50,720] bf16` | Residual 1 |
| 117 | `clone` | `[1,50,720] bf16` | `[1,50,720] bf16` | Clone |
| 118 | `to` | `[1,50,720] bf16` | `[1,50,720] float32` | Cast post-attn RMSNorm |
| 119 | `pow` | `[1,50,720] float32` | `[1,50,720] float32` | Post-attn RMSNorm: `x²` |
| 120 | `mean` | `[1,50,720] float32` | `[1,50,1] float32` | mean |
| 121 | `add` | `[1,50,1] float32` | `[1,50,1] float32` | +eps |
| 122 | `rsqrt` | `[1,50,1] float32` | `[1,50,1] float32` | rsqrt |
| 123 | `mul` | `[1,50,720] float32 × [1,50,1] float32` | `[1,50,720] float32` | × rsqrt |
| 124 | `to` | `[1,50,720] float32` | `[1,50,720] float32 (cast)` | cast |
| 125 | `mul` | `[720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | × weight |
| 126 | **`linear`** | `[1,50,720] bf16 × [2048,720] bf16` | `[1,50,2048] bf16` | **MLP gate** `720→2048` |
| 127 | **`silu`** | `[1,50,2048] bf16` | `[1,50,2048] bf16` | **SiLU** |
| 128 | **`linear`** | `[1,50,720] bf16 × [2048,720] bf16` | `[1,50,2048] bf16` | **MLP up** `720→2048` |
| 129 | `mul` | `[1,50,2048] bf16 × [1,50,2048] bf16` | `[1,50,2048] bf16` | Gate ⊗ up |
| 130 | **`linear`** | `[1,50,2048] bf16 × [720,2048] bf16` | `[1,50,720] bf16` | **MLP down** `2048→720` |
| 131 | `add_` | `[1,50,720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | Residual 2 |

### 4.3 Odd expert layer — cross-attention to prefix (template; ×`8`/step → `80`/chunk)

Example: first odd layer **`#132–222`**. Q from suffix; **Cross-K / Cross-V** `320→320` **FP32** on cached prefix; RoPE on **Q only**; scores `[…,50,241]`; **`A·V` is FP32**.

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 132 | `transpose` | `[1,5,241,64] bf16` | `[1,5,241,64] bf16` | Transpose cached prefix K |
| 133 | `transpose` | `[1,5,241,64] bf16` | `[1,50,720] bf16` | Transpose cached prefix V |
| 134 | `to` | `[1,50,720] bf16` | `[1,50,720] float32` | Cast for RMSNorm |
| 135 | `pow` | `[1,50,720] float32` | `[1,50,720] float32` | RMSNorm: `x²` |
| 136 | `mean` | `[1,50,720] float32` | `[1,50,1] float32` | mean |
| 137 | `add` | `[1,50,1] float32` | `[1,50,1] float32` | +eps |
| 138 | `rsqrt` | `[1,50,1] float32` | `[1,50,1] float32` | rsqrt |
| 139 | `mul` | `[1,50,720] float32 × [1,50,1] float32` | `[1,50,720] float32` | × rsqrt |
| 140 | `to` | `[1,50,720] float32` | `[1,50,720] float32 (cast)` | cast |
| 141 | `mul` | `[720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | × weight |
| 142 | `to` | `[1,50,720] bf16` | `[1,50,720] bf16` | cast bf16 |
| 143 | **`linear`** | `[1,50,720] bf16 × [960,720] bf16` | `[1,50,960] bf16` | **Q Linear** `720→960` (cross) |
| 144 | `view` | `[1,50,960] bf16` | `[1,241,5,64] bf16` | Reshape Q |
| 145 | `to` | `[1,241,5,64] bf16` | `[1,241,5,64] float32` | Read prefix K cache |
| 146 | `reshape` | `[1,241,5,64] float32` | `[1,241,320] float32` | Flatten K → 320 FP32 |
| 147 | **`linear`** | `[1,241,320] float32 × [320,320] float32` | `[1,241,320] float32` | **Cross-K Linear** `320→320` **FP32** |
| 148 | `view` | `[1,241,320] float32` | `[1,241,5,64] float32` | Reshape cross-K |
| 149 | `to` | `[1,241,5,64] bf16` | `[1,241,5,64] float32` | Read prefix V cache |
| 150 | `reshape` | `[1,241,5,64] float32` | `[1,241,320] float32` | Flatten V → 320 FP32 |
| 151 | **`linear`** | `[1,241,320] float32 × [320,320] float32` | `[1,241,320] float32` | **Cross-V Linear** `320→320` **FP32** |
| 152 | `view` | `[1,241,320] float32` | `[1,50] float32` | Reshape cross-V |
| 153 | `min` | `[1,50] int64` | `reduced` | Mask length `min` |
| 154 | `sub` | `[1,50] int64 × [1,1] int64` | `[1,50] int64` | Adjust |
| 155 | `slice` | `[1,50,291] bool` | `[1,50,291] bool` | Slice mask |
| 156 | `slice` | `[1,50,291] bool` | `[1,50,15,64] bool` | Slice mask |
| 157 | `to` | `[1,50,15,64] bf16` | `[1,50,15,64] bf16 (cast)` | Cast Q for RoPE |
| 158 | `arange` | `(size/scalar args)` | `[32] float32 (new)` | RoPE(Q only) |
| 159 | `mul` | `[32] float32` | `[32] float32` | RoPE(Q only) |
| 160 | `pow` | `[32] float32` | `[32] float32` | RoPE(Q only) |
| 161 | `unsqueeze` | `[1,50] int64` | `[1,50,1] int64` | RoPE(Q only) |
| 162 | `to` | `[1,50,1] int64` | `[1,50,1] int64 (cast)` | RoPE(Q only) |
| 163 | `unsqueeze` | `[32] float32` | `[1,32] float32` | RoPE(Q only) |
| 164 | `unsqueeze` | `[1,32] float32` | `[1,1,32] float32` | RoPE(Q only) |
| 165 | `to` | `[1,1,32] float32` | `[1,1,32] float32 (cast)` | RoPE(Q only) |
| 166 | `div` | `[1,50,1] float32 × [1,1,32] float32` | `[1,1,32] float32` | RoPE(Q only) |
| 167 | `unsqueeze` | `[1,50,32] float32` | `[1,50,1,32] float32` | RoPE(Q only) |
| 168 | `sin` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(Q only) |
| 169 | `cos` | `[1,50,1,32] float32` | `[1,50,1,32] float32` | RoPE(Q only) |
| 170 | `split` | `[1,50,15,64] float32` | `2× half-dim (RoPE)` | RoPE(Q only) |
| 171 | `empty_like` | `[1,50,15,64] float32` | `[1,50,15,32] float32 (new)` | RoPE(Q only) |
| 172 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 173 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 174 | `sub` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 175 | `slice` | `[1,50,15,64] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 176 | `copy_` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32 (in-place)` | RoPE(Q only) |
| 177 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 178 | `mul` | `[1,50,15,32] float32 × [1,50,1,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 179 | `add` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 180 | `slice` | `[1,50,15,64] float32` | `[1,50,15,32] float32` | RoPE(Q only) |
| 181 | `copy_` | `[1,50,15,32] float32 × [1,50,15,32] float32` | `[1,50,15,32] float32 (in-place)` | RoPE(Q only) |
| 182 | `to` | `[1,50,15,64] float32` | `[1,50,15,64] float32 (cast)` | RoPE(Q only) |
| 183 | `unsqueeze` | `[1,241,5,64] float32` | `[1,241,5,1,64] float32` | Expand cross K/V 5→15 |
| 184 | `expand` | `[1,241,5,1,64] float32` | `[1,241,5,3,64] float32` | Expand cross K/V 5→15 |
| 185 | `reshape` | `[1,241,5,3,64] float32` | `[1,241,5,64] float32` | Expand cross K/V 5→15 |
| 186 | `unsqueeze` | `[1,241,5,64] float32` | `[1,241,5,1,64] float32` | Expand cross K/V 5→15 |
| 187 | `expand` | `[1,241,5,1,64] float32` | `[1,241,5,3,64] float32` | Expand cross K/V 5→15 |
| 188 | `reshape` | `[1,241,5,3,64] float32` | `[1,50,15,64] float32` | Expand cross K/V 5→15 |
| 189 | `to` | `[1,50,15,64] bf16` | `[1,50,15,64] bf16 (cast)` | Cast/transpose for scores |
| 190 | `to` | `[1,241,15,64] float32` | `[1,241,15,64] float32 (cast)` | Cast/transpose for scores |
| 191 | `transpose` | `[1,50,15,64] float32` | `[1,241,15,64] float32` | Cast/transpose for scores |
| 192 | `transpose` | `[1,241,15,64] float32` | `[1,15,241,64] float32` | Cast/transpose for scores |
| 193 | `transpose` | `[1,15,241,64] float32` | `[1,15,50,64] float32` | Cast/transpose for scores |
| 194 | **`matmul`** | `[1,15,50,64] float32 × [1,15,64,241] float32` | `[1,15,50,241] float32` | **Attn MatMul `QKᵀ`** `[…,50,241]` FP32 |
| 195 | `mul_` | `[1,15,50,241] float32` | `[1,15,50,241] float32` | Scale |
| 196 | `to` | `[1,15,50,241] float32` | `[1,15,50,241] float32 (cast)` | Scores path |
| 197 | `unsqueeze` | `[1,50,241] bool` | `[1,1,50,241] bool` | Unsqueeze mask |
| 198 | `where` | `[1,1,50,241] bool × [1,15,50,241] float32` | `[1,15,50,241] float32` | `where` |
| 199 | **`softmax`** | `[1,15,50,241] float32` | `[1,15,50,241] float32` | **Softmax** |
| 200 | `to` | `[1,15,50,241] float32` | `[1,15,50,241] float32 (cast)` | Cast probs |
| 201 | `permute` | `[1,241,15,64] float32` | `[1,15,50,241] float32` | Permute V |
| 202 | **`matmul`** | `[1,15,50,241] float32 × [1,15,241,64] float32` | `[1,15,50,64] float32` | **Attn MatMul `A·V`** **FP32** |
| 203 | `permute` | `[1,15,50,64] float32` | `[1,50,15,64] float32` | Permute |
| 204 | `reshape` | `[1,50,15,64] float32` | `[1,50,960] float32` | Concat heads |
| 205 | `to` | `[1,50,960] float32` | `[1,50,960] bf16` | Cast attn → bf16 for out Linear |
| 206 | **`linear`** | `[1,50,960] bf16 × [720,960] bf16` | `[1,50,720] bf16` | **Attn-out Linear** `960→720` |
| 207 | `add_` | `[1,50,720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | Residual 1 |
| 208 | `clone` | `[1,50,720] bf16` | `[1,50,720] bf16` | Clone |
| 209 | `to` | `[1,50,720] bf16` | `[1,50,720] float32` | Cast post-attn RMSNorm |
| 210 | `pow` | `[1,50,720] float32` | `[1,50,720] float32` | RMSNorm `x²` |
| 211 | `mean` | `[1,50,720] float32` | `[1,50,1] float32` | mean |
| 212 | `add` | `[1,50,1] float32` | `[1,50,1] float32` | +eps |
| 213 | `rsqrt` | `[1,50,1] float32` | `[1,50,1] float32` | rsqrt |
| 214 | `mul` | `[1,50,720] float32 × [1,50,1] float32` | `[1,50,720] float32` | × rsqrt |
| 215 | `to` | `[1,50,720] float32` | `[1,50,720] float32 (cast)` | cast |
| 216 | `mul` | `[720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | × weight |
| 217 | **`linear`** | `[1,50,720] bf16 × [2048,720] bf16` | `[1,50,2048] bf16` | **MLP gate** `720→2048` |
| 218 | **`silu`** | `[1,50,2048] bf16` | `[1,50,2048] bf16` | **SiLU** |
| 219 | **`linear`** | `[1,50,720] bf16 × [2048,720] bf16` | `[1,50,2048] bf16` | **MLP up** |
| 220 | `mul` | `[1,50,2048] bf16 × [1,50,2048] bf16` | `[1,50,2048] bf16` | Gate ⊗ up |
| 221 | **`linear`** | `[1,50,2048] bf16 × [720,2048] bf16` | `[1,50,720] bf16` | **MLP down** |
| 222 | `add_` | `[1,50,720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | Residual 2 |

### 4.4 Velocity head + Euler + cache crop (`#1665–1693`; ×`M`/chunk)

After the 16th expert layer MLP residual:

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1665 | `to` | `[1,50,720] bf16` | `[1,50,720] float32` | Expert final RMSNorm |
| 1666 | `pow` | `[1,50,720] float32` | `[1,50,720] float32` | Expert final RMSNorm |
| 1667 | `mean` | `[1,50,720] float32` | `[1,50,1] float32` | Expert final RMSNorm |
| 1668 | `add` | `[1,50,1] float32` | `[1,50,1] float32` | Expert final RMSNorm |
| 1669 | `rsqrt` | `[1,50,1] float32` | `[1,50,1] float32` | Expert final RMSNorm |
| 1670 | `mul` | `[1,50,720] float32 × [1,50,1] float32` | `[1,50,720] float32` | Expert final RMSNorm |
| 1671 | `to` | `[1,50,720] float32` | `[1,50,720] float32 (cast)` | Expert final RMSNorm |
| 1672 | `mul` | `[720] bf16 × [1,50,720] bf16` | `[1,50,720] bf16` | Expert final RMSNorm |
| 1673 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1674 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1675 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1676 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1677 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1678 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1679 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1680 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1681 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1682 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1683 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1684 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1685 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1686 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1687 | `slice` | `[1,5,291,64] bf16` | `[1,5,291,64] bf16` | Cache crop `slice` → prefix len 241 |
| 1688 | `slice` | `[1,5,291,64] bf16` | `[1,50,720] bf16` | Cache crop `slice` → prefix len 241 |
| 1689 | `slice` | `[1,50,720] bf16` | `[1,50,720] bf16` | Slice expert hidden |
| 1690 | `to` | `[1,50,720] bf16` | `[1,50,720] float32` | Cast → FP32 for velocity head |
| 1691 | **`linear`** | `[1,50,720] float32 × [32,720] float32 × [32] float32` | `[1,50,32] float32` | **Velocity Linear** `720→32` |
| 1692 | `mul` | `[1,50,32] float32` | `[1,50,32] float32` | Euler: `dt · v` |
| 1693 | `add` | `[1,50,32] float32 × [1,50,32] float32` | `[1,50,32] float32` | Euler: `x ← x + dt·v` |

### 4.5 Stage-3 reading

- **1693** aten / Euler step; ×`M=10` ≈ **16.9k**/chunk.
- Even vs odd fingerprint: even `matmul` shapes use **291** keys (bf16 `A·V`); odd uses **241** keys and **FP32** `A·V` + FP32 cross-K/V.
- Linear-only GEMM budget /chunk (excl. attn QK/AV) ≈ **1130**. Wall Stage 3 dominates chunk latency.

---

## 5. Stage 4 — crop / queue / postprocess (GPU)

**Function:** after chunk fill, crop action dims, build the action queue, pop one step, run `postprocess` (unnormalize) → env action `(B,6)`.

**Source:** chrono Stage **`4_crop_queue_post`** — **5** launches / `select_action` pop (§0.1). Mirrors `actions.transpose(0,1)[:n_action_steps]` → `queue.pop(0)` → `postprocess(one)`. Wall ≈ **0.05 ms** (negligible vs Stages 0–3).

### 5.1 Full path (`#1–5`; 1× per popped action)

| Seq | aten:: | Input (shape, dtype) | Output (shape, dtype) | Semantic role |
|---:|---|---|---|---|
| 1 | `slice` | `[1,50,32] float32` | `[1,50,6] float32` | Crop action dim `32→6` |
| 2 | `transpose` | `[1,50,6] float32` | `[50,1,6] float32` | Horizon-first for queue (`transpose(0,1)`) |
| 3 | `slice` | `[50,1,6] float32` | `[50,1,6] float32` (prefix `n_action_steps`) | Take first `n_action_steps` of queue |
| 4 | `unbind` | `[50,1,6] float32` | list of `[1,6]` steps | Materialize queue entries |
| 5 | `to` | `[1,6] float32` | `[1,6] float32` | Popped step on device / dtype path into `postprocess` |

### 5.2 Stage-4 reading

- **5** aten launches; almost all view/slice/unbind (little or no heavy GEMM).
- Explicit `mul`/`add` unnormalize may be inside `postprocess` with few extra GPU dispatches (or CPU); chrono shows only the `#5` `to` on the popped `[1,6]` — compare Nsight if a cast/elementwise kernel appears in the Stage‑4 window.
- No fusion opportunity that matters for latency here; Stage 4 is queue bookkeeping.

---

## 6. Potential fusion analysis (aten → larger op)

Cross-check **chrono aten rows** (this doc §1–5) against **Nsight kernels** (`smolVLA_kernel_gpu_list.md`). Goal: which **adjacent aten chains** could collapse into **one** fused op/kernel (today’s eager path often does **not**).

Legend: **Today** = what this capture already does; **Potential** = what a fused rewrite / Inductor / custom kernel could merge.

### 6.1 Elementwise chains

| Case | Stage / seq | Aten ops that could fuse → one op | Today | Potential fused op |
|---|---|---|---|---|
| Image `2x−1` | §1.0 | `mul` + `add` on `[B,3,512,512] float32` | Separate ATen / tiny kernels | `fused_scale_bias` / pointwise |
| Euler step | §4.4 `#1692–1693` | `mul` (`dt·v`) + `add` (`x+…`) | Two elementwise | `x = axpy(x, v, dt)` one kernel |
| Residual | §3.2 `#112`, `#127`; §1.2 `#60`,`#65` | lone `add` / `add_` after Linear | Often separate (or GEMM epilogue if bias-only) | Fuse into preceding GEMM epilogue when producer is Linear |
| Attn score scale | §3.2 `#101` | `mul_` after `QKᵀ` | Separate | Epilogue of `QKᵀ` GEMM (`scale` in-kernel) |
| Gate ⊗ up | §3.2 `#125`; §4.3 `#220` | `mul` between gate-SiLU and up | Separate | Fuse into SwiGLU epilogue (see §6.7) |
| Stage‑1 time emb | §2.1 `#4–8`,`#11` | several `mul` / `pow` / `reciprocal` on freqs | Many tiny FP64 kernels | One `sincos_embedding` kernel |

**Nsight signal:** many `BinaryFunctor` / `mul` / `add` / `direct_copy` rows between big GEMMs = unfused elementwise.

### 6.2 RMSNorm

VLM/expert norms are **not** a single `aten::rms_norm` in this chrono — they expand to a long chain.

| Case | Stage / seq | Aten chain (fuse candidates) | Today | Potential |
|---|---|---|---|---|
| Prefill input RMSNorm | §3.2 `#11–19` | `to` → `pow` → `mean` → `add`(eps) → `rsqrt` → `mul` → `to` → `mul`(weight) → `to` | **Many** kernels (mean/rsqrt/muls/casts) | One `rms_norm` kernel (stats in FP32, out bf16) |
| Prefill post-attn RMSNorm | §3.2 `#114–121` | same pattern | Same | Same |
| Expert RMSNorm | §4.2 `#17–25`, `#118–125`; §4.4 `#1665–1672` | same | Same | Same |
| Contrast: ViT LayerNorm | §1.2 `#46`,`#61` | single `aten::layer_norm` | Already **fused** (`vectorized_layer_norm_kernel`) | — |

**Potential fused aten:** `rms_norm(x, weight, eps)` replacing `#11–19`-style sequences (drop redundant `to` if weight/act stay bf16 with FP32 accum).

### 6.3 RoPE

| Case | Stage / seq | Aten ops that could fuse | Today | Potential |
|---|---|---|---|---|
| Angle build (Q) | §3.2 `#30–41` | `arange`/`mul`/`pow`/`unsqueeze`/`div`/`sin`/`cos` | Separate sin/cos + helpers (Nsight: `sin_kernel`/`cos_kernel`) | `rope_freqs(pos)` → sin/cos in one kernel |
| Apply rotate (Q) | §3.2 `#42–54` | `split`/`mul`×4/`sub`/`add`/`slice`/`copy_`/`to` | Many elementwise + copies | `rope_apply(x, sin, cos)` one kernel (in-place rotate halves) |
| Same for K | §3.2 `#55–80` | duplicate of Q path on 5 heads | Same | Share freqs; one apply kernel for K |
| Expert RoPE | §4.2 `#36–86`; odd Q-only `#158–182` | same pattern on len 50 | Same | Same fused `rope_apply` |

**Potential fused aten:** `apply_rotary_pos_emb(q, k, cos, sin)` (HF-style) instead of ~25 aten rows per of Q/K.

### 6.4 Matmul + transpose

| Case | Stage / seq | Aten ops | Today | Potential |
|---|---|---|---|---|
| Prefill `QKᵀ` layout | §3.2 `#97–100` | `transpose`×2–3 then `matmul` `[1,15,241,64]×[1,15,64,241]` | Transposes often **view** (no kernel); matmul is FP32 magma | cuBLASLt / CUTLASS with **transpose flags** (`TN`/`NT`) — avoid materializing Kᵀ if any copy appears |
| Prefill `A·V` | §3.2 `#107–108` | `permute` V then `matmul` | permute may be view; bf16 matmul | Batched GEMM with layout operands; fuse permute into GEMM args |
| Even expert `QKᵀ` | §4.2 `#101–104` | `transpose` then `matmul` scores `[…,50,291]` | Same | Same |
| ViT heads | §1.2 `#48–49` before SDPA | `view`+`transpose` | Views; SDPA fused | Already OK (no extra GEMM) |

**Potential fused aten:** `matmul(a, b.transpose(-1,-2))` lowered as one GEMM with `transB` — already true if transpose is a view; fuse only needed when a prior `to`/copy made layout contiguous explicitly (`#95–99` cast+transpose chain).

### 6.5 Matmul + cast

| Case | Stage / seq | Aten ops | Today | Potential |
|---|---|---|---|---|
| Prefill scores path | §3.2 `#95–100` | `to` (bf16→fp32 on Q/K) → `transpose` → **`matmul` FP32** | Cast kernels (`direct_copy`) **before** FP32 `QKᵀ` | GEMM that accepts bf16 inputs + FP32 accumulate (TC) — **one** op, drop pre-casts |
| Prefill `A·V` cast | §3.2 `#105–108` | `softmax` FP32 → `to` bf16 → `matmul` bf16 | Explicit cast between softmax and `A·V` | Keep probs FP32 into a mixed GEMM, or softmax→bf16 epilogue fused into `A·V` preload |
| Odd expert `A·V` | §4.3 `#199–202` | softmax → cast path → **FP32** `matmul` | All FP32 | Less cast pressure; still fuse scale/`where` (below) |
| Velocity head | §4.4 `#1690–1691` | `to` bf16→fp32 then `linear` FP32 | Cast then GEMM | FP32 GEMM epilogue reading bf16 act (or keep expert out FP32) |

**Potential fused aten:** `matmul_fp32acc(q_bf16, k_bf16)` / Linear with in-kernel cast — matches “BF16 IO, FP32 accum” TC math without a separate `aten::to`.

### 6.6 Matmul + expand (GQA)

| Case | Stage / seq | Aten ops | Today | Potential |
|---|---|---|---|---|
| Prefill K/V 5→15 | §3.2 `#89–94` then used in `#100`/`#108` | `unsqueeze`+`expand`+`reshape` (×2 for K,V) before attn | Expand often **view** (no DRAM); still many aten rows | Flash/SDPA or GEMM with **GQA** (repeat-KV in-kernel) — no expand aten |
| Even expert 5→15 over len 291 | §4.2 `#93–98` | same | Same | Same |
| Odd cross 5→15 | §4.3 `#183–188` | expand on prefix K/V | Same | Same |

**Potential fused aten:** attention/GEMM API that takes `num_kv_heads=5`, `num_q_heads=15` and broadcasts K/V inside the kernel (as Flash GQA does).

### 6.7 MLP (SwiGLU / GELU-MLP)

| Case | Stage / seq | Aten ops that could fuse | Today | Potential |
|---|---|---|---|---|
| VLM SwiGLU | §3.2 `#122–127` | `linear`(gate) + `silu` + `linear`(up) + `mul` + `linear`(down) + `add_` | **3 GEMMs** + SiLU + mul + residual; SiLU/mul separate (Nsight) | (a) fuse `silu`+`mul` into gate GEMM epilogue; (b) dual-GEMM SwiGLU kernel; (c) fuse residual into down GEMM |
| Expert SwiGLU | §4.2 `#126–131`; §4.3 `#217–222` | same on `[1,50,*]` | Same | Same |
| Stage‑1 fusion MLP | §2.1 `#19–21` | `linear` + `silu` + `linear` | SiLU separate (`silu_kernel`) | `linear_silu_linear` or epilogue SiLU on first GEMM |
| ViT GELU-MLP | §1.2 `#62–65` | `linear` + `gelu` + `linear` + `add` | GELU separate (`GeluCUDAKernelImpl`) | Fuse GELU into up-GEMM epilogue; residual into down |

**Potential fused aten:** `swiglu_mlp(x)` / `gelu_mlp(x)` as one or two launches instead of 5–6.

### 6.8 Attention

| Case | Stage / seq | Aten ops that could fuse → one large op | Today | Potential |
|---|---|---|---|---|
| **ViT self-attn (already fused)** | §1.2 `#56` | single `scaled_dot_product_attention` | **Flash** one kernel | Reference pattern |
| Prefill eager attn | §3.2 `#100–110` | `matmul`(QKᵀ) + `mul_` + `where` + `softmax` + `to` + `matmul`(A·V) + layout ops | **Unfused** (magma + where + softmax_warp + cutlass) — see kernel list §1 | `scaled_dot_product_attention` / Flash (or memory-efficient) like ViT |
| Even expert self-attn | §4.2 `#104–114` | same pattern, scores `[…,50,291]` | Unfused | Flash-decoding / SDPA with KV cache |
| Odd expert cross-attn | §4.3 `#194–204` | `matmul` + `mul_` + `where` + `softmax` + `matmul` (FP32 A·V) | Unfused | Fused cross-attn kernel; keep FP32 acc if needed |
| + QKV projections | §3.2 `#20–25`; §4.2 `#26–31` | three `linear`+`view` before attn | Three GEMMs | Fused QKV packed Linear (one GEMM, split Q/K/V) |
| + out projection | `#111` / `#115` / `#206` | `linear` after concat heads | Separate | Often kept separate; optional fuse with residual |

**Largest win vs chrono:** replace Stage 2/3 eager chains (`matmul`…`softmax`…`matmul`) with one SDPA/Flash-style aten — same as Stage 0 already does — and optionally pack QKV Linears.

### 6.9 How to verify a fusion claim

1. List the aten subsequence in chrono (this doc).
2. Find the Stage window in `smolVLA_kernel_gpu_list.md` / Nsight.
3. If **N atens → N kernels** → unfused (§6.1–6.8 “Today”).
4. If **N semantic steps → 1 kernel** (Flash, `vectorized_layer_norm`, CUTLASS epilogue) → already fused.
5. Re-run chrono after a change: fused APIs often **shrink** the aten list (e.g. many RoPE rows → one `apply_rotary_pos_emb`).

---

## 7. Skill — produce an AtenOp list for any GPU inference path

This section is the **playbook** used to build *this* file (`SmolVLA_AtenOp_List_gpu_backend.md`) from a live model. Apply the same steps to another policy / backbone / decode loop to get an equivalent ATen operator inventory (shapes, dtypes, call order, semantic roles).

### 7.1 Goal and deliverables

| Deliverable | Content |
|---|---|
| **Chrono JSON** | Per-stage list of `{i, name, input_shapes, input_dtypes}` in **call order** |
| **Chrono log** | Human-readable dump of the same events |
| **AtenOp markdown** | Staged tables: `Seq \| aten:: \| Input \| Output \| Semantic role`, plus repeat notes and reading tips |

**In scope:** what PyTorch’s ATen dispatcher ran (eager).  
**Out of scope here:** GPU kernel times / Nsight (pair later with a kernel list). Do **not** use this capture for latency.

### 7.2 Prerequisites

1. **Eager CUDA path** — `model.eval()` + `@torch.inference_mode()`; **no** `torch.compile`, CUDA graphs, or TensorRT for the capture run.
2. **Reproducible dummy batch** — fixed seed, known `B`, image size, sequence lengths (so shapes match across runs).
3. **Stage boundaries** — named callables that match how you think about the pipeline (e.g. prefix embed / prefill / suffix / expert step / postprocess).
4. **Reference script pattern** — this repo: `src/smolvla_aten_profile.py` (`ChronoAtenDispatch` via `TorchDispatchMode`).

### 7.3 Step-by-step (SmolVLA → generalize)

#### Step 1 — Split the inference into stages

Draw the production `select_action` / forward path as **numbered stages**. For SmolVLA:

| Stage | Callable wrapped | Repeat |
|---|---|---|
| 0 | `embed_prefix(...)` | 1× / chunk |
| 2 | VLM prefill `forward(..., use_cache)` | 1× / chunk |
| 1 | `embed_suffix(x_t, t)` | ×`M` Euler (capture **1** step) |
| 3 | expert `forward` + velocity + Euler | ×`M` (capture **1** step) |
| 4 | crop / queue / `postprocess` | 1× pop |

For another model: replace with *your* stages (e.g. encode / decode-layer / LM-head). Prefer **one** representative repeat unit when a loop would ×10–×N the log.

#### Step 2 — Implement chronological `TorchDispatchMode`

Copy the pattern from `smolvla_aten_profile.py`:

1. Subclass `torch.utils._python_dispatch.TorchDispatchMode`.
2. In `__torch_dispatch__`, append one event per call: normalized `aten::*` name, walked tensor **input** shapes/dtypes.
3. Optionally skip noise: `empty*` / `empty_strided` / `alias` / `detach` (re-enable if you need alloc detail).
4. **Do not** use `profiler.key_averages` (aggregates, loses order) or bare `TorchFunctionMode` (misses many C++ ATen ops).

#### Step 3 — Warmup, then record each stage separately

```text
load model → prep batch → warmup full path (no recorder)
for each stage:
    sync → with ChronoAtenDispatch: stage_fn() → sync
    store events under stages["k_name"]
write JSON + log
```

Rules:

- Enter the mode **only** around the stage callable (keeps stages separable).
- Wrap **preprocess** too if it matters (SmolVLA §1.0 `prepare_images` was *outside* `embed_prefix` — document gaps explicitly when a stage is missing from chrono).
- Keep dtype/device the same as production inference.

#### Step 4 — Run on the target GPU and archive artifacts

```bash
# Example (this repo)
source /venv/main/bin/activate   # or project venv
export HF_HOME=... SMOKE_DEVICE=cuda
python src/smolvla_aten_profile.py
# → doc/gpu/smolvla_aten_chrono.json
# → doc/gpu/smolvla_aten_chrono.log
```

For another project: point `OUT_JSON` / `OUT_LOG` at that model’s doc folder; keep `meta` (model id, torch, CUDA, GPU name, seed, step counts).

#### Step 5 — Find templates (repeating blocks)

From the log/JSON:

1. Count events per stage.
2. Locate the **first** layer / block / Euler unit by landmark ops (`linear` QKV, `matmul` scores, `softmax`, `silu`, …).
3. Measure the aten stride to the next identical landmark → **template length**.
4. Document: `example #A–#B`; repeat `×L` or `×M`.

SmolVLA examples: ViT block ×36; prefill layer **117** aten ×16; expert even `#17–131` / odd `#132–222` ×8 each per Euler step.

#### Step 6 — Build the markdown AtenOp tables

For each stage / template:

| Column | How to fill |
|---|---|
| **Seq** | Chrono `i` (or renumbered template-local index) |
| **aten::** | `name` from JSON (`aten::linear`, `aten::matmul`, …) |
| **Input** | `input_shapes` + `input_dtypes` (join multi-tensor with `×`) |
| **Output** | **Infer** from the next op’s inputs and known module math (chrono records **inputs only**) |
| **Semantic role** | Short label: `Q Linear`, `RMSNorm x²`, `Attn MatMul QKᵀ`, `RoPE (Q only)`, … |

Conventions that keep the doc usable:

- Bold **compute** rows (`linear`, `matmul`, `conv2d`, `softmax`, `scaled_dot_product_attention`, `silu`, …).
- Keep views (`view` / `reshape` / `transpose` / `expand`) when they explain layout (GQA 5→15, head merge).
- Note dtype surprises (e.g. odd expert Cross-K/V **FP32**, scores FP32, `A·V` FP32 vs even bf16 `A·V`).
- Mark capture gaps (`prepare_*` not wrapped, CPU tokenize, etc.).

#### Step 7 — Add stage reading + cross-checks

Per stage, write a short “reading” blurb:

- Event count; what repeats; fused vs eager attn (`sdpa` vs `matmul`+`softmax`).
- Fingerprints that separate variants (even KV **291** vs odd cross **241**).
- Optional: Nsight / kernel-list cross-ref for “1 aten → N kernels” and fusion candidates (§6 style).

#### Step 8 — Freeze the skill outputs

Check in (or publish):

1. `*_aten_chrono.json` + `.log`
2. `*_AtenOp_List_gpu_backend.md` (this document’s structure)
3. Pointer to the recorder script and §7 (this skill)

### 7.4 Checklist for a new inference case

- [ ] Stages named and wrapable as callables  
- [ ] Eager CUDA, no compile/graphs for capture  
- [ ] Dummy batch shapes documented in `meta`  
- [ ] `TorchDispatchMode` append-only recorder (not `key_averages`)  
- [ ] Noise filter decided (and documented)  
- [ ] Warmup once; record stages separately with CUDA sync  
- [ ] JSON + log written  
- [ ] Templates identified (first unit + stride + repeat count)  
- [ ] Markdown tables with inferred outputs + semantic roles  
- [ ] Gaps called out (code outside the wrapper)  
- [ ] Optional: pair with Nsight kernel list for fusion / dtype checks  

### 7.5 What to copy vs rewrite

| Keep as-is (skill) | Rewrite per model |
|---|---|
| `ChronoAtenDispatch` idea + JSON schema | Stage callables / model load / dummy batch |
| Table column layout + template method | Landmark ops, seq ranges, semantic labels |
| Limits in §0.1 (fusion, no timing, 1-step Euler) | Stage map, counts, dtype fingerprints |

### 7.6 Minimal adapter sketch (other model)

```python
# Pseudocode — same skill, new stages
def record(fn):
    rec = ChronoAtenDispatch(skip_noise=True)
    torch.cuda.synchronize()
    with rec:
        fn()
        torch.cuda.synchronize()
    return rec.events

ops_enc = record(lambda: model.encode(batch))
ops_dec = record(lambda: model.decode_one_layer(h, cache))  # one layer template
# write JSON/log → find template → fill AtenOp markdown (§7.3 steps 5–7)
```

**Bottom line:** wrap each logical stage with chronological `TorchDispatchMode`, archive JSON/log, extract the first repeating unit into typed tables, and label semantics — that is how this SmolVLA AtenOp file was produced and how to reproduce it for another inference stack.

---

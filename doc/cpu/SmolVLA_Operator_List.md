# SmolVLA Operator List

This document follows the inference path described in `SmolVLA_Algorithm_Architecture.md`:

1. prefix embedding;
2. action-suffix embedding;
3. prefix prefill and KV-cache construction;
4. iterative expert decode/denoise;
5. action post-processing and queue execution.

It describes the installed `lerobot/smolvla_base` checkpoint. Training-only loss operators are outside the main scope.

## 0. Notation, fixed dimensions, and datatype rules

| Symbol | Value | Meaning |
|---|---:|---|
| `B` | variable; normally 1 on a robot | Batch size |
| `C` | 3 | Number of camera views in this setup |
| `H_i × W_i` | `256 × 256` | Raw image size |
| `H_v × W_v` | `512 × 512` | Vision-model image size |
| `p` | 16 | Vision patch side |
| `S_v` | `(512/16)^2 = 1024` | ViT tokens before PixelShuffle |
| `d_v` | 768 | Vision hidden width |
| `h_v × d_h` | `12 × 64` | Vision attention heads and head width |
| `d_vff` | 3072 | Vision MLP width |
| `L_v` | 12 | Vision encoder blocks per camera |
| `r` | 4 | PixelShuffle spatial reduction factor |
| `S_img` | `1024/4^2 = 64` | Image tokens per camera |
| `S_lang` | 48 | Padded language-token count |
| `S_state` | 1 | State-token count |
| `S_p` | `3×64 + 48 + 1 = 241` | Prefix length |
| `d_p` | 960 | Prefix/VLM hidden width |
| `h_q / h_kv` | `15 / 5` | VLM and expert query/KV heads |
| `d_h` | 64 | VLM/expert attention-head width |
| `d_pff` | 2560 | VLM SwiGLU intermediate width |
| `L_p` | 16 | Truncated VLM layers |
| `N` | 50 | Action-chunk length |
| `D_max` | 32 | Padded action/state dimension |
| `D_robot` | 6 | Real robot action/state dimension here |
| `d_e` | 720 | Action-expert hidden width |
| `d_eff` | 2048 | Expert SwiGLU intermediate width |
| `L_e` | 16 | Expert layers: 8 even self-attention, 8 odd cross-attention |
| `M` | 10 | Euler denoising steps per inferred chunk |

### 0.1 Datatype legend

The installed checkpoint genuinely uses mixed precision:

- raw images, robot state, noisy actions, timestep, velocity, and returned actions: `float32`;
- vision, connector, token embedding, VLM, and most expert weights: `bfloat16`;
- state/action projections and time-fusion MLP weights: `float32`;
- odd expert cross-attention `K/V` projections: `float32`;
- token IDs and position IDs: `int64`;
- validity and attention masks: `bool`;
- installed text/expert code explicitly computes attention logits and softmax probabilities in `float32`;
- **documentation-only prefill assumption:** Section 3.2 instead models prefix `QKᵀ` in `bfloat16`, followed by an explicit score cast to `float32` before masking/softmax. The installed code is intentionally left unchanged;
- vision softmax is internally evaluated in `float32`, then converted back to `bfloat16`.

`RMSNorm` computes its variance in `float32`, then returns the activation datatype expected by the surrounding path. “View” operations (`reshape`, `transpose`, `permute`, `expand`) do not change datatype.

### 0.2 Repeat-count convention

Here, **an action chunk is one sequence of future robot actions produced by one full model inference**. For this checkpoint:

```text
model output before cropping: (B, 50, 32)
robot action chunk:            (B, 50, 6)
one robot action:              (B, 6)
```

Thus, one chunk contains `N = 50` consecutive robot actions. The robot normally places these 50 actions in a queue and executes one action per control tick. In every expression below, **`/chunk` means “for each complete 50-action sequence produced,” not “for each individual robot action.”**

- `1/chunk`: the operator runs once per complete 50-action sequence.
- `C/chunk = 3`: the operator runs three times per action chunk—once independently for each camera.
- `L_v/camera = 12`: the operator runs in 12 ViT blocks for each camera. With three cameras, this is `C × L_v = 3 × 12 = 36/chunk`.
- `L_p/chunk = 16`: the operator runs once in each of the 16 VLM prefill layers, for a total of 16 executions per action chunk.
- `M/chunk = 10`: the operator runs once in each of the 10 Euler denoising steps, for a total of 10 executions per action chunk.
- `L_e × M = 16 × 10 = 160/chunk`: the operator runs in all 16 expert layers during every one of the 10 Euler steps.
- `8 × M = 8 × 10 = 80/chunk`: the operator runs only in the eight even expert layers or only in the eight odd expert layers during every Euler step.

### 0.3 FLOP-counting convention for Stages 0 and 1

The rightmost column in Stages 0 and 1 reports operations for **one complete action chunk** and already includes the **Repeat** count. `B` remains symbolic; on a robot, set `B=1`.

Counts are separated into:

- **Basic:** addition, subtraction, multiplication, comparison, max/min, and similar low-level arithmetic, all assigned equal cost;
- **Div:** division or reciprocal;
- **Exp:** exponential;
- **Sin:** sine;
- **Cos:** cosine;
- **Sqrt:** square root.

Each count is prefixed by its performance-accounting datatype (`BF16`, `FP32`, `INT64`, or `Bool`). This prefix may differ from the actual input/output datatype.

For the Stage 0–2 performance model, **all arithmetic that the current implementation executes in FP64 is intentionally charged to the FP32 bucket**. The input/output datatype columns continue to show the real runtime tensors—for example, Stage 1 sinusoidal construction is still shown as `float64 → float64`—but its `Basic`, `Div`, `Exp`, `Sin`, `Cos`, and `Sqrt` counts are grouped with FP32. Casts and data movement still count as zero arithmetic FLOPs.

The estimates use these rules:

- one multiply-accumulate is `1 multiply + 1 add = 2 Basic`;
- a bias Linear with input width `K` costs `2K Basic` per output;
- a bias-free Linear or MatMul reduction of width `K` costs `2K-1 Basic` per output;
- bilinear resize is approximated as `4 multiplies + 3 adds = 7 Basic` per output pixel-channel;
- LayerNorm over width `D` is modeled as `7D-1 Basic + 3 Div + 1 Sqrt` per vector, using one inverse standard deviation;
- softmax over `K` values is modeled as `3K-2 Basic + K Exp + K Div` per row;
- vision GELU (`gelu_pytorch_tanh`) is modeled by rewriting `tanh` through exponential/division: `11 Basic + 1 Exp + 1 Div` per element;
- SiLU is modeled as `x/(1+exp(-x))`: `3 Basic + 1 Exp + 1 Div` per element;
- floating-point power in the period construction is modeled as `exp(fraction × precomputed_log_ratio)`;
- reshape, transpose, permute, expand, cast, concatenate, slice, allocation, index lookup, and data copy have `0` arithmetic FLOPs. They can still be bandwidth or latency bottlenecks.

These are **algorithmic workload estimates**, not exact CPU instructions. Vector kernels may fuse operations or use hardware approximations. The counts are suitable for comparing model blocks, while final performance analysis must also include memory traffic, datatype, kernel efficiency, and measured latency.

## 1. Stage 0 — prefix embedding

**Function:** convert three images, one language instruction, and one robot state into prefix `P`.

**Stage input**

```text
images:      3 × (B, 3, 256, 256), float32, range [0, 1]
instruction: B strings
state:       (B, 6), float32
```

**Stage output**

```text
P:           (B, 241, 960), float32 at concatenation
padding mask:(B, 241), bool
attention ar:(B, 241), bool
```

The mixed `bfloat16` image/language tokens and `float32` state token promote `P` to `float32` during `torch.cat`. The prefill projections cast their inputs to the corresponding `bfloat16` weights.

### 1.1 Image resize and normalization

Source block: `SmolVLAPolicy.prepare_images` and `vla_utils.resize_with_pad`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Select latest frame when a time axis exists | `(B,T,3,256,256) → (B,3,256,256)` | `float32 → float32` | `C/chunk` | `0` |
| 2 | Bilinear interpolate | `(B,3,256,256) → (B,3,512,512)` for square inputs | `float32 → float32` | `C/chunk` | `≈16,515,072B FP32 Basic` |
| 3 | Left/top constant pad | resized image → `(B,3,512,512)` | `float32 → float32` | `C/chunk` | `0` |
| 4 | Multiply and subtract, `2x-1` | `(B,3,512,512) →` same | `float32 → float32` | `C/chunk` | `4,718,592B FP32 Basic` |
| 5 | Create/cast camera-valid mask | `(B,) → (B,)` | source mask → `bool` | `C/chunk` | `0` |

For the square `256×256` inputs, interpolation directly reaches `512×512`, so the pad widths are zero. The padding operator still belongs to the general path.

**Section 1.1 total:** approximately `21,233,664B FP32 Basic`; no tensor-level `Div`, `Exp`, `Sin`, `Cos`, or `Sqrt`. Scalar shape calculations and interpolation address generation are excluded.

### 1.2 ViT patch embedding

Source block: `SmolVLMVisionEmbeddings`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Cast to vision-weight dtype | `(B,3,512,512) →` same | `float32 → bfloat16` | `C/chunk` | `0` |
| 2 | `Conv2d`, kernel/stride `16`, `3→768` | `(B,3,512,512) → (B,768,32,32)` | `bfloat16 → bfloat16` | `C/chunk` | `3,623,878,656B BF16 Basic` |
| 3 | Flatten spatial axes | `(B,768,32,32) → (B,768,1024)` | `bfloat16 → bfloat16` | `C/chunk` | `0` |
| 4 | Transpose token/feature axes | `(B,768,1024) → (B,1024,768)` | `bfloat16 → bfloat16` | `C/chunk` | `0` |
| 5 | Position-ID construction (`arange`, bucketize, index assignment) | patch mask `(B,32,32) → (B,1024)` | `bool → int64` | `C/chunk` | `≈3,354B INT64 Basic + 1,344B FP32 Basic + 6B FP32 Div` |
| 6 | Position-embedding lookup | `(B,1024) → (B,1024,768)` | `int64 → bfloat16` | `C/chunk` | `0` |
| 7 | Elementwise add | patch + position embeddings, both `(B,1024,768)` | `bfloat16 → bfloat16` | `C/chunk` | `2,359,296B BF16 Basic` |

**Section 1.2 total:** approximately `3,626,237,952B BF16 Basic + 1,344B FP32 Basic + 6B FP32 Div + 3,354B INT64 Basic`. The position-ID estimate assumes about five FP32 comparisons per `bucketize` binary search over 31 boundaries; lookup and tensor-layout operations contribute memory traffic but no arithmetic FLOPs.

### 1.3 ViT encoder: 12 blocks per camera

Source block: `SmolVLMEncoder` → `SmolVLMEncoderLayer`.

Every row below runs `12/camera = 36/chunk`. Sequence length `1024` and hidden width `768` do not change.

| Order | Operator/source sub-block | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | `LayerNorm1` | `(B,1024,768) →` same | `bfloat16 → bfloat16` | `36/chunk` | `198,144,000B FP32 Basic + 110,592B FP32 Div + 36,864B FP32 Sqrt` |
| 2 | Three Linear projections, `Q/K/V` | each `(B,1024,768) → (B,1024,768)` | `bfloat16 → bfloat16` | `3×36/chunk` | `130,459,631,616B BF16 Basic` |
| 3 | Reshape + transpose into heads | each `(B,1024,768) → (B,12,1024,64)` | `bfloat16 → bfloat16` | `3×36/chunk` | `0` |
| 4 | Batched MatMul, `QKᵀ` | `(B,12,1024,64) × (B,12,64,1024) → (B,12,1024,1024)` | `bfloat16 → bfloat16` | `36/chunk` | `57,529,073,664B BF16 Basic` |
| 5 | Multiply by `1/√64 = 1/8` | `(B,12,1024,1024) →` same | `bfloat16 → bfloat16` | `36/chunk` | `452,984,832B BF16 Basic` |
| 6 | Add bidirectional padding mask | `(B,12,1024,1024) →` same | `bfloat16 → bfloat16` | `36/chunk` | `≤452,984,832B BF16 Basic` |
| 7 | Softmax over key positions | `(B,12,1024,1024) →` same | internal `float32`, output `bfloat16` | `36/chunk` | `1,358,069,760B FP32 Basic + 452,984,832B FP32 Exp + 452,984,832B FP32 Div` |
| 8 | Batched MatMul, `A·V` | `(B,12,1024,1024) × (B,12,1024,64) → (B,12,1024,64)` | `bfloat16 → bfloat16` | `36/chunk` | `57,953,746,944B BF16 Basic` |
| 9 | Transpose + reshape, concatenate 12 heads | `(B,12,1024,64) → (B,1024,768)` | `bfloat16 → bfloat16` | `36/chunk` | `0` |
| 10 | Attention output Linear, `768→768` | `(B,1024,768) →` same | `bfloat16 → bfloat16` | `36/chunk` | `43,486,543,872B BF16 Basic` |
| 11 | First residual add | two `(B,1024,768) → (B,1024,768)` | `bfloat16 → bfloat16` | `36/chunk` | `28,311,552B BF16 Basic` |
| 12 | `LayerNorm2` | `(B,1024,768) →` same | `bfloat16 → bfloat16` | `36/chunk` | `198,144,000B FP32 Basic + 110,592B FP32 Div + 36,864B FP32 Sqrt` |
| 13 | MLP `fc1`, `768→3072` | `(B,1024,768) → (B,1024,3072)` | `bfloat16 → bfloat16` | `36/chunk` | `173,946,175,488B BF16 Basic` |
| 14 | GELU activation | `(B,1024,3072) →` same | `bfloat16 → bfloat16` | `36/chunk` | `1,245,708,288B BF16 Basic + 113,246,208B BF16 Exp + 113,246,208B BF16 Div` |
| 15 | MLP `fc2`, `3072→768` | `(B,1024,3072) → (B,1024,768)` | `bfloat16 → bfloat16` | `36/chunk` | `173,946,175,488B BF16 Basic` |
| 16 | Second residual add | two `(B,1024,768) → (B,1024,768)` | `bfloat16 → bfloat16` | `36/chunk` | `28,311,552B BF16 Basic` |

After all 12 blocks, `post_layernorm` runs once per camera:

```text
(B,1024,768) bfloat16 → (B,1024,768) bfloat16
```

Its cost across the three cameras is:

```text
16,512,000B FP32 Basic + 9,216B FP32 Div + 3,072B FP32 Sqrt
```

**Section 1.3 total by dtype:**

```text
BF16: 639,529,648,128B Basic
       113,246,208B Div
       113,246,208B Exp

FP32:    1,770,869,760B Basic
         453,215,232B Div
         452,984,832B Exp
              76,800B Sqrt
```

`Sin=0`, `Cos=0`. The BF16 Basic total includes the mask-add upper bound; an optimized all-valid bidirectional-mask path may omit that add.

### 1.4 PixelShuffle and connector

Source block: `SmolVLMConnector`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Reshape sequence to grid | `(B,1024,768) → (B,32,32,768)` | `bfloat16 → bfloat16` | `C/chunk` | `0` |
| 2 | Reshape/permute PixelShuffle groups | `(B,32,32,768) → (B,8,8,12288)` | `bfloat16 → bfloat16` | `C/chunk` | `0` |
| 3 | Flatten grid to sequence | `(B,8,8,12288) → (B,64,12288)` | `bfloat16 → bfloat16` | `C/chunk` | `0` |
| 4 | Bias-free connector Linear, `12288→960` | `(B,64,12288) → (B,64,960)` | `bfloat16 → bfloat16` | `C/chunk` | `4,529,664,000B BF16 Basic` |
| 5 | Multiply by `√960` | `(B,64,960) →` same | `bfloat16 → bfloat16` | `C/chunk` | `184,320B BF16 Basic + 3 scalar FP32-accounted Sqrt` |
| 6 | Expand camera-valid mask | `(B,) → (B,64)` | `bool → bool` | `C/chunk` | `0` |

PixelShuffle is only a data rearrangement; it has no learned weights and performs no arithmetic on token values.

**Section 1.4 total:** `4,529,848,320B BF16 Basic + 3 scalar FP32-accounted Sqrt`. Although PixelShuffle has zero arithmetic FLOPs, its rearrangement can incur memory/cache costs.

### 1.5 Language-token embedding

Source block: processor/tokenizer and `SmolVLMWithExpertModel.embed_language_tokens`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Tokenize, truncate, and pad | `B strings → (B,48)` | text → `int64` token IDs | `1/chunk` | `0` tensor FLOPs |
| 2 | Create language validity mask | `B token lists → (B,48)` | — → `bool` | `1/chunk` | `0` |
| 3 | Embedding-table gather, vocabulary `49280`, width `960` | `(B,48) → (B,48,960)` | `int64 → bfloat16` | `1/chunk` | `0` |
| 4 | Multiply by `√960` | `(B,48,960) →` same | `bfloat16 → bfloat16` | `1/chunk` | `46,080B BF16 Basic + 1 scalar FP32-accounted Sqrt` |

**Section 1.5 total:** `46,080B BF16 Basic + 1 scalar FP32-accounted Sqrt`. Tokenization and embedding lookup are excluded from arithmetic FLOPs but can contribute CPU and memory latency.

### 1.6 Robot-state embedding

Source block: input feature padding and `VLAFlowMatching.state_proj`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Zero allocation + slice copy | `(B,6) → (B,32)` | `float32 → float32` | `1/chunk` | `0` |
| 2 | Linear, `32→960`, with bias | `(B,32) → (B,960)` | `float32 → float32` | `1/chunk` | `61,440B FP32 Basic` |
| 3 | Unsqueeze sequence axis | `(B,960) → (B,1,960)` | `float32 → float32` | `1/chunk` | `0` |
| 4 | Create state-valid mask | — → `(B,1)` | — → `bool` | `1/chunk` | `0` |

**Section 1.6 total:** `61,440B FP32 Basic`; all exception categories are zero.

### 1.7 Prefix assembly

Source block: `VLAFlowMatching.embed_prefix`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Concatenate token sequences | `3×(B,64,960)`, `(B,48,960)`, `(B,1,960) → (B,241,960)` | mixed `bfloat16/float32 → float32` | `1/chunk` | `0` |
| 2 | Concatenate validity masks | `3×(B,64)`, `(B,48)`, `(B,1) → (B,241)` | `bool → bool` | `1/chunk` | `0` |
| 3 | Construct attention-group mask | 240 zeros + 1 one → `(B,241)` | `bool → bool` | `1/chunk` | `0` |

`add_image_special_tokens=false` and `prefix_length=0` for this checkpoint, so no extra special-image tokens or final prefix padding are added.

**Section 1.7 total:** `0` arithmetic FLOPs. Concatenation and mask construction still write `O(B×241×960)` activation data.

### 1.8 Stage-0 FLOP summary

For all three cameras and all 12 ViT blocks:

```text
BF16:
  Basic: 647,685,780,480B
  Div:       113,246,208B
  Exp:       113,246,208B

FP32:
  Basic:   1,792,166,208B
  Div:       453,215,238B
  Exp:       452,984,832B
  Sqrt:           76,800B + 4 scalar

INT64:
  Basic:           3,354B

Sin: 0
Cos: 0
```

For the normal robot case `B=1`, the Basic categories still sum to approximately `649.478 billion`, but their datatype split matters for throughput modeling. The 12-block vision encoder contributes over 98% of the Stage-0 Basic count and is the clear arithmetic bottleneck within prefix embedding.

## 2. Stage 1 — action-suffix embedding

**Function:** combine current noisy action sample `x_t` and Euler flow time `t` into expert input tokens `U_t`.

**Stage input**

```text
x_t: (B,50,32), float32
t:   (B,), float32
```

**Stage output**

```text
U_t:         (B,50,720), float32
suffix masks:(B,50), bool/float32 as noted below
```

Source block: `VLAFlowMatching.embed_suffix` and `create_sinusoidal_pos_embedding`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Action input Linear, `32→720` | `(B,50,32) → (B,50,720)` | `float32 → float32` | `M/chunk` | `23,040,000B FP32 Basic` |
| 2 | `linspace` for 360 frequency fractions | scalar endpoints → `(360,)` | — → `float64` on CPU | `M/chunk` | `≈7,200 FP32-accounted Basic + 10 FP32-accounted Div` |
| 3 | Power/multiply to construct periods `p_k` | `(360,) → (360,)` | `float64 → float64` | `M/chunk` | `7,200 FP32-accounted Basic + 3,600 FP32-accounted Exp` |
| 4 | Reciprocal/multiply, `2π/p_k` | `(360,) → (360,)` | `float64 → float64` | `M/chunk` | `3,600 FP32-accounted Basic + 3,600 FP32-accounted Div` |
| 5 | Broadcast multiply with `t` | `(B,1) × (1,360) → (B,360)` | promotes to `float64` | `M/chunk` | `3,600B FP32-accounted Basic` |
| 6 | Elementwise `sin` and `cos` | each `(B,360) → (B,360)` | `float64 → float64` | `2×M/chunk` | `3,600B FP32-accounted Sin + 3,600B FP32-accounted Cos` |
| 7 | Concatenate sine/cosine features | `2×(B,360) → (B,720)` | `float64 → float64` | `M/chunk` | `0` |
| 8 | Cast time embedding | `(B,720) →` same | `float64 → float32` | `M/chunk` | `0` |
| 9 | Unsqueeze + broadcast across horizon | `(B,720) → (B,50,720)` | `float32 → float32` | `M/chunk` | `0` |
| 10 | Concatenate action/time features | `2×(B,50,720) → (B,50,1440)` | `float32 → float32` | `M/chunk` | `0` |
| 11 | Fusion Linear, `1440→720` | `(B,50,1440) → (B,50,720)` | `float32 → float32` | `M/chunk` | `1,036,800,000B FP32 Basic` |
| 12 | SiLU | `(B,50,720) →` same | `float32 → float32` | `M/chunk` | `1,080,000B FP32 Basic + 360,000B FP32 Exp + 360,000B FP32 Div` |
| 13 | Fusion Linear, `720→720` | `(B,50,720) → (B,50,720)` | `float32 → float32` | `M/chunk` | `518,400,000B FP32 Basic` |
| 14 | Create suffix padding mask | — → `(B,50)` | — → `bool` | `M/chunk` | `0` |
| 15 | Create/expand suffix attention-group mask | — → `(B,50)` | — → `float32` | `M/chunk` | `0` |

Thus every learned Stage-1 operator runs 10 times per chunk with the default Euler solver.

### 2.1 Stage-1 FLOP summary

Across all `M=10` suffix reconstructions:

```text
FP32:
  Basic: 1,579,323,600B + 18,000
  Div:         360,000B +  3,610
  Exp:         360,000B +  3,600
  Sin:           3,600B
  Cos:           3,600B

Sqrt: 0
```

For `B=1`:

```text
FP32 Basic: 1,579,341,600
FP32 Div:         363,610
FP32 Exp:         363,600
FP32 Sin:           3,600
FP32 Cos:           3,600
```

The first fusion Linear (`1440→720`) contributes about 65.6% of Stage-1 Basic FLOPs, the second fusion Linear contributes about 32.8%, and the action-input Linear contributes about 1.5%. The sinusoidal construction is charged to FP32 in this analysis even though the current CPU implementation uses `float64`; its transcendental-kernel latency should still be measured.

## 3. Stage 2 — prefix prefill and KV-cache construction

**Function:** process `P` once through the 16-layer VLM text stack and cache each layer's prefix keys and values.

**Stage input**

```text
P:            (B,241,960), float32 immediately after assembly
prefix masks: (B,241), bool
```

**Stage output**

```text
16 cache entries:
K_prefix[ell], V_prefix[ell]: (B,5,241,64), bfloat16
```

### 3.1 Prefix masks and positions

Source block: `make_att_2d_masks` and `VLAFlowMatching.sample_actions`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Cumulative sum of attention-group mask | `(B,241) → (B,241)` | `bool → int64` | `1/chunk` | `240B INT64 Basic` |
| 2 | Broadcast comparison to build 2-D attention mask | `(B,241) → (B,241,241)` | `int64 → bool` | `1/chunk` | `58,081B INT64 Basic` |
| 3 | Construct padding-pair validity and AND with group permission | `(B,241,241) →` same | `bool → bool` | `1/chunk` | `116,162B Bool Basic` |
| 4 | Cumulative sum minus one for position IDs | `(B,241) → (B,241)` | `bool → int64` | `1/chunk` | `481B INT64 Basic` |
| 5 | Allocate empty `DynamicCache` | no entries → 16 layer slots as filled | — | `1/chunk` | `0` |

**Section 3.1 total:** `58,802B INT64 Basic + 116,162B Bool Basic`. These mask operations are small arithmetically but materialize a `(B,241,241)` Boolean tensor.

### 3.2 One VLM prefill layer

Source block: `SmolVLMWithExpertModel.forward_attn_layer`, `eager_attention_forward`, and the corresponding Llama layer.

Every row in this subsection runs `L_p = 16` times per chunk unless its repeat cell says otherwise. Layer 0 may receive the `float32` mixed prefix; projection inputs are explicitly converted to `bfloat16`. Later hidden states are `bfloat16`.

> **Documentation-only performance assumption:** rows 9–14 below model `QKᵀ` and score scaling in `bfloat16`, then cast the score tensor to `float32` immediately before masking and softmax. This is the requested target precision flow for analysis. It does **not** describe the unchanged installed implementation, which currently casts Q and K to `float32` before `QKᵀ`. Stage 3 continues to document the actual installed-code precision flow.

Stage-2 RMSNorm is modeled per 960-wide token as `3D FP32 Basic + 2 FP32 Div + 1 FP32 Sqrt` for RMS statistics/normalization, plus `D` scale multiplies in the residual-stream dtype. Bias-free Linears and MatMuls use `2K-1 Basic` per output.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Input RMSNorm | `(B,241,960) →` same | variance in `float32`; activation preserved, then cast for projections | `16/chunk` | `11,336,640B FP32 Basic + 3,470,400B BF16 Basic + 7,712B FP32 Div + 3,856B FP32 Sqrt` |
| 2 | Q Linear, `960→960` | `(B,241,960) → (B,241,960)` | cast to `bfloat16 → bfloat16` | `16/chunk` | `7,103,677,440B BF16 Basic` |
| 3 | K and V Linears, each `960→320` | each `(B,241,960) → (B,241,320)` | cast to `bfloat16 → bfloat16` | `2×16/chunk` | `4,735,784,960B BF16 Basic` |
| 4 | Reshape into heads | `Q→(B,241,15,64)`; `K,V→(B,241,5,64)` | `bfloat16 → bfloat16` | `3×16/chunk` | `0` |
| 5 | RoPE angle construction (`arange`, power, divide) + `sin/cos` | positions `(B,241) →` sine/cosine terms `(B,241,1,32)` | internal `float32` | `2×16/chunk` | `2,048 FP32 Basic + (246,784B + 32) FP32 Div + 1,024 FP32 Exp + 246,784B FP32 Sin + 246,784B FP32 Cos` |
| 6 | RoPE multiply/add on Q and K | shapes unchanged | internal `float32`, output `bfloat16` | `2×16/chunk` | `14,807,040B FP32 Basic` |
| 7 | Cache append/update | empty layer slot + `(B,5,241,64)` → same cached shape | `bfloat16 → bfloat16` | `16/chunk` | `0` |
| 8 | Expand/reshape each K and V from 5 to 15 heads | `(B,241,5,64) → (B,241,15,64)` | `bfloat16 → bfloat16` | `2×16/chunk` | `0` |
| 9 | Transpose Q/K heads first; no precision cast | each → `Q:(B,15,241,64)`, `K:(B,15,241,64)` | `bfloat16 → bfloat16` | `2×16/chunk` | `0` |
| 10 | Batched MatMul, `QKᵀ` | `(B,15,241,64) × (B,15,64,241) → (B,15,241,241)` | `bfloat16 → bfloat16` | `16/chunk` | `1,770,308,880B BF16 Basic` |
| 11 | Multiply by `1/8` | `(B,15,241,241) →` same | `bfloat16 → bfloat16` | `16/chunk` | `13,939,440B BF16 Basic` |
| 12 | Cast attention scores before stable masking/softmax | `(B,15,241,241) →` same | `bfloat16 → float32` | `16/chunk` | `0` |
| 13 | `where(mask, score, -∞)` | mask `(B,1,241,241)`, scores `(B,15,241,241)` | `bool/float32 → float32` | `16/chunk` | `13,939,440B FP32 Basic` |
| 14 | Softmax over 241 keys | `(B,15,241,241) →` same | `float32 → float32`, then cast `bfloat16` | `16/chunk` | `41,702,640B FP32 Basic + 13,939,440B FP32 Exp + 13,939,440B FP32 Div` |
| 15 | Batched MatMul, `A·V` | `(B,15,241,241) × (B,15,241,64) → (B,15,241,64)` | `bfloat16 → bfloat16` | `16/chunk` | `1,780,546,560B BF16 Basic` |
| 16 | Permute + reshape, concatenate 15 heads | `(B,15,241,64) → (B,241,960)` | `bfloat16 → bfloat16` | `16/chunk` | `0` |
| 17 | Attention output Linear, `960→960` | `(B,241,960) →` same | `bfloat16 → bfloat16` | `16/chunk` | `7,103,677,440B BF16 Basic` |
| 18 | First residual add | two `(B,241,960) → (B,241,960)` | output path becomes `bfloat16` | `16/chunk` | `3,701,760B BF16 Basic` |
| 19 | Post-attention RMSNorm | `(B,241,960) →` same | stats `float32`, output `bfloat16` | `16/chunk` | `11,105,280B FP32 Basic + 3,701,760B BF16 Basic + 7,712B FP32 Div + 3,856B FP32 Sqrt` |
| 20 | SwiGLU gate Linear and up Linear, each `960→2560` | each `(B,241,960) → (B,241,2560)` | `bfloat16 → bfloat16` | `2×16/chunk` | `37,886,279,680B BF16 Basic` |
| 21 | SiLU on gate branch | `(B,241,2560) →` same | `bfloat16 → bfloat16` | `16/chunk` | `29,614,080B BF16 Basic + 9,871,360B BF16 Exp + 9,871,360B BF16 Div` |
| 22 | Elementwise gate multiply | `2×(B,241,2560) → (B,241,2560)` | `bfloat16 → bfloat16` | `16/chunk` | `9,871,360B BF16 Basic` |
| 23 | Down Linear, `2560→960` | `(B,241,2560) → (B,241,960)` | `bfloat16 → bfloat16` | `16/chunk` | `18,949,309,440B BF16 Basic` |
| 24 | Second residual add | two `(B,241,960) → (B,241,960)` | `bfloat16 → bfloat16` | `16/chunk` | `3,701,760B BF16 Basic` |

**Section 3.2 total by dtype:**

```text
BF16:
  Basic: 67,558,122,560B
  Div:        9,871,360B
  Exp:        9,871,360B

FP32:
  Basic:     92,891,040B + 2,048
  Div:       14,201,648B + 32
  Exp:       13,939,440B + 1,024
  Sin:          246,784B
  Cos:          246,784B
  Sqrt:           7,712B
```

### 3.3 Final prefix norm

The text model's final RMSNorm runs once after layer 15:

```text
(B,241,960) bfloat16 → (B,241,960) bfloat16
```

Its arithmetic cost is:

```text
BF16: 231,360B Basic
FP32: 694,080B Basic + 482B Div + 241B Sqrt
```

The normalized hidden-state output is not needed by action sampling; the persistent result of prefill is the 16-layer KV cache.

### 3.4 Stage-2 FLOP summary

Including mask construction, all 16 prefill layers, and final RMSNorm:

```text
BF16:
  Basic: 67,558,353,920B
  Div:        9,871,360B
  Exp:        9,871,360B

FP32:
  Basic:     93,585,120B + 2,048
  Div:       14,202,130B + 32
  Exp:       13,939,440B + 1,024
  Sin:          246,784B
  Cos:          246,784B
  Sqrt:           7,953B

INT64:
  Basic:          58,802B

Bool:
  Basic:         116,162B
```

For `B=1`, all Basic categories sum to approximately `67.652 billion` operations. About 99.86% of those Basic operations are BF16 under the documentation-only BF16-score-MatMul assumption. The two SwiGLU input projections and the down projection dominate Stage-2 arithmetic.

## 4. Stage 3 — expert decode and Euler denoising

**Function:** start from Gaussian action noise and evaluate the 16-layer expert `M=10` times while integrating from `t=1` to `t=0`.

### 4.1 Euler-loop setup

Source block: `sample_noise` and `euler_integrate`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Gaussian random sampling | requested `(B,50,32) → x₁ (B,50,32)` | — → `float32` | `1/chunk` | `0` modeled arithmetic FLOPs; RNG cost is separate |
| 2 | Scalar divide, `dt=-1/M` | scalar → scalar `-0.1` | Python float | `1/chunk` | `1 FP32-accounted Div` |
| 3 | Scalar multiply/add, `t=1+step·dt` | loop index → scalar | Python float | `M/chunk` | `20 FP32-accounted Basic` |
| 4 | Tensor construction + expand | scalar `t → (B,)` | — → `float32` | `M/chunk` | `0` |
| 5 | Stage-1 suffix embedding | `x_t,t → U_t (B,50,720)` | `float32 → float32` | `M/chunk` | Stage-1 total; included once in the Stage-3 grand total |
| 6 | Expert velocity evaluation | `U_t + prefix cache → v_t (B,50,32)` | mixed → `float32` | `M/chunk` | Sum of Sections 4.4–4.7; not added again in this row |
| 7 | Multiply, `dt·v_t` | `(B,50,32) →` same | `float32 → float32` | `M/chunk` | `16,000B FP32 Basic` |
| 8 | Add, `x_t ← x_t + dt·v_t` | `2×(B,50,32) → (B,50,32)` | `float32 → float32` | `M/chunk` | `16,000B FP32 Basic` |

**Section 4.1 direct loop-overhead total, excluding the referenced suffix/expert evaluations:** `32,000B FP32 Basic + 20 scalar FP32-accounted Basic + 1 scalar FP32-accounted Div`.

### 4.2 Suffix masks and positions in each Euler step

Source block: `VLAFlowMatching.denoise_step`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Unsqueeze + expand prefix-valid mask | `(B,241) → (B,50,241)` | `bool → bool` | `M/chunk` | `0` |
| 2 | Build suffix 2-D mask | `(B,50)` masks → `(B,50,50)` | mask logic → `bool` | `M/chunk` | `25,490B FP32 Basic + 50,000B Bool Basic` |
| 3 | Concatenate prefix/suffix key masks | `(B,50,241)` + `(B,50,50) → (B,50,291)` | `bool → bool` | `M/chunk` | `0` |
| 4 | Sum valid prefix positions | `(B,241) → (B,1)` | `bool → int64` | `M/chunk` | `2,400B INT64 Basic` |
| 5 | Cumulative sum + prefix offset | `(B,50) → (B,50)` | `bool → int64` | `M/chunk` | `1,490B INT64 Basic` |

**Section 4.2 total:** `25,490B FP32 Basic + 3,890B INT64 Basic + 50,000B Bool Basic`.

### 4.3 Expert layer schedule

For every Euler step:

```text
layer 0,2,...,14: suffix self-attention with prefix + temporary suffix K/V
layer 1,3,...,15: suffix-to-prefix cross-attention
```

Each expert layer then applies its own attention output projection, residual, RMSNorm, SwiGLU MLP, and second residual.

For Stage 3, RMSNorm over expert width `D=720` uses the same model as Stage 2: `3D FP32 Basic + 2 FP32 Div + 1 FP32 Sqrt` for statistics/normalization per token, plus `D` scale multiplies in the residual-stream dtype. Bias-free Linear and MatMul reductions use `2K-1 Basic` per output.

### 4.4 Even expert layers: self-attention and temporary cache append

Source block: `forward_attn_layer`; applies to 8 even layers per Euler step, or `80/chunk`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Expert input RMSNorm | `(B,50,720) →` same | stats `float32`; first ingress may be `float32`, later path `bfloat16` | `80/chunk` | `9,000,000B FP32 Basic + 2,520,000B BF16 Basic + 8,000B FP32 Div + 4,000B FP32 Sqrt` |
| 2 | Q Linear, `720→960` | `(B,50,720) → (B,50,960)` | cast to `bfloat16 → bfloat16` | `80/chunk` | `5,525,760,000B BF16 Basic` |
| 3 | K and V Linears, each `720→320` | each `(B,50,720) → (B,50,320)` | cast to `bfloat16 → bfloat16` | `2×80/chunk` | `3,683,840,000B BF16 Basic` |
| 4 | Reshape into Q15/K5/V5 heads | `Q:(B,50,15,64)`; `K,V:(B,50,5,64)` | `bfloat16 → bfloat16` | `3×80/chunk` | `0` |
| 5 | RoPE generation and application on Q and K | head shapes unchanged | internal frequencies/arithmetic `float32`, Q/K output `bfloat16` | `2×80/chunk` | `15,360,000B FP32 Basic + 10,240 scalar FP32 Basic + (256,000B + 160) FP32 Div + 5,120 FP32 Exp + 256,000B FP32 Sin + 256,000B FP32 Cos` |
| 6 | `cache.update`: append suffix K/V | prefix `(B,5,241,64)` + suffix `(B,5,50,64) → (B,5,291,64)` | `bfloat16 → bfloat16` | `80/chunk` | `0` |
| 7 | Expand/reshape K and V, 5→15 heads | each `(B,291,5,64) → (B,291,15,64)` | `bfloat16 → bfloat16` | `2×80/chunk` | `0` |
| 8 | Cast/transpose Q and K | `Q→(B,15,50,64)`, `K→(B,15,291,64)` | `bfloat16 → float32` | `2×80/chunk` | `0` |
| 9 | MatMul `QKᵀ`, scale `1/8` | → `(B,15,50,291)` | `float32 → float32` | `80/chunk` | `2,234,880,000B FP32 Basic` |
| 10 | Mask with `where` | mask `(B,1,50,291)` and scores → same score shape | `bool/float32 → float32` | `80/chunk` | `17,460,000B FP32 Basic` |
| 11 | Softmax over 291 keys | `(B,15,50,291) →` same | `float32`, then cast `bfloat16` | `80/chunk` | `52,260,000B FP32 Basic + 17,460,000B FP32 Exp + 17,460,000B FP32 Div` |
| 12 | MatMul `A·V` | → `(B,15,50,64)` | `bfloat16 → bfloat16` | `80/chunk` | `2,231,040,000B BF16 Basic` |
| 13 | Permute + reshape, concatenate heads | `(B,15,50,64) → (B,50,960)` | `bfloat16 → bfloat16` | `80/chunk` | `0` |

The 291 keys are exactly `241` fixed prefix positions plus `50` temporary suffix positions. Each layer owns a separate cache entry.

**Section 4.4 total by dtype:**

```text
BF16:
  Basic: 11,443,160,000B

FP32:
  Basic:  2,328,960,000B + 10,240
  Div:       17,724,000B + 160
  Exp:       17,460,000B + 5,120
  Sin:          256,000B
  Cos:          256,000B
  Sqrt:           4,000B
```

### 4.5 Odd expert layers: cross-attention to fixed prefix

Source block: `forward_cross_attn_layer`; applies to 8 odd layers per Euler step, or `80/chunk`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Expert input RMSNorm | `(B,50,720) →` same | stats `float32`, output path `bfloat16` | `80/chunk` | `8,640,000B FP32 Basic + 2,880,000B BF16 Basic + 8,000B FP32 Div + 4,000B FP32 Sqrt` |
| 2 | Q Linear, `720→960`, reshape Q15 | `(B,50,720) → (B,50,15,64)` | cast to `bfloat16 → bfloat16` | `80/chunk` | `5,525,760,000B BF16 Basic` |
| 3 | Read and flatten prefix K5/V5 | each `(B,5,241,64) → (B,241,320)` | `bfloat16 → bfloat16` | `2×80/chunk` | `0` |
| 4 | Cross K/V Linear, each `320→320` | each `(B,241,320) → (B,241,320)` | cast `bfloat16→float32`; output `float32` | `2×80/chunk` | `7,884,748,800B FP32 Basic` |
| 5 | Reshape K5/V5 | each `(B,241,320) → (B,241,5,64)` | `float32 → float32` | `2×80/chunk` | `0` |
| 6 | Reduce-min and subtract to rebase suffix position IDs at zero | `(B,50) → (B,50)` | `int64 → int64` | `80/chunk` | `7,920B INT64 Basic` |
| 7 | Apply RoPE to suffix Q only | `Q:(B,50,15,64) →` same | internal `float32`, output `bfloat16` | `80/chunk` | `11,520,000B FP32 Basic + 5,120 scalar FP32 Basic + (128,000B + 80) FP32 Div + 2,560 FP32 Exp + 128,000B FP32 Sin + 128,000B FP32 Cos` |
| 8 | Expand/reshape K and V, 5→15 heads | each `(B,241,5,64) → (B,241,15,64)` | `float32 → float32` | `2×80/chunk` | `0` |
| 9 | Cast/transpose Q and K | `Q→(B,15,50,64)`, `K→(B,15,241,64)` | both become `float32` | `2×80/chunk` | `0` |
| 10 | MatMul `QKᵀ`, scale `1/8` | → `(B,15,50,241)` | `float32 → float32` | `80/chunk` | `1,850,880,000B FP32 Basic` |
| 11 | Mask with `where` | scores remain `(B,15,50,241)` | `bool/float32 → float32` | `80/chunk` | `14,460,000B FP32 Basic` |
| 12 | Softmax over 241 prefix keys | `(B,15,50,241) →` same | `float32 → float32` | `80/chunk` | `43,260,000B FP32 Basic + 14,460,000B FP32 Exp + 14,460,000B FP32 Div` |
| 13 | MatMul `A·V` | → `(B,15,50,64)` | `float32 → float32` | `80/chunk` | `1,847,040,000B FP32 Basic` |
| 14 | Permute + reshape, concatenate heads | `(B,15,50,64) → (B,50,960)` | `float32 → float32` | `80/chunk` | `0` |

Odd layers do not append suffix K/V to the cache. They derive cross-attention keys and values from the fixed prefix cache for that layer. The cached prefix K was already RoPE-rotated during prefill; after the learned cross-K projection, the implementation does not apply RoPE to it again.

**Section 4.5 total by dtype:**

```text
BF16:
  Basic:  5,528,640,000B

FP32:
  Basic: 11,660,548,800B + 5,120
  Div:       14,596,000B + 80
  Exp:       14,460,000B + 2,560
  Sin:          128,000B
  Cos:          128,000B
  Sqrt:           4,000B

INT64:
  Basic:           7,920B
```

### 4.6 Common expert attention output, residual, and MLP

Source block: shared tail in `SmolVLMWithExpertModel.forward`.

These rows run in every expert layer: `16 × 10 = 160/chunk`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Cast attention result to output-weight dtype | `(B,50,960) →` same | even `bfloat16`; odd `float32→bfloat16` | `160/chunk` | `0` |
| 2 | Attention output Linear, `960→720` | `(B,50,960) → (B,50,720)` | `bfloat16 → bfloat16` | `160/chunk` | `11,053,440,000B BF16 Basic` |
| 3 | First residual add | two `(B,50,720) → (B,50,720)` | result `bfloat16` | `160/chunk` | `5,760,000B BF16 Basic` |
| 4 | Clone residual for second skip | `(B,50,720) →` same | `bfloat16 → bfloat16` | `160/chunk` | `0` |
| 5 | Post-attention RMSNorm | `(B,50,720) →` same | stats `float32`, output `bfloat16` | `160/chunk` | `17,280,000B FP32 Basic + 5,760,000B BF16 Basic + 16,000B FP32 Div + 8,000B FP32 Sqrt` |
| 6 | SwiGLU gate and up Linears, each `720→2048` | each `(B,50,720) → (B,50,2048)` | `bfloat16 → bfloat16` | `2×160/chunk` | `47,153,152,000B BF16 Basic` |
| 7 | SiLU on gate branch | `(B,50,2048) →` same | `bfloat16 → bfloat16` | `160/chunk` | `49,152,000B BF16 Basic + 16,384,000B BF16 Exp + 16,384,000B BF16 Div` |
| 8 | Elementwise gate multiply | `2×(B,50,2048) → (B,50,2048)` | `bfloat16 → bfloat16` | `160/chunk` | `16,384,000B BF16 Basic` |
| 9 | Down Linear, `2048→720` | `(B,50,2048) → (B,50,720)` | `bfloat16 → bfloat16` | `160/chunk` | `23,587,200,000B BF16 Basic` |
| 10 | Second residual add | two `(B,50,720) → (B,50,720)` | `bfloat16 → bfloat16` | `160/chunk` | `5,760,000B BF16 Basic` |

**Section 4.6 total by dtype:**

```text
BF16:
  Basic: 81,876,608,000B
  Div:       16,384,000B
  Exp:       16,384,000B

FP32:
  Basic:     17,280,000B
  Div:           16,000B
  Sqrt:           8,000B
```

### 4.7 Expert final norm, velocity head, and cache restoration

Source block: end of `SmolVLMWithExpertModel.forward` and `VLAFlowMatching.denoise_step`.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Expert final RMSNorm | `(B,50,720) →` same | stats `float32`, output `bfloat16` | `M/chunk` | `1,080,000B FP32 Basic + 360,000B BF16 Basic + 1,000B FP32 Div + 500B FP32 Sqrt` |
| 2 | Slice last `N=50` suffix tokens | `(B,50,720) → (B,50,720)` | `bfloat16 → bfloat16` | `M/chunk` | `0` |
| 3 | Cast expert output | `(B,50,720) →` same | `bfloat16 → float32` | `M/chunk` | `0` |
| 4 | Action output Linear, `720→32` | `(B,50,720) → (B,50,32)` | `float32 → float32` | `M/chunk` | `23,040,000B FP32 Basic` |
| 5 | `cache.crop(241)` | even-layer cache `(B,5,291,64) → (B,5,241,64)` | `bfloat16 → bfloat16` | `M/chunk` | `0` |

Cropping occurs after all 16 expert layers in an Euler step. It removes only the temporary 50-token suffix entries appended by even layers; the original prefix cache remains fixed for the next Euler step.

**Section 4.7 total by dtype:**

```text
BF16:
  Basic:    360,000B

FP32:
  Basic: 24,120,000B
  Div:        1,000B
  Sqrt:         500B
```

### 4.8 Stage-3 FLOP summary

#### Expert-only total

This first total covers Sections 4.4–4.7: all 160 expert-layer executions, final expert normalization, and 10 velocity projections. It excludes suffix rebuilding, mask preparation, and Euler updates.

```text
BF16:
  Basic: 98,848,768,000B
  Div:       16,384,000B
  Exp:       16,384,000B

FP32:
  Basic: 14,030,908,800B + 15,360
  Div:       32,337,000B +    240
  Exp:       31,920,000B +  7,680
  Sin:          384,000B
  Cos:          384,000B
  Sqrt:          16,500B

INT64:
  Basic:           7,920B
```

#### Inclusive Stage-3 total

This total adds:

- the Stage-1 suffix rebuilds, already counted across all 10 Euler steps;
- Section 4.2 suffix-mask/position construction;
- direct Euler scalar and `x_t` update operations from Section 4.1.

```text
BF16:
  Basic: 98,848,768,000B
  Div:       16,384,000B
  Exp:       16,384,000B

FP32:
  Basic: 15,610,289,890B + 33,380 scalar
  Div:       32,697,000B +  3,851 scalar
  Exp:       32,280,000B + 11,280 scalar
  Sin:          387,600B
  Cos:          387,600B
  Sqrt:          16,500B

INT64:
  Basic:          11,810B

Bool:
  Basic:          50,000B
```

For `B=1`, all inclusive Basic categories sum to approximately `114.459 billion` operations. The common expert SwiGLU/output stack in Section 4.6 is the largest arithmetic contributor. Unlike the documentation-only Stage-2 assumption, Stage 3 here follows the installed implementation: attention score MatMuls are FP32.

## 5. Stage 4 — action post-processing and execution queue

**Function:** retain the real robot dimensions, queue the generated horizon, convert each selected normalized action to robot units, and execute one action per control tick.

Source blocks: `SmolVLAPolicy._get_action_chunk`, `SmolVLAPolicy.select_action`, the LeRobot output processor, and the action queue.

This checkpoint uses `adapt_to_pi_aloha=false` and `ACTION` normalization mode `MEAN_STD`. Unnormalization is therefore:

```text
action_robot = action_norm * std + mean
```

with one multiply and one add per action dimension. The FLOPs/chunk column already includes the full horizon of control ticks when an operator runs once per tick.

| Order | Operator | Input → output shape | Input → output dtype | Repeat | FLOPs/chunk |
|---:|---|---|---|---:|---:|
| 1 | Slice/crop padded action dimension | `(B,50,32) → (B,50,6)` | `float32 → float32` | `1/chunk` | `0` |
| 2 | Optional Pi-Aloha action-coordinate conversion | `(B,50,6) → (B,50,6)` | `float32 → float32` | `1/chunk` when enabled | `0` for this checkpoint (`adapt_to_pi_aloha=false`) |
| 3 | Transpose batch/horizon and queue insertion | `(B,50,6) → 50 × (B,6)` | `float32 → float32` | `1/chunk` | `0` |
| 4 | Queue pop | queue head → `(B,6)` | `float32 → float32` | once per control tick, up to `N/chunk` | `0` |
| 5 | Output-processor unnormalization, MEAN_STD | `(B,6) → (B,6)` | `float32 → float32` | once per control tick, up to `N/chunk` | `600B FP32 Basic` |
| 6 | Send robot command | `(B,6) → robot interface` | normally `float32` before device conversion | once per control tick, up to `N/chunk` | `0` arithmetic FLOPs |

Unnormalization arithmetic detail for one full chunk that consumes all `N=50` queued actions:

```text
per tick:   B × 6 × (1 multiply + 1 add) = 12B FP32 Basic
per chunk:  50 × 12B = 600B FP32 Basic
```

Slice, transpose, queue push/pop, and robot I/O are data movement / control-path operations. They contribute memory and latency, but no modeled arithmetic FLOPs under the Stage 0–3 convention.

If `adapt_to_pi_aloha` were enabled on a larger dual-arm action space, Stage 4 would additionally include joint sign flips and Aloha gripper angular conversions (`unnormalize`, `normalize`, and `arcsin`). That path is inactive for `lerobot/smolvla_base`.

The policy queue stores model-space actions; the normal LeRobot runtime applies the output postprocessor after each pop. The 50 queue pops are not model evaluations. In synchronous operation, the next full Stage-0-to-Stage-4 model inference begins when the queue is empty.

### 5.1 Stage-4 FLOP summary

For one complete chunk that executes all `N=50` queued robot commands (`B` symbolic; set `B=1` on a robot):

```text
FP32:
  Basic: 600B

BF16:  0
Div:   0
Exp:   0
Sin:   0
Cos:   0
Sqrt:  0
INT64: 0
Bool:  0
```

For `B=1`:

```text
FP32 Basic: 600
```

Stage 4 is arithmetically negligible relative to Stages 0–3. Its cost is dominated by queue/runtime I/O latency, not FLOPs. Across one full model inference that produces and then executes one 50-action chunk, Stage 4 contributes about `600` FP32 Basic operations versus hundreds of billions in the earlier stages.

## 6. Compact repetition summary

For one default action-chunk inference (`C=3`, `L_v=12`, `L_p=16`, `L_e=16`, `M=10`):

| Block | Repetitions per chunk |
|---|---:|
| Image preprocessing | 3 |
| ViT patch embedding | 3 |
| ViT encoder block | `3×12 = 36` |
| PixelShuffle + connector | 3 |
| Language/state embedding and prefix assembly | 1 |
| VLM prefill layer | 16 |
| Suffix embedding | 10 |
| Even expert self-attention layer | `8×10 = 80` |
| Odd expert cross-attention layer | `8×10 = 80` |
| Common expert residual + MLP block | `16×10 = 160` |
| Velocity projection | 10 |
| Euler update | 10 |
| Full model inference | 1 per produced chunk |
| Queue pop / robot command | up to 50 control ticks |

## 7. Cross-stage FLOP comparison (Stages 0–4)

All numbers below are for **one complete action-chunk inference** with `B=1`, already including each stage’s loop multiplicities (`C`, `L_v`, `L_p`, `M`, and control-tick count). FP64 arithmetic is charged to the FP32 bucket, as in Sections 1–5.

To avoid double-counting Stage 1, Stage 3 is split into:

- **Stage 3 expert**: Sections 4.4–4.7 only;
- **Stage 3 overhead**: Euler `x`/`t` updates and suffix-mask/position construction from Sections 4.1–4.2.

Stage 1 is listed separately because it is the suffix rebuild that runs inside the Euler loop. The end-to-end total is the sum of all disjoint columns.

### 7.1 Basic FLOPs by stage

| Category | Stage 0 prefix embed | Stage 1 suffix embed | Stage 2 prefill | Stage 3 expert | Stage 3 overhead | Stage 4 postprocess | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 Basic | 647,685,780,480 | 0 | 67,558,353,920 | 98,848,768,000 | 0 | 0 | 814,092,902,400 |
| FP32 Basic | 1,792,166,208 | 1,579,341,600 | 93,587,168 | 14,030,924,160 | 57,510 | 600 | 17,496,077,246 |
| INT64 Basic | 3,354 | 0 | 58,802 | 7,920 | 3,890 | 0 | 73,966 |
| Bool Basic | 0 | 0 | 116,162 | 0 | 50,000 | 0 | 166,162 |
| **All Basic** | **649,477,950,042** | **1,579,341,600** | **67,652,116,052** | **112,879,700,080** | **111,400** | **600** | **831,589,219,774** |
| Share of end-to-end Basic | 78.10% | 0.19% | 8.14% | 13.57% | ≈0% | ≈0% | 100% |

### 7.2 Exception operators by stage

| Category | Stage 0 | Stage 1 | Stage 2 | Stage 3 expert | Stage 3 overhead | Stage 4 | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 Div | 113,246,208 | 0 | 9,871,360 | 16,384,000 | 0 | 0 | 139,501,568 |
| BF16 Exp | 113,246,208 | 0 | 9,871,360 | 16,384,000 | 0 | 0 | 139,501,568 |
| FP32 Div | 453,215,238 | 363,610 | 14,202,162 | 32,337,240 | 1 | 0 | 500,118,251 |
| FP32 Exp | 452,984,832 | 363,600 | 13,940,464 | 31,927,680 | 0 | 0 | 499,216,576 |
| FP32 Sin | 0 | 3,600 | 246,784 | 384,000 | 0 | 0 | 634,384 |
| FP32 Cos | 0 | 3,600 | 246,784 | 384,000 | 0 | 0 | 634,384 |
| FP32 Sqrt | 76,800 + 4 scalar | 0 | 7,953 | 16,500 | 0 | 0 | 101,253 + 4 scalar |

### 7.3 Bottleneck reading

```text
End-to-end Basic ≈ 831.589 billion

Stage 0  ≈ 78.1%   vision encoder dominates
Stage 3  ≈ 13.6%   expert decode / denoise (expert layers only)
Stage 2  ≈  8.1%   VLM prefix prefill
Stage 1  ≈  0.2%   suffix rebuild across 10 Euler steps
Stage 4  ≈  0.0%   unnormalize / queue / I/O
```

Among floating-point Basic operations only:

```text
BF16 Basic ≈ 97.9% of (BF16 + FP32) Basic
FP32 Basic ≈  2.1% of (BF16 + FP32) Basic
```

So the arithmetic bottleneck for one chunk inference is Stage 0 (especially the 12-block ViT), then Stage 3 expert layers, then Stage 2 prefill. Stage 1 and Stage 4 are negligible in FLOP count; Stage 4 may still matter for measured latency because of queueing and robot I/O.

Equivalence check against earlier summaries:

```text
Stage 3 inclusive Basic
  = Stage 1 + Stage 3 expert + Stage 3 overhead
  = 1,579,341,600 + 112,879,700,080 + 111,400
  = 114,459,153,080
```

```text
End-to-end Basic
  = Stage 0 + Stage 2 + Stage 3 inclusive + Stage 4
  = 649,477,950,042 + 67,652,116,052 + 114,459,153,080 + 600
  = 831,589,219,774
```

## 8. Code-source map

| Logical block | Installed source |
|---|---|
| Image preparation, prefix/suffix embedding, sampling, denoise step | `.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py` |
| Joint VLM/expert layer scheduling and custom attention | `.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py` |
| Resize/pad, sinusoidal embedding, 2-D mask construction | `.venv/lib/python3.12/site-packages/lerobot/policies/common/vla_utils.py` |
| Euler integration and Gaussian-noise sampling | `.venv/lib/python3.12/site-packages/lerobot/policies/common/flow_matching.py` |
| Vision patch embedding, ViT blocks, PixelShuffle, connector | `.venv/lib/python3.12/site-packages/transformers/models/smolvlm/modeling_smolvlm.py` |
| Llama RMSNorm, RoPE, grouped-query attention, SwiGLU | `.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py` |

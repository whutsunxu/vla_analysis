# SmolVLA Algorithm Architecture

## 0. End-to-end workflow (prefix embed → suffix embed → prefill → decode/denoise)


Generally, two embedding products feed the rest of the net: prefix **`P`** (once per observation) and suffix **`U_t`** (every Euler step).

```
prepare (LeRobot)     cameras, state, task string
        |
        v
0  prefix embed       img | lang | state tokens              ->  P     (B, 241, 960)
        |
        v
1  suffix embed       noisy actions + flow time              ->  U_t   (B, 50, 720)
        |                                                        (rebuilt each Euler step)
        v
2  prefill            VLM self-attn on P                     ->  KV cache (once)
        |
        v
3  decode / denoise   expert Q=U_t  x  prefix KV, Euler x 10 ->  action chunk (B, 50, 6)
        |
        v
       select_action queue   pop one step                        ->  (B, 6)
```


The **runtime control flow** is:

```
New observation arrives: cameras, state, task string
        |
        v
Stage 0: build prefix P
        |
        v
Stage 2: prefill P -> fixed prefix KV cache
        |
        v
Initialize x_1 ~ N(0, I)
        |
        v
+--------------------------------------------------+
| Euler loop: k = 0 ... 9                          |
|                                                  |
|   Stage 1: (current x_t, current t) -> U_t       |
|        |                                         |
|        v                                         |
|   Stage 3: (U_t, fixed prefix cache) -> v_t      |
|        |                                         |
|        v                                         |
|   Update x_t <- x_t + dt * v_t                   |
+--------------------------------------------------+
        |
        v
Crop x_0: (B, 50, 32) -> action chunk (B, 50, 6)
        |
        v
Put 50 actions into queue
        |
        v
Each robot tick: pop and execute one (B, 6) command
        |
        v
Queue empty? -- no --> keep popping
        |
       yes
        |
        +----> acquire a new observation and return to Stage 0
```

Therefore:

- **Stages 0 and 2 run once** when a new action chunk is needed.
- **Stages 1 and the denoising part of Stage 3 run 10 times**, once per Euler step.
- The prefix KV cache from Stage 2 stays fixed throughout those 10 iterations.
- Post-processing and queue creation happen once after Euler finishes.
- The robot executes the 50 queued commands without rerunning the model; after the queue empties, the cycle starts again with a new observation.

Detailed formulas, attention internals, and tensor transformations are documented separately in `SmolVLA_Algorithm_Architecture.md`.

## 1. Prefix embed (out: `P`)

**Meaning.** Turn the three observation modalities (cameras, instruction, robot state) into VLM tokens of width `d = 960` and concatenate them on the sequence axis. **Out:** prefix `P` with shape `(B, 241, 960)`. Built **once** per new observation; not rebuilt inside the Euler loop.

### 1.1 Image (each camera, independently)

Run 1.1.1–1.1.5 once per camera (`c = 1, 2, 3`). `B` is usually 1. The three `I_c` are concatenated in 1.4; this is **not** a stacked tensor `3×3×256×256`. Overall:

```
v_c = ViT(x_c)          # 1.1.1–1.1.3
z_c = PixelShuffle_4(v_c)
I_c = W_conn(z_c)
```

#### 1.1.1 Prepare (before the ViT weights)

```
x_c = 2 * ResizePad_512(raw_c) - 1
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `raw_c` | One camera RGB, float in `[0, 1]` | `(B, 3, 256, 256)` | `3` = RGB, **not** 3 cameras. `256` = `input_features` camera size |
| This step | `ResizePad_512` | Scale so the long side fits 512, pad leftover on **left/top** with 0 (aspect ratio kept) | `(B, 3, 512, 512)` | SmolVLM2 `image_size = 512`. Square 256→256 scale ×2, so this is just upsample + no pad |
| This step | `2x-1` | Map pixels `[0, 1] → [-1, 1]` | `(B, 3, 512, 512)` | What SigLIP / SmolVLM vision expects |
| Output | `x_c` | Prepared image that actually enters `ViT` | `(B, 3, 512, 512)` in `[-1, 1]` | Same spatial size as `image_size` |

#### 1.1.2 Patch embedding (this is where 1024 and 768 appear)

```
g_c = Conv2d(x_c; in=3, out=768, kernel=16, stride=16)
e_c = FlattenHW(g_c) + PE
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x_c` | Prepared RGB image | `(B, 3, 512, 512)` | 1.1.1 |
| This step | `Conv2d` | Non-overlapping 16×16 tiles. Each tile (16×16×3 numbers) is linearly mapped to one 768-vector. Implemented as conv, not an explicit unfold | `(B, 768, 32, 32)` | Spatial: `(512/16)×(512/16) = 32×32`. Channels: vision `hidden_size = 768` |
| This step | `FlattenHW` | Collapse the 32×32 grid into a sequence (no CLS token) | `(B, 1024, 768)` | `1024 = 32×32` patches. Order is row-major over the grid |
| This step | `PE` | Add a learned position embedding, one vector per patch index | `(B, 1024, 768)` | Table size `1024 × 768`; addition does not change shape |
| Output | `e_c` | Patch tokens + positions | `(B, 1024, 768)` | Already the ViT sequence length and width |

There is **no class / `[CLS]` token**. SigLIP-style SmolVLM keeps all 1024 patches.

#### 1.1.3 ViT encoder (shape stays `(B, 1024, 768)`)

```
v_c = LayerNorm( Encoder_12(e_c) )
```

The encoder has 12 independent Pre-LN transformer blocks. It uses ordinary 12-head attention: unlike the later language/action stacks, the vision encoder does **not** use 5-head grouped-query attention. It also does not add RoPE here because learned 2-D patch-position embeddings were already added in 1.1.2.

| Symbol | Value | Meaning |
|---|---|---|
| `S_v` | 1024 | Number of image-patch tokens |
| `d_v` | 768 | Vision hidden width |
| `n_h` | 12 | Attention heads |
| `d_h` | 64 | Width of each head: `768 / 12` |
| `d_ff` | 3072 | Vision MLP intermediate width |
| `L_v` | 12 | Number of encoder blocks |

##### 1.1.3.1 Loop across the 12 blocks

```
H^0 = e_c                                      # (B, 1024, 768)

for ell = 0 .. 11:
    N1 = LayerNorm1_ell(H^ell)
    T  = SelfAttention_ell(N1)
    R  = H^ell + T

    N2 = LayerNorm2_ell(R)
    M  = MLP_ell(N2)
    H^{ell+1} = R + M

v_c = PostLayerNorm(H^12)                     # (B, 1024, 768)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^0 = e_c` | Patch embeddings plus learned positions from 1.1.2 | `(B, 1024, 768)` | 1024 patches, each represented by 768 features |
| Loop | `ell` | Apply one attention block with its own weights | 12 iterations | `num_hidden_layers = 12` |
| Per block | `H^{ell+1}` | Updated patch representations | `(B, 1024, 768)` | Attention and MLP both project back to width 768 |
| Final step | `PostLayerNorm` | Normalize the output of the twelfth block | `(B, 1024, 768)` | Shape unchanged |
| Output | `v_c` | Contextualized patch sequence | `(B, 1024, 768)` | Input to pixel shuffle in 1.1.4 |

Each block has different learned weights. The 1024 tokens are not reduced inside `Encoder_12`; token reduction happens later in pixel shuffle.

##### 1.1.3.2 First LayerNorm

For every patch token independently, `LayerNorm(768)` computes statistics over its last 768 features:

```
mu     = mean(H^ell, dim=-1, keepdim=True)            # (B, 1024, 1)
var    = mean((H^ell - mu)^2, dim=-1, keepdim=True)  # (B, 1024, 1)
N1     = gamma1 ⊙ (H^ell - mu) / sqrt(var + eps) + beta1
                                                        # (B, 1024, 768)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell` | Residual stream entering block `ell` | `(B, 1024, 768)` | One 768-vector per patch |
| This step | `mu, var` | Mean and variance over hidden features only | each `(B, 1024, 1)` | Batch items and patch positions are not mixed |
| This step | `gamma1, beta1` | Learned feature-wise scale and bias | each `(768,)` | Broadcast across `B` and 1024 tokens |
| Output | `N1` | Normalized tokens used to form Q, K, V | `(B, 1024, 768)` | LayerNorm does not change shape |

This is regular LayerNorm with mean subtraction and bias—not the RMSNorm used later by the Llama-style text and action stacks.

##### 1.1.3.3 Q, K, V projections and head split

```
Q_flat = N1 W_Q                              # (B, 1024, 768)
K_flat = N1 W_K                              # (B, 1024, 768)
V_flat = N1 W_V                              # (B, 1024, 768)

Q = split_heads(Q_flat)                      # (B, 12, 1024, 64)
K = split_heads(K_flat)                      # (B, 12, 1024, 64)
V = split_heads(V_flat)                      # (B, 12, 1024, 64)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `N1` | Normalized patch tokens | `(B, 1024, 768)` | Output of 1.1.3.2 |
| This step | `W_Q` | Learned linear `768 → 768` | `Q_flat`: `(B, 1024, 768)` | One query vector per patch |
| This step | `W_K` | Learned linear `768 → 768` | `K_flat`: `(B, 1024, 768)` | One key vector per patch |
| This step | `W_V` | Learned linear `768 → 768` | `V_flat`: `(B, 1024, 768)` | One value vector per patch |
| This step | split + transpose | Reshape 768 into 12 heads × 64 features | each `(B, 12, 1024, 64)` | `12 × 64 = 768` |

There are 12 Q heads, 12 K heads, and 12 V heads—no KV-head repetition. The learned position information is already present in `N1`; there is no additional RoPE operation in this vision-attention block.

##### 1.1.3.4 Scaled dot-product and softmax

```
scores = (Q K^T) / sqrt(64)                  # (B, 12, 1024, 1024)
       = (Q K^T) / 8

A = softmax(scores + mask, dim=key)          # (B, 12, 1024, 1024)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `Q, K` | Per-head queries and keys | each `(B, 12, 1024, 64)` | 12 heads, 1024 patches, head width 64 |
| This step | `Q K^T` | Compare every query patch with every key patch | `(B, 12, 1024, 1024)` | One 1024×1024 score matrix per head |
| This step | `/ 8` | Scale by `sqrt(d_h) = sqrt(64)` | same | Prevent excessively large softmax logits |
| This step | mask | Exclude padded/invalid patches when present | same | For these full 512×512 inputs, all 1024 patches are valid |
| Output | `A` | Attention weights over the key-patch axis | `(B, 12, 1024, 1024)` | Every row sums to 1 over 1024 keys |

Vision attention is **bidirectional**, not causal: every valid patch may attend to every other valid patch.

##### 1.1.3.5 Weighted values, concatenate heads, and output projection

```
O_heads = A V                                # (B, 12, 1024, 64)
O = O_heads.transpose(1, 2)                  # (B, 1024, 12, 64)
O = O.reshape(B, 1024, 12*64)                # (B, 1024, 768)
T = O W_O                                    # (B, 1024, 768)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `A V` | Weighted sum of all value patches, independently per head | `(B, 12, 1024, 64)` | One 64-D result per head and query patch |
| This step | transpose + reshape | Concatenate the 12 head outputs per patch | `(B, 1024, 768)` | `12 × 64 = 768` |
| This step | `W_O` | Learned output projection `768 → 768` | `T`: `(B, 1024, 768)` | Mixes information across heads |

The concatenation joins **head feature vectors**, not patch tokens. Sequence length remains 1024.

##### 1.1.3.6 First residual, second LayerNorm, and MLP

```
R  = H^ell + T                              # first residual
N2 = LayerNorm2_ell(R)

F  = GELU_tanh(N2 W_1)                      # W_1: 768 → 3072
M  = F W_2                                  # W_2: 3072 → 768

H^{ell+1} = R + M                           # second residual
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell, T` | Original block input and attention output | each `(B, 1024, 768)` | Same shape permits elementwise residual addition |
| This step | `R = H^ell + T` | Skip connection around LayerNorm1 + attention | `(B, 1024, 768)` | Preserves the incoming representation |
| This step | `LayerNorm2` | New LayerNorm with its own `gamma2, beta2` | `N2`: `(B, 1024, 768)` | Normalizes each patch’s 768 features |
| This step | `W_1` | Expand `768 → 3072` independently per patch | `(B, 1024, 3072)` | `intermediate_size = 3072` |
| This step | `GELU_tanh` | Vision MLP nonlinearity | `(B, 1024, 3072)` | Elementwise; shape unchanged |
| This step | `W_2` | Contract `3072 → 768` | `M`: `(B, 1024, 768)` | Return to residual width |
| Output | `H^{ell+1} = R + M` | Skip connection around LayerNorm2 + MLP | `(B, 1024, 768)` | Feeds the next encoder block |

The MLP does not mix patch positions; it transforms each patch independently. Self-attention is the operation that mixes information among the 1024 patches.

##### 1.1.3.7 Final post-LayerNorm

After all 12 blocks:

```
v_c = PostLayerNorm(H^12)                    # (B, 1024, 768)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^12` | Output of the last encoder block | `(B, 1024, 768)` | Sequence and width were preserved across all blocks |
| This step | `PostLayerNorm` | Final `LayerNorm(768)` | `(B, 1024, 768)` | Same per-token formula as 1.1.3.2, with separate learned parameters |
| Output | `v_c` | Final contextualized vision tokens | `(B, 1024, 768)` | Passed to pixel shuffle in 1.1.4 |

#### 1.1.4 Pixel shuffle

```
z_c = PixelShuffle_4(v_c)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `v_c` | ViT patches from 1.1.3 | `(B, 1024, 768)` | Same as 1.1.3 output |
| This step | `PixelShuffle_4` | Pack every **4×4** neighboring patches into **one** token (sequence ÷ 16, channels × 16) | — | Factor `4` is the SmolVLM connector downsample |
| Output | `z_c` | Fewer, wider visual tokens | `(B, 64, 12288)` | `64 = 1024 / 4²`; `12288 = 768 × 16` |

#### 1.1.5 Connector MLP

```
I_c = W_conn(z_c)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `z_c` | Shuffled tokens from 1.1.4 | `(B, 64, 12288)` | Same as 1.1.4 output |
| This step | `W_conn` | Learned linear projection (connector MLP) from vision width into VLM width | — | So visual tokens can sit next to language tokens |
| Output | `I_c` | Visual token sequence for camera `c` | `(B, 64, 960)` | Sequence length stays 64; `960` = text `hidden_size` |

Three cameras give `3 × 64 = 192` visual tokens that later become the image part of `P`.

### 1.2 Instruction

| | Tensor |
|---|---|
| In | `task` string → token ids `(B, 48)` (`B` usually 1) |
| Embedding table | `(B, 48, 960)` |

```
L = Embed(token_ids)     # lookup table, not a 48-D vector
```

### 1.3 State

| | Tensor |
|---|---|
| In | `(B, 6)` (or `(B, 1, 6)`) |
| Pad | `(B, 32)` |
| `state_proj` | `(B, 960)` → unsqueeze → `(B, 1, 960)` |

```
s_32 = pad(s_6)          # 6 real DoF, 26 zeros
S    = W_s * s_32        # then unsqueeze to (B, 1, 960)
```

### 1.4 Prefix concat (sequence axis, not pixel stack)

```
P = concat(I_1, I_2, I_3, L, S)     # shape (B, 241, 960)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `I_1, I_2, I_3` | 64 visual tokens per camera | each `(B, 64, 960)` | 1.1.5, three views |
| Input | `L` | Instruction embeddings | `(B, 48, 960)` | `tokenizer_max_length = 48` |
| Input | `S` | Projected state, one token | `(B, 1, 960)` | 6-DoF padded to 32 then `W_s` |
| Output | `P` | Observation prefix | `(B, 241, 960)` | `192 + 48 + 1 = 241`. This is **the** 1 output |

## 2. Suffix embed (out: `U_t`)

**Meaning.** Turn the current noisy action chunk and the flow-matching time `t` into action-expert tokens. **Out:** suffix `U_t` with shape `(B, 50, 720)`. This is **not** a tokenizer path and not VLM width 960 (`720 = 0.75 × 960`). Rebuilt **every** Euler step because `x_t` and `t` change; `P` from 1 stays fixed.

`t` is **flow time** (a scalar in `[0, 1]` for the whole chunk), **not** the 50-step action index. Euler runs `num_steps=10` with `dt = -1/10`, `t = 1, 0.9, …, 0.1`. Start from Gaussian noise at `t=1`; at `t=0` the chunk should be the action.

### 2.1 Noisy action chunk

```
x_t ∈ R^{B × 50 × 32}
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x_t` | Current guess of the **whole** future chunk. Inference starts as `N(0, I)`. Each Euler step: `x ← x + dt * v_t` | `(B, 50, 32)` | `50 = chunk_size` (horizon). `32 = max_action_dim` (6 real DoF padded with 26 zeros, same idea as state) |
| Time | `t` | One scalar per batch item: how far along the noise→action path | `(B,)` | Shared by all 50 horizon slots. **Not** “step 0..49 of the robot” |

### 2.2 Project actions into expert width

```
a_t = W_a x_t
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x_t` | Noisy padded actions | `(B, 50, 32)` | 2.1 |
| This step | `W_a` | `action_in_proj`: linear `32 → 720`, applied at **each** of the 50 time slots | — | Expert `hidden_size = 720` |
| Output | `a_t` | One expert-width vector per horizon step | `(B, 50, 720)` | Sequence length stays 50; width matches the action transformer |

### 2.3 Sinusoidal time embedding, then broadcast

`p` and `τ` are **tensors**, not one scalar. `p` is a fixed length-360 vector of periods (not learned). `t` is `(B,)`. Sine/cosine are **elementwise** on a `(B, 360)` phase matrix.

#### 2.3.1 Period vector

```
α = linspace(0, 1, 360)                         # shape (360,)
p = min_period * (max_period / min_period)^α    # shape (360,), elementwise power
  = 4e-3 * 1000^α
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `α` | Even samples in `[0, 1]` | `(360,)` | `720/2 = 360` frequency channels |
| This step | `min_period` | Fastest period (scalar config) | `4e-3` | Scales the whole vector |
| This step | `max_period` | Slowest period (scalar config) | `4` | `1000 = 4 / 4e-3` |
| Output | `p` | Period of every sin/cos pair | `(360,)` | `p[0] = 4e-3`, `p[359] = 4`. Geometric, so log-spacing |

#### 2.3.2 Phase, PE vector, then broadcast to the chunk

```
phase = 2π * t[:, None] / p[None, :]            # (B, 1) / (1, 360) → (B, 360)
τ     = concat( sin(phase), cos(phase), dim=-1) # (B, 720)
τ_B   = τ[:, None, :].expand(B, 50, 720)        # (B, 50, 720)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `t` | Flow time | `(B,)` | One float per batch item |
| Input | `p` | Period vector from 2.3.1 | `(360,)` | Shared across the batch |
| This step | `t[:, None] / p[None, :]` | Outer divide: every batch `t` against every period | `(B, 360)` | Broadcast `(B,1)` with `(1,360)` |
| This step | `phase` | `2π` times that ratio | `(B, 360)` | Elementwise |
| This step | `τ` | `sin` and `cos` of `phase`, concat on last axis | `(B, 720)` | `(B,360)` + `(B,360)`. One PE vector per batch item |
| This step | `τ_B` | Repeat that vector on the horizon axis | `(B, 50, 720)` | Same `τ` for all 50 action slots |

### 2.4 Fuse action + time (2-layer MLP)

```
h   = SiLU( W_in  [ a_t | τ_B ] )
U_t =        W_out h
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `[a_t \| τ_B]` | Concat last dim: action features beside time features | `(B, 50, 1440)` | `720 + 720` |
| This step | `W_in` | `action_time_mlp_in`: linear `1440 → 720` | `(B, 50, 720)` | Mix action and time into expert width |
| This step | `SiLU` | Swish nonlinearity | `(B, 50, 720)` | Shape unchanged |
| This step | `W_out` | `action_time_mlp_out`: linear `720 → 720` | `(B, 50, 720)` | Second MLP layer |
| Output | `U_t` | Suffix tokens for the action expert | `(B, 50, 720)` | These are the expert’s input embeddings for this Euler step |

The one-line form is the same pipeline:

```
U_t = W_out( SiLU( W_in( concat(W_a x_t,  broadcast(PE(t))) ) ) )
```

`U_t` is **the** 2 output. 4 later lets the expert attend **from** `U_t` **to** the cached prefix `K, V`, and `action_out_proj` maps `(B, 50, 720) → (B, 50, 32)` to get velocity `v_t`. Only the first 6 of those 32 dims are real robot commands after unpadding.

## 3. Prefill (out: prefix KV cache)

**Meaning.** Run the **VLM text stack** (16 Llama-style blocks) **once** on `P` with **self-attention**. Image, language, and state mix here (language/image do not attend to state). Store **K, V** per layer. The action expert is idle. **Out:** 16-layer KV cache over sequence length 241. Runs once per observation, not per Euler step.

Sizes used in every block:

| Symbol | Value | Role |
|---|---|---|
| `S` | 241 | prefix length |
| `d` | 960 | VLM hidden |
| `n_h` | 15 | query heads |
| `n_kv` | 5 | key/value heads (GQA: each KV head is shared by 3 query heads) |
| `d_h` | 64 | `head_dim` |
| `d_ff` | 2560 | MLP width |

The one-line `K_ell, V_ell = SelfAttn_ell(P)` is **not** attention in isolation. Each layer is a full Pre-LN block; only the **K, V** of that layer are cached for 4.

### 3.1 Layer loop

```
H^0 = P                                 # (B, 241, 960)
for ell = 0 .. 15:
    H^{ell+1},  (K_ell, V_ell) = Block_ell(H^ell)
cache = { (K_ell, V_ell)  for ell = 0 .. 15 }
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^0` | Prefix from 1 | `(B, 241, 960)` | `P` |
| This step | `ell` | Same block recipe, **new weights** each layer | 16 iterations | `num_vlm_layers = 16` (SmolVLM2 truncated from 32) |
| Output | `H^{16}` | Final prefix hidden (then a last RMSNorm) | `(B, 241, 960)` | Shape never changes |
| Output | cache | What 4 actually reads | 16 × `(K, V)` | One pair per layer; **not** three separate image/lang/state caches |

Decode does **not** concat `K_img`, `K_lang`, `K_state` after the fact. Those tokens already sit at different **positions** of this one sequence.

Inside `Block_ell`, in order:

### 3.2 Input RMSNorm

Llama text uses **RMSNorm**, not mean-and-variance LayerNorm. Same slot as “pre-attn layernorm”: last axis only, per token.

```
H̃ = RMSNorm(H^ell)     # no mean subtract
  = γ ⊙ H^ell / sqrt( mean(H^ell²) + ε )
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell` | Residual stream into this layer | `(B, 241, 960)` | Previous block, or `P` if `ell=0` |
| This step | `RMSNorm` | Scale each token’s 960 channels by RMS; multiply learned `γ` | `(B, 241, 960)` | `γ` length 960. No β, no mean |
| Output | `H̃` | Normalized tokens for QKV | `(B, 241, 960)` | Same shape |

### 3.3 QKV projection

Two layouts show up. **Do not mix them:**

- **seq layout** `(B, S, heads, d_h)` after `view` — `S` is axis 1
- **head layout** `(B, heads, S, d_h)` after `transpose(1, 2)` — used for `Q K^T` and for the cache

`K` is **5-head** until an explicit GQA repeat. It is never 15-head at the same time as being 5-head.

```
Q  = reshape(H̃ W_Q)     # W_Q: (960, 15*64) → (B, 241, 15, 64)
K5 = reshape(H̃ W_K)     # W_K: (960,  5*64) → (B, 241,  5, 64)
V5 = reshape(H̃ W_V)     # W_V: (960,  5*64) → (B, 241,  5, 64)
Q  = RoPE(Q);  K5 = RoPE(K5)     # V5 not rotated; shapes unchanged
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `W_Q` | Linear `960 → 960`, view as heads | `Q`: `(B, 241, 15, 64)` | `15×64 = 960`. Seq layout |
| This step | `W_K` | Linear `960 → 320`, view as heads | `K5`: `(B, 241, 5, 64)` | `5×64 = 320`. **5 heads, not 15** |
| This step | `W_V` | Linear `960 → 320`, view as heads | `V5`: `(B, 241, 5, 64)` | Same as `K5` |
| This step | RoPE | Rotate `Q` and `K5` by position | same shapes | `V5` unchanged |
| Output | `Q, K5, V5` | Native projections, seq layout | `Q` 15-head; `K5,V5` 5-head | This is what 3.9 caches (`K5,V5` transposed) |

### 3.4 Scaled dot-product

First make `K` match the 15 Q heads, then move heads to axis 1 so the matmul is over `d_h`.

> **Documentation-only performance assumption:** this prefill description models the score MatMul in `bfloat16`, followed by an explicit cast of the score tensor to `float32` before masking and softmax. The installed code remains unchanged and currently upcasts Q and K before the MatMul. This assumption is recorded here for performance analysis; it is not a claim about the current runtime implementation.

```
K15 = repeat(K5, ×3)                 # (B, 241,  5, 64) → (B, 241, 15, 64)
Q̂   = Q.transpose(1, 2)              # (B, 15, 241, 64), bfloat16
K̂   = K15.transpose(1, 2)            # (B, 15, 241, 64), bfloat16
scores_bf16 = (Q̂ K̂^T) / sqrt(64)    # (B, 15, 241, 241), bfloat16
scores = scores_bf16.float()           # (B, 15, 241, 241), float32
```

`K̂^T` here means transpose of the last two axes: `(B, 15, 64, 241)`.

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `Q` | Queries from 3.3 | `(B, 241, 15, 64)` | Seq layout, 15 heads |
| Input | `K5` | Keys from 3.3 | `(B, 241, 5, 64)` | Seq layout, **5** heads |
| This step | `K15 = repeat(K5, 3)` | GQA: copy each KV head to 3 Q heads | `(B, 241, 15, 64)` | `15/5 = 3`. First time `K` has 15 heads |
| This step | `Q̂, K̂` | `transpose(1, 2)`: heads before sequence | both `(B, 15, 241, 64)` | Head layout. **Self**-attn: both seq axes are 241 |
| This step | `Q̂ K̂^T` | Dot over `d_h=64` in the documentation's assumed precision path | `(B, 15, 241, 241)`, `bfloat16` | One `S×S` score matrix per head |
| This step | `/ 8` | Scale by `sqrt(64)` | `(B, 15, 241, 241)`, `bfloat16` | Stops softmax from saturating |
| This step | cast | Convert scores before stable masking and softmax | `(B, 15, 241, 241)`, `float32` | Softmax input is FP32 |
| This step | mask | Image/lang may attend to each other; they **cannot** attend to state. Invalid pairs → large negative | same, `float32` | From `make_att_2d_masks` on the prefix |
| Output | `scores` | Scaled masked dots | `(B, 15, 241, 241)`, `float32` | Head layout, no `d_h` left |

### 3.5 Softmax

```
A_fp32 = softmax(scores, dim=key)     # over the last 241
A = A_fp32.to(bfloat16)                # before A·V
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `scores` | Scaled, cast, and masked dots | `(B, 15, 241, 241)`, `float32` | 3.4 |
| This step | `softmax` | Turn each query’s 241 FP32 scores into weights that sum to 1 | `(B, 15, 241, 241)`, `float32` | Softmax on the **key** axis only |
| Output | `A` | Attention weights cast for `A·V` | `(B, 15, 241, 241)`, `bfloat16` | `A[b,h,i,j]` = how much token `i` reads token `j` |

### 3.6 Attend to V (`dot`) + output projection

`V` follows the same 5 → 15 → head-layout path as `K`, then the weights mix values:

```
V15    = repeat(V5, ×3)              # (B, 241, 5, 64) → (B, 241, 15, 64)
V̂     = V15.transpose(1, 2)         # (B, 15, 241, 64)
O_heads = A V̂                       # (B, 15, 241, 64)
O      = O_heads.transpose(1, 2).reshape(B, 241, 960)
attn   = O W_O                       # W_O: (960, 960)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `V5` | Values from 3.3 | `(B, 241, 5, 64)` | Seq layout, 5 heads — same as `K5` |
| This step | `V̂` | GQA ×3 then `transpose(1, 2)` | `(B, 15, 241, 64)` | Same head layout as `A`’s value axis |
| This step | `A V̂` | Weighted sum of value vectors | `(B, 15, 241, 64)` | Mixes the 241 tokens **inside** each head |
| This step | concat | `transpose` + reshape 15 heads | `(B, 241, 960)` | `15×64 = 960` |
| This step | `W_O` | `o_proj`: linear back to residual width | `(B, 241, 960)` | Mix heads together |
| Output | `attn` | Attention output of this layer | `(B, 241, 960)` | Same as residual stream |

### 3.7 Residual + post RMSNorm

```
H'  = H^ell + attn
H'' = RMSNorm(H')
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | residual | Add skip from **before** 3.2 | `(B, 241, 960)` | Pre-LN: residual is the un-normalized `H^ell` |
| This step | `RMSNorm` | Same as 3.2, new `γ` (`post_attention_layernorm`) | `(B, 241, 960)` | Prepares tokens for the MLP |
| Output | `H''` | Normalized post-attn stream | `(B, 241, 960)` | — |

### 3.8 MLP + residual

SwiGLU (Llama MLP), not a single linear:

```
MLP(x) = W_down( SiLU(x W_gate) ⊙ (x W_up) )
H^{ell+1} = H' + MLP(H'')
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `W_gate`, `W_up` | Linear `960 → 2560` each | `(B, 241, 2560)` | `intermediate_size = 2560` |
| This step | `SiLU ⊙` | SwiGLU gate | `(B, 241, 2560)` | Elementwise |
| This step | `W_down` | Linear `2560 → 960` | `(B, 241, 960)` | Back to hidden size |
| This step | residual | Add skip from `H'` (after attn residual, before this RMSNorm) | `(B, 241, 960)` | `H^{ell+1}` feeds the next `ell` |
| Output | `H^{ell+1}` | Residual stream for layer `ell+1` | `(B, 241, 960)` | Sequence length still 241 |

### 3.9 Write cache

Cache the **5-head** tensors from 3.3 (post-RoPE), **not** the GQA-repeated `K15`. Transpose into head layout because `DynamicCache` stores `(B, heads, S, d_h)`:

```
cache[ell].K = K5.transpose(1, 2)     # (B, 241, 5, 64) → (B, 5, 241, 64)
cache[ell].V = V5.transpose(1, 2)     # (B, 241, 5, 64) → (B, 5, 241, 64)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `K5, V5` | Native KV from 3.3 | `(B, 241, 5, 64)` | Seq layout, 5 heads |
| This step | `transpose(1, 2)` | Heads before sequence | `(B, 5, 241, 64)` | Cache / head layout |
| Output | `cache[ell].K, .V` | Prefix KV for this layer | `(B, 5, 241, 64)` each | Still 5 GQA heads. Repeat-to-15 happens again when 4 attends |
| Why cache | — | 4 will **not** re-run QKV on `P` | 16 layers | Prefill once; denoise 10 Euler steps |

After `ell = 15`, a final RMSNorm is applied to `H^{16}`. That hidden is **not** what the action expert consumes; the expert in 4 reads the **cache**.

## 4. Decode / denoise (out: action chunk)

**Meaning.** Start from Gaussian noise and **Euler-integrate** the learned flow for a **fixed** `num_steps=10`. This is **not** next-token generation and **not** an iterative residual threshold. Each Euler step rebuilds 2 `U_t`, runs the **action expert** (16 layers), and steps `x` toward the action. Even layers temporarily append suffix K/V to the prefix cache and attend over prefix + suffix; odd layers **cross-attend** to the fixed prefix cache only. **Out:** after crop and unnormalize, a robot command `(B, 6)` (via a 50-step queue).

Expert sizes (thinner than the VLM):

| Symbol | Value | Role |
|---|---|---|
| `S_a` | 50 | action chunk / expert sequence |
| `S_p` | 241 | prefix length in the cache |
| `d_e` | 720 | expert hidden (`0.75 × 960`) |
| `n_h` | 15 | query heads (same count as VLM) |
| `n_kv` | 5 | KV heads (GQA) |
| `d_h` | 64 | `head_dim` |
| `d_ff^e` | 2048 | expert MLP width |

`Q` is still 15×64 = **960** channels even though the residual is 720: `q_proj` is `720 → 960`.

### 4.1 Input set

What this stage reads. `P` itself is **not** an input; the expert reads the **cache** that 3 already computed.

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `cache[ell].K, .V` | Prefix keys/values from 3 | 16 × `(B, 5, 241, 64)` | One pair per VLM layer; 5 GQA heads |
| Input | prefix pad mask | Which of the 241 prefix slots are real | `(B, 241)` | Builds the cross-attn mask |
| Input | `x` at start | Noisy action chunk | `(B, 50, 32)` | Drawn `N(0, I)`. 50 horizon × 32 padded DoF |
| Input | `t` | Flow time, one float per batch | `(B,)` | Set by the Euler index, see 4.2 |
| Built each step | `U_t` | Expert token sequence (2) | `(B, 50, 720)` | From `x` and `t`; **not** cached across Euler steps |

VLM text weights are **idle** here (`inputs_embeds = [None, U_t]`). Only the expert residual stream moves.

### 4.2 Control flow (Euler × layers) and stop

Two nested loops. The **outer** loop is flow time. The **inner** loop is the 16 expert blocks.

```
x  ~ N(0, I)                            # (B, 50, 32)
dt = -1 / 10
for k = 0 .. 9:                         # outer: Euler  (STOP: k hits 10)
    t  = 1 + k * dt                     # 1.0, 0.9, …, 0.1
    U  = embed_suffix(x, t)             # 2 → (B, 50, 720)
    H  = U
    for ell = 0 .. 15:                  # inner: expert layers
        if ell % 2 == 0:
            H = SelfAttnBlock_ell(H, cache[ell]) # Q: 50; KV: 241 prefix + 50 suffix
        else:
            H = CrossAttnBlock_ell(H, cache[ell])  # 50 attend to 241
    cache.crop(241)                     # drop any suffix KV self-attn appended
    v  = H W_out                        # (B, 50, 32)
    x  = x + dt * v                     # step toward t = 0
# STOP. No ||v|| < ε test. After 10 updates, x is the t ≈ 0 chunk.
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Outer | `k` | Euler index | `0 … 9` | `num_steps = 10` **exactly** |
| Outer | `t` | Flow time at this evaluation | `(B,)` | `t = 1 + k·dt`. Last eval is `t=0.1`, then one `dt` lands at `t=0` |
| Stop | — | `for` ends when `k==10` | — | Fixed-step integrator, not a residual/confidence stop |
| Inner | `ell` | Expert block index | `0 … 15` | Same 16 as VLM. Even = append suffix KV and attend over prefix + suffix; odd = prefix-only cross-attn |
| During even layer | `cache.update` | Append this Euler step’s 50 suffix K/V entries | cache seq `241 → 291` at that layer | Temporary; prefix K/V are not overwritten |
| After inner | `cache.crop(241)` | Delete the temporary 50 suffix entries | cache seq `291 → 241` | Next Euler step starts from the original prefix-only cache |

### 4.3 Rebuild suffix `U_t`

`embed_suffix(x, t)` runs once at the beginning of **every Euler iteration**. With `M=10`, this section therefore runs 10 times per action-chunk inference.

The learned weights are reused, but the activations are rebuilt because both inputs change:

```text
Euler k     flow time t_k     action tensor supplied to embed_suffix
0           1.0               x^(0) = initial Gaussian noise
1           0.9               x^(1) = x^(0) + dt·v^(0)
2           0.8               x^(2) = x^(1) + dt·v^(1)
...
9           0.1               x^(9) = x^(8) + dt·v^(8)
```

By contrast, the observation-dependent prefix `P` and its 16-layer KV cache remain fixed throughout all 10 iterations.

#### 4.3.1 Inputs at Euler step `k`

```text
x^(k): (B, 50, 32), float32
t_k:   (B,),         float32
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x^(k)` | Current noisy/partially denoised action chunk | `(B, 50, 32)` | 50 future action slots; each action is padded to `max_action_dim=32` |
| Input | `t_k` | Current flow time | `(B,)` | One noise-level value per batch item |
| Change from previous `k` | `x^(k), t_k` | Euler changes both the action estimate and flow time | same shapes | Therefore the previous `U_t` cannot be reused |
| Fixed across `k` | learned weights | `W_a`, `W_in^time`, `W_out^time` | parameter tensors | The model parameters do not change during inference |
| Fixed across `k` | prefix cache | 16 K/V pairs over 241 positions | each K/V `(B,5,241,64)` | Cameras, instruction, and state were prefetched once |

#### 4.3.2 Project every action slot to expert width

The same action projection is independently applied at each of the 50 horizon positions:

```text
a^(k) = W_a x^(k) + b_a

W_a: (720, 32)
x^(k): (B, 50, 32)
b_a: (720,)

action_in_proj, 32 → 720
a^(k): (B, 50, 720), float32
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x^(k)` | 32 padded action values at every horizon position | `(B,50,32)` | Input action width |
| This step | `W_a` | Learned affine projection `32→720` | applied to every `(b,n)` independently | Matches expert hidden width |
| Output | `a^(k)` | Action-content embedding | `(B,50,720)` | Sequence length remains 50 |

This Linear does **not** mix information between action positions. Position `n=0` and position `n=49` use the same weights but different 32-value input vectors.

#### 4.3.3 Recompute the flow-time embedding

The implementation constructs 360 logarithmically spaced periods on every call:

```text
fraction[j] = j / 359                              for j = 0,...,359
period[j]   = 0.004 × (4 / 0.004)^fraction[j]
            = 0.004 × 1000^fraction[j]
omega[j]    = 2π / period[j]
```

The endpoints are:

```text
period[0]   = 0.004     # highest frequency
period[359] = 4.0       # lowest frequency
```

For every batch item:

```text
phase[b,j] = t_k[b] × omega[j]                     # (B,360)

tau[b,:] = concat(
    sin(phase[b,0:360]),
    cos(phase[b,0:360])
)                                                    # (B,720)
```

On CPU, period/phase construction and `sin/cos` are initially evaluated in `float64`; the resulting 720-dimensional vector is then cast to the action-embedding datatype:

```text
tau: (B,720), float64 → float32
```

The same flow time applies to all 50 action slots, so `tau` is broadcast without creating 50 different time values:

```text
tau_B = tau[:,None,:].expand(B,50,720)              # (B,50,720), float32
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `fraction` | 360 values uniformly spaced from 0 to 1 | `(360,)` | Half of expert width: `720/2=360` |
| This step | `period`, `omega` | Convert fractions to 360 time scales/frequencies | each `(360,)` | Covers fast and slow variation in flow time |
| This step | `phase` | Outer multiply of `t_k` with all frequencies | `(B,360)` | One phase per batch item and frequency |
| This step | `sin`, `cos` | Encode every phase in two complementary channels | two `(B,360)` tensors | Preserves periodic phase information |
| This step | concat | Put all sine channels beside all cosine channels | `(B,720)` | `360+360=720` |
| This step | cast | Match action/fusion MLP datatype | `(B,720)`, `float32` | Fusion MLP weights are FP32 |
| This step | broadcast | Attach the same time vector to all horizon slots | `(B,50,720)` | One common Euler time for the complete chunk |

Although `tau_B[b,n,:]` is identical for all `n`, `a^(k)[b,n,:]` is normally different at every action position.

#### 4.3.4 Concatenate and fuse action/time features

Action and time are concatenated on the **feature axis**, not the sequence axis:

```text
z^(k) = concat(a^(k), tau_B, dim=-1)

a^(k): (B,50,720)
tau_B: (B,50,720)
z^(k): (B,50,1440)
```

The two-layer fusion MLP is:

```
g^(k) = W_in^time z^(k) + b_in              # 1440 → 720
h^(k) = SiLU(g^(k))                          # shape unchanged
U_t    = W_out^time h^(k) + b_out            # 720 → 720
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `z^(k)` | Action and flow-time features placed side by side | `(B,50,1440)` | `720 action + 720 time` |
| This step | `W_in^time` | Learned Linear `1440→720` | `(B,50,720)` | Every output feature can mix all action and time features |
| This step | `SiLU` | Elementwise nonlinear gate `g·sigmoid(g)` | `(B,50,720)` | Shape does not change |
| This step | `W_out^time` | Learned Linear `720→720` | `(B,50,720)` | Produces expert-width tokens |
| Output | `U_t` | New suffix/expert input for Euler step `k` | `(B,50,720)`, `float32` | One token per future action position |


This explains the word **fusion**: every resulting 720-dimensional token combines its slot-specific action content with the common flow-time context. The fusion Linears still act independently at each of the 50 sequence positions; interaction between positions happens later in the expert attention blocks.

The complete one-line form is:

```text
U_t =
W_out^time(
    SiLU(
        W_in^time(
            concat(
                W_a x^(k),
                broadcast(PE(t_k))
            )
        )
    )
)
```

`W_out^time` here is the second **time-fusion MLP** Linear. It is not the later `action_out_proj`, which maps final expert tokens from `720→32` to produce Euler velocity.

#### 4.3.5 Suffix masks produced with `U_t`

`embed_suffix` also creates two length-50 masks:

```text
suffix_pad_mask: (B,50), bool, all True
suffix_att_group:(B,50), float32, all ones
```

| Mask | Meaning | How it is used later |
|---|---|---|
| `suffix_pad_mask` | All 50 action-token positions are valid | Combined with prefix validity in 4.2/4.4 |
| `suffix_att_group` | Every action position starts a new causal group | `make_att_2d_masks` converts it into a lower-triangular `50×50` suffix mask |

Consequently, suffix position `n` can read suffix positions `0...n`, but not future suffix positions `n+1...49`. All suffix queries may also read valid prefix keys.

#### 4.3.6 What changes and what remains fixed

| Item | Rebuilt every Euler step? | Reason |
|---|---|---|
| `x^(k)` | Yes | Updated by the previous velocity |
| `t_k` | Yes | Decreases by `0.1` |
| `a^(k)` | Yes | Depends on current `x^(k)` |
| `PE(t_k)`, `tau_B` | Yes | Depends on current flow time |
| `U_t` and suffix masks | Yes | Inputs to the current expert evaluation |
| Prefix `P` | No | Observation is unchanged during this chunk inference |
| Prefix K/V cache | No | Prefilled once before the Euler loop |
| Model weights | No | Shared across all Euler steps |

After this rebuild, `H^0 = U_t` enters the 16-layer expert loop in 4.4.

### 4.4 Expert layer loop

```
H^0 = U_t
for ell = 0 .. 15:
    H^{ell+1} = Block_ell(H^ell)
H_out = RMSNorm(H^{16})
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^0` | Suffix tokens | `(B, 50, 720)` | 4.3 |
| Even `ell` | cached self-attn path | Q and new suffix K/V come from `H^ell`; cached prefix K/V are prepended | Q seq 50, KV seq 291 | Recipe in 4.5 |
| Odd `ell` | cross-attn | Q from `H^ell`; K, V from `cache[ell]` then expert `W_K, W_V` | Q seq 50, KV seq 241 | Recipe in 4.6 |
| Output | `H_out` | Final expert tokens | `(B, 50, 720)` | Then 4.8 maps to velocity |

Each `Block_ell` is still Pre-LN: RMSNorm → attn → residual → RMSNorm → MLP → residual. The difference is that even layers use `[prefix KV; current suffix KV]`, whereas odd layers read transformed prefix KV only.

### 4.5 Even `ell`: temporarily append suffix KV

`ell ∈ {0, 2, …, 14}`. Queries come from the 50 suffix tokens. Their newly projected K/V are appended to that layer’s 241 cached prefix entries, producing 291 K/V positions for this attention call.

**Projection**

```
H̃  = RMSNorm(H^ell)                         # (B, 50, 720)
Q   = reshape(H̃ W_Q)                        # W_Q: (720, 15*64) → (B, 50, 15, 64)
K5^s = reshape(H̃ W_K)                       # W_K: (720, 5*64) → (B, 50, 5, 64)
V5^s = reshape(H̃ W_V)                       # same as K5^s
Q    = RoPE(Q);  K5^s = RoPE(K5^s)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell` | Expert residual | `(B, 50, 720)` | 50 action slots |
| This step | RMSNorm | Last-axis RMS, per token | `(B, 50, 720)` | Same slot as 3.2 |
| This step | `W_Q` | Linear `720 → 960` | `Q`: `(B, 50, 15, 64)` | `15×64=960` even though hidden is 720 |
| This step | `W_K, W_V` | Linear `720 → 320` | `K5^s,V5^s`: `(B, 50, 5, 64)` | Current suffix K/V; 5 GQA heads |
| This step | cache append | Append `K5^s,V5^s` after the 241 prefix positions | `(B, 291, 5, 64)` | `291 = 241 prefix + 50 suffix`; prefix entries remain unchanged |

**Scaled dot, softmax, `A V`, `o_proj`**

```
# DynamicCache stores head-first tensors:
K5^s_h = K5^s.transpose(1, 2)                 # (B, 5, 50, 64)
V5^s_h = V5^s.transpose(1, 2)                 # (B, 5, 50, 64)

# Mutating append: 241 prefix positions + 50 suffix positions = 291
K5^all_h, V5^all_h = cache.update(
    K5^s_h, V5^s_h, layer=ell
)                                               # each (B, 5, 291, 64)

# Return to seq layout, then expand 5 KV heads to 15 query heads:
K5^all = K5^all_h.transpose(1, 2)              # (B, 291, 5, 64)
V5^all = V5^all_h.transpose(1, 2)              # (B, 291, 5, 64)
K15 = repeat(K5^all, ×3)                       # (B, 291, 15, 64)
V15 = repeat(V5^all, ×3)                       # (B, 291, 15, 64)

# Calculate the scores
Q̂ = Q.transpose(1, 2)                          # (B, 15, 50, 64)
K̂ = K15.transpose(1, 2)                        # (B, 15, 291, 64)
V̂ = V15.transpose(1, 2)                        # (B, 15, 291, 64)
scores = (Q̂ K̂^T) / 8                        # (B, 15, 50, 291)
A      = softmax(scores, dim=key)

# Calculate the atten
O_heads = A V̂                               # (B, 15, 50, 64)
O = O_heads.transpose(1, 2)                  # (B, 50, 15, 64)
O = O.reshape(B, 50, 15*64)                  # (B, 50, 960): concat 15 heads
attn = O W_O                                 # (B, 50, 720), W_O: (960, 720)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| This step | `cache.update` | Append current suffix K/V to the 241 cached prefix entries | `K5^all_h,V5^all_h`: `(B, 5, 291, 64)` | `291 = 241 + 50`; this mutates the layer cache temporarily |
| This step | GQA + transpose | Repeat the combined 5-head KV to 15 heads | `K̂,V̂`: `(B, 15, 291, 64)` | Prefix and suffix are both available as keys/values |
| This step | `Q̂ K̂^T / 8` | Score 50 suffix queries against 291 positions | `(B, 15, 50, 291)` | 241 prefix + 50 suffix keys |
| This step | mask + softmax | Prefix keys are visible; suffix keys use the suffix causal mask | `(B, 15, 50, 291)` | A suffix position cannot read a later suffix position |
| This step | `A V̂` | Weighted value sum, independently for each head | `(B, 15, 50, 64)` | 15 separate 64-D head outputs |
| This step | transpose + reshape | Put heads beside one another on the feature axis | `(B, 50, 960)` | Concatenate `15 × 64 = 960`; no sequence tokens are concatenated |
| This step | `W_O` | Mix the concatenated heads and project to expert width | `(B, 50, 720)` | `960 → 720` |

After attention, continue with the shared residual/RMSNorm/MLP block in 4.7.

This append is temporary. After all 16 layers finish for the current Euler step, `cache.crop(241)` removes the 50 suffix entries from every even layer. Therefore the next Euler step starts with exactly the same prefix cache produced by 3.

### 4.6 Odd `ell`: cross-attn to prefix cache

`ell ∈ {1, 3, …, 15}`. Queries are the 50 action tokens. Keys/values are the **241 prefix** tokens already in the cache (not recomputed). Expert `W_K, W_V` here are `320 → 320`: they map VLM KV width into expert KV width.

**Projection (Q from expert, K/V from cache)**

```
H̃   = RMSNorm(H^ell)                         # (B, 50, 720)
Q    = reshape(H̃ W_Q)                        # (B, 50, 15, 64)   W_Q: 720 → 960
Q    = RoPE(Q)

K5^p = cache[ell].K.transpose(1, 2)          # (B, 5, 241, 64) → (B, 241, 5, 64)
V5^p = cache[ell].V.transpose(1, 2)
K5   = reshape( flatten(K5^p) W_K )          # flatten 5*64=320; W_K: 320 → 320
V5   = reshape( flatten(V5^p) W_V )          # → (B, 241, 5, 64)
```

`K5, V5` are **not** RoPE’d again (cache already has post-RoPE VLM KV). No Q from the prefix.

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell` | Expert residual | `(B, 50, 720)` | Action tokens only |
| Input | `cache[ell].K, .V` | Prefix KV | `(B, 5, 241, 64)` | Head layout from 3.9 |
| This step | `W_Q` | Expert queries | `Q`: `(B, 50, 15, 64)` | Seq **50** |
| This step | `W_K, W_V` | Linear `320 → 320` on flattened prefix KV | `K5,V5`: `(B, 241, 5, 64)` | Seq **241**. This is **cross**-attn, not 50×50 |
| Output | `Q` vs `K5` | Different sequence lengths | 50 vs 241 | Scores will be rectangular |

**Scaled dot, softmax, `A V`, `o_proj`**

```
K15 = repeat(K5, ×3)                         # (B, 241, 15, 64)
Q̂   = Q.transpose(1, 2)                      # (B, 15,  50, 64)
K̂   = K15.transpose(1, 2)                    # (B, 15, 241, 64)
V̂   = same path for V5
scores = (Q̂ K̂^T) / 8                        # (B, 15, 50, 241)
A      = softmax(scores, dim=key)            # over 241
O_heads = A V̂                               # (B, 15, 50, 64)
O = O_heads.transpose(1, 2)                  # (B, 50, 15, 64)
O = O.reshape(B, 50, 15*64)                  # (B, 50, 960): concat 15 heads
attn = O W_O                                 # (B, 50, 720), W_O: (960, 720)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `Q` | Expert queries | `(B, 50, 15, 64)` | Seq layout |
| Input | `K5` | Prefix keys after expert `W_K` | `(B, 241, 5, 64)` | 5 heads until GQA |
| This step | `K15`, `K̂` | Repeat ×3, then head layout | `K̂`: `(B, 15, 241, 64)` | Matches 15 Q heads; **241** keys |
| This step | `Q̂ K̂^T / 8` | Each of 50 queries scores all 241 prefix keys | `(B, 15, 50, 241)` | Rectangular. **Not** `(15, 50, 50)` and **not** `(15, 241, 241)` |
| This step | softmax | Weights over the **241** key axis | `(B, 15, 50, 241)` | Each action token’s weights sum to 1 over the prefix |
| This step | `A V̂` | Mix prefix values | `(B, 15, 50, 64)` | Output seq stays 50 |
| This step | transpose + reshape | Concatenate the 15 head outputs per action token | `(B, 50, 960)` | `15 × 64 = 960` |
| This step | `W_O` | Mix heads and project `960 → 720` | `(B, 50, 720)` | Back to expert residual |

Mask: suffix queries may read the whole prefix (pad-masked). Image/lang still do not attend to actions — the VLM is not running.

After attention, continue with the shared residual/RMSNorm/MLP block in 4.7.

### 4.7 Residual + RMSNorm + MLP

This block runs after the attention computation in **every** expert layer, whether 4.5 (even, cached self-attention path) or 4.6 (odd, cross-attention path). It preserves shape `(B, 50, 720)`.

#### 4.7.1 First residual: add the attention result

```
R = H^ell + attn
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H^ell` | Expert residual stream before the layer’s input RMSNorm | `(B, 50, 720)` | Skip path preserves the original layer input |
| Input | `attn` | Output of 4.5 or 4.6 after `W_O` | `(B, 50, 720)` | Same width as the residual stream |
| This step | `+` | Elementwise residual addition | `(B, 50, 720)` | Every token and feature is added independently |
| Output | `R` | Post-attention residual stream | `(B, 50, 720)` | Becomes both the MLP input path and the second skip path |

The residual path lets a layer retain `H^ell` even if the learned attention contribution is small. It also gives gradients a direct path through the stack.

#### 4.7.2 Post-attention RMSNorm

RMSNorm is applied independently to the 720 features of each of the `B × 50` action tokens:

```
N = RMSNorm(R)
  = gamma ⊙ R / sqrt(mean(R^2, dim=-1) + epsilon)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `R` | Post-attention residual | `(B, 50, 720)` | Output of 4.7.1 |
| This step | `mean(R², dim=-1)` | Compute one mean square per token | `(B, 50, 1)` | Normalize over hidden features only, not batch or sequence |
| This step | divide by RMS | Rescale each token | `(B, 50, 720)` | RMSNorm does not subtract a mean |
| This step | `gamma` | Learned feature-wise scale | `(720,)`, broadcast to `(B, 50, 720)` | One learned scale per expert hidden feature; no learned bias |
| Output | `N` | Normalized MLP input | `(B, 50, 720)` | Shape unchanged |

#### 4.7.3 SwiGLU MLP expansion and gate

The MLP has two separate `720 → 2048` projections. One produces a nonlinear gate; the other produces values:

```
G = N W_gate                         # (B, 50, 2048)
U = N W_up                           # (B, 50, 2048)
Z = SiLU(G) ⊙ U                     # (B, 50, 2048)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `N` | Normalized expert tokens | `(B, 50, 720)` | Output of 4.7.2 |
| This step | `W_gate` | Linear `720 → 2048` | `G`: `(B, 50, 2048)` | Expert `intermediate_size = 2048` |
| This step | `W_up` | Independent linear `720 → 2048` | `U`: `(B, 50, 2048)` | Supplies the values controlled by the gate |
| This step | `SiLU(G)` | Smooth nonlinear gate | `(B, 50, 2048)` | Applied elementwise |
| This step | `⊙ U` | Gate the up-projection elementwise | `Z`: `(B, 50, 2048)` | SwiGLU-style multiplicative interaction |

`W_gate` and `W_up` do not share weights. The MLP operates independently at each of the 50 sequence positions; attention is the operation that mixes positions.

#### 4.7.4 Down projection and second residual

```
M = Z W_down                         # (B, 50, 720)
H^{ell+1} = R + M                    # (B, 50, 720)
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `Z` | Gated intermediate features | `(B, 50, 2048)` | Output of 4.7.3 |
| This step | `W_down` | Linear `2048 → 720` | `M`: `(B, 50, 720)` | Return to expert residual width |
| This step | second residual | Add the pre-MLP stream `R` | `(B, 50, 720)` | Skip around RMSNorm + MLP |
| Output | `H^{ell+1}` | Input residual stream for the next expert layer | `(B, 50, 720)` | Sequence length and hidden width remain fixed |

Complete shared post-attention block:

```
R         = H^ell + attn
N         = RMSNorm(R)
Z         = SiLU(N W_gate) ⊙ (N W_up)
M         = Z W_down
H^{ell+1} = R + M
```

After `ell = 15`, the loop applies one additional final expert RMSNorm before the velocity head.

### 4.8 Velocity head `v_t`

After `ell=15` and a final expert RMSNorm:

```
v_t = H_out W_out          # action_out_proj: 720 → 32
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `H_out` | Expert tokens at this `t` | `(B, 50, 720)` | 4.4 |
| This step | `W_out` | Linear `720 → 32` | `(B, 50, 32)` | Same layout as `x`: 50 horizon × 32 padded DoF |
| Output | `v_t` | Predicted flow velocity | `(B, 50, 32)` | Trained to match `noise - action`. **Not** an action yet |

### 4.9 Euler update

```
x ← x + dt * v_t          # dt = -0.1
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x` | Chunk at current `t` | `(B, 50, 32)` | Previous Euler state |
| Input | `v_t` | Velocity from 4.8 | `(B, 50, 32)` | Same shape, so the add is elementwise |
| This step | `dt=-1/10` | One tenth of the path noise→action, **backward** in `t` | scalar | `t` decreases |
| Output | `x` | Chunk at `t+dt` | `(B, 50, 32)` | After `k=9`, this is the `t≈0` sample |

Then go back to 4.2 until `k=10`.

### 4.10 Post-process → robot command

The network’s last tensor is still padded and normalized. LeRobot then crops, queues, and unnormalizes.

```
a_chunk = x[:, :, :6]                 # drop 26 pad zeros
queue.extend(a_chunk.transpose(0,1))  # 50 steps, each (B, 6)
a_now   = queue.popleft()             # (B, 6)
a_cmd   = unnormalize(a_now)          # mean/std from dataset stats
# send a_cmd to motors: shoulder_pan, shoulder_lift, elbow_flex,
#                       wrist_flex, wrist_roll, gripper
```

| Step | Symbol | What it does | Output shape | Why that size |
|---|---|---|---|---|
| Input | `x` at `t≈0` | Finished Euler chunk | `(B, 50, 32)` | 4.9 after 10 steps |
| This step | crop | Keep real action DoF | `(B, 50, 6)` | `action_feature.shape[0] = 6` |
| This step | queue | Store the horizon; `select_action` pops **one** step per control tick | 50 × `(B, 6)` | `n_action_steps = 50` (= `chunk_size` here) |
| This step | `popleft` | Command for **this** robot tick | `(B, 6)` | Not the whole chunk |
| This step | unnormalize | Invert training mean/std; move to CPU | `(B, 6)` | Postprocessor. Values are now joint/gripper units |
| Output | `a_cmd` | What the robot executes | `(B, 6)` | 5 arm joints + 1 gripper |

When the queue is empty (50 ticks later), the loop returns to 1 and 3 with a **new** camera/state observation. 2 `U_t` is only rebuilt inside Euler, not between robot ticks while the queue still has actions.

The Hub snippet `policy.select_action(raw_frame)` is incomplete: the policy expects the **preprocessed** batch (`observation.language.tokens` must exist).

---

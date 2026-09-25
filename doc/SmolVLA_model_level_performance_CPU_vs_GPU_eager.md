# SmolVLA model-level performance: CPU vs GPU eager vs GPU compile

**Scope.** Same checkpoint [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), same fixed observation + `SMOKE_SEED=0` noise, batch **B=1**. Compare three backends:

| Backend | Device | Path in this repo | Primary timing source |
|---|---|---|---|
| **CPU** | host CPU | [`doc/cpu/`](cpu/) | `smoke_test_report.json` stage profile |
| **GPU eager** | RTX 5060 Ti (SM 12.0) | [`doc/gpu/eager_mode/`](gpu/eager_mode/) | [`smolVLA_profiling.md`](gpu/eager_mode/smolVLA_profiling.md) (5 warm runs, CUDA-synchronized) |
| **GPU compile** | same GPU | [`doc/gpu/compile_mode/`](gpu/compile_mode/) | [`smolVLA_kerne_list_gpu_backend.md`](gpu/compile_mode/smolVLA_kerne_list_gpu_backend.md) (Nsight steady chunk #4; `torch.compile` **`reduce-overhead`**) |

Secondary GPU views: eager CUPTI chunk #3 in [`eager kerne_list`](gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md) §1; compile correctness / smoke in [`smolVLA_compile_mode_report.md`](gpu/compile_mode/smolVLA_compile_mode_report.md).

---

## 1. Metrics (model-level)

SmolVLA is a **flow-matching VLA**, not an autoregressive LLM. Classic “decode tokens/s” does not apply directly. Metrics used here:

| Metric | Definition | Why |
|---|---|---|
| **End-to-end latency** $T_{\mathrm{e2e}}$ | Wall time for one full **chunk fill**: Stages 0→2→(1+3)×$M{=}10$→4 (CUDA-synced on GPU eager; Nsight CUPTI wall-span on GPU compile) | Time until the 50-step action queue is ready |
| **Chunk throughput** | $1 / T_{\mathrm{e2e}}$ (chunks/s) | How often the policy can refill actions |
| **Action throughput** | $N / T_{\mathrm{e2e}}$ with $N{=}50$ (actions/s) | Upper bound if the queue is drained one step per control tick with **no** extra model calls |
| **Prefix token throughput** | $S_p / (T_0 + T_2)$, $S_p{=}241$ (tok/s) | Tokens through ViT+connector+lang/state + VLM prefill (Stages 0+2) |
| **Suffix token-step throughput** | $(N \cdot M) / T_{1+3}$, $N{=}50$, $M{=}10$ (tok-step/s) | Action-expert tokens × Euler steps (Stages 1+3) |

Shapes: prefix $S_p{=}241$ (192 visual + 48 language + 1 state); suffix chunk $N{=}50$; Euler steps $M{=}10$; action dim used = 6.

**Compile note.** Per-stage $T_0$, $T_2$, $T_{1+3}$ are **not** available through the compiled `sample_actions` wrapper (manual Stage 0–4 profiler bypasses Inductor). For compile, report $T_{\mathrm{e2e}}$ and chunk/action throughput only; prefix/suffix tok/s stay blank until a compiled stage profile exists.

---

## 2. End-to-end latency

Warm inference (excludes model load, GPU first-run warmup, and compile cold start).

| | **CPU** | **GPU eager** | **GPU compile** | Speedup vs CPU | vs eager |
|---|---:|---:|---:|---:|---:|
| **$T_{\mathrm{e2e}}$ (wall)** | **20.129 s** | **0.0913 s** (**91.3 ms**) | **0.0480 s** (**48.0 ms**) | **≈419×** | **≈1.90×** |
| Stages 0–4 sum | 20.128 s | 0.0903 s | — † | — | — |
| Stage 0 prefix embed | 15.009 s | 23.9 ms | — † | — | — |
| Stage 2 VLM prefill | 1.933 s | 7.18 ms | — † | — | — |
| Stage 1+3 (suffix + expert ×10) | 3.187 s | 59.1 ms | — † | — | — |
| Stage 4 crop/queue | ~0.1 ms | ~0.05 ms | — † | — | — |

† Compile path merges Stages 0–3 into Inductor / CUDA-graph launches; use Nsight landmarks in compile kerne_list §4 instead of per-stage host timers.

**Compile $T_{\mathrm{e2e}}$ source:** Nsight CUPTI wall-span of steady `select_action` chunk **#4** (**48.03 ms**; busy Σ **43.78 ms**; **4101** launches). Host warm `select_action` in that session ≈ **47–50 ms**. Smoke harness second call was **77 ms** (post–cold-compile residual; prefer the Nsight steady number for model-level latency).

**Cold compile (once):** first `select_action` ≈ **98 s** (Inductor + CUDA-graph capture) — excluded from the table above.

**Smoke `select_action` (includes load/overhead, 2 runs):** CPU **42.87 s** total vs GPU eager **0.48 s** total — same order of gap; prefer the warm $T_{\mathrm{e2e}}$ row for model latency.

### Stage share (latency mix shifts on GPU)

| Stage | CPU share | GPU eager share | GPU compile |
|---|---:|---:|---|
| 0 prefix | **74.6%** | 26.4% | fused into Inductor graphs (no host stage split) |
| 2 prefill | 9.6% | 7.9% | same |
| 3 expert denoise | 15.8% | **63.5%** | busy still dominated by GEMM + FMHA (compile kerne_list §2) |
| 1 suffix + 4 | ~0.1% | ~2.2% | preprocess upsample still outside heavy fusion |

On CPU, **vision/prefix** dominates. On GPU eager, **expert Euler loop** dominates. On GPU compile, launch overhead collapses (gap **8.8%** vs eager ~**55%**); remaining busy is ~**61%** CUTLASS BF16 GEMM + ~**16%** FMHA.

---

## 3. Throughput

| Metric | **CPU** | **GPU eager** | **GPU compile** | Speedup vs CPU | vs eager |
|---|---:|---:|---:|---:|---:|
| Chunk throughput | **0.050** /s | **10.95** /s | **20.8** /s | ≈419× | ≈1.90× |
| Action throughput (queue drain) | **2.48** actions/s | **547** actions/s | **1042** actions/s | ≈419× | ≈1.90× |
| Prefix token throughput (Stages 0+2) | **14.2** tok/s | **7 760** tok/s | — † | — | — |
| Suffix token-step throughput (Stages 1+3) | **157** tok-step/s | **8 450** tok-step/s | — † | — | — |

† Needs compiled stage timers; see §1.

**Real-time control note.** At 30 Hz control with $N{=}50$, one chunk lasts $50/30 \approx 1.67$ s of robot time. CPU $T_{\mathrm{e2e}} \approx 20$ s cannot keep up. GPU eager (~91 ms) and GPU compile (~48 ms) both leave ample margin for synchronous refill (budget $N / f_{\mathrm{control}}$).

---

## 4. Same GPU path — host wall vs Nsight

Eager and compile measure different clocks; Nsight rows are CUPTI first→last kernel wall-span of a steady chunk.

| View | Latency / chunk | Chunk throughput | Notes |
|---|---:|---:|---|
| **Eager** PyTorch CUDA-sync $T_{\mathrm{e2e}}$ | **91.3 ms** | **10.95** /s | Host-visible stage barriers; best eager model-level latency |
| **Eager** Nsight CUPTI wall (chunk #3) | **128.7 ms** | **7.77** /s | First upsample → last kernel; includes capture/stream idle |
| **Eager** Nsight busy Σ | **57.4 ms** | **17.4** /s *ceiling* | No inter-launch gaps |
| **Compile** Nsight CUPTI wall (chunk #4) | **48.0 ms** | **20.8** /s | Steady `reduce-overhead` + CUDA graphs (`--cuda-graph-trace=node`) |
| **Compile** Nsight busy Σ | **43.8 ms** | **22.8** /s *ceiling* | Gap only **8.8%** (vs eager ~55%) |

Compile vs eager on the **same Nsight metric** (wall-span): **128.7 → 48.0 ms** (~**2.7×**). Busy drops only **57.4 → 43.8 ms** (~**1.3×**) — most of the win is fewer launches / less idle, not cheaper math kernels.

---

## 5. Action parity (correctness, not speed)

Same inputs on CPU, GPU eager, and GPU compile (`reduce-overhead`):

| Check | GPU eager vs CPU | GPU compile vs CPU |
|---|---|---|
| max \|Δ\| | **1.66×10⁻³** | **6.56×10⁻³** |
| mean abs error (eager) | 8.71×10⁻⁴ | — |
| Within 1e-2 / 1e-1 | **PASS** | **PASS** |
| Compile vs eager max \|Δ\| | — | **7.91×10⁻³** (still ≤ 1e-2) |

Residuals are expected bf16 / Inductor / TF32-high matmul numeric paths, not throughput artifacts. Compile run-to-run (2× `select_action`) is bit-identical.

---

## 6. Out of scope

| Mode | Status |
|---|---|
| TensorRT / export / non–`reduce-overhead` compile modes | Not in the latency columns above (`max-autotune` smoke: ~75 ms warm, ~141 s cold — see compile report §6) |

---

## 7. Sources

| Artifact | Role |
|---|---|
| [`smoke_test_report.json`](../smoke_test_report.json) | CPU stage wall |
| [`smoke_test_report_gpu.json`](../smoke_test_report_gpu.json) | GPU eager smoke + coarse stages |
| [`smoke_test_report_gpu_compile.json`](../smoke_test_report_gpu_compile.json) | GPU compile smoke (correctness + cold/warm host) |
| [`doc/gpu/eager_mode/smolVLA_profiling.md`](gpu/eager_mode/smolVLA_profiling.md) | GPU eager warm means (authoritative for eager §2–3) |
| [`doc/gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md`](gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md) §1, §11 | Eager Nsight wall/busy + fusion gaps |
| [`doc/gpu/compile_mode/smolVLA_kerne_list_gpu_backend.md`](gpu/compile_mode/smolVLA_kerne_list_gpu_backend.md) | Compile Nsight $T_{\mathrm{e2e}}$ / busy / launch mix (authoritative for compile §2–4) |
| [`doc/gpu/compile_mode/smolVLA_compile_mode_report.md`](gpu/compile_mode/smolVLA_compile_mode_report.md) | Compile enablement, correctness, cold cost |
| [`doc/cpu/smolVLA_cpu_report.md`](cpu/smolVLA_cpu_report.md) | CPU model / control-loop context |

---

## 8. Bottom line

| | CPU | GPU eager | GPU compile (`reduce-overhead`) |
|---|---|---|---|
| **Latency** | ~**20.1 s** / chunk | ~**91 ms** / chunk (**≈220×** vs CPU) | ~**48 ms** / chunk (**≈419×** vs CPU, **≈1.9×** vs eager) |
| **Throughput** | ~**0.05** chunk/s · **14** prefix tok/s | ~**11** chunk/s · **~7.8k** prefix tok/s | ~**21** chunk/s (prefix tok/s n/a) |
| **Bottleneck** | Stage 0 (vision) | Stage 3 (expert ×10) | GEMM + FMHA busy; launch gap already small |
| **Launches / Nsight gap** | — | ~13.8k / ~55% idle | **4101** / **8.8%** idle |

**GPU eager** is the LeRobot default (`compile_model=False`). **GPU compile** is the measured faster path on this GPU once the ~98 s cold compile is paid; prefer it when warm chunk rate matters and a one-time compile cost is acceptable.

# SmolVLA model-level performance: CPU vs GPU (eager)

**Scope.** Same checkpoint [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), same fixed observation + `SMOKE_SEED=0` noise, batch **B=1**. Compare **CPU** vs **GPU eager** (LeRobot default: `compile_model=False`). There is no separate Hub “GPU non-eager” product path; `doc/gpu/compile_mode/` is reserved for a future `torch.compile` capture and is **not** included here.

| Backend | Device | Path in this repo | Primary timing source |
|---|---|---|---|
| **CPU** | host CPU | [`doc/cpu/`](cpu/) | `smoke_test_report.json` stage profile |
| **GPU eager** | RTX 5060 Ti (SM 12.0) | [`doc/gpu/eager_mode/`](gpu/eager_mode/) | [`smolVLA_profiling.md`](gpu/eager_mode/smolVLA_profiling.md) (5 warm runs, CUDA-synchronized) |

Secondary GPU view (same eager kernels): Nsight CUPTI chunk #3 in [`smolVLA_kerne_list_gpu_backend.md`](gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md) §1.

---

## 1. Metrics (model-level)

SmolVLA is a **flow-matching VLA**, not an autoregressive LLM. Classic “decode tokens/s” does not apply directly. Metrics used here:

| Metric | Definition | Why |
|---|---|---|
| **End-to-end latency** \(T_{\mathrm{e2e}}\) | Wall time for one full **chunk fill**: Stages 0→2→(1+3)×`M=10`→4, CUDA-synced on GPU | Time until the 50-step action queue is ready |
| **Chunk throughput** | \(1 / T_{\mathrm{e2e}}\) (chunks/s) | How often the policy can refill actions |
| **Action throughput** | \(N / T_{\mathrm{e2e}}\) with \(N=50\) (actions/s) | Upper bound if the queue is drained one step per control tick with **no** extra model calls |
| **Prefix token throughput** | \(S_p / (T_0+T_2)\), \(S_p=241\) (tok/s) | Tokens through ViT+connector+lang/state + VLM prefill (Stages 0+2) |
| **Suffix token-step throughput** | \((N\cdot M) / T_{1+3}\), \(N=50\), \(M=10\) (tok-step/s) | Action-expert tokens × Euler steps (Stages 1+3) |

Shapes: prefix \(S_p=241\) (192 visual + 48 language + 1 state); suffix chunk \(N=50\); Euler steps \(M=10\); action dim used = 6.

---

## 2. End-to-end latency

Warm inference (excludes model load and GPU first-run warmup).

| | **CPU** | **GPU eager** | Speedup |
|---|---:|---:|---:|
| **\(T_{\mathrm{e2e}}\) (wall)** | **20.129 s** | **0.0913 s** (**91.3 ms**) | **≈220×** |
| Stages 0–4 sum | 20.128 s | 0.0903 s | ≈223× |
| Stage 0 prefix embed | 15.009 s | 23.9 ms | ≈628× |
| Stage 2 VLM prefill | 1.933 s | 7.18 ms | ≈269× |
| Stage 1+3 (suffix + expert ×10) | 3.187 s | 59.1 ms | ≈54× |
| Stage 4 crop/queue | ~0.1 ms | ~0.05 ms | ~2× |

**Smoke `select_action` (includes load/overhead, 2 runs):** CPU **42.87 s** total vs GPU **0.48 s** total — same order of gap; prefer the warm \(T_{\mathrm{e2e}}\) row for model latency.

### Stage share (latency mix shifts on GPU)

| Stage | CPU share | GPU eager share |
|---|---:|---:|
| 0 prefix | **74.6%** | 26.4% |
| 2 prefill | 9.6% | 7.9% |
| 3 expert denoise | 15.8% | **63.5%** |
| 1 suffix + 4 | ~0.1% | ~2.2% |

On CPU, **vision/prefix** dominates. On GPU eager, **expert Euler loop** dominates — fusion/compile wins should target Stage 3 (see eager kernel list §11).

---

## 3. Throughput

| Metric | **CPU** | **GPU eager** | Speedup |
|---|---:|---:|---:|
| Chunk throughput | **0.050** /s | **10.95** /s | ≈220× |
| Action throughput (queue drain) | **2.48** actions/s | **547** actions/s | ≈220× |
| Prefix token throughput (Stages 0+2) | **14.2** tok/s | **7 760** tok/s | ≈545× |
| Suffix token-step throughput (Stages 1+3) | **157** tok-step/s | **8 450** tok-step/s | ≈54× |

**Real-time control note.** At 30 Hz control with \(N=50\), one chunk lasts \(50/30\approx1.67\) s of robot time. CPU \(T_{\mathrm{e2e}}\approx20\) s cannot keep up. GPU eager \(T_{\mathrm{e2e}}\approx91\) ms leaves ample margin for synchronous refill (budget \(N/f_{\mathrm{control}}\)).

---

## 4. Same GPU eager path — host wall vs Nsight

Both rows are **eager**; they measure different clocks.

| View | Latency / chunk | Chunk throughput | Notes |
|---|---:|---:|---|
| PyTorch CUDA-sync \(T_{\mathrm{e2e}}\) | **91.3 ms** | **10.95** /s | Host-visible stage barriers; best model-level latency |
| Nsight CUPTI wall-span (chunk #3) | **128.7 ms** | **7.77** /s | First upsample → last kernel; includes capture/stream idle |
| Nsight busy Σ (kernel runtimes only) | **57.4 ms** | **17.4** /s *ceiling* | No inter-launch gaps; not achievable as e2e without fusion/graphs |

Fusion gap analysis (eager §11) estimates ~**50 ms** of deletable inter-kernel idle on this chunk — consistent with busy ≪ CUPTI wall.

---

## 5. Action parity (correctness, not speed)

Same inputs on CPU and GPU eager:

| Check | Value |
|---|---|
| max \|CPU − GPU\| | **1.66×10⁻³** |
| mean abs error | 8.71×10⁻⁴ |
| Within 1e-2 / 1e-1 | **PASS** |

Residual is expected bf16 GPU vs CPU numeric paths, not a throughput artifact.

---

## 6. Not in this comparison

| Mode | Status |
|---|---|
| **GPU `torch.compile`** (`compile_model=True`) | Folder [`doc/gpu/compile_mode/`](gpu/compile_mode/) exists; **no warm latency / throughput capture yet** |
| TensorRT / CUDA graphs / export | Out of scope |

When compile numbers exist, add a third column with the same \(T_{\mathrm{e2e}}\) and tok/s definitions.

---

## 7. Sources

| Artifact | Role |
|---|---|
| [`smoke_test_report.json`](../smoke_test_report.json) | CPU stage wall |
| [`smoke_test_report_gpu.json`](../smoke_test_report_gpu.json) | GPU smoke + coarse stages |
| [`doc/gpu/eager_mode/smolVLA_profiling.md`](gpu/eager_mode/smolVLA_profiling.md) | GPU eager warm means (authoritative for §2–3) |
| [`doc/gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md`](gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md) §1, §11 | Nsight wall/busy + fusion gaps |
| [`doc/cpu/smolVLA_cpu_report.md`](cpu/smolVLA_cpu_report.md) | CPU model / control-loop context |

---

## 8. Bottom line

| | CPU | GPU eager |
|---|---|---|
| **Latency** | ~**20.1 s** / chunk | ~**91 ms** / chunk (**≈220×** faster) |
| **Throughput** | ~**0.05** chunk/s · **14** prefix tok/s | ~**11** chunk/s · **~7.8k** prefix tok/s |
| **Bottleneck stage** | Stage 0 (vision) | Stage 3 (expert ×10) |

**GPU eager is the measured production-default GPU path.** Next model-level delta to capture: **eager vs compile** under `doc/gpu/compile_mode/`.

# SmolVLA GPU compile-mode report

Date: 2026-09-25  
Host: Vast.ai (`162.249.226.242:32130`), container hostname `5cc711215e18`  
GPU: NVIDIA GeForce RTX 5060 Ti (16 GB, SM 12.0)  
Script: `src/smolvla_test_infer.py` with `SMOKE_DEVICE=cuda` + `SMOKE_COMPILE=1`  
Baseline: CPU `smoke_test_report.json` / remote `smoke_test_report_cpu.json` (same fixed observation + `SMOKE_SEED=0` noise)

## Verdict

**PASS** for `torch.compile` with **`reduce-overhead`**.

| Check | Result |
|---|---|
| Finite / shape / magnitude | OK |
| Run-to-run determinism (2× `select_action`) | **bit-identical** (`max_abs_diff=0`) |
| vs CPU abs error | **max 6.56×10⁻³** ≤ 1e-2 ✓ (≤ 1e-1 ✓) |
| vs GPU eager abs error | max **7.91×10⁻³** (numeric path differs under Inductor; still within CPU tol band) |

LeRobot’s default `compile_mode=max-autotune` was also attempted on the same host; see §6.

## How to enable

LeRobot 0.6.1 already exposes:

```python
config.compile_model = True
config.compile_mode = "reduce-overhead"  # or max-autotune / default / ...
```

This wraps `VLAFlowMatching.sample_actions` and `forward` with `torch.compile` and sets `torch.set_float32_matmul_precision("high")`.

This repo:

```bash
export SMOKE_DEVICE=cuda
export SMOKE_COMPILE=1
export SMOKE_COMPILE_MODE=reduce-overhead   # default in the smoke script
python src/smolvla_test_infer.py
# → smoke_test_report_gpu_compile.json
```

## Environment

| Item | Value |
|---|---|
| Python | 3.12.14 (`/venv/main`) |
| PyTorch | 2.11.0+cu128 |
| LeRobot | 0.6.1 (`compile_model` / `compile_mode` present) |
| Transformers | 5.5.4 |
| Model | `lerobot/smolvla_base` (~450M params) |

## Correctness vs CPU

Same inputs as the CPU / GPU-eager smokes.

| Dim | CPU | GPU compile (`reduce-overhead`) | \|Δ\| |
|---:|---:|---:|---:|
| 0 | -0.063721 | -0.064164 | 4.43e-4 |
| 1 | -0.029679 | -0.031366 | 1.69e-3 |
| 2 | -0.193380 | -0.191901 | 1.48e-3 |
| 3 | 0.190984 | 0.189634 | 1.35e-3 |
| 4 | -0.081199 | -0.074638 | **6.56e-3** |
| 5 | -0.396449 | -0.399840 | 3.39e-3 |

- **max abs error vs CPU:** `6.56e-3` (eager GPU was `1.66e-3`)
- Residual is expected from Inductor / TF32-high matmul vs eager bf16 path; not an input mismatch.

## Latency (model-level)

| Metric | GPU eager (prior smoke) | GPU compile `reduce-overhead` |
|---|---:|---:|
| First `select_action` (cold / compile) | (eager load only) | **98.0 s** (Inductor + CUDA graphs) |
| Second `select_action` (warm chunk fill) | ~0.09–0.48 s* | **0.077 s** |
| Stage 0–4 manual profile wall† | 0.0906 s | 0.0895 s |

\* Eager smoke “inference.seconds=0.48” covered two runs and overhead; warm stage sum ≈ **91 ms**.  
† Stage profiler walks Stage 0–4 **without** going through the compiled `sample_actions` wrapper, so it is **not** a compile-speed benchmark — use the warm `select_action` row for compile latency.

**Warm chunk throughput (compile):** \(1/0.077 \approx\) **13.0 chunks/s** (vs eager ~11.0 /s from 91 ms e2e).

## Issues encountered / debug notes

1. **SSH port** — instance moved from `:32983` → **`:32130`** (old port refused).
2. **Inductor warning:** `Not enough SMs to use max_autotune_gemm mode` on RTX 5060 Ti (36 SMs). Harmless; GEMM autotune falls back.
3. **`reduce-overhead` first call ~98 s** — expected; subsequent calls reuse the compiled graph / CUDA graphs.
4. **Stage profiler ≠ compiled path** — for fair stage timing under compile, wrap or call `predict_action_chunk` / `sample_actions` as a whole (future work).
5. **CLI probe `lerobot-record` dry-run** still fails without robot hardware (unchanged; unrelated to compile).

## `max-autotune` (LeRobot default compile_mode)

Also **PASS** on the same host / same CPU baseline.

| Metric | `reduce-overhead` | `max-autotune` |
|---|---:|---:|
| Cold first `select_action` | **98.0 s** | **140.8 s** |
| Warm second `select_action` | **0.077 s** | **0.075 s** |
| vs CPU max abs | 6.56e-3 | **4.73e-3** |
| Run-to-run identical | yes | yes |

Warm latency is essentially the same; `max-autotune` costs ~**43 s** more cold compile for ~**2 ms** warm gain on this GPU. Prefer **`reduce-overhead`** unless chasing last-millisecond kernel choices.

## Artifacts

| Path | Content |
|---|---|
| [`smoke_test_report_gpu_compile.json`](../../../smoke_test_report_gpu_compile.json) | Full smoke JSON (`reduce-overhead`, repo root) |
| [`smoke_test_report_gpu_compile_reduce_overhead.json`](smoke_test_report_gpu_compile_reduce_overhead.json) | Same, under this folder |
| [`smoke_test_run_compile.log`](smoke_test_run_compile.log) | `reduce-overhead` console log |
| [`smoke_test_report_gpu_compile_max_autotune.json`](smoke_test_report_gpu_compile_max_autotune.json) | `max-autotune` smoke JSON |
| [`smoke_test_run_compile_max_autotune.log`](smoke_test_run_compile_max_autotune.log) | `max-autotune` console log |
| [`SmolVLA_CompileOp_List_gpu_backend.md`](SmolVLA_CompileOp_List_gpu_backend.md) | **Optimized FX/AOT operator list** (AtenOp-style) |
| [`smolvla_compile_fx_graphs.json`](smolvla_compile_fx_graphs.json) | Raw captured graphs (19 334 compute ops / 8 graphs) |

## Bottom line

`torch.compile` on SmolVLA **works on this GPU** with LeRobot’s built-in flag. Prefer **`reduce-overhead`** for a shorter cold compile (~98 s) and warm latency ≈ **77 ms / chunk**, with actions matching CPU within **1e-2**. `max-autotune` also passes (~141 s cold, ~75 ms warm) but is not worth the extra compile time here.

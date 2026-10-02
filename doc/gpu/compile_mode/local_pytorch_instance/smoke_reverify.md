# Smoke re-verify — local editable PyTorch

**Date:** 2026-09-28  
**Baseline report:** [`../smolVLA_compile_mode_smoke_test_report.md`](../smolVLA_compile_mode_smoke_test_report.md)  
**Command:** `SMOKE_DEVICE=cuda SMOKE_COMPILE=1 SMOKE_COMPILE_MODE=reduce-overhead python src/smolvla_test_infer.py`  
**Torch:** `2.11.0a0+git70d99e9` from `/workspace/pytorch` (editable)

## Verdict

**PASS** (`status=PASS`, total wall **323.5 s**)

| Check | Wheel baseline (report) | Local editable | Result |
|-------|------------------------:|---------------:|--------|
| Finite / shape / magnitude | OK | OK | OK |
| Run-to-run determinism | bit-identical | **max_abs_diff=0** | OK |
| vs CPU abs error | 6.56e-3 ≤ 1e-2 | **4.71e-3 ≤ 1e-2** | OK |
| Cold first `select_action` | ~98 s (prior host) | **245.8 s** | slower cold (this host / cold caches) |
| Warm second `select_action` | ~0.077 s | **0.862 s** | slower warm; still functional |

Cold/warm absolute times differ from the earlier Vast host numbers (new instance, cold HF/Inductor caches, editable build). Correctness gates match the smoke report criteria.

## Environment

| Item | Value |
|------|------|
| Python | 3.12 (`/venv/main`) |
| PyTorch | `2.11.0a0+git70d99e9` · `/workspace/pytorch/torch` |
| Git | `v2.11.0` / `70d99e9` |
| GPU | RTX 5060 Ti · CUDA 12.8 |
| Compile | `reduce-overhead` |

## vs CPU (per-dim)

| Dim | CPU | Local compile | \|Δ\| |
|---:|---:|---:|---:|
| 0 | -0.063721 | -0.067072 | 3.35e-3 |
| 1 | -0.029679 | -0.028974 | 7.06e-4 |
| 2 | -0.193380 | -0.193207 | 1.73e-4 |
| 3 | 0.190984 | 0.190287 | 6.96e-4 |
| 4 | -0.081199 | -0.076491 | **4.71e-3** |
| 5 | -0.396449 | -0.398619 | 2.17e-3 |

**max abs vs CPU:** `4.71e-3` (baseline wheel report had `6.56e-3`).

## Artifacts

| Path | Content |
|------|---------|
| [`smoke_test_report_gpu_compile_local_pytorch.json`](smoke_test_report_gpu_compile_local_pytorch.json) | Full smoke JSON |
| [`smoke_reverify_run.log`](smoke_reverify_run.log) | Console log (gitignored `*.log` may not sync) |
| Remote logs | `/workspace/local_pytorch_logs/build_full.log` |

## Bottom line

Editable local PyTorch **replaces** the wheel in `/venv/main` and **passes** the same compile-mode smoke gates. Dynamo / Inductor sources under `/workspace/pytorch/torch/_dynamo` and `_inductor` are now editable for diving into compile internals.

# vla_analysis

Study and analysis of Vision-Language-Action (VLA) models, using **SmolVLA** ([`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)) as the primary case study.

## Goals

1. **Learn the VLA stack** — how cameras, language, and proprioception map into continuous robot actions.
2. **Analyze structure and performance** — algorithm stages, ATen ops, GPU kernels, FLOPs/IO, and wall-clock profiles.
3. **Optimize later** — use that analysis at **framework**, **model-graph**, and **IR** levels (fusion, compile, custom kernels).

## Repository layout

```
doc/
  SmolVLA_Algorithm_Architecture.md   # stage-level algorithm / tensor flow
  cpu/                                # CPU study (operators + report)
  gpu/
    eager_mode/                       # GPU eager-path capture (default product path)
src/                                  # inference / profiling harnesses
```

### Architecture & model-level perf

| Path | Content |
|------|---------|
| [`doc/SmolVLA_Algorithm_Architecture.md`](doc/SmolVLA_Algorithm_Architecture.md) | Stage 0–4 algorithm and tensor flow |
| [`doc/SmolVLA_model_level_performance_CPU_vs_GPU_eager.md`](doc/SmolVLA_model_level_performance_CPU_vs_GPU_eager.md) | CPU vs GPU eager: e2e latency + throughput (chunks/s, actions/s, tok/s) |

### CPU

| Path | Content |
|------|---------|
| [`doc/cpu/smolVLA_cpu_report.md`](doc/cpu/smolVLA_cpu_report.md) | Model report, runtime, stage timing |
| [`doc/cpu/SmolVLA_Operator_List_cpu.md`](doc/cpu/SmolVLA_Operator_List_cpu.md) | Operators and FLOP accounting |

### GPU — eager mode (current focus)

Public SmolVLA / LeRobot default is **eager** (`compile_model=False`). All GPU docs below are from that path (Nsight + ATen chrono on RTX 5060 Ti).

| Path | Content |
|------|---------|
| [`doc/gpu/eager_mode/smolVLA_gpu_report.md`](doc/gpu/eager_mode/smolVLA_gpu_report.md) | GPU smoke test vs CPU baseline |
| [`doc/gpu/eager_mode/smolVLA_profiling.md`](doc/gpu/eager_mode/smolVLA_profiling.md) | Stage / CUDA event profiling notes |
| [`doc/gpu/eager_mode/SmolVLA_AtenOp_List_gpu_backend.md`](doc/gpu/eager_mode/SmolVLA_AtenOp_List_gpu_backend.md) | ATen ops in call order + fusion candidates (§6) |
| [`doc/gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md`](doc/gpu/eager_mode/smolVLA_kerne_list_gpu_backend.md) | CUPTI kernels, roofline util, **fusion gap benefit (§11)** |
| [`doc/gpu/eager_mode/smolvla_aten_chrono.json`](doc/gpu/eager_mode/smolvla_aten_chrono.json) / [`.log`](doc/gpu/eager_mode/smolvla_aten_chrono.log) | ATen chrono capture |
| [`doc/gpu/eager_mode/nsight/`](doc/gpu/eager_mode/nsight/) | `smolvla_nsys.nsys-rep` / `.sqlite` / `.log` |

**Fusion takeaway (eager, chunk #3):** inter-kernel **gap** savings from candidate fusions ≈ **50 ms / chunk** (dominated by RoPE + RMSNorm); view-only transpose/GQA add little gap but can still fold cast/layout **busy** into GEMM/Flash (~2–4 ms upper bound). See kernel list **§11**.

### Scripts

| Path | Role |
|------|------|
| [`src/smolvla_test_infer.py`](src/smolvla_test_infer.py) | CPU/CUDA `select_action` smoke test + Stage 0–4 timing |
| [`src/smolvla_profile_detail.py`](src/smolvla_profile_detail.py) | Finer stage / event profiling |
| [`src/smolvla_aten_profile.py`](src/smolvla_aten_profile.py) | `TorchDispatchMode` ATen chrono → JSON/log |
| [`src/smolvla_nsight_target.py`](src/smolvla_nsight_target.py) | Nsight Systems capture target (`cudaProfilerStart/Stop`) |

## Eager vs compile

| Mode | Status in this repo | Notes |
|------|---------------------|--------|
| **Eager** (default) | Documented under `doc/gpu/eager_mode/` | Matches Hub / LeRobot default |
| **`torch.compile`** | Not captured yet | Optional in current LeRobot (`compile_model`, default `False`, mode `max-autotune`). Sibling VLAs (e.g. openpi π₀ PyTorch) often compile `sample_actions` by default |

## Status

- **Done:** SmolVLA architecture + CPU report; GPU eager ATen/kernel lists, util/roofline, fusion gap analysis verified from nsys.
- **Next:** Optional `doc/gpu/compile_mode/` capture; then framework / graph / IR optimizations guided by §11 priorities (RoPE, RMSNorm, eager→Flash, cast/epilogue fold).

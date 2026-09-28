# Local PyTorch instance (Dynamo / Inductor dive)

Goal: replace the venv’s wheel `torch==2.11.0+cu128` with a **same-commit source build** on the GPU host so Dynamo / Inductor can be edited and stepped through.

Smoke baseline: [`../smolVLA_compile_mode_smoke_test_report.md`](../smolVLA_compile_mode_smoke_test_report.md).

## Host layout (remote only — not synced into this git repo’s runtime tree beyond this doc)

| Path | Role |
|------|------|
| `/workspace/pytorch` | PyTorch git checkout (outside `vla_analysis`) |
| `/workspace/pytorch-build` | CMake / object build dir (`TORCH_BUILD_DIR`) |
| `/venv/main` | Target venv (editable install) |
| This folder | Process notes, timings, smoke re-verify log |

## Target version

| Item | Value |
|------|------|
| Installed wheel (before) | `2.11.0+cu128` |
| `torch.version.git_version` | `70d99e998b4955e0049d13a98d77ae1b14db1f45` |
| CUDA toolkit | 12.8 |
| GPU arch | `TORCH_CUDA_ARCH_LIST=12.0` (RTX 5060 Ti / SM120) |
| Python | 3.12.14 |

## Time & disk estimate (before build)

| Phase | Expected wall time (32 vCPU) | Disk |
|-------|------------------------------:|------|
| Clone (+ needed submodules) | **5–20 min** | ~1.5–3 GB |
| CMake configure | **2–5 min** | — |
| Compile + link CUDA `libtorch` | **1.5–3 h** (often ~90–150 min for single-arch Release) | **peak ~25–45 GB** typical |
| Editable install into venv | **1–5 min** | +~1–2 GB |
| Smoke re-verify (`reduce-overhead`) | **~2–3 min** warm path; cold first compile **~1.5–2 min+** (was ~98 s on wheel) | — |

**Bottom line:** plan on about **2–3.5 hours** end-to-end for a minimized CUDA build.  
**Risk on this Vast container:** root overlay is only **~32 GB** (~19 GB free after cache clean) — a full CUDA build can **OOM the disk**. Build uses aggressive feature cuts; if disk fills, fall back to the Python-overlay mode in `NOTES.md` (still enough for Dynamo/Inductor logic).

## Build flags (minimized)

```text
USE_CUDA=1
USE_CUDNN=1
TORCH_CUDA_ARCH_LIST=12.0
CMAKE_BUILD_TYPE=Release
BUILD_TEST=0
USE_DISTRIBUTED=0
USE_NCCL=0
USE_MPI=0
USE_FBGEMM=0
USE_KINETO=1
USE_MKLDNN=0
USE_QNNPACK=0
USE_PYTORCH_QNNPACK=0
USE_XNNPACK=0
MAX_JOBS=8          # lower peak /tmp object pressure
CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

## Status

See [`STATUS.md`](STATUS.md) for live progress, [`NOTES.md`](NOTES.md) for commands/logs, [`smoke_reverify.md`](smoke_reverify.md) after install.

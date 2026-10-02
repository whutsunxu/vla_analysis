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

## Time & disk (actual on this host)

| Phase | Actual |
|-------|--------|
| Clone + deps | ~minutes |
| Compile + editable install | **~2 h** (`ninja -j 6`, 2317 steps) |
| Peak free disk remaining | **~17 GB** of 32 GB overlay (survived) |
| Smoke re-verify cold `select_action` | **245.8 s** |
| Smoke re-verify warm | **0.86 s** · **PASS** vs CPU max abs **4.71e-3** |

See [`STATUS.md`](STATUS.md) and [`smoke_reverify.md`](smoke_reverify.md).

## Build flags (used)

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
USE_FLASH_ATTENTION=0
USE_MEM_EFF_ATTENTION=0
MAX_JOBS=6
CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

## Status

**Complete.** Editable torch is live in `/venv/main`. Details: [`STATUS.md`](STATUS.md), smoke: [`smoke_reverify.md`](smoke_reverify.md).

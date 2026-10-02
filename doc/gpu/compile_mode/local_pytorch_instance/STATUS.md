# Status — local PyTorch build

| Field | Value |
|-------|------|
| Started | 2026-09-28 |
| Finished | 2026-09-28 (~2h compile + install) |
| Target | `v2.11.0` / `70d99e998b4955e0049d13a98d77ae1b14db1f45` |
| Source | `/workspace/pytorch` (remote only, outside repo) |
| Build dir | `/workspace/pytorch/build` (~1.5 GB objects) |
| Venv | `/venv/main` editable: `torch-2.11.0a0+git70d99e9` |
| Smoke re-verify | **PASS** (see [`smoke_reverify.md`](smoke_reverify.md)) |

## Timeline

| Step | Status | Notes |
|------|--------|-------|
| Free disk / estimate | done | Started ~20 GB free; finished with ~17 GB free |
| Clone `v2.11.0` | done | `/workspace/pytorch` @ `70d99e9` |
| Install build deps | done | pip requirements + cmake/ninja; lerobot pin warnings on cmake/setuptools |
| CMake + compile | done | `pip install -e .` → ninja `-j 6` · 2317 steps · ~2h |
| Editable install | done | `Successfully installed torch-2.11.0a0+git70d99e9` |
| Smoke re-verify | done | `SMOKE_COMPILE=1 reduce-overhead` → **PASS** |

## Torch identity (post-install)

```
version: 2.11.0a0+git70d99e9
file:    /workspace/pytorch/torch/__init__.py
dynamo:  /workspace/pytorch/torch/_dynamo/__init__.py
inductor:/workspace/pytorch/torch/_inductor/__init__.py
cuda:    True 12.8 · RTX 5060 Ti
```

## Mode

- **Used:** full CUDA editable build (same git commit as the prior `2.11.0+cu128` wheel family).
- torchvision warns `requires torch==2.11.0` vs `2.11.0a0+git…` — expected for source builds; smoke still PASS.

# Pre · prepare_images — upsample / scale / mask (eager)

Standalone **eager** unit test for the Pre stage documented in
[`SmolVLA_op_list_inductor_nsight.md`](../../inductor/SmolVLA_op_list_inductor_nsight.md) §Pre.

This path runs **outside** Inductor (`sample_actions` compile scope). Full-model
scripts only call `policy.prepare_images(batch)`; the op body lives in lerobot.

## Ops under test (cam0)

| Order | Op | Shapes |
|------:|----|--------|
| 1 | `upsample_bilinear2d` | `[1,3,256,256] f32` → `[1,3,512,512] f32` |
| 2 | `mul` (×2) | `[1,3,512,512] f32` |
| 3 | `add` (−1) | `[1,3,512,512] f32` |
| 4 | `ones` → bool | `[1,32,32] bool` (ViT patch mask; nsight Pre table) |

Also asserts the camera-valid mask from `prepare_images`: `ones([B], bool)`.

For square 256→512 inputs, `resize_with_pad` pad widths are 0 (pure bilinear).

## Source mapping

| Piece | Location |
|-------|----------|
| Call site in tests | `src/smolvla_test_infer.py`, `smolvla_compile_*_dump.py`, aten/profile scripts |
| Policy body | `SmolVLAPolicy.prepare_images` |
| Resize helper | `lerobot.policies.common.vla_utils.resize_with_pad` |

## Run

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/pre-process-upsample
source /venv/main/bin/activate   # GPU host
SMOKE_DEVICE=cuda python test_prepare_images_upsample.py
# optional profiler name dump:
PROFILE=1 SMOKE_DEVICE=cuda python test_prepare_images_upsample.py
```

Writes `prepare_images_upsample_report.json` next to the script.

## Nsight ablation profiles

See [`nsys/`](nsys/) — three `.nsys-rep` captures (full / no_scale / no_upsample), 10 iters each.

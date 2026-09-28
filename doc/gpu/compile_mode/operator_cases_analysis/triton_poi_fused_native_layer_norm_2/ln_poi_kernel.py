"""Faithful standalone of Inductor `triton_poi_fused_native_layer_norm_2`.

Source: inductor_debug_samples/model__4_inference_4.4/output_code.py
Call site: .run(arg1_1, buf3, buf4, arg2_1, arg3_1, buf6, 1024, 768)

Baseline (in-model):
  x     [1,Y,X] bf16  strides (Y*X, 1, Y)   # feature-major load
  mean  [1,Y,1] f32
  var   [1,Y,1] f32   # sum of squared diffs (Inductor red stage)
  gamma [X]     bf16
  beta  [X]     bf16
  out   [1,Y,X] bf16  strides (Y*X, X, 1)   # contiguous store

Grid2D: program_id(0)=x tiles, program_id(1)=y tiles — same as Inductor.
Default blocks match an Inductor-compiled TTIR variant for this kernel:
  XBLOCK=64, YBLOCK=64, num_warps=16 → grid (12, 16) at baseline Y=1024,X=768.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# Baseline Inductor shapes / launch (GRAPH5 · ViT L0 LN epilogue)
BASE_Y = 1024
BASE_X = 768
DEFAULT_XBLOCK = 64
DEFAULT_YBLOCK = 64
DEFAULT_NUM_WARPS = 16
EPS = 1e-6

# FLOPs heuristic (smolvla_compile_fused_table_metrics.py): poi LN = 7N
FLOPS_PER_ELEM = 7.0
DRAM_PEAK_GB_S = 448.0
FP32_PEAK_TFLOP_S = 23.7


@dataclass(frozen=True)
class LnPoiShapes:
    y: int
    x: int
    xblock: int = DEFAULT_XBLOCK
    yblock: int = DEFAULT_YBLOCK
    num_warps: int = DEFAULT_NUM_WARPS

    @property
    def grid(self) -> tuple[int, int]:
        return (triton.cdiv(self.x, self.xblock), triton.cdiv(self.y, self.yblock))

    @property
    def numel(self) -> int:
        return self.y * self.x


@triton.jit
def triton_poi_fused_native_layer_norm_2(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    in_ptr3,
    in_ptr4,
    out_ptr0,
    ynumel,
    xnumel,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """Pointwise LN affine epilogue — same math/indexing as Inductor codegen."""
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    mask = ymask & xmask
    x1 = xindex
    y0 = yindex

    # Feature-major load: index = y + ynumel * x  (Inductor: y + 1024*x)
    tmp0 = tl.load(in_ptr0 + (y0 + ynumel * x1), mask, eviction_policy="evict_last").to(
        tl.float32
    )
    tmp2 = tl.load(in_ptr1 + y0, ymask, eviction_policy="evict_last")
    tmp4 = tl.load(in_ptr2 + y0, ymask, eviction_policy="evict_last")
    tmp11 = tl.load(in_ptr3 + x1, xmask, eviction_policy="evict_last").to(tl.float32)
    tmp14 = tl.load(in_ptr4 + x1, xmask, eviction_policy="evict_last").to(tl.float32)

    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = xnumel.to(tl.float32)
    tmp6 = tmp4 / tmp5
    tmp7 = tl.full([1, 1], 1e-6, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    # Contiguous store: index = x + xnumel * y  (Inductor: x + 768*y)
    tl.store(out_ptr0 + (x1 + xnumel * y0), tmp16, mask)


def algo_io_bytes(y: int, x: int) -> int:
    """Σ bytes(inputs) + bytes(output) — same IO model as Inductor §GRAPH5 row."""
    return (
        y * x * 2  # x bf16
        + y * 4  # mean f32
        + y * 4  # var f32
        + x * 2  # gamma bf16
        + x * 2  # beta bf16
        + y * x * 2  # out bf16
    )


def shape_for_target_io_mb(io_mb: float, x: int = BASE_X) -> LnPoiShapes:
    """Grow Y (tokens) so algo IO ≈ io_mb decimal MB; keep hidden dim X fixed."""
    target = io_mb * 1e6
    # Dominant: 4*Y*X + 8*Y + 4*X  (two bf16 activations + mean/var + gamma/beta)
    # Y ≈ (target - 4*X) / (4*X + 8)
    denom = 4 * x + 8
    y = max(int(round((target - 4 * x) / denom)), 1)
    # Snap Y to multiple of YBLOCK for clean grids (same as Inductor ymask=True path)
    step = DEFAULT_YBLOCK
    y = max(step, (y // step) * step)
    best = y
    best_err = abs(algo_io_bytes(y, x) - target)
    for cand in range(max(step, y - 4 * step), y + 4 * step + 1, step):
        err = abs(algo_io_bytes(cand, x) - target)
        if err < best_err:
            best, best_err = cand, err
    return LnPoiShapes(y=best, x=x)


def make_buffers(
    shapes: LnPoiShapes, device: torch.device, *, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Allocate tensors with Inductor strides; fill with reproducible noise."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    y, x = shapes.y, shapes.x

    x_in = torch.empty_strided(
        (1, y, x), (y * x, 1, y), dtype=torch.bfloat16, device=device
    )
    # Fill via contiguous then copy into strided view
    tmp = torch.randn(1, y, x, generator=g, dtype=torch.float32).to(torch.bfloat16)
    x_in.copy_(tmp.to(device))

    # mean / var from a cheap host-side LN prep on contiguous float view
    x_f = tmp.float()
    mean = x_f.mean(dim=-1, keepdim=True)  # [1,Y,1]
    # Inductor red stores sum of squared diffs (var * X), not mean variance
    var_sum = ((x_f - mean) ** 2).sum(dim=-1, keepdim=True)

    mean_t = torch.empty_strided(
        (1, y, 1), (y, 1, y), dtype=torch.float32, device=device
    )
    var_t = torch.empty_strided(
        (1, y, 1), (y, 1, y), dtype=torch.float32, device=device
    )
    mean_t.copy_(mean.to(device))
    var_t.copy_(var_sum.to(device))

    gamma = torch.randn(x, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    beta = torch.randn(x, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    out = torch.empty_strided(
        (1, y, x), (y * x, x, 1), dtype=torch.bfloat16, device=device
    )
    return x_in, mean_t, var_t, gamma, beta, out


def reference_out(
    x_in: torch.Tensor,
    mean: torch.Tensor,
    var_sum: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference matching the Triton epilogue math."""
    y, x = x_in.shape[1], x_in.shape[2]
    # Gather feature-major → logical [1,Y,X]
    # x_in[b,y,x] lives at offset y + y*x_feat  wait: stride (Y*X, 1, Y)
    x_logical = x_in.transpose(-1, -2).contiguous().transpose(-1, -2)  # make contiguous logical
    # Actually transpose trick: as_strided already defines logical indexing
    x_f = x_in.float()
    m = mean.float()
    rstd = torch.rsqrt(var_sum.float() / x + EPS)
    out = (x_f - m) * rstd * gamma.float() + beta.float()
    return out.to(torch.bfloat16).contiguous()


def launch(
    x_in: torch.Tensor,
    mean: torch.Tensor,
    var_sum: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    shapes: LnPoiShapes,
) -> None:
    grid = shapes.grid
    triton_poi_fused_native_layer_norm_2[grid](
        x_in,
        mean,
        var_sum,
        gamma,
        beta,
        out,
        shapes.y,
        shapes.x,
        YBLOCK=shapes.yblock,
        XBLOCK=shapes.xblock,
        num_warps=shapes.num_warps,
        num_stages=1,
    )


def roofline(shapes: LnPoiShapes, time_s: float) -> dict:
    io_b = algo_io_bytes(shapes.y, shapes.x)
    flops = FLOPS_PER_ELEM * shapes.numel
    bd = (io_b / time_s) / 1e9 if time_s > 0 else float("nan")
    gps = (flops / time_s) / 1e9 if time_s > 0 else float("nan")
    return {
        "io_bytes": io_b,
        "io_gb": io_b / 1e9,
        "io_mb": io_b / 1e6,
        "flops": flops,
        "gflops": flops / 1e9,
        "bd_gb_s": bd,
        "bd_util": bd / DRAM_PEAK_GB_S,
        "gflops_s": gps,
        "flops_util": gps / (FP32_PEAK_TFLOP_S * 1e3),
        "ai": flops / io_b if io_b else float("nan"),
    }


def baseline_shapes() -> LnPoiShapes:
    return LnPoiShapes(y=BASE_Y, x=BASE_X)

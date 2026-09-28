#!/usr/bin/env python3
"""Unit test for Inductor `triton_poi_fused_native_layer_norm_2`.

Matches GRAPH5 ViT L0 LN epilogue:
  launch Grid2D (ceil(768/64), ceil(1024/64)) = (12, 16) with XBLOCK=64, YBLOCK=64, num_warps=16
  inputs/outputs dims + dtypes + strides from model__4 output_code.py

Usage:
  SMOKE_DEVICE=cuda python test_triton_poi_fused_native_layer_norm_2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from ln_poi_kernel import (
    BASE_X,
    BASE_Y,
    DEFAULT_XBLOCK,
    DEFAULT_YBLOCK,
    algo_io_bytes,
    baseline_shapes,
    launch,
    make_buffers,
    reference_out,
    roofline,
)

OUT_DIR = Path(__file__).resolve().parent
REPORT_PATH = OUT_DIR / "ln_poi_baseline_report.json"


def resolve_device() -> torch.device:
    req = os.environ.get("SMOKE_DEVICE", "cuda").lower()
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if req == "cuda":
            raise RuntimeError("SMOKE_DEVICE=cuda but CUDA is not available")
    return torch.device("cpu")


def main() -> int:
    device = resolve_device()
    if device.type != "cuda":
        print("CUDA required for Triton LN poi kernel", file=sys.stderr)
        return 2

    shapes = baseline_shapes()
    assert shapes.y == BASE_Y and shapes.x == BASE_X
    assert shapes.xblock == DEFAULT_XBLOCK and shapes.yblock == DEFAULT_YBLOCK
    assert shapes.grid == (12, 16), f"unexpected grid {shapes.grid}"

    x_in, mean, var_sum, gamma, beta, out = make_buffers(shapes, device, seed=0)

    # Stride / dtype contracts from Inductor call site
    assert x_in.shape == (1, BASE_Y, BASE_X)
    assert x_in.stride() == (BASE_Y * BASE_X, 1, BASE_Y)
    assert x_in.dtype == torch.bfloat16
    assert mean.shape == (1, BASE_Y, 1) and mean.dtype == torch.float32
    assert var_sum.shape == (1, BASE_Y, 1) and var_sum.dtype == torch.float32
    assert gamma.shape == (BASE_X,) and gamma.dtype == torch.bfloat16
    assert beta.shape == (BASE_X,) and beta.dtype == torch.bfloat16
    assert out.stride() == (BASE_Y * BASE_X, BASE_X, 1)

    # Warmup compile
    launch(x_in, mean, var_sum, gamma, beta, out, shapes)
    torch.cuda.synchronize()

    ref = reference_out(x_in, mean, var_sum, gamma, beta)
    out.zero_()
    launch(x_in, mean, var_sum, gamma, beta, out, shapes)
    torch.cuda.synchronize()

    # bf16 epilogue — allow modest ulp slack vs fp32 reference path
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < 2e-2
    print(
        f"correctness max_abs={max_diff:.6g} mean_abs={mean_diff:.6g} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        return 1

    # Quick timing (host wall, 10 iters after 5 warmup) — nsys is authoritative
    n_warmup = int(os.environ.get("N_WARMUP", "5"))
    n_iters = int(os.environ.get("N_ITERS", "10"))
    for _ in range(n_warmup):
        launch(x_in, mean, var_sum, gamma, beta, out, shapes)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for i in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        launch(x_in, mean, var_sum, gamma, beta, out, shapes)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt)
        print(f"  iter {i:02d}: {dt:.4f} ms")

    mean_ms = sum(times_ms) / len(times_ms)
    # Rough GPU busy ≈ host wall for this single-kernel launch (still launch+sync heavy)
    rf = roofline(shapes, mean_ms / 1e3)
    report = {
        "kernel": "triton_poi_fused_native_layer_norm_2",
        "device": str(device),
        "y": shapes.y,
        "x": shapes.x,
        "XBLOCK": shapes.xblock,
        "YBLOCK": shapes.yblock,
        "grid": list(shapes.grid),
        "dtypes": {
            "x": "bfloat16",
            "mean": "float32",
            "var": "float32",
            "gamma": "bfloat16",
            "beta": "bfloat16",
            "out": "bfloat16",
        },
        "strides": {
            "x": list(x_in.stride()),
            "out": list(out.stride()),
        },
        "algo_io_bytes": algo_io_bytes(shapes.y, shapes.x),
        "algo_io_mb": algo_io_bytes(shapes.y, shapes.x) / 1e6,
        "correctness_max_abs": max_diff,
        "host_mean_ms": mean_ms,
        "host_iters_ms": times_ms,
        "roofline_from_host_mean": rf,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"grid={shapes.grid} XBLOCK={shapes.xblock} YBLOCK={shapes.yblock} "
        f"algo_io={report['algo_io_mb']:.3f} MB  host_mean={mean_ms:.4f} ms"
    )
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

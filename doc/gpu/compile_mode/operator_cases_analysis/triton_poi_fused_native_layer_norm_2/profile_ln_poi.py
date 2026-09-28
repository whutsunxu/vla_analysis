#!/usr/bin/env python3
"""Nsight-ready profiling harness for `triton_poi_fused_native_layer_norm_2`.

No correctness checks. Warmup then N measured iters (default 10).

Modes:
  baseline (default)     — Y=1024, X=768, Grid2D (12,16), XBLOCK=64, YBLOCK=64, num_warps=16
  TARGET_IO_MB=<N>       — grow Y so algo IO (Σ in+out bytes) ≈ N decimal MB;
                           X stays 768; XBLOCK/YBLOCK/num_warps unchanged (grid Y grows)

Usage:
  SMOKE_DEVICE=cuda python profile_ln_poi.py
  TARGET_IO_MB=32 SMOKE_DEVICE=cuda python profile_ln_poi.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

from ln_poi_kernel import (
    algo_io_bytes,
    baseline_shapes,
    launch,
    make_buffers,
    shape_for_target_io_mb,
)

N_WARMUP = int(os.environ.get("N_WARMUP", "5"))
N_ITERS = int(os.environ.get("N_ITERS", "10"))
TARGET_IO_MB = os.environ.get("TARGET_IO_MB", "").strip()


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
        print("CUDA required", file=sys.stderr)
        return 2

    if TARGET_IO_MB:
        shapes = shape_for_target_io_mb(float(TARGET_IO_MB))
        label = f"io{TARGET_IO_MB}"
    else:
        shapes = baseline_shapes()
        label = "baseline"

    x_in, mean, var_sum, gamma, beta, out = make_buffers(shapes, device, seed=0)
    io_b = algo_io_bytes(shapes.y, shapes.x)
    print(
        f"label={label} device={device} warmup={N_WARMUP} iters={N_ITERS} "
        f"Y={shapes.y} X={shapes.x} grid={shapes.grid} "
        f"XBLOCK={shapes.xblock} YBLOCK={shapes.yblock} "
        f"algo_io_mb={io_b/1e6:.3f} buf_x_mb={shapes.y*shapes.x*2/1e6:.3f}"
    )

    for _ in range(N_WARMUP):
        launch(x_in, mean, var_sum, gamma, beta, out, shapes)
    torch.cuda.synchronize()

    use_cupti = os.environ.get("CUPTI_RANGE", "1") == "1"
    if use_cupti:
        torch.cuda.cudart().cudaProfilerStart()

    times_ms: list[float] = []
    for i in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        launch(x_in, mean, var_sum, gamma, beta, out, shapes)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt)
        print(f"  iter {i:02d}: {dt:.4f} ms  out={tuple(out.shape)}")

    if use_cupti:
        torch.cuda.cudart().cudaProfilerStop()

    times_sorted = sorted(times_ms)
    mid = times_sorted[len(times_sorted) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    print(
        f"summary label={label} mean={mean_ms:.4f} ms  "
        f"median={mid:.4f} ms  min={times_sorted[0]:.4f} max={times_sorted[-1]:.4f}"
    )
    print(f"live out_ptr={out.data_ptr()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

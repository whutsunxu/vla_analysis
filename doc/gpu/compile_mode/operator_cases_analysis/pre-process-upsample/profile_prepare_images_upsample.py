#!/usr/bin/env python3
"""Nsight-ready profiling harness for Pre · prepare_images ablations.

No correctness checks. Warmup then N measured iters (default 10).

Variants (PROFILE_VARIANT):
  full         — F.interpolate 256→512 + *2-1 + cam/patch masks
  no_scale     — F.interpolate only + masks
  no_upsample  — *2-1 only (default 512², or sized via TARGET_IO_MB)

Sized no_upsample (scale-only buffer sweep):
  PROFILE_VARIANT=no_upsample TARGET_IO_MB=12|24|30|32|36
  TARGET_IO_MB = algorithmic IO for one elementwise (bytes_in + bytes_out), decimal MB.
  Each buffer ≈ TARGET_IO_MB/2; shape [1,3,H,W] f32 chosen to match.
  INCLUDE_MASKS=0 (default for TARGET_IO_MB set) skips tiny mask fills.

Usage:
  PROFILE_VARIANT=full SMOKE_DEVICE=cuda python profile_prepare_images_upsample.py
  PROFILE_VARIANT=no_upsample TARGET_IO_MB=32 SMOKE_DEVICE=cuda python profile_prepare_images_upsample.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

TARGET_HW = (512, 512)
IN_HW = (256, 256)
PATCH_MASK_HW = (32, 32)
B, C = 1, 3
N_WARMUP = int(os.environ.get("N_WARMUP", "5"))
N_ITERS = int(os.environ.get("N_ITERS", "10"))
VARIANT = os.environ.get("PROFILE_VARIANT", "full").strip().lower()
TARGET_IO_MB = os.environ.get("TARGET_IO_MB", "").strip()


def resolve_device() -> torch.device:
    req = os.environ.get("SMOKE_DEVICE", "cuda").lower()
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if req == "cuda":
            raise RuntimeError("SMOKE_DEVICE=cuda but CUDA is not available")
    return torch.device("cpu")


def shape_for_io_mb(io_mb: float) -> tuple[int, int, int, int]:
    """Pick [1,3,H,W] f32 so algo IO (in+out) ≈ io_mb decimal MB."""
    target_io = io_mb * 1e6
    buf_bytes = target_io / 2.0
    numel = max(int(buf_bytes / 4.0), C)
    hw = max(int(round(math.sqrt(numel / C))), 1)
    # Refine hw so 2 * 3 * hw * hw * 4 is closest to target_io.
    best = hw
    best_err = abs(2 * C * hw * hw * 4 - target_io)
    for cand in range(max(1, hw - 8), hw + 9):
        err = abs(2 * C * cand * cand * 4 - target_io)
        if err < best_err:
            best, best_err = cand, err
    return (B, C, best, best)


def run_once(
    img: torch.Tensor, variant: str, *, include_masks: bool
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if variant == "full":
        x = F.interpolate(img, size=TARGET_HW, mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
    elif variant == "no_scale":
        x = F.interpolate(img, size=TARGET_HW, mode="bilinear", align_corners=False)
    elif variant == "no_upsample":
        x = img * 2.0 - 1.0
    else:
        raise ValueError(f"unknown PROFILE_VARIANT={variant!r}")

    if not include_masks:
        return x, None, None
    cam_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    patch_mask = torch.ones(B, *PATCH_MASK_HW, dtype=torch.bool, device=x.device)
    return x, cam_mask, patch_mask


def main() -> int:
    if VARIANT not in ("full", "no_scale", "no_upsample"):
        print(f"bad PROFILE_VARIANT={VARIANT}", file=sys.stderr)
        return 2

    device = resolve_device()
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    sized = VARIANT == "no_upsample" and bool(TARGET_IO_MB)
    include_masks = os.environ.get(
        "INCLUDE_MASKS", "0" if sized else "1"
    ).strip() not in ("0", "false", "False")

    if sized:
        io_mb = float(TARGET_IO_MB)
        shape = shape_for_io_mb(io_mb)
        img = torch.rand(*shape, dtype=torch.float32, device=device)
        buf_mb = shape[1] * shape[2] * shape[3] * 4 / 1e6
        algo_io_mb = 2 * buf_mb
    elif VARIANT == "no_upsample":
        img = torch.rand(B, C, *TARGET_HW, dtype=torch.float32, device=device)
        buf_mb = C * TARGET_HW[0] * TARGET_HW[1] * 4 / 1e6
        algo_io_mb = 2 * buf_mb
        io_mb = None
    else:
        img = torch.rand(B, C, *IN_HW, dtype=torch.float32, device=device)
        buf_mb = algo_io_mb = None
        io_mb = None

    print(
        f"variant={VARIANT} device={device} warmup={N_WARMUP} iters={N_ITERS} "
        f"in={tuple(img.shape)} include_masks={include_masks}"
        + (
            f" target_io_mb={io_mb} buf_mb={buf_mb:.3f} algo_io_mb={algo_io_mb:.3f}"
            if buf_mb is not None
            else ""
        )
    )

    for _ in range(N_WARMUP):
        out, cam, patch = run_once(img, VARIANT, include_masks=include_masks)
    if device.type == "cuda":
        torch.cuda.synchronize()

    use_cupti_range = os.environ.get("CUPTI_RANGE", "1") == "1" and device.type == "cuda"
    if use_cupti_range:
        torch.cuda.cudart().cudaProfilerStart()

    times_ms: list[float] = []
    for i in range(N_ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, cam, patch = run_once(img, VARIANT, include_masks=include_masks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt)
        extra = ""
        if cam is not None:
            extra = f" cam={tuple(cam.shape)} patch={tuple(patch.shape)}"
        print(f"  iter {i:02d}: {dt:.4f} ms  out={tuple(out.shape)}{extra}")

    if use_cupti_range:
        torch.cuda.cudart().cudaProfilerStop()

    times_ms.sort()
    mid = times_ms[len(times_ms) // 2]
    mean = sum(times_ms) / len(times_ms)
    print(
        f"summary variant={VARIANT} mean={mean:.4f} ms  "
        f"median={mid:.4f} ms  min={times_ms[0]:.4f} max={times_ms[-1]:.4f}"
    )
    live = f"live out_ptr={out.data_ptr()}"
    if cam is not None:
        live += f" cam_ptr={cam.data_ptr()} patch_ptr={patch.data_ptr()}"
    print(live)
    return 0


if __name__ == "__main__":
    sys.exit(main())

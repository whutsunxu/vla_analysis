#!/usr/bin/env python3
"""Nsight-ready profiling harness for Pre · prepare_images ablations.

No correctness checks. Warmup then N measured iters (default 10).

Variants (PROFILE_VARIANT):
  full         — F.interpolate 256→512 + *2-1 + cam/patch masks  (case 1)
  no_scale     — F.interpolate only + masks                      (case 2: drop *2-1)
  no_upsample  — *2-1 on **512²** input + masks                  (case 3: drop interpolate;
                 input is already TARGET_HW so scale dims match full)

Usage:
  PROFILE_VARIANT=full SMOKE_DEVICE=cuda python profile_prepare_images_upsample.py
"""

from __future__ import annotations

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


def resolve_device() -> torch.device:
    req = os.environ.get("SMOKE_DEVICE", "cuda").lower()
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if req == "cuda":
            raise RuntimeError("SMOKE_DEVICE=cuda but CUDA is not available")
    return torch.device("cpu")


def run_once(img: torch.Tensor, variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variant == "full":
        x = F.interpolate(img, size=TARGET_HW, mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
    elif variant == "no_scale":
        x = F.interpolate(img, size=TARGET_HW, mode="bilinear", align_corners=False)
    elif variant == "no_upsample":
        # Scale only at 512² (same spatial size as full after upsample).
        if img.shape[-2:] != TARGET_HW:
            raise ValueError(
                f"no_upsample expects input {TARGET_HW}, got {tuple(img.shape[-2:])}"
            )
        x = img * 2.0 - 1.0
    else:
        raise ValueError(f"unknown PROFILE_VARIANT={variant!r}")

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

    # no_upsample: feed native 512² so ×2−1 matches full's post-upsample size.
    in_hw = TARGET_HW if VARIANT == "no_upsample" else IN_HW
    img = torch.rand(B, C, *in_hw, dtype=torch.float32, device=device)
    print(
        f"variant={VARIANT} device={device} warmup={N_WARMUP} iters={N_ITERS} "
        f"in={tuple(img.shape)}"
    )

    for _ in range(N_WARMUP):
        out, cam, patch = run_once(img, VARIANT)
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
        out, cam, patch = run_once(img, VARIANT)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt)
        print(
            f"  iter {i:02d}: {dt:.4f} ms  out={tuple(out.shape)} "
            f"cam={tuple(cam.shape)} patch={tuple(patch.shape)}"
        )

    if use_cupti_range:
        torch.cuda.cudart().cudaProfilerStop()

    times_ms.sort()
    mid = times_ms[len(times_ms) // 2]
    mean = sum(times_ms) / len(times_ms)
    print(
        f"summary variant={VARIANT} mean={mean:.4f} ms  "
        f"median={mid:.4f} ms  min={times_ms[0]:.4f} max={times_ms[-1]:.4f}"
    )
    # Keep last tensors live until process exit (no extra GPU math).
    print(f"live out_ptr={out.data_ptr()} cam_ptr={cam.data_ptr()} patch_ptr={patch.data_ptr()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

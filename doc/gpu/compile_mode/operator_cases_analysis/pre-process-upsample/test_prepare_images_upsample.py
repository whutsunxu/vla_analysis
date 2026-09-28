#!/usr/bin/env python3
"""Eager unit test for SmolVLA Pre · prepare_images upsample / scale / mask.

Mirrors the Pre section of:
  doc/gpu/compile_mode/inductor/SmolVLA_op_list_inductor_nsight.md

Ops (CUPTI calling order, first camera):
  1. upsample_bilinear2d  [1,3,256,256] f32 → [1,3,512,512] f32
  2. mul                  ×2
  3. add                  −1
  4. fill_bool            [1,32,32] bool  (ViT patch mask ones→bool;
                                           chronologically next to prepare_images
                                           in nsight; camera mask from prepare_images
                                           itself is [B] bool)

Source of truth in lerobot:
  SmolVLAPolicy.prepare_images  (resize_with_pad + img*2-1 + ones[B])
  lerobot.policies.common.vla_utils.resize_with_pad

Usage:
  SMOKE_DEVICE=cuda python test_prepare_images_upsample.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent
REPORT_PATH = OUT_DIR / "prepare_images_upsample_report.json"

TARGET_HW = (512, 512)  # (height, width); config stores (width, height)=(512,512)
IN_HW = (256, 256)
PATCH_MASK_HW = (32, 32)  # 512/16 ViT patches
B, C = 1, 3


def resize_with_pad(
    img: torch.Tensor, height: int, width: int, *, pad_value: float
) -> torch.Tensor:
    """Same contract as lerobot.policies.common.vla_utils.resize_with_pad."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def try_import_lerobot_resize():
    try:
        from lerobot.policies.common.vla_utils import resize_with_pad as rp

        return rp
    except Exception:
        return None


def prepare_images_cam0(
    img: torch.Tensor,
    *,
    resize_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-camera path from SmolVLAPolicy.prepare_images (no empty cams)."""
    resize_fn = resize_fn or resize_with_pad
    # config.resize_imgs_with_padding is (width, height); helper wants (h, w)
    out = resize_fn(img, TARGET_HW[0], TARGET_HW[1], pad_value=0)
    out = out * 2.0 - 1.0
    cam_mask = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
    return out, cam_mask


def vit_patch_mask(device: torch.device) -> torch.Tensor:
    """Pre table order-4: ones → bool [1,32,32] (ViT patch attention mask)."""
    return torch.ones(B, *PATCH_MASK_HW, dtype=torch.bool, device=device)


def resolve_device() -> torch.device:
    req = os.environ.get("SMOKE_DEVICE", "cuda").lower()
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if req == "cuda":
            raise RuntimeError("SMOKE_DEVICE=cuda but CUDA is not available")
    return torch.device("cpu")


def assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype {a.dtype} != {b.dtype}")
    if a.dtype == torch.bool:
        if not torch.equal(a, b):
            raise AssertionError(f"{name}: bool tensors differ")
        return
    diff = (a.float() - b.float()).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{name}: max abs diff {diff} > {atol}")


def run_reference_path(img: torch.Tensor) -> dict:
    """Reference using local resize_with_pad (always available)."""
    out, cam_mask = prepare_images_cam0(img, resize_fn=resize_with_pad)
    patch = vit_patch_mask(img.device)
    return {"image": out, "cam_mask": cam_mask, "patch_mask": patch}


def run_lerobot_path(img: torch.Tensor) -> dict | None:
    rp = try_import_lerobot_resize()
    if rp is None:
        return None
    out, cam_mask = prepare_images_cam0(img, resize_fn=rp)
    patch = vit_patch_mask(img.device)
    return {"image": out, "cam_mask": cam_mask, "patch_mask": patch}


def run_profiled(img: torch.Tensor) -> list[str]:
    """Optional CUDA profiler: collect CUDA kernel / CPU op names seen."""
    if img.device.type != "cuda":
        return []
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    names: list[str] = []
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        _ = run_reference_path(img)
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        n = ev.key
        if any(
            k in n
            for k in (
                "upsample",
                "mul",
                "add",
                "fill",
                "ones",
                "interpolate",
                "pad",
            )
        ):
            names.append(n)
    return names


def main() -> int:
    device = resolve_device()
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # Observation convention: float32 pixels in [0, 1], square 256 (no pad needed).
    img = torch.rand(B, C, *IN_HW, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ref = run_reference_path(img)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append({"name": name, "ok": ok, "detail": detail})
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    out = ref["image"]
    cam = ref["cam_mask"]
    patch = ref["patch_mask"]

    print(f"device={device}  torch={torch.__version__}")
    print(f"input  {tuple(img.shape)} {img.dtype}")
    print(f"output {tuple(out.shape)} {out.dtype}  cam_mask={tuple(cam.shape)}  patch_mask={tuple(patch.shape)}")
    print(f"wall   {elapsed_ms:.3f} ms (eager, sync'd)")

    check("out_shape", out.shape == (B, C, *TARGET_HW), str(tuple(out.shape)))
    check("out_dtype", out.dtype == torch.float32, str(out.dtype))
    check("cam_mask_shape", cam.shape == (B,), str(tuple(cam.shape)))
    check("cam_mask_all_true", bool(cam.all().item()))
    check("patch_mask_shape", patch.shape == (B, *PATCH_MASK_HW), str(tuple(patch.shape)))
    check("patch_mask_all_true", bool(patch.all().item()) and patch.dtype == torch.bool)

    # Range: input in [0,1] → after 2x−1 should be in [-1,1]
    check("out_range_lo", float(out.min()) >= -1.0 - 1e-5, f"min={float(out.min()):.6f}")
    check("out_range_hi", float(out.max()) <= 1.0 + 1e-5, f"max={float(out.max()):.6f}")

    # Direct bilinear path (square 256→512 ⇒ pad widths 0) must match resize_with_pad.
    direct = F.interpolate(img, size=TARGET_HW, mode="bilinear", align_corners=False)
    direct = direct * 2.0 - 1.0
    try:
        assert_close("direct_vs_resize_with_pad", out, direct, atol=1e-5)
        check("direct_vs_resize_with_pad", True, "atol=1e-5")
    except AssertionError as e:
        check("direct_vs_resize_with_pad", False, str(e))

    mid_in = img[0, 0, IN_HW[0] // 2, IN_HW[1] // 2]
    sample = torch.tensor([0.25, 0.5, 0.75], device=device)
    sample_out = sample * 2.0 - 1.0
    check(
        "scale_formula",
        torch.allclose(sample_out, torch.tensor([-0.5, 0.0, 0.5], device=device)),
        f"mid_in={float(mid_in):.4f}",
    )

    # Cross-check against installed lerobot helper when present.
    lerobot = run_lerobot_path(img)
    if lerobot is None:
        check("lerobot_resize_import", False, "lerobot not importable; local resize used")
    else:
        try:
            assert_close("lerobot_image", out, lerobot["image"], atol=1e-5)
            assert_close("lerobot_cam_mask", cam, lerobot["cam_mask"])
            assert_close("lerobot_patch_mask", patch, lerobot["patch_mask"])
            check("lerobot_parity", True, "resize_with_pad from lerobot matches local")
        except AssertionError as e:
            check("lerobot_parity", False, str(e))

    profile_names: list[str] = []
    if device.type == "cuda" and os.environ.get("PROFILE", "0") == "1":
        profile_names = run_profiled(img)
        print("profiler hits:", profile_names[:40])

    passed = all(c["ok"] for c in checks if c["name"] != "lerobot_resize_import")

    report = {
        "verdict": "PASS" if passed else "FAIL",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "shapes": {
            "input": list(img.shape),
            "output": list(out.shape),
            "cam_mask": list(cam.shape),
            "patch_mask": list(patch.shape),
        },
        "elapsed_ms": elapsed_ms,
        "checks": checks,
        "profile_names": profile_names,
        "source_doc": "doc/gpu/compile_mode/inductor/SmolVLA_op_list_inductor_nsight.md §Pre",
        "ops": [
            {"order": 1, "op": "upsample_bilinear2d", "in": "[1,3,256,256] f32", "out": "[1,3,512,512] f32"},
            {"order": 2, "op": "mul", "meaning": "×2"},
            {"order": 3, "op": "add", "meaning": "−1"},
            {"order": 4, "op": "fill_/ones", "out": "[1,32,32] bool", "note": "ViT patch mask (nsight Pre)"},
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nverdict={report['verdict']}  report={REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Minimal SmolVLA CUDA workload for Nsight Systems / Compute.

Warmup outside the capture range, then one Stage0-4 chunk fill between
cudaProfilerStart/Stop so nsys/ncu can isolate steady-state kernels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("SMOKE_DEVICE", "cuda")

import torch

assert torch.cuda.is_available(), "CUDA required for Nsight capture"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_test_infer import (  # noqa: E402
    MODEL_ID,
    DEVICE,
    make_dummy_frame,
    make_fixed_noise,
    profile_stage_timings,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

WARMUP = int(os.environ.get("NSIGHT_WARMUP", "3"))
REPEATS = int(os.environ.get("NSIGHT_REPEATS", "1"))  # measured iterations inside capture


def main() -> int:
    device_str = str(DEVICE)
    print(f"[nsight-target] device={device_str} warmup={WARMUP} repeats={REPEATS}", flush=True)

    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = device_str
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to(DEVICE).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": device_str}},
    )
    frame = make_dummy_frame(policy.config)
    batch = preprocess(frame)
    noise = make_fixed_noise(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        device=DEVICE,
    )

    print("[nsight-target] warmup...", flush=True)
    for _ in range(WARMUP):
        policy.reset()
        with torch.inference_mode():
            profile_stage_timings(
                policy, batch, postprocess, noise=noise.clone(), warmup=False
            )
    torch.cuda.synchronize()

    print("[nsight-target] cudaProfilerStart", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for i in range(REPEATS):
            policy.reset()
            with torch.inference_mode():
                out = profile_stage_timings(
                    policy, batch, postprocess, noise=noise.clone(), warmup=False
                )
            torch.cuda.synchronize()
            print(
                f"[nsight-target] measured[{i}] wall={out['wall_clock_total_s']:.4f}s",
                flush=True,
            )
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        print("[nsight-target] cudaProfilerStop", flush=True)

    print("[nsight-target] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""SmolVLA CUDA workload for Nsight — **torch.compile** path.

Warmup (incl. cold Inductor + CUDA-graph capture) outside the capture range,
then measure `select_action` between cudaProfilerStart/Stop.

Env:
  SMOKE_COMPILE_MODE=reduce-overhead|max-autotune|...
  NSIGHT_WARMUP=3   # outside capture (min 2 recommended after compile)
  NSIGHT_REPEATS=3  # measured select_action calls inside capture
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("SMOKE_DEVICE", "cuda")
# Force compile path for this target
os.environ["SMOKE_COMPILE"] = "1"

import torch

assert torch.cuda.is_available(), "CUDA required for Nsight capture"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_test_infer import (  # noqa: E402
    MODEL_ID,
    DEVICE,
    make_dummy_frame,
    make_fixed_noise,
    _SMOKE_COMPILE_MODE,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

WARMUP = int(os.environ.get("NSIGHT_WARMUP", "3"))
REPEATS = int(os.environ.get("NSIGHT_REPEATS", "3"))


def main() -> int:
    device_str = str(DEVICE)
    mode = _SMOKE_COMPILE_MODE
    print(
        f"[nsight-compile] device={device_str} mode={mode!r} "
        f"warmup={WARMUP} repeats={REPEATS}",
        flush=True,
    )

    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = device_str
    if not hasattr(config, "compile_model"):
        raise RuntimeError("LeRobot SmolVLAConfig missing compile_model")
    config.compile_model = True
    config.compile_mode = mode
    torch.set_float32_matmul_precision("high")

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

    def one_select() -> float:
        policy.reset()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = postprocess(policy.select_action(batch, noise=noise.clone()))
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print("[nsight-compile] warmup (pays cold compile + cudagraph)...", flush=True)
    for i in range(WARMUP):
        w = one_select()
        print(f"[nsight-compile] warmup[{i}] wall_s={w:.4f}", flush=True)

    print("[nsight-compile] cudaProfilerStart", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for i in range(REPEATS):
            w = one_select()
            print(f"[nsight-compile] measured[{i}] wall_s={w:.4f}", flush=True)
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        print("[nsight-compile] cudaProfilerStop", flush=True)

    print("[nsight-compile] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

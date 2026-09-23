#!/usr/bin/env python
"""CPU-only SmolVLA inference smoke test.

Fixed observation + fixed flow-matching noise (SMOKE_SEED) for reproducible
actions across runs. Checks shape/finiteness/magnitude and run-to-run equality.
Does not train. Does not use GPU.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import torch

assert not torch.cuda.is_available(), "CUDA unexpectedly available; this smoke test must stay CPU-only"
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
OUT_PATH = Path(__file__).resolve().parents[1] / "smoke_test_report.json"
# Fixed seed for reproducible inputs + flow-matching noise across runs.
SMOKE_SEED = 0
FIXED_TASK = "Put lego brick into the transparent box"
# Post-unnormalize action sanity bounds (SO-100-scale joint/gripper units).
ACTION_ABS_MAX = 500.0


def _now() -> float:
    return time.perf_counter()


def _summarize_tensor(x):
    if not torch.is_tensor(x):
        return {"type": type(x).__name__, "value": str(x)[:200]}
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x.min()) if x.numel() else None,
        "max": float(x.max()) if x.numel() else None,
        "mean": float(x.float().mean()) if x.numel() else None,
        "finite": bool(torch.isfinite(x.float()).all()) if x.numel() else True,
        "sample": x.detach().float().flatten()[:8].tolist(),
    }


def _make_generator(device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(SMOKE_SEED)
    return g


def make_fixed_noise(shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
    """Deterministic N(0,1) flow-matching noise (same across runs)."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
        generator=_make_generator(device),
    )


def make_dummy_frame(config) -> dict:
    """Fixed observation (not random). Same tensors every call / every process."""
    # Explicit seed so any future stochastic helper stays aligned.
    torch.manual_seed(SMOKE_SEED)
    frame = {
        "observation.state": torch.tensor(
            [0.0, 0.1, -0.1, 0.2, -0.2, 0.0], dtype=torch.float32
        ),
        "task": FIXED_TASK,
    }
    for key, feat in config.input_features.items():
        if key == "observation.state":
            continue
        c, h, w = tuple(feat.shape)
        # Deterministic [0, 1] image: channel planes + spatial ramp (no torch.rand).
        yy = torch.linspace(0.0, 1.0, h, dtype=torch.float32).view(1, h, 1)
        xx = torch.linspace(0.0, 1.0, w, dtype=torch.float32).view(1, 1, w)
        base = 0.25 + 0.5 * (0.5 * yy + 0.5 * xx)
        chans = []
        for ci in range(c):
            chans.append(torch.clamp(base + 0.05 * (ci - 1), 0.0, 1.0))
        frame[key] = torch.cat(chans, dim=0)
    return frame


def assert_action_reasonable(action: torch.Tensor, *, label: str) -> dict:
    """Shape / finiteness / magnitude checks for a postprocessed action."""
    checks = {
        "label": label,
        "shape_ok": list(action.shape) == [1, 6],
        "dtype_ok": action.dtype == torch.float32,
        "finite_ok": bool(torch.isfinite(action).all()),
        "not_all_zero": bool(action.abs().sum() > 0),
        "abs_max": float(action.abs().max()),
        "abs_max_ok": bool(action.abs().max() <= ACTION_ABS_MAX),
    }
    checks["ok"] = all(
        checks[k] for k in ("shape_ok", "dtype_ok", "finite_ok", "not_all_zero", "abs_max_ok")
    )
    if not checks["ok"]:
        raise AssertionError(f"unreasonable action ({label}): {checks} sample={_summarize_tensor(action)}")
    return checks


def try_libero_frame() -> tuple[dict | None, dict]:
    meta = {"ok": False}
    t0 = _now()
    try:
        dataset = LeRobotDataset(DATASET_ID)
        episode_index = 0
        from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
        frame = dict(dataset[from_idx])
        meta.update(
            {
                "ok": True,
                "seconds": round(_now() - t0, 2),
                "keys": sorted(str(k) for k in frame.keys()),
                "num_frames": int(len(dataset)),
            }
        )
        return frame, meta
    except Exception as exc:
        meta.update(
            {
                "ok": False,
                "seconds": round(_now() - t0, 2),
                "error_type": type(exc).__name__,
                "error": str(exc)[:800],
                "traceback": traceback.format_exc()[-1500:],
            }
        )
        return None, meta


def probe_real_world_cli() -> dict:
    """Verify Real-World Inference CLIs exist. Hardware is not present."""
    result = {
        "lerobot_record_help": None,
        "lerobot_rollout_help": None,
        "serial_ports": [],
        "video_devices": [],
        "record_dry_run": None,
        "hardware_present": False,
    }
    for name, dest in [
        ("lerobot-record", "lerobot_record_help"),
        ("lerobot-rollout", "lerobot_rollout_help"),
    ]:
        try:
            proc = subprocess.run(
                [name, "--help"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            stdout = proc.stdout or ""
            result[dest] = {
                "returncode": proc.returncode,
                "has_so100": "so100_follower" in stdout,
                "has_policy_path_or_type": ("--policy" in stdout) or ("policy.path" in stdout),
                "first_line": stdout.strip().splitlines()[0] if stdout.strip() else "",
            }
        except Exception as exc:
            result[dest] = {"error": str(exc)}

    result["serial_ports"] = sorted(
        p for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"] if os.path.exists(p)
    )
    result["video_devices"] = sorted(
        str(p) for p in Path("/dev").glob("video*") if p.exists()
    )
    result["hardware_present"] = bool(result["serial_ports"] or result["video_devices"])

    # The documented real-world command cannot connect without a robot. Capture the
    # expected hardware failure instead of hanging on serial I/O.
    try:
        proc = subprocess.run(
            [
                "lerobot-record",
                "--robot.type=so100_follower",
                "--robot.port=/dev/ttyACM1",
                "--dataset.repo_id=local/eval_so100_smoke",
                "--dataset.single_task=Put lego brick into the transparent box",
                "--dataset.push_to_hub=false",
                "--display_data=false",
            ],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        result["record_dry_run"] = {
            "returncode": proc.returncode,
            "timed_out": False,
            "output_tail": combined[-1500:],
        }
    except subprocess.TimeoutExpired as exc:
        combined = ((exc.stdout or b"") + b"\n" + (exc.stderr or b"")).decode("utf-8", "replace")
        result["record_dry_run"] = {
            "timed_out": True,
            "output_tail": combined[-1500:],
            "note": "CLI started but did not finish within 25s (likely waiting on missing robot hardware).",
        }
    except Exception as exc:
        result["record_dry_run"] = {"error": str(exc)}
    return result


def profile_stage_timings(
    policy: SmolVLAPolicy,
    batch: dict,
    postprocess,
    *,
    noise: torch.Tensor | None = None,
    warmup: bool = True,
) -> dict:
    """Wall-clock Stage 0-4 timings for one action-chunk fill on CPU.

    Stage boundaries follow SmolVLA_Operator_List.md / Architecture.md:
      Stage 0: prefix embedding (images + language + state -> P)
      Stage 1: suffix embedding (x_t + t -> U_t), summed over Euler steps
      Stage 2: VLM prefill / prefix KV cache
      Stage 3: expert denoise excluding Stage 1 rebuilds (expert forward + Euler update)
      Stage 4: crop, queue, and postprocess unnormalization of one popped action

    Stage 1 runs inside every Euler step; Stage 3 inclusive = Stage 1 + Stage 3 expert.
    """
    model = policy.model
    cfg = policy.config
    num_steps = int(cfg.num_steps)
    original_action_dim = cfg.action_feature.shape[0]

    # Match select_action batch preparation.
    batch = policy._prepare_batch(batch)
    policy.reset()

    def _run_once(record: bool) -> dict | None:
        times = {
            "stage0_prefix_embed_s": 0.0,
            "stage1_suffix_embed_s": 0.0,
            "stage2_prefill_s": 0.0,
            "stage3_expert_denoise_s": 0.0,
            "stage3_euler_update_s": 0.0,
            "stage4_crop_queue_postprocess_s": 0.0,
            "num_euler_steps": num_steps,
        }

        t_all0 = _now()

        # --- Stage 0: prepare + embed_prefix ---
        t0 = _now()
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        if record:
            times["stage0_prefix_embed_s"] = _now() - t0

        # --- Stage 2: prefill KV cache ---
        t0 = _now()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=cfg.use_cache,
        )
        if record:
            times["stage2_prefill_s"] = _now() - t0

        # --- Stage 1 + Stage 3: Euler loop with split timing ---
        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, cfg.chunk_size, cfg.max_action_dim)
        if noise is None:
            x_t = make_fixed_noise(actions_shape, device)
        else:
            x_t = noise.to(device=device, dtype=torch.float32).clone()
        dt = -1.0 / num_steps
        prefix_len = prefix_pad_masks.shape[1]

        for step in range(num_steps):
            time_value = 1.0 + step * dt
            time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(bsize)

            # Stage 1: rebuild suffix
            t0 = _now()
            suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)
            if record:
                times["stage1_suffix_embed_s"] += _now() - t0

            # Stage 3: expert denoise (masks + expert forward + velocity head + cache crop)
            t0 = _now()
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = model.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=cfg.use_cache,
            )
            if past_key_values is not None:
                past_key_values.crop(prefix_len)
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -cfg.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = model.action_out_proj(suffix_out)
            if record:
                times["stage3_expert_denoise_s"] += _now() - t0

            t0 = _now()
            x_t = x_t + dt * v_t
            if record:
                times["stage3_euler_update_s"] += _now() - t0

        actions = x_t

        # --- Stage 4: crop, optional aloha, queue, pop one, postprocess ---
        t0 = _now()
        actions = actions[:, :, :original_action_dim]
        if cfg.adapt_to_pi_aloha:
            actions = policy._pi_aloha_encode_actions(actions)
        queue = list(actions.transpose(0, 1)[: cfg.n_action_steps])
        one = queue.pop(0)
        one = postprocess(one)
        if record:
            times["stage4_crop_queue_postprocess_s"] = _now() - t0

        if not record:
            return None

        times["stage3_expert_plus_euler_s"] = (
            times["stage3_expert_denoise_s"] + times["stage3_euler_update_s"]
        )
        times["stage3_inclusive_with_stage1_s"] = (
            times["stage1_suffix_embed_s"] + times["stage3_expert_plus_euler_s"]
        )
        times["stages_0_to_4_sum_s"] = (
            times["stage0_prefix_embed_s"]
            + times["stage2_prefill_s"]
            + times["stage1_suffix_embed_s"]
            + times["stage3_expert_plus_euler_s"]
            + times["stage4_crop_queue_postprocess_s"]
        )
        times["wall_clock_total_s"] = _now() - t_all0
        times["action"] = _summarize_tensor(one)

        # Shares of measured model path (exclude queue/I/O noise in stage4 if tiny).
        denom = times["stages_0_to_4_sum_s"] or 1.0
        times["share_pct"] = {
            "stage0": round(100.0 * times["stage0_prefix_embed_s"] / denom, 2),
            "stage1": round(100.0 * times["stage1_suffix_embed_s"] / denom, 2),
            "stage2": round(100.0 * times["stage2_prefill_s"] / denom, 2),
            "stage3_expert": round(100.0 * times["stage3_expert_plus_euler_s"] / denom, 2),
            "stage4": round(100.0 * times["stage4_crop_queue_postprocess_s"] / denom, 2),
        }
        # Round for JSON readability.
        for k, v in list(times.items()):
            if isinstance(v, float):
                times[k] = round(v, 4)
        return times

    with torch.inference_mode():
        if warmup:
            _run_once(record=False)
        return _run_once(record=True)


def main() -> int:
    report: dict = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_forced": "cpu",
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "steps": {},
    }
    overall_t0 = _now()
    print("=== SmolVLA CPU inference smoke test ===", flush=True)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    print("\n[1/6] Probe real-world CLIs (no robot hardware expected)", flush=True)
    t0 = _now()
    report["real_world_cli"] = probe_real_world_cli()
    report["steps"]["cli_probe_s"] = round(_now() - t0, 2)
    print(json.dumps(report["real_world_cli"], indent=2)[:2000], flush=True)

    print("\n[2/6] Load SmolVLAPolicy.from_pretrained on CPU", flush=True)
    t0 = _now()
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = "cpu"
    print(f"config.device={config.device} type={config.type}", flush=True)
    print(f"input_features={list(config.input_features)}", flush=True)
    print(f"output_features={list(config.output_features)}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to("cpu").eval()
    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    devices = sorted({str(p.device) for p in policy.parameters()})
    dtypes = sorted({str(p.dtype) for p in policy.parameters()})
    report["model"] = {
        "load_seconds": round(_now() - t0, 2),
        "param_count": n_params,
        "trainable_param_count": n_trainable,
        "parameter_devices": devices,
        "parameter_dtypes": dtypes,
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "num_steps": getattr(config, "num_steps", None),
        "input_features": {k: list(v.shape) for k, v in config.input_features.items()},
        "output_features": {k: list(v.shape) for k, v in config.output_features.items()},
    }
    print(f"loaded params={n_params:,} devices={devices} dtypes={dtypes} in {report['model']['load_seconds']}s", flush=True)
    if any(d.startswith("cuda") for d in devices):
        raise RuntimeError(f"Model parameters landed on GPU: {devices}")

    print("\n[3/6] Build pre/post processors", flush=True)
    t0 = _now()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    report["processors"] = {"seconds": round(_now() - t0, 2)}

    print("\n[4/6] Build fixed model input (deterministic observation + noise)", flush=True)
    # Always use the fixed dummy observation for reproducibility.
    frame = make_dummy_frame(policy.config)
    source = "fixed_dummy_smolvla_base"
    report["dataset"] = {
        "ok": False,
        "skipped": True,
        "reason": "using fixed deterministic dummy observation (SMOKE_SEED); libero not required",
        "smoke_seed": SMOKE_SEED,
        "task": FIXED_TASK,
        "state": frame["observation.state"].tolist(),
        "image_keys": sorted(k for k in frame if k.startswith("observation.images")),
    }
    print(json.dumps(report["dataset"], indent=2), flush=True)

    # Fixed flow-matching noise for select_action / profile (B, chunk, max_action_dim).
    noise = make_fixed_noise(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        device="cpu",
    )

    print("\n[5/6] Run select_action twice on CPU (fixed input + noise)", flush=True)
    t0 = _now()
    batch = preprocess(frame)
    reasonableness = []
    with torch.inference_mode():
        policy.reset()
        pred1 = postprocess(policy.select_action(batch, noise=noise.clone()))
        reasonableness.append(assert_action_reasonable(pred1, label="run1"))

        policy.reset()
        pred2 = postprocess(policy.select_action(batch, noise=noise.clone()))
        reasonableness.append(assert_action_reasonable(pred2, label="run2"))

    max_abs_diff = float((pred1 - pred2).abs().max())
    identical = bool(torch.equal(pred1, pred2))
    # Allow tiny float noise if backends ever differ; CPU eager should be exact.
    deterministic_ok = identical or max_abs_diff <= 1e-6
    if not deterministic_ok:
        raise AssertionError(
            f"non-deterministic actions across two runs: max_abs_diff={max_abs_diff} "
            f"run1={pred1.tolist()} run2={pred2.tolist()}"
        )

    pred_summary = _summarize_tensor(pred1)
    report["inference"] = {
        "seconds": round(_now() - t0, 2),
        "batch_source": source,
        "smoke_seed": SMOKE_SEED,
        "fixed_noise": True,
        "action": pred_summary,
        "determinism": {
            "runs": 2,
            "identical": identical,
            "max_abs_diff": max_abs_diff,
            "ok": deterministic_ok,
        },
        "reasonableness": reasonableness,
    }
    print(json.dumps(report["inference"], indent=2), flush=True)

    print("\n[6/6] Profile Stage 0-4 wall-clock timings on CPU", flush=True)
    policy.reset()
    stage_profile = profile_stage_timings(
        policy, batch, postprocess, noise=noise.clone(), warmup=True
    )
    report["stage_profile"] = stage_profile
    print(json.dumps(stage_profile, indent=2), flush=True)

    report["total_seconds"] = round(_now() - overall_t0, 2)
    report["status"] = (
        "PASS"
        if pred_summary
        and pred_summary.get("finite")
        and deterministic_ok
        and all(r["ok"] for r in reasonableness)
        else "FAIL"
    )
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSTATUS={report['status']} total_s={report['total_seconds']}", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

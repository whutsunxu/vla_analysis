#!/usr/bin/env python
"""Detailed SmolVLA Stage 0-4 profiler (CUDA preferred).

Produces a rich JSON report with:
  - environment / GPU / driver metadata
  - model config + param counts by dtype/device
  - tensor shapes at each stage boundary
  - wall-clock timings with cuda.synchronize (warmup + N measured runs)
  - per-Euler-step Stage1/Stage3 breakdown
  - CUDA memory (allocated / reserved / peak)
  - derived rates (chunk Hz, action Hz, ms per Euler step)
  - optional CPU baseline comparison if smoke_test_report.json is present

Usage:
  SMOKE_DEVICE=cuda python src/smolvla_profile_detail.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

_SMOKE_DEVICE_REQ = os.environ.get("SMOKE_DEVICE", "cuda").strip().lower()
if _SMOKE_DEVICE_REQ == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# Reuse helpers from the smoke test.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_test_infer import (  # noqa: E402
    MODEL_ID,
    SMOKE_SEED,
    FIXED_TASK,
    make_dummy_frame,
    make_fixed_noise,
    _summarize_tensor,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "doc" / "smolVLA_profiling.json"
OUT_MD = ROOT / "doc" / "smolVLA_profiling.md"
CPU_BASELINE = ROOT / "smoke_test_report_cpu.json"
if not CPU_BASELINE.exists():
    CPU_BASELINE = ROOT / "smoke_test_report.json"

NUM_WARMUP = int(os.environ.get("PROFILE_WARMUP", "2"))
NUM_RUNS = int(os.environ.get("PROFILE_RUNS", "5"))


def _now() -> float:
    return time.perf_counter()


def _resolve_device(req: str) -> torch.device:
    if req in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("SMOKE_DEVICE=cuda but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(req)


DEVICE = _resolve_device(_SMOKE_DEVICE_REQ)


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _shape(x) -> list | None:
    return list(x.shape) if torch.is_tensor(x) else None


def _dtype(x) -> str | None:
    return str(x.dtype) if torch.is_tensor(x) else None


def _mem_snapshot() -> dict | None:
    if DEVICE.type != "cuda":
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 2),
        "free_mb": round(free_b / 1e6, 2),
        "total_mb": round(total_b / 1e6, 2),
    }


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "n": n,
        "min_s": round(s[0], 6),
        "max_s": round(s[-1], 6),
        "mean_s": round(mean, 6),
        "std_s": round(var**0.5, 6),
        "p50_s": round(s[n // 2], 6),
        "p90_s": round(s[max(0, int(0.9 * (n - 1)))], 6),
        "all_s": [round(x, 6) for x in xs],
    }


def _param_breakdown(policy) -> dict:
    by_dtype: dict[str, dict] = {}
    by_device: dict[str, dict] = {}
    total = 0
    trainable = 0
    for p in policy.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        dt = str(p.dtype)
        dv = str(p.device)
        by_dtype.setdefault(dt, {"params": 0, "bytes": 0})
        by_dtype[dt]["params"] += n
        by_dtype[dt]["bytes"] += n * p.element_size()
        by_device.setdefault(dv, {"params": 0, "bytes": 0})
        by_device[dv]["params"] += n
        by_device[dv]["bytes"] += n * p.element_size()
    for d in (by_dtype, by_device):
        for v in d.values():
            v["mb"] = round(v["bytes"] / 1e6, 2)
    return {
        "param_count": total,
        "trainable_param_count": trainable,
        "by_dtype": by_dtype,
        "by_device": by_device,
        "weight_mb_approx": round(sum(v["bytes"] for v in by_dtype.values()) / 1e6, 2),
    }


@torch.inference_mode()
def profile_once(policy, batch, postprocess, noise, *, record_shapes: bool) -> dict:
    model = policy.model
    cfg = policy.config
    num_steps = int(cfg.num_steps)
    original_action_dim = cfg.action_feature.shape[0]

    batch = policy._prepare_batch(batch)
    policy.reset()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out: dict = {
        "num_euler_steps": num_steps,
        "euler_steps": [],
        "shapes": {},
        "memory": {},
    }

    _sync()
    t_all0 = _now()
    mem_start = _mem_snapshot()

    # --- Stage 0 ---
    _sync()
    t0 = _now()
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    _sync()
    out["stage0_prefix_embed_s"] = _now() - t0
    out["memory"]["after_stage0"] = _mem_snapshot()
    if record_shapes:
        out["shapes"]["images"] = [_shape(im) for im in images]
        out["shapes"]["img_masks"] = [_shape(m) for m in img_masks]
        out["shapes"]["state"] = _shape(state)
        out["shapes"]["lang_tokens"] = _shape(lang_tokens)
        out["shapes"]["lang_masks"] = _shape(lang_masks)
        out["shapes"]["prefix_embs"] = _shape(prefix_embs)
        out["shapes"]["prefix_pad_masks"] = _shape(prefix_pad_masks)
        out["shapes"]["prefix_att_masks"] = _shape(prefix_att_masks)
        out["shapes"]["prefix_embs_dtype"] = _dtype(prefix_embs)

    # --- Stage 2 ---
    _sync()
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
    _sync()
    out["stage2_prefill_s"] = _now() - t0
    out["memory"]["after_stage2"] = _mem_snapshot()
    if record_shapes:
        out["shapes"]["prefix_att_2d_masks"] = _shape(prefix_att_2d_masks)
        out["shapes"]["prefix_position_ids"] = _shape(prefix_position_ids)
        out["shapes"]["use_cache"] = bool(cfg.use_cache)

    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, cfg.chunk_size, cfg.max_action_dim)
    x_t = noise.to(device=device, dtype=torch.float32).clone()
    dt = -1.0 / num_steps
    prefix_len = prefix_pad_masks.shape[1]

    s1_sum = 0.0
    s3_sum = 0.0
    s3e_sum = 0.0

    for step in range(num_steps):
        time_value = 1.0 + step * dt
        time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(bsize)
        step_rec: dict = {"step": step, "flow_time": time_value}

        _sync()
        t0 = _now()
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)
        _sync()
        s1 = _now() - t0
        s1_sum += s1
        step_rec["stage1_suffix_embed_s"] = s1

        _sync()
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
        _sync()
        s3 = _now() - t0
        s3_sum += s3
        step_rec["stage3_expert_denoise_s"] = s3

        _sync()
        t0 = _now()
        x_t = x_t + dt * v_t
        _sync()
        s3e = _now() - t0
        s3e_sum += s3e
        step_rec["stage3_euler_update_s"] = s3e
        step_rec["stage3_total_s"] = s3 + s3e
        step_rec["euler_step_total_s"] = s1 + s3 + s3e

        if record_shapes and step == 0:
            out["shapes"]["suffix_embs"] = _shape(suffix_embs)
            out["shapes"]["suffix_pad_masks"] = _shape(suffix_pad_masks)
            out["shapes"]["full_att_2d_masks"] = _shape(full_att_2d_masks)
            out["shapes"]["v_t"] = _shape(v_t)
            out["shapes"]["x_t"] = _shape(x_t)
            out["shapes"]["suffix_embs_dtype"] = _dtype(suffix_embs)
            out["shapes"]["v_t_dtype"] = _dtype(v_t)

        out["euler_steps"].append(step_rec)

    out["stage1_suffix_embed_s"] = s1_sum
    out["stage3_expert_denoise_s"] = s3_sum
    out["stage3_euler_update_s"] = s3e_sum
    out["stage3_expert_plus_euler_s"] = s3_sum + s3e_sum
    out["stage3_inclusive_with_stage1_s"] = s1_sum + s3_sum + s3e_sum
    out["memory"]["after_euler_loop"] = _mem_snapshot()

    # --- Stage 4 ---
    _sync()
    t0 = _now()
    actions = x_t[:, :, :original_action_dim]
    if cfg.adapt_to_pi_aloha:
        actions = policy._pi_aloha_encode_actions(actions)
    queue = list(actions.transpose(0, 1)[: cfg.n_action_steps])
    one = queue.pop(0)
    one = postprocess(one)
    _sync()
    out["stage4_crop_queue_postprocess_s"] = _now() - t0
    if record_shapes:
        out["shapes"]["action_chunk"] = _shape(actions)
        out["shapes"]["action_step"] = _shape(one)

    out["stages_0_to_4_sum_s"] = (
        out["stage0_prefix_embed_s"]
        + out["stage2_prefill_s"]
        + out["stage1_suffix_embed_s"]
        + out["stage3_expert_plus_euler_s"]
        + out["stage4_crop_queue_postprocess_s"]
    )
    _sync()
    out["wall_clock_total_s"] = _now() - t_all0
    out["action"] = _summarize_tensor(one)
    out["memory"]["start"] = mem_start
    out["memory"]["end"] = _mem_snapshot()

    denom = out["stages_0_to_4_sum_s"] or 1.0
    out["share_pct"] = {
        "stage0": round(100.0 * out["stage0_prefix_embed_s"] / denom, 2),
        "stage1": round(100.0 * out["stage1_suffix_embed_s"] / denom, 2),
        "stage2": round(100.0 * out["stage2_prefill_s"] / denom, 2),
        "stage3_expert": round(100.0 * out["stage3_expert_plus_euler_s"] / denom, 2),
        "stage4": round(100.0 * out["stage4_crop_queue_postprocess_s"] / denom, 2),
    }

    # Round floats for readability (keep euler_steps precise-ish).
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 6)
    for step_rec in out["euler_steps"]:
        for k, v in list(step_rec.items()):
            if isinstance(v, float) and k != "flow_time":
                step_rec[k] = round(v, 6)
            elif k == "flow_time":
                step_rec[k] = round(v, 6)
    return out


def _aggregate_runs(runs: list[dict]) -> dict:
    keys = [
        "stage0_prefix_embed_s",
        "stage1_suffix_embed_s",
        "stage2_prefill_s",
        "stage3_expert_denoise_s",
        "stage3_euler_update_s",
        "stage3_expert_plus_euler_s",
        "stage3_inclusive_with_stage1_s",
        "stage4_crop_queue_postprocess_s",
        "stages_0_to_4_sum_s",
        "wall_clock_total_s",
    ]
    agg = {k: _stats([r[k] for r in runs]) for k in keys}

    # Per-euler-step aggregation across runs.
    n_steps = runs[0]["num_euler_steps"]
    per_step = []
    for i in range(n_steps):
        per_step.append(
            {
                "step": i,
                "flow_time": runs[0]["euler_steps"][i]["flow_time"],
                "stage1_suffix_embed": _stats([r["euler_steps"][i]["stage1_suffix_embed_s"] for r in runs]),
                "stage3_expert_denoise": _stats([r["euler_steps"][i]["stage3_expert_denoise_s"] for r in runs]),
                "stage3_euler_update": _stats([r["euler_steps"][i]["stage3_euler_update_s"] for r in runs]),
                "euler_step_total": _stats([r["euler_steps"][i]["euler_step_total_s"] for r in runs]),
            }
        )
    agg["per_euler_step"] = per_step

    wall_mean = agg["wall_clock_total_s"]["mean_s"]
    chunk_hz = (1.0 / wall_mean) if wall_mean > 0 else None
    action_hz = (50.0 / wall_mean) if wall_mean > 0 else None  # n_action_steps=50
    agg["derived"] = {
        "chunk_fill_hz_mean": round(chunk_hz, 3) if chunk_hz else None,
        "action_step_hz_if_queue_drained_mean": round(action_hz, 3) if action_hz else None,
        "ms_per_euler_step_mean": round(
            1000.0 * agg["stage3_inclusive_with_stage1_s"]["mean_s"] / n_steps, 3
        ),
        "ms_stage0_mean": round(1000.0 * agg["stage0_prefix_embed_s"]["mean_s"], 3),
        "ms_stage2_mean": round(1000.0 * agg["stage2_prefill_s"]["mean_s"], 3),
        "ms_stage3_loop_mean": round(1000.0 * agg["stage3_inclusive_with_stage1_s"]["mean_s"], 3),
        "note": (
            "chunk_fill_hz = 1 / wall_clock for one full 50-action chunk fill "
            "(prefix+prefill+10 Euler). action_step_hz assumes the 50 queued "
            "actions are consumed one-per-control-tick with no extra model calls."
        ),
    }
    return agg


def _compare_cpu(gpu_agg: dict) -> dict | None:
    if not CPU_BASELINE.exists():
        return None
    cpu = json.loads(CPU_BASELINE.read_text())
    sp = cpu.get("stage_profile") or {}
    mapping = {
        "stage0_prefix_embed_s": "stage0_prefix_embed_s",
        "stage1_suffix_embed_s": "stage1_suffix_embed_s",
        "stage2_prefill_s": "stage2_prefill_s",
        "stage3_expert_denoise_s": "stage3_expert_denoise_s",
        "stage3_euler_update_s": "stage3_euler_update_s",
        "stage3_expert_plus_euler_s": "stage3_expert_plus_euler_s",
        "stage4_crop_queue_postprocess_s": "stage4_crop_queue_postprocess_s",
        "stages_0_to_4_sum_s": "stages_0_to_4_sum_s",
        "wall_clock_total_s": "wall_clock_total_s",
    }
    rows = []
    for k, ck in mapping.items():
        cpu_v = sp.get(ck)
        gpu_v = gpu_agg.get(k, {}).get("mean_s")
        if cpu_v is None or gpu_v is None:
            continue
        rows.append(
            {
                "metric": k,
                "cpu_s": cpu_v,
                "gpu_mean_s": gpu_v,
                "speedup": round(cpu_v / gpu_v, 2) if gpu_v > 0 else None,
                "cpu_share_of_wall_pct": round(100.0 * cpu_v / sp["wall_clock_total_s"], 2)
                if sp.get("wall_clock_total_s")
                else None,
                "gpu_share_of_wall_pct": round(100.0 * gpu_v / gpu_agg["wall_clock_total_s"]["mean_s"], 2)
                if gpu_agg["wall_clock_total_s"]["mean_s"]
                else None,
            }
        )
    ca = (cpu.get("inference") or {}).get("action", {}).get("sample")
    return {
        "cpu_baseline_file": str(CPU_BASELINE),
        "cpu_host": cpu.get("hostname"),
        "cpu_torch": cpu.get("torch"),
        "cpu_wall_s": sp.get("wall_clock_total_s"),
        "cpu_share_pct": sp.get("share_pct"),
        "rows": rows,
        "cpu_action_sample": ca,
    }


def _write_markdown(report: dict) -> str:
    env = report["environment"]
    model = report["model"]
    shapes = report["shapes"]
    agg = report["aggregate"]
    mem = report["memory_peak_across_runs"]
    cmp_ = report.get("cpu_comparison")
    der = agg["derived"]

    lines = []
    lines.append("# SmolVLA detailed profiling report")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at_utc']}`")
    lines.append(f"Host: `{env['hostname']}`")
    lines.append(f"Device: `{env['device']}` — {env.get('cuda', {}).get('device_name', 'CPU')}")
    lines.append(f"Model: `{report['model_id']}`")
    lines.append(f"Warmup runs: {report['num_warmup']} · Measured runs: {report['num_runs']}")
    lines.append(f"Smoke seed: `{report['smoke_seed']}` (fixed observation + CPU-sampled noise)")
    lines.append("")
    lines.append("## 1. Environment")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Python | {env['python']} |")
    lines.append(f"| Torch | {env['torch']} |")
    lines.append(f"| CUDA available | {env['cuda_available']} |")
    if env.get("cuda"):
        c = env["cuda"]
        lines.append(f"| GPU | {c['device_name']} |")
        lines.append(f"| Compute capability | {c['capability']} |")
        lines.append(f"| Torch CUDA | {c['torch_cuda']} |")
        lines.append(f"| VRAM total | {c['memory_gb']} GB |")
        lines.append(f"| SMs | {c['multi_processor_count']} |")
    lines.append("")
    lines.append("## 2. Model")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Params | {model['param_count']:,} |")
    lines.append(f"| Trainable | {model['trainable_param_count']:,} |")
    lines.append(f"| Weight footprint (approx) | {model['weight_mb_approx']} MB |")
    lines.append(f"| chunk_size / n_action_steps / num_steps | {model['chunk_size']} / {model['n_action_steps']} / {model['num_steps']} |")
    lines.append(f"| Load time | {model['load_seconds']} s |")
    lines.append("")
    lines.append("### Params by dtype")
    lines.append("")
    lines.append("| dtype | params | MB |")
    lines.append("|---|---:|---:|")
    for dt, v in model["by_dtype"].items():
        lines.append(f"| {dt} | {v['params']:,} | {v['mb']} |")
    lines.append("")
    lines.append("### Params by device")
    lines.append("")
    lines.append("| device | params | MB |")
    lines.append("|---|---:|---:|")
    for dv, v in model["by_device"].items():
        lines.append(f"| {dv} | {v['params']:,} | {v['mb']} |")
    lines.append("")
    lines.append("## 3. Tensor shapes (first measured run)")
    lines.append("")
    lines.append("Architecture reference dims (from Operator List): prefix `S_p=241, d_p=960`; suffix chunk `N=50`; Euler `M=10`.")
    lines.append("")
    lines.append("| Tensor | Shape / value |")
    lines.append("|---|---|")
    for k, v in shapes.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## 4. Stage wall-clock (CUDA-synchronized)")
    lines.append("")
    lines.append("Timings exclude warmup. Each stage boundary calls `torch.cuda.synchronize()`.")
    lines.append("")
    lines.append("| Stage | mean (s) | std | min | max | p50 | p90 | share % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    share = report["runs"][-1]["share_pct"]
    stage_rows = [
        ("0 prefix embed", "stage0_prefix_embed_s", "stage0"),
        ("1 suffix embed (sum×M)", "stage1_suffix_embed_s", "stage1"),
        ("2 VLM prefill", "stage2_prefill_s", "stage2"),
        ("3 expert denoise (sum×M)", "stage3_expert_denoise_s", "stage3_expert"),
        ("3 euler update (sum×M)", "stage3_euler_update_s", None),
        ("3 expert+euler", "stage3_expert_plus_euler_s", None),
        ("3 inclusive (+stage1)", "stage3_inclusive_with_stage1_s", None),
        ("4 crop/queue/post", "stage4_crop_queue_postprocess_s", "stage4"),
        ("stages 0–4 sum", "stages_0_to_4_sum_s", None),
        ("wall clock total", "wall_clock_total_s", None),
    ]
    for label, key, share_key in stage_rows:
        s = agg[key]
        sh = share.get(share_key, "") if share_key else ""
        lines.append(
            f"| {label} | {s['mean_s']:.6f} | {s['std_s']:.6f} | {s['min_s']:.6f} | "
            f"{s['max_s']:.6f} | {s['p50_s']:.6f} | {s['p90_s']:.6f} | {sh} |"
        )
    lines.append("")
    lines.append("### Derived rates")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Chunk-fill Hz (mean) | {der['chunk_fill_hz_mean']} |")
    lines.append(f"| Action-step Hz if queue drained (mean) | {der['action_step_hz_if_queue_drained_mean']} |")
    lines.append(f"| ms / Euler step (stage1+3 mean) | {der['ms_per_euler_step_mean']} |")
    lines.append(f"| ms Stage0 mean | {der['ms_stage0_mean']} |")
    lines.append(f"| ms Stage2 mean | {der['ms_stage2_mean']} |")
    lines.append(f"| ms Stage3 loop mean | {der['ms_stage3_loop_mean']} |")
    lines.append("")
    lines.append(f"_{der['note']}_")
    lines.append("")
    lines.append("## 5. Per-Euler-step breakdown (mean over measured runs)")
    lines.append("")
    lines.append("| step | flow_t | stage1 mean (ms) | stage3 expert mean (ms) | euler upd mean (ms) | step total mean (ms) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in agg["per_euler_step"]:
        lines.append(
            f"| {row['step']} | {row['flow_time']} | "
            f"{1000*row['stage1_suffix_embed']['mean_s']:.3f} | "
            f"{1000*row['stage3_expert_denoise']['mean_s']:.3f} | "
            f"{1000*row['stage3_euler_update']['mean_s']:.3f} | "
            f"{1000*row['euler_step_total']['mean_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 6. CUDA memory")
    lines.append("")
    if mem:
        lines.append("| Snapshot | allocated MB | reserved MB | max_allocated MB | max_reserved MB | free MB |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, snap in mem.items():
            if not isinstance(snap, dict):
                lines.append(f"| {name} | {snap} |  |  |  |  |")
                continue
            lines.append(
                f"| {name} | {snap.get('allocated_mb')} | {snap.get('reserved_mb')} | "
                f"{snap.get('max_allocated_mb')} | {snap.get('max_reserved_mb')} | {snap.get('free_mb')} |"
            )
    else:
        lines.append("_CPU run — no CUDA memory stats._")
    lines.append("")
    lines.append("## 7. Action output (last measured run)")
    lines.append("")
    act = report["runs"][-1]["action"]
    lines.append("```json")
    lines.append(json.dumps(act, indent=2))
    lines.append("```")
    lines.append("")
    if cmp_:
        lines.append("## 8. CPU vs GPU comparison")
        lines.append("")
        lines.append(f"CPU baseline: `{cmp_['cpu_baseline_file']}` ({cmp_['cpu_host']}, torch `{cmp_['cpu_torch']}`)")
        lines.append("")
        lines.append("| Metric | CPU (s) | GPU mean (s) | Speedup | CPU share % | GPU share % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in cmp_["rows"]:
            lines.append(
                f"| {r['metric']} | {r['cpu_s']} | {r['gpu_mean_s']} | {r['speedup']}× | "
                f"{r['cpu_share_of_wall_pct']} | {r['gpu_share_of_wall_pct']} |"
            )
        lines.append("")
        if cmp_.get("cpu_action_sample") and act.get("sample"):
            diffs = [abs(a - b) for a, b in zip(cmp_["cpu_action_sample"], act["sample"])]
            lines.append("### Action parity (CPU vs last GPU run)")
            lines.append("")
            lines.append(f"- max abs diff: `{max(diffs):.6g}`")
            lines.append(f"- mean abs diff: `{sum(diffs)/len(diffs):.6g}`")
            lines.append(f"- within 1e-2: `{max(diffs) <= 1e-2}`")
            lines.append(f"- within 1e-1: `{max(diffs) <= 1e-1}`")
            lines.append("")
            lines.append("| dim | CPU | GPU | abs diff |")
            lines.append("|---:|---:|---:|---:|")
            for i, (a, b, d) in enumerate(zip(cmp_["cpu_action_sample"], act["sample"], diffs)):
                lines.append(f"| {i} | {a} | {b} | {d:.6g} |")
            lines.append("")
    lines.append("## 9. Method notes")
    lines.append("")
    lines.append("- Profiler mirrors `select_action` chunk-fill path (Stages 0→2→Euler{1,3}×M→4).")
    lines.append("- Noise is sampled on CPU with `SMOKE_SEED` then moved to device (bit-identical across CPU/GPU).")
    lines.append("- Observation is the fixed dummy frame (not libero).")
    lines.append("- Stage 3 expert timing includes mask build + expert forward + `action_out_proj` + KV crop.")
    lines.append("- First warmup run pays CUDA kernel compile / cudnn autotune cost; excluded from aggregates.")
    lines.append("")
    lines.append(f"Machine-readable twin: `{OUT_JSON.name}`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import datetime as _dt

    device_str = str(DEVICE)
    print(f"=== SmolVLA detailed profiler on {device_str} ===", flush=True)
    print(f"warmup={NUM_WARMUP} runs={NUM_RUNS}", flush=True)

    report: dict = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "smoke_seed": SMOKE_SEED,
        "fixed_task": FIXED_TASK,
        "num_warmup": NUM_WARMUP,
        "num_runs": NUM_RUNS,
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "smoke_device_req": _SMOKE_DEVICE_REQ,
            "device": device_str,
        },
    }
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        report["environment"]["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda,
            "memory_gb": round(props.total_memory / 1e9, 2),
            "multi_processor_count": props.multi_processor_count,
        }

    t0 = _now()
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = device_str
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to(DEVICE).eval()
    load_s = round(_now() - t0, 3)
    breakdown = _param_breakdown(policy)
    report["model"] = {
        **breakdown,
        "load_seconds": load_s,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "num_steps": config.num_steps,
        "max_action_dim": config.max_action_dim,
        "input_features": {k: list(v.shape) for k, v in config.input_features.items()},
        "output_features": {k: list(v.shape) for k, v in config.output_features.items()},
    }
    print(f"loaded in {load_s}s params={breakdown['param_count']:,}", flush=True)

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

    print(f"warmup ×{NUM_WARMUP} ...", flush=True)
    for _ in range(NUM_WARMUP):
        profile_once(policy, batch, postprocess, noise, record_shapes=False)

    runs = []
    print(f"measure ×{NUM_RUNS} ...", flush=True)
    for i in range(NUM_RUNS):
        rec = profile_once(policy, batch, postprocess, noise, record_shapes=(i == 0))
        print(
            f"  run{i}: wall={rec['wall_clock_total_s']:.4f}s "
            f"s0={rec['stage0_prefix_embed_s']:.4f} s2={rec['stage2_prefill_s']:.4f} "
            f"s3incl={rec['stage3_inclusive_with_stage1_s']:.4f}",
            flush=True,
        )
        runs.append(rec)

    report["shapes"] = runs[0].get("shapes", {})
    report["runs"] = runs
    report["aggregate"] = _aggregate_runs(runs)

    # Peak memory across runs (end snapshots).
    mem_peak = {
        "after_stage0": runs[0]["memory"].get("after_stage0"),
        "after_stage2": runs[0]["memory"].get("after_stage2"),
        "after_euler_loop": runs[0]["memory"].get("after_euler_loop"),
        "end_last_run": runs[-1]["memory"].get("end"),
        "max_allocated_mb_across_runs": max(
            (r["memory"].get("end") or {}).get("max_allocated_mb") or 0 for r in runs
        ),
        "max_reserved_mb_across_runs": max(
            (r["memory"].get("end") or {}).get("max_reserved_mb") or 0 for r in runs
        ),
    }
    report["memory_peak_across_runs"] = mem_peak
    report["cpu_comparison"] = _compare_cpu(report["aggregate"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    md = _write_markdown(report)
    OUT_MD.write_text(md)
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)
    print(
        f"mean wall={report['aggregate']['wall_clock_total_s']['mean_s']:.4f}s "
        f"chunk_Hz={report['aggregate']['derived']['chunk_fill_hz_mean']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

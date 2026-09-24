#!/usr/bin/env python
"""Record SmolVLA Stage 0–4 ATen ops in chronological call order.

Uses TorchDispatchMode (aten dispatcher) with an append-only event list —
NOT aggregated. Each event: name, input shapes, input dtypes.

Usage:
  source /venv/main/bin/activate
  export HF_HOME=/workspace/.hf_home SMOKE_DEVICE=cuda
  python src/smolvla_aten_profile.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

_SMOKE_DEVICE_REQ = os.environ.get("SMOKE_DEVICE", "cuda").strip().lower()
if _SMOKE_DEVICE_REQ == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_test_infer import (  # noqa: E402
    DEVICE,
    MODEL_ID,
    SMOKE_SEED,
    _sync,
    make_dummy_frame,
    make_fixed_noise,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "gpu"
OUT_JSON = OUT_DIR / "smolvla_aten_chrono.json"
OUT_LOG = OUT_DIR / "smolvla_aten_chrono.log"

# Skip pure factory / meta ops that drown the log (still chronological for real compute).
SKIP_SUBSTRINGS = (
    "empty.memory_format",
    "empty_strided",
    "new_empty",
    "new_empty_strided",
    "alias",
    "detach",
)


def _tensor_meta(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
    }


def _arg_metas(args, kwargs) -> tuple[list, list]:
    shapes, dtypes = [], []

    def add_t(t: torch.Tensor):
        shapes.append(list(t.shape))
        dtypes.append(str(t.dtype).replace("torch.", ""))

    def walk(obj):
        if isinstance(obj, torch.Tensor):
            add_t(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                walk(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                walk(x)

    for a in args:
        walk(a)
    for v in (kwargs or {}).values():
        walk(v)
    return shapes, dtypes


class ChronoAtenDispatch(TorchDispatchMode):
    """Append every ATen dispatcher call in chronological order (no aggregation)."""

    def __init__(self, *, skip_noise: bool = True):
        super().__init__()
        self.events: list[dict] = []
        self.skip_noise = skip_noise

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(func)
        # Normalize:aten.xxx.default -> aten::xxx
        if name.startswith("aten."):
            short = name[len("aten.") :].split(".")[0]
            full = f"aten::{short}"
        else:
            full = name

        shapes, dtypes = _arg_metas(args, kwargs)
        skip = self.skip_noise and any(s in name for s in SKIP_SUBSTRINGS)
        if not skip:
            self.events.append(
                {
                    "i": len(self.events) + 1,
                    "name": full,
                    "op_overload": name,
                    "input_shapes": shapes,
                    "input_dtypes": dtypes,
                }
            )
        return func(*args, **kwargs)


def _write_stage_log(fh, label: str, ops: list[dict]) -> None:
    fh.write(f"\n{'=' * 72}\n")
    fh.write(f"{label}  —  {len(ops)} aten launches (chronological)\n")
    fh.write(f"{'=' * 72}\n")
    fh.write(f"{'#':>6}  {'name':<36}  shapes  dtypes\n")
    for e in ops:
        sh = json.dumps(e["input_shapes"], separators=(",", ":"))
        dt = json.dumps(e["input_dtypes"], separators=(",", ":"))
        fh.write(f"{e['i']:6d}  {e['name']:<36}  {sh}  {dt}\n")
    print(f"{label}: {len(ops)} chronological aten events", flush=True)


def _record_callable(fn, *, label: str) -> list[dict]:
    rec = ChronoAtenDispatch(skip_noise=True)
    _sync()
    with rec:
        fn()
        _sync()
    # renumber after skip filter already applied in-mode
    for i, e in enumerate(rec.events, 1):
        e["i"] = i
    return rec.events


@torch.inference_mode()
def main() -> int:
    print(f"Loading {MODEL_ID} on {DEVICE}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(DEVICE)
    policy.eval()
    cfg = policy.config
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    frame = make_dummy_frame(cfg)
    batch = preprocess(frame)
    noise = make_fixed_noise((1, cfg.chunk_size, cfg.max_action_dim), DEVICE)
    model = policy.model
    num_steps = int(cfg.num_steps)

    batch = policy._prepare_batch(batch)
    policy.reset()

    print("Warmup…", flush=True)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    pam = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=pam,
        position_ids=pids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=cfg.use_cache,
    )
    bsize = state.shape[0]
    device = state.device
    x_t = noise.to(device=device, dtype=torch.float32).clone()
    dt = -1.0 / num_steps
    prefix_len = prefix_pad_masks.shape[1]
    time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)
    sl = suffix_pad_masks.shape[1]
    pp = prefix_pad_masks[:, None, :].expand(bsize, sl, prefix_len)
    sa = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full = torch.cat([pp, sa], dim=2)
    offs = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = offs + torch.cumsum(suffix_pad_masks, dim=1) - 1
    outputs_embeds, _ = model.vlm_with_expert.forward(
        attention_mask=full,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=cfg.use_cache,
    )
    if past_key_values is not None:
        past_key_values.crop(prefix_len)
    _sync()

    policy.reset()
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    result: dict = {
        "meta": {
            "model_id": MODEL_ID,
            "device": str(DEVICE),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "smoke_seed": SMOKE_SEED,
            "num_euler_steps": num_steps,
            "chunk_size": int(cfg.chunk_size),
            "method": (
                "TorchDispatchMode chronological append-only list "
                "(not aggregated; one row per aten dispatch). "
                "Skips empty/empty_strided/alias/detach noise. "
                "Stage 1/3 = one Euler step (×M per chunk)."
            ),
        },
        "stages": {},
    }
    log_chunks: list[tuple[str, list[dict]]] = []

    def stage0():
        nonlocal prefix_embs, prefix_pad_masks, prefix_att_masks
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

    ops0 = _record_callable(stage0, label="Stage 0")
    result["stages"]["0_prefix_embed"] = {
        "title": "Stage 0 — prefix embedding",
        "repeat_note": "1× per chunk",
        "ops": ops0,
        "tensors": {
            "images": [_tensor_meta(im) for im in images],
            "state": _tensor_meta(state),
            "lang_tokens": _tensor_meta(lang_tokens),
            "prefix_embs": _tensor_meta(prefix_embs),
            "prefix_pad_masks": _tensor_meta(prefix_pad_masks),
        },
    }
    log_chunks.append(("Stage 0 — prefix embedding", ops0))

    def stage2():
        nonlocal past_key_values
        pam = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=pam,
            position_ids=pids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=cfg.use_cache,
        )

    ops2 = _record_callable(stage2, label="Stage 2")
    result["stages"]["2_prefill"] = {
        "title": "Stage 2 — prefix prefill / KV cache",
        "repeat_note": "1× per chunk (L_p=16 layers inside)",
        "ops": ops2,
        "tensors": {
            "prefix_embs": _tensor_meta(prefix_embs),
            "prefix_pad_masks": _tensor_meta(prefix_pad_masks),
        },
    }
    log_chunks.append(("Stage 2 — prefix prefill / KV cache", ops2))

    x_t = noise.to(device=device, dtype=torch.float32).clone()
    time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)

    def stage1():
        nonlocal suffix_embs, suffix_pad_masks, suffix_att_masks
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)

    ops1 = _record_callable(stage1, label="Stage 1")
    result["stages"]["1_suffix_embed"] = {
        "title": "Stage 1 — action-suffix embedding (one Euler step)",
        "repeat_note": f"×{num_steps} Euler steps per chunk",
        "ops": ops1,
        "tensors": {
            "x_t": _tensor_meta(x_t),
            "time_tensor": _tensor_meta(time_tensor),
            "suffix_embs": _tensor_meta(suffix_embs),
        },
    }
    log_chunks.append(("Stage 1 — suffix embed (1 Euler step)", ops1))

    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)

    def stage3():
        nonlocal past_key_values, x_t
        sl = suffix_pad_masks.shape[1]
        pp = prefix_pad_masks[:, None, :].expand(bsize, sl, prefix_len)
        sa = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full = torch.cat([pp, sa], dim=2)
        offs = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        pids = offs + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full,
            position_ids=pids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=cfg.use_cache,
        )
        if past_key_values is not None:
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1][:, -cfg.chunk_size :].to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        x_t = x_t + dt * v_t

    ops3 = _record_callable(stage3, label="Stage 3")
    result["stages"]["3_expert_euler"] = {
        "title": "Stage 3 — expert decode + Euler update (one Euler step)",
        "repeat_note": f"×{num_steps} Euler steps per chunk",
        "ops": ops3,
        "tensors": {
            "suffix_embs": _tensor_meta(suffix_embs),
            "x_t_after": _tensor_meta(x_t),
        },
    }
    log_chunks.append(("Stage 3 — expert + Euler (1 step)", ops3))

    for step in range(1, num_steps):
        time_value = 1.0 + step * dt
        time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(bsize)
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)
        sl = suffix_pad_masks.shape[1]
        pp = prefix_pad_masks[:, None, :].expand(bsize, sl, prefix_len)
        sa = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full = torch.cat([pp, sa], dim=2)
        offs = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        pids = offs + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full,
            position_ids=pids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=cfg.use_cache,
        )
        if past_key_values is not None:
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1][:, -cfg.chunk_size :].to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        x_t = x_t + dt * v_t

    original_action_dim = cfg.action_feature.shape[0]
    actions_chunk = x_t

    def stage4():
        actions = actions_chunk[:, :, :original_action_dim]
        if cfg.adapt_to_pi_aloha:
            actions = policy._pi_aloha_encode_actions(actions)
        queue = list(actions.transpose(0, 1)[: cfg.n_action_steps])
        one = queue.pop(0)
        _ = postprocess(one)

    ops4 = _record_callable(stage4, label="Stage 4")
    result["stages"]["4_crop_queue_post"] = {
        "title": "Stage 4 — crop / queue / postprocess",
        "repeat_note": "1× pop per select_action after chunk fill",
        "ops": ops4,
        "tensors": {"actions_chunk": _tensor_meta(actions_chunk)},
    }
    log_chunks.append(("Stage 4 — crop / queue / postprocess", ops4))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    with OUT_LOG.open("w") as fh:
        fh.write("SmolVLA ATen chronological op log (TorchDispatchMode)\n")
        fh.write(json.dumps(result["meta"], indent=2) + "\n")
        for label, ops in log_chunks:
            _write_stage_log(fh, label, ops)

    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_LOG}", flush=True)
    for k, st in result["stages"].items():
        print(f"  {k}: {len(st['ops'])} events", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise

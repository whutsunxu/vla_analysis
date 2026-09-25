#!/usr/bin/env python
"""Dump torch.compile optimized FX / AOT graphs for SmolVLA sample_actions.

Captures the post-Dynamo / post-AOT FX graphs that Inductor compiles (aten-level
ops after decompositions), analogous to the eager TorchDispatchMode chrono list
but for the *compiled* path.

Usage (remote GPU):
  source /venv/main/bin/activate
  export HF_HOME=/workspace/.hf_home SMOKE_DEVICE=cuda
  export SMOKE_COMPILE_MODE=reduce-overhead   # optional
  python src/smolvla_compile_graph_dump.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("SMOKE_DEVICE", "cuda")

import torch
import torch._dynamo as dynamo
from torch._inductor import compile_fx as compile_fx_mod

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_test_infer import (  # noqa: E402
    DEVICE,
    MODEL_ID,
    SMOKE_SEED,
    _sync,
    make_dummy_frame,
    make_fixed_noise,
)

_COMPILE_MODE = os.environ.get("SMOKE_COMPILE_MODE", "reduce-overhead").strip()
OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "gpu" / "compile_mode"
OUT_JSON = OUT_DIR / "smolvla_compile_fx_graphs.json"
OUT_LOG = OUT_DIR / "smolvla_compile_fx_graphs.log"
OUT_MD = OUT_DIR / "SmolVLA_CompileOp_List_gpu_backend.md"


def _symint_to_py(x):
    try:
        return int(x)
    except Exception:
        return str(x)


def _shape_dtype_from_meta(val) -> tuple:
    """Return (shape, dtype) from FakeTensor / Tensor / TensorMetadata / nested."""
    if val is None:
        return None, None
    if isinstance(val, torch.Tensor):
        try:
            return [_symint_to_py(s) for s in val.shape], str(val.dtype).replace("torch.", "")
        except Exception:
            return list(val.size()) if hasattr(val, "size") else None, str(getattr(val, "dtype", ""))
    # torch.fx.passes.shape_prop.TensorMetadata (namedtuple-like)
    if type(val).__name__ == "TensorMetadata" or (
        hasattr(val, "shape") and hasattr(val, "dtype") and not callable(val)
    ):
        try:
            shape = getattr(val, "shape", None)
            dtype = getattr(val, "dtype", None)
            if shape is not None:
                shape = [_symint_to_py(s) for s in shape]
            dtype_s = str(dtype).replace("torch.", "") if dtype is not None else None
            return shape, dtype_s
        except Exception:
            pass
    if isinstance(val, (list, tuple)):
        shapes, dtypes = [], []
        for x in val:
            s, d = _shape_dtype_from_meta(x)
            if s is not None:
                shapes.append(s)
            if d is not None:
                dtypes.append(d)
        if not shapes:
            return None, None
        # single tensor wrapped
        if len(shapes) == 1 and len(dtypes) <= 1:
            return shapes[0], (dtypes[0] if dtypes else None)
        return shapes, dtypes if dtypes else None
    return None, None


def _node_meta_val(n) -> object:
    if not hasattr(n, "meta") or not isinstance(n.meta, dict):
        return None
    for key in ("val", "tensor_meta", "example_value"):
        if key in n.meta and n.meta[key] is not None:
            return n.meta[key]
    return None


def _normalize_target(target) -> str:
    if target is None:
        return ""
    name = str(target)
    if name.startswith("aten."):
        short = name[len("aten.") :].split(".")[0]
        return f"aten::{short}"
    import re

    m = re.search(r"built-in (?:function|method) (\w+)", name)
    if m:
        return f"aten::{m.group(1)}"
    m = re.search(r"<function (\w+)", name)
    if m:
        return f"aten::{m.group(1)}"
    if "aten." in name:
        part = name.split("aten.")[-1].split(".")[0]
        return f"aten::{part}"
    return name


def _fmt_pair(shapes, dtypes) -> str:
    if shapes is None:
        return "—"
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], int):
        dt = dtypes if isinstance(dtypes, str) else (dtypes[0] if isinstance(dtypes, list) and dtypes else "")
        return f"{shapes} {dt}".strip()
    # list of shapes
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], list):
        parts = []
        dts = dtypes if isinstance(dtypes, list) else [dtypes] * len(shapes)
        for i, sh in enumerate(shapes):
            d = dts[i] if i < len(dts) else ""
            parts.append(f"{sh} {d}".strip())
        return " | ".join(parts)
    return f"{shapes} {dtypes}".strip()


def _collect_arg_shapes(a, arg_shapes: list, arg_dtypes: list) -> None:
    """Walk FX Node / Tensor / nested list|tuple args (e.g. aten.cat([a,b]))."""
    if isinstance(a, torch.fx.Node):
        s, d = _shape_dtype_from_meta(_node_meta_val(a))
        if s is None:
            return
        if isinstance(s, list) and s and isinstance(s[0], list):
            arg_shapes.extend(s)
            if isinstance(d, list):
                arg_dtypes.extend(d)
            elif d:
                arg_dtypes.append(d)
        else:
            arg_shapes.append(s)
            if d:
                arg_dtypes.append(d if isinstance(d, str) else d)
    elif isinstance(a, torch.Tensor):
        arg_shapes.append(list(a.shape))
        arg_dtypes.append(str(a.dtype).replace("torch.", ""))
    elif isinstance(a, (list, tuple)):
        for x in a:
            _collect_arg_shapes(x, arg_shapes, arg_dtypes)


def _node_record(n, idx: int) -> dict:
    meta_val = _node_meta_val(n)
    shapes, dtypes = _shape_dtype_from_meta(meta_val)

    arg_shapes, arg_dtypes = [], []
    for a in n.args:
        _collect_arg_shapes(a, arg_shapes, arg_dtypes)

    # kwargs tensor nodes (including nested)
    for v in (n.kwargs or {}).values():
        _collect_arg_shapes(v, arg_shapes, arg_dtypes)

    name = _normalize_target(n.target) if n.op in ("call_function", "call_method") else n.op
    # In-place / setitem often leave no example_value; treat mutated tensor as output.
    if shapes is None and arg_shapes and name in ("aten::setitem", "aten::copy_", "aten::iadd", "aten::imul"):
        shapes = arg_shapes[0]
        dtypes = arg_dtypes[0] if arg_dtypes else None

    return {
        "i": idx,
        "fx_op": n.op,
        "name": name,
        "target": str(n.target) if n.target is not None else n.op,
        "node_name": n.name,
        "args_repr": [repr(a)[:120] for a in n.args[:8]],
        "kwargs_keys": list((n.kwargs or {}).keys()),
        "meta_keys": sorted(n.meta.keys()) if hasattr(n, "meta") and isinstance(n.meta, dict) else [],
        "output_shapes": shapes,
        "output_dtypes": dtypes,
        "input_shapes": arg_shapes or None,
        "input_dtypes": arg_dtypes or None,
        "input_str": _fmt_pair(arg_shapes or None, arg_dtypes or None),
        "output_str": _fmt_pair(shapes, dtypes),
    }


def _enrich_ops_from_node_names(graph: dict) -> None:
    """Fill missing cat inputs / setitem outputs using sibling node metas (offline repair)."""
    import re

    by_name = {}
    for o in graph.get("ops") or []:
        by_name[o["node_name"]] = o
    for n in graph.get("io_nodes") or []:
        by_name[n["node_name"]] = n

    for o in graph.get("ops") or []:
        if not o.get("input_shapes") and o.get("name") == "aten::cat" and o.get("args_repr"):
            ar = o["args_repr"][0]
            names = re.findall(r"[A-Za-z_][\w]*", ar)
            shapes, dts = [], []
            ok = True
            for name in names:
                if name in ("Ellipsis", "None", "True", "False"):
                    continue
                src = by_name.get(name)
                if not src or src.get("output_shapes") is None:
                    ok = False
                    break
                sh, d = src["output_shapes"], src.get("output_dtypes")
                if isinstance(sh, list) and sh and isinstance(sh[0], list):
                    shapes.extend(sh)
                    if isinstance(d, list):
                        dts.extend(d)
                    elif d:
                        dts.append(d)
                else:
                    shapes.append(sh)
                    if d:
                        dts.append(d)
            if ok and shapes:
                o["input_shapes"] = shapes
                o["input_dtypes"] = dts or None
                o["input_str"] = _fmt_pair(shapes, dts or None)

        if o.get("output_shapes") is None and o.get("input_shapes"):
            if o.get("name") in ("aten::setitem", "aten::copy_", "aten::iadd", "aten::imul"):
                o["output_shapes"] = o["input_shapes"][0]
                dts = o.get("input_dtypes") or []
                o["output_dtypes"] = dts[0] if dts else None
                o["output_str"] = _fmt_pair(o["output_shapes"], o["output_dtypes"])

    ops = graph.get("ops") or []
    graph["shape_meta_coverage"] = {
        "ops_with_output": sum(1 for e in ops if e.get("output_shapes") is not None),
        "ops_with_input": sum(1 for e in ops if e.get("input_shapes")),
    }


def _propagate_shapes(gm: torch.fx.GraphModule, example_inputs) -> None:
    """Fill node.meta['val'] when missing so I/O dims are available."""
    # Already populated?
    compute = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module")]
    if compute and _node_meta_val(compute[0]) is not None:
        return
    try:
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        FakeTensorProp(gm).propagate(*example_inputs)
        return
    except Exception as exc:
        print(f"[capture] FakeTensorProp failed: {exc}", flush=True)
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(gm).propagate(*example_inputs)
    except Exception as exc:
        print(f"[capture] ShapeProp failed: {exc}", flush=True)


class GraphCaptureBackend:
    """Record each FX graph Inductor receives, then compile normally."""

    def __init__(self):
        self.graphs: list[dict] = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs):
        # Ensure FakeTensor / shape meta exists before reading nodes (call order = graph order).
        _propagate_shapes(gm, example_inputs)

        nodes = []
        compute_i = 0
        # Iterate gm.graph.nodes in topological / calling order
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                rec = _node_record(n, 0)
                rec["i"] = f"in:{n.name}"
                rec["name"] = "placeholder"
                nodes.append(rec)
                continue
            if n.op == "output":
                rec = _node_record(n, 0)
                rec["i"] = "out"
                rec["name"] = "output"
                nodes.append(rec)
                continue
            if n.op == "get_attr":
                continue
            compute_i += 1
            nodes.append(_node_record(n, compute_i))

        try:
            readable = gm.print_readable(print_output=False)
        except Exception as exc:
            readable = f"<print_readable failed: {exc}>"

        ops = [n for n in nodes if isinstance(n["i"], int)]
        # sanity: seq must be 1..N in order (FX calling / topo order)
        for i, e in enumerate(ops, 1):
            e["i"] = i

        gdict = {
            "graph_id": len(self.graphs) + 1,
            "num_fx_nodes": len(list(gm.graph.nodes)),
            "num_compute_ops": len(ops),
            "backend_kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
            "ops": ops,  # chronological calling order
            "io_nodes": [n for n in nodes if not isinstance(n["i"], int)],
            "readable_head": readable[:8000],
            "example_input_metas": [
                {
                    "shape": list(t.shape) if torch.is_tensor(t) else None,
                    "dtype": str(t.dtype).replace("torch.", "") if torch.is_tensor(t) else type(t).__name__,
                }
                for t in example_inputs
            ],
        }
        _enrich_ops_from_node_names(gdict)
        self.graphs.append(gdict)
        cov = self.graphs[-1]["shape_meta_coverage"]
        print(
            f"[capture] graph #{self.graphs[-1]['graph_id']}: "
            f"{len(ops)} compute ops, {len(list(gm.graph.nodes))} fx nodes "
            f"out_meta={cov['ops_with_output']}/{len(ops)} in_meta={cov['ops_with_input']}/{len(ops)}",
            flush=True,
        )
        fx_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        return compile_fx_mod.compile_fx(gm, example_inputs, **fx_kwargs)


def _op_histogram(ops: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for e in ops:
        hist[e["name"]] = hist.get(e["name"], 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))


# Explicit meanings for non-parameter Dynamo placeholders (activations / graph I/O).
_PLACEHOLDER_MEANING = {
    "l_image_": "camera image (pre-ViT / float input)",
    "l_pixel_values_": "ViT pixel values (bf16 image)",
    "l_patch_attention_mask_": "ViT patch attention mask",
    "l_position_ids_": "ViT position ids",
    "l_embeddings_": "ViT patch embeddings (+ pos)",
    "l_tokens_": "language token ids",
    "l_noise_": "action diffusion noise",
    "l_stack0_": "Dynamo intermediate (prior-graph activation)",
    "l_stack0_0_": "Dynamo intermediate #0 (prior-graph activation)",
    "l_stack0_1_": "Dynamo intermediate #1 (prior-graph activation)",
    "l_stack0_2_": "Dynamo intermediate #2 (prior-graph activation)",
    "l_stack0_last_hidden_state": "ViT last_hidden_state (into connector)",
}

# Short role label per Dynamo FX graph (sample_actions partition).
_GRAPH_MEANING = {
    1: (
        "Cast camera image f32 → bf16 (ViT dtype).",
        "`image` `[1,3,512,512] f32` → `[…] bf16`",
    ),
    2: (
        "Build full ViT patch attention mask (`ones` → bool).",
        "`ones` → mask `[1,32,32] bool`",
    ),
    3: (
        "ViT patch embed (`conv2d` 16×16) + mask-derived position-id bookkeeping.",
        "`pixel_values` + patch weight/bias + mask → patch tokens `[1,1024,768]`, "
        "`position_ids`, flat mask",
    ),
    4: (
        "Add learned position embedding to patch tokens.",
        "`embeddings` + `position_embedding.weight` → `embeddings` `[1,1024,768] bf16`",
    ),
    5: (
        "ViT encoder stack (12× LayerNorm / SDPA / MLP) → last_hidden_state.",
        "pos-aware embeddings + mask + layer params → "
        "`last_hidden_state` `[1,1024,768] bf16`",
    ),
    6: (
        "Vision→language connector (spatial pack / PixelShuffle-style reshape + Linear).",
        "`last_hidden_state` → `image_hidden_states` `[1,64,960] bf16`",
    ),
    7: (
        "Language token embedding lookup.",
        "`tokens` `[1,48]` + `embed_tokens.weight` → `embedding` `[1,48,960] bf16`",
    ),
    8: (
        "Prefix + VLM/expert flow-matching (attn / RoPE / Euler) → action chunk.",
        "prefix acts + masks + weights → `x_t` / actions `[1,50,32] f32`",
    ),
}


def _placeholder_meaning(ph_id: str) -> str:
    """Human-readable role for an FX placeholder id (`in:l_…`)."""
    import re

    raw = str(ph_id)
    if raw.startswith("in:"):
        raw = raw[3:]
    if raw in _PLACEHOLDER_MEANING:
        return _PLACEHOLDER_MEANING[raw]

    s = raw[2:] if raw.startswith("l_") else raw
    s = s.rstrip("_")

    m = re.match(r"self_modules_(.+)_parameters_(weight|bias)$", s)
    if m:
        path, kind = m.group(1), m.group(2)
        return f"parameter · {'.'.join(path.split('_modules_'))}.{kind}"

    m = re.match(r"self_modules_(.+)_buffers_(.+)$", s)
    if m:
        path, buf = m.group(1), m.group(2)
        return f"buffer · {'.'.join(path.split('_modules_'))}.{buf}"

    return raw.rstrip("_").replace("_", " ")


def _merged_graph_inputs(g: dict) -> list[dict]:
    """Zip example_input_metas with FX placeholders (same order / length when both present)."""
    ph = [n for n in (g.get("io_nodes") or []) if str(n.get("i", "")).startswith("in:")]
    ex = g.get("example_input_metas") or []
    n = max(len(ph), len(ex))
    rows = []
    for i in range(n):
        p = ph[i] if i < len(ph) else None
        e = ex[i] if i < len(ex) else None
        ph_id = p.get("i") if p else None
        if p and p.get("output_shapes") is not None:
            shape = p.get("output_shapes")
            dtype = p.get("output_dtypes")
        elif e:
            shape = e.get("shape")
            dtype = e.get("dtype")
        else:
            shape, dtype = None, None
        rows.append(
            {
                "i": i,
                "placeholder": ph_id or "—",
                "meaning": _placeholder_meaning(ph_id) if ph_id else "example input (no FX name)",
                "shape": shape,
                "dtype": dtype,
            }
        )
    return rows


def _write_markdown(payload: dict) -> str:
    meta = payload["meta"]
    graphs = payload["graphs"]
    lines: list[str] = []
    lines.append("# SmolVLA Compile Operator List (GPU backend)")
    lines.append("")
    lines.append(
        "This document is the **torch.compile** counterpart of "
        "`doc/gpu/eager_mode/SmolVLA_AtenOp_List_gpu_backend.md`."
    )
    lines.append("")
    lines.append(
        "Eager chrono = every ATen dispatcher call. "
        "**Compile list = FX / AOT graph nodes** that Inductor receives after Dynamo "
        "tracing + decompositions (optimized graph *before* Triton/CUTLASS codegen). "
        "One Inductor kernel may still fuse many of these nodes."
    )
    lines.append("")
    lines.append("| Artifact | Role |")
    lines.append("|---|---|")
    lines.append("| `doc/gpu/compile_mode/smolvla_compile_fx_graphs.json` | structured FX graphs + ops |")
    lines.append("| `doc/gpu/compile_mode/smolvla_compile_fx_graphs.log` | readable graph dump |")
    lines.append(f"| `doc/gpu/compile_mode/{OUT_MD.name}` | this operator list |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 0. Capture method and scope")
    lines.append("")
    lines.append("### 0.1 What was compiled")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{meta['model_id']}` |")
    lines.append(f"| Device | `{meta['device']}` — {meta.get('gpu')} |")
    lines.append(f"| Torch / CUDA | {meta['torch']} / {meta.get('cuda')} |")
    lines.append(f"| `torch.compile` mode | **`{meta['compile_mode']}`** |")
    lines.append("| Compiled callable | `VLAFlowMatching.sample_actions` (LeRobot compile target) |")
    lines.append("| Not in this graph | `prepare_images` / processors (run eager outside `sample_actions`) |")
    lines.append(f"| Graphs captured | **{meta['num_graphs']}** |")
    lines.append(f"| Graph breaks (Dynamo explain) | {meta.get('graph_breaks', 'n/a')} |")
    lines.append("")
    lines.append("### 0.2 Method")
    lines.append("")
    lines.append("1. Load `lerobot/smolvla_base` on CUDA (eager), build fixed batch + noise (`SMOKE_SEED=0`).")
    lines.append(
        "2. Wrap `policy.model.sample_actions` with `torch.compile(..., backend=GraphCaptureBackend, "
        f"mode={meta['compile_mode']!r})`."
    )
    lines.append(
        "3. `GraphCaptureBackend` records each `torch.fx.GraphModule` Inductor would compile "
        "(node order, normalized `aten::*` name, output shapes/dtypes from FakeTensor meta), "
        "then calls `torch._inductor.compile_fx.compile_fx`."
    )
    lines.append("4. Run one cold + one warm `sample_actions` under `torch.inference_mode()`.")
    lines.append("")
    lines.append("**How to read vs eager AtenOp list**")
    lines.append("")
    lines.append("| Eager (`TorchDispatchMode`) | Compile (this file) |")
    lines.append("|---|---|")
    lines.append("| Every runtime aten dispatch | FX nodes after trace + AOT decomp |")
    lines.append("| Includes Python-visible copies/views | Many views may be elided or folded |")
    lines.append("| Stage tables split 0/1/2/3/4 | Graphs follow Dynamo partitions of `sample_actions` |")
    lines.append("| No fusion | Nodes are **pre-codegen**; Inductor may still fuse them into fewer kernels |")
    lines.append("")
    lines.append(
        "Ops in each graph are listed in **FX calling order** (`gm.graph.nodes` topo order). "
        "Input/Output dims+dtypes come from FakeTensor / ShapeProp meta on each FX node."
    )
    lines.append("")
    lines.append("### 0.3 FX graphs — roles and cross-graph dataflow")
    lines.append("")
    lines.append(
        "Dynamo split `sample_actions` into **8** Inductor compile units (7 breaks on "
        "`aten.nonzero` / data-dependent shapes). Eager glue between graphs stays in Python; "
        "each graph below is one FX module. Scope is **`sample_actions` only** — "
        "`prepare_images` / processors are outside."
    )
    lines.append("")
    lines.append("| Graph | Compute ops | Role (what this FX unit does) | Key tensors in → out |")
    lines.append("|---:|---:|---|---|")
    for gid in sorted(_GRAPH_MEANING):
        role, io = _GRAPH_MEANING[gid]
        nops = next((g["num_compute_ops"] for g in graphs if g["graph_id"] == gid), "?")
        lines.append(f"| **{gid}** | {nops} | {role} | {io} |")
    lines.append("")
    lines.append("**Dataflow across graphs** (smoke shape; single-camera path in this dump):")
    lines.append("")
    lines.append("```text")
    lines.append("  [eager] prepare_images / processors / batch → sample_actions(…)")
    lines.append("")
    lines.append("  G1  image f32 ──to──► image bf16")
    lines.append("                         │")
    lines.append("  G2  ───────────────────┼──► patch_attention_mask [1,32,32] bool")
    lines.append("                         ▼")
    lines.append("  G3  pixel_values + mask ──► patch embeddings [1,1024,768]")
    lines.append("                              + position_ids / flat mask")
    lines.append("                         ▼")
    lines.append("  G4  embeddings + pos_emb table ──► embeddings (+ position)")
    lines.append("                         ▼")
    lines.append("  G5  ViT encoder ──► last_hidden_state [1,1024,768]")
    lines.append("                         ▼")
    lines.append("  G6  connector ──► image_hidden_states [1,64,960]")
    lines.append("                         ╲")
    lines.append("  G7  tokens ──► lang embedding [1,48,960]  ╲")
    lines.append("                         ╲                 ╲")
    lines.append("                          ╲──── eager cat / state / mask ────╲")
    lines.append("                                              ▼")
    lines.append("  G8  prefix (≈241 tok) + suffix/noise + VLM+expert blocks")
    lines.append("       + Euler steps ──► action chunk [1,50,32]")
    lines.append("```")
    lines.append("")
    lines.append(
        "- **G1→G5** = vision tower (cast → mask → patch/pos → encoder). "
        "**G6** projects vision into the VLM width (960). **G7** embeds language. "
        "Python between G6/G7 and G8 builds the **prefix** (image tokens + language + state) "
        "and attention masks — those show up in G8 as `l_stack0_*` placeholders."
    )
    lines.append(
        "- **G8** is the bulk of compile work: multimodal transformer + action expert "
        "and the flow-matching / Euler loop that emits the action chunk "
        f"(captured out shape `{meta.get('action_out_shape')}`)."
    )
    lines.append(
        "- Graph breaks are **not** semantic stage boundaries; they are Dynamo "
        "partition points. Semantic Stage 0–4 in the eager docs map roughly to "
        "G1–G7 (prefix vision/lang) + G8 (prefill + expert Euler), with eager glue "
        "still owning some cats/masks."
    )
    lines.append("")
    lines.append("### 0.4 Global op histogram (all graphs, compute nodes only)")
    lines.append("")
    lines.append("| aten / op | count |")
    lines.append("|---|---:|")
    for name, cnt in list(payload["histogram"].items())[:60]:
        lines.append(f"| `{name}` | {cnt} |")
    if len(payload["histogram"]) > 60:
        lines.append(f"| … | ({len(payload['histogram']) - 60} more distinct ops) |")
    lines.append("")
    lines.append(f"**Total compute ops (sum over graphs):** {payload['total_compute_ops']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for g in graphs:
        gid = g["graph_id"]
        cov = g.get("shape_meta_coverage") or {}
        role, io = _GRAPH_MEANING.get(gid, ("(unknown partition)", "—"))
        # Short title for outline / TOC scanning
        short = role.split("(")[0].strip().rstrip(".")
        if len(short) > 72:
            short = short[:69] + "…"
        lines.append(f"## {gid}. Compiled FX graph #{gid} — {short}")
        lines.append("")
        lines.append(f"**Meaning:** {role}")
        lines.append("")
        lines.append(f"**Dataflow (this graph):** {io}")
        lines.append("")
        lines.append(
            f"**Compute ops (calling order):** {g['num_compute_ops']} · "
            f"**FX nodes (all):** {g['num_fx_nodes']} · "
            f"**shape meta:** out {cov.get('ops_with_output', '?')}/{g['num_compute_ops']}, "
            f"in {cov.get('ops_with_input', '?')}/{g['num_compute_ops']}"
        )
        lines.append("")
        lines.append(
            "Ops below are in **FX calling order** (`Seq` = 1 … N). "
            "Input / Output = shape + dtype from FakeTensor / ShapeProp meta on each node "
            "(and its tensor args)."
        )
        lines.append("")
        lines.append("| Seq | op | Input (shape, dtype) | Output (shape, dtype) |")
        lines.append("|---:|---|---|---|")
        for e in g["ops"]:
            # calling order: e['i'] is 1..N assigned in graph order
            inn = e.get("input_str") or _fmt_pair(e.get("input_shapes"), e.get("input_dtypes"))
            out = e.get("output_str") or _fmt_pair(e.get("output_shapes"), e.get("output_dtypes"))
            lines.append(f"| {e['i']} | `{e['name']}` | `{inn}` | `{out}` |")
        lines.append("")

        inputs = _merged_graph_inputs(g)
        if inputs:
            lines.append(f"### Graph #{gid} — inputs (Dynamo example ↔ FX placeholder)")
            lines.append("")
            lines.append("| # | placeholder | meaning | shape | dtype |")
            lines.append("|---:|---|---|---|---|")
            for row in inputs:
                ph = row["placeholder"]
                ph_cell = f"`{ph}`" if ph != "—" else "—"
                sh = row["shape"]
                dt = row["dtype"]
                sh_cell = f"`{sh}`" if sh is not None else "—"
                dt_cell = f"`{dt}`" if dt is not None else "—"
                lines.append(
                    f"| {row['i']} | {ph_cell} | {row['meaning']} | {sh_cell} | {dt_cell} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Comparison notes vs eager AtenOp list")
    lines.append("")
    lines.append(
        "- Eager Stage 0–3 chrono is split by Python stage helpers; compile captures "
        "**`sample_actions` only**, so `prepare_images` upsample/`2x−1` casts are absent here."
    )
    lines.append(
        "- Expect **fewer** distinct micro-ops for patterns Inductor decomposes differently "
        "(e.g. RMSNorm / RoPE may still appear as several aten nodes here, then fuse at codegen)."
    )
    lines.append(
        "- Multiple graphs ⇒ Dynamo **graph breaks** (typical around data-dependent control or "
        "unsupported ops). Each graph is a separate Inductor compile unit; within a graph, "
        "**Seq** is strict calling order."
    )
    lines.append(
        "- For GPU *kernel* identity after codegen, pair with Nsight on a compile-mode capture; "
        "this file stays at **graph / aten** level like the eager AtenOp doc."
    )
    lines.append("")
    lines.append("## Skill — rebuild this list")
    lines.append("")
    lines.append("```bash")
    lines.append("export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home")
    lines.append("export SMOKE_COMPILE_MODE=reduce-overhead")
    lines.append("python src/smolvla_compile_graph_dump.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Outputs land in `doc/gpu/compile_mode/`: JSON + log + this markdown. "
        "Change `SMOKE_COMPILE_MODE` to dump `max-autotune` graphs (cold compile longer)."
    )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID} on {DEVICE} (eager load, then compile sample_actions)", flush=True)
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = str(DEVICE)
    # Do NOT set compile_model on config — we wrap sample_actions ourselves to capture graphs.
    config.compile_model = False
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to(DEVICE).eval()
    cfg = policy.config
    preprocess, _post = make_pre_post_processors(
        cfg,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )
    frame = make_dummy_frame(cfg)
    batch = preprocess(frame)
    batch = policy._prepare_batch(batch)
    noise = make_fixed_noise((1, cfg.chunk_size, cfg.max_action_dim), DEVICE)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    model = policy.model
    backend = GraphCaptureBackend()

    print(f"Compiling sample_actions mode={_COMPILE_MODE!r}…", flush=True)
    torch.set_float32_matmul_precision("high")
    compiled_sample = torch.compile(
        model.sample_actions,
        backend=backend,
        mode=_COMPILE_MODE,
        fullgraph=False,
    )

    # Dynamo explain (optional; may re-trace)
    explain_meta = {}
    try:
        expl = dynamo.explain(model.sample_actions)(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )
        explain_meta = {
            "graph_count": getattr(expl, "graph_count", None),
            "graph_break_count": getattr(expl, "graph_break_count", None),
            "break_reasons": [str(x)[:200] for x in (getattr(expl, "break_reasons", None) or [])][:20],
        }
        print(f"dynamo.explain: graphs={explain_meta.get('graph_count')} breaks={explain_meta.get('graph_break_count')}", flush=True)
    except Exception as exc:
        explain_meta = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        print(f"dynamo.explain failed: {explain_meta['error']}", flush=True)

    print("Cold compiled sample_actions…", flush=True)
    _sync()
    out1 = compiled_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
    _sync()
    print(f"cold done out={tuple(out1.shape)} graphs_so_far={len(backend.graphs)}", flush=True)

    print("Warm compiled sample_actions…", flush=True)
    out2 = compiled_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
    _sync()
    print(f"warm done out={tuple(out2.shape)}", flush=True)

    all_ops = []
    for g in backend.graphs:
        all_ops.extend(g["ops"])
    hist = _op_histogram(all_ops)

    payload = {
        "meta": {
            "model_id": MODEL_ID,
            "device": str(DEVICE),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "smoke_seed": SMOKE_SEED,
            "compile_mode": _COMPILE_MODE,
            "compiled_fn": "VLAFlowMatching.sample_actions",
            "num_graphs": len(backend.graphs),
            "graph_breaks": explain_meta,
            "method": (
                "torch.compile custom backend recording FX GraphModule nodes "
                "then inductor compile_fx; shapes from FakeTensor meta"
            ),
            "action_out_shape": list(out2.shape),
            "action_max_abs_diff_run1_run2": float((out1 - out2).abs().max()),
        },
        "histogram": hist,
        "total_compute_ops": len(all_ops),
        "graphs": backend.graphs,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    with OUT_LOG.open("w") as fh:
        fh.write(f"SmolVLA compile FX dump  mode={_COMPILE_MODE}  graphs={len(backend.graphs)}\n")
        fh.write(json.dumps(payload["meta"], indent=2, default=str))
        fh.write("\n")
        for g in backend.graphs:
            fh.write("\n" + "=" * 72 + "\n")
            fh.write(f"GRAPH {g['graph_id']}  compute_ops={g['num_compute_ops']}\n")
            fh.write("=" * 72 + "\n")
            for e in g["ops"]:
                fh.write(
                    f"{e['i']:5d}  {e['name']:<40}  out={e.get('output_shapes')} {e.get('output_dtypes')}\n"
                )
            fh.write("\n--- readable head ---\n")
            fh.write(g.get("readable_head") or "")
            fh.write("\n")

    md = _write_markdown(payload)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_LOG}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)
    print(f"TOTAL compute ops={len(all_ops)} distinct={len(hist)} graphs={len(backend.graphs)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

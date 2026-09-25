#!/usr/bin/env python
"""Regenerate SmolVLA_Fused_CompileOp_List_gpu_backend.md from Inductor debug dumps.

Produces CompileOp-style per-graph tables: Seq in **calling order**, with meaning,
fused aten origins, and Input/Output (shape, dtype) when recoverable from
output_code.py / provenance.

Usage on GPU host (after smolvla_compile_fused_dump.py):
  python src/smolvla_compile_fused_regen_md.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "gpu" / "compile_mode"
DEBUG_ROOT = OUT_DIR / "inductor_debug"
OUT_JSON = OUT_DIR / "smolvla_compile_fused_kernels.json"
OUT_MD = OUT_DIR / "SmolVLA_Fused_CompileOp_List_gpu_backend.md"
OUT_TRACE = OUT_DIR / "smolvla_compile_chrome_trace.json"

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

_EXTERN_MEANING = {
    "addmm": "GEMM + bias (aten.linear / addmm) via cuBLAS",
    "mm": "GEMM matmul (aten.mm / linear without fused bias) via cuBLAS",
    "bmm": "Batched matmul (aten.bmm) via cuBLAS",
    "convolution": "Conv (aten.convolution / conv2d) via cuDNN/cuBLAS",
}

_DTYPE_SHORT = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i8": "int8",
    "b8": "bool",
    "bool": "bool",
}


def _fmt_pair(shape, dtype) -> str:
    if shape is None and dtype is None:
        return "—"
    if shape is None:
        return f"? {dtype}" if dtype else "—"
    dt = dtype or ""
    return f"{shape} {dt}".strip()


def _parse_empty_strided(line: str) -> tuple[str | None, dict] | tuple[None, None]:
    m = re.match(
        r"(buf\d+|arg\d+_1)\s*=\s*empty_strided_(?:cuda|cpu)\(\s*(\([^)]*\)|\[\s*[^\]]*\s*\])\s*,"
        r"\s*(\([^)]*\))\s*,\s*(torch\.\w+)",
        line.strip(),
    )
    if not m:
        # alternate: empty_strided_cuda((1, 3, 512, 512), (786432, 1, 1536, 3), torch.bfloat16)
        m = re.match(
            r"(buf\d+|arg\d+_1)\s*=\s*empty_strided_(?:cuda|cpu)\((.+)\)$",
            line.strip(),
        )
        if not m:
            return None, None
        name, insides = m.group(1), m.group(2)
        sm = re.match(
            r"\(([^)]*)\),\s*\([^)]*\),\s*(torch\.\w+)",
            insides.strip(),
        )
        if not sm:
            return None, None
        shape_s, dt = sm.group(1), sm.group(2).replace("torch.", "")
        shape = "[" + ",".join(s.strip() for s in shape_s.split(",") if s.strip() != "") + "]"
        if shape == "[]" and shape_s.strip() == "":
            shape = "[]"
        return name, {"shape": shape, "dtype": dt}
    name = m.group(1)
    shape_raw = m.group(2).strip()
    dt = m.group(4).replace("torch.", "")
    if shape_raw.startswith("("):
        shape_s = shape_raw[1:-1]
        shape = "[" + ",".join(s.strip() for s in shape_s.split(",") if s.strip() != "") + "]"
    else:
        shape = shape_raw.replace(" ", "")
    return name, {"shape": shape, "dtype": dt}


def _parse_fx_comment_tensors(text: str) -> dict[str, dict]:
    """Map FX %name / aten target → shape/dtype from Inductor header comments."""
    out: dict[str, dict] = {}
    # %foo : Tensor "bf16[1, 768, 32, 32][stride]cuda:0" = call_function[target=torch.ops.aten.xxx
    for m in re.finditer(
        r"%([\w]+)\s*:\s*Tensor\s*\"([^\"]+)\"\s*=\s*call_function\[target=torch\.ops\.aten\.([\w.]+)\]",
        text,
    ):
        name, meta, target = m.group(1), m.group(2), m.group(3)
        # meta like bf16[1, 768, 32, 32][786432, ...]cuda:0
        mm = re.match(r"([a-z0-9]+)\\?\[([^\]]*)\]", meta)
        if not mm:
            mm = re.match(r"([a-z0-9]+)\[([^\]]*)\]", meta)
        if not mm:
            continue
        dt = _DTYPE_SHORT.get(mm.group(1), mm.group(1))
        shape = "[" + mm.group(2).replace(" ", "") + "]"
        short = target.split(".")[0]
        rec = {"shape": shape, "dtype": dt, "aten": f"aten::{short}", "fx": name}
        out[name] = rec
        out.setdefault(f"aten::{short}", rec)
        out.setdefault(short, rec)
    return out


def _kernel_ops_hint(name: str) -> tuple[str, str]:
    m = re.match(r"triton_(\w+?)_fused_(.+?)(?:_(\d+))?$", name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"triton_(\w+)_(.+)$", name)
    if m:
        return m.group(1), m.group(2)
    return "—", name


def _meaning_for(kind: str, name: str, fused_ops: list[str]) -> str:
    if kind == "extern":
        short = name.split(".")[-1]
        return _EXTERN_MEANING.get(short, f"Extern kernel `{short}`")
    if kind == "aten":
        return f"ATen op `{name}` (Inductor wrapper / mutation)"
    cat, hint = _kernel_ops_hint(name)
    cat_map = {
        "poi": "pointwise Triton fusion",
        "per": "persistent-reduction Triton fusion",
        "red": "reduction Triton fusion",
    }
    cat_s = cat_map.get(cat, f"Triton ({cat})")
    if fused_ops:
        ops = ", ".join(fused_ops[:12])
        if len(fused_ops) > 12:
            ops += f", …(+{len(fused_ops) - 12})"
        return f"{cat_s}: {ops}"
    # readable hint
    hint_r = hint.replace("_", " ").strip()
    return f"{cat_s}: {hint_r}" if hint_r else cat_s


def _extract_partition_bodies(text: str) -> dict[int, str]:
    bodies: dict[int, str] = {}
    for m in re.finditer(
        r"^def partition_(\d+)\((?:self, )?args\):([\s\S]*?)"
        r"(?=^def |^class |\nrunner = |\nif __name__|\Z)",
        text,
        re.M,
    ):
        bodies[int(m.group(1))] = m.group(2)
    return bodies


def _extract_runner_call(text: str) -> str:
    m = re.search(
        r"class Runner[\s\S]*?def call\(self, args\):([\s\S]*?)(?=\nrunner = |\nif __name__|\Z)",
        text,
    )
    if m:
        return m.group(1)
    # fallback: some graphs only have partition_0 used directly
    return ""


def _parse_launches_in_body(body: str, buf_meta: dict) -> list[dict]:
    launches: list[dict] = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name_meta = _parse_empty_strided(s)
        if name_meta[0]:
            buf_meta[name_meta[0]] = name_meta[1]
            continue
        m = re.match(
            r"(buf\d+)\s*=\s*reinterpret_tensor\((\w+),\s*(\([^)]*\)|\[[^\]]*\])",
            s,
        )
        if m:
            dst, src, shape_raw = m.group(1), m.group(2), m.group(3)
            base = dict(buf_meta.get(src) or {})
            if shape_raw.startswith("("):
                shape_s = shape_raw[1:-1]
                base["shape"] = (
                    "["
                    + ",".join(x.strip() for x in shape_s.split(",") if x.strip() != "")
                    + "]"
                )
            elif shape_raw.startswith("["):
                base["shape"] = shape_raw.replace(" ", "")
            buf_meta[dst] = base
            continue

        if ".run(" in s and "triton_" in s:
            k = re.search(r"(triton_[\w]+)\.run\((.*)\)\s*$", s)
            if not k:
                continue
            raw_args = [a.strip() for a in k.group(2).split(",")]
            args = [
                a
                for a in raw_args
                if a
                and not a.startswith("stream=")
                and not re.fullmatch(r"\d+", a)
                and not re.fullmatch(r"[\d.]+", a)
            ]
            launches.append(
                {
                    "kind": "triton",
                    "name": k.group(1),
                    "args": args,
                    "line": s[:240],
                }
            )
            continue

        if "extern_kernels." in s:
            # buf = extern_kernels.foo(...)  OR  extern_kernels.foo(..., out=buf)
            k = re.search(
                r"(?:(buf\d+)\s*=\s*)?extern_kernels\.(\w+)\((.*)\)\s*$",
                s,
            )
            if k:
                out = k.group(1)
                rest = k.group(3)
                om = re.search(r"\bout\s*=\s*(buf\d+)", rest)
                if om:
                    out = om.group(1)
                targs = re.findall(
                    r"reinterpret_tensor\([^)]*(?:\([^)]*\)[^)]*)*\)|buf\d+|arg\d+_1",
                    rest,
                )
                # drop out=buf from tensor args list if present as bare buf
                targs = [t for t in targs if t != out]
                arg_metas = []
                for t in targs:
                    if t.startswith("reinterpret_tensor"):
                        rm = re.search(
                            r"reinterpret_tensor\(\s*(\w+)\s*,\s*\(([^)]*)\)",
                            t,
                        )
                        if rm:
                            src = rm.group(1)
                            shape = (
                                "["
                                + ",".join(
                                    x.strip()
                                    for x in rm.group(2).split(",")
                                    if x.strip() != ""
                                )
                                + "]"
                            )
                            dt = (buf_meta.get(src) or {}).get("dtype")
                            arg_metas.append(_fmt_pair(shape, dt))
                        else:
                            arg_metas.append("—")
                    elif t in buf_meta:
                        arg_metas.append(
                            _fmt_pair(buf_meta[t].get("shape"), buf_meta[t].get("dtype"))
                        )
                    else:
                        arg_metas.append("—")
                launches.append(
                    {
                        "kind": "extern",
                        "name": f"extern_kernels.{k.group(2)}",
                        "out": out,
                        "args": targs,
                        "arg_metas": arg_metas,
                        "line": s[:240],
                    }
                )
            continue

        if "aten." in s:
            k = re.search(r"aten\.(\w+)\(", s)
            if k:
                out_m = re.match(r"(buf\d+|arg\d+_1)\s*=\s*aten\.", s)
                launches.append(
                    {
                        "kind": "aten",
                        "name": f"aten::{k.group(1)}",
                        "out": out_m.group(1) if out_m else None,
                        "args": re.findall(r"(buf\d+|arg\d+_1)", s),
                        "line": s[:240],
                    }
                )
    return launches


def _seed_arg_meta_from_fx(text: str, buf_meta: dict) -> None:
    for m in re.finditer(r"%?(arg\d+_1)\s*:?\s*Tensor\s*\"([^\"]+)\"", text):
        name, meta = m.group(1), m.group(2)
        mm = re.match(r"([a-z0-9]+)\[([^\]]*)\]", meta)
        if not mm:
            continue
        dt = _DTYPE_SHORT.get(mm.group(1), mm.group(1))
        shape = "[" + mm.group(2).replace(" ", "") + "]"
        buf_meta.setdefault(name, {"shape": shape, "dtype": dt})


def _expand_runner_order(text: str) -> list[dict]:
    """Full calling order: Runner.call steps, expanding partition bodies."""
    buf_meta: dict = {}
    for line in text.splitlines():
        n, meta = _parse_empty_strided(line.strip())
        if n:
            buf_meta[n] = meta
    _seed_arg_meta_from_fx(text, buf_meta)

    partitions = _extract_partition_bodies(text)
    runner = _extract_runner_call(text)
    ordered: list[dict] = []

    if runner.strip():
        for line in runner.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n, meta = _parse_empty_strided(s)
            if n:
                buf_meta[n] = meta
                continue
            pm = re.search(r"self\.partitions\[(\d+)\]\(", s)
            if pm:
                pid = int(pm.group(1))
                body = partitions.get(pid, "")
                ordered.extend(_parse_launches_in_body(body, buf_meta))
                continue
            ordered.extend(_parse_launches_in_body(s + "\n", buf_meta))
    else:
        for pid in sorted(partitions):
            ordered.extend(_parse_launches_in_body(partitions[pid], buf_meta))

    for L in ordered:
        L["_buf_meta"] = buf_meta
    return ordered


def _io_from_launch(L: dict, fx_meta: dict, fused_ops: list[str]) -> tuple[str, str]:
    buf_meta = L.get("_buf_meta") or {}
    if L.get("arg_metas"):
        ins = [x for x in L["arg_metas"] if x != "—"] or L["arg_metas"]
        in_s = " | ".join(L["arg_metas"]) if L["arg_metas"] else "—"
    else:
        args = L.get("args") or []
        tensor_args = [a for a in args if re.match(r"^(buf\d+|arg\d+_1)$", a)]
        ins = []
        for a in tensor_args:
            if L.get("out") and a == L["out"]:
                continue
            m = buf_meta.get(a)
            ins.append(_fmt_pair(m.get("shape"), m.get("dtype")) if m else "—")
        in_s = " | ".join(ins) if ins else "—"
        tensor_args_for_out = tensor_args
    out_s = "—"
    if L.get("out") and L["out"] in buf_meta:
        m = buf_meta[L["out"]]
        out_s = _fmt_pair(m.get("shape"), m.get("dtype"))
    elif L["kind"] == "triton":
        args = L.get("args") or []
        tensor_args = [a for a in args if re.match(r"^(buf\d+|arg\d+_1)$", a)]
        if tensor_args:
            last = tensor_args[-1]
            if last in buf_meta and last.startswith("buf"):
                out_s = _fmt_pair(buf_meta[last].get("shape"), buf_meta[last].get("dtype"))
                # rebuild inputs without last out buf
                ins2 = []
                for a in tensor_args[:-1]:
                    m = buf_meta.get(a)
                    ins2.append(_fmt_pair(m.get("shape"), m.get("dtype")) if m else "—")
                in_s = " | ".join(ins2) if ins2 else "—"

    if (not in_s or in_s == "—" or set(in_s.split(" | ")) <= {"—"}) and fused_ops:
        for op in fused_ops:
            key = op if op.startswith("aten::") else f"aten::{op}"
            if key in fx_meta:
                in_s = _fmt_pair(fx_meta[key].get("shape"), fx_meta[key].get("dtype"))
                break
            # also try bare name match in fx values
            for v in fx_meta.values():
                if v.get("aten") == key or v.get("aten") == f"aten::{op}":
                    # prefer as output hint below; skip
                    pass
    if out_s == "—" and L["kind"] == "extern":
        short = L["name"].split(".")[-1]
        for key in (f"aten::{short}", short, "convolution", "addmm", "mm", "bmm"):
            if key in fx_meta:
                out_s = _fmt_pair(fx_meta[key].get("shape"), fx_meta[key].get("dtype"))
                break
            for v in fx_meta.values():
                if v.get("aten") == f"aten::{short}":
                    out_s = _fmt_pair(v.get("shape"), v.get("dtype"))
                    break
            if out_s != "—":
                break
    if out_s == "—" and fused_ops:
        for op in reversed(fused_ops):
            key = op if op.startswith("aten::") else op
            for v in fx_meta.values():
                if v.get("aten") in (f"aten::{key}", key) or v.get("fx") == key:
                    out_s = _fmt_pair(v.get("shape"), v.get("dtype"))
                    break
            if out_s != "—":
                break
    return in_s, out_s


def _load_provenance(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fused_ops_for(name: str, kind: str, prov: dict) -> list[str]:
    cpp = prov.get("cppCodeToPost") or {}
    hits = []
    for k, v in cpp.items():
        base = k.split(":")[0]
        if kind == "triton" and base == name:
            hits = list(v)
            break
        if kind == "extern" and (
            base == name
            or base.endswith("." + name.split(".")[-1])
            or base == name.split(".")[-1]
            or base.endswith(name)
        ):
            hits = list(v)
            break
    out = []
    for h in hits:
        h = str(h)
        if h.startswith("aten::"):
            out.append(h)
        elif h.startswith("aten."):
            out.append("aten::" + h.split(".")[1])
        else:
            # FX node names from provenance — keep readable, also try as aten::
            if re.match(r"^[a-z_]+$", h) and h not in ("clone",):
                out.append(h)
            else:
                out.append(h)
    # Dedup preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _summarize_graph(debug_dir: Path, gid: int) -> dict:
    code_path = debug_dir / "output_code.py"
    text = code_path.read_text(errors="replace") if code_path.exists() else ""
    prov = _load_provenance(debug_dir / "inductor_provenance_tracking_node_mappings.json")
    fx_meta = _parse_fx_comment_tensors(text)
    launches = _expand_runner_order(text)

    ops = []
    for i, L in enumerate(launches, 1):
        fused = _fused_ops_for(L["name"], L["kind"], prov)
        # also from "Original ATen: [...]" near kernel — already in provenance
        if L["kind"] == "extern" and not fused:
            short = L["name"].split(".")[-1]
            fused = [f"aten::{short}"]
        if L["kind"] == "aten" and not fused:
            fused = [L["name"]]
        meaning = _meaning_for(L["kind"], L["name"], fused)
        inn, out = _io_from_launch(L, fx_meta, fused)
        ops.append(
            {
                "i": i,
                "kind": L["kind"],
                "name": L["name"],
                "meaning": meaning,
                "fused_aten": fused,
                "fused_aten_str": ", ".join(fused) if fused else "—",
                "input_str": inn,
                "output_str": out,
            }
        )

    role, io = _GRAPH_MEANING.get(gid, ("(unknown)", "—"))
    return {
        "graph_id": gid,
        "meaning": role,
        "dataflow": io,
        "debug_dir": str(debug_dir),
        "num_launches": len(ops),
        "kind_histogram": dict(Counter(o["kind"] for o in ops)),
        "ops": ops,  # calling order
    }


def _find_debug_dirs() -> list[Path]:
    found = []
    if not DEBUG_ROOT.exists():
        return found
    for p in DEBUG_ROOT.rglob("output_code.py"):
        if "inference" in p.parent.name:
            found.append(p.parent)
    found = sorted(set(found), key=lambda p: p.stat().st_mtime)
    # last 8 inference dirs in graph order by name model__N
    found = sorted(found, key=lambda p: p.name)
    return found[-8:] if len(found) >= 8 else found


def _parse_chrome(path: Path) -> dict:
    if not path.exists():
        return {"kernels": [], "unique_kernel_events": 0, "total_kernel_events": 0}
    try:
        data = json.loads(path.read_text())
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
    except Exception as exc:
        return {"error": str(exc), "kernels": []}
    names: Counter = Counter()
    for e in events or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or ""
        if "triton" in name.lower() or name.startswith("extern") or "gemm" in name.lower():
            names[name] += 1
    return {
        "unique_kernel_events": len(names),
        "total_kernel_events": sum(names.values()),
        "kernels": [{"name": n, "count": c} for n, c in names.most_common(40)],
    }


def _write_markdown(payload: dict) -> str:
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# SmolVLA Fused Compile Operator List (GPU backend)")
    lines.append("")
    lines.append(
        "This document is the **Inductor post-fusion / codegen** counterpart of "
        "`SmolVLA_CompileOp_List_gpu_backend.md` (pre-codegen FX aten nodes)."
    )
    lines.append("")
    lines.append("| Artifact | Role |")
    lines.append("|---|---|")
    lines.append("| `smolvla_compile_fused_kernels.json` | structured fused launch list |")
    lines.append("| `smolvla_compile_chrome_trace.json` | warm-run chrome / Kineto trace |")
    lines.append("| `inductor_debug/**/output_code.py` | Inductor codegen source of truth |")
    lines.append(f"| `{OUT_MD.name}` | this operator list |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 0. Capture method and scope")
    lines.append("")
    lines.append("### 0.1 What was compiled")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{meta.get('model_id')}` |")
    lines.append(f"| Device | `{meta.get('device')}` — {meta.get('gpu')} |")
    lines.append(f"| Torch / CUDA | {meta.get('torch')} / {meta.get('cuda')} |")
    lines.append(f"| `torch.compile` mode | **`{meta.get('compile_mode')}`** |")
    lines.append("| Compiled callable | `VLAFlowMatching.sample_actions` |")
    lines.append(
        "| Env | `TORCH_COMPILE_DEBUG=1`, `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`, "
        "`TORCH_LOGS=output_code` |"
    )
    lines.append(f"| Graphs / debug dirs | **{meta.get('num_graphs')}** |")
    lines.append("")
    lines.append("### 0.2 Method")
    lines.append("")
    lines.append(
        "1. Compile `sample_actions` with Inductor debug dumps "
        "(`ir_post_fusion.txt`, `output_code.py`, provenance JSON)."
    )
    lines.append(
        "2. Walk each graph’s `Runner.call` / `partition_*` body in **source order** "
        "to list launches: `aten::*`, `extern_kernels.*`, `triton_*_fused_*.run`."
    )
    lines.append(
        "3. Attach fused aten origins from `inductor_provenance_tracking_node_mappings.json` "
        "(`cppCodeToPost`) and I/O shapes/dtypes from `empty_strided_*` / FX header comments."
    )
    lines.append("")
    lines.append("| FX list (`CompileOp`) | Fused list (this file) |")
    lines.append("|---|---|")
    lines.append("| Pre-codegen aten / view / transpose | Post-fusion **GPU launches** |")
    lines.append("| One row per FX node | One row per Triton / extern / leftover aten |")
    lines.append("| Fusion not visible | `_fused_` names + provenance = merged atens |")
    lines.append("")
    lines.append("### 0.3 FX graphs — roles and cross-graph dataflow")
    lines.append("")
    lines.append(
        "Same Dynamo partitions as the FX compile list (8 graphs). "
        "See `SmolVLA_CompileOp_List_gpu_backend.md` §0.3 for the full ascii dataflow."
    )
    lines.append("")
    lines.append("| Graph | Fused launches | Role | Key tensors in → out |")
    lines.append("|---:|---:|---|---|")
    for g in payload["graphs"]:
        role, io = g.get("meaning"), g.get("dataflow")
        lines.append(
            f"| **{g['graph_id']}** | {g['num_launches']} | {role} | {io} |"
        )
    lines.append("")

    # global histogram of launch names
    hist: Counter = Counter()
    for g in payload["graphs"]:
        for o in g["ops"]:
            hist[o["name"]] += 1
    lines.append("### 0.4 Global fused-launch histogram (top 40)")
    lines.append("")
    lines.append(f"**Total fused launches (sum over graphs):** {sum(hist.values())}")
    lines.append("")
    lines.append("| kernel / op | count |")
    lines.append("|---|---:|")
    for name, cnt in hist.most_common(40):
        lines.append(f"| `{name}` | {cnt} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for g in payload["graphs"]:
        gid = g["graph_id"]
        short = (g.get("meaning") or "").split("(")[0].strip().rstrip(".")
        if len(short) > 72:
            short = short[:69] + "…"
        lines.append(f"## {gid}. Fused Inductor graph #{gid} — {short}")
        lines.append("")
        lines.append(f"**Meaning:** {g.get('meaning')}")
        lines.append("")
        lines.append(f"**Dataflow (this graph):** {g.get('dataflow')}")
        lines.append("")
        kh = g.get("kind_histogram") or {}
        lines.append(
            f"**Launches (calling order):** {g['num_launches']} · "
            f"triton={kh.get('triton', 0)} · extern={kh.get('extern', 0)} · "
            f"aten={kh.get('aten', 0)}"
        )
        lines.append("")
        lines.append(
            "Ops below are in **Inductor codegen calling order** "
            "(`Runner.call` / `partition_*`). "
            "**Fused aten** = origins merged into this launch (provenance). "
            "Input/Output from buffer allocs / FX meta when available."
        )
        lines.append("")
        lines.append(
            "| Seq | kernel / op | kind | meaning | fused aten | "
            "Input (shape, dtype) | Output (shape, dtype) |"
        )
        lines.append("|---:|---|---|---|---|---|---|")
        for e in g["ops"]:
            fused_list = e.get("fused_aten") or []
            fused = ", ".join(f"`{x}`" for x in fused_list) if fused_list else "—"
            lines.append(
                f"| {e['i']} | `{e['name']}` | `{e['kind']}` | {e['meaning']} | "
                f"{fused} | `{e.get('input_str') or '—'}` | `{e.get('output_str') or '—'}` |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## How to rebuild")
    lines.append("")
    lines.append("```bash")
    lines.append("export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home")
    lines.append("export SMOKE_COMPILE_MODE=reduce-overhead")
    lines.append("python src/smolvla_compile_fused_dump.py   # cold compile + debug dumps")
    lines.append("python src/smolvla_compile_fused_regen_md.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    dirs = _find_debug_dirs()
    if not dirs:
        print(f"No debug dirs under {DEBUG_ROOT}", flush=True)
        return 1
    print(f"Found {len(dirs)} debug dirs", flush=True)
    graphs = [_summarize_graph(d, i) for i, d in enumerate(dirs, 1)]
    for g in graphs:
        print(
            f"g{g['graph_id']}: launches={g['num_launches']} kinds={g['kind_histogram']}",
            flush=True,
        )

    meta = {
        "model_id": "lerobot/smolvla_base",
        "device": "cuda",
        "gpu": None,
        "torch": None,
        "cuda": None,
        "compile_mode": "reduce-overhead",
        "num_graphs": len(graphs),
        "method": "output_code.py Runner/partition calling order + provenance + empty_strided I/O",
    }
    if OUT_JSON.exists():
        try:
            old = json.loads(OUT_JSON.read_text())
            meta = {**old.get("meta", {}), **meta}
            meta["num_graphs"] = len(graphs)
        except Exception:
            pass

    chrome = _parse_chrome(OUT_TRACE)
    payload = {"meta": meta, "chrome_trace": chrome, "graphs": graphs}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    OUT_MD.write_text(_write_markdown(payload))
    print(f"Wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

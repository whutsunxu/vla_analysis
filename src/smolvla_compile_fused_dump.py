#!/usr/bin/env python
"""Dump Inductor *post-fusion* kernels for SmolVLA sample_actions.

Captures what happens *after* the FX aten list (SmolVLA_CompileOp_List…):
  - TORCH_COMPILE_DEBUG=1 → ir_pre_fusion.txt / ir_post_fusion.txt / output_code.py
  - TORCH_LOGS=output_code → same codegen text in a log
  - TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 + chrome trace → runtime fused kernel names

Usage (remote GPU):
  source /venv/main/bin/activate
  export HF_HOME=/workspace/.hf_home SMOKE_DEVICE=cuda
  export SMOKE_COMPILE_MODE=reduce-overhead
  python src/smolvla_compile_fused_dump.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Must set before inductor compiles heavily.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("SMOKE_DEVICE", "cuda")
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
# Keep debug dumps under the repo so we can scp them.
_OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "gpu" / "compile_mode"
_DEBUG_ROOT = _OUT_DIR / "inductor_debug"
_CACHE_ROOT = _OUT_DIR / "inductor_cache_fused_dump"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(_DEBUG_ROOT)
# Fresh cache so we actually regenerate output_code / ir_post_fusion (no hit skip).
if _CACHE_ROOT.exists():
    shutil.rmtree(_CACHE_ROOT, ignore_errors=True)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_CACHE_ROOT)

import torch
import torch._dynamo as dynamo
import torch._logging

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
OUT_DIR = _OUT_DIR
OUT_JSON = OUT_DIR / "smolvla_compile_fused_kernels.json"
OUT_LOG = OUT_DIR / "smolvla_compile_fused_dump.log"
OUT_CODE_LOG = OUT_DIR / "smolvla_inductor_output_code.log"
OUT_TRACE = OUT_DIR / "smolvla_compile_chrome_trace.json"
OUT_MD = OUT_DIR / "SmolVLA_Fused_CompileOp_List_gpu_backend.md"

_GRAPH_MEANING = {
    1: "Cast camera image f32 → bf16 (ViT dtype).",
    2: "Build full ViT patch attention mask (`ones` → bool).",
    3: "ViT patch embed (`conv2d` 16×16) + mask-derived position-id bookkeeping.",
    4: "Add learned position embedding to patch tokens.",
    5: "ViT encoder stack (12× LayerNorm / SDPA / MLP) → last_hidden_state.",
    6: "Vision→language connector (spatial pack / PixelShuffle-style reshape + Linear).",
    7: "Language token embedding lookup.",
    8: "Prefix + VLM/expert flow-matching (attn / RoPE / Euler) → action chunk.",
}


def _collect_debug_dirs(root: Path) -> list[Path]:
    """Find Inductor debug dump dirs (contain output_code.py / ir_post_fusion.txt)."""
    found: list[Path] = []
    search_roots = []
    for base in (root, root / "torchinductor", Path("/tmp")):
        if base.exists():
            search_roots.append(base)
    for base in search_roots:
        for code in base.rglob("output_code.py"):
            found.append(code.parent)
    # Prefer newest; dedupe
    found = sorted(set(found), key=lambda p: p.stat().st_mtime)
    return found


def _parse_post_fusion_ir(text: str) -> list[dict]:
    """Extract top-level fused / extern / scheduler nodes from ir_post_fusion.txt.

    Nested `snodes[i] =\\n opX: SchedulerNode` blocks are ignored so counts reflect
    Inductor's post-fusion schedule (not pre-fusion leaf buffers).
    """
    nodes = []
    # Top-level only: name at column 0.
    block_re = re.compile(
        r"^(?P<name>[\w]+):\s*(?P<kind>FusedSchedulerNode|SchedulerNode|NopKernelSchedulerNode|"
        r"ExternKernelSchedulerNode)\(",
        re.M,
    )
    matches = list(block_re.finditer(text))
    for i, m in enumerate(matches):
        # Skip nested leaves that appear after ".snodes[" in the preceding few lines
        pretxt = text[max(0, m.start() - 80) : m.start()]
        if ".snodes[" in pretxt:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), start + 8000)
        body = text[start:end]
        # Prefer explicit aten. origins; also harvest from Triton-style name fragments in group
        origins = sorted(set(re.findall(r"aten\.[\w.]+", body)))
        # snodes count inside a fused node
        snodes = len(re.findall(r"\.snodes\[\d+\]\s*=", body))
        nodes.append(
            {
                "name": m.group("name"),
                "kind": m.group("kind"),
                "aten_origins": origins[:40],
                "n_snodes": snodes,
                "body_chars": len(body),
            }
        )
    return nodes


def _parse_output_code(text: str) -> dict:
    """Pull Triton / extern / cudagraph call sites from output_code.py."""
    # def triton_poi_fused_... or async_compile.triton('triton_...', ...)
    triton_defs = sorted(set(re.findall(r"\b(triton_[\w]+)\s*=\s*async_compile\.triton", text)))
    triton_defs += sorted(set(re.findall(r"^def (triton_[\w]+)\(", text, re.M)))
    triton_defs = sorted(set(triton_defs))

    extern = sorted(set(re.findall(r"extern_kernels\.(\w+)\(", text)))
    aten_calls = sorted(set(re.findall(r"aten\.(\w+)\.", text)))
    # fused name hints in kernel names: triton_poi_fused_add_mul_0
    fused_tags = []
    for name in triton_defs:
        m = re.match(r"triton_(\w+?)_fused_(.+?)_\d+$", name)
        if m:
            fused_tags.append({"kernel": name, "category": m.group(1), "ops_hint": m.group(2)})
        else:
            m2 = re.match(r"triton_(\w+)_(.+)$", name)
            if m2:
                fused_tags.append({"kernel": name, "category": m2.group(1), "ops_hint": m2.group(2)})

    return {
        "triton_kernels": triton_defs,
        "extern_kernels": extern,
        "aten_direct": aten_calls[:80],
        "fused_name_hints": fused_tags,
        "n_triton": len(triton_defs),
        "n_extern": len(extern),
        "code_chars": len(text),
    }


def _parse_chrome_trace(path: Path) -> dict:
    if not path.exists():
        return {"error": "missing", "kernels": []}
    try:
        events = json.loads(path.read_text())
        if isinstance(events, dict):
            events = events.get("traceEvents") or events.get("events") or []
    except Exception as exc:
        return {"error": str(exc), "kernels": []}
    names = Counter()
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("ph") not in ("X", "i", None) and e.get("cat") not in ("kernel", "gpu_user_annotation", "cuda"):
            # still count gpu kernels by name heuristics
            pass
        name = e.get("name") or ""
        cat = str(e.get("cat") or "")
        if not name:
            continue
        # CUDA / Triton kernels
        if (
            "triton" in name.lower()
            or name.startswith("void ")
            or "gemm" in name.lower()
            or "cutlass" in name.lower()
            or "cudnn" in name.lower()
            or "cublas" in name.lower()
            or cat in ("kernel", "gpu_user_annotation", "cuda_runtime", "gpu")
            or "sm_" in name
        ):
            # skip very generic CPU
            if name in ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize"):
                continue
            names[name] += 1
    top = names.most_common(200)
    return {
        "unique_kernel_events": len(names),
        "total_kernel_events": sum(names.values()),
        "kernels": [{"name": n, "count": c} for n, c in top],
    }


def _summarize_debug_dir(d: Path, idx: int) -> dict:
    rec = {
        "graph_id": idx,
        "debug_dir": str(d),
        "files": sorted(x.name for x in d.iterdir() if x.is_file())[:40],
        "meaning": _GRAPH_MEANING.get(idx, ""),
    }
    post = d / "ir_post_fusion.txt"
    pre = d / "ir_pre_fusion.txt"
    code = d / "output_code.py"
    if post.exists():
        nodes = _parse_post_fusion_ir(post.read_text(errors="replace"))
        kinds = Counter(n["kind"] for n in nodes)
        rec["post_fusion"] = {
            "n_nodes": len(nodes),
            "kind_histogram": dict(kinds),
            "nodes": nodes,
        }
        rec["n_pre_fusion_bytes"] = pre.stat().st_size if pre.exists() else 0
        rec["n_post_fusion_bytes"] = post.stat().st_size
    if code.exists():
        rec["output_code"] = _parse_output_code(code.read_text(errors="replace"))
        # copy slim pointer
        rec["output_code_path"] = str(code)
        rec["ir_post_fusion_path"] = str(post) if post.exists() else None
    return rec


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
    lines.append("| Layer | Artifact | What you see |")
    lines.append("|---|---|---|")
    lines.append(
        "| FX / AOT (pre-codegen) | `SmolVLA_CompileOp_List_gpu_backend.md` | "
        "Many small `aten::*` / `view` / `transpose` / `add` |"
    )
    lines.append(
        "| **Inductor fused (this file)** | `ir_post_fusion.txt` + `output_code.py` + chrome trace | "
        "**FusedSchedulerNode** / Triton `triton_*_fused_*` / cuBLAS extern |"
    )
    lines.append("")
    lines.append("### Capture method")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{meta['model_id']}` |")
    lines.append(f"| Device | `{meta['device']}` — {meta.get('gpu')} |")
    lines.append(f"| Torch / CUDA | {meta['torch']} / {meta.get('cuda')} |")
    lines.append(f"| `torch.compile` mode | **`{meta['compile_mode']}`** |")
    lines.append("| Compiled callable | `VLAFlowMatching.sample_actions` |")
    lines.append("| Env | `TORCH_COMPILE_DEBUG=1`, `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`, `TORCH_LOGS=output_code` |")
    lines.append(f"| Debug root | `{meta.get('debug_root')}` |")
    lines.append(f"| Graphs / debug dirs | **{meta.get('num_graphs')}** |")
    lines.append(f"| Chrome trace | `{OUT_TRACE.name}` |")
    lines.append("")
    lines.append(
        "Fusion of adjacent FX ops (e.g. `linear`+`add`, pointwise chains) happens in Inductor "
        "**after** the FX list. Kernel names with `_fused_` encode which aten origins were merged."
    )
    lines.append("")
    lines.append(
        "**Chrome trace note:** CUPTI may fail on some cloud GPUs "
        "(`CUPTI_ERROR_INVALID_DEVICE`); Triton kernel names below still appear via "
        "Kineto/profiler annotations when `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`."
    )
    lines.append("")
    lines.append(
        "**How to read vs FX list:** FX graph #5 had ~284 aten nodes; this file’s graph #5 "
        "shows ~16 Triton kernels (many `*_fused_*`) plus `extern_kernels.addmm` for GEMMs. "
        "FX `transpose`/`view`/`add` next to `linear` are typically epilogue-fused or "
        "metadata and do **not** each launch a kernel."
    )
    lines.append("")

    # Global histogram of triton kernels
    all_triton: Counter = Counter()
    all_extern: Counter = Counter()
    all_kinds: Counter = Counter()
    for g in payload["graphs"]:
        oc = g.get("output_code") or {}
        for k in oc.get("triton_kernels") or []:
            all_triton[k] += 1
        for k in oc.get("extern_kernels") or []:
            all_extern[k] += 1
        pf = g.get("post_fusion") or {}
        for kind, c in (pf.get("kind_histogram") or {}).items():
            all_kinds[kind] += c

    lines.append("### 0. Global fused-kernel summary")
    lines.append("")
    lines.append(
        f"**Post-fusion scheduler nodes (sum):** {sum(all_kinds.values())} · "
        f"**Distinct Triton kernel defs (sum over graphs):** {sum(all_triton.values())} · "
        f"**Extern kernel kinds:** {dict(all_extern)}"
    )
    lines.append("")
    lines.append("| Scheduler node kind | count |")
    lines.append("|---|---:|")
    for k, c in all_kinds.most_common():
        lines.append(f"| `{k}` | {c} |")
    lines.append("")

    chrome = payload.get("chrome_trace") or {}
    lines.append("#### Chrome-trace GPU kernel names (warm run, top 40)")
    lines.append("")
    lines.append(
        f"Unique named kernel events: **{chrome.get('unique_kernel_events', '?')}** · "
        f"total events: **{chrome.get('total_kernel_events', '?')}**"
    )
    lines.append("")
    lines.append("| count | kernel name |")
    lines.append("|---:|---|")
    for row in (chrome.get("kernels") or [])[:40]:
        lines.append(f"| {row['count']} | `{row['name'][:160]}` |")
    if not chrome.get("kernels"):
        lines.append("| — | *(no kernel events parsed — see JSON)* |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for g in payload["graphs"]:
        gid = g["graph_id"]
        meaning = g.get("meaning") or _GRAPH_MEANING.get(gid, "")
        lines.append(f"## {gid}. Fused Inductor graph #{gid} — {meaning.split('(')[0].strip().rstrip('.')}")
        lines.append("")
        lines.append(f"**Meaning:** {meaning}")
        lines.append("")
        lines.append(f"**Debug dir:** `{g.get('debug_dir')}`")
        lines.append("")
        pf = g.get("post_fusion") or {}
        oc = g.get("output_code") or {}
        lines.append(
            f"**Post-fusion nodes:** {pf.get('n_nodes', '?')} · "
            f"**Triton kernels:** {oc.get('n_triton', '?')} · "
            f"**Extern:** {oc.get('extern_kernels', [])}"
        )
        lines.append("")
        lines.append("### Scheduler nodes (`ir_post_fusion.txt`)")
        lines.append("")
        nodes = pf.get("nodes") or []
        # Prefer fused / extern first for readability; cap long graphs.
        fused = [n for n in nodes if n["kind"] == "FusedSchedulerNode"]
        extern_n = [n for n in nodes if n["kind"] == "ExternKernelSchedulerNode"]
        other = [n for n in nodes if n["kind"] not in ("FusedSchedulerNode", "ExternKernelSchedulerNode")]
        ordered = fused + extern_n + other
        max_rows = 120 if gid >= 8 else 200
        lines.append(
            f"Showing **{min(len(ordered), max_rows)}** / {len(ordered)} nodes "
            f"(fused={len(fused)}, extern={len(extern_n)}, other={len(other)}; "
            f"fused listed first)."
        )
        lines.append("")
        lines.append("| # | name | kind | #snodes | aten origins (from IR body) |")
        lines.append("|---:|---|---|---:|---|")
        for i, n in enumerate(ordered[:max_rows], 1):
            origins = ", ".join(f"`{o}`" for o in (n.get("aten_origins") or [])[:10]) or "—"
            if len(n.get("aten_origins") or []) > 10:
                origins += f" … (+{len(n['aten_origins']) - 10})"
            lines.append(
                f"| {i} | `{n['name']}` | `{n['kind']}` | {n.get('n_snodes', 0)} | {origins} |"
            )
        if len(ordered) > max_rows:
            lines.append(
                f"| … | … | … | … | *({len(ordered) - max_rows} more nodes — see "
                f"`ir_post_fusion.txt` / JSON)* |"
            )
        if not nodes:
            lines.append("| — | — | — | — | *(ir_post_fusion missing or unparsed)* |")
        lines.append("")

        lines.append("### Codegen kernels (`output_code.py`)")
        lines.append("")
        lines.append("| # | Triton / fused kernel | category | ops hint (from name) |")
        lines.append("|---:|---|---|---|")
        hints = {h["kernel"]: h for h in (oc.get("fused_name_hints") or [])}
        for i, kname in enumerate(oc.get("triton_kernels") or [], 1):
            h = hints.get(kname) or {}
            lines.append(
                f"| {i} | `{kname}` | `{h.get('category', '—')}` | `{h.get('ops_hint', '—')}` |"
            )
        if not oc.get("triton_kernels"):
            lines.append("| — | — | — | *(no triton_* parsed)* |")
        lines.append("")
        if oc.get("extern_kernels"):
            lines.append("Extern (cuBLAS / ATen) calls in this graph:")
            lines.append("")
            for e in oc["extern_kernels"]:
                lines.append(f"- `extern_kernels.{e}`")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## How to rebuild")
    lines.append("")
    lines.append("```bash")
    lines.append("export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home")
    lines.append("export SMOKE_COMPILE_MODE=reduce-overhead")
    lines.append("python src/smolvla_compile_fused_dump.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Outputs: `smolvla_compile_fused_kernels.json`, `smolvla_inductor_output_code.log`, "
        f"`{OUT_TRACE.name}`, `inductor_debug/`, and this markdown."
    )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

    # Fast path: re-parse existing debug dirs + chrome trace (no recompile).
    if os.environ.get("REGEN_ONLY", "").strip() in ("1", "true", "yes"):
        debug_dirs = _collect_debug_dirs(_DEBUG_ROOT)
        if not debug_dirs:
            print("REGEN_ONLY: no debug dirs under", _DEBUG_ROOT, flush=True)
            return 1
        debug_dirs = sorted(debug_dirs, key=lambda p: p.stat().st_mtime)
        # Keep last 8 matching model__*_inference_*.*
        debug_dirs = [d for d in debug_dirs if "inference" in d.name][-8:] or debug_dirs[-8:]
        graphs = [_summarize_debug_dir(d, i) for i, d in enumerate(debug_dirs, 1)]
        chrome = _parse_chrome_trace(OUT_TRACE)
        meta = {
            "model_id": MODEL_ID,
            "device": str(DEVICE),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" and torch.cuda.is_available() else None,
            "smoke_seed": SMOKE_SEED,
            "compile_mode": _COMPILE_MODE,
            "compiled_fn": "VLAFlowMatching.sample_actions",
            "num_graphs": len(graphs),
            "debug_root": str(_DEBUG_ROOT),
            "method": "REGEN_ONLY re-parse of TORCH_COMPILE_DEBUG dumps + chrome trace",
            "action_out_shape": None,
        }
        if OUT_JSON.exists():
            try:
                old = json.loads(OUT_JSON.read_text())
                meta = {**old.get("meta", {}), **meta, "action_out_shape": old.get("meta", {}).get("action_out_shape")}
            except Exception:
                pass
        payload = {"meta": meta, "chrome_trace": chrome, "graphs": graphs}
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        OUT_MD.write_text(_write_markdown(payload))
        print(f"REGEN wrote {OUT_MD} graphs={len(graphs)}", flush=True)
        return 0

    # Capture output_code to a dedicated log via TORCH_LOGS API.
    torch._logging.set_logs(output_code=True)
    # Also tee via env for any child paths
    os.environ["TORCH_LOGS"] = "output_code"

    log_fh = OUT_LOG.open("w")
    code_fh = OUT_CODE_LOG.open("w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"Loading {MODEL_ID} on {DEVICE}; compile_mode={_COMPILE_MODE}")
    log(f"TORCH_COMPILE_DEBUG_DIR={_DEBUG_ROOT}")

    # Clear previous debug dumps for a clean graph index
    if _DEBUG_ROOT.exists():
        for child in _DEBUG_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = str(DEVICE)
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

    torch.set_float32_matmul_precision("high")
    dynamo.reset()
    compiled = torch.compile(
        model.sample_actions,
        mode=_COMPILE_MODE,
        fullgraph=False,
    )

    t0 = time.time()
    log("Cold compiled sample_actions (triggers Inductor debug dumps)…")
    # Redirect inductor verbose output_code prints by capturing stderr? logging goes to stderr.
    import logging

    class _CodeFilter(logging.Handler):
        def emit(self, record):  # noqa: ANN001
            msg = record.getMessage()
            if "output_code" in msg.lower() or "triton_" in msg or "async_compile" in msg:
                code_fh.write(msg + "\n")

    root_logger = logging.getLogger("torch._inductor")
    handler = _CodeFilter()
    handler.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    out1 = compiled(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
    _sync()
    log(f"cold done in {time.time() - t0:.1f}s out={tuple(out1.shape)}")

    log("Warm run + chrome trace…")
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        out2 = compiled(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
        _sync()
    log(f"warm done out={tuple(out2.shape)}")
    try:
        prof.export_chrome_trace(str(OUT_TRACE))
        log(f"Wrote chrome trace {OUT_TRACE}")
    except Exception as exc:
        log(f"chrome trace export failed: {exc}")

    root_logger.removeHandler(handler)
    code_fh.flush()

    # Collect debug dirs created during this run (mtime after t0)
    debug_dirs = [
        d
        for d in _collect_debug_dirs(_DEBUG_ROOT)
        if d.stat().st_mtime >= t0 - 5
    ]
    # Fallback: all under DEBUG_ROOT
    if not debug_dirs:
        debug_dirs = _collect_debug_dirs(_DEBUG_ROOT)
    if not debug_dirs:
        # last resort: newest under /tmp
        debug_dirs = _collect_debug_dirs(Path("/tmp"))[-16:]

    log(f"Found {len(debug_dirs)} debug dirs")
    for d in debug_dirs:
        log(f"  {d}")

    # Sort by mtime = compile order ≈ graph order
    debug_dirs = sorted(debug_dirs, key=lambda p: p.stat().st_mtime)
    graphs = [_summarize_debug_dir(d, i) for i, d in enumerate(debug_dirs, 1)]

    chrome = _parse_chrome_trace(OUT_TRACE)

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
            "num_graphs": len(graphs),
            "debug_root": str(_DEBUG_ROOT),
            "method": (
                "TORCH_COMPILE_DEBUG=1 ir_post_fusion + output_code; "
                "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1; chrome trace warm run; "
                "TORCH_LOGS=output_code"
            ),
            "action_out_shape": list(out2.shape),
        },
        "chrome_trace": chrome,
        "graphs": graphs,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    md = _write_markdown(payload)
    OUT_MD.write_text(md)
    log(f"Wrote {OUT_JSON}")
    log(f"Wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")
    log(
        f"TOTAL graphs={len(graphs)} chrome_unique={chrome.get('unique_kernel_events')} "
        f"post_fusion_nodes={sum((g.get('post_fusion') or {}).get('n_nodes', 0) for g in graphs)}"
    )
    log_fh.close()
    code_fh.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

#!/usr/bin/env python
"""Normalize soft-wraps + Input/Output formatting in Fused CompileOp tables.

Input / Output rules:
  - One tensor: `[shape] dtype`
  - GEMM / addmm / mm / bmm / conv / affine: operands joined with ` × `
  - Other multi-buffer fused ops: joined with ` ; ` (never raw `|` — Markdown splits columns)
  - Missing slot: standalone `—` (never glued as `—[`)
  - Soft-wrap: `` `part` ``<br>`` `; part` `` / `` `× part` `` (separator on the new line)
  - LayerNorm split (Inductor): red = act→2×stats; per = stats→mean/rstd;
    poi = x×mean×rstd×w×b→y; full fused writing hidden keeps act(+res)×w×b→y

Usage:
  python src/smolvla_compile_fused_table_wrap.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md"

NAME_W = 40
LIST_W = 52
DTYPE = r"(?:float32|float16|bfloat16|int64|int32|int8|bool)"
DTYPE_CANON = {
    "f32": "float32", "fp32": "float32", "float32": "float32",
    "f16": "float16", "fp16": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "i64": "int64", "int64": "int64",
    "i32": "int32", "int32": "int32",
    "i8": "int8", "int8": "int8",
    "bool": "bool",
}


def split_fields(line: str):
    body = line[1:-1]
    fields, buf, in_bt = [], [], False
    for ch in body:
        if ch == "`":
            in_bt = not in_bt
            buf.append(ch)
        elif ch == "|" and not in_bt:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    fields.append("".join(buf).strip())
    return fields


def join_fields(fields):
    return "| " + " | ".join(fields) + " |"


def strip_br(s: str) -> str:
    return s.replace("<br>", "")


def bare_name(cell: str) -> str:
    parts = re.findall(r"`([^`]*)`", strip_br(cell))
    return "".join(parts) if parts else strip_br(cell).strip("`")


def wrap_name(cell: str, width: int = NAME_W) -> str:
    s = strip_br(cell).strip()
    parts = re.findall(r"`([^`]*)`", s)
    if not parts:
        return s
    ident = "".join(parts)
    prefix = s[: s.find("`")] if "`" in s else ""
    suffix = s[s.rfind("`") + 1 :] if "`" in s else ""
    if len(ident) <= width:
        return f"{prefix}`{ident}`{suffix}"
    tokens, buf = [], ""
    for ch in ident:
        buf += ch
        if ch == "_":
            tokens.append(buf)
            buf = ""
    if buf:
        tokens.append(buf)
    chunks, cur = [], ""
    for tok in tokens:
        if not cur:
            cur = tok
        elif len(cur) + len(tok) <= width:
            cur += tok
        else:
            chunks.append(cur)
            cur = tok
    if cur:
        chunks.append(cur)
    if len(chunks) >= 2 and len(chunks[-1]) < 8:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return prefix + "<br>".join(f"`{c}`" for c in chunks) + suffix


def is_gemmish(name: str, kind: str) -> bool:
    n = name.lower()
    return any(x in n for x in ("addmm", ".mm", ".bmm", "convolution")) or (
        kind == "extern" and any(x in n for x in ("mm", "bmm", "conv"))
    )


def extract_io_tokens(cell: str) -> list[str]:
    if not cell or cell.strip() in ("—", "-"):
        return []
    s = strip_br(cell).replace("`", "")
    s = s.replace("–", "—")
    # missing sep: dtype[shape] or ][
    s = re.sub(rf"({DTYPE})\s*(?=\[)", r"\1 | ", s, flags=re.I)
    s = re.sub(r"\]\s*\[", "] | [", s)
    s = re.sub(r"—\s*\[", "— | [", s)
    s = re.sub(r"\]\s*—", "] | —", s)
    s = re.sub(r"—{2,}", "—", s)
    s = s.replace(" × ", " | ").replace("×", " | ").replace(" + ", " | ")
    s = s.replace("; ", " | ").replace(";", " | ")
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"( \| ){2,}", " | ", s).strip(" |")

    tokens: list[str] = []
    for part in s.split(" | "):
        part = part.strip()
        if not part or part in ("—", "-"):
            if not tokens or tokens[-1] != "—":
                tokens.append("—")
            continue
        for m in re.finditer(rf"\[([^\]]*)\]\s*({DTYPE})?", part, flags=re.I):
            dims = m.group(1).strip()
            dt = (m.group(2) or "").lower()
            if not dims:
                continue
            try:
                tuple(int(x.strip()) for x in dims.split(",") if x.strip() != "")
            except ValueError:
                continue
            if dt:
                dt = DTYPE_CANON.get(dt, dt)
                tok = f"[{dims}] {dt}"
            else:
                tok = f"[{dims}]"
            tokens.append(tok)
    while tokens and tokens[0] == "—":
        tokens.pop(0)
    while tokens and tokens[-1] == "—":
        tokens.pop()
    return tokens


def fill_dtypes(tokens: list[str]) -> list[str]:
    dts = []
    for t in tokens:
        m = re.search(rf"\]\s*({DTYPE})$", t)
        if m:
            dts.append(m.group(1))
    if not dts:
        return tokens
    default, last = dts[0], dts[0]
    out = []
    for t in tokens:
        if t == "—":
            out.append(t)
            continue
        m = re.search(rf"\]\s*({DTYPE})$", t)
        if m:
            last = m.group(1)
            out.append(t)
        else:
            out.append(f"{t} {last}")
    # forward fill from later if still missing
    for i, t in enumerate(out):
        if t != "—" and not re.search(rf"\]\s*{DTYPE}$", t):
            fwd = next(
                (
                    re.search(rf"\]\s*({DTYPE})$", out[j]).group(1)
                    for j in range(i + 1, len(out))
                    if out[j] != "—" and re.search(rf"\]\s*{DTYPE}$", out[j])
                ),
                default,
            )
            out[i] = f"{t} {fwd}"
    return out


def wrap_tokens(tokens: list[str], sep: str, width: int = LIST_W) -> str:
    if not tokens:
        return "—"
    if len(tokens) == 1 and tokens[0] == "—":
        return "—"
    # Never use raw '|' inside cells — Markdown tables split on it before backticks.
    if sep.strip() == "|":
        sep = "; "
    body = sep.join(tokens)
    if len(body) <= width:
        return f"`{body}`"
    lines, cur, cur_len = [], [], 0
    for t in tokens:
        add = len(t) if not cur else len(sep) + len(t)
        if cur and cur_len + add > width:
            lines.append(sep.join(cur))
            cur, cur_len = [t], len(t)
        else:
            cur.append(t)
            cur_len += add
    if cur:
        lines.append(sep.join(cur))
    out = [f"`{lines[0]}`"]
    for ln in lines[1:]:
        out.append(f"`{sep.strip()} {ln}`")
    return "<br>".join(out)


def _ln_dims(t: str):
    m = re.match(r"\[([^\]]+)\]", t)
    if not m:
        return None
    try:
        return tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip() != "")
    except ValueError:
        return None


def _is_ln_stats(t: str) -> bool:
    """Partial/final LN mean|rstd buffers: last dim ≤6, float32."""
    if t == "—":
        return False
    dims = _ln_dims(t)
    if not dims or dims[-1] > 6:
        return False
    dt = re.search(rf"\]\s*({DTYPE}|f32|fp32)$", t, flags=re.I)
    return bool(dt and dt.group(1).lower() in ("float32", "f32", "fp32"))


def _is_ln_act(t: str) -> bool:
    dims = _ln_dims(t)
    return bool(dims and len(dims) >= 2 and dims[-1] >= 32)


def _is_ln_affine(t: str) -> bool:
    dims = _ln_dims(t)
    return bool(dims and len(dims) == 1 and dims[0] in (768, 960, 720, 2048, 512, 256))


def normalize_layer_norm_io(name: str, inn_cell: str, out_cell: str) -> tuple[str, str]:
    """Role-correct Input/Output for Inductor-split and fused LayerNorm kernels."""
    if "layer_norm" not in name:
        return inn_cell, out_cell

    inn = fill_dtypes(extract_io_tokens(inn_cell))
    out = fill_dtypes(extract_io_tokens(out_cell))
    acts_in = [t for t in inn if _is_ln_act(t)]
    stats_in = [t for t in inn if _is_ln_stats(t)]
    aff_in = [t for t in inn if _is_ln_affine(t)][:2]
    acts_out = [t for t in out if _is_ln_act(t)]
    stats_out = [t for t in out if _is_ln_stats(t)]

    # --- red: var_mean → write 2× partial stats (Inductor parks scratch under Input)
    if "triton_red_fused" in name and not acts_out:
        stats = list(stats_in)
        for t in stats_out:
            if t not in stats:
                stats.append(t)
        if len(stats) == 1:
            stats = [stats[0], stats[0]]
        elif len(stats) > 2:
            # keep first two distinct shapes; else duplicate first
            stats = stats[:2] if stats[0] != stats[1] else [stats[0], stats[0]]
        if not stats and stats_out:
            stats = [stats_out[0], stats_out[0]]
        primary = acts_in[:1] if acts_in else ([inn[0]] if inn else [])
        if primary and stats:
            return wrap_tokens(primary, "; "), wrap_tokens(stats, "; ")

    # --- red that also materializes hidden (add+LN fused reduction writing y)
    if "triton_red_fused" in name and acts_out:
        toks = acts_in + aff_in
        if not toks:
            toks = [t for t in inn if t != "—"]
        y = acts_out[:1]
        if toks and y:
            return wrap_tokens(toks, "; "), wrap_tokens(y, "; ")

    # --- per: refine partial stats → final mean/rstd (no hidden I/O)
    if "triton_per_fused" in name and not acts_out and not acts_in:
        stats = list(stats_in) or [t for t in inn if t != "—"]
        # drop trailing final-mean scratch often listed as last input
        stats = [t for t in stats if _is_ln_stats(t) or _ln_dims(t)]
        # prefer two partial buffers as inputs
        partial = [t for t in stats if _ln_dims(t) and _ln_dims(t)[-1] > 1][:2]
        if not partial:
            partial = stats[:2]
        if len(partial) == 1:
            partial = [partial[0], partial[0]]
        finals = list(stats_out)
        if not finals:
            # synthesize [1,T,1] from partial [1,T,1,K]
            if partial:
                d = _ln_dims(partial[0])
                if d and len(d) >= 3:
                    finals = [f"[{d[0]},{d[1]},1] float32", f"[{d[0]},{d[1]},1] float32"]
        if len(finals) == 1:
            finals = [finals[0], finals[0]]
        if partial and finals:
            return wrap_tokens(partial[:2], "; "), wrap_tokens(finals[:2], "; ")

    # --- per: full LN (act + affine → y)
    if "triton_per_fused" in name and (acts_out or acts_in):
        toks = acts_in + aff_in
        if not toks:
            toks = [t for t in inn if t != "—" and not _is_ln_stats(t)]
        # ensure both weight and bias when one affine present and hidden D known
        if len(aff_in) == 1 and acts_in:
            d = _ln_dims(acts_in[0])
            if d:
                toks = acts_in + [aff_in[0], f"[{d[-1]}] bfloat16"]
        y = acts_out[:1] if acts_out else (
            [acts_in[0]] if acts_in else [t for t in out if t != "—"][:1]
        )
        y = [
            (t if re.search(rf"\]\s*({DTYPE})$", t) else f"{t} bfloat16")
            for t in y
        ]
        if toks and y:
            return wrap_tokens(toks, " × "), wrap_tokens(y, "; ")

    # --- poi: epilogue x × mean × rstd × w × b → y
    if "triton_poi_fused" in name:
        mean_rstd = [t for t in inn if _is_ln_stats(t)]
        # also accept [1,T,1] float32
        if len(mean_rstd) < 2:
            extra = [
                t for t in inn
                if t not in mean_rstd
                and _ln_dims(t)
                and len(_ln_dims(t)) == 3
                and _ln_dims(t)[-1] == 1
            ]
            mean_rstd = (mean_rstd + extra)[:2]
        # Inductor dump sometimes omits mean/rstd — synthesize from act shape
        if len(mean_rstd) < 2 and acts_in:
            d = _ln_dims(acts_in[0])
            if d and len(d) >= 2:
                mean_rstd = [
                    f"[{d[0]},{d[1]},1] float32",
                    f"[{d[0]},{d[1]},1] float32",
                ]
        toks = acts_in[:1] + mean_rstd[:2] + aff_in
        if acts_in and len([t for t in toks if _is_ln_affine(t)]) < 2:
            d = _ln_dims(acts_in[0])
            if d:
                aff = [t for t in toks if _is_ln_affine(t)]
                while len(aff) < 2:
                    aff.append(f"[{d[-1]}] bfloat16")
                toks = acts_in[:1] + mean_rstd[:2] + aff[:2]
        y = acts_out[:1] if acts_out else (
            [acts_in[0]] if acts_in else [t for t in out if t != "—"][:1]
        )
        # ensure y has dtype
        y = fill_dtypes(y) if y else y
        if toks and y:
            return wrap_tokens([t for t in toks if t != "—"], " × "), wrap_tokens(y, "; ")

    return inn_cell, out_cell


def format_io_cell(cell: str, name: str, kind: str, gflops: str, is_out: bool) -> str:
    # preserve G7 embedding special phrasing
    if (not is_out) and "embedding_0" in name and "ids" in cell:
        return "`[49280,960] bfloat16 × ids [1,48] int64`"

    tokens = fill_dtypes(extract_io_tokens(cell))
    if is_out:
        tokens = [t for t in tokens if t != "—"]
        return wrap_tokens(tokens, "; ")

    gemm = is_gemmish(name, kind)
    affine = (
        ("poi_fused" in name and "layer_norm" in name)
        or ("add_embedding" in name)
    )
    sep = " × " if (gemm or affine) else "; "
    if gemm or affine:
        tokens = [t for t in tokens if t != "—"]

    # addmm with 1D FLOPs → ensure bias vector present
    if "addmm" in name and "1D" in gflops:
        real = [t for t in tokens if t != "—"]
        if len(real) == 2:
            m = re.search(r"\[([^\]]+)\]", real[0])
            # bias length = out features ≈ weight N = second mat's dim0 or dim1
            wm = re.search(r"\[(\d+),\s*(\d+)\]", real[1])
            n = wm.group(1) if wm else (m.group(1).split(",")[-1].strip() if m else None)
            dt_m = re.search(rf"\]\s*({DTYPE})$", real[0])
            dt = dt_m.group(1) if dt_m else "bfloat16"
            if n:
                tokens = real + [f"[{n}] {dt}"]
        sep = " × "

    return wrap_tokens(tokens, sep)


def wrap_fused(cell: str, width: int = LIST_W) -> str:
    s = strip_br(cell).strip()
    if not s or s == "—":
        return "—"
    s = re.sub(r"``+", "`, `", s)
    items = [it.strip() for it in re.findall(r"`([^`]+)`", s) if it.strip()]
    if not items:
        return "—"
    # Split accidental glue inside a single tick using JSON-like known boundaries
    expanded = []
    for it in items:
        if re.search(
            r"(mean|type|clone|view|embedding|permute|rsqrt|silu|gelu|softmax|convert|mul|add|sub|pow|cat|slice)"
            r"(?=convert|clone|view|embedding|permute|rsqrt|mul|add|var_|mean|pow|silu|gelu)",
            it,
        ):
            # cannot safely split without source list — keep as-is; restore script fixes from JSON
            expanded.append(it)
        else:
            expanded.append(it)
    items = [f"`{it}`" for it in expanded]
    lines, cur, cur_len = [], [], 0
    for it in items:
        extra = 0 if not cur else 2
        if cur and cur_len + extra + len(it) > width:
            lines.append(", ".join(cur))
            cur, cur_len = [it], len(it)
        else:
            cur.append(it)
            cur_len += extra + len(it)
    if cur:
        lines.append(", ".join(cur))
    if len(lines) == 1:
        return lines[0]
    # trailing comma on every line but last so soft-wrap stays valid lists
    return "<br>".join(
        (ln + "," if i < len(lines) - 1 else ln) for i, ln in enumerate(lines)
    )


def clean_meaning(cell: str) -> str:
    s = strip_br(cell)
    s = re.sub(r":\s*:\s*", "::", s)
    s = re.sub(r"::\s+", "::", s)
    s = re.sub(r"\s*…\s*", "…", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<!:):(?!:)(?=\S)", ": ", s)
    s = re.sub(r":\s*:\s*", "::", s)
    s = re.sub(r",(?=\S)", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()

    def fix_code(m):
        inner = m.group(1)
        inner = re.sub(r":\s*:\s*", "::", inner)
        inner = re.sub(r"::\s+", "::", inner)
        inner = re.sub(r"\s+::", "::", inner)
        inner = re.sub(r"\s*…\s*", "…", inner)
        return f"`{inner}`"

    return re.sub(r"`([^`]+)`", fix_code, s)


def wrap_meaning(cell: str, width: int = LIST_W) -> str:
    s = clean_meaning(cell)
    s = re.sub(r"fusion:(?=\S)", "fusion: ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= width:
        return s
    # Prefer breaks after ", " so the comma stays on the previous line
    chunks, cur = [], ""
    i = 0
    while i < len(s):
        # take next segment up to and including ", " if present, else a word
        if s[i] == "`":
            j = s.find("`", i + 1)
            if j < 0:
                cur += s[i:]
                break
            token = s[i : j + 1]
            if cur and len(cur) + len(token) > width:
                chunks.append(cur.rstrip())
                cur = token
            else:
                cur += token
            i = j + 1
            continue
        j = s.find(", ", i)
        if j >= i:
            piece = s[i : j + 2]  # include ", "
            nxt_i = j + 2
        else:
            # no more commas: take rest as words
            k = i
            while k < len(s) and s[k] == " ":
                k += 1
            m = k
            while m < len(s) and s[m] not in " ,`":
                m += 1
            piece = s[i:m]
            nxt_i = m
        if cur and len(cur) + len(piece) > width and cur.strip():
            chunks.append(cur.rstrip())
            cur = piece.lstrip()
        else:
            cur += piece
        i = nxt_i
    if cur:
        chunks.append(cur.rstrip())
    cleaned = []
    for c in chunks:
        if cleaned and len(c) <= 2 and c.strip() in ")].":
            cleaned[-1] += c
        else:
            cleaned.append(c)
    return "<br>".join(cleaned)


def main() -> int:
    import json

    json_path = PATH.parent / "smolvla_compile_fused_kernels.json"
    by_g: dict[int, dict[int, dict]] = {}
    by_g_name: dict[int, dict[str, dict]] = {}
    # Pre-seed LN templates from first complete JSON dump of each kernel name
    ln_templates: dict[str, tuple[list[str], list[str]]] = {}
    if json_path.exists():
        data = json.loads(json_path.read_text())
        graphs = data["graphs"] if isinstance(data, dict) and "graphs" in data else data
        by_g = {g["graph_id"]: {e["i"]: e for e in g["ops"]} for g in graphs}
        by_g_name = {}
        for g in graphs:
            namemap: dict[str, dict] = {}
            for e in g["ops"]:
                nm = e.get("name") or ""
                if nm and nm not in namemap:
                    namemap[nm] = e
            by_g_name[g["graph_id"]] = namemap
        for g in graphs:
            for e in g["ops"]:
                nm = e.get("name") or ""
                if "layer_norm" not in nm or nm in ln_templates:
                    continue
                inn = fill_dtypes(extract_io_tokens(e.get("input_str") or ""))
                out = fill_dtypes(extract_io_tokens(e.get("output_str") or ""))
                real = [t for t in inn if t != "—"]
                if not real or not out:
                    continue
                # require completeness by class
                if "poi_fused" in nm and len(real) < 3:
                    continue
                if "per_fused" in nm and any(_is_ln_act(t) for t in out) and len(real) < 2:
                    continue
                nin, nout = normalize_layer_norm_io(
                    nm,
                    wrap_tokens(real, "; "),
                    wrap_tokens([t for t in out if t != "—"], "; "),
                )
                ln_templates[nm] = (
                    extract_io_tokens(nin),
                    extract_io_tokens(nout),
                )

    lines = PATH.read_text().splitlines()
    out, n = [], 0
    section = ""
    section_g = None
    for line in lines:
        m = re.match(r"^## (\d+)\.", line)
        if line.startswith("## Pre"):
            section, section_g = line, None
        elif m:
            section, section_g = line, int(m.group(1))
        elif line.startswith("## "):
            section, section_g = line, None

        if not re.match(r"^\| \d+ \|", line):
            out.append(line)
            continue
        f = split_fields(line)
        if len(f) != 17 and len(f) != 16:
            out.append(line)
            continue
        if len(f) == 16:
            f = f[:3] + ["—"] + f[3:]
        if not f[0].isdigit():
            out.append(line)
            continue
        name = bare_name(f[1])
        kind = f[2].strip("`")
        order = int(f[0])
        f[1] = wrap_name(f[1])

        # Restore meaning / fused aten from Inductor dump when available.
        # Prefer name match: after G8 representative collapse, Order is renumbered
        # and no longer equals JSON `i`.
        e = None
        if section_g and section_g in by_g_name and name in by_g_name[section_g]:
            e = by_g_name[section_g][name]
        elif section_g and section_g in by_g and order in by_g[section_g]:
            e = by_g[section_g][order]
        if e is not None:
            if e.get("meaning"):
                f[15] = e["meaning"]
            if e.get("fused_aten"):
                f[16] = ", ".join(f"`{x}`" for x in e["fused_aten"])
            js_in = e.get("input_str") or ""
            js_out = e.get("output_str") or ""
            # Prefer JSON I/O for LayerNorm (role rewrite below) or when richer
            if "layer_norm" in name:
                js_real = [t for t in extract_io_tokens(js_in) if t != "—"]
                tpl = ln_templates.get(name)
                tpl_n = len([t for t in tpl[0] if t != "—"]) if tpl else 0
                if tpl and len(js_real) < max(2, tpl_n - 1):
                    # sparse Inductor dump — keep template; don't clobber with dashes
                    pass
                else:
                    if js_in.strip():
                        f[4] = js_in
                    if js_out.strip() and "[" in js_out:
                        f[5] = js_out
            else:
                md_in = "".join(re.findall(r"`([^`]*)`", f[4]))
                if js_in.count("[") > md_in.count("[") and not is_gemmish(name, kind):
                    f[4] = js_in

        f[4] = format_io_cell(f[4], name, kind, f[9], is_out=False)
        f[5] = format_io_cell(f[5], name, kind, f[9], is_out=True)

        if "layer_norm" in name:
            # Fill sparse / wrong-role later-block dumps from first complete template
            inn_toks = extract_io_tokens(f[4])
            out_toks = extract_io_tokens(f[5])
            real_in = [t for t in inn_toks if t != "—"]
            if name in ln_templates:
                tin, tout = ln_templates[name]
                tin_real = [t for t in tin if t != "—"]
                out_hid = any(_is_ln_act(t) for t in (out_toks + tout))
                in_has_act = any(_is_ln_act(t) for t in real_in)
                need_tpl = (
                    len(real_in) < len(tin_real)
                    or (out_hid and any(_is_ln_act(t) for t in tin_real) and not in_has_act)
                    or (out_hid and any(_is_ln_affine(t) for t in tin_real) and not any(_is_ln_affine(t) for t in real_in))
                )
                if need_tpl and tin_real:
                    sep = (
                        " × "
                        if (
                            "poi_fused" in name
                            or (
                                "per_fused" in name
                                and any(_is_ln_act(t) for t in tin_real)
                            )
                        )
                        else "; "
                    )
                    f[4] = wrap_tokens(fill_dtypes(tin_real), sep)
                if (
                    not out_toks
                    or all(t == "—" for t in out_toks)
                    or (
                        any(_is_ln_act(t) for t in tout)
                        and not any(_is_ln_act(t) for t in out_toks)
                    )
                ):
                    f[5] = wrap_tokens(fill_dtypes([t for t in tout if t != "—"]), "; ")
            f[4], f[5] = normalize_layer_norm_io(name, f[4], f[5])
            # force dtype on hidden outputs
            ot = extract_io_tokens(f[5])
            ot2 = []
            for t in ot:
                if _is_ln_act(t) and not re.search(rf"\]\s*({DTYPE})$", t):
                    ot2.append(f"{t} bfloat16")
                else:
                    ot2.append(t)
            if ot2:
                f[5] = wrap_tokens(fill_dtypes(ot2), "; ")
            # remember first complete template
            nt_in, nt_out = extract_io_tokens(f[4]), extract_io_tokens(f[5])
            n_real = [t for t in nt_in if t != "—"]
            if len(n_real) >= 1 and nt_out and name not in ln_templates:
                ok = True
                if "poi_fused" in name and len(n_real) < 4:
                    ok = False
                if "per_fused" in name and any(_is_ln_act(t) for t in nt_out) and len(n_real) < 2:
                    ok = False
                if ok:
                    ln_templates[name] = (nt_in, nt_out)

        if section.startswith("## 4.") and "add_embedding" in name:
            f[4] = "`[1,1024,768] bfloat16 × [1,1024,768] bfloat16`"
            f[5] = "`[1,1024,768] bfloat16`"
        if section.startswith("## 7.") and name.endswith("embedding_0"):
            f[4] = "`[49280,960] bfloat16 × ids [1,48] int64`"
            f[5] = "`[1,48,960] bfloat16`"
        f[15] = wrap_meaning(f[15])
        f[16] = wrap_fused(f[16])
        out.append(join_fields(f))
        n += 1
    PATH.write_text("\n".join(out) + "\n")
    print(f"Wrapped/normalized {n} rows in {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

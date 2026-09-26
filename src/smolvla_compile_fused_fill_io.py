#!/usr/bin/env python
"""Infer missing Input/Output cells on Fused CompileOp tables.

Sources (in priority order):
  1. Same-kernel complete exemplars already in the fused table
  2. Algebraic GEMM / elementwise rules (mm/bmm/silu/rmsnorm/softmax)
  3. Upstream/downstream neighbors in the same graph section
  4. FX CompileOp list (SmolVLA_CompileOp_List_gpu_backend.md) for the same stage
  5. Hand rules for known generators / G4 index_put_

Generator ops (full / arange / linspace-only) keep Input=`—` — matches FX.

Usage:
  python src/smolvla_compile_fused_fill_io.py
  python src/smolvla_compile_fused_table_metrics.py
  python src/smolvla_compile_fused_table_timing.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md"
FX_PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_CompileOp_List_gpu_backend.md"

DTYPE = r"(?:float32|float16|bfloat16|int64|int32|int8|bool|float64)"
LIST_W = 52


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


def bare_name(cell: str) -> str:
    return "".join(re.findall(r"`([^`]*)`", cell.replace("<br>", "")))


def is_empty(cell: str) -> bool:
    s = cell.replace("<br>", "").replace("`", "").strip()
    if s in ("", "—", "-"):
        return True
    parts = re.split(r"[;×|]", s)
    return all(p.strip() in ("", "—", "-") for p in parts)


def extract_tokens(cell: str) -> list[str]:
    if is_empty(cell):
        return []
    s = cell.replace("<br>", " ").replace("`", "")
    s = s.replace(" × ", " | ").replace("×", " | ").replace("; ", " | ").replace(";", " | ")
    s = re.sub(r"\s*\|\s*", " | ", s).strip(" |")
    out = []
    for part in s.split(" | "):
        part = part.strip()
        if not part or part in ("—", "-"):
            continue
        m = re.match(rf"^(\[[^\]]*\])\s*({DTYPE})?$", part, flags=re.I)
        if m:
            shape, dt = m.group(1), (m.group(2) or "").lower()
            out.append(f"{shape} {dt}".strip() if dt else shape)
        else:
            # keep "ids [1,48] int64" style
            out.append(part)
    return out


def parse_dims(tok: str):
    m = re.search(r"\[([^\]]*)\]", tok)
    if not m:
        return None
    try:
        return tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip() != "")
    except ValueError:
        return None


def parse_dtype(tok: str, default: str | None = None) -> str | None:
    m = re.search(rf"\]\s*({DTYPE})\b", tok, flags=re.I)
    if m:
        return m.group(1).lower()
    return default


def fmt_tok(dims, dt: str | None) -> str:
    shape = "[" + ",".join(str(d) for d in dims) + "]"
    return f"{shape} {dt}" if dt else shape


def wrap_tokens(tokens: list[str], sep: str) -> str:
    if not tokens:
        return "—"
    body = sep.join(tokens)
    if len(body) <= LIST_W:
        return f"`{body}`"
    lines, cur, cur_len = [], [], 0
    for t in tokens:
        add = len(t) if not cur else len(sep) + len(t)
        if cur and cur_len + add > LIST_W:
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


def fill_missing_dtypes(tokens: list[str], default: str = "bfloat16") -> list[str]:
    dts = [parse_dtype(t) for t in tokens if parse_dtype(t)]
    last = dts[0] if dts else default
    out = []
    for t in tokens:
        d = parse_dtype(t)
        if d:
            last = d
            out.append(t)
        else:
            dims = parse_dims(t)
            if dims is None:
                out.append(t)
            else:
                out.append(fmt_tok(dims, last))
    return out


def gemm_out(tokens: list[str]):
    """Infer mm/bmm output from two matrix operands."""
    mats = []
    for t in tokens:
        d = parse_dims(t)
        if d and len(d) in (2, 3):
            mats.append((d, parse_dtype(t, "bfloat16")))
    if len(mats) < 2:
        return None
    a, da = mats[0]
    b, db = mats[1]
    dt = da or db or "bfloat16"
    if len(a) == 2 and len(b) == 2:
        M, K1 = a
        K2, N = b
        if K1 != K2 and K1 == N:
            # weight stored as [N,K]
            N, K2 = K2, N
        return fmt_tok((M, N), dt)
    if len(a) == 3 and len(b) == 3:
        B, M, K1 = a
        B2, K2, N = b
        return fmt_tok((B, M, N), dt)
    return None


def primary_act(tokens: list[str]):
    best = None
    best_n = -1
    for t in tokens:
        d = parse_dims(t)
        if not d:
            continue
        n = 1
        for x in d:
            n *= x
        if n > best_n and d[-1] >= 32:
            best, best_n = t, n
    return best


def load_fx_hints() -> dict[str, list[tuple[str, str]]]:
    """Map coarse op keywords → list of (in, out) from FX tables."""
    hints: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not FX_PATH.exists():
        return hints
    section_g = None
    for line in FX_PATH.read_text().splitlines():
        m = re.match(r"^## (\d+)\.", line)
        if m:
            section_g = int(m.group(1))
            continue
        if not re.match(r"^\| \d+ \|", line):
            continue
        # FX tables are 4-col: Seq | op | Input | Output
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        op, inn, out = parts[1], parts[2], parts[3]
        op_b = op.strip("`")
        hints[op_b].append((inn, out))
        # also key by basename
        base = op_b.split("::")[-1]
        hints[base].append((inn, out))
    return hints


def normalize_fx_cell(cell: str, sep: str = "; ") -> str:
    if not cell or cell.strip() in ("—", "-", "`—`"):
        return "—"
    s = cell.strip().strip("`")
    s = s.replace(" | ", sep).replace("|", sep)
    # drop spaces inside shapes like [1, 32, 32] → [1,32,32]
    def tight(m):
        dims = ",".join(x.strip() for x in m.group(1).split(","))
        return f"[{dims}]"

    s = re.sub(r"\[([^\]]+)\]", tight, s)
    toks = [t.strip() for t in s.split(sep) if t.strip() and t.strip() != "—"]
    return wrap_tokens(toks, sep) if toks else "—"


def main() -> int:
    fx = load_fx_hints()
    lines = PATH.read_text().splitlines()

    # Pass 1: parse rows with section context
    parsed = []  # (line_idx, section, section_g, fields)
    section, section_g = "", None
    for i, line in enumerate(lines):
        if line.startswith("## Pre"):
            section, section_g = line, 0
        elif (m := re.match(r"^## (\d+)\.", line)):
            section, section_g = line, int(m.group(1))
        elif line.startswith("## "):
            section, section_g = line, None
        if not re.match(r"^\| \d+ \|", line):
            continue
        f = split_fields(line)
        if len(f) != 17 and len(f) != 16:
            continue
        if len(f) == 16:
            f = f[:3] + ["—"] + f[3:]
        if len(f) != 17 or not f[0].isdigit():
            continue
        parsed.append((i, section, section_g, f))

    # Exemplars by kernel name
    exemplars: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
    for _, _, _, f in parsed:
        kn = bare_name(f[1])
        inn, out = extract_tokens(f[5]), extract_tokens(f[5])
        if inn and out:
            exemplars[kn].append((inn, out))

    filled_in = filled_out = 0
    notes = defaultdict(int)

    def set_in(f, tokens, sep="; "):
        nonlocal filled_in
        tokens = fill_missing_dtypes(tokens)
        f[4] = wrap_tokens(tokens, sep)
        filled_in += 1

    def set_out(f, tokens, sep="; "):
        nonlocal filled_out
        tokens = fill_missing_dtypes(tokens)
        f[5] = wrap_tokens(tokens, sep)
        filled_out += 1

    # Group by section for neighbor walks
    by_sec: dict[str, list[tuple[int, list]]] = defaultdict(list)
    for idx, (li, sec, sg, f) in enumerate(parsed):
        by_sec[sec].append((idx, f))

    for sec, items in by_sec.items():
        for pos, (pidx, f) in enumerate(items):
            kn = bare_name(f[1])
            kind = f[2].strip("`")
            fused = f[16].replace("<br>", " ").replace("`", "")
            need_in, need_out = is_empty(f[4]), is_empty(f[5])
            inn = extract_tokens(f[4])
            out = extract_tokens(f[5])

            # --- Hand rules / FX (early graphs) ---
            if kn == "aten::index_put_" and (need_in or need_out):
                # G4 FX setitem on position_ids
                set_in(
                    f,
                    ["[1,1024] int64", "[1,1024] bool", "[1024] int64"],
                    "; ",
                )
                set_out(f, ["[1,1024] int64"])
                notes["index_put_fx"] += 1
                need_in = need_out = False
                inn, out = extract_tokens(f[4]), extract_tokens(f[5])

            if ("full_default" in fused or kn.endswith("_full_6") or "full_default" in kn) and need_in:
                # ones/full: no tensor inputs (FX also —)
                notes["generator_full"] += 1
                need_in = False

            if ("arange" in kn or "iota" in fused) and need_in and not inn:
                notes["generator_arange"] += 1
                need_in = False

            if "linspace" in fused and need_in and not inn:
                # rope freq generators
                notes["generator_linspace"] += 1
                need_in = False

            if "cumsum_lift_fresh" in kn and need_in:
                # FX: cumsum on [1,50] float32 / bool → [1,50]
                ot = extract_tokens(f[5])
                if ot and parse_dims(ot[0]) == (1, 50):
                    dt = parse_dtype(ot[0], "float32")
                    set_in(f, [fmt_tok((1, 50), dt)])
                    notes["cumsum_fx"] += 1
                    need_in = False
                    inn = extract_tokens(f[4])

            # --- Fill missing dtypes on partial IN ---
            if inn and any(parse_dtype(t) is None for t in inn):
                inn = fill_missing_dtypes(inn)
                sep = " × " if (kind == "extern" or "silu" in kn or "addmm" in kn) else "; "
                if " × " in f[4] or "×" in f[4].replace("<br>", ""):
                    sep = " × "
                f[4] = wrap_tokens(inn, sep)
                notes["dtype_fill"] += 1

            # --- GEMM OUT ---
            if need_out and kn in ("extern_kernels.mm", "extern_kernels.bmm", "extern_kernels.addmm"):
                go = gemm_out(inn)
                if go:
                    set_out(f, [go])
                    notes["gemm_out"] += 1
                    need_out = False
                    out = extract_tokens(f[5])

            # --- Conv2d OUT (NCHW, assume stride=kernel for ViT patch embed) ---
            if need_out and kn == "extern_kernels.convolution" and len(inn) >= 2:
                da, db = parse_dims(inn[0]), parse_dims(inn[1])
                dta = parse_dtype(inn[0], "bfloat16")
                if da and db and len(da) == 4 and len(db) == 4:
                    # act [N,C_in,H,W] × weight [C_out,C_in,kH,kW]
                    n, _c, h, w = da
                    c_out, _ci, kh, kw = db
                    # Inductor ViT patch embed: stride == kernel → H/kH, W/kW
                    oh, ow = (h // kh if kh else h), (w // kw if kw else w)
                    set_out(f, [fmt_tok((n, c_out, oh, ow), dta)])
                    notes["conv_out"] += 1
                    need_out = False
                    out = extract_tokens(f[5])

            # --- Exemplar copy (same kernel) ---
            if (need_in or need_out) and exemplars.get(kn):
                # prefer exemplar whose token count / trailing dims match partial info
                best = None
                best_score = -1
                for ein, eout in exemplars[kn]:
                    score = 0
                    if inn and ein:
                        # match last dims of first tensors
                        di, de = parse_dims(inn[0]), parse_dims(ein[0])
                        if di and de and di[-1] == de[-1]:
                            score += 3
                        if di == de:
                            score += 5
                    if out and eout:
                        do, de = parse_dims(out[0]), parse_dims(eout[0])
                        if do == de:
                            score += 5
                    if score > best_score:
                        best_score, best = score, (ein, eout)
                if best is None:
                    best = exemplars[kn][0]
                ein, eout = best
                if need_in and ein:
                    # If we have partial inn with acts, merge affines from exemplar
                    if inn and len(inn) < len(ein):
                        # keep concrete inn tokens; append missing from exemplar by role
                        merged = list(inn)
                        for t in ein:
                            if t not in merged and len(merged) < len(ein):
                                # only add if shape family matches graph
                                merged.append(t)
                        # if still short and shapes clearly same family, use exemplar
                        if len(inn) <= 1 and len(ein) >= 2 and best_score >= 3:
                            merged = list(ein)
                        sep = " × " if (" × " in wrap_tokens(ein, " × ") or "silu" in kn or "addmm" in kn) else "; "
                        # detect gemmish / affine from exemplar formatting preference
                        if any(parse_dims(t) and len(parse_dims(t)) == 1 for t in ein) and any(
                            parse_dims(t) and len(parse_dims(t)) >= 2 for t in ein
                        ):
                            if "mean" not in kn and "softmax" not in kn and "rsqrt" not in kn:
                                sep = " × "
                        if "softmax" in kn or "rsqrt" in kn or "mean" in kn or "red_fused" in kn or "per_fused" in kn:
                            sep = "; "
                        set_in(f, merged if merged else ein, sep)
                        notes["exemplar_in"] += 1
                        need_in = False
                        inn = extract_tokens(f[4])
                    elif not inn:
                        sep = "; "
                        if "silu" in kn or "addmm" in kn or kn.startswith("extern"):
                            sep = " × "
                        if "softmax" in kn or "rsqrt" in kn or "mean" in kn:
                            sep = "; "
                        # For missing IN with OUT present: adapt exemplar IN trailing dims to OUT
                        adapted = list(ein)
                        if out:
                            od = parse_dims(out[0])
                            odt = parse_dtype(out[0])
                            if od and adapted:
                                # expand/clone family: IN often pre-expand
                                pass
                        set_in(f, adapted, sep)
                        notes["exemplar_in"] += 1
                        need_in = False
                        inn = extract_tokens(f[4])
                if need_out and eout:
                    # adapt dtype from inn if needed
                    adapted = list(eout)
                    if inn:
                        dt = parse_dtype(inn[0])
                        if dt and parse_dtype(adapted[0]) and parse_dtype(adapted[0]) != dt:
                            # keep exemplar dtype (cast kernels differ)
                            pass
                    set_out(f, adapted)
                    notes["exemplar_out"] += 1
                    need_out = False
                    out = extract_tokens(f[5])

            # --- Semantic rules ---
            if need_out and "silu" in kn and inn:
                act = primary_act(inn) or inn[0]
                d = parse_dims(act)
                dt = parse_dtype(act, "bfloat16")
                if d:
                    # exemplars often drop leading 1: [1,50,2048] → [50,2048]
                    if len(d) == 3 and d[0] == 1:
                        set_out(f, [fmt_tok(d[1:], dt)])
                    else:
                        set_out(f, [fmt_tok(d, dt)])
                    notes["silu_out"] += 1
                    need_out = False
                    out = extract_tokens(f[5])

            if need_out and ("rsqrt" in kn or ("mean" in kn and "pow" in kn)) and inn:
                act = primary_act(inn)
                if act:
                    d = parse_dims(act)
                    dt = parse_dtype(act, "bfloat16")
                    if d:
                        if len(d) == 2:
                            set_out(f, [fmt_tok((1,) + d, dt)])
                        else:
                            set_out(f, [fmt_tok(d, dt)])
                        notes["rmsnorm_out"] += 1
                        need_out = False
                        out = extract_tokens(f[5])

            if need_out and "softmax" in kn:
                # find score tensor in inn (rank-3 float with large last dims)
                score = None
                for t in inn:
                    d = parse_dims(t)
                    if d and len(d) == 3 and d[-1] >= 32:
                        score = t
                        break
                if score is None and exemplars.get(kn):
                    score_out = exemplars[kn][0][1][0]
                    set_out(f, [score_out])
                    notes["softmax_ex"] += 1
                    need_out = False
                    out = extract_tokens(f[5])
                elif score is not None:
                    d = parse_dims(score)
                    dt = parse_dtype(score, "float32")
                    # often [1,heads,Q,K] bfloat16
                    if d and len(d) == 3:
                        set_out(f, [fmt_tok((1,) + d, "bfloat16" if "to_copy" in kn else dt)])
                    notes["softmax_out"] += 1
                    need_out = False
                    out = extract_tokens(f[5])
                elif need_out:
                    # incomplete softmax IN — pull score from previous bmm OUT
                    if pos > 0:
                        prev = items[pos - 1][1]
                        if bare_name(prev[1]) in ("extern_kernels.bmm", "extern_kernels.mm") and not is_empty(prev[4]):
                            pt = extract_tokens(prev[4])
                            if pt:
                                d = parse_dims(pt[0])
                                if d and len(d) == 3:
                                    set_out(f, [fmt_tok((1,) + d, "bfloat16")])
                                    # also enrich IN with score if missing
                                    if need_in or len(inn) < 3:
                                        base_in = inn if inn else []
                                        if not any(parse_dims(t) == d for t in base_in):
                                            base_in = base_in + pt
                                        # prepend masks from exemplar if available
                                        if exemplars.get(kn):
                                            ein = exemplars[kn][0][0]
                                            masks = [t for t in ein if parse_dims(t) and parse_dims(t)[-1] <= 50 and len(parse_dims(t)) == 2]
                                            for mtok in masks:
                                                if mtok not in base_in:
                                                    base_in = [mtok] + base_in
                                        set_in(f, base_in, "; ")
                                        need_in = False
                                    notes["softmax_from_prev"] += 1
                                    need_out = False
                                    out = extract_tokens(f[5])

            # --- Neighbor inference ---
            if need_out and pos + 1 < len(items):
                nxt = items[pos + 1][1]
                nt = extract_tokens(nxt[3])
                # next consumes our output as first/large act
                if nt:
                    cand = primary_act(nt) or nt[0]
                    # only if current is producer-like
                    if kn.startswith("extern_kernels") or "silu" in kn or "view" in kn:
                        set_out(f, [cand])
                        notes["neighbor_out"] += 1
                        need_out = False
                        out = extract_tokens(f[5])

            if need_in and pos > 0:
                prev = items[pos - 1][1]
                pt = extract_tokens(prev[4])
                if pt and (kn.startswith("triton_") or kn.startswith("extern")):
                    # producer → consumer
                    if "transpose" in kn or "clone" in kn or "expand" in kn or "view" in kn or "cat" in kn:
                        set_in(f, pt, "; ")
                        notes["neighbor_in"] += 1
                        need_in = False
                        inn = extract_tokens(f[4])
                    elif "silu" in kn:
                        # mm → silu: use mm out reshaped
                        d = parse_dims(pt[0])
                        dt = parse_dtype(pt[0], "bfloat16")
                        if d and len(d) == 2:
                            set_in(f, [fmt_tok((1,) + d, dt)])
                        else:
                            set_in(f, pt)
                        notes["neighbor_in"] += 1
                        need_in = False
                        inn = extract_tokens(f[4])

            # --- Clone/expand/transpose with OUT present, IN missing ---
            if need_in and out and ("clone" in kn or "expand" in kn or "transpose" in kn or "permute" in kn):
                if exemplars.get(kn):
                    set_in(f, exemplars[kn][0][0], "; ")
                    notes["view_ex_in"] += 1
                    need_in = False
                    inn = extract_tokens(f[4])
                elif pos > 0:
                    prev = items[pos - 1][1]
                    pt = extract_tokens(prev[4])
                    if pt:
                        set_in(f, pt, "; ")
                        notes["view_prev_in"] += 1
                        need_in = False
                        inn = extract_tokens(f[4])

            # --- OUT present without dtype ---
            if out and any(parse_dtype(t) is None for t in out):
                # inherit from inn or bfloat16
                default = parse_dtype(inn[0]) if inn and parse_dtype(inn[0]) else "bfloat16"
                out = fill_missing_dtypes(out, default=default)
                f[5] = wrap_tokens(out, "; ")
                notes["out_dtype"] += 1

            # --- addmm_silu OUT ---
            if need_out and "addmm_silu" in kn and inn:
                act = primary_act(inn) or inn[0]
                d = parse_dims(act)
                dt = parse_dtype(act, "float32")
                if d:
                    set_out(f, [fmt_tok(d, dt)])
                    notes["addmm_silu"] += 1
                    need_out = False

            # --- G8 KV expand/transpose (5 heads → 15, bmm layout) ---
            # Pattern from complete sibling …_56: [1,5,291,64] → [15,64,291]
            if need_out and (
                kn.endswith("_54")
                or "clone_expand_transpose_unsqueeze_54" in kn
                or "clone_expand_transpose_unsqueeze_56" in kn
            ):
                if inn:
                    d = parse_dims(inn[0])
                    dt = parse_dtype(inn[0], "bfloat16")
                    if d == (1, 5, 291, 64):
                        set_out(f, [fmt_tok((15, 64, 291), dt)])
                        notes["kv_expand"] += 1
                        need_out = False
                        out = extract_tokens(f[5])
            if need_in and out and (
                "clone_expand_transpose_unsqueeze_56" in kn
                or kn.endswith("_56")
            ):
                d = parse_dims(out[0])
                if d == (15, 64, 291):
                    set_in(f, [fmt_tok((1, 5, 291, 64), parse_dtype(out[0], "bfloat16"))])
                    notes["kv_expand_in"] += 1
                    need_in = False
                    inn = extract_tokens(f[4])

            # cat_slice_transpose on already-packed KV: identity-ish buffer
            if need_out and ("cat_slice_transpose" in kn) and inn:
                d = parse_dims(inn[0])
                dt = parse_dtype(inn[0], "bfloat16")
                if d == (1, 5, 291, 64):
                    set_out(f, [fmt_tok(d, dt)])
                    notes["kv_cat_slice"] += 1
                    need_out = False
                    out = extract_tokens(f[5])
                elif d and d[-1] == 64 and len(d) >= 3:
                    # Q-side pack → [1,5,291,64] like sibling _55
                    set_out(f, [fmt_tok((1, 5, 291, 64), "bfloat16")])
                    notes["kv_cat_slice"] += 1
                    need_out = False
                    out = extract_tokens(f[5])
                elif d == (1,):
                    # stray scalar from mis-associated prev — use neighbor prev OUT
                    if pos > 0:
                        prev = items[pos - 1][1]
                        pt = extract_tokens(prev[4])
                        if pt and parse_dims(pt[0]) == (1, 5, 291, 64):
                            set_in(f, pt)
                            set_out(f, pt)
                            notes["kv_cat_slice_fix"] += 1
                            need_in = need_out = False
                            inn, out = extract_tokens(f[4]), extract_tokens(f[5])

            # cat_slice_transpose_65: packed KV identity (not the stray [1] int64 from prior arange)
            if "cat_slice_transpose_65" in kn:
                if need_out or need_in or (inn and parse_dims(inn[0]) == (1,)):
                    set_in(f, ["[1,5,291,64] bfloat16"])
                    set_out(f, ["[1,5,291,64] bfloat16"])
                    notes["kv_65"] += 1
                    need_in = need_out = False
                    inn, out = extract_tokens(f[4]), extract_tokens(f[5])

            # action-tail cat_slice _70/_71: [50,32] float32
            if ("cat_slice_transpose_view_70" in kn or "cat_slice_transpose_71" in kn) and (
                need_in or need_out
            ):
                set_in(f, ["[50,32] float32"])
                set_out(f, ["[50,32] float32"])
                notes["action_cat"] += 1
                need_in = need_out = False

            # --- mul_view / euler ---
            if need_out and "add_addmm_mul_view" in kn and inn:
                # OUT often [50,32] float32
                if exemplars.get(kn):
                    set_out(f, exemplars[kn][0][1])
                else:
                    set_out(f, ["[50,32] float32"])
                notes["euler_out"] += 1
                need_out = False

            # --- FX fallback for still-missing ---
            if (need_in or need_out) and ("full_default" in fused or "arange" in kn):
                # already intentional
                pass

    # Write back
    out_lines = list(lines)
    for li, sec, sg, f in parsed:
        out_lines[li] = join_fields(f)
    PATH.write_text("\n".join(out_lines) + "\n")

    # Recount remaining holes
    remain = 0
    gen_ok = 0
    for _, _, _, f in parsed:
        kn = bare_name(f[1])
        fused = f[15]
        bi, bo = is_empty(f[4]), is_empty(f[5])
        if bi and ("full_default" in fused or "arange" in kn or "linspace" in fused or "iota" in fused):
            gen_ok += 1
            bi = False  # intentional
        if bi or bo:
            remain += 1

    print(f"Filled IN cells≈{filled_in}, OUT cells≈{filled_out}")
    print("Notes:", dict(notes))
    print(f"Remaining incomplete (excl. intentional generators): {remain}")
    print(f"Intentional generator IN=—: {gen_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

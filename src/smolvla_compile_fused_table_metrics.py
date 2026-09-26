#!/usr/bin/env python
"""Fill IO / GFLOPs / AI / Theoretical bottleneck on Fused CompileOp tables.

Rewrites rows in doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md
in place (Pre + graphs #1–#8). Safe to re-run.

Accepts 11-col (legacy), 16-col, or 17-col (with stage) layout. On 16/17-col rows,
refreshes IO / GFLOPs / AI / bottleneck and leaves BD / util / GFLOPs/sec / GPU time
(and stage) untouched (re-fill timing with smolvla_compile_fused_table_timing.py).

| Order | kernel / op | kind | stage / loop / layer | Input | Output | IO Volume /GB | BD /GB/s |
| BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | AI | bottleneck |
| GPU time per launch | meaning | fused aten |

Usage:
  python src/smolvla_compile_fused_table_metrics.py
  python src/smolvla_compile_fused_table_timing.py  # BD / util / GPU time
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md"

DRAM_GBS = 448.0
PEAK_BF16 = 94.8e12
PEAK_FP32 = 23.7e12
DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "f32": 4,
    "float16": 2, "fp16": 2, "f16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "int64": 8, "long": 8, "i64": 8,
    "int32": 4, "i32": 4,
    "int8": 1, "i8": 1,
    "bool": 1, "b8": 1,
}


def _fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    if x >= 1:
        return f"{x:.3g}"
    return f"{x:.2e}"


def _parse_shape_dtype(tok: str, default_dt: str | None = None):
    tok = tok.strip().strip("`").strip()
    if not tok or tok in ("—", "-"):
        return None
    m = re.match(r"^\[([^\]]*)\]\s*([A-Za-z0-9_]+)?$", tok)
    if not m:
        return None
    dims_s, dt = m.group(1), (m.group(2) or "").lower()
    if not dims_s.strip():
        return None
    try:
        dims = tuple(int(x.strip()) for x in dims_s.split(",") if x.strip() != "")
    except ValueError:
        return None
    if not dt:
        dt = default_dt or ""
    return dims, dt


def _tensor_specs(cell: str):
    cell = cell.strip().replace("<br>", " ")
    # join multi-backtick soft-wraps
    parts_bt = re.findall(r"`([^`]*)`", cell)
    if parts_bt:
        cell = " ".join(parts_bt)
    cell = cell.strip("`").strip()
    # accept list (; or |) and matmul (×) separators — never emit raw | into MD cells
    cell = cell.replace(" × ", " | ").replace("×", " | ").replace("; ", " | ").replace(";", " | ")
    # continuation soft-wrap may leave leading "| " / "× "
    cell = re.sub(r"^\|\s*", "", cell)
    raw_parts = [p.strip() for p in cell.split(" | ")]
    dts = []
    for p in raw_parts:
        parsed = _parse_shape_dtype(p)
        if parsed and parsed[1]:
            dts.append(parsed[1])
    default_dt = dts[0] if dts else "float32"
    out = []
    for p in raw_parts:
        if p in ("", "—", "-", "ids"):
            continue
        # G7: "ids [1,48] int64" → "[1,48] int64"
        if p.startswith("ids "):
            p = p[4:].strip()
        parsed = _parse_shape_dtype(p, default_dt=default_dt)
        if parsed:
            dims, dt = parsed
            out.append((dims, dt or default_dt))
    return out


def _numel(dims):
    n = 1
    for d in dims:
        n *= max(d, 0)
    return n


def _bytes(dims, dt):
    b = DTYPE_BYTES.get(dt)
    return None if b is None else _numel(dims) * b


def estimate_metrics(name: str, kind: str, fused: str, inn: str, out: str):
    ins = _tensor_specs(inn)
    outs = _tensor_specs(out)
    io_bytes = 0
    known = False
    for dims, dt in ins + outs:
        b = _bytes(dims, dt)
        if b is None:
            continue
        io_bytes += b
        known = True
    if not known:
        return "—", "—", "—", "—"

    name_l, fused_l = name.lower(), fused.lower()
    flops = flops_2d = flops_1d = None
    tag = ""

    def out_n():
        return _numel(outs[0][0]) if outs else None

    def out_dt():
        return outs[0][1] if outs else ""

    if "upsample_bilinear" in name_l or "upsample_bilinear" in fused_l:
        n = out_n()
        if n is not None:
            flops, tag = 7.0 * n, "FP32"
    elif kind == "extern" and ("addmm" in name_l or name_l.endswith(".mm") or ".bmm" in name_l):
        mats = [x for x in ins if len(x[0]) == 2]
        if len(mats) >= 2:
            a, b = mats[-2], mats[-1]
            M, K1 = a[0]
            d0, d1 = b[0]
            K, N = (K1, d1) if K1 == d0 else ((K1, d0) if K1 == d1 else (K1, d1))
            flops_2d = 2.0 * M * N * K
            tag = "BF16" if "bf" in (a[1] + b[1] + out_dt()) else "FP32"
            if "addmm" in name_l and outs:
                flops_1d = float(out_n())
    elif kind == "extern" and "convolution" in name_l:
        w = next((x for x in ins if len(x[0]) == 4), None)
        if w and outs and len(outs[0][0]) == 4:
            Cout, Cin, kH, kW = w[0]
            N, _, Ho, Wo = outs[0][0]
            flops_2d = 2.0 * N * Cout * Ho * Wo * Cin * kH * kW
            tag = "BF16" if "bf" in (w[1] + out_dt()) else "FP32"
    elif "embedding" in fused_l and "add" in fused_l:
        n = out_n()
        if n is not None:
            flops = float(n)
            tag = "BF16" if "bf" in out_dt() else "FP32"
    elif "layer_norm" in fused_l or "rsqrt" in fused_l or "var_mean" in fused_l or "layer_norm" in name_l:
        # Prefer activation / hidden numel — not tiny mean/rstd buffers.
        n = None
        for dims, dt in outs + ins:
            if len(dims) >= 2 and dims[-1] >= 32:
                n = _numel(dims)
                break
        if n is None:
            n = out_n()
        out_is_hidden = bool(outs and len(outs[0][0]) >= 2 and outs[0][0][-1] >= 32)
        # Eager LN ≈ (7D+3)/token ≈ 7N. Inductor split: red≈2N, poi≈7N
        # (epilogue row carries full-LN comparable FLOPs like eager), per(stats)≈2·|stats|,
        # full fused writing y (add+LN) ≈ 8N.
        if n is not None:
            if "triton_red_fused" in name_l and not out_is_hidden:
                flops, tag = 2.0 * n, "FP32"
            elif "triton_poi_fused" in name_l:
                flops, tag = 7.0 * n, "FP32"
            elif "triton_per_fused" in name_l and not out_is_hidden:
                sn = sum(_numel(d) for d, _ in (outs + ins) if len(d) >= 2 and d[-1] <= 6)
                flops, tag = 2.0 * (sn or n), "FP32"
            elif out_is_hidden:
                flops, tag = 8.0 * n, "FP32"
            else:
                flops, tag = 2.0 * n, "FP32"
    elif any(x in fused_l for x in ("convert_element_type", "to_copy", "clone", "full_default", "embedding")) and not any(
        x in fused_l for x in ("mul", "add", "gelu", "silu", "softmax", "layer_norm", "rsqrt", "var_mean")
    ):
        flops = 0.0
    elif any(x in fused_l or x in name_l for x in ("mul", "add", "sub", "gelu", "silu", "exp", "softmax")):
        n = out_n()
        if n is not None:
            mult = 8.0 if "gelu" in fused_l else (4.0 if "silu" in fused_l else (5.0 if "softmax" in fused_l else 1.0))
            flops = mult * n
            tag = "BF16" if "bf" in out_dt() else "FP32"
    elif "sum" in fused_l or "mean" in fused_l:
        if ins:
            flops, tag = float(_numel(ins[0][0])), "FP32"

    io_gb_s = _fmt_sci(io_bytes / 1e9)
    total = 0.0
    parts = []
    if flops_2d is not None:
        total += flops_2d
        parts.append(f"2D {_fmt_sci(flops_2d/1e9)} ({tag or 'BF16'})")
        if flops_1d:
            total += flops_1d
            parts.append(f"1D {_fmt_sci(flops_1d/1e9)}")
    elif flops is not None:
        total += flops
        parts.append("0" if flops == 0 else f"{_fmt_sci(flops/1e9)}" + (f" ({tag})" if tag else ""))
    else:
        parts.append("—")
        total = None
    gflops_s = " + ".join(parts)
    ai_s = "—" if (total is None or total == 0 or io_bytes == 0) else _fmt_sci(total / io_bytes)

    time_bd = io_bytes / (DRAM_GBS * 1e9)
    time_2d = (flops_2d / (PEAK_BF16 if (tag or "BF16") == "BF16" else PEAK_FP32)) if flops_2d else 0.0
    time_1d = (flops_1d / PEAK_FP32) if flops_1d else 0.0
    time_calc = (flops / PEAK_FP32) if (flops_2d is None and flops and flops > 0) else 0.0
    best_t, bn = -1.0, "bd-bounded"
    for t, lab in (
        (time_2d, "2D-calc-bounded"),
        (time_1d, "1D-calc-bounded"),
        (time_calc, "calc-bounded"),
        (time_bd, "bd-bounded"),
    ):
        if t > best_t + 1e-30:
            best_t, bn = t, lab
    if best_t <= 0 and total is None:
        bn = "—"
    elif best_t <= 0:
        bn = "bd-bounded"
    return io_gb_s, gflops_s, ai_s, bn


def split_pipe_fields(line: str):
    body = line[1:-1]
    fields, buf, in_bt, i = [], [], False, 0
    while i < len(body):
        ch = body[i]
        if ch == "`":
            in_bt = not in_bt
            buf.append(ch)
        elif ch == "|" and not in_bt:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    fields.append("".join(buf).strip())
    return fields


def main() -> int:
    lines = PATH.read_text().splitlines()
    out, n = [], 0
    for line in lines:
        if not re.match(r"^\| \d+ \|", line):
            out.append(line)
            continue
        fields = split_pipe_fields(line)
        if not fields or not fields[0].isdigit():
            out.append(line)
            continue
        if len(fields) == 11:
            seq, name, kind, inn, out_s, _io, _g, _a, _b, meaning, fused = fields
            name_bare = "".join(re.findall(r"`([^`]*)`", name.replace("<br>", ""))) or (
                name.split("`")[1] if "`" in name else name
            )
            kind_bare = kind.strip("`")
            io_gb, gflops, ai, bn = estimate_metrics(
                name_bare, kind_bare, fused, inn, out_s
            )
            # Expand to 17-col placeholders; timing script fills BD/util/time.
            out.append(
                "| "
                + " | ".join(
                    [
                        seq, name, kind, "—", inn, out_s,
                        io_gb, "—", "—", gflops, "—", "—", ai, bn, "—",
                        meaning, fused,
                    ]
                )
                + " |"
            )
            n += 1
            continue
        if len(fields) == 16:
            (
                seq, name, kind, inn, out_s,
                _io, bd, bdu, _g, gps, fu, _a, _b, tcell, meaning, fused,
            ) = fields
            stage = "—"
            name_bare = "".join(re.findall(r"`([^`]*)`", name.replace("<br>", ""))) or (
                name.split("`")[1] if "`" in name else name
            )
            kind_bare = kind.strip("`")
            io_gb, gflops, ai, bn = estimate_metrics(
                name_bare, kind_bare, fused, inn, out_s
            )
            out.append(
                "| "
                + " | ".join(
                    [
                        seq, name, kind, stage, inn, out_s,
                        io_gb, bd, bdu, gflops, gps, fu, ai, bn, tcell,
                        meaning, fused,
                    ]
                )
                + " |"
            )
            n += 1
            continue
        if len(fields) == 17:
            (
                seq, name, kind, stage, inn, out_s,
                _io, bd, bdu, _g, gps, fu, _a, _b, tcell, meaning, fused,
            ) = fields
            name_bare = "".join(re.findall(r"`([^`]*)`", name.replace("<br>", ""))) or (
                name.split("`")[1] if "`" in name else name
            )
            kind_bare = kind.strip("`")
            io_gb, gflops, ai, bn = estimate_metrics(
                name_bare, kind_bare, fused, inn, out_s
            )
            out.append(
                "| "
                + " | ".join(
                    [
                        seq, name, kind, stage, inn, out_s,
                        io_gb, bd, bdu, gflops, gps, fu, ai, bn, tcell,
                        meaning, fused,
                    ]
                )
                + " |"
            )
            n += 1
            continue
        out.append(line)
    PATH.write_text("\n".join(out) + "\n")
    print(f"Updated {n} rows in {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fill BD / GFLOPs/sec / util / GPU time on Fused CompileOp tables from nsys.

Reads CUPTI kernels from the compile-mode sqlite export, windows warm chunk #3
(prepare_images → next chunk upsample), matches Pre / G1–G8 template rows to
kernels in the §0.6 landmark regions (G1–G6 = first camera), and rewrites:

  BD /GB/s | BD util ratio | GFLOPs/sec | FLOPs util ratio | GPU time per launch

Expects 16- or 17-column tables (run after smolvla_compile_fused_table_metrics.py or
when placeholders already exist). Safe to re-run.

Usage:
  python src/smolvla_compile_fused_table_timing.py
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md"
DB = ROOT / "doc/gpu/compile_mode/nsight/smolvla_nsys_compile.sqlite"

DRAM_GBS = 448.0
PEAK_BF16 = 94.8e12
PEAK_FP32 = 23.7e12

HEADER = (
    "| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | Output (shape, dtype) | "
    "IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | "
    "Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | "
    "meaning | fused aten |"
)
SEP = "|---:|---|---|:---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|"

# Warm chunk index among upsample clusters: coldA=0, coldB=1, warm0..=2.. → warm#3 = 5
WARM_CHUNK_IDX = 5


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


def unwrap(s: str) -> str:
    return s.replace("<br>", "")


def bare_name(cell: str) -> str:
    """Kernel name with <br> removed; join multi-backtick soft-wraps."""
    s = unwrap(cell)
    parts = re.findall(r"`([^`]*)`", s)
    if parts:
        return "".join(parts)
    return s.strip()


def load_warm_chunk():
    con = sqlite3.connect(DB)
    rows = list(
        con.execute(
            """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"""
        )
    )
    con.close()
    ups = [(i, r[0], r[1]) for i, r in enumerate(rows) if "upsample_bilinear2d" in r[2]]
    clusters: list[list[tuple[int, int, int]]] = []
    for i, s, e in ups:
        if not clusters or (s - clusters[-1][-1][1]) / 1e6 > 5:
            clusters.append([(i, s, e)])
        else:
            clusters[-1].append((i, s, e))
    ci = WARM_CHUNK_IDX
    c0 = clusters[ci][0][0]
    c1 = clusters[ci + 1][0][0]
    return rows[c0:c1]


def find_all(chunk, pred, lo=0, hi=None):
    hi = len(chunk) if hi is None else hi
    return [j for j in range(lo, hi) if pred(chunk[j][2])]


def find_first(chunk, pred, after=0):
    for j in range(after, len(chunk)):
        if pred(chunk[j][2]):
            return j
    return None


def build_regions(chunk):
    g1 = find_all(chunk, lambda n: n.startswith("triton_poi_fused__to_copy_0"))
    arange = find_all(chunk, lambda n: n.startswith("triton_poi_fused_arange_0"))
    add_emb = find_all(chunk, lambda n: "triton_poi_fused_add_embedding" in n)
    ln0 = find_all(chunk, lambda n: n.startswith("triton_red_fused_native_layer_norm_0"))
    conn = find_all(
        chunk, lambda n: n.startswith("triton_poi_fused__unsafe_view_clone_permute_view_0")
    )
    g7 = find_all(chunk, lambda n: n.startswith("triton_poi_fused_embedding_0"))
    sum0 = find_first(chunk, lambda n: n.startswith("triton_per_fused_sum_0"))
    idx_before = add_emb[0] - 1
    return {
        "Pre": (0, g1[0]),
        "G1": (g1[0], g1[0] + 1),
        "G2": (g1[1], g1[1] + 1),
        "G3": (arange[0], idx_before),
        "G4": (idx_before, ln0[0]),
        "G5": (ln0[0], conn[0]),
        "G6": (conn[0], g1[2]),
        "G7": (g7[0], sum0),
        "G8": (sum0, len(chunk)),
    }


def match_cupti(op_name: str, kind: str, cupti_slice: list, start_j: int):
    name = bare_name(op_name)
    n = len(cupti_slice)
    lookahead = min(n, start_j + 80)

    def dur(j):
        s, e, _ = cupti_slice[j]
        return (e - s) / 1e3

    if kind == "triton" or name.startswith("triton_"):
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2]
            if cn.startswith(name) or cn.split()[0] == name:
                return j, dur(j)
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2]
            if cn.startswith("triton_") and name in cn:
                return j, dur(j)
        return None, None

    if name.startswith("extern_kernels.convolution") or (
        kind == "extern" and "convolution" in name
    ):
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2].lower()
            if "padding" in cn:
                continue
            if (
                "convolution" in cn
                or "fprop" in cn
                or "implicit_convolve" in cn
                or ("cutlass" in cn and "conv" in cn)
                or ("cudnn" in cn and ("conv" in cn or "fprop" in cn or "nhwc" in cn))
            ):
                return j, dur(j)
        for j in range(start_j, min(n, start_j + 15)):
            cn = cupti_slice[j][2]
            if "cutlass" in cn.lower() and "gemm" not in cn.lower():
                return j, dur(j)
        return None, None

    if name.startswith(
        ("extern_kernels.addmm", "extern_kernels.mm", "extern_kernels.bmm")
    ):
        for j in range(start_j, lookahead):
            cl = cupti_slice[j][2].lower()
            if "cutlass" in cl and "gemm" in cl:
                return j, dur(j)
            if "cublas" in cl and ("gemm" in cl or "gemv" in cl):
                return j, dur(j)
            if "magma_sgemm" in cl or "bmm" in cl:
                return j, dur(j)
        return None, None

    if "index_put" in name or name.startswith("aten::index"):
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2]
            if "index_put" in cn or "index_elementwise" in cn or "IndexPut" in cn:
                return j, dur(j)
        return None, None

    if "upsample" in name:
        for j in range(start_j, lookahead):
            if "upsample_bilinear2d" in cupti_slice[j][2]:
                return j, dur(j)
        return None, None

    if "Mul" in op_name or (kind == "aten" and "mul" in name.lower()):
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2]
            if "MulFunctor" in cn or ("vectorized_elementwise" in cn and "Mul" in cn):
                return j, dur(j)
        return None, None

    if "Add" in op_name or (kind == "aten" and name == "add"):
        for j in range(start_j, lookahead):
            cn = cupti_slice[j][2]
            if "CUDAFunctorOnSelf_add" in cn or "CUDAFunctor_add" in cn:
                return j, dur(j)
        return None, None

    short = name.split(".")[-1][:40]
    for j in range(start_j, lookahead):
        if short and short in cupti_slice[j][2]:
            return j, dur(j)
    return None, None


def parse_io_gb(cell: str):
    cell = cell.strip()
    if cell in ("—", "-", ""):
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", cell.replace(",", ""), flags=re.I)
    return float(m.group()) if m else None


def parse_gflops_total(cell: str):
    """Do not count digits inside FP32/BF16/2D/1D labels."""
    cell = cell.strip()
    if cell in ("—", "-", ""):
        return None
    if cell == "0":
        return 0.0
    cleaned = re.sub(r"\((?:FP32|BF16|FP16|TF32)\)", " ", cell, flags=re.I)
    cleaned = re.sub(r"\b(?:2D|1D)\b", " ", cleaned)
    cleaned = cleaned.replace("+", " ")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", cleaned, flags=re.I)
    if not nums:
        return None
    return sum(float(x) for x in nums)


def is_bf16_2d(gflops_cell: str) -> bool:
    return bool(re.search(r"\b2D\b", gflops_cell)) and bool(
        re.search(r"BF16", gflops_cell, re.I)
    )


def fmt_num(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.4g}"
    return f"{x:.3g}"


def bd_util_tag(bd: float, io_gb, t_us: float) -> str:
    util = 100.0 * bd / DRAM_GBS
    s = f"{util:.1f}%"
    if util >= 100:
        if t_us <= 1.5 and io_gb and io_gb > 1e-4:
            s += " ⚠launch-floor+algo-IO≠DRAM"
        elif io_gb and io_gb > 1e-3:
            s += " ⚠L2/cache·algo-IO≠DRAM"
        else:
            s += " ⚠algo-IO≠DRAM"
    return s


def compute_perf(io_cell, gflops_cell, t_us: float | None):
    if t_us is None or t_us <= 0:
        return "—", "—", "—", "—", "—"
    t_s = t_us * 1e-6
    io_gb = parse_io_gb(io_cell)
    gflops = parse_gflops_total(gflops_cell)

    if io_gb is None:
        bd_s, bd_u = "—", "—"
    else:
        bd = io_gb / t_s
        bd_s = fmt_num(bd)
        bd_u = bd_util_tag(bd, io_gb, t_us)

    if gflops is None:
        gps, fu = "—", "—"
    elif gflops == 0:
        gps, fu = "0", "—"
    else:
        gps_val = gflops / t_s
        gps = fmt_num(gps_val)
        peak = PEAK_BF16 if is_bf16_2d(gflops_cell) else PEAK_FP32
        util = 100.0 * (gps_val * 1e9) / peak
        if util >= 10:
            fu = f"{util:.1f}%"
        elif util >= 1:
            fu = f"{util:.2f}%"
        else:
            fu = f"{util:.3g}%"

    return bd_s, bd_u, gps, fu, f"**{t_us:.2f} µs**"


def section_key(header: str) -> str | None:
    if header.startswith("## Pre"):
        return "Pre"
    m = re.match(r"## (\d+)\.", header)
    if m:
        return f"G{m.group(1)}"
    return None


def normalize_row(fields: list[str]) -> list[str] | None:
    """Return 17-col row fields, or None if not a data row we understand."""
    if not fields or not fields[0].isdigit():
        return None
    if len(fields) == 17:
        return fields
    if len(fields) == 16:
        return fields[:3] + ["—"] + fields[3:]
    if len(fields) == 11:
        # Order,name,kind,inn,out,io,gflops,ai,bn,meaning,fused
        o, name, kind, inn, out_s, io, gflops, ai, bn, meaning, fused = fields
        return [
            o, name, kind, "—", inn, out_s, io, "—", "—", gflops, "—", "—", ai, bn, "—", meaning, fused
        ]
    return None


def main() -> int:
    if not DB.exists():
        print(f"Missing nsys sqlite: {DB}", flush=True)
        return 1

    chunk = load_warm_chunk()
    regions = build_regions(chunk)
    # clusters: coldA=0, coldB=1, warm0=2 … → warm#3 uses index 5
    warm_num = WARM_CHUNK_IDX - 2
    print(
        f"warm chunk#{warm_num} kernels={len(chunk)} "
        f"abs {chunk[0][0]/1e9:.6f}->{chunk[-1][1]/1e9:.6f}",
        flush=True,
    )
    for k, (a, b) in regions.items():
        busy = sum((e - s) for s, e, _ in chunk[a:b]) / 1e3
        print(f"  {k}: [{a}:{b}] n={b - a} busy={busy:.1f} us", flush=True)

    lines = PATH.read_text().splitlines()
    out: list[str] = []
    section = ""
    matched = missed = rows = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line

        if line.startswith("| Order | kernel / op |"):
            out.append(HEADER)
            i += 1
            if i < len(lines) and re.match(r"^\|\s*:?---", lines[i]):
                out.append(SEP)
                i += 1

            sk = section_key(section)
            cupti: list = []
            cursor = 0
            if sk and sk in regions:
                a, b = regions[sk]
                cupti = chunk[a:b]

            while i < len(lines) and re.match(r"^\| \d+ \|", lines[i]):
                fields = normalize_row(split_fields(lines[i]))
                if fields is None:
                    out.append(lines[i])
                    i += 1
                    continue
                rows += 1
                name, kind = fields[1], fields[2].strip("`")
                t_us = None
                if cupti:
                    j, t_us = match_cupti(name, kind, cupti, cursor)
                    if j is not None:
                        cursor = j + 1
                        matched += 1
                    else:
                        missed += 1
                else:
                    missed += 1

                bd, bdu, gps, fu, tcell = compute_perf(fields[6], fields[9], t_us)
                fields[7], fields[8], fields[10], fields[11], fields[14] = (
                    bd, bdu, gps, fu, tcell
                )
                out.append(join_fields(fields))
                i += 1
            continue

        out.append(line)
        i += 1

    PATH.write_text("\n".join(out) + "\n")
    print(f"Updated {rows} rows (matched={matched} missed={missed}) in {PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

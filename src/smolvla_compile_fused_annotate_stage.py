#!/usr/bin/env python
"""Insert `stage / loop / layer` column (right of kind) on Fused CompileOp tables.

G8 labels follow the unrolled structure:
  prefix | prefill · L#/15 | euler{k} · suffix | euler{k} · expert L#/16
  | euler{k} · update | tail

Other graphs get coarser stage tags (Pre / G1… / ViT L#/12 / …).

Usage:
  python src/smolvla_compile_fused_annotate_stage.py
  python src/smolvla_compile_fused_annotate_stage.py --full   # keep all unrolled G8 rows
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "doc/gpu/compile_mode/SmolVLA_Fused_CompileOp_List_gpu_backend.md"

NEW_COL = "stage / loop / layer"
HEADER_OLD = (
    "| Order | kernel / op | kind | Input (shape, dtype) | Output (shape, dtype) |"
)
HEADER_NEW = (
    "| Order | kernel / op | kind | stage / loop / layer | Input (shape, dtype) | "
    "Output (shape, dtype) |"
)
# tolerate shorter header variants
HEADER_PAT = re.compile(
    r"^\| Order \| kernel / op \| kind \| (?:stage / loop / layer \| )?"
    r"Input.*\| Output"
)


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


def ensure_stage_col(fields: list[str]) -> list[str]:
    """Return 17-col fields with stage at index 3."""
    if len(fields) == 17:
        return fields
    if len(fields) == 16:
        return fields[:3] + ["—"] + fields[3:]
    return fields


def annotate_g8(rows: list[list[str]]) -> list[str]:
    """Return stage label per row for graph #8."""
    names = [bare_name(f[1]) for f in rows]
    n = len(rows)
    labels = ["other"] * n

    silu22 = [i for i, nm in enumerate(names) if nm.endswith("silu_27") is False and nm.endswith("silu_22")]
    # fix: silu_22
    silu22 = [i for i, nm in enumerate(names) if nm.endswith("silu_22")]
    silu27 = [i for i, nm in enumerate(names) if nm.endswith("silu_27")]
    euler = [
        i
        for i, nm in enumerate(names)
        if "add_addmm_mul_view_51" in nm or "add_addmm_mul_view_47" in nm
    ]
    suffix = [i for i, nm in enumerate(names) if "addmm_silu_view_6" in nm]

    # map marker index → (kind, loop, layer)
    prefill_layer_at = {}
    for li, idx in enumerate(silu22):
        prefill_layer_at[idx] = li  # 0..14

    # 10 packs of 16 silu_27 → euler steps 0..9
    expert_at = {}  # index → (euler_k, layer)
    for pi in range(0, len(silu27), 16):
        chunk = silu27[pi : pi + 16]
        ek = pi // 16
        for lj, idx in enumerate(chunk):
            expert_at[idx] = (ek, lj)

    # Euler update markers in order → step k (0..9)
    euler_sorted = sorted(euler)
    euler_step_at = {idx: k for k, idx in enumerate(euler_sorted)}

    # Suffix markers: assign to nearest following expert pack / euler step
    suffix_step_at = {}
    for idx in suffix:
        # next silu27 pack's euler k, or next euler update
        after = [ek for j, (ek, _) in ((j, expert_at[j]) for j in silu27 if j > idx)]
        if after:
            suffix_step_at[idx] = after[0]
        else:
            after_e = [euler_step_at[j] for j in euler_sorted if j > idx]
            suffix_step_at[idx] = after_e[0] if after_e else 0

    # Region boundaries
    first_compute = min(
        [silu22[0] if silu22 else n, silu27[0] if silu27 else n, suffix[0] if suffix else n]
    )
    tail_start = next(
        (
            i
            for i, nm in enumerate(names)
            if "cat_slice_transpose_view_70" in nm or nm.endswith("transpose_71")
        ),
        n,
    )

    # Anchor intervals from expert packs + euler updates for filling gaps
    # For each euler step k, span = from previous euler+1 (or first expert of pack k) to euler update k
    pack_ranges = []
    for pi in range(0, len(silu27), 16):
        chunk = silu27[pi : pi + 16]
        ek = pi // 16
        start = chunk[0]
        # start a bit earlier to include layer attention before first silu of pack
        # use midpoint between previous pack end and this pack start
        pack_ranges.append((ek, start, chunk[-1]))

    def nearest_expert(i):
        best = None
        best_d = 10**9
        for idx, (ek, lj) in expert_at.items():
            d = abs(idx - i)
            if d < best_d:
                best_d, best = d, (ek, lj, idx)
        return best, best_d

    def nearest_prefill(i):
        if not silu22:
            return None, 10**9
        best = min(silu22, key=lambda j: abs(j - i))
        return prefill_layer_at[best], abs(best - i)

    def io_blob(f):
        """Input+Output cells (17-col: idx 4–5; 16-col pre-stage: idx 3–4)."""
        if len(f) >= 17:
            return (f[4] + " " + f[5]).replace("<br>", "")
        if len(f) >= 5:
            return (f[3] + " " + f[4]).replace("<br>", "")
        return ""

    def is_vlm_shaped(f):
        blob = io_blob(f)
        return bool(
            re.search(r"\[1,241[,\]].*2560|\[241,.*960\]|\[1,241,960\]|\[1,241,5", blob)
            or re.search(r"\[15,241,", blob)
            or re.search(r"\[241,320\] bfloat16", blob)
        )

    def is_expert_shaped(f):
        blob = io_blob(f)
        return bool(
            re.search(r"\[1,50[,\]].*2048|\[50,720\]|\[50,2048\]|\[50,320\]|\[50,960\]", blob)
            or re.search(r"\[15,50,", blob)
            or re.search(r"\[1,50,720\]", blob)
            # expert cross-attn over VLM tokens (float32 path)
            or re.search(r"\[241,320\] float32", blob)
        )

    for i in range(n):
        if i >= tail_start:
            labels[i] = "tail"
            continue

        if i in euler_step_at:
            labels[i] = f"euler{euler_step_at[i]} · update"
            continue

        if i in suffix_step_at:
            labels[i] = f"euler{suffix_step_at[i]} · suffix"
            continue

        if i in prefill_layer_at:
            labels[i] = f"prefill · L{prefill_layer_at[i]}/15"
            continue

        if i in expert_at:
            ek, lj = expert_at[i]
            labels[i] = f"euler{ek} · expert L{lj}/16"
            continue

        if i < first_compute:
            labels[i] = "prefix"
            continue

        labels[i] = "other"

    def is_suffix_shaped(f):
        """Action/time-embed path; exclude expert attn/MLP that sit before silu_27."""
        blob = io_blob(f)
        nm = bare_name(f[1])
        if is_vlm_shaped(f) and not is_expert_shaped(f):
            return False
        if re.search(r"\[241,320\] float32|\[50,2048\]|720,2048|2048,720", blob):
            return False
        if "expand_unsqueeze_view_25" in nm or nm.endswith("mul_silu_27"):
            return False
        return bool(
            is_expert_shaped(f) or "720" in blob or re.search(r"\[15,50,|\[50,", blob)
        )

    # Suffix windows: from each suffix marker → first expert silu of that Euler step.
    # Only claim action/time-embed ops (never VLM or expert attn/MLP).
    for idx, ek in suffix_step_at.items():
        for j in range(idx + 1, min(idx + 40, n)):
            if j in expert_at and expert_at[j][0] == ek:
                break
            if j in euler_step_at or j >= tail_start:
                break
            if j in prefill_layer_at or j in expert_at:
                continue
            if labels[j] != "other":
                continue
            if is_suffix_shaped(rows[j]):
                labels[j] = f"euler{ek} · suffix"

    # Prefill windows: claim VLM-shaped ops around each silu_22 as that layer.
    for idx, li in prefill_layer_at.items():
        for j in range(max(0, idx - 25), min(n, idx + 5)):
            if labels[j] != "other":
                continue
            if is_vlm_shaped(rows[j]) or (not is_expert_shaped(rows[j]) and "241" in io_blob(rows[j])):
                labels[j] = f"prefill · L{li}/15"

    # Expert windows: claim expert-shaped ops around each silu_27 as that layer.
    for idx, (ek, lj) in expert_at.items():
        for j in range(max(0, idx - 12), min(n, idx + 8)):
            if labels[j] != "other":
                continue
            if j in prefill_layer_at or j in suffix_step_at:
                continue
            if is_expert_shaped(rows[j]) or (
                not is_vlm_shaped(rows[j]) and ("50," in io_blob(rows[j]) or "720" in io_blob(rows[j]))
            ):
                labels[j] = f"euler{ek} · expert L{lj}/16"

    # Gap fill remaining "other"
    for i in range(n):
        if labels[i] != "other":
            continue
        f = rows[i]
        vlm_l, dv = nearest_prefill(i)
        exp, de = nearest_expert(i)

        if is_vlm_shaped(f) and (not is_expert_shaped(f) or dv <= de):
            if vlm_l is not None and dv < 80:
                labels[i] = f"prefill · L{vlm_l}/15"
            else:
                labels[i] = "prefill"
            continue

        if is_expert_shaped(f) and exp is not None and de < 80:
            ek, lj, _ = exp
            labels[i] = f"euler{ek} · expert L{lj}/16"
            continue

        if exp is not None and de <= dv and de < 120:
            ek, lj, _ = exp
            labels[i] = f"euler{ek} · expert L{lj}/16"
        elif vlm_l is not None and dv < 120:
            labels[i] = f"prefill · L{vlm_l}/15"
        elif exp is not None:
            ek, lj, _ = exp
            labels[i] = f"euler{ek} · expert L{lj}/16"
        else:
            labels[i] = "other"

    return labels


def annotate_g5(rows: list[list[str]]) -> list[str]:
    """ViT encoder: 12 blocks × (LN / attn / MLP), keyed off gelu_view_7 markers."""
    names = [bare_name(f[1]) for f in rows]
    n = len(rows)
    labels = ["ViT"] * n
    gelu = [i for i, nm in enumerate(names) if "gelu" in nm]
    if not gelu:
        return ["ViT · block"] * n

    def part_for(nm: str) -> str:
        if "layer_norm" in nm:
            return "LN"
        if "gelu" in nm:
            return "MLP"
        if "scaled_dot" in nm or "where" in nm:
            return "attn"
        if nm in ("extern_kernels.addmm", "extern_kernels.mm", "extern_kernels.bmm") or nm.startswith(
            "extern_kernels."
        ):
            return "attn"  # refined per-layer below
        return "block"

    for k, g in enumerate(gelu):
        start = 0 if k == 0 else gelu[k - 1] + 2  # after previous MLP down
        end = g + 1  # inclusive MLP down after gelu
        # MLP region: last LN before gelu → gelu → down proj
        ln_before = [i for i in range(start, g) if "layer_norm" in names[i]]
        mlp_start = ln_before[-1] + 1 if ln_before else g
        for i in range(start, min(end + 1, n)):
            nm = names[i]
            if "layer_norm" in nm:
                part = "LN"
            elif i >= mlp_start:
                part = "MLP"
            elif "gelu" in nm:
                part = "MLP"
            elif "scaled_dot" in nm or "where" in nm:
                part = "attn"
            elif "addmm" in nm or "mm" in nm or "bmm" in nm:
                part = "attn"
            else:
                part = "attn"
            labels[i] = f"ViT · L{k}/12 · {part}"

    for i in range(gelu[-1] + 2, n):
        labels[i] = "ViT · epilogue"
    return labels


def annotate_simple(section_g: int | None, n: int) -> list[str]:
    mapping = {
        0: "Pre · prepare_images",
        1: "G1 · cast",
        2: "G2 · mask",
        3: "G3 · patch_embed",
        4: "G4 · pos_embed",
        6: "G6 · connector",
        7: "G7 · lang_embed",
    }
    base = mapping.get(section_g, "other")
    return [base] * n


def g5_is_representative(text: str) -> bool:
    """True when G5 already has only ViT L0 (small row count)."""
    lines = text.splitlines()
    section = ""
    count = 0
    saw_l0 = False
    for line in lines:
        if line.startswith("## "):
            section = line
        if "graph #5" not in section and not section.startswith("## 5."):
            continue
        if not re.match(r"^\| \d+ \|", line):
            continue
        f = split_fields(line)
        if len(f) != 17:
            return False
        st = f[3]
        if not st.startswith("ViT"):
            continue
        count += 1
        if re.search(r"ViT · L[1-9]/", st) or re.search(r"ViT · L1[0-1]/", st):
            return False
        if st.startswith("ViT · L0/"):
            saw_l0 = True
    # Full encoder is 116 rows; one layer is ~14. Mis-labeled all-L0 full table → False.
    return saw_l0 and 0 < count <= 20


def filter_g5_representative(text: str) -> str:
    """Keep only ViT · L0/12 (+ epilogue if any)."""
    lines = text.splitlines()
    out: list[str] = []
    section = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line
            out.append(line)
            i += 1
            continue
        if ("graph #5" in section or section.startswith("## 5.")) and line.startswith("| Order |"):
            out.append(line)
            i += 1
            if i < len(lines) and re.match(r"^\|\s*:?---", lines[i]):
                out.append(lines[i])
                i += 1
            kept = []
            while i < len(lines) and re.match(r"^\| \d+ \|", lines[i]):
                f = split_fields(lines[i])
                if len(f) == 17:
                    st = f[3]
                    if st.startswith("ViT · L0/"):
                        kept.append(f)
                i += 1
            for n, f in enumerate(kept, 1):
                f[0] = str(n)
                out.append(join_fields(f))
            continue
        out.append(line)
        i += 1

    # refresh G5 launch counts
    triton = extern = aten = nrows = 0
    section = ""
    for line in out:
        if line.startswith("## "):
            section = line
        if not (section.startswith("## 5.") or "graph #5" in section):
            continue
        if not re.match(r"^\| \d+ \|", line):
            continue
        f = split_fields(line)
        if len(f) != 17:
            continue
        nrows += 1
        k = f[2].strip("`")
        if k == "triton":
            triton += 1
        elif k == "extern":
            extern += 1
        else:
            aten += 1

    final = []
    section = ""
    for line in out:
        if line.startswith("## "):
            section = line
        if (section.startswith("## 5.") or "graph #5" in section) and line.startswith(
            "**Launches (calling order):**"
        ):
            final.append(
                f"**Launches (calling order):** {nrows} · triton={triton} · extern={extern} · aten={aten} "
                f"(*(representative)* **ViT L0/12** only)"
            )
            continue
        if (section.startswith("## 5.") or "graph #5" in section) and line.startswith(
            "**Stage column (G5):**"
        ):
            final.append(
                "**Stage column (G5):** collapsed to one representative **`ViT · L0/12`** "
                "(LN / attn / MLP). Pass `--full` to annotate without collapsing."
            )
            continue
        if re.match(r"^\| \*\*5\*\* \|", line):
            line = re.sub(r"^\| \*\*5\*\* \| \d+ \|", f"| **5** | {nrows} |", line)
        final.append(line)

    # Insert stage note under G5 meaning if missing
    text2 = "\n".join(final)
    if "**Stage column (G5):**" not in text2:
        needle = f"**Launches (calling order):** {nrows}"
        # only first G5 launches line
        note = (
            "**Stage column (G5):** collapsed to one representative **`ViT · L0/12`** "
            "(LN / attn / MLP). Pass `--full` to annotate without collapsing.\n\n" + needle
        )
        # replace only within G5 — do a section-aware replace
        parts = text2.split("## 5.")
        if len(parts) >= 2:
            head, rest = parts[0], "## 5." + parts[1]
            rest = rest.replace(needle, note, 1)
            text2 = head + rest
    return text2 if text2.endswith("\n") else text2 + "\n"


def g8_is_representative(text: str) -> bool:
    """True when G8 table already has only euler0 / L0 (plus prefix/suffix/update/tail)."""
    lines = text.splitlines()
    section = ""
    saw_stage = False
    for line in lines:
        if line.startswith("## "):
            section = line
        if "graph #8" not in section and not section.startswith("## 8."):
            continue
        if not re.match(r"^\| \d+ \|", line):
            continue
        f = split_fields(line)
        if len(f) != 17:
            return False
        st = f[3]
        if st in ("", "—"):
            return False
        saw_stage = True
        if re.search(r"euler[1-9]", st):
            return False
        if re.search(r"prefill · L[1-9]/", st) or re.search(r"expert L[1-9]/", st):
            return False
    return saw_stage


def filter_g8_representative(text: str) -> str:
    """Keep prefix/tail (deduped), prefill L0, euler0 expert L0 + suffix/update only."""
    lines = text.splitlines()

    def keep_stage(stage: str) -> bool:
        s = stage.strip()
        # Bare "prefill" is an unlabeled gap — do not promote to L0.
        if s in ("prefix", "tail", "other"):
            return True
        if s.startswith("prefill · L0/"):
            return True
        if s.startswith("prefill"):
            return False
        m = re.match(r"euler(\d+)\s*·\s*(.*)$", s)
        if not m:
            return True
        if int(m.group(1)) != 0:
            return False
        rest = m.group(2)
        if rest.startswith("expert L0/") or rest in ("suffix", "update"):
            return True
        if rest.startswith("expert L"):
            return False
        return True

    def bare(cell: str) -> str:
        return "".join(re.findall(r"`([^`]*)`", cell.replace("<br>", "")))

    out: list[str] = []
    section = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line
            out.append(line)
            i += 1
            continue
        if "graph #8" in section and line.startswith("| Order |"):
            out.append(line)
            i += 1
            if i < len(lines) and re.match(r"^\|\s*:?---", lines[i]):
                out.append(lines[i])
                i += 1
            raw = []
            while i < len(lines) and re.match(r"^\| \d+ \|", lines[i]):
                f = split_fields(lines[i])
                if len(f) == 17 and keep_stage(f[3]):
                    raw.append(f)
                i += 1
            seen_p, seen_t, seen_s, seen_pf = set(), set(), set(), set()
            seen_update = False
            kept = []
            for f in raw:
                st, nm = f[3], bare(f[1])
                if st == "prefix":
                    if nm in seen_p:
                        continue
                    seen_p.add(nm)
                    kept.append(f)
                    continue
                if st == "tail":
                    if nm in seen_t:
                        continue
                    seen_t.add(nm)
                    kept.append(f)
                    continue
                if st.startswith("euler0 · update"):
                    seen_update = True
                    kept.append(f)
                    continue
                if st.startswith("prefill"):
                    # Drop only after euler update (post-step mislabels).
                    if seen_update:
                        continue
                    if nm in seen_pf:
                        continue
                    seen_pf.add(nm)
                    kept.append(f)
                    continue
                if st.startswith("euler0 · expert"):
                    kept.append(f)
                    continue
                if st.startswith("euler0 · suffix"):
                    if nm in seen_s:
                        continue
                    seen_s.add(nm)
                    kept.append(f)
                    continue
                kept.append(f)
            for n, f in enumerate(kept, 1):
                f[0] = str(n)
                out.append(join_fields(f))
            continue
        out.append(line)
        i += 1

    # refresh G8 launch counts
    triton = extern = aten = nrows = 0
    section = ""
    for line in out:
        if line.startswith("## "):
            section = line
        if "graph #8" not in section or not re.match(r"^\| \d+ \|", line):
            continue
        f = split_fields(line)
        if len(f) != 17:
            continue
        nrows += 1
        k = f[2].strip("`")
        if k == "triton":
            triton += 1
        elif k == "extern":
            extern += 1
        else:
            aten += 1
    final = []
    section = ""
    for line in out:
        if line.startswith("## "):
            section = line
        if "graph #8" in section and line.startswith("**Launches (calling order):**"):
            final.append(
                f"**Launches (calling order):** {nrows} · triton={triton} · extern={extern} · aten={aten} "
                f"(*(representative)* **prefill L0** + **euler0 / expert L0** only)"
            )
            continue
        if line.startswith("**Stage column (G8):**"):
            final.append(
                "**Stage column (G8):** collapsed to one representative **`prefill · L0/15`**, one "
                "**`euler0 · expert L0/16`** (plus `prefix` / `euler0 · suffix` / `euler0 · update` / `tail`). "
                "Pass `--full` to annotate without collapsing. Full ×10×16 list requires regen from JSON first."
            )
            continue
        if re.match(r"^\| \*\*8\*\* \|", line):
            line = re.sub(r"^\| \*\*8\*\* \| \d+ \|", f"| **8** | {nrows} |", line)
        final.append(line)
    return "\n".join(final) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="Keep full unrolled G5 (12 ViT layers) and G8 (all Euler steps / layers). "
        "Default collapses to ViT L0 + G8 L0/euler0.",
    )
    args = ap.parse_args(argv)

    lines = PATH.read_text().splitlines()
    already_g8 = g8_is_representative("\n".join(lines))
    already_g5 = g5_is_representative("\n".join(lines))
    section = ""
    section_g = None

    # Pass 1: parse all data rows with section
    parsed: list[tuple[int, int | None, list[str]]] = []  # line_idx, section_g, fields
    for i, line in enumerate(lines):
        if line.startswith("## Pre"):
            section, section_g = line, 0
        elif (m := re.match(r"^## (\d+)\.", line)):
            section, section_g = line, int(m.group(1))
        elif line.startswith("## "):
            section, section_g = line, None

        if HEADER_PAT.match(line):
            # normalize header to 17-col
            if "stage / loop / layer" not in line:
                line = line.replace(
                    "| kind | Input",
                    "| kind | stage / loop / layer | Input",
                )
            out_line = line
            # fix sep line handled below
            lines[i] = line
            continue

        if re.match(r"^\|---", line) and "Input" not in line:
            # separator under header — count dashes groups
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) == 16:
                # insert stage sep after kind (3rd col)
                parts = parts[:3] + [":---"] + parts[3:]
                lines[i] = "| " + " | ".join(parts) + " |"
            continue

        if re.match(r"^\| \d+ \|", line):
            f = ensure_stage_col(split_fields(line))
            parsed.append((i, section_g, f))

    # Group by section_g contiguous
    from itertools import groupby

    # Annotate
    annotations: dict[int, str] = {}
    for sg, group in groupby(parsed, key=lambda x: x[1]):
        group = list(group)
        rows = [f for _, _, f in group]
        idxs = [i for i, _, _ in group]
        if sg == 8 and already_g8 and not args.full:
            # Keep existing stage labels — re-annotate on a collapsed table is lossy.
            labs = [
                (f[3] if f[3] not in ("", "—") else "other") for f in rows
            ]
        elif sg == 8:
            labs = annotate_g8(rows)
        elif sg == 5 and already_g5 and not args.full:
            labs = [(f[3] if f[3] not in ("", "—") else "ViT") for f in rows]
        elif sg == 5:
            labs = annotate_g5(rows)
        else:
            labs = annotate_simple(sg, len(rows))
        for li, lab in zip(idxs, labs):
            annotations[li] = lab

    # Rewrite file
    new_lines = []
    for i, line in enumerate(lines):
        if HEADER_PAT.match(line):
            if "stage / loop / layer" not in line:
                line = line.replace("| kind | Input", "| kind | stage / loop / layer | Input")
            new_lines.append(line)
            continue
        if re.match(r"^\|---", line) and i > 0 and HEADER_PAT.match(lines[i - 1] if False else ""):
            pass
        if re.match(r"^\|---", line):
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) == 16:
                parts = parts[:3] + [":---"] + parts[3:]
                new_lines.append("| " + " | ".join(parts) + " |")
                continue
            if len(parts) == 17:
                new_lines.append(line)
                continue
            new_lines.append(line)
            continue
        if re.match(r"^\| \d+ \|", line):
            f = ensure_stage_col(split_fields(line))
            f[3] = annotations.get(i, f[3] if f[3] != "—" else "other")
            new_lines.append(join_fields(f))
            continue
        # column docs line
        if "Op-table columns" in line and "stage / loop / layer" not in line:
            line = line.replace(
                "| kind | Input",
                "| kind | stage / loop / layer | Input",
            )
        new_lines.append(line)

    # Doc note under G8 meaning if missing
    text = "\n".join(new_lines)
    if "stage / loop / layer" in text and "**Stage column (G8):**" not in text:
        needle = "**Launches (calling order):** 3410"
        note = (
            "**Stage column (G8):** `prefix` (masks/bookkeeping) → `prefill · L#/15` (VLM, once; "
            "interleaved early with euler0) → `euler{k} · suffix` (action/time embed) → "
            "`euler{k} · expert L#/16` (16-layer action expert nested in each of 10 Euler steps) → "
            "`euler{k} · update` → `tail` (Stage‑4 crop). Loops are **unrolled** (one row per launch).\n\n"
            + needle
        )
        text = text.replace(needle, note, 1)

    if args.full:
        print("Kept full unrolled G5/G8 (--full)")
    else:
        if already_g8:
            print("G8 already representative (euler0 / L0); skipped collapse filter")
        else:
            text = filter_g8_representative(text)
            print("Collapsed G8 to representative prefill L0 + euler0 / expert L0")
        if already_g5:
            print("G5 already representative (ViT L0); skipped collapse filter")
        else:
            text = filter_g5_representative(text)
            print("Collapsed G5 to representative ViT L0/12")

    PATH.write_text(text if text.endswith("\n") else text + "\n")

    # stats
    from collections import Counter

    c = Counter(annotations.values())
    print(f"Annotated {len(annotations)} rows → 17-col tables")
    print("Top labels (pre-filter annotation):")
    for k, v in c.most_common(15):
        print(f"  {v:4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

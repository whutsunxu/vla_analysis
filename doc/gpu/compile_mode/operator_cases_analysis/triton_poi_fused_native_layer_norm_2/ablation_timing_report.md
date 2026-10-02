# `triton_poi_fused_native_layer_norm_2` — ablation / size-sweep timing report

**Date:** 2026-09-28  
**GPU:** NVIDIA GeForce RTX 5060 Ti (DRAM peak **448 GB/s**, FP32 CUDA peak **23.7 TFLOP/s**, L2 **32 MB**)  
**Nsight:** Systems 2025.1.3 · CUPTI range = measured window only  
**Harness:** `profile_ln_poi.py` + `ln_poi_kernel.py`  
**Protocol:** warmup **5** (outside CUPTI) + measured **10** sync’d iters (inside `cudaProfilerApi`)

Artifacts: `nsys/ln_poi_{baseline,io8,io16,io24,io29…io35,io48,io64,io96,io128}.nsys-rep` + `*_cuda_gpu_kern_sum.csv`.  
ncu (§7): `ncu/io{24,31,32,33,34,35,48,64}_ncu.log` + `io*_app.csv` (`ncu` 2026.3.1).

Roofline / IO rules match `SmolVLA_op_list_inductor_nsight.md` §GRAPH5 and `smolvla_compile_fused_table_metrics.py`:  
poi LN **FLOPs = 7·N**; **IO = Σ bytes(inputs)+Σ bytes(output)**; BD util ⚠ when algo-IO ≫ DRAM (L2/cache).

---

## 0. Experiment matrix

| Variant | What | Shape | Launch |
|---------|------|------:|--------|
| **baseline** | Inductor in-model dims | `Y=1024`, `X=768` | Grid2D **(12, 16)** · `XBLOCK=64` · `YBLOCK=64` · `num_warps=16` |
| **io8…io128** | Grow **Y** (tokens); keep **X=768** + same tile/warps | see §6 | Grid2D `(12, ceil(Y/64))` |
| **io29…io35** | L2-boundary zoom (±3 MB around 32) | see §6.5 | same |

**Tensors / dtypes / strides** (same as `model__4_inference_4.4/output_code.py` call site):

| Buffer | Shape | Dtype | Stride (baseline) |
|--------|------:|-------|-------------------|
| `x` (in_ptr0) | `[1,Y,X]` | bf16 | `(Y·X, 1, Y)` feature-major load |
| `mean` | `[1,Y,1]` | f32 | `(Y, 1, Y)` |
| `var` | `[1,Y,1]` | f32 | `(Y, 1, Y)` — Inductor **sum of squared diffs** |
| `γ` / `β` | `[X]` | bf16 | contiguous |
| `out` | `[1,Y,X]` | bf16 | `(Y·X, X, 1)` contiguous store |

Math: `(x − mean) · rsqrt(var/X + 1e−6) · γ + β` → bf16.

Unit test (`test_triton_poi_fused_native_layer_norm_2.py`): correctness **PASS** (max abs vs fp32 ref **7.8e−3**).

---

## 1. Average time across each 10-iter batch

### 1.1 Host wall (CUDA synchronize around each iter) — baseline + size sweep means

| Target | Host mean (ms) | median | min | max | mean excl. iter0 |
|-------:|---------------:|-------:|----:|----:|-----------------:|
| baseline (~3.16 MB) | **0.1201** | 0.1045 | 0.0947 | 0.2621 | 0.1044 |
| **8 MB** | **0.1146** | 0.0990 | 0.0943 | 0.2515 | 0.0994 |
| **16 MB** | **0.1340** | 0.1176 | 0.1117 | 0.2700 | 0.1189 |
| **24 MB** | **0.1518** | 0.1285 | 0.1208 | 0.3081 | 0.1345 |
| **29 MB** | **0.1437** | 0.1275 | 0.1183 | 0.2921 | 0.1272 |
| **30 MB** | **0.1300** | 0.1107 | 0.1021 | 0.3029 | 0.1108 |
| **31 MB** | **0.1457** | 0.1256 | 0.1186 | 0.2871 | 0.1299 |
| **32 MB** | **0.1557** | 0.1367 | 0.1279 | 0.3237 | 0.1370 |
| **33 MB** | **0.1543** | 0.1354 | 0.1283 | 0.3169 | 0.1363 |
| **34 MB** | **0.1590** | 0.1399 | 0.1349 | 0.2971 | 0.1436 |
| **35 MB** | **0.1569** | 0.1439 | 0.1379 | 0.2807 | 0.1432 |
| **48 MB** | **0.2018** | 0.1879 | 0.1821 | 0.3190 | 0.1887 |
| **64 MB** | **0.2840** | 0.2697 | 0.2611 | 0.4065 | 0.2704 |
| **96 MB** | **0.3714** | 0.3608 | 0.3470 | 0.5054 | 0.3565 |
| **128 MB** | **0.4540** | 0.4419 | 0.4296 | 0.5856 | 0.4394 |

Host wall ≫ GPU busy at small sizes (launch / driver / sync). Gap closes as GPU work grows.

### 1.2 GPU busy time per iter (CUPTI Σ kernel duration / 10)

| Target | GPU busy / iter | Instances in window |
|-------:|----------------:|--------------------:|
| baseline | **4.832 µs** | 10 |
| 8 MB | **10.218 µs** | 10 |
| 16 MB | **18.449 µs** | 10 |
| 24 MB | **26.420 µs** | 10 |
| 29 MB | **31.429 µs** | 10 |
| 30 MB | **32.373 µs** | 10 |
| 31 MB | **33.730 µs** | 10 |
| 32 MB | **37.653 µs** | 10 |
| 33 MB | **38.907 µs** | 10 |
| 34 MB | **44.530 µs** | 10 |
| 35 MB | **55.414 µs** | 10 |
| 48 MB | **112.165 µs** | 10 |
| 64 MB | **165.118 µs** | 10 |
| 96 MB | **252.351 µs** | 10 |
| 128 MB | **336.959 µs** | 10 |

---

## 2. Baseline roofline (same columns as Inductor §GRAPH5)

**GPU time per launch** = `cuda_gpu_kern_sum` Avg over the 10-iter CUPTI window.  
**IO / GFLOPs / AI / Theoretical bottleneck** from shapes + heuristics (poi LN = **7** FLOP/elem).  
**BD / GFLOPs/sec / util** from IO÷time and GFLOPs÷time vs 448 GB/s and 23.7 TFLOP/s.

| Order | kernel / op | Input (shape, dtype) | Output (shape, dtype) | IO Volume /GB | BD /GB/s | BD util ratio | GFLOPs | GFLOPs/sec | FLOPs util ratio | Arithmetic intensity (FLOP/byte) | Theoretical bottleneck | GPU time per launch | meaning |
|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| 1 | `triton_poi_fused_native_layer_norm_2` | `[1,1024,768] bf16 × [1,1024,1] f32 × [1,1024,1] f32 × [768] bf16 × [768] bf16` | `[1,1024,768] bf16` | 3.16e-03 | 653.3 | 145.8% ⚠L2/cache·algo-IO≠DRAM | 5.51e-03 (FP32) | 1139.2 | 4.81% | 1.74 | bd-bounded | **4.83 µs** | LN affine epilogue (standalone microbench) |

Launch config: Grid2D **(12, 16)**, `XBLOCK=64`, `YBLOCK=64`, `num_warps=16` (matches an Inductor TTIR tile for this kernel: `tensor<64x64>` + `num_warps=16`).

---

## 3. Timing stability (per-launch StdDev, CUPTI)

| Target | Avg (µs) | StdDev (µs) | CV |
|-------:|---------:|------------:|---:|
| baseline | 4.83 | 0.072 | 1.5% |
| 8 MB | 10.22 | 0.100 | 1.0% |
| 16 MB | 18.45 | 0.073 | 0.4% |
| 24 MB | 26.42 | 0.089 | 0.3% |
| 29 MB | 31.43 | 0.095 | 0.3% |
| 30 MB | 32.37 | 0.156 | 0.5% |
| 31 MB | 33.73 | 0.455 | 1.3% |
| 32 MB | 37.65 | 0.712 | 1.9% |
| 33 MB | 38.91 | 1.16 | 3.0% |
| 34 MB | 44.53 | 1.62 | 3.6% |
| 35 MB | 55.41 | 0.646 | 1.2% |
| 48 MB | 112.2 | 2.35 | 2.1% |
| 64 MB | 165.1 | 3.33 | 2.0% |
| 96 MB | 252.4 | 2.97 | 1.2% |
| 128 MB | 337.0 | 4.04 | 1.2% |

---

## 4. Findings

1. **Kernel is `bd-bounded`** (AI ≈ **1.74–1.75** ≪ ridge ~52.9). Same classification as Inductor §GRAPH5.
2. **Baseline microbench ≈ 4.83 µs** vs in-model §GRAPH5 **3.33 µs** (~**1.45×**) — same pattern as Pre upsample microbench vs in-model (colder L2 / isolated launch cadence). IO / GFLOPs / AI labels identical by construction.
3. **⚠ BD util >100%** while algo IO is L2-resident. Fine scan (§6.5): util stays **~205%** through **29–31 MB**, first soft drop at **32–33 MB** (~189%), then steepens **34→35 MB** (170%→141%), and reaches **~95%** by 48 MB.
4. The discontinuity is **not a hard cliff exactly at 32 MB** — onset aligns with when **`x`+`out` ≈ L2** (each buffer ≈ 16 MB at target 32), then worsens as both activations exceed L2 together.
5. **Feature-major load** still reaches **~380 GB/s** post-L2 (~**85%** of 448) at 64–128 MB.
6. Absolute in-model µs remain authoritative in §GRAPH5; this report is for **tile-matched microbench + size-sweep BD-util clarification**.

---

## 5. Relation to Inductor §GRAPH5 (same IO model)

| Metric | §GRAPH5 (in-model) | This baseline microbench |
|--------|-------------------:|-------------------------:|
| GPU time | **3.33 µs** | **4.83 µs** |
| BD | 949.5 GB/s | 653.3 GB/s |
| BD util | 211.9% ⚠ | 145.8% ⚠ |
| GFLOPs/sec | 1655.6 | 1139.2 |
| FLOPs util | 6.99% | 4.81% |
| IO / GFLOPs / AI | 3.16e-03 / 5.51e-03 / 1.74 | **identical** |

---

## 6. Buffer-size sweep — why BD util drops toward ≤100%

**Goal:** Grow algo IO (`Σ bytes(in)+bytes(out)`) through **8 → 128 MB**, with a **±3 MB zoom around L2 = 32 MB** (29–35), and watch BD util / GPU time discontinuity.

**Setup:** `TARGET_IO_MB=<N>` grows **Y** only (`X=768` fixed; same `XBLOCK/YBLOCK/num_warps`). Warmup 5 + 10 CUPTI iters.  
Artifacts: `nsys/ln_poi_io{8,16,24,29,30,31,32,33,34,35,48,64,96,128}.nsys-rep`.

### 6.1 Shapes and host wall

| Target IO | Y | Shape | Buffer `x` (bf16) | Algo IO (Σ in+out) | Grid | Host mean (10 iters) |
|----------:|--:|------:|------------------:|-------------------:|------|---------------------:|
| ~3.16 MB (baseline) | 1024 | `[1,1024,768]` | 1.57 MB | 3.16 MB | (12, 16) | 0.120 ms |
| **8 MB** | 2624 | `[1,2624,768]` | 4.03 MB | 8.09 MB | (12, 41) | 0.115 ms |
| **16 MB** | 5184 | `[1,5184,768]` | 7.96 MB | 15.97 MB | (12, 81) | 0.134 ms |
| **24 MB** | 7808 | `[1,7808,768]` | 11.99 MB | 24.05 MB | (12, 122) | 0.152 ms |
| **29 MB** | 9408 | `[1,9408,768]` | 14.45 MB | 28.98 MB | (12, 147) | 0.144 ms |
| **30 MB** | 9728 | `[1,9728,768]` | 14.94 MB | 29.97 MB | (12, 152) | 0.130 ms |
| **31 MB** | 10048 | `[1,10048,768]` | 15.43 MB | 30.95 MB | (12, 157) | 0.146 ms |
| **32 MB** | 10368 | `[1,10368,768]` | 15.93 MB | 31.94 MB | (12, 162) | 0.156 ms |
| **33 MB** | 10688 | `[1,10688,768]` | 16.42 MB | 32.92 MB | (12, 167) | 0.154 ms |
| **34 MB** | 11008 | `[1,11008,768]` | 16.91 MB | 33.91 MB | (12, 172) | 0.159 ms |
| **35 MB** | 11392 | `[1,11392,768]` | 17.50 MB | 35.09 MB | (12, 178) | 0.157 ms |
| **48 MB** | 15552 | `[1,15552,768]` | 23.89 MB | 47.90 MB | (12, 243) | 0.202 ms |
| **64 MB** | 20800 | `[1,20800,768]` | 31.95 MB | 64.07 MB | (12, 325) | 0.284 ms |
| **96 MB** | 31168 | `[1,31168,768]` | 47.87 MB | 96.00 MB | (12, 487) | 0.371 ms |
| **128 MB** | 41536 | `[1,41536,768]` | 63.80 MB | 127.93 MB | (12, 649) | 0.454 ms |

### 6.2 Roofline vs size (AI ≈ 1.75; all `bd-bounded`)

Same IO model as §2: `IO = 2·Y·X·2 + 2·Y·4 + 2·X·2`, `FLOPs = 7·Y·X` (FP32).

| Algo IO | GPU time | BD /GB/s | BD util | GFLOPs/sec | FLOPs util |
|--------:|---------:|---------:|--------:|-----------:|-----------:|
| 3.16 MB | **4.83 µs** | 653 | **146%** ⚠ | 1139 | 4.81% |
| 8.09 MB | **10.22 µs** | 791 | **177%** ⚠ | 1381 | 5.83% |
| 15.97 MB | **18.45 µs** | 866 | **193%** ⚠ | 1511 | 6.37% |
| 24.05 MB | **26.42 µs** | 910 | **203%** ⚠ | 1589 | 6.70% |
| 28.98 MB | **31.43 µs** | 922 | **206%** ⚠ | 1609 | 6.79% |
| 29.97 MB | **32.37 µs** | 926 | **207%** ⚠ | 1616 | 6.82% |
| 30.95 MB | **33.73 µs** | 918 | **205%** ⚠ | 1602 | 6.76% |
| 31.94 MB | **37.65 µs** | 848 | **189%** ⚠ | 1480 | 6.25% |
| 32.92 MB | **38.91 µs** | 846 | **189%** ⚠ | 1477 | 6.23% |
| 33.91 MB | **44.53 µs** | 761 | **170%** ⚠ | 1329 | 5.61% |
| 35.09 MB | **55.41 µs** | 633 | **141%** ⚠ | 1105 | 4.66% |
| 47.90 MB | **112.2 µs** | 427 | **95.3%** | 745 | 3.15% |
| 64.07 MB | **165.1 µs** | 388 | **86.6%** | 677 | 2.86% |
| 96.00 MB | **252.4 µs** | 380 | **84.9%** | 664 | 2.80% |
| 127.93 MB | **337.0 µs** | 380 | **84.7%** | 663 | 2.80% |

### 6.3 Does BD util stabilize at larger sizes?

| Regime | Algo IO | BD util | Behavior |
|--------|--------:|--------:|----------|
| L2-resident | 3–31 MB | **146–207%** ⚠ | Flat high util; time ≈ linear in IO |
| Soft onset | 32–33 MB | **~189%** ⚠ | First clear drop (−16 pp vs 31 MB) |
| Steep spill | 34–35 MB | **170% → 141%** ⚠ | Time jumps (+14% / +24% step-to-step) |
| Crossing | 35 → 48 MB | 141% → **95%** | Continues toward DRAM |
| Post-L2 | 64–128 MB | **86.6% → 84.7%** | **Flat plateau** (~85%) |

**Verdict:** Perf discontinuity **does start around 32 MB**, but as a **ramp**, not a single-step cliff: soft at 32–33, steep at 34–35, then settles ~85% by 64+. Same qualitative story as Pre `mul` (plateau ~88–95%), slightly lower asymptote from feature-major loads.

```text
BD util vs algo IO  (L2 zoom highlighted)

  210% |              * * *
  200% |           *
  190% |                    * *
  170% |                        *
  140% |                          *
  100% |                             *
   85% |                                *──*──*   ← plateau ~85%
       +--24-29-30-31-32-33-34-35----48-64-96-128  MB
                      ↑↑ soft     ↑↑ steep
                         L2=32MB (x+out ≈ 32 MB)
```

### 6.4 Interpretation (short)

1. **⚠ util >100%** while `x`+`out` (plus small mean/var/γ/β) still fit / reuse **L2**.
2. Onset when each activation ≈ **16 MB** (algo IO ≈ **32 MB**) — both no longer fit hot together.
3. Past L2, util **stabilizes below 100%** (~**85%** here).
4. In-model absolute times remain valid; the size sweep only clarifies the BD-util metric.

### 6.5 L2-boundary zoom (29–35 MB) — detail

Step-to-step GPU time / util (CUPTI Avg):

| Step | Δ algo IO | Δ GPU time | Δ BD util | Note |
|------|----------:|-----------:|----------:|------|
| 29 → 30 | +0.99 MB | +0.94 µs (**+3.0%**) | +0.8 pp | still L2-hot |
| 30 → 31 | +0.99 MB | +1.36 µs (**+4.2%**) | −1.8 pp | still L2-hot |
| 31 → 32 | +0.99 MB | +3.92 µs (**+11.6%**) | **−15.5 pp** | **first soft break** |
| 32 → 33 | +0.99 MB | +1.25 µs (**+3.3%**) | −0.4 pp | holds ~189% |
| 33 → 34 | +0.99 MB | +5.62 µs (**+14.5%**) | **−18.9 pp** | steepening |
| 34 → 35 | +1.18 MB | +10.88 µs (**+24.4%**) | **−28.6 pp** | steepest step in zoom |

At target 32 MB: `x` ≈ `out` ≈ **15.93 MB** → together ≈ **L2**. Beyond that, producer→consumer L2 reuse collapses over a few MB, not in one binary flip.

---

## 7. Nsight Compute confirmation (L2 hit vs DRAM)

**Yes.** `ncu` **2026.3.1** profiles this GB206 (RTX 5060 Ti). `ncu` 2025.1.1 did not (`LibraryNotLoaded`; `--list-chips` had no `gb206`).

**Protocol:** application replay, `--cache-control none`, `--clock-control none`, `--profile-from-start off`. Warmup **5** is outside `cudaProfilerStart`; **1** measured launch inside the range. Kernel `triton_poi_fused_native_layer_norm_2`. Unit test on this host: max abs **7.8e−3**, **PASS**.

Artifacts: `ncu/io{24,31,32,33,34,35,48,64}_ncu.log` + matching `io*_app.csv`.

ncu times are one warmed launch under unlocked clocks, so they are not the §6 CUPTI averages. Hit rate and DRAM bytes are the L2 evidence.

### 7.1 Counters

`x`+`out` = 2 × bf16 activation. **DRAM read / x** = `dram__bytes_op_read` / bytes(`x`). **Read / write hit** = L2 sector lookup hits from L1TEX (`lts__t_sectors_srcunit_tex_op_{read,write}_lookup_hit` / `_sum`). **L2 hit** = `lts__t_sector_hit_rate.pct`. **DRAM memory throughput** is the NCU Speed-of-Light **Memory Throughput** row: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` against the **448 GB/s** DRAM peak, with GB/s = (DRAM read + DRAM write) / kernel time. That column is DRAM, not L2.

| Algo IO | `x` buffer | `x`+`out` | Time | DRAM read | DRAM write | DRAM read / `x` | L2 hit | Read hit | Write hit | DRAM memory throughput |
|--------:|-----------:|----------:|-----:|----------:|-----------:|----------------:|-------:|---------:|----------:|-----------------------:|
| 24.05 MB | 11.99 MB | 24.0 MB | **18.0 µs** | 0.00 MB | 3.57 MB | **0%** | **100%** | **100%** | 100% | **198 GB/s · 45%** |
| 30.95 MB | 15.43 MB | 30.9 MB | **31.5 µs** | 2.50 MB | 7.65 MB | 16% | 86% | 84% | 88% | 322 GB/s · 73% |
| 31.94 MB | 15.93 MB | 31.9 MB | **40.4 µs** | 5.35 MB | 8.64 MB | 34% | 75% | **69%** | 81% | 346 GB/s · 79% |
| 32.92 MB | 16.42 MB | 32.8 MB | **49.0 µs** | 9.61 MB | 8.23 MB | 59% | 54% | 47% | 63% | 364 GB/s · 83% |
| 33.91 MB | 16.91 MB | 33.8 MB | **60.5 µs** | 14.0 MB | 8.36 MB | 83% | 37% | 25% | 49% | 370 GB/s · 84% |
| 35.09 MB | 17.50 MB | 35.0 MB | **67.8 µs** | 16.6 MB | 8.96 MB | **95%** | 29% | **13%** | 47% | 377 GB/s · 85% |
| 47.90 MB | 23.89 MB | 47.8 MB | **100 µs** | 24.0 MB | 13.9 MB | 101% | 11% | 6.7% | 15% | **379 GB/s · 86%** |
| 64.07 MB | 31.95 MB | 63.9 MB | **149 µs** | 32.1 MB | 24.1 MB | 101% | 3.5% | 6.7% | **0%** | **377 GB/s · 86%** |

24 MB has the highest L2 hit and the **lowest** DRAM throughput: loads never reach DRAM, and the 3.57 MB write-back is only ~198 GB/s (45% of 448). From 35 MB on, DRAM throughput sits at ~377 GB/s (85–86% of peak).

L1 global-load bytes stay ~**1.6×** the `x` buffer at every size. L2 read-sector bytes stay within ~**8%** of the real tensor, so that extra is intra-tile reuse of the same L2 sectors, not extra DRAM.

### 7.2 What this confirms in §6

| Regime | ncu | Matches §6 |
|--------|-----|------------|
| L2-resident (24 MB) | DRAM read **0**; L2 hit **100%**. The 3.57 MB DRAM write is dirty output flushed during the launch. | ⚠ BD util **>100%**: algo IO counts loads/stores L2 never sent to DRAM. |
| Onset (31→35 MB) | Read hit **84% → 13%**. DRAM read goes from **2.5 MB** to **16.6 / 17.5 MB** of `x`. | Soft break at ~32 MB, then a ramp, not one step. `x`+`out` ≈ L2 at the 32 MB target (each buffer ≈ 15.9 MB). |
| Post-L2 (48–64 MB) | DRAM read ≈ all of `x`. At 64 MB, DRAM write is **24.1 / 32.0 MB** and write hit is **0**. DRAM memory throughput **377 GB/s (86%)**. | Plateau ~**85%** of 448 GB/s. |

Writes lag reads through the zoom: DRAM write stays ~**8–9 MB** from 31 MB to 35 MB while DRAM read climbs **2.5 → 16.6 MB**. Stores allocate in L2 and many lines are still dirty at kernel exit. Algo IO grows **2.7×** from 24→64 MB; ncu time grows **8.3×**.

### 7.3 Reproduce

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/triton_poi_fused_native_layer_norm_2
source /venv/main/bin/activate
mkdir -p ncu
export PATH="/usr/local/bin:/usr/local/cuda/bin:$PATH"

METRICS="gpu__time_duration.avg,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum,\
lts__t_sector_hit_rate.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

for mb in 24 31 32 33 34 35 48 64; do
  ncu --target-processes all \
    --profile-from-start off \
    --cache-control none \
    --clock-control none \
    --replay-mode application \
    --kernel-name regex:triton_poi_fused_native_layer_norm \
    --launch-count 1 \
    --metrics "$METRICS" \
    --csv \
    --log-file "ncu/io${mb}_ncu.log" \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=1 CUPTI_RANGE=1 TARGET_IO_MB=$mb \
    python profile_ln_poi.py > "ncu/io${mb}_app.csv"
done
```

---

## 8. How to reproduce (nsys)

```bash
cd doc/gpu/compile_mode/operator_cases_analysis/triton_poi_fused_native_layer_norm_2
source /venv/main/bin/activate
export PATH=/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH

# Unit test (shapes / dtypes / strides / grid)
SMOKE_DEVICE=cuda python test_triton_poi_fused_native_layer_norm_2.py

mkdir -p nsys

# Baseline
nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-event-trace=false \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none \
  --output=nsys/ln_poi_baseline \
  env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 \
  python profile_ln_poi.py
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  --output=nsys/ln_poi_baseline nsys/ln_poi_baseline.nsys-rep

# Buffer-size sweep (+ L2 zoom)
for mb in 8 16 24 29 30 31 32 33 34 35 48 64 96 128; do
  nsys profile --force-overwrite=true \
    --trace=cuda,nvtx,osrt --cuda-event-trace=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --output=nsys/ln_poi_io${mb} \
    env SMOKE_DEVICE=cuda N_WARMUP=5 N_ITERS=10 CUPTI_RANGE=1 TARGET_IO_MB=$mb \
    python profile_ln_poi.py
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    --output=nsys/ln_poi_io${mb} nsys/ln_poi_io${mb}.nsys-rep
done
```

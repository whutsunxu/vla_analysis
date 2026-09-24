# SmolVLA detailed profiling report

Generated: `2026-09-23T11:47:45.007201+00:00`
Host: `a8b9265fc194`
Device: `cuda` — NVIDIA GeForce RTX 5060 Ti
Model: `lerobot/smolvla_base`
Warmup runs: 2 · Measured runs: 5
Smoke seed: `0` (fixed observation + CPU-sampled noise)

## 1. Environment

| Field | Value |
|---|---|
| Python | 3.12.14 |
| Torch | 2.11.0+cu128 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute capability | [12, 0] |
| Torch CUDA | 12.8 |
| VRAM total | 16.62 GB |
| SMs | 36 |

## 2. Model

| Field | Value |
|---|---|
| Params | 450,046,176 |
| Trainable | 99,880,992 |
| Weight footprint (approx) | 906.64 MB |
| chunk_size / n_action_steps / num_steps | 50 / 50 / 10 |
| Load time | 6.39 s |

### Params by dtype

| dtype | params | MB |
|---|---:|---:|
| torch.bfloat16 | 446,772,624 | 893.55 |
| torch.float32 | 3,273,552 | 13.09 |

### Params by device

| device | params | MB |
|---|---:|---:|
| cuda:0 | 450,046,176 | 906.64 |

## 3. Tensor shapes (first measured run)

Architecture reference dims (from Operator List): prefix `S_p=241, d_p=960`; suffix chunk `N=50`; Euler `M=10`.

| Tensor | Shape / value |
|---|---|
| `images` | `[[1, 3, 512, 512], [1, 3, 512, 512], [1, 3, 512, 512]]` |
| `img_masks` | `[[1], [1], [1]]` |
| `state` | `[1, 32]` |
| `lang_tokens` | `[1, 48]` |
| `lang_masks` | `[1, 48]` |
| `prefix_embs` | `[1, 241, 960]` |
| `prefix_pad_masks` | `[1, 241]` |
| `prefix_att_masks` | `[1, 241]` |
| `prefix_embs_dtype` | `torch.float32` |
| `prefix_att_2d_masks` | `[1, 241, 241]` |
| `prefix_position_ids` | `[1, 241]` |
| `use_cache` | `True` |
| `suffix_embs` | `[1, 50, 720]` |
| `suffix_pad_masks` | `[1, 50]` |
| `full_att_2d_masks` | `[1, 50, 291]` |
| `v_t` | `[1, 50, 32]` |
| `x_t` | `[1, 50, 32]` |
| `suffix_embs_dtype` | `torch.float32` |
| `v_t_dtype` | `torch.float32` |
| `action_chunk` | `[1, 50, 6]` |
| `action_step` | `[1, 6]` |

## 4. Stage wall-clock (CUDA-synchronized)

Timings exclude warmup. Each stage boundary calls `torch.cuda.synchronize()`.

| Stage | mean (s) | std | min | max | p50 | p90 | share % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 prefix embed | 0.023885 | 0.000052 | 0.023852 | 0.023987 | 0.023857 | 0.023875 | 26.39 |
| 1 suffix embed (sum×M) | 0.001926 | 0.000011 | 0.001917 | 0.001946 | 0.001921 | 0.001925 | 2.13 |
| 2 VLM prefill | 0.007184 | 0.000007 | 0.007172 | 0.007190 | 0.007187 | 0.007190 | 7.94 |
| 3 expert denoise (sum×M) | 0.057090 | 0.000154 | 0.056917 | 0.057319 | 0.057104 | 0.057184 | 63.49 |
| 3 euler update (sum×M) | 0.000124 | 0.000006 | 0.000118 | 0.000136 | 0.000123 | 0.000125 |  |
| 3 expert+euler | 0.057215 | 0.000156 | 0.057037 | 0.057442 | 0.057241 | 0.057310 |  |
| 3 inclusive (+stage1) | 0.059140 | 0.000152 | 0.058964 | 0.059367 | 0.059158 | 0.059230 |  |
| 4 crop/queue/post | 0.000052 | 0.000000 | 0.000052 | 0.000053 | 0.000052 | 0.000052 | 0.06 |
| stages 0–4 sum | 0.090262 | 0.000173 | 0.090039 | 0.090481 | 0.090321 | 0.090387 |  |
| wall clock total | 0.091327 | 0.000166 | 0.091123 | 0.091579 | 0.091347 | 0.091415 |  |

### Derived rates

| Metric | Value |
|---|---|
| Chunk-fill Hz (mean) | 10.95 |
| Action-step Hz if queue drained (mean) | 547.483 |
| ms / Euler step (stage1+3 mean) | 5.914 |
| ms Stage0 mean | 23.885 |
| ms Stage2 mean | 7.184 |
| ms Stage3 loop mean | 59.14 |

_chunk_fill_hz = 1 / wall_clock for one full 50-action chunk fill (prefix+prefill+10 Euler). action_step_hz assumes the 50 queued actions are consumed one-per-control-tick with no extra model calls._

## 5. Per-Euler-step breakdown (mean over measured runs)

| step | flow_t | stage1 mean (ms) | stage3 expert mean (ms) | euler upd mean (ms) | step total mean (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.211 | 5.828 | 0.015 | 6.054 |
| 1 | 0.9 | 0.192 | 5.738 | 0.012 | 5.942 |
| 2 | 0.8 | 0.192 | 5.723 | 0.011 | 5.926 |
| 3 | 0.7 | 0.190 | 5.700 | 0.012 | 5.902 |
| 4 | 0.6 | 0.188 | 5.690 | 0.012 | 5.891 |
| 5 | 0.5 | 0.191 | 5.698 | 0.012 | 5.902 |
| 6 | 0.4 | 0.193 | 5.683 | 0.011 | 5.887 |
| 7 | 0.3 | 0.190 | 5.684 | 0.012 | 5.886 |
| 8 | 0.2 | 0.189 | 5.669 | 0.013 | 5.871 |
| 9 | 0.1 | 0.188 | 5.678 | 0.012 | 5.879 |

## 6. CUDA memory

| Snapshot | allocated MB | reserved MB | max_allocated MB | max_reserved MB | free MB |
|---|---:|---:|---:|---:|---:|
| after_stage0 | 948.54 | 1006.63 | 969.09 | 1006.63 | 15411.25 |
| after_stage2 | 954.01 | 1006.63 | 970.96 | 1006.63 | 15411.25 |
| after_euler_loop | 954.45 | 1006.63 | 970.96 | 1006.63 | 15411.25 |
| end_last_run | 954.45 | 1006.63 | 970.96 | 1006.63 | 15411.25 |
| max_allocated_mb_across_runs | 970.96 |  |  |  |  |
| max_reserved_mb_across_runs | 1006.63 |  |  |  |  |

## 7. Action output (last measured run)

```json
{
  "shape": [
    1,
    6
  ],
  "dtype": "torch.float32",
  "device": "cpu",
  "min": -0.39810889959335327,
  "max": 0.19139112532138824,
  "mean": -0.09570863842964172,
  "finite": true,
  "sample": [
    -0.06281795352697372,
    -0.02878165990114212,
    -0.1933826208114624,
    0.19139112532138824,
    -0.08255180716514587,
    -0.39810889959335327
  ]
}
```

## 8. CPU vs GPU comparison

CPU baseline: `/workspace/vla_analysis/smoke_test_report_cpu.json` (575cd14fb466, torch `2.11.0+cpu`)

| Metric | CPU (s) | GPU mean (s) | Speedup | CPU share % | GPU share % |
|---|---:|---:|---:|---:|---:|
| stage0_prefix_embed_s | 15.0085 | 0.023885 | 628.37× | 74.56 | 26.15 |
| stage1_suffix_embed_s | 0.0114 | 0.001926 | 5.92× | 0.06 | 2.11 |
| stage2_prefill_s | 1.9332 | 0.007184 | 269.1× | 9.6 | 7.87 |
| stage3_expert_denoise_s | 3.1749 | 0.05709 | 55.61× | 15.77 | 62.51 |
| stage3_euler_update_s | 0.0003 | 0.000124 | 2.42× | 0.0 | 0.14 |
| stage3_expert_plus_euler_s | 3.1752 | 0.057215 | 55.5× | 15.77 | 62.65 |
| stage4_crop_queue_postprocess_s | 0.0001 | 5.2e-05 | 1.92× | 0.0 | 0.06 |
| stages_0_to_4_sum_s | 20.1284 | 0.090262 | 223.0× | 100.0 | 98.83 |
| wall_clock_total_s | 20.1288 | 0.091327 | 220.4× | 100.0 | 100.0 |

### Action parity (CPU vs last GPU run)

- max abs diff: `0.00165981`
- mean abs diff: `0.000870512`
- within 1e-2: `True`
- within 1e-1: `True`

| dim | CPU | GPU | abs diff |
|---:|---:|---:|---:|
| 0 | -0.06372098624706268 | -0.06281795352697372 | 0.000903033 |
| 1 | -0.029679380357265472 | -0.02878165990114212 | 0.00089772 |
| 2 | -0.19337990880012512 | -0.1933826208114624 | 2.71201e-06 |
| 3 | 0.19098371267318726 | 0.19139112532138824 | 0.000407413 |
| 4 | -0.08119942247867584 | -0.08255180716514587 | 0.00135238 |
| 5 | -0.39644908905029297 | -0.39810889959335327 | 0.00165981 |

## 9. Method notes

- Profiler mirrors `select_action` chunk-fill path (Stages 0→2→Euler{1,3}×M→4).
- Noise is sampled on CPU with `SMOKE_SEED` then moved to device (bit-identical across CPU/GPU).
- Observation is the fixed dummy frame (not libero).
- Stage 3 expert timing includes mask build + expert forward + `action_out_proj` + KV crop.
- First warmup run pays CUDA kernel compile / cudnn autotune cost; excluded from aggregates.

Machine-readable twin: `smolVLA_profiling.json`.

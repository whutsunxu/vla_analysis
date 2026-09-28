# SmolVLA GPU smoke test report

Date: 2026-09-23  
Host: Vast.ai instance `C.52216532` (`162.249.226.242:32983`)  
Script: `src/smolvla_test_infer.py` with `SMOKE_DEVICE=cuda`  
Baseline CPU: local `smoke_test_report.json` (same fixed observation + `SMOKE_SEED=0` noise)

## Verdict

**PASS.** GPU inference matches the CPU baseline within **~1.7e-3** max abs error (tighter than the 1e-2 / 1e-1 tolerance). Two consecutive GPU runs were bit-identical.

## CUDA environment setup

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB), compute capability **12.0 (Blackwell)** |
| Driver | 590.48.01 (driver max CUDA 13.1) |
| Host toolkit | CUDA **12.8** at `/usr/local/cuda` |
| Python | `/venv/main` (3.12.14) |
| PyTorch | **2.11.0+cu128** (required: Blackwell needs CUDA ≥ 12.8 wheels) |
| LeRobot | 0.6.1 |
| Transformers | 5.5.4 |
| Accelerate | 1.15.0 |

Install commands used on the remote:

```bash
source /venv/main/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot==0.6.1" "transformers==5.5.4" "accelerate==1.15.0" \
  "einops==0.8.2" "av==15.1.0" "datasets==4.8.5" "draccus==0.11.6"
pip install num2words sentencepiece protobuf
```

Torch CUDA smoke (matmul on `cuda:0`) succeeded before the SmolVLA run.

## Dependency issues fixed

1. **No PyTorch in `/venv/main`** — installed `torch==2.11.0+cu128` (not cu124; Blackwell would fail with “no kernel image”).
2. **Missing SmolVLM processor dep** — `ImportError: Package num2words is required` during `SmolVLAPolicy.from_pretrained`; fixed with `pip install num2words` (also installed `sentencepiece`, `protobuf`).
3. **CPU-only script** — updated `src/smolvla_test_infer.py` to honor `SMOKE_DEVICE=auto|cpu|cuda`, write `smoke_test_report_gpu.json` on CUDA, sample flow-matching noise on CPU then `.to(device)` so CPU/GPU share identical noise, and `torch.cuda.synchronize()` around stage timings.

## How the GPU run was executed

```bash
cd /workspace/vla_analysis
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export SMOKE_DEVICE=cuda
python src/smolvla_test_infer.py 2>&1 | tee smoke_test_run_gpu.log
```

Artifacts on the remote (and mirrored locally):

- `smoke_test_report_gpu.json`
- `smoke_test_run_gpu.log`
- `smoke_test_report_cpu.json` (copy of the local CPU baseline for side-by-side compare)

## GPU run summary

| Metric | GPU | CPU (baseline) |
|--------|-----|----------------|
| Status | PASS | PASS |
| Params | 450,046,176 on `cuda:0` | same on `cpu` |
| Dtypes | bfloat16 + float32 | bfloat16 + float32 |
| Model load | 9.56 s | ~19.7 s |
| `select_action` ×2 | **0.46 s** | 42.87 s |
| Stage 0–4 wall (warm) | **0.0897 s** | 20.13 s (~224×) |
| Determinism (2 runs) | identical, max_abs_diff=0 | identical |

### Stage profile (GPU, synchronized)

| Stage | Seconds | Share |
|-------|---------|-------|
| 0 prefix embed | 0.0238 | 26.7% |
| 1 suffix embed | 0.0019 | 2.1% |
| 2 prefill | 0.0071 | 8.0% |
| 3 expert+Euler | 0.0565 | 63.2% |
| 4 crop/queue/post | ~0 | ~0% |

## CPU vs GPU action comparison

Same inputs: fixed dummy observation, `SMOKE_SEED=0`, CPU-sampled noise moved to device.

| Dim | CPU | GPU | \|Δ\| |
|-----|-----|-----|------|
| 0 | -0.063721 | -0.062818 | 9.03e-4 |
| 1 | -0.029679 | -0.028782 | 8.98e-4 |
| 2 | -0.193380 | -0.193383 | 2.71e-6 |
| 3 | 0.190984 | 0.191391 | 4.07e-4 |
| 4 | -0.081199 | -0.082552 | 1.35e-3 |
| 5 | -0.396449 | -0.398109 | 1.66e-3 |

- **max abs error:** `1.66e-3`
- **mean abs error:** `8.71e-4`
- Tolerance check: **≤ 1e-2** ✓, **≤ 1e-1** ✓

Residual difference is expected from bf16 GPU kernels vs CPU bf16/float32 paths; not from input/noise mismatch.

## Notes

- Real-world CLI probe: `lerobot-record` / `lerobot-rollout` are present on the GPU box; `record` dry-run fails without robot hardware (expected).
- Local CPU default remains `SMOKE_DEVICE=cpu` (or unset with no GPU). On the remote, use `SMOKE_DEVICE=cuda` explicitly.

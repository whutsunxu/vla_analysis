# vla_analysis

Study and analysis of Vision-Language-Action (VLA) models, using **SmolVLA** as the primary case study.

## Goals

1. **Learn the VLA stack** — understand how a VLA maps cameras, language, and proprioception into continuous robot actions, with SmolVLA as a concrete, runnable reference.
2. **Analyze structure and performance** — document algorithm stages, operators, FLOPs, and wall-clock profiles so cost and bottlenecks are explicit.
3. **Optimize later** — use that analysis to improve the model at **framework**, **model-graph**, and **IR** levels (planned follow-on work).

## Repository layout

```
doc/   Architecture, operator list, and condensed report (SmolVLA)
src/   CPU smoke-test / inference harness for SmolVLA
```

| Path | Content |
|------|---------|
| [`doc/SmolVLA_Algorithm_Architecture.md`](doc/SmolVLA_Algorithm_Architecture.md) | Stage-level algorithm and tensor flow |
| [`doc/SmolVLA_Operator_List.md`](doc/SmolVLA_Operator_List.md) | Operators and FLOP accounting |
| [`doc/smolVLA_report.md`](doc/smolVLA_report.md) | Condensed report (runtime, stages, CPU profile) |
| [`src/smolvla_test_infer.py`](src/smolvla_test_infer.py) | CPU `select_action` smoke test + Stage 0–4 timing |

## Status

- **Now:** SmolVLA study case — structure docs + CPU inference / stage profiling.
- **Next:** Optimize at framework, model-graph, and IR levels based on the measured structure and performance.

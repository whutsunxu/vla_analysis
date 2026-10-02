# Dynamo → Inductor Pipeline (ATen Extern + Triton)

Walkthrough of how `torch.compile(..., backend="inductor")` turns Python / ATen ops into either **ATen external kernels** or **generated Triton kernels**.

Pinned to local tree: PyTorch **v2.11.0** (`70d99e9`) under `/root/workspace/pytorch`.

Paths below are relative to that tree (e.g. `torch/__init__.py` → `/root/workspace/pytorch/torch/__init__.py`) unless noted otherwise.

Related SmolVLA artifacts:

- Dynamo / FX ops: [`dynamo/SmolVLA_op_list_dynamo.md`](dynamo/SmolVLA_op_list_dynamo.md)
- Inductor / nsys kernels: [`inductor/SmolVLA_op_list_inductor_nsight.md`](inductor/SmolVLA_op_list_inductor_nsight.md)
- Sample generated wrapper: [`inductor_debug_samples/model__4_inference_4.4/output_code.py`](inductor_debug_samples/model__4_inference_4.4/output_code.py)

---

## 0. End-to-end map

```text
torch.compile(fn, backend="inductor")
        │
        ▼
┌─────────────────────── Dynamo ───────────────────────┐
│ eval_frame (CPython hook)                            │
│   → convert_frame / trace_frame                      │
│   → InstructionTranslator (bytecode → FX)            │
│   → OutputGraph.compile_subgraph                     │
│   → OutputGraph.call_user_compiler  (= inductor)    │
└──────────────────────────┬───────────────────────────┘
                           │  FX GraphModule (aten / prims)
                           ▼
┌────────────────── Inductor entry ────────────────────┐
│ compile_fx  (torch/_inductor/compile_fx.py)          │
│   (1) pre-grad FX passes                             │
│   (2) aot_autograd → joint / fw / bw / inference     │
│   (3) compile_fx_inner: post-grad FX passes          │
│   (4) GraphLowering → IR → codegen                   │
└──────────────────────────┬───────────────────────────┘
                           │  Inductor IR (after GraphLowering)
                           ▼
┌────────────────── GraphLowering ─────────────────────┐
│ GraphLowering.run  (FX Interpreter)                  │
│   call_function → lowering.py registry               │
│     ├─ make_pointwise / make_reduction → IR buffers  │
│     ├─ templates / autotune (mm, conv, …)            │
│     └─ FallbackKernel / ExternKernel* → ATen call    │
└──────────────────────────┬───────────────────────────┘
                           │  Inductor IR (buffers + ops)
                           ▼
┌──────────────────── Scheduler ───────────────────────┐
│ Scheduler: deps, fusion, scheduling                  │
│   SchedulerNode          → Triton / SIMD path        │
│   ExternKernelSchedulerNode → wrapper ATen call      │
└──────────────────────────┬───────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   TritonKernel.codegen_kernel    PythonWrapperCodegen
   TritonScheduling.define_kernel   generate_extern_kernel_*
   async_compile.triton(...)        extern_kernels.<op>(...)
            │                             │
            └──────────────┬──────────────┘
                           ▼
              output_code.py  →  PyCodeCache module
              call = compiled_module.call
```

Two kernel families in the final wrapper:

| Path | IR node | Scheduler node | Wrapper emit | Runtime |
|---|---|---|---|---|
| **Triton (fused)** | `ComputedBuffer` / `Pointwise` / `Reduction` / `TritonTemplateBuffer` | `SchedulerNode` / fused | `async_compile.triton(...)` + grid launch | Generated Triton → CUDA |
| **ATen extern** | `ExternKernel` / `ExternKernelOut` / `ExternKernelAlloc` / `FallbackKernel` | `ExternKernelSchedulerNode` | `extern_kernels.<name>(...)` or `aten.*.default(...)` | Existing ATen / cuDNN / cuBLAS / custom op |

### Stage I/O summary

| Stage | § | Input | Output |
|---|---|---|---|
| `torch.compile` wrap | 1 | Python callable (`sample_actions`) + `mode`/`backend` | Dynamo-wrapped callable (eval-frame armed); **no FX / no kernels yet** |
| Dynamo capture | 2 | Function bytecode + real/FakeTensor example args on first call | One or more FX `GraphModule`s (`aten`/`prims` nodes) + example inputs; eager glue between graph breaks |
| Inductor `compile_fx` + AOT | 3 | Dynamo FX `GraphModule` + `example_inputs` | After pre-grad / AOT / post-grad FX rewrites: inference or fw(+bw) FX ready for lowering. Final return: AOT boxed compiled callable |
| Pre-grad passes | 3 | Pre-AOT Dynamo FX | Rewritten FX (`pre_grad_passes`: fusion / sink / normalize) |
| Joint graph passes | 3 | Joint FX (training path) | Canonicalized / optimized joint FX (`joint_graph_passes`) |
| Post-grad passes | 3 | Partitioned or inference FX (inside `compile_fx_inner`) | Rewritten FX still (`post_grad_passes`: pattern fusion) — **not** Inductor IR yet |
| `GraphLowering` / lowering | 4 | Post-grad FX + example inputs | Inductor IR: buffers + ops (`ComputedBuffer`, `ExternKernel*`, templates) |
| Scheduler fuse + dispatch | 5 | Inductor IR ops/buffers | Ordered / fused schedule; path split to Triton vs extern emitters |
| Triton codegen | 6 | Fused / Loop `SchedulerNode`(s) | Triton `@triton.jit` **body** + **launch meta** (grid_type / BLOCKs / heuristics); tune → grid + warps at compile/launch |
| ATen extern emit | 7 | `ExternKernelSchedulerNode` | Wrapper calls: `extern_kernels.*` / `aten.*.default(...)` |
| Wrapper + cache | 8 | All kernel defs + launch schedule | `output_code.py` → `PyCodeCache` module with `.call` |

Chaining (output of one stage is input of the next):

```text
callable
  → wrapped callable
    → FX GraphModule(s) + example_inputs
      → pre-grad FX → AOT → post-grad FX     # §3 (all still FX)
        → Inductor IR                          # §4 GraphLowering
          → scheduled / fused nodes
            → Triton src + extern calls inside wrapper text
              → loaded Python module (.call)
```

---

## 1. Entry: `torch.compile` → Dynamo eval frame

| | |
|---|---|
| **Input** | User callable (e.g. `VLAFlowMatching.sample_actions`); kwargs: `backend`, `mode`/`options`, `fullgraph`, `dynamic`, `disable` |
| **Output** | Wrapped callable (`OptimizeContext(fn)`). Inductor config patches stored on `_TorchCompileInductorWrapper`. **Not** an FX graph yet |

**Files:** `torch/__init__.py` (`compile`, `_TorchCompileInductorWrapper`), `torch/_dynamo/eval_frame.py` (`optimize`, `OptimizeContext`)

`torch.compile` is the public API. It does **not** run Inductor immediately: it builds a Dynamo-optimized wrapper around the callable. Compilation / lowering starts on the **first real call** (or on guard failure).

SmolVLA (LeRobot) wires it as:

```python
# modeling_smolvla.py — VLAFlowMatching.__init__
self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
self.forward = torch.compile(self.forward, mode=config.compile_mode)
# backend defaults to "inductor"; mode e.g. "reduce-overhead"
```

### 1.1 Key components of `torch.compile`

Defined on `torch.compile` in `torch/__init__.py`. Helpers for resolving presets live under Inductor.

| Component | Type / values | Source | Role |
|---|---|---|---|
| **`model` / callable** | function or bound method (e.g. `sample_actions`) | caller (LeRobot: `.../lerobot/policies/smolvla/modeling_smolvla.py`) | Region whose frames Dynamo will intercept |
| **`backend`** | default `"inductor"`; or custom callable / other registered name | arg to `torch.compile`; registry `torch/_dynamo/backends/registry.py` (`lookup_backend`) | Who compiles each captured FX `GraphModule` |
| **`mode`** | `"default"` / `"reduce-overhead"` / `"max-autotune"` / `"max-autotune-no-cudagraphs"` | applied via `_TorchCompileInductorWrapper.apply_mode` → `torch/_inductor/__init__.py` (`list_mode_options`) | Preset Inductor config (CUDA graphs, autotune, …) |
| **`options`** | `dict` (mutually exclusive with `mode`) | applied via `_TorchCompileInductorWrapper.apply_options` → knobs listed by `torch/_inductor/__init__.py` (`list_options`) | Fine-grained Inductor knobs (`triton.cudagraphs`, `max_autotune`, `epilogue_fusion`, …) |
| **`fullgraph`** | `bool` → Dynamo `nopython` | `torch.compile` → `torch/_dynamo/eval_frame.py` (`optimize` / `_optimize`) | If `True`, graph breaks raise; aim for one whole-program graph |
| **`dynamic`** | `True` / `False` / `None` | same (`optimize` / `_optimize`) | Dynamic-shape policy: force dynamic, always specialize, or auto-detect on recompile |
| **`disable`** | `bool` | same (`optimize` / `_optimize` → `_NullDecorator`) | Turn compile into a no-op (testing) |

Notes:

- Pass **either** `mode` **or** `options`, not both (`torch.compile` raises if both are set).
- If both are omitted, `mode` becomes `"default"`.
- `fullgraph=True` also opts into stronger unbacked / dynamic-output capture defaults (see docstring on `torch.compile`).

### 1.2 What `torch.compile` builds internally

For `backend="inductor"` (the SmolVLA path):

1. Construct **`_TorchCompileInductorWrapper(mode, options, dynamic)`** — `torch/__init__.py`
   - `apply_mode(mode)` → patches from `list_mode_options(mode, dynamic)` (`torch/_inductor/__init__.py`) into `self.config`
   - `apply_options(options)` → extra Inductor config overrides
   - Example: `mode="reduce-overhead"` turns on CUDA-graph-related Inductor settings to cut Python overhead
2. Call Dynamo (`torch/_dynamo/eval_frame.py`):

```python
return torch._dynamo.optimize(
    backend=backend,      # the Inductor wrapper above
    nopython=fullgraph,
    dynamic=dynamic,
    disable=disable,
    guard_filter_fn=...,
)(model)
```

3. `optimize` → `_optimize` → **`OptimizeContext`** (`torch/_dynamo/eval_frame.py`) — see §1.2.1. This chain does **not** run Inductor yet; it builds a context/decorator that wraps the callable so Dynamo can intercept it later.

When Dynamo later finishes an FX subgraph, the Inductor wrapper’s `__call__` is the backend (`torch/__init__.py`):

```python
# _TorchCompileInductorWrapper.__call__
return compile_fx(model_, inputs_, config_patches=self.config)
# compile_fx ← torch/_inductor/compile_fx.py
```

So: **`mode` / `options` become `config_patches` on every Inductor `compile_fx` invoke.**

### 1.2.1 `optimize` → `_optimize` → `OptimizeContext`

All three live in `torch/_dynamo/eval_frame.py`. End of `torch.compile`:

```text
optimize(backend=...)(fn)   →   wrapped fn
```

That chain **does not compile yet**. It builds a **context/decorator** that, when applied to `fn`, returns a callable Dynamo can intercept on the next real call.

#### `optimize` — thin public entry

- User-facing Dynamo API (`@torch._dynamo.optimize` / what `torch.compile` calls).
- Mostly forwards into `_optimize`.
- Keeps a small `rebuild_ctx` closure so Dynamo can recreate the same optimize setup later (e.g. compiled autograd).

**Job:** “Start Dynamo optimize with these kwargs.” Almost no real work.

#### `_optimize` — configure *how* Dynamo will compile

Docstring: *graph capture, then call `backend()` on extracted graphs.* Roughly:

1. **Sanity / kill switches** — Dynamo supported? `disable` / `TORCHDYNAMO_DISABLE`? → maybe return a no-op (`_NullDecorator`).
2. **Pack guard hooks** — optional callbacks when guards are exported / fail / filtered.
3. **`fullgraph` / `nopython`** — if true, take the strict path (`optimize_assert`: graph breaks = errors).
4. **Resolve backend** — `get_compiler_fn(backend)` → string/wrapper becomes a real compiler (SmolVLA: Inductor wrapper → eventually `compile_fx`).
5. **Wire convert_frame** — build `convert_frame(backend, hooks)` = “when a frame is captured, turn it into FX and call the backend.”
6. **Return `OptimizeContext`** — via `_optimize_catch_errors(...)`, wrapping that convert path in error-catching.

**Job:** Assemble the policy (backend, graph-break rules, dynamic shapes, guard hooks) and hand back a context object. Still no Inductor run.

#### `OptimizeContext` — wrap `fn` and arm the eval-frame hook

Subclass of `_TorchDynamoContext`.

**On construction** it stores the convert-frame **callback**, backend context, `dynamic`, `fullgraph`, compiler config, etc.

**When you do `OptimizeContext(fn)`** (the `(model)` at the end of `torch.compile`):

- If `fn` is an `nn.Module` → wrap `forward` via `OptimizedModule`.
- If `fn` is a function/method (SmolVLA: `sample_actions`) → return a **wrapper** around that callable.

**When that wrapper actually runs** (first `sample_actions(...)`):

1. Enter context → **`set_eval_frame(callback)`** (`torch._C._dynamo.eval_frame`) so CPython asks Dynamo before interpreting bytecode.
2. Run the original function under that hook → Dynamo may capture graphs and call the backend.
3. Exit → restore the previous eval-frame callback.

**Job:** Turn “optimize this callable” into “whenever this callable runs, Dynamo’s frame callback is active.”

| Piece | Source | Does |
|---|---|---|
| **`optimize`** | `torch/_dynamo/eval_frame.py` | Public entry; forwards to `_optimize` |
| **`_optimize`** | `torch/_dynamo/eval_frame.py` | Validates options, resolves backend, builds convert_frame pipeline, returns context |
| **`_optimize_catch_errors`** | `torch/_dynamo/eval_frame.py` | Wraps convert_frame in catch-errors; constructs `OptimizeContext` |
| **`OptimizeContext`** | `torch/_dynamo/eval_frame.py` | Wraps your function; on call, installs eval-frame hook so Dynamo can intercept bytecode |
| **`set_eval_frame`** | `torch._C._dynamo.eval_frame` (used from `eval_frame.py`) | CPython hook that routes frames into Dynamo |

```text
optimize          →  “please optimize with these settings”
_optimize         →  “here is the compiler + convert_frame wiring”
OptimizeContext   →  “here is your wrapped fn; on run, arm Dynamo”
(later call)      →  bytecode intercepted → FX → backend (Inductor)
```

### 1.3 Lowering from this stage (what happens next)

```text
torch.compile(fn, mode=...)                    # torch/__init__.py
        │
        ├─ _TorchCompileInductorWrapper         # torch/__init__.py
        │     (mode/options → inductor config_patches)
        │
        ▼
torch._dynamo.optimize(backend=wrapper)(fn)     # torch/_dynamo/eval_frame.py
        │
        ├─ OptimizeContext / set_eval_frame     # eval_frame.py + torch._C._dynamo.eval_frame
        │     returns wrapped fn   ← after LeRobot lines 536–537; Inductor not run yet
        │
        ▼  first call to sample_actions (or guard fail)
eval_frame intercepts that frame’s bytecode
        │
        ▼
convert_frame / InstructionTranslator           # §2 — convert_frame.py / symbolic_convert.py
        │  FX GraphModule(s)
        ▼
wrapper(gm, example_inputs)
  → compile_fx(..., config_patches=...)         # §3 — torch/_inductor/compile_fx.py
```

| Step | Function / class | Source | Output |
|---|---|---|---|
| Public API | `torch.compile` | `torch/__init__.py` | config + call into `_dynamo.optimize` |
| Backend object | `_TorchCompileInductorWrapper` | `torch/__init__.py` | callable backend; holds Inductor `config_patches` |
| Mode presets | `list_mode_options` | `torch/_inductor/__init__.py` | dict merged into wrapper `config` |
| Dynamo entry | `optimize` → `_optimize` → `OptimizeContext` | `torch/_dynamo/eval_frame.py` | wrapped callable; eval-frame hook armed |
| Frame hook (C) | `set_eval_frame` | `torch._C._dynamo.eval_frame` (used from `eval_frame.py`) | intercept next bytecode of compiled region |
| Backend resolve | `lookup_backend` / `get_compiler_fn` | `torch/_dynamo/backends/registry.py`, `torch/_dynamo/eval_frame.py` | resolved compiler callable |
| On first call | `convert_frame` / `trace_frame` (§2) | `torch/_dynamo/convert_frame.py` | one or more FX graphs |
| Backend invoke | `wrapper.__call__` → `compile_fx` (§3) | `torch/__init__.py` → `torch/_inductor/compile_fx.py` | AOT + Inductor compiled module |

Guards (shape, dtype, device, constants, …) decide whether a cached compiled code object can be reused. Guard failure → recompile (up to `torch._dynamo.config.recompile_limit`, default 8); beyond that Dynamo may fall back to eager.

**Part 1 takeaway:** `torch.compile` is a thin **config + Dynamo wiring** layer. The important objects are the **callable**, the **Inductor backend wrapper** (`mode`/`options` → `config_patches`), and the **eval-frame hook**. Actual graph lowering starts only when the wrapped method runs and Dynamo hands FX to `compile_fx`.

---

## 2. Dynamo: bytecode → FX graph

| | |
|---|---|
| **Input** | Wrapped callable’s **CPython bytecode** + live frame locals/args (tensors become FakeTensors while tracing) |
| **Output** | Per capturable stretch: FX `GraphModule` (mostly `aten.*` / `prims.*` nodes) + `example_inputs`. On graph break: that subgraph is compiled, eager runs the break site, then a **new** FX graph may start. SmolVLA: often **multiple** FX graphs from one `sample_actions` call |

**Files:** `torch/_dynamo/convert_frame.py`, `torch/_dynamo/symbolic_convert.py`, `torch/_dynamo/output_graph.py`

### 2.1 Convert frame

| Function | Source | Role |
|---|---|---|
| `convert_frame` / `convert_frame_assert` | `torch/_dynamo/convert_frame.py` | Wrap compiler; catch / assert on graph breaks |
| `trace_frame` | `torch/_dynamo/convert_frame.py` | Build `InstructionTranslator`, run bytecode analysis |
| `InstructionTranslator` | `torch/_dynamo/symbolic_convert.py` | Symbolic execution of Python bytecode |
| `OutputGraph` | `torch/_dynamo/output_graph.py` | Owns FX graph construction for one frame |

`trace_frame` constructs an `InstructionTranslator` and runs it under Dynamo’s tracing context. Each tensor op becomes an FX node (typically `aten.*` / `prims.*` after decompositions inside Dynamo/AOT).

### 2.2 Graph break vs fullgraph

- Default `fullgraph=False`: unsupported / data-dependent ops (e.g. `aten.nonzero`) **break** the graph; Dynamo resumes eager then starts a new graph.
- `fullgraph=True`: break → error.

SmolVLA `sample_actions` under `reduce-overhead` saw multiple graphs largely from dynamic-shape ops (see Dynamo op list).

### 2.3 Hand-off to Inductor

When Dynamo finishes a subgraph:

1. `OutputGraph.compile_subgraph` (`torch/_dynamo/output_graph.py`) finalizes the FX `GraphModule`.
2. `OutputGraph.call_user_compiler` (`torch/_dynamo/output_graph.py`) invokes the backend (`compile_fx` in `torch/_inductor/compile_fx.py`) with `(gm, example_inputs)`.
3. Backend returns a callable; Dynamo generates bytecode that calls it and resumes the frame.

At this point Inductor still sees **high-level FX**, not Triton yet.

---

## 3. Inductor entry + AOT Autograd (+ FX passes)

| | |
|---|---|
| **Input** | Dynamo FX `GraphModule` + `example_inputs` (from `call_user_compiler` → `compile_fx`) |
| **Output** | AOT boxed callable that runs compiled inference (or fw+bw). FX stays FX through **pre-grad → AOT → post-grad**; IR lowering starts only in §4 (`GraphLowering`) |

**File:** `torch/_inductor/compile_fx.py`
**AOT:** `torch/_functorch/aot_autograd.py`
**FX passes:** `torch/_inductor/fx_passes/{pre_grad,joint_graph,post_grad}.py`

`compile_fx` is the Inductor-registered Dynamo backend. It does **not** lower to Triton itself; it orchestrates AOT Autograd and FX rewrite passes, then `compile_fx_inner` (which still runs post-grad FX fusion before §4 lowering).

Sub-stage I/O:

| Sub-stage | Where | Input | Output |
|---|---|---|---|
| `run_pre_grad_passes` / `pre_grad_passes` | `_compile_fx_main` | Dynamo FX | Rewritten FX (fusion / sink / normalize) |
| AOT capture + decomps | AOT | Pre-grad FX + examples | Joint fw+bw FX **or** inference FX (`needs_autograd`) |
| `joint_graph_passes` | AOT (training) | Joint FX | Canonicalized / optimized joint FX |
| Partition | AOT (training) | Joint FX | Separate fw FX + bw FX |
| `_recursive_post_grad_passes` / `post_grad_passes` | `_compile_fx_inner` (~L1344) | Partitioned or inference FX | Rewritten FX (pattern matcher / fusion) — **still FX** |
| `GraphLowering` + codegen | §4 onward | Post-grad FX | IR → wrapper / kernels |
| AOT stitch | AOT | Compiled piece(s) | Single runtime callable |

Documented steps:

1. **Pre-grad passes** — `run_pre_grad_passes(gm, example_inputs)` on the Dynamo FX graph (`_compile_fx_main`).
2. Build **`fw_compiler` / `bw_compiler` / `inference_compiler`** closures around `inner_compile` (default `compile_fx_inner`).
3. Call **`aot_autograd`** / `aot_module_simplified`:
   - **(3a)** Trace joint forward+backward with **decompositions** (`select_decomp_table()`), or inference-only when no autograd.
   - **(3b)** Partition joint graph into fw / bw; run `joint_graph_passes`.
   - **(3c)** Invoke compilers → each enters `compile_fx_inner`.
4. **Post-grad passes** (still §3 / FX) — at the start of real compile inside `compile_fx_inner` / `_InProcessFxCompile.codegen_and_compile`:
   - `view_to_reshape` / FakeTensor prop, then **`_recursive_post_grad_passes(gm)` → `post_grad_passes`** (~L1344).
   - Same *kind* of work as pre-grad: pattern fusion on an FX graph, not Triton IR.
5. Then §4: `GraphLowering(gm).run(...)` lowers that post-grad FX to Inductor IR.

### 3.1 FX pass layers (all still GraphModule)

| Pass | Timing | Role |
|---|---|---|
| `pre_grad_passes` | Before AOT | Fusion / sink / normalize on Dynamo FX |
| `joint_graph_passes` | On joint graph (train) | Canonicalize + joint opts before partition |
| `post_grad_passes` | Inside `compile_fx_inner`, before `GraphLowering` | Post-AOT pattern matcher / fusion on fw, bw, or inference FX |

```text
Dynamo FX
  → pre_grad_passes          # FX → FX
  → AOT (+ joint_graph_passes / partition or inference)
  → post_grad_passes         # FX → FX   ← belongs with §3, not §4
  → GraphLowering.run        # FX → Inductor IR  ← §4
```

Key functions:

| Function | Source | Role |
|---|---|---|
| `compile_fx` | `torch/_inductor/compile_fx.py` | Dynamo backend entry; ownership of `GraphModule` |
| `_compile_fx_main` | `torch/_inductor/compile_fx.py` | Pre-grad + AOT wiring |
| `run_pre_grad_passes` / `_recursive_pre_grad_passes` | `torch/_inductor/compile_fx.py` | Invoke pre-grad FX pass pipeline on Dynamo graph |
| `pre_grad_passes` | `torch/_inductor/fx_passes/pre_grad.py` | Pre-grad fusion / sink / normalize (e.g. `group_batch_fusion`, `sink_cat_after_pointwise`, `fuse_conv_bn`) |
| `joint_graph_passes` | `torch/_inductor/fx_passes/joint_graph.py` | Canonicalize + opts on joint fw+bw graph (before partition) |
| `select_decomp_table` | `torch/_inductor/decomposition.py` | ATen decompositions for AOT |
| `compile_fx_forward` / `compile_fx_backward` | `torch/_inductor/compile_fx.py` | Per-direction Inductor compile |
| `compile_fx_inner` / `_compile_fx_inner` | `torch/_inductor/compile_fx.py` | Single-graph Inductor compile entry (cache + `fx_codegen_and_compile`) |
| `_recursive_post_grad_passes` | `torch/_inductor/compile_fx.py` | Invoke post-grad FX pass pipeline **before** `GraphLowering` (~L1344) |
| `post_grad_passes` | `torch/_inductor/fx_passes/post_grad.py` | Post-grad pattern matcher / fusion (FX → FX) |
| `aot_module_simplified` | `torch/_functorch/aot_autograd.py` | AOT dispatcher used by Inductor |
| `create_aot_state` | `torch/_functorch/aot_autograd.py` | Joint graph + partition state |

Inference (`torch.inference_mode` / no grad) still goes through AOT with an inference compiler path (no BW compile); freezing can use `fw_compiler_freezing` (`torch/_inductor/compile_fx.py`).

---

## 4. FX → Inductor IR (`GraphLowering`)

| | |
|---|---|
| **Input** | **Post-grad** FX `GraphModule` + `example_inputs` (post-grad already applied in §3) |
| **Output** | Inductor **IR** graph: buffers + ops (`Pointwise`/`ComputedBuffer`, `ExternKernel*`, templates, views, …) — still not CUDA binaries |

**Files:**

| Role | Path |
|---|---|
| Call site | `torch/_inductor/compile_fx.py` (~L1451–1485) |
| Driver | `torch/_inductor/graph.py` (`GraphLowering`) |
| Op → IR rules | `torch/_inductor/lowering.py` (`lowerings`, `make_pointwise`, `make_fallback`, …) |
| IR type system | `torch/_inductor/ir.py` |
| Template / autotune choices | `torch/_inductor/select_algorithm.py` |

Call site:

```python
graph = GraphLowering(gm, example_inputs=example_inputs, ...)
graph.run(*example_inputs)   # FX Interpreter → IR
```

### 4.0 Workflow overview

```text
                    GraphLowering.run  (DRIVER)
                            │
              for each FX node (placeholders → call_function → output)
                            │
                            ▼
                  call_function(target, args, kwargs)
                            │
                            ▼
                  lowerings[target](...)     # registry in lowering.py
                            │
          ┌─────────────────┼──────────────────┬──────────────────┐
          ▼                 ▼                  ▼                  ▼
   make_pointwise /   make_reduction /   select_algorithm /   make_fallback /
   register_pointwise views / layouts    templates            implicit fallback
          │                 │                  │                  │
          └────────┬────────┴────────┬─────────┴────────┬─────────┘
                   ▼                 ▼                  ▼
            classes in ir.py  (Pointwise, ComputedBuffer, ExternKernel*, …)
                   │
                   ▼
            IR graph owned by GraphLowering  →  hand off to Scheduler (§5)
```

| Layer | Role |
|---|---|
| **`GraphLowering` / `run`** | **Driver** — walks FX; does not invent IR shapes itself |
| **`ir.py` classes** | **Foundation** — vocabulary of IR nodes the rest of Inductor understands |
| **`lowering.py` helpers** | **Per-op mapping** — build those IR nodes for each ATen target |

§4 ends when IR exists. `codegen` / `compile_to_module` on the same class run later (§5–8).

---

### 4.1 Driver: `GraphLowering` / `run`

`GraphLowering` subclasses `torch.fx.Interpreter` (`torch/_inductor/graph.py`).

| Method | What it does in §4 |
|---|---|
| `GraphLowering(...)` | Bind FX `gm`, shape env, example inputs, device/wrapper flags |
| `run(*example_inputs)` | Drive the Interpreter over all FX nodes in order |
| `run_node` | Per-node dispatch; keep origins / meta for provenance |
| `call_function` | Resolve `target` in `lowerings` dict and call the registered handler; if missing → `make_fallback` / implicit fallback (when enabled) |

Key behavior in `call_function` (~L1293+):

1. Look up `lowerings[target]`.
2. If absent and `config.implicit_fallbacks` (or allow-list): register a fallback on the fly.
3. Else error / missing-op path.
4. Handler returns IR (`TensorBox` wrapping buffers / views / externs), stored as the FX node’s value for downstream users.

**Mental model:** `run` is the loop; `call_function` is the switch that turns each FX `aten`/`prims` node into IR.

---

### 4.2 Foundation: key classes in `ir.py`

`torch/_inductor/ir.py` defines the **IR type system**. Lowerings construct these; Scheduler / codegen consume them.

#### Base / graph plumbing

| Class | Key features | Role |
|---|---|---|
| `IRNode` | Abstract base; `origins` / `origin_node` / traceback; read deps | Common interface for everything in the IR graph |
| `Operation` | Logical compute / side-effecting op | Marks nodes that “do work” (vs pure views) |
| `Buffer` | Named storage; layout (size/stride/device/dtype) | Memory object codegen will allocate or bind |
| `OperationBuffer` | `Buffer` + `Operation` | Buffer whose contents are produced by an op |
| `OutputSpec` / `Layout` / `FixedLayout` / `FlexibleLayout` | Size, stride, contiguity constraints | How tensors are laid out; drives fusion / copy decisions |

#### Loop-level compute (fuseable into Triton later)

| Class | Key features | Role |
|---|---|---|
| `Loops` | Index ranges + `inner_fn` over symbolic indices | Base for elementwise / reduction bodies |
| `Pointwise` | `inner_fn(index) → value`; no reduction dims | One elementwise map — **produced by `make_pointwise`**; Scheduler may fuse many of these |
| `Reduction` | Reduction ranges + type (`sum`, `max`, …) | Reductions over dims — from `make_reduction` |
| `Scatter` | Pointwise write with scatter indexer | Sparse / indexed stores |
| `Scan` / `Sort` | Specialized loop patterns | Prefix / sort style ops when lowered natively |

#### Realized / input buffers

| Class | Key features | Role |
|---|---|---|
| `ComputedBuffer` | `data: Loops` (Pointwise/Reduction/…) | Realized result of loop IR — main **Triton fusion candidate** in §5 |
| `InputBuffer` / `DonatedBuffer` / `ConstantBuffer` | Graph inputs / constants | Edges from outside the compiled subgraph |
| `TensorBox` / `StorageBox` | Mutable box around storage/views | What most lowerings return/pass between FX nodes (user-facing IR handle) |

#### Views (usually no new kernel)

| Class | Key features | Role |
|---|---|---|
| `BaseView` + `ExpandView` / `PermuteView` / `View` / `SliceView` / `DtypeView` / … | Reinterpret size/stride/dtype without copying | Cheap metadata ops; often fused away or become indexing |

#### Templates (autotuned kernels)

| Class | Key features | Role |
|---|---|---|
| `TemplateBuffer` | Fixed kernel template body | Abstract templated op |
| `TritonTemplateBuffer` | Triton template choice | GEMM-like / attention templates → Triton path |
| `CUTLASSTemplateBuffer` / `CppTemplateBuffer` / … | Other backends | Alternate template realizations |
| `ChoiceCaller` / template callers | Autotune alternatives | Used with `select_algorithm` to pick winner |

#### Extern / fallback (call ATen, don’t generate body)

| Class | Key features | Role |
|---|---|---|
| `ExternKernel` | Not Loop IR; `op_overload` / kernel name | Schedule a call into ATen / library |
| `ExternKernelOut` | Out= / in-place style | Extern writing into provided storage |
| `ExternKernelAlloc` | Allocates its output | Extern that returns a new tensor |
| `FallbackKernel` | Generic unsupported / hard ops | Default “run eager ATen” IR for `make_fallback` |
| `MultiOutput` | Unpack multi-return extern | Split tuple outputs of fallbacks / multi-out kernels |
| `NopKernel` / `ConcatKernel` | Special-cased structural ops | Sometimes avoid a real kernel |

**Mapping rule of thumb:** Loop IR (`Pointwise`/`Reduction` → `ComputedBuffer`) → later Triton fusion; `ExternKernel*` / `FallbackKernel` → wrapper ATen call (§7).

---

### 4.3 Particular op mapping: `make_pointwise` (+ `register_pointwise`)

**Does not fuse.** It builds **one** `Pointwise` IR node for **one** elementwise ATen op. Fusion of several pointwise nodes is §5 (`Scheduler`).

How it works (`lowering.py`):

1. `make_pointwise(fn)` returns a closure over IR inputs.
2. Loads inputs at a symbolic index, runs `fn` (usually `ops.add` / `ops.relu` / …), returns `Pointwise.create(...)`.
3. Typically wrapped by `register_pointwise(aten_fn)` → registers into `lowerings` via `@register_lowering`.

Examples (all register into `lowerings` and ultimately create `Pointwise`):

| FX / ATen target | Registration style | IR result |
|---|---|---|
| `aten.add` | `add = register_pointwise(aten.add, allow_alpha=True, …)` | `Pointwise` with `ops.add` (or logical_or on bool) |
| `aten.mul` | custom `@register_lowering` using `make_pointwise` | `Pointwise` multiply |
| `aten.relu` | `relu = register_pointwise(aten.relu)` | `Pointwise` relu |
| `aten.sigmoid` / `aten.exp` / `aten.sin` / `aten.cos` | `register_pointwise_numeric*` | `Pointwise` numeric ops |
| `aten.sub` | `register_pointwise(aten.sub, allow_alpha=True)` | `Pointwise` sub |
| `aten.abs` / `maximum` / `minimum` / bitwise / logical | `register_pointwise(...)` | Same pattern |

Example mental lowerings:

```text
FX:  y = relu(x + 1)
IR:  Pointwise(add, x, 1) → ComputedBuffer
     Pointwise(relu, …)   → ComputedBuffer
     # still two IR ops; Scheduler may fuse into one Triton kernel later
```

Related: `make_reduction("sum"|"max"|…)` builds `Reduction` IR the same way for `sum`/`mean`/`amax`/… — also fuseable later, not fused inside the helper.

---

### 4.4 `make_fallback`: what it does

`make_fallback(op, …)` (`lowering.py` ~L2306) **registers an ATen op as “don’t lower to Loop IR — call ATen at runtime.”**

Mechanically:

1. For each overload of `op`, wrap `fallback_handler(op_overload)`.
2. `register_lowering(...)` so `GraphLowering.call_function` finds it.
3. At lower time, that handler builds a **`FallbackKernel` / `ExternKernel*`** IR node (inputs must often be realized).
4. Later, wrapper codegen emits something like `torch.ops.aten.<op>.default(...)` or `extern_kernels.*` — **no Triton body**.

Also:

- Prefer a **decomp** over a fallback when one exists (CI asserts if both).
- If an op is **not** registered and `implicit_fallbacks` is on, `call_function` can call `make_fallback` on the fly.
- Layout constraints (`require_dense`, `require_contiguous`, …) can be attached so inputs are copied into a form ATen expects.

Examples of intentional fallbacks (illustrative; list evolves):

- Some pool / upsample **backward** ops  
- Certain RNG / histogram / niche ops  
- Ops where eager / vendor kernel is preferred over a fragile Inductor body  

Contrast with SmolVLA upsample **forward**: many `upsample_nearest*` paths have **real** Inductor lowerings; only some variants / backwards use fallback.

---

### 4.5 Other mapping paths (don’t miss)

| Mechanism | File | When | IR / outcome |
|---|---|---|---|
| `make_reduction` | `lowering.py` | Reductions (`sum`, `prod`, `amax`, …) | `Reduction` → `ComputedBuffer` |
| View lowerings | `lowering.py` + `ir.py` | `expand`, `view`, `permute`, `slice`, … | `*View` nodes (often no kernel) |
| `select_algorithm` / templates | `select_algorithm.py` | `mm`, `bmm`, `convolution`, some attention | `TritonTemplateBuffer` / CUTLASS / **or** extern choice after autotune |
| Custom `@register_lowering` | `lowering.py` | Hand-written bodies (pools, upsample, …) | Mix of Loop IR, templates, or conditional fallback |
| Implicit fallback | `graph.py` `call_function` | Unknown `OpOverload` | Same as `make_fallback` at runtime |

Three-way summary (what §4 decides per FX node):

| Path | Helper | IR family | Later fate |
|---|---|---|---|
| Elementwise / reduction | `make_pointwise` / `make_reduction` | `Pointwise`/`Reduction` → `ComputedBuffer` | §5 fuse → §6 Triton |
| Template / GEMM | `select_algorithm` | `TritonTemplateBuffer` / … | Autotune → Triton or extern |
| Unsupported / library call | `make_fallback` | `FallbackKernel` / `ExternKernel*` | §7 ATen call in wrapper |

So the **ATen-extern vs Triton** split is chosen here at **lowering time**; §5 only fuses among IR that is already Loop/template-shaped.

---
## 5. Scheduler: fusion + which backend emits the node

| | |
|---|---|
| **Input** | Inductor IR operations / buffers from `GraphLowering.run` (§4) |
| **Output** | Dependency-aware **schedule** of `BaseSchedulerNode`s (fused / standalone / extern) ready for device codegen (§6) or ATen emission (§7) |

**Files:**

| Role | Path |
|---|---|
| Outer call site | `torch/_inductor/graph.py` (`GraphLowering.codegen` → `_update_scheduler`) |
| Driver | `torch/_inductor/scheduler.py` (`Scheduler`) |
| Node type system | `torch/_inductor/scheduler.py` (`SchedulerNode`, `FusedSchedulerNode`, …) |
| CUDA backend (later) | `torch/_inductor/codegen/triton.py` (`TritonScheduling`) |

Call site (outer driver still on `GraphLowering`, inner driver is `Scheduler`):

```python
# graph.py — GraphLowering.codegen
self.init_wrapper_code()
self._update_scheduler()          # Scheduler(self.operations)
self.scheduler.codegen()          # fuse already done in Scheduler.__init__; now emit
result = self.wrapper_code.generate(...)
```

### 5.0 Workflow overview

```text
                 GraphLowering.codegen  (outer)
                            │
                            ▼
              Scheduler(operations)     (DRIVER for §5)
                            │
          create_scheduler_node per IR op
                            │
          deps → topo sort → DCE → foreach grouping
                            │
                            ▼
                     fusion passes
              (can_fuse / score_fusion / fuse)
                            │
                            ▼
              schedule: FusedSchedulerNode | SchedulerNode | ExternKernelSchedulerNode | …
                            │
                            ▼
                   Scheduler.codegen / _codegen
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   is_template →     is_extern →        else → get_backend(device)
   codegen_template  codegen_extern_call   .codegen_node (§6 Triton on CUDA)
                            │
                            ▼
              lines accumulate in PythonWrapperCodegen (§7–8)
```

| Layer | Role |
|---|---|
| **`Scheduler`** | **Driver** — wrap IR ops as scheduler nodes, fuse, order, dispatch codegen |
| **`BaseSchedulerNode` hierarchy** | **Foundation** — schedule vocabulary consumed by backends / wrapper |
| **`fuse` / `can_fuse` / `score_fusion_*`** | **Fusion mapping** — which Loop IR nodes become one kernel |
| **`get_backend` / `codegen_extern_call`** | **Path split** — Triton (§6) vs ATen wrapper (§7) |

§5 does **not** emit Triton source or ATen call text itself for the happy path; it decides **groups** and **which emitter** runs.

---

### 5.1 Driver: `Scheduler`

| Method | What it does in §5 |
|---|---|
| `Scheduler(operations)` | Build scheduler nodes from IR ops; compute deps; fuse; produce final `self.nodes` |
| `create_scheduler_node` | Map one `ir.Operation` → `SchedulerNode` / `ExternKernelSchedulerNode` / `NopKernelSchedulerNode` |
| `compute_dependencies` / `topological_sort_schedule` | Fix legal execution order |
| `dead_node_elimination` | Drop unused nodes |
| `fuse` / `can_fuse` / `score_fusion_*` | Merge compatible nodes into `FusedSchedulerNode` |
| `codegen` / `_codegen` | Walk final schedule; dispatch template / extern / foreach / device backend |
| `get_backend(device)` | Lazy-create device scheduling (`TritonScheduling` on CUDA) |
| `codegen_extern_call` | Hand extern node to `ir.ExternKernel.codegen(wrapper)` (§7) |

Key behavior in `__init__` (~L2880+): fusion runs **before** `codegen`. By the time `_codegen` runs, the node list is already fused/reordered.

Key behavior in `_codegen` (~L6746+): for each node —

1. Device guard / flush if device or extern/template boundary.
2. `is_template()` → `backend.codegen_template(...)`.
3. `is_extern()` → `codegen_extern_call` (§7).
4. Else → `get_backend(device).codegen_node(...)` (§6 on CUDA).

**Mental model:** `Scheduler` is the loop over IR ops for **fusion + dispatch**; backends / wrapper do the actual text emission.

---

### 5.2 Foundation: key classes in `scheduler.py`

#### Schedule nodes

| Class | Key features | Role |
|---|---|---|
| `BaseSchedulerNode` | Deps, reads/writes, origins, estimated runtime | Common schedule interface |
| `SchedulerNode` | Wraps `ComputedBuffer` or `TemplateBuffer`; loop sizes / `LoopBody` | Fuseable native / template unit |
| `FusedSchedulerNode` | `snodes: list[...]`; unmet deps = union of children | One codegen unit for many IR ops → typically **one Triton kernel** |
| `ExternKernelSchedulerNode` | Wraps `ir.ExternKernel*`; `is_extern()=True` | Discrete ATen / library launch (§7) |
| `NopKernelSchedulerNode` | No-op IR | Skipped / structural placeholder |
| `GroupedSchedulerNode` | Keep children together for ordering | Group without necessarily fusing bodies |
| `ForeachKernelSchedulerNode` | List of similar nodes | Combo / foreach kernel codegen |

#### Buffer / dep helpers

| Class / concept | Key features | Role |
|---|---|---|
| `SchedulerBuffer` / `SchedulerDonatedBuffer` | Named buffer users, weak users | Track liveness / who reads a buf |
| `MemoryDep` / `StarDep` (deps) | Index / size / mode of a read or write | Fusion legality + memory score |
| `name_to_node` / `name_to_fused_node` / `name_to_buf` | Lookups after fusion | Resolve fused identity of a buffer |

**Mapping rule of thumb:** `ComputedBuffer` / `TemplateBuffer` → `SchedulerNode` (fuseable); `ExternKernel*` → `ExternKernelSchedulerNode` (usually not fused into Triton body).

---

### 5.3 Particular fusion mapping: `can_fuse` / `fuse` / `score_fusion_*`

**This is where fusion actually happens** (not in `make_pointwise`).

How it works (`scheduler.py`):

1. After topo sort, fusion passes try producer/consumer (vertical) and sibling (horizontal) pairs.
2. Legality is checked by several **`can_fuse`** helpers (see below); profitability by `score_fusion_memory` / `score_fusion_key`.
3. `Scheduler.fuse` → `FusedSchedulerNode.fuse` (or foreach / mix-order variants) builds a new fused node.

#### Several `can_fuse` helpers — who owns what

There is **one entry point** for a fusion round (`Scheduler.can_fuse`), plus **specialized** helpers for narrow cases. They do not all mean the same thing.

| Helper | Where | Kind | Scope / field | Role |
|---|---|---|---|---|
| **`Scheduler.can_fuse`** | `scheduler.py` ~L5412 | instance method on `Scheduler` | **General fusion gate** for almost every candidate pair | Top-level legality: same device, no cycles, not extern/nop (unless template), order/ancestors, prologue/epilogue template rules, vertical vs horizontal, then asks backend hooks. Also short-circuits into mix-order / grouped cases below. |
| **`MixOrderReduction.can_fuse`** | `scheduler.py` ~L269 | `@classmethod` | **Two reductions** that share reads but use **different reduce / loop orders** | Returns whether those two reductions may form a `FusedMixOrderReductions` (needs `config.triton.mix_order_reduction`, GPU+Triton, no producer/consumer link, mixed orders, common buffer access). Called from scoring / SIMD reduction mismatch paths — not the everyday pointwise path. |
| **`FusedMixOrderReductions.can_fuse_with`** | `scheduler.py` ~L2187 | instance method | **Already-built** mix-order fused node + another node | How to attach more nodes onto an existing mix-order pair (fuse into `node1` or `node2` side without illegally linking the two reductions). `Scheduler.can_fuse` delegates here when `node1` is already `FusedMixOrderReductions`. |
| **`ForeachKernelSchedulerNode.can_fuse`** | `scheduler.py` ~L2251 | `@classmethod` | At least one side is a **foreach / combo** group | Foreach-specific rules: zip two foreach lists of equal length; or fuse a single outside node into the matching subnode (`get_producer_subnode_for` / `get_consumer_subnode_for`); no reduction↔foreach today. Invoked from **`SIMDScheduling.can_fuse`** when either arg `is_foreach()`. |
| **`GroupedSchedulerNode.can_fuse`** | `scheduler.py` ~L2690 | `@classmethod` | **Grouped** schedule blocks (e.g. FSDP collective clusters) | Always **`False`**. Grouped nodes must not fuse with outsiders; they only keep children contiguous until unpack. `Scheduler.can_fuse` also rejects any pair involving a `GroupedSchedulerNode` early. |

Related (not separate “op families”, but part of the same gate):

| Helper | Where | Role |
|---|---|---|
| `SIMDScheduling.can_fuse` / `BaseScheduling.can_fuse_vertical` / `can_fuse_horizontal` | `codegen/simd.py`, `scheduler.py` backend API | **Device/backend** constraints after `Scheduler.can_fuse`’s graph-level checks (numel/rnumel match, Triton limits, foreach dispatch, …) |
| `Scheduler.fuse` / `FusedSchedulerNode.fuse` / `ForeachKernelSchedulerNode.fuse` | `scheduler.py` | **Commit** a legal pair into a fused / foreach fused node (runs after `can_fuse` + score / optional speedup check) |

**Call relationship (simplified):**

```text
fuse_nodes_once → get_possible_fusions
                      │
                      ▼
              Scheduler.can_fuse(node1, node2)          # general gate
                      │
        ┌─────────────┼──────────────────┬────────────────────┐
        ▼             ▼                  ▼                    ▼
 FusedMixOrder*.   Grouped* → False   template /          backend.can_fuse_*
 can_fuse_with                        extern / order /    (SIMDScheduling.can_fuse
                                      vertical rules       → may call Foreach*.can_fuse
                                                           or MixOrderReduction.can_fuse)
```

Examples (illustrative of what becomes one nsys kernel):

| Before fusion (IR / scheduler nodes) | After | Typical CUDA name |
|---|---|---|
| `Pointwise(add)` + `Pointwise(relu)` | One `FusedSchedulerNode` | `triton_poi_fused_...` |
| LayerNorm pieces: mean / var / affine pointwise+reduction | Fused poi+red group | `triton_poi_fused_native_layer_norm_2` (case study) |
| Many elementwise epilogue ops after a GEMM template | Template + epilogue fusion | Template kernel with epilogue inlined |
| `FallbackKernel(upsample)` next to pointwise | **Usually not fused** | Separate ATen launch + separate Triton |

```text
IR:   Pointwise(add) → ComputedBuffer A
      Pointwise(relu) → ComputedBuffer B (reads A)
Sched before: SchedulerNode(A), SchedulerNode(B)
Sched after:  FusedSchedulerNode([A, B])  → one Triton kernel in §6
```

Extern nodes generally **do not** fuse into Triton bodies (exception: limited multi-output / template epilogue cases).

---

### 5.4 Backend dispatch after fusion: how `Scheduler` drives codegen

Fusion (§5.3) only rewrites `self.nodes`. Codegen starts when `GraphLowering.codegen` calls **`Scheduler.codegen` → `_codegen(self.nodes)`**.

#### Workflow

```text
 GraphLowering.codegen
        │
        ▼
 Scheduler.codegen() / _codegen(nodes)          ← DRIVER (walk schedule in order)
        │
        for each schedule node (already fused):
            flush / device guard if needed
            │
            ├─ is_template() ────────────► backend.codegen_template(prologue, template, epilogue)
            │                                      └─ §6 template Triton / CUTLASS path
            │
            ├─ is_extern() ──────────────► Scheduler.codegen_extern_call
            │                                      └─ ir.ExternKernel.codegen(wrapper)  → §7
            │
            ├─ is_foreach() ─────────────► backend.codegen_combo_kernel(foreach_node)
            │                                      └─ one combo / foreach Triton kernel (§6)
            │
            ├─ FusedMixOrderReductions ──► backend.codegen_mix_order_reduction(...)
            │
            ├─ FusedSchedulerNode
            │   or SchedulerNode ────────► backend.codegen_node(node)   ← usual fused path (§6)
            │                                      │
            │                                      └─ TritonScheduling / SIMDScheduling
            │                                           builds TritonKernel, then for each
            │                                           child SchedulerNode calls
            │                                           child.codegen(index_vars)  (*)
            │
            └─ NopKernelSchedulerNode ───► mark_run() only
        │
        ▼
 lines accumulate in PythonWrapperCodegen  → finalized in §8
```

`get_backend(device)` → `create_backend` → `get_scheduling_for_device(device.type)`. On CUDA with Triton: **`TritonScheduling`** (`SIMDScheduling` subclass).

| Condition in `_codegen` | Call | Next |
|---|---|---|
| `node.is_template()` | `backend.codegen_template(...)` | §6 template |
| `node.is_extern()` | `codegen_extern_call` | §7 |
| `node.is_foreach()` | `backend.codegen_combo_kernel` | §6 combo |
| `FusedMixOrderReductions` | `backend.codegen_mix_order_reduction` | §6 special |
| `FusedSchedulerNode` / `SchedulerNode` | `backend.codegen_node` | §6 |
| `NopKernelSchedulerNode` | `mark_run` | none |

#### Relation: `Scheduler` vs `SchedulerNode.codegen` vs foreach

Two different meanings of “codegen” sit on different layers:

| API | Owner | What it does |
|---|---|---|
| **`Scheduler.codegen` / `_codegen`** | `Scheduler` | **Outer driver** — iterate fused schedule; pick backend / extern / foreach path |
| **`BaseScheduling.codegen_node` / `codegen_combo_kernel` / `codegen_template`** | Device backend (`TritonScheduling`, …) | **Kernel builder** — create `TritonKernel`, emit `@triton.jit` source, register in wrapper |
| **`SchedulerNode.codegen(index_vars)`** | Single loop node | **Inner body emit** — replay this node’s `LoopBody` into the *already open* kernel (loads / `ops.*` / stores). Called from the backend while generating one Triton kernel, not from `Scheduler._codegen` directly |
| **`ForeachKernelSchedulerNode.codegen`** | Foreach group | **`NotImplementedError` on purpose** — foreach is never “codegen’d as itself”; `Scheduler._codegen` routes to **`backend.codegen_combo_kernel(node)`**, which uses `get_subkernel_nodes()` / `snodes` and may call each sub-`SchedulerNode.codegen` inside the combo kernel |

```text
Scheduler._codegen                    # schedule-level driver
    │
    ├─ codegen_node(FusedSchedulerNode)
    │       backend opens one TritonKernel
    │       for each child SchedulerNode:
    │           child.codegen(index_vars)   # (*) body into that kernel
    │
    └─ codegen_combo_kernel(ForeachKernelSchedulerNode)
            backend opens one combo/foreach kernel
            for each subkernel in foreach.snodes:
                sub.codegen(...)            # same (*) idea, parallel siblings
```

So: **`SchedulerNode.codegen` is not an alternate top-level path** — it is the per-op body hook used *inside* backend kernel generation. **`ForeachKernelSchedulerNode` does not implement top-level codegen**; it is a grouping that `_codegen` sends to the combo-kernel backend API.

---

### 5.5 Other scheduler aspects (don’t miss)

| Mechanism | When | Outcome |
|---|---|---|
| Comm / collective ordering | Distributed graphs | Global order before fusion |
| `create_foreach_nodes` | `V.graph.lists` foreach patterns | `ForeachKernelSchedulerNode` |
| Graph partition (`config.graph_partition`) | Large / CUDA-graph friendly splits | `_codegen_partitions` + subgraph wrappers |
| Custom `_pre_fusion_custom_pass` | Config hook | User/reorder fusion input |
| Why-not-fuse / slow-fusion logging | Debug | Explains why two nodes stayed separate |

---

## 6. Triton kernel generation

| | |
|---|---|
| **Input** | Fused / standalone `SchedulerNode` (Loop / template / foreach) dispatched by `Scheduler._codegen` |
| **Output** | (1) Triton **kernel body** (`@triton.jit` source) + (2) **launch-parameter meta** (grid type, size hints, BLOCK / warps configs) wired into the wrapper; later compiled and launched |

§6 has two coupled jobs after fusion:

1. **Kernel codegen** — emit the Triton function body (loads / compute / stores).  
2. **Launch-parameter codegen / tune** — emit meta + heuristics so grid, `XBLOCK`/`RBLOCK`, `num_warps`, `num_stages`, … can be chosen and applied at compile/launch.

**Files:**

| Role | Path |
|---|---|
| Driver | `torch/_inductor/codegen/triton.py` (`TritonScheduling`) |
| Kernel IR → source + meta | `torch/_inductor/codegen/triton.py` (`TritonKernel`) |
| SIMD foundation | `torch/_inductor/codegen/simd.py` (`SIMDScheduling`, `SIMDKernel`) |
| Heuristics / autotune / grid eval | `torch/_inductor/runtime/triton_heuristics.py` |
| Async compile | `torch/_inductor/async_compile.py` (`AsyncCompile`) |
| Wrapper call stub | `torch/_inductor/codegen/wrapper.py` (`generate_kernel_call`) |
| Template choices | `torch/_inductor/select_algorithm.py` |

### 6.0 Workflow overview

```text
           Scheduler.get_backend(cuda).codegen_node(fused_node)
                            │
                            ▼
                 TritonScheduling  (DRIVER)
                            │
              build TritonKernel; walk fused schedule
              (each SchedulerNode.codegen(index_vars) fills body)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   (A) KERNEL CODEGEN              (B) LAUNCH-PARAM CODEGEN / TUNE
   TritonKernel.codegen_kernel     same codegen_kernel emits:
     @triton.jit body                triton_meta / inductor_meta
     imports, tl.load/store          grid_type, size_hints
                                     XBLOCK/RBLOCK constexprs
                                     heuristics decorator configs
              │                           │
              └─────────────┬─────────────┘
                            ▼
              TritonScheduling.define_kernel(src, ...)
                   → name kernel; optional async_compile.triton early
                   → TritonKernel.call_kernel → wrapper.generate_kernel_call
                            │
                            ▼
              at compile / first run (triton_heuristics):
                   pick num_warps / num_stages / BLOCKs (default or autotune)
                   GridExpr.from_meta(...).eval → concrete grid
                            │
                            ▼
              launch: kernel[grid](*args, num_warps=..., ...)
```

| Layer | Role |
|---|---|
| **`TritonScheduling`** | **Driver** — schedule node → named kernel + wrapper call |
| **`SIMDKernel` / `TritonKernel`** | **Foundation** — body emission + meta for launch |
| **(A) `codegen_kernel` body** | **Kernel codegen** — Loop schedule → `@triton.jit` text |
| **(B) meta + `triton_heuristics`** | **Launch-param codegen / tune** — grid / BLOCK / warps |
| **`AsyncCompile.triton`** | **Compile** — Triton source → GPU binary |
| **`call_kernel` / `generate_kernel_call`** | **Wrapper hook** — call site that will supply launch args |

---

### 6.1 Driver: `TritonScheduling`

Subclass of `SIMDScheduling` (`triton.py` ~L6036).

| Method | What it does in §6 |
|---|---|
| `codegen_node` (via `SIMDScheduling`) | Create kernel, drive node schedule body, then define/launch hooks |
| `codegen_template` | Template + optional prologue/epilogue fusion path |
| `define_kernel` | Dedup by `src_code`; name kernel; `_emit_kernel_to_wrapper` |
| `codegen_combo_kernel` | Foreach / combo kernels |
| `benchmark_fused_nodes` | Optional fusion / kernel timing |

`define_kernel` naming (~L6147+):

```text
triton_{category[:3]}_{fused_descriptive_name}_{suffix}
# e.g. triton_poi_fused_native_layer_norm_2
```

---

### 6.2 Foundation: SIMD / Triton kernel classes

| Class | Key features | Role |
|---|---|---|
| `SIMDScheduling` | Device-agnostic scheduling for SIMD-style backends | Base driver API (`codegen_node`, indexing groups) |
| `SIMDKernel` | Index ranges, loads/stores, CSE, masking | Intermediate “SIMD IR” while generating a kernel |
| `TritonKernel` | Triton overrides; builds `triton_meta` / `inductor_meta` | Body + launch meta toward Triton |
| `TritonKernelOverrides` / `TritonOverrides` | `ops.add` → `tl.*` mappings | Lower Inductor `ops.*` to Triton ops |
| `triton_heuristics` configs / `GridExpr` | Autotune configs; grid from meta | Choose / evaluate launch params |
| `AsyncCompile` | `triton()`, `wait()` | Out-of-process compile + barrier before first use |

---

### 6.3 Kernel codegen: body (`codegen_kernel` → `define_kernel`)

How a fused pointwise group becomes a Triton **function body**:

1. **`TritonScheduling`** constructs a **`TritonKernel`** and replays fused nodes’ loop bodies (`SchedulerNode.codegen(index_vars)` → index math, `tl.load` / compute / `tl.store`, reductions).
2. **`TritonKernel.codegen_kernel`** emits the Python module string: imports, `@triton.jit def ...`, kernel body.
3. **`TritonScheduling.define_kernel`** assigns the public name, may kick **`async_compile.triton(subs_name, src_code)`** early, and records the def in the wrapper.

Emitted shape (conceptual):

```python
async_compile.triton('triton_poi_fused_...', '''
@triton_heuristics.pointwise(size_hints=..., ...)
@triton.jit
def triton_poi_fused_...( ..., XBLOCK : tl.constexpr ):
    ...
''', device_str='cuda')
```

Examples:

| Schedule input | Category | Typical name |
|---|---|---|
| Fused elementwise only | `poi` | `triton_poi_fused_add_relu_...` |
| Pointwise + reduction (LN-like) | `poi` / `red` mix | `triton_poi_fused_native_layer_norm_2` |
| Reduction-heavy | `red` | `triton_red_fused_...` |
| Triton GEMM template win | template | Named template kernel (still Triton source) |

---

### 6.4 Launch-parameter codegen / tune

Fusion (§5) does **not** pick grid / warps. That happens here, in two phases: **emit meta with the kernel**, then **select / evaluate at compile or launch**.

#### (B1) Codegen-time: meta written beside the kernel

Inside the same `TritonKernel.codegen_kernel` that emits the body, Inductor also builds:

| Meta / artifact | Examples | Role |
|---|---|---|
| `inductor_meta` | `grid_type`, `size_hints`, `no_x_dim`, tiling hints, … | How to derive the launch grid and heuristic search space |
| `triton_meta` | signature, device props, `launch_cooperative_grid`, … | Triton compile / launch metadata |
| Constexpr launch knobs in signature | `XBLOCK`, `RBLOCK`, `RSPLIT`, … | Block sizes passed as `tl.constexpr` |
| Heuristics decorator | `@triton_heuristics.pointwise/reduction/...` | Declares candidate configs (`num_warps`, `num_stages`, BLOCK values) |

Then **`TritonKernel.call_kernel`** → **`wrapper.generate_kernel_call(..., triton_meta=..., inductor_meta=...)`** writes the wrapper call site that will eventually launch with those metas.

#### (B2) Tune / finalize: heuristics + runtime grid

| Step | Where | What gets fixed |
|---|---|---|
| Config selection | `triton_heuristics` (default heuristics or autotune) | Concrete `num_warps`, `num_stages`, `XBLOCK` / `RBLOCK`, … |
| Grid evaluation | `GridExpr.from_meta(inductor_meta, config).eval...` | Concrete `(grid_x, grid_y, …)` from numel + chosen BLOCKs |
| Launch | wrapper / heuristics launcher | `kernel[grid](*args, num_warps=..., num_stages=...)` |

```text
codegen_kernel          →  body + meta + heuristics decorator
async_compile / first call → pick config (tune or default)
GridExpr                →  grid from meta + config + runtime sizes
launch                  →  GPU grid + warps/stages
```

So launch params are **configured in §6 alongside kernel codegen**, refined by **heuristics/autotune**, and the **grid is computed at launch** — not during Scheduler fusion.

---

### 6.5 `AsyncCompile.triton`: what it does

`AsyncCompile.triton(kernel_name, source_code, device_str="cuda")` (`async_compile.py`):

1. Submits Triton source (body + heuristics wrapper) to a **process pool** (when enabled) so compile overlaps with further codegen.
2. Caches compiled kernel objects (`CompiledTritonKernels`) so a second identical `triton(...)` is cheap.
3. `wait(scope)` blocks until outstanding compiles finish and binds kernels into the wrapper module globals before launch.

**Not** the same as Inductor’s FX/`GraphLowering` compile — this is only **Triton → GPU binary** (configs from heuristics participate in that compile).

---

### 6.6 Other Triton / template paths (don’t miss)

| Mechanism | File | When | Outcome |
|---|---|---|---|
| `select_algorithm` + `ChoiceCaller` | `select_algorithm.py` | `mm` / `bmm` / conv / attention templates | Autotune among Triton template, CUTLASS, **or** `ExternKernelCaller` (also a launch/config choice) |
| `codegen_template` prologue/epilogue | `triton.py` / scheduler | Pointwise fused into template | One template kernel with fused edges |
| Combo / foreach | `triton_combo_kernel.py` | `ForeachKernelSchedulerNode` | Combo kernel body + its own partition/launch meta |
| CPU Triton / other devices | device scheduling registry | Non-CUDA | Same SIMD pattern, different backend class |
| Autotune-at-compile-time | wrapper + config | Tuning block embedded in output code | Extra defs/calls before main `call` |

If autotune picks **`ExternKernelCaller`**, that node leaves the Triton path and is emitted as §7 ATen/library call instead.

---

## 7. ATen external kernel emission

| | |
|---|---|
| **Input** | `ExternKernelSchedulerNode` whose IR is `ExternKernel*` / `FallbackKernel` (chosen at §4 lowering, optionally reinforced by §6 autotune losing to extern) |
| **Output** | Python wrapper lines that **call** existing ATen / library kernels — **no** generated Triton body |

**Files:**

| Role | Path |
|---|---|
| Dispatch from schedule | `torch/_inductor/scheduler.py` (`codegen_extern_call`) |
| IR ops | `torch/_inductor/ir.py` (`ExternKernel*`, `FallbackKernel`) |
| Driver / emitters | `torch/_inductor/codegen/wrapper.py` (`PythonWrapperCodegen.generate_*`) |
| Autotune extern namespace | `torch/_inductor/select_algorithm.py` (`extern_kernels`, `ExternKernelCaller`) |

### 7.0 Workflow overview

```text
           Scheduler._codegen sees node.is_extern()
                            │
                            ▼
              codegen_extern_call(scheduler_node)   (DRIVER entry)
                            │
              decide_inplace_update / mark_run / free_buffers
                            │
                            ▼
              ir_node.codegen(V.graph.wrapper_code)
                            │
                            ▼
         PythonWrapperCodegen methods  (FOUNDATION emitters)
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   generate_fallback_   generate_extern_   generate_extern_
   kernel               kernel_alloc       kernel_out
          │                 │                  │
          └────────┬────────┴────────┬─────────┘
                   ▼
         wrapper lines, e.g.
           buf0 = extern_kernels.mm(...)
           buf1 = torch.ops.aten.upsample_nearest2d.default(...)
```

| Layer | Role |
|---|---|
| **`codegen_extern_call`** | **Driver entry** — bookkeeping then IR `codegen` |
| **`ExternKernel*` IR** | **Foundation op** — carries kernel name, args, layout, origins |
| **`PythonWrapperCodegen.generate_*`** | **Emission mapping** — IR → concrete Python call lines |
| **`extern_kernels` / `torch.ops.aten`** | **Callee** — real GPU work happens inside ATen / cuBLAS / … |

---

### 7.1 Driver: `codegen_extern_call` → IR `codegen`

| Step | Where | What |
|---|---|---|
| Detect extern | `Scheduler._codegen` | `node.is_extern()` |
| `codegen_extern_call` | `scheduler.py` ~L5876 | Inplace decisions, `mark_run`, then `node.codegen(wrapper)` |
| `FallbackKernel.codegen` / `ExternKernel*.codegen` | `ir.py` | Call the matching `wrapper.generate_*` helper |
| Wrapper writeline | `wrapper.py` | Append `ExternKernelAllocLine` / `ExternKernelOutLine` / custom |

**Mental model:** Inductor **schedules and allocates**, but **does not generate** the GPU kernel body; the ATen dispatcher / cuDNN / cuBLAS / custom library does.

---

### 7.2 Foundation: extern IR → wrapper emitters

| IR class (`ir.py`) | Wrapper method (`wrapper.py`) | Key features | Role |
|---|---|---|---|
| `FallbackKernel` | `generate_fallback_kernel` | Generic unsupported / hard op | Emit `torch.ops.aten.*.default(...)` (or custom codegen hook) |
| `ExternKernelAlloc` | `generate_extern_kernel_alloc` | Allocates output tensor | `out = kernel(args)` |
| `ExternKernelOut` | `generate_extern_kernel_out` | Writes into provided `out=` | `kernel(..., out=buf)` |
| Specialized fallbacks | `generate_scatter_fallback` / `generate_index_put_fallback` / … | Odd calling conventions | Keep wrapper correct without Triton |
| `MultiOutput` | multi-output unpack paths | Tuple returns | Split extern results for consumers |

Also:

| Symbol | File | Role |
|---|---|---|
| `extern_kernels` (`KernelNamespace`) | `select_algorithm.py` | Attributes like `extern_kernels.mm` used in wrapper imports / calls |
| `ExternKernelCaller` | `select_algorithm.py` | Autotune choice that means “call extern, don’t use Triton template” |
| `CUSTOM_EXTERN_KERNEL_CODEGEN` | `wrapper.py` | Escape hatch for hand-written extern emit |

---

### 7.3 Particular emission examples

Wrapper preamble typically includes:

```python
from torch._inductor.select_algorithm import extern_kernels
async_compile = AsyncCompile()
```

Emitted calls:

| Situation | Example wrapper line |
|---|---|
| Autotune / explicit extern GEMM | `buf0 = extern_kernels.mm(arg0_1, arg1_1)` |
| §4 `make_fallback` / `FallbackKernel` | `buf1 = torch.ops.aten.upsample_nearest2d.default(...)` |
| Out-variant extern | `extern_kernels.something(..., out=buf2)` |
| SmolVLA-style upsample that **did** lower natively | **Not** this path — would be Triton §6 instead |

Contrast: an op is on this path because §4 built `ExternKernel*`/`FallbackKernel`, **or** because §6 template autotune selected `ExternKernelCaller`.

---

### 7.4 Other extern aspects (don’t miss)

| Mechanism | Role |
|---|---|
| Layout constraints from `make_fallback` | May insert copies so ATen sees contiguous / dense inputs |
| `codegen_comment` / provenance | Keep FX origins in wrapper for debugging |
| Inplace update decisions | Reuse buffers across extern calls when safe |
| Cpp wrapper / AOTI | Parallel emitters for non-Python wrappers (same IR idea) |

---

## 8. Wrapper assemble → executable module

| | |
|---|---|
| **Input** | Accumulated wrapper buffer after §5–7: Triton defs, extern calls, allocs, launch order |
| **Output** | `output_code.py` text → `PyCodeCache` → importable module with `.call`; returned through AOT / Dynamo as the compiled subgraph callable |

**Files:**

| Role | Path |
|---|---|
| Outer driver | `torch/_inductor/graph.py` (`GraphLowering.codegen`, `compile_to_module`) |
| Wrapper foundation | `torch/_inductor/codegen/wrapper.py` (`PythonWrapperCodegen`) |
| Code cache | `torch/_inductor/codecache.py` (`PyCodeCache`) |
| Return type glue | `torch/_inductor/output_code.py`, `compile_fx.py` |

### 8.0 Workflow overview

```text
                 GraphLowering.codegen / compile_to_module  (DRIVER)
                            │
              init_wrapper_code() → PythonWrapperCodegen
                            │
              Scheduler.codegen()     # fills wrapper with §6 + §7 lines
                            │
                            ▼
              PythonWrapperCodegen.generate(is_inference)
                   → finalize header, kernel defs, call body, returns
                   → ValueWithLineMap (full output_code.py text)
                            │
                            ▼
              PyCodeCache.write(code) → path on disk
              import / load CompiledModule
                            │
                            ▼
              module.call(*args)  ← used on every subsequent run
```

| Layer | Role |
|---|---|
| **`GraphLowering.codegen` / `compile_to_module`** | **Driver** — orchestrate wrapper init → schedule codegen → cache module |
| **`PythonWrapperCodegen`** | **Foundation** — buffers lines for imports, kernels, allocs, launches, returns |
| **`generate` / `PyCodeCache.write`** | **Materialize** — string → cached Python module with `.call` |

---

### 8.1 Driver: `GraphLowering.codegen` → `compile_to_module`

| Method | What it does in §8 |
|---|---|
| `init_wrapper_code` | Construct `PythonWrapperCodegen` (or C++ / subgraph variant) |
| `codegen` | `_update_scheduler` → `scheduler.codegen` → `wrapper_code.generate` |
| `compile_to_module` / `_compile_to_module` | Take generated text, write via `PyCodeCache`, return `CompiledModule` |
| `_compile_to_module_lines` | Optional autotune-at-compile-time preamble; `PyCodeCache.write`; load module |

`GraphLowering` is again the outer driver — same class as §4 — but here it runs **after** IR exists and drives **code emission + module load**, not FX→IR.

---

### 8.2 Foundation: wrapper + cache classes

| Class / API | Key features | Role |
|---|---|---|
| `PythonWrapperCodegen` | `header`, `kernel` defs, `lines`, `wrapper_call`, imports | Owns all generated Python for one graph |
| `WrapperLine` / `ExternKernelAllocLine` / … | Deferred line objects | Structured emit; finalized in `generate` |
| `SubgraphPythonWrapperCodegen` | Partition / subgraph variant | Nested graphs / partitions |
| `PyCodeCache` | Hash → path; write + import | Durable cache of `output_code.py` modules |
| `CompiledModule` / `OutputCode` | `.call`, `__file__` | Object AOT/Dynamo invokes at runtime |
| `AsyncCompile` instance in wrapper | `wait(globals())` | Sync Triton compiles before launches |

---

### 8.3 Particular materialization: `generate` → `PyCodeCache` → `.call`

How the executable appears:

1. During §5–7, emitters **append** to the wrapper (kernel defs, `extern_kernels.*`, `empty_strided`, launches).
2. **`PythonWrapperCodegen.generate`** finalizes: run wrapper IR passes, render `WrapperLine`s into the `call` body, splice multi-kernel defs, emit returns / sync hooks → full file text + line map.
3. **`PyCodeCache.write`** stores the text; Inductor imports it as a module.
4. Return **`CompiledModule`** whose entry is typically **`.call`**.

Debug: `TORCH_LOGS=output_code` or Inductor dumps under paths like `model__N_inference_.../output_code.py`.

Runtime path after compile:

```text
user call → Dynamo cached bytecode → AOT boxed call → inductor module.call
              → triton kernels + aten extern launches on CUDA stream
```

---

### 8.4 Other assemble aspects (don’t miss)

| Mechanism | Role |
|---|---|
| `cpp_wrapper` / AOTI | Alternate `codegen_with_cpp_wrapper` path; same Scheduler IR, different wrapper language |
| Graph partition subgraphs | Nested `generate` + launcher fns inside parent wrapper |
| Autotune-at-compile-time block | Extra tuning code prepended in `_compile_to_module_lines` |
| Line maps / provenance | Map wrapper lines back to FX nodes for profiling / errors |
| Constant folding / memory planning passes inside `generate` | Last wrapper-level opts before string freeze |

After §8, Inductor’s job for this subgraph is done: Dynamo/AOT just **calls** the cached module.

---

## 9. Stage checklist (functions to set breakpoints)

Useful order when stepping a local editable build (`/root/workspace/pytorch`):

| # | Break / inspect | Source |
|---|---|---|
| 1 | `torch.compile` → `OptimizeContext` / `set_eval_frame` | `torch/__init__.py`, `torch/_dynamo/eval_frame.py` |
| 2 | `trace_frame` → `InstructionTranslator.run` | `torch/_dynamo/convert_frame.py`, `torch/_dynamo/symbolic_convert.py` |
| 3 | `OutputGraph.compile_subgraph` → `call_user_compiler` | `torch/_dynamo/output_graph.py` |
| 4 | `compile_fx` → `_compile_fx_main` → `aot_module_simplified` | `torch/_inductor/compile_fx.py`, `torch/_functorch/aot_autograd.py` |
| 5 | `compile_fx_inner` → `_recursive_post_grad_passes` (FX fusion) then `GraphLowering` | `torch/_inductor/compile_fx.py`, `fx_passes/post_grad.py`, `graph.py` |
| 6 | `GraphLowering.run` / `call_function` → lowering handlers | `torch/_inductor/graph.py`, `torch/_inductor/lowering.py` |
| 7 | IR: `ComputedBuffer` vs `FallbackKernel` / `ExternKernelOut` | `torch/_inductor/ir.py` |
| 8 | `Scheduler` fusion → `TritonScheduling.define_kernel` vs `generate_extern_*` | `torch/_inductor/scheduler.py`, `torch/_inductor/codegen/triton.py`, `torch/_inductor/codegen/wrapper.py` |
| 9 | `TritonKernel.codegen_kernel` → `AsyncCompile.triton` | `torch/_inductor/codegen/triton.py`, `torch/_inductor/async_compile.py` |
| 10 | `GraphLowering.compile_to_module` → `PyCodeCache.write` → `.call` | `torch/_inductor/graph.py`, `torch/_inductor/codecache.py` |

---

## 10. How this maps to SmolVLA GPU compile work

| Observation in our docs / nsys | Pipeline stage |
|---|---|
| Multiple FX graphs / `nonzero` breaks | Dynamo graph break (`convert_frame`) |
| Long aten op lists in Dynamo MD | Post-Dynamo / pre-Inductor FX nodes |
| `triton_poi_fused_*` kernels | Fused `ComputedBuffer` → `TritonKernel` |
| `triton_poi_fused_native_layer_norm_2` case study | Pointwise+reduction fusion + Triton codegen |
| Upsample / some pool ops as discrete ATen | `FallbackKernel` / extern (little or no Triton body) |
| `output_code.py` under inductor debug | `PythonWrapperCodegen.generate` + `PyCodeCache` |
| `extern_kernels.mm` / cuBLAS-like launches | Autotune chose `ExternKernelCaller` or template extern |

---

## 11. Quick reference — primary source files

| Area | Key symbols | Path |
|---|---|---|
| Public compile API | `torch.compile`, `_TorchCompileInductorWrapper` | `torch/__init__.py` |
| Inductor mode / options lists | `list_mode_options`, `list_options` | `torch/_inductor/__init__.py` |
| Eval frame / optimize | `optimize`, `_optimize`, `OptimizeContext` | `torch/_dynamo/eval_frame.py` |
| Eval-frame C API | `set_eval_frame` | `torch._C._dynamo.eval_frame` |
| Backend registry | `lookup_backend`, `register_backend` | `torch/_dynamo/backends/registry.py` |
| Bytecode → FX | `convert_frame`, `trace_frame` | `torch/_dynamo/convert_frame.py` |
| Symbolic convert | `InstructionTranslator` | `torch/_dynamo/symbolic_convert.py` |
| FX ownership / backend call | `OutputGraph`, `compile_subgraph`, `call_user_compiler` | `torch/_dynamo/output_graph.py` |
| Inductor backend + AOT glue | `compile_fx`, `compile_fx_inner`, `_compile_fx_main` | `torch/_inductor/compile_fx.py` |
| Decompositions | `select_decomp_table` | `torch/_inductor/decomposition.py` |
| AOT Autograd | `aot_module_simplified`, `create_aot_state` | `torch/_functorch/aot_autograd.py` |
| FX → IR | `GraphLowering` | `torch/_inductor/graph.py` |
| Lowering registry | `register_lowering`, `make_pointwise`, `make_fallback` | `torch/_inductor/lowering.py` |
| IR node types | `ComputedBuffer`, `ExternKernel*`, `FallbackKernel` | `torch/_inductor/ir.py` |
| Fusion / schedule | `Scheduler`, `SchedulerNode`, `ExternKernelSchedulerNode` | `torch/_inductor/scheduler.py` |
| Triton emit | `TritonScheduling`, `TritonKernel` | `torch/_inductor/codegen/triton.py` |
| SIMD base | `SIMDKernel` | `torch/_inductor/codegen/simd.py` |
| Wrapper / extern emit | `PythonWrapperCodegen`, `generate_extern_*` | `torch/_inductor/codegen/wrapper.py` |
| Autotune + `extern_kernels` | `ChoiceCaller`, `ExternKernelCaller`, `extern_kernels` | `torch/_inductor/select_algorithm.py` |
| Async Triton compile | `AsyncCompile.triton`, `wait` | `torch/_inductor/async_compile.py` |
| Code cache | `PyCodeCache` | `torch/_inductor/codecache.py` |
| SmolVLA compile wiring (site-packages) | `torch.compile(self.sample_actions, …)` | `lerobot/policies/smolvla/modeling_smolvla.py` |

# tensor-optix — Living Implementation Plan

> This document is the single source of truth for building tensor-optix.
> Update it as decisions are made, issues are found, and tasks complete.

---

## Project Identity

- **Package name:** `tensor-optix`
- **Import name:** `tensor_optix`
- **Root directory:** `d:\development\AugData\tensor-optix\`
- **Python:** `>=3.11`
- **Framework:** TensorFlow `>=2.18.0` (TF only, no framework abstraction)
- **Environment API:** Gymnasium `>=1.0.0` (modern API: `terminated | truncated`, not `done`)

---

## What This Is

A PyPI-distributable Python library that replaces the conventional RL training loop with an autonomous, continuously-learning optimization system. The user builds their TF model and Gymnasium environment. The library owns the training loop, evaluation, hyperparameter tuning, checkpointing, and adaptation lifecycle.

**Core philosophy:** We own the loop. The user owns the model.

---

## Architecture Summary

```
RLOptimizer (main entry point)
    └── LoopController (state machine + loop orchestration)
            ├── BaseAgent          ← user implements this
            ├── BaseEvaluator      ← user implements or use TFEvaluator default
            ├── BaseOptimizer      ← BackoffOptimizer or PBTOptimizer
            ├── BasePipeline       ← BatchPipeline or LivePipeline
            ├── CheckpointRegistry ← snapshot storage
            └── BackoffScheduler   ← interval + state management
```

### Loop States
| State | Behavior |
|-------|----------|
| ACTIVE | Aggressive tuning, eval every episode |
| COOLING | Recent improvement, exponential backoff |
| DORMANT | Plateau, minimal intervention |
| WATCHDOG | Monitoring for degradation |

---

## Repository Structure

```
tensor-optix/
├── PLAN.md                            ← this file
├── pyproject.toml
├── README.md
├── LICENSE
│
├── tensor_optix/
│   ├── __init__.py                    # Public API surface
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                   # EpisodeData, EvalMetrics, HyperparamSet, PolicySnapshot, LoopState
│   │   ├── base_agent.py              # Abstract BaseAgent
│   │   ├── base_evaluator.py          # Abstract BaseEvaluator
│   │   ├── base_optimizer.py          # Abstract BaseOptimizer
│   │   ├── base_pipeline.py           # Abstract BasePipeline + EpisodeBoundaryFn
│   │   ├── loop_controller.py         # LoopController + LoopCallback
│   │   ├── checkpoint_registry.py     # CheckpointRegistry
│   │   └── backoff_scheduler.py       # BackoffScheduler
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── tensorflow/
│   │       ├── __init__.py
│   │       ├── tf_agent.py            # TFAgent(BaseAgent)
│   │       └── tf_evaluator.py        # TFEvaluator(BaseEvaluator)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── batch_pipeline.py          # BatchPipeline — Gymnasium env, static/episodic
│   │   └── live_pipeline.py           # LivePipeline — real-time streaming source
│   │
│   └── optimizers/
│       ├── __init__.py
│       ├── backoff_optimizer.py       # BackoffOptimizer (default, perturbation-based)
│       └── pbt_optimizer.py           # PBTOptimizer (pseudo population-based training)
│
└── tests/
    ├── conftest.py
    ├── test_core/
    │   ├── test_types.py
    │   ├── test_backoff_scheduler.py
    │   ├── test_checkpoint_registry.py
    │   └── test_loop_controller.py
    ├── test_adapters/
    │   ├── test_tf_agent.py
    │   └── test_tf_evaluator.py
    ├── test_pipeline/
    │   ├── test_batch_pipeline.py
    │   └── test_live_pipeline.py
    ├── test_optimizers/
    │   ├── test_backoff_optimizer.py
    │   └── test_pbt_optimizer.py
    └── test_integration/
        └── test_end_to_end.py
```

---

## Critical Rules (never violate)

1. **Gymnasium API only.** `env.reset()` → `(obs, info)`. `env.step()` → `(obs, reward, terminated, truncated, info)`. Never use legacy `done` flag internally — merge `terminated | truncated` at the pipeline boundary.
2. **`BaseAgent` is the only contract.** `LoopController` calls only: `act()`, `learn()`, `get_hyperparams()`, `set_hyperparams()`, `save_weights()`, `load_weights()`.
3. **`HyperparamSet.params` is an open dict.** Core never reads specific key names. Opaque blob passed between optimizer and agent.
4. **`EpisodeData` carries raw interaction data only.** No algorithm-specific fields.
5. **No algorithm-specific code in `core/` or `loop_controller.py`.** PPO, DQN, SAC, etc. are never referenced there.
6. **`LoopController` is algorithm-blind.** run episode → get score → compare → tune → repeat.

---

## Implementation Tasks

### Phase 1 — Core Foundation
- [ ] `pyproject.toml`
- [ ] `tensor_optix/core/types.py`
- [ ] `tensor_optix/core/base_agent.py`
- [ ] `tensor_optix/core/base_evaluator.py`
- [ ] `tensor_optix/core/base_optimizer.py`
- [ ] `tensor_optix/core/base_pipeline.py`
- [ ] `tensor_optix/core/backoff_scheduler.py`
- [ ] `tensor_optix/core/checkpoint_registry.py`
- [ ] `tensor_optix/core/loop_controller.py`

### Phase 2 — TensorFlow Adapter
- [ ] `tensor_optix/adapters/tensorflow/tf_agent.py`
- [ ] `tensor_optix/adapters/tensorflow/tf_evaluator.py`

### Phase 3 — Pipelines
- [ ] `tensor_optix/pipeline/batch_pipeline.py`
- [ ] `tensor_optix/pipeline/live_pipeline.py`

### Phase 4 — Optimizers
- [ ] `tensor_optix/optimizers/backoff_optimizer.py`
- [ ] `tensor_optix/optimizers/pbt_optimizer.py`

### Phase 5 — Wiring
- [ ] `tensor_optix/optimizer.py` (RLOptimizer entry point)
- [ ] `tensor_optix/__init__.py` (public API surface)
- [ ] All `core/__init__.py`, `adapters/__init__.py`, `pipeline/__init__.py`, `optimizers/__init__.py`

### Phase 6 — Tests
- [ ] `tests/conftest.py`
- [ ] `tests/test_core/test_types.py`
- [ ] `tests/test_core/test_backoff_scheduler.py`
- [ ] `tests/test_core/test_checkpoint_registry.py`
- [ ] `tests/test_core/test_loop_controller.py`
- [ ] `tests/test_adapters/test_tf_agent.py`
- [ ] `tests/test_adapters/test_tf_evaluator.py`
- [ ] `tests/test_pipeline/test_batch_pipeline.py`
- [ ] `tests/test_pipeline/test_live_pipeline.py`
- [ ] `tests/test_optimizers/test_backoff_optimizer.py`
- [ ] `tests/test_optimizers/test_pbt_optimizer.py`
- [ ] `tests/test_integration/test_end_to_end.py`

---

## Known Issues / Decisions Log

| Date | Issue | Decision |
|------|-------|----------|
| 2026-03-27 | Blueprint said "framework-agnostic" | Corrected: TensorFlow only |
| 2026-03-27 | Blueprint used legacy gym API | Corrected: Gymnasium >=1.0.0 |
| 2026-03-27 | Blueprint hardcoded TF as required dep in a "framework-agnostic" core | N/A — TF-only removes the contradiction |
| 2026-03-27 | Degradation check `score < best * threshold` breaks for negative scores | Fixed: use `score < best - abs(best) * (1 - threshold)` |

---

## Notes

- `BatchPipeline` wraps a Gymnasium-compatible env for episodic/batch training. Not a static dataset loader.
- `LivePipeline` wraps a streaming data source (e.g. websocket feed). User provides a `stream()` generator.
- `TFAgent.learn()` provides a generic gradient update baseline. Users subclass and override for specific algorithms (PPO clipping, SAC entropy tuning, etc.).
- `PBTOptimizer` approximates population-based training for single-agent use via a virtual population from history.

---

## Optimizer Math — BackoffOptimizer (Running Finite Difference)

### Core Idea
Estimate the gradient of `primary_score` w.r.t. each hyperparam using finite differences accumulated across episodes. Step in the direction that increases score.

### Per-param gradient estimate
```
∂score/∂θᵢ ≈ (score_avg_after - score_avg_before) / Δθᵢ
```
Where `score_avg` is a rolling mean over the last N episodes (noise reduction).

### Update rule
```
θᵢ_new = clip(θᵢ + α * ∂score/∂θᵢ, low_bound, high_bound)
```

### Step size α (adaptive)
```
α = base_lr / (1 + β * score_variance)
```
High variance in recent scores → smaller steps. Low variance → larger steps.

### Perturbation size δ (per param)
- Multiplicative: `δᵢ = perturbation_scale * |θᵢ|` (scale-invariant)
- Clamped: `δᵢ = max(δᵢ, min_delta)` to avoid zero delta on small params

### Directional memory
- Track last direction moved per param (`+1` or `-1`)
- Track whether that move improved score
- If improvement: continue in same direction (momentum)
- If no improvement: reverse direction, halve step size

### Score buffer
- Rolling window of last `score_window` (default: 5) primary scores
- Use mean of buffer as the stable score signal for gradient estimation
- Do not update params until buffer has at least `min_samples` entries

### Bounds enforcement
- User provides `param_bounds: dict[str, tuple[float, float]]`
- Params not in bounds are left unchanged
- All updates clipped to `[low, high]` after step

### Variance-gated updates
- If `score_variance > high_variance_threshold`: skip update this cycle (too noisy to trust)
- Log skipped updates for observability

---

## Optimizer Math — PBTOptimizer (Pseudo Population-Based Training)

### Core Idea
Maintain a history of `(HyperparamSet, primary_score)` pairs as a virtual population. Use exploit/explore logic from PBT without parallel workers.

### Exploit condition
```
if current_score < percentile(history_scores, 20):
    # bottom 20% — exploit top 20%
    best_params = params from top 20% of history (by score)
    new_params = perturb(best_params, scale=small)
```

### Explore condition
```
else:
    # not bottom 20% — explore
    new_params = perturb(current_params, scale=medium)
```

### Perturbation function (shared with BackoffOptimizer)
```
perturb(θ, scale) → for each param:
    δ = scale * (high - low)          # fraction of param range
    new_val = θ + uniform(-δ, +δ)
    new_val = clip(new_val, low, high)
```

### History management
- Keep last `history_size` (default: 50) `(params, score)` pairs
- FIFO eviction
- Percentile computed over this window only

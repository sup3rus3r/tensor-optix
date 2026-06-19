# Distributed training (IMPALA + V-trace)

`AsyncActorLearner` implements IMPALA-style asynchronous actor-learner training. N actor subprocesses read weights from shared memory (lock-free) and collect trajectories; the learner dequeues trajectories, applies V-trace importance-sampling correction, and writes updated weights back to shared memory - actors see the new weights immediately, with no explicit broadcast step.

```python
from tensor_optix.distributed import AsyncActorLearner

learner = AsyncActorLearner(
    actor=actor,
    critic=critic,
    optimizer=optimizer,
    env_factory=lambda: gym.make("ALE/Pong-v5"),
    n_actors=8,
    trajectory_len=64,
)
stats = learner.run(max_steps=10_000_000)
# stats["steps_per_second"] -> ~4x single-process throughput on CPU
```

## Why V-trace

Actors run a policy μ that may lag behind the learner's current policy θ (staleness from async collection). The importance ratio `ρ_t = π_θ(a_t|s_t) / π_μ(a_t|s_t)` corrects for this mismatch, with two clips to control variance:

- `ρ̄_t = min(ρ̄, ρ_t)` - clips the IS weight applied to the TD error
- `c̄_t = min(c̄, ρ_t)` - clips the IS weight applied to the trace (controls the bias/variance tradeoff)

With `ρ̄ = c̄ = 1` and synchronous actors (`μ = θ`), V-trace reduces to standard on-policy GAE with `λ = c̄`. See Espeholt et al. 2018 ("IMPALA").

## Platform notes

`fork` is used automatically on Linux (no pickling required - subprocess objects are inherited via `os.fork()`); `spawn` is used on Windows/macOS, which requires `env_factory` to be picklable (a module-level function or `functools.partial`, not a lambda capturing a non-picklable closure).

## When to reach for this

This is for environments where single-process throughput is the bottleneck and your policy/critic are plain `nn.Module`s with a discrete action space - Atari-scale problems, not small classic-control envs where `VectorBatchPipeline` (synchronous, in `RLOptimizer`) is simpler and sufficient. `AsyncActorLearner` is a standalone training loop, not wired into `RLOptimizer`'s convergence/checkpoint/SPSA machinery - use it when you specifically need the throughput and are comfortable managing your own stopping condition via `max_steps`.

## Reference

Full constructor signature and `compute_vtrace_targets` math: [Distributed reference](../reference/distributed.md).

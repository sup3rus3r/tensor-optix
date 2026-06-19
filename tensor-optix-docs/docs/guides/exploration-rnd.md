# Add exploration bonuses (RND)

Random Network Distillation (RND) adds an intrinsic exploration reward to any pipeline without modifying the agent. Two small networks - a frozen random target and a trained predictor - both map observations to a fixed embedding space. Novel states produce high prediction error (high intrinsic reward); visited states are fitted well (low bonus).

```
r_int(s) = ||f_θ(s) - g(s)||²       (g frozen, f_θ trained)
r_total  = r_ext + η · r_int / σ(r_int)   (intrinsic normalized per episode)
```

## Usage

```python
from tensor_optix.exploration.rnd import RNDPipeline
from tensor_optix import RLOptimizer, BatchPipeline
import gymnasium as gym

base = BatchPipeline(env=gym.make("LunarLander-v2"), agent=agent, window_size=2048)
pipeline = RNDPipeline(base, obs_dim=8, embedding_dim=64, eta=0.1)

optimizer = RLOptimizer(agent=agent, pipeline=pipeline)
optimizer.run()
```

`RNDPipeline` wraps any `BasePipeline`. It intercepts each `EpisodeData` after collection and injects the intrinsic bonus into `episode_data.rewards` before the agent sees them, then trains the predictor network on the current batch's observations via plain SGD. Implementation is pure numpy (a minimal two-layer MLP) - no TF or Torch dependency, and `RNDPipeline` proxies any other attribute access through to the wrapped pipeline (`__getattr__`), so it's a drop-in wrapper.

## Loop-controlled exploration schedule

`η` (the intrinsic reward scale) is **not** static - `LoopController` calls `pipeline.set_eta(...)` automatically at state transitions, with zero cost for pipelines that don't implement `set_eta`:

| Event | Effect on η |
|---|---|
| improvement | `η *= 0.9` - pull back exploration while genuinely learning |
| plateau (`COOLING`) | `η *= 1.5`, capped at `4 × η_base` - push exploration harder when stuck |
| `DORMANT` | `η = 0` - converged, stop injecting noise |
| restart (after spawn) | `η = η_base` - reset for the new policy variant |

This means you set the initial `eta` once and the loop adapts it automatically as training progresses - no manual exploration-decay schedule needed.

## Reference

Constructor parameters and the `_MLP` internals: [Exploration reference](../reference/exploration.md).

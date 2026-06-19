# tensor_optix.exploration.rnd

```
Random Network Distillation (RND) exploration bonus.

RND adds an intrinsic reward signal to any pipeline without modifying any agent.
Two small networks - a frozen random target and a trained predictor - both map
observations to a fixed embedding space. Novel states produce high prediction
error (high intrinsic reward). Visited states are fitted well (low bonus).

Math:
    r_int(s) = ||f_θ(s) - g(s)||²       (g frozen, f_θ trained)
    r_total  = r_ext + η · r_int / σ(r_int)   (intrinsic normalized per episode)

η is controlled by the loop via set_eta():
    ACTIVE  → η = η_base        (default exploration level)
    COOLING → η *= 1.5          (stuck - push exploration harder)
    DORMANT → η = 0             (converged - stop injecting noise)
    improvement → η *= 0.9      (getting better - reduce exploration)

RNDPipeline wraps any BasePipeline. It intercepts EpisodeData after each episode
and injects the intrinsic bonus into episode_data.rewards before the agent sees them.
The predictor network is trained on the current batch each episode.

Framework: pure numpy + a minimal two-layer MLP. No TF or Torch dependency.
The target network is random and fixed; the predictor is updated via SGD.
```

## RNDPipeline

```python
class RNDPipeline(BasePipeline):
    """
    Wraps any BasePipeline and injects RND intrinsic rewards into each episode.

    After each episode, before returning EpisodeData:
    1. Compute r_int(s) = ||predictor(s) - target(s)||² for each step
    2. Normalize r_int by its running std
    3. Inject: rewards[t] += eta * r_int[t]
    4. Train predictor on current batch observations

    The loop controls eta via set_eta() at state transitions.

    Args:
        pipeline:      The wrapped pipeline (any BasePipeline).
        obs_dim:       Observation dimension (flattened).
        embedding_dim: RND embedding size. Default 64 - larger = slower but richer.
        eta:           Initial intrinsic reward scale. Default 0.1.
        predictor_lr:  SGD learning rate for the predictor. Default 1e-3.
        norm_eps:      Small constant for std normalization. Default 1e-8.
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        obs_dim: int,
        embedding_dim: int = 64,
        eta: float = 0.1,
        predictor_lr: float = 1e-3,
        norm_eps: float = 1e-8,
    ): ...

    def set_eta(self, eta: float) -> None:
        """Called by LoopController at state transitions to adjust exploration scale."""

    # BasePipeline interface: setup, teardown, episodes, is_live - all delegate
    # to the wrapped pipeline, with episodes() injecting the intrinsic bonus.

    # Any other attribute access (e.g. set_agent, n_steps) proxies through
    # to the wrapped pipeline via __getattr__.
```

Running stats for intrinsic-reward normalization use Welford's online algorithm (mean and variance updated incrementally, one `r_int` value at a time).

See [Add exploration bonuses (RND)](../guides/exploration-rnd.md) for usage with `RLOptimizer`.

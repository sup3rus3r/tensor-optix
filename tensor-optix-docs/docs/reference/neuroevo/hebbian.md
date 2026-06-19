# HebbianHook

```
HebbianHook - local Hebbian learning running alongside PPO gradient updates.

Rule (Oja-style with decay to prevent unbounded growth):
    Δw(u→v) = η · mean_t(h_u^t · h_v^t) - λ · w(u→v)

    η  : hebbian_lr   - how fast co-activation strengthens connections
    λ  : weight_decay - prevents weights from growing without bound
    t  : timestep index within the current episode
```

## HebbianHook

```python
class HebbianHook(LoopCallback):
    """
    Accumulates co-activation statistics across a full episode, then applies
    a local Hebbian weight update to every edge in one or more NeuronGraphs.

    Parameters
    ----------
    graphs : NeuronGraph | list[NeuronGraph]
        The graph(s) to apply Hebbian updates to. Pass a list to cover all
        regions of a BrainNetwork, or use HebbianHook.from_brain().
    hebbian_lr : float
        Hebbian learning rate η. Typical range: 1e-4 – 1e-2.
    weight_decay : float
        Decay coefficient λ. Prevents unbounded growth. Typical: 1e-4 – 1e-3.
    clip_weight : float | None
        If set, clamps all weights to [-clip_weight, +clip_weight] after update.
    respect_dale : bool
        If True, calls graph.enforce_dale() after each Hebbian update so
        Dale's Law constraints are maintained.
    """

    @classmethod
    def from_brain(cls, brain, hebbian_lr=1e-3, weight_decay=1e-4,
                    clip_weight=None, respect_dale=True) -> "HebbianHook":
        """Create a HebbianHook covering all regions of a BrainNetwork."""

    def record(self) -> None:
        """
        Snapshot the current co-activation product (h_pre · h_post) for every
        edge across all tracked graphs.

        Call this immediately after each forward pass / agent.act() while
        neurons still hold their _current activations for this timestep.
        """

    def apply(self) -> None:
        """
        Apply the Hebbian update using accumulated co-activation statistics.
        Δw = η · mean(h_pre · h_post) - λ · w

        Call once per episode, after agent.learn() so PPO and Hebbian updates
        are both applied before the next episode starts.
        """

    def reset(self) -> None:
        """Clear all accumulated co-activation data. Call at episode start or end."""

    def apply_and_reset(self) -> None:
        """Apply the Hebbian update then immediately clear accumulators."""

    @property
    def n_steps_recorded(self) -> int:
        """Number of timesteps recorded since last reset."""

    def mean_coactivation(self) -> Dict[str, float]:
        """
        Return a dict of edge_id -> mean co-activation value across all graphs.
        Useful for logging how strongly pairs of neurons are correlating.
        """
```

### Usage with GraphAgent

```python
hook = HebbianHook(graph, hebbian_lr=1e-3, weight_decay=1e-4)

for step in episode:
    action = agent.act(obs)
    hook.record()
    obs, reward, done, _ = env.step(action)

agent.learn(episode_data)
hook.apply()
hook.reset()
```

`HebbianHook` is a `LoopCallback` - pass it directly to `Optimizer` or `RLOptimizer` via `callbacks=` and it wires itself automatically (you don't need to call `record()`/`apply()`/`reset()` manually when using the loop).

# NeuromodulatorSignal

```
NeuromodulatorSignal - translates training regime into global parameter changes
across the neuroevo stack, analogous to dopamine / norepinephrine / acetylcholine.

    RegimeDetector
          │
          ▼  (regime: "trending" | "ranging" | "volatile")
    NeuromodulatorSignal
          │
          ├── HebbianHook.hebbian_lr      (local plasticity)
          ├── GraphAgent entropy_coef     (exploration breadth)
          └── TopologyController         (grow / prune aggressiveness)
```

| Regime | Biological analogue | Effect |
|---|---|---|
| `trending` | Dopamine ↑ | Reward signal strong → consolidate, exploit. Lower `hebbian_lr` (don't overwrite good patterns), lower entropy (exploit known policy), raise prune threshold (trim redundant neurons). |
| `ranging` | Acetylcholine ↑, Dopamine ↓ | Plateau → shift attention to new structure. Raise `hebbian_lr` (explore local correlations), raise entropy (broaden action distribution), lower grow threshold (allow topology to expand). |
| `volatile` | Norepinephrine ↑ | High noise → arousal, cautious plasticity. Lower `hebbian_lr` (don't lock in noisy patterns), raise entropy (explore to escape noise), raise prune threshold (don't prune during instability). |

## NeuromodulatorSignal

```python
class NeuromodulatorSignal(LoopCallback):
    """
    Reads the current training regime and modulates learning parameters
    across HebbianHook, GraphAgent, and TopologyController.

    Parameters
    ----------
    detector : RegimeDetector
        Detects the current regime from EvalMetrics history.
    hebbian_hook : HebbianHook, optional
        If provided, hebbian_lr is scaled per regime.
    agent : GraphAgent, optional
        If provided, the entropy_coef hyperparameter is scaled per regime.
    topology_controller : TopologyController, optional
        If provided, grow_gap_threshold and prune_neuron_threshold are scaled.
    hebbian_lr_scale, entropy_scale, grow_gap_scale, prune_threshold_scale : dict, optional
        Override per-regime scale factor dicts.
        Keys: "trending", "ranging", "volatile". Values: float multipliers.
    """

    def step(self, metrics_history) -> str:
        """
        Detect the current regime and apply parameter modulation.
        Call once per episode after metrics are appended.
        Returns the detected regime string.
        """

    @property
    def current_regime(self) -> str:
        """The most recently detected regime."""

    @property
    def state(self) -> dict:
        """Snapshot of all currently active modulated values."""

    def reset_to_base(self) -> None:
        """Restore all modulated parameters to their original base values."""
```

### Default scale factors

```python
_DEFAULT_HEBBIAN_LR_SCALE       = {"trending": 0.5, "ranging": 2.0, "volatile": 0.3}
_DEFAULT_ENTROPY_SCALE          = {"trending": 0.5, "ranging": 2.0, "volatile": 1.5}
_DEFAULT_GROW_GAP_SCALE         = {"trending": 1.5, "ranging": 0.5, "volatile": 1.2}
_DEFAULT_PRUNE_THRESHOLD_SCALE  = {"trending": 1.5, "ranging": 0.8, "volatile": 0.3}
```

### Usage

```python
from tensor_optix.neuroevo import NeuromodulatorSignal
from tensor_optix.core import RegimeDetector

signal = NeuromodulatorSignal(
    detector=RegimeDetector(),
    hebbian_hook=hook, agent=agent, topology_controller=tc,
)

# In your training loop, after each episode:
signal.step(metrics_history)

# Or wire as a callback - called automatically each episode
opt = Optimizer(agent, env, callbacks=[hook, signal])
opt.run()
```

`NeuromodulatorSignal` is itself a `LoopCallback` and can be passed directly to `Optimizer`/`RLOptimizer` via `callbacks=`.

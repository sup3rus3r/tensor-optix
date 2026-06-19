# Write a custom agent

The loop never inspects an agent's internals - it calls exactly six methods. Any algorithm at all (a custom evolutionary method, CMA-ES, an algorithm that doesn't exist yet) is usable as long as it implements `BaseAgent`:

```python
from tensor_optix import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet

class MyAgent(BaseAgent):
    def act(self, observation):
        """Given an observation, return an action. Must be fast - hot path."""
        ...

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Update weights given a completed episode's data.
        Return a dict of training diagnostics (loss, entropy, ...) - may be empty.
        """
        ...

    def get_hyperparams(self) -> HyperparamSet:
        ...

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        ...

    def save_weights(self, path: str) -> None:
        ...

    def load_weights(self, path: str) -> None:
        ...
```

## Declaring on-policy vs. off-policy

```python
class MyOffPolicyAgent(MyAgent):
    is_on_policy = False
```

Off-policy agents (those with a replay buffer accumulating experience from past policies, like DQN/SAC/TD3) **must** override `is_on_policy` to `False`. `LoopController` skips weight rollback for off-policy agents even when `rollback_on_degradation=True` - restoring weights without clearing the buffer would immediately corrupt Bellman targets and drag the policy back down. The default is `True`.

## Enabling SPSA out of the box

If your agent declares `default_param_bounds` and `default_log_params` as class attributes, `RLOptimizer`/`Optimizer` will build an `SPSAOptimizer` automatically without you having to construct one:

```python
class MyAgent(BaseAgent):
    default_param_bounds = {
        "learning_rate": (1e-5, 1e-2),
        "entropy_coef":  (0.0, 0.1),
    }
    default_log_params = ["learning_rate"]
```

`HyperparamSet.params` is treated as a fully opaque dict everywhere in core - never assume the optimizer or loop reads specific keys. Your agent's `set_hyperparams()` is solely responsible for knowing how to apply each key (e.g. updating an optimizer's learning rate group).

## Optional hooks

```python
def average_weights(self, paths: list) -> None:
    """SWA: θ_avg = mean of weights loaded from each path. Default: no-op."""

def perturb_weights(self, noise_scale: float) -> None:
    """θ_new = θ × (1 + noise_scale × ε). Used by PolicyManager spawning. Default: no-op."""

def export_onnx(self, path: str) -> None:
    """Export the actor network. Default: raises NotImplementedError."""

def teardown(self) -> None:
    """Release GPU memory / file handles. Called by PolicyManager.prune(). Default: no-op."""
```

Implement `average_weights` if you want `CheckpointRegistry.load_ensemble()` (stochastic weight averaging across top-k checkpoints) to work for your agent. Implement `perturb_weights` if you want `PolicyManager.spawn_variant()`'s weight-space mutation to actually touch your network rather than being a no-op.

## Architectural rules (if contributing upstream)

From `CONTRIBUTING.md`:

1. No algorithm-specific code in `core/` - PPO, DQN, SAC, etc. must never be referenced there or in `loop_controller.py`.
2. Gymnasium API only - `(obs, info) = env.reset()`, `(obs, reward, terminated, truncated, info) = env.step()`. Never the legacy `done` flag.
3. `HyperparamSet.params` is opaque - core code must never read or hardcode specific key names.
4. Separation of concerns - the optimizer tunes hyperparameters; `PolicyManager` evolves models. Don't mix these.
5. Framework-specific code belongs in `adapters/<framework>/`, not scattered through core.

## Reference

[BaseAgent reference](../reference/core/base_agent.md), [BaseOptimizer](../reference/core/base_optimizer.md), [BaseEvaluator](../reference/core/base_evaluator.md), [BasePipeline](../reference/core/base_pipeline.md).

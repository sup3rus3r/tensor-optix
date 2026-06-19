# tensor_optix.core.ensemble_agent

## EnsembleAgent

```python
class EnsembleAgent(BaseAgent):
    """
    Wraps a PolicyManager to present multiple agents as a single BaseAgent.

    The pipeline and LoopController see one agent. Internally, act() delegates
    to PolicyManager.ensemble_action(), combining multiple policies.

    learn() trains ALL registered ensemble agents on the same episode.
    This is mathematically necessary: if non-primary agents do not update,
    their action distributions diverge from the primary as training progresses.
    The ensemble then degrades - you pay the cost of multiple agents while
    only the primary improves. Each agent learns independently; their
    contributions are combined only at act() time via weighted averaging.

    Diagnostics returned by learn() are from the primary agent. Per-agent
    diagnostics are logged at DEBUG level as "agent_{i}_{key}".

    get/set_hyperparams(), save/load_weights() delegate to primary only -
    the primary is the authoritative agent for checkpointing.

    Usage:
        pm = PolicyManager(registry)
        pm.add_agent(agent_a, weight=1.0)
        pm.add_agent(agent_b, weight=0.5)
        ensemble = EnsembleAgent(pm, primary_agent=agent_a)
        pipeline.set_agent(ensemble)
        rl_opt = RLOptimizer(agent=ensemble, pipeline=pipeline, ...)
    """

    def __init__(self, policy_manager: PolicyManager, primary_agent: BaseAgent): ...

    def act(self, observation) -> any:
        return self._pm.ensemble_action(observation)

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Train all registered ensemble agents on the same episode data.

        Returns primary agent's diagnostics for the LoopController/evaluator.
        Non-primary diagnostics are logged at DEBUG level to avoid polluting
        the main metrics stream.
        """

    def get_hyperparams(self) -> HyperparamSet: ...
    def set_hyperparams(self, hyperparams: HyperparamSet) -> None: ...
    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...

    @property
    def policy_manager(self) -> PolicyManager: ...
```

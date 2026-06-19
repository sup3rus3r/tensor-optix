# tensor_optix.core.policy_manager

## PolicyManager

```python
class PolicyManager:
    """
    Manages model evolution and ensemble logic.

    Separation of concerns:
        Optimizer  → tunes hyperparameters
        PolicyManager → evolves models (swap, variant spawn)

    Core responsibilities:
    1. evolve(): On DORMANT, compare current score vs registry best.
                 If current < best: rollback agent to best checkpoint.
                 If current >= best: no-op (system is at its best known state).
    2. spawn_variant(): Clone best checkpoint into a new agent shell and apply
                        hyperparam mutation. Produces a candidate for the ensemble.
    3. prune(): Remove the lowest-performing agents to keep the ensemble lean.
    4. boost(): Multiply a specific agent's weight - use after regime detection.
    5. ensemble_action(): Combine actions from multiple registered agents
                          via weighted averaging.
    6. update_weights() / auto_update_weights(): Adjust agent weights based on
                          externally provided or internally recorded score history.
    7. adaptive_noise_scale(): Compute a dynamic noise_scale for spawn_variant
                          based on recent improvement - high noise on plateau,
                          low noise when improving.
    8. status(): Structured snapshot of all ensemble state for observability.

    Minimal usage (evolution only):
        pm = PolicyManager(registry)
        cb = pm.as_callback(agent)
        optimizer = RLOptimizer(..., callbacks=[cb])

    Ensemble with spawning:
        pm = PolicyManager(registry)
        pm.add_agent(primary_agent, weight=1.0)
        variant = pm.spawn_variant(SecondAgent(...), noise_scale=0.05)
        pm.add_agent(variant, weight=1.0)
        ensemble = EnsembleAgent(pm, primary_agent=primary_agent)
    """

    def __init__(
        self,
        registry: CheckpointRegistry,
        score_window: int = 10,
        max_spawns: Optional[int] = None,
        max_ensemble_size: Optional[int] = None,
    ): ...
```

### Ensemble management

```python
def add_agent(self, agent: BaseAgent, weight: float = 1.0) -> None:
    """Add an agent to the ensemble pool."""

def ensemble_action(self, obs) -> any:
    """
    Return a combined action from all registered agents.

    Single agent: equivalent to agent.act(obs).

    Multiple agents - two modes depending on whether agents expose action_probs():
      Discrete (agents implement action_probs):
        Weighted average is computed in probability space AFTER softmax,
        not over sampled actions or logits. Averaging logits before softmax
        is not equivalent to averaging policies and produces incorrect results.
        Returns a probability array; caller argmaxes or samples from it.
      Continuous (agents do not implement action_probs):
        Falls back to weighted average of sampled actions. This is correct
        for unimodal continuous policies but loses distributional information
        for multimodal policies.
    """

def update_weights(self, agent_scores: Dict[int, float]) -> None:
    """
    Update ensemble weights based on recent performance.

    agent_scores: {index_in_ensemble: score}
    Weights are set proportional to shifted scores (ensuring positivity).
    Agents not in agent_scores keep their current weight.
    """

def record_agent_score(self, agent_idx: int, score: float) -> None:
    """
    Record a performance score for a specific agent.
    Scores are stored in a rolling window of size score_window.
    Call auto_update_weights() to apply the recorded history to weights.
    """

def auto_update_weights(self) -> None:
    """
    Recompute ensemble weights from internally recorded score history.
    For each agent with recorded scores, weight = mean(recent_scores).
    """

def prune(self, bottom_k: int = 1) -> List[BaseAgent]:
    """
    Remove the bottom_k agents by current weight from the ensemble.
    Returns the list of removed agents (in ascending weight order).
    Calls agent.teardown() on each removed agent.
    """

def boost(self, agent: BaseAgent, factor: float = 2.0) -> None:
    """
    Multiply the weight of a specific agent by factor.

    Use after regime detection to shift action weight toward the most
    relevant policy without zeroing out the others.

    Example:
        regime = detector.detect(metrics_history)
        if regime == "volatile":
            pm.boost(agent_volatile, factor=2.0)
    """

def set_regime(self, regime: str) -> None:
    """Record the current regime label for observability (see status())."""
```

### Adaptive noise

```python
def adaptive_noise_scale(
    self,
    metrics_history: List[EvalMetrics],
    min_scale: float = 0.001,
    max_scale: float = 0.1,
    window: int = 10,
) -> float:
    """
    Compute a dynamic noise_scale for spawn_variant().

    Driven by three signals, not one:

    1. val_score slope: improving val → less noise (don't disrupt)
    2. generalization_gap: large train-val gap → more noise (overfitting,
       need to explore different solutions)
    3. train/val correlation: low correlation → more noise (train is lying -
       val is not following the training signal)

    Without val data (no val_pipeline), falls back to slope-only mode.
    """
```

### Evolution

```python
def evolve(self, agent: BaseAgent, current_score: float) -> bool:
    """
    Called when the loop enters DORMANT state.

    Compares current_score against the best checkpoint in the registry.
    - If current < best: loads best weights into agent. Returns True.
    - If current >= best: no-op. Returns False.
    """

def spawn_variant(
    self,
    agent_shell: BaseAgent,
    noise_scale: float = 0.01,
    mutation_fn: Optional[Callable[[BaseAgent], None]] = None,
) -> BaseAgent:
    """
    Clone best checkpoint into agent_shell and apply mutation.

    Loads the best known weights into agent_shell, then perturbs its
    hyperparameters with multiplicative Gaussian noise. If mutation_fn
    is provided, it is called after weight loading for custom weight
    perturbation.

    noise_scale: std dev for multiplicative Gaussian noise on each
                 numeric hyperparam. Default 0.01 (1% perturbation).

    Returns agent_shell (mutated in place, also returned for chaining).
    """
```

### Reporting

```python
def status(self) -> dict:
    """
    Structured snapshot of current ensemble state: ensemble_size, per-agent
    weight/mean_score/recent_scores, regime, spawn_count, prune_count,
    budget_exhausted, spawns_remaining.
    """

def training_report(self) -> dict:
    """
    Structured training report for display after training completes.

    Superset of status() enriched with best_score, best_val_score,
    best_generalization_gap (all read from the best registry checkpoint),
    and termination_reason.
    """
```

### Properties

```python
@property
def ranked_snapshots(self) -> List[dict]:
    """All registry snapshots ranked by primary_score (highest first)."""

@property
def ensemble_size(self) -> int: ...

@property
def budget_exhausted(self) -> bool:
    """True when max_spawns is set and the spawn budget has been fully used."""

@property
def spawns_remaining(self) -> Optional[int]:
    """Remaining spawns in budget. None if no budget was set."""
```

### as_callback

```python
def as_callback(
    self,
    agent: BaseAgent,
    agent_factory: Optional[Callable[[], BaseAgent]] = None,
    meta_controller: Optional[Any] = None,
) -> "PolicyManagerCallback":
    """
    Returns a LoopCallback that wires PolicyManager into the loop.

    agent_factory: callable that returns a fresh agent shell for spawning.
        When provided, DORMANT triggers spawn-until-budget-exhausted.
        When omitted, DORMANT stops immediately.

    meta_controller: optional MetaController that overrides spawn/stop
        decisions with custom logic.

    Wiring:
        pm = PolicyManager(registry, max_spawns=3)
        cb = pm.as_callback(agent, agent_factory=lambda: MyAgent(...))
        opt = RLOptimizer(..., callbacks=[cb])
        cb.set_stop_fn(opt.stop)
    """
```

## PolicyManagerCallback

```python
class PolicyManagerCallback(LoopCallback):
    """
    Wires PolicyManager into the LoopController event system.

    On every DORMANT event:
    1. Auto-rebalance ensemble weights from score history
    2. Rollback to best checkpoint if current score < best
    3. If agent_factory provided:
       - If MetaController present: delegate SPAWN/PRUNE/STOP/NO_OP decision to it
       - If no MetaController: spawn whenever budget allows (default behaviour)
    4. If budget exhausted: call stop_fn → opt.run() returns cleanly
    """

    def set_stop_fn(self, fn: Callable[[], None]) -> None:
        """
        Register a callable invoked when the spawn budget is exhausted.
        Typically: cb.set_stop_fn(optimizer.stop)
        """
```

`_do_spawn()` grows the weight-perturbation scale based on recent improvement: when at least two metrics are recorded, the scale is `clip(0.01 / (1 + improvement_ratio), 0.005, 0.1)`; otherwise it falls back to `min(0.1, 0.01 × 2^spawn_count)` (exponential growth with spawn count).

See also: [Ensembles and policy evolution](../../guides/ensembles-evolution.md), [MetaController](meta_controller.md).

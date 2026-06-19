# tensor_optix.core.loop_controller

The heart of the library. Most users interact with `RLOptimizer` or `Optimizer` rather than this class directly, but every behavior described here is what those wrappers configure.

## LoopCallback

```python
class LoopCallback:
    """
    Hook into the loop at key events.
    Subclass and override any methods you care about.

    Example:
        class MyLogger(LoopCallback):
            def on_improvement(self, snapshot):
                print(f"New best: {snapshot.eval_metrics.primary_score:.4f}")
    """

    def on_loop_start(self) -> None: ...
    def on_loop_stop(self) -> None: ...
    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None: ...
    def on_improvement(self, snapshot: PolicySnapshot) -> None: ...
    def on_plateau(self, episode_id: int, state: LoopState) -> None: ...
    def on_dormant(self, episode_id: int) -> None: ...
    def on_degradation(self, episode_id: int, eval_metrics: EvalMetrics) -> None: ...
    def on_hyperparam_update(self, old_params: dict, new_params: dict) -> None: ...
```

## LoopController

```python
class LoopController:
    """
    The heart of the library.

    Orchestrates the full autonomous improvement loop:
    1. Cold start: run baseline episode, store snapshot
    2. Main loop: interact → evaluate → compare → tune → repeat
    3. State management via BackoffScheduler
    4. Checkpoint management via CheckpointRegistry
    5. Graceful shutdown on stop(), DORMANT, or max_episodes

    Default behaviour (no configuration needed):
    - DORMANT fires → loop stops. Best known weights are restored before
      run() returns. Caller always gets the optimal policy, not the last one.
    - Callbacks may call stop() to halt early (e.g. PolicyManager after
      spawning variants). Core behaviour is independent of callbacks.

    This class has NO knowledge of TensorFlow or any ML framework.
    This class has NO knowledge of PPO, SAC, DQN, or any RL algorithm.
    All interaction is via the abstract interfaces exclusively.
    """
```

### Constructor

```python
def __init__(
    self,
    agent: BaseAgent,
    evaluator: BaseEvaluator,
    optimizer: BaseOptimizer,
    pipeline: BasePipeline,
    checkpoint_registry: CheckpointRegistry,
    backoff_scheduler: BackoffScheduler,
    rollback_on_degradation: bool = False,
    improvement_margin: float = 0.0,
    max_episodes: Optional[int] = None,
    callbacks: Optional[List[LoopCallback]] = None,
    val_pipeline: Optional[BasePipeline] = None,
    score_smoothing: int = 2,
    checkpoint_score_fn=None,
    verbose: bool = False,
    verbose_log_file: Optional[str] = None,
    diagnostic_controller: Optional["DiagnosticController"] = None,
    min_consecutive_degradations: int = 3,
    convergence_patience: int = 5,
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.20,
    target_score: Optional[float] = None,
)
```

Three separate concerns are kept deliberately independent inside the controller:

1. **Checkpoint saving** - driven by `checkpoint_score` (the output of `checkpoint_score_fn(agent)` when provided, otherwise raw `primary_score`). The best checkpoint is the one with the highest true policy quality; an external deterministic eval is more accurate than the noisy training-window mean.
2. **Convergence / degradation detection** - driven by the *smoothed* `primary_score` (rolling mean of the last `score_smoothing` evals), so a single lucky window can't set an unreachable "best" that permanently blocks `DORMANT`.
3. **`checkpoint_score_fn`** - optional `Callable[[BaseAgent], float]` called after every eval episode to measure true policy quality independently of the training signal.

### Methods

```python
def run(self) -> None:
    """
    Start the loop. Blocks until convergence, stop(), or max_episodes.

    When run() returns, the agent always holds the best known weights -
    whether stopped by convergence, budget, or manual stop().
    """

def stop(self) -> None:
    """Signal the loop to stop cleanly after the current episode."""
```

### Properties

```python
@property
def state(self) -> LoopState: ...

@property
def best_snapshot(self) -> Optional[PolicySnapshot]: ...
```

## Degradation handling

`_handle_degradation()`:

```
Called when watchdog detects performance drop.

Rollback is skipped for off-policy agents (agent.is_on_policy == False)
even when rollback_on_degradation=True. Off-policy agents maintain a
replay buffer containing experience from all past policies - restoring
weights without clearing the buffer produces corrupted Bellman targets
that immediately drag the policy back down. Their buffers naturally
smooth through degradations without rollback.
```

Rollback fires only when all of: `rollback_on_degradation=True`, the agent is on-policy, a best snapshot exists, and the scheduler is currently in `DORMANT`.

## Degradation suppression during optimizer probes

When an online optimizer (SPSA, Backoff) applies a finite-difference probe perturbation, the resulting score drop is self-inflicted, not a genuine policy collapse. The main loop checks `optimizer.is_probing` and `agent.is_on_policy` before firing degradation handling - probes and off-policy noise are both excluded from the degradation path.

## RND eta scheduling

`_update_rnd_eta(event)` adjusts `RNDPipeline`'s intrinsic-reward scale at loop state transitions, when the active pipeline exposes `set_eta`:

```
improvement → eta *= 0.9   (getting better, pull back exploration)
plateau     → eta *= 1.5   (stuck, push exploration; capped at 4x base)
dormant     → eta = 0      (converged, stop injecting noise)
restart     → eta = base   (reset after dormant restart)
```

This is a no-op (zero cost) for pipelines that don't implement `set_eta`. See [Exploration (RND)](../exploration.md).

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
    def on_convergence(self, episode_id: int, report: ConvergenceReport) -> None: ...
```

Callbacks need not subclass `LoopCallback`: hooks a duck-typed callback does not define are skipped.

| Hook | Fires |
|---|---|
| `on_plateau(episode_id, state)` | once, when the run **enters** COOLING |
| `on_dormant(episode_id)` | once, when the run enters DORMANT |
| `on_degradation(episode_id, eval_metrics)` | once per confirmed collapse, for on-policy **and** off-policy agents (rollback only for on-policy) |
| `on_convergence(episode_id, report)` | whenever the convergence regime or the loop state changes; `report` is a [`ConvergenceReport`](convergence.md#convergencereport) |

## LoopController

```python
class LoopController:
    """
    The heart of the library.

    Orchestrates the full autonomous improvement loop:
    1. Cold start: run baseline episode, store snapshot
    2. Main loop: interact → evaluate → compare → tune → repeat
    3. Convergence via ConvergenceMonitor (what the score curve is doing,
       ACTIVE / COOLING / DORMANT); evaluation cadence via BackoffScheduler
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
    cv_threshold: Optional[float] = None,      # deprecated, no effect
    gap_threshold: Optional[float] = None,     # deprecated, no effect
    target_score: Optional[float] = None,
    criteria_mode: str = "none",
    criteria_k: float = 2.0,
    checkpoint_confirmation: bool = False,
    checkpoint_confirm_window: int = 3,
    checkpoint_noise_k: float = 2.0,
    min_completed_episodes_before_dormant: int = 10,
    convergence_monitor: Optional[ConvergenceMonitor] = None,
)
```

- `convergence_monitor`: the [`ConvergenceMonitor`](convergence.md) driving convergence. When omitted, one is built from the scheduler:
  - **Patience:** an explicitly set `plateau_threshold` / `dormant_threshold` becomes the monitor's cooling / dormant patience, in episodes. Otherwise the monitor defaults apply (32 / 160).
  - **Significance:** `noise_k` sets the significance `z`.
  - **Earliest DORMANT:** `min_episodes_before_dormant` becomes `min_at_before_dormant`.
  - **Persistence:** `min_consecutive_degradations` sets how many observations a new regime must hold.
- `cv_threshold` / `gap_threshold` are deprecated and do nothing (they fed an `is_converged()` check that the loop never called); passing either emits a `DeprecationWarning`.

Three separate concerns are kept deliberately independent inside the controller:

1. **Checkpoint saving** - driven by `checkpoint_score` (the output of `checkpoint_score_fn(agent)` when provided, otherwise raw `primary_score`, then run through composite scoring and confirmation - see below). The best checkpoint is the one with the highest true policy quality; an external deterministic eval is more accurate than the noisy training-window mean.
2. **Convergence / degradation detection** - the raw `primary_score` of every observed window feeds the `ConvergenceMonitor`, which does its own multi-timescale smoothing and peak tracking (see [convergence](convergence.md)).
3. **`checkpoint_score_fn`** - optional `Callable[[BaseAgent], float]` called after every eval episode to measure true policy quality independently of the training signal.

### Avoiding "unicorn" checkpoints

By default (`criteria_mode="none"`), checkpoint acceptance is a raw point
comparison: `checkpoint_score > best + improvement_margin`. This is exactly
what earlier versions did, kept as the default so upgrading never silently
changes what gets checkpointed. The problem with it: a single lucky episode
during early high-entropy exploration - reward concentrated on one step
rather than earned consistently - can produce a raw score that no later,
genuinely better policy ever beats, because that lucky episode becomes a
permanent, unbeatable "best."

Two opt-in parameters address this, both documented in full on
[BaseEvaluator](base_evaluator.md#measuring-good-as-more-than-one-number):

- `criteria_mode`: `"none"` (default) | `"manual"` | `"auto"` | `"both"`.
  Widens the measurement itself - `"auto"` needs no domain code at all (it
  reads the shape of the reward stream that already exists on every
  episode); `"manual"` uses a `BaseEvaluator.criteria()` rubric you define.
  `criteria_k` (default 2.0) controls how sharply an unsatisfied fraction is
  penalized.
- `checkpoint_confirmation` (default `False`, implicitly `True` whenever
  `criteria_mode != "none"`): gates acceptance through a
  `CheckpointConfirmationTracker` requiring either low enough measured noise
  or `checkpoint_confirm_window` (default 3) consecutive corroborating evals
  before trusting an improvement - the same trend-over-point philosophy
  `BackoffScheduler` already uses for improvement/degradation state, applied
  here to checkpoint acceptance instead. `checkpoint_noise_k` (default 2.0,
  same semantics as `BackoffScheduler.noise_k`) scales the adaptive floor.

When `checkpoint_score_fn` is provided, composite scoring is skipped (the
external eval is already presumed trustworthy) but confirmation still
applies, guarding against a fluke in the external eval itself.

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

@property
def monitor(self) -> ConvergenceMonitor: ...        # pass to TopologyController.set_scheduler

@property
def convergence(self) -> Optional[ConvergenceReport]: ...   # most recent report
```

## Observation cadence

When scoring is free (no `val_pipeline`, no `checkpoint_score_fn`), **every window is scored and observed**, so the monitor sees the whole curve. Checkpoint decisions run on every observed window too. The `BackoffScheduler` interval then paces only the costly steps:

- `optimizer.suggest`;
- the validation rollout, when a `val_pipeline` is set. In that case the monitor observes the validation scores at their real episode positions.

The interval resets to `base_interval` on improvement and backs off by `backoff_factor` on each due evaluation without improvement.

## Degradation handling

A collapse is confirmed by the monitor: the `COLLAPSING` regime must hold for `min_consecutive_degradations` (default 3) consecutive observations. It is never confirmed on a window scored under an optimizer probe (`optimizer.is_probing`), because that drop is self-inflicted. On confirmation `_handle_degradation()` runs once:

```
Rollback is skipped for off-policy agents (agent.is_on_policy == False)
even when rollback_on_degradation=True. Off-policy agents maintain a
replay buffer containing experience from all past policies - restoring
weights without clearing the buffer produces corrupted Bellman targets
that immediately drag the policy back down. Their buffers naturally
smooth through degradations without rollback.
```

`on_degradation` fires for every agent. Rollback happens only when all three hold: `rollback_on_degradation=True`, the agent is on-policy, and a best snapshot exists. After a rollback the monitor is told (`note_restored`):

- **History:** scores from the collapsed policy are dropped from the trend history.
- **Peak and patience:** both are kept, since restoring is not progress.
- **No early DORMANT:** the run cannot be declared DORMANT until the restored policy's scores can be read.
- **Renewed collapse:** if the restored policy collapses again before it has been back at the peak, the collapse is reported (`on_degradation`) but **not** rolled back a second time. A restore that did not help points at something other than the weights (environment, hyperparameters), and repeating it would loop forever.

## Convergence actions

Each action runs **once, on entry**:

- **Entering COOLING:** `optimizer.on_cooling()` (by default the same reaction as a plateau: widen the search, early enough for it to run), RND η × 1.5, and `on_plateau` fires.
- **Entering DORMANT:** `optimizer.on_plateau()`, RND η = 0, `on_dormant` fires, then the DORMANT rule below.
- **Every `IMPROVING` observation:** `optimizer.on_improvement()` and RND η × 0.9.

A regressed run is **reported** (`on_convergence`, `report.gap_to_peak`, `report.eta_to_peak`) and keeps counting toward DORMANT, because climbing back to the peak is not progress.

- **While collapsing:** DORMANT is withheld while a collapse is being confirmed or is in progress.
- **Once the fall is past:** a run that stays down turns into `REGRESSED_STALLED`. Patience keeps running from the last real improvement, so the run is eventually declared DORMANT and stops, and `run()` restores the best weights.

**DORMANT behaviour:**

- **Sticky:** once DORMANT, only new ground or a confirmed collapse leaves it.
- **Not during warm-up:** DORMANT is never entered while the trends are still in warm-up. During warm-up the evaluation interval does not back off either, so the monitor gets the observations it needs.
- **Final state:** when the loop stops at DORMANT, `state` and `convergence.state` both read DORMANT.
- **Running again:** a repeated `run()` starts a new monitor segment and keeps the peak.

## What happens at DORMANT

After `on_dormant` callbacks have run, the loop stops unless a callback asks for more training:

- A callback may expose `consume_dormant_resume() -> bool`. It is polled once per DORMANT event on every callback. If any returns `True` (and no callback called `stop()`), the scheduler and monitor are restarted (`record_restart()`) and training continues. The new variant must still beat the run's peak.
- `PolicyManagerCallback` returns `True` when that DORMANT event spawned or pruned a variant. Budget exhausted, a `MetaController` `STOP`/`NO_OP`, or passive mode (no `agent_factory`) leave the default in place.
- Otherwise the loop stops and the best checkpoint is restored.

## RND eta scheduling

`_update_rnd_eta(event)` adjusts `RNDPipeline`'s intrinsic-reward scale at loop state transitions, when the **training** pipeline exposes `set_eta` (the validation pipeline is only consulted if the training pipeline does not):

```
improvement → eta *= 0.9   (every IMPROVING observation: pull back exploration)
plateau     → eta *= 1.5   (once, on entering COOLING: push exploration; capped at 4x base)
dormant     → eta = 0      (converged, stop injecting noise)
restart     → eta = base   (reset after dormant restart)
```

This is a no-op (zero cost) for pipelines that don't implement `set_eta`. See [Exploration (RND)](../exploration.md).

## Verbose output

`verbose=True` prints a per-eval diagnostic block: the score, each timescale's level and slope ± SE, the regime with peak / gap / noise, the state with patience, and the optimizer step. Output goes through `tensor_optix.utils.console.safe_print`, which replaces glyphs the output stream cannot encode (e.g. box-drawing characters on a Windows cp1252 console or a redirected stdout) with `?` instead of raising `UnicodeEncodeError`. Pass `verbose_log_file=` to write the same output to a UTF-8 file.

## Scores are finished-episode returns

A window is a fixed number of steps, not an episode, so the loop keeps track of which windows actually contain something to score.

- **Returns are tracked across windows.** `BatchPipeline`, `VectorBatchPipeline` and the multi-agent pipelines keep a running return per environment (or seat) between windows and publish the full returns of the episodes that *finished* in a window as `EpisodeData.metadata["episode_returns"]`. An episode that spans five windows is scored once, with its whole return, in the window where it ends.
- **The default evaluators** (`TorchEvaluator` / `TFEvaluator`) use the mean of those returns as `primary_score` and report `metrics["episodes_completed"]`. With a custom `primary_score_fn` the count is not reported and none of the rules below apply.
- **A window in which no episode finished is not an evaluation.** The agent still learns from it, but there is no checkpoint decision, no monitor observation and no hyperparameter step; `on_episode_end` fires with `eval_metrics=None`.
- **Before the first episode ever finishes** there is nothing better to score with, so windows are scored by mean per-step reward. When the first episode finishes the scale changes from per-step reward to episode return, so the best score, the monitor (`reset_scale()`) and the metrics history are reset at that point - otherwise a per-step best such as `-1.0` could never be beaten by a return such as `-200`.
- **`min_completed_episodes_before_dormant`** (default `10`): DORMANT is held (`monitor.hold_dormant()`: neither entered nor reported) until that many episodes have finished. A flat score built on a handful of episodes - or on none - is not treated as convergence. For a genuinely continuing task whose episodes never end, set it to `0`.

With a validation pipeline the same rules are applied to the validation windows (`val_episodes_completed`), since the validation score is what drives the loop.

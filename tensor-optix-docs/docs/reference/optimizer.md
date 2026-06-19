# Top-level API: RLOptimizer and Optimizer

## RLOptimizer

```python
class RLOptimizer:
    """
    Main public API. This is what users import.

    Assembles all components and delegates to LoopController.
    Sensible defaults for everything - minimal required args: agent, pipeline.

    Minimal usage:
        optimizer = RLOptimizer(agent=my_tf_agent, pipeline=my_pipeline)
        optimizer.run()

    With automatic trial-level hyperparameter search (TrialOrchestrator):
        optimizer = RLOptimizer(
            agent_factory=lambda params: MyAgent(params),
            pipeline_factory=lambda: MyPipeline(),
            param_space={
                "learning_rate": ("log_float", 1e-4, 3e-3),
                "clip_ratio":    ("float", 0.1, 0.3),
            },
            n_trials=20,
            trial_steps_fraction=0.15,   # 15% of max_steps per trial
        )
        optimizer.run()   # trial search runs first, then full training with best params

    When param_space + agent_factory + pipeline_factory are provided:
      1. TrialOrchestrator runs n_trials independent short runs to find best params
      2. agent_factory(best_params) creates the agent for the full run
      3. The main RLOptimizer loop runs with those params + SPSA online adaptation

    The agent and pipeline args become optional when factories are provided.
    """
```

### Constructor

```python
def __init__(
    self,
    agent: Optional[BaseAgent] = None,
    pipeline: Optional[BasePipeline] = None,
    evaluator: Optional[BaseEvaluator] = None,
    optimizer: Optional[BaseOptimizer] = None,
    checkpoint_dir: str = "./tensor_optix_checkpoints",
    max_snapshots: int = 10,
    rollback_on_degradation: bool = False,
    improvement_margin: float = 0.0,
    max_episodes: Optional[int] = None,
    base_interval: int = 1,
    backoff_factor: float = 2.0,
    max_interval_episodes: int = 100,
    plateau_threshold: int = 5,
    dormant_threshold: int = 20,
    degradation_threshold: float = 0.95,
    min_degradation_drop: float = 1e-4,
    noise_k: float = 2.0,
    score_window: int = 20,
    trend_window: int = 8,
    min_episodes_before_dormant: int = 0,
    min_episodes_before_degradation: int = 5,
    callbacks: Optional[List[LoopCallback]] = None,
    val_pipeline: Optional[BasePipeline] = None,
    score_smoothing: int = 2,
    checkpoint_score_fn=None,
    verbose: bool = False,
    verbose_log_file: Optional[str] = None,
    # ── DiagnosticController ──
    diag_loss_spike_factor: float = 5.0,
    diag_entropy_floor: Optional[float] = 0.05,
    diag_target_kl: Optional[float] = 0.02,
    diag_epsilon_patience: int = 20,
    diag_epsilon_reset_value: float = 0.3,
    diag_epsilon_score_threshold: float = 20.0,
    diag_min_episodes: int = 5,
    min_consecutive_degradations: int = 3,
    convergence_patience: int = 5,
    cv_threshold: float = 0.05,
    gap_threshold: float = 0.20,
    target_score: Optional[float] = None,
    # ── Trial-level search (TrialOrchestrator) ──
    agent_factory: Optional[Callable[[Dict[str, Any]], BaseAgent]] = None,
    pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
    param_space: Optional[Dict[str, tuple]] = None,
    n_trials: int = 20,
    trial_steps_fraction: float = 0.01,
    val_pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
    trial_agent_factory: Optional[Callable[[Dict[str, Any]], BaseAgent]] = None,
)
```

Either `agent` or `agent_factory` must be provided, and either `pipeline` or `pipeline_factory`.

`RLOptimizer` builds, internally: a `TFEvaluator` (default, if none given), an `SPSAOptimizer` seeded from the agent's `default_param_bounds`/`default_log_params` (if no optimizer given, or if a given `SPSAOptimizer` has no bounds yet), a `CheckpointRegistry`, a `BackoffScheduler`, and a `DiagnosticController` - then wraps them all in a `LoopController`.

Off-policy agents (`agent.is_on_policy == False`) get more patient defaults automatically: if `dormant_threshold` is left at its default `20`, it's lowered to `15`; if `min_episodes_before_dormant` is left at `0`, it's raised to `agent.default_min_episodes_before_dormant` (default `100`).

### Methods

```python
def run(self) -> None:
    """
    Start the autonomous loop. Blocks until stopped or max_episodes reached.

    If param_space + agent_factory + pipeline_factory were provided:
      1. Runs TrialOrchestrator to find best hyperparameter configuration.
      2. Calls agent_factory(best_params) and pipeline_factory() to build
         the agent and pipeline for the full training run.
      3. Runs the main loop with those params + SPSA online adaptation.
    """

def add_callback(self, cb: LoopCallback) -> None:
    """
    Register an additional callback. Safe to call before or after run().
    Callbacks added before run() are merged into the controller when it is built.
    """

def stop(self) -> None:
    """Signal graceful shutdown after current episode."""
```

### Properties

```python
@property
def state(self) -> LoopState:
    """Current LoopState."""

@property
def best_snapshot(self) -> Optional[PolicySnapshot]:
    """Best known PolicySnapshot."""
```

When the trial-search path runs, the best trial's weights are warm-started into the main agent via `load_weights()` before the full run begins, and all trial checkpoint directories are deleted afterward.

---

## Optimizer

```python
class Optimizer:
    """
    Simplified entry point for training. Wraps RLOptimizer with sensible defaults.

    Parameters
    ----------
    agent:
        A fully-constructed agent from make_agent() or built manually.
    env:
        A Gymnasium environment. Used for window_size estimation and as the
        training environment when n_envs=1.
    n_envs:
        Number of parallel environments. When >1, pass env as a callable
        (``lambda: gym.make("EnvName")``) or a list of callables.
    window_size:
        Number of steps per training window. When None, computed automatically
        from env and agent type.
    max_episodes:
        Stop after this many episodes. None means run indefinitely.
    callbacks:
        Additional LoopCallback instances to attach.
    rollback_on_degradation:
        Roll back to the best checkpoint when performance degrades.
    checkpoint_dir:
        Directory for saving policy snapshots.
    kwargs:
        Additional keyword arguments forwarded to RLOptimizer.
    """

    def __init__(
        self,
        agent,
        env,
        n_envs: int = 1,
        window_size: Optional[int] = None,
        max_episodes: Optional[int] = None,
        callbacks: Optional[List] = None,
        rollback_on_degradation: bool = False,
        checkpoint_dir: str = "./tensor_optix_checkpoints",
        optimizer=None,
        loss="auto",
        batch_size: int = 64,
        shuffle: bool = True,
        **kwargs,
    ): ...

    def run(self):
        """Start training. Blocks until max_episodes or KeyboardInterrupt."""
```

`Optimizer` auto-detects **ML mode**: if `agent` is an `nn.Module` (and not a `BaseAgent`) and `env` is a `torch.utils.data.Dataset` or `DataLoader`, it builds an `MLAgent` + `DatasetPipeline` instead of an RL pipeline. See [ML mode](ml.md).

For the RL path, it:

1. Infers the algorithm name from the agent's class name (for window-size lookup): `"ppo"` if `"ppo"` or `"graph"` is in the class name, `"sac"`, `"td3"`, `"dqn"`/`"rainbow"`, else falls back on `agent.is_on_policy`.
2. Computes `window_size` via `optimal_window_size(env, algorithm=alg_name)` if not given explicitly.
3. Builds `BatchPipeline` (`n_envs=1`) or `VectorBatchPipeline` (`n_envs>1`).
4. Collects any `agent._neuroevo_callbacks` (set by `make_agent(..., neuroevo=True)`) and adds them to `callbacks`.
5. Builds an `SPSAOptimizer` from `agent.default_param_bounds` unless `optimizer=` is passed explicitly or the agent declares no bounds (in which case SPSA is left inactive, with a warning logged).
6. Assembles and stores an `RLOptimizer`.

`opt.run()` simply delegates to the wrapped `RLOptimizer.run()`.

# Learning-to-Learn Engine

```
Six composable layers, each reading/writing the shared experiment ledger:

  1. tensor_optix.ledger      - ExperimentStore (SQLite-backed, idempotent, provenance-tracked)
  2. tensor_optix.scheduler   - ConcurrentTrialScheduler (real mp.Process worker isolation)
  3. tensor_optix.selection   - StrategySelector (non-learned baseline allocation)
  4. tensor_optix.nanonets    - NanonetsSelector (cross-task pooling, conservative abstention gate)
  5. tensor_optix.strategy    - typed grammar, StrategyTrialRunner, PromotionPolicy
  6. tensor_optix.runtime     - LiveAdaptiveRuntime (recommend-only observe/experiment/evaluate loop)

No layer uses generative AI or executes arbitrary generated code. Strategy
candidates are reconstructed only from a fixed OP_REGISTRY via keyword
arguments matching each dataclass's declared fields -- never eval()/exec().

See the guide for narrative usage: ../guides/learning-to-learn.md
```

## 1. Ledger (`tensor_optix.ledger`)

```python
class ExperimentStore:
    def __init__(self, db_path: str): ...
    def close(self) -> None: ...

    # Experiments
    def create_experiment(
        self, *, task_env: str, algorithm: str, framework: str, seed: int,
        checkpoint_dir: str, solve_threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        parent_experiment_id: Optional[str] = None, study_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """Idempotent on experiment_id -- a second call with the same id
        returns the existing row rather than raising or duplicating it."""

    def start_experiment(self, experiment_id: str) -> None: ...
    def record_progress(
        self, experiment_id: str, *, best_primary_score: Optional[float] = None,
        best_train_score: Optional[float] = None, best_val_score: Optional[float] = None,
        generalization_gap: Optional[float] = None,
    ) -> None: ...
    def finalize_experiment(
        self, experiment_id: str, *, state: ExperimentState,
        evaluation_status: EvaluationStatus = EvaluationStatus.PRELIMINARY,
        final_metrics: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None,
    ) -> None:
        """Idempotent -- guarded by WHERE finished_at IS NULL."""

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]: ...
    def query_experiments(
        self, *, task_env: Optional[str] = None, algorithm: Optional[str] = None,
        framework: Optional[str] = None, state: Optional[ExperimentState] = None,
        evaluation_status: Optional[EvaluationStatus] = None, study_id: Optional[str] = None,
        created_after: Optional[float] = None, created_before: Optional[float] = None,
        min_best_primary_score: Optional[float] = None, limit: int = 100,
    ) -> List[ExperimentRecord]: ...

    # Events (append-only, per-experiment)
    def append_event(self, experiment_id: str, event: EventRecord) -> bool: ...
    def get_events(self, experiment_id: str) -> List[EventRecord]: ...

    # Artifacts
    def record_artifact(self, artifact: ArtifactRef) -> None: ...
    def get_artifacts(self, experiment_id: str) -> List[ArtifactRef]: ...

    # Strategy candidates / evaluations / registry (sec 5 below)
    def create_strategy_candidate(self, *, candidate_id: str, op_name: str, op_version: int,
                                   grammar_schema_version: int, params: Optional[Dict] = None,
                                   created_by: Optional[str] = None) -> StrategyCandidateRecord: ...
    def get_strategy_candidate(self, candidate_id: str) -> Optional[StrategyCandidateRecord]: ...
    def record_strategy_evaluation(self, *, candidate_id: str, experiment_id: str, task_env: str,
                                    algorithm: str, framework: str, seed: int,
                                    baseline_experiment_id: Optional[str] = None,
                                    outcome_delta: Optional[float] = None,
                                    evaluation_id: Optional[str] = None) -> StrategyEvaluationRecord: ...
    def query_strategy_evaluations(self, *, candidate_id: Optional[str] = None,
                                    task_env: Optional[str] = None, limit: int = 1000) -> List[StrategyEvaluationRecord]: ...
    def get_strategy_registry_entry(self, candidate_id: str) -> Optional[StrategyRegistryRecord]: ...
    def approve_strategy(self, candidate_id: str, *, reviewer: str, scope: Optional[Dict] = None) -> None:
        """The only call that can set a candidate's status to APPROVED.
        Never called by PromotionPolicy.evaluate() in approval_mode="human"."""
    def demote_strategy(self, candidate_id: str, reason: str) -> None:
        """Sets status to DEMOTED. Every consumer of the registry filters
        status=='approved' live on every read, never cached -- a demotion
        takes effect immediately everywhere."""
    def query_approved_strategies(self, *, limit: int = 100) -> List[StrategyRegistryRecord]: ...
    def record_strategy_scope(self, candidate_id: str, scope: Dict[str, Any]) -> None: ...
    def get_strategy_scope(self, candidate_id: str) -> Optional[Dict[str, Any]]: ...
```

### ExperimentLedgerCallback

```python
class ExperimentLedgerCallback(LoopCallback):
    def __init__(
        self, store: ExperimentStore, *, task_env: str, algorithm: str,
        framework: str = "torch", seed: int = 0, checkpoint_dir: str,
        solve_threshold: Optional[float] = None, config: Optional[Dict[str, Any]] = None,
        parent_experiment_id: Optional[str] = None, study_id: Optional[str] = None,
        experiment_id: Optional[str] = None, checkpoint_registry: Optional[Any] = None,
    ): ...

    @property
    def experiment_id(self) -> str: ...

    def mark_failed(self, exc: Exception) -> None:
        """Finalize the experiment as FAILED / INCONCLUSIVE (idempotent)."""
```

Attach it to any `RLOptimizer` / `LoopController` run to record that run in the ledger: the
experiment row (with provenance: tensor-optix version, git revision, library versions, hostname),
one event per loop hook, a progress update and an `ArtifactRef` (with a SHA-256 of the weights)
per improvement.

Terminal state is decided in `on_loop_stop`, which the loop fires from `run()`'s `finally`:

| How the run ended | `state` | `evaluation_status` |
|---|---|---|
| returned normally | `COMPLETED` | `PRELIMINARY` |
| an exception is propagating | `FAILED` (with `error_message`) | `INCONCLUSIVE` |
| `KeyboardInterrupt` | `CANCELLED` | `INCONCLUSIVE` |

So a crashed run is recorded as failed without any caller-side handling; calling
`mark_failed(exc)` yourself afterwards is harmless (finalization is idempotent).

Library versions are read from package metadata; frameworks that the run has not imported
(e.g. TensorFlow in a torch-only run) are not imported just to stamp the row.

Schema version 3. Tables: `experiments`, `events`, `artifacts`, `strategy_candidates`,
`strategy_evaluations`, `strategy_registry`. Migrations are version-gated in
`tensor_optix/ledger/schema.py::_migrate()` — one `if from_version == N:` branch per real,
shipped hop.

## 2. Concurrent scheduler (`tensor_optix.scheduler`)

```python
class ConcurrentTrialScheduler:
    def __init__(
        self, agent_factory: Callable[[Dict[str, Any]], BaseAgent],
        pipeline_factory: Callable[[], BasePipeline],
        param_space: Optional[ParamSpace] = None, n_trials: int = 20,
        trial_steps: int = 50_000, direction: str = "maximize",
        n_startup_trials: int = 10, pruner_warmup_steps: int = 5, pruner_interval: int = 1,
        rloptimizer_kwargs: Optional[Dict[str, Any]] = None,
        val_pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
        checkpoint_score_fn: Optional[Callable[[BaseAgent], float]] = None,
        study_name: Optional[str] = None, storage: Optional[str] = None,
        sampler: Optional["optuna.samplers.BaseSampler"] = None, n_workers: int = 2,
        max_wall_clock_s: Optional[float] = None, trial_timeout_s: Optional[float] = None,
        max_retries: int = 2, ledger_store: Optional[ExperimentStore] = None,
        task_env: Optional[str] = None, algorithm: Optional[str] = None,
        framework: str = "torch", solve_threshold: Optional[float] = None,
        base_seed: int = 0, poll_interval_s: float = 0.5,
        device_placement_fn: Optional[Callable[[int], str]] = None,
    ): ...

    def run(self) -> Tuple[Dict[str, Any], float]:
        """Dispatches up to n_workers concurrent trials as real mp.Process
        workers. Returns (best_params, best_score)."""

    @property
    def study(self) -> "optuna.Study": ...
```

Workers never touch the Optuna `Study` or the `ExperimentStore` directly — they receive a
plain sampled-params dict and report scores back over a queue; all ledger/study writes happen
in the parent process.

Failure handling:

- A worker that **raises** reports `status="failed"`; that is a property of the trial, not of
  the infrastructure, so it is not retried - the trial is told to Optuna as `FAIL` and
  finalized `FAILED` in the ledger.
- A worker process that **dies without reporting** (OOM-kill, segfault, external kill) is
  detected by the parent on the next poll. It is treated as transient and re-dispatched with
  a new seed up to `max_retries` times, then finalized `FAILED` with the exit code in
  `error_message`. `run()` therefore always terminates, with or without `trial_timeout_s`.
- A trial exceeding `trial_timeout_s` is cancelled and finalized `TIMED_OUT`.

## 3. Baseline selection (`tensor_optix.selection`)

```python
@dataclass
class MethodStats:
    method: str
    n_experiments: int
    n_completed: int
    best_score: Optional[float]
    score_interval: Optional[ScoreInterval]
    success_rate: float

@dataclass
class SelectionDecision:
    allocations: Dict[str, int]
    rationale: str

class StrategySelector:
    def __init__(self, store: ExperimentStore): ...
    def stats_for(self, *, task_env: str, method: str) -> MethodStats: ...
    def compare(self, *, task_env: str, methods: List[str]) -> Dict[str, MethodStats]: ...
    def propose(
        self, *, task_env: str, methods: List[str], budget: int,
        task_features: Optional[Dict[str, Any]] = None,
    ) -> SelectionDecision:
        """task_features must be None in this layer -- raises NotImplementedError
        otherwise; reserved for the Nanonets layer above."""
    def observe(
        self, *, orchestrator, task_env: str, algorithm: str, framework: str = "torch",
        seed: int = 0, solve_threshold: Optional[float] = None,
    ) -> List[Any]:
        """Backfills a completed TrialOrchestrator study into the ledger."""
```

`tensor_optix.selection.uncertainty.score_interval(scores: List[float]) -> ScoreInterval`
computes a normal-approximation 95% CI (`mean`, `std`, `n`, `lo95`, `hi95`) — descriptive
uncertainty over repeated ledger observations, not a predictive model.

## 4. Cross-task transfer (`tensor_optix.nanonets`)

```python
@dataclass
class PooledEstimate:
    method: str
    pooled_mean: float
    pooled_interval: SmallNInterval
    sources_used: List[str]
    sources_gated_out: List[str]
    fallback_triggered: bool

class NanonetsSelector:
    def __init__(self, store: ExperimentStore, *, source_task_envs: List[str]):
        """source_task_envs is always explicit -- never auto-discovered,
        so leave-one-task-out leakage discipline is enforced at the call
        site, not inferred."""

    def task_features(self, task_env: str) -> TaskFeatures: ...
    def pooled_stats(self, *, task_env: str, method: str) -> PooledEstimate: ...
    def propose(self, *, task_env: str, methods: List[str], budget: int) -> SelectionDecision:
        """Falls back to StrategySelector.propose() if every method's
        pooled estimate triggers fallback_triggered."""
```

Abstention gate (`tensor_optix.nanonets.abstention`):

```python
MIN_TARGET_N = 2

def normalize_score(raw_score: float, solve_threshold: float) -> Optional[float]:
    """(raw_score - solve_threshold) / abs(solve_threshold). None if
    solve_threshold == 0.0. Plain division is unsafe across a sign
    boundary -- this signed-distance form is not."""

def normalize_scores(raw_scores: List[float], solve_threshold: Optional[float]) -> Optional[List[float]]: ...

def passes_percentile_gate(source_scores: List[float], target_scores: List[float]) -> bool:
    """A source is eligible iff its median >= the target's own 95th
    percentile, both on NORMALIZED scores."""

def similarity_weight(*, source_algorithm: str, target_algorithm: str,
                       source_n: int, target_n: int) -> float:
    """Algorithm-family match contributes full weight (mismatch halves
    it); extreme sample-size imbalance scales the weight down further."""
```

Pooling (`tensor_optix.nanonets.pooling.pool_estimates`) combines the target's own scores with
gate-passed sources via inverse-variance + similarity weighting. The caller (`NanonetsSelector`)
normalizes inputs the same way the gate does before pooling, then de-normalizes the pooled
result back to the target's own raw units — pooling in raw, mixed-scale units was a real bug
found and fixed via live validation; pooling must happen in the same comparable space the gate
uses, not a different one.

## 5. Strategy discovery (`tensor_optix.strategy`)

```python
@dataclass(frozen=True)
class StrategyOp:
    op_name: ClassVar[str] = ""
    def required_capabilities(self) -> Tuple[str, ...]: ...
    def supports(self, agent: BaseAgent) -> bool: ...
    def attach(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        """Returns None (never raises) if `agent` lacks a capability this
        op needs -- self-exclusion is the primary safety mechanism."""

OP_REGISTRY: Dict[str, Type[StrategyOp]]  # the 7 concrete ops, nothing else reachable
```

The 7 concrete ops — `LearningRateTrendOp`, `CheckpointRollbackOp`, `ExplorationBoundsOp`,
`OptimizationMethodChoiceOp`, `BudgetAllocationOp`, `ScheduleChangeOp`,
`PolicyVariantSelectionOp` — each a frozen dataclass with named, typed fields; see the guide for
field-level detail on each.

```python
@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str            # sha256 content hash, first 16 hex chars
    op_name: str
    op_version: int
    params: Dict[str, Any]
    grammar_schema_version: int = 1

    @staticmethod
    def from_op(op: StrategyOp, op_version: int = 1) -> "StrategyCandidate": ...
    def to_op(self) -> StrategyOp:
        """Reconstructs via OP_REGISTRY[self.op_name](**self.params) only
        -- never eval()/exec(). Raises ValueError for an unknown op_name,
        TypeError for an unknown param key (dataclass construction itself
        enforces the closed field set)."""
```

```python
class StrategyTrialRunner:
    def __init__(
        self, *, agent_factory: Callable[[Dict[str, Any]], BaseAgent],
        pipeline_factory: Callable[[], BasePipeline],
        base_params: Optional[Dict[str, Any]] = None,
        rloptimizer_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_score_fn: Optional[Callable[[BaseAgent], float]] = None,
        trial_steps: int = 50_000, ledger_store: Optional[ExperimentStore] = None,
        task_env: str = "", algorithm: str = "", framework: str = "torch",
        solve_threshold: Optional[float] = None, trial_timeout_s: Optional[float] = None,
        poll_interval_s: float = 0.5,
    ): ...

    def run_candidate(self, candidate: StrategyCandidate, *, seeds: List[int]) -> List[Dict[str, Any]]:
        """For each seed, dispatches a PAIRED candidate + no-strategy-
        baseline mp.Process trial (same seed/task/budget). Returns a list
        of {seed, candidate_score, baseline_score, outcome_delta,
        candidate_experiment_id, baseline_experiment_id, candidate_status,
        baseline_status} dicts. Writes ExperimentRecords and one
        StrategyEvaluationRecord per seed if a ledger_store is supplied."""
```

Comparator (`tensor_optix.strategy.comparator`):

```python
MIN_EVALUATIONS = 3

def compare_candidate(store: ExperimentStore, candidate_id: str, task_env: str,
                       *, margin: float = 0.0) -> StrategyComparison:
    """Gates on lo95(outcome_delta) > margin AND n_evaluations >= 3 --
    the pessimistic CI bound, not the mean; hard floor on distinct seeds."""

def compare_against_approved(store: ExperimentStore, candidate_id: str, task_env: str,
                              *, margin: float = 0.0) -> StrategyComparison:
    """As above, plus recomputes (never cached) each currently-approved
    strategy's own interval for the same task_env for a head-to-head check."""
```

Promotion (`tensor_optix.strategy.promotion`):

```python
class PromotionPolicy:
    def __init__(
        self, store: ExperimentStore, *, margin: float = 0.0, min_seeds: int = 3,
        min_held_out_tasks: int = 1, max_compute_ratio: float = 1.5,
        approval_mode: str = "human",   # "human" | "auto" -- ValueError otherwise
    ): ...

    def evaluate(self, candidate_id: str, *, train_task_envs: List[str],
                 held_out_task_envs: List[str]) -> PromotionVerdict:
        """Six criteria: beats_baseline_margin, repeats_across_seeds,
        generalizes_held_out, meets_compute_constraints, reproducible,
        human_reviewed. In approval_mode="human" (default), human_reviewed
        can only become True via a prior, separate store.approve_strategy()
        call -- evaluate() never calls it. In approval_mode="auto", if the
        other five all pass, evaluate() calls approve_strategy() itself
        with reviewer="auto:PromotionPolicy" -- always traceable, never silent."""

@dataclass
class PromotionVerdict:
    candidate_id: str
    approved: bool
    criteria: Dict[str, bool]
    rationale: str

@dataclass
class StrategyScope:
    task_envs: List[str]
    held_out_task_envs: List[str]
    algorithms: List[str]          # exact strings tested, never "all"
    frameworks: List[str]
    budget_range_episodes: Tuple[int, int]
    n_seeds_per_task: int
    excluded_agents: List[str]
    confound_notes: str

def record_candidate_scope(store, candidate, *, train_task_envs, held_out_task_envs,
                            algorithms, frameworks, budget_range_episodes,
                            n_seeds_per_task, excluded_agents=None, confound_notes="") -> StrategyScope: ...
```

## 6. Live adaptive runtime (`tensor_optix.runtime`)

```python
def establish_baseline(
    store: ExperimentStore, *, agent_factory, pipeline_factory, evaluator,
    task_env: str, algorithm: str, study_id: str, trial_steps: int,
    seed: int = 0, checkpoint_dir: Optional[str] = None,
) -> str:
    """Trains directly (no LoopController/strategy-op), records one
    ExperimentRecord with config['role']='deployed_baseline' and the
    real weights_path. Returns the experiment_id."""

def get_current_baseline_experiment_id(store, *, task_env, algorithm, study_id) -> Optional[str]:
    """Live lookup, never cached -- the deployed baseline can change at
    any time via promote.py's separate human-invoked path."""

def run_reevaluation(store, *, baseline_experiment_id, agent_factory, evaluator,
                      task_env, algorithm, study_id, eval_episodes: int = 5) -> float:
    """Loads the baseline's weights into a fresh agent, runs a real,
    inference-only rollout (no learn() calls), records one
    deployed_baseline_eval row, returns the real score."""
```

```python
class DeploymentObserver:
    def __init__(self, store, *, task_env, algorithm, study_id, scheduler: Optional[BackoffScheduler] = None):
        """Wraps BackoffScheduler (reused as-is, store-free, numpy-only --
        not reimplemented) with a per-re-evaluation-cycle calling
        convention instead of per-episode."""

    def observe(self, score: float) -> ObservationResult: ...
    def score_history(self, *, limit: int = 100) -> List[float]: ...
    def replay_history(self) -> None:
        """Reconstructs ledger-backed history and replays it through the
        scheduler -- makes Observer state durable across process restarts
        with zero new persistence code."""

@dataclass
class ObservationResult:
    cycle: int
    score: float
    state: LoopState
    is_degrading: bool
    degradation_fired: bool
    best_score: Optional[float]
```

```python
def propose_and_run(store, *, agent_factory, pipeline_factory, task_env, algorithm,
                     solve_threshold, source_task_envs, budget, trial_steps, seeds,
                     trial_timeout_s: float = 180.0) -> List[Tuple[StrategyCandidate, List[Dict]]]:
    """Ranks a fixed, agent-appropriate shortlist of StrategyOps via
    NanonetsSelector.propose() (never invents a new op), runs whichever
    receive nonzero allocation through StrategyTrialRunner."""

def evaluate_and_recommend(store, candidate, trial_results, *, task_env, algorithm,
                            trial_steps, seeds) -> PromotionVerdict:
    """Records scope, evaluates via a hardcoded approval_mode="human"
    PromotionPolicy, issues a durable recommendation event. Never calls
    approve_strategy."""

def issue_recommendation(store, candidate, verdict, *, experiment_id) -> None:
    """Appends one EventRecord(event_type="recommendation_issued") --
    an auditable record, not a vanishing print line."""
```

```python
@dataclass
class RuntimeConfig:
    task_env: str = "CartPole-v1"
    algorithm: str = "PPO"
    study_id: str = "live_runtime_cartpole_v1"
    solve_threshold: float = 475.0
    eval_episodes: int = 5
    cycle_sleep_s: float = 1.0
    research_cycle_every: int = 3
    experiment_budget_trials: int = 2
    experiment_trial_steps: int = 3_000
    experiment_seeds: Tuple[int, ...] = (0, 1, 2)
    max_cycles: int = 6
    source_task_envs: Tuple[str, ...] = ()
    # Deliberately NO approval_mode field -- no parameter path reaches
    # PromotionPolicy's approval_mode from here.

class LiveAdaptiveRuntime:
    def __init__(self, store: ExperimentStore, config: RuntimeConfig, *,
                 agent_factory, pipeline_factory): ...

    def run(self) -> Tuple[List[ObservationResult], List[CycleRecommendation]]:
        """Single-threaded sequential loop, bounded by max_cycles. Each
        cycle: re-evaluate -> observe -> maybe propose_and_run ->
        evaluate_and_recommend -> sleep. Raises RuntimeError if no
        deployed baseline exists yet (call establish_baseline first)."""

@dataclass
class CycleRecommendation:
    cycle: int
    candidate_id: str
    op_name: str
    automatable_criteria_passed: bool
    rationale: str
```

```python
def promote_candidate_to_baseline(
    store: ExperimentStore, candidate_id: str, *, reviewer: str,
    task_env: str, algorithm: str, study_id: str,
) -> str:
    """The ONLY function that can change the deployed baseline. Raises
    ValueError unless the candidate's registry status is already
    APPROVED -- refuses to act on anything not already human-reviewed.
    Never imported by live_runtime.py, experiment_stage.py, observer.py,
    or baseline.py. Picks the candidate's best-scoring recorded
    weights_path and writes a new deployed_baseline ExperimentRecord."""
```

See the guide for the full end-to-end wiring: [Learning-to-Learn Engine guide](../guides/learning-to-learn.md).

## Behaviour notes

- **Method evidence.** `StrategySelector` and `NanonetsSelector` count a ledger experiment as evidence for a method when `config["sampler"] == method` (trial-search runs) or `config["op_name"] == method` with `is_baseline` false (strategy-candidate runs). The live runtime's proposals over strategy ops therefore learn from earlier strategy trials. Untried methods are ranked last.
- **`CycleRecommendation.automatable_criteria_passed`** is `True` when every evidence-based criterion passed, ignoring `human_reviewed` (which the recommend-only loop never sets).
- **Trial workers** (`ConcurrentTrialScheduler`, `StrategyTrialRunner`) seed `torch`, `numpy` and `random` from the trial seed.
- **`CheckpointRollbackOp`** in a strategy trial rolls back to the run's current best checkpoint (it shares the run's `CheckpointRegistry`).
- **`OptimizationMethodChoiceOp`** switches at most once per stall; after the run improves again, the next stall advances to the next entry of `candidate_methods`.
- **`run_reevaluation`** evaluates epsilon-greedy agents greedily (`epsilon = 0`).

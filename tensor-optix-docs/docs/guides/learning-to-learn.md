# Learning-to-Learn Engine

Everything else in tensor-optix tunes one training run. The Learning-to-Learn Engine is a layer on top that learns *across* runs: it records every experiment with enough structure to compare reliably, lets you search and discover better training procedures, and — for a long-running deployed agent — continuously proposes evidence-based interventions for a human to review. It is built in six layers, each composing over the one before it, and each layer can be used on its own.

No layer here uses generative AI or runs arbitrary generated code. Strategy discovery works over a small, closed set of hand-written typed operations, reconstructed only from a fixed registry — never `eval`/`exec`.

## 1. Experiment ledger

The foundation every other layer reads from and writes to: a SQLite-backed `ExperimentStore` that records each trial's provenance (code revision, library versions, hostname), lifecycle state, and scores.

```python
from tensor_optix.ledger import ExperimentStore, ExperimentState, EvaluationStatus

store = ExperimentStore("./experiments.sqlite3")
record = store.create_experiment(
    task_env="CartPole-v1", algorithm="PPO", framework="torch",
    seed=0, checkpoint_dir="./ckpt", solve_threshold=475.0,
)
store.start_experiment(record.experiment_id)
store.record_progress(record.experiment_id, best_primary_score=480.0)
store.finalize_experiment(
    record.experiment_id, state=ExperimentState.COMPLETED,
    evaluation_status=EvaluationStatus.VALIDATED,
)
```

Every write is idempotent on `experiment_id`, and every method opens a short, single-statement transaction — nothing here holds a lock for the duration of training. The schema also carries three tables for strategy discovery (below): `strategy_candidates`, `strategy_evaluations`, `strategy_registry`.

## 2. Concurrent trial scheduler

`ConcurrentTrialScheduler` runs multiple Optuna trials as real, isolated OS processes — the parent process is the sole owner of the Optuna `Study` and the ledger; workers never touch either directly, they just train and report scores back over a queue.

```python
from tensor_optix.scheduler import ConcurrentTrialScheduler

scheduler = ConcurrentTrialScheduler(
    agent_factory=make_agent, pipeline_factory=make_pipeline,
    param_space={"learning_rate": ("log_float", 1e-4, 3e-3)},
    n_trials=20, n_workers=4, trial_steps=50_000,
    ledger_store=store, task_env="CartPole-v1", algorithm="PPO",
    solve_threshold=475.0,
)
best_params, best_score = scheduler.run()
```

`TrialOrchestrator` (see [Run trial-level search](trial-search.md)) remains the simpler, correct tool when you don't need real process-level concurrency.

## 3. Baseline selection

`StrategySelector` is the non-learned baseline: given a task's ledger history, it allocates a trial budget across competing methods by ranking their confidence-interval upper bound (`hi95`), with a minimum exploration share for any method that hasn't been tried yet.

```python
from tensor_optix.selection import StrategySelector

baseline = StrategySelector(store)
decision = baseline.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)
print(decision.rationale)
```

This is also the fallback every higher layer uses when it doesn't have enough evidence to do better.

## 4. Cross-task transfer (Nanonets)

Tuning a new task from scratch throws away everything you already know about similar tasks. `NanonetsSelector` pools a source task's ledger evidence into a target task's estimate — but only after a conservative abstention gate confirms the source is actually comparable.

The gate compares each task's scores *normalized against its own solve threshold* (`(score - threshold) / abs(threshold)`), not raw scores — comparing raw scores across tasks with different scales and signs (e.g. CartPole's ~0–500 against Acrobot's ~–500–0) is a real, demonstrated bug class, not a hypothetical one: a positive-scale task's data can numerically pass a percentile check against a negative-scale target purely by coincidence. A source passes only if its normalized median is at or above the target's own normalized 95th percentile.

```python
from tensor_optix.nanonets import NanonetsSelector

selector = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
decision = selector.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)
# rationale states explicitly whether it pooled cross-task evidence or
# fell back to the plain baseline because nothing passed the gate.
```

The pooling itself (precision-weighted shrinkage, not a Gaussian Process or RGPE — deliberately simple, since multi-task GP correlation estimation is unreliable at the sample sizes real experiments produce) also normalizes before combining estimates, then de-normalizes back to the target's own raw units for display — pooling must happen in the same comparable space as the gate, or the same scale-mixing problem the gate exists to prevent resurfaces in the pooled number itself.

`source_task_envs` is always supplied explicitly by the caller, never auto-discovered — for leave-one-task-out validation, the caller must already know which tasks are training sources versus held out.

## 5. Strategy discovery

Hyperparameter search tunes values; strategy discovery searches *procedures* — a small, closed grammar of typed training-time interventions:

| Operation | What it does |
|---|---|
| `LearningRateTrendOp` | Adjusts learning rate based on a measured score trend |
| `CheckpointRollbackOp` | Rolls back to the best checkpoint on a validation condition |
| `ExplorationBoundsOp` | Adjusts an algorithm's exploration knob (`entropy_coef`, `epsilon`, TD3's target-smoothing noise) within bounds |
| `OptimizationMethodChoiceOp` | Switches which online tuning `BaseOptimizer` drives the run |
| `BudgetAllocationOp` | Reallocates ensemble budget toward a promising policy variant |
| `ScheduleChangeOp` | Overrides a schedule (DQN's `epsilon_decay`, RND's eta) |
| `PolicyVariantSelectionOp` | Boosts or prunes ensemble members by evidence |

Every operation wraps an already-existing mutation primitive (`agent.set_hyperparams`, `checkpoint_registry.load_best`, `PolicyManager.boost`/`prune`) as a `LoopCallback`. An operation that doesn't apply to a given agent (e.g. exploration bounds on `TorchSACAgent`, whose temperature is learned rather than a static hyperparameter) returns `None` from `attach()` and is silently skipped — self-exclusion, not a crash.

```python
from tensor_optix.strategy import LearningRateTrendOp, StrategyCandidate, StrategyTrialRunner

op = LearningRateTrendOp(window=5, increase_factor=1.2, decrease_factor=0.5)
candidate = StrategyCandidate.from_op(op)   # content-hashed, versioned, serializable

runner = StrategyTrialRunner(
    agent_factory=make_agent, pipeline_factory=make_pipeline,
    trial_steps=50_000, ledger_store=store, task_env="CartPole-v1",
    algorithm="PPO", solve_threshold=475.0,
)
results = runner.run_candidate(candidate, seeds=[0, 1, 2])
```

Each seed runs a **paired** candidate-vs-baseline trial (same seed/task/budget, baseline has no strategy callback attached) so the comparison is honest, not candidate-vs-stale-history.

### Comparing and promoting

`compare_candidate` gates on the *pessimistic* end of the outcome-delta confidence interval (`lo95 > margin`), not the mean, and hard-requires at least 3 distinct seeds before it can ever pass — the same conservative discipline as Nanonets' abstention gate, applied to "is this strategy's benefit trustworthy" instead of "is this source task comparable."

`PromotionPolicy` checks six independent, inspectable criteria:

```python
from tensor_optix.strategy import PromotionPolicy, record_candidate_scope

record_candidate_scope(
    store, candidate, train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"],
    algorithms=["PPO"], frameworks=["torch"], budget_range_episodes=(50_000, 50_000),
    n_seeds_per_task=3,
)

policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1, approval_mode="human")
verdict = policy.evaluate(
    candidate_id=candidate.candidate_id,
    train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"],
)
for name, passed in verdict.criteria.items():
    print(name, passed)
```

`beats_baseline_margin`, `repeats_across_seeds`, `generalizes_held_out`, `meets_compute_constraints`, and `reproducible` are all computed from ledger evidence. The sixth, `human_reviewed`, is different: in the default `approval_mode="human"`, `evaluate()` never sets it itself — it can only become `True` after a separate, explicit `store.approve_strategy(candidate_id, reviewer=...)` call made by an actual human. `approval_mode="auto"` exists for contexts that explicitly want full automation (it still requires all five automatable criteria to pass, and leaves a traceable `reviewer="auto:PromotionPolicy"` record — never a silent self-approval) — but the live adaptive runtime below never uses it.

## 6. Live adaptive runtime

The final layer wires everything above into the proposal's observe → detect → propose → experiment → evaluate → recommend loop, run continuously against a long-running agent that stands in for a deployed model.

```python
from tensor_optix.runtime import LiveAdaptiveRuntime, RuntimeConfig, establish_baseline
from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator

baseline_id = establish_baseline(
    store, agent_factory=make_agent, pipeline_factory=make_pipeline,
    evaluator=TorchEvaluator(), task_env="CartPole-v1", algorithm="PPO",
    study_id="my_deployment", trial_steps=20_000,
)

config = RuntimeConfig(task_env="CartPole-v1", algorithm="PPO", study_id="my_deployment", max_cycles=20)
runtime = LiveAdaptiveRuntime(store, config, agent_factory=make_agent, pipeline_factory=make_pipeline)
observations, recommendations = runtime.run()
```

Each cycle: re-evaluates the deployed agent with a short, real, inference-only rollout (standing in for live traffic); feeds the score through a `DeploymentObserver` (a thin wrapper around the existing `BackoffScheduler` trend/degradation state machine — reused exactly as-is, not reimplemented); and, on degradation or a scheduled research cycle, consults `NanonetsSelector` to rank a fixed shortlist of strategy operations, runs the winners through `StrategyTrialRunner`, and issues a durable recommendation.

**The loop is recommend-only by construction, not by convention.** `PromotionPolicy` is hardcoded inside the runtime with `approval_mode="human"` — there is no field on `RuntimeConfig`, no CLI flag, no environment variable that reaches that value. Recommendations are written as auditable ledger `EventRecord`s, never just a line of stdout that scrolls away. Nothing in the loop ever calls `store.approve_strategy`.

Promoting a candidate to actually become the deployed baseline requires two separate, explicit actions outside the loop:

```python
from tensor_optix.runtime import promote_candidate_to_baseline

# 1. A human reviews the recommendation and approves it.
store.approve_strategy(candidate_id, reviewer="your-name")

# 2. A separate, explicit call promotes the approved candidate's best weights.
new_baseline_id = promote_candidate_to_baseline(
    store, candidate_id, reviewer="your-name",
    task_env="CartPole-v1", algorithm="PPO", study_id="my_deployment",
)
```

`promote_candidate_to_baseline` refuses to act — raises `ValueError` — unless the candidate was already approved. It is never imported by any of the runtime's own loop modules.

## Honest results so far

Real validation runs against this engine (CartPole-v1/Acrobot-v1/Pendulum-v1 with PPO and TD3) have mostly produced **negative or inconclusive results at the tight, cheap probe budgets used so far** — a few thousand training steps and 2–3 seeds, deliberately small to prove the mechanism works before spending a larger budget. Most candidate strategies and cross-task transfer attempts correctly triggered fallback or failed the promotion margin, rather than being accepted on noisy, low-powered evidence. That is the conservative gates working as designed, not a disappointing outcome to downplay — the proposal's own exit criteria explicitly accept "a documented negative result that informs the next research direction" as a legitimate deliverable, and this project treats it that way throughout.

## Reference

Full constructor and method signatures: [Learning-to-Learn Engine reference](../reference/learning-to-learn.md).

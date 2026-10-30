"""ConcurrentTrialScheduler — a bounded, multi-process trial scheduler.

Runs multiple Optuna trials concurrently as real OS processes (mirroring
tensor_optix/distributed/async_learner.py's proven mp.Process pattern),
while the parent process remains the sole owner of the optuna.Study and
the ExperimentStore. Workers never touch either — they receive a plain
sampled-params dict and report back over queues (see
tensor_optix/scheduler/worker.py).

TrialOrchestrator (tensor_optix/orchestrator.py) is untouched and remains
the correct tool for simple sequential trial search. This class is for
when genuine process-level concurrency is wanted.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.orchestrator import ParamSpace, _sample_params
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore

from ._mp_common import mp_context, shutdown_worker
from .budget import BudgetTracker
from .dedup import fingerprint
from .isolation import trial_checkpoint_dir, trial_seed
from .retry import classify as classify_retry
from .worker import _trial_worker

logger = logging.getLogger(__name__)


@dataclass
class _ActiveTrial:
    trial: "optuna.Trial"
    process: Any
    cancel_event: Any
    result_queue: Any
    report_queue: Any
    params: Dict[str, Any]
    experiment_id: Optional[str]
    attempt: int = 0


class ConcurrentTrialScheduler:
    """Runs up to `n_workers` Optuna trials concurrently as real OS processes.

    device_placement_fn is an explicit, documented hook for future
    heterogeneous-hardware placement; it defaults to always "cpu" since
    this environment has no GPU to develop or test against. Coordinating
    fair resource allocation ACROSS multiple concurrent studies/scheduler
    instances is explicitly out of scope — each instance only respects its
    own n_workers.
    """

    def __init__(
        self,
        agent_factory: Callable[[Dict[str, Any]], BaseAgent],
        pipeline_factory: Callable[[], BasePipeline],
        param_space: Optional[ParamSpace] = None,
        n_trials: int = 20,
        trial_steps: int = 50_000,
        direction: str = "maximize",
        n_startup_trials: int = 10,
        pruner_warmup_steps: int = 5,
        pruner_interval: int = 1,
        rloptimizer_kwargs: Optional[Dict[str, Any]] = None,
        val_pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
        checkpoint_score_fn: Optional[Callable[[BaseAgent], float]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler: Optional["optuna.samplers.BaseSampler"] = None,
        n_workers: int = 2,
        max_wall_clock_s: Optional[float] = None,
        trial_timeout_s: Optional[float] = None,
        max_retries: int = 2,
        ledger_store: Optional[ExperimentStore] = None,
        task_env: Optional[str] = None,
        algorithm: Optional[str] = None,
        framework: str = "torch",
        solve_threshold: Optional[float] = None,
        base_seed: int = 0,
        poll_interval_s: float = 0.5,
        device_placement_fn: Optional[Callable[[int], str]] = None,
    ):
        self._agent_factory = agent_factory
        self._pipeline_factory = pipeline_factory

        if param_space is None:
            try:
                probe_agent = agent_factory({})
                raw_bounds = getattr(probe_agent, "default_param_bounds", {})
                log_ps = set(getattr(probe_agent, "default_log_params", []))
                param_space = {
                    k: ("log_float" if k in log_ps else "float", lo, hi)
                    for k, (lo, hi) in raw_bounds.items()
                }
            except Exception:
                param_space = {}
        self._param_space = param_space

        self._n_trials = n_trials
        self._trial_steps = trial_steps
        self._direction = direction
        self._rloptimizer_kwargs = rloptimizer_kwargs or {}
        self._val_pipeline_factory = val_pipeline_factory
        self._checkpoint_score_fn = checkpoint_score_fn
        self._study_name = study_name or "tensor_optix_concurrent_study"
        self._n_workers = max(1, n_workers)
        self._max_retries = max_retries
        self._ledger_store = ledger_store
        self._task_env = task_env or ""
        self._algorithm = algorithm or ""
        self._framework = framework
        self._solve_threshold = solve_threshold
        self._base_seed = base_seed
        self._poll_interval_s = poll_interval_s
        self._device_placement_fn = device_placement_fn or (lambda worker_id: "cpu")

        self._sampler = sampler or TPESampler(n_startup_trials=n_startup_trials)
        self._study = optuna.create_study(
            study_name=self._study_name,
            direction=direction,
            storage=storage,
            sampler=self._sampler,
            pruner=MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=pruner_warmup_steps,
                interval_steps=pruner_interval,
            ),
            load_if_exists=(storage is not None),
        )

        self._budget = BudgetTracker(max_wall_clock_s=max_wall_clock_s, trial_timeout_s=trial_timeout_s)
        self._run_ckpt_dir: Optional[str] = None
        self._active: Dict[int, _ActiveTrial] = {}
        self._active_procs: List[Any] = []
        self._told_count = 0
        self._dispatched_count = 0
        self._seen_fingerprints: Dict[str, float] = {}  # fingerprint -> score
        self.max_concurrent_observed = 0  # high-water mark, for tests/introspection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Dict[str, Any], float]:
        ctx = mp_context()
        self._run_ckpt_dir = tempfile.mkdtemp(prefix="to_concurrent_")

        try:
            while True:
                self._dispatch_new_trials(ctx)
                self._poll_reports()
                self._poll_results(ctx)
                self._enforce_timeouts(ctx)

                if self._told_count >= self._n_trials and not self._active:
                    break
                if self._budget.global_budget_exhausted() and not self._active and self._dispatched_count >= self._told_count:
                    break
                time.sleep(self._poll_interval_s)
        finally:
            self._shutdown_all(ctx)

        try:
            best = self._study.best_trial
        except ValueError:
            direction_worst = float("-inf") if self._direction == "maximize" else float("inf")
            return {}, direction_worst
        return best.params, best.value

    @property
    def study(self) -> "optuna.Study":
        return self._study

    # ------------------------------------------------------------------
    # Scheduling loop internals
    # ------------------------------------------------------------------

    def _dispatch_new_trials(self, ctx) -> None:
        while (
            len(self._active) < self._n_workers
            and self._dispatched_count < self._n_trials
            and not self._budget.global_budget_exhausted()
        ):
            trial = self._study.ask()
            params = _sample_params(trial, self._param_space)

            fp = fingerprint(params)
            if fp in self._seen_fingerprints:
                # Duplicate sampled params — tell Optuna the earlier result
                # without re-running training.
                self._study.tell(trial, self._seen_fingerprints[fp], skip_if_finished=True)
                self._told_count += 1
                self._dispatched_count += 1
                continue

            self._dispatch_trial(ctx, trial, params)
            self._dispatched_count += 1
            self.max_concurrent_observed = max(self.max_concurrent_observed, len(self._active))

    def _dispatch_trial(self, ctx, trial, params: Dict[str, Any], attempt: int = 0) -> None:
        trial_number = trial.number
        seed = trial_seed(self._base_seed, trial_number, retry_attempt=attempt)
        ckpt_dir = trial_checkpoint_dir(self._run_ckpt_dir, trial_number)

        experiment_id = None
        if self._ledger_store is not None:
            experiment_id = f"{self._study_name}-trial-{trial_number}"
            record = self._ledger_store.create_experiment(
                experiment_id=experiment_id,
                task_env=self._task_env,
                algorithm=self._algorithm,
                framework=self._framework,
                solve_threshold=self._solve_threshold,
                seed=seed,
                checkpoint_dir=ckpt_dir,
                config={"params": params, "trial_number": trial_number},
                study_id=self._study_name,
            )
            self._ledger_store.start_experiment(experiment_id)

        report_q = ctx.Queue()
        result_q = ctx.Queue()
        cancel_event = ctx.Event()

        proc = ctx.Process(
            target=_trial_worker,
            kwargs=dict(
                trial_number=trial_number,
                params=params,
                agent_factory=self._agent_factory,
                pipeline_factory=self._pipeline_factory,
                val_pipeline_factory=self._val_pipeline_factory,
                rloptimizer_kwargs=self._rloptimizer_kwargs,
                checkpoint_score_fn=self._checkpoint_score_fn,
                trial_steps=self._trial_steps,
                ckpt_dir=ckpt_dir,
                seed=seed,
                report_queue=report_q,
                cancel_event=cancel_event,
                result_queue=result_q,
            ),
            daemon=True,
            name=f"trial-worker-{trial_number}",
        )
        proc.start()
        self._active_procs.append(proc)
        self._budget.start_trial(trial_number)

        self._active[trial_number] = _ActiveTrial(
            trial=trial,
            process=proc,
            cancel_event=cancel_event,
            result_queue=result_q,
            report_queue=report_q,
            params=params,
            experiment_id=experiment_id,
            attempt=attempt,
        )

    def _poll_reports(self) -> None:
        for trial_number, active in list(self._active.items()):
            drained = 0
            while drained < 32:
                try:
                    _tn, episode_id, score = active.report_queue.get_nowait()
                except Exception:
                    break
                drained += 1
                try:
                    active.trial.report(score, step=episode_id)
                except Exception:
                    pass
                if self._ledger_store is not None and active.experiment_id is not None:
                    self._ledger_store.record_progress(active.experiment_id, best_primary_score=score)
                if active.trial.should_prune():
                    active.cancel_event.set()

    def _poll_results(self, ctx) -> None:
        for trial_number, active in list(self._active.items()):
            try:
                result = active.result_queue.get_nowait()
            except Exception:
                continue
            self._handle_result(ctx, trial_number, active, result)

    def _enforce_timeouts(self, ctx) -> None:
        for trial_number, active in list(self._active.items()):
            if self._budget.trial_timed_out(trial_number):
                active.cancel_event.set()
                self._shutdown_worker(active.process)
                self._finalize_trial(
                    trial_number, active, result=None, state=ExperimentState.TIMED_OUT,
                    evaluation_status=EvaluationStatus.INCONCLUSIVE, score=None,
                    error="trial exceeded trial_timeout_s",
                )
                self._told_count += 1
                del self._active[trial_number]

    def _handle_result(self, ctx, trial_number: int, active: _ActiveTrial, result: Dict[str, Any]) -> None:
        status = result.get("status")

        if status == "failed":
            decision = classify_retry(
                result_received=True, status=status,
                max_retries=self._max_retries, attempt=active.attempt,
            )
            if decision.should_retry:
                self._shutdown_worker(active.process)
                del self._active[trial_number]
                self._dispatch_trial(ctx, active.trial, active.params, attempt=active.attempt + 1)
                return

            self._finalize_trial(
                trial_number, active, result=result, state=ExperimentState.FAILED,
                evaluation_status=EvaluationStatus.INCONCLUSIVE, score=None,
                error=result.get("error"),
            )
            self._told_count += 1
            del self._active[trial_number]
            return

        if status == "cancelled":
            self._finalize_trial(
                trial_number, active, result=result, state=ExperimentState.TIMED_OUT,
                evaluation_status=EvaluationStatus.INCONCLUSIVE, score=result.get("score"),
                error=None,
            )
            self._told_count += 1
            del self._active[trial_number]
            return

        # completed
        score = result.get("score")
        self._seen_fingerprints[fingerprint(active.params)] = score if score is not None else float("-inf")
        self._finalize_trial(
            trial_number, active, result=result, state=ExperimentState.COMPLETED,
            evaluation_status=EvaluationStatus.PRELIMINARY, score=score, error=None,
        )
        self._told_count += 1
        del self._active[trial_number]

    def _finalize_trial(
        self, trial_number: int, active: _ActiveTrial, *,
        result: Optional[Dict[str, Any]], state: ExperimentState,
        evaluation_status: EvaluationStatus, score: Optional[float], error: Optional[str],
    ) -> None:
        try:
            if score is not None:
                self._study.tell(active.trial, score, skip_if_finished=True)
            else:
                self._study.tell(active.trial, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
        except Exception as exc:
            logger.warning("study.tell failed for trial %d: %s", trial_number, exc)

        if self._ledger_store is not None and active.experiment_id is not None:
            final_metrics = {"trial_number": trial_number}
            if score is not None:
                final_metrics["score"] = score
            if score is not None:
                self._ledger_store.record_progress(active.experiment_id, best_primary_score=score)
            self._ledger_store.finalize_experiment(
                active.experiment_id, state=state, evaluation_status=evaluation_status,
                final_metrics=final_metrics, error_message=error,
            )

    def _shutdown_worker(self, proc) -> Tuple[Optional[int], bool]:
        return shutdown_worker(proc, self._active_procs)

    def _shutdown_all(self, ctx) -> None:
        for trial_number, active in list(self._active.items()):
            active.cancel_event.set()
        for trial_number, active in list(self._active.items()):
            self._shutdown_worker(active.process)
        self._active.clear()
        self._active_procs = []

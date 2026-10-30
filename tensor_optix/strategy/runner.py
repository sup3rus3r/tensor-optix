"""StrategyTrialRunner -- runs a fixed strategy candidate against real
tasks, as paired (candidate, baseline) trials, writing results to the
experiment ledger.

Deliberately bypasses ConcurrentTrialScheduler's Optuna ask/tell
machinery: the grammar is a small, closed, enumerable set, not a
continuous space worth modeling with TPE. Reuses the proven mp.Process
spawn/teardown pattern via scheduler._mp_common, extracted for exactly
this reuse.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.scheduler._mp_common import mp_context, shutdown_worker

from .candidate import StrategyCandidate
from .worker import _strategy_trial_worker

logger = logging.getLogger(__name__)


@dataclass
class _ActiveStrategyTrial:
    process: Any
    cancel_event: Any
    result_queue: Any
    report_queue: Any
    experiment_id: Optional[str]
    is_baseline: bool


class StrategyTrialRunner:
    def __init__(
        self,
        *,
        agent_factory: Callable[[Dict[str, Any]], BaseAgent],
        pipeline_factory: Callable[[], BasePipeline],
        base_params: Optional[Dict[str, Any]] = None,
        rloptimizer_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_score_fn: Optional[Callable[[BaseAgent], float]] = None,
        trial_steps: int = 50_000,
        ledger_store: Optional[ExperimentStore] = None,
        task_env: str = "",
        algorithm: str = "",
        framework: str = "torch",
        solve_threshold: Optional[float] = None,
        trial_timeout_s: Optional[float] = None,
        poll_interval_s: float = 0.5,
    ):
        self._agent_factory = agent_factory
        self._pipeline_factory = pipeline_factory
        self._base_params = base_params or {}
        self._rloptimizer_kwargs = rloptimizer_kwargs or {}
        self._checkpoint_score_fn = checkpoint_score_fn
        self._trial_steps = trial_steps
        self._ledger_store = ledger_store
        self._task_env = task_env
        self._algorithm = algorithm
        self._framework = framework
        self._solve_threshold = solve_threshold
        self._trial_timeout_s = trial_timeout_s
        self._poll_interval_s = poll_interval_s

    def run_candidate(self, candidate: StrategyCandidate, *, seeds: List[int]) -> List[Dict[str, Any]]:
        """For each seed, dispatch a paired (candidate, baseline) trial.
        Returns a list of {seed, candidate_score, baseline_score,
        outcome_delta, candidate_experiment_id, baseline_experiment_id}
        dicts -- one per seed. Writes ExperimentRecords and
        StrategyEvaluationRecords to the ledger if one was supplied.
        """
        ctx = mp_context()
        run_ckpt_dir = tempfile.mkdtemp(prefix="to_strategy_")
        results: List[Dict[str, Any]] = []

        for seed in seeds:
            cand_result = self._run_one(
                ctx, run_ckpt_dir, candidate=candidate, seed=seed, is_baseline=False,
            )
            base_result = self._run_one(
                ctx, run_ckpt_dir, candidate=candidate, seed=seed, is_baseline=True,
            )

            cand_score = cand_result.get("score")
            base_score = base_result.get("score")
            outcome_delta = (
                cand_score - base_score if cand_score is not None and base_score is not None else None
            )

            if self._ledger_store is not None:
                self._ledger_store.record_strategy_evaluation(
                    candidate_id=candidate.candidate_id,
                    experiment_id=cand_result["experiment_id"],
                    baseline_experiment_id=base_result["experiment_id"],
                    task_env=self._task_env, algorithm=self._algorithm,
                    framework=self._framework, seed=seed, outcome_delta=outcome_delta,
                )

            results.append({
                "seed": seed, "candidate_score": cand_score, "baseline_score": base_score,
                "outcome_delta": outcome_delta,
                "candidate_experiment_id": cand_result["experiment_id"],
                "baseline_experiment_id": base_result["experiment_id"],
                "candidate_status": cand_result["status"], "baseline_status": base_result["status"],
            })

        return results

    def _run_one(
        self, ctx, run_ckpt_dir: str, *, candidate: StrategyCandidate, seed: int, is_baseline: bool,
    ) -> Dict[str, Any]:
        label = "baseline" if is_baseline else "candidate"
        ckpt_dir = f"{run_ckpt_dir}/{label}_seed{seed}"

        experiment_id = None
        if self._ledger_store is not None:
            task_slug = self._task_env.replace("-", "_").replace(" ", "_") or "unknown_task"
            experiment_id = f"strategy_{candidate.candidate_id}_{task_slug}_{label}_seed{seed}"
            algorithm = self._algorithm if is_baseline else f"{self._algorithm}+{candidate.op_name}"
            record = self._ledger_store.create_experiment(
                experiment_id=experiment_id, task_env=self._task_env, algorithm=algorithm,
                framework=self._framework, seed=seed, checkpoint_dir=ckpt_dir,
                solve_threshold=self._solve_threshold,
                config={"candidate_id": candidate.candidate_id, "op_name": candidate.op_name,
                        "is_baseline": is_baseline, "sampler": "TPESampler"},
            )
            experiment_id = record.experiment_id
            self._ledger_store.start_experiment(experiment_id)

        report_q = ctx.Queue()
        result_q = ctx.Queue()
        cancel_event = ctx.Event()

        proc = ctx.Process(
            target=_strategy_trial_worker,
            kwargs=dict(
                trial_index=seed,
                candidate_params=None if is_baseline else candidate.params,
                op_name=None if is_baseline else candidate.op_name,
                agent_factory=self._agent_factory,
                pipeline_factory=self._pipeline_factory,
                base_params=self._base_params,
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
            name=f"strategy-trial-{label}-seed{seed}",
        )
        proc.start()

        start = time.monotonic()
        result = None
        while True:
            try:
                result = result_q.get_nowait()
                break
            except Exception:
                pass
            if self._trial_timeout_s is not None and time.monotonic() - start >= self._trial_timeout_s:
                cancel_event.set()
                shutdown_worker(proc)
                result = {"trial_index": seed, "status": "timed_out", "score": None, "error": "trial_timeout_s exceeded"}
                break
            if not proc.is_alive():
                # Process exited without putting a result -- treat as failed.
                result = {"trial_index": seed, "status": "failed", "score": None, "error": "worker exited without a result"}
                break
            time.sleep(self._poll_interval_s)

        shutdown_worker(proc)

        score = result.get("score")
        status = result.get("status", "failed")

        if self._ledger_store is not None and experiment_id is not None:
            state_map = {
                "completed": ExperimentState.COMPLETED,
                "cancelled": ExperimentState.TIMED_OUT,
                "timed_out": ExperimentState.TIMED_OUT,
                "skipped": ExperimentState.CANCELLED,
                "failed": ExperimentState.FAILED,
            }
            eval_status = EvaluationStatus.PRELIMINARY if status == "completed" else EvaluationStatus.INCONCLUSIVE
            if score is not None:
                self._ledger_store.record_progress(experiment_id, best_primary_score=score)
            self._ledger_store.finalize_experiment(
                experiment_id, state=state_map.get(status, ExperimentState.FAILED),
                evaluation_status=eval_status,
                final_metrics={"trial_index": float(seed)},
                error_message=result.get("error") or result.get("reason"),
            )

        return {"experiment_id": experiment_id, "score": score, "status": status}

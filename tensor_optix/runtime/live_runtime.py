"""LiveAdaptiveRuntime -- the controller loop tying together the
Observer, Experiment, and Evaluate/Recommend stages into the proposal's
sec 19.4 feedback loop: observe -> detect -> propose -> experiment ->
evaluate -> record -> recommend -> repeat. Recommend-only: never
auto-promotes a candidate to become the deployed baseline.

Single-threaded sequential loop, not concurrent -- StrategyTrialRunner
already spawns its own mp.Process workers for actual training trials,
so "the slow stage runs separately" is already true at the trial level
without the controller itself needing concurrency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.strategy import PromotionPolicy

from .baseline import get_current_baseline_experiment_id, run_reevaluation
from .experiment_stage import evaluate_and_recommend, propose_and_run
from .observer import DeploymentObserver, ObservationResult
from tensor_optix.utils.console import safe_print


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
    # Deliberately NO approval_mode field here -- there is no parameter
    # path from config to PromotionPolicy's approval_mode. See
    # LiveAdaptiveRuntime.__init__, which hardcodes "human".


@dataclass
class CycleRecommendation:
    cycle: int
    candidate_id: str
    op_name: str
    automatable_criteria_passed: bool
    rationale: str


class LiveAdaptiveRuntime:
    def __init__(
        self,
        store: ExperimentStore,
        config: RuntimeConfig,
        *,
        agent_factory: Callable[[Any], Any],
        pipeline_factory: Callable[[], Any],
    ):
        self._store = store
        self._cfg = config
        self._agent_factory = agent_factory
        self._pipeline_factory = pipeline_factory
        self._observer = DeploymentObserver(
            store, task_env=config.task_env, algorithm=config.algorithm, study_id=config.study_id,
        )
        # Safety-critical: evaluate_and_recommend() (experiment_stage.py)
        # hardcodes PromotionPolicy(..., approval_mode="human") itself --
        # there is no parameter path from RuntimeConfig/CLI/env to that
        # value anywhere in this class or its config.

    def run(self) -> Tuple[List[ObservationResult], List[CycleRecommendation]]:
        baseline_id = get_current_baseline_experiment_id(
            self._store, task_env=self._cfg.task_env, algorithm=self._cfg.algorithm, study_id=self._cfg.study_id,
        )
        if baseline_id is None:
            raise RuntimeError(
                f"No deployed baseline found for task_env={self._cfg.task_env!r}, "
                f"algorithm={self._cfg.algorithm!r}, study_id={self._cfg.study_id!r}. "
                "Call runtime.baseline.establish_baseline() first."
            )

        observations: List[ObservationResult] = []
        recommendations: List[CycleRecommendation] = []

        for cycle in range(1, self._cfg.max_cycles + 1):
            score = run_reevaluation(
                self._store, baseline_experiment_id=baseline_id, agent_factory=self._agent_factory,
                evaluator=self._make_evaluator(), task_env=self._cfg.task_env,
                algorithm=self._cfg.algorithm, study_id=self._cfg.study_id, eval_episodes=self._cfg.eval_episodes,
            )
            obs = self._observer.observe(score)
            observations.append(obs)
            safe_print(f"[cycle {cycle}] score={score:.1f} state={obs.state.name} degrading={obs.is_degrading}")

            should_trigger = obs.degradation_fired or (cycle % self._cfg.research_cycle_every == 0)
            if should_trigger:
                outcomes = propose_and_run(
                    self._store, agent_factory=self._agent_factory, pipeline_factory=self._pipeline_factory,
                    task_env=self._cfg.task_env, algorithm=self._cfg.algorithm,
                    solve_threshold=self._cfg.solve_threshold, source_task_envs=list(self._cfg.source_task_envs),
                    budget=self._cfg.experiment_budget_trials, trial_steps=self._cfg.experiment_trial_steps,
                    seeds=list(self._cfg.experiment_seeds),
                )
                for candidate, trial_results in outcomes:
                    verdict = evaluate_and_recommend(
                        self._store, candidate, trial_results, task_env=self._cfg.task_env,
                        algorithm=self._cfg.algorithm, trial_steps=self._cfg.experiment_trial_steps,
                        seeds=list(self._cfg.experiment_seeds),
                    )
                    # verdict.approved also requires human_reviewed, which this
                    # recommend-only loop never sets — so it is always False
                    # here. The flag reports the evidence-based criteria only.
                    automatable_pass = all(
                        passed for name, passed in verdict.criteria.items()
                        if name != "human_reviewed"
                    )
                    recommendations.append(CycleRecommendation(
                        cycle=cycle, candidate_id=candidate.candidate_id, op_name=candidate.op_name,
                        automatable_criteria_passed=automatable_pass, rationale=verdict.rationale,
                    ))
                    safe_print(f"  recommendation {candidate.candidate_id[:8]} ({candidate.op_name}): "
                          f"automatable_pass={automatable_pass} -- {verdict.rationale}")

            if cycle < self._cfg.max_cycles:
                time.sleep(self._cfg.cycle_sleep_s)

        return observations, recommendations

    @staticmethod
    def _make_evaluator():
        from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
        return TorchEvaluator()

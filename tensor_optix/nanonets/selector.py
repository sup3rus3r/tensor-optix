"""NanonetsSelector -- Phase 4 cross-task statistical pooling.

This is a fully separate code path from tensor_optix.selection.StrategySelector,
composing over it rather than editing or subclassing it. StrategySelector's
propose() still raises NotImplementedError on a non-None task_features
argument (tensor_optix/selection/selector.py lines 90-92) -- that guard is
left untouched in this pass. Wiring Nanonets' output back through
StrategySelector.propose(task_features=...) is a plausible v2 ergonomic
improvement once Nanonets has validated evidence behind it; doing it now
would mean editing Phase 3's frozen, tested contract to serve a v1 that
hasn't yet earned that integration. StrategySelector is used here only for
the fallback path, called directly and unchanged.

source_task_envs is supplied explicitly by the caller rather than
auto-discovered from the ledger (no list_distinct_task_envs() exists on
ExperimentStore, and none is added here) -- for leave-one-task-out
validation, the caller must already know exactly which task_envs are
training vs. held-out, so auto-discovery would work against that
discipline rather than help it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tensor_optix.ledger.records import ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.selection.selector import SelectionDecision, StrategySelector

from .abstention import MIN_TARGET_N, normalize_scores, passes_percentile_gate, similarity_weight
from .features import TaskFeatures, extract_task_features
from .pooling import PooledResult, pool_estimates
from .smalln import SmallNInterval


@dataclass
class PooledEstimate:
    method: str
    pooled_mean: float
    pooled_interval: SmallNInterval
    sources_used: List[str]
    sources_gated_out: List[str]
    fallback_triggered: bool


def _denormalize(value: float, threshold: float) -> float:
    """Inverse of abstention.normalize_score -- maps a normalized
    'fraction of solve_threshold magnitude above/below solved' value
    back to the target task's own raw score units."""
    return value * abs(threshold) + threshold


class NanonetsSelector:
    def __init__(self, store: ExperimentStore, *, source_task_envs: List[str]):
        self._store = store
        self._baseline = StrategySelector(store)
        self._source_task_envs = list(source_task_envs)

    def task_features(self, task_env: str) -> TaskFeatures:
        return extract_task_features(self._store, task_env=task_env)

    def _scores_for(self, task_env: str, method: str) -> List[float]:
        records = self._store.query_experiments(task_env=task_env, limit=1000)
        matching = [r for r in records if r.config.get("sampler") == method]
        completed = [r for r in matching if r.state == ExperimentState.COMPLETED]
        return [r.best_primary_score for r in completed if r.best_primary_score is not None]

    def _solve_threshold_for(self, task_env: str) -> Optional[float]:
        """The solve_threshold recorded for this task_env, if any record
        carries one. Any row for a given task_env carries the same
        threshold value per the wiring at every ledger-writing call site,
        so the first non-None value found is sufficient."""
        records = self._store.query_experiments(task_env=task_env, limit=1000)
        for r in records:
            if r.solve_threshold is not None:
                return r.solve_threshold
        return None

    def pooled_stats(self, *, task_env: str, method: str) -> PooledEstimate:
        """Compute a pooled estimate for one (task_env, method) pair,
        applying the abstention gate to every candidate source task
        before including it. Falls back to a target-only estimate
        (fallback_triggered=True) if no source passes the gate, the
        target has fewer than MIN_TARGET_N observations, or there are
        no source task_envs configured at all.
        """
        target_scores = self._scores_for(task_env, method)
        target_features = self.task_features(task_env)

        if len(target_scores) < MIN_TARGET_N:
            # Not enough target data to even run the gate meaningfully;
            # pool_estimates still works with whatever target data exists
            # (possibly raising if there is truly none), but no sources
            # are ever admitted in this regime.
            if not target_scores:
                return PooledEstimate(
                    method=method, pooled_mean=float("nan"),
                    pooled_interval=SmallNInterval(mean=float("nan"), std=0.0, n=0, lo95=float("nan"), hi95=float("nan")),
                    sources_used=[], sources_gated_out=list(self._source_task_envs),
                    fallback_triggered=True,
                )
            result = pool_estimates(target_scores=target_scores, source_scores_by_task={}, similarity_weights={})
            return PooledEstimate(
                method=method, pooled_mean=result.pooled_mean, pooled_interval=result.pooled_interval,
                sources_used=[], sources_gated_out=list(self._source_task_envs), fallback_triggered=True,
            )

        sources_used: List[str] = []
        sources_gated_out: List[str] = []
        gated_scores: Dict[str, List[float]] = {}  # normalized units, not raw
        sim_weights: Dict[str, float] = {}

        target_threshold = self._solve_threshold_for(task_env)
        normalized_target = normalize_scores(target_scores, target_threshold)

        for source_env in self._source_task_envs:
            if source_env == task_env:
                continue
            source_scores = self._scores_for(source_env, method)
            source_threshold = self._solve_threshold_for(source_env)
            normalized_source = normalize_scores(source_scores, source_threshold)

            if normalized_source is None or normalized_target is None:
                # Honest fallback: thresholds unknown for one or both tasks
                # means a comparable scale cannot be established. Do NOT
                # fall back to raw-score comparison (that was the bug) --
                # treat this exactly like a genuine gate failure.
                sources_gated_out.append(source_env)
                continue

            if passes_percentile_gate(normalized_source, normalized_target):
                sources_used.append(source_env)
                gated_scores[source_env] = normalized_source
                source_features = self.task_features(source_env)
                sim_weights[source_env] = similarity_weight(
                    source_algorithm=source_features.algorithm,
                    target_algorithm=target_features.algorithm,
                    source_n=source_features.n_completed,
                    target_n=target_features.n_completed,
                )
            else:
                sources_gated_out.append(source_env)

        fallback_triggered = len(sources_used) == 0

        if normalized_target is None:
            # Target's own threshold is unknown -- no normalized space to
            # pool in at all. gated_scores is always empty in this case
            # (the loop above gates out every source when normalized_target
            # is None), so this is equivalent to target-only pooling on the
            # raw scores, same as the pre-existing no-sources-passed path.
            result = pool_estimates(target_scores=target_scores, source_scores_by_task={}, similarity_weights={})
            return PooledEstimate(
                method=method, pooled_mean=result.pooled_mean, pooled_interval=result.pooled_interval,
                sources_used=sources_used, sources_gated_out=sources_gated_out, fallback_triggered=fallback_triggered,
            )

        # Pool in normalized space -- mixing raw scores across tasks of
        # different scale/sign here was the same bug class already fixed
        # for the gate itself (see normalize_scores above); pooling needs
        # the same treatment since inverse-variance weighting on raw
        # scores is meaningless across incompatible units.
        result = pool_estimates(
            target_scores=normalized_target, source_scores_by_task=gated_scores, similarity_weights=sim_weights,
        )
        raw_pooled_mean = _denormalize(result.pooled_mean, target_threshold)
        raw_pooled_interval = SmallNInterval(
            mean=_denormalize(result.pooled_interval.mean, target_threshold),
            std=result.pooled_interval.std * abs(target_threshold),
            n=result.pooled_interval.n,
            lo95=_denormalize(result.pooled_interval.lo95, target_threshold),
            hi95=_denormalize(result.pooled_interval.hi95, target_threshold),
        )
        return PooledEstimate(
            method=method, pooled_mean=raw_pooled_mean, pooled_interval=raw_pooled_interval,
            sources_used=sources_used, sources_gated_out=sources_gated_out, fallback_triggered=fallback_triggered,
        )

    def propose(self, *, task_env: str, methods: List[str], budget: int) -> SelectionDecision:
        """Allocate `budget` trials across `methods` using pooled
        cross-task evidence, falling back to StrategySelector's plain
        non-learned propose() if every method's pooled estimate
        triggered the abstention fallback.
        """
        estimates = {m: self.pooled_stats(task_env=task_env, method=m) for m in methods}

        if all(e.fallback_triggered for e in estimates.values()):
            decision = self._baseline.propose(task_env=task_env, methods=methods, budget=budget)
            return SelectionDecision(
                allocations=decision.allocations,
                rationale=(
                    "[nanonets: fallback to baseline selector — no source task passed "
                    f"the percentile gate] {decision.rationale}"
                ),
            )

        ranked = sorted(methods, key=lambda m: estimates[m].pooled_interval.hi95, reverse=True)
        allocations = {m: 0 for m in methods}
        min_exploration = max(1, budget // len(methods)) if budget else 0

        untried = [m for m in methods if estimates[m].pooled_interval.n == 0]
        remaining = budget
        for m in untried:
            take = min(min_exploration, remaining)
            allocations[m] += take
            remaining -= take

        tried = [m for m in ranked if m not in untried]
        if tried and remaining > 0:
            if len(tried) == 1:
                allocations[tried[0]] += remaining
            else:
                best = tried[0]
                majority = int(round(remaining * 0.7))
                allocations[best] += majority
                rest_remaining = remaining - majority
                rest = tried[1:]
                share = rest_remaining // len(rest)
                leftover = rest_remaining - share * len(rest)
                for i, m in enumerate(rest):
                    allocations[m] += share + (1 if i < leftover else 0)
        elif remaining > 0:
            share = remaining // len(methods)
            leftover = remaining - share * len(methods)
            for i, m in enumerate(methods):
                allocations[m] += share + (1 if i < leftover else 0)

        parts = [
            f"{m} pooled_hi95={estimates[m].pooled_interval.hi95:.2f} "
            f"(sources_used={estimates[m].sources_used})"
            for m in ranked
        ]
        rationale = "[nanonets: pooled cross-task estimate] " + "; ".join(parts) + f"; allocating {allocations}"

        return SelectionDecision(allocations=allocations, rationale=rationale)

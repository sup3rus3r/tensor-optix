"""Phase 5 strategy grammar -- a small, closed set of typed training-
procedure operations, each wrapping an already-existing mutation
primitive (agent.set_hyperparams, checkpoint_registry.load_best, ...)
via a LoopCallback.

No operation generates code, text, or free-form parameters outside its
own typed dataclass fields. OP_REGISTRY is the concrete enforcement that
only these known classes can ever be reconstructed from serialized data
(candidate.py's to_op() looks up op_name here -- never eval()/exec()).

An operation that needs a capability an agent lacks returns None from
attach() rather than raising -- self-exclusion is the primary safety
mechanism; LoopController._fire()'s exception-swallowing is a backstop,
not something this grammar relies on.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Optional, Tuple, Type

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, HyperparamSet, LoopState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyOp:
    """Base for one typed grammar operation. Frozen -- a candidate's
    identity is its op type plus these exact parameter values (see
    candidate.py's content-hash fingerprint)."""

    op_name: ClassVar[str] = ""

    def required_capabilities(self) -> Tuple[str, ...]:
        """Agent attributes/methods this op needs. Checked by supports()."""
        return ()

    def supports(self, agent: BaseAgent) -> bool:
        return all(hasattr(agent, cap) for cap in self.required_capabilities())

    def attach(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        """Return a LoopCallback wired to `agent` (and whatever else
        **kwargs provides), or None if `agent` doesn't support this op.
        Never raises for an unsupported agent."""
        if not self.supports(agent):
            return None
        return self._build_callback(agent, **kwargs)

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        raise NotImplementedError


@dataclass(frozen=True)
class LearningRateTrendOp(StrategyOp):
    """Change learning rate according to a measured trend in the
    episode-score history."""

    op_name: ClassVar[str] = "lr_trend"
    window: int = 5
    increase_factor: float = 1.2
    decrease_factor: float = 0.5
    min_lr: float = 1e-6
    max_lr: float = 1e-2

    def required_capabilities(self) -> Tuple[str, ...]:
        return ("set_hyperparams", "get_hyperparams")

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        return _LRTrendCallback(agent, self)


class _LRTrendCallback(LoopCallback):
    def __init__(self, agent: BaseAgent, op: LearningRateTrendOp):
        self._agent = agent
        self._op = op
        self._history: deque = deque(maxlen=op.window)

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is None:
            return
        self._history.append(eval_metrics.primary_score)
        if len(self._history) < self._op.window:
            return

        slope = _trend_slope(self._history)
        hp = self._agent.get_hyperparams()
        current_lr = float(hp.params.get("learning_rate", 0.0))
        if current_lr <= 0.0:
            return

        factor = self._op.increase_factor if slope > 0 else self._op.decrease_factor
        new_lr = min(max(current_lr * factor, self._op.min_lr), self._op.max_lr)
        if new_lr == current_lr:
            return

        new_params = dict(hp.params)
        new_params["learning_rate"] = new_lr
        self._agent.set_hyperparams(HyperparamSet(params=new_params, episode_id=episode_id))


def _trend_slope(values) -> float:
    """Least-squares slope of `values` against their index. Mirrors the
    linear-regression-over-window idiom used elsewhere in the loop
    controller's own backoff scheduling, re-derived here (not imported)
    since that implementation is private to its own module."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


@dataclass(frozen=True)
class CheckpointRollbackOp(StrategyOp):
    """Trigger a checkpoint rollback under a defined validation
    condition (degradation or plateau)."""

    op_name: ClassVar[str] = "checkpoint_rollback"
    trigger: str = "on_degradation"  # "on_degradation" | "on_plateau"
    min_consecutive: int = 1
    cooldown_episodes: int = 5

    def __post_init__(self):
        if self.trigger not in ("on_degradation", "on_plateau"):
            raise ValueError(f"CheckpointRollbackOp.trigger must be 'on_degradation' or 'on_plateau', got {self.trigger!r}")

    def required_capabilities(self) -> Tuple[str, ...]:
        return ()

    def supports(self, agent: BaseAgent) -> bool:
        # Off-policy agents' replay buffers can go stale on a weight
        # rollback -- the core loop's own rollback gate excludes them
        # for the same reason; mirror that here.
        return getattr(agent, "is_on_policy", True)

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        registry = kwargs.get("checkpoint_registry")
        if registry is None:
            return None
        return _RollbackCallback(agent, registry, self)


class _RollbackCallback(LoopCallback):
    def __init__(self, agent: BaseAgent, registry, op: CheckpointRollbackOp):
        self._agent = agent
        self._registry = registry
        self._op = op
        self._consecutive = 0
        self._last_rollback_episode: Optional[int] = None

    def _maybe_rollback(self, episode_id: int) -> None:
        self._consecutive += 1
        if self._consecutive < self._op.min_consecutive:
            return
        if (
            self._last_rollback_episode is not None
            and episode_id - self._last_rollback_episode < self._op.cooldown_episodes
        ):
            return
        snapshot = self._registry.load_best(self._agent)
        if snapshot is not None:
            self._last_rollback_episode = episode_id
            self._consecutive = 0
            logger.info("CheckpointRollbackOp: rolled back to best snapshot at episode %d", episode_id)

    def on_degradation(self, episode_id: int, eval_metrics: EvalMetrics) -> None:
        if self._op.trigger == "on_degradation":
            self._maybe_rollback(episode_id)
        else:
            self._consecutive = 0

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        if self._op.trigger == "on_plateau":
            self._maybe_rollback(episode_id)
        else:
            self._consecutive = 0

    def on_improvement(self, snapshot) -> None:
        self._consecutive = 0


@dataclass(frozen=True)
class ExplorationBoundsOp(StrategyOp):
    """Adjust exploration within defined bounds. "Exploration" is three
    distinct real hyperparameters depending on algorithm family --
    PPO-family's entropy_coef, DQN-family's epsilon, TD3's
    target_noise/target_noise_clip (target-policy-smoothing noise, not
    behavioral exploration). SAC has no addressable static knob (its
    temperature is learned, overwritten every step by its own gradient
    update) -- this op self-excludes for SAC."""

    op_name: ClassVar[str] = "exploration_bounds"
    param_name: str = "entropy_coef"
    lo: float = 0.0
    hi: float = 0.1
    step_fn: str = "plateau_increase"  # "plateau_increase" | "improvement_decay"

    def __post_init__(self):
        if self.step_fn not in ("plateau_increase", "improvement_decay"):
            raise ValueError(f"ExplorationBoundsOp.step_fn must be 'plateau_increase' or 'improvement_decay', got {self.step_fn!r}")

    def required_capabilities(self) -> Tuple[str, ...]:
        return ("set_hyperparams", "get_hyperparams")

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        if self.param_name not in agent.get_hyperparams().params:
            return None  # agent's hyperparam schema doesn't expose this knob (e.g. SAC)
        return _ExplorationBoundsCallback(agent, self)


class _ExplorationBoundsCallback(LoopCallback):
    def __init__(self, agent: BaseAgent, op: ExplorationBoundsOp):
        self._agent = agent
        self._op = op

    def _adjust(self, episode_id: int, factor: float) -> None:
        hp = self._agent.get_hyperparams()
        current = hp.params.get(self._op.param_name)
        if current is None:
            return
        new_value = min(max(current * factor, self._op.lo), self._op.hi)
        if new_value == current:
            return
        new_params = dict(hp.params)
        new_params[self._op.param_name] = new_value
        self._agent.set_hyperparams(HyperparamSet(params=new_params, episode_id=episode_id))

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        if self._op.step_fn == "plateau_increase":
            self._adjust(episode_id, factor=1.5)

    def on_improvement(self, snapshot) -> None:
        if self._op.step_fn == "improvement_decay":
            self._adjust(getattr(snapshot, "episode_id", 0), factor=0.9)


@dataclass(frozen=True)
class OptimizationMethodChoiceOp(StrategyOp):
    """Choose among supported BaseOptimizer tuning strategies -- switches
    WHICH BaseOptimizer drives the run's own hyperparameter search, not
    torch.optim.Adam-vs-SGD (there is no generic hook for that, and this
    op does not invent one). candidate_methods names a closed set of
    optimizer factories the caller supplies via attach()'s
    optimizer_factories kwarg -- the op orchestrates WHEN to switch, not
    HOW to construct each BaseOptimizer (their constructors are too
    varied to duplicate here)."""

    op_name: ClassVar[str] = "optimizer_method_choice"
    candidate_methods: Tuple[str, ...] = ("spsa",)
    switch_condition: str = "on_plateau"  # "on_plateau" | "on_dormant"

    def __post_init__(self):
        if self.switch_condition not in ("on_plateau", "on_dormant"):
            raise ValueError(f"OptimizationMethodChoiceOp.switch_condition must be 'on_plateau' or 'on_dormant', got {self.switch_condition!r}")

    def required_capabilities(self) -> Tuple[str, ...]:
        return ()

    def supports(self, agent: BaseAgent) -> bool:
        return True

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        loop_controller = kwargs.get("loop_controller")
        optimizer_factories = kwargs.get("optimizer_factories")
        if loop_controller is None or not optimizer_factories:
            return None
        missing = [m for m in self.candidate_methods if m not in optimizer_factories]
        if missing:
            logger.info("OptimizationMethodChoiceOp: skipping, no factory for %s", missing)
            return None
        return _OptimizerSwitchCallback(loop_controller, optimizer_factories, self)


class _OptimizerSwitchCallback(LoopCallback):
    def __init__(self, loop_controller, optimizer_factories: Dict[str, Callable], op: "OptimizationMethodChoiceOp"):
        self._controller = loop_controller
        self._factories = optimizer_factories
        self._op = op
        self._next_idx = 0
        self._switched = False

    def _switch(self) -> None:
        if self._switched or self._next_idx >= len(self._op.candidate_methods):
            return
        method = self._op.candidate_methods[self._next_idx]
        self._next_idx += 1
        self._controller.set_optimizer(self._factories[method]())
        self._switched = True
        logger.info("OptimizationMethodChoiceOp: switched to %r", method)

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        if self._op.switch_condition == "on_plateau":
            self._switch()

    def on_dormant(self, episode_id: int) -> None:
        if self._op.switch_condition == "on_dormant":
            self._switch()


@dataclass(frozen=True)
class BudgetAllocationOp(StrategyOp):
    """Allocate more (within-run ensemble) budget to a promising
    candidate -- a thin typed wrapper around PolicyManager's already-
    existing record_agent_score/auto_update_weights, which already
    implement this verbatim. The trial-level reading of this same
    operation (allocate more trial/seed budget to a promising strategy
    CANDIDATE) happens naturally in StrategyTrialRunner's own seed
    count, not as a second grammar class."""

    op_name: ClassVar[str] = "budget_allocation"
    promotion_factor: float = 2.0
    evaluation_window: int = 10

    def required_capabilities(self) -> Tuple[str, ...]:
        return ()

    def supports(self, agent: BaseAgent) -> bool:
        return True

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        pm = kwargs.get("policy_manager")
        if pm is None or getattr(pm, "ensemble_size", 0) < 1:
            return None
        return _BudgetAllocationCallback(pm, agent, self)


def _ensemble_index_of(policy_manager, agent: BaseAgent) -> Optional[int]:
    """PolicyManager has no public index-lookup -- its ensemble is a
    private List[Tuple[BaseAgent, float]], positional by add_agent()
    call order. Scan for the agent's identity (not equality -- two
    agents with identical state are still distinct ensemble members)."""
    for i, (member, _) in enumerate(getattr(policy_manager, "_ensemble", [])):
        if member is agent:
            return i
    return None


class _BudgetAllocationCallback(LoopCallback):
    def __init__(self, policy_manager, agent: BaseAgent, op: BudgetAllocationOp):
        self._pm = policy_manager
        self._agent = agent
        self._op = op
        self._episode_count = 0

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is None:
            return
        agent_idx = _ensemble_index_of(self._pm, self._agent)
        if agent_idx is None:
            return
        self._pm.record_agent_score(agent_idx, eval_metrics.primary_score)
        self._episode_count += 1
        if self._episode_count % self._op.evaluation_window == 0:
            self._pm.auto_update_weights()


@dataclass(frozen=True)
class ScheduleChangeOp(StrategyOp):
    """Change a supported schedule -- grounded in the two real schedules
    in the codebase: DQN-family's epsilon_decay (via set_hyperparams)
    and RND's pipeline.set_eta-driven eta schedule (called directly on
    the pipeline, independent of LoopController's own internal
    _update_rnd_eta scheduling)."""

    op_name: ClassVar[str] = "schedule_change"
    target: str = "epsilon_decay"  # "epsilon_decay" | "rnd_eta"
    new_value: float = 0.0

    def __post_init__(self):
        if self.target not in ("epsilon_decay", "rnd_eta"):
            raise ValueError(f"ScheduleChangeOp.target must be 'epsilon_decay' or 'rnd_eta', got {self.target!r}")

    def required_capabilities(self) -> Tuple[str, ...]:
        return ("set_hyperparams", "get_hyperparams") if self.target == "epsilon_decay" else ()

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        if self.target == "epsilon_decay":
            if "epsilon_decay" not in agent.get_hyperparams().params:
                return None
            return _ScheduleChangeCallback(agent=agent, pipeline=None, op=self)
        pipeline = kwargs.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "set_eta"):
            return None
        return _ScheduleChangeCallback(agent=agent, pipeline=pipeline, op=self)


class _ScheduleChangeCallback(LoopCallback):
    def __init__(self, *, agent: BaseAgent, pipeline, op: ScheduleChangeOp):
        self._agent = agent
        self._pipeline = pipeline
        self._op = op

    def on_loop_start(self) -> None:
        if self._op.target == "epsilon_decay":
            hp = self._agent.get_hyperparams()
            new_params = dict(hp.params)
            new_params["epsilon_decay"] = self._op.new_value
            self._agent.set_hyperparams(HyperparamSet(params=new_params, episode_id=0))
        else:
            self._pipeline.set_eta(self._op.new_value)


@dataclass(frozen=True)
class PolicyVariantSelectionOp(StrategyOp):
    """Select a policy variant based on defined evidence -- a thin
    wrapper around PolicyManager.boost/prune, reading evidence from
    pm.status()["agents"] (real fields: index, weight, mean_score,
    recent_scores -- evidence_metric is restricted to what's actually
    there, not an invented generic concept). Requires an active
    multi-agent ensemble (ensemble_size >= 2); self-excludes for
    single-agent runs."""

    op_name: ClassVar[str] = "policy_variant_selection"
    evidence_metric: str = "mean_score"  # "mean_score" | "weight"
    selection_rule: str = "boost_best"  # "boost_best" | "prune_worst"
    boost_factor: float = 2.0
    evaluation_window: int = 10

    def __post_init__(self):
        if self.evidence_metric not in ("mean_score", "weight"):
            raise ValueError(f"PolicyVariantSelectionOp.evidence_metric invalid: {self.evidence_metric!r}")
        if self.selection_rule not in ("boost_best", "prune_worst"):
            raise ValueError(f"PolicyVariantSelectionOp.selection_rule invalid: {self.selection_rule!r}")

    def required_capabilities(self) -> Tuple[str, ...]:
        return ()

    def supports(self, agent: BaseAgent) -> bool:
        return True

    def _build_callback(self, agent: BaseAgent, **kwargs) -> Optional[LoopCallback]:
        pm = kwargs.get("policy_manager")
        if pm is None or getattr(pm, "ensemble_size", 0) < 2:
            return None
        return _VariantSelectionCallback(pm, self)


class _VariantSelectionCallback(LoopCallback):
    def __init__(self, policy_manager, op: PolicyVariantSelectionOp):
        self._pm = policy_manager
        self._op = op
        self._episode_count = 0

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        self._episode_count += 1
        if self._episode_count % self._op.evaluation_window != 0:
            return
        status = self._pm.status()
        agents_info = status.get("agents", [])
        if not agents_info:
            return

        metric = self._op.evidence_metric
        ranked = sorted(
            (a for a in agents_info if a.get(metric) is not None),
            key=lambda a: a[metric],
            reverse=True,
        )
        if not ranked:
            return

        if self._op.selection_rule == "boost_best":
            best_idx = ranked[0]["index"]
            member = self._pm._ensemble[best_idx][0]
            self._pm.boost(member, factor=self._op.boost_factor)
        elif self._op.selection_rule == "prune_worst":
            self._pm.prune(bottom_k=1)


OP_REGISTRY: Dict[str, Type[StrategyOp]] = {
    LearningRateTrendOp.op_name: LearningRateTrendOp,
    CheckpointRollbackOp.op_name: CheckpointRollbackOp,
    ExplorationBoundsOp.op_name: ExplorationBoundsOp,
    OptimizationMethodChoiceOp.op_name: OptimizationMethodChoiceOp,
    BudgetAllocationOp.op_name: BudgetAllocationOp,
    ScheduleChangeOp.op_name: ScheduleChangeOp,
    PolicyVariantSelectionOp.op_name: PolicyVariantSelectionOp,
}

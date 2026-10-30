import pytest

from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.loop_controller import LoopController
from tensor_optix.core.policy_manager import PolicyManager
from tensor_optix.core.types import EvalMetrics, HyperparamSet, LoopState
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer
from tensor_optix.strategy.grammar import (
    BudgetAllocationOp,
    ExplorationBoundsOp,
    OptimizationMethodChoiceOp,
    PolicyVariantSelectionOp,
    ScheduleChangeOp,
)
from tests.conftest import DummyAgent, DummyEvaluator, DummyPipeline


def test_loop_controller_set_optimizer_actually_swaps_the_active_optimizer(tmp_path):
    """Direct verification of the one necessary core-framework change
    this grammar relies on: LoopController.set_optimizer() must
    genuinely replace self._optimizer, not just appear to."""
    from tensor_optix.core.base_optimizer import BaseOptimizer

    class _MarkerOptimizer(BaseOptimizer):
        def suggest(self, current_hyperparams, metrics_history):
            return current_hyperparams

    agent = DummyAgent()
    original_optimizer = BackoffOptimizer()
    controller = LoopController(
        agent=agent, evaluator=DummyEvaluator(), optimizer=original_optimizer,
        pipeline=DummyPipeline(), checkpoint_registry=CheckpointRegistry(str(tmp_path)),
        backoff_scheduler=BackoffScheduler(base_interval=1, plateau_threshold=50, dormant_threshold=100),
        max_episodes=1,
    )
    assert controller._optimizer is original_optimizer

    new_optimizer = _MarkerOptimizer()
    controller.set_optimizer(new_optimizer)

    assert controller._optimizer is new_optimizer
    assert controller._optimizer is not original_optimizer


# ------------------------------------------------------------------ #
# ExplorationBoundsOp
# ------------------------------------------------------------------ #

def test_exploration_bounds_rejects_invalid_step_fn():
    with pytest.raises(ValueError):
        ExplorationBoundsOp(step_fn="bogus")


def test_exploration_bounds_skips_agent_without_the_param():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = ExplorationBoundsOp(param_name="entropy_coef")
    assert op.attach(agent) is None  # no entropy_coef on this agent's hyperparams


def test_exploration_bounds_increases_on_plateau():
    agent = DummyAgent(HyperparamSet(params={"entropy_coef": 0.01}, episode_id=0))
    op = ExplorationBoundsOp(param_name="entropy_coef", lo=0.0, hi=0.1, step_fn="plateau_increase")
    cb = op.attach(agent)
    assert cb is not None

    cb.on_plateau(5, LoopState.COOLING)

    assert agent.get_hyperparams().params["entropy_coef"] == pytest.approx(0.015)


def test_exploration_bounds_respects_hi_clamp():
    agent = DummyAgent(HyperparamSet(params={"entropy_coef": 0.09}, episode_id=0))
    op = ExplorationBoundsOp(param_name="entropy_coef", lo=0.0, hi=0.1, step_fn="plateau_increase")
    cb = op.attach(agent)
    cb.on_plateau(5, LoopState.COOLING)
    assert agent.get_hyperparams().params["entropy_coef"] == pytest.approx(0.1)


def test_exploration_bounds_decreases_on_improvement():
    agent = DummyAgent(HyperparamSet(params={"entropy_coef": 0.05}, episode_id=0))
    op = ExplorationBoundsOp(param_name="entropy_coef", lo=0.0, hi=0.1, step_fn="improvement_decay")
    cb = op.attach(agent)

    snapshot = type("S", (), {"episode_id": 3})()
    cb.on_improvement(snapshot)

    assert agent.get_hyperparams().params["entropy_coef"] == pytest.approx(0.045)


# ------------------------------------------------------------------ #
# OptimizationMethodChoiceOp
# ------------------------------------------------------------------ #

def test_optimizer_method_choice_rejects_invalid_switch_condition():
    with pytest.raises(ValueError):
        OptimizationMethodChoiceOp(switch_condition="bogus")


def test_optimizer_method_choice_requires_loop_controller_and_factories():
    agent = DummyAgent()
    op = OptimizationMethodChoiceOp(candidate_methods=("spsa",))
    assert op.attach(agent) is None  # no loop_controller/optimizer_factories kwargs
    assert op.attach(agent, loop_controller=object()) is None  # no factories


def test_optimizer_method_choice_skips_when_factory_missing():
    agent = DummyAgent()
    op = OptimizationMethodChoiceOp(candidate_methods=("nonexistent_method",))
    cb = op.attach(agent, loop_controller=object(), optimizer_factories={"spsa": lambda: None})
    assert cb is None


def test_optimizer_method_choice_switches_optimizer_on_plateau():
    from tensor_optix.core.base_optimizer import BaseOptimizer

    class _FakeOptimizer(BaseOptimizer):
        def suggest(self, current_hyperparams, metrics_history):
            return current_hyperparams

    class _StubController:
        def __init__(self):
            self.set_calls = []

        def set_optimizer(self, optimizer):
            self.set_calls.append(optimizer)

    agent = DummyAgent()
    controller = _StubController()

    switched_to = {}

    def _factory():
        opt = _FakeOptimizer()
        switched_to["opt"] = opt
        return opt

    op = OptimizationMethodChoiceOp(candidate_methods=("fake",), switch_condition="on_plateau")
    cb = op.attach(agent, loop_controller=controller, optimizer_factories={"fake": _factory})
    assert cb is not None

    cb.on_plateau(1, LoopState.COOLING)

    assert controller.set_calls == [switched_to["opt"]]


def test_optimizer_method_choice_only_switches_once():
    from tensor_optix.core.base_optimizer import BaseOptimizer

    class _FakeOptimizer(BaseOptimizer):
        def suggest(self, current_hyperparams, metrics_history):
            return current_hyperparams

    agent = DummyAgent()
    call_count = {"n": 0}

    def _factory():
        call_count["n"] += 1
        return _FakeOptimizer()

    op = OptimizationMethodChoiceOp(candidate_methods=("fake",), switch_condition="on_plateau")
    cb = op.attach(agent, loop_controller=type("C", (), {"set_optimizer": lambda self, o: None})(),
                   optimizer_factories={"fake": _factory})

    cb.on_plateau(1, LoopState.COOLING)
    cb.on_plateau(2, LoopState.COOLING)

    assert call_count["n"] == 1


# ------------------------------------------------------------------ #
# BudgetAllocationOp
# ------------------------------------------------------------------ #

def test_budget_allocation_skips_without_policy_manager():
    agent = DummyAgent()
    op = BudgetAllocationOp()
    assert op.attach(agent) is None


def test_budget_allocation_records_score_and_updates_weights(tmp_path):
    registry = CheckpointRegistry(str(tmp_path))
    pm = PolicyManager(registry)
    agent = DummyAgent()
    pm.add_agent(agent, weight=1.0)

    op = BudgetAllocationOp(evaluation_window=2)
    cb = op.attach(agent, policy_manager=pm)
    assert cb is not None

    cb.on_episode_end(0, EvalMetrics(primary_score=10.0, metrics={}, episode_id=0))
    cb.on_episode_end(1, EvalMetrics(primary_score=20.0, metrics={}, episode_id=1))

    status = pm.status()
    assert status["agents"][0]["mean_score"] == pytest.approx(15.0)


def test_budget_allocation_ignores_agent_not_in_ensemble(tmp_path):
    registry = CheckpointRegistry(str(tmp_path))
    pm = PolicyManager(registry)
    other_agent = DummyAgent()
    pm.add_agent(other_agent)

    unrelated_agent = DummyAgent()
    op = BudgetAllocationOp()
    cb = op.attach(unrelated_agent, policy_manager=pm)
    cb.on_episode_end(0, EvalMetrics(primary_score=10.0, metrics={}, episode_id=0))

    assert pm.status()["agents"][0]["mean_score"] is None  # unrelated_agent's score never recorded


# ------------------------------------------------------------------ #
# ScheduleChangeOp
# ------------------------------------------------------------------ #

def test_schedule_change_rejects_invalid_target():
    with pytest.raises(ValueError):
        ScheduleChangeOp(target="bogus")


def test_schedule_change_epsilon_decay_skips_agent_without_key():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = ScheduleChangeOp(target="epsilon_decay", new_value=0.99)
    assert op.attach(agent) is None


def test_schedule_change_epsilon_decay_sets_value():
    agent = DummyAgent(HyperparamSet(params={"epsilon_decay": 0.95}, episode_id=0))
    op = ScheduleChangeOp(target="epsilon_decay", new_value=0.99)
    cb = op.attach(agent)
    assert cb is not None
    cb.on_loop_start()
    assert agent.get_hyperparams().params["epsilon_decay"] == pytest.approx(0.99)


def test_schedule_change_rnd_eta_skips_pipeline_without_set_eta():
    agent = DummyAgent()
    op = ScheduleChangeOp(target="rnd_eta", new_value=0.5)
    assert op.attach(agent, pipeline=DummyPipeline()) is None


def test_schedule_change_rnd_eta_calls_set_eta():
    class _RNDPipeline(DummyPipeline):
        def __init__(self):
            super().__init__()
            self.eta_calls = []

        def set_eta(self, value):
            self.eta_calls.append(value)

    agent = DummyAgent()
    pipeline = _RNDPipeline()
    op = ScheduleChangeOp(target="rnd_eta", new_value=0.42)
    cb = op.attach(agent, pipeline=pipeline)
    assert cb is not None
    cb.on_loop_start()
    assert pipeline.eta_calls == [0.42]


# ------------------------------------------------------------------ #
# PolicyVariantSelectionOp
# ------------------------------------------------------------------ #

def test_policy_variant_selection_rejects_invalid_fields():
    with pytest.raises(ValueError):
        PolicyVariantSelectionOp(evidence_metric="bogus")
    with pytest.raises(ValueError):
        PolicyVariantSelectionOp(selection_rule="bogus")


def test_policy_variant_selection_skips_single_agent_ensemble(tmp_path):
    registry = CheckpointRegistry(str(tmp_path))
    pm = PolicyManager(registry)
    agent = DummyAgent()
    pm.add_agent(agent)

    op = PolicyVariantSelectionOp()
    assert op.attach(agent, policy_manager=pm) is None


def test_policy_variant_selection_boosts_best_agent(tmp_path):
    registry = CheckpointRegistry(str(tmp_path))
    pm = PolicyManager(registry)
    agent_a = DummyAgent()
    agent_b = DummyAgent()
    pm.add_agent(agent_a, weight=1.0)
    pm.add_agent(agent_b, weight=1.0)

    pm.record_agent_score(0, 10.0)
    pm.record_agent_score(1, 50.0)  # agent_b is clearly better

    op = PolicyVariantSelectionOp(evidence_metric="mean_score", selection_rule="boost_best",
                                   boost_factor=3.0, evaluation_window=1)
    cb = op.attach(agent_a, policy_manager=pm)
    assert cb is not None

    weight_before = pm.status()["agents"][1]["weight"]
    cb.on_episode_end(0, EvalMetrics(primary_score=1.0, metrics={}, episode_id=0))
    weight_after = pm.status()["agents"][1]["weight"]

    assert weight_after == pytest.approx(weight_before * 3.0)


def test_policy_variant_selection_prunes_worst(tmp_path):
    registry = CheckpointRegistry(str(tmp_path))
    pm = PolicyManager(registry)
    agent_a = DummyAgent()
    agent_b = DummyAgent()
    pm.add_agent(agent_a, weight=5.0)
    pm.add_agent(agent_b, weight=1.0)

    pm.record_agent_score(0, 10.0)
    pm.record_agent_score(1, 1.0)

    op = PolicyVariantSelectionOp(evidence_metric="weight", selection_rule="prune_worst", evaluation_window=1)
    cb = op.attach(agent_a, policy_manager=pm)

    assert pm.ensemble_size == 2
    cb.on_episode_end(0, EvalMetrics(primary_score=1.0, metrics={}, episode_id=0))
    assert pm.ensemble_size == 1

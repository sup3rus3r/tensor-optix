import pytest

from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet
from tensor_optix.strategy import OP_REGISTRY, CheckpointRollbackOp, LearningRateTrendOp
from tests.conftest import DummyAgent


def test_op_registry_contains_both_implemented_ops():
    assert OP_REGISTRY["lr_trend"] is LearningRateTrendOp
    assert OP_REGISTRY["checkpoint_rollback"] is CheckpointRollbackOp


def test_checkpoint_rollback_rejects_unknown_trigger():
    with pytest.raises(ValueError):
        CheckpointRollbackOp(trigger="on_whatever")


def test_lr_trend_requires_set_and_get_hyperparams():
    op = LearningRateTrendOp()
    assert op.required_capabilities() == ("set_hyperparams", "get_hyperparams")
    agent = DummyAgent()
    assert op.supports(agent) is True


def test_lr_trend_attach_returns_callback_for_supported_agent():
    op = LearningRateTrendOp(window=3)
    agent = DummyAgent()
    cb = op.attach(agent)
    assert cb is not None


def test_lr_trend_increases_lr_on_improving_trend():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = LearningRateTrendOp(window=3, increase_factor=1.5, decrease_factor=0.5)
    cb = op.attach(agent)

    for i, score in enumerate([10.0, 20.0, 30.0]):
        cb.on_episode_end(i, EvalMetrics(primary_score=score, metrics={}, episode_id=i))

    assert agent.get_hyperparams().params["learning_rate"] == pytest.approx(1.5e-3)


def test_lr_trend_decreases_lr_on_degrading_trend():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = LearningRateTrendOp(window=3, increase_factor=1.5, decrease_factor=0.5)
    cb = op.attach(agent)

    for i, score in enumerate([30.0, 20.0, 10.0]):
        cb.on_episode_end(i, EvalMetrics(primary_score=score, metrics={}, episode_id=i))

    assert agent.get_hyperparams().params["learning_rate"] == pytest.approx(0.5e-3)


def test_lr_trend_respects_min_and_max_bounds():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = LearningRateTrendOp(window=3, increase_factor=100.0, max_lr=2e-3)
    cb = op.attach(agent)

    for i, score in enumerate([10.0, 20.0, 30.0]):
        cb.on_episode_end(i, EvalMetrics(primary_score=score, metrics={}, episode_id=i))

    assert agent.get_hyperparams().params["learning_rate"] == pytest.approx(2e-3)


def test_lr_trend_does_nothing_before_window_fills():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = LearningRateTrendOp(window=5)
    cb = op.attach(agent)

    cb.on_episode_end(0, EvalMetrics(primary_score=10.0, metrics={}, episode_id=0))
    cb.on_episode_end(1, EvalMetrics(primary_score=20.0, metrics={}, episode_id=1))

    assert agent.get_hyperparams().params["learning_rate"] == pytest.approx(1e-3)


def test_lr_trend_skips_when_no_eval_metrics():
    agent = DummyAgent(HyperparamSet(params={"learning_rate": 1e-3}, episode_id=0))
    op = LearningRateTrendOp(window=2)
    cb = op.attach(agent)

    cb.on_episode_end(0, None)
    cb.on_episode_end(1, None)

    assert agent.get_hyperparams().params["learning_rate"] == pytest.approx(1e-3)


def test_checkpoint_rollback_supports_on_policy_agent():
    op = CheckpointRollbackOp()
    agent = DummyAgent()
    assert op.supports(agent) is True


def test_checkpoint_rollback_attach_requires_registry_kwarg():
    op = CheckpointRollbackOp()
    agent = DummyAgent()
    assert op.attach(agent) is None  # no checkpoint_registry kwarg supplied
    registry = CheckpointRegistry(str(pytest.importorskip("tempfile").mkdtemp()))
    assert op.attach(agent, checkpoint_registry=registry) is not None


def test_checkpoint_rollback_restores_weights_on_degradation(tmp_path):
    agent = DummyAgent()
    registry = CheckpointRegistry(str(tmp_path))

    agent._weights = {"w": 1.0}
    registry.save(agent, eval_metrics=EvalMetrics(primary_score=100.0, metrics={}, episode_id=0),
                   hyperparams=agent.get_hyperparams())

    agent._weights = {"w": 999.0}  # simulate degraded training

    op = CheckpointRollbackOp(trigger="on_degradation", min_consecutive=1, cooldown_episodes=0)
    cb = op.attach(agent, checkpoint_registry=registry)
    assert cb is not None

    cb.on_degradation(1, EvalMetrics(primary_score=1.0, metrics={}, episode_id=1))

    assert agent._weights == {"w": 1.0}


def test_checkpoint_rollback_respects_min_consecutive(tmp_path):
    agent = DummyAgent()
    registry = CheckpointRegistry(str(tmp_path))
    agent._weights = {"w": 1.0}
    registry.save(agent, eval_metrics=EvalMetrics(primary_score=100.0, metrics={}, episode_id=0),
                   hyperparams=agent.get_hyperparams())
    agent._weights = {"w": 999.0}

    op = CheckpointRollbackOp(trigger="on_degradation", min_consecutive=3, cooldown_episodes=0)
    cb = op.attach(agent, checkpoint_registry=registry)

    cb.on_degradation(1, EvalMetrics(primary_score=1.0, metrics={}, episode_id=1))
    cb.on_degradation(2, EvalMetrics(primary_score=1.0, metrics={}, episode_id=2))
    assert agent._weights == {"w": 999.0}  # not yet rolled back -- only 2 consecutive

    cb.on_degradation(3, EvalMetrics(primary_score=1.0, metrics={}, episode_id=3))
    assert agent._weights == {"w": 1.0}  # 3rd consecutive triggers rollback


def test_checkpoint_rollback_respects_cooldown(tmp_path):
    agent = DummyAgent()
    registry = CheckpointRegistry(str(tmp_path))
    agent._weights = {"w": 1.0}
    registry.save(agent, eval_metrics=EvalMetrics(primary_score=100.0, metrics={}, episode_id=0),
                   hyperparams=agent.get_hyperparams())

    op = CheckpointRollbackOp(trigger="on_degradation", min_consecutive=1, cooldown_episodes=10)
    cb = op.attach(agent, checkpoint_registry=registry)

    agent._weights = {"w": 999.0}
    cb.on_degradation(1, EvalMetrics(primary_score=1.0, metrics={}, episode_id=1))
    assert agent._weights == {"w": 1.0}  # first rollback succeeds

    agent._weights = {"w": 888.0}
    cb.on_degradation(2, EvalMetrics(primary_score=1.0, metrics={}, episode_id=2))
    assert agent._weights == {"w": 888.0}  # still within cooldown, no second rollback


def test_checkpoint_rollback_on_improvement_resets_consecutive_counter(tmp_path):
    agent = DummyAgent()
    registry = CheckpointRegistry(str(tmp_path))
    agent._weights = {"w": 1.0}
    registry.save(agent, eval_metrics=EvalMetrics(primary_score=100.0, metrics={}, episode_id=0),
                   hyperparams=agent.get_hyperparams())
    agent._weights = {"w": 999.0}

    op = CheckpointRollbackOp(trigger="on_degradation", min_consecutive=2, cooldown_episodes=0)
    cb = op.attach(agent, checkpoint_registry=registry)

    cb.on_degradation(1, EvalMetrics(primary_score=1.0, metrics={}, episode_id=1))
    cb.on_improvement(None)  # resets the counter
    cb.on_degradation(2, EvalMetrics(primary_score=1.0, metrics={}, episode_id=2))

    assert agent._weights == {"w": 999.0}  # counter reset, so this is only the 1st consecutive again

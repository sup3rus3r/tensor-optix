import pytest
import numpy as np
import tempfile
from tensor_optix.core.policy_manager import PolicyManager
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_registry(tmp_path):
    return CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)


def make_metrics(score, episode_id=0):
    return EvalMetrics(primary_score=score, metrics={}, episode_id=episode_id)


def make_hp(lr=1e-3, gamma=0.99):
    return HyperparamSet(params={"learning_rate": lr, "gamma": gamma}, episode_id=0)


# -----------------------------------------------------------------------
# PolicyManager.spawn_variant()
# -----------------------------------------------------------------------

def test_spawn_variant_loads_best_weights(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    source = DummyAgent()
    source._weights["w"] = 42.0
    registry.save(source, make_metrics(10.0), make_hp())

    shell = DummyAgent()
    shell._weights["w"] = 0.0  # different from saved

    pm = PolicyManager(registry)
    pm.spawn_variant(shell, noise_scale=0.0)

    assert shell._weights["w"] == 42.0


def test_spawn_variant_perturbs_float_hyperparams(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(10.0), make_hp(lr=1e-3, gamma=0.99))

    shell = DummyAgent()
    pm = PolicyManager(registry)
    pm.spawn_variant(shell, noise_scale=0.5)  # large noise to ensure change

    hp = shell.get_hyperparams()
    # At least one param should differ from original with high-noise perturbation
    original_lr = 1e-3
    original_gamma = 0.99
    changed = (
        hp.params["learning_rate"] != original_lr
        or hp.params["gamma"] != original_gamma
    )
    assert changed


def test_spawn_variant_no_perturbation_at_zero_noise(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(10.0), make_hp(lr=1e-3, gamma=0.99))

    shell = DummyAgent()
    pm = PolicyManager(registry)
    pm.spawn_variant(shell, noise_scale=0.0)

    hp = shell.get_hyperparams()
    assert hp.params["learning_rate"] == pytest.approx(1e-3)
    assert hp.params["gamma"] == pytest.approx(0.99)


def test_spawn_variant_returns_agent_shell(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    shell = DummyAgent()
    pm = PolicyManager(registry)
    result = pm.spawn_variant(shell)

    assert result is shell


def test_spawn_variant_calls_mutation_fn(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    shell = DummyAgent()
    mutation_called_with = []

    def my_mutation(a):
        mutation_called_with.append(a)

    pm = PolicyManager(registry)
    pm.spawn_variant(shell, mutation_fn=my_mutation)

    assert mutation_called_with == [shell]


def test_spawn_variant_empty_registry_returns_shell(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)

    shell = DummyAgent()
    pm = PolicyManager(registry)
    result = pm.spawn_variant(shell)

    assert result is shell


def test_spawn_variant_int_hyperparams_stay_positive(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    hp = HyperparamSet(params={"batch_size": 32, "n_steps": 128}, episode_id=0)
    registry.save(agent, make_metrics(10.0), hp)

    shell = DummyAgent()
    pm = PolicyManager(registry)
    pm.spawn_variant(shell, noise_scale=2.0)  # extreme noise

    result_hp = shell.get_hyperparams()
    assert result_hp.params["batch_size"] >= 1
    assert result_hp.params["n_steps"] >= 1


# -----------------------------------------------------------------------
# record_agent_score() + auto_update_weights()
# -----------------------------------------------------------------------

class FixedActionAgent:
    def __init__(self, action):
        self._action = action

    def act(self, obs):
        return np.array(self._action, dtype=float)


def _make_pm():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    return pm


def test_auto_update_weights_no_history_is_noop():
    pm = _make_pm()
    pm.add_agent(FixedActionAgent([0.0]), weight=2.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=3.0)
    pm.auto_update_weights()
    assert pm._ensemble[0][1] == 2.0
    assert pm._ensemble[1][1] == 3.0


def test_record_and_auto_update_shifts_weights():
    pm = _make_pm()
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)

    # Agent 0 consistently outperforms agent 1
    for _ in range(5):
        pm.record_agent_score(0, 10.0)
        pm.record_agent_score(1, 2.0)

    pm.auto_update_weights()

    w0 = pm._ensemble[0][1]
    w1 = pm._ensemble[1][1]
    assert w0 > w1


def test_score_window_limits_history():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 3

    for score in [1.0, 2.0, 3.0, 4.0, 5.0]:
        pm.record_agent_score(0, score)

    # Only last 3 scores kept
    assert list(pm._score_history[0]) == [3.0, 4.0, 5.0]


def test_unrecorded_agents_keep_their_weights():
    pm = _make_pm()
    pm.add_agent(FixedActionAgent([0.0]), weight=5.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=7.0)

    pm.record_agent_score(0, 10.0)  # only record for agent 0
    pm.auto_update_weights()

    # Agent 1 weight unchanged
    assert pm._ensemble[1][1] == 7.0


def test_callback_calls_auto_update_weights_on_dormant(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(100.0), make_hp())
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)

    for _ in range(5):
        pm.record_agent_score(0, 10.0)
        pm.record_agent_score(1, 2.0)

    original_w1 = pm._ensemble[1][1]

    cb = pm.as_callback(agent)
    cb.on_episode_end(1, make_metrics(30.0))
    cb.on_dormant(1)

    # Weights should have been rebalanced
    new_w0 = pm._ensemble[0][1]
    new_w1 = pm._ensemble[1][1]
    assert new_w0 > new_w1

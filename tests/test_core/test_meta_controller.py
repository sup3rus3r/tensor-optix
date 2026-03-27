import pytest
import numpy as np
from tensor_optix.core.meta_controller import MetaController, MetaAction
from tensor_optix.core.types import EvalMetrics
from tensor_optix.core.policy_manager import PolicyManager, PolicyManagerCallback
from tensor_optix.core.types import HyperparamSet
from tensor_optix.core.checkpoint_registry import CheckpointRegistry


def make_metrics(primary, train=None, val=None, episode_id=0):
    m = {"score": primary}
    if train is not None:
        m["train_score"] = train
    if val is not None:
        m["val_score"] = val
        m["generalization_gap"] = (train or primary) - val
    return EvalMetrics(primary_score=primary, metrics=m, episode_id=episode_id)


def make_registry(tmp_path):
    return CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)


def make_hp():
    return HyperparamSet(params={"lr": 1e-3}, episode_id=0)


# -----------------------------------------------------------------------
# MetaAction.STOP — budget exhausted
# -----------------------------------------------------------------------

def test_stop_when_budget_exhausted():
    mc = MetaController()
    history = [make_metrics(10.0) for _ in range(5)]
    status = {"budget_exhausted": True}
    assert mc.decide(history, status) == MetaAction.STOP


# -----------------------------------------------------------------------
# MetaAction.PRUNE — large generalization gap
# -----------------------------------------------------------------------

def test_prune_on_large_gap():
    mc = MetaController(gap_threshold=0.1)
    # train=10, val=5 → normalized gap = (10-5)/5 = 1.0 >> 0.1
    history = [make_metrics(5.0, train=10.0, val=5.0, episode_id=i) for i in range(5)]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action == MetaAction.PRUNE


def test_no_prune_when_gap_small():
    mc = MetaController(gap_threshold=0.5)
    # gap ≈ 0
    history = [make_metrics(10.0, train=10.0, val=10.0, episode_id=i) for i in range(5)]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action != MetaAction.PRUNE


# -----------------------------------------------------------------------
# MetaAction.SPAWN — low correlation
# -----------------------------------------------------------------------

def test_spawn_on_low_correlation():
    mc = MetaController(gap_threshold=10.0, corr_threshold=0.5)
    # Train zigzags, val stays flat → low correlation
    history = [
        make_metrics(5.0, train=10.0 + (-1)**i * 5, val=5.0, episode_id=i)
        for i in range(6)
    ]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action == MetaAction.SPAWN


# -----------------------------------------------------------------------
# MetaAction.SPAWN — plateau (low improvement rate)
# -----------------------------------------------------------------------

def test_spawn_on_plateau():
    mc = MetaController(gap_threshold=10.0, corr_threshold=-1.0, improvement_threshold=0.05)
    # Flat scores → slope ≈ 0
    history = [make_metrics(10.0, episode_id=i) for i in range(6)]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action == MetaAction.SPAWN


# -----------------------------------------------------------------------
# MetaAction.NO_OP — healthy system
# -----------------------------------------------------------------------

def test_no_op_when_improving_and_generalizing():
    mc = MetaController(gap_threshold=0.5, corr_threshold=0.3, improvement_threshold=0.01)
    # Both train and val improving together
    history = [
        make_metrics(5.0 + i, train=5.0 + i, val=5.0 + i, episode_id=i)
        for i in range(6)
    ]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action == MetaAction.NO_OP


def test_no_op_with_insufficient_history():
    mc = MetaController()
    history = [make_metrics(10.0), make_metrics(10.0)]
    action = mc.decide(history, {"budget_exhausted": False})
    assert action == MetaAction.NO_OP


# -----------------------------------------------------------------------
# Phase 2: autonomous spawning via agent_factory
# -----------------------------------------------------------------------

class FixedAgent:
    def __init__(self, action=0.0):
        self._action = action
        self._hyperparams = HyperparamSet(params={"lr": 1e-3}, episode_id=0)
        self._weights = {"w": 1.0}

    def act(self, obs):
        return np.array([self._action])

    def learn(self, episode_data):
        return {}

    def get_hyperparams(self):
        return self._hyperparams.copy()

    def set_hyperparams(self, hp):
        self._hyperparams = hp

    def save_weights(self, path):
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "weights.json"), "w") as f:
            json.dump(self._weights, f)

    def load_weights(self, path):
        import json, os
        p = os.path.join(path, "weights.json")
        if os.path.exists(p):
            with open(p) as f:
                self._weights = json.load(f)


def test_autonomous_spawn_on_dormant(tmp_path):
    registry = make_registry(tmp_path)
    agent = FixedAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=2)
    pm.add_agent(agent, weight=1.0)

    factory_calls = []
    def factory():
        a = FixedAgent()
        factory_calls.append(a)
        return a

    cb = pm.as_callback(agent, agent_factory=factory)
    cb.on_episode_end(1, make_metrics(10.0))
    cb.on_dormant(1)

    assert len(factory_calls) == 1
    assert pm.ensemble_size == 2
    assert pm._spawn_count == 1


def test_autonomous_spawn_respects_max_ensemble_size(tmp_path):
    registry = make_registry(tmp_path)
    agent = FixedAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=5, max_ensemble_size=2)
    pm.add_agent(agent, weight=1.0)

    cb = pm.as_callback(agent, agent_factory=FixedAgent)
    cb.on_episode_end(1, make_metrics(10.0))
    cb.on_dormant(1)

    assert pm.ensemble_size <= 2


def test_meta_controller_prune_action_executed(tmp_path):
    registry = make_registry(tmp_path)
    agent = FixedAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=5)
    pm.add_agent(agent, weight=1.0)
    pm.add_agent(FixedAgent(), weight=0.1)  # weak agent — will be pruned

    # Controller that always PRUNEs
    class AlwaysPrune:
        def decide(self, history, status):
            return MetaAction.PRUNE

    cb = pm.as_callback(agent, agent_factory=FixedAgent, meta_controller=AlwaysPrune())
    cb.on_episode_end(1, make_metrics(10.0))
    cb.on_dormant(1)

    assert pm.ensemble_size == 1


def test_meta_controller_no_op_does_not_spawn(tmp_path):
    registry = make_registry(tmp_path)
    agent = FixedAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=5)
    pm.add_agent(agent, weight=1.0)

    class AlwaysNoOp:
        def decide(self, history, status):
            return MetaAction.NO_OP

    cb = pm.as_callback(agent, agent_factory=FixedAgent, meta_controller=AlwaysNoOp())
    cb.on_episode_end(1, make_metrics(10.0))
    cb.on_dormant(1)

    assert pm._spawn_count == 0
    assert pm.ensemble_size == 1


def test_stop_fn_called_by_meta_stop_action(tmp_path):
    registry = make_registry(tmp_path)
    agent = FixedAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=5)

    class AlwaysStop:
        def decide(self, history, status):
            return MetaAction.STOP

    stop_calls = []
    cb = pm.as_callback(agent, agent_factory=FixedAgent, meta_controller=AlwaysStop())
    cb.set_stop_fn(lambda: stop_calls.append(1))
    cb.on_episode_end(1, make_metrics(10.0))
    cb.on_dormant(1)

    assert len(stop_calls) == 1

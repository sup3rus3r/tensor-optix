import pytest
import numpy as np
from tensor_optix.core.types import (
    EpisodeData, EvalMetrics, HyperparamSet, PolicySnapshot, LoopState
)


def test_episode_data_dones():
    ep = EpisodeData(
        observations=np.zeros((3, 4)),
        actions=np.zeros(3),
        rewards=[1.0, 2.0, 3.0],
        terminated=[False, False, True],
        truncated=[False, True, False],
        infos=[{}, {}, {}],
        episode_id=0,
    )
    assert ep.dones == [False, True, True]


def test_episode_data_total_reward():
    ep = EpisodeData(
        observations=np.zeros((3, 4)),
        actions=np.zeros(3),
        rewards=[1.0, 2.0, 3.0],
        terminated=[False, False, True],
        truncated=[False, False, False],
        infos=[{}, {}, {}],
        episode_id=0,
    )
    assert ep.total_reward == 6.0
    assert ep.length == 3


def test_eval_metrics_beats():
    a = EvalMetrics(primary_score=10.0, metrics={}, episode_id=0)
    b = EvalMetrics(primary_score=9.0, metrics={}, episode_id=1)
    assert a.beats(b)
    assert not b.beats(a)
    assert not a.beats(b, margin=1.5)
    assert a.beats(b, margin=0.5)


def test_hyperparam_set_copy_is_independent():
    hp = HyperparamSet(params={"lr": 1e-3, "gamma": 0.99}, episode_id=0)
    hp2 = hp.copy()
    hp2.params["lr"] = 9999.0
    assert hp.params["lr"] == 1e-3


def test_loop_state_enum():
    assert LoopState.ACTIVE != LoopState.DORMANT
    states = list(LoopState)
    assert len(states) == 4

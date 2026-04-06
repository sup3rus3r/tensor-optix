"""
Tests for the cache partial-clear fix in TorchPPOAgent and TFPPOAgent.

Key behaviours verified:
  - After learn(T), exactly T entries are removed; excess entries survive.
  - Cache underflow raises RuntimeError.
  - Normal learn() cycle (act T times → learn T steps) leaves cache empty.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet

OBS_DIM   = 4
N_ACTIONS = 2


def make_agent():
    actor  = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh(), nn.Linear(16, N_ACTIONS))
    critic = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh(), nn.Linear(16, 1))
    opt    = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    return TorchPPOAgent(
        actor=actor, critic=critic, optimizer=opt,
        hyperparams=HyperparamSet(params={
            "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
            "vf_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95,
            "n_epochs": 1, "minibatch_size": 16, "max_grad_norm": 0.5,
        }, episode_id=0),
    )


def make_episode(agent, T):
    obs_list, act_list = [], []
    for _ in range(T):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        obs_list.append(obs)
        act_list.append(agent.act(obs))
    return EpisodeData(
        observations=np.array(obs_list),
        actions=np.array(act_list),
        rewards=[1.0] * T,
        terminated=[False] * (T - 1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


# ── Normal cycle ──────────────────────────────────────────────────────────────

def test_normal_cycle_empties_cache():
    """act T × learn T → cache is empty."""
    agent = make_agent()
    T = 32
    ep = make_episode(agent, T)
    assert len(agent._cache_obs) == T
    agent.learn(ep)
    assert len(agent._cache_obs) == 0
    assert len(agent._cache_log_probs) == 0
    assert len(agent._cache_values) == 0


# ── Residual preservation ─────────────────────────────────────────────────────

def test_residual_entries_survive_after_learn():
    """
    If act() is called T+5 times but learn() only consumes T steps,
    the 5 extra entries must remain in the cache.
    """
    agent  = make_agent()
    T      = 32
    EXTRA  = 5
    ep     = make_episode(agent, T)       # fills cache with T entries

    # Simulate 5 extra act() calls (e.g. from a partially-started next window)
    for _ in range(EXTRA):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    assert len(agent._cache_obs) == T + EXTRA

    agent.learn(ep)   # consumes only T entries

    assert len(agent._cache_obs)      == EXTRA
    assert len(agent._cache_log_probs) == EXTRA
    assert len(agent._cache_values)    == EXTRA


def test_second_learn_uses_residual_correctly():
    """
    Two back-to-back learn() calls: second call must use residual from first.
    """
    agent = make_agent()
    T     = 20

    # First window
    ep1 = make_episode(agent, T)
    agent.learn(ep1)
    assert len(agent._cache_obs) == 0

    # Second window
    ep2 = make_episode(agent, T)
    agent.learn(ep2)
    assert len(agent._cache_obs) == 0


# ── Cache underflow ───────────────────────────────────────────────────────────

def test_learn_raises_on_cache_underflow():
    """learn() with more rewards than act() calls → RuntimeError."""
    agent = make_agent()
    T_act = 10
    T_ep  = 20   # more steps than cached

    for _ in range(T_act):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))

    ep = EpisodeData(
        observations=np.random.rand(T_ep, OBS_DIM).astype(np.float32),
        actions=np.zeros(T_ep, dtype=np.int64),
        rewards=[1.0] * T_ep,
        terminated=[False] * (T_ep - 1) + [True],
        truncated=[False] * T_ep,
        infos=[{}] * T_ep,
        episode_id=0,
    )
    with pytest.raises(RuntimeError, match="Cache underflow"):
        agent.learn(ep)


def test_empty_cache_raises_on_learn():
    """learn() with empty cache → RuntimeError."""
    agent = make_agent()
    ep = EpisodeData(
        observations=np.random.rand(5, OBS_DIM).astype(np.float32),
        actions=np.zeros(5, dtype=np.int64),
        rewards=[1.0] * 5,
        terminated=[False] * 4 + [True],
        truncated=[False] * 5,
        infos=[{}] * 5,
        episode_id=0,
    )
    with pytest.raises(RuntimeError, match="Cache underflow"):
        agent.learn(ep)

"""
Tests for TorchGaussianPPOAgent — continuous action PPO.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from tensor_optix.algorithms.torch_ppo_continuous import TorchGaussianPPOAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet

OBS_DIM    = 8
ACTION_DIM = 2
T          = 64


@pytest.fixture
def hyperparams():
    return HyperparamSet(params={
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       2,
        "minibatch_size": 16,
        "max_grad_norm":  0.5,
    }, episode_id=0)


@pytest.fixture
def agent(hyperparams):
    actor  = nn.Sequential(
        nn.Linear(OBS_DIM, 32), nn.Tanh(),
        nn.Linear(32, 2 * ACTION_DIM),  # mean || log_std
    )
    critic = nn.Sequential(
        nn.Linear(OBS_DIM, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    )
    return TorchGaussianPPOAgent(
        actor=actor, critic=critic,
        optimizer=optimizer,
        action_dim=ACTION_DIM,
        hyperparams=hyperparams,
    )


@pytest.fixture
def episode(agent):
    obs_list, act_list = [], []
    for _ in range(T):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        obs_list.append(obs)
        action = agent.act(obs)
        act_list.append(action)
    return EpisodeData(
        observations=np.array(obs_list),
        actions=np.array(act_list),
        rewards=[float(np.random.randn()) for _ in range(T)],
        terminated=[False] * (T - 1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


# ── act() ─────────────────────────────────────────────────────────────────────

def test_act_returns_float_array(agent):
    obs    = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32


def test_act_bounded_to_minus_one_plus_one(agent):
    """tanh squashing must bound all actions to (-1, 1)."""
    for _ in range(50):
        obs    = np.random.rand(OBS_DIM).astype(np.float32)
        action = agent.act(obs)
        assert np.all(action > -1.0) and np.all(action < 1.0)


def test_act_populates_cache(agent):
    N = 10
    for _ in range(N):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    assert len(agent._cache_obs)       == N
    assert len(agent._cache_log_probs) == N
    assert len(agent._cache_values)    == N


# ── learn() ───────────────────────────────────────────────────────────────────

def test_learn_returns_required_keys(agent, episode):
    diag = agent.learn(episode)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "n_updates"):
        assert key in diag


def test_learn_clears_cache(agent, episode):
    agent.learn(episode)
    assert len(agent._cache_obs) == 0


def test_learn_entropy_positive(agent, episode):
    """Gaussian entropy should be positive (it's differential entropy, unbounded below
    only for degenerate σ→0; with random init σ > 0 so entropy > some constant)."""
    diag = agent.learn(episode)
    # Differential entropy of N(μ, σ) = 0.5*log(2πe σ²) — positive for σ > e^(-0.5)/√(2π) ≈ 0.24
    # Random init σ ≈ exp(0) = 1, so entropy should be clearly positive
    assert diag["entropy"] > 0.0


def test_learn_updates_weights(agent, episode):
    params_before = [p.detach().clone() for p in agent._actor.parameters()]
    agent.learn(episode)
    params_after = list(agent._actor.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed


def test_learn_n_updates_positive(agent, episode):
    diag = agent.learn(episode)
    assert diag["n_updates"] > 0


# ── Residual cache preservation ───────────────────────────────────────────────

def test_residual_entries_preserved(agent):
    """Extra act() calls beyond T must survive learn()."""
    T_ep  = 32
    EXTRA = 7
    obs_list, act_list = [], []

    for _ in range(T_ep + EXTRA):
        obs    = np.random.rand(OBS_DIM).astype(np.float32)
        action = agent.act(obs)
        obs_list.append(obs)
        act_list.append(action)

    ep = EpisodeData(
        observations=np.array(obs_list[:T_ep]),
        actions=np.array(act_list[:T_ep]),
        rewards=[1.0] * T_ep,
        terminated=[False] * (T_ep - 1) + [True],
        truncated=[False] * T_ep,
        infos=[{}] * T_ep,
        episode_id=0,
    )
    agent.learn(ep)
    assert len(agent._cache_obs) == EXTRA


def test_cache_underflow_raises(agent):
    for _ in range(5):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    ep = EpisodeData(
        observations=np.random.rand(20, OBS_DIM).astype(np.float32),
        actions=np.random.rand(20, ACTION_DIM).astype(np.float32),
        rewards=[1.0] * 20,
        terminated=[False] * 19 + [True],
        truncated=[False] * 20,
        infos=[{}] * 20,
        episode_id=0,
    )
    with pytest.raises(RuntimeError, match="Cache underflow"):
        agent.learn(ep)


# ── Hyperparams ───────────────────────────────────────────────────────────────

def test_set_hyperparams_updates_lr(agent):
    new_hp = HyperparamSet(params={"learning_rate": 1e-5}, episode_id=1)
    agent.set_hyperparams(new_hp)
    assert abs(agent._optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12


def test_get_hyperparams_reflects_lr(agent):
    hp = agent.get_hyperparams()
    assert "learning_rate" in hp.params
    assert abs(hp.params["learning_rate"] - 3e-4) < 1e-8


# ── Save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(agent, episode, tmp_path):
    agent.learn(episode)
    original = [p.detach().clone() for p in agent._actor.parameters()]
    path = str(tmp_path / "gaussian_ppo")
    agent.save_weights(path)
    for p in agent._actor.parameters():
        with torch.no_grad():
            p.zero_()
    agent.load_weights(path)
    restored = list(agent._actor.parameters())
    for o, r in zip(original, restored):
        assert torch.allclose(o, r.detach())

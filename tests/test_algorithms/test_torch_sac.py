import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from tensor_optix.algorithms.torch_sac import TorchSACAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM    = 4
ACTION_DIM = 2
T          = 60


def build_actor():
    return nn.Sequential(
        nn.Linear(OBS_DIM, 16), nn.ReLU(),
        nn.Linear(16, ACTION_DIM * 2),
    )


def build_critic():
    return nn.Sequential(
        nn.Linear(OBS_DIM + ACTION_DIM, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


@pytest.fixture
def agent():
    actor   = build_actor()
    c1      = build_critic()
    c2      = build_critic()
    log_alpha = torch.tensor(0.0, requires_grad=True)
    hp = HyperparamSet(params={
        "learning_rate":   3e-4,
        "gamma":           0.99,
        "tau":             0.005,
        "batch_size":      16,
        "updates_per_step": 1,
        "replay_capacity": 1000,
    }, episode_id=0)
    return TorchSACAgent(
        actor=actor, critic1=c1, critic2=c2,
        action_dim=ACTION_DIM,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(
            list(c1.parameters()) + list(c2.parameters()), lr=3e-4
        ),
        alpha_optimizer=torch.optim.Adam([log_alpha], lr=3e-4),
        hyperparams=hp,
    )


@pytest.fixture
def episode():
    return EpisodeData(
        observations=np.random.rand(T, OBS_DIM).astype(np.float32),
        actions=np.random.uniform(-1, 1, (T, ACTION_DIM)).astype(np.float32),
        rewards=[float(np.random.randn()) for _ in range(T)],
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


def test_act_shape_and_range(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)


def test_act_is_stochastic(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    a1 = agent.act(obs)
    a2 = agent.act(obs)
    assert not np.allclose(a1, a2)


def test_learn_fills_buffer(agent, episode):
    agent.learn(episode)
    assert len(agent._buffer) == T - 1


def test_learn_returns_diagnostics(agent, episode):
    diag = agent.learn(episode)
    assert "actor_loss" in diag
    assert "critic_loss" in diag
    assert "alpha" in diag


def test_learn_no_update_when_buffer_small(agent):
    tiny = EpisodeData(
        observations=np.random.rand(3, OBS_DIM).astype(np.float32),
        actions=np.random.uniform(-1, 1, (3, ACTION_DIM)).astype(np.float32),
        rewards=[0.1, 0.2, 0.3],
        terminated=[False, False, True],
        truncated=[False] * 3,
        infos=[{}] * 3,
        episode_id=0,
    )
    diag = agent.learn(tiny)
    assert diag["actor_loss"] == 0.0


def test_alpha_remains_positive(agent, episode):
    for _ in range(4):
        diag = agent.learn(episode)
    assert diag["alpha"] > 0.0


def test_save_load_roundtrip(agent, episode, tmp_path):
    for _ in range(3):
        agent.learn(episode)
    original = [p.detach().clone() for p in agent._actor.parameters()]
    path = str(tmp_path / "sac")
    agent.save_weights(path)
    for p in agent._actor.parameters():
        with torch.no_grad():
            p.zero_()
    agent.load_weights(path)
    for o, r in zip(original, agent._actor.parameters()):
        assert torch.allclose(o, r.detach())

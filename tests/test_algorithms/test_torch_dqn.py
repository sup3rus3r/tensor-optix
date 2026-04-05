import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM   = 4
N_ACTIONS = 2
T         = 50


@pytest.fixture
def agent():
    q_net = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.ReLU(), nn.Linear(16, N_ACTIONS))
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    hp = HyperparamSet(params={
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "epsilon":            1.0,
        "epsilon_min":        0.05,
        "epsilon_decay":      0.9,
        "batch_size":         16,
        "target_update_freq": 5,
        "replay_capacity":    500,
    }, episode_id=0)
    return TorchDQNAgent(q_network=q_net, n_actions=N_ACTIONS,
                         optimizer=optimizer, hyperparams=hp)


@pytest.fixture
def episode():
    return EpisodeData(
        observations=np.random.rand(T, OBS_DIM).astype(np.float32),
        actions=np.random.randint(0, N_ACTIONS, T),
        rewards=[1.0] * (T-1) + [0.0],
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


def test_act_valid_action(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    assert agent.act(obs) in range(N_ACTIONS)


def test_act_random_at_eps_one(agent):
    agent._hyperparams.params["epsilon"] = 1.0
    actions = {agent.act(np.random.rand(OBS_DIM).astype(np.float32)) for _ in range(30)}
    assert len(actions) > 1


def test_act_greedy_at_eps_zero(agent):
    agent._hyperparams.params["epsilon"] = 0.0
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    assert len({agent.act(obs) for _ in range(10)}) == 1


def test_learn_fills_buffer(agent, episode):
    agent.learn(episode)
    assert len(agent._buffer) == T - 1


def test_learn_decays_epsilon(agent, episode):
    eps_before = float(agent._hyperparams.params["epsilon"])
    agent.learn(episode)
    assert float(agent._hyperparams.params["epsilon"]) < eps_before


def test_learn_no_update_when_buffer_small(agent):
    tiny = EpisodeData(
        observations=np.random.rand(3, OBS_DIM).astype(np.float32),
        actions=np.zeros(3, dtype=int),
        rewards=[1.0, 1.0, 0.0],
        terminated=[False, False, True],
        truncated=[False] * 3,
        infos=[{}] * 3,
        episode_id=0,
    )
    diag = agent.learn(tiny)
    assert diag["loss"] == 0.0


def test_save_load_roundtrip(agent, episode, tmp_path):
    for _ in range(3):
        agent.learn(episode)
    original = [p.detach().clone() for p in agent._q.parameters()]
    path = str(tmp_path / "dqn")
    agent.save_weights(path)
    for p in agent._q.parameters():
        with torch.no_grad():
            p.zero_()
    agent.load_weights(path)
    for o, r in zip(original, agent._q.parameters()):
        assert torch.allclose(o, r.detach())

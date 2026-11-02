import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.adapters.pytorch.torch_agent import TorchAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


@pytest.fixture
def agent():
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    hp = HyperparamSet(params={"learning_rate": 1e-3, "gamma": 0.99}, episode_id=0)
    return TorchAgent(model=model, optimizer=optimizer, hyperparams=hp)


@pytest.fixture
def episode():
    T = 16
    return EpisodeData(
        observations=torch.randn(T, 4).numpy(),
        actions=torch.randint(0, 2, (T,)).numpy(),
        rewards=[1.0] * (T - 1) + [0.0],
        terminated=[False] * (T - 1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


def test_resumable_components_declared(agent):
    assert agent.resumable_components == frozenset({"model", "optimizer", "rng"})


def test_save_load_weights_roundtrip(agent, episode, tmp_path):
    agent.learn(episode)
    original = [p.detach().clone() for p in agent.model.parameters()]
    path = str(tmp_path / "weights")
    agent.save_weights(path)
    for p in agent.model.parameters():
        with torch.no_grad():
            p.zero_()
    agent.load_weights(path)
    for o, r in zip(original, agent.model.parameters()):
        assert torch.allclose(o, r.detach())


def test_save_full_state_restores_optimizer_momentum(agent, episode, tmp_path):
    agent.learn(episode)
    assert agent.optimizer.state

    path = str(tmp_path / "full_state")
    agent.save_full_state(path)

    original_state = {
        id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        for p, s in agent.optimizer.state.items()
    }
    for s in agent.optimizer.state.values():
        for k, v in s.items():
            if torch.is_tensor(v):
                v.zero_()

    agent.load_full_state(path)

    for p, s in agent.optimizer.state.items():
        orig = original_state[id(p)]
        for k, v in s.items():
            if torch.is_tensor(v):
                assert torch.allclose(v, orig[k])


def test_load_full_state_is_a_no_op_when_resume_file_missing(agent, episode, tmp_path):
    # A checkpoint saved via plain save_weights() (no resume_state.pt) must
    # not crash load_full_state() -- it should just restore weights, same
    # as calling load_weights() directly.
    agent.learn(episode)
    path = str(tmp_path / "weights_only")
    agent.save_weights(path)  # NOT save_full_state -- no resume_state.pt written

    agent.load_full_state(path)  # must not raise


def test_get_weights_returns_detached_clone(agent):
    weights = agent.get_weights()
    assert set(weights.keys()) == set(agent.model.state_dict().keys())
    first_key = next(iter(weights))
    weights[first_key].zero_()
    assert not torch.equal(weights[first_key], agent.model.state_dict()[first_key])

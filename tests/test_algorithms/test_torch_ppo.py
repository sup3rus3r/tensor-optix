import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM   = 4
N_ACTIONS = 2
T         = 64


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
        "minibatch_size": 32,
        "max_grad_norm":  0.5,
    }, episode_id=0)


@pytest.fixture
def agent(hyperparams):
    actor  = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh(), nn.Linear(16, N_ACTIONS))
    critic = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    )
    return TorchPPOAgent(actor=actor, critic=critic,
                         optimizer=optimizer, hyperparams=hyperparams)


@pytest.fixture
def episode(agent):
    obs_list = []
    act_list = []
    for _ in range(T):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        obs_list.append(obs)
        act_list.append(agent.act(obs))
    return EpisodeData(
        observations=np.array(obs_list),
        actions=np.array(act_list),
        rewards=[1.0] * (T-1) + [0.0],
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


def test_act_returns_valid_action(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert action in range(N_ACTIONS)


def test_act_populates_cache(agent):
    for _ in range(5):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    assert len(agent._cache_obs) == 5
    assert len(agent._cache_log_probs) == 5
    assert len(agent._cache_values) == 5


def test_learn_returns_required_keys(agent, episode):
    diag = agent.learn(episode)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "n_updates"):
        assert key in diag


def test_learn_clears_cache(agent, episode):
    agent.learn(episode)
    assert len(agent._cache_obs) == 0


def test_learn_entropy_positive(agent, episode):
    diag = agent.learn(episode)
    assert diag["entropy"] > 0.0


def test_learn_updates_weights(agent, episode):
    params_before = [p.detach().clone() for p in agent._actor.parameters()]
    agent.learn(episode)
    params_after = list(agent._actor.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed


def test_set_hyperparams_updates_lr(agent):
    new_hp = HyperparamSet(params={"learning_rate": 1e-5}, episode_id=1)
    agent.set_hyperparams(new_hp)
    assert abs(agent._optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12


def test_save_load_roundtrip(agent, episode, tmp_path):
    agent.learn(episode)
    original = [p.detach().clone() for p in agent._actor.parameters()]
    path = str(tmp_path / "torch_ppo")
    agent.save_weights(path)
    for p in agent._actor.parameters():
        with torch.no_grad():
            p.zero_()
    agent.load_weights(path)
    restored = list(agent._actor.parameters())
    for o, r in zip(original, restored):
        assert torch.allclose(o, r.detach())


def test_resumable_components_declared(agent):
    assert agent.resumable_components == frozenset({"model", "optimizer", "rng"})


def test_save_full_state_restores_optimizer_momentum(agent, episode, tmp_path):
    # Adam's momentum (exp_avg) only exists after at least one step().
    agent.learn(episode)
    assert agent._optimizer.state  # sanity: optimizer actually has state now

    path = str(tmp_path / "torch_ppo_full")
    agent.save_full_state(path)

    original_state = {
        id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        for p, s in agent._optimizer.state.items()
    }

    # Corrupt the live optimizer state to prove load_full_state restores it,
    # not just weights.
    for s in agent._optimizer.state.values():
        for k, v in s.items():
            if torch.is_tensor(v):
                v.zero_()

    agent.load_full_state(path)

    for p, s in agent._optimizer.state.items():
        orig = original_state[id(p)]
        for k, v in s.items():
            if torch.is_tensor(v):
                assert torch.allclose(v, orig[k]), f"optimizer state {k!r} was not restored"


def test_load_full_state_restores_rng_state(agent, episode, tmp_path):
    agent.learn(episode)
    path = str(tmp_path / "torch_ppo_rng")
    agent.save_full_state(path)

    torch.manual_seed(999)  # perturb global RNG state
    _ = torch.randn(5)

    agent.load_full_state(path)
    after_load = torch.get_rng_state()

    # A fresh save right after load should reproduce the identical RNG
    # state blob -- proving load actually set it, not left it perturbed.
    path2 = str(tmp_path / "torch_ppo_rng2")
    agent.save_full_state(path2)
    import torch as _torch
    blob1 = _torch.load(str(tmp_path / "torch_ppo_rng" / "resume_state.pt"), weights_only=False)
    blob2 = _torch.load(str(tmp_path / "torch_ppo_rng2" / "resume_state.pt"), weights_only=False)
    assert torch.equal(blob1["torch_rng_state"], blob2["torch_rng_state"])


def test_get_weights_matches_state_dict_without_disk_round_trip(agent, episode):
    agent.learn(episode)
    weights = agent.get_weights()
    assert set(weights["actor"].keys()) == set(agent._actor.state_dict().keys())
    for k, v in agent._actor.state_dict().items():
        assert torch.equal(weights["actor"][k], v)
    # Must be a detached clone, not the live tensor -- mutating it must not
    # affect the model.
    first_key = next(iter(weights["actor"]))
    weights["actor"][first_key].zero_()
    assert not torch.equal(weights["actor"][first_key], agent._actor.state_dict()[first_key])

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensor_optix.algorithms.tf_dqn import TFDQNAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM   = 4
N_ACTIONS = 2
T         = 50


@pytest.fixture
def dqn_hyperparams():
    return HyperparamSet(params={
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "epsilon":            1.0,
        "epsilon_min":        0.05,
        "epsilon_decay":      0.9,   # fast decay for tests
        "batch_size":         16,
        "target_update_freq": 5,
        "replay_capacity":    500,
    }, episode_id=0)


@pytest.fixture
def q_net():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(OBS_DIM,)),
        tf.keras.layers.Dense(N_ACTIONS),
    ])


@pytest.fixture
def agent(q_net, dqn_hyperparams):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    return TFDQNAgent(q_network=q_net, n_actions=N_ACTIONS,
                     optimizer=optimizer, hyperparams=dqn_hyperparams)


@pytest.fixture
def episode():
    obs  = np.random.rand(T, OBS_DIM).astype(np.float32)
    acts = np.random.randint(0, N_ACTIONS, size=T)
    return EpisodeData(
        observations=obs,
        actions=acts,
        rewards=[1.0] * (T - 1) + [0.0],
        terminated=[False] * (T - 1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


# ── act() ──────────────────────────────────────────────────────────────────

def test_act_returns_valid_action(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert action in range(N_ACTIONS)


def test_act_is_random_when_epsilon_one(agent):
    """With epsilon=1.0 every act() should be a random draw."""
    agent._hyperparams.params["epsilon"] = 1.0
    actions = {agent.act(np.random.rand(OBS_DIM).astype(np.float32)) for _ in range(30)}
    assert len(actions) > 1   # should see both actions eventually


def test_act_is_greedy_when_epsilon_zero(agent):
    """With epsilon=0 the action should be deterministic (argmax Q)."""
    agent._hyperparams.params["epsilon"] = 0.0
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    actions = [agent.act(obs) for _ in range(10)]
    assert len(set(actions)) == 1


# ── learn() ────────────────────────────────────────────────────────────────

def test_learn_returns_diagnostics(agent, episode):
    diag = agent.learn(episode)
    assert "epsilon" in diag
    assert "buffer_size" in diag


def test_learn_fills_buffer(agent, episode):
    agent.learn(episode)
    assert len(agent._buffer) == T - 1  # T-1 transitions (no next_obs for last step)


def test_learn_decays_epsilon(agent, episode):
    initial_eps = float(agent._hyperparams.params["epsilon"])
    agent.learn(episode)
    new_eps = float(agent._hyperparams.params["epsilon"])
    assert new_eps < initial_eps


def test_learn_no_update_when_buffer_small(agent):
    """If buffer < batch_size, learn() should return loss=0 and not crash."""
    tiny_episode = EpisodeData(
        observations=np.random.rand(3, OBS_DIM).astype(np.float32),
        actions=np.zeros(3, dtype=int),
        rewards=[1.0, 1.0, 0.0],
        terminated=[False, False, True],
        truncated=[False] * 3,
        infos=[{}] * 3,
        episode_id=0,
    )
    diag = agent.learn(tiny_episode)
    assert diag["loss"] == 0.0


def test_learn_updates_weights_when_buffer_full(agent):
    """After enough samples, weights should change."""
    # Fill buffer first
    for _ in range(5):
        agent.learn(EpisodeData(
            observations=np.random.rand(T, OBS_DIM).astype(np.float32),
            actions=np.random.randint(0, N_ACTIONS, T),
            rewards=[1.0] * T,
            terminated=[False] * (T-1) + [True],
            truncated=[False] * T,
            infos=[{}] * T,
            episode_id=0,
        ))
    weights_before = [v.numpy().copy() for v in agent._q.trainable_variables]
    agent.learn(EpisodeData(
        observations=np.random.rand(T, OBS_DIM).astype(np.float32),
        actions=np.random.randint(0, N_ACTIONS, T),
        rewards=[1.0] * T,
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=1,
    ))
    weights_after = [v.numpy() for v in agent._q.trainable_variables]
    assert any(not np.allclose(b, a) for b, a in zip(weights_before, weights_after))


def test_target_network_updates(agent):
    """After target_update_freq learn() calls, target should match online network."""
    freq = int(agent._hyperparams.params["target_update_freq"])
    ep = EpisodeData(
        observations=np.random.rand(T, OBS_DIM).astype(np.float32),
        actions=np.random.randint(0, N_ACTIONS, T),
        rewards=[1.0] * T,
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )
    # Fill buffer first
    for _ in range(3):
        agent.learn(ep)
    # Now call learn() until the target update happens
    for i in range(freq - (agent._learn_calls % freq)):
        diag = agent.learn(ep)
    assert diag.get("target_updated") == 1


# ── save / load ────────────────────────────────────────────────────────────

def test_save_load_roundtrip(agent, episode, tmp_path):
    agent.learn(episode)
    original = [v.numpy().copy() for v in agent._q.trainable_variables]
    path = str(tmp_path / "dqn")
    agent.save_weights(path)
    for v in agent._q.trainable_variables:
        v.assign(tf.zeros_like(v))
    agent.load_weights(path)
    restored = [v.numpy() for v in agent._q.trainable_variables]
    for o, r in zip(original, restored):
        np.testing.assert_array_almost_equal(o, r)

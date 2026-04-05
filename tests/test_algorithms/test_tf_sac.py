import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensor_optix.algorithms.tf_sac import TFSACAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM    = 4
ACTION_DIM = 2
T          = 60


def build_actor():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(OBS_DIM,)),
        tf.keras.layers.Dense(ACTION_DIM * 2),   # [mean || log_std]
    ])


def build_critic():
    # Input: [obs || action] → scalar Q
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(OBS_DIM + ACTION_DIM,)),
        tf.keras.layers.Dense(1),
    ])


@pytest.fixture
def sac_hyperparams():
    return HyperparamSet(params={
        "learning_rate":   3e-4,
        "gamma":           0.99,
        "tau":             0.005,
        "batch_size":      16,
        "updates_per_step": 1,
        "replay_capacity": 1000,
        "log_alpha_init":  0.0,
    }, episode_id=0)


@pytest.fixture
def agent(sac_hyperparams):
    actor   = build_actor()
    critic1 = build_critic()
    critic2 = build_critic()
    return TFSACAgent(
        actor=actor, critic1=critic1, critic2=critic2,
        action_dim=ACTION_DIM,
        actor_optimizer=tf.keras.optimizers.Adam(3e-4),
        critic_optimizer=tf.keras.optimizers.Adam(3e-4),
        alpha_optimizer=tf.keras.optimizers.Adam(3e-4),
        hyperparams=sac_hyperparams,
    )


@pytest.fixture
def episode():
    obs  = np.random.rand(T, OBS_DIM).astype(np.float32)
    acts = np.random.uniform(-1, 1, size=(T, ACTION_DIM)).astype(np.float32)
    return EpisodeData(
        observations=obs,
        actions=acts,
        rewards=[float(np.random.randn()) for _ in range(T)],
        terminated=[False] * (T-1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
    )


# ── act() ──────────────────────────────────────────────────────────────────

def test_act_returns_continuous_action(agent):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_act_is_stochastic(agent):
    """Two calls on the same obs should produce different actions (almost surely)."""
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    a1 = agent.act(obs)
    a2 = agent.act(obs)
    assert not np.allclose(a1, a2)


# ── learn() ────────────────────────────────────────────────────────────────

def test_learn_returns_diagnostics(agent, episode):
    diag = agent.learn(episode)
    assert "actor_loss" in diag
    assert "critic_loss" in diag
    assert "alpha" in diag
    assert "buffer_size" in diag


def test_learn_fills_buffer(agent, episode):
    agent.learn(episode)
    assert len(agent._buffer) == T - 1


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


def test_learn_updates_actor_weights(agent, episode):
    # Fill buffer first
    for _ in range(3):
        agent.learn(episode)
    before = [v.numpy().copy() for v in agent._actor.trainable_variables]
    agent.learn(episode)
    after = [v.numpy() for v in agent._actor.trainable_variables]
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_alpha_is_positive(agent, episode):
    diag = agent.learn(episode)
    assert diag["alpha"] > 0.0


def test_soft_update_target_differs_from_source(agent, episode):
    """After a soft update, target weights should be a blend (not identical)."""
    # Fill buffer and do updates
    for _ in range(5):
        agent.learn(episode)
    c1_weights = [v.numpy() for v in agent._c1.trainable_variables]
    c1t_weights = [v.numpy() for v in agent._c1_tgt.trainable_variables]
    # Target should NOT be identical to source (tau < 1)
    assert any(not np.allclose(c, t) for c, t in zip(c1_weights, c1t_weights))


# ── save / load ────────────────────────────────────────────────────────────

def test_save_load_roundtrip(agent, episode, tmp_path):
    for _ in range(3):
        agent.learn(episode)
    original = [v.numpy().copy() for v in agent._actor.trainable_variables]
    path = str(tmp_path / "sac")
    agent.save_weights(path)
    for v in agent._actor.trainable_variables:
        v.assign(tf.zeros_like(v))
    agent.load_weights(path)
    restored = [v.numpy() for v in agent._actor.trainable_variables]
    for o, r in zip(original, restored):
        np.testing.assert_array_almost_equal(o, r)


# ── hyperparams ────────────────────────────────────────────────────────────

def test_set_hyperparams_updates_all_optimizers(agent):
    new_hp = HyperparamSet(params={"learning_rate": 1e-5}, episode_id=1)
    agent.set_hyperparams(new_hp)
    for opt in (agent._actor_opt, agent._critic_opt, agent._alpha_opt):
        lr = float(opt.learning_rate.numpy()
                   if hasattr(opt.learning_rate, "numpy") else opt.learning_rate)
        assert abs(lr - 1e-5) < 1e-10

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensor_optix.algorithms.tf_ppo import TFPPOAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


OBS_DIM   = 4
N_ACTIONS = 2
T         = 64  # steps per window


@pytest.fixture
def ppo_hyperparams():
    return HyperparamSet(params={
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       2,        # small for test speed
        "minibatch_size": 32,
        "max_grad_norm":  0.5,
    }, episode_id=0)


@pytest.fixture
def actor():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(OBS_DIM,)),
        tf.keras.layers.Dense(N_ACTIONS),
    ])


@pytest.fixture
def critic():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(OBS_DIM,)),
        tf.keras.layers.Dense(1),
    ])


@pytest.fixture
def agent(actor, critic, ppo_hyperparams):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    return TFPPOAgent(actor=actor, critic=critic,
                     optimizer=optimizer, hyperparams=ppo_hyperparams)


@pytest.fixture
def episode(agent):
    """Drive act() for T steps to populate the cache, then return EpisodeData."""
    obs_list  = []
    act_list  = []
    for _ in range(T):
        obs = np.random.rand(OBS_DIM).astype(np.float32)
        obs_list.append(obs)
        act_list.append(agent.act(obs))
    return EpisodeData(
        observations=np.array(obs_list),
        actions=np.array(act_list),
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


def test_act_populates_cache(agent):
    for _ in range(5):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    assert len(agent._cache_obs) == 5
    assert len(agent._cache_log_probs) == 5
    assert len(agent._cache_values) == 5


def test_act_log_probs_are_negative(agent):
    """Log probabilities under a valid distribution must be ≤ 0."""
    for _ in range(10):
        agent.act(np.random.rand(OBS_DIM).astype(np.float32))
    for lp in agent._cache_log_probs:
        assert lp <= 0.0


# ── learn() ────────────────────────────────────────────────────────────────

def test_learn_returns_required_keys(agent, episode):
    diag = agent.learn(episode)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "n_updates"):
        assert key in diag, f"Missing key: {key}"


def test_learn_clears_cache(agent, episode):
    agent.learn(episode)
    assert len(agent._cache_obs) == 0
    assert len(agent._cache_log_probs) == 0
    assert len(agent._cache_values) == 0


def test_learn_n_updates_matches_epochs(agent, episode):
    diag = agent.learn(episode)
    expected_batches = agent._hyperparams.params["n_epochs"] * (
        T // agent._hyperparams.params["minibatch_size"] + (1 if T % agent._hyperparams.params["minibatch_size"] else 0)
    )
    assert diag["n_updates"] == expected_batches


def test_learn_entropy_is_positive(agent, episode):
    diag = agent.learn(episode)
    assert diag["entropy"] > 0.0


def test_learn_updates_weights(agent, episode):
    weights_before = [v.numpy().copy() for v in agent._actor.trainable_variables]
    agent.learn(episode)
    weights_after = [v.numpy() for v in agent._actor.trainable_variables]
    changed = any(not np.allclose(b, a) for b, a in zip(weights_before, weights_after))
    assert changed, "Actor weights should change after learn()"


# ── hyperparams ────────────────────────────────────────────────────────────

def test_set_hyperparams_updates_lr(agent):
    new_hp = HyperparamSet(params={"learning_rate": 1e-5, "clip_ratio": 0.1}, episode_id=1)
    agent.set_hyperparams(new_hp)
    assert abs(float(agent._optimizer.learning_rate.numpy()) - 1e-5) < 1e-10


def test_get_hyperparams_returns_copy(agent):
    hp1 = agent.get_hyperparams()
    hp1.params["learning_rate"] = 9999.0
    assert agent.get_hyperparams().params["learning_rate"] != 9999.0


# ── save / load ────────────────────────────────────────────────────────────

def test_save_load_roundtrip(agent, episode, tmp_path):
    agent.learn(episode)
    original = [v.numpy().copy() for v in agent._actor.trainable_variables]

    path = str(tmp_path / "ppo_weights")
    agent.save_weights(path)

    # Zero out weights
    for v in agent._actor.trainable_variables:
        v.assign(tf.zeros_like(v))

    agent.load_weights(path)
    restored = [v.numpy() for v in agent._actor.trainable_variables]
    for o, r in zip(original, restored):
        np.testing.assert_array_almost_equal(o, r)

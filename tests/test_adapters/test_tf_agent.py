import pytest
import numpy as np
tf = pytest.importorskip("tensorflow")
from tensor_optix.adapters.tensorflow.tf_agent import TFAgent
from tensor_optix.core.types import HyperparamSet, EpisodeData


@pytest.fixture
def simple_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(2),
    ])


@pytest.fixture
def tf_agent(simple_model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    hyperparams = HyperparamSet(
        params={"learning_rate": 1e-3, "gamma": 0.99},
        episode_id=0,
    )
    return TFAgent(model=simple_model, optimizer=optimizer, hyperparams=hyperparams)


@pytest.fixture
def episode():
    return EpisodeData(
        observations=np.random.rand(10, 4).astype(np.float32),
        actions=np.random.randint(0, 2, size=10),
        rewards=[1.0] * 9 + [0.0],
        terminated=[False] * 9 + [True],
        truncated=[False] * 10,
        infos=[{}] * 10,
        episode_id=0,
    )


def test_act_returns_array(tf_agent):
    obs = np.random.rand(4).astype(np.float32)
    action = tf_agent.act(obs)
    assert action is not None
    assert hasattr(action, "__len__") or isinstance(action, (int, float, np.ndarray))


def test_act_returns_discrete_action(tf_agent):
    obs = np.random.rand(4).astype(np.float32)
    result = tf_agent.act(obs)
    assert isinstance(result, int)
    assert result in (0, 1)


def test_learn_returns_diagnostics(tf_agent, episode):
    diagnostics = tf_agent.learn(episode)
    assert "loss" in diagnostics
    assert "grad_norm" in diagnostics
    assert isinstance(diagnostics["loss"], float)


def test_get_hyperparams_returns_copy(tf_agent):
    hp1 = tf_agent.get_hyperparams()
    hp2 = tf_agent.get_hyperparams()
    hp1.params["learning_rate"] = 9999.0
    assert tf_agent.get_hyperparams().params["learning_rate"] != 9999.0


def test_set_hyperparams_updates_lr(tf_agent):
    new_hp = HyperparamSet(params={"learning_rate": 5e-4, "gamma": 0.95}, episode_id=1)
    tf_agent.set_hyperparams(new_hp)
    lr = float(tf_agent.optimizer.learning_rate.numpy())
    assert abs(lr - 5e-4) < 1e-8


def test_save_load_weights(tf_agent, tmp_path):
    weights_dir = str(tmp_path / "weights")
    tf_agent.save_weights(weights_dir)
    # Corrupt weights to verify load restores them
    original_weights = [w.numpy().copy() for w in tf_agent.model.trainable_variables]
    for var in tf_agent.model.trainable_variables:
        var.assign(tf.zeros_like(var))
    tf_agent.load_weights(weights_dir)
    restored_weights = [w.numpy() for w in tf_agent.model.trainable_variables]
    for orig, restored in zip(original_weights, restored_weights):
        np.testing.assert_array_almost_equal(orig, restored)


def test_custom_loss_fn_is_called(simple_model):
    call_count = {"n": 0}

    def my_loss(model, episode_data, returns):
        call_count["n"] += 1
        obs = tf.cast(episode_data.observations, tf.float32)
        out = model(obs, training=True)
        return tf.reduce_mean(tf.square(out))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    hyperparams = HyperparamSet(params={"learning_rate": 1e-3, "gamma": 0.99}, episode_id=0)
    agent = TFAgent(
        model=simple_model,
        optimizer=optimizer,
        hyperparams=hyperparams,
        compute_loss_fn=my_loss,
    )
    episode = EpisodeData(
        observations=np.random.rand(5, 4).astype(np.float32),
        actions=np.zeros(5, dtype=int),
        rewards=[1.0] * 5,
        terminated=[False] * 4 + [True],
        truncated=[False] * 5,
        infos=[{}] * 5,
        episode_id=0,
    )
    agent.learn(episode)
    assert call_count["n"] == 1

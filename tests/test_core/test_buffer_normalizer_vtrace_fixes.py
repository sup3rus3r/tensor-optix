"""
Replay buffer keeps its contents across uniform <-> prioritized switches;
off-policy agents store the last step of every window (terminal rewards are
no longer dropped); RunningMeanStd ignores empty batches; V-trace accepts an
empty trajectory.
"""
import warnings

import gymnasium as gym
import numpy as np
import pytest

from tensor_optix.core.normalizers import ObsNormalizer, RunningMeanStd
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer, window_transitions
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.distributed.vtrace import compute_vtrace_targets


def _push(buf, start, n):
    for i in range(start, start + n):
        buf.push(np.array([i], dtype=np.float32), 0, float(i),
                 np.array([i + 1], dtype=np.float32), False)


def _stored(buf):
    if buf._alpha == 0:
        return list(buf._uniform_buf)
    return list(buf._tree._data[:buf._tree.size])


# ---------------------------------------------------------------------------
# 8.1 switching sampling modes keeps every stored transition
# ---------------------------------------------------------------------------

class TestModeSwitch:

    def test_uniform_to_prioritized_keeps_everything(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        _push(buf, 0, 50)
        buf.set_params(alpha=0.6)
        assert len(buf) == 50
        assert sorted(t[2] for t in _stored(buf)) == [float(i) for i in range(50)]
        obs, *_ = buf.sample(8)
        assert len(obs) == 8

    def test_prioritized_to_uniform_keeps_everything_in_order(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        _push(buf, 0, 30)
        buf.set_params(alpha=0.6)
        _push(buf, 30, 20)
        buf.set_params(alpha=0.0)
        assert len(buf) == 50
        assert [t[2] for t in _stored(buf)] == [float(i) for i in range(50)]
        buf.sample(8)

    def test_prioritized_to_uniform_after_wraparound(self):
        buf = PrioritizedReplayBuffer(capacity=10, alpha=0.6)
        _push(buf, 0, 15)
        buf.set_params(alpha=0.0)
        assert [t[2] for t in _stored(buf)] == [float(i) for i in range(5, 15)]

    def test_round_trip_twice(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        _push(buf, 0, 10)
        for alpha in (0.5, 0.0, 0.7, 0.0):
            buf.set_params(alpha=alpha)
            assert len(buf) == 10

    def test_changing_alpha_between_positive_values_keeps_tree(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.4)
        _push(buf, 0, 10)
        tree = buf._tree
        buf.set_params(alpha=0.8)
        assert buf._tree is tree and len(buf) == 10

    def test_dqn_set_hyperparams_switch_keeps_buffer(self):
        torch = pytest.importorskip("torch")
        from tensor_optix.factory import make_agent
        agent = make_agent("DQN", gym.make("CartPole-v1"))
        agent.learn(_episode(4, discrete=True, T=40))
        n = len(agent._buffer)
        hp = agent.get_hyperparams()
        hp.params["per_alpha"] = 0.6
        agent.set_hyperparams(hp)
        assert len(agent._buffer) == n


# ---------------------------------------------------------------------------
# 8.3 the last step of every window is stored
# ---------------------------------------------------------------------------

def _episode(obs_dim, discrete, T=6, last_done=True, final_obs=None, act_dim=1):
    rewards = [0.0] * (T - 1) + [10.0]
    actions = (np.zeros(T, dtype=np.int64) if discrete
               else np.zeros((T, act_dim), dtype=np.float32))
    return EpisodeData(
        observations=np.random.randn(T, obs_dim).astype(np.float32),
        actions=actions,
        rewards=rewards,
        terminated=[False] * (T - 1) + [last_done],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
        final_obs=final_obs,
    )


class TestWindowTransitions:

    def test_terminal_last_step_pairs_with_itself(self):
        obs, pairs = window_transitions(_episode(2, True, T=4))
        assert len(obs) == 4
        assert pairs == [[(0, 1), (1, 2), (2, 3), (3, 3)]]

    def test_mid_episode_last_step_pairs_with_final_obs(self):
        final = np.full(2, 7.0, dtype=np.float32)
        obs, pairs = window_transitions(_episode(2, True, T=4, last_done=False, final_obs=final))
        assert len(obs) == 5 and np.array_equal(obs[4], final)
        assert pairs[0][-1] == (3, 4)

    def test_no_final_obs_and_not_done_skips_only_the_last_step(self):
        _, pairs = window_transitions(_episode(2, True, T=4, last_done=False))
        assert pairs == [[(0, 1), (1, 2), (2, 3)]]

    def test_vector_window_per_env_final_obs(self):
        # 2 envs interleaved, 3 steps each: env0 ends terminal, env1 mid-episode
        T, n_envs = 6, 2
        final_env1 = np.full(2, 9.0, dtype=np.float32)
        ep = EpisodeData(
            observations=np.arange(T * 2, dtype=np.float32).reshape(T, 2),
            actions=np.zeros(T, dtype=np.int64),
            rewards=[0.0] * T,
            terminated=[False, False, False, False, True, False],
            truncated=[False] * T, infos=[{}] * T, episode_id=0,
            final_obs=[None, final_env1],
            metadata={"n_envs": n_envs},
        )
        obs, pairs = window_transitions(ep)
        assert pairs[0] == [(0, 2), (2, 4), (4, 4)]
        assert pairs[1] == [(1, 3), (3, 5), (5, 6)]
        assert np.array_equal(obs[6], final_env1)


def _agents():
    yield "torch-DQN", "DQN", "torch", "CartPole-v1"
    yield "torch-Rainbow", "RAINBOW", "torch", "CartPole-v1"
    yield "torch-SAC", "SAC", "torch", "Pendulum-v1"
    yield "torch-TD3", "TD3", "torch", "Pendulum-v1"
    yield "tf-SAC", "SAC", "tf", "Pendulum-v1"
    yield "tf-TD3", "TD3", "tf", "Pendulum-v1"


@pytest.mark.parametrize("name,algo,framework,env_id", list(_agents()), ids=[a[0] for a in _agents()])
def test_agent_keeps_terminal_reward_of_last_step(name, algo, framework, env_id):
    if framework == "tf":
        pytest.importorskip("tensorflow")
    else:
        pytest.importorskip("torch")
    from tensor_optix.factory import make_agent
    env = gym.make(env_id)
    agent = make_agent(algo, env, framework=framework)
    discrete = env_id == "CartPole-v1"
    obs_dim = env.observation_space.shape[0]
    T = 6
    agent.learn(_episode(obs_dim, discrete, T=T))
    stored = _stored(agent._buffer)
    # n-step agents fold the terminal reward into earlier returns too; either
    # way it must reach the buffer.
    assert max(t[2] for t in stored) > 0
    assert len(stored) == T


def test_tf_dqn_keeps_terminal_reward_of_last_step():
    tf = pytest.importorskip("tensorflow")
    from tensor_optix.algorithms.tf_dqn import TFDQNAgent
    q = tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(2)])
    agent = TFDQNAgent(q_network=q, n_actions=2, optimizer=tf.keras.optimizers.Adam(1e-3),
                       hyperparams=HyperparamSet(params={"batch_size": 10**6}, episode_id=0))
    agent.learn(_episode(4, True, T=6))
    stored = _stored(agent._buffer)
    assert len(stored) == 6 and stored[-1][2] == 10.0


def test_dqn_mid_episode_window_stores_final_obs_as_next_state():
    pytest.importorskip("torch")
    from tensor_optix.factory import make_agent
    agent = make_agent("DQN", gym.make("CartPole-v1"))
    final = np.full(4, 3.0, dtype=np.float32)
    agent.learn(_episode(4, True, T=6, last_done=False, final_obs=final))
    stored = _stored(agent._buffer)
    assert len(stored) == 6
    assert np.allclose(stored[-1][3], final) and stored[-1][4] == 0.0


def test_batch_pipeline_window_stores_every_step():
    pytest.importorskip("torch")
    from tensor_optix.factory import make_agent
    from tensor_optix.pipeline.batch_pipeline import BatchPipeline
    env = gym.make("CartPole-v1")
    agent = make_agent("DQN", env)
    pipe = BatchPipeline(env=env, agent=agent, window_size=50)
    pipe.setup()
    window = next(iter(pipe.episodes()))
    agent.learn(window)
    assert len(agent._buffer) == 50
    pipe.teardown()


# ---------------------------------------------------------------------------
# 8.4 empty batches leave the running statistics untouched
# ---------------------------------------------------------------------------

class TestNormalizerEmptyBatch:

    def test_empty_batch_is_ignored(self):
        norm = ObsNormalizer(obs_shape=(3,))
        norm.update(np.random.randn(100, 3))
        mean, var = norm.mean.copy(), norm.var.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # no "mean of empty slice" either
            norm.update(np.zeros((0, 3)))
            norm.update(np.array([]))
        assert np.array_equal(norm.mean, mean) and np.array_equal(norm.var, var)
        norm.update(np.random.randn(100, 3))
        assert np.isfinite(norm.mean).all() and np.isfinite(norm.normalize(np.ones(3))).all()

    def test_scalar_stats_ignore_empty(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([1.0, 2.0, 3.0]))
        rms.update(np.array([]))
        assert rms.mean == pytest.approx(2.0, abs=1e-3)

    def test_nonempty_batches_unchanged(self):
        x = np.random.randn(500, 3)
        rms = RunningMeanStd(shape=(3,))
        for chunk in np.array_split(x, 7):
            rms.update(chunk)
        assert np.allclose(rms.mean, x.mean(0), atol=1e-4)
        assert np.allclose(rms.var, x.var(0), atol=1e-3)


# ---------------------------------------------------------------------------
# 8.5 V-trace on an empty trajectory
# ---------------------------------------------------------------------------

def test_vtrace_empty_trajectory_returns_empty_arrays():
    vs, adv = compute_vtrace_targets(
        rewards=np.zeros(0, np.float32), values=np.zeros(1, np.float32),
        behaviour_log_probs=np.zeros(0, np.float32), current_log_probs=np.zeros(0, np.float32),
        dones=np.zeros(0, bool),
    )
    assert vs.shape == (0,) and adv.shape == (0,)
    assert vs.dtype == np.float32 and adv.dtype == np.float32


def test_vtrace_nonempty_still_computes():
    T = 5
    vs, adv = compute_vtrace_targets(
        rewards=np.ones(T, np.float32), values=np.zeros(T + 1, np.float32),
        behaviour_log_probs=np.zeros(T, np.float32), current_log_probs=np.zeros(T, np.float32),
        dones=np.zeros(T, bool), gamma=0.0,
    )
    assert np.allclose(vs, 1.0) and np.allclose(adv, 1.0)

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class _ReplayBuffer:
    """Fixed-capacity circular buffer of (obs, action, reward, next_obs, done) tuples."""

    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int32),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class TFDQNAgent(BaseAgent):
    """
    DQN (Deep Q-Network) agent for TensorFlow with discrete action spaces.

    Implements: experience replay, target network with periodic hard updates,
    and epsilon-greedy exploration.

    Architecture: a single Q-network mapping obs → Q-values [n_actions].
    The target network is a copy updated every target_update_freq learn() calls.

    Usage:
        q_net = tf.keras.Sequential([...])   # obs → Q-values [n_actions]
        agent = TFDQNAgent(
            q_network=q_net,
            n_actions=4,
            optimizer=tf.keras.optimizers.Adam(1e-3),
            hyperparams=HyperparamSet(params={
                "learning_rate":      1e-3,
                "gamma":              0.99,
                "epsilon":            1.0,
                "epsilon_min":        0.05,
                "epsilon_decay":      0.995,
                "batch_size":         64,
                "target_update_freq": 100,
                "replay_capacity":    100_000,
            }, episode_id=0),
        )

    Note: epsilon decays multiplicatively each time learn() is called
    (once per window/episode). Set epsilon_decay=1.0 to disable decay.

    Hyperparams tuned by the framework:
        learning_rate, gamma, epsilon, epsilon_min, epsilon_decay,
        batch_size, target_update_freq
    """

    def __init__(
        self,
        q_network: tf.keras.Model,
        n_actions: int,
        optimizer: tf.keras.optimizers.Optimizer,
        hyperparams: HyperparamSet,
    ):
        self._q = q_network
        self._q_target = tf.keras.models.clone_model(q_network)
        self._q_target.set_weights(q_network.get_weights())
        self._n_actions = n_actions
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

        capacity = int(hyperparams.params.get("replay_capacity", 100_000))
        self._buffer = _ReplayBuffer(capacity)
        self._learn_calls = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> int:
        """
        Epsilon-greedy action selection.
        With probability epsilon: random action.
        Otherwise: argmax Q(s, ·) from the online network.
        """
        epsilon = float(self._hyperparams.params.get("epsilon", 0.1))
        if np.random.random() < epsilon:
            return np.random.randint(self._n_actions)
        obs = tf.cast(np.atleast_2d(observation), tf.float32)
        q_values = self._q(obs, training=False)
        return int(tf.argmax(q_values, axis=-1).numpy()[0])

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Push the collected window to the replay buffer, then perform
        one gradient update (if buffer has enough samples).

        Target:  y_t = r_t + γ · max_a Q_target(s_{t+1}, a) · (1 - done_t)
        Loss:    MSE(Q(s_t, a_t), y_t)

        Target network is hard-updated every target_update_freq calls.
        """
        hp = self._hyperparams.params
        gamma              = float(hp.get("gamma",              0.99))
        batch_size         = int(hp.get("batch_size",           64))
        target_update_freq = int(hp.get("target_update_freq",   100))
        epsilon_decay      = float(hp.get("epsilon_decay",      0.995))
        epsilon_min        = float(hp.get("epsilon_min",        0.05))

        # Push transitions to replay buffer
        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = episode_data.actions
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones

        T = len(rew_arr)
        for t in range(T - 1):
            self._buffer.push(
                obs_arr[t],
                int(act_arr[t]),
                float(rew_arr[t]),
                obs_arr[t + 1],
                bool(done_arr[t]),
            )

        self._learn_calls += 1

        # Decay epsilon
        current_eps = float(hp.get("epsilon", 1.0))
        new_eps = max(epsilon_min, current_eps * epsilon_decay)
        self._hyperparams.params["epsilon"] = new_eps

        if len(self._buffer) < batch_size:
            return {
                "loss":         0.0,
                "epsilon":      new_eps,
                "buffer_size":  len(self._buffer),
                "target_updates": 0,
            }

        obs_b, act_b, rew_b, next_obs_b, done_b = self._buffer.sample(batch_size)
        obs_b      = tf.cast(obs_b,      tf.float32)
        act_b      = tf.cast(act_b,      tf.int32)
        rew_b      = tf.cast(rew_b,      tf.float32)
        next_obs_b = tf.cast(next_obs_b, tf.float32)
        done_b     = tf.cast(done_b,     tf.float32)

        # Compute TD targets using the frozen target network
        next_q   = self._q_target(next_obs_b, training=False)  # [batch, n_actions]
        max_next = tf.reduce_max(next_q, axis=-1)               # [batch]
        targets  = rew_b + gamma * max_next * (1.0 - done_b)   # [batch]

        with tf.GradientTape() as tape:
            q_vals = self._q(obs_b, training=True)              # [batch, n_actions]
            # Gather Q(s, a) for the taken actions
            batch_idx = tf.range(tf.shape(act_b)[0])
            indices   = tf.stack([batch_idx, act_b], axis=1)
            q_taken   = tf.gather_nd(q_vals, indices)           # [batch]
            loss      = tf.reduce_mean(tf.square(q_taken - tf.stop_gradient(targets)))

        grads = tape.gradient(loss, self._q.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q.trainable_variables))

        # Hard target network update
        target_updated = False
        if self._learn_calls % target_update_freq == 0:
            self._q_target.set_weights(self._q.get_weights())
            target_updated = True

        return {
            "loss":           float(loss.numpy()),
            "epsilon":        new_eps,
            "buffer_size":    len(self._buffer),
            "target_updated": int(target_updated),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._optimizer.learning_rate.numpy()
            if hasattr(self._optimizer.learning_rate, "numpy")
            else self._optimizer.learning_rate
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            self._optimizer.learning_rate.assign(float(hyperparams.params["learning_rate"]))

    def save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._q.save(os.path.join(path, "q_network.keras"))
        self._q_target.save(os.path.join(path, "q_target.keras"))

    def load_weights(self, path: str) -> None:
        loaded_q = tf.keras.models.load_model(os.path.join(path, "q_network.keras"))
        for v, lv in zip(self._q.trainable_variables, loaded_q.trainable_variables):
            v.assign(lv)
        # Sync target from loaded online network
        self._q_target.set_weights(self._q.get_weights())

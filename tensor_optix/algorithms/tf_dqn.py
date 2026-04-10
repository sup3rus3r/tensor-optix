import os
from typing import Optional

import numpy as np
import tensorflow as tf

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer


class TFDQNAgent(BaseAgent):
    """
    DQN (Deep Q-Network) agent for TensorFlow with discrete action spaces.

    Implements: Prioritized Experience Replay (PER), n-step returns,
    target network with periodic hard updates, epsilon-greedy exploration.

    Architecture: a single Q-network mapping obs → Q-values [n_actions].
    The target network is a copy updated every target_update_freq learn() calls.

    PER and n-step params are exposed in hyperparams so SPSA can adapt them:
        per_alpha      — prioritization strength (0=uniform, 1=full PER)
        per_beta       — IS correction exponent
        n_step         — multi-step TD target length

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
                "per_alpha":          0.0,
                "per_beta":           0.4,
                "n_step":             1,
            }, episode_id=0),
        )
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

        hp = hyperparams.params
        capacity  = int(hp.get("replay_capacity", 100_000))
        per_alpha = float(hp.get("per_alpha", 0.0))
        per_beta  = float(hp.get("per_beta",  0.4))
        n_step    = int(hp.get("n_step",    1))
        gamma     = float(hp.get("gamma",   0.99))

        self._buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=per_alpha,
            beta=per_beta,
            n_step=n_step,
            gamma=gamma,
        )
        self._learn_calls = 0

    default_param_bounds = {
        "learning_rate": (1e-4, 1e-3),
        "gamma":         (0.95, 0.999),
        # epsilon_decay intentionally excluded: schedule parameter, not optimisable.
        # SPSA random-walks it toward 0.999, preventing epsilon from decaying.
    }
    default_log_params = ["learning_rate"]

    # With default epsilon_decay=0.95, epsilon reaches floor at episode ~58.
    # DORMANT must not fire before then — a stalled agent before the floor is reached
    # is still exploring, not genuinely stuck.
    default_min_episodes_before_dormant = 60

    @property
    def is_on_policy(self) -> bool:
        return False

    def act(self, observation) -> int:
        epsilon = float(self._hyperparams.params.get("epsilon", 0.1))
        if np.random.random() < epsilon:
            return np.random.randint(self._n_actions)
        obs = tf.cast(np.atleast_2d(observation), tf.float32)
        q_values = self._q(obs, training=False)
        return int(tf.argmax(q_values, axis=-1).numpy()[0])

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        gamma              = float(hp.get("gamma",              0.99))
        batch_size         = int(hp.get("batch_size",           64))
        target_update_freq = int(hp.get("target_update_freq",   100))
        epsilon_decay      = float(hp.get("epsilon_decay",      0.995))
        epsilon_min        = float(hp.get("epsilon_min",        0.05))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = episode_data.actions
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones

        T = len(rew_arr)
        for t in range(T - 1):
            self._buffer.push(
                obs_arr[t], int(act_arr[t]), float(rew_arr[t]),
                obs_arr[t + 1], bool(done_arr[t]),
            )
        self._buffer.flush_episode()

        self._learn_calls += 1
        new_eps = max(epsilon_min, float(hp.get("epsilon", 1.0)) * epsilon_decay)
        self._hyperparams.params["epsilon"] = new_eps

        if len(self._buffer) < batch_size:
            return {"loss": 0.0, "epsilon": new_eps, "buffer_size": len(self._buffer)}

        # Gradient updates proportional to steps collected: 1 update per batch_size steps.
        n_updates = max(1, (T - 1) // batch_size)

        total_loss     = 0.0
        total_td_error = 0.0
        target_updated = False

        for _ in range(n_updates):
            obs_b, act_b, rew_b, next_obs_b, done_b, weights, indices, n_steps = self._buffer.sample(batch_size)
            obs_b      = tf.cast(obs_b,      tf.float32)
            act_b      = tf.cast(act_b,      tf.int32)
            rew_b      = tf.cast(rew_b,      tf.float32)
            next_obs_b = tf.cast(next_obs_b, tf.float32)
            done_b     = tf.cast(done_b,     tf.float32)
            weights    = tf.cast(weights,    tf.float32)
            gammas_n   = tf.cast(gamma ** n_steps.astype(np.float32), tf.float32)

            next_q   = self._q_target(next_obs_b, training=False)
            max_next = tf.reduce_max(next_q, axis=-1)
            targets  = rew_b + gammas_n * max_next * (1.0 - done_b)

            with tf.GradientTape() as tape:
                q_vals    = self._q(obs_b, training=True)
                batch_idx = tf.range(tf.shape(act_b)[0])
                gather_idx = tf.stack([batch_idx, act_b], axis=1)
                q_taken   = tf.gather_nd(q_vals, gather_idx)
                td_errors_tf = q_taken - tf.stop_gradient(targets)
                loss      = tf.reduce_mean(weights * tf.square(td_errors_tf))

            grads = tape.gradient(loss, self._q.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            self._optimizer.apply_gradients(zip(grads, self._q.trainable_variables))

            td_errors = td_errors_tf.numpy()
            self._buffer.update_priorities(indices, np.abs(td_errors))

            total_loss     += float(loss.numpy())
            total_td_error += float(np.abs(td_errors).mean())

        # Target network update counted per window (not per gradient step)
        if self._learn_calls % target_update_freq == 0:
            self._q_target.set_weights(self._q.get_weights())
            target_updated = True

        episode_reward = float(np.sum(episode_data.rewards)) if episode_data.rewards is not None else None
        return {
            "loss":           total_loss / n_updates,
            "epsilon":        new_eps,
            "buffer_size":    len(self._buffer),
            "target_updated": int(target_updated),
            "td_error_mean":  total_td_error / n_updates,
            "episode_reward": episode_reward,
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._optimizer.learning_rate.numpy()
            if hasattr(self._optimizer.learning_rate, "numpy")
            else self._optimizer.learning_rate
        )
        self._hyperparams.params["per_alpha"] = self._buffer._alpha
        self._hyperparams.params["per_beta"]  = self._buffer._beta
        self._hyperparams.params["n_step"]    = self._buffer._n_step
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        hp = hyperparams.params
        if "learning_rate" in hp:
            self._optimizer.learning_rate.assign(float(hp["learning_rate"]))
        self._buffer.set_params(
            alpha=hp.get("per_alpha"),
            beta=hp.get("per_beta"),
            n_step=hp.get("n_step"),
            gamma=hp.get("gamma"),
        )

    def save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._q.save(os.path.join(path, "q_network.keras"))
        self._q_target.save(os.path.join(path, "q_target.keras"))

    def load_weights(self, path: str) -> None:
        loaded_q = tf.keras.models.load_model(os.path.join(path, "q_network.keras"))
        for v, lv in zip(self._q.trainable_variables, loaded_q.trainable_variables):
            v.assign(lv)
        self._q_target.set_weights(self._q.get_weights())

    def average_weights(self, paths: list) -> None:
        loaded = [tf.keras.models.load_model(os.path.join(p, "q_network.keras")) for p in paths]
        for v, *lvs in zip(self._q.trainable_variables, *[m.trainable_variables for m in loaded]):
            v.assign(tf.reduce_mean(tf.stack([lv for lv in lvs], axis=0), axis=0))
        self._q_target.set_weights(self._q.get_weights())

    def perturb_weights(self, noise_scale: float) -> None:
        for v in self._q.trainable_variables:
            v.assign(v * (1.0 + noise_scale * tf.random.normal(v.shape)))
        self._q_target.set_weights(self._q.get_weights())

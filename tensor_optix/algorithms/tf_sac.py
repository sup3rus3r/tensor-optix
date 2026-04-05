import os
import random
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class _ReplayBuffer:
    """Replay buffer for continuous-action off-policy methods."""

    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self._buf.append((
            np.array(obs,      dtype=np.float32),
            np.array(action,   dtype=np.float32),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.float32),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class TFSACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) agent for TensorFlow continuous action spaces.

    Implements the entropy-regularized actor-critic (Haarnoja et al. 2018) with:
    - Gaussian actor with reparameterization trick and tanh squashing
    - Twin Q-critics (clipped double-Q) to reduce overestimation bias
    - Soft target network updates (Polyak averaging)
    - Automatic entropy temperature tuning (learnable log_alpha)

    Architecture:
        actor:   tf.keras.Model, obs → [mean, log_std] concatenated, shape [batch, 2*action_dim]
        critic1: tf.keras.Model, [obs, action] concatenated → Q-value, shape [batch, 1]
        critic2: tf.keras.Model, same as critic1 (independent weights for twin-Q)

    Target entropy heuristic (Haarnoja et al.): -action_dim

    Usage:
        actor   = build_actor(obs_dim, action_dim)    # outputs [mean||log_std]
        critic1 = build_critic(obs_dim, action_dim)   # takes [obs||action]
        critic2 = build_critic(obs_dim, action_dim)
        agent = TFSACAgent(
            actor=actor, critic1=critic1, critic2=critic2,
            action_dim=action_dim,
            actor_optimizer=tf.keras.optimizers.Adam(3e-4),
            critic_optimizer=tf.keras.optimizers.Adam(3e-4),
            alpha_optimizer=tf.keras.optimizers.Adam(3e-4),
            hyperparams=HyperparamSet(params={
                "learning_rate":   3e-4,
                "gamma":           0.99,
                "tau":             0.005,
                "batch_size":      256,
                "log_alpha_init":  0.0,
                "replay_capacity": 1_000_000,
                "updates_per_step": 1,
            }, episode_id=0),
        )

    Actions are squashed to (-1, 1) via tanh. Scale to your env's action range in a wrapper.

    Hyperparams tuned by the framework:
        learning_rate (applied to all three optimizers), gamma, tau, batch_size
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        actor: tf.keras.Model,
        critic1: tf.keras.Model,
        critic2: tf.keras.Model,
        action_dim: int,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        alpha_optimizer: tf.keras.optimizers.Optimizer,
        hyperparams: HyperparamSet,
    ):
        self._actor    = actor
        self._c1       = critic1
        self._c2       = critic2
        self._c1_tgt   = tf.keras.models.clone_model(critic1)
        self._c2_tgt   = tf.keras.models.clone_model(critic2)
        self._c1_tgt.set_weights(critic1.get_weights())
        self._c2_tgt.set_weights(critic2.get_weights())

        self._action_dim    = action_dim
        self._actor_opt     = actor_optimizer
        self._critic_opt    = critic_optimizer
        self._alpha_opt     = alpha_optimizer
        self._hyperparams   = hyperparams.copy()

        log_alpha_init = float(hyperparams.params.get("log_alpha_init", 0.0))
        self._log_alpha = tf.Variable(log_alpha_init, dtype=tf.float32, trainable=True)
        self._target_entropy = tf.constant(-float(action_dim), dtype=tf.float32)

        capacity = int(hyperparams.params.get("replay_capacity", 1_000_000))
        self._buffer = _ReplayBuffer(capacity)

        # Track last obs for building (s, a, r, s') transitions
        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> np.ndarray:
        """
        Sample action from the squashed Gaussian policy.
        Returns np.ndarray of shape [action_dim], values in (-1, 1).
        """
        obs = tf.cast(np.atleast_2d(observation), tf.float32)
        action, _ = self._sample_action(obs, training=False)
        self._last_obs = np.array(observation, dtype=np.float32)
        return action.numpy()[0]

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Push window transitions to the replay buffer, then perform
        updates_per_step gradient steps per window step (if buffer ready).
        """
        hp = self._hyperparams.params
        gamma             = float(hp.get("gamma",             0.99))
        tau               = float(hp.get("tau",               0.005))
        batch_size        = int(hp.get("batch_size",          256))
        updates_per_step  = int(hp.get("updates_per_step",    1))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = np.array(episode_data.actions,      dtype=np.float32)
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        for t in range(T - 1):
            self._buffer.push(
                obs_arr[t],
                act_arr[t],
                float(rew_arr[t]),
                obs_arr[t + 1],
                float(done_arr[t]),
            )

        if len(self._buffer) < batch_size:
            return {
                "actor_loss":  0.0,
                "critic_loss": 0.0,
                "alpha":       float(tf.exp(self._log_alpha).numpy()),
                "buffer_size": len(self._buffer),
            }

        n_updates = T * updates_per_step
        total_actor_loss  = 0.0
        total_critic_loss = 0.0

        for _ in range(n_updates):
            al, cl = self._update_step(batch_size, gamma, tau)
            total_actor_loss  += al
            total_critic_loss += cl

        return {
            "actor_loss":  total_actor_loss  / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "alpha":       float(tf.exp(self._log_alpha).numpy()),
            "buffer_size": len(self._buffer),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._actor_opt.learning_rate.numpy()
            if hasattr(self._actor_opt.learning_rate, "numpy")
            else self._actor_opt.learning_rate
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            self._actor_opt.learning_rate.assign(lr)
            self._critic_opt.learning_rate.assign(lr)
            self._alpha_opt.learning_rate.assign(lr)

    def save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.keras"))
        self._c1.save(os.path.join(path,    "critic1.keras"))
        self._c2.save(os.path.join(path,    "critic2.keras"))
        np.save(os.path.join(path, "log_alpha.npy"), self._log_alpha.numpy())

    def load_weights(self, path: str) -> None:
        la = tf.keras.models.load_model(os.path.join(path, "actor.keras"))
        lc1 = tf.keras.models.load_model(os.path.join(path, "critic1.keras"))
        lc2 = tf.keras.models.load_model(os.path.join(path, "critic2.keras"))
        for v, lv in zip(self._actor.trainable_variables, la.trainable_variables):
            v.assign(lv)
        for v, lv in zip(self._c1.trainable_variables, lc1.trainable_variables):
            v.assign(lv)
        for v, lv in zip(self._c2.trainable_variables, lc2.trainable_variables):
            v.assign(lv)
        log_alpha_path = os.path.join(path, "log_alpha.npy")
        if os.path.exists(log_alpha_path):
            self._log_alpha.assign(float(np.load(log_alpha_path)))
        # Sync target networks from loaded weights
        self._c1_tgt.set_weights(self._c1.get_weights())
        self._c2_tgt.set_weights(self._c2.get_weights())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_action(self, obs_tf, training: bool):
        """
        Reparameterized squashed Gaussian sample.

        Returns:
            action:   tanh(mean + std * ε),  shape [batch, action_dim]
            log_prob: log π(action|obs) with tanh correction, shape [batch]
        """
        out = self._actor(obs_tf, training=training)        # [batch, 2*action_dim]
        mean, log_std = tf.split(out, 2, axis=-1)
        log_std = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.exp(log_std)

        # Reparameterization: u ~ N(mean, std)
        eps    = tf.random.normal(tf.shape(mean))
        u      = mean + std * eps
        action = tf.tanh(u)

        # Log prob under squashed Gaussian (sum over action dims)
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u_i) + ε)
        gaussian_log_prob = -0.5 * (
            ((u - mean) / (std + 1e-8)) ** 2
            + 2.0 * log_std
            + tf.math.log(2.0 * np.pi)
        )
        log_prob = tf.reduce_sum(gaussian_log_prob, axis=-1)
        log_prob -= tf.reduce_sum(
            tf.math.log(1.0 - action ** 2 + 1e-6), axis=-1
        )

        return action, log_prob

    def _critic_input(self, obs, action):
        """Concatenate obs and action for the Q-network input."""
        return tf.concat([obs, action], axis=-1)

    def _update_step(self, batch_size: int, gamma: float, tau: float):
        """One full SAC gradient step (critic + actor + alpha + soft update)."""
        obs_b, act_b, rew_b, next_obs_b, done_b = self._buffer.sample(batch_size)
        obs_b      = tf.cast(obs_b,      tf.float32)
        act_b      = tf.cast(act_b,      tf.float32)
        rew_b      = tf.cast(rew_b,      tf.float32)
        next_obs_b = tf.cast(next_obs_b, tf.float32)
        done_b     = tf.cast(done_b,     tf.float32)
        alpha      = tf.exp(self._log_alpha)

        # ---- Critic update ----
        with tf.GradientTape() as tape:
            # Target: r + γ(1-d)(min Q_tgt(s', ã') - α log π(ã'|s'))
            next_a, next_lp = self._sample_action(next_obs_b, training=False)
            ci_next = self._critic_input(next_obs_b, next_a)
            q1_tgt  = tf.squeeze(self._c1_tgt(ci_next, training=False), -1)
            q2_tgt  = tf.squeeze(self._c2_tgt(ci_next, training=False), -1)
            min_q   = tf.minimum(q1_tgt, q2_tgt)
            q_tgt   = rew_b + gamma * (1.0 - done_b) * (min_q - alpha * next_lp)
            q_tgt   = tf.stop_gradient(q_tgt)

            ci      = self._critic_input(obs_b, act_b)
            q1      = tf.squeeze(self._c1(ci, training=True), -1)
            q2      = tf.squeeze(self._c2(ci, training=True), -1)
            c_loss  = tf.reduce_mean(tf.square(q1 - q_tgt)) + \
                      tf.reduce_mean(tf.square(q2 - q_tgt))

        c_vars = self._c1.trainable_variables + self._c2.trainable_variables
        c_grads = tape.gradient(c_loss, c_vars)
        self._critic_opt.apply_gradients(zip(c_grads, c_vars))

        # ---- Actor update ----
        with tf.GradientTape() as tape:
            new_a, new_lp = self._sample_action(obs_b, training=True)
            ci_new = self._critic_input(obs_b, new_a)
            q1_new = tf.squeeze(self._c1(ci_new, training=False), -1)
            q2_new = tf.squeeze(self._c2(ci_new, training=False), -1)
            a_loss = tf.reduce_mean(alpha * new_lp - tf.minimum(q1_new, q2_new))

        a_grads = tape.gradient(a_loss, self._actor.trainable_variables)
        self._actor_opt.apply_gradients(zip(a_grads, self._actor.trainable_variables))

        # ---- Alpha (entropy temperature) update ----
        with tf.GradientTape() as tape:
            _, new_lp2 = self._sample_action(obs_b, training=False)
            alpha_loss = -tf.reduce_mean(
                self._log_alpha * tf.stop_gradient(new_lp2 + self._target_entropy)
            )

        alpha_grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_opt.apply_gradients(zip(alpha_grads, [self._log_alpha]))

        # ---- Soft target update (Polyak averaging) ----
        self._soft_update(self._c1, self._c1_tgt, tau)
        self._soft_update(self._c2, self._c2_tgt, tau)

        return float(a_loss.numpy()), float(c_loss.numpy())

    @staticmethod
    def _soft_update(source: tf.keras.Model, target: tf.keras.Model, tau: float) -> None:
        """θ_target ← τ·θ_source + (1-τ)·θ_target"""
        for sv, tv in zip(source.trainable_variables, target.trainable_variables):
            tv.assign(tau * sv + (1.0 - tau) * tv)

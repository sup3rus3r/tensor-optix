"""
TFTDDAgent — Twin Delayed DDPG for TensorFlow continuous action spaces.

See tensor_optix/algorithms/torch_td3.py for the full mathematical derivation.
This module is the TensorFlow mirror of TorchTD3Agent using the same algorithm
and identical hyperparameter interface.

Architecture:
    actor:   tf.keras.Model, obs → tanh(action), shape [batch, action_dim]
    critic1: tf.keras.Model, [obs || action] → Q-value, shape [batch, 1]
    critic2: tf.keras.Model, same architecture, independent weights (twin-Q)

Usage::

    import tensorflow as tf
    from tensor_optix.algorithms.tf_td3 import TFTDDAgent

    obs_dim, act_dim = 8, 2

    actor = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(obs_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(act_dim, activation="tanh"),
    ])
    critic1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(obs_dim + act_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    critic2 = tf.keras.Sequential([...])  # independent weights

    agent = TFTDDAgent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=tf.keras.optimizers.Adam(3e-4),
        critic_optimizer=tf.keras.optimizers.Adam(3e-4),
        hyperparams=HyperparamSet(params={
            "learning_rate":     3e-4,
            "gamma":             0.99,
            "tau":               0.005,
            "batch_size":        256,
            "updates_per_step":  1,
            "replay_capacity":   1_000_000,
            "policy_delay":      2,
            "target_noise":      0.2,
            "target_noise_clip": 0.5,
            "per_alpha":         0.0,
            "per_beta":          0.4,
        }, episode_id=0),
    )
"""

import os

import numpy as np
import tensorflow as tf

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer


class TFTDDAgent(BaseAgent):
    """TD3 agent for TensorFlow continuous action spaces. See module docstring."""

    default_param_bounds = {
        "learning_rate": (1e-4, 1e-3),
        "gamma":         (0.97, 0.999),
        "tau":           (1e-3, 1e-1),
    }
    default_log_params = ["learning_rate", "tau"]
    default_min_episodes_before_dormant = 30

    def __init__(
        self,
        actor: tf.keras.Model,
        critic1: tf.keras.Model,
        critic2: tf.keras.Model,
        action_dim: int,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        hyperparams: HyperparamSet,
    ):
        self._actor   = actor
        self._c1      = critic1
        self._c2      = critic2

        # Target networks: cloned and weight-copied from online networks.
        self._actor_tgt = tf.keras.models.clone_model(actor)
        self._c1_tgt    = tf.keras.models.clone_model(critic1)
        self._c2_tgt    = tf.keras.models.clone_model(critic2)
        self._actor_tgt.set_weights(actor.get_weights())
        self._c1_tgt.set_weights(critic1.get_weights())
        self._c2_tgt.set_weights(critic2.get_weights())

        self._action_dim  = action_dim
        self._actor_opt   = actor_optimizer
        self._critic_opt  = critic_optimizer
        self._hyperparams = hyperparams.copy()
        self._update_count: int = 0

        hp = hyperparams.params
        self._buffer = PrioritizedReplayBuffer(
            capacity=int(hp.get("replay_capacity", 1_000_000)),
            alpha=float(hp.get("per_alpha", 0.0)),
            beta=float(hp.get("per_beta", 0.4)),
            n_step=int(hp.get("n_step", 1)),
            gamma=float(hp.get("gamma", 0.99)),
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        return False

    def act(self, observation) -> np.ndarray:
        """Deterministic action: a = π_θ(s) = tanh(actor(s))."""
        obs = tf.cast(np.atleast_2d(observation), tf.float32)
        action = self._actor(obs, training=False)
        return action.numpy()[0]

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        gamma            = float(hp.get("gamma",            0.99))
        tau              = float(hp.get("tau",              0.005))
        batch_size       = int(hp.get("batch_size",         256))
        updates_per_step = int(hp.get("updates_per_step",   1))
        policy_delay     = int(hp.get("policy_delay",       2))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = np.array(episode_data.actions,      dtype=np.float32)
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        for t in range(T - 1):
            self._buffer.push(
                obs_arr[t], act_arr[t], float(rew_arr[t]),
                obs_arr[t + 1], float(done_arr[t]),
            )
        self._buffer.flush_episode()

        if len(self._buffer) < batch_size:
            return {
                "actor_loss":    0.0,
                "critic_loss":   0.0,
                "buffer_size":   len(self._buffer),
                "policy_update": 0,
            }

        n_updates = T * updates_per_step
        total_al = 0.0
        total_cl = 0.0
        n_actor_updates = 0

        for _ in range(n_updates):
            self._update_count += 1
            update_actor = (self._update_count % policy_delay == 0)
            al, cl = self._update_step(batch_size, gamma, tau, update_actor)
            total_cl += cl
            if update_actor:
                total_al += al
                n_actor_updates += 1

        return {
            "actor_loss":    total_al / max(n_actor_updates, 1),
            "critic_loss":   total_cl / n_updates,
            "buffer_size":   len(self._buffer),
            "policy_update": int(n_actor_updates > 0),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._actor_opt.learning_rate.numpy()
            if hasattr(self._actor_opt.learning_rate, "numpy")
            else self._actor_opt.learning_rate
        )
        self._hyperparams.params["per_alpha"] = self._buffer._alpha
        self._hyperparams.params["per_beta"]  = self._buffer._beta
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        hp = hyperparams.params
        if "learning_rate" in hp:
            lr = float(hp["learning_rate"])
            self._actor_opt.learning_rate.assign(lr)
            self._critic_opt.learning_rate.assign(lr)
        self._buffer.set_params(
            alpha=hp.get("per_alpha"),
            beta=hp.get("per_beta"),
            gamma=hp.get("gamma"),
        )

    def save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.keras"))
        self._c1.save(os.path.join(path,    "critic1.keras"))
        self._c2.save(os.path.join(path,    "critic2.keras"))

    def load_weights(self, path: str) -> None:
        la  = tf.keras.models.load_model(os.path.join(path, "actor.keras"))
        lc1 = tf.keras.models.load_model(os.path.join(path, "critic1.keras"))
        lc2 = tf.keras.models.load_model(os.path.join(path, "critic2.keras"))
        for v, lv in zip(self._actor.trainable_variables, la.trainable_variables):
            v.assign(lv)
        for v, lv in zip(self._c1.trainable_variables, lc1.trainable_variables):
            v.assign(lv)
        for v, lv in zip(self._c2.trainable_variables, lc2.trainable_variables):
            v.assign(lv)
        self._actor_tgt.set_weights(self._actor.get_weights())
        self._c1_tgt.set_weights(self._c1.get_weights())
        self._c2_tgt.set_weights(self._c2.get_weights())

    def average_weights(self, paths: list) -> None:
        for net, fname in (
            (self._actor, "actor.keras"),
            (self._c1,    "critic1.keras"),
            (self._c2,    "critic2.keras"),
        ):
            loaded = [tf.keras.models.load_model(os.path.join(p, fname)) for p in paths]
            for v, *lvs in zip(
                net.trainable_variables,
                *[m.trainable_variables for m in loaded],
            ):
                v.assign(tf.reduce_mean(tf.stack(list(lvs), axis=0), axis=0))
        self._actor_tgt.set_weights(self._actor.get_weights())
        self._c1_tgt.set_weights(self._c1.get_weights())
        self._c2_tgt.set_weights(self._c2.get_weights())

    def perturb_weights(self, noise_scale: float) -> None:
        for module in (self._actor, self._c1, self._c2):
            for v in module.trainable_variables:
                v.assign(v * (1.0 + noise_scale * tf.random.normal(v.shape)))
        self._actor_tgt.set_weights(self._actor.get_weights())
        self._c1_tgt.set_weights(self._c1.get_weights())
        self._c2_tgt.set_weights(self._c2.get_weights())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_step(
        self,
        batch_size: int,
        gamma: float,
        tau: float,
        update_actor: bool,
    ):
        """One TD3 gradient step. See TorchTD3Agent._update_step for derivation."""
        hp = self._hyperparams.params
        target_noise      = float(hp.get("target_noise",      0.2))
        target_noise_clip = float(hp.get("target_noise_clip", 0.5))

        obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = \
            self._buffer.sample(batch_size)

        obs_b    = tf.cast(obs_b,    tf.float32)
        act_b    = tf.cast(act_b,    tf.float32)
        rew_b    = tf.cast(rew_b,    tf.float32)
        next_b   = tf.cast(next_b,   tf.float32)
        done_b   = tf.cast(done_b,   tf.float32)
        weights  = tf.cast(weights,  tf.float32)
        gammas_n = tf.cast(gamma ** n_steps.astype(np.float32), tf.float32)

        # ---- Critic update ----
        with tf.GradientTape() as tape:
            # Fix 3: target policy smoothing
            noise = tf.clip_by_value(
                tf.random.normal(tf.shape(act_b), stddev=target_noise),
                -target_noise_clip,
                target_noise_clip,
            )
            next_action = tf.clip_by_value(
                self._actor_tgt(next_b, training=False) + noise, -1.0, 1.0
            )

            # Fix 1: twin critics
            ci_next = tf.concat([next_b, next_action], axis=-1)
            q1_tgt  = tf.squeeze(self._c1_tgt(ci_next, training=False), -1)
            q2_tgt  = tf.squeeze(self._c2_tgt(ci_next, training=False), -1)
            q_tgt   = tf.stop_gradient(
                rew_b + gammas_n * (1.0 - done_b) * tf.minimum(q1_tgt, q2_tgt)
            )

            ci = tf.concat([obs_b, act_b], axis=-1)
            q1 = tf.squeeze(self._c1(ci, training=True), -1)
            q2 = tf.squeeze(self._c2(ci, training=True), -1)
            td_errors = tf.stop_gradient((q1 + q2) / 2.0 - q_tgt)
            c_loss = tf.reduce_mean(
                weights * (tf.square(q1 - q_tgt) + tf.square(q2 - q_tgt))
            )

        c_vars  = self._c1.trainable_variables + self._c2.trainable_variables
        c_grads = tape.gradient(c_loss, c_vars)
        c_grads, _ = tf.clip_by_global_norm(c_grads, 10.0)
        self._critic_opt.apply_gradients(zip(c_grads, c_vars))
        self._buffer.update_priorities(indices, np.abs(td_errors.numpy()))

        # ---- Actor update (Fix 2: delayed) ----
        a_loss = 0.0
        if update_actor:
            with tf.GradientTape() as tape:
                new_a  = self._actor(obs_b, training=True)
                ci_new = tf.concat([obs_b, new_a], axis=-1)
                # Deterministic policy gradient: maximise Q1(s, π(s))
                a_loss_tf = -tf.reduce_mean(
                    tf.squeeze(self._c1(ci_new, training=False), -1)
                )

            a_grads = tape.gradient(a_loss_tf, self._actor.trainable_variables)
            a_grads, _ = tf.clip_by_global_norm(a_grads, 10.0)
            self._actor_opt.apply_gradients(
                zip(a_grads, self._actor.trainable_variables)
            )
            a_loss = float(a_loss_tf.numpy())
            self._soft_update(self._actor, self._actor_tgt, tau)

        self._soft_update(self._c1, self._c1_tgt, tau)
        self._soft_update(self._c2, self._c2_tgt, tau)

        return a_loss, float(c_loss.numpy())

    @staticmethod
    def _soft_update(source: tf.keras.Model, target: tf.keras.Model, tau: float) -> None:
        """θ_target ← τ·θ_source + (1−τ)·θ_target  (Polyak averaging)"""
        for sv, tv in zip(source.trainable_variables, target.trainable_variables):
            tv.assign(tau * sv + (1.0 - tau) * tv)

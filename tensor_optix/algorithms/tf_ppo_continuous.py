import os
import numpy as np
import tensorflow as tf

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches

LOG_STD_MIN = -5
LOG_STD_MAX = 2
_LOG_2PI = float(np.log(2 * np.pi))


class TFGaussianPPOAgent(BaseAgent):
    """
    PPO for continuous action spaces (TensorFlow) using a squashed Gaussian policy.

    Actions are bounded to (-1, 1)^action_dim via tanh. Suitable for continuous
    control tasks such as position sizing in trading (negative = short, positive = long).

    Does NOT require tensorflow_probability — Gaussian log prob and tanh correction
    are implemented manually.

    Actor network: obs → [batch, 2 * action_dim]
        First  action_dim outputs = pre-tanh mean
        Second action_dim outputs = log_std  (clamped to [LOG_STD_MIN, LOG_STD_MAX])

    Critic network: obs → [batch, 1] or [batch]  (scalar value estimate)

    Typical usage (position sizing, action_dim=1):

        import tensorflow as tf
        from tensor_optix.algorithms.tf_ppo_continuous import TFGaussianPPOAgent

        obs_dim    = 13
        action_dim = 1    # scalar position in (-1, 1)

        actor = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(2 * action_dim),   # mean || log_std
        ])
        critic = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(1),
        ])
        agent = TFGaussianPPOAgent(
            actor=actor,
            critic=critic,
            optimizer=tf.keras.optimizers.Adam(3e-4),
            action_dim=action_dim,
            hyperparams=HyperparamSet(params={
                "learning_rate": 3e-4,
                "clip_ratio":    0.2,
                "entropy_coef":  0.01,
                "vf_coef":       0.5,
                "gamma":         0.99,
                "gae_lambda":    0.95,
                "n_epochs":      10,
                "minibatch_size": 64,
                "max_grad_norm": 0.5,
            }, episode_id=0),
        )

    Hyperparams tuned by the framework (all optional, defaults shown above):
        learning_rate, clip_ratio, entropy_coef, vf_coef, gamma,
        gae_lambda, n_epochs, minibatch_size, max_grad_norm
    """

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
        # entropy_coef lo=0.001: continuous PPO std can collapse to zero when entropy_coef=0.
        # gamma included: SPSA can adapt the discount horizon per-environment.
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        action_dim: int,
        hyperparams: HyperparamSet,
        reward_normalizer=None,
    ):
        self._actor = actor
        self._critic = critic
        self._optimizer = optimizer
        self._action_dim = action_dim
        self._hyperparams = hyperparams.copy()
        self._reward_normalizer = reward_normalizer

        # Rollout cache — populated by act(), consumed by learn()
        self._cache_obs: list = []
        self._cache_log_probs: list = []
        self._cache_values: list = []

    # ------------------------------------------------------------------
    # Gaussian log prob with tanh squashing correction (no tfp required)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_log_prob(pre_tanh, mean, log_std):
        """
        Log probability of a squashed Gaussian action.

        log π(a|s) = Σᵢ [ log N(uᵢ; μᵢ, σᵢ) − log(1 − tanh²(uᵢ) + ε) ]

        where a = tanh(u), u ~ N(μ, σ).
        Returns a scalar per sample: shape [batch].
        """
        std = tf.exp(log_std)
        log_prob_normal = (
            -0.5 * tf.square((pre_tanh - mean) / (std + 1e-8))
            - log_std
            - 0.5 * _LOG_2PI
        )
        tanh_correction = tf.math.log(1.0 - tf.square(tf.tanh(pre_tanh)) + 1e-6)
        return tf.reduce_sum(log_prob_normal - tanh_correction, axis=-1)  # [batch]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> np.ndarray:
        """
        Sample action ∈ (-1, 1)^action_dim; cache log_prob and V(obs) for learn().
        Returns a float numpy array of shape [action_dim].
        """
        obs = tf.cast(np.atleast_2d(observation), tf.float32)

        out     = self._actor(obs, training=False)             # [1, 2*action_dim]
        mean    = out[:, :self._action_dim]
        log_std = tf.clip_by_value(out[:, self._action_dim:], LOG_STD_MIN, LOG_STD_MAX)
        std     = tf.exp(log_std)

        eps      = tf.random.normal(tf.shape(mean))
        pre_tanh = mean + std * eps                            # [1, action_dim]
        action   = tf.tanh(pre_tanh)                          # [1, action_dim]

        log_prob = self._gaussian_log_prob(pre_tanh, mean, log_std)  # [1]
        value    = float(tf.squeeze(self._critic(obs, training=False)).numpy())

        self._cache_obs.append(np.squeeze(observation))
        self._cache_log_probs.append(float(tf.squeeze(log_prob).numpy()))
        self._cache_values.append(value)

        return tf.squeeze(action, axis=0).numpy()  # [action_dim]

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Run PPO update on the collected rollout.

        1. Recover pre_tanh = atanh(actions.clamp(-1+ε, 1-ε))
        2. Compute GAE advantages
        3. Run n_epochs of minibatch clipped surrogate updates
        4. Partially clear cache (del [:T] preserves any residual entries)
        """
        hp = self._hyperparams.params
        clip_ratio    = float(hp.get("clip_ratio",    0.2))
        entropy_coef  = float(hp.get("entropy_coef",  0.01))
        vf_coef       = float(hp.get("vf_coef",       0.5))
        gamma         = float(hp.get("gamma",          0.99))
        gae_lambda    = float(hp.get("gae_lambda",     0.95))
        n_epochs      = int(hp.get("n_epochs",         10))
        mb_size       = int(hp.get("minibatch_size",   64))
        max_grad_norm = float(hp.get("max_grad_norm",  0.5))

        T = len(episode_data.rewards)
        if len(self._cache_obs) < T:
            raise RuntimeError(
                f"Cache underflow: expected >= {T} entries but got {len(self._cache_obs)}. "
                "Ensure act() is called exactly once per environment step."
            )

        obs_arr    = np.array(self._cache_obs[:T],       dtype=np.float32)
        old_lp_arr = np.array(self._cache_log_probs[:T], dtype=np.float32)
        val_arr    = np.array(self._cache_values[:T],    dtype=np.float32)
        act_arr    = np.array(episode_data.actions,      dtype=np.float32)
        rewards    = list(episode_data.rewards)
        dones      = episode_data.dones

        if self._reward_normalizer is not None:
            for r, done in zip(rewards, dones):
                self._reward_normalizer.step(r)
                if done:
                    self._reward_normalizer.reset()
            rewards = list(self._reward_normalizer.normalize(np.array(rewards, dtype=np.float32)))

        last_value = 0.0
        if not dones[-1] and episode_data.final_obs is not None:
            final_obs_t = tf.cast(np.atleast_2d(episode_data.final_obs), tf.float32)
            last_value = float(tf.squeeze(self._critic(final_obs_t, training=False)).numpy())

        advantages, returns = compute_gae(rewards, val_arr, dones, gamma, gae_lambda, last_value)
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout = {
            "obs":        obs_arr,
            "actions":    act_arr,
            "old_lp":     old_lp_arr,
            "advantages": advantages,
            "returns":    returns,
        }

        all_vars = self._actor.trainable_variables + self._critic.trainable_variables
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_approx_kl   = 0.0
        n_updates         = 0

        for _ in range(n_epochs):
            for batch in make_minibatches(rollout, mb_size):
                obs_b = tf.cast(batch["obs"],        tf.float32)
                act_b = tf.cast(batch["actions"],    tf.float32)
                old_b = tf.cast(batch["old_lp"],     tf.float32)
                adv_b = tf.cast(batch["advantages"], tf.float32)
                ret_b = tf.cast(batch["returns"],    tf.float32)

                # Recover pre_tanh from stored actions (atanh, numerically safe)
                pre_tanh_b = tf.math.atanh(tf.clip_by_value(act_b, -1 + 1e-6, 1 - 1e-6))

                with tf.GradientTape() as tape:
                    out_b    = self._actor(obs_b, training=True)
                    mean_b   = out_b[:, :self._action_dim]
                    log_std_b = tf.clip_by_value(
                        out_b[:, self._action_dim:], LOG_STD_MIN, LOG_STD_MAX
                    )

                    new_lp   = self._gaussian_log_prob(pre_tanh_b, mean_b, log_std_b)

                    ratio    = tf.exp(new_lp - old_b)
                    s1       = ratio * adv_b
                    s2       = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
                    pol_loss = -tf.reduce_mean(tf.minimum(s1, s2))

                    new_val  = tf.squeeze(self._critic(obs_b, training=True), axis=-1)
                    val_loss = tf.reduce_mean(tf.square(new_val - ret_b))

                    # Differential entropy of Gaussian: H = 0.5 Σ log(2πe σ²)
                    std_b   = tf.exp(log_std_b)
                    entropy = tf.reduce_mean(
                        tf.reduce_sum(0.5 * (1.0 + tf.math.log(2.0 * np.pi * tf.square(std_b))),
                                      axis=-1)
                    )

                    loss = pol_loss + vf_coef * val_loss - entropy_coef * entropy

                grads = tape.gradient(loss, all_vars)
                if max_grad_norm > 0:
                    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
                self._optimizer.apply_gradients(zip(grads, all_vars))

                approx_kl = float(tf.reduce_mean(old_b - new_lp).numpy())
                total_policy_loss += float(pol_loss.numpy())
                total_value_loss  += float(val_loss.numpy())
                total_entropy     += float(entropy.numpy())
                total_approx_kl   += approx_kl
                n_updates         += 1

        # Partial clear: preserve residual entries beyond T
        del self._cache_obs[:T]
        del self._cache_log_probs[:T]
        del self._cache_values[:T]

        n = max(n_updates, 1)
        explained_var = self._explained_variance(val_arr, returns)
        return {
            "policy_loss":   total_policy_loss / n,
            "value_loss":    total_value_loss  / n,
            "entropy":       total_entropy     / n,
            "approx_kl":     total_approx_kl  / n,
            "explained_var": explained_var,
            "n_updates":     n_updates,
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
        self._actor.save(os.path.join(path, "actor.keras"))
        self._critic.save(os.path.join(path, "critic.keras"))

    def load_weights(self, path: str) -> None:
        loaded_actor  = tf.keras.models.load_model(os.path.join(path, "actor.keras"))
        loaded_critic = tf.keras.models.load_model(os.path.join(path, "critic.keras"))
        for v, lv in zip(self._actor.trainable_variables,  loaded_actor.trainable_variables):
            v.assign(lv)
        for v, lv in zip(self._critic.trainable_variables, loaded_critic.trainable_variables):
            v.assign(lv)

    def average_weights(self, paths: list) -> None:
        for net, fname in ((self._actor, "actor.keras"), (self._critic, "critic.keras")):
            loaded = [tf.keras.models.load_model(os.path.join(p, fname)) for p in paths]
            for v, *lvs in zip(net.trainable_variables, *[m.trainable_variables for m in loaded]):
                v.assign(tf.reduce_mean(tf.stack([lv for lv in lvs], axis=0), axis=0))

    def perturb_weights(self, noise_scale: float) -> None:
        for module in (self._actor, self._critic):
            for v in module.trainable_variables:
                v.assign(v * (1.0 + noise_scale * tf.random.normal(v.shape)))

    @staticmethod
    def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return float("nan")
        return float(1.0 - np.var(returns - values) / var_returns)

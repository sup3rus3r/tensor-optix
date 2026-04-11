import os
import numpy as np
import tensorflow as tf
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches


class TFPPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization) agent for TensorFlow discrete action spaces.

    Implements the clipped surrogate objective (Schulman et al. 2017) with:
    - GAE-λ advantage estimation
    - Entropy bonus for exploration regularization
    - Value function loss (MSE) with optional clipping
    - Multiple epochs of minibatch gradient descent per rollout
    - Global gradient norm clipping

    Architecture: separate actor and critic networks.
        actor:  tf.keras.Model, obs → logits [batch, n_actions]
        critic: tf.keras.Model, obs → value  [batch, 1] or [batch]

    Usage:
        actor  = tf.keras.Sequential([...])   # outputs logits
        critic = tf.keras.Sequential([...])   # outputs scalar value
        agent  = TFPPOAgent(
            actor=actor,
            critic=critic,
            optimizer=tf.keras.optimizers.Adam(3e-4),
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

    The agent caches (log_prob, value) during act() calls inside the pipeline's
    rollout loop. learn() consumes the cache, runs GAE, and performs n_epochs of
    minibatch PPO updates, then clears the cache.

    Hyperparams tuned by the framework (all optional, defaults shown above):
        learning_rate, clip_ratio, entropy_coef, vf_coef, gamma,
        gae_lambda, n_epochs, minibatch_size, max_grad_norm
    """

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
        # entropy_coef lo=0.001: prevents SPSA from zeroing entropy and collapsing the policy.
        # gamma included: PPO advantage estimation is sensitive to discount horizon;
        # SPSA can adapt it per-environment consistently with DQN/SAC.
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        hyperparams: HyperparamSet,
        reward_normalizer=None,
    ):
        self._actor = actor
        self._critic = critic
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()
        self._reward_normalizer = reward_normalizer

        # Rollout cache — populated by act(), consumed by learn()
        self._cache_obs: list = []
        self._cache_log_probs: list = []
        self._cache_values: list = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> int:
        """
        Sample action from π(·|obs), cache log_prob and V(obs) for learn().
        Returns a Python int (action index).
        """
        obs = tf.cast(np.atleast_2d(observation), tf.float32)

        logits = self._actor(obs, training=False)                 # [1, n_actions]
        log_probs_all = tf.nn.log_softmax(logits)                 # [1, n_actions]
        action = int(tf.squeeze(tf.random.categorical(logits, 1)).numpy())
        log_prob = float(log_probs_all[0, action].numpy())

        value = float(tf.squeeze(self._critic(obs, training=False)).numpy())

        self._cache_obs.append(np.squeeze(observation))
        self._cache_log_probs.append(log_prob)
        self._cache_values.append(value)

        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Run PPO update on the collected rollout.

        1. Compute GAE advantages from cached values + episode rewards
        2. Normalize advantages (zero mean, unit std)
        3. Run n_epochs of shuffled minibatch updates:
           - Policy loss: clipped surrogate
           - Value loss:  MSE(new_value, returns)
           - Entropy:     bonus to encourage exploration
        4. Clear rollout cache
        5. Return training diagnostics
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

        # Align cache with episode_data (BatchPipeline steps == len(cache))
        T = len(episode_data.rewards)
        if len(self._cache_obs) < T:
            raise RuntimeError(
                f"Cache underflow: expected >= {T} entries but got {len(self._cache_obs)}. "
                "Ensure act() is called exactly once per environment step."
            )
        obs_arr      = np.array(self._cache_obs[:T],      dtype=np.float32)
        old_lp_arr   = np.array(self._cache_log_probs[:T], dtype=np.float32)
        values_arr   = np.array(self._cache_values[:T],    dtype=np.float32)
        rewards      = list(episode_data.rewards)
        dones        = episode_data.dones

        # Reward normalizer: update running stats and reset at episode boundaries.
        # Done internally so subclasses and callers don't have to manage it.
        if self._reward_normalizer is not None:
            for r, done in zip(rewards, dones):
                self._reward_normalizer.step(r)
                if done:
                    self._reward_normalizer.reset()
            rewards = list(self._reward_normalizer.normalize(np.array(rewards, dtype=np.float32)))

        # Bootstrap: if the window ended mid-episode, V(s_T) from the critic
        # corrects the TD target at the last step. Zero when terminal.
        last_value = 0.0
        if not dones[-1] and episode_data.final_obs is not None:
            final_obs_t = tf.cast(np.atleast_2d(episode_data.final_obs), tf.float32)
            last_value = float(tf.squeeze(self._critic(final_obs_t, training=False)).numpy())

        advantages, returns = compute_gae(rewards, values_arr, dones, gamma, gae_lambda, last_value)

        # Normalize advantages across the rollout
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout = {
            "obs":        obs_arr,
            "actions":    np.array(episode_data.actions, dtype=np.int32),
            "old_lp":     old_lp_arr,
            "advantages": advantages,
            "returns":    returns,
        }

        all_vars = self._actor.trainable_variables + self._critic.trainable_variables
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_approx_kl   = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for batch in make_minibatches(rollout, mb_size):
                obs_b   = tf.cast(batch["obs"],        tf.float32)
                act_b   = tf.cast(batch["actions"],    tf.int32)
                old_b   = tf.cast(batch["old_lp"],     tf.float32)
                adv_b   = tf.cast(batch["advantages"], tf.float32)
                ret_b   = tf.cast(batch["returns"],    tf.float32)

                with tf.GradientTape() as tape:
                    # Actor forward pass
                    logits   = self._actor(obs_b, training=True)         # [mb, n_actions]
                    lp_all   = tf.nn.log_softmax(logits)                  # [mb, n_actions]
                    # Gather log_prob of each taken action: lp_all[i, act_b[i]]
                    batch_idx = tf.range(tf.shape(act_b)[0])
                    gather_idx = tf.stack([batch_idx, act_b], axis=1)    # [mb, 2]
                    new_lp   = tf.gather_nd(lp_all, gather_idx)          # [mb]

                    # Importance ratio and clipped surrogate
                    ratio    = tf.exp(new_lp - old_b)
                    s1       = ratio * adv_b
                    s2       = tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio) * adv_b
                    pol_loss = -tf.reduce_mean(tf.minimum(s1, s2))

                    # Critic forward pass
                    new_val  = tf.squeeze(self._critic(obs_b, training=True), axis=-1)
                    val_loss = tf.reduce_mean(tf.square(new_val - ret_b))

                    # Entropy bonus (categorical: -Σ p·log p)
                    probs    = tf.nn.softmax(logits)
                    entropy  = -tf.reduce_mean(
                        tf.reduce_sum(probs * lp_all, axis=-1)
                    )

                    loss = pol_loss + vf_coef * val_loss - entropy_coef * entropy

                grads = tape.gradient(loss, all_vars)
                if max_grad_norm > 0:
                    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
                self._optimizer.apply_gradients(zip(grads, all_vars))

                # Diagnostics accumulation
                approx_kl = float(tf.reduce_mean(old_b - new_lp).numpy())
                total_policy_loss += float(pol_loss.numpy())
                total_value_loss  += float(val_loss.numpy())
                total_entropy     += float(entropy.numpy())
                total_approx_kl   += approx_kl
                n_updates += 1

        self._clear_cache(T)

        n = max(n_updates, 1)
        explained_var = self._explained_variance(values_arr, returns)
        return {
            "policy_loss":    total_policy_loss / n,
            "value_loss":     total_value_loss  / n,
            "entropy":        total_entropy      / n,
            "approx_kl":      total_approx_kl   / n,
            "explained_var":  explained_var,
            "n_updates":      n_updates,
        }

    def action_probs(self, observation) -> np.ndarray:
        """
        Return softmax action probabilities without sampling.
        Used by PolicyManager.ensemble_action() to average distributions
        across ensemble members rather than averaging sampled actions.
        """
        obs = tf.cast(np.atleast_2d(observation), tf.float32)
        logits = self._actor(obs, training=False)
        return tf.nn.softmax(logits)[0].numpy()

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
        import numpy as np
        n = len(paths)
        for net, fname in ((self._actor, "actor.keras"), (self._critic, "critic.keras")):
            loaded = [tf.keras.models.load_model(os.path.join(p, fname)) for p in paths]
            for v, *lvs in zip(net.trainable_variables, *[m.trainable_variables for m in loaded]):
                v.assign(tf.reduce_mean(tf.stack([lv for lv in lvs], axis=0), axis=0))

    def perturb_weights(self, noise_scale: float) -> None:
        for module in (self._actor, self._critic):
            for v in module.trainable_variables:
                v.assign(v * (1.0 + noise_scale * tf.random.normal(v.shape)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        """
        Discard all entries in the rollout cache without learning from them.
        Called by LoopController after a val-pipeline window is collected, so
        that val-rollout entries never bleed into the next training learn() call.
        """
        self._cache_obs.clear()
        self._cache_log_probs.clear()
        self._cache_values.clear()

    def _clear_cache(self, T: int) -> None:
        del self._cache_obs[:T]
        del self._cache_log_probs[:T]
        del self._cache_values[:T]

    @staticmethod
    def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
        """Fraction of return variance explained by the value function. 1.0 = perfect."""
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return float("nan")
        return float(1.0 - np.var(returns - values) / var_returns)

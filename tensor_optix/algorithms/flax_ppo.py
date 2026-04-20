"""
tensor_optix.algorithms.flax_ppo — PPO agent implemented with Flax NNX.

Architecture
------------
A single ``_ActorCritic`` module holds two independent MLP heads that share no
weights: an *actor* head (obs → action logits) and a *critic* head
(obs → scalar value).  A single ``nnx.Optimizer`` covers all parameters.

PPO update
----------
After each episode the rollout cache (observations, log-probs, values) is used
to compute GAE-λ advantages and discounted returns.  ``n_epochs`` passes of
mini-batch gradient descent apply the clipped surrogate objective:

    L = E[ min(r_t·A_t,  clip(r_t, 1−ε, 1+ε)·A_t) ]
      − vf_coef · MSE(V(s_t), Rₜ)
      + entropy_coef · H[π(·|s_t)]

where  r_t = π_θ(a_t|s_t) / π_μ(a_t|s_t).

JAX/Flax specifics
------------------
* ``nnx.value_and_grad(loss_fn, has_aux=True)(model)`` differentiates through
  the combined actor+critic module in a single pass — no separate backward
  calls per head.
* ``nnx.Optimizer(model, tx, wrt=nnx.Param)`` ensures only trainable ``Param``
  variables are updated; ``BatchStat`` variables (future BN/LN) are skipped.
* Weight persistence: ``nnx.to_pure_dict(nnx.state(model))`` → plain nested
  dict of numpy arrays; restore via ``nnx.replace_by_pure_dict`` + ``nnx.update``.
"""

from __future__ import annotations

import os
import pickle
import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _ActorCritic:
    """
    Lightweight wrapper that creates two independent MLP heads via Flax NNX.

    Built lazily inside ``FlaxPPOAgent.__init__`` to keep JAX/Flax as an
    optional dependency — the class definition itself does not import them.
    """
    pass  # defined at runtime; see FlaxPPOAgent.__init__


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FlaxPPOAgent(BaseAgent):
    """
    PPO agent for discrete action spaces using Flax NNX + optax.

    Parameters
    ----------
    obs_dim      int   — observation vector length
    n_actions    int   — number of discrete actions
    hyperparams  HyperparamSet — training hyper-parameters (see below)
    hidden_size  int   — width of each hidden layer (default 64)
    seed         int   — PRNG seed for weight initialisation

    Hyper-parameter keys (all optional, defaults shown)
    ---------------------------------------------------
    learning_rate   3e-4   — Adam step size
    clip_ratio      0.2    — PPO ε clipping
    entropy_coef    0.01   — entropy bonus coefficient
    vf_coef         0.5    — value function loss weight
    gamma           0.99   — discount factor
    gae_lambda      0.95   — GAE trace decay
    n_epochs        10     — gradient epochs per rollout
    minibatch_size  64     — mini-batch size
    """

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hyperparams: HyperparamSet,
        hidden_size: int = 64,
        seed: int = 0,
    ) -> None:
        import jax
        import optax
        from flax import nnx

        self._obs_dim     = obs_dim
        self._n_actions   = n_actions
        self._hidden_size = hidden_size
        self._seed        = seed
        self._hyperparams = hyperparams.copy()
        self._np_rng      = np.random.default_rng(seed)

        # ── Build actor-critic network ────────────────────────────────
        rngs = nnx.Rngs(seed)

        class _AC(nnx.Module):
            def __init__(self, obs_d, n_act, hidden, rngs):
                self.actor_fc1  = nnx.Linear(obs_d,  hidden, rngs=rngs)
                self.actor_fc2  = nnx.Linear(hidden, n_act,  rngs=rngs)
                self.critic_fc1 = nnx.Linear(obs_d,  hidden, rngs=rngs)
                self.critic_fc2 = nnx.Linear(hidden, 1,      rngs=rngs)

            def actor(self, x):
                import jax
                return self.actor_fc2(jax.nn.tanh(self.actor_fc1(x)))

            def critic(self, x):
                import jax
                return self.critic_fc2(jax.nn.tanh(self.critic_fc1(x))).squeeze(-1)

        self._model = _AC(obs_dim, n_actions, hidden_size, rngs)

        lr = float(hyperparams.params.get("learning_rate", 3e-4))
        self._optimizer = nnx.Optimizer(
            self._model, optax.adam(lr), wrt=nnx.Param
        )

        self._cache_obs:      list[np.ndarray] = []
        self._cache_log_probs: list[float]     = []
        self._cache_values:    list[float]     = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> int:
        """
        Sample action from the current policy.

        Caches the observation, log-prob, and value estimate for the
        subsequent ``learn()`` call.
        """
        import jax
        import jax.numpy as jnp

        obs = jnp.array(np.atleast_2d(observation), dtype=jnp.float32)
        logits = self._model.actor(obs)
        lp_all = jax.nn.log_softmax(logits, axis=-1)
        probs  = np.array(jax.nn.softmax(logits, axis=-1)[0])
        action = int(self._np_rng.choice(len(probs), p=probs))
        logp   = float(lp_all[0, action])
        value  = float(self._model.critic(obs)[0])

        self._cache_obs.append(np.squeeze(observation))
        self._cache_log_probs.append(logp)
        self._cache_values.append(value)
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        """PPO update on the cached rollout."""
        import jax
        import jax.numpy as jnp
        from flax import nnx

        hp            = self._hyperparams.params
        clip_ratio    = float(hp.get("clip_ratio",    0.2))
        entropy_coef  = float(hp.get("entropy_coef",  0.01))
        vf_coef       = float(hp.get("vf_coef",       0.5))
        gamma         = float(hp.get("gamma",          0.99))
        gae_lambda    = float(hp.get("gae_lambda",     0.95))
        n_epochs      = int(hp.get("n_epochs",         10))
        mb_size       = int(hp.get("minibatch_size",   64))

        T = len(episode_data.rewards)
        if len(self._cache_obs) < T:
            raise RuntimeError(
                f"Cache underflow: expected >= {T} entries, got {len(self._cache_obs)}. "
                "Ensure act() is called once per environment step."
            )

        obs_arr    = np.array(self._cache_obs[:T],       dtype=np.float32)
        old_lp_arr = np.array(self._cache_log_probs[:T], dtype=np.float32)
        val_arr    = np.array(self._cache_values[:T],    dtype=np.float32)
        rewards    = list(episode_data.rewards)
        dones      = episode_data.dones

        # Bootstrap value when trajectory ends mid-episode.
        last_value = 0.0
        if not dones[-1] and episode_data.final_obs is not None:
            obs_final  = jnp.array(
                np.atleast_2d(episode_data.final_obs), dtype=jnp.float32
            )
            last_value = float(self._model.critic(obs_final)[0])

        advantages, returns = compute_gae(
            rewards, val_arr, dones, gamma, gae_lambda, last_value
        )
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout = {
            "obs":        obs_arr,
            "actions":    np.array(episode_data.actions, dtype=np.int32),
            "old_lp":     old_lp_arr,
            "advantages": advantages,
            "returns":    returns,
        }

        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        n_updates      = 0

        for _ in range(n_epochs):
            for batch in make_minibatches(rollout, mb_size):
                obs_b  = jnp.array(batch["obs"],        dtype=jnp.float32)
                act_b  = jnp.array(batch["actions"],    dtype=jnp.int32)
                old_b  = jnp.array(batch["old_lp"],     dtype=jnp.float32)
                adv_b  = jnp.array(batch["advantages"], dtype=jnp.float32)
                ret_b  = jnp.array(batch["returns"],    dtype=jnp.float32)

                # Capture hyper-params in a closure (Python scalars, safe for JAX).
                cr, vc, ec = clip_ratio, vf_coef, entropy_coef

                def loss_fn(model, obs_b=obs_b, act_b=act_b, old_b=old_b,
                            adv_b=adv_b, ret_b=ret_b):
                    logits   = model.actor(obs_b)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    new_lp   = log_probs[jnp.arange(len(act_b)), act_b]

                    ratio = jnp.exp(new_lp - old_b)
                    s1    = ratio * adv_b
                    s2    = jnp.clip(ratio, 1 - cr, 1 + cr) * adv_b
                    pol_loss = -jnp.mean(jnp.minimum(s1, s2))

                    new_val  = model.critic(obs_b)
                    val_loss = jnp.mean((new_val - ret_b) ** 2)

                    probs   = jax.nn.softmax(logits, axis=-1)
                    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

                    total = pol_loss + vc * val_loss - ec * entropy
                    return total, (pol_loss, val_loss, entropy)

                (_, (pl, vl, ent)), grads = nnx.value_and_grad(
                    loss_fn, has_aux=True
                )(self._model)
                self._optimizer.update(self._model, grads)

                total_pol_loss += float(pl)
                total_val_loss += float(vl)
                total_entropy  += float(ent)
                n_updates += 1

        del self._cache_obs[:T]
        del self._cache_log_probs[:T]
        del self._cache_values[:T]

        n = max(n_updates, 1)
        ev = _explained_variance(val_arr, returns)
        return {
            "policy_loss":   total_pol_loss / n,
            "value_loss":    total_val_loss / n,
            "entropy":       total_entropy  / n,
            "explained_var": ev,
            "n_updates":     n_updates,
        }

    def action_probs(self, observation) -> np.ndarray:
        """Softmax action probabilities (for ensemble averaging)."""
        import jax
        import jax.numpy as jnp
        obs = jnp.array(np.atleast_2d(observation), dtype=jnp.float32)
        return np.array(jax.nn.softmax(self._model.actor(obs), axis=-1)[0])

    def get_hyperparams(self) -> HyperparamSet:
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        import optax
        from flax import nnx
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            self._optimizer = nnx.Optimizer(
                self._model, optax.adam(lr), wrt=nnx.Param
            )

    def save_weights(self, path: str) -> None:
        from flax import nnx
        os.makedirs(path, exist_ok=True)
        state     = nnx.state(self._model)
        pure_dict = nnx.to_pure_dict(state)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(pure_dict, f, protocol=4)

    def load_weights(self, path: str) -> None:
        from flax import nnx
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            pure_dict = pickle.load(f)
        state = nnx.state(self._model)
        nnx.replace_by_pure_dict(state, pure_dict)
        nnx.update(self._model, state)

    def reset_cache(self) -> None:
        """Discard rollout cache without learning."""
        self._cache_obs.clear()
        self._cache_log_probs.clear()
        self._cache_values.clear()

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_r = float(np.var(returns))
    if var_r < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_r)

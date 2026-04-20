"""
tensor_optix.adapters.jax.flax_agent — Base agent for Flax NNX models.

Mirrors the role of TorchAgent for PyTorch: wraps a ``flax.nnx.Module`` and
implements the ``BaseAgent`` interface so any Flax model can participate in
``LoopController``, ``TrialOrchestrator``, and hyper-parameter optimisers
without modification to any core component.

Weight persistence
------------------
JAX arrays are not directly pickle-safe across JAX/XLA versions, so weights
are serialised via ``flax.nnx.to_pure_dict(nnx.state(model))`` which returns a
plain nested dict of raw numpy arrays, and restored via
``nnx.replace_by_pure_dict(state, pure_dict)`` + ``nnx.update(model, state)``.
"""

from __future__ import annotations

import os
import pickle
import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class FlaxAgent(BaseAgent):
    """
    Base agent for Flax NNX models (discrete action spaces).

    Parameters
    ----------
    model       flax.nnx.Module  — policy network (obs → logits)
    optimizer   flax.nnx.Optimizer — wraps optax transform
    hyperparams HyperparamSet — must include at least ``gamma``

    Subclass and override ``learn()`` for algorithm-specific updates
    (PPO, A2C, DQN, …).  The base ``learn()`` does a REINFORCE update with
    normalised returns.
    """

    def __init__(self, model, optimizer, hyperparams: HyperparamSet) -> None:
        self._model = model
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> int:
        """Greedy action (argmax over logits).  No sampling, no caching."""
        import jax
        import jax.numpy as jnp

        obs = jnp.array(np.atleast_2d(observation), dtype=jnp.float32)
        logits = self._model(obs)
        return int(jnp.argmax(logits, axis=-1)[0])

    def learn(self, episode_data: EpisodeData) -> dict:
        """REINFORCE with normalised returns."""
        import jax
        import jax.numpy as jnp
        from flax import nnx

        gamma = float(self._hyperparams.params.get("gamma", 0.99))

        rewards = episode_data.rewards
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_arr = np.array(returns, dtype=np.float32)

        ret_t = jnp.array(returns_arr)
        if len(returns_arr) > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        obs = jnp.array(np.array(episode_data.observations, dtype=np.float32))
        actions = jnp.array(np.array(episode_data.actions, dtype=np.int32))

        def loss_fn(model):
            logits = model(obs)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            act_lp = log_probs[jnp.arange(len(actions)), actions]
            return -jnp.mean(act_lp * ret_t)

        loss_val, grads = nnx.value_and_grad(loss_fn)(self._model)
        self._optimizer.update(self._model, grads)

        return {"loss": float(loss_val)}

    def get_hyperparams(self) -> HyperparamSet:
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()

    def save_weights(self, path: str) -> None:
        """Serialise model parameters to ``<path>/model.pkl`` as a plain dict."""
        from flax import nnx
        os.makedirs(path, exist_ok=True)
        state = nnx.state(self._model)
        pure_dict = nnx.to_pure_dict(state)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(pure_dict, f, protocol=4)

    def load_weights(self, path: str) -> None:
        """Restore model parameters from ``<path>/model.pkl``."""
        from flax import nnx
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            pure_dict = pickle.load(f)
        state = nnx.state(self._model)
        nnx.replace_by_pure_dict(state, pure_dict)
        nnx.update(self._model, state)

"""
tests/test_improvements/test_12_jax.py — JAX/Flax Adapter (item 12)

Tests
-----
1.  FlaxAgent imports cleanly and satisfies the BaseAgent interface.
2.  FlaxEvaluator produces an EvalMetrics with total_reward as primary score.
3.  FlaxPPOAgent.act() returns a valid discrete action and caches rollout data.
4.  FlaxPPOAgent.learn() returns diagnostics keys and clears the rollout cache.
5.  FlaxPPOAgent.save_weights() / load_weights() round-trip: logits identical.
6.  FlaxPPOAgent.set_hyperparams() accepts a new learning rate without error.
7.  FlaxPPOAgent.reset_cache() empties the cache.
8.  FlaxPPOAgent.action_probs() returns a valid probability distribution.
9.  Multiple learn() calls on sequential episodes do not raise (cache management).
10. FlaxPPOAgent trains on CartPole-v1 and reaches mean reward ≥ 150 in 200 episodes.
11. FlaxPPOAgent parity: score within 10 % of TorchPPOAgent given the same budget.
"""

from __future__ import annotations

import sys
import importlib.machinery
from unittest.mock import MagicMock

# ── TF stub — must run before any tensor_optix import ──────────────────────
def _make_tf_mock(name: str) -> MagicMock:
    m = MagicMock()
    m.__spec__    = importlib.machinery.ModuleSpec(name=name, loader=None, origin=None)
    m.__version__ = "2.18.0"
    m.__name__    = name
    m.__package__ = name.split(".")[0]
    m.__path__    = []
    m.__loader__  = None
    return m

for _tf_mod in [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.layers",
    "tensorflow.keras.optimizers", "tensorflow.keras.models",
]:
    if _tf_mod not in sys.modules:
        sys.modules[_tf_mod] = _make_tf_mock(_tf_mod)

# ── stdlib / third-party ───────────────────────────────────────────────────
import os
import tempfile
import numpy as np
import pytest

# ── Project imports ────────────────────────────────────────────────────────
from tensor_optix.core.types import EpisodeData, HyperparamSet

# Conditionally skip if JAX is not installed.
jax = pytest.importorskip("jax", reason="jax not installed")
pytest.importorskip("flax",  reason="flax not installed")
pytest.importorskip("optax", reason="optax not installed")

from tensor_optix.adapters.jax.flax_agent    import FlaxAgent
from tensor_optix.adapters.jax.flax_evaluator import FlaxEvaluator
from tensor_optix.algorithms.flax_ppo         import FlaxPPOAgent


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _default_hp(**overrides) -> HyperparamSet:
    params = dict(
        learning_rate  = 3e-4,
        clip_ratio     = 0.2,
        entropy_coef   = 0.01,
        vf_coef        = 0.5,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        n_epochs       = 4,
        minibatch_size = 32,
    )
    params.update(overrides)
    return HyperparamSet(params=params, episode_id=0)


@pytest.fixture
def ppo_agent():
    return FlaxPPOAgent(obs_dim=4, n_actions=2, hyperparams=_default_hp(), seed=0)


def _make_episode(n: int = 10, obs_dim: int = 4) -> EpisodeData:
    return EpisodeData(
        observations = np.random.rand(n, obs_dim).astype(np.float32),
        actions      = np.random.randint(0, 2, n).tolist(),
        rewards      = [1.0] * (n - 1) + [0.0],
        terminated   = [False] * (n - 1) + [True],
        truncated    = [False] * n,
        infos        = [{}] * n,
        episode_id   = 0,
    )


def _prime_cache(agent: FlaxPPOAgent, episode: EpisodeData) -> None:
    """Call act() for every step so the cache is populated before learn()."""
    for obs in episode.observations:
        agent.act(obs)


# ===========================================================================
# Tests 1–9: unit / integration
# ===========================================================================

class TestFlaxAgentInterface:
    """FlaxAgent satisfies the BaseAgent interface."""

    def test_import(self):
        from tensor_optix.core.base_agent import BaseAgent
        assert issubclass(FlaxAgent, BaseAgent)

    def test_flax_ppo_is_base_agent(self, ppo_agent):
        from tensor_optix.core.base_agent import BaseAgent
        assert isinstance(ppo_agent, BaseAgent)

    def test_get_hyperparams_roundtrip(self, ppo_agent):
        hp = ppo_agent.get_hyperparams()
        assert abs(hp.params["learning_rate"] - 3e-4) < 1e-9

    def test_set_hyperparams(self, ppo_agent):
        new_hp = _default_hp(learning_rate=1e-4)
        ppo_agent.set_hyperparams(new_hp)
        assert abs(ppo_agent.get_hyperparams().params["learning_rate"] - 1e-4) < 1e-9

    def test_teardown_no_error(self, ppo_agent):
        ppo_agent.teardown()  # must not raise


class TestFlaxEvaluator:
    def test_score_returns_eval_metrics(self):
        ev = FlaxEvaluator()
        ep = _make_episode()
        m  = ev.score(ep, {"loss": 0.5})
        assert m.primary_score == pytest.approx(sum(ep.rewards))
        assert "total_reward" in m.metrics
        assert "loss" in m.metrics


class TestFlaxPPOAct:
    def test_act_returns_valid_action(self, ppo_agent):
        obs = np.zeros(4, dtype=np.float32)
        action = ppo_agent.act(obs)
        assert action in (0, 1)

    def test_act_populates_cache(self, ppo_agent):
        obs = np.zeros(4, dtype=np.float32)
        for _ in range(5):
            ppo_agent.act(obs)
        assert len(ppo_agent._cache_obs) == 5
        assert len(ppo_agent._cache_log_probs) == 5
        assert len(ppo_agent._cache_values) == 5

    def test_action_probs_sums_to_one(self, ppo_agent):
        obs = np.zeros(4, dtype=np.float32)
        probs = ppo_agent.action_probs(obs)
        assert probs.shape == (2,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_reset_cache(self, ppo_agent):
        ppo_agent.act(np.zeros(4, np.float32))
        ppo_agent.reset_cache()
        assert len(ppo_agent._cache_obs) == 0


class TestFlaxPPOLearn:
    def test_learn_returns_diagnostics(self, ppo_agent):
        ep = _make_episode()
        _prime_cache(ppo_agent, ep)
        diag = ppo_agent.learn(ep)
        for key in ("policy_loss", "value_loss", "entropy", "n_updates"):
            assert key in diag

    def test_learn_clears_cache(self, ppo_agent):
        ep = _make_episode()
        _prime_cache(ppo_agent, ep)
        ppo_agent.learn(ep)
        assert len(ppo_agent._cache_obs) == 0

    def test_multiple_sequential_episodes(self, ppo_agent):
        for i in range(3):
            ep = _make_episode()
            _prime_cache(ppo_agent, ep)
            ppo_agent.learn(ep)
        assert len(ppo_agent._cache_obs) == 0


class TestFlaxPPOWeights:
    def test_save_load_roundtrip(self, ppo_agent):
        import jax.numpy as jnp

        obs = jnp.ones((1, 4))
        logits_before = np.array(ppo_agent._model.actor(obs))

        with tempfile.TemporaryDirectory() as tmp:
            ppo_agent.save_weights(tmp)
            # Corrupt the model then restore.
            ppo_agent2 = FlaxPPOAgent(obs_dim=4, n_actions=2,
                                      hyperparams=_default_hp(), seed=99)
            ppo_agent2.load_weights(tmp)
            logits_after = np.array(ppo_agent2._model.actor(obs))

        np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5)


# ===========================================================================
# Tests 10–11: training quality
# ===========================================================================

def _train_flax_ppo(n_episodes: int = 200, seed: int = 0) -> float:
    """Train FlaxPPOAgent on CartPole-v1; return mean eval reward (5 episodes)."""
    import gymnasium as gym

    hp = _default_hp(n_epochs=4, minibatch_size=32)
    agent = FlaxPPOAgent(obs_dim=4, n_actions=2, hyperparams=hp, seed=seed)
    env   = gym.make("CartPole-v1")

    for _ in range(n_episodes):
        obs_t, _ = env.reset(seed=seed)
        obs_list, act_list, rew_list = [], [], []
        done = False
        while not done:
            action = agent.act(obs_t)
            obs_list.append(obs_t.copy())
            act_list.append(action)
            obs_t, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            done = terminated or truncated

        ep = EpisodeData(
            observations = np.array(obs_list, dtype=np.float32),
            actions      = act_list,
            rewards      = rew_list,
            terminated   = [False] * (len(rew_list) - 1) + [bool(terminated)],
            truncated    = [False] * (len(rew_list) - 1) + [bool(truncated)],
            infos        = [{}] * len(rew_list),
            episode_id   = _,
        )
        agent.learn(ep)

    env.close()

    # Evaluate
    eval_env = gym.make("CartPole-v1")
    rewards  = []
    for _ in range(5):
        obs_e, _ = eval_env.reset(seed=seed + 1000)
        total = 0.0
        done  = False
        while not done:
            action = agent.act(obs_e)
            obs_e, r, term, trunc, _ = eval_env.step(action)
            total += r
            done = term or trunc
        agent.reset_cache()
        rewards.append(total)
    eval_env.close()
    return float(np.mean(rewards))


def _train_torch_ppo(n_episodes: int = 200, seed: int = 0) -> float:
    """Train TorchPPOAgent on CartPole-v1; return mean eval reward (5 episodes)."""
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from tensor_optix.algorithms.torch_ppo import TorchPPOAgent

    torch.manual_seed(seed)
    obs_dim, n_actions = 4, 2
    actor  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, n_actions))
    critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))
    hp = HyperparamSet(params=dict(
        learning_rate=3e-4, clip_ratio=0.2, entropy_coef=0.01,
        vf_coef=0.5, gamma=0.99, gae_lambda=0.95,
        n_epochs=4, minibatch_size=32, max_grad_norm=0.5,
    ), episode_id=0)
    agent = TorchPPOAgent(
        actor=actor, critic=critic,
        optimizer=torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=3e-4
        ),
        hyperparams=hp, device="cpu",
    )

    env = gym.make("CartPole-v1")
    for _ in range(n_episodes):
        obs_t, _ = env.reset(seed=seed)
        obs_list, act_list, rew_list = [], [], []
        done = False
        while not done:
            action = agent.act(obs_t)
            obs_list.append(obs_t.copy())
            act_list.append(action)
            obs_t, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            done = terminated or truncated

        ep = EpisodeData(
            observations = np.array(obs_list, dtype=np.float32),
            actions      = act_list,
            rewards      = rew_list,
            terminated   = [False] * (len(rew_list) - 1) + [bool(terminated)],
            truncated    = [False] * (len(rew_list) - 1) + [bool(truncated)],
            infos        = [{}] * len(rew_list),
            episode_id   = _,
        )
        agent.learn(ep)
    env.close()

    eval_env = gym.make("CartPole-v1")
    rewards  = []
    for _ in range(5):
        obs_e, _ = eval_env.reset(seed=seed + 1000)
        total = 0.0
        done  = False
        while not done:
            action = agent.act(obs_e)
            obs_e, r, term, trunc, _ = eval_env.step(action)
            total += r
            done = term or trunc
        agent.reset_cache()
        rewards.append(total)
    eval_env.close()
    return float(np.mean(rewards))


@pytest.mark.slow
def test_flax_ppo_convergence():
    """FlaxPPOAgent reaches mean reward ≥ 150 on CartPole-v1 in 200 episodes."""
    score = _train_flax_ppo(n_episodes=200, seed=42)
    assert score >= 150, f"FlaxPPOAgent mean eval reward {score:.1f} < 150"


@pytest.mark.slow
def test_flax_ppo_parity_with_torch():
    """
    FlaxPPOAgent score is within 10% of TorchPPOAgent on CartPole-v1.

    Both agents train for 200 episodes with the same hyper-parameters.
    Convergence varies with seed; the 10% tolerance is deliberately loose to
    account for JAX/PyTorch PRNG differences and JIT warm-up overhead.
    """
    torch = pytest.importorskip("torch", reason="torch not installed — skipping parity test")

    seed = 7
    flax_score  = _train_flax_ppo(n_episodes=200,  seed=seed)
    torch_score = _train_torch_ppo(n_episodes=200, seed=seed)

    if torch_score < 10:
        pytest.skip("TorchPPO baseline degenerated — skip parity check")

    ratio = abs(flax_score - torch_score) / max(torch_score, 1.0)
    assert ratio < 0.10, (
        f"Parity failed: FlaxPPO={flax_score:.1f}, TorchPPO={torch_score:.1f}, "
        f"relative gap={ratio:.3f} (threshold 0.10)"
    )

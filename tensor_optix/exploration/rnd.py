"""
Random Network Distillation (RND) exploration bonus.

RND adds an intrinsic reward signal to any pipeline without modifying any agent.
Two small networks — a frozen random target and a trained predictor — both map
observations to a fixed embedding space. Novel states produce high prediction
error (high intrinsic reward). Visited states are fitted well (low bonus).

Math:
    r_int(s) = ||f_θ(s) - g(s)||²       (g frozen, f_θ trained)
    r_total  = r_ext + η · r_int / σ(r_int)   (intrinsic normalized per episode)

η is controlled by the loop via set_eta():
    ACTIVE  → η = η_base        (default exploration level)
    COOLING → η *= 1.5          (stuck — push exploration harder)
    DORMANT → η = 0             (converged — stop injecting noise)
    improvement → η *= 0.9      (getting better — reduce exploration)

Usage:
    from tensor_optix.exploration.rnd import RNDPipeline

    base = BatchPipeline(env=gym.make("LunarLander-v2"), agent=agent, window_size=2048)
    pipeline = RNDPipeline(base, obs_dim=8, embedding_dim=64, eta=0.1)

    optimizer = RLOptimizer(agent=agent, pipeline=pipeline)
    optimizer.run()

RNDPipeline wraps any BasePipeline. It intercepts EpisodeData after each episode
and injects the intrinsic bonus into episode_data.rewards before the agent sees them.
The predictor network is trained on the current batch each episode.

Framework: pure numpy + a minimal two-layer MLP. No TF or Torch dependency.
The target network is random and fixed; the predictor is updated via SGD.
"""

import numpy as np
from typing import Generator, Optional

from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.types import EpisodeData


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class _MLP:
    """Tiny two-layer MLP in pure numpy. Used for both target and predictor."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)
        self.W1 = rng.standard_normal((in_dim,     hidden_dim)).astype(np.float32) * scale(in_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32)    * scale(hidden_dim)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [batch, in_dim] → [batch, out_dim]"""
        h = _relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward_and_grad(self, x: np.ndarray):
        """Returns output and gradient of MSE loss wrt W1, b1, W2, b2."""
        h = _relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, h, x

    def sgd_step(self, x: np.ndarray, target: np.ndarray, lr: float) -> None:
        """One SGD step minimizing MSE(forward(x), target)."""
        h = _relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        n = x.shape[0]

        d_out = 2.0 * (out - target) / n            # [batch, out_dim]
        dW2 = h.T @ d_out                            # [hidden, out_dim]
        db2 = d_out.sum(axis=0)
        d_h = d_out @ self.W2.T                      # [batch, hidden]
        d_h_relu = d_h * (h > 0).astype(np.float32) # ReLU gradient
        dW1 = x.T @ d_h_relu
        db1 = d_h_relu.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


class RNDPipeline(BasePipeline):
    """
    Wraps any BasePipeline and injects RND intrinsic rewards into each episode.

    After each episode, before returning EpisodeData:
    1. Compute r_int(s) = ||predictor(s) - target(s)||² for each step
    2. Normalize r_int by its running std
    3. Inject: rewards[t] += eta * r_int[t]
    4. Train predictor on current batch observations

    The loop controls eta via set_eta() at state transitions.

    Args:
        pipeline:      The wrapped pipeline (any BasePipeline).
        obs_dim:       Observation dimension (flattened).
        embedding_dim: RND embedding size. Default 64 — larger = slower but richer.
        eta:           Initial intrinsic reward scale. Default 0.1.
        predictor_lr:  SGD learning rate for the predictor. Default 1e-3.
        norm_eps:      Small constant for std normalization. Default 1e-8.
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        obs_dim: int,
        embedding_dim: int = 64,
        eta: float = 0.1,
        predictor_lr: float = 1e-3,
        norm_eps: float = 1e-8,
    ):
        self._pipeline = pipeline
        self._eta = eta
        self._predictor_lr = predictor_lr
        self._norm_eps = norm_eps

        # Fixed random target (seed=0 for reproducibility across runs)
        self._target    = _MLP(obs_dim, embedding_dim, embedding_dim, seed=0)
        self._predictor = _MLP(obs_dim, embedding_dim, embedding_dim, seed=42)

        # Running stats for intrinsic reward normalization
        self._int_reward_var = 1.0
        self._int_reward_mean = 0.0
        self._int_reward_count = 0

    # ------------------------------------------------------------------
    # BasePipeline interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._pipeline.setup()

    def teardown(self) -> None:
        self._pipeline.teardown()

    @property
    def is_live(self) -> bool:
        return self._pipeline.is_live

    def episodes(self) -> Generator[EpisodeData, None, None]:
        for episode_data in self._pipeline.episodes():
            if self._eta > 0.0 and len(episode_data.observations) > 1:
                episode_data = self._inject_intrinsic(episode_data)
            yield episode_data

    # ------------------------------------------------------------------
    # Loop control
    # ------------------------------------------------------------------

    def set_eta(self, eta: float) -> None:
        """Called by LoopController at state transitions to adjust exploration scale."""
        self._eta = max(0.0, float(eta))

    # Proxy any other pipeline attributes (e.g. set_agent, n_steps)
    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _inject_intrinsic(self, episode_data: EpisodeData) -> EpisodeData:
        obs = np.array(episode_data.observations[:-1], dtype=np.float32)  # [T-1, obs_dim]
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Compute intrinsic rewards
        tgt_emb  = self._target.forward(obs)     # [T-1, embed]
        pred_emb = self._predictor.forward(obs)  # [T-1, embed]
        r_int = np.sum((pred_emb - tgt_emb) ** 2, axis=-1).astype(np.float32)  # [T-1]

        # Update running stats (Welford online)
        for r in r_int:
            self._int_reward_count += 1
            delta = r - self._int_reward_mean
            self._int_reward_mean += delta / self._int_reward_count
            self._int_reward_var  += delta * (r - self._int_reward_mean)

        std = np.sqrt(self._int_reward_var / max(1, self._int_reward_count) + self._norm_eps)
        r_int_norm = r_int / std

        # Inject into rewards
        rewards = list(episode_data.rewards)
        T = min(len(rewards), len(r_int_norm))
        for t in range(T):
            rewards[t] = float(rewards[t]) + self._eta * float(r_int_norm[t])
        episode_data.rewards = rewards

        # Train predictor on current batch
        self._predictor.sgd_step(obs, tgt_emb, lr=self._predictor_lr)

        return episode_data

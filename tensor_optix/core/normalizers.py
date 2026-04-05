import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's parallel algorithm.
    Numerically stable for online updates with arbitrary batch sizes.

    shape: shape of a single sample (e.g. () for scalars, (obs_dim,) for vectors).
    epsilon: small initial count to prevent division by zero at startup.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of samples (shape [N, ...] or scalar)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            # Single sample — wrap in batch dimension
            x = x[np.newaxis]
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        # Parallel variance formula (Chan et al. 1979)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize x to approximately zero mean, unit variance, then clip."""
        x = np.asarray(x, dtype=np.float32)
        normed = (x - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        if clip > 0:
            normed = np.clip(normed, -clip, clip)
        return normed


class ObsNormalizer:
    """
    Wraps RunningMeanStd for observation normalization.

    Usage:
        norm = ObsNormalizer(obs_shape=(4,))
        norm.update(obs_batch)          # update stats from a batch
        obs_normed = norm.normalize(obs)  # normalize a single obs or batch

    Typically update() is called on each collected rollout before normalize()
    is used inside act().
    """

    def __init__(self, obs_shape, clip: float = 10.0):
        self._rms = RunningMeanStd(shape=obs_shape)
        self._clip = clip

    def update(self, obs: np.ndarray) -> None:
        self._rms.update(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return self._rms.normalize(obs, clip=self._clip)

    @property
    def mean(self) -> np.ndarray:
        return self._rms.mean

    @property
    def var(self) -> np.ndarray:
        return self._rms.var


class RewardNormalizer:
    """
    Normalizes rewards by tracking a running estimate of the return variance
    (not raw reward variance). This is the approach used in OpenAI baselines:
    maintain a running mean/std of discounted returns, divide raw rewards by
    the return std. Does NOT subtract the mean (to preserve reward sign).

    Usage:
        norm = RewardNormalizer(gamma=0.99)
        for r in rewards:
            norm.step(r)                    # update running return
        scaled_rewards = norm.normalize(rewards)
    """

    def __init__(self, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8):
        self._rms = RunningMeanStd(shape=())
        self._gamma = gamma
        self._clip = clip
        self._epsilon = epsilon
        self._running_return = 0.0

    def step(self, reward: float) -> None:
        """Update running return with a single step reward."""
        self._running_return = self._running_return * self._gamma + reward
        self._rms.update(np.array([self._running_return]))

    def normalize(self, rewards) -> np.ndarray:
        """Divide rewards by running return std (does not subtract mean)."""
        rewards = np.asarray(rewards, dtype=np.float32)
        std = float(np.sqrt(self._rms.var + self._epsilon))
        normed = rewards / std
        if self._clip > 0:
            normed = np.clip(normed, -self._clip, self._clip)
        return normed

    def reset(self) -> None:
        """Reset running return at episode boundary."""
        self._running_return = 0.0

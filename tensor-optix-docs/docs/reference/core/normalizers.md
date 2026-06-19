# tensor_optix.core.normalizers

## RunningMeanStd

```python
class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's parallel algorithm.
    Numerically stable for online updates with arbitrary batch sizes.

    shape: shape of a single sample (e.g. () for scalars, (obs_dim,) for vectors).
    epsilon: small initial count to prevent division by zero at startup.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4): ...

    def update(self, x) -> None:
        """Update statistics with a batch of samples (shape [N, ...] or scalar)."""

    def normalize(self, x, clip: float = 10.0):
        """Normalize x to approximately zero mean, unit variance, then clip."""
```

## ObsNormalizer

```python
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

    def __init__(self, obs_shape, clip: float = 10.0): ...
    def update(self, obs) -> None: ...
    def normalize(self, obs): ...

    @property
    def mean(self): ...
    @property
    def var(self): ...
```

## RewardNormalizer

```python
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

    def __init__(self, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8): ...

    def step(self, reward: float) -> None:
        """Update running return with a single step reward."""

    def normalize(self, rewards):
        """Divide rewards by running return std (does not subtract mean)."""

    def reset(self) -> None:
        """Reset running return at episode boundary."""
```

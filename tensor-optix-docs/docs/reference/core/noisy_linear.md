# tensor_optix.core.noisy_linear

```
tensor_optix.core.noisy_linear - NoisyLinear layer for Rainbow DQN.

Replaces standard nn.Linear with a stochastic counterpart where both the
weight mean and noise scale are learned:

    y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)

This is **factorized Gaussian noise** (Fortunato et al. 2017, §3.2).
For a layer mapping p inputs to q outputs:

  * Independent noise would require p*q + q random samples.
  * Factorized noise draws p + q samples and constructs:

        ε_w_{ij} = f(ε_i) ⊗ f(ε_j)
        ε_b_j    = f(ε_j)

    where f(x) = sgn(x) * sqrt(|x|)

This reduces the noise sampling cost from O(pq) to O(p+q) while
preserving the key property: each weight gets its own independent noise
(not shared across the whole layer).

**Initialization (Fortunato et al. 2017, §3.1):**

    μ  ~ U(-1/√p, +1/√p)      (same as Glorot for uniform)
    σ  = σ_0 / √p              (σ_0 = 0.5 for factorized noise)

**Exploration semantics:**

    σ → 0  on states the network has learned → exploit (μ is reliable).
    σ → ∞  on uncertain states              → explore (variance is large).

The gradient flows through σ via the reparameterisation trick - exploration
is adaptive, not a fixed schedule. ε-greedy is entirely replaced.

During evaluation (no_noise=True, or model.eval()):
    y = μ_w x + μ_b    (deterministic, ε = 0)
```

## NoisyLinear

```python
class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al. 2017).

    Parameters
    ----------
    in_features  : int
    out_features : int
    sigma_0      : float  - initial σ scale factor (0.5 per paper)
    """

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5): ...

    def reset_noise(self) -> None:
        """
        Sample fresh factorized noise.

        Call once per training step (or per forward pass during training).
        Sharing a single noise sample across the entire forward pass is the
        factorized Gaussian approximation from §3.2 of the paper.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        During training (self.training=True):  y = (μ + σ⊙ε)x + (μ_b + σ_b⊙ε_b)
        During evaluation (self.training=False): y = μx + μ_b  (ε = 0, deterministic)
        """
```

Used by `TorchRainbowDQNAgent`'s `RainbowQNetwork` in place of standard `nn.Linear` layers - see [Algorithms](../algorithms.md).

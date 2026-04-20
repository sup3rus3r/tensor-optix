"""
tensor_optix.core.noisy_linear — NoisyLinear layer for Rainbow DQN.

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

The gradient flows through σ via the reparameterisation trick — exploration
is adaptive, not a fixed schedule. ε-greedy is entirely replaced.

During evaluation (no_noise=True, or model.eval()):
    y = μ_w x + μ_b    (deterministic, ε = 0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al. 2017).

    Parameters
    ----------
    in_features  : int
    out_features : int
    sigma_0      : float  — initial σ scale factor (0.5 per paper)
    """

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_0      = sigma_0

        # Learnable mean parameters
        self.mu_w    = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b    = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers — reset each forward pass during training
        self.register_buffer("eps_w", torch.zeros(out_features, in_features))
        self.register_buffer("eps_b", torch.zeros(out_features))

        self._reset_parameters()
        self.reset_noise()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)

        sigma_init = self.sigma_0 / math.sqrt(self.in_features)
        self.sigma_w.data.fill_(sigma_init)
        self.sigma_b.data.fill_(sigma_init)

    # ------------------------------------------------------------------
    # Noise management
    # ------------------------------------------------------------------

    @staticmethod
    def _factorized_noise(size: int, device=None) -> torch.Tensor:
        """Draw size samples, apply f(x) = sgn(x)·√|x|."""
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Sample fresh factorized noise.

        Call once per training step (or per forward pass during training).
        Sharing a single noise sample across the entire forward pass is the
        factorized Gaussian approximation from §3.2 of the paper.
        """
        device = self.mu_w.device
        eps_i = self._factorized_noise(self.in_features, device=device)
        eps_j = self._factorized_noise(self.out_features, device=device)
        # ε_w_ij = f(ε_i) ⊗ f(ε_j)   [outer product]
        self.eps_w = eps_j.unsqueeze(1) * eps_i.unsqueeze(0)
        self.eps_b = eps_j

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        During training (self.training=True):  y = (μ + σ⊙ε)x + (μ_b + σ_b⊙ε_b)
        During evaluation (self.training=False): y = μx + μ_b  (ε = 0, deterministic)
        """
        if self.training:
            weight = self.mu_w + self.sigma_w * self.eps_w
            bias   = self.mu_b + self.sigma_b * self.eps_b
        else:
            weight = self.mu_w
            bias   = self.mu_b
        return F.linear(x, weight, bias)

    # ------------------------------------------------------------------
    # nn.Module interface helpers
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, sigma_0={self.sigma_0}")

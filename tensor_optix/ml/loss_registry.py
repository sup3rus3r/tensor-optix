from __future__ import annotations

"""
Loss function registry and auto-detection for MLAgent.

Available loss strings
----------------------
Supervised
  "auto"           Detect from dataset (default).
  "cross_entropy"  nn.CrossEntropyLoss       — multi-class classification (int targets).
  "bce"            nn.BCEWithLogitsLoss       — binary classification (float {0,1} targets).
  "mse"            nn.MSELoss                 — regression (float targets, any range).
  "mae"            nn.L1Loss                  — regression, less sensitive to outliers.
  "huber"          nn.HuberLoss               — regression, robust to large outliers.

Unsupervised / self-supervised
  "reconstruction" nn.MSELoss on (output, input) — autoencoders.
  "vae"            ELBO: reconstruction + KL divergence — variational autoencoders.
                   Model must return (reconstruction, mu, logvar).
  "contrastive"    NT-Xent loss — SimCLR-style contrastive learning.
                   Dataset must yield pairs of augmented views (x1, x2).
  "cosine"         nn.CosineEmbeddingLoss     — embedding / similarity tasks.

Any nn.Module or callable can be passed directly as the loss argument.
"""

import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom unsupervised losses
# ------------------------------------------------------------------

class _ReconstructionLoss(nn.Module):
    """MSE between model output and input — for autoencoders."""
    def forward(self, output, target):
        return F.mse_loss(output, target)


class _VAELoss(nn.Module):
    """
    ELBO loss for variational autoencoders.
    Model must return (reconstruction, mu, logvar).
    target is the original input.
    """
    def forward(self, model_output, target):
        if not (isinstance(model_output, (tuple, list)) and len(model_output) == 3):
            raise ValueError(
                "VAE loss expects model to return (reconstruction, mu, logvar). "
                "Got: " + repr(type(model_output))
            )
        recon, mu, logvar = model_output
        recon_loss = F.mse_loss(recon, target, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl) / target.size(0)


class _NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR.
    Expects model_output to be a tuple (z1, z2) of projected embeddings.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, model_output, target=None):
        if not (isinstance(model_output, (tuple, list)) and len(model_output) == 2):
            raise ValueError(
                "Contrastive loss expects model to return (z1, z2) embeddings. "
                "Dataset must yield (view1, view2) pairs."
            )
        z1, z2 = model_output
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                      # [2N, D]
        sim = torch.mm(z, z.T) / self.temperature            # [2N, 2N]
        # Mask out self-similarity
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
        return F.cross_entropy(sim, labels)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

LOSS_MAP: dict = {
    # supervised
    "cross_entropy":  nn.CrossEntropyLoss,
    "bce":            nn.BCEWithLogitsLoss,
    "mse":            nn.MSELoss,
    "mae":            nn.L1Loss,
    "huber":          nn.HuberLoss,
    "cosine":         nn.CosineEmbeddingLoss,
    # unsupervised / self-supervised
    "reconstruction": _ReconstructionLoss,
    "vae":            _VAELoss,
    "contrastive":    _NTXentLoss,
}

_LOSS_DESCRIPTIONS = {
    "cross_entropy":  "CrossEntropyLoss       — multi-class classification (integer targets)",
    "bce":            "BCEWithLogitsLoss       — binary classification (float targets in {0,1})",
    "mse":            "MSELoss                 — regression (float targets, any range)",
    "mae":            "L1Loss                  — regression, less sensitive to outliers",
    "huber":          "HuberLoss               — regression, robust to large outliers",
    "cosine":         "CosineEmbeddingLoss     — embedding / similarity tasks",
    "reconstruction": "MSE(output, input)      — autoencoders (no labels needed)",
    "vae":            "ELBO: recon + KL        — variational autoencoders; model returns (recon, mu, logvar)",
    "contrastive":    "NT-Xent                 — SimCLR contrastive; dataset yields (view1, view2) pairs",
}


def available_losses() -> str:
    """Human-readable list of all supported loss strings."""
    lines = ["Available loss options:", ""]
    lines.append("  Supervised:")
    for k in ("cross_entropy", "bce", "mse", "mae", "huber", "cosine"):
        lines.append(f'    "{k}"  →  {_LOSS_DESCRIPTIONS[k]}')
    lines.append("")
    lines.append("  Unsupervised / self-supervised:")
    for k in ("reconstruction", "vae", "contrastive"):
        lines.append(f'    "{k}"  →  {_LOSS_DESCRIPTIONS[k]}')
    lines.append("")
    lines.append('  "auto"  →  detect from data (default)')
    lines.append("  Any nn.Module or callable  →  used directly")
    return "\n".join(lines)


def resolve_loss(
    loss: Union[str, nn.Module],
    dataset=None,
    n_samples: int = 64,
) -> nn.Module:
    """
    Return an instantiated loss nn.Module.

    Parameters
    ----------
    loss:
        "auto", a loss key string, or an nn.Module instance.
    dataset:
        Required when loss="auto". Used to sample items for detection.
    n_samples:
        Number of samples to inspect for auto-detection.
    """
    if isinstance(loss, nn.Module):
        return loss

    if callable(loss) and not isinstance(loss, str):
        # bare function/lambda — wrap it
        class _FnLoss(nn.Module):
            def forward(self, pred, target):
                return loss(pred, target)
        return _FnLoss()

    if not isinstance(loss, str):
        raise TypeError(
            f"loss must be a string, nn.Module, or callable — got {type(loss).__name__}.\n"
            f"{available_losses()}"
        )

    key = loss.lower()

    if key == "auto":
        if dataset is None:
            logger.warning(
                "loss='auto' but no dataset provided — defaulting to 'mse'. "
                "Pass the dataset to Optimizer to enable auto-detection."
            )
            return nn.MSELoss()
        detected = _auto_detect(dataset, n_samples)
        logger.info("loss='auto' detected: '%s'", detected)
        print(
            f"  [tensor-optix] loss='auto' detected: '{detected}'  "
            f"(set loss='{detected}' explicitly to suppress this message)",
            flush=True,
        )
        return LOSS_MAP[detected]()

    if key not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{loss}'.\n{available_losses()}")

    return LOSS_MAP[key]()


# ------------------------------------------------------------------
# Auto-detection
# ------------------------------------------------------------------

def _auto_detect(dataset, n_samples: int) -> str:
    """
    Sample items from the dataset and infer the appropriate loss.

    Rules (in order):
    1. Items are single tensors (no label)       → "reconstruction"
    2. Items are pairs (x1, x2) of same shape   → "contrastive"
    3. Targets are int/long                      → "cross_entropy"
    4. Targets are float, values in {0, 1}       → "bce"
    5. Targets are float, any range              → "mse"
    """
    sample = _get_sample(dataset)
    if sample is None:
        logger.warning("Could not sample from dataset — defaulting to 'mse'.")
        return "mse"

    # Single tensor or single-element tuple — no labels → autoencoder
    if isinstance(sample, torch.Tensor):
        return "reconstruction"

    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        return "reconstruction"

    x, y = sample[0], sample[1]

    # Both elements are tensors of equal shape → contrastive pair
    if (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.shape == y.shape
    ):
        return "contrastive"

    # Label-based detection
    targets = _sample_targets(dataset, n_samples)
    if targets is None:
        return "mse"

    if targets.dtype in (torch.int32, torch.int64):
        return "cross_entropy"

    unique = targets.unique()
    if unique.numel() <= 2 and set(unique.tolist()).issubset({0.0, 1.0}):
        return "bce"

    return "mse"


def _get_sample(dataset):
    """Get the first item from a Dataset or DataLoader."""
    try:
        from torch.utils.data import Dataset
        if isinstance(dataset, Dataset):
            item = dataset[0]
            if isinstance(item, (tuple, list)):
                return tuple(torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in item)
            return torch.as_tensor(item)
        # DataLoader — grab first batch, take first element
        for batch in dataset:
            if isinstance(batch, (tuple, list)):
                return tuple(v[0] if isinstance(v, torch.Tensor) else torch.as_tensor(v)[0] for v in batch)
            return batch[0]
    except Exception as e:
        logger.debug("Dataset sampling failed: %s", e)
    return None


def _sample_targets(dataset, n_samples: int):
    """Extract up to n_samples target values as a flat tensor."""
    from torch.utils.data import Dataset
    try:
        if isinstance(dataset, Dataset):
            targets = []
            for i in range(min(n_samples, len(dataset))):
                item = dataset[i]
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    targets.append(torch.as_tensor(item[1]).reshape(-1))
            return torch.cat(targets)[:n_samples] if targets else None
        # DataLoader
        targets = []
        for batch in dataset:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                targets.append(torch.as_tensor(batch[1]).reshape(-1))
            if sum(t.numel() for t in targets) >= n_samples:
                break
        return torch.cat(targets)[:n_samples] if targets else None
    except Exception as e:
        logger.debug("Target sampling failed: %s", e)
        return None

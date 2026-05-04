"""
Global device registry for tensor-optix.

Set once at the start of your script and all tensor-optix components
(neuroevo, adapters, pipelines) will use it automatically.

Usage::

    import tensor_optix as tx
    tx.set_device("cuda")          # or "cpu", "cuda:1", torch.device(...)

    # Or read the current device:
    dev = tx.get_device()
"""

from __future__ import annotations
import torch

_DEVICE: torch.device = torch.device("cpu")


def set_device(device) -> None:
    """Set the global tensor-optix device."""
    global _DEVICE
    _DEVICE = torch.device(device)


def get_device() -> torch.device:
    """Return the global tensor-optix device."""
    return _DEVICE


def auto_device() -> torch.device:
    """Return CUDA if available, else CPU, and set it as the global device."""
    global _DEVICE
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

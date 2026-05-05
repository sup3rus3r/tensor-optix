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

# Lazy torch import — core package must not require PyTorch at import time.
_DEVICE: str = "cpu"


def set_device(device) -> None:
    """Set the global tensor-optix device.

    Accepts ``str``, ``torch.device``, or any object with a ``type`` attribute.
    The internal value is stored as a string so the module loads without PyTorch.
    """
    global _DEVICE
    if hasattr(device, "type"):
        idx = getattr(device, "index", None)
        device = f"{device.type}:{idx}" if idx is not None else device.type
    _DEVICE = str(device)


def get_device() -> "torch.device":
    """Return the global tensor-optix device as a ``torch.device``.

    Lazily imports ``torch`` so the call works once PyTorch is installed.
    """
    import torch

    return torch.device(_DEVICE)


def auto_device() -> "torch.device":
    """Return CUDA if available, else CPU, and set it as the global device."""
    import torch

    global _DEVICE
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(_DEVICE)

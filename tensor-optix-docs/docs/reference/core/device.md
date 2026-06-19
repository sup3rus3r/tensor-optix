# tensor_optix.core.device

```
Global device registry for tensor-optix.

Set once at the start of your script and all tensor-optix components
(neuroevo, adapters, pipelines) will use it automatically.

Usage::

    import tensor_optix as tx
    tx.set_device("cuda")          # or "cpu", "cuda:1", torch.device(...)

    # Or read the current device:
    dev = tx.get_device()
```

```python
def set_device(device) -> None:
    """Set the global tensor-optix device.

    Accepts ``str``, ``torch.device``, or any object with a ``type`` attribute.
    The internal value is stored as a string so the module loads without PyTorch.
    """

def get_device() -> "torch.device":
    """Return the global tensor-optix device as a ``torch.device``.

    Lazily imports ``torch`` so the call works once PyTorch is installed.
    """

def auto_device() -> "torch.device":
    """Return CUDA if available, else CPU, and set it as the global device."""
```

These three functions are re-exported at the top level: `tensor_optix.set_device`, `tensor_optix.get_device`, `tensor_optix.auto_device`.

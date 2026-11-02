"""
Generic helpers for capturing/restoring full resumable training state
(model + optimizer/scheduler + RNG) for torch-based agents.

Exact training continuation requires more than model weights — optimizer
momentum, LR-scheduler state, and RNG state all affect what happens on the
next gradient step. These helpers let a torch-based agent opt into full
resumability with a few lines, by passing in whichever stateful objects it
owns (anything implementing the standard torch state_dict()/load_state_dict()
protocol — nn.Module, optim.Optimizer, and LR schedulers all qualify).

This does NOT attempt to capture arbitrary custom state (a replay buffer, a
running normalizer) automatically — only objects implementing that standard
protocol. An agent with additional non-standard state must capture that
separately and must not report resumable_components as including it unless
it actually does — see BaseAgent.resumable_components.
"""
import os
from typing import Dict

RESUME_FILENAME = "resume_state.pt"


def save_torch_resume_state(path: str, **named_stateful) -> None:
    """
    Save state_dict() of every passed object, plus torch/numpy (and CUDA,
    when available) RNG state, into <path>/resume_state.pt.

    named_stateful: keyword args mapping a name to an object with
    .state_dict() — nn.Module, Optimizer, LR scheduler, etc.
    """
    import numpy as np
    import torch

    os.makedirs(path, exist_ok=True)
    blob: Dict[str, object] = {
        "components": {name: obj.state_dict() for name, obj in named_stateful.items()},
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        blob["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(blob, os.path.join(path, RESUME_FILENAME))


def load_torch_resume_state(path: str, **named_stateful) -> bool:
    """
    Restore state_dict() for every passed object from <path>/resume_state.pt,
    plus torch/numpy/CUDA RNG state.

    Returns False (no-op) if the file doesn't exist — e.g. a checkpoint
    saved before this agent adopted full resume, or by an agent that never
    implemented it — so callers can distinguish "restored" from
    "weights-only checkpoint, nothing more to restore."
    """
    import numpy as np
    import torch

    resume_path = os.path.join(path, RESUME_FILENAME)
    if not os.path.exists(resume_path):
        return False

    blob = torch.load(resume_path, map_location="cpu", weights_only=False)
    components = blob.get("components", {})
    for name, obj in named_stateful.items():
        if name in components:
            obj.load_state_dict(components[name])
    torch.set_rng_state(blob["torch_rng_state"])
    np.random.set_state(blob["numpy_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in blob:
        torch.cuda.set_rng_state_all(blob["torch_cuda_rng_state_all"])
    return True

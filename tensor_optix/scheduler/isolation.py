"""Per-worker isolation helpers: seeds and checkpoint directories.

Pure functions, no multiprocessing. Mirrors the exact conventions already
proven in tensor_optix/distributed/async_learner.py (seed offset) and
tensor_optix/orchestrator.py (checkpoint directory layout).
"""

import os

_SEED_PRIME = 7919  # matches async_learner's `seed + actor_id * 7919` convention


def trial_seed(base_seed: int, trial_number: int, retry_attempt: int = 0) -> int:
    """Deterministic, distinct seed per (trial_number, retry_attempt)."""
    return base_seed + trial_number * _SEED_PRIME + retry_attempt


def trial_checkpoint_dir(run_ckpt_dir: str, trial_number: int) -> str:
    """Identical path construction to orchestrator.py's _objective (unchanged)."""
    return os.path.join(run_ckpt_dir, f"trial_{trial_number}")

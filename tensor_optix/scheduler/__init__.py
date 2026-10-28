"""Phase 2 — concurrent experiment scheduler.

ConcurrentTrialScheduler runs multiple Optuna trials as real, isolated OS
processes while the parent process remains the sole owner of the
optuna.Study and the experiment ledger. Workers never touch either —
see tensor_optix/scheduler/worker.py for the picklable worker entry point
and the module docstring of pool.py for the full architecture.

TrialOrchestrator (tensor_optix/orchestrator.py) is untouched and remains
the simpler, correct tool for sequential trial search.
"""

from .pool import ConcurrentTrialScheduler

__all__ = ["ConcurrentTrialScheduler"]

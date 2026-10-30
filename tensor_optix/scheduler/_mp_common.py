"""Process-lifecycle helpers shared between ConcurrentTrialScheduler
(Optuna-driven) and StrategyTrialRunner (Phase 5, fixed-candidate-list
driven). Both own their own dispatch/polling loop and worker entry
point -- this module only extracts the genuinely identical mechanics:
choosing an mp context and the graceful-then-forceful process teardown
sequence.
"""

import multiprocessing as mp
import sys
from typing import List, Optional, Tuple


def mp_context():
    return mp.get_context("fork") if sys.platform.startswith("linux") else mp.get_context("spawn")


def shutdown_worker(proc, active_procs: Optional[List] = None) -> Tuple[Optional[int], bool]:
    """Join with an escalating grace period, then terminate/kill if the
    process hasn't exited on its own. Returns (exitcode, reaped)."""
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
    exitcode = proc.exitcode
    reaped = not proc.is_alive()
    if active_procs is not None and proc in active_procs:
        active_procs.remove(proc)
    return exitcode, reaped

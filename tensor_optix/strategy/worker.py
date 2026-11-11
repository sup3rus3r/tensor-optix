"""Module-level, picklable strategy-trial worker entry point.

Mirrors tensor_optix/scheduler/worker.py's _trial_worker shape exactly
(plain module-level function for picklability under Windows spawn, heavy
imports deferred inside the body, one thread per process). The one
structural difference: this worker optionally attaches a StrategyOp's
LoopCallback alongside the usual reporting callback, and runs with a
no-op BaseOptimizer when the active op touches learning_rate (confound
avoidance -- see candidate.py/comparator.py's design notes).
"""

import os
import time
from typing import Any, Callable, Dict, Optional


def _build_noop_optimizer():
    """Returns a BaseOptimizer whose suggest() returns hyperparams
    unchanged every call. Used for the paired candidate/baseline runs
    when the active strategy op mutates a hyperparameter the run's own
    BaseOptimizer would otherwise also adjust -- avoids confounding the
    comparison between two competing adjustment mechanisms."""
    from tensor_optix.core.base_optimizer import BaseOptimizer

    class _NoOpOptimizer(BaseOptimizer):
        def suggest(self, current_hyperparams, metrics_history):
            return current_hyperparams.copy()

    return _NoOpOptimizer()


def _strategy_trial_worker(
    trial_index: int,
    candidate_params: Optional[Dict[str, Any]],  # None => baseline run, no op attached
    op_name: Optional[str],
    agent_factory: Callable[[Dict[str, Any]], Any],
    pipeline_factory: Callable[[], Any],
    base_params: Dict[str, Any],
    rloptimizer_kwargs: Dict[str, Any],
    checkpoint_score_fn: Optional[Callable[[Any], float]],
    trial_steps: int,
    ckpt_dir: str,
    seed: int,
    report_queue,
    cancel_event,
    result_queue,
) -> None:
    import torch

    from tensor_optix.core.checkpoint_registry import CheckpointRegistry
    from tensor_optix.core.loop_controller import LoopCallback
    from tensor_optix.optimizer import RLOptimizer
    from tensor_optix.strategy.candidate import StrategyCandidate

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    # Agents and replay buffers also draw from numpy / stdlib RNGs
    # (epsilon-greedy, PER sampling, minibatch shuffles) — seed them too so a
    # trial is reproducible from its seed, not just its torch stream.
    import random as _random
    import numpy as _np
    _np.random.seed(seed % (2**32))
    _random.seed(seed)

    started_at = time.time()
    pid = os.getpid()

    class _ReportingCallback(LoopCallback):
        def on_episode_end(self, episode_id, eval_metrics):
            if eval_metrics is None:
                return
            try:
                report_queue.put((trial_index, episode_id, eval_metrics.primary_score), timeout=0.1)
            except Exception:
                pass
            if cancel_event.is_set():
                self._stop_fn()

    try:
        os.makedirs(ckpt_dir, exist_ok=True)

        agent = agent_factory(base_params)
        pipeline = pipeline_factory()
        registry = CheckpointRegistry(ckpt_dir)

        steps_per_episode = getattr(pipeline, "n_steps", None) or getattr(pipeline, "steps_per_episode", None)
        if steps_per_episode is not None and steps_per_episode > 0:
            max_episodes = max(1, trial_steps // steps_per_episode)
        else:
            max_episodes = trial_steps

        callbacks = [_ReportingCallback()]
        skipped = False
        if candidate_params is not None:
            candidate = StrategyCandidate(
                candidate_id="", op_name=op_name, op_version=1, params=candidate_params,
            )
            op = candidate.to_op()
            cb = op.attach(
                agent, checkpoint_registry=registry, policy_manager=None,
                loop_controller=None, pipeline=pipeline,
            )
            if cb is None:
                skipped = True
            else:
                callbacks.append(cb)

        if skipped:
            result_queue.put({
                "trial_index": trial_index, "status": "skipped",
                "reason": "agent/pipeline does not support this op", "score": None,
                "weights_path": None, "error": None, "pid": pid,
                "started_at": started_at, "finished_at": time.time(),
            })
            return

        # Confound avoidance: when a strategy op touches learning_rate,
        # run with a no-op tuning optimizer so the comparison isn't
        # confounded by two competing LR-adjustment mechanisms.
        touches_lr = candidate_params is not None and op_name == "lr_trend"
        rl_kwargs = dict(rloptimizer_kwargs)
        if touches_lr:
            rl_kwargs = {k: v for k, v in rl_kwargs.items() if k != "optimizer"}
            rl_kwargs["optimizer"] = _build_noop_optimizer()

        rl = RLOptimizer(
            agent=agent, pipeline=pipeline, max_episodes=max_episodes,
            checkpoint_dir=ckpt_dir, checkpoint_score_fn=checkpoint_score_fn,
            callbacks=callbacks, **rl_kwargs,
        )
        callbacks[0]._stop_fn = rl.stop
        # A registry-backed op (CheckpointRollbackOp) must share the run's own
        # CheckpointRegistry instance. A second instance on the same directory
        # caches the first "best" it loads and never sees later saves, so a
        # rollback would restore a stale snapshot instead of the current best.
        for cb in callbacks[1:]:
            if getattr(cb, "_registry", None) is registry:
                cb._registry = rl._controller._registry

        rl.run()

        best = rl.best_snapshot
        status = "cancelled" if cancel_event.is_set() else "completed"
        result_queue.put({
            "trial_index": trial_index, "status": status,
            "score": best.eval_metrics.primary_score if best is not None else None,
            "weights_path": best.weights_path if best is not None else None,
            "error": None, "pid": pid, "started_at": started_at, "finished_at": time.time(),
        })
    except Exception as exc:
        result_queue.put({
            "trial_index": trial_index, "status": "failed", "score": None,
            "weights_path": None, "error": str(exc), "pid": pid,
            "started_at": started_at, "finished_at": time.time(),
        })

"""Module-level, picklable trial worker entry point for the concurrent scheduler.

Mirrors tensor_optix/distributed/async_learner.py's _actor_loop shape: a
plain module-level function (required for picklability under Windows
spawn), heavy imports deferred inside the function body, one thread per
process (torch.set_num_threads(1)) to avoid BLAS contention across workers.

The worker never touches an optuna Trial/Study or an ExperimentStore — it
only receives a plain sampled-params dict and reports back over queues,
exactly like async_learner's actors only read shared memory and write
trajectories to a queue. All Study/ledger interaction happens in the
parent process (tensor_optix/scheduler/pool.py).
"""

import os
import time
from typing import Any, Callable, Dict, Optional


def _trial_worker(
    trial_number: int,
    params: Dict[str, Any],
    agent_factory: Callable[[Dict[str, Any]], Any],
    pipeline_factory: Callable[[], Any],
    val_pipeline_factory: Optional[Callable[[], Any]],
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

    from tensor_optix.core.loop_controller import LoopCallback
    from tensor_optix.optimizer import RLOptimizer

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
                report_queue.put(
                    (trial_number, episode_id, eval_metrics.primary_score), timeout=0.1
                )
            except Exception:
                pass  # queue full — drop, matches async_learner's drop-on-full discipline

            if cancel_event.is_set():
                self._stop_fn()

    try:
        os.makedirs(ckpt_dir, exist_ok=True)

        agent = agent_factory(params)
        pipeline = pipeline_factory()
        val_pipeline = val_pipeline_factory() if val_pipeline_factory else None

        steps_per_episode = getattr(pipeline, "n_steps", None) or getattr(pipeline, "steps_per_episode", None)
        if steps_per_episode is not None and steps_per_episode > 0:
            max_episodes = max(1, trial_steps // steps_per_episode)
        else:
            max_episodes = trial_steps

        reporter = _ReportingCallback()

        rl = RLOptimizer(
            agent=agent,
            pipeline=pipeline,
            max_episodes=max_episodes,
            checkpoint_dir=ckpt_dir,
            val_pipeline=val_pipeline,
            checkpoint_score_fn=checkpoint_score_fn,
            callbacks=[reporter],
            **rloptimizer_kwargs,
        )
        reporter._stop_fn = rl.stop

        rl.run()

        best = rl.best_snapshot
        status = "cancelled" if cancel_event.is_set() else "completed"
        result_queue.put({
            "trial_number": trial_number,
            "status": status,
            "score": best.eval_metrics.primary_score if best is not None else None,
            "weights_path": best.weights_path if best is not None else None,
            "error": None,
            "pid": pid,
            "started_at": started_at,
            "finished_at": time.time(),
        })
    except Exception as exc:
        result_queue.put({
            "trial_number": trial_number,
            "status": "failed",
            "score": None,
            "weights_path": None,
            "error": str(exc),
            "pid": pid,
            "started_at": started_at,
            "finished_at": time.time(),
        })

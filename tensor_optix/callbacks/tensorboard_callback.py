"""
TensorBoardCallback — TensorBoard logging for tensor-optix training loops.

Requires: pip install torch  (SummaryWriter ships with torch.utils.tensorboard)
          or: pip install tensor-optix[tensorboard]

Logs the same signal groups as WandbCallback using TensorBoard's SummaryWriter:

    score/primary          — eval primary_score at every eval window
    score/best             — best score seen so far (on_improvement)
    score/at_degradation   — score at degradation event

    metrics/*              — all keys from EvalMetrics.metrics (algorithm
                             diagnostics + reward stats + generalization_gap)

    hyperparams/*          — current hyperparam values on improvement and
                             on every SPSA update

    spsa/step_magnitude    — L2 norm of relative per-param changes per SPSA step

    events/improvement     — 1 on new best checkpoint
    events/plateau         — 1 on COOLING state entry
    events/convergence     — 1 on DORMANT declaration
    events/degradation     — 1 on detected performance drop

TensorBoard scalar tags use '/' as the group separator, which TensorBoard
renders as grouped scalar panels in the UI.
"""

import math
import os
from typing import Optional

from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, PolicySnapshot


class TensorBoardCallback(LoopCallback):
    """
    Logs tensor-optix loop events to TensorBoard via SummaryWriter.

    Args:
        log_dir:      Directory for TensorBoard event files.
                      Defaults to ``./runs/tensor_optix``.
        comment:      Suffix appended to log_dir when log_dir is not set.
                      Passed to SummaryWriter(comment=...).
        flush_secs:   How often (seconds) the writer flushes to disk.
                      Lower values give more real-time visibility at the cost
                      of more I/O. Default 10.
        **writer_kwargs: Any additional kwargs forwarded to SummaryWriter().

    Example::

        from tensor_optix.callbacks import TensorBoardCallback

        opt = RLOptimizer(
            agent=agent,
            pipeline=pipeline,
            callbacks=[TensorBoardCallback(log_dir="./runs/cartpole-ppo")],
        )
        opt.run()
        # Then: tensorboard --logdir ./runs
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        flush_secs: int = 10,
        **writer_kwargs,
    ):
        self._log_dir = log_dir
        self._comment = comment
        self._flush_secs = flush_secs
        self._writer_kwargs = writer_kwargs

        self._writer = None
        self._last_episode_id: int = 0

    # ------------------------------------------------------------------
    # LoopCallback hooks
    # ------------------------------------------------------------------

    def on_loop_start(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "torch is required for TensorBoardCallback (SummaryWriter). "
                "Install it with: pip install torch  "
                "or: pip install tensor-optix[tensorboard]"
            )
        log_dir = self._log_dir or os.path.join("runs", "tensor_optix")
        self._writer = SummaryWriter(
            log_dir=log_dir,
            comment=self._comment,
            flush_secs=self._flush_secs,
            **self._writer_kwargs,
        )

    def on_loop_stop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        self._last_episode_id = episode_id
        if eval_metrics is None or self._writer is None:
            return

        self._writer.add_scalar("score/primary", eval_metrics.primary_score, episode_id)

        for k, v in eval_metrics.metrics.items():
            if k == "primary_score":
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"metrics/{k}", float(v), episode_id)

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        if self._writer is None:
            return

        ep = snapshot.episode_id
        self._writer.add_scalar("score/best", snapshot.eval_metrics.primary_score, ep)
        self._writer.add_scalar("events/improvement", 1, ep)

        for k, v in snapshot.hyperparams.params.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"hyperparams/{k}", float(v), ep)

    def on_plateau(self, episode_id: int, state) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("events/plateau", 1, episode_id)

    def on_dormant(self, episode_id: int) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("events/convergence", 1, episode_id)

    def on_degradation(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("events/degradation", 1, episode_id)
        if eval_metrics is not None:
            self._writer.add_scalar("score/at_degradation", eval_metrics.primary_score, episode_id)

    def on_hyperparam_update(self, old_params: dict, new_params: dict) -> None:
        if self._writer is None:
            return

        sq_sum = 0.0
        n_changed = 0

        for k, new_v in new_params.items():
            if not isinstance(new_v, (int, float)):
                continue
            new_f = float(new_v)
            self._writer.add_scalar(f"hyperparams/{k}", new_f, self._last_episode_id)

            if k in old_params and isinstance(old_params[k], (int, float)):
                old_f = float(old_params[k])
                rel = (new_f - old_f) / (abs(old_f) + 1e-12)
                sq_sum += rel * rel
                n_changed += 1

        if sq_sum > 0.0:
            self._writer.add_scalar(
                "spsa/step_magnitude", math.sqrt(sq_sum), self._last_episode_id
            )

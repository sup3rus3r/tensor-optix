"""
WandbCallback — Weights & Biases logging for tensor-optix training loops.

Requires: pip install wandb  (or pip install tensor-optix[wandb])

Hooks into every LoopCallback event and logs the following signal groups:

    score/primary          — eval primary_score at every eval window
    score/best             — best score seen so far (on_improvement)
    score/at_degradation   — score at degradation event

    metrics/*              — all keys from EvalMetrics.metrics, which includes:
                             algorithm diagnostics (policy_loss, entropy, approx_kl,
                             explained_var for PPO; actor_loss, critic_loss, alpha
                             for SAC), reward stats, and generalization_gap when
                             a val_pipeline is active

    hyperparams/*          — current hyperparam values logged on every improvement
                             and on every SPSA update

    spsa/step_magnitude    — L2 norm of relative per-param changes from each SPSA
                             step. Measures SPSA aggression: large during plateau
                             (probe scale widens), small during active improvement
                             (probe scale shrinks). Computed as:
                                 ||Δx / |x_old|||₂   across all tuned params

    events/improvement     — 1 whenever a new best checkpoint is saved
    events/plateau         — 1 on each COOLING state entry
    events/convergence     — 1 when DORMANT is declared (training complete)
    events/degradation     — 1 on each detected performance drop
"""

import math
from typing import Optional

from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, PolicySnapshot


class WandbCallback(LoopCallback):
    """
    Logs tensor-optix loop events to Weights & Biases.

    Args:
        project:      W&B project name. Passed to wandb.init().
        name:         Run name. Passed to wandb.init().
        config:       Dict of static run configuration (e.g. env name, arch).
                      Merged with hyperparams logged during training.
        tags:         List of string tags for the W&B run.
        group:        W&B run group (for grouping seeds in the UI).
        resume:       Whether to resume an existing run ("allow", "must", etc.).
                      Passed through to wandb.init().
        **init_kwargs: Any additional kwargs forwarded to wandb.init().

    Example::

        from tensor_optix.callbacks import WandbCallback

        opt = RLOptimizer(
            agent=agent,
            pipeline=pipeline,
            callbacks=[WandbCallback(project="my-project", name="cartpole-ppo")],
        )
        opt.run()
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        resume: Optional[str] = None,
        **init_kwargs,
    ):
        self._project = project
        self._name = name
        self._config = config or {}
        self._tags = tags or []
        self._group = group
        self._resume = resume
        self._init_kwargs = init_kwargs

        self._wandb = None
        self._run = None
        self._last_episode_id: int = 0   # tracks step for on_hyperparam_update

    # ------------------------------------------------------------------
    # LoopCallback hooks
    # ------------------------------------------------------------------

    def on_loop_start(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install it with: pip install wandb  "
                "or: pip install tensor-optix[wandb]"
            )
        self._wandb = wandb
        kwargs = dict(
            project=self._project,
            name=self._name,
            config=self._config,
            tags=self._tags or None,
            group=self._group,
            resume=self._resume,
            **self._init_kwargs,
        )
        # Remove None values — wandb.init() treats None differently from absent
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._run = wandb.init(**kwargs)

    def on_loop_stop(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
            self._wandb = None

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        self._last_episode_id = episode_id
        if eval_metrics is None or self._wandb is None:
            return

        data = {"score/primary": eval_metrics.primary_score}

        # All algorithm diagnostics and reward stats from the evaluator.
        # TorchEvaluator / TFEvaluator merge train_diagnostics into metrics,
        # so policy_loss, entropy, approx_kl, explained_var (PPO), actor_loss,
        # critic_loss, alpha, buffer_size (SAC) appear here automatically.
        for k, v in eval_metrics.metrics.items():
            if k == "primary_score":
                continue  # already logged above
            if isinstance(v, (int, float)):
                data[f"metrics/{k}"] = float(v)

        self._wandb.log(data, step=episode_id)

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        if self._wandb is None:
            return

        data = {
            "score/best": snapshot.eval_metrics.primary_score,
            "events/improvement": 1,
        }
        for k, v in snapshot.hyperparams.params.items():
            if isinstance(v, (int, float)):
                data[f"hyperparams/{k}"] = float(v)

        self._wandb.log(data, step=snapshot.episode_id)

    def on_plateau(self, episode_id: int, state) -> None:
        if self._wandb is None:
            return
        self._wandb.log({"events/plateau": 1}, step=episode_id)

    def on_dormant(self, episode_id: int) -> None:
        if self._wandb is None:
            return
        self._wandb.log({"events/convergence": 1}, step=episode_id)

    def on_degradation(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if self._wandb is None:
            return
        data: dict = {"events/degradation": 1}
        if eval_metrics is not None:
            data["score/at_degradation"] = eval_metrics.primary_score
        self._wandb.log(data, step=episode_id)

    def on_hyperparam_update(self, old_params: dict, new_params: dict) -> None:
        if self._wandb is None:
            return

        data: dict = {}
        sq_sum = 0.0
        n_changed = 0

        for k, new_v in new_params.items():
            if not isinstance(new_v, (int, float)):
                continue
            new_f = float(new_v)
            data[f"hyperparams/{k}"] = new_f

            if k in old_params and isinstance(old_params[k], (int, float)):
                old_f = float(old_params[k])
                rel = (new_f - old_f) / (abs(old_f) + 1e-12)
                sq_sum += rel * rel
                n_changed += 1

        # SPSA step magnitude: L2 norm of relative per-param changes.
        # Invariant to parameter scale — a 10% shift in lr and a 10% shift in
        # clip_ratio contribute equally. Large during plateau (probe widens),
        # small during active improvement (probe shrinks on_improvement).
        # Only logged when at least one param actually moved (sq_sum > 0);
        # identical old/new params produce sq_sum = 0 and are not logged.
        if sq_sum > 0.0:
            data["spsa/step_magnitude"] = math.sqrt(sq_sum)

        if data:
            self._wandb.log(data, step=self._last_episode_id)

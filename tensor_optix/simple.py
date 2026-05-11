"""
tensor_optix.simple — one-line Optimizer entry point.

    opt = optix.Optimizer(agent, env)
    opt = optix.Optimizer(agent, env, n_envs=8)
    opt.run()

Handles:
  - window_size auto-computation via optimal_window_size
  - n_envs=1 → BatchPipeline, n_envs>1 → VectorBatchPipeline
  - neuroevo callbacks wired automatically when agent has _neuroevo_callbacks
  - SPSA wired when agent has default_param_bounds
  - Loud diagnostics for silent failures
"""

from __future__ import annotations

import logging
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Simplified entry point for training. Wraps RLOptimizer with sensible defaults.

    Parameters
    ----------
    agent:
        A fully-constructed agent from make_agent() or built manually.
    env:
        A Gymnasium environment. Used for window_size estimation and as the
        training environment when n_envs=1.
    n_envs:
        Number of parallel environments. When >1, pass env as a callable
        (``lambda: gym.make("EnvName")``) or a list of callables.
    window_size:
        Number of steps per training window. When None, computed automatically
        from env and agent type.
    max_episodes:
        Stop after this many episodes. None means run indefinitely.
    callbacks:
        Additional LoopCallback instances to attach.
    rollback_on_degradation:
        Roll back to the best checkpoint when performance degrades.
    checkpoint_dir:
        Directory for saving policy snapshots.
    kwargs:
        Additional keyword arguments forwarded to RLOptimizer.
    """

    def __init__(
        self,
        agent,
        env,
        n_envs: int = 1,
        window_size: Optional[int] = None,
        max_episodes: Optional[int] = None,
        callbacks: Optional[List] = None,
        rollback_on_degradation: bool = False,
        checkpoint_dir: str = "./tensor_optix_checkpoints",
        optimizer=None,
        **kwargs,
    ):
        from tensor_optix.optimizer import RLOptimizer
        from tensor_optix.pipeline.batch_pipeline import BatchPipeline
        from tensor_optix.pipeline.vector_pipeline import VectorBatchPipeline
        from tensor_optix.utils.window_size import optimal_window_size
        from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer

        # ------------------------------------------------------------------
        # Determine algorithm name for window_size computation
        # ------------------------------------------------------------------
        alg_name = _infer_algorithm_name(agent)

        # ------------------------------------------------------------------
        # Compute window_size
        # ------------------------------------------------------------------
        if window_size is None:
            _ref_env = env() if callable(env) else env
            window_size = optimal_window_size(_ref_env, algorithm=alg_name)
            logger.info(
                "Optimizer: auto window_size=%d for algorithm=%s", window_size, alg_name
            )

        # ------------------------------------------------------------------
        # Build pipeline
        # ------------------------------------------------------------------
        if n_envs == 1:
            _env = env() if callable(env) else env
            pipeline = BatchPipeline(env=_env, agent=agent, window_size=window_size)
        else:
            if callable(env) and not isinstance(env, list):
                env_fns = [env] * n_envs
            elif isinstance(env, list):
                env_fns = env
            else:
                import gymnasium as gym
                env_id = env.spec.id if hasattr(env, "spec") and env.spec else None
                if env_id is None:
                    raise ValueError(
                        "For n_envs>1, pass env as a callable (lambda: gym.make('EnvId')) "
                        "or a list of callables."
                    )
                env_fns = [lambda: gym.make(env_id)] * n_envs
            pipeline = VectorBatchPipeline(
                env_fns=env_fns, agent=agent, window_size=window_size
            )

        # ------------------------------------------------------------------
        # Collect callbacks
        # ------------------------------------------------------------------
        all_callbacks = list(callbacks or [])

        # Neuroevo callbacks (HebbianHook + TopologyController)
        if hasattr(agent, "_neuroevo_callbacks"):
            all_callbacks.extend(agent._neuroevo_callbacks)
            logger.info(
                "Optimizer: wiring %d neuroevo callbacks (HebbianHook, TopologyController)",
                len(agent._neuroevo_callbacks),
            )

        # ------------------------------------------------------------------
        # SPSA optimizer — explicit takes priority over auto-detection
        # ------------------------------------------------------------------
        if optimizer is not None:
            spsa_opt = optimizer
            logger.info("Optimizer: using provided optimizer %s", type(optimizer).__name__)
        elif hasattr(agent, "default_param_bounds") and agent.default_param_bounds:
            spsa_opt = SPSAOptimizer(
                param_bounds=agent.default_param_bounds,
                log_params=getattr(agent, "default_log_params", []),
            )
            logger.info("Optimizer: SPSA active with bounds: %s", list(agent.default_param_bounds.keys()))
        else:
            spsa_opt = None
            logger.warning(
                "Optimizer: SPSA inactive — agent has no default_param_bounds. "
                "Add default_param_bounds to enable online hyperparameter tuning."
            )

        # ------------------------------------------------------------------
        # Assemble RLOptimizer
        # ------------------------------------------------------------------
        self._rl_optimizer = RLOptimizer(
            agent=agent,
            pipeline=pipeline,
            optimizer=spsa_opt,
            callbacks=all_callbacks if all_callbacks else None,
            max_episodes=max_episodes,
            rollback_on_degradation=rollback_on_degradation,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )

    def run(self):
        """Start training. Blocks until max_episodes or KeyboardInterrupt."""
        return self._rl_optimizer.run()


def _infer_algorithm_name(agent) -> str:
    """Infer algorithm string from agent class name for window_size lookup."""
    cls = type(agent).__name__.lower()
    if "ppo" in cls or "graph" in cls:
        return "ppo"
    if "sac" in cls:
        return "sac"
    if "td3" in cls:
        return "td3"
    if "dqn" in cls or "rainbow" in cls:
        return "dqn"
    # Fallback: on-policy agents benefit from larger windows
    try:
        return "ppo" if agent.is_on_policy else "sac"
    except AttributeError:
        return "default"

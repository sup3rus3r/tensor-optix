from typing import Optional, List
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.loop_controller import LoopController, LoopCallback
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.types import LoopState, PolicySnapshot
from tensor_optix.adapters.tensorflow.tf_evaluator import TFEvaluator
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer


class RLOptimizer:
    """
    Main public API. This is what users import.

    Assembles all components and delegates to LoopController.
    Sensible defaults for everything — minimal required args: agent, pipeline.

    Minimal usage:
        optimizer = RLOptimizer(agent=my_tf_agent, pipeline=my_pipeline)
        optimizer.run()

    Custom usage:
        optimizer = RLOptimizer(
            agent=my_tf_agent,
            pipeline=LivePipeline(data_source=feed),
            evaluator=MyCustomEvaluator(),
            optimizer=PBTOptimizer(param_bounds={...}),
            rollback_on_degradation=True,
            checkpoint_dir="./checkpoints",
        )
        optimizer.run()
    """

    def __init__(
        self,
        agent: BaseAgent,
        pipeline: BasePipeline,
        evaluator: Optional[BaseEvaluator] = None,
        optimizer: Optional[BaseOptimizer] = None,
        checkpoint_dir: str = "./tensor_optix_checkpoints",
        max_snapshots: int = 10,
        rollback_on_degradation: bool = False,
        improvement_margin: float = 0.0,
        max_episodes: Optional[int] = None,
        base_interval: int = 1,
        backoff_factor: float = 2.0,
        max_interval_episodes: int = 100,
        plateau_threshold: int = 5,
        dormant_threshold: int = 20,
        degradation_threshold: float = 0.95,
        callbacks: Optional[List[LoopCallback]] = None,
        val_pipeline: Optional[BasePipeline] = None,
    ):
        # Inject agent into pipeline if pipeline supports it
        if hasattr(pipeline, "set_agent"):
            pipeline.set_agent(agent)

        _evaluator = evaluator or TFEvaluator()
        _optimizer = optimizer or BackoffOptimizer()

        _registry = CheckpointRegistry(
            checkpoint_dir=checkpoint_dir,
            max_snapshots=max_snapshots,
        )
        _scheduler = BackoffScheduler(
            base_interval=base_interval,
            backoff_factor=backoff_factor,
            max_interval_episodes=max_interval_episodes,
            plateau_threshold=plateau_threshold,
            dormant_threshold=dormant_threshold,
            degradation_threshold=degradation_threshold,
        )

        self._controller = LoopController(
            agent=agent,
            evaluator=_evaluator,
            optimizer=_optimizer,
            pipeline=pipeline,
            checkpoint_registry=_registry,
            backoff_scheduler=_scheduler,
            rollback_on_degradation=rollback_on_degradation,
            improvement_margin=improvement_margin,
            max_episodes=max_episodes,
            callbacks=callbacks,
            val_pipeline=val_pipeline,
        )

    def run(self) -> None:
        """Start the autonomous loop. Blocks until stopped or max_episodes reached."""
        self._controller.run()

    def stop(self) -> None:
        """Signal graceful shutdown after current episode."""
        self._controller.stop()

    @property
    def state(self) -> LoopState:
        """Current LoopState."""
        return self._controller.state

    @property
    def best_snapshot(self) -> Optional[PolicySnapshot]:
        """Best known PolicySnapshot."""
        return self._controller.best_snapshot

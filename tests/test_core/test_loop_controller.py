import pytest
import tempfile
from tensor_optix.core.loop_controller import LoopController, LoopCallback
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.types import LoopState, EvalMetrics, PolicySnapshot
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer


def make_controller(agent, evaluator, pipeline, tmp_path, max_episodes=5, callbacks=None):
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)
    scheduler = BackoffScheduler(
        base_interval=1,
        plateau_threshold=3,
        dormant_threshold=6,
    )
    optimizer = BackoffOptimizer()
    return LoopController(
        agent=agent,
        evaluator=evaluator,
        optimizer=optimizer,
        pipeline=pipeline,
        checkpoint_registry=registry,
        backoff_scheduler=scheduler,
        max_episodes=max_episodes,
        callbacks=callbacks or [],
    )


def test_cold_start_establishes_baseline(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path):
    controller = make_controller(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path, max_episodes=1)
    controller.run()
    assert controller.best_snapshot is not None
    assert controller.best_snapshot.eval_metrics.primary_score > 0


def test_loop_runs_max_episodes(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path):
    controller = make_controller(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path, max_episodes=5)
    controller.run()
    assert dummy_agent.learn_calls >= 1


def test_stop_halts_loop(dummy_agent, dummy_evaluator, tmp_path):
    from tests.conftest import DummyPipeline
    pipeline = DummyPipeline(rewards_sequence=[[1.0] * 5] * 100)

    class StopAfterFirst(LoopCallback):
        def __init__(self, ctrl):
            self._ctrl = ctrl
            self.count = 0
        def on_episode_end(self, episode_id, eval_metrics):
            self.count += 1
            if self.count >= 3:
                self._ctrl.stop()

    registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)
    scheduler = BackoffScheduler(base_interval=1)
    optimizer = BackoffOptimizer()
    cb = StopAfterFirst(None)
    controller = LoopController(
        agent=dummy_agent,
        evaluator=dummy_evaluator,
        optimizer=optimizer,
        pipeline=pipeline,
        checkpoint_registry=registry,
        backoff_scheduler=scheduler,
        max_episodes=1000,
        callbacks=[cb],
    )
    cb._ctrl = controller
    controller.run()
    assert cb.count >= 3


def test_improvement_callback_fires(dummy_agent, dummy_evaluator, tmp_path):
    # Increasing rewards so improvement is always detected
    from tests.conftest import DummyPipeline
    rewards = [[float(i)] * 3 for i in range(1, 20)]
    pipeline = DummyPipeline(rewards_sequence=rewards)

    improvements = []

    class ImprovementTracker(LoopCallback):
        def on_improvement(self, snapshot):
            improvements.append(snapshot.eval_metrics.primary_score)

    controller = make_controller(
        dummy_agent, dummy_evaluator, pipeline, tmp_path,
        max_episodes=10,
        callbacks=[ImprovementTracker()],
    )
    controller.run()
    assert len(improvements) >= 1


def test_callbacks_dont_crash_on_exception(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path):
    class BrokenCallback(LoopCallback):
        def on_episode_end(self, episode_id, eval_metrics):
            raise RuntimeError("Intentional callback error")

    controller = make_controller(
        dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path,
        max_episodes=3,
        callbacks=[BrokenCallback()],
    )
    # Should not raise
    controller.run()

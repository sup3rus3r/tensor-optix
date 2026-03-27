"""
End-to-end integration tests.
These tests use real TF models and real Gymnasium environments.
"""
import pytest
import numpy as np
import tensorflow as tf
import gymnasium as gym
from tensor_optix import (
    RLOptimizer, TFAgent, TFEvaluator,
    BatchPipeline, HyperparamSet,
    BackoffOptimizer, PBTOptimizer,
    LoopCallback,
)


def make_cartpole_agent():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(2),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    hyperparams = HyperparamSet(
        params={"learning_rate": 1e-3, "gamma": 0.99},
        episode_id=0,
    )
    return TFAgent(model=model, optimizer=optimizer, hyperparams=hyperparams)


def make_cartpole_pipeline(agent):
    env = gym.make("CartPole-v1")
    return BatchPipeline(env=env, agent=agent, window_size=50)


def test_rl_optimizer_runs_to_completion(tmp_path):
    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=5,
    )
    opt.run()
    assert opt.best_snapshot is not None
    assert opt.best_snapshot.eval_metrics.primary_score is not None


def test_best_snapshot_score_is_float(tmp_path):
    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=3,
    )
    opt.run()
    assert isinstance(opt.best_snapshot.eval_metrics.primary_score, float)


def test_with_pbt_optimizer(tmp_path):
    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    pbt = PBTOptimizer(
        param_bounds={"learning_rate": (1e-5, 1e-2)},
        explore_scale=0.1,
    )
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        optimizer=pbt,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=5,
    )
    opt.run()
    assert opt.best_snapshot is not None


def test_custom_evaluator(tmp_path):
    from tensor_optix import BaseEvaluator
    from tensor_optix.core.types import EpisodeData, EvalMetrics

    class TotalRewardEvaluator(BaseEvaluator):
        def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
            score = float(sum(episode_data.rewards))
            return EvalMetrics(primary_score=score, metrics={"total": score}, episode_id=episode_data.episode_id)

    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        evaluator=TotalRewardEvaluator(),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=4,
    )
    opt.run()
    assert opt.best_snapshot is not None


def test_callbacks_fire(tmp_path):
    fired = {"episode_end": 0, "loop_start": 0, "loop_stop": 0}

    class TrackingCallback(LoopCallback):
        def on_loop_start(self):
            fired["loop_start"] += 1
        def on_loop_stop(self):
            fired["loop_stop"] += 1
        def on_episode_end(self, episode_id, eval_metrics):
            fired["episode_end"] += 1

    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=4,
        callbacks=[TrackingCallback()],
    )
    opt.run()
    assert fired["loop_start"] == 1
    assert fired["loop_stop"] == 1
    assert fired["episode_end"] >= 1


def test_rollback_on_degradation(tmp_path):
    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        rollback_on_degradation=True,
        degradation_threshold=0.99,  # Very sensitive — will trigger rollback
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=6,
    )
    # Should not raise
    opt.run()


def test_stop_method_works(tmp_path):
    import threading

    agent = make_cartpole_agent()
    pipeline = make_cartpole_pipeline(agent)
    opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_episodes=1000,
    )

    def stop_after_delay():
        import time
        time.sleep(0.5)
        opt.stop()

    t = threading.Thread(target=stop_after_delay)
    t.start()
    opt.run()
    t.join()
    # If we reach here without hanging, stop() works

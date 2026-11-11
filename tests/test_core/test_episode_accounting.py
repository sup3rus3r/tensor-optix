"""
Finished-episode accounting: scores are full episode returns tracked across
windows; windows without a finished episode are not evaluations; a minimum
number of finished episodes is required before a plateau can be declared.
"""
import numpy as np

from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.loop_controller import LoopCallback, LoopController
from tensor_optix.core.reward_shape import EpisodeReturnTracker
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.multi_agent.multi_agent_runtime import MultiAgentRunner
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.pipeline.vector_pipeline import VectorBatchPipeline


class FixedLengthEnv:
    """Reward 1.0 per step; episode ends after `length` steps."""

    def __init__(self, length):
        self.length = length
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(1, dtype=np.float32), 1.0, self.t >= self.length, False, {}

    def close(self):
        pass


class NullAgent(BaseAgent):
    def __init__(self):
        self._hp = HyperparamSet(params={}, episode_id=0)

    def act(self, observation):
        return 0

    def learn(self, episode_data):
        return {}

    def get_hyperparams(self):
        return self._hp.copy()

    def set_hyperparams(self, hyperparams):
        self._hp = hyperparams.copy()

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass


class NoOpOptimizer(BaseOptimizer):
    def suggest(self, current_hyperparams, metrics_history):
        return current_hyperparams.copy()

    def on_improvement(self, metrics):
        pass

    def on_plateau(self, metrics_history):
        pass


class Recorder(LoopCallback):
    def __init__(self):
        self.evals = []        # (episode_id, score) for evaluated windows
        self.skipped = 0
        self.dormant = []

    def on_episode_end(self, episode_id, eval_metrics):
        if eval_metrics is None:
            self.skipped += 1
        else:
            self.evals.append((episode_id, eval_metrics.primary_score))

    def on_dormant(self, episode_id):
        self.dormant.append(episode_id)


def _controller(tmp_path, env, window_size, max_episodes, **kw):
    agent = NullAgent()
    rec = Recorder()
    kw.setdefault("backoff_scheduler", BackoffScheduler(
        backoff_factor=1.0, plateau_threshold=2, dormant_threshold=4,
    ))
    controller = LoopController(
        agent=agent, evaluator=TorchEvaluator(), optimizer=NoOpOptimizer(),
        pipeline=BatchPipeline(env, agent, window_size=window_size),
        checkpoint_registry=CheckpointRegistry(str(tmp_path)),
        max_episodes=max_episodes, callbacks=[rec], **kw,
    )
    return controller, rec


# ----------------------------------------------------------------------------
# Tracker / pipelines / evaluator
# ----------------------------------------------------------------------------

def test_tracker_carries_partial_episode_across_windows():
    tr = EpisodeReturnTracker()
    assert tr.update([1.0, 1.0, 1.0], [False, False, False]) == []
    assert tr.update([1.0, 1.0, 5.0], [False, True, False]) == [5.0]
    assert tr.update([1.0], [True]) == [6.0]


def test_tracker_keeps_interleaved_envs_separate():
    tr = EpisodeReturnTracker(n_envs=2)
    # layout [t * n_envs + env]: env0 gets 1s, env1 gets 10s
    assert tr.update([1.0, 10.0, 1.0, 10.0], [False, False, False, True]) == [20.0]
    assert tr.update([1.0, 10.0], [True, False]) == [3.0]


def test_batch_pipeline_reports_full_returns_of_long_episodes():
    env = FixedLengthEnv(length=25)
    pipeline = BatchPipeline(env, NullAgent(), window_size=10)
    pipeline.setup()
    gen = pipeline.episodes()
    reported = [next(gen).metadata["episode_returns"] for _ in range(5)]
    # 50 steps: episodes end at steps 25 and 50 -> windows 3 and 5
    assert reported == [[], [], [25.0], [], [25.0]]


def test_vector_pipeline_reports_full_returns_per_env():
    pipeline = VectorBatchPipeline([lambda: FixedLengthEnvGym(7)] * 2, NullAgent(), window_size=5)
    pipeline.setup()
    gen = pipeline.episodes()
    first, second = next(gen), next(gen)
    pipeline.teardown()
    assert first.metadata["episode_returns"] == []
    assert second.metadata["episode_returns"] == [7.0, 7.0]


def test_evaluator_scores_full_return_and_reports_episode_count():
    ev = TorchEvaluator()

    def window(returns):
        return EpisodeData(
            observations=np.zeros((3, 1)), actions=np.zeros(3), rewards=[1.0, 1.0, 1.0],
            terminated=[False, False, False], truncated=[False, False, False],
            infos=[{}] * 3, episode_id=0, metadata={"episode_returns": returns},
        )

    done = ev.score(window([25.0, 35.0]), {})
    assert done.primary_score == 30.0 and done.metrics["episodes_completed"] == 2
    none = ev.score(window([]), {})
    assert none.metrics["episodes_completed"] == 0
    assert none.primary_score == 1.0          # per-step fallback, flagged by the count


def test_custom_score_fn_does_not_report_episode_count():
    ev = TorchEvaluator(primary_score_fn=lambda ep, diag: 42.0)
    ep = EpisodeData(observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[1.0],
                     terminated=[False], truncated=[False], infos=[{}], episode_id=0)
    assert "episodes_completed" not in ev.score(ep, {}).metrics


# ----------------------------------------------------------------------------
# LoopController
# ----------------------------------------------------------------------------

def test_windows_without_finished_episode_are_not_evaluated(tmp_path):
    # 25-step episodes, 10-step windows: an episode finishes in windows 3, 5, 8, 10, ...
    controller, rec = _controller(tmp_path, FixedLengthEnv(25), window_size=10, max_episodes=21,
                                  min_completed_episodes_before_dormant=1000)
    controller.run()
    scores_after_first_episode = [s for _, s in rec.evals if s == 25.0]
    assert scores_after_first_episode, "no window was scored by episode return"
    # After the first finished episode every evaluated window is on the return scale;
    # the per-step fallback (1.0) only appears before it.
    first_return_idx = next(i for i, (_, s) in enumerate(rec.evals) if s == 25.0)
    assert all(s == 25.0 for _, s in rec.evals[first_return_idx:])
    assert all(s == 1.0 for _, s in rec.evals[:first_return_idx])
    assert rec.skipped > 0


def test_score_scale_change_resets_best(tmp_path):
    # Negative rewards: per-step fallback (-1) is "better" than the return (-25).
    class NegEnv(FixedLengthEnv):
        def step(self, action):
            obs, _, term, trunc, info = super().step(action)
            return obs, -1.0, term, trunc, info

    controller, _ = _controller(tmp_path, NegEnv(25), window_size=10, max_episodes=12,
                                min_completed_episodes_before_dormant=1000)
    controller.run()
    # Without the reset the per-step -1.0 would remain the unbeatable best forever.
    assert controller.best_snapshot.eval_metrics.primary_score == -25.0


def test_no_dormant_before_min_completed_episodes(tmp_path):
    # One finished episode per window, perfectly flat score.
    controller, rec = _controller(tmp_path, FixedLengthEnv(10), window_size=10, max_episodes=200,
                                  min_completed_episodes_before_dormant=30)
    controller.run()
    assert rec.dormant, "flat score never reached DORMANT"
    # cold start = episode 0; 30 finished episodes are reached at episode 29,
    # then dormant_threshold=4 flat evals are needed.
    assert rec.dormant[0] >= 30


def test_env_whose_episodes_never_finish_is_scored_but_not_declared_converged(tmp_path):
    # Default: no finished episodes -> the per-step fallback keeps the loop
    # evaluating, but a flat fallback score is never called convergence.
    controller, rec = _controller(tmp_path, FixedLengthEnv(10 ** 9), window_size=10, max_episodes=100)
    controller.run()
    assert rec.skipped == 0
    assert len(rec.evals) == 100
    assert rec.dormant == []


def test_continuing_task_opts_out_with_zero_minimum(tmp_path):
    controller, rec = _controller(tmp_path, FixedLengthEnv(10 ** 9), window_size=10, max_episodes=200,
                                  min_completed_episodes_before_dormant=0)
    controller.run()
    assert rec.dormant and rec.dormant[0] < 200


# ----------------------------------------------------------------------------
# MultiAgentRunner
# ----------------------------------------------------------------------------

class TrackedSeatPipeline:
    """Single seat; an episode of `ep_len` windows, return = ep_len."""

    def __init__(self, ep_len, n_windows):
        self.ep_len, self.n_windows = ep_len, n_windows

    def setup(self):
        pass

    def teardown(self):
        pass

    def episodes(self):
        for i in range(self.n_windows):
            finished = (i + 1) % self.ep_len == 0
            yield {"a": EpisodeData(
                observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[1.0],
                terminated=[finished], truncated=[False], infos=[{}], episode_id=i,
                metadata={"episode_returns": [float(self.ep_len)] if finished else []},
            )}


def test_runner_skips_windows_without_finished_episode():
    runner = MultiAgentRunner(TrackedSeatPipeline(ep_len=5, n_windows=40), {"a": NullAgent()},
                              stop_on_convergence=False)
    result = runner.run()
    seat = result.seats["a"]
    assert seat.completed_episodes == 8
    # 4 per-step-scored windows before the first episode, then one per finished episode
    assert seat.windows_scored == 4 + 8
    assert seat.best_score == 5.0                # reset from the per-step 1.0 scale


def test_runner_requires_min_completed_episodes_before_convergence():
    runner = MultiAgentRunner(TrackedSeatPipeline(ep_len=1, n_windows=200), {"a": NullAgent()},
                              plateau_threshold=2, dormant_threshold=4,
                              min_completed_episodes_before_converged=50)
    result = runner.run()
    assert result.stop_reason == "converged"
    assert result.windows_run >= 50


# gymnasium-registered variant for the vector pipeline test
import gymnasium as gym  # noqa: E402


class FixedLengthEnvGym(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, length):
        self.length = length
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(1, dtype=np.float32), 1.0, self.t >= self.length, False, {}

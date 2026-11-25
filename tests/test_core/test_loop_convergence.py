"""
LoopController wiring of the ConvergenceMonitor: every window is observed when
scoring is free, actions fire once on entry, collapses are reported for every
agent (rollback only for on-policy ones), and the backoff interval only paces
the costly steps.
"""
import numpy as np
import pytest

from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.convergence import ConvergenceMonitor, ConvergenceRegime
from tensor_optix.core.loop_controller import LoopCallback, LoopController
from tensor_optix.core.types import EpisodeData, HyperparamSet, LoopState


class ScriptPipeline:
    """One finished episode per window with a scripted return. Stateful, so the
    cold start's generator and the main loop's generator share one position."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def setup(self):
        pass

    def teardown(self):
        pass

    def episodes(self):
        while self.i < len(self.script):
            score = float(self.script[self.i])
            self.i += 1
            yield EpisodeData(
                observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[score],
                terminated=[True], truncated=[False], infos=[{}], episode_id=0,
                metadata={"episode_returns": [score]},
            )


class Agent(BaseAgent):
    is_on_policy = True

    def __init__(self):
        self._hp = HyperparamSet(params={}, episode_id=0)
        self.w = 0
        self.loads = 0

    def act(self, observation):
        return 0

    def learn(self, episode_data):
        self.w += 1
        return {}

    def get_hyperparams(self):
        return self._hp.copy()

    def set_hyperparams(self, hyperparams):
        self._hp = hyperparams.copy()

    def save_weights(self, path):
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "w.json"), "w") as f:
            json.dump({"w": self.w}, f)

    def load_weights(self, path):
        import json, os
        self.loads += 1
        with open(os.path.join(path, "w.json")) as f:
            self.w = json.load(f)["w"]


class OffPolicyAgent(Agent):
    is_on_policy = False


class CountingOptimizer(BaseOptimizer):
    def __init__(self):
        self.suggests = self.improvements = self.coolings = self.plateaus = 0

    def suggest(self, current_hyperparams, metrics_history):
        self.suggests += 1
        return current_hyperparams.copy()

    def on_improvement(self, metrics):
        self.improvements += 1

    def on_cooling(self, metrics_history):
        self.coolings += 1

    def on_plateau(self, metrics_history):
        self.plateaus += 1


class Recorder(LoopCallback):
    def __init__(self):
        self.plateau, self.dormant, self.degradation, self.reports = [], [], [], []

    def on_plateau(self, episode_id, state):
        self.plateau.append((episode_id, state))

    def on_dormant(self, episode_id):
        self.dormant.append(episode_id)

    def on_degradation(self, episode_id, eval_metrics):
        self.degradation.append(episode_id)

    def on_convergence(self, episode_id, report):
        self.reports.append((episode_id, report))


def _controller(tmp_path, script, agent=None, scheduler=None, **kw):
    agent = agent or Agent()
    opt = CountingOptimizer()
    rec = Recorder()
    controller = LoopController(
        agent=agent, evaluator=TorchEvaluator(), optimizer=opt,
        pipeline=ScriptPipeline(script),
        checkpoint_registry=CheckpointRegistry(str(tmp_path)),
        backoff_scheduler=scheduler or BackoffScheduler(plateau_threshold=5, dormant_threshold=15),
        callbacks=[rec], min_completed_episodes_before_dormant=0, **kw,
    )
    return controller, agent, opt, rec


def test_flat_run_cools_once_then_stops_at_dormant(tmp_path):
    controller, _, opt, rec = _controller(tmp_path, [5.0] * 200)
    controller.run()
    assert len(rec.plateau) == 1 and rec.plateau[0][1] == LoopState.COOLING
    assert len(rec.dormant) == 1
    assert opt.coolings == 1 and opt.plateaus == 1
    assert controller.state != LoopState.ACTIVE or controller.monitor.n_observations > 0
    # stopped at DORMANT, well before the script ran out
    assert rec.dormant[0] < 60


def test_every_window_is_observed_but_suggest_follows_the_backoff(tmp_path):
    sched = BackoffScheduler(backoff_factor=2.0, max_interval_episodes=8,
                             plateau_threshold=100, dormant_threshold=200)
    controller, _, opt, _ = _controller(tmp_path, [5.0] * 120, scheduler=sched)
    controller.run()
    # 1 cold-start + 119 main windows observed
    assert controller.monitor.n_observations == 120
    # suggest only on evaluation-due windows: far fewer than one per window
    assert 0 < opt.suggests < 40


def test_regime_changes_reach_on_convergence(tmp_path):
    script = [float(i) for i in range(40)] + [40.0] * 80
    controller, _, _, rec = _controller(tmp_path, script)
    controller.run()
    regimes = [r.regime for _, r in rec.reports]
    assert ConvergenceRegime.IMPROVING in regimes
    assert controller.convergence is controller.monitor.last_report


def test_collapse_is_reported_for_off_policy_agents_without_rollback(tmp_path):
    script = [float(i) for i in range(40)] + [0.0] * 40
    agent = OffPolicyAgent()
    controller, agent, _, rec = _controller(tmp_path, script, agent=agent,
                                            rollback_on_degradation=True)
    controller.run()
    assert rec.degradation, "collapse was never reported"
    # off-policy: never rolled back mid-run (the final restore in run() is the only load)
    assert agent.loads == 1


def test_on_policy_collapse_rolls_back_and_restarts_the_trend(tmp_path):
    script = [float(i) for i in range(40)] + [0.0] * 40
    controller, agent, _, rec = _controller(tmp_path, script, rollback_on_degradation=True)
    controller.run()
    assert rec.degradation
    # mid-run rollback(s) plus the final restore in run()
    assert agent.loads >= 2
    assert controller.monitor.peak is not None and controller.monitor.peak > 20


def test_explicit_thresholds_become_patience_in_episodes(tmp_path):
    controller, _, _, _ = _controller(
        tmp_path, [1.0] * 5, scheduler=BackoffScheduler(plateau_threshold=7, dormant_threshold=21),
    )
    assert controller.monitor.cooling_patience == 7
    assert controller.monitor.dormant_patience == 21


def test_default_thresholds_use_monitor_defaults(tmp_path):
    controller, _, _, _ = _controller(tmp_path, [1.0] * 5, scheduler=BackoffScheduler())
    defaults = ConvergenceMonitor()
    assert controller.monitor.cooling_patience == defaults.cooling_patience
    assert controller.monitor.dormant_patience == defaults.dormant_patience


def test_custom_monitor_is_used_and_attached(tmp_path):
    mon = ConvergenceMonitor(cooling_patience=3, dormant_patience=9)
    sched = BackoffScheduler()
    controller, _, _, _ = _controller(tmp_path, [1.0] * 5, scheduler=sched, convergence_monitor=mon)
    assert controller.monitor is mon
    mon.observe(1.0, at=0)
    sched.record_restart()            # legacy reset path reaches the monitor
    assert mon.state == LoopState.ACTIVE and mon.n_observations == 1


def test_cv_and_gap_thresholds_are_deprecated(tmp_path):
    with pytest.warns(DeprecationWarning):
        _controller(tmp_path, [1.0] * 5, cv_threshold=0.05)


def test_duck_typed_callback_without_new_hooks_is_fine(tmp_path, caplog):
    class Minimal:
        def on_dormant(self, episode_id):
            self.hit = episode_id

    cb = Minimal()
    controller, _, _, _ = _controller(tmp_path, [5.0] * 200)
    controller._callbacks.append(cb)
    controller.run()
    assert hasattr(cb, "hit")
    assert "raised during on_convergence" not in caplog.text

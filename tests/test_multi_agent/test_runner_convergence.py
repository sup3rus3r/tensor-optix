"""
MultiAgentRunner convergence + checkpoint logic, driven by scripted seats
(no PettingZoo needed).
"""
import json
import os

import numpy as np
import pytest

from tensor_optix.core.types import EpisodeData, HyperparamSet, LoopState
from tensor_optix.multi_agent.multi_agent_runtime import MultiAgentRunner


class ScriptedAgent:
    """Agent whose 'weights' are just the number of learn() calls so far."""

    def __init__(self):
        self.w = 0

    def act(self, obs):
        return 0

    def learn(self, episode_data):
        self.w += 1
        return {}

    def get_hyperparams(self):
        return HyperparamSet(params={}, episode_id=0)

    def set_hyperparams(self, hp):
        pass

    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "w.json"), "w") as f:
            json.dump({"w": self.w}, f)

    def load_weights(self, path):
        with open(os.path.join(path, "w.json")) as f:
            self.w = json.load(f)["w"]


class ScriptedPipeline:
    """Yields one window per step; each seat's window return follows a script
    (None = that seat got no turn this window)."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.torn_down = False

    def setup(self):
        pass

    def teardown(self):
        self.torn_down = True

    def episodes(self):
        n = max(len(s) for s in self._scripts.values())
        for i in range(n):
            window = {}
            for aid, script in self._scripts.items():
                score = script[i] if i < len(script) else script[-1]
                if score is None:
                    window[aid] = EpisodeData(
                        observations=np.array([]), actions=np.array([]), rewards=[],
                        terminated=[], truncated=[], infos=[], episode_id=i,
                    )
                else:
                    window[aid] = EpisodeData(
                        observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[float(score)],
                        terminated=[True], truncated=[False], infos=[{}], episode_id=i,
                    )
            yield window


def _runner(scripts, tmp_path=None, **kw):
    agents = {aid: ScriptedAgent() for aid in scripts}
    pipeline = ScriptedPipeline(scripts)
    kw.setdefault("plateau_threshold", 2)
    kw.setdefault("dormant_threshold", 4)
    runner = MultiAgentRunner(
        pipeline, agents,
        checkpoint_dir=str(tmp_path) if tmp_path is not None else None, **kw,
    )
    return runner, agents, pipeline


def test_stops_when_every_seat_has_converged():
    flat = [1.0] * 200
    runner, _, pipeline = _runner({"a": flat, "b": flat})
    result = runner.run()
    assert result.stop_reason == "converged" and result.converged
    assert result.windows_run < 200
    assert all(s.converged and s.state == LoopState.DORMANT for s in result.seats.values())
    assert pipeline.torn_down


def test_stop_when_all_waits_for_the_seat_still_improving():
    flat = [1.0] * 60
    rising = [float(i) for i in range(60)]
    runner, _, _ = _runner({"a": flat, "b": rising})
    result = runner.run()
    assert result.stop_reason == "pipeline_exhausted"
    assert result.seats["a"].converged
    assert not result.seats["b"].converged


def test_stop_when_any_stops_on_first_converged_seat():
    flat = [1.0] * 60
    rising = [float(i) for i in range(60)]
    runner, _, _ = _runner({"a": flat, "b": rising}, stop_when="any")
    result = runner.run()
    assert result.stop_reason == "converged"
    assert result.seats["a"].converged and not result.seats["b"].converged
    assert result.windows_run < 60


def test_max_windows_and_disabled_convergence():
    flat = [1.0] * 200
    runner, _, _ = _runner({"a": flat}, stop_on_convergence=False)
    result = runner.run(max_windows=50)
    assert result.stop_reason == "max_windows" and result.windows_run == 50


def test_checkpoints_best_and_restores_it(tmp_path):
    # Peak at window index 3, then worse. learn() ran 4 times at the peak.
    script = [1.0, 2.0, 3.0, 9.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    runner, agents, _ = _runner({"a": script}, tmp_path)
    result = runner.run()
    seat = result.seats["a"]
    assert seat.best_score == 9.0
    assert seat.best_snapshot is not None
    assert seat.best_snapshot.eval_metrics.primary_score == 9.0
    assert agents["a"].w == 4                      # restored to the peak snapshot
    assert os.path.isdir(os.path.join(str(tmp_path), "a"))


def test_restore_best_false_keeps_final_weights(tmp_path):
    script = [1.0, 9.0, 2.0, 2.0, 2.0, 2.0]
    runner, agents, _ = _runner({"a": script}, tmp_path, restore_best=False,
                                stop_on_convergence=False)
    result = runner.run()
    assert agents["a"].w == len(script)
    assert result.seats["a"].best_snapshot is not None


def test_no_checkpoint_dir_means_no_snapshots_and_no_restore():
    script = [1.0, 9.0, 2.0, 2.0]
    runner, agents, _ = _runner({"a": script}, stop_on_convergence=False)
    result = runner.run()
    assert result.seats["a"].best_score == 9.0
    assert result.seats["a"].best_snapshot is None
    assert agents["a"].w == len(script)


def test_seat_without_a_turn_is_not_learned_scored_or_counted():
    runner, agents, _ = _runner({"a": [1.0] * 6, "b": [1.0, None, None, 1.0, None, 1.0]},
                                stop_on_convergence=False)
    result = runner.run()
    assert agents["b"].w == 3
    assert result.seats["b"].windows_scored == 3
    assert result.seats["a"].windows_scored == 6


def test_callbacks_fire():
    events = []

    class CB:
        def on_run_start(self):
            events.append("start")

        def on_window_end(self, window_id, metrics):
            events.append(("window", window_id, sorted(metrics)))

        def on_improvement(self, agent_id, snapshot, score):
            events.append(("improved", agent_id, score))

        def on_converged(self, agent_id, window_id):
            events.append(("converged", agent_id))

        def on_run_stop(self, reason):
            events.append(("stop", reason))

    runner, _, _ = _runner({"a": [1.0] * 100}, callbacks=[CB()])
    runner.run()
    assert events[0] == "start"
    assert ("improved", "a", 1.0) in events
    assert ("converged", "a") in events
    assert events[-1] == ("stop", "converged")
    assert ("window", 0, ["a"]) in events


def test_stop_method_ends_run():
    runner, _, _ = _runner({"a": [float(i) for i in range(100)]})

    class Stopper:
        def on_window_end(self, window_id, metrics):
            if window_id == 4:
                runner.stop()

    runner.add_callback(Stopper())
    result = runner.run()
    assert result.stop_reason == "stopped" and result.windows_run == 5


def test_invalid_stop_when_rejected():
    with pytest.raises(ValueError):
        MultiAgentRunner(ScriptedPipeline({"a": [1.0]}), {"a": ScriptedAgent()}, stop_when="most")

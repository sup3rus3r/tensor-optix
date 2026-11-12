"""
MultiAgentRunner rollback on degradation. Scripted agents — no PettingZoo.
"""
import json
import os

import numpy as np
import pytest

from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.multi_agent import MultiAgentRunner, SelfPlayConfig


class Agent:
    is_on_policy = True

    def __init__(self):
        self.w = 0
        self.loads = 0

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
        self.loads += 1
        with open(os.path.join(path, "w.json")) as f:
            self.w = json.load(f)["w"]


class OffPolicyAgent(Agent):
    is_on_policy = False


class Pipeline:
    """Each seat finishes one episode per window with a scripted return."""

    def __init__(self, scripts):
        self._scripts = scripts
        self._agents = {}
        self._env = type("Env", (), {"possible_agents": list(scripts)})()

    def set_agents(self, agents):
        self._agents = agents

    def setup(self):
        pass

    def teardown(self):
        pass

    def episodes(self):
        n = max(len(s) for s in self._scripts.values())
        for i in range(n):
            window = {}
            for seat, script in self._scripts.items():
                score = float(script[min(i, len(script) - 1)])
                if seat in self._agents:
                    self._agents[seat].act(np.zeros(1))
                window[seat] = EpisodeData(
                    observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[score],
                    terminated=[True], truncated=[False], infos=[{}], episode_id=i,
                    metadata={"episode_returns": [score]},
                )
            yield window


# good for 6 windows (best = 10 at window 5, 6 learn() calls), then collapses
COLLAPSE = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0] + [1.0] * 6


def _run(scripts, tmp_path, agents=None, **kw):
    agents = agents or {seat: Agent() for seat in scripts}
    pipeline = Pipeline(scripts)
    pipeline.set_agents(dict(agents))
    kw.setdefault("optimizer", None)
    kw.setdefault("stop_on_convergence", False)
    kw.setdefault("restore_best", False)
    runner = MultiAgentRunner(pipeline, agents, checkpoint_dir=str(tmp_path), **kw)
    return agents, runner.run()


def test_rolls_back_to_best_after_confirmed_degradation(tmp_path):
    events = []

    class CB:
        def on_degradation(self, agent_id, window_id, score, rolled_back):
            events.append((agent_id, window_id, score, rolled_back))

    agents, result = _run({"a": COLLAPSE}, tmp_path, rollback_on_degradation=True,
                          min_consecutive_degradations=3, callbacks=[CB()])
    seat = result.seats["a"]
    assert seat.degradations == 2 and seat.rollbacks == 2
    # first confirmation on the 3rd degraded window (index 8)
    assert events[0] == ("a", 8, 1.0, True)
    assert agents["a"].loads == 2
    # after the last rollback (window 11) no further learn() ran: weights = best snapshot
    assert agents["a"].w == 6


def test_disabled_by_default(tmp_path):
    agents, result = _run({"a": COLLAPSE}, tmp_path)
    assert result.seats["a"].degradations == 0 and agents["a"].loads == 0
    assert agents["a"].w == len(COLLAPSE)


def test_needs_consecutive_degraded_windows(tmp_path):
    script = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 10.0]
    agents, result = _run({"a": script}, tmp_path, rollback_on_degradation=True,
                          min_consecutive_degradations=3)
    assert result.seats["a"].degradations == 0 and agents["a"].loads == 0


def test_small_dips_within_threshold_do_not_count(tmp_path):
    script = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0] + [9.8] * 8      # 2% below best, threshold 5%
    agents, result = _run({"a": script}, tmp_path, rollback_on_degradation=True)
    assert result.seats["a"].degradations == 0


def test_off_policy_agents_are_reported_but_never_rolled_back(tmp_path):
    events = []

    class CB:
        def on_degradation(self, agent_id, window_id, score, rolled_back):
            events.append(rolled_back)

    agents, result = _run({"a": COLLAPSE}, tmp_path, agents={"a": OffPolicyAgent()},
                          rollback_on_degradation=True, callbacks=[CB()])
    assert result.seats["a"].degradations >= 1
    assert result.seats["a"].rollbacks == 0 and agents["a"].loads == 0
    assert events and not any(events)


def test_degradation_is_not_counted_as_convergence(tmp_path):
    agents, result = _run({"a": COLLAPSE + [1.0] * 40}, tmp_path, rollback_on_degradation=True,
                          plateau_threshold=2, dormant_threshold=4,
                          min_completed_episodes_before_converged=0, stop_on_convergence=True)
    # every third collapsed window is a confirmed degradation that resets the
    # seat to ACTIVE, so a collapsed seat keeps being rolled back instead of
    # being declared converged at its collapsed level
    assert result.stop_reason != "converged"
    assert result.seats["a"].rollbacks >= 5


def test_requires_checkpoint_dir():
    with pytest.raises(ValueError, match="checkpoint_dir"):
        MultiAgentRunner(Pipeline({"a": [1.0]}), {"a": Agent()}, rollback_on_degradation=True)


def test_with_self_play_only_windows_against_frozen_opponents_count(tmp_path):
    # "a" collapses; "b" is flat. With self-play on, a's drop only counts in
    # windows where b was played by a pool snapshot.
    scripts = {"a": COLLAPSE + [1.0] * 60, "b": [5.0] * 72}

    # opponent_prob=0 -> b is never frozen -> a's collapse must never be confirmed
    agents, result = _run(scripts, tmp_path / "never", rollback_on_degradation=True,
                          self_play=SelfPlayConfig(snapshot_every=2, opponent_prob=0.0, seed=0))
    assert result.seats["a"].degradations == 0

    # opponent_prob=0.5 -> some windows have b frozen -> the collapse is confirmed
    agents, result = _run(scripts, tmp_path / "some", rollback_on_degradation=True,
                          self_play=SelfPlayConfig(snapshot_every=2, opponent_prob=0.5, seed=0))
    assert result.seats["a"].rollbacks >= 1

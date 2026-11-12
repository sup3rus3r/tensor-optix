"""
Per-seat hyperparameter tuning and the frozen-opponent self-play pool
(MultiAgentRunner) plus the single-learner SelfPlayPipeline. Scripted
agents/pipelines — no PettingZoo needed.
"""
import json
import os

import numpy as np
import pytest

from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.multi_agent import (
    MultiAgentRunner,
    OpponentPool,
    SelfPlayConfig,
    SelfPlayPipeline,
)


class Agent:
    """'Weights' = number of learn() calls. Counts act() calls too."""

    default_param_bounds = {"learning_rate": (1e-4, 1e-2)}
    default_log_params = ["learning_rate"]

    def __init__(self, name="agent"):
        self.name = name
        self.w = 0
        self.acts = 0
        self.cache = []
        self.hp = HyperparamSet(params={"learning_rate": 1e-3, "flag": True}, episode_id=0)

    def act(self, obs):
        self.acts += 1
        self.cache.append(obs)
        return 0

    def learn(self, episode_data):
        self.w += 1
        self.cache.clear()
        return {}

    def reset_cache(self):
        self.cache.clear()

    def get_hyperparams(self):
        return self.hp.copy()

    def set_hyperparams(self, hp):
        self.hp = hp.copy()

    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "w.json"), "w") as f:
            json.dump({"w": self.w}, f)

    def load_weights(self, path):
        with open(os.path.join(path, "w.json")) as f:
            self.w = json.load(f)["w"]


class NoBoundsAgent(Agent):
    default_param_bounds = {}


class Pipeline:
    """Two-seat pipeline: every window each seat's CURRENT acting agent takes
    one step and finishes one episode with return `score`."""

    def __init__(self, seats=("a", "b"), n_windows=50, score=1.0):
        self._seats = list(seats)
        self._n = n_windows
        self._score = score
        self._agents = {}
        self._env = type("Env", (), {"possible_agents": list(seats)})()
        self._window_size = 1
        self.acting_log = []      # per window: {seat: acting agent object}

    def set_agents(self, agents):
        self._agents = agents

    def setup(self):
        pass

    def teardown(self):
        pass

    def episodes(self):
        for i in range(self._n):
            window, acting = {}, {}
            for seat in self._seats:
                agent = self._agents[seat]
                agent.act(np.zeros(1))
                acting[seat] = agent
                window[seat] = EpisodeData(
                    observations=np.zeros((1, 1)), actions=np.zeros(1), rewards=[self._score],
                    terminated=[True], truncated=[False], infos=[{}], episode_id=i,
                    metadata={"episode_returns": [self._score]},
                )
            self.acting_log.append(acting)
            yield window


class CountingOptimizer(BaseOptimizer):
    def __init__(self):
        self.calls = 0
        self.improvements = 0
        self.plateaus = 0

    def suggest(self, current_hyperparams, metrics_history):
        self.calls += 1
        hp = current_hyperparams.copy()
        hp.params["learning_rate"] = hp.params["learning_rate"] * 0.5
        return hp

    def on_improvement(self, metrics):
        self.improvements += 1

    def on_plateau(self, metrics_history):
        self.plateaus += 1


def _run(agents, pipeline, **kw):
    pipeline.set_agents(dict(agents))
    runner = MultiAgentRunner(pipeline, agents, **kw)
    return runner, runner.run()


# ----------------------------------------------------------------------------
# Per-seat hyperparameter tuning
# ----------------------------------------------------------------------------

def test_auto_gives_each_seat_its_own_spsa_optimizer():
    agents = {"a": Agent(), "b": Agent()}
    runner, _ = _run(agents, Pipeline(n_windows=3), stop_on_convergence=False)
    opt_a, opt_b = runner._seats["a"].optimizer, runner._seats["b"].optimizer
    assert type(opt_a).__name__ == "SPSAOptimizer"
    assert opt_a is not opt_b


def test_auto_skips_agents_without_bounds_and_none_disables():
    agents = {"a": NoBoundsAgent()}
    runner, _ = _run(agents, Pipeline(seats=("a",), n_windows=3), stop_on_convergence=False)
    assert runner._seats["a"].optimizer is None

    agents = {"a": Agent()}
    runner, result = _run(agents, Pipeline(seats=("a",), n_windows=5), optimizer=None,
                          stop_on_convergence=False)
    assert runner._seats["a"].optimizer is None
    assert result.seats["a"].hyperparams["learning_rate"] == 1e-3


def test_optimizer_steps_once_per_scored_window_and_updates_the_agent():
    opts = {"a": CountingOptimizer(), "b": CountingOptimizer()}
    agents = {"a": Agent(), "b": Agent()}
    updates = []

    class CB:
        def on_hyperparam_update(self, agent_id, old, new):
            updates.append((agent_id, old["learning_rate"], new["learning_rate"]))

    _, result = _run(agents, Pipeline(n_windows=4), optimizer=opts, callbacks=[CB()],
                     stop_on_convergence=False)
    assert opts["a"].calls == 4 and opts["b"].calls == 4
    assert agents["a"].hp.params["learning_rate"] == pytest.approx(1e-3 * 0.5 ** 4)
    assert result.seats["a"].hyperparams["learning_rate"] == pytest.approx(1e-3 * 0.5 ** 4)
    assert len(updates) == 8 and updates[0] == ("a", 1e-3, 5e-4)
    assert agents["a"].hp.params["flag"] is True       # non-tuned params untouched


def test_optimizer_factory_and_shared_instance_rejected():
    built = []

    def factory(agent):
        built.append(agent)
        return CountingOptimizer()

    agents = {"a": Agent(), "b": Agent()}
    runner, _ = _run(agents, Pipeline(n_windows=2), optimizer=factory, stop_on_convergence=False)
    assert len(built) == 2
    assert runner._seats["a"].optimizer is not runner._seats["b"].optimizer

    with pytest.raises(ValueError, match="cannot be shared"):
        MultiAgentRunner(Pipeline(), {"a": Agent(), "b": Agent()}, optimizer=CountingOptimizer())
    with pytest.raises(ValueError):
        MultiAgentRunner(Pipeline(), {"a": Agent()}, optimizer="spsa")


def test_optimizer_is_told_about_improvement_and_plateau():
    opt = CountingOptimizer()
    agents = {"a": Agent()}
    _run(agents, Pipeline(seats=("a",), n_windows=200), optimizer={"a": opt},
         plateau_threshold=2, dormant_threshold=4, min_completed_episodes_before_converged=0)
    assert opt.improvements >= 1      # the first scored window
    assert opt.plateaus == 1          # once, when the seat converged


def test_real_spsa_changes_only_bounded_params_and_stays_in_bounds():
    agents = {"a": Agent()}
    _, result = _run(agents, Pipeline(seats=("a",), n_windows=30), stop_on_convergence=False)
    lr = result.seats["a"].hyperparams["learning_rate"]
    assert 1e-4 <= lr <= 1e-2
    assert result.seats["a"].hyperparams["flag"] is True


# ----------------------------------------------------------------------------
# OpponentPool
# ----------------------------------------------------------------------------

def test_pool_is_fifo_capped_and_cleans_up(tmp_path):
    pool = OpponentPool(pool_size=3, pool_dir=str(tmp_path / "pool"), seed=0)
    agent = Agent()
    paths = []
    for i in range(5):
        agent.w = i
        paths.append(pool.add("a", agent))
    assert pool.size("a") == 3
    assert not os.path.exists(paths[0]) and not os.path.exists(paths[1])
    assert all(os.path.exists(p) for p in paths[2:])

    stand_in = Agent()
    seen = set()
    for _ in range(50):
        assert pool.load_into("a", stand_in)
        seen.add(stand_in.w)
    assert seen == {2, 3, 4}
    assert pool.sample("missing") is None and not pool.load_into("missing", stand_in)


def test_pool_owned_temp_dir_is_removed():
    pool = OpponentPool(pool_size=2)
    pool.add("a", Agent())
    root = pool._dir
    assert os.path.isdir(root)
    pool.cleanup()
    assert not os.path.exists(root)


def test_load_into_syncs_hyperparams_from_live_agent(tmp_path):
    pool = OpponentPool(pool_size=2, pool_dir=str(tmp_path))
    live, stand_in = Agent(), Agent()
    pool.add("a", live)
    live.hp.params["learning_rate"] = 0.5
    pool.load_into("a", stand_in, live)
    assert stand_in.hp.params["learning_rate"] == 0.5


# ----------------------------------------------------------------------------
# MultiAgentRunner self-play
# ----------------------------------------------------------------------------

def _self_play_run(n_windows=60, **cfg):
    agents = {"a": Agent("a"), "b": Agent("b")}
    pipeline = Pipeline(n_windows=n_windows)
    cfg.setdefault("seed", 0)
    cfg.setdefault("snapshot_every", 5)
    runner, result = _run(agents, pipeline, self_play=SelfPlayConfig(**cfg),
                          optimizer=None, stop_on_convergence=False)
    return agents, pipeline, runner, result


def test_frozen_seats_do_not_learn_and_are_played_by_a_stand_in():
    agents, pipeline, runner, result = _self_play_run()
    for seat in ("a", "b"):
        frozen = result.seats[seat].frozen_windows
        assert frozen > 0
        # every window the seat was live, its own agent acted and learned
        live_windows = sum(1 for acting in pipeline.acting_log if acting[seat] is agents[seat])
        assert live_windows == 60 - frozen
        assert agents[seat].acts == live_windows
        assert result.seats[seat].windows_scored == live_windows
    # a frozen seat is never the live agent object
    assert any(acting["a"] is not agents["a"] for acting in pipeline.acting_log)


def test_at_least_one_seat_is_always_live():
    agents, pipeline, _, _ = _self_play_run(n_windows=200, opponent_prob=1.0)
    for acting in pipeline.acting_log:
        assert any(acting[s] is agents[s] for s in ("a", "b"))


def test_nobody_is_frozen_before_the_first_snapshot():
    agents, pipeline, _, _ = _self_play_run(n_windows=40, snapshot_every=10, opponent_prob=1.0)
    for acting in pipeline.acting_log[:10]:
        assert acting["a"] is agents["a"] and acting["b"] is agents["b"]
    assert any(acting["a"] is not agents["a"] or acting["b"] is not agents["b"]
               for acting in pipeline.acting_log[10:])


def test_stand_in_plays_past_weights_and_never_accumulates_rollout():
    agents, pipeline, runner, _ = _self_play_run(n_windows=80)
    stand_in = runner._stand_ins["a"]
    assert stand_in is not agents["a"]
    assert stand_in.w <= agents["a"].w        # a past snapshot, never ahead of the live agent
    assert len(stand_in.cache) <= 1           # cleared after every frozen window


def test_pool_size_capped_and_live_agents_restored_to_pipeline():
    agents, pipeline, _, result = _self_play_run(n_windows=100, pool_size=3, snapshot_every=5)
    assert result.pool_sizes == {"a": 3, "b": 3}
    assert pipeline._agents["a"] is agents["a"] and pipeline._agents["b"] is agents["b"]


def test_opponent_prob_zero_means_plain_training():
    agents, pipeline, _, result = _self_play_run(opponent_prob=0.0)
    assert result.seats["a"].frozen_windows == 0 and result.seats["b"].frozen_windows == 0
    assert agents["a"].w == 60


def test_opponent_factory_is_used():
    built = []

    def factory(agent_id):
        built.append(agent_id)
        return Agent(f"standin-{agent_id}")

    agents, _, runner, _ = _self_play_run(opponent_factory=factory)
    assert set(built) == {"a", "b"}
    assert runner._stand_ins["a"].name == "standin-a"


def test_self_play_requires_set_agents():
    class Bare:
        def setup(self): pass
        def teardown(self): pass
        def episodes(self): return iter(())

    with pytest.raises(TypeError, match="set_agents"):
        MultiAgentRunner(Bare(), {"a": Agent()}, self_play=SelfPlayConfig())


def test_config_validation():
    for bad in ({"pool_size": 0}, {"snapshot_every": 0}, {"opponent_prob": 1.5}):
        with pytest.raises(ValueError):
            SelfPlayConfig(**bad)


# ----------------------------------------------------------------------------
# SelfPlayPipeline (single learner)
# ----------------------------------------------------------------------------

def test_single_learner_pipeline_yields_only_the_learner_seat_and_freezes_opponents():
    inner = Pipeline(n_windows=30)
    learner = Agent("learner")
    sp = SelfPlayPipeline(inner, learner_seat="a", agent=learner,
                          config=SelfPlayConfig(snapshot_every=5, pool_size=4, seed=0))
    sp.setup()
    n = 0
    for data in sp.episodes():
        learner.learn(data)
        n += 1
    assert n == 30
    for acting in inner.acting_log:
        assert acting["a"] is learner
        assert acting["b"] is not learner
    opponent = sp._stand_ins["b"]
    assert opponent.w <= learner.w
    assert len(opponent.cache) <= 1
    assert sp.pool.size("a") == 4        # initial + 6 snapshots, capped at 4
    sp.teardown()
    assert sp.pool is None


def test_single_learner_pipeline_works_under_rloptimizer(tmp_path):
    from tensor_optix.optimizer import RLOptimizer

    class BaseAgentLearner(Agent):
        is_on_policy = True

    inner = Pipeline(n_windows=500)
    learner = BaseAgentLearner("learner")
    sp = SelfPlayPipeline(inner, learner_seat="a", config=SelfPlayConfig(snapshot_every=3, seed=0))
    opt = RLOptimizer(agent=learner, pipeline=sp, max_episodes=12,
                      checkpoint_dir=str(tmp_path / "ckpt"))
    opt.run()
    assert opt.best_snapshot is not None
    assert all(acting["a"] is learner and acting["b"] is not learner for acting in inner.acting_log)


def test_single_learner_pipeline_rejects_unknown_seat_and_bad_pipeline():
    sp = SelfPlayPipeline(Pipeline(), learner_seat="zzz", agent=Agent())
    sp.setup()
    with pytest.raises(ValueError, match="learner_seat"):
        next(sp.episodes())

    class Bare:
        pass

    with pytest.raises(TypeError):
        SelfPlayPipeline(Bare(), learner_seat="a")

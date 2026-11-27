"""RL discovery: probes, wrappers, trials, recommendation, build, guardrails."""
import os
import sys

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.masked_cartpole import MaskedCartPoleEnv  # noqa: E402

from tensor_optix.discovery import Budget, DiscoveryReport, discover  # noqa: E402
from tensor_optix.discovery.findings import Severity  # noqa: E402
from tensor_optix.discovery.rl import env as E  # noqa: E402
from tensor_optix.discovery.rl.probes import history_gain, probe_env, random_rollout  # noqa: E402
from tensor_optix.discovery.rl.trials import Variant, run_trial  # noqa: E402

TINY = Budget(time_s=300, env_steps=14_000)


def _ids(fs):
    return {f.id for f in fs}


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def test_cartpole_probe_dense_deterministic_markov():
    f = lambda: gym.make("CartPole-v1")              # noqa: E731
    sec, fs, facts = probe_env(f, f(), steps=5000, seed=0)
    ids = _ids(fs)
    assert "env.spaces" in ids and "env.determinism" in ids
    assert "env.sparse_reward" not in ids             # +1 per step with varying lengths is dense
    assert "env.partial_observability" not in ids
    assert facts["supported"] and facts["flat_obs"]
    assert 10 < sec["rollout"]["random_return_mean"] < 40


def test_masked_cartpole_is_partially_observable():
    sec, fs, facts = probe_env(MaskedCartPoleEnv, MaskedCartPoleEnv(), steps=8000, seed=0)
    f = {x.id: x for x in fs}.get("env.partial_observability")
    assert f is not None and f.severity == Severity.ADVISORY and facts.get("pomdp")
    assert f.evidence[0].interval[0] > 0.10


def test_mountaincar_is_sparse():
    f = lambda: gym.make("MountainCar-v0")           # noqa: E731
    sec, fs, facts = probe_env(f, f(), steps=4000, seed=0)
    assert "env.sparse_reward" in _ids(fs) and facts["sparse"]


def test_pendulum_needs_action_rescale():
    f = lambda: gym.make("Pendulum-v1")              # noqa: E731
    sec, fs, facts = probe_env(f, f(), steps=2000, seed=0)
    r = {x.id: x for x in fs}["env.action_rescale"]
    assert r.default_applied and r.suggested_action.kind == "wrap_action_rescale"
    assert r.suggested_action.params == {"low": [-2.0], "high": [2.0]}


def test_badly_scaled_observations_are_flagged():
    class Scaled(gym.ObservationWrapper):
        def __init__(self):
            super().__init__(gym.make("CartPole-v1"))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

        def observation(self, o):
            return (np.asarray(o) * np.array([1000.0, 1.0, 0.001, 1.0])).astype(np.float32)
    sec, fs, facts = probe_env(Scaled, Scaled(), steps=3000, seed=0)
    assert "env.obs_scale" in _ids(fs) and facts.get("obs_badly_scaled")


def test_history_gain_null_on_markov_random_data():
    f = gym.make("CartPole-v1")
    ro = random_rollout(f, 6000, seed=1)
    hg = history_gain(ro, seed=1)
    assert hg is not None and hg["interval"][0] <= 0.10


# ---------------------------------------------------------------------------
# env handling and wrappers
# ---------------------------------------------------------------------------

def test_env_factory_variants():
    fac, inst, note = E.env_factory(gym.make("CartPole-v1"))
    assert note is None and fac() is not inst
    fac, inst, note = E.env_factory("CartPole-v1")
    assert note is None
    fac, inst, note = E.env_factory(MaskedCartPoleEnv)
    assert note is None and isinstance(inst, MaskedCartPoleEnv)
    e = MaskedCartPoleEnv()
    fac, inst, note = E.env_factory(e)
    assert fac() is e or note is None


def test_wrappers_map_ranges():
    env = E.make_wrapped(lambda: gym.make("Pendulum-v1"),
                         {"action_rescale": {"low": [-2.0], "high": [2.0]},
                          "obs_normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 8.0]}})
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0
    assert np.isclose(env.unwrapped.__class__.__name__ != "", True)
    inner = env.env                                          # ActionRescale under FixedObsNormalize
    assert np.allclose(inner.action(np.array([1.0])), [2.0]) and np.allclose(inner.action(np.array([-1.0])), [-2.0])
    o = E.reset(env, seed=0)
    assert np.all(np.abs(o) <= 10.0)


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------

def test_trial_is_reproducible_and_leaves_global_rng_alone():
    f = lambda: gym.make("CartPole-v1")              # noqa: E731
    np.random.seed(5)
    torch.manual_seed(5)
    s_np = np.random.get_state()[1].copy()
    s_t = torch.random.get_rng_state().clone()
    a = run_trial(f, Variant("PPO"), wrappers={}, seed=3, steps=2048, window=512, eval_episodes=3)
    b = run_trial(f, Variant("PPO"), wrappers={}, seed=3, steps=2048, window=512, eval_episodes=3)
    assert a.status == "ok" and a.eval_returns == b.eval_returns
    assert np.array_equal(np.random.get_state()[1], s_np) and torch.equal(torch.random.get_rng_state(), s_t)


def test_failing_algorithm_is_a_result_not_a_crash():
    f = lambda: gym.make("CartPole-v1")              # noqa: E731
    r = run_trial(f, Variant("NOT_AN_ALGO"), wrappers={}, seed=0, steps=1024, window=512, eval_episodes=1)
    assert r.status == "failed" and r.score is None and r.error


# ---------------------------------------------------------------------------
# end to end (tiny budgets: plumbing, not quality)
# ---------------------------------------------------------------------------

def test_discover_rl_end_to_end_cartpole():
    rep = discover(gym.make("CartPole-v1"), budget=TINY, seed=0)
    assert rep.domain == "rl" and rep.input_summary["metric"] == "return"
    fams = {x.family for x in rep.experiments}
    assert fams >= {"PPO", "DQN"}
    assert all(len(x.scores) >= 3 for x in rep.experiments if x.status == "ok")
    assert rep.recommendation.verdict in ("clear_winner", "tie_simplest_chosen", "undetermined", "abstain")
    assert rep.budget_used["env_steps"] <= TINY.env_steps * 1.05
    again = DiscoveryReport.from_html(rep.to_html())
    assert again.to_dict() == rep.to_dict()
    if rep.recommendation.family:
        sol = rep.build()
        env = sol.make_env(gym.make("CartPole-v1"))
        agent = sol.make_agent(env)
        assert agent.act(E.reset(env, seed=0)) in (0, 1)


def test_discover_rl_restricted_algorithms_and_box_rescale():
    rep = discover("Pendulum-v1", kind="rl", families=["sac"], budget=Budget(time_s=300, env_steps=9_000), seed=0)
    assert {x.family for x in rep.experiments} == {"SAC"}
    assert "env.action_rescale" in rep.profile.sections["applied_findings"]
    assert rep.profile.sections["trials"]["wrappers_in_trials"] == ["action_rescale"]
    with pytest.raises(ValueError, match="applies to this action space"):
        discover("Pendulum-v1", kind="rl", families=["dqn"], budget=TINY)


def test_multi_agent_is_profiled_not_trialled():
    class FakeAEC:
        possible_agents = ["a", "b"]
        agents = ["a", "b"]
        observation_spaces = {}

        def reset(self, seed=None):
            pass

        def step(self, action):
            pass

        def last(self):
            pass

    rep = discover(FakeAEC(), kind="rl", budget=TINY)
    assert rep.recommendation.verdict == "abstain"
    assert "limitation.multi_agent" in _ids(rep.findings)


@pytest.mark.slow
def test_ppo_learns_cartpole_and_recurrent_helps_masked_cartpole():
    rep = discover(gym.make("CartPole-v1"), budget=Budget(time_s=900, env_steps=150_000), seed=0)
    assert rep.recommendation.family is not None and rep.recommendation.expected_score > 100
    rep2 = discover(MaskedCartPoleEnv, kind="rl", families=["ppo"],
                    budget=Budget(time_s=900, env_steps=150_000), seed=0)
    assert "env.partial_observability" in _ids(rep2.findings)
    assert any(x.family == "RECURRENT_PPO" for x in rep2.experiments)

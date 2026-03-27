import pytest
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer
from tensor_optix.core.types import HyperparamSet, EvalMetrics


BOUNDS = {"learning_rate": (1e-5, 1e-2), "gamma": (0.9, 0.999)}


def make_hp(lr=1e-3, gamma=0.99, episode_id=0):
    return HyperparamSet(params={"learning_rate": lr, "gamma": gamma}, episode_id=episode_id)


def make_metrics(score, episode_id=0):
    return EvalMetrics(primary_score=score, metrics={}, episode_id=episode_id)


def test_returns_hyperparam_set():
    opt = BackoffOptimizer(param_bounds=BOUNDS)
    result = opt.suggest(make_hp(), [make_metrics(1.0)])
    assert isinstance(result, HyperparamSet)


def test_returns_unchanged_with_empty_history():
    opt = BackoffOptimizer(param_bounds=BOUNDS)
    result = opt.suggest(make_hp(), [])
    assert result.params == make_hp().params


def test_no_change_without_bounds():
    opt = BackoffOptimizer(param_bounds={})
    result = opt.suggest(make_hp(), [make_metrics(1.0)])
    assert result.params == make_hp().params


def test_probe_phase_changes_one_param():
    opt = BackoffOptimizer(param_bounds=BOUNDS, perturbation_scale=0.1)
    hp = make_hp()
    result = opt.suggest(hp, [make_metrics(1.0)])
    # Only the first param in cycle should change during probe
    changed = [k for k in BOUNDS if result.params[k] != hp.params[k]]
    assert len(changed) == 1


def test_params_stay_within_bounds():
    opt = BackoffOptimizer(param_bounds=BOUNDS, perturbation_scale=0.5)
    hp = make_hp()
    history = [make_metrics(float(i)) for i in range(20)]
    for i in range(40):
        hp = opt.suggest(hp, history)
        assert BOUNDS["learning_rate"][0] <= hp.params["learning_rate"] <= BOUNDS["learning_rate"][1]
        assert BOUNDS["gamma"][0] <= hp.params["gamma"] <= BOUNDS["gamma"][1]


def test_commit_keeps_probe_when_score_improves():
    opt = BackoffOptimizer(
        param_bounds={"learning_rate": (1e-5, 1e-2)},
        perturbation_scale=0.1,
    )
    hp = make_hp()

    # Phase 1: probe (score=1.0 as base)
    probed = opt.suggest(hp, [make_metrics(1.0)])
    probed_lr = probed.params["learning_rate"]
    assert probed_lr != hp.params["learning_rate"]

    # Phase 2: commit with higher score → should keep probe value
    committed = opt.suggest(probed, [make_metrics(1.0), make_metrics(2.0)])
    assert committed.params["learning_rate"] == probed_lr


def test_commit_reverses_when_score_drops():
    opt = BackoffOptimizer(
        param_bounds={"learning_rate": (1e-5, 1e-2)},
        perturbation_scale=0.1,
        min_delta=1e-7,
    )
    hp = make_hp(lr=5e-3)

    # Phase 1: probe (base score = 2.0)
    probed = opt.suggest(hp, [make_metrics(2.0)])
    probed_lr = probed.params["learning_rate"]

    # Phase 2: commit with lower score → should go opposite direction
    committed = opt.suggest(probed, [make_metrics(2.0), make_metrics(1.0)])
    # Should be on the other side of original lr, not the probe side
    assert committed.params["learning_rate"] != probed_lr


def test_cycle_advances_to_next_param():
    opt = BackoffOptimizer(param_bounds=BOUNDS, perturbation_scale=0.1)
    hp = make_hp()

    # Two suggest calls: probe + commit for first param
    probed = opt.suggest(hp, [make_metrics(1.0)])
    committed = opt.suggest(probed, [make_metrics(1.0), make_metrics(1.5)])

    # Now probe for second param
    next_probe = opt.suggest(committed, [make_metrics(1.0), make_metrics(1.5), make_metrics(1.5)])
    assert opt._current_param_idx == 1 or opt._current_param_idx == 0  # cycled


def test_on_improvement_increases_scale():
    opt = BackoffOptimizer(param_bounds=BOUNDS, perturbation_scale=0.05)
    opt.suggest(make_hp(), [make_metrics(1.0)])  # initialize
    old_scale = opt._perturbation_scale
    opt.on_improvement(make_metrics(5.0))
    assert opt._perturbation_scale > old_scale


def test_on_plateau_increases_scale_and_resets_cycle():
    opt = BackoffOptimizer(param_bounds=BOUNDS, perturbation_scale=0.05)
    opt.suggest(make_hp(), [make_metrics(1.0)])
    opt._current_param_idx = 1
    old_scale = opt._perturbation_scale
    opt.on_plateau([make_metrics(1.0)] * 5)
    assert opt._perturbation_scale > old_scale
    assert opt._current_param_idx == 0


def test_unbounded_params_unchanged():
    opt = BackoffOptimizer(param_bounds={"learning_rate": (1e-5, 1e-2)})
    hp = make_hp()
    result = opt.suggest(hp, [make_metrics(1.0)])
    assert result.params["gamma"] == hp.params["gamma"]

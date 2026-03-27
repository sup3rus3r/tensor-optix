import pytest
from tensor_optix.optimizers.pbt_optimizer import PBTOptimizer
from tensor_optix.core.types import HyperparamSet, EvalMetrics


BOUNDS = {"learning_rate": (1e-5, 1e-2), "gamma": (0.9, 0.999)}


def make_hp(lr=1e-3, gamma=0.99, episode_id=0):
    return HyperparamSet(params={"learning_rate": lr, "gamma": gamma}, episode_id=episode_id)


def make_metrics(score, episode_id=0):
    return EvalMetrics(primary_score=score, metrics={}, episode_id=episode_id)


def test_returns_hyperparam_set():
    opt = PBTOptimizer(param_bounds=BOUNDS)
    hp = make_hp()
    history = [make_metrics(1.0)]
    result = opt.suggest(hp, history)
    assert isinstance(result, HyperparamSet)


def test_returns_unchanged_with_empty_history():
    opt = PBTOptimizer(param_bounds=BOUNDS)
    hp = make_hp()
    result = opt.suggest(hp, [])
    assert result.params == hp.params


def test_params_stay_within_bounds():
    opt = PBTOptimizer(param_bounds=BOUNDS, explore_scale=0.3)
    hp = make_hp()
    history = [make_metrics(float(i)) for i in range(20)]
    for _ in range(30):
        hp = opt.suggest(hp, history)
        assert BOUNDS["learning_rate"][0] <= hp.params["learning_rate"] <= BOUNDS["learning_rate"][1]
        assert BOUNDS["gamma"][0] <= hp.params["gamma"] <= BOUNDS["gamma"][1]


def test_exploit_copies_from_top_performer():
    opt = PBTOptimizer(
        param_bounds=BOUNDS,
        history_size=20,
        exploit_percentile=0.2,
        top_percentile=0.2,
        exploit_scale=0.0,  # zero perturbation to verify exact copy
    )
    # Build history: best params have lr=9e-3
    best_hp = HyperparamSet(params={"learning_rate": 9e-3, "gamma": 0.99}, episode_id=0)
    for i in range(15):
        opt._history.append((make_hp(lr=1e-4), float(i)))  # poor performers
    opt._history.append((best_hp, 1000.0))  # single top performer

    # Current agent has very low score (bottom 20%)
    current_hp = make_hp(lr=1e-4)
    history = [make_metrics(0.1)]
    result = opt.suggest(current_hp, history)
    # With exploit_scale=0.0, should copy best_hp lr exactly
    assert abs(result.params["learning_rate"] - 9e-3) < 1e-8


def test_plateau_clears_history():
    opt = PBTOptimizer(param_bounds=BOUNDS)
    hp = make_hp()
    history = [make_metrics(float(i)) for i in range(10)]
    opt.suggest(hp, history)
    assert len(opt._history) > 0
    opt.on_plateau(history)
    assert len(opt._history) == 0


def test_unbounded_params_unchanged():
    opt = PBTOptimizer(param_bounds={"learning_rate": (1e-5, 1e-2)})
    hp = make_hp()
    history = [make_metrics(1.0)] * 5
    result = opt.suggest(hp, history)
    assert result.params["gamma"] == hp.params["gamma"]


def test_history_respects_maxlen():
    opt = PBTOptimizer(param_bounds=BOUNDS, history_size=5)
    hp = make_hp()
    for i in range(20):
        opt.suggest(hp, [make_metrics(float(i))])
    assert len(opt._history) <= 5

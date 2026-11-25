"""
ConvergenceMonitor: multi-timescale convergence detection against a global
peak. Unit behaviour first, then seeded statistical scenarios (noise sigma = 5
around a level of 100) with deliberately loose bounds.
"""
import math

import numpy as np
import pytest

from tensor_optix.core.convergence import ConvergenceMonitor, ConvergenceRegime as R
from tensor_optix.core.types import LoopState

SIGMA, B = 5.0, 100.0


def _stream(fn, n, seed, ar=0.0):
    rng = np.random.default_rng(seed)
    e, out = 0.0, []
    for i in range(n):
        e = ar * e + math.sqrt(1 - ar * ar) * rng.normal(0, SIGMA)
        out.append(fn(i) + e)
    return out


def _run(scores, **kw):
    m = ConvergenceMonitor(**kw)
    return m, [m.observe(s, at=i) for i, s in enumerate(scores)]


def _first(reports, pred, after=0):
    for r in reports:
        if r.at >= after and pred(r):
            return r.at - after
    return None


# ----------------------------------------------------------------------------
# Construction / input validation
# ----------------------------------------------------------------------------

def test_rejects_bad_timescales_and_patience():
    with pytest.raises(ValueError):
        ConvergenceMonitor(timescales=(2, 8, 16))
    with pytest.raises(ValueError):
        ConvergenceMonitor(timescales=(8, 4, 16))
    with pytest.raises(ValueError):
        ConvergenceMonitor(cooling_patience=10, dormant_patience=5)


def test_rejects_non_finite_scores_and_decreasing_positions():
    m = ConvergenceMonitor()
    with pytest.raises(ValueError):
        m.observe(float("nan"), at=0)
    m.observe(1.0, at=5)
    with pytest.raises(ValueError):
        m.observe(1.0, at=4)


def test_first_observation_is_the_baseline():
    m = ConvergenceMonitor()
    r = m.observe(42.0, at=0)
    assert r.regime == R.IMPROVING and r.improved and r.peak_raised
    assert r.peak == 42.0 and r.best_score == 42.0 and r.state == LoopState.ACTIVE


def test_default_position_is_the_observation_count():
    m = ConvergenceMonitor()
    assert [m.observe(1.0).at for _ in range(3)] == [0.0, 1.0, 2.0]


# ----------------------------------------------------------------------------
# Deterministic behaviour
# ----------------------------------------------------------------------------

def test_deterministic_rise_is_improving_throughout():
    _, reps = _run([B + i for i in range(200)])
    assert all(r.regime == R.IMPROVING for r in reps[5:])
    assert all(r.state == LoopState.ACTIVE for r in reps)


def test_noise_estimate_ignores_curvature():
    # Rise then flat with unit noise: one straight line fits this badly (its
    # residuals look large and autocorrelated), but first differences barely
    # see the bend, so neither noise nor autocorrelation is invented.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        m = ConvergenceMonitor()
        for i in range(60):
            r = m.observe(2.0 * min(i, 20) + rng.normal(0, 1.0), at=i)
        assert 0.6 < r.sigma < 1.6
        assert r.rho < 0.4


def test_flat_series_cools_then_goes_dormant_exactly_once():
    _, reps = _run([7.0] * 100, cooling_patience=10, dormant_patience=30)
    assert sum(r.entered_cooling for r in reps) == 1
    assert sum(r.entered_dormant for r in reps) == 1
    first_dormant = _first(reps, lambda r: r.entered_dormant)
    # the patience clock starts once a noise estimate exists (5 observations)
    assert 30 <= first_dormant <= 40
    assert reps[-1].state == LoopState.DORMANT


def test_min_at_before_dormant_is_respected():
    _, reps = _run([7.0] * 200, cooling_patience=5, dormant_patience=10, min_at_before_dormant=150)
    assert _first(reps, lambda r: r.entered_dormant) >= 150


def test_defer_dormant_lets_dormant_be_entered_again():
    m = ConvergenceMonitor(cooling_patience=2, dormant_patience=4)
    entered = []
    for i in range(30):
        r = m.observe(1.0, at=i)
        if r.entered_dormant:
            entered.append(i)
            m.defer_dormant()
    assert len(entered) >= 2 and m.state == LoopState.COOLING


def test_reset_scale_forgets_the_peak():
    m = ConvergenceMonitor()
    for i in range(50):
        m.observe(100.0 + i, at=i)
    m.reset_scale()
    r = m.observe(-5.0, at=50)
    assert r.peak == -5.0 and r.best_score == -5.0 and m.n_observations == 1


def test_restart_keeps_the_peak_and_returning_to_it_is_not_improvement():
    m = ConvergenceMonitor(timescales=(4, 8, 16))
    for i in range(40):
        m.observe(10.0 + (i % 2) * 0.1, at=i)
    peak = m.peak
    m.restart(at=40)
    assert m.state == LoopState.ACTIVE and m.peak == peak
    reps = [m.observe(10.0 + (i % 2) * 0.1, at=40 + i) for i in range(20)]
    assert not any(r.regime == R.IMPROVING for r in reps)
    # genuinely higher is new ground
    reps = [m.observe(30.0 + i, at=60 + i) for i in range(10)]
    assert any(r.regime == R.IMPROVING for r in reps)


def test_restored_run_is_not_declared_dormant_while_unreadable():
    m = ConvergenceMonitor(timescales=(8, 16, 32), cooling_patience=2, dormant_patience=4)
    for i in range(20):
        m.observe(10.0 + (i % 3) * 0.1, at=i)
    m.note_restored(at=20)
    reps = [m.observe(10.0, at=21 + i) for i in range(5)]
    assert all(r.state != LoopState.DORMANT for r in reps)


def test_probing_observations_never_confirm_a_collapse():
    scores = [10.0 + 0.1 * (i % 2) for i in range(60)] + [0.0] * 20
    m = ConvergenceMonitor()
    reps = [m.observe(s, at=i, probing=i >= 60) for i, s in enumerate(scores)]
    assert not any(r.collapse_confirmed for r in reps)


# ----------------------------------------------------------------------------
# Seeded statistical scenarios
# ----------------------------------------------------------------------------

SEEDS = range(8)


def test_plateau_rarely_claims_improvement_or_collapse():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B, 1500, seed))
        late = reps[200:]
        assert sum(r.regime == R.IMPROVING for r in late) <= 25
        assert sum(r.regime == R.COLLAPSING for r in late) <= 10


def test_plateau_reaches_dormant_within_patience_bounds():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B, 1500, seed))
        d = _first(reps, lambda r: r.entered_dormant)
        assert d is not None and 160 <= d <= 600


def test_dip_and_recovery_to_the_same_level_is_never_improvement():
    def dip(i):
        if 400 <= i < 420:
            return B - 30
        if 420 <= i < 440:
            return B - 30 + 1.5 * (i - 420)
        return B

    for seed in SEEDS:
        _, reps = _run(_stream(dip, 800, seed))
        # the climb back (and the windows that still contain it) is never new
        # ground; afterwards the plateau's ordinary false-alarm rate applies
        assert not any(r.regime == R.IMPROVING for r in reps[400:460])
        onsets = sum(r.regime == R.IMPROVING and r.regime_changed for r in reps[400:560])
        assert onsets <= 2
        regressed_at = _first(reps, lambda r: r.is_regressed or r.regime == R.COLLAPSING, 400)
        assert regressed_at is not None and regressed_at <= 15


def test_drop_that_stays_is_reported_as_stalled_below_peak():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B if i < 400 else B - 15, 700, seed))
        d = _first(reps, lambda r: r.regime == R.REGRESSED_STALLED, 400)
        assert d is not None and d <= 45
        r = next(r for r in reps[400:] if r.regime == R.REGRESSED_STALLED)
        assert r.gap_to_peak > 1.0 and r.eta_to_peak == math.inf and r.below_peak_for >= 0


def test_climbing_back_is_reported_with_an_eta():
    def rise_then_dip(i):
        if i < 300:
            return B + 0.2 * i
        if i < 320:
            return B + 35
        return B + 35 + min(25, 0.1 * (i - 320))

    climbing = 0
    for seed in SEEDS:
        _, reps = _run(_stream(rise_then_dip, 600, seed))
        hits = [r for r in reps[320:570] if r.regime == R.REGRESSED_CLIMBING]
        climbing += len(hits)
        assert all(r.eta_to_peak is not None and r.eta_to_peak < math.inf for r in hits)
    assert climbing > 0


def test_collapse_is_confirmed_fast_and_once():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B if i < 600 else B - 30, 900, seed))
        d = _first(reps, lambda r: r.collapse_confirmed, 600)
        assert d is not None and d <= 12
        assert sum(r.collapse_confirmed for r in reps[600:700]) == 1


@pytest.mark.parametrize("rate, bound", [(0.05, 200), (0.2, 70), (0.5, 40)])
def test_resumed_learning_is_detected(rate, bound):
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B + max(0, i - 600) * rate, 1200, seed))
        d = _first(reps, lambda r: r.regime == R.IMPROVING, 600)
        assert d is not None and d <= bound


def test_slow_steady_learner_is_never_declared_dormant():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B + 0.05 * i, 2000, seed))
        assert not any(r.state == LoopState.DORMANT for r in reps)


def test_autocorrelated_noise_is_detected_and_does_not_fake_improvement():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B, 1500, seed, ar=0.6))
        assert reps[-1].rho > 0.3
        assert sum(r.regime == R.IMPROVING for r in reps[200:]) <= 25


def test_warmup_fast_rise_is_improving_and_never_regressed():
    for seed in SEEDS:
        _, reps = _run(_stream(lambda i: B + 2.0 * i if i < 25 else B + 50, 120, seed))
        assert sum(r.regime == R.IMPROVING for r in reps[:32]) >= 3
        assert not any(r.is_regressed or r.regime == R.COLLAPSING for r in reps)

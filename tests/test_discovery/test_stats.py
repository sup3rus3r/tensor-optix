"""
Statistics used by discovery: validated against tabulated reference values and
exact identities (scipy is deliberately not a dependency).
"""
import math

import numpy as np
import pytest

from tensor_optix.discovery.stats import (
    autocorrelation, betainc, block_permutation, bootstrap_interval, compare_paired,
    corrected_resampled_t, default_rope, holm, kolmogorov_sf, ks_2samp, norm_cdf, norm_ppf,
    paired_deltas, permutation_test, psi, rankdata, rope_verdict, score, t_cdf, t_interval,
    t_ppf, t_sf,
)
from tensor_optix.discovery.stats import metrics as M
from tensor_optix.nanonets.smalln import _T_CRITICAL_975

# Two-sided critical values from standard t tables (df -> t_{0.995}).
T_995 = {1: 63.657, 2: 9.925, 5: 4.032, 10: 3.169, 20: 2.845, 30: 2.750}


# ---------------------------------------------------------------- distributions

@pytest.mark.parametrize("df,crit", sorted(_T_CRITICAL_975.items()))
def test_t_ppf_matches_the_tabulated_975_values(df, crit):
    assert t_ppf(0.975, df) == pytest.approx(crit, abs=2e-4)
    assert t_cdf(crit, df) == pytest.approx(0.975, abs=2e-5)


@pytest.mark.parametrize("df,crit", sorted(T_995.items()))
def test_t_ppf_matches_the_tabulated_995_values(df, crit):
    assert t_ppf(0.995, df) == pytest.approx(crit, abs=2e-3)


def test_t_distribution_identities():
    for df in (1, 2, 3, 7.5, 30, 200):
        assert t_cdf(0.0, df) == 0.5
        for t in (0.3, 1.0, 2.5, 7.0):
            assert t_cdf(t, df) + t_cdf(-t, df) == pytest.approx(1.0, abs=1e-12)
            assert t_sf(t, df) == pytest.approx(1.0 - t_cdf(t, df), abs=1e-12)
    # df=1 is the Cauchy distribution: CDF = 1/2 + atan(t)/pi
    for t in (-3.0, -0.5, 0.7, 4.0):
        assert t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-12)
    # df=2 has a closed form: CDF = 1/2 + t / (2 sqrt(2 + t^2))
    for t in (-2.0, 0.4, 3.3):
        assert t_cdf(t, 2) == pytest.approx(0.5 + t / (2 * math.sqrt(2 + t * t)), abs=1e-12)
    # large df approaches the normal
    assert t_cdf(1.96, 1e6) == pytest.approx(norm_cdf(1.96), abs=1e-6)


def test_t_ppf_inverts_t_cdf():
    for df in (1, 4, 12, 50):
        for p in (0.01, 0.2, 0.5, 0.8, 0.999):
            assert t_cdf(t_ppf(p, df), df) == pytest.approx(p, abs=1e-10)


def test_betainc_known_values():
    assert betainc(1.0, 1.0, 0.37) == pytest.approx(0.37, abs=1e-14)          # uniform
    assert betainc(2.0, 1.0, 0.5) == pytest.approx(0.25, abs=1e-14)           # x^a
    assert betainc(1.0, 3.0, 0.2) == pytest.approx(1 - 0.8 ** 3, abs=1e-14)   # 1-(1-x)^b
    for a, b, x in ((2.5, 0.7, 0.3), (0.5, 0.5, 0.9)):
        assert betainc(a, b, x) == pytest.approx(1.0 - betainc(b, a, 1.0 - x), abs=1e-12)


def test_normal_functions():
    assert norm_cdf(0.0) == 0.5
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert norm_ppf(norm_cdf(-1.234)) == pytest.approx(-1.234, abs=1e-10)


def test_kolmogorov_sf_known_values():
    # Tabulated Kolmogorov critical values: P(K > 1.3581) = 0.05, P(K > 1.6276) = 0.01
    assert kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=5e-4)
    assert kolmogorov_sf(1.6276) == pytest.approx(0.01, abs=2e-4)
    assert kolmogorov_sf(0.0) == 1.0


# ---------------------------------------------------------------- intervals

def test_t_interval_matches_hand_computation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    iv = t_interval(x)
    se = np.std(x, ddof=1) / math.sqrt(5)
    assert iv.mean == 3.0
    assert iv.lo == pytest.approx(3.0 - _T_CRITICAL_975[4] * se, abs=1e-3)
    assert iv.hi == pytest.approx(3.0 + _T_CRITICAL_975[4] * se, abs=1e-3)


def test_corrected_t_is_wider_than_naive_and_matches_formula():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, size=10)
    naive = t_interval(d)
    corr = corrected_resampled_t(d, n_train=800, n_test=200)
    assert corr.hi - corr.lo > naive.hi - naive.lo
    expected_se = math.sqrt((1 / 10 + 200 / 800) * np.var(d, ddof=1))
    assert corr.se == pytest.approx(expected_se, rel=1e-12)


def test_corrected_t_type_one_error_is_near_nominal_under_the_null():
    """With independent deltas (no overlap) and n_test/n_train -> 0 the correction
    vanishes, so the interval must cover 0 about 95% of the time."""
    rng = np.random.default_rng(1)
    misses = 0
    trials = 2000
    for _ in range(trials):
        iv = corrected_resampled_t(rng.normal(0, 1, size=8), n_train=1e9, n_test=1.0)
        misses += not (iv.lo <= 0.0 <= iv.hi)
    rate = misses / trials
    assert 0.03 < rate < 0.07


def test_bootstrap_variants_cover_the_mean():
    rng = np.random.default_rng(2)
    vals = rng.normal(5.0, 1.0, size=300)
    for kw in ({}, {"block": 10}, {"groups": np.repeat(np.arange(30), 10)}):
        iv = bootstrap_interval(vals, np.random.default_rng(3), n_boot=500, **kw)
        assert iv.lo < 5.0 < iv.hi or abs(iv.mean - 5.0) < 0.3


def test_degenerate_inputs():
    with pytest.raises(ValueError):
        t_interval([])
    one = t_interval([3.0])
    assert one.mean == 3.0 and math.isinf(one.lo)
    const = corrected_resampled_t([0.2, 0.2, 0.2], 100, 20)
    assert const.se == 0.0 and const.p_two_sided == 0.0


# ---------------------------------------------------------------- comparison

def test_rope_verdicts():
    assert rope_verdict(0.02, 0.05, 0.01) == "better"
    assert rope_verdict(-0.05, -0.02, 0.01) == "worse"
    assert rope_verdict(-0.004, 0.006, 0.01) == "equivalent"
    assert rope_verdict(-0.02, 0.03, 0.01) == "undetermined"
    assert rope_verdict(-math.inf, math.inf, 0.01) == "undetermined"


def test_paired_deltas_orientation_and_missing_pairs():
    d = paired_deltas([0.9, None, 0.8], [0.8, 0.7, 0.85], higher_is_better=True)
    assert list(d) == pytest.approx([0.1, -0.05])
    d = paired_deltas([0.3, 0.2], [0.4, 0.1], higher_is_better=False)    # losses
    assert list(d) == pytest.approx([0.1, -0.1])
    with pytest.raises(ValueError):
        paired_deltas([1.0], [1.0, 2.0])


def test_compare_paired_identical_families_are_never_better():
    rng = np.random.default_rng(4)
    better = 0
    for _ in range(300):
        base = rng.normal(0.8, 0.02, size=10)
        a = base + rng.normal(0, 0.005, size=10)
        b = base + rng.normal(0, 0.005, size=10)
        verdict, _ = compare_paired(a, b, rope=0.005, n_train=800, n_test=200)
        better += verdict == "better"
    assert better / 300 <= 0.05


def test_compare_paired_detects_a_real_difference():
    rng = np.random.default_rng(5)
    base = rng.normal(0.8, 0.02, size=10)
    verdict, iv = compare_paired(base + 0.05, base + rng.normal(0, 0.002, 10),
                                 rope=0.01, n_train=800, n_test=200)
    assert verdict == "better" and iv.lo > 0.01


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, None])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max -> 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert adj[3] is None


def test_default_rope_values():
    assert default_rope("auc") == 0.005
    assert default_rope("rmse", target_std=10.0) == pytest.approx(0.1)
    assert default_rope("log_loss", scale=0.69) == pytest.approx(0.0069)


# ---------------------------------------------------------------- hypothesis tests

def test_permutation_test_is_calibrated_and_powerful():
    rng = np.random.default_rng(6)
    x = rng.normal(size=200)

    def corr(lbl):
        return float(np.corrcoef(x, lbl)[0, 1])

    null_p = [permutation_test(corr, rng.normal(size=200), np.random.default_rng(s), n_perm=99).p_value
              for s in range(200)]
    assert 0.02 < np.mean(np.asarray(null_p) < 0.05) < 0.09
    strong = permutation_test(corr, x + rng.normal(scale=0.5, size=200), np.random.default_rng(7), n_perm=199)
    assert strong.p_value == pytest.approx(1 / 200)


def test_block_permutation_preserves_values():
    x = np.arange(23)
    y = block_permutation(5)(x, np.random.default_rng(0))
    assert sorted(y) == list(x) and len(y) == 23


def test_ks_2samp():
    rng = np.random.default_rng(8)
    same = ks_2samp(rng.normal(size=500), rng.normal(size=400))
    diff = ks_2samp(rng.normal(size=500), rng.normal(0.5, 1, size=400))
    assert same.p_value > 0.01 and diff.p_value < 1e-6
    a = np.array([0, 0, 1, 1, 2.0])
    assert ks_2samp(a, a).statistic == 0.0


def test_autocorrelation_and_psi():
    rng = np.random.default_rng(9)
    e = rng.normal(size=5000)
    ar = np.zeros(5000)
    for i in range(1, 5000):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    assert autocorrelation(ar, 2)[0] == pytest.approx(0.7, abs=0.05)
    assert abs(autocorrelation(e, 1)[0]) < 0.05
    assert psi(e, e) < 0.01
    assert psi(e, e + 1.0) > 0.25


# ---------------------------------------------------------------- metrics

def test_rankdata_ties():
    assert list(rankdata(np.array([10, 20, 20, 5]))) == [2.0, 3.5, 3.5, 1.0]


def test_auc():
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc_binary(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert M.roc_auc_binary(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert math.isnan(M.roc_auc_binary(np.array([1, 1]), np.array([0.2, 0.3])))
    # AUC is invariant to monotone transforms of the score
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert M.roc_auc_binary(y, np.log(s)) == M.roc_auc_binary(y, s)


def test_log_loss_and_brier():
    y = np.array([0, 1])
    p = np.array([0.2, 0.9])
    expected = -(math.log(0.8) + math.log(0.9)) / 2
    assert M.log_loss(y, p, 2) == pytest.approx(expected)
    assert M.brier(y, p, 2) == pytest.approx(((0.2 ** 2) * 2 + (0.1 ** 2) * 2) / 2)
    assert M.log_loss(np.array([0]), np.array([1.0]), 2) < 40   # clipped, finite


def test_label_and_value_metrics():
    y = np.array([0, 0, 0, 1])
    yhat = np.array([0, 0, 0, 0])
    assert M.accuracy(y, yhat) == 0.75
    assert M.balanced_accuracy(y, yhat) == 0.5
    assert M.macro_f1(y, yhat) == pytest.approx((6 / 7 + 0.0) / 2)
    yr = np.array([1.0, 2.0, 3.0])
    assert M.rmse(yr, yr + 1) == pytest.approx(1.0)
    assert M.mae(yr, yr - 2) == pytest.approx(2.0)
    assert M.r2(yr, yr) == 1.0
    assert M.r2(yr, np.full(3, 2.0)) == pytest.approx(0.0)
    assert M.poisson_deviance(np.array([2.0, 0.0]), np.array([2.0, 1e-12])) == pytest.approx(0.0, abs=1e-8)


def test_score_dispatch():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])
    assert score("auc", y, proba=p) == 1.0
    assert score("accuracy", y, proba=p) == 1.0
    with pytest.raises(ValueError):
        score("rmse", y, proba=p)
    with pytest.raises(ValueError):
        score("nope", y, proba=p)

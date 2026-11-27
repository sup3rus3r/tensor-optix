"""Statistics used by tensor_optix.discovery (numpy only, no scipy)."""
from .compare import compare_paired, default_rope, holm, paired_deltas, rope_verdict
from .distributions import betainc, chi2_sf, gammainc_upper, kolmogorov_sf, norm_cdf, norm_ppf, norm_sf, t_cdf, t_ppf, t_sf
from .hypothesis import TestResult, autocorrelation, block_permutation, ks_2samp, permutation_test, psi
from .intervals import Interval, bootstrap_interval, corrected_resampled_t, t_interval
from .metrics import METRICS, Metric, get_metric, rankdata, score

__all__ = [
    "compare_paired", "default_rope", "holm", "paired_deltas", "rope_verdict",
    "betainc", "chi2_sf", "gammainc_upper", "kolmogorov_sf", "norm_cdf", "norm_ppf", "norm_sf", "t_cdf", "t_ppf", "t_sf",
    "TestResult", "autocorrelation", "block_permutation", "ks_2samp", "permutation_test", "psi",
    "Interval", "bootstrap_interval", "corrected_resampled_t", "t_interval",
    "METRICS", "Metric", "get_metric", "rankdata", "score",
]

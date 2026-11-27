"""
Distribution functions discovery needs, in numpy / math only (scipy is not a
tensor-optix dependency).

* Student-t CDF via the regularised incomplete beta function, evaluated with
  the modified Lentz continued fraction (Numerical Recipes, 3rd ed., 6.4).
  Accurate to ~1e-12 for the df and |t| ranges used here.
* Student-t quantile by bisection on the CDF (monotone, so bisection is
  robust; 200 iterations bound the error far below 1e-10).
* Normal CDF / quantile.
* Kolmogorov distribution survival function (for the KS test).
* Chi-square survival function via the regularised upper incomplete gamma.

Each function is validated in tests against tabulated values (the same table
``nanonets.smalln`` hardcodes) and against identities.
"""
from __future__ import annotations

import math

_EPS = 3.0e-16
_FPMIN = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) for a, b > 0, 0 <= x <= 1."""
    if a <= 0 or b <= 0:
        raise ValueError("betainc requires a > 0 and b > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be > 0")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)       # P(T > |t|)
    return 1.0 - tail if t > 0 else tail


def t_sf(t: float, df: float) -> float:
    """P(T > t) (survival function), computed without cancellation for large t."""
    if t == 0.0:
        return 0.5
    if t > 0:
        x = df / (df + t * t)
        return 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - t_sf(-t, df)


def t_ppf(p: float, df: float) -> float:
    """Quantile: the t with P(T <= t) = p."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -norm_ppf(1.0 - p)
    lo, hi = 0.0, 1.0
    while norm_cdf(hi) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def kolmogorov_sf(lam: float) -> float:
    """
    Q_KS(lambda) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lambda^2): the
    asymptotic P(sqrt(n) D > lambda) of the Kolmogorov-Smirnov statistic.
    """
    if lam <= 0.0:
        return 1.0
    if lam < 0.2:
        # The alternating series converges slowly here; the value is ~1.
        return 1.0
    total, sign = 0.0, 1.0
    for k in range(1, 200):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += sign * term
        if term < 1e-16:
            break
        sign = -sign
    return float(min(max(2.0 * total, 0.0), 1.0))


def gammainc_upper(a: float, x: float) -> float:
    """
    Regularised upper incomplete gamma Q(a, x) = Gamma(a, x) / Gamma(a):
    series for x < a + 1, Lentz continued fraction otherwise (Numerical
    Recipes, 3rd ed., 6.2).
    """
    if a <= 0.0:
        raise ValueError("a must be > 0")
    if x <= 0.0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * _EPS:
                break
        p = s * math.exp(-x + a * math.log(x) - lg)
        return float(min(max(1.0 - p, 0.0), 1.0))
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return float(min(max(math.exp(-x + a * math.log(x) - lg) * h, 0.0), 1.0))


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) for a chi-square variable with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be > 0")
    return gammainc_upper(df / 2.0, x / 2.0)

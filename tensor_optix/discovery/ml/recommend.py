"""
The family recommendation.

Every candidate family (its best configuration) is compared with the leader
-- the family with the best mean CV score -- on the folds they share, paired,
with the Nadeau-Bengio corrected interval and the metric's ROPE; p-values are
Holm-adjusted over all comparisons with the leader.

Decision rule (stated in the report):
  * nothing beats the dummy by more than the ROPE           -> ``abstain``
  * the simplest family *equivalent* to the leader          -> ``tie_simplest_chosen``
    (equivalence = the whole interval of the difference lies inside the ROPE)
  * the leader beats every other family                     -> ``clear_winner``
  * otherwise                                               -> ``undetermined``: the leader is
                                                              named, the families it could not be
                                                              separated from are listed as alternatives
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..findings import Category, Evidence, Finding, Severity
from ..schema import Comparison, ExperimentResult, Recommendation
from ..stats.compare import holm, rope_verdict
from ..stats.intervals import corrected_resampled_t
from .families.base import get_family


def aligned(a: ExperimentResult, b: ExperimentResult) -> Tuple[List[float], List[float]]:
    """Scores of ``a`` and ``b`` on the folds both have (by fold id), both finite."""
    bmap = dict(zip(b.fold_ids, b.scores))
    xa, xb = [], []
    for fid, s in zip(a.fold_ids, a.scores):
        t = bmap.get(fid)
        if s is None or t is None or not (math.isfinite(s) and math.isfinite(t)):
            continue
        xa.append(float(s))
        xb.append(float(t))
    return xa, xb


@dataclass
class PooledLosses:
    """
    Per-row losses of each experiment on forward-chaining test rows (NaN where a
    row was never a test row), the time order of the rows and the HAC lag.
    With these, two families are compared by a Diebold-Mariano test on the pooled
    test rows instead of on a handful of fold means -- far more power when the
    CV cannot be repeated.
    """
    losses: Dict[str, np.ndarray]
    order: np.ndarray
    lag: int
    metric: str


def _dm_compare(a: ExperimentResult, b: ExperimentResult, pooled: PooledLosses, rope: float
                ) -> Optional[Comparison]:
    from ..stats.distributions import norm_sf
    from .structure import andrews_lag, newey_west_se
    la, lb = pooled.losses.get(a.key), pooled.losses.get(b.key)
    if la is None or lb is None:
        return None
    o = pooled.order
    la, lb = la[o], lb[o]
    ok = np.isfinite(la) & np.isfinite(lb)
    if ok.sum() < 30:
        return None
    la, lb = la[ok], lb[ok]
    d = lb - la                                           # positive = a has the lower loss = a better
    lag = andrews_lag(d, floor=max(pooled.lag, int(4 * (len(d) / 100.0) ** (2.0 / 9.0))))
    se = newey_west_se(d, lag)
    mean = float(d.mean())
    # report on the metric's scale: RMSE differences ~ MSE differences / (2 RMSE)
    scale = 2.0 * math.sqrt(max(0.5 * float(la.mean() + lb.mean()), 1e-300)) if pooled.metric == "rmse" else 1.0
    z = 1.959963984540054
    lo, hi = (mean - z * se) / scale, (mean + z * se) / scale
    p = float(min(1.0, 2.0 * norm_sf(abs(mean) / se))) if se > 0 else (0.0 if mean != 0 else 1.0)
    return Comparison(a=a.key, b=b.key, n_pairs=int(len(d)), delta_mean=mean / scale, delta_interval=(lo, hi),
                      rope=rope, verdict=rope_verdict(lo, hi, rope), p_value=p,
                      method=f"diebold_mariano (pooled test rows, Newey-West lag {lag})")


def compare_results(a: ExperimentResult, b: ExperimentResult, *, rope: float, n_train: float,
                    n_test: float, pooled: Optional[PooledLosses] = None) -> Optional[Comparison]:
    if pooled is not None:
        c = _dm_compare(a, b, pooled, rope)
        if c is not None:
            return c
    xa, xb = aligned(a, b)
    if len(xa) < 2:
        return None
    d = np.array(xa) - np.array(xb)
    if not a.higher_is_better:
        d = -d
    iv = corrected_resampled_t(d, n_train, n_test)
    finite = math.isfinite(iv.lo) and math.isfinite(iv.hi)
    return Comparison(a=a.key, b=b.key, n_pairs=len(d), delta_mean=float(d.mean()),
                      delta_interval=(iv.lo, iv.hi) if finite else None, rope=rope,
                      verdict=rope_verdict(iv.lo, iv.hi, rope) if finite else "undetermined",
                      p_value=iv.p_two_sided, method="corrected_resampled_t (paired by fold id)")


def leader_of(best: Dict[str, ExperimentResult]) -> Optional[str]:
    ok = [(f, r) for f, r in best.items() if r.mean is not None]
    if not ok:
        return None
    hib = ok[0][1].higher_is_better
    return max(ok, key=lambda fr: fr[1].mean if hib else -fr[1].mean)[0]


def compare_with_leader(best: Dict[str, ExperimentResult], *, rope: float, n_train: float, n_test: float,
                        pooled: Optional[PooledLosses] = None
                        ) -> Tuple[Optional[str], Dict[str, Comparison]]:
    """Leader vs each other family, delta oriented as *other minus leader* (positive = other better)."""
    lead = leader_of(best)
    out: Dict[str, Comparison] = {}
    if lead is None:
        return None, out
    for f, r in best.items():
        if f == lead or r.mean is None:
            continue
        c = compare_results(r, best[lead], rope=rope, n_train=n_train, n_test=n_test, pooled=pooled)
        if c is not None:
            out[f] = c
    adj = holm([c.p_value for c in out.values()])
    out = {f: replace(c, p_holm=q) for (f, c), q in zip(out.items(), adj)}
    return lead, out


def undetermined_vs_leader(best: Dict[str, ExperimentResult], *, rope: float, n_train: float, n_test: float,
                           pooled: Optional[PooledLosses] = None
                           ) -> List[str]:
    """Families to refine: the leader plus those it cannot yet be separated from (best first)."""
    lead, comps = compare_with_leader(best, rope=rope, n_train=n_train, n_test=n_test, pooled=pooled)
    if lead is None:
        return []
    und = [f for f, c in comps.items() if c.verdict == "undetermined"]
    if not und:
        return []
    hib = best[lead].higher_is_better
    und.sort(key=lambda f: -(best[f].mean if hib else -best[f].mean))
    return [lead] + und


def recommend(best: Dict[str, ExperimentResult], dummy: Optional[ExperimentResult], *, rope: float,
              n_train: float, n_test: float, metric: str, pooled: Optional[PooledLosses] = None
              ) -> Tuple[Recommendation, List[Comparison], List[Finding]]:
    findings: List[Finding] = []
    lead, comps = compare_with_leader(best, rope=rope, n_train=n_train, n_test=n_test, pooled=pooled)
    all_comps = list(comps.values())
    if lead is None:
        rec = Recommendation(family=None, verdict="abstain", metric=metric,
                             rationale=("No candidate family produced a usable score.",))
        return rec, all_comps, findings
    lr = best[lead]

    # ---- does anything beat the dummy?
    beats_dummy = True
    vs_dummy = None
    if dummy is not None and dummy.mean is not None:
        vs_dummy = compare_results(lr, dummy, rope=rope, n_train=n_train, n_test=n_test, pooled=pooled)
        if vs_dummy is not None:
            all_comps.append(vs_dummy)
            beats_dummy = vs_dummy.verdict == "better"
    if not beats_dummy:
        what = "mean" if metric in ("rmse", "mae", "r2") else "class prior"
        inconclusive = vs_dummy is not None and vs_dummy.verdict == "undetermined"
        if inconclusive:
            lo, hi = vs_dummy.delta_interval or (float("nan"), float("nan"))
            why = (f"The best family ({lead}, {metric} {lr.mean:.4f}) could not be shown to beat predicting the "
                   f"{what} ({dummy.mean:.4f}): the 95% interval of its advantage, [{lo:+.4f}, {hi:+.4f}], "
                   f"does not clear the practical threshold ({rope:g}). The evidence is inconclusive, not "
                   "proof that there is no signal.")
            statement = ("Discovery does not recommend a model family: the evidence that any family beats the "
                         "dummy baseline is inconclusive with the data and folds available.")
        else:
            why = (f"The best family ({lead}, {metric} {lr.mean:.4f}) is not better than predicting the {what} "
                   f"({dummy.mean:.4f}) by more than the practical threshold ({rope:g}).")
            statement = ("Discovery does not recommend a model family: none beats the dummy baseline by a "
                         "margin that matters.")
        rec = Recommendation(
            family=None, verdict="abstain", metric=metric, expected_score=None,
            rationale=(why, "Revisit the features or the target before choosing a model (see the data findings)"
                            + ("; more rows, or a less volatile target, would sharpen the comparison."
                               if inconclusive else "."),
                       f"build(family=...) still builds any family explicitly (best scoring: {lead})."),
            alternatives=tuple(f for f in best if f != lead) if not inconclusive else (lead,))
        findings.append(Finding(
            id="family.abstain", category=Category.FAMILY, severity=Severity.WARNING,
            statement=statement,
            evidence=(Evidence(test="paired_corrected_t", statistic=vs_dummy.delta_mean if vs_dummy else None,
                               interval=vs_dummy.delta_interval if vs_dummy else None,
                               details={"leader": lead, "rope": rope,
                                        "verdict_vs_dummy": vs_dummy.verdict if vs_dummy else None}),),
            confidence="low" if inconclusive else "moderate",
            confidence_basis="paired interval vs the dummy overlaps the ROPE" if inconclusive
            else "paired interval vs the dummy lies within the ROPE or below it"))
        return rec, all_comps, findings

    def rank(f: str) -> Tuple[int, str]:
        return (get_family(f).cost_rank, f)

    equivalent = [f for f, c in comps.items() if c.verdict == "equivalent"]
    undetermined = [f for f, c in comps.items() if c.verdict == "undetermined"]
    better_than_leader = [f for f, c in comps.items() if c.verdict == "better"]   # cannot happen; guarded
    simplest = min([lead] + equivalent, key=rank)

    def fmt(f: str) -> str:
        r = best[f]
        return f"{f} ({metric} {r.mean:.4f})"

    rationale: List[str] = []
    if simplest != lead:
        verdict = "tie_simplest_chosen"
        c = comps[simplest]
        rationale.append(f"{fmt(lead)} scored best, but {fmt(simplest)} is equivalent to it: the 95% interval of "
                         f"the difference [{c.delta_interval[0]:+.4f}, {c.delta_interval[1]:+.4f}] lies inside "
                         f"the practical threshold of ±{rope:g}. The simpler family is chosen.")
    elif not undetermined and not equivalent and not better_than_leader:
        verdict = "clear_winner"
        rationale.append(f"{fmt(lead)} is better than every other family tried by more than the practical "
                         f"threshold (±{rope:g}), on identical test rows.")
    else:
        verdict = "undetermined"
        rationale.append(f"{fmt(lead)} scored best, but it cannot be separated from "
                         f"{', '.join(fmt(f) for f in undetermined)} with the data and budget available.")
    if vs_dummy is not None:
        rationale.append(f"It beats the dummy baseline ({metric} {dummy.mean:.4f}) by "
                         f"{vs_dummy.delta_mean:+.4f} (95% CI {vs_dummy.delta_interval[0]:+.4f} to "
                         f"{vs_dummy.delta_interval[1]:+.4f})." if vs_dummy.delta_interval else "")
    worse = [f for f, c in comps.items() if c.verdict == "worse"]
    if worse:
        rationale.append(f"Ruled out (worse than the leader by more than the threshold): {', '.join(sorted(worse))}.")
    chosen = best[simplest]
    alternatives = tuple(sorted(set(undetermined + equivalent + ([lead] if simplest != lead else [])) - {simplest},
                                key=rank))
    rec = Recommendation(family=simplest, config=dict(chosen.config), verdict=verdict,
                         rationale=tuple(r for r in rationale if r), metric=metric,
                         alternatives=alternatives)
    return rec, all_comps, findings

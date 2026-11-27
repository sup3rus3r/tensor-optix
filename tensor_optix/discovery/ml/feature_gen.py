"""
Feature-family generators for discovery's feature-opportunity stage and for
``MLSolution`` (so a built model computes exactly the features discovery
measured).

Every generator is leakage-safe by construction:
  * row-wise transforms (calendar cycles, log, products) use only the row;
  * temporal families use only *earlier* rows of the same series (rows ordered
    by time, one series per group when groups are given); features of the
    current row are allowed (they are known at prediction time), the target of
    the current row never is -- target lags start at lag 1;
  * group aggregates use features only, never the target.

A ``FeatureSpec`` is JSON-native (it is the parameter set of the
``add_feature_family`` action), so a report fully describes the features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .data import Table
from .preprocess import _as_datetime

PREFIX = "fx__"
FAMILIES = ("target_lags", "rolling_features", "volatility", "calendar_cycles", "log_transform",
            "interactions", "group_aggregates")
TARGET_FAMILIES = ("target_lags", "volatility")          # need past target values at prediction time


@dataclass(frozen=True)
class FeatureSpec:
    family: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown feature family {self.family!r}; known: {FAMILIES}")

    @property
    def needs_target(self) -> bool:
        return self.family in TARGET_FAMILIES

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, **self.params}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        d = dict(d)
        return cls(d.pop("family"), d)


# ---------------------------------------------------------------------------
# series helpers
# ---------------------------------------------------------------------------

def series_positions(n: int, order: Optional[np.ndarray], groups: Optional[np.ndarray]) -> List[np.ndarray]:
    """Row indices of each series, in time order."""
    order = np.arange(n) if order is None else np.asarray(order)
    if groups is None:
        return [order]
    g = np.asarray(groups)[order]
    codes = np.unique(g, return_inverse=True)[1]
    out = []
    for c in range(int(codes.max()) + 1 if len(codes) else 0):
        out.append(order[codes == c])                      # stable: keeps time order inside the group
    return out


def _past_rolling(s: np.ndarray, w: int, stat: str) -> np.ndarray:
    """Statistic of the previous ``w`` values (excluding the current one), NaN-aware."""
    n = len(s)
    out = np.full(n, np.nan)
    v = np.where(np.isfinite(s), s, 0.0)
    m = np.isfinite(s).astype(np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cm = np.concatenate([[0.0], np.cumsum(m)])
    for i in range(1, n):
        lo = max(0, i - w)
        k = cm[i] - cm[lo]
        if k < 1 or (stat == "std" and k < 2):
            continue
        mean = (c1[i] - c1[lo]) / k
        if stat == "mean":
            out[i] = mean
        else:
            var = max((c2[i] - c2[lo]) / k - mean * mean, 0.0) * k / (k - 1)
            out[i] = np.sqrt(var)
    return out


def _incl_rolling(s: np.ndarray, w: int, stat: str) -> np.ndarray:
    """Statistic of the last ``w`` values *including* the current one (for features)."""
    ext = np.append(s, np.nan)
    shifted = _past_rolling(ext, w, stat)[1:]
    return shifted


def _num(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind in "biuf":
        return col.astype(np.float64)
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate(spec: FeatureSpec, table: Table, y: Optional[np.ndarray] = None,
             order: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """New columns (name -> float array) for every row of ``table``."""
    n = table.n_rows
    p = spec.params
    fam = spec.family
    out: Dict[str, np.ndarray] = {}
    if fam in TARGET_FAMILIES:
        if y is None:
            raise ValueError(f"feature family {fam!r} needs the target's past values (pass the target)")
        yv = np.asarray(y, dtype=np.float64)
    if fam == "target_lags":
        lags = list(p.get("lags", (1, 2, 3)))
        windows = list(p.get("windows", (5, 20)))
        cols = {f"{PREFIX}y_lag{k}": np.full(n, np.nan) for k in lags}
        cols.update({f"{PREFIX}y_mean{w}": np.full(n, np.nan) for w in windows})
        for pos in series_positions(n, order, groups):
            s = yv[pos]
            for k in lags:
                v = np.full(len(s), np.nan)
                v[k:] = s[:-k] if k < len(s) else v[k:]
                cols[f"{PREFIX}y_lag{k}"][pos] = v
            for w in windows:
                cols[f"{PREFIX}y_mean{w}"][pos] = _past_rolling(s, w, "mean")
        out.update(cols)
    elif fam == "volatility":
        windows = list(p.get("windows", (10, 30)))
        for w in windows:
            out[f"{PREFIX}y_std{w}"] = np.full(n, np.nan)
        out[f"{PREFIX}y_absdiff1"] = np.full(n, np.nan)
        for pos in series_positions(n, order, groups):
            s = yv[pos]
            for w in windows:
                out[f"{PREFIX}y_std{w}"][pos] = _past_rolling(s, w, "std")
            d = np.full(len(s), np.nan)
            if len(s) > 2:
                d[2:] = np.abs(s[1:-1] - s[:-2])               # |y[t-1] - y[t-2]|: past only
            out[f"{PREFIX}y_absdiff1"][pos] = d
    elif fam == "rolling_features":
        windows = list(p.get("windows", (5, 20)))
        for c in p["columns"]:
            x = _num(table.columns[c])
            for w in windows:
                m = np.full(n, np.nan)
                sd = np.full(n, np.nan)
                for pos in series_positions(n, order, groups):
                    m[pos] = _incl_rolling(x[pos], w, "mean")
                    sd[pos] = _incl_rolling(x[pos], w, "std")
                out[f"{PREFIX}{c}_rmean{w}"] = m
                out[f"{PREFIX}{c}_rstd{w}"] = sd
            dx = np.full(n, np.nan)
            for pos in series_positions(n, order, groups):
                s = x[pos]
                v = np.full(len(s), np.nan)
                v[1:] = s[1:] - s[:-1]
                dx[pos] = v
            out[f"{PREFIX}{c}_diff1"] = dx
    elif fam == "calendar_cycles":
        for c in p["columns"]:
            t = _as_datetime(table.columns[c])
            nat = np.isnat(t)
            safe = np.where(nat, np.datetime64("1970-01-01", "ns"), t).astype("datetime64[ns]")
            days = safe.astype("datetime64[D]")
            dow = ((days.astype(np.int64) + 3) % 7).astype(np.float64)          # Monday = 0
            month = (safe.astype("datetime64[M]").astype(np.int64) % 12).astype(np.float64)
            doy = (days - safe.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64).astype(np.float64)
            hour = ((safe - days.astype("datetime64[ns]")).astype("timedelta64[h]").astype(np.int64)
                    ).astype(np.float64)
            parts = {"dow": (dow, 7.0), "month": (month, 12.0), "doy": (doy, 365.25)}
            if np.unique(hour[~nat]).size > 1:
                parts["hour"] = (hour, 24.0)
            for name, (v, period) in parts.items():
                ang = 2.0 * np.pi * v / period
                s_, c_ = np.sin(ang), np.cos(ang)
                s_[nat] = np.nan
                c_[nat] = np.nan
                out[f"{PREFIX}{c}_{name}_sin"] = s_
                out[f"{PREFIX}{c}_{name}_cos"] = c_
    elif fam == "log_transform":
        for c in p["columns"]:
            x = _num(table.columns[c])
            out[f"{PREFIX}{c}_slog"] = np.sign(x) * np.log1p(np.abs(x))
    elif fam == "interactions":
        cols = list(p["columns"])
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = _num(table.columns[cols[i]]), _num(table.columns[cols[j]])
                out[f"{PREFIX}{cols[i]}_x_{cols[j]}"] = a * b
    elif fam == "group_aggregates":
        if groups is None:
            raise ValueError("feature family 'group_aggregates' needs the groups")
        codes = np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
        size = np.bincount(codes).astype(np.float64)
        out[f"{PREFIX}group_size"] = size[codes]
        for c in p["columns"]:
            x = _num(table.columns[c])
            ok = np.isfinite(x)
            s = np.bincount(codes, weights=np.where(ok, x, 0.0))
            s2 = np.bincount(codes, weights=np.where(ok, x * x, 0.0))
            k = np.bincount(codes, weights=ok.astype(np.float64))
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = s / k
                var = np.clip(s2 / k - mean * mean, 0.0, None) * k / np.maximum(k - 1, 1)
            out[f"{PREFIX}{c}_gmean"] = mean[codes]
            out[f"{PREFIX}{c}_gstd"] = np.sqrt(var)[codes]
            out[f"{PREFIX}{c}_gdev"] = x - mean[codes]
    return out


def augment(table: Table, specs: Sequence[FeatureSpec], y: Optional[np.ndarray] = None,
            order: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> Table:
    cols = dict(table.columns)
    for s in specs:
        for k, v in generate(s, table, y, order, groups).items():
            cols[k] = v
    return Table(columns=cols, target=table.target, target_name=table.target_name, groups=table.groups,
                 groups_name=table.groups_name, time=table.time, time_name=table.time_name,
                 declared_types=dict(table.declared_types))

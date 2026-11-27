"""
Fold-safe preprocessing: turn a ``Table`` into a float matrix, learning every
statistic (medians, scales, category levels, target encodings) from the
*training rows of the current fold only*.

``ColumnPipeline.fit_transform(table, train_idx, y_train)`` learns and
transforms the training rows; ``transform(table, idx)`` applies the learned
state to any other rows. Target encoding of high-cardinality categoricals is
computed *out-of-fold inside the training rows* (an inner K-fold), so a
training row never sees its own target through its encoding; test rows use
statistics from all training rows.

Per-column recipe by inferred type:
  numeric / discrete     median impute (+ missing indicator) -> scale
  boolean                0/1 (+ missing indicator)
  categorical_low        one-hot of training levels (rare levels pooled; unseen -> all zeros)
                         or ordinal codes
  categorical_high       smoothed target encoding (per class for multiclass) or ordinal codes
  datetime               year, month, day of week, day of year, hour (when it varies), days since epoch
  embedded_number        parsed to a number, then as numeric
  identifier / text / url / list / constant / unknown -> dropped
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..rng import rng_for
from .data import Table
from .types import ColumnType, parse_embedded_number

_DROP = {"identifier", "text", "url", "list", "constant", "unknown"}
_BOOL_TRUE = {"true", "yes", "y", "t", "1"}


@dataclass(frozen=True)
class PipelineOptions:
    scale: str = "standard"                  # "standard" | "robust" | "none"
    categorical: str = "onehot"              # "onehot" | "ordinal"
    high_cardinality: str = "target"         # "target" | "ordinal"
    missing_indicators: bool = True
    min_level_frequency: int = 5             # one-hot: rarer training levels pooled into "other"
    max_onehot_levels: int = 50
    target_smoothing: float = 10.0
    inner_folds: int = 5
    robust_columns: Tuple[str, ...] = ()     # scale these robustly even when scale="standard"
    quantile_columns: Tuple[str, ...] = ()   # map these to normal scores via training-fold quantiles
    drop: Tuple[str, ...] = ()               # extra columns to drop (accepted findings)
    type_overrides: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.scale not in ("standard", "robust", "none"):
            raise ValueError("scale must be 'standard', 'robust' or 'none'")
        if self.categorical not in ("onehot", "ordinal"):
            raise ValueError("categorical must be 'onehot' or 'ordinal'")
        if self.high_cardinality not in ("target", "ordinal"):
            raise ValueError("high_cardinality must be 'target' or 'ordinal'")


def _as_float(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind in "biuf":
        return col.astype(np.float64)
    if col.dtype.kind == "M":
        out = col.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
        out[np.isnat(col)] = np.nan
        return out
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _as_bool(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind in "biuf":
        x = col.astype(np.float64)
        return np.where(np.isnan(x), np.nan, (x != 0).astype(np.float64))
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            out[i] = float(v)
        else:
            out[i] = 1.0 if str(v).strip().lower() in _BOOL_TRUE else 0.0
    return out


def _as_datetime(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind == "M":
        return col.astype("datetime64[ns]")
    out = np.full(len(col), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i, v in enumerate(col):
        if v is None:
            continue
        s = str(v).strip().replace("/", "-")
        try:
            out[i] = np.datetime64(s.replace("Z", ""), "ns")
        except ValueError:
            parts = s.split("-")
            if len(parts) == 3 and len(parts[2]) == 4:                   # d-m-yyyy / m-d-yyyy
                try:
                    out[i] = np.datetime64(f"{parts[2]}-{int(parts[0]):02d}-{int(parts[1]):02d}", "ns")
                except ValueError:
                    pass
    return out


def _keys(col: np.ndarray) -> np.ndarray:
    """Category keys as strings; missing gets its own key."""
    if col.dtype == object:
        return np.array(["\x00missing" if v is None else str(v) for v in col], dtype=object)
    if col.dtype.kind == "f":
        return np.array(["\x00missing" if np.isnan(v) else repr(float(v)) for v in col], dtype=object)
    return np.array([repr(v.item() if hasattr(v, "item") else v) for v in col], dtype=object)


# ---------------------------------------------------------------------------
# Column steps
# ---------------------------------------------------------------------------

class _Numeric:
    def __init__(self, name, scale, indicator, parse=None):
        self.name, self.scale, self.indicator, self.parse = name, scale, indicator, parse

    def _x(self, col):
        if self.parse == "embedded":
            return np.array([parse_embedded_number(v) for v in col], dtype=np.float64)
        if self.parse == "bool":
            return _as_bool(col)
        return _as_float(col)

    def fit(self, col, y, ctx):
        x = self._x(col)
        ok = ~np.isnan(x)
        self.fill = float(np.median(x[ok])) if ok.any() else 0.0
        self.has_missing = bool((~ok).any()) and self.indicator
        xf = np.where(ok, x, self.fill)
        if self.scale == "quantile":
            # normal scores from the training rows' quantiles (rank-based: outliers and skew vanish)
            from ..stats.distributions import norm_ppf
            probs = np.linspace(0.001, 0.999, 201)
            q = np.quantile(xf, probs)
            uq, first = np.unique(q, return_index=True)
            z = np.array([norm_ppf(p_) for p_ in probs])
            self.qx = uq
            self.qz = np.array([z[q == v].mean() for v in uq]) if len(uq) < len(q) else z[first]
            self.center, self.spread = 0.0, 1.0
        elif self.scale == "robust":
            self.center = float(np.median(xf))
            q75, q25 = np.percentile(xf, [75, 25])
            s = float(q75 - q25)
            self.spread = s if s > 0 else (float(xf.std()) or 1.0)
        elif self.scale == "standard":
            self.center = float(xf.mean())
            self.spread = float(xf.std()) or 1.0
        else:
            self.center, self.spread = 0.0, 1.0
        names = [self.name] + ([f"{self.name}__missing"] if self.has_missing else [])
        return names

    def transform(self, col, ctx):
        x = self._x(col)
        miss = np.isnan(x)
        xv = np.where(miss, self.fill, x)
        if self.scale == "quantile":
            v = np.interp(xv, self.qx, self.qz) if len(self.qx) > 1 else np.zeros(len(xv))
        else:
            v = (xv - self.center) / self.spread
        if self.has_missing:
            return np.column_stack([v, miss.astype(np.float64)])
        return v[:, None]


class _DateParts:
    PARTS = ("year", "month", "dayofweek", "dayofyear", "hour", "days")

    def __init__(self, name, scale, indicator):
        self.name, self.scale, self.indicator = name, scale, indicator

    def _parts(self, col):
        t = _as_datetime(col)
        nat = np.isnat(t)
        safe = np.where(nat, np.datetime64("1970-01-01", "ns"), t)
        days = safe.astype("datetime64[D]")
        years = safe.astype("datetime64[Y]").astype(np.int64) + 1970
        months = safe.astype("datetime64[M]").astype(np.int64) % 12 + 1
        dow = (days.astype(np.int64) + 3) % 7                        # 1970-01-01 was a Thursday
        doy = (days - safe.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64) + 1
        hour = (safe.astype("datetime64[h]").astype(np.int64) % 24)
        epoch_days = safe.astype("datetime64[s]").astype(np.int64) / 86400.0
        cols = {"year": years, "month": months, "dayofweek": dow, "dayofyear": doy, "hour": hour,
                "days": epoch_days}
        out = {k: np.where(nat, np.nan, v.astype(np.float64)) for k, v in cols.items()}
        return out

    def fit(self, col, y, ctx):
        parts = self._parts(col)
        self.keep = [p for p in self.PARTS
                     if np.nanstd(parts[p]) > 0 or (p == "days" and np.isfinite(parts[p]).any())]
        self.steps = {}
        names = []
        for p in self.keep:
            st = _Numeric(f"{self.name}__{p}", self.scale, self.indicator and p == self.keep[0])
            names += st.fit(parts[p], y, ctx)
            self.steps[p] = st
        return names

    def transform(self, col, ctx):
        parts = self._parts(col)
        blocks = [self.steps[p].transform(parts[p], ctx) for p in self.keep]
        return np.column_stack(blocks) if blocks else np.zeros((len(col), 0))


class _OneHot:
    def __init__(self, name, min_freq, max_levels):
        self.name, self.min_freq, self.max_levels = name, min_freq, max_levels

    def fit(self, col, y, ctx):
        keys = _keys(col)
        lv, cnt = np.unique(keys, return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        kept = [lv[i] for i in order if cnt[i] >= self.min_freq][: self.max_levels]
        self.levels = kept
        self.index = {k: i for i, k in enumerate(kept)}
        self.has_other = len(kept) < len(lv)
        names = [f"{self.name}={k if not k.startswith(chr(0)) else '<missing>'}" for k in kept]
        return names + ([f"{self.name}=<other>"] if self.has_other else [])

    def transform(self, col, ctx):
        keys = _keys(col)
        width = len(self.levels) + (1 if self.has_other else 0)
        out = np.zeros((len(keys), width))
        for i, k in enumerate(keys):
            j = self.index.get(k)
            if j is not None:
                out[i, j] = 1.0
            elif self.has_other:
                out[i, -1] = 1.0                      # rare or unseen level
        return out


class _Ordinal:
    """Codes by training frequency (most frequent = 0); unseen -> -1. Missing is a level."""
    def __init__(self, name):
        self.name = name

    def fit(self, col, y, ctx):
        lv, cnt = np.unique(_keys(col), return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        self.index = {lv[i]: float(r) for r, i in enumerate(order)}
        return [self.name]

    def transform(self, col, ctx):
        return np.array([self.index.get(k, -1.0) for k in _keys(col)])[:, None]


class _TargetEncode:
    """Smoothed mean target per level (per class for multiclass). Training rows get
    out-of-fold values (inner K-fold); other rows get full-training statistics."""
    def __init__(self, name, smoothing, inner_folds):
        self.name, self.m, self.inner = name, smoothing, inner_folds

    def _targets(self, y, ctx):
        if ctx["task"] == "multiclass":
            return np.eye(ctx["n_classes"])[y.astype(int)]           # (n, K)
        return y.astype(np.float64)[:, None]                        # binary / regression: (n, 1)

    def _stats(self, keys, T):
        prior = T.mean(axis=0)
        lv, inv = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(lv), T.shape[1]))
        np.add.at(sums, inv, T)
        counts = np.bincount(inv, minlength=len(lv))[:, None]
        enc = (sums + self.m * prior) / (counts + self.m)
        return {k: enc[i] for i, k in enumerate(lv)}, prior

    def fit(self, col, y, ctx):
        if y is None:
            raise ValueError("target encoding needs the training target")
        keys = _keys(col)
        T = self._targets(y, ctx)
        self.table, self.prior = self._stats(keys, T)
        # out-of-fold encodings for the training rows themselves
        n = len(keys)
        k = max(2, min(self.inner, n))
        perm = rng_for(ctx["seed"], "target_encoding", self.name).permutation(n)
        oof = np.empty_like(T, dtype=np.float64)
        for part in np.array_split(perm, k):
            rest = np.setdiff1d(perm, part, assume_unique=True)
            tab, pr = self._stats(keys[rest], T[rest])
            oof[part] = np.array([tab.get(kk, pr) for kk in keys[part]])
        self.train_values = oof
        width = T.shape[1]
        return [f"{self.name}__te" if width == 1 else f"{self.name}__te{c}" for c in range(width)]

    def transform(self, col, ctx):
        keys = _keys(col)
        return np.array([self.table.get(k, self.prior) for k in keys])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ColumnPipeline:
    def __init__(self, types: Dict[str, ColumnType], options: PipelineOptions = PipelineOptions(),
                 *, task: str = "regression", n_classes: int = 0, seed: int = 0):
        self.types = dict(types)
        for name, t in options.type_overrides:
            if name in self.types:
                self.types[name] = ColumnType(name=name, type=t, confidence="high", declared=True)
        self.options = options
        self.ctx = {"task": task, "n_classes": n_classes, "seed": seed}
        self.steps: List[Tuple[str, Any]] = []
        self.feature_names: List[str] = []
        self.fitted_rows: Optional[np.ndarray] = None       # for audits: rows the fit saw
        self._fitted = False

    def _make_step(self, name: str, ct: ColumnType):
        o = self.options
        if name in o.drop or ct.type in _DROP:
            return None
        scale = o.scale
        if o.scale != "none":
            if name in o.quantile_columns:
                scale = "quantile"
            elif name in o.robust_columns:
                scale = "robust"
        if ct.type in ("numeric_continuous", "numeric_discrete"):
            return _Numeric(name, scale, o.missing_indicators)
        if ct.type == "embedded_number":
            return _Numeric(name, scale, o.missing_indicators, parse="embedded")
        if ct.type == "boolean":
            return _Numeric(name, "none", o.missing_indicators, parse="bool")
        if ct.type == "datetime":
            return _DateParts(name, scale, o.missing_indicators)
        if ct.type == "categorical_low":
            if o.categorical == "onehot":
                return _OneHot(name, o.min_level_frequency, o.max_onehot_levels)
            return _Ordinal(name)
        if ct.type == "categorical_high":
            if o.high_cardinality == "target" and self.ctx["task"] != "none":
                return _TargetEncode(name, o.target_smoothing, o.inner_folds)
            return _Ordinal(name)
        raise ValueError(f"no preprocessing recipe for type {ct.type!r}")

    def fit_transform(self, table: Table, rows: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Learn from ``rows`` only and return their transformed features.
        ``y`` is the encoded target for ``rows`` (needed for target encoding)."""
        rows = np.asarray(rows)
        self.fitted_rows = rows.copy()
        self.steps, self.feature_names, blocks = [], [], []
        for name in table.names:
            ct = self.types.get(name)
            if ct is None:
                continue
            step = self._make_step(name, ct)
            if step is None:
                continue
            col = table.columns[name][rows]
            names = step.fit(col, y, self.ctx)
            if isinstance(step, _TargetEncode):
                block = step.train_values
            else:
                block = step.transform(col, self.ctx)
            if block.ndim == 1:
                block = block[:, None]
            self.steps.append((name, step))
            self.feature_names += names
            blocks.append(block)
        self._fitted = True
        return self._stack(blocks, len(rows))

    def transform(self, table: Table, rows: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ColumnPipeline.transform() before fit_transform()")
        rows = np.asarray(rows)
        blocks = []
        for name, step in self.steps:
            block = step.transform(table.columns[name][rows], self.ctx)
            blocks.append(block[:, None] if block.ndim == 1 else block)
        return self._stack(blocks, len(rows))

    @staticmethod
    def _stack(blocks: List[np.ndarray], n: int) -> np.ndarray:
        if not blocks:
            return np.zeros((n, 0), dtype=np.float64)
        X = np.column_stack(blocks).astype(np.float64)
        X[~np.isfinite(X)] = 0.0
        return X

    def spec(self) -> Dict[str, Any]:
        """JSON-native description (what would be built), for reports / plans."""
        o = self.options
        return {"scale": o.scale, "categorical": o.categorical, "high_cardinality": o.high_cardinality,
                "missing_indicators": o.missing_indicators, "drop": list(o.drop),
                "robust_columns": list(o.robust_columns), "quantile_columns": list(o.quantile_columns),
                "type_overrides": [list(p) for p in o.type_overrides]}


def log1p_target(y: np.ndarray) -> np.ndarray:
    if np.any(y < 0):
        raise ValueError("log1p target transform needs a non-negative target")
    return np.log1p(y)


def expm1_target(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)

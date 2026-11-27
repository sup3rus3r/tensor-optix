"""
``Table``: the one data container discovery's ML side works on.

Accepts a pandas DataFrame (pandas is imported only if one is passed), a 2-D
numpy array (optionally with column names / types), a structured array, an
``(X, y)`` pair, or a dict of 1-D arrays. Columns keep their original kind
(numbers, strings / objects, datetimes, booleans) so feature-type inference
sees what the user actually has, instead of everything cast to float32.

Missing values are normalised: NaN for float columns, ``None`` for object
columns, NaT for datetimes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def is_dataframe(obj: Any) -> bool:
    """Duck-typed pandas DataFrame check that never imports pandas."""
    mod = type(obj).__module__ or ""
    return mod.startswith("pandas") and hasattr(obj, "columns") and hasattr(obj, "iloc")


def _is_series(obj: Any) -> bool:
    mod = type(obj).__module__ or ""
    return mod.startswith("pandas") and hasattr(obj, "to_numpy") and not hasattr(obj, "columns")


def _normalise_column(values: Any) -> np.ndarray:
    """1-D numpy array with consistent missing markers."""
    if _is_series(values):
        values = _series_to_numpy(values)
    a = np.asarray(values)
    if a.ndim != 1:
        a = a.reshape(len(a), -1)
        if a.shape[1] != 1:
            raise ValueError("each column must be 1-D")
        a = a[:, 0]
    if a.dtype.kind in "iub":
        return a
    if a.dtype.kind == "f":
        return a.astype(np.float64, copy=False)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]", copy=False)
    if a.dtype.kind == "m":
        return a.astype("timedelta64[ns]").astype(np.int64).astype(np.float64)
    # strings / objects: one object array, missing -> None
    out = np.empty(len(a), dtype=object)
    for i, v in enumerate(a.tolist() if a.dtype.kind in "US" else a):
        out[i] = None if _is_missing_scalar(v) else v
    return out


def _is_missing_scalar(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    t = type(v).__name__
    if t in ("NAType", "NaTType"):
        return True
    if isinstance(v, np.datetime64) and np.isnat(v):
        return True
    return False


def _series_to_numpy(s) -> np.ndarray:
    """pandas Series -> numpy with missing preserved and categories as values."""
    dtype = str(s.dtype)
    if dtype == "category":
        return np.asarray(s.astype(object).to_numpy(), dtype=object)
    if dtype in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
                 "Float32", "Float64"):
        return s.astype("float64").to_numpy()
    if dtype == "boolean":
        if s.isna().any():
            return np.asarray(s.astype(object).to_numpy(), dtype=object)
        return s.astype(bool).to_numpy()
    if dtype.startswith("datetime64"):
        try:
            s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
        except (AttributeError, TypeError):
            pass
        return s.to_numpy(dtype="datetime64[ns]")
    if dtype in ("string", "str"):
        return np.asarray(s.astype(object).to_numpy(), dtype=object)
    return s.to_numpy()


def missing_mask(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind == "f":
        return np.isnan(col)
    if col.dtype.kind == "M":
        return np.isnat(col)
    if col.dtype == object:
        return np.fromiter((_is_missing_scalar(v) for v in col), dtype=bool, count=len(col))
    return np.zeros(len(col), dtype=bool)


@dataclass
class Table:
    columns: Dict[str, np.ndarray]
    target: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    groups: Optional[np.ndarray] = None
    groups_name: Optional[str] = None
    time: Optional[np.ndarray] = None
    time_name: Optional[str] = None
    declared_types: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        lengths = {name: len(c) for name, c in self.columns.items()}
        for extra in ("target", "groups", "time"):
            v = getattr(self, extra)
            if v is not None:
                lengths[f"<{extra}>"] = len(v)
        if len(set(lengths.values())) > 1:
            raise ValueError(f"all columns must have the same length, got {lengths}")
        if not self.columns:
            raise ValueError("the data has no feature columns")

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.columns.values())))

    @property
    def names(self) -> List[str]:
        return list(self.columns)

    def take(self, idx: np.ndarray) -> "Table":
        idx = np.asarray(idx)
        return Table(
            columns={k: v[idx] for k, v in self.columns.items()},
            target=None if self.target is None else self.target[idx], target_name=self.target_name,
            groups=None if self.groups is None else self.groups[idx], groups_name=self.groups_name,
            time=None if self.time is None else self.time[idx], time_name=self.time_name,
            declared_types=dict(self.declared_types),
        )

    def drop(self, names: Sequence[str]) -> "Table":
        keep = {k: v for k, v in self.columns.items() if k not in set(names)}
        return Table(columns=keep, target=self.target, target_name=self.target_name,
                     groups=self.groups, groups_name=self.groups_name, time=self.time,
                     time_name=self.time_name,
                     declared_types={k: v for k, v in self.declared_types.items() if k in keep})

    def fingerprint(self, max_rows: int = 2000) -> str:
        """
        sha256 over the schema, the row count and a hash of the first
        ``max_rows`` rows. Stable for identical data; contains no raw values.
        """
        h = hashlib.sha256()
        h.update(str(self.n_rows).encode())
        for name, col in self.columns.items():
            h.update(b"\x1f" + name.encode("utf-8", "replace") + b"\x1e" + str(col.dtype).encode())
            head = col[:max_rows]
            h.update(repr(head.tolist()).encode("utf-8", "replace"))
        if self.target is not None:
            h.update(b"\x1dtarget" + repr(self.target[:max_rows].tolist()).encode("utf-8", "replace"))
        return h.hexdigest()


def _default_names(k: int, given: Optional[Sequence[str]]) -> List[str]:
    if given is None:
        return [f"x{i}" for i in range(k)]
    names = [str(n) for n in given]
    if len(names) != k:
        raise ValueError(f"column_names has {len(names)} names for {k} columns")
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique")
    return names


def _resolve_side(spec: Any, columns: Dict[str, np.ndarray], n: int, what: str
                  ) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """``spec`` is a column name (removed from the features) or an array."""
    if spec is None:
        return None, None
    if isinstance(spec, str):
        if spec not in columns:
            raise KeyError(f"{what} column {spec!r} not found; columns: {list(columns)[:20]}")
        return columns.pop(spec), spec
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        names = list(columns)
        if not 0 <= int(spec) < len(names):
            raise IndexError(f"{what} index {spec} out of range")
        name = names[int(spec)]
        return columns.pop(name), name
    arr = _normalise_column(spec)
    if len(arr) != n:
        raise ValueError(f"{what} has {len(arr)} values for {n} rows")
    return arr, None


def to_table(data: Any, target: Any = None, *, groups: Any = None, time: Any = None,
             column_names: Optional[Sequence[str]] = None,
             column_types: Optional[Dict[str, str]] = None) -> Table:
    """Normalise any supported dataset input into a ``Table``."""
    if isinstance(data, tuple) and len(data) == 2 and target is None:
        data, target = data
    if is_dataframe(data):
        if column_names is not None:
            raise ValueError("column_names is only for numpy input; a DataFrame has its own")
        columns = {str(c): _normalise_column(_series_to_numpy(data[c])) for c in data.columns}
        if len(columns) != len(data.columns):
            raise ValueError("DataFrame column names must be unique after str()")
    elif isinstance(data, dict):
        columns = {str(k): _normalise_column(v) for k, v in data.items()}
    else:
        a = np.asarray(data)
        if a.dtype.names:                                   # structured array
            columns = {n: _normalise_column(a[n]) for n in a.dtype.names}
        else:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.ndim != 2:
                raise ValueError(f"expected 2-D data, got shape {a.shape}")
            names = _default_names(a.shape[1], column_names)
            columns = {name: _normalise_column(a[:, j]) for j, name in enumerate(names)}
    if not columns:
        raise ValueError("the data has no columns")
    n = len(next(iter(columns.values())))
    y, y_name = _resolve_side(target, columns, n, "target")
    g, g_name = _resolve_side(groups, columns, n, "groups")
    t, t_name = _resolve_side(time, columns, n, "time")
    declared = {}
    for name, typ in (column_types or {}).items():
        if name not in columns:
            raise KeyError(f"column_types names unknown column {name!r}")
        declared[name] = typ
    return Table(columns=columns, target=y, target_name=y_name, groups=g, groups_name=g_name,
                 time=t, time_name=t_name, declared_types=declared)

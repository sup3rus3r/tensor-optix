"""
Feature-type inference.

Wrong types are a common, silent source of bad models (a zip code read as a
quantity, an id read as a feature, "$1,200" read as a category). Each column
gets one inferred type, a confidence, and -- for every non-obvious call -- a
reversible DATA_TYPES finding so the user can overrule it. Types the user
declares via ``column_types`` are taken as given.

Types:
    numeric_continuous, numeric_discrete, boolean, categorical_low,
    categorical_high, datetime, identifier, text, embedded_number, constant,
    url, list, unknown
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from .data import Table, missing_mask

TYPES = (
    "numeric_continuous", "numeric_discrete", "boolean", "categorical_low", "categorical_high",
    "datetime", "identifier", "text", "embedded_number", "constant", "url", "list", "unknown",
)
NUMERIC_TYPES = ("numeric_continuous", "numeric_discrete", "boolean")
CATEGORICAL_TYPES = ("categorical_low", "categorical_high")
UNUSABLE_TYPES = ("identifier", "text", "constant", "url", "list", "unknown")

LOW_CARDINALITY = 32
DISCRETE_MAX_LEVELS = 20
SAMPLE_FOR_PATTERNS = 5000

_BOOL_STRINGS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
_NUM = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$")
_EMBEDDED = re.compile(
    r"^\s*[-+]?[$€£¥]?\s*[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s*(%|[a-zA-Z]{1,5})?\s*$"
    r"|^\s*[-+]?[$€£¥]?\s*[-+]?\d+(?:\.\d+)?\s*(%|[a-zA-Z]{1,5})?\s*$"
)
_UUID = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")
_HEX = re.compile(r"^[0-9a-fA-F]{16,}$")
_URL = re.compile(r"^(https?|ftp)://\S+$|^www\.\S+\.\S+$", re.IGNORECASE)
_DATE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^\d{4}/\d{2}/\d{2}$|^\d{1,2}/\d{1,2}/\d{4}$"
)
_LIST = re.compile(r"^\s*\[.*\]\s*$|^[^;|]+([;|][^;|]+){2,}$")


@dataclass(frozen=True)
class ColumnType:
    name: str
    type: str
    confidence: str                          # "low" | "moderate" | "high"
    declared: bool = False
    stored_as: str = ""                      # original storage kind: "float", "int", "object", ...
    n_unique: int = 0
    missing_fraction: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "confidence": self.confidence,
                "declared": self.declared, "stored_as": self.stored_as, "n_unique": self.n_unique,
                "missing_fraction": round(self.missing_fraction, 4), **self.details}


def _storage_kind(col: np.ndarray) -> str:
    return {"b": "bool", "i": "int", "u": "int", "f": "float", "M": "datetime", "O": "object"}.get(
        col.dtype.kind, str(col.dtype))


def _frac(pattern: re.Pattern, strings: List[str]) -> float:
    if not strings:
        return 0.0
    return sum(1 for s in strings if pattern.match(s)) / len(strings)


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_embedded_number(s: Any) -> float:
    """'$1,200' -> 1200.0, '12kg' -> 12.0, '15%' -> 15.0; NaN if not parseable."""
    if s is None:
        return float("nan")
    if isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, bool):
        return float(s)
    t = str(s).strip()
    t = re.sub(r"^[-+]?[$€£¥]\s*", lambda m: "-" if m.group(0).lstrip().startswith("-") else "", t)
    t = re.sub(r"\s*(%|[a-zA-Z]{1,5})\s*$", "", t)
    t = t.replace(",", "").replace(" ", "")
    v = _parse_float(t)
    return float("nan") if v is None else v


def _is_sequential(values: np.ndarray) -> bool:
    """Integers forming (nearly) a consecutive run in row order -- a row counter."""
    if len(values) < 3:
        return False
    d = np.diff(values.astype(np.float64))
    return bool(np.mean(d == 1) > 0.95 or np.mean(d == -1) > 0.95)


def _numeric_type(name: str, x: np.ndarray, n: int) -> Tuple[str, str, Dict[str, Any]]:
    x = x[np.isfinite(x)]
    uniq = np.unique(x)
    nu = len(uniq)
    details: Dict[str, Any] = {}
    integral = bool(np.all(np.floor(x) == x)) if len(x) else False
    if nu == 2 and set(uniq.tolist()) <= {0.0, 1.0}:
        return "boolean", "high", details
    if integral and nu >= 50 and nu / max(len(x), 1) > 0.95:
        if _is_sequential(x) or (uniq[-1] - uniq[0] + 1 == nu):
            details["reason"] = "unique, consecutive integers (a row counter or id)"
            return "identifier", "high", details
    if integral and nu <= DISCRETE_MAX_LEVELS:
        details["levels"] = int(nu)
        return "numeric_discrete", "moderate", details
    return "numeric_continuous", "high", details


def _object_type(name: str, col: np.ndarray, rng: np.random.Generator) -> Tuple[str, str, Dict[str, Any]]:
    present = [v for v in col if v is not None]
    if not present:
        return "unknown", "low", {"reason": "all values missing"}
    if all(isinstance(v, (bool, np.bool_)) for v in present):
        return "boolean", "high", {}
    if len(present) > SAMPLE_FOR_PATTERNS:
        pick = rng.choice(len(present), SAMPLE_FOR_PATTERNS, replace=False)
        sample = [present[i] for i in pick]
    else:
        sample = present
    strings = [str(v).strip() for v in sample]
    nu = len(set(str(v) for v in present))
    unique_ratio = nu / len(present)
    lower = {s.lower() for s in strings}
    if len(lower) <= 2 and lower <= _BOOL_STRINGS:
        return "boolean", "high", {"values": sorted(lower)}
    f_num = _frac(_NUM, strings)
    if f_num >= 0.98:
        nums = np.array([float(s) for s in strings if _NUM.match(s)])
        t, conf, d = _numeric_type(name, nums, len(nums))
        d["stored_as_text"] = True
        return t, "moderate" if conf == "high" else conf, d
    if _frac(_UUID, strings) >= 0.9 or _frac(_HEX, strings) >= 0.9:
        return "identifier", "high", {"reason": "UUID / hash-like values"}
    if _frac(_URL, strings) >= 0.9:
        return "url", "high", {}
    if _frac(_DATE, strings) >= 0.9:
        return "datetime", "moderate", {"stored_as_text": True}
    if _frac(_EMBEDDED, strings) >= 0.9:
        return "embedded_number", "moderate", {"example": strings[0][:40]}
    tokens = np.mean([len(s.split()) for s in strings])
    if tokens > 3 and unique_ratio > 0.5:
        return "text", "moderate", {"mean_tokens": round(float(tokens), 1)}
    if _frac(_LIST, strings) >= 0.9 and tokens <= 3:
        return "list", "low", {}
    if unique_ratio > 0.95 and len(present) >= 50:
        return "identifier", "moderate", {"reason": "almost every value is unique"}
    if nu <= LOW_CARDINALITY:
        return "categorical_low", "high", {"levels": nu}
    return "categorical_high", "high", {"levels": nu}


def infer_column_type(name: str, col: np.ndarray, rng: np.random.Generator,
                      declared: Optional[str] = None) -> ColumnType:
    miss = missing_mask(col)
    n = len(col)
    mfrac = float(miss.mean()) if n else 0.0
    present = col[~miss]
    try:
        n_unique = int(len(np.unique(present))) if present.dtype != object else len(set(map(str, present)))
    except TypeError:
        n_unique = len(set(map(str, present)))
    base = dict(name=name, stored_as=_storage_kind(col), n_unique=n_unique, missing_fraction=mfrac)
    if declared is not None:
        if declared not in TYPES:
            raise ValueError(f"unknown column type {declared!r} for {name!r}; known: {TYPES}")
        return ColumnType(type=declared, confidence="high", declared=True, **base)
    if len(present) == 0:
        return ColumnType(type="unknown", confidence="low", details={"reason": "all values missing"}, **base)
    # constant / near-constant (on present values)
    if n_unique <= 1:
        return ColumnType(type="constant", confidence="high", details={"reason": "single value"}, **base)
    if present.dtype != object:
        _, counts = np.unique(present, return_counts=True)
    else:
        _, counts = np.unique(np.array([str(v) for v in present], dtype=object), return_counts=True)
    top = float(counts.max() / len(present))
    if top >= 0.995 and len(present) >= 200:
        return ColumnType(type="constant", confidence="moderate",
                          details={"reason": f"one value in {top:.1%} of rows"}, **base)
    kind = col.dtype.kind
    if kind == "M":
        return ColumnType(type="datetime", confidence="high", **base)
    if kind == "b":
        return ColumnType(type="boolean", confidence="high", **base)
    if kind in "iuf":
        t, conf, d = _numeric_type(name, present.astype(np.float64), n)
        return ColumnType(type=t, confidence=conf, details=d, **base)
    t, conf, d = _object_type(name, col, rng)
    return ColumnType(type=t, confidence=conf, details=d, **base)


def infer_types(table: Table, rng: np.random.Generator) -> Dict[str, ColumnType]:
    return {name: infer_column_type(name, col, rng, table.declared_types.get(name))
            for name, col in table.columns.items()}


def type_findings(types: Dict[str, ColumnType]) -> List[Finding]:
    """One finding per non-obvious inference; declared types produce none."""
    out: List[Finding] = []
    for name, ct in types.items():
        if ct.declared:
            continue
        ev = (Evidence(test="type_inference", n=None, details={
            "inferred": ct.type, "stored_as": ct.stored_as, "n_unique": ct.n_unique,
            "missing_fraction": round(ct.missing_fraction, 4), **ct.details}),)
        fid = f"data_types.{ct.type}:col={name}"
        if ct.type == "identifier":
            out.append(Finding(
                id=fid, category=Category.DATA_TYPES, severity=Severity.ADVISORY,
                statement=f"'{name}' looks like an identifier ({ct.details.get('reason', 'unique values')}); "
                          "it is excluded from modelling.",
                evidence=ev, confidence=ct.confidence, confidence_basis="rule-based type inference",
                suggested_action=Action("drop_column", {"column": name}), default_applied=True))
        elif ct.type == "constant":
            out.append(Finding(
                id=fid, category=Category.QUALITY, severity=Severity.ADVISORY,
                statement=f"'{name}' is (nearly) constant ({ct.details.get('reason', '')}); it carries "
                          "no signal and is excluded.",
                evidence=ev, confidence=ct.confidence, confidence_basis="value frequency",
                suggested_action=Action("drop_column", {"column": name}), default_applied=True))
        elif ct.type in ("text", "url", "list", "unknown"):
            what = {"text": "free text", "url": "URLs", "list": "list-like values",
                    "unknown": "values that could not be typed"}[ct.type]
            out.append(Finding(
                id=fid, category=Category.DATA_TYPES, severity=Severity.ADVISORY,
                statement=f"'{name}' holds {what}; this version of discovery does not model it, so it "
                          "is excluded (consider deriving features from it yourself).",
                evidence=ev, confidence=ct.confidence, confidence_basis="rule-based type inference",
                suggested_action=Action("drop_column", {"column": name}), default_applied=True))
        elif ct.type == "embedded_number":
            out.append(Finding(
                id=fid, category=Category.DATA_TYPES, severity=Severity.ADVISORY,
                statement=f"'{name}' stores numbers inside text (e.g. '{ct.details.get('example', '')}'); "
                          "it is parsed into a number.",
                evidence=ev, confidence=ct.confidence, confidence_basis="pattern match on values",
                suggested_action=Action("set_column_type", {"column": name, "type": "embedded_number"}),
                default_applied=True))
        elif ct.details.get("stored_as_text"):
            out.append(Finding(
                id=f"data_types.stored_as_text:col={name}", category=Category.DATA_TYPES,
                severity=Severity.INFO,
                statement=f"'{name}' is stored as text but holds {ct.type.replace('_', ' ')} values; it is "
                          "treated as such.",
                evidence=ev, confidence=ct.confidence, confidence_basis="pattern match on values",
                suggested_action=Action("set_column_type", {"column": name, "type": ct.type}),
                default_applied=True))
        elif ct.type == "numeric_discrete":
            out.append(Finding(
                id=fid, category=Category.DATA_TYPES, severity=Severity.INFO,
                statement=f"'{name}' has only {ct.details.get('levels')} integer levels; it is treated as an "
                          "ordered number. If the codes are labels with no order, declare it categorical.",
                evidence=ev, confidence=ct.confidence, confidence_basis="level count",
                suggested_action=Action("set_column_type", {"column": name, "type": "categorical_low"}),
                default_applied=False))
    return out

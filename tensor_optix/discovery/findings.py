"""
Findings: what discovery learned about the data / environment, with the
evidence behind it and the action it suggests.

A finding never changes anything by itself. Its suggested ``Action`` is only
applied by ``DiscoveryReport.build()`` once the user has ACCEPTED it, which is
why every finding has a stable ``id``: accept / reject decisions survive a
re-run of ``discover()`` on the same data (``decisions=report.decisions()``).

Everything here is JSON-native so a report can be saved, shared as HTML and
loaded back without loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Severity(str, Enum):
    """How much a finding matters. Ordered: INFO < ADVISORY < WARNING < CRITICAL."""

    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.INFO: 0, Severity.ADVISORY: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}


class Category(str, Enum):
    DATA_TYPES = "data_types"
    QUALITY = "quality"
    STRUCTURE = "structure"
    LEAKAGE = "leakage"
    TARGET = "target"
    SHIFT = "shift"
    FAMILY = "family"
    BUDGET = "budget"
    LIMITATION = "limitation"
    FEATURES = "features"
    ENV = "env"


class DecisionState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


CONFIDENCE_LEVELS = ("low", "moderate", "high")


def _clean_number(x: Any) -> Any:
    """Floats must survive JSON: NaN / inf become None."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int,)):
        return int(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return f if math.isfinite(f) else None


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays / tuples to JSON-native types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (int, float)):
        return jsonable(obj.tolist())
    return _clean_number(obj)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """One statistical result supporting a finding."""

    test: str                                   # e.g. "permutation_auc", "split_gap"
    statistic: Optional[float] = None
    n: Optional[int] = None
    null_value: Optional[float] = None
    p_value: Optional[float] = None
    interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "test": self.test, "statistic": self.statistic, "n": self.n,
            "null_value": self.null_value, "p_value": self.p_value,
            "interval": list(self.interval) if self.interval is not None else None,
            "effect_size": self.effect_size, "details": self.details, "seed": self.seed,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        iv = d.get("interval")
        return cls(
            test=d["test"], statistic=d.get("statistic"), n=d.get("n"),
            null_value=d.get("null_value"), p_value=d.get("p_value"),
            interval=(iv[0], iv[1]) if iv is not None else None,
            effect_size=d.get("effect_size"), details=dict(d.get("details") or {}),
            seed=d.get("seed"),
        )


# ---------------------------------------------------------------------------
# Actions: a closed, versioned grammar
# ---------------------------------------------------------------------------

ACTION_GRAMMAR_VERSION = 2

# kind -> grammar version that introduced it (1 when absent)
ACTION_KIND_SINCE: Dict[str, int] = {"add_feature_family": 2}

# kind -> required parameter names. Anything else is rejected, never ignored.
ACTION_KINDS: Dict[str, Tuple[str, ...]] = {
    # ML
    "drop_column": ("column",),
    "set_column_type": ("column", "type"),
    "use_split": ("kind",),                     # optional: groups, time, gap, n_splits
    "impute": ("column", "strategy"),
    "encode": ("column", "encoding"),
    "scale": ("column", "method"),
    "transform_target": ("transform",),
    "class_weight": ("scheme",),
    "dedupe": ("strategy",),
    "set_metric": ("metric",),
    "use_family": ("family",),
    "add_feature_family": ("family",),          # params: a feature_gen.FeatureSpec as a dict
    # RL (phase 2)
    "wrap_obs_normalize": (),
    "wrap_action_rescale": ("low", "high"),
    "use_recurrent": (),
    "add_rnd": (),
    "frame_stack": ("k",),
}


@dataclass(frozen=True)
class Action:
    """A concrete, reversible change discovery suggests."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    grammar_version: Optional[int] = None       # None -> the version that introduced ``kind``

    def __post_init__(self):
        if self.grammar_version is not None and self.grammar_version > ACTION_GRAMMAR_VERSION:
            raise ValueError(
                f"action grammar version {self.grammar_version} is newer than this "
                f"tensor-optix understands ({ACTION_GRAMMAR_VERSION}); upgrade tensor-optix"
            )
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}; known: {sorted(ACTION_KINDS)}")
        missing = [p for p in ACTION_KINDS[self.kind] if p not in self.params]
        if missing:
            raise ValueError(f"action {self.kind!r} is missing parameter(s) {missing}")
        if self.grammar_version is None:
            # the lowest grammar version that can read this action, so reports stay
            # readable by older tensor-optix unless they use a newer kind
            object.__setattr__(self, "grammar_version", ACTION_KIND_SINCE.get(self.kind, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "params": jsonable(self.params),
                "grammar_version": self.grammar_version}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(kind=d["kind"], params=dict(d.get("params") or {}),
                   grammar_version=int(d.get("grammar_version", 1)))


# ---------------------------------------------------------------------------
# Decision (the only mutable part)
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    state: DecisionState = DecisionState.PENDING
    note: Optional[str] = None
    at: Optional[str] = None                    # ISO timestamp of the decision

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state.value, "note": self.note, "at": self.at}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Decision":
        if not d:
            return cls()
        return cls(state=DecisionState(d.get("state", "pending")), note=d.get("note"), at=d.get("at"))


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """
    Args:
        id: stable across re-runs on the same data, e.g.
            ``"leakage.single_feature:col=refund_flag"``.
        statement: one plain-language sentence for the report.
        confidence: "low" | "moderate" | "high", derived from the evidence by
            the detector's documented rule (``confidence_basis`` says which).
        default_applied: whether the configuration discovery used for its own
            experiments already applied ``suggested_action`` (e.g. dropping a
            suspected leak). If the user later rejects such a finding, the
            evidence behind the recommendation was gathered under a different
            configuration, and ``build()`` says so.
    """

    id: str
    category: Category
    severity: Severity
    statement: str
    evidence: Tuple[Evidence, ...] = ()
    confidence: str = "moderate"
    confidence_basis: str = ""
    suggested_action: Optional[Action] = None
    default_applied: bool = False

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError("a finding needs a non-empty string id")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence must be one of {CONFIDENCE_LEVELS}, got {self.confidence!r}")
        object.__setattr__(self, "category", Category(self.category))
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "category": self.category.value, "severity": self.severity.value,
            "statement": self.statement, "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence, "confidence_basis": self.confidence_basis,
            "suggested_action": self.suggested_action.to_dict() if self.suggested_action else None,
            "default_applied": bool(self.default_applied),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Finding":
        act = d.get("suggested_action")
        return cls(
            id=d["id"], category=Category(d["category"]), severity=Severity(d["severity"]),
            statement=d["statement"],
            evidence=tuple(Evidence.from_dict(e) for e in d.get("evidence") or ()),
            confidence=d.get("confidence", "moderate"), confidence_basis=d.get("confidence_basis", ""),
            suggested_action=Action.from_dict(act) if act else None,
            default_applied=bool(d.get("default_applied", False)),
        )

    def with_(self, **changes) -> "Finding":
        return replace(self, **changes)


def confidence_from_p(p_value: Optional[float], effect_ok: bool = True) -> str:
    """
    Default confidence rule for permutation / hypothesis-test based findings:
    high when p < 0.001 and the effect size passed its threshold, moderate when
    p < 0.01, low otherwise. Detectors that use another rule say so in
    ``confidence_basis``.
    """
    if p_value is None:
        return "low"
    if p_value < 0.001 and effect_ok:
        return "high"
    if p_value < 0.01 and effect_ok:
        return "moderate"
    return "low"

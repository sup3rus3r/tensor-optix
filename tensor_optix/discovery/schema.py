"""
The discovery report data model.

``DiscoveryReport.to_dict()`` is the single source of truth: the HTML report is
rendered from it, the JSON file is it, and ``from_dict`` / ``from_json`` /
``from_html`` rebuild an identical report. Only the per-finding ``Decision``s
are mutable (``accept`` / ``reject``); everything discovery measured is frozen.

The schema is versioned (``DISCOVERY_SCHEMA_VERSION``). Reports written by an
older tensor-optix are migrated forward by ``_MIGRATIONS``; reports written by
a newer one are refused with a clear message instead of being misread.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .findings import (
    Category, Decision, DecisionState, Finding, Severity, jsonable,
)

DISCOVERY_SCHEMA_VERSION = 1

# from_version -> function(dict) -> dict at from_version + 1
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

VERDICTS = ("better", "worse", "equivalent", "undetermined")
RECOMMENDATION_VERDICTS = ("clear_winner", "tie_simplest_chosen", "undetermined", "abstain")
EXPERIMENT_STATUSES = ("ok", "failed", "skipped_budget", "unavailable_dependency")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """
    What discovery measured about the input, before any model was fit.

    ``sections`` holds the domain-specific parts as JSON-native dicts, e.g. for
    ML: ``columns`` (list of per-column profiles), ``target``, ``structure``,
    ``quality``, ``shift``; for RL: ``spaces``, ``rollout``, ``dynamics``.
    """

    domain: str                                 # "ml" | "rl"
    n_total: Optional[int] = None               # rows (ML) or steps probed (RL)
    n_sampled: Optional[int] = None
    sampling: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None           # hashed schema + sample, no raw values

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "domain": self.domain, "n_total": self.n_total, "n_sampled": self.n_sampled,
            "sampling": self.sampling, "sections": self.sections, "fingerprint": self.fingerprint,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Profile":
        return cls(domain=d["domain"], n_total=d.get("n_total"), n_sampled=d.get("n_sampled"),
                   sampling=dict(d.get("sampling") or {}), sections=dict(d.get("sections") or {}),
                   fingerprint=d.get("fingerprint"))


# ---------------------------------------------------------------------------
# Experiments and comparisons
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentResult:
    """One family / configuration evaluated on a fixed set of folds (or seeds)."""

    family: str
    config: Dict[str, Any] = field(default_factory=dict)
    stage: str = "trial"                        # "landmark" | "trial" | "confirm"
    metric: str = ""
    higher_is_better: bool = True
    fold_ids: Tuple[str, ...] = ()              # shared ids -> paired across families
    scores: Tuple[Optional[float], ...] = ()    # one per fold id
    secondary: Dict[str, Tuple[Optional[float], ...]] = field(default_factory=dict)
    interval: Optional[Tuple[float, float]] = None
    mean: Optional[float] = None
    fit_time_s: float = 0.0
    stop_reasons: Tuple[str, ...] = ()
    status: str = "ok"
    error: Optional[str] = None

    def __post_init__(self):
        if self.status not in EXPERIMENT_STATUSES:
            raise ValueError(f"status must be one of {EXPERIMENT_STATUSES}, got {self.status!r}")
        if len(self.fold_ids) != len(self.scores):
            raise ValueError("fold_ids and scores must have the same length")

    @property
    def key(self) -> str:
        """Unique id of this experiment inside a report."""
        cfg = json.dumps(jsonable(self.config), sort_keys=True)
        return f"{self.stage}:{self.family}:{cfg}"

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "family": self.family, "config": self.config, "stage": self.stage,
            "metric": self.metric, "higher_is_better": self.higher_is_better,
            "fold_ids": list(self.fold_ids), "scores": list(self.scores),
            "secondary": {k: list(v) for k, v in self.secondary.items()},
            "interval": list(self.interval) if self.interval else None, "mean": self.mean,
            "fit_time_s": self.fit_time_s, "stop_reasons": list(self.stop_reasons),
            "status": self.status, "error": self.error,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        iv = d.get("interval")
        return cls(
            family=d["family"], config=dict(d.get("config") or {}), stage=d.get("stage", "trial"),
            metric=d.get("metric", ""), higher_is_better=bool(d.get("higher_is_better", True)),
            fold_ids=tuple(d.get("fold_ids") or ()), scores=tuple(d.get("scores") or ()),
            secondary={k: tuple(v) for k, v in (d.get("secondary") or {}).items()},
            interval=(iv[0], iv[1]) if iv else None, mean=d.get("mean"),
            fit_time_s=float(d.get("fit_time_s") or 0.0),
            stop_reasons=tuple(d.get("stop_reasons") or ()),
            status=d.get("status", "ok"), error=d.get("error"),
        )


@dataclass(frozen=True)
class Comparison:
    """Paired comparison of experiment ``a`` against ``b`` (delta = a - b, oriented
    so that a positive delta always means ``a`` is better)."""

    a: str                                      # ExperimentResult.key
    b: str
    n_pairs: int
    delta_mean: Optional[float]
    delta_interval: Optional[Tuple[float, float]]
    rope: float
    verdict: str
    p_value: Optional[float] = None
    p_holm: Optional[float] = None
    method: str = ""

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "a": self.a, "b": self.b, "n_pairs": self.n_pairs, "delta_mean": self.delta_mean,
            "delta_interval": list(self.delta_interval) if self.delta_interval else None,
            "rope": self.rope, "verdict": self.verdict, "p_value": self.p_value,
            "p_holm": self.p_holm, "method": self.method,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Comparison":
        iv = d.get("delta_interval")
        return cls(a=d["a"], b=d["b"], n_pairs=int(d.get("n_pairs") or 0),
                   delta_mean=d.get("delta_mean"), delta_interval=(iv[0], iv[1]) if iv else None,
                   rope=float(d.get("rope") or 0.0), verdict=d["verdict"],
                   p_value=d.get("p_value"), p_holm=d.get("p_holm"), method=d.get("method", ""))


# ---------------------------------------------------------------------------
# Recommendation and plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    family: Optional[str]
    config: Dict[str, Any] = field(default_factory=dict)
    verdict: str = "undetermined"
    rationale: Tuple[str, ...] = ()             # plain-language bullets (may cite ids)
    expected_score: Optional[float] = None      # on the confirmation holdout
    expected_interval: Optional[Tuple[float, float]] = None
    metric: str = ""
    alternatives: Tuple[str, ...] = ()          # families still in contention
    conditions: Tuple[str, ...] = ()            # e.g. "valid only if column X is dropped"

    def __post_init__(self):
        if self.verdict not in RECOMMENDATION_VERDICTS:
            raise ValueError(f"verdict must be one of {RECOMMENDATION_VERDICTS}, got {self.verdict!r}")

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "family": self.family, "config": self.config, "verdict": self.verdict,
            "rationale": list(self.rationale), "expected_score": self.expected_score,
            "expected_interval": list(self.expected_interval) if self.expected_interval else None,
            "metric": self.metric, "alternatives": list(self.alternatives),
            "conditions": list(self.conditions),
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Recommendation":
        iv = d.get("expected_interval")
        return cls(family=d.get("family"), config=dict(d.get("config") or {}),
                   verdict=d.get("verdict", "undetermined"), rationale=tuple(d.get("rationale") or ()),
                   expected_score=d.get("expected_score"),
                   expected_interval=(iv[0], iv[1]) if iv else None, metric=d.get("metric", ""),
                   alternatives=tuple(d.get("alternatives") or ()),
                   conditions=tuple(d.get("conditions") or ()))


@dataclass(frozen=True)
class PlanStep:
    kind: str                                   # "split" | "preprocess" | "train" | "monitor" | ...
    description: str
    finding_refs: Tuple[str, ...] = ()
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "description": self.description,
                "finding_refs": list(self.finding_refs), "code": self.code}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanStep":
        return cls(kind=d["kind"], description=d["description"],
                   finding_refs=tuple(d.get("finding_refs") or ()), code=d.get("code"))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class UnknownFindingError(KeyError):
    pass


@dataclass
class DiscoveryReport:
    domain: str
    profile: Profile
    findings: List[Finding] = field(default_factory=list)
    experiments: List[ExperimentResult] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    plan: List[PlanStep] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    input_summary: Dict[str, Any] = field(default_factory=dict)
    budget_requested: Dict[str, Any] = field(default_factory=dict)
    budget_used: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)
    ledger_refs: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    tensor_optix_version: str = ""
    schema_version: int = DISCOVERY_SCHEMA_VERSION
    _decisions: Dict[str, Decision] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        ids = [f.id for f in self.findings]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate finding ids: {sorted(dup)}")
        for f in self.findings:
            self._decisions.setdefault(f.id, Decision())
        unknown = set(self._decisions) - set(ids)
        if unknown:
            raise ValueError(f"decisions for unknown finding ids: {sorted(unknown)}")
        if not self.tensor_optix_version:
            # importlib.metadata, not ``import tensor_optix``: the top-level
            # package pulls in heavy optional frameworks.
            self.tensor_optix_version = _package_version()

    # ---- findings and decisions -------------------------------------------

    def finding(self, finding_id: str) -> Finding:
        for f in self.findings:
            if f.id == finding_id:
                return f
        raise UnknownFindingError(finding_id)

    def decision(self, finding_id: str) -> Decision:
        self.finding(finding_id)
        return self._decisions[finding_id]

    def _decide(self, finding_id: str, state: DecisionState, note: Optional[str]) -> None:
        self.finding(finding_id)                 # raises for unknown ids
        self._decisions[finding_id] = Decision(state=state, note=note, at=_now_iso())

    def accept(self, finding_id: str, note: Optional[str] = None) -> None:
        """Accept one finding: ``build()`` will apply its suggested action."""
        self._decide(finding_id, DecisionState.ACCEPTED, note)

    def reject(self, finding_id: str, note: Optional[str] = None) -> None:
        """Reject one finding: ``build()`` will not apply its suggested action."""
        self._decide(finding_id, DecisionState.REJECTED, note)

    def accept_where(self, *, category: Optional[str] = None, min_severity: Optional[str] = None,
                     note: Optional[str] = None) -> List[str]:
        """
        Accept every PENDING finding matching the filters (at least one filter
        is required -- there is deliberately no ``accept_all``). Returns the
        accepted ids.
        """
        if category is None and min_severity is None:
            raise ValueError("accept_where() needs category= and/or min_severity=")
        cat = Category(category) if category is not None else None
        sev = Severity(min_severity) if min_severity is not None else None
        done = []
        for f in self.findings:
            if self._decisions[f.id].state != DecisionState.PENDING:
                continue
            if cat is not None and f.category != cat:
                continue
            if sev is not None and f.severity.rank < sev.rank:
                continue
            self._decide(f.id, DecisionState.ACCEPTED, note)
            done.append(f.id)
        return done

    def decisions(self) -> Dict[str, Decision]:
        """Copy of all decisions, to pass back to ``discover(decisions=...)``."""
        return {k: Decision(v.state, v.note, v.at) for k, v in self._decisions.items()}

    def pending(self, min_severity: str = "info") -> List[Finding]:
        sev = Severity(min_severity)
        return [f for f in self.findings
                if self._decisions[f.id].state == DecisionState.PENDING and f.severity.rank >= sev.rank]

    def findings_by(self, category: str) -> List[Finding]:
        cat = Category(category)
        return [f for f in self.findings if f.category == cat]

    def apply_decisions(self, decisions: Dict[str, Decision]) -> List[str]:
        """Carry decisions over from an earlier run; ids that no longer exist are
        returned (not silently dropped) so the caller can tell the user."""
        stale = []
        for fid, d in decisions.items():
            if any(f.id == fid for f in self.findings):
                self._decisions[fid] = Decision(d.state, d.note, d.at)
            else:
                stale.append(fid)
        return stale

    # ---- experiments -------------------------------------------------------

    def experiment(self, key: str) -> ExperimentResult:
        for e in self.experiments:
            if e.key == key:
                return e
        raise KeyError(key)

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_optix_version": self.tensor_optix_version,
            "created_at": self.created_at,
            "domain": self.domain,
            "seed": self.seed,
            "input_summary": jsonable(self.input_summary),
            "budget_requested": jsonable(self.budget_requested),
            "budget_used": jsonable(self.budget_used),
            "profile": self.profile.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "decisions": {k: v.to_dict() for k, v in self._decisions.items()},
            "experiments": [e.to_dict() for e in self.experiments],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "plan": [p.to_dict() for p in self.plan],
            "limitations": list(self.limitations),
            "provenance": jsonable(self.provenance),
            "ledger_refs": jsonable(self.ledger_refs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryReport":
        d = _migrate(dict(d))
        rec = d.get("recommendation")
        return cls(
            domain=d["domain"], profile=Profile.from_dict(d["profile"]),
            findings=[Finding.from_dict(f) for f in d.get("findings") or ()],
            experiments=[ExperimentResult.from_dict(e) for e in d.get("experiments") or ()],
            comparisons=[Comparison.from_dict(c) for c in d.get("comparisons") or ()],
            recommendation=Recommendation.from_dict(rec) if rec else None,
            plan=[PlanStep.from_dict(p) for p in d.get("plan") or ()],
            limitations=list(d.get("limitations") or ()),
            input_summary=dict(d.get("input_summary") or {}),
            budget_requested=dict(d.get("budget_requested") or {}),
            budget_used=dict(d.get("budget_used") or {}),
            seed=int(d.get("seed") or 0), provenance=dict(d.get("provenance") or {}),
            ledger_refs=dict(d.get("ledger_refs") or {}),
            created_at=d.get("created_at") or _now_iso(),
            tensor_optix_version=d.get("tensor_optix_version") or "",
            schema_version=int(d["schema_version"]),
            _decisions={k: Decision.from_dict(v) for k, v in (d.get("decisions") or {}).items()},
        )

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=False, allow_nan=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text: str) -> "DiscoveryReport":
        text = path_or_text
        if not path_or_text.lstrip().startswith("{"):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def to_html(self, path: Optional[str] = None) -> str:
        from .report.render import render_html
        html = render_html(self.to_dict())
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    @classmethod
    def from_html(cls, path_or_text: str) -> "DiscoveryReport":
        from .report.io import extract_json
        text = path_or_text
        if "<html" not in path_or_text[:2000].lower():
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(extract_json(text))

    def summary(self) -> str:
        from .report.render import render_text
        return render_text(self.to_dict())

    def __str__(self) -> str:                    # pragma: no cover - convenience
        return self.summary()

    # ---- build -------------------------------------------------------------

    def build(self, *args, **kwargs):
        """Build the recommended setup from the ACCEPTED findings (opt-in)."""
        from .build import build_solution
        return build_solution(self, *args, **kwargs)


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("tensor-optix")
    except Exception:
        return "unknown"


def _migrate(d: Dict[str, Any]) -> Dict[str, Any]:
    v = int(d.get("schema_version") or 0)
    if v < 1:
        raise ValueError("not a discovery report (missing schema_version)")
    if v > DISCOVERY_SCHEMA_VERSION:
        raise ValueError(
            f"this discovery report uses schema version {v}, newer than this tensor-optix "
            f"understands ({DISCOVERY_SCHEMA_VERSION}); upgrade tensor-optix to read it"
        )
    while v < DISCOVERY_SCHEMA_VERSION:
        d = _MIGRATIONS[v](d)
        v += 1
        d["schema_version"] = v
    return d

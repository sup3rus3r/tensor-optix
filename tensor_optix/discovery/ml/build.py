"""
``report.build()`` for datasets: turn the ACCEPTED findings of an ML discovery
report into an ``MLSolution`` -- evaluation split, feature types, excluded
columns, metric and model family -- that trains on the user's full data.

Contract:
  * only findings whose decision is ACCEPTED change anything; a finding that
    discovery applied by default in its own experiments but that is still
    pending (or rejected) is *not* applied, and is listed in
    ``solution.evidence_mismatch`` (the report's numbers assumed it);
  * build refuses while any CRITICAL finding is pending, unless
    ``acknowledge_pending=True``;
  * a report that abstained needs an explicit ``family=``.

Preprocessing is the same fold-safe ``ColumnPipeline`` discovery used, so a
solution's ``cross_validate`` reproduces what the report measured when the
same decisions are in force.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..build import PendingCriticalFindings, register_builder
from ..findings import DecisionState, Severity
from ..rng import int_seed_for
from ..schema import DiscoveryReport, ExperimentResult
from .data import Table, to_table
from .evaluate import evaluate_family
from .families.base import CLASSIFICATION, get_family
from .feature_gen import PREFIX, FeatureSpec, generate
from .preprocess import ColumnPipeline
from .splits import (
    ForwardChaining, GroupKFold, KFold, RepeatedSplitter, StratifiedGroupKFold, StratifiedKFold,
)
from .structure import _group_codes, _strat, _time_numeric
from .target import encode_classes, profile_target
from .types import TYPES, ColumnType, infer_column_type

SUPPORTED_ACTIONS = ("drop_column", "set_column_type", "use_split", "dedupe", "set_metric", "use_family",
                     "add_feature_family", "scale")


class UnsupportedAction(NotImplementedError):
    pass


@dataclass
class SplitSpec:
    kind: str = "random"                         # "random" | "group" | "forward"
    groups: Optional[str] = None                 # column name, or "<groups>" (pass groups= to the solution)
    time: Optional[str] = None                   # column name, "<time>" or "<row order>"
    gap: int = 0
    group_duplicates: bool = False
    n_splits: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class MLSolution:
    """A ready-to-train setup built from a discovery report's accepted findings."""

    family: str
    config: Dict[str, Any]
    task: str
    metric: str
    column_types: Dict[str, str]
    drop: List[str]
    split: SplitSpec
    target: Optional[str]
    applied: List[str]
    evidence_mismatch: List[str] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)      # accepted feature families (FeatureSpec dicts)
    scaling: Dict[str, str] = field(default_factory=dict)             # column -> "robust" | "quantile"
    seed: int = 0
    classes_: Optional[tuple] = None
    _pipe: Any = field(default=None, repr=False)
    _est: Any = field(default=None, repr=False)
    _features: Optional[List[str]] = field(default=None, repr=False)

    # ---- data handling ------------------------------------------------------

    def _table(self, data, target=None, groups=None, time=None) -> Table:
        target = self.target if target is None and self.target is not None and _has_col(data, self.target) \
            else target
        g = groups if groups is not None else (self.split.groups if self.split.groups and
                                               not self.split.groups.startswith("<") and
                                               _has_col(data, self.split.groups) else None)
        # an explicit time= array is used as given; a time *column* stays a feature (datetime parts),
        # exactly as in discovery's own experiments, and is read from the table when splitting
        return to_table(data, target, groups=g, time=time)

    def _types(self, table: Table) -> Dict[str, ColumnType]:
        rng = np.random.default_rng(int_seed_for(self.seed, "solution_types"))
        out: Dict[str, ColumnType] = {}
        for name, col in table.columns.items():
            if name in self.drop:
                continue
            declared = self.column_types.get(name) or ("numeric_continuous" if name.startswith(PREFIX) else None)
            out[name] = infer_column_type(name, col, rng, declared)
        if self.split.kind == "group" and self.split.groups in out:
            out.pop(self.split.groups)                       # the group id is not a feature
        return out

    # ---- engineered feature families ---------------------------------------

    @staticmethod
    def _side(table: Table, name: Optional[str], kind: str) -> Optional[np.ndarray]:
        if not name:
            return None
        arr, arr_name = (table.time, table.time_name) if kind == "time" else (table.groups, table.groups_name)
        if arr is not None and (name.startswith("<") or arr_name == name):
            return arr
        if name in table.columns:
            return table.columns[name]
        raise ValueError(f"an accepted feature family needs the {kind} column {name!r}; include it or pass {kind}=")

    def _augment(self, table: Table, y_num: Optional[np.ndarray]) -> Table:
        if not self.features:
            return table
        cols = dict(table.columns)
        for d in self.features:
            spec = FeatureSpec.from_dict(d)
            t = spec.params.get("time")
            order = None
            if t == "<row order>":
                order = np.arange(table.n_rows)
            elif t:
                order = np.argsort(_time_numeric(self._side(table, t, "time")), kind="stable")
            g = self._side(table, spec.params.get("groups"), "groups")
            groups = None if g is None else _group_codes(g)
            if spec.needs_target and y_num is None:
                raise ValueError(f"the accepted feature family {spec.family!r} uses past values of the target: "
                                 "pass target= with the known history (NaN for the rows to predict)")
            cols.update(generate(spec, table, y_num if spec.needs_target else None, order, groups))
        return Table(columns=cols, target=table.target, target_name=table.target_name, groups=table.groups,
                     groups_name=table.groups_name, time=table.time, time_name=table.time_name,
                     declared_types=dict(table.declared_types))

    def _target_numeric(self, y_raw: np.ndarray) -> np.ndarray:
        from .data import missing_mask
        miss = missing_mask(y_raw)
        out = np.full(len(y_raw), np.nan)
        if self.task in CLASSIFICATION:
            index = {c: i for i, c in enumerate(self.classes_ or ())}
            for i, v in enumerate(y_raw.tolist()):
                if not miss[i] and v in index:
                    out[i] = index[v]
        else:
            ok = ~miss
            out[ok] = np.asarray(y_raw[ok], dtype=np.float64)
        return out

    def _encode(self, y_raw: np.ndarray):
        prof, y, keep, _ = profile_target(y_raw, self.task, self.metric)
        if self.task in CLASSIFICATION:
            _, classes = encode_classes(y_raw[keep])
            self.classes_ = classes
        return y, keep

    # ---- public API ---------------------------------------------------------

    def _options(self):
        opts = get_family(self.family).preprocessing
        if not self.scaling:
            return opts
        return replace(opts,
                       robust_columns=tuple(c for c, m in self.scaling.items() if m == "robust"),
                       quantile_columns=tuple(c for c, m in self.scaling.items() if m == "quantile"))

    def splitter(self, n_splits: Optional[int] = None):
        n_splits = n_splits or self.split.n_splits
        if self.split.kind == "forward":
            return ForwardChaining(n_splits, gap=self.split.gap, min_train_frac=0.3)
        if self.split.kind == "group" or self.split.group_duplicates:
            return (StratifiedGroupKFold(n_splits, self.seed) if self.task in CLASSIFICATION
                    else GroupKFold(n_splits, self.seed))
        return StratifiedKFold(n_splits, self.seed) if self.task in CLASSIFICATION else KFold(n_splits, self.seed)

    def _split_inputs(self, table: Table, keep: np.ndarray):
        groups = None
        if self.split.kind == "group":
            if table.groups is None:
                raise ValueError(f"this solution splits by groups ({self.split.groups!r}); pass groups=")
            groups = _group_codes(table.groups[keep])
        elif self.split.group_duplicates:
            from .quality import row_group_ids
            groups = row_group_ids(table.take(np.flatnonzero(keep)), [c for c in table.names if c not in self.drop])
        time = None
        if self.split.kind == "forward" and self.split.time != "<row order>":
            if table.time is not None:
                raw = table.time
            elif self.split.time in table.columns:
                raw = table.columns[self.split.time]
            else:
                raise ValueError(f"this solution splits by time ({self.split.time!r}); pass time=")
            time = _time_numeric(raw[keep])
        return groups, time

    def cross_validate(self, data, target=None, *, groups=None, time=None, repeats: int = 1,
                       n_splits: Optional[int] = None) -> ExperimentResult:
        """Cross-validate this setup on ``data`` with its split (fold-safe preprocessing)."""
        table = self._table(data, target, groups, time)
        if table.target is None:
            raise ValueError("cross_validate needs a target")
        y, keep = self._encode(table.target)
        sub = self._augment(table.take(np.flatnonzero(keep)), np.asarray(y, dtype=np.float64))
        g, t = self._split_inputs(table, keep)
        folds = RepeatedSplitter(self.splitter(n_splits), repeats).folds(len(y), y=_strat(y, self.task),
                                                                        groups=g, time=t)
        return evaluate_family(sub, self._types(sub), y, folds, get_family(self.family), self.config,
                               task=self.task, n_classes=len(self.classes_ or ()), metric=self.metric,
                               seed=self.seed, stage="confirm", preprocessing=self._options())

    def fit(self, data, target=None, *, groups=None, time=None) -> "MLSolution":
        table = self._table(data, target, groups, time)
        if table.target is None:
            raise ValueError("fit needs a target: pass target= (name, index or array)")
        y, keep = self._encode(table.target)
        sub = self._augment(table.take(np.flatnonzero(keep)), np.asarray(y, dtype=np.float64))
        types = self._types(sub)
        fam = get_family(self.family)
        k = len(self.classes_ or ())
        opts = self._options()
        self._pipe = ColumnPipeline(types, opts, task=self.task, n_classes=k,
                                    seed=int_seed_for(self.seed, "solution_pipeline"))
        rows = np.arange(sub.n_rows)
        X = self._pipe.fit_transform(sub, rows, y)
        self._est = fam.make(self.config, int_seed_for(self.seed, "solution_estimator"), self.task, k)
        self._est.fit(X, y)
        self._features = list(types)
        return self

    def _X(self, data, target=None) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("call fit() first")
        if isinstance(data, Table):
            table = data
        else:
            tgt = target if target is not None else (
                self.target if self.target is not None and _has_col(data, self.target) else None)
            table = to_table(data, tgt)
        y_num = None if table.target is None else self._target_numeric(table.target)
        table = self._augment(table, y_num)
        missing = [c for c in self._features if c not in table.columns]
        if missing:
            raise KeyError(f"columns missing from the data: {missing}")
        cols = {c: table.columns[c] for c in self._features}
        t = Table(columns=cols)
        return self._pipe.transform(t, np.arange(t.n_rows))

    def predict_proba(self, data, target=None) -> np.ndarray:
        """``target``: only needed when an accepted feature family uses past target values."""
        if self.task not in CLASSIFICATION:
            raise ValueError("predict_proba is for classification")
        return self._est.predict_proba(self._X(data, target))

    def predict(self, data, target=None) -> np.ndarray:
        """``target``: only needed when an accepted feature family uses past target values
        (the known history, NaN for the rows to predict)."""
        X = self._X(data, target)
        if self.task in CLASSIFICATION:
            idx = np.argmax(self._est.predict_proba(X), axis=1)
            out = np.empty(len(idx), dtype=object)
            out[:] = [self.classes_[i] for i in idx]
            try:
                return out.astype(np.asarray(self.classes_).dtype)
            except (TypeError, ValueError):
                return out
        pred = self._est.predict(X)
        return np.clip(pred, 0.0, None) if self.task == "count" else pred

    def describe(self) -> Dict[str, Any]:
        return {"family": self.family, "config": self.config, "task": self.task, "metric": self.metric,
                "split": self.split.to_dict(), "drop": list(self.drop), "column_types": dict(self.column_types),
                "applied_findings": list(self.applied), "evidence_mismatch": list(self.evidence_mismatch),
                "features": list(self.features), "scaling": dict(self.scaling)}


def _has_col(data, name: str) -> bool:
    try:
        return name in (data.columns if hasattr(data, "columns") and not isinstance(data, dict) else data)
    except TypeError:
        return False


def _fallback_type(col: Dict[str, Any]) -> str:
    stored = str(col.get("stored_as", ""))
    if stored in ("float", "int", "bool", "uint"):
        return "numeric_continuous"
    return "categorical_low" if int(col.get("n_unique") or 0) <= 50 else "categorical_high"


def build_ml(report: DiscoveryReport, *, family: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
             acknowledge_pending: bool = False, seed: Optional[int] = None) -> MLSolution:
    pending_critical = [f.id for f in report.findings
                        if f.severity == Severity.CRITICAL and report.decision(f.id).state == DecisionState.PENDING]
    if pending_critical and not acknowledge_pending:
        raise PendingCriticalFindings(
            "decide on these CRITICAL findings first (report.accept(id) / report.reject(id)), or pass "
            f"acknowledge_pending=True: {pending_critical}")
    sec = report.profile.sections
    target_sec = sec.get("target") or {}
    task = target_sec.get("task") or report.input_summary.get("task")
    metric = report.input_summary.get("metric") or target_sec.get("metric")
    columns = {c["name"]: c for c in sec.get("columns") or []}
    col_types = {name: c["type"] for name, c in columns.items()}
    drop: List[str] = []
    feats: List[Dict[str, Any]] = []
    scaling: Dict[str, str] = {}
    split = SplitSpec(n_splits=int((sec.get("structure") or {}).get("splitter", {}).get("n_splits", 5)))
    applied: List[str] = []
    fam_from_action = None
    for f in report.findings:
        act = f.suggested_action
        if act is None:
            continue
        state = report.decision(f.id).state
        if state != DecisionState.ACCEPTED:
            if f.default_applied and act.kind == "set_column_type":
                c = act.params["column"]
                if c in columns:
                    col_types[c] = _fallback_type(columns[c])       # discovery's inference, not accepted
            continue
        if act.kind not in SUPPORTED_ACTIONS:
            raise UnsupportedAction(f"{f.id}: action {act.kind!r} is not supported by the ML builder yet")
        p = act.params
        if act.kind == "drop_column":
            if p["column"] not in drop:
                drop.append(p["column"])
        elif act.kind == "set_column_type":
            if p["type"] not in TYPES:
                raise ValueError(f"{f.id}: unknown column type {p['type']!r}")
            col_types[p["column"]] = p["type"]
        elif act.kind == "use_split":
            if p["kind"] == "group":
                split = replace(split, kind="group" if split.kind != "forward" else "forward", groups=p["groups"])
            elif p["kind"] == "forward":
                split = replace(split, kind="forward", time=p.get("time"), gap=int(p.get("gap", 0)))
            else:
                raise UnsupportedAction(f"{f.id}: split kind {p['kind']!r} is not supported")
        elif act.kind == "dedupe":
            split = replace(split, group_duplicates=True)
        elif act.kind == "set_metric":
            metric = p["metric"]
        elif act.kind == "use_family":
            fam_from_action = p["family"]
        elif act.kind == "scale":
            if p["method"] not in ("robust", "quantile", "standard"):
                raise UnsupportedAction(f"{f.id}: scaling method {p['method']!r} is not supported")
            for c in ([p["column"]] if isinstance(p["column"], str) else list(p["column"])):
                scaling[c] = p["method"]
        elif act.kind == "add_feature_family":
            FeatureSpec.from_dict(p)                         # validates
            feats.append(dict(p))
        applied.append(f.id)
    rec = report.recommendation
    fam = family or fam_from_action or (rec.family if rec is not None else None)
    if fam is None:
        raise ValueError("the report recommends no family (it abstained); pass build(family=...) to choose one")
    get_family(fam)                                          # validates the name / availability
    if config is None:
        config = dict(rec.config) if rec is not None and rec.family == fam else dict(
            get_family(fam).configs(10_000, len(col_types), task)[0])
    in_experiments = {fid for fid in sec.get("applied_findings") or []}
    with_actions = {f.id for f in report.findings if f.suggested_action is not None}
    mismatch = sorted(((in_experiments ^ set(applied)) & with_actions))
    if mismatch:
        warnings.warn(
            "the report's experiments and this build apply different findings "
            f"({mismatch}); its scores may not describe this setup. Re-run discover(..., "
            "decisions=report.decisions()) to measure it.", stacklevel=3)
    for c in drop:
        col_types.pop(c, None)
    return MLSolution(family=fam, config=config, task=task, metric=metric, column_types=col_types, drop=drop,
                      split=split, target=report.input_summary.get("target"), applied=applied,
                      evidence_mismatch=mismatch, features=feats, scaling=scaling, seed=report.seed if seed is None else seed)


register_builder("ml", build_ml)

"""
Opt-in recording of a discovery run in the tensor-optix experiment ledger
(``discover(..., store=ExperimentStore(path) or path)``).

Layout (no ledger schema change -- plain experiments and events):
  * one parent experiment: ``algorithm="discovery"``, ``task_env="discovery:<data fingerprint>"``,
    ``config={"role": "discovery", ...}``;
  * one child per family trial: ``algorithm="discovery_trial"``,
    ``parent_experiment_id=<parent>``, ``config={"role": "discovery_trial", "family": ...,
    "family_config": {...}}``, CV mean / interval in ``final_metrics``;
  * parent events: ``discovery.profile`` (versioned meta-features), one ``discovery.finding``
    per finding, ``discovery.decision`` (decisions at record time), ``discovery.recommendation``,
    ``discovery.budget``.

Guardrails:
  * records live under their own ``task_env`` namespace, so selectors that
    query a task's history never see them;
  * no config uses the keys the method selector treats as evidence
    (``sampler``, ``op_name``, ``is_baseline``) -- checked before writing;
  * ``best_primary_score`` is never set (discovery metrics are not all
    higher-is-better); scores go to ``final_metrics`` with the metric name.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Union

from .findings import jsonable
from .schema import DiscoveryReport

RESERVED_CONFIG_KEYS = ("sampler", "op_name", "is_baseline")


def _check(config: Dict[str, Any]) -> Dict[str, Any]:
    bad = [k for k in RESERVED_CONFIG_KEYS if k in config]
    if bad:
        raise ValueError(f"discovery ledger config must not use {bad} (they mark method evidence)")
    return config


def _open(store: Union[str, Any]):
    from ..ledger import ExperimentStore
    if isinstance(store, ExperimentStore):
        return store, False
    if isinstance(store, str):
        return ExperimentStore(store), True
    raise TypeError("store must be an ExperimentStore or a path to a ledger database")


def record_report(report: DiscoveryReport, store: Union[str, Any]) -> Dict[str, Any]:
    from ..ledger import EventRecord, EvaluationStatus, ExperimentState
    st, owned = _open(store)
    try:
        fp = (report.profile.fingerprint or "unknown")[:16]
        rec = report.recommendation
        parent = st.create_experiment(
            task_env=f"discovery:{fp}", algorithm="discovery", framework="numpy", seed=report.seed,
            checkpoint_dir="",
            config=_check(jsonable({"role": "discovery", "domain": report.domain,
                                    "task": report.input_summary.get("task"),
                                    "metric": report.input_summary.get("metric"),
                                    "schema_version": report.schema_version})))
        pid = parent.experiment_id
        st.start_experiment(pid)
        trial_ids: Dict[str, str] = {}
        for x in report.experiments:
            if x.stage != "trial":
                continue
            child = st.create_experiment(
                task_env=f"discovery:{fp}", algorithm="discovery_trial", framework="numpy", seed=report.seed,
                checkpoint_dir="", parent_experiment_id=pid,
                config=_check(jsonable({"role": "discovery_trial", "family": x.family,
                                        "family_config": x.config, "metric": x.metric})))
            st.start_experiment(child.experiment_id)
            metrics = {f"cv_{x.metric}_mean": x.mean}
            if x.interval:
                metrics[f"cv_{x.metric}_lo"], metrics[f"cv_{x.metric}_hi"] = x.interval
            st.finalize_experiment(
                child.experiment_id,
                state=ExperimentState.COMPLETED if x.status == "ok" else ExperimentState.FAILED,
                evaluation_status=EvaluationStatus.PRELIMINARY,
                final_metrics={k: float(v) for k, v in metrics.items() if v is not None},
                error_message=x.error)
            trial_ids[x.key] = child.experiment_id
        seq = 0

        def emit(kind: str, payload: Dict[str, Any]) -> None:
            nonlocal seq
            st.append_event(pid, EventRecord(event_id=str(uuid.uuid4()), experiment_id=pid, sequence=seq,
                                             event_type=kind, source="discovery", payload=jsonable(payload),
                                             timestamp=time.time()))
            seq += 1

        emit("discovery.profile", {"meta_features": report.profile.sections.get("meta_features"),
                                   "fingerprint": report.profile.fingerprint,
                                   "n_total": report.profile.n_total, "n_sampled": report.profile.n_sampled})
        for f in report.findings:
            emit("discovery.finding", f.to_dict())
        emit("discovery.decision", {k: v.to_dict() for k, v in report.decisions().items()})
        emit("discovery.recommendation", rec.to_dict() if rec is not None else {})
        emit("discovery.budget", {"requested": report.budget_requested, "used": report.budget_used})
        final = {}
        if rec is not None and rec.expected_score is not None:
            final[f"holdout_{rec.metric}"] = float(rec.expected_score)
        st.finalize_experiment(pid, state=ExperimentState.COMPLETED, evaluation_status=EvaluationStatus.PRELIMINARY,
                               final_metrics=final)
        return {"experiment_id": pid, "trials": trial_ids}
    finally:
        if owned:
            st.close()

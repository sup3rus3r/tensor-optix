"""Structure detection against planted ground truth and look-alike controls."""
import numpy as np
import pytest

from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.structure import detect_structure
from tensor_optix.discovery.ml.target import profile_target
from tensor_optix.discovery.ml.types import infer_types

import synth


def _run(cols, y, **kw):
    table = to_table(cols, target=y, **{k: v for k, v in kw.items() if k in ("groups", "time")})
    types = infer_types(table, np.random.default_rng(0))
    prof, yenc, keep, _ = profile_target(table.target)
    return detect_structure(table, types, yenc, task=prof.task, n_classes=prof.n_classes,
                            metric=prof.metric, seed=kw.get("seed", 0))


def _ids(findings):
    return {f.id: f for f in findings}


@pytest.mark.parametrize("seed", [0, 1])
def test_planted_groups_require_a_group_split(seed):
    cols, y = synth.grouped(seed=seed)
    prof, fs = _run(cols, y, seed=seed)
    f = _ids(fs).get("structure.grouped:col=customer_id")
    assert prof.kind == "grouped" and prof.group_column == "customer_id"
    assert f is not None and f.severity == Severity.CRITICAL and f.default_applied
    assert f.suggested_action.params == {"kind": "group", "groups": "customer_id"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_repeating_ids_without_group_effect_are_not_flagged(seed):
    cols, y = synth.iid_with_repeating_ids(seed=seed)
    prof, fs = _run(cols, y, seed=seed)
    assert prof.kind == "iid"
    assert not any(f.severity == Severity.CRITICAL for f in fs)


def test_declared_groups_are_used_and_reported():
    cols, y = synth.iid_with_repeating_ids()
    prof, fs = _run(cols, y, groups="customer_id")
    assert prof.kind == "grouped" and prof.groups is not None
    assert _ids(fs)["structure.grouped:col=customer_id"].severity == Severity.INFO


@pytest.mark.parametrize("seed", [0, 1])
def test_planted_drift_requires_forward_chaining(seed):
    cols, y = synth.drifting(seed=seed)
    prof, fs = _run(cols, y, seed=seed)
    f = _ids(fs).get("structure.temporal:col=when")
    assert prof.kind == "temporal" and prof.time_column == "when" and prof.time is not None
    assert f is not None and f.severity == Severity.CRITICAL
    assert f.suggested_action.kind == "use_split" and f.suggested_action.params["kind"] == "forward"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dates_without_time_structure_are_not_flagged(seed):
    cols, y = synth.iid_with_dates(seed=seed)
    prof, fs = _run(cols, y, seed=seed)
    assert prof.kind == "iid"
    assert not any(f.id.startswith("structure.temporal") for f in fs)


def test_row_order_sorted_by_target_is_reported_not_acted_on():
    cols, y = synth.sorted_by_target()
    prof, fs = _run(cols, y)
    f = _ids(fs).get("structure.row_order_informative")
    assert f is not None and f.suggested_action is None
    assert prof.kind == "iid"


def test_declared_time_is_used():
    cols, y = synth.iid_with_dates()
    prof, fs = _run(cols, y, time="when")
    assert prof.kind == "temporal" and _ids(fs)["structure.temporal:col=when"].severity == Severity.INFO


def test_andrews_lag_grows_with_persistence():
    from tensor_optix.discovery.ml.structure import andrews_lag
    r = np.random.default_rng(0)
    white = r.normal(size=2000)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.95 * ar[t - 1] + r.normal()
    assert andrews_lag(white) <= 5
    assert andrews_lag(ar) > 30
    assert andrews_lag(ar) <= 500 and andrews_lag(white, floor=12) == 12

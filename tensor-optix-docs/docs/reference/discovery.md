# tensor_optix.discovery

Understand a dataset, find the model family that fits it, and explain why. Guide: [Understand your data and choose a model family](../guides/discovery.md); methods: [Discovery statistics](../guides/discovery-statistics.md).

```python
from tensor_optix.discovery import (
    discover, Budget, DiscoveryReport, Finding, Evidence, Action, Decision,
    DecisionState, Severity, Category, Recommendation, ExperimentResult, Comparison,
)
```

## discover

```python
discover(data, target=None, *, kind="auto", task=None, groups=None, time=None, test=None,
         column_names=None, column_types=None, metric=None, families=None, budget=None,
         decisions=None, seed=0, holdout=0.15, store=None, verbose=False) -> DiscoveryReport
```

| Argument | |
|---|---|
| `data` | pandas DataFrame, dict of columns, 2-D numpy array, structured array, or an `(X, y)` tuple |
| `target` | column name, column index, or an array of targets |
| `kind` | `"auto"` (default), `"ml"`, `"rl"` (RL environments are not supported yet: `NotImplementedError`) |
| `task` | force `"binary"`, `"multiclass"`, `"regression"` or `"count"` |
| `groups`, `time` | declare the group / time column (name, index or array) |
| `test` | features of the rows the model will be used on (enables train-vs-test shift checks) |
| `column_names` | names for numpy columns |
| `column_types` | `{column: type}` overrides; types: `numeric_continuous`, `numeric_discrete`, `boolean`, `categorical_low`, `categorical_high`, `datetime`, `identifier`, `text`, `embedded_number`, `constant`, `url`, `list` |
| `metric` | force the comparison metric (`auc`, `macro_auc`, `accuracy`, `balanced_accuracy`, `macro_f1`, `log_loss`, `brier`, `rmse`, `mae`, `r2`, `poisson_deviance`) |
| `families` | restrict candidates: `linear`, `naive_bayes`, `knn`, `mlp` (torch), `lightgbm`, `xgboost` |
| `budget` | `Budget`; default `Budget.default_for_rows(n_rows)` |
| `decisions` | `{finding_id: "accept" \| "reject" \| Decision}`, typically `report.decisions()` |
| `seed` | reproducibility seed |
| `holdout` | fraction kept aside for the final estimate; `0` = none |
| `store` | `ExperimentStore` or ledger path; the run is recorded there |

## Budget

```python
Budget(time_s=600.0, max_rows=None, max_fits=None, env_steps=None)
Budget.default_for_rows(n_rows)     # ~60 s at 10k rows, growing with log size, capped at 15 min
```

`max_rows` caps the rows used for experiments (default 50 000; a stratified or whole-group sample). When an experiment would not fit in the remaining time it is skipped and reported as a `budget.skipped:family=...` finding; `report.budget_used["cuts"]` lists every cut.

## DiscoveryReport

| Member | |
|---|---|
| `findings` | `List[Finding]` |
| `experiments` | `List[ExperimentResult]` -- every landmark and trial, per-fold scores included |
| `comparisons` | `List[Comparison]` -- every paired comparison |
| `recommendation` | `Recommendation` |
| `plan` | `List[PlanStep]` |
| `profile` | `Profile`: `sections` holds `columns`, `target`, `structure`, `quality`, `leakage`, `landmarks`, `shift`, `meta_features`, `applied_findings`, `excluded_columns` |
| `limitations` | `List[str]` |
| `budget_requested`, `budget_used`, `seed`, `ledger_refs`, `schema_version` | |
| `finding(id)`, `decision(id)` | look-ups |
| `accept(id, note=None)`, `reject(id, note=None)` | record a decision |
| `accept_where(*, category=None, min_severity=None, note=None)` | bulk accept of pending findings (at least one filter required) |
| `pending(min_severity="info")` | undecided findings |
| `decisions()` | copy of all decisions, for `discover(decisions=...)` |
| `to_html(path=None)`, `from_html(path_or_text)` | self-contained HTML (the JSON is embedded) |
| `to_json(path=None)`, `from_json(path_or_text)`, `to_dict()`, `from_dict(d)` | |
| `summary()` | plain-text rendering |
| `build(...)` | see below |

Reports are versioned (`DISCOVERY_SCHEMA_VERSION`); older reports are migrated on load, newer ones are refused with a clear message.

## Finding

| Field | |
|---|---|
| `id` | stable id, e.g. `leakage.single_feature:col=refund_flag` |
| `category` | `data_types`, `quality`, `structure`, `leakage`, `target`, `shift`, `family`, `budget`, `limitation` |
| `severity` | `info` < `advisory` < `warning` < `critical` |
| `statement` | plain-language finding |
| `evidence` | `Tuple[Evidence]`: `test`, `statistic`, `n`, `null_value`, `p_value`, `interval`, `effect_size`, `details`, `seed` |
| `confidence`, `confidence_basis` | `low` / `moderate` / `high`, and what it rests on |
| `suggested_action` | `Action(kind, params)` from a closed grammar: `drop_column`, `set_column_type`, `use_split`, `dedupe`, `set_metric`, `use_family`, ... |
| `default_applied` | whether discovery's own experiments applied it (unless rejected) |

## Recommendation

`family`, `config`, `verdict` (`clear_winner`, `tie_simplest_chosen`, `undetermined`, `abstain`), `rationale`, `expected_score` / `expected_interval` (holdout, or CV when there is no holdout), `metric`, `alternatives`, `conditions`.

## build and MLSolution

```python
solution = report.build(family=None, config=None, acknowledge_pending=False, seed=None)
```

Applies the findings whose decision is **accepted**. Raises `PendingCriticalFindings` while a critical finding is pending (unless `acknowledge_pending=True`), `ValueError` if the report abstained and no `family` is given. Warns when the applied findings differ from those discovery's experiments applied (`solution.evidence_mismatch`).

| `MLSolution` | |
|---|---|
| `fit(data, target=None, *, groups=None, time=None)` | train on all rows |
| `predict(data)` | predictions; original class labels for classification |
| `predict_proba(data)` | class probabilities (classification) |
| `cross_validate(data, target=None, *, groups=None, time=None, repeats=1, n_splits=None)` | `ExperimentResult` with the built split |
| `splitter(n_splits=None)` | the CV splitter (`KFold`, `StratifiedKFold`, `GroupKFold`, `StratifiedGroupKFold`, `ForwardChaining`) |
| `describe()` | family, config, split, dropped columns, types, applied findings, mismatches |

## Ledger records

With `store=`: a parent experiment (`algorithm="discovery"`, `task_env="discovery:<fingerprint>"`, `config["role"]="discovery"`), a child per trial (`algorithm="discovery_trial"`, `config={"role": "discovery_trial", "family": ..., "family_config": ...}`, CV mean and interval in `final_metrics`), and parent events `discovery.profile`, `discovery.finding`, `discovery.decision`, `discovery.recommendation`, `discovery.budget`. No ledger schema change; configs never use the keys the method selector reads as evidence (`sampler`, `op_name`, `is_baseline`).

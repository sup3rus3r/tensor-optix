# Discovery statistics

This page states exactly how discovery reaches each conclusion, so a report can be checked rather than trusted. Everything is implemented in numpy (no scipy); every random choice derives from the run's `seed`.

## Principles

- **Our own evaluation must not leak.** All preprocessing statistics are learned inside training folds; target encoding is out-of-fold *within* the training rows; duplicate rows never straddle a split; the confirmation holdout is cut before any model selection and used once.
- **Identical folds for every family.** All comparisons are paired by fold id (or by test row), which removes most of the fold-to-fold noise from the difference.
- **Practical significance, not just statistical.** Each metric has a *region of practical equivalence* (ROPE): differences smaller than it do not matter. A verdict needs the interval of the difference to clear it.
- **Conservative detectors.** A false "leak" or "grouped data" alarm costs trust, so every detector is calibrated (below).

## Comparing two families

For per-fold scores `a_j`, `b_j` on the same folds, the difference `d_j = a_j - b_j` (oriented so positive means `a` is better) gets the **Nadeau-Bengio corrected resampled t interval**:

```
se = sqrt((1/J + n_test/n_train) * var(d))
```

The `n_test/n_train` term corrects for the overlap between CV training sets, which makes the plain t interval far too narrow.

The verdict compares the 95% interval `[lo, hi]` with the ROPE `r`:

| Interval | Verdict |
|---|---|
| `lo > r` | `better` |
| `hi < -r` | `worse` |
| `-r <= lo` and `hi <= r` | `equivalent` |
| otherwise | `undetermined` |

Default ROPEs: AUC 0.005; accuracy / balanced accuracy / macro-F1 / R² 0.01; RMSE / MAE 1% of the target's standard deviation; log-loss / Brier / Poisson deviance 1% of the dummy model's loss.

p-values of all comparisons against the leader are **Holm-adjusted**.

### Time-ordered data

Forward-chaining CV (train on the past, test on the next block) cannot be repeated, so it gives only a handful of fold scores. Discovery uses 5-10 test blocks of at least 100 rows each, and -- when the metric is a mean of per-row losses (RMSE, MAE, log-loss, Brier, Poisson deviance) -- compares families with a **Diebold-Mariano test on the pooled test rows**: the per-row loss differences, in time order, with a Newey-West (HAC) standard error.

The HAC bandwidth is chosen from the data (**Andrews 1991**, AR(1) plug-in: `1.1447 (α n)^{1/3}` with `α = 4ρ² / ((1-ρ)²(1+ρ)²)`), never shorter than the forward-chaining gap. A fixed short bandwidth badly understates the uncertainty when the loss differences are persistent. RMSE differences are reported on the RMSE scale (MSE difference / 2·RMSE).

## The recommendation rule

1. The **leader** is the family with the best mean score.
2. Every other family is compared with the leader.
3. If the leader is not `better` than the dummy baseline: **abstain**, saying whether the evidence shows no meaningful signal (`equivalent`/`worse`) or is merely inconclusive (`undetermined`).
4. Among the leader and the families `equivalent` to it, the **simplest** wins (simplicity order: linear < naive Bayes < k-NN < boosted trees < MLP). If that is not the leader, the verdict is `tie_simplest_chosen`.
5. Otherwise `clear_winner` if every other family is `worse`, else `undetermined` with the unresolved families as alternatives.
6. While comparisons with the leader are `undetermined` and the budget allows, CV repeats are added for the leader and those families (up to 5 repeats; not for forward chaining).

The chosen family is refit on all development rows and scored on the holdout. Its interval is a percentile bootstrap: whole groups for grouped data, moving blocks for temporal data, rows otherwise.

## Structure detectors

**Grouped data.** For each candidate group column (repeating values, not unique per row), the same reference model is scored with a random split and with a group split. The *split gap* (random minus group score) gets an interval from both sets of resampled scores; with several candidates the level is Bonferroni-corrected. A gap that clears the ROPE is `critical` (discovery switches to a group split); a gap whose interval excludes 0 but not the ROPE is an `advisory` *possible groups* finding, applied only if you accept it.

**Temporal data.** For a datetime column (or the row order), a model trained on the past and one trained on an equal-size random sample of all other rows predict the same future blocks; the time column is excluded from both, so a date trend cannot be extrapolated. The per-row loss differences get a Diebold-Mariano test as above. A past-only model that is worse by more than the ROPE means the data drifts: validate with forward chaining.

**Row order.** The same Diebold-Mariano result on the row index, plus a Spearman correlation between row position and the target (|ρ| ≥ 0.3, p < 0.001) for files sorted by outcome.

## Leakage detectors

All scores are out-of-fold on the structure-correct split, expressed as **skill**: the fraction of the possible improvement over the dummy, `(score - dummy) / (perfect - dummy)`.

| Check | Rule | Severity |
|---|---|---|
| Single feature near-perfect | depth-3 tree on that feature alone, skill >= 0.90 (AUC >= 0.95) | `warning`, `critical` if its name matches the target's; excluded from discovery's experiments; the full model's score without it is attached |
| One feature dominates | >= 90% of the full model's skill and twice the next feature's | `info` |
| Identifier predicts | an id, used as a number, skill >= 0.2 with an interval above 0 | `warning` |
| Missingness predicts | the missing indicator alone, skill >= 0.1 | `advisory` |
| Future target | a feature equals the target 1-3 steps later in >= 90% of rows (time-ordered data) | `critical` |

## Shift detectors

With `test=`: an out-of-fold depth-3 tree trying to tell training rows from test rows (adversarial validation; `advisory` when the AUC's lower bound exceeds 0.55, `warning` at AUC >= 0.70), plus per-feature tests -- Kolmogorov-Smirnov and PSI for numeric features, chi-square (rare categories pooled so expected counts are >= 5) for categoricals. Per-feature results are Holm-adjusted and reported only when the adjusted p < 0.01 *and* the effect is large enough to matter (KS D or PSI >= 0.1, total-variation distance >= 0.1). Exact train/test row overlap and columns missing from `test` are reported too. Datetime columns are compared by range only.

For time-ordered data, the same per-feature tests compare the earliest third with the latest third, plus the target.

## Calibration

Each detector is run on 200 seeded *null* datasets (no planted effect). The exact (Clopper-Pearson) upper 95% bound of its false-alarm rate must be <= 0.08; the test suite enforces this (`tests/test_discovery/test_calibration.py`, marked `slow`). The same suite checks that two exchangeable families on strongly autocorrelated series (AR coefficient 0.95) are rarely declared different by the pooled comparison.

Planted-effect tests check the other direction: planted leaks, groups, drift, covariate shift, non-linearity and noise-only targets are each reported (or, for noise, lead to abstaining).

## Benchmark

`benchmarks/validate_discovery.py` runs discovery on six realistic synthetic datasets -- churn with a leaked column, repeated patient visits, daily sales with drift, a 600-level categorical, 2% fraud, Friedman-1 -- each with a fresh test set from the same generator (new customers, new patients, the future). It checks that every planted issue is reported, that the recommended family's test score is within the ROPE of the best family's on at least 80% of datasets and never worse than the dummy, that the run stays within its budget, and that the report renders in under a second.

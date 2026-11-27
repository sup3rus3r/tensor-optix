# Gradient-boosted trees (tensor_optix.gbdt)

A native histogram gradient-boosted decision tree implementation in numpy -- no compiled dependency -- with early stopping by tensor-optix's `ConvergenceMonitor`.

```python
from tensor_optix.gbdt import GBDTClassifier, GBDTRegressor

clf = GBDTClassifier(learning_rate=0.05, max_depth=6).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
labels = clf.predict(X_test)                      # original labels (any hashable type)

reg = GBDTRegressor(objective="l2").fit(X_train, y_train, eval_set=(X_val, y_val))
counts = GBDTRegressor(objective="poisson").fit(X_train, n_events)
```

## What it does

- **Histogram splits**: numeric features are binned once into up to `max_bins` quantile bins; trees grow depth-wise and every level's histograms come from one `bincount` per feature.
- **Second-order gains** with L2 regularisation: `G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) - γ`; leaves are `-G/(H+λ)` times the learning rate.
- **Missing values** need no imputation: each feature has a missing bin, and every split learns whether missing values go left or right (unseen values at predict time follow the same direction).
- **Native categorical splits**: pass `categorical_features=[column indices]` (integer-coded columns). Within a node the categories are ordered by `G/(H+cat_smooth)` and the best prefix goes left -- the optimal binary partition for these losses -- instead of treating codes as numbers. Regularised as in LightGBM: categories with fewer than `cat_smooth` rows stay out of the search, categorical gains get an extra `cat_l2`, each side needs `min_data_per_group` rows, and at most `max_cat_threshold` categories go to one side. Unseen categories map to an "other" bin.
- **Objectives**: binary and multiclass log loss (classifier), squared error and Poisson (regressor).
- **Row / column subsampling**: `subsample`, `colsample` (per boosting round), seeded by `seed`.
- **Early stopping**: with `early_stopping=True` (default) a `validation_fraction` of the training rows -- or `eval_set` -- is scored after every round, and a `ConvergenceMonitor` on the negated validation loss stops training once it reaches DORMANT (no new ground for its patience, no slow rise in progress). The model is truncated to the round with the lowest validation loss (`best_iteration_`); `stop_reason_` says `"converged"` or `"n_estimators"` (the size cap).

| Parameter | Default | |
|---|---|---|
| `n_estimators` | 2000 | maximum boosting rounds |
| `learning_rate` | 0.1 | shrinkage |
| `max_depth` | 6 | |
| `min_samples_leaf` | 20 | |
| `min_child_weight` | 1e-3 | minimum hessian per child |
| `reg_lambda`, `gamma` | 1.0, 0.0 | L2 on leaf values, minimum split gain |
| `subsample`, `colsample` | 1.0, 1.0 | |
| `max_bins` | 255 | |
| `categorical_features` | `()` | column indices |
| `cat_smooth`, `cat_l2`, `min_data_per_group`, `max_cat_threshold` | 10, 10, 100, 32 | categorical regularisation |
| `early_stopping`, `validation_fraction` | True, 0.1 | |
| `seed` | 0 | |

## Accuracy and speed

`benchmarks/validate_gbdt.py` compares it with LightGBM on six problems (Friedman-1, interactions with 10% missing values, a 200-level categorical, multiclass, Poisson counts, a 2%-positive binary target) over three seeds with the same rows, validation rows and learning rate. The native implementation is within each metric's practical threshold of LightGBM on all of them; it is pure numpy and typically 10-20x slower.

## In discovery

Discovery offers it as the `gbdt` family. It is tried by default only when neither LightGBM nor XGBoost is installed -- so boosted trees are always part of the comparison -- and on request otherwise:

```python
discover(df, target="y", families=["linear", "gbdt", "lightgbm"])
```

"""
tensor_optix.gbdt -- native histogram gradient-boosted decision trees (numpy).

    from tensor_optix.gbdt import GBDTClassifier, GBDTRegressor
    model = GBDTClassifier(learning_rate=0.1, max_depth=6).fit(X, y)
    proba = model.predict_proba(X_new)

Second-order (Newton) splits with L2 regularisation, native missing-value
handling (learned default direction), native categorical splits
(``categorical_features=[...]``), row / column subsampling, and early stopping
by tensor-optix's ``ConvergenceMonitor`` on an inner validation split (or
``eval_set``). Objectives: binary / multiclass log loss, squared error, Poisson.
"""
from .booster import Booster, GBDTClassifier, GBDTParams, GBDTRegressor

__all__ = ["Booster", "GBDTClassifier", "GBDTParams", "GBDTRegressor"]

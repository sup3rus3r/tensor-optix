"""
tensor_optix.ml_core — the ML-native training core.

A genuinely separate contract from the RL core's BaseAgent/BasePipeline/
EpisodeData: fit()/predict()/evaluate() instead of act()/learn(), epochs
and batches instead of episodes, and real classification/regression
metrics (accuracy, F1, confusion matrix, MAE, R2) instead of mean episode
return. See each module's docstring for which pieces of the RL core are
reused as-is (CheckpointRegistry, BackoffScheduler, SPSAOptimizer, the
core dataclasses) versus genuinely new (BaseMLModel, Trainer, the metrics
registry, the framework adapters).

This does not replace or remove tensor_optix.ml.MLAgent/DatasetPipeline/
simple.Optimizer's existing ML path -- that path keeps working unchanged.
ml_core is the new, documented, first-class way to do supervised ML with
tensor-optix going forward.
"""
from tensor_optix.ml_core.base_model import BaseMLModel, FitHistory
from tensor_optix.ml_core.metrics import MLMetrics, available_metrics, resolve_metrics

__all__ = [
    "BaseMLModel",
    "FitHistory",
    "MLMetrics",
    "available_metrics",
    "resolve_metrics",
]

try:
    from tensor_optix.ml_core.adapters.torch_model import TorchMLModel
    __all__.append("TorchMLModel")
except ImportError:
    pass

try:
    from tensor_optix.ml_core.adapters.tf_model import TFMLModel
    __all__.append("TFMLModel")
except ImportError:
    pass

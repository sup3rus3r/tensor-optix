# tensor_optix.ml_core

An ML-native training core: `fit()`/`predict()`/`evaluate()` instead of `act()`/`learn()`, epochs and batches instead of episodes, real classification/regression metrics instead of mean episode return. Reuses `CheckpointRegistry`, `BackoffScheduler`, and `SPSAOptimizer` from the RL core unmodified. All three are algorithm-agnostic already. Does not replace or remove [`tensor_optix.ml`](ml.md)'s `MLAgent`/`DatasetPipeline` path, which still works unchanged.

## BaseMLModel

```python
class BaseMLModel(ABC):
    """
    The only contract between a trained ML model and tensor-optix's
    ML-native checkpointing/metrics machinery.
    """

    def predict(self, X: Any) -> Any:
        """Run inference on a batch of inputs. Must not update weights."""

    def train_step(self, batch: Any) -> float:
        """One gradient step on a single batch. Returns the scalar loss
        (a Python float, not a framework tensor)."""

    def evaluate(self, data) -> MLMetrics:
        """Run inference over an entire dataset/loader and compute MLMetrics
        via resolve_metrics()."""

    def iter_batches(self, data, batch_size: int):
        """Yield framework-appropriate batches from a raw (X, y) pair or a
        native Dataset/DataLoader/tf.data.Dataset -- auto-detected."""

    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...

    def get_hyperparams(self) -> HyperparamSet:
        """Default: no tunable hyperparameters. Override to enable SPSA."""

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        """Default: no-op."""

    default_param_bounds: Dict[str, tuple] = {}
    default_log_params: List[str] = []

    def average_weights(self, paths: List[str]) -> None:
        """Stochastic Weight Averaging across checkpoints. Default: no-op."""

    def fit(
        self,
        train_data,
        val_data=None,
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping_patience: Optional[int] = None,
        metrics="auto",
        callbacks: Optional[List] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> FitHistory:
        """
        Train for up to `epochs` epochs. If val_data is given and
        early_stopping_patience is set, stops early once the validation
        primary_score hasn't improved for that many consecutive epochs,
        and restores the best checkpoint's weights before returning.
        "New best" is tracked independently of whether checkpoint_dir is
        set -- early stopping works even with no disk checkpointing.
        """
```

## FitHistory

```python
@dataclass
class FitHistory:
    """Per-epoch record of training, returned by fit()."""
    train_loss: List[float]          # mean training loss per epoch
    val_metrics: List[MLMetrics]      # one MLMetrics per epoch (only if val_data given)
    best_epoch: Optional[int]         # index of the best-scoring epoch
    stopped_early: bool                # True if early stopping fired
    epochs_run: int                    # actual epochs completed (<= epochs if stopped early)
```

## Trainer

```python
class Trainer:
    """
    Drives BaseMLModel.fit()'s actual epoch loop. Not part of the public
    BaseMLModel API -- constructed and run internally by fit().

    Per epoch:
      1. Iterate train_data in batches, call model.train_step() on each.
      2. If val_data given, call model.evaluate(val_data) -> MLMetrics.
      3. Compare MLMetrics.primary_score against the best score seen this
         run. On improvement: reset the early-stopping counter, record the
         epoch as best, and (if checkpoint_dir is set) call
         CheckpointRegistry.save(). On no improvement: advance the
         early-stopping counter and feed BackoffScheduler.record_non_improvement().
      4. If an SPSA-tunable model (default_param_bounds set), call
         SPSAOptimizer.suggest() with the accumulated EvalMetrics history
         and apply the suggestion via model.set_hyperparams().
      5. Stop if early_stopping_patience epochs have passed with no
         improvement.

    After the loop, if checkpoint_dir was set and at least one checkpoint
    was saved, restores the best checkpoint's weights via
    CheckpointRegistry.load_best() before returning.
    """

    def __init__(
        self,
        model,
        epochs: int,
        batch_size: int,
        early_stopping_patience: Optional[int],
        metrics,
        callbacks: Optional[List],
        checkpoint_dir: Optional[str],
    ): ...

    def run(self, train_data, val_data) -> FitHistory: ...
```

```python
# BackoffScheduler defaults used by Trainer -- re-tuned from the RL core's
# defaults (trend_window=8, score_window=20, noise_k=2.0) for per-epoch
# validation curves, which are typically smoother and evaluated far less
# often than per-episode RL reward.
ML_BACKOFF_DEFAULTS = dict(
    trend_window=4,
    score_window=8,
    noise_k=1.0,
    plateau_threshold=5,
    dormant_threshold=20,
    min_episodes_before_degradation=3,
)
```

## MLMetrics

```python
@dataclass
class MLMetrics:
    """Scored output from a BaseMLModel's evaluate() call."""
    primary_score: float   # higher-is-better; feeds CheckpointRegistry/BackoffScheduler
    loss: float              # raw loss value, for display only
    task_type: str            # "classification" | "regression" | "other"
    metrics: Dict[str, float]  # accuracy, f1_macro, confusion_matrix_<i>_<j>, mae, r2, ...
    epoch: int

    def to_eval_metrics(self) -> EvalMetrics:
        """Bridge to the RL core's EvalMetrics, so CheckpointRegistry
        (which only knows EvalMetrics) needs zero changes."""
```

## Metrics registry

```python
def resolve_metrics(
    metrics: Union[str, List[Callable]],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
    loss: float = 0.0,
    epoch: int = 0,
) -> MLMetrics:
    """
    Compute an MLMetrics from (y_true, y_pred), auto-detecting task type
    from y_true's dtype/values when metrics="auto" (default): integer or
    whole-number-valued-with-few-unique-values targets -> classification,
    continuous float targets -> regression. Mirrors
    tensor_optix.ml.loss_registry.resolve_loss()'s "auto" pattern.
    """

def available_metrics() -> str:
    """Human-readable list of all supported metric task types."""
```

| `metrics=` | Computed keys (`MLMetrics.metrics`) | `primary_score` |
|---|---|---|
| `"classification"` | `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, per-class `precision_<i>`/`recall_<i>`/`f1_<i>`, flattened `confusion_matrix_<i>_<j>` | `accuracy` |
| `"regression"` | `mae`, `rmse`, `r2` | `r2` |
| a list of `(y_true, y_pred) -> float` callables | one key per callable, named by `__name__` | `-loss` |

`metrics_impl/classification.py` and `metrics_impl/regression.py` implement the underlying metric math in pure numpy. Both adapters convert their tensors to numpy before calling these, so the implementation is framework-agnostic:

```python
def confusion_matrix(y_true, y_pred, n_classes) -> np.ndarray: ...
def accuracy(y_true, y_pred) -> float: ...
def precision_recall_f1(cm) -> Dict[str, np.ndarray]: ...
def classification_metrics(y_true, y_pred, n_classes) -> Dict[str, float]: ...
def unflatten_confusion_matrix(metrics, n_classes) -> np.ndarray:
    """Inverse of classification_metrics()'s confusion_matrix_<i>_<j>
    flattening, for display/plotting."""

def mean_absolute_error(y_true, y_pred) -> float: ...
def root_mean_squared_error(y_true, y_pred) -> float: ...
def r2_score(y_true, y_pred) -> float: ...
def regression_metrics(y_true, y_pred) -> Dict[str, float]: ...
```

## Adapters

```python
class TorchMLModel(BaseMLModel):
    """
    Parameters
    ----------
    model:    Any nn.Module.
    loss_fn:  Instantiated loss nn.Module, or any callable (pred, target) -> scalar.
    learning_rate, weight_decay: Adam hyperparameters. Tunable online via SPSA.
    max_grad_norm: Gradient clipping norm. None disables.
    device:   "auto", "cpu", or "cuda".
    metrics:  Passed through to evaluate() as resolve_metrics()'s metrics arg.

    default_param_bounds = {"learning_rate": (1e-5, 1e-2), "weight_decay": (0.0, 1e-2)}
    """
```

```python
class TFMLModel(BaseMLModel):
    """
    Parameters
    ----------
    model:    Any tf.keras.Model.
    loss_fn:  A tf.keras.losses.Loss instance, or any callable (y_true, y_pred) -> scalar.
    learning_rate, weight_decay: AdamW when weight_decay > 0, plain Adam otherwise
                                   (TF's base Adam has no weight_decay kwarg).
    max_grad_norm: Gradient clipping norm. None disables.
    metrics:  Passed through to evaluate() as resolve_metrics()'s metrics arg.

    default_param_bounds = {"learning_rate": (1e-5, 1e-2), "weight_decay": (0.0, 1e-2)}
    """
```

Both adapters' `iter_batches()` accept a raw `(X, y)` array pair or a native `Dataset`/`DataLoader` (Torch) / `tf.data.Dataset` (TF), auto-detected by type.

See [Train an ML model](../guides/ml-core.md) for the full guide.

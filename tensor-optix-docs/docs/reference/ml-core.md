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

    def iter_batches(self, data, batch_size: int, augment=None):
        """Yield framework-appropriate batches from a raw (X, y) pair or a
        native Dataset/DataLoader/tf.data.Dataset -- auto-detected.
        augment, if given, is applied per-batch and only takes effect for
        the raw (X, y) array form."""

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
        epochs: Optional[int] = None,
        batch_size: int = 32,
        early_stopping_patience: Optional[int] = None,
        metrics="auto",
        callbacks: Optional[List] = None,
        checkpoint_dir: Optional[str] = None,
        lr_schedule: Optional[Callable[[int], float]] = None,
        augment: Optional[Callable] = None,
        max_epochs: Optional[int] = None,
        stop_on_convergence: Optional[bool] = None,
        convergence: Optional[ConvergenceMonitor] = None,
    ) -> FitHistory:
        """
        epochs: None (default) -- train until the ConvergenceMonitor reaches
        DORMANT (no new ground for its dormant patience, no material slow
        rise still in progress). It reads the validation primary_score when
        val_data is given, otherwise the negated training loss. An int N
        runs exactly N epochs; convergence does not end it early unless
        stop_on_convergence=True.

        max_epochs: optional hard cap; off by default.
        early_stopping_patience: stop after that many epochs without a new
        validation best; off by default.
        stop_on_convergence: None (default) = on exactly when epochs is None.
        convergence: a ConvergenceMonitor replacing the ML defaults; may be
        reused across fit() calls (each call starts a new segment).

        With checkpoint_dir set, the best checkpoint's weights are restored
        before returning, however the run ended (also on an exception).
        Without it nothing is persisted and the final weights are returned.
        "New best" is tracked independently of whether checkpoint_dir is set.

        lr_schedule: callable(epoch) -> float, applied once per epoch.
        Mutually exclusive with SPSA tuning of learning_rate for this run
        (see tensor_optix.ml_core.lr_schedules).

        augment: callable(X_batch, y_batch) -> (X_batch, y_batch), applied
        to every training batch (never validation). Only takes effect on
        the raw (X, y) array input form.
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
    stopped_early: bool                # True if early stopping or convergence ended the run
    epochs_run: int                    # actual epochs completed
    stop_reason: Optional[str]         # "epochs" | "max_epochs" | "early_stopping" | "converged"
    convergence: List[ConvergenceReport]   # one report per validated epoch
```

## Trainer

```python
class Trainer:
    """
    Drives BaseMLModel.fit()'s actual epoch loop. Not part of the public
    BaseMLModel API -- constructed and run internally by fit().

    Per epoch:
      1. Iterate train_data in batches, call model.train_step() on each.
         A raw (X, y) pair is re-shuffled with a fresh random permutation
         every epoch (numpy global RNG); native Dataset / DataLoader /
         tf.data.Dataset inputs are passed through untouched.
      2. If val_data given, call model.evaluate(val_data) -> MLMetrics.
      3. Compare MLMetrics.primary_score against the best score seen this
         run. On a new best: reset the early-stopping counter, record the
         epoch as best, (if checkpoint_dir is set) CheckpointRegistry.save(),
         fire on_improvement. Otherwise advance the counter and fire
         on_no_improvement.
      4. Feed the score to the ConvergenceMonitor (at=epoch). Fire
         on_convergence on regime / state changes; on IMPROVING tell SPSA
         (on_improvement); on entering COOLING tell SPSA (on_cooling) and
         fire on_plateau once.
      5. If an SPSA-tunable model (default_param_bounds set), call
         SPSAOptimizer.suggest() with the accumulated EvalMetrics history
         and apply the suggestion via model.set_hyperparams().
      6. Stop on early_stopping_patience, or -- when stop_on_convergence --
         when the monitor enters DORMANT (on_dormant fires first).
    Without val_data (and with convergence on) only the negated training
    loss is fed to the monitor; nothing is checkpointed.

    After the loop, if checkpoint_dir was set and at least one checkpoint
    was saved, restores the best checkpoint's weights via
    CheckpointRegistry.load_best() before returning.
    """

    def __init__(
        self,
        model,
        epochs: Optional[int],
        batch_size: int,
        early_stopping_patience: Optional[int],
        metrics,
        callbacks: Optional[List],
        checkpoint_dir: Optional[str],
        lr_schedule: Optional[Callable[[int], float]] = None,
        augment: Optional[Callable] = None,
        max_epochs: Optional[int] = None,
        stop_on_convergence: Optional[bool] = None,
        convergence: Optional[ConvergenceMonitor] = None,
    ): ...

    def run(self, train_data, val_data) -> FitHistory: ...
```

```python
# ConvergenceMonitor settings used by Trainer -- per-epoch validation
# curves are far smoother than per-episode RL returns and a run is tens of
# epochs, so every window is short; patience keeps COOLING after 5 /
# DORMANT after 20 epochs without new ground.
ML_CONVERGENCE_DEFAULTS = dict(
    timescales=(4, 12, 48),
    cooling_patience=5,
    dormant_patience=20,
    persist=2,
    rel_delta=1e-4,     # gains < 0.01 % of the peak are not new ground
)
# (replaces the former ML_BACKOFF_DEFAULTS)
```

## MLMetrics

```python
@dataclass
class MLMetrics:
    """Scored output from a BaseMLModel's evaluate() call."""
    primary_score: float   # higher-is-better; feeds CheckpointRegistry/ConvergenceMonitor
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

```python
class FlaxMLModel(BaseMLModel):
    """
    JAX/Flax NNX adapter. Conventions mirror tensor_optix.algorithms.flax_ppo,
    the only other Flax NNX code in the codebase: nnx.Optimizer for the
    update step, nnx.value_and_grad for the gradient, nnx.state/to_pure_dict
    + replace_by_pure_dict/update for weight save/load.

    Parameters
    ----------
    model:    Any flax.nnx.Module. Called as model(x).
    loss_fn:  callable(pred, target) -> scalar jax array. There is no Flax
              equivalent of an instantiated loss object, so this is always
              a plain function (e.g. wrapping optax.softmax_cross_entropy_
              with_integer_labels to reduce to a scalar).
    learning_rate, weight_decay: optax.adamw when weight_decay > 0, plain
                                   optax.adam otherwise.
    max_grad_norm: applied via optax.clip_by_global_norm in the optax.chain.
                     None disables.
    metrics:  Passed through to evaluate() as resolve_metrics()'s metrics arg.

    default_param_bounds = {"learning_rate": (1e-5, 1e-2), "weight_decay": (0.0, 1e-2)}
    """
```

Shuffling: the adapters' `iter_batches()` walk a raw `(X, y)` pair in storage order - `Trainer` permutes the pair once per epoch before handing it over, so training batches differ between epochs while `evaluate()` stays deterministic. Seed with `np.random.seed(...)` for reproducible runs. A `DataLoader` / `tf.data.Dataset` keeps whatever shuffle configuration you built it with.

All three adapters' `iter_batches()` accept a raw `(X, y)` array pair, or a native `Dataset`/`DataLoader` (Torch) / `tf.data.Dataset` (TF), auto-detected by type. `FlaxMLModel` accepts only the raw `(X, y)` form -- there is no Dataset/DataLoader equivalent in this codebase's JAX dependency set.

## lr_schedules

```python
class StepLR:
    """base_lr * gamma**(epoch // step_size)."""
    def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1): ...
    def __call__(self, epoch: int) -> float: ...

class CosineAnnealingLR:
    """Cosine decay from base_lr to min_lr over `epochs` total epochs."""
    def __init__(self, base_lr: float, epochs: int, min_lr: float = 0.0): ...
    def __call__(self, epoch: int) -> float: ...

class WarmupLR:
    """Linear warmup from 0 to base_lr over warmup_epochs, then delegates
    to `after` (any other schedule) for epochs >= warmup_epochs, re-based
    to 0 at that point."""
    def __init__(self, base_lr: float, warmup_epochs: int, after: Optional[object] = None): ...
    def __call__(self, epoch: int) -> float: ...
```

Any `callable(epoch: int) -> float` works as `lr_schedule` -- these three are provided for convenience, not a closed interface.

## transfer

```python
def freeze_layers(model, layers: Optional[Sequence[str]] = None) -> None:
    """Freeze parameters/layers in-place. model: nn.Module or tf.keras.Model.
    layers: submodule names (Torch, model.named_modules()) or layer names
    (TF, layer.name) to target. None freezes the whole model."""

def unfreeze_layers(model, layers: Optional[Sequence[str]] = None) -> None:
    """Inverse of freeze_layers()."""

def load_pretrained_backbone(
    name: str,
    framework: str = "torch",
    pretrained: bool = True,
    freeze: bool = False,
    input_shape: Optional[tuple] = None,
):
    """
    Load a common pretrained backbone with its classifier head removed.
    Torch: name is a torchvision.models function name (resnet18, resnet34,
    resnet50, resnet101, resnet152, vgg11/13/16/19, densenet121/169/201,
    mobilenet_v2, mobilenet_v3_small/large, efficientnet_b0/b1/b2) -- head
    replaced with nn.Identity(). TF: name is a tf.keras.applications class
    name (e.g. ResNet50, MobileNetV2, EfficientNetB0) -- loaded with
    include_top=False, which requires input_shape. Any other architecture:
    build it directly via torchvision.models/tf.keras.applications and
    strip the head yourself.
    """
```

See [Train an ML model](../guides/ml-core.md) for the full guide.

## Classification metrics and single-output heads

`resolve_metrics(..., "classification")` works on flat arrays of class indices. Two shapes are normalised before the confusion matrix is built:

- targets of shape `(N, 1)` are flattened;
- a single-output binary head, whose per-sample output is a raw logit or probability rather than a class index, is thresholded - at `0.5` when every value lies in `[0, 1]` (probabilities), at `0` otherwise (logits).

Multi-class heads are already reduced with `argmax` by the adapters' `evaluate()`.

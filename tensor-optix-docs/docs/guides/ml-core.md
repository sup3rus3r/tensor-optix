# Train an ML model (ml_core)

`tensor_optix.ml_core` is the first-class way to train a supervised or unsupervised model with tensor-optix. The older `Optimizer(model, dataset)` path (see [ML mode](ml-mode.md)) is built on top of the RL loop, treating each mini-batch as a fake episode. `BaseMLModel`/`Trainer` are a separate, epoch-driven contract with real classification/regression metrics and validation-based early stopping.

```python
import torch.nn as nn
from tensor_optix.ml_core import TorchMLModel

model = TorchMLModel(
    nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)),
    loss_fn=nn.CrossEntropyLoss(),
    learning_rate=1e-3,
)

history = model.fit(
    (X_train, y_train), (X_val, y_val),
    epochs=50,
    early_stopping_patience=5,
    checkpoint_dir="./checkpoints",
)

print(history.epochs_run, history.stopped_early, history.best_epoch)
print(history.val_metrics[-1].metrics)  # {'accuracy': 0.94, 'f1_macro': 0.93, 'confusion_matrix_0_0': ..., ...}
```

`TorchMLModel` and `TFMLModel` are provided (`from tensor_optix.ml_core import TorchMLModel, TFMLModel`); each wraps an `nn.Module` / `tf.keras.Model` respectively. There is no JAX/Flax adapter yet.

## Data input

`fit()`, `evaluate()`, and `predict()` all accept either form, auto-detected:

```python
# Raw (X, y) array pair, simplest case
model.fit((X_train, y_train), (X_val, y_val), epochs=20)

# A native Dataset/DataLoader (Torch) or tf.data.Dataset (TF)
model.fit(train_dataset, val_dataset, epochs=20, batch_size=64)
```

## Metrics

`metrics="auto"` (the default) detects task type from the target dtype. Integer/class-index targets get classification metrics, float targets get regression metrics:

```python
model = TorchMLModel(net, loss_fn=nn.CrossEntropyLoss())             # metrics="auto"
model = TorchMLModel(net, loss_fn=nn.MSELoss(), metrics="regression") # explicit
model = TorchMLModel(net, loss_fn=nn.CrossEntropyLoss(),
                      metrics=[my_custom_metric])                     # custom callables
```

| `metrics=` | Computed keys |
|---|---|
| `"classification"` | `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, per-class `precision_<i>`/`recall_<i>`/`f1_<i>`, flattened `confusion_matrix_<i>_<j>` |
| `"regression"` | `mae`, `rmse`, `r2` |
| a list of `(y_true, y_pred) -> float` callables | one key per callable, named by `__name__` |

Every call to `evaluate()` returns an `MLMetrics`: `primary_score` (higher-is-better; `accuracy` for classification, `r2` for regression, `-loss` for custom metrics), `loss`, `task_type`, the full `metrics` dict, and `epoch`. This is the value checkpointing and early stopping compare against.

## Checkpointing and early stopping

Pass `checkpoint_dir` to `fit()` to persist the best validation checkpoint via the same `CheckpointRegistry` the RL core uses. `fit()` restores the best checkpoint's weights automatically before returning, regardless of whether training ran to completion or stopped early:

```python
history = model.fit(
    (X_train, y_train), (X_val, y_val),
    epochs=100,
    early_stopping_patience=10,   # stop after 10 epochs with no new best
    checkpoint_dir="./checkpoints",
)
```

`early_stopping_patience` works even without `checkpoint_dir` set. "New best" is tracked independently of whether checkpoints are being persisted to disk. `checkpoint_dir` only controls whether that best state is saved and restorable later.

Convergence/plateau detection reuses `BackoffScheduler` from the RL core, with defaults re-tuned for per-epoch validation curves instead of the RL defaults. Per-epoch validation curves are typically smoother and evaluated far less often than per-episode RL reward. See [`tensor_optix.ml_core.trainer.ML_BACKOFF_DEFAULTS`](../reference/ml-core.md) for the exact values.

## Hyperparameter tuning

If your model overrides `default_param_bounds` (the same mechanism `MLAgent`/RL agents use), `Trainer` activates `SPSAOptimizer` automatically, tuning hyperparameters once per epoch from the validation score:

```python
class MyModel(TorchMLModel):
    default_param_bounds = {"learning_rate": (1e-5, 1e-2), "weight_decay": (0.0, 1e-2)}
```

`TorchMLModel`/`TFMLModel` both already set this for `learning_rate`/`weight_decay` by default.

## What's not here yet

- No multi-GPU / distributed training.
- No data augmentation hooks. Apply augmentation in your `Dataset`/`tf.data.Dataset` pipeline before it reaches `fit()`.
- No pretrained-backbone / transfer-learning helpers. Load/freeze layers on your `nn.Module`/`tf.keras.Model` yourself before wrapping it.
- No learning-rate schedules beyond SPSA's online tuning.
- No JAX/Flax adapter.

## Reference

`BaseMLModel`, `Trainer`, `MLMetrics`, `resolve_metrics()`, and the adapters: [ml_core reference](../reference/ml-core.md).

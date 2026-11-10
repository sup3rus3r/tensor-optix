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

`TorchMLModel`, `TFMLModel`, and `FlaxMLModel` are provided (`from tensor_optix.ml_core import TorchMLModel, TFMLModel, FlaxMLModel`); each wraps an `nn.Module` / `tf.keras.Model` / `flax.nnx.Module` respectively.

```python
from flax import nnx
import jax.numpy as jnp
import optax
from tensor_optix.ml_core import FlaxMLModel

class MLP(nnx.Module):
    def __init__(self, din, dh, dout, rngs):
        self.fc1 = nnx.Linear(din, dh, rngs=rngs)
        self.fc2 = nnx.Linear(dh, dout, rngs=rngs)
    def __call__(self, x):
        return self.fc2(jnp.tanh(self.fc1(x)))

def cross_entropy(pred, y):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))

model = FlaxMLModel(MLP(784, 256, 10, nnx.Rngs(0)), loss_fn=cross_entropy, learning_rate=1e-3)
history = model.fit((X_train, y_train), (X_val, y_val), epochs=50, early_stopping_patience=5)
```

`FlaxMLModel.iter_batches()` only accepts a raw `(X, y)` array pair -- there's no Dataset/DataLoader equivalent in this codebase's JAX dependency set.

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

## Learning-rate schedules

Pass `lr_schedule` to `fit()` for a per-epoch learning-rate schedule, independent of SPSA:

```python
from tensor_optix.ml_core import StepLR, CosineAnnealingLR, WarmupLR

model.fit((X_train, y_train), (X_val, y_val), epochs=50,
          lr_schedule=CosineAnnealingLR(base_lr=1e-3, epochs=50))

model.fit((X_train, y_train), (X_val, y_val), epochs=50,
          lr_schedule=StepLR(base_lr=1e-3, step_size=10, gamma=0.5))

# Linear warmup for 5 epochs, then cosine decay over the remaining 45
model.fit((X_train, y_train), (X_val, y_val), epochs=50,
          lr_schedule=WarmupLR(base_lr=1e-3, warmup_epochs=5,
                                after=CosineAnnealingLR(base_lr=1e-3, epochs=45)))
```

`lr_schedule` and SPSA are mutually exclusive for `learning_rate` within a single `fit()` call: when `lr_schedule` is set, `Trainer` drops `learning_rate` from the bounds it hands `SPSAOptimizer` for that run (a schedule and a score-driven probe both mutating the same hyperparameter every epoch would fight each other). `weight_decay` and any other SPSA-tunable hyperparameter are unaffected. A schedule is any `callable(epoch: int) -> float`, so writing a custom one is a plain function.

## Data augmentation

```python
def augment(X_batch, y_batch):
    # Torch: X_batch is a tensor; TF: X_batch is a numpy array
    return X_batch + torch.randn_like(X_batch) * 0.05, y_batch

model.fit((X_train, y_train), (X_val, y_val), epochs=20, augment=augment)
```

`augment` is a `callable(X_batch, y_batch) -> (X_batch, y_batch)` applied to every training batch, never to validation batches. It only takes effect when `train_data` is a raw `(X, y)` array pair — if you pass a native `Dataset`/`DataLoader`/`tf.data.Dataset`, build augmentation into that pipeline instead (a `transform` on the `Dataset`, or `.map()` on the `tf.data.Dataset`), which is where augmentation naturally belongs for those input forms.

## Pretrained backbones and transfer learning

```python
from tensor_optix.ml_core import load_pretrained_backbone, freeze_layers, unfreeze_layers
import torch.nn as nn

backbone = load_pretrained_backbone("resnet18", framework="torch", pretrained=True, freeze=True)
model = nn.Sequential(backbone, nn.Linear(512, 10))  # resnet18's head is already stripped to nn.Identity()

ml_model = TorchMLModel(model, loss_fn=nn.CrossEntropyLoss())
ml_model.fit((X_train, y_train), epochs=10)  # only the new head trains while backbone is frozen

unfreeze_layers(backbone)  # fine-tune the whole thing afterward
```

`load_pretrained_backbone(name, framework="torch"|"tf", pretrained=True, freeze=False, input_shape=None)` loads a backbone from `torchvision.models` (Torch) or `tf.keras.applications` (TF) with its classifier head removed, ready for a new head. Only a small set of common architectures is supported by name (see [reference](../reference/ml-core.md)); anything else, build it with `torchvision.models`/`tf.keras.applications` directly and pass it straight to `TorchMLModel`/`TFMLModel`.

`freeze_layers(model, layers=None)`/`unfreeze_layers(model, layers=None)` work on any `nn.Module` or `tf.keras.Model`, not just backbones from `load_pretrained_backbone()`. `layers=None` freezes/unfreezes the whole model; pass submodule names (Torch, from `model.named_modules()`) or layer names (TF, from `layer.name`) to target specific parts. Freezing works before *or* after wrapping in `TorchMLModel`/`TFMLModel` -- a frozen parameter accumulates no gradient and is never updated, whether or not it was already in the optimizer's param group when `fit()` starts.

## What's not here yet

- No multi-GPU / distributed training.

## Reference

`BaseMLModel`, `Trainer`, `MLMetrics`, `resolve_metrics()`, and the adapters: [ml_core reference](../reference/ml-core.md).

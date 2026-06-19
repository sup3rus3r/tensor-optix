# Train any PyTorch model (ML mode)

tensor-optix's loop isn't RL-specific. Pass any `nn.Module` and a PyTorch `Dataset` or `DataLoader` to `Optimizer`, and the same SPSA hyperparameter tuning, rollback-on-degradation, convergence detection, and checkpointing apply automatically - the loop treats one mini-batch as one "episode."

```python
import tensor_optix as optix
import torch.nn as nn

# Supervised - loss auto-detected from dataset
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = optix.Optimizer(model, train_dataset)
opt.run()

# Explicit loss
opt = optix.Optimizer(model, train_dataset, loss="cross_entropy")
opt = optix.Optimizer(model, train_dataset, loss="mse")

# Autoencoder
opt = optix.Optimizer(autoencoder, train_dataset, loss="reconstruction")

# VAE - model must return (reconstruction, mu, logvar)
opt = optix.Optimizer(vae, train_dataset, loss="vae")

# Contrastive (SimCLR) - dataset yields (view1, view2) pairs
opt = optix.Optimizer(encoder, pairs_dataset, loss="contrastive")

# DataLoader works too
opt = optix.Optimizer(model, DataLoader(dataset, batch_size=64))

# Custom loss
opt = optix.Optimizer(model, dataset, loss=nn.HuberLoss())
opt.run()
```

`Optimizer` detects ML mode automatically: if `agent` is an `nn.Module` (not a `BaseAgent`) and `env` is a `Dataset`/`DataLoader`, it builds an `MLAgent` wrapping your model and loss, and a `DatasetPipeline` wrapping your dataset, instead of an RL pipeline.

## Available `loss=` strings

| String | Criterion | Use case |
|---|---|---|
| `"auto"` | detected from data | default |
| `"cross_entropy"` | `CrossEntropyLoss` | multi-class classification |
| `"bce"` | `BCEWithLogitsLoss` | binary classification |
| `"mse"` | `MSELoss` | regression |
| `"mae"` | `L1Loss` | regression, outlier-robust |
| `"huber"` | `HuberLoss` | regression, very outlier-robust |
| `"reconstruction"` | MSE(output, input) | autoencoders |
| `"vae"` | ELBO: reconstruction + KL | variational autoencoders |
| `"contrastive"` | NT-Xent | SimCLR-style contrastive learning |
| `"cosine"` | `CosineEmbeddingLoss` | embedding / similarity tasks |

Any `nn.Module` or callable can be passed directly as `loss=` instead of a string.

`loss="auto"` samples a few items from the dataset and infers the right loss: single-tensor items → `"reconstruction"`; paired same-shape tensors → `"contrastive"`; integer targets → `"cross_entropy"`; float targets in `{0,1}` → `"bce"`; float targets otherwise → `"mse"`.

## Save and load

```python
agent = optix.Optimizer(model, dataset, loss="cross_entropy")
# Checkpoints written automatically to checkpoint_dir
# Load back:
from tensor_optix.ml import MLAgent
ml_agent.load_weights("checkpoint.pt")
```

`MLAgent.is_on_policy = True` - each mini-batch is independent data, so rollback on validation degradation is always safe (unlike off-policy RL agents with replay buffers).

## Reference

`MLAgent`, `DatasetPipeline`, and the loss registry internals: [ML mode reference](../reference/ml.md).

# tensor_optix.ml

Adapts the training loop to general (non-RL) ML tasks: supervised, autoencoder, VAE, and contrastive learning.

## MLAgent

```python
class MLAgent(BaseAgent):
    """
    BaseAgent adapter for any nn.Module training task.

    Wraps any nn.Module and gives it the full tensor-optix feature set:
      - SPSA online hyperparameter tuning
      - Rollback on validation degradation
      - Convergence / plateau detection
      - Checkpointing (save / load weights)
      - Weight averaging (SWA)

    Works for: supervised classification/regression, unsupervised
    autoencoders/VAE, self-supervised contrastive learning (SimCLR), or
    anything with a loss.

    Parameters
    ----------
    model:    Any nn.Module.
    loss_fn:  Instantiated loss nn.Module (build via loss_registry.resolve_loss()).
    learning_rate: Initial Adam learning rate. Tunable online via SPSA.
    weight_decay:  Adam L2 regularisation. Tunable online via SPSA.
    max_grad_norm: Gradient clipping norm. None disables.
    device: "auto", "cpu", or "cuda".
    """

    is_on_policy = True  # safe for rollback - each batch is independent data
    default_param_bounds = {"learning_rate": (1e-5, 1e-2), "weight_decay": (0.0, 1e-2)}
    default_log_params = ["learning_rate"]

    def act(self, observation):
        """Inference - returns raw model output as a numpy array."""

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        One gradient update on the batch packed into episode_data.

        Mapping from EpisodeData fields:
          observations  → model input  (X)
          actions       → target       (y, or X again for reconstruction/VAE)

        For unsupervised tasks the pipeline sets actions = observations so
        the loss receives the original input as the target automatically.
        """
```

## DatasetPipeline

```python
class DatasetPipeline(BasePipeline):
    """
    Adapts a PyTorch Dataset or DataLoader to BasePipeline.

    Each "episode" is one mini-batch. Loops through the dataset indefinitely
    - the loop controller decides when to stop, never the pipeline.

    Mapping to EpisodeData:
        observations  → input tensor X
        actions       → target tensor y, OR X again for unsupervised tasks
        rewards       → [-loss] placeholder filled by MLAgent.learn()
        terminated    → [True] - one "episode" per batch.
        truncated     → [False]

    Parameters
    ----------
    dataset:    Dataset or DataLoader. A Dataset is wrapped in a DataLoader
                automatically using batch_size and shuffle.
    batch_size: Batch size when building a DataLoader from a Dataset.
    shuffle:    Whether to shuffle each epoch. Default True.
    loss_key:   When an unsupervised key ("reconstruction", "vae",
                "contrastive"), sets actions = observations automatically.
    num_workers, pin_memory: DataLoader passthrough.
    """
```

## Loss registry

```python
def resolve_loss(loss, dataset=None, n_samples: int = 64) -> nn.Module:
    """
    Return an instantiated loss nn.Module.

    loss: "auto", a loss key string, or an nn.Module instance.
    dataset: required when loss="auto" - used to sample items for detection.
    """

def available_losses() -> str:
    """Human-readable list of all supported loss strings."""
```

Auto-detection rules (`loss="auto"`), in order:

1. Items are single tensors (no label) → `"reconstruction"`
2. Items are pairs `(x1, x2)` of same shape → `"contrastive"`
3. Targets are int/long → `"cross_entropy"`
4. Targets are float, values in `{0, 1}` → `"bce"`
5. Targets are float, any range → `"mse"`

| String | Criterion | Use case |
|---|---|---|
| `"cross_entropy"` | `nn.CrossEntropyLoss` | multi-class classification |
| `"bce"` | `nn.BCEWithLogitsLoss` | binary classification |
| `"mse"` | `nn.MSELoss` | regression |
| `"mae"` | `nn.L1Loss` | regression, outlier-robust |
| `"huber"` | `nn.HuberLoss` | regression, very outlier-robust |
| `"cosine"` | `nn.CosineEmbeddingLoss` | embedding / similarity |
| `"reconstruction"` | MSE(output, input) | autoencoders |
| `"vae"` | ELBO (reconstruction + KL) | VAEs - model must return `(reconstruction, mu, logvar)` |
| `"contrastive"` | NT-Xent | SimCLR - dataset yields `(x1, x2)` pairs |

Any `nn.Module` or callable can be passed directly as `loss=` instead of a string.

See [Train any PyTorch model (ML mode)](../guides/ml-mode.md) for the simplified `Optimizer(model, dataset, ...)` entry point.

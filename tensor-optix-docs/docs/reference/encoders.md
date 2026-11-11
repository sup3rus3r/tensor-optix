# tensor_optix.encoders

Observation encoders that let `make_agent()` build agents for non-flat observation spaces. Every network `make_agent()` constructs sits on top of an encoder mapping a raw observation to a flat feature vector.

| Observation space | Encoder | Output width |
|---|---|---|
| flat 1-D `Box` | identity (pass-through - network dimensions identical to a hand-built MLP) | `shape[0]` |
| image `Box` `(H, W, C)`, `C ∈ {1, 3, 4}` | CNN: Nature-DQN stack for `min(H, W) ≥ 36`, a smaller 3×3 stack down to 12 | `feature_dim` (256) |
| `Dict` of the above | one sub-encoder per key, outputs concatenated | sum of the parts |

Anything else (`MultiDiscrete`, `Tuple`, `MultiBinary`, nested `Dict`, other Box ranks) raises `NotImplementedError` - build the agent manually.

## Torch

```python
from tensor_optix.encoders import (
    EncodedHead, build_torch_encoder, encoder_output_dim, is_flat_box, is_image_box,
)

def build_torch_encoder(space, feature_dim: int = 256) -> nn.Module:
    """
    Build a fresh nn.Module mapping an observation from `space` to a flat
    [batch, encoder_output_dim(space, feature_dim)] feature tensor.
    Each call returns an independently-initialised module - actor and
    critics never share encoder weights.
    """

def encoder_output_dim(space, feature_dim: int = 256) -> int:
    """The flat width build_torch_encoder() will produce, without building it."""

class EncodedHead(nn.Module):
    """encoder followed by head; exposes .encoder and .head separately."""
    def __init__(self, encoder, head): ...
```

## TensorFlow

```python
from tensor_optix.encoders.obs_encoder_tf import build_tf_encoder

def build_tf_encoder(space, feature_dim: int = 256, input_name: str = None): ...
```

Same architecture constants as the Torch builder.

## How agents use encoders

- **PPO family (on-policy).** The encoder is part of the actor/critic and trains end-to-end through the usual minibatch updates.
- **SAC / TD3.** Observations are encoded with the actor's encoder *before* entering the replay buffer, so the buffer always stores flat `float32` vectors and the critics only ever see encoded features. The encoding happens once at collection time; replayed Q-loss gradients do not flow back into the encoder.
- **DQN / Rainbow.** A single Q-network serves both `act()` and the Bellman update, so the buffer stores the raw (possibly `Dict`) observation and the network's own `EncodedHead` runs on every call - trained end-to-end.

Helpers for batching flat/image/`Dict` observations uniformly live in `tensor_optix.core.obs_utils` (`to_batch_numpy`, `to_torch_batch`, `to_tf_batch`, `batch_index`).

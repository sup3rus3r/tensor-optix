"""
tensor_optix.encoders — observation encoders for non-flat observation spaces.

make_agent() builds every network (actor/critic/Q-network) on top of an
encoder that maps a raw observation to a flat feature vector. For the
existing flat 1-D Box case this is IdentityEncoder, a pass-through that
reproduces today's exact behavior. CNNEncoder and DictEncoder extend this
to images and structured (Dict) observations without touching any existing
flat-obs code path.

Off-policy agents (SAC, TD3, DQN, Rainbow) encode the observation to a flat
vector BEFORE pushing into PrioritizedReplayBuffer, so the buffer itself
never changes -- it always stores plain float32 arrays. This means those
agents' encoders are NOT end-to-end trainable through replayed Q-loss
gradients (the encoding happens once, at collection time, and the detached
result is what gets replayed). On-policy agents (PPO, GaussianPPO) never
replay, so their encoder trains end-to-end normally through ordinary
minibatch gradient steps -- no special handling needed there.

Each network (actor, critic1, critic2, ...) gets its OWN encoder instance
via build_torch_encoder(), matching the existing pattern where actor/critic
are already fully separate nn.Module instances with no weight sharing.
"""
from __future__ import annotations

from typing import Callable, Dict as TDict, Union

# Small, Nature-DQN-style conv stack, tuned for Atari-scale (84x84+)
# inputs. Shared configuration so the Torch and TF builders
# (obs_encoder.py / obs_encoder_tf.py) agree on the same architecture --
# exported (no leading underscore) since obs_encoder_tf.py imports these
# directly rather than redefining them.
CNN_CHANNELS = (32, 64, 64)
CNN_KERNELS = (8, 4, 3)
CNN_STRIDES = (4, 2, 1)
# Smaller conv stack for inputs below Nature-DQN's working range -- the
# 8/4/3-kernel, 4/2/1-stride stack above shrinks spatial dims below the
# next kernel size well before the third layer on anything smaller than
# roughly 36x36, which raises a cryptic "kernel size can't be greater
# than actual input size" error deep inside PyTorch/Keras instead of a
# clear one. This smaller stack (3x3 kernels, stride 1, with 2x2 maxpool
# downsampling instead of strided convs) works down to about 12x12.
SMALL_CNN_CHANNELS = (16, 32)
SMALL_CNN_KERNEL = 3
# Minimum spatial dimension (the smaller of H, W) required for each stack.
MIN_SIZE_FOR_NATURE_STACK = 36
MIN_SIZE_FOR_SMALL_STACK = 12
ACCEPTED_IMAGE_CHANNELS = (1, 3, 4)


def is_image_box(space) -> bool:
    """A Box is treated as an image when it is rank-3 with a plausible
    channel count in the last axis (H, W, C) -- the standard Gymnasium
    image-observation convention."""
    from gymnasium.spaces import Box

    return (
        isinstance(space, Box)
        and len(space.shape) == 3
        and space.shape[-1] in ACCEPTED_IMAGE_CHANNELS
    )


def is_flat_box(space) -> bool:
    from gymnasium.spaces import Box

    return isinstance(space, Box) and len(space.shape) == 1


# ---------------------------------------------------------------------------
# Torch encoders
# ---------------------------------------------------------------------------
#
# IdentityEncoder / CNNEncoder / DictEncoder are not separate classes here --
# build_torch_encoder() returns the right nn.Module directly (an identity
# pass-through, a small CNN stack, or a per-key Dict concatenation) based on
# the observation_space it's given, matching this codebase's lazy-import
# convention (torch is only imported inside functions, never at module
# level). Use build_torch_encoder()/encoder_output_dim() directly; there is
# no separate class to instantiate.


def _lazy_nn_module_base():
    """Returns torch.nn.Module, imported lazily. EncodedHead needs to
    subclass nn.Module at class-definition time, which forces the import
    here rather than inside a function body -- still scoped to this single
    call site, not a module-level import."""
    import torch.nn as nn
    return nn.Module


class EncodedHead(_lazy_nn_module_base()):
    """
    Composes an observation encoder with a downstream MLP head: a single
    nn.Module computing head(encoder(observation)).

    Used wherever a network must accept a raw (possibly non-flat)
    observation directly -- e.g. every actor, and every on-policy critic,
    since those always see raw observations from act()/learn(), never
    pre-encoded replay-buffer samples. Off-policy critics (SAC/TD3 critic1/
    critic2) do NOT use this -- they're built as plain head-only MLPs on
    encoder_output_dim + act_dim, since by the time they run, the
    observation has already been encoded to a flat vector by the time it
    reached the replay buffer (see core/obs_utils.py's module docstring).
    """

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


def encoder_output_dim(space, feature_dim: int = 256) -> int:
    """
    The flat feature width build_torch_encoder() will produce for `space`,
    without building any network. Used by factory.py to size the next
    layer (e.g. SAC/TD3 critic's obs+action concat width) before the actual
    encoder instance exists.

    IdentityEncoder (flat Box): output width == space.shape[0], exactly
    matching today's obs_dim -- this is the invariant that keeps the
    existing flat-Box path's network dimensions bit-identical.
    """
    from gymnasium.spaces import Box, Dict as GymDict

    if is_flat_box(space):
        return int(space.shape[0])
    if is_image_box(space):
        return feature_dim
    if isinstance(space, GymDict):
        return sum(encoder_output_dim(sub, feature_dim) for sub in space.spaces.values())
    raise NotImplementedError(
        f"encoder_output_dim() has no rule for observation space type "
        f"{type(space).__name__} with shape "
        f"{getattr(space, 'shape', None)}. Build the agent manually."
    )


def build_torch_encoder(space, feature_dim: int = 256):
    """
    Build a fresh nn.Module mapping an observation from `space` to a flat
    [batch, encoder_output_dim(space, feature_dim)] feature tensor.

    Call this once per network that needs an encoder (actor, critic1,
    critic2, ...) -- each call returns an independently-initialized module,
    matching the existing no-weight-sharing pattern between actor/critic.
    """
    import torch
    import torch.nn as nn
    from gymnasium.spaces import Dict as GymDict

    if is_flat_box(space):
        class _Identity(nn.Module):
            def forward(self, x):
                return x
        return _Identity()

    if is_image_box(space):
        h, w, c = space.shape
        min_side = min(h, w)
        layers = []
        in_ch = c
        if min_side >= MIN_SIZE_FOR_NATURE_STACK:
            for out_ch, k, s in zip(CNN_CHANNELS, CNN_KERNELS, CNN_STRIDES):
                layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s), nn.ReLU()]
                in_ch = out_ch
        elif min_side >= MIN_SIZE_FOR_SMALL_STACK:
            for out_ch in SMALL_CNN_CHANNELS:
                layers += [
                    nn.Conv2d(in_ch, out_ch, kernel_size=SMALL_CNN_KERNEL, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
                in_ch = out_ch
        else:
            raise NotImplementedError(
                f"build_torch_encoder() has no CNN stack small enough for "
                f"image observations with spatial size {h}x{w} (minimum "
                f"supported side is {MIN_SIZE_FOR_SMALL_STACK}). Build a "
                f"custom encoder and pass it via make_agent(obs_encoder=...)."
            )
        conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = conv(dummy).numel()

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv
                self.proj = nn.Linear(conv_out_dim, feature_dim)

            def forward(self, x):
                # x: [batch, H, W, C] (Gymnasium's image convention) ->
                # [batch, C, H, W] for Conv2d.
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = self.conv(x)
                x = x.flatten(start_dim=1)
                return self.proj(x)

        return _CNN()

    if isinstance(space, GymDict):
        sub_encoders = {key: build_torch_encoder(sub, feature_dim) for key, sub in space.spaces.items()}
        keys = list(space.spaces.keys())

        class _Dict(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub_encoders = nn.ModuleDict(sub_encoders)
                self._keys = keys

            def forward(self, x: TDict[str, "torch.Tensor"]):
                parts = [self.sub_encoders[k](x[k]) for k in self._keys]
                return torch.cat(parts, dim=-1)

        return _Dict()

    raise NotImplementedError(
        f"build_torch_encoder() has no rule for observation space type "
        f"{type(space).__name__} with shape {getattr(space, 'shape', None)}. "
        f"Build the agent manually."
    )

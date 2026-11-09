"""
tensor_optix.core.obs_utils — shared observation batching/tensoring helper.

Every agent's act()/learn() previously called torch.as_tensor(np.atleast_2d(
observation), ...) or np.array(episode_data.observations, dtype=np.float32)
directly, which only works for a flat numeric array. to_batch() generalizes
this to also accept:

  - a single image array / list of image arrays (any rank Box)
  - a dict / list of dicts (Dict observation space)

while reproducing today's exact output for the flat-array case (verified by
tests/test_improvements/test_09_obs_encoders.py's IdentityEncoder regression
test) -- this is the mechanism that keeps the existing flat-Box agent code
path bit-identical after this change.

The encoder module (tensor_optix.encoders) is responsible for turning
whatever to_batch() produces into a flat feature vector; to_batch() itself
never flattens or encodes -- it only stacks/batches/casts to a tensor.
"""
from __future__ import annotations

from typing import Any, List, Union

import numpy as np


def _stack_leaf(values: List[Any], dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def to_batch_numpy(observation_or_list: Union[Any, List[Any]]) -> Any:
    """
    Normalize a single observation or a list of observations into a batched
    numpy structure:

      - flat/image array, or list of such arrays -> np.ndarray, shape
        [batch, ...] (a single observation is wrapped to batch size 1,
        exactly matching today's np.atleast_2d behavior for flat arrays).
      - dict, or list of dicts -> dict[str, np.ndarray], each value stacked
        per-key the same way.

    This function never casts dtype beyond float32 for numeric leaves --
    image uint8 arrays are cast to float32 here too (scaling to [0, 1] is
    the encoder's job, consistent with build_torch_encoder's CNN branch
    dividing by 255.0 itself).
    """
    is_list = isinstance(observation_or_list, list)
    sample = observation_or_list[0] if is_list else observation_or_list

    if isinstance(sample, dict):
        items = observation_or_list if is_list else [observation_or_list]
        keys = sample.keys()
        return {k: _stack_leaf([item[k] for item in items]) for k in keys}

    items = observation_or_list if is_list else [observation_or_list]
    return _stack_leaf(items)


def to_torch_batch(observation_or_list, device):
    """to_batch_numpy() result, converted to torch tensor(s) on `device`.
    Dict observations become dict[str, Tensor]; everything else becomes a
    single Tensor -- exactly what build_torch_encoder()'s forward()
    expects for each observation-space branch."""
    import torch

    batched = to_batch_numpy(observation_or_list)
    if isinstance(batched, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batched.items()}
    return torch.as_tensor(batched, dtype=torch.float32).to(device)


def already_batched_to_torch(value, device):
    """Cast an ALREADY-batched numpy structure (an ndarray of shape
    [B, ...], or a dict of such arrays) straight to torch tensor(s), with
    no re-batching/wrapping. Use this for data that is already a rollout
    minibatch slice (e.g. make_minibatches()'s output) -- to_torch_batch()
    would incorrectly treat a [B, ...] array as a single unbatched
    observation and wrap it with an extra leading dim."""
    import torch

    if isinstance(value, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in value.items()}
    return torch.as_tensor(value, dtype=torch.float32).to(device)


def batch_index(value, t):
    """Index a single timestep t out of an already-batched structure: a
    plain ndarray of shape [T, ...] is indexed as value[t] (today's exact
    behavior); a dict of such arrays (Dict observation space, from
    to_batch_numpy()) is indexed per-key, returning a single-timestep dict
    observation -- exactly the shape a replay buffer's push() or an
    encoder's forward() expects for one transition."""
    if isinstance(value, dict):
        return {k: v[t] for k, v in value.items()}
    return value[t]


def to_tf_batch(observation_or_list):
    """to_batch_numpy() result, converted to tf.Tensor(s). Dict
    observations become dict[str, Tensor] (matching build_tf_encoder()'s
    functional-API multi-input Model, whose .input is itself a dict of
    named Input layers); everything else becomes a single Tensor."""
    import tensorflow as tf

    batched = to_batch_numpy(observation_or_list)
    if isinstance(batched, dict):
        return {k: tf.cast(v, tf.float32) for k, v in batched.items()}
    return tf.cast(batched, tf.float32)


def already_batched_to_tf(value):
    """Cast an ALREADY-batched numpy structure straight to tf.Tensor(s),
    with no re-batching/wrapping -- the TF counterpart of
    already_batched_to_torch()."""
    import tensorflow as tf

    if isinstance(value, dict):
        return {k: tf.cast(v, tf.float32) for k, v in value.items()}
    return tf.cast(value, tf.float32)

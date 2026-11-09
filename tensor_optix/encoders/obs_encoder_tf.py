"""
tensor_optix.encoders.obs_encoder_tf — TF/Keras counterpart of obs_encoder.py.

Same architecture and semantics as the Torch builders in obs_encoder.py:
IdentityEncoder for flat Box (bit-identical to today's behavior), a small
Nature-DQN-style CNN for image Box, and a per-key sub-encoder + concat for
Dict. See obs_encoder.py's module docstring for the off-policy
encode-before-buffer tradeoff, which applies identically here.
"""
from __future__ import annotations

from tensor_optix.encoders.obs_encoder import (
    CNN_CHANNELS,
    CNN_KERNELS,
    CNN_STRIDES,
    SMALL_CNN_CHANNELS,
    SMALL_CNN_KERNEL,
    MIN_SIZE_FOR_NATURE_STACK,
    MIN_SIZE_FOR_SMALL_STACK,
    is_flat_box,
    is_image_box,
)


def build_tf_encoder(space, feature_dim: int = 256, input_name: str = None):
    """
    Build a fresh tf.keras.Model mapping an observation from `space` to a
    flat [batch, encoder_output_dim(space, feature_dim)] feature tensor.

    Call once per network needing an encoder (actor, critic1, critic2, ...),
    mirroring build_torch_encoder()'s one-instance-per-network contract.

    input_name: passed to the Input layer's `name=` for a Dict sub-encoder,
    so the sub-model's input tensor name matches its dict key -- avoids a
    Keras UserWarning about dict input keys not matching tensor names when
    the composed Dict model is later called with a {key: array} dict.
    """
    import tensorflow as tf
    from gymnasium.spaces import Dict as GymDict

    if is_flat_box(space):
        inputs = tf.keras.layers.Input(shape=space.shape, name=input_name)
        return tf.keras.Model(inputs=inputs, outputs=inputs, name="identity_encoder")

    if is_image_box(space):
        h, w, c = space.shape
        min_side = min(h, w)
        inputs = tf.keras.layers.Input(shape=(h, w, c), name=input_name)
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        if min_side >= MIN_SIZE_FOR_NATURE_STACK:
            for out_ch, k, s in zip(CNN_CHANNELS, CNN_KERNELS, CNN_STRIDES):
                x = tf.keras.layers.Conv2D(out_ch, kernel_size=k, strides=s, activation="relu")(x)
        elif min_side >= MIN_SIZE_FOR_SMALL_STACK:
            for out_ch in SMALL_CNN_CHANNELS:
                x = tf.keras.layers.Conv2D(out_ch, kernel_size=SMALL_CNN_KERNEL, padding="same", activation="relu")(x)
                x = tf.keras.layers.MaxPooling2D(2)(x)
        else:
            raise NotImplementedError(
                f"build_tf_encoder() has no CNN stack small enough for "
                f"image observations with spatial size {h}x{w} (minimum "
                f"supported side is {MIN_SIZE_FOR_SMALL_STACK}). Build a "
                f"custom encoder and pass it via make_agent(obs_encoder=...)."
            )
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(feature_dim)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_encoder")

    if isinstance(space, GymDict):
        keys = list(space.spaces.keys())
        sub_models = {k: build_tf_encoder(sub, feature_dim, input_name=k) for k, sub in space.spaces.items()}
        inputs = {k: sub_models[k].input for k in keys}
        parts = [sub_models[k].output for k in keys]
        outputs = tf.keras.layers.Concatenate()(parts) if len(parts) > 1 else parts[0]
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="dict_encoder")

    raise NotImplementedError(
        f"build_tf_encoder() has no rule for observation space type "
        f"{type(space).__name__} with shape {getattr(space, 'shape', None)}. "
        f"Build the agent manually."
    )

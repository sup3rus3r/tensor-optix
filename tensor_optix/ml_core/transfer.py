"""
tensor_optix.ml_core.transfer — pretrained-backbone and transfer-learning
helpers: freeze/unfreeze any Torch or TF model's parameters, and load a
handful of common pretrained backbones with their classifier head removed.

Framework-agnostic at the freeze_layers()/unfreeze_layers() level (dispatch
on model type); load_pretrained_backbone() is necessarily framework-
specific since it goes through torchvision.models / tf.keras.applications.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

# Head attribute name for each supported torchvision architecture family --
# torchvision has no single convention ("fc" for resnet, "classifier" for
# vgg/densenet/mobilenet/efficientnet), so this is an explicit, small,
# honestly-incomplete registry rather than a guess-the-attribute heuristic.
_TORCHVISION_HEAD_ATTR = {
    "resnet18": "fc", "resnet34": "fc", "resnet50": "fc", "resnet101": "fc", "resnet152": "fc",
    "vgg11": "classifier", "vgg13": "classifier", "vgg16": "classifier", "vgg19": "classifier",
    "densenet121": "classifier", "densenet169": "classifier", "densenet201": "classifier",
    "mobilenet_v2": "classifier", "mobilenet_v3_small": "classifier", "mobilenet_v3_large": "classifier",
    "efficientnet_b0": "classifier", "efficientnet_b1": "classifier", "efficientnet_b2": "classifier",
}


def _is_torch_module(model) -> bool:
    try:
        import torch.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def freeze_layers(model, layers: Optional[Sequence[str]] = None) -> None:
    """
    Freeze parameters in-place so they're excluded from gradient updates.

    model: an nn.Module or tf.keras.Model.
    layers: names of submodules (Torch, via model.named_modules()) or
        layers (TF, via model.layers' .name) to freeze. None freezes the
        whole model.
    """
    if _is_torch_module(model):
        _set_torch_trainable(model, layers, trainable=False)
    else:
        _set_tf_trainable(model, layers, trainable=False)


def unfreeze_layers(model, layers: Optional[Sequence[str]] = None) -> None:
    """Inverse of freeze_layers() -- re-enables gradient updates."""
    if _is_torch_module(model):
        _set_torch_trainable(model, layers, trainable=True)
    else:
        _set_tf_trainable(model, layers, trainable=True)


def _set_torch_trainable(model, layers, trainable: bool) -> None:
    if layers is None:
        for p in model.parameters():
            p.requires_grad_(trainable)
        return
    named = dict(model.named_modules())
    for name in layers:
        if name not in named:
            raise ValueError(f"No submodule named '{name}' in model. Available: {list(named)}")
        for p in named[name].parameters():
            p.requires_grad_(trainable)


def _set_tf_trainable(model, layers, trainable: bool) -> None:
    if layers is None:
        model.trainable = trainable
        return
    by_name = {layer.name: layer for layer in model.layers}
    for name in layers:
        if name not in by_name:
            raise ValueError(f"No layer named '{name}' in model. Available: {list(by_name)}")
        by_name[name].trainable = trainable


def load_pretrained_backbone(
    name: str,
    framework: str = "torch",
    pretrained: bool = True,
    freeze: bool = False,
    input_shape: Optional[tuple] = None,
):
    """
    Load a common pretrained backbone with its classifier head removed,
    ready to attach a new head onto for transfer learning.

    name: architecture name. Torch: a torchvision.models function name
        (e.g. "resnet18", "mobilenet_v2"). TF: a tf.keras.applications
        class name (e.g. "MobileNetV2", "ResNet50").
    framework: "torch" or "tf".
    pretrained: load ImageNet weights (Torch) / "imagenet" weights (TF).
        False gives a randomly-initialized backbone of the same architecture.
    freeze: freeze all backbone parameters immediately after loading --
        equivalent to calling freeze_layers(backbone) yourself.
    input_shape: TF only, required by tf.keras.applications (e.g.
        (224, 224, 3)). Ignored for Torch, which is shape-agnostic until
        the first forward pass.

    Returns the backbone with its head removed (Torch: nn.Identity() in
    place of the original head; TF: include_top=False, no head to begin
    with) so you attach your own head on top.
    """
    if framework == "torch":
        return _load_torch_backbone(name, pretrained, freeze)
    if framework == "tf":
        return _load_tf_backbone(name, pretrained, freeze, input_shape)
    raise ValueError(f"framework must be 'torch' or 'tf', got {framework!r}")


def _load_torch_backbone(name: str, pretrained: bool, freeze: bool):
    import torch.nn as nn
    import torchvision.models as tv_models

    if name not in _TORCHVISION_HEAD_ATTR:
        raise ValueError(
            f"Unsupported backbone '{name}'. Supported: {sorted(_TORCHVISION_HEAD_ATTR)}. "
            f"For any other torchvision model, build it yourself via "
            f"torchvision.models.{name}(weights=...) and strip its head manually."
        )
    builder = getattr(tv_models, name)
    weights = "DEFAULT" if pretrained else None
    model = builder(weights=weights)

    head_attr = _TORCHVISION_HEAD_ATTR[name]
    setattr(model, head_attr, nn.Identity())

    if freeze:
        freeze_layers(model)
    return model


def _load_tf_backbone(name: str, pretrained: bool, freeze: bool, input_shape):
    import tensorflow as tf

    if not hasattr(tf.keras.applications, name):
        raise ValueError(
            f"Unsupported backbone '{name}'. See tf.keras.applications for "
            f"available names (e.g. 'ResNet50', 'MobileNetV2', 'EfficientNetB0')."
        )
    builder = getattr(tf.keras.applications, name)
    weights = "imagenet" if pretrained else None
    model = builder(include_top=False, weights=weights, input_shape=input_shape)

    if freeze:
        freeze_layers(model)
    return model

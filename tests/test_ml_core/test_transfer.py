import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.ml_core.transfer import freeze_layers, unfreeze_layers, load_pretrained_backbone
from tensor_optix.ml_core.adapters.torch_model import TorchMLModel


def _build_two_layer_model():
    model = nn.Sequential()
    model.add_module("backbone", nn.Linear(10, 8))
    model.add_module("head", nn.Linear(8, 3))
    return model


def test_freeze_layers_whole_model_sets_all_requires_grad_false():
    model = _build_two_layer_model()
    freeze_layers(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_layers_specific_submodule_only_freezes_that_one():
    model = _build_two_layer_model()
    freeze_layers(model, layers=["backbone"])
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_unfreeze_layers_reverses_freeze():
    model = _build_two_layer_model()
    freeze_layers(model)
    unfreeze_layers(model, layers=["head"])
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_freeze_layers_unknown_submodule_raises():
    model = _build_two_layer_model()
    with pytest.raises(ValueError, match="No submodule named"):
        freeze_layers(model, layers=["nonexistent"])


def test_torch_ml_model_optimizer_only_updates_trainable_params():
    model = _build_two_layer_model()
    freeze_layers(model, layers=["backbone"])

    backbone_before = model.backbone.weight.clone()
    head_before = model.head.weight.clone()

    ml_model = TorchMLModel(model, loss_fn=nn.CrossEntropyLoss(), learning_rate=1e-1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10)).astype(np.float32)
    y = rng.integers(0, 3, 40).astype(np.int64)
    ml_model.fit((X, y), epochs=5, batch_size=8)

    assert torch.allclose(model.backbone.weight, backbone_before)
    assert not torch.allclose(model.head.weight, head_before)


def test_load_pretrained_backbone_torch_resnet18_strips_head():
    backbone = load_pretrained_backbone("resnet18", framework="torch", pretrained=False)
    assert isinstance(backbone.fc, nn.Identity)

    x = torch.randn(2, 3, 64, 64)
    out = backbone(x)
    assert out.shape == (2, 512)  # resnet18's pre-head feature width


def test_load_pretrained_backbone_torch_unsupported_name_raises():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        load_pretrained_backbone("not_a_real_model", framework="torch")


def test_load_pretrained_backbone_bad_framework_raises():
    with pytest.raises(ValueError, match="framework must be"):
        load_pretrained_backbone("resnet18", framework="jax")


def test_load_pretrained_backbone_torch_freeze_true_freezes_all_params():
    backbone = load_pretrained_backbone("resnet18", framework="torch", pretrained=False, freeze=True)
    assert all(not p.requires_grad for p in backbone.parameters())

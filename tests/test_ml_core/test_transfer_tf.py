import pytest

tf = pytest.importorskip("tensorflow")

from tensor_optix.ml_core.transfer import freeze_layers, unfreeze_layers, load_pretrained_backbone


def _build_two_layer_model():
    inputs = tf.keras.Input(shape=(10,))
    backbone_out = tf.keras.layers.Dense(8, name="backbone")(inputs)
    head_out = tf.keras.layers.Dense(3, name="head")(backbone_out)
    return tf.keras.Model(inputs, head_out)


def test_freeze_layers_whole_model_sets_trainable_false():
    model = _build_two_layer_model()
    freeze_layers(model)
    assert model.trainable is False


def test_freeze_layers_specific_layer_only_freezes_that_one():
    model = _build_two_layer_model()
    freeze_layers(model, layers=["backbone"])
    by_name = {layer.name: layer for layer in model.layers}
    assert by_name["backbone"].trainable is False
    assert by_name["head"].trainable is True


def test_unfreeze_layers_reverses_freeze():
    model = _build_two_layer_model()
    freeze_layers(model, layers=["backbone", "head"])
    unfreeze_layers(model, layers=["head"])
    by_name = {layer.name: layer for layer in model.layers}
    assert by_name["backbone"].trainable is False
    assert by_name["head"].trainable is True


def test_freeze_layers_unknown_layer_raises():
    model = _build_two_layer_model()
    with pytest.raises(ValueError, match="No layer named"):
        freeze_layers(model, layers=["nonexistent"])


def test_load_pretrained_backbone_tf_mobilenet_has_no_top():
    backbone = load_pretrained_backbone(
        "MobileNetV2", framework="tf", pretrained=False, input_shape=(32, 32, 3),
    )
    import numpy as np
    out = backbone(np.zeros((1, 32, 32, 3), dtype="float32"))
    assert len(out.shape) == 4  # feature map, not flattened class logits


def test_load_pretrained_backbone_tf_unsupported_name_raises():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        load_pretrained_backbone("NotARealModel", framework="tf", input_shape=(32, 32, 3))


def test_load_pretrained_backbone_tf_freeze_true_freezes_model():
    backbone = load_pretrained_backbone(
        "MobileNetV2", framework="tf", pretrained=False, freeze=True, input_shape=(32, 32, 3),
    )
    assert backbone.trainable is False

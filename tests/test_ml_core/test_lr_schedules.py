import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.ml_core.adapters.torch_model import TorchMLModel
from tensor_optix.ml_core.lr_schedules import StepLR, CosineAnnealingLR, WarmupLR


def _make_separable_dataset(n=200, n_features=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    for i in range(n):
        X[i, y[i]] += 3.0
    return X, y


def _build_model(n_features=10, n_classes=3):
    return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, n_classes))


def test_step_lr_decays_at_boundaries():
    schedule = StepLR(base_lr=1.0, step_size=10, gamma=0.1)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(10) == pytest.approx(0.1)
    assert schedule(20) == pytest.approx(0.01)


def test_cosine_annealing_lr_endpoints_and_monotonic_decrease():
    schedule = CosineAnnealingLR(base_lr=1.0, epochs=10, min_lr=0.0)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(10) == pytest.approx(0.0, abs=1e-9)
    values = [schedule(e) for e in range(11)]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_warmup_lr_ramps_linearly_then_delegates():
    after = CosineAnnealingLR(base_lr=1.0, epochs=5)
    schedule = WarmupLR(base_lr=1.0, warmup_epochs=5, after=after)
    assert schedule(0) == pytest.approx(0.2)
    assert schedule(4) == pytest.approx(1.0)
    assert schedule(5) == pytest.approx(after(0))
    assert schedule(9) == pytest.approx(after(4))


def test_warmup_lr_without_after_holds_base_lr():
    schedule = WarmupLR(base_lr=1.0, warmup_epochs=3, after=None)
    assert schedule(0) == pytest.approx(1.0 / 3)
    assert schedule(2) == pytest.approx(1.0)
    assert schedule(10) == pytest.approx(1.0)


def test_fit_with_lr_schedule_actually_changes_model_lr_each_epoch():
    X, y = _make_separable_dataset(n=60)
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss(), learning_rate=1.0)

    seen_lrs = []
    original_train_step = model.train_step

    def recording_train_step(batch):
        seen_lrs.append(model._optimizer.param_groups[0]["lr"])
        return original_train_step(batch)

    model.train_step = recording_train_step
    schedule = StepLR(base_lr=1.0, step_size=2, gamma=0.5)
    model.fit((X, y), epochs=6, batch_size=20, lr_schedule=schedule)

    # Each epoch runs 3 batches (60/20); the recorded lr at the start of
    # each epoch's batches must match the schedule's value for that epoch.
    epoch0_lr = seen_lrs[0]
    epoch2_lr = seen_lrs[6]
    epoch4_lr = seen_lrs[12]
    assert epoch0_lr == pytest.approx(1.0)
    assert epoch2_lr == pytest.approx(0.5)
    assert epoch4_lr == pytest.approx(0.25)


def test_lr_schedule_disables_spsa_tuning_of_learning_rate():
    X, y = _make_separable_dataset(n=40)
    X_train, y_train = X[:30], y[:30]
    X_val, y_val = X[30:], y[30:]

    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss(), learning_rate=1e-2)
    schedule = StepLR(base_lr=1e-2, step_size=5, gamma=0.5)
    model.fit(
        (X_train, y_train), (X_val, y_val),
        epochs=10, batch_size=10, lr_schedule=schedule,
    )

    # The schedule is deterministic given the epoch alone; if SPSA were
    # still tuning learning_rate alongside it, the final lr would not
    # cleanly equal schedule(9).
    final_lr = model._optimizer.param_groups[0]["lr"]
    assert final_lr == pytest.approx(schedule(9))

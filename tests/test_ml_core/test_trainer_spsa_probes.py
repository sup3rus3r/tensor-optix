"""
Regression tests for the GitHub issue "ML trainer does not pass SPSA probe
status to the ConvergenceMonitor": epochs trained under an SPSA probe are not
progress readings, COOLING does not widen the tuner in the ML trainer, and
fit(spsa=...) controls the tuner.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.ml_core.adapters.torch_model import TorchMLModel
from tensor_optix.ml_core.lr_schedules import CosineAnnealingLR
from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = rng.integers(0, 2, 60).astype(np.int64)
    return (X[:40], y[:40]), (X[40:], y[40:])


def _model(monkeypatch, score_fn):
    """TorchMLModel whose validation primary_score is score_fn(epoch_index)."""
    model = TorchMLModel(nn.Sequential(nn.Linear(4, 2)), loss_fn=nn.CrossEntropyLoss())
    original = model.evaluate
    calls = {"n": 0}

    def scripted(data):
        m = original(data)
        m.primary_score = float(score_fn(calls["n"]))
        calls["n"] += 1
        return m

    monkeypatch.setattr(model, "evaluate", scripted)
    return model


class CountingTuner(BaseOptimizer):
    """Never probes; counts lifecycle calls."""

    def __init__(self):
        self.suggests = self.coolings = self.plateaus = self.improvements = 0

    def suggest(self, current_hyperparams, metrics_history):
        self.suggests += 1
        return current_hyperparams.copy()

    def on_improvement(self, metrics):
        self.improvements += 1

    def on_cooling(self, metrics_history):
        self.coolings += 1

    def on_plateau(self, metrics_history):
        self.plateaus += 1


def test_probe_epochs_are_not_fed_to_the_monitor(monkeypatch):
    train, val = _data()
    model = _model(monkeypatch, lambda e: 0.5 + 0.01 * e)
    history = model.fit(train, val, epochs=12, batch_size=8)
    # SPSA cycle: epoch e's suggest() sets the hyperparameters of epoch e+1:
    # idle -> plus probe -> minus probe -> update. Epoch 0 runs on the base values.
    assert history.probe_epochs == [1, 2, 4, 5, 7, 8, 10, 11]
    assert [int(r.at) for r in history.convergence] == [0, 3, 6, 9]
    assert len(history.val_metrics) == 12                        # every epoch is still evaluated


def test_probe_epochs_cannot_create_a_false_plateau(monkeypatch):
    """The issue's scenario: probe epochs score much worse (a probed learning
    rate hurts the model). Base epochs keep improving. The monitor must keep
    seeing progress -- no COOLING, no DORMANT."""
    probe = {1, 2}

    def score(e):
        base = 0.5 + 0.002 * e
        return base - 0.3 if (e % 3) in probe else base

    train, val = _data()
    model = _model(monkeypatch, score)
    events = {"plateau": 0}

    class CB:
        def on_plateau(self, metrics):
            events["plateau"] += 1

    history = model.fit(train, val, epochs=90, batch_size=8, callbacks=[CB()], stop_on_convergence=True)
    assert history.stop_reason == "epochs"
    assert events["plateau"] == 0
    assert not any(r.entered_cooling or r.entered_dormant for r in history.convergence)


def test_probe_epochs_cannot_fake_progress(monkeypatch):
    """The mirror case: only probe epochs look better. A genuinely flat run must
    still converge (probe scores are not new ground)."""
    def score(e):
        return 0.9 if (e % 3) in (1, 2) else 0.5 + 1e-4 * np.sin(e)

    train, val = _data()
    model = _model(monkeypatch, score)
    history = model.fit(train, val, epochs=300, batch_size=8)       # fixed budget, convergence off
    assert not any(r.improved and r.at > 0 for r in history.convergence)


def test_cooling_does_not_widen_the_tuner_in_the_ml_trainer(monkeypatch):
    train, val = _data()
    model = _model(monkeypatch, lambda e: 0.7)
    tuner = CountingTuner()
    events = {"plateau": 0}

    class CB:
        def on_plateau(self, metrics):
            events["plateau"] += 1

    history = model.fit(train, val, batch_size=8, spsa=tuner, callbacks=[CB()])
    assert history.stop_reason == "converged"
    assert events["plateau"] == 1                                 # COOLING still reported to callbacks
    assert tuner.coolings == 0 and tuner.plateaus == 0            # ...but never widens the tuner
    assert tuner.suggests == history.epochs_run
    assert history.probe_epochs == []


def test_spsa_false_disables_tuning(monkeypatch):
    train, val = _data()
    model = _model(monkeypatch, lambda e: 0.5 + 0.01 * e)
    before = dict(model.get_hyperparams().params)
    history = model.fit(train, val, epochs=9, batch_size=8, spsa=False)
    assert model.get_hyperparams().params == before
    assert history.probe_epochs == [] and len(history.convergence) == 9


def test_custom_spsa_instance_is_used(monkeypatch):
    train, val = _data()
    model = _model(monkeypatch, lambda e: 0.5 + 0.01 * e)
    tuner = SPSAOptimizer(param_bounds={"weight_decay": (0.0, 1e-2)}, perturbation_scale=0.01)
    lr = model.get_hyperparams().params["learning_rate"]
    history = model.fit(train, val, epochs=6, batch_size=8, spsa=tuner)
    assert history.probe_epochs == [1, 2, 4, 5]
    assert model.get_hyperparams().params["learning_rate"] == lr     # not in this tuner's bounds


def test_bad_spsa_arguments():
    train, val = _data()
    model = TorchMLModel(nn.Sequential(nn.Linear(4, 2)), loss_fn=nn.CrossEntropyLoss())
    with pytest.raises(TypeError):
        model.fit(train, val, epochs=2, spsa="yes please")
    with pytest.raises(ValueError, match="fight"):
        model.fit(train, val, epochs=2, lr_schedule=CosineAnnealingLR(1e-3, epochs=2),
                  spsa=SPSAOptimizer(param_bounds={"learning_rate": (1e-5, 1e-2)}))

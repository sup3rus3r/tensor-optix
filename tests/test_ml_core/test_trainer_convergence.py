"""
ML Trainer acting on the ConvergenceMonitor: open-ended training
(epochs=None), stopping at DORMANT, the optional max_epochs cap, stop
reasons, and once-on-entry callbacks.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.core.convergence import ConvergenceMonitor, ConvergenceRegime
from tensor_optix.ml_core.adapters.torch_model import TorchMLModel


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int64)
    return (X[:40], y[:40]), (X[40:], y[40:])


def _model_with_scores(monkeypatch, scores):
    """A real TorchMLModel whose validation primary_score follows a script
    (last value repeated), independent of what the network learns."""
    model = TorchMLModel(nn.Sequential(nn.Linear(4, 2)), loss_fn=nn.CrossEntropyLoss())
    original = model.evaluate
    calls = {"n": 0}

    def scripted(data):
        m = original(data)
        i = min(calls["n"], len(scores) - 1)
        calls["n"] += 1
        m.primary_score = float(scores[i])
        return m

    monkeypatch.setattr(model, "evaluate", scripted)
    return model


def test_open_ended_fit_stops_when_converged(monkeypatch):
    train, val = _data()
    # rises for 10 epochs, then flat forever
    model = _model_with_scores(monkeypatch, [0.5 + 0.03 * i for i in range(10)] + [0.8])
    history = model.fit(train, val, epochs=None, batch_size=8)
    assert history.stop_reason == "converged" and history.stopped_early
    # The default SPSA tuner probes 2 of every 3 epochs and probe epochs are
    # not progress readings, so the monitor's 12-reading warm-up spans ~36
    # epochs before DORMANT can be judged.
    assert 20 <= history.epochs_run <= 80
    assert len(history.convergence) + len(history.probe_epochs) == history.epochs_run
    assert history.convergence[-1].entered_dormant
    # without the tuner every epoch is a reading and the run stops sooner
    model = _model_with_scores(monkeypatch, [0.5 + 0.03 * i for i in range(10)] + [0.8])
    fast = model.fit(train, val, epochs=None, batch_size=8, spsa=False)
    assert fast.stop_reason == "converged" and fast.epochs_run < history.epochs_run
    assert len(fast.convergence) == fast.epochs_run


def test_default_fit_without_validation_converges_on_training_loss():
    train, _ = _data()
    model = TorchMLModel(nn.Sequential(nn.Linear(4, 2)), loss_fn=nn.CrossEntropyLoss(),
                         learning_rate=1e-2)
    history = model.fit(train, batch_size=8, max_epochs=2000)     # no epochs given
    assert history.stop_reason == "converged"
    assert len(history.convergence) == history.epochs_run
    assert not history.val_metrics


def test_fixed_epochs_keep_the_old_behaviour(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])   # converges fast
    history = model.fit(train, val, epochs=60, batch_size=8)
    assert history.stop_reason == "epochs" and history.epochs_run == 60


def test_best_weights_are_restored_even_on_an_exception(monkeypatch, tmp_path):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.9, 0.6, 0.6, 0.6])
    loads = {"n": 0}
    original_load = model.load_weights

    def counting_load(path):
        loads["n"] += 1
        return original_load(path)

    monkeypatch.setattr(model, "load_weights", counting_load)

    class Boom:
        def on_epoch_end(self, epoch, loss):
            if epoch == 3:
                raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        model.fit(train, val, epochs=10, batch_size=8, callbacks=[Boom()],
                  checkpoint_dir=str(tmp_path))
    assert loads["n"] == 1


def test_a_monitor_can_be_reused_across_fits(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    mon = ConvergenceMonitor(timescales=(3, 6, 12), cooling_patience=3, dormant_patience=8)
    model.fit(train, val, epochs=5, batch_size=8, convergence=mon)
    model.fit(train, val, epochs=5, batch_size=8, convergence=mon)   # must not raise


def test_open_ended_fit_with_convergence_off_needs_another_stop():
    train, val = _data()
    model = TorchMLModel(nn.Sequential(nn.Linear(4, 2)), loss_fn=nn.CrossEntropyLoss())
    with pytest.raises(ValueError, match="never stop"):
        model.fit(train, val, epochs=None, stop_on_convergence=False)


def test_max_epochs_is_an_optional_cap(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.1 * i for i in range(1000)])   # never converges
    history = model.fit(train, val, epochs=None, max_epochs=12, batch_size=8)
    assert history.stop_reason == "max_epochs" and history.epochs_run == 12


def test_fixed_budget_still_runs_to_the_end_when_not_converged(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.1 * i for i in range(100)])
    history = model.fit(train, val, epochs=8, batch_size=8)
    assert history.stop_reason == "epochs" and history.epochs_run == 8


def test_convergence_stops_a_fixed_budget_run_early_when_asked(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    history = model.fit(train, val, epochs=200, batch_size=8, stop_on_convergence=True)
    assert history.stop_reason == "converged" and history.epochs_run < 200


def test_stop_on_convergence_false_ignores_dormant(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    history = model.fit(train, val, epochs=40, batch_size=8, stop_on_convergence=False)
    assert history.stop_reason == "epochs" and history.epochs_run == 40


def test_explicit_early_stopping_still_wins(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    history = model.fit(train, val, epochs=None, early_stopping_patience=3, batch_size=8)
    assert history.stop_reason == "early_stopping"


def test_custom_monitor_and_callbacks_fire_once(monkeypatch):
    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    events = {"plateau": 0, "dormant": 0, "no_improvement": 0, "convergence": []}

    class CB:
        def on_plateau(self, metrics):
            events["plateau"] += 1

        def on_dormant(self, epoch):
            events["dormant"] += 1

        def on_no_improvement(self, metrics):
            events["no_improvement"] += 1

        def on_convergence(self, epoch, report):
            events["convergence"].append(report.regime)

    mon = ConvergenceMonitor(timescales=(3, 6, 12), cooling_patience=3, dormant_patience=8)
    history = model.fit(train, val, epochs=None, batch_size=8, callbacks=[CB()], convergence=mon)
    assert history.stop_reason == "converged"
    assert events["plateau"] == 1 and events["dormant"] == 1
    assert events["no_improvement"] >= 8
    assert ConvergenceRegime.IMPROVING in events["convergence"]


def test_fixed_length_lr_schedule_warns_on_open_ended_runs(monkeypatch):
    from tensor_optix.ml_core.lr_schedules import CosineAnnealingLR

    train, val = _data()
    model = _model_with_scores(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    with pytest.warns(UserWarning, match="fixed length"):
        model.fit(train, val, epochs=None, batch_size=8,
                  lr_schedule=CosineAnnealingLR(1e-2, epochs=5))

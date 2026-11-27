"""
Multi-layer perceptron family, trained with tensor-optix's own ML loop
(``TorchMLModel.fit``): open-ended training that the ConvergenceMonitor ends
when the inner validation loss stops making progress, with the best
checkpoint restored. Online SPSA tuning is off so every fold trains the same
way (a discovery trial measures the family, not the tuner).

Scores come from ``predict()`` (raw outputs -> probabilities here), never from
``TorchMLModel.evaluate()``, which reduces multi-output predictions to hard
labels.

Determinism: the network is built and trained inside
``torch.random.fork_rng`` with a seed derived from the fold, on the CPU, so
results are reproducible and the caller's global torch RNG is untouched.

Importing this module raises ImportError when torch is missing; the family
registry then simply does not offer it.
"""
from __future__ import annotations

import tempfile
from typing import Any, Dict, List

import numpy as np
import torch

from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register, softmax
from ..preprocess import PipelineOptions

VAL_FRACTION = 0.15
MAX_EPOCHS = 300                     # discovery's own spend cap per fit (the library loop has none by default)


def _net(p: int, out: int, hidden: int, layers: int, dropout: float) -> "torch.nn.Module":
    mods: List[torch.nn.Module] = []
    width = p
    for _ in range(layers):
        mods += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
        if dropout > 0:
            mods.append(torch.nn.Dropout(dropout))
        width = hidden
    mods.append(torch.nn.Linear(width, out))
    return torch.nn.Sequential(*mods)


class MLP(Estimator):
    def __init__(self, cfg: Dict[str, Any], seed: int, task: str, k: int):
        self.cfg = dict(cfg)
        self.seed = int(seed) % (2 ** 31)
        self.task = task
        self.k = int(k)
        self.stop_reason = "not_fitted"
        self.epochs_run = 0

    def _split(self, n: int, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n_val = max(1, int(round(VAL_FRACTION * n)))
        if self.task in CLASSIFICATION:
            val = []
            for c in np.unique(y):
                rows = np.flatnonzero(y == c)
                take = int(round(len(rows) * VAL_FRACTION))
                if take >= 1 and len(rows) - take >= 1:
                    val.append(rng.choice(rows, take, replace=False))
            val = np.concatenate(val) if val else rng.choice(n, n_val, replace=False)
        else:
            val = rng.choice(n, n_val, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[val] = True
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    def fit(self, X, y, X_val=None, y_val=None):
        from torch.utils.data import DataLoader, TensorDataset
        from ....ml_core.adapters.torch_model import TorchMLModel

        X = np.asarray(X, dtype=np.float32)
        classification = self.task in CLASSIFICATION
        if classification:
            yt = np.asarray(y).astype(np.int64)
        else:
            yf = np.asarray(y, dtype=np.float64)
            self._mu, self._sd = float(yf.mean()), float(yf.std()) or 1.0
            yt = ((yf - self._mu) / self._sd).astype(np.float32).reshape(-1, 1)
        if X_val is None:
            tr, va = self._split(len(X), np.asarray(y))
            Xtr, ytr, Xva, yva = X[tr], yt[tr], X[va], yt[va]
        else:
            Xtr, ytr = X, yt
            Xva = np.asarray(X_val, dtype=np.float32)
            yva = (np.asarray(y_val).astype(np.int64) if classification else
                   ((np.asarray(y_val, dtype=np.float64) - self._mu) / self._sd).astype(np.float32).reshape(-1, 1))
        out = self.k if classification else 1
        bs = int(self.cfg.get("batch_size", min(256, max(32, len(Xtr) // 20))))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _net(X.shape[1], out, int(self.cfg.get("hidden", 64)), int(self.cfg.get("layers", 2)),
                       float(self.cfg.get("dropout", 0.0)))
            loss = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
            self._model = TorchMLModel(net, loss, learning_rate=float(self.cfg.get("lr", 1e-3)),
                                       weight_decay=float(self.cfg.get("weight_decay", 1e-4)),
                                       device="cpu", metrics=[])           # primary score = -val loss
            gen = torch.Generator().manual_seed(self.seed)
            loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                                batch_size=bs, shuffle=True, generator=gen)
            with tempfile.TemporaryDirectory(prefix="optix_discovery_mlp_") as ckpt:
                hist = self._model.fit(loader, (Xva, yva), epochs=None, max_epochs=int(self.cfg.get(
                    "max_epochs", MAX_EPOCHS)), batch_size=bs, checkpoint_dir=ckpt, spsa=False)
        self.stop_reason = hist.stop_reason or "unknown"
        self.epochs_run = hist.epochs_run
        return self

    def _raw(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def predict_proba(self, X):
        return softmax(self._raw(X))

    def predict(self, X):
        z = self._raw(X).reshape(-1)
        return z * self._sd + self._mu


def _configs(n: int, p: int, task: str) -> List[Dict[str, Any]]:
    return [{"hidden": 64, "layers": 2, "weight_decay": 1e-4},
            {"hidden": 128, "layers": 2, "weight_decay": 1e-3, "dropout": 0.1}]


register(Family(
    name="mlp", cost_rank=6, tasks=CLASSIFICATION + REGRESSION,
    make=lambda cfg, seed, task, k: MLP(cfg, seed, task, k),
    configs=_configs,
    preprocessing=PipelineOptions(scale="standard", categorical="onehot", high_cardinality="target"),
    landmark=False, candidate=True,
    description="Feed-forward neural network (2 hidden layers), trained by tensor-optix until convergence.",
))

"""
Cross-validation splitters and the confirmation holdout (pure numpy).

The splitter decides whether an evaluation can be trusted: a random split of
grouped rows (several rows per customer / patient) or of time-ordered rows
lets the model see near-copies of the test rows during training, and the
score looks better than it will ever be in use. Discovery picks the splitter
from the detected structure; ``BuiltSolution`` reuses the same one.

All splitters are deterministic given their seed and use discovery's
independent random streams (never numpy's global RNG).

Guarantees (property-tested):
  * train and test of a fold never overlap; folds' test sets partition the
    evaluated rows (except forward chaining, whose earliest block is train-only);
  * group splitters never put rows of one group on both sides;
  * forward chaining only ever trains on rows that come (at least ``gap``
    rows) before every test row;
  * stratified splitters keep each class's count per fold within 1 of even.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rng import rng_for

Fold = Tuple[np.ndarray, np.ndarray]


def _check_n_splits(n_splits: int) -> None:
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")


def _codes(a: np.ndarray) -> np.ndarray:
    if a.dtype == object:
        a = np.array([repr(v) for v in a], dtype=object)
    return np.unique(a, return_inverse=True)[1].reshape(-1)


@dataclass
class KFold:
    n_splits: int = 5
    seed: int = 0
    kind: str = field(default="kfold", init=False)

    def split(self, n: int, *, y=None, groups=None, time=None, repeat: int = 0) -> List[Fold]:
        _check_n_splits(self.n_splits)
        if n < self.n_splits:
            raise ValueError(f"cannot make {self.n_splits} folds from {n} rows")
        perm = rng_for(self.seed, "kfold", repeat).permutation(n)
        parts = np.array_split(perm, self.n_splits)
        return [(np.sort(np.concatenate(parts[:k] + parts[k + 1:])), np.sort(parts[k]))
                for k in range(self.n_splits)]


@dataclass
class StratifiedKFold:
    """Each class's rows are spread over the folds as evenly as possible.
    For regression pass quantile-binned targets as ``y``."""
    n_splits: int = 5
    seed: int = 0
    kind: str = field(default="stratified", init=False)

    def split(self, n: int, *, y=None, groups=None, time=None, repeat: int = 0) -> List[Fold]:
        _check_n_splits(self.n_splits)
        if y is None:
            raise ValueError("StratifiedKFold needs y")
        y = _codes(np.asarray(y))
        rng = rng_for(self.seed, "stratified", repeat)
        fold_of = np.empty(n, dtype=np.int64)
        offset = int(rng.integers(0, self.n_splits))
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            fold_of[idx] = (np.arange(len(idx)) + offset) % self.n_splits
            offset = (offset + len(idx)) % self.n_splits      # keeps fold sizes balanced overall
        return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(self.n_splits)]


@dataclass
class GroupKFold:
    """Whole groups go to one fold; folds balanced by row count (greedy, largest
    group first, random tie-break)."""
    n_splits: int = 5
    seed: int = 0
    kind: str = field(default="group", init=False)

    def split(self, n: int, *, y=None, groups=None, time=None, repeat: int = 0) -> List[Fold]:
        _check_n_splits(self.n_splits)
        if groups is None:
            raise ValueError("GroupKFold needs groups")
        g = _codes(np.asarray(groups))
        n_groups = int(g.max()) + 1
        if n_groups < self.n_splits:
            raise ValueError(f"cannot make {self.n_splits} group folds from {n_groups} groups")
        sizes = np.bincount(g, minlength=n_groups)
        rng = rng_for(self.seed, "group", repeat)
        order = rng.permutation(n_groups)
        order = order[np.argsort(-sizes[order], kind="stable")]
        load = np.zeros(self.n_splits, dtype=np.int64)
        fold_of_group = np.empty(n_groups, dtype=np.int64)
        for gi in order:
            k = int(np.argmin(load))
            fold_of_group[gi] = k
            load[k] += sizes[gi]
        fold_of = fold_of_group[g]
        return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(self.n_splits)]


@dataclass
class StratifiedGroupKFold:
    """Whole groups per fold, chosen greedily to keep class proportions per fold
    close to the overall ones."""
    n_splits: int = 5
    seed: int = 0
    kind: str = field(default="stratified_group", init=False)

    def split(self, n: int, *, y=None, groups=None, time=None, repeat: int = 0) -> List[Fold]:
        _check_n_splits(self.n_splits)
        if groups is None or y is None:
            raise ValueError("StratifiedGroupKFold needs y and groups")
        g = _codes(np.asarray(groups))
        yc = _codes(np.asarray(y))
        n_groups, n_cls = int(g.max()) + 1, int(yc.max()) + 1
        if n_groups < self.n_splits:
            raise ValueError(f"cannot make {self.n_splits} group folds from {n_groups} groups")
        gc = np.zeros((n_groups, n_cls), dtype=np.int64)
        np.add.at(gc, (g, yc), 1)
        sizes = gc.sum(axis=1)
        target = gc.sum(axis=0) / self.n_splits
        rng = rng_for(self.seed, "stratified_group", repeat)
        order = rng.permutation(n_groups)
        order = order[np.argsort(-sizes[order], kind="stable")]
        fc = np.zeros((self.n_splits, n_cls), dtype=np.float64)
        fold_of_group = np.empty(n_groups, dtype=np.int64)
        scale = np.maximum(target, 1.0)
        for gi in order:
            cand = fc + gc[gi]
            # Increase in imbalance caused by adding this group to each fold
            # (absolute cost would favour already-full folds).
            cost = ((((cand - target) / scale) ** 2).sum(axis=1)
                    - (((fc - target) / scale) ** 2).sum(axis=1))
            cost += 1e-9 * cand.sum(axis=1)               # tie-break: emptier fold
            k = int(np.argmin(cost))
            fold_of_group[gi] = k
            fc[k] += gc[gi]
        fold_of = fold_of_group[g]
        return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(self.n_splits)]


def _time_order(n: int, time) -> np.ndarray:
    if time is None:
        return np.arange(n)
    t = np.asarray(time)
    if t.dtype.kind == "M":
        t = t.astype("datetime64[ns]").astype(np.int64)
    return np.argsort(t, kind="stable")


@dataclass
class ForwardChaining:
    """
    Expanding-window time-series CV: the data (ordered by ``time``, else by row
    order) is cut into ``n_splits`` consecutive test blocks after an initial
    ``min_train_frac`` of rows; each block is predicted from everything before
    it, minus a ``gap`` of rows right before the block (so autocorrelated
    neighbours cannot leak). With ``groups``, a group is never on both sides:
    groups are ordered by their first appearance and moved as units.
    """
    n_splits: int = 5
    gap: int = 0
    min_train_frac: float = 0.3
    seed: int = 0                                     # unused (deterministic); kept for a uniform API
    kind: str = field(default="forward", init=False)

    def split(self, n: int, *, y=None, groups=None, time=None, repeat: int = 0) -> List[Fold]:
        _check_n_splits(self.n_splits)
        if not 0.0 < self.min_train_frac < 1.0:
            raise ValueError("min_train_frac must be in (0, 1)")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")
        order = _time_order(n, time)
        if groups is not None:
            return self._grouped(n, order, _codes(np.asarray(groups)))
        start = int(np.floor(self.min_train_frac * n))
        blocks = np.array_split(np.arange(start, n), self.n_splits)
        folds = []
        for b in blocks:
            if len(b) == 0:
                continue
            train_end = b[0] - self.gap
            if train_end <= 0:
                raise ValueError("gap / min_train_frac leave no training rows for the first block")
            folds.append((np.sort(order[:train_end]), np.sort(order[b])))
        return folds

    def _grouped(self, n: int, order: np.ndarray, g: np.ndarray) -> List[Fold]:
        first_pos = np.full(int(g.max()) + 1, n, dtype=np.int64)
        np.minimum.at(first_pos, g[order], np.arange(n))
        unit_order = np.argsort(first_pos, kind="stable")
        n_units = len(unit_order)
        start = int(np.floor(self.min_train_frac * n_units))
        blocks = np.array_split(np.arange(start, n_units), self.n_splits)
        folds = []
        for b in blocks:
            if len(b) == 0:
                continue
            train_units = unit_order[:max(0, b[0] - self.gap)]
            test_units = unit_order[b]
            if len(train_units) == 0:
                raise ValueError("not enough groups before the first test block")
            folds.append((np.flatnonzero(np.isin(g, train_units)), np.flatnonzero(np.isin(g, test_units))))
        return folds


_SPLITTERS = {
    "kfold": KFold, "stratified": StratifiedKFold, "group": GroupKFold,
    "stratified_group": StratifiedGroupKFold, "forward": ForwardChaining,
}


@dataclass
class RepeatedSplitter:
    """``n_repeats`` independent repetitions of ``base`` (forward chaining is
    deterministic, so it is never repeated). Fold ids are ``r{rep}f{k}``."""
    base: Any
    n_repeats: int = 1

    def folds(self, n: int, *, y=None, groups=None, time=None,
              repeats: Optional[Sequence[int]] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        reps = list(range(self.n_repeats)) if repeats is None else list(repeats)
        if getattr(self.base, "kind", "") == "forward":
            reps = reps[:1]
        out = []
        for r in reps:
            for k, (tr, te) in enumerate(self.base.split(n, y=y, groups=groups, time=time, repeat=r)):
                out.append((f"r{r}f{k}", tr, te))
        return out


def make_splitter(kind: str, **params) -> Any:
    try:
        cls = _SPLITTERS[kind]
    except KeyError:
        raise ValueError(f"unknown splitter kind {kind!r}; known: {sorted(_SPLITTERS)}") from None
    return cls(**params)


def splitter_to_dict(s: Any) -> Dict[str, Any]:
    d = {k: v for k, v in s.__dict__.items() if k != "kind"}
    return {"kind": s.kind, **d}


def splitter_from_dict(d: Dict[str, Any]) -> Any:
    d = dict(d)
    return make_splitter(d.pop("kind"), **d)


# ---------------------------------------------------------------------------
# Confirmation holdout
# ---------------------------------------------------------------------------

def carve_holdout(n: int, *, structure: str, frac: float, seed: int, y=None, groups=None,
                  time=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split rows into (development, holdout) *before* any model selection, using
    the same structure the CV uses:

    * ``iid``: stratified on ``y`` when given;
    * ``grouped``: whole groups, chosen at random;
    * ``temporal``: the most recent ``frac`` of rows by time;
    * ``grouped_temporal``: the groups that appear last.
    """
    if not 0.0 < frac < 0.5:
        raise ValueError("holdout frac must be in (0, 0.5)")
    rng = rng_for(seed, "holdout")
    k = max(1, int(round(frac * n)))
    if structure == "temporal":
        order = _time_order(n, time)
        hold = np.sort(order[n - k:])
    elif structure in ("grouped", "grouped_temporal"):
        if groups is None:
            raise ValueError(f"structure {structure!r} needs groups")
        g = _codes(np.asarray(groups))
        n_groups = int(g.max()) + 1
        if structure == "grouped":
            unit_order = rng.permutation(n_groups)
        else:
            order = _time_order(n, time)
            first_pos = np.full(n_groups, n, dtype=np.int64)
            np.minimum.at(first_pos, g[order], np.arange(n))
            unit_order = np.argsort(-first_pos, kind="stable")      # latest groups first
        sizes = np.bincount(g, minlength=n_groups)
        chosen, total = [], 0
        for u in unit_order:
            if total >= k:
                break
            chosen.append(u)
            total += sizes[u]
        if len(chosen) >= n_groups:
            raise ValueError("too few groups to carve a holdout")
        hold = np.flatnonzero(np.isin(g, chosen))
    elif structure == "iid":
        if y is not None:
            yc = _codes(np.asarray(y))
            picks = []
            for c in np.unique(yc):
                idx = np.flatnonzero(yc == c)
                m = int(round(frac * len(idx)))
                if m > 0:
                    picks.append(rng.choice(idx, m, replace=False))
            hold = np.sort(np.concatenate(picks)) if picks else np.sort(rng.choice(n, k, replace=False))
        else:
            hold = np.sort(rng.choice(n, k, replace=False))
    else:
        raise ValueError(f"unknown structure {structure!r}")
    dev = np.setdiff1d(np.arange(n), hold, assume_unique=True)
    return dev, hold

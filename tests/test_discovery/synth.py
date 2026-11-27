"""
Synthetic datasets with known ground truth for discovery's detector tests.
Every generator is seeded and returns plain dicts of numpy columns.
"""
import numpy as np

from tensor_optix.discovery.ml.families.base import sigmoid


def _obj(values):
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def grouped(n_groups=120, per_group=10, effect=2.0, seed=0, regression=False):
    """Rows repeat per customer; an unobserved per-customer effect drives the
    target. The customer id (as a categorical) lets a random split memorise it."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    u = rng.normal(scale=effect, size=n_groups)[g]
    x1 = rng.normal(size=len(g))
    x2 = rng.normal(size=len(g))
    z = 0.8 * x1 - 0.5 * x2 + u
    y = z + rng.normal(scale=0.5, size=len(g)) if regression else (rng.random(len(g)) < sigmoid(z)).astype(int)
    cols = {"customer_id": _obj([f"cust{v:04d}" for v in g]), "x1": x1, "x2": x2}
    return cols, y


def iid_with_repeating_ids(n=1200, seed=0):
    """Ids repeat but carry no signal: must NOT be flagged as grouped."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 120, n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < sigmoid(0.8 * x1 - 0.5 * x2)).astype(int)
    return {"customer_id": _obj([f"cust{v:04d}" for v in ids]), "x1": x1, "x2": x2}, y


def drifting(n=1500, drift=3.0, seed=0, shuffle_rows=True):
    """Regression target whose level wanders over time as a random walk (not
    observable from the features and not extrapolable from the date). Rows
    shuffled, time in a datetime column."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = rng.normal(size=n)
    level = np.cumsum(rng.normal(scale=drift / np.sqrt(n) * 3, size=n))
    y = x + level + rng.normal(scale=0.5, size=n)
    when = np.datetime64("2023-01-01") + t.astype("timedelta64[D]")
    perm = rng.permutation(n) if shuffle_rows else t
    return {"when": when[perm], "x": x[perm]}, y[perm]


def iid_with_dates(n=1500, seed=0):
    """A date column that carries no time structure: must NOT be flagged."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)
    when = np.datetime64("2023-01-01") + rng.integers(0, 1000, n).astype("timedelta64[D]")
    return {"when": when, "x": x}, y


def sorted_by_target(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (rng.random(n) < sigmoid(x[:, 0])).astype(int)
    order = np.argsort(y, kind="stable")
    return {f"x{i}": x[order, i] for i in range(3)}, y[order]


def leaky(n=1500, seed=0):
    """'refund_flag' is (almost) the target itself; 'income' is a strong but
    legitimate feature; 'signup_id' a sequential id."""
    rng = np.random.default_rng(seed)
    income = rng.normal(size=n)
    other = rng.normal(size=n)
    y = (rng.random(n) < sigmoid(1.5 * income + 0.5 * other)).astype(int)
    refund = np.where(rng.random(n) < 0.98, y, 1 - y).astype(float)
    return {"income": income, "other": other, "refund_flag": refund, "signup_id": np.arange(n)}, y


def shifted(n=1200, seed=0):
    """Test data where 2 of 8 features shifted."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    T = rng.normal(size=(n, 8))
    T[:, 2] += 0.8
    T[:, 5] *= 2.0
    y = (rng.random(n) < sigmoid(X[:, 0])).astype(int)
    names = [f"f{i}" for i in range(8)]
    return ({k: X[:, i] for i, k in enumerate(names)}, y, {k: T[:, i] for i, k in enumerate(names)})

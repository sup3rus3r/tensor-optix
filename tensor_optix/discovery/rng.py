"""
Deterministic randomness for discovery.

Every random choice (sampling, folds, permutations, model init) draws from a
``numpy.random.Generator`` derived from the user's ``seed`` plus a *named
purpose*, e.g. ``rng_for(seed, "folds", repeat)``. Derivation goes through
``numpy.random.SeedSequence`` with a stable (sha256-based, not Python-hash)
key, so

  * the same seed and purpose always give the same stream, on every platform
    and Python version;
  * different purposes give statistically independent streams (adding a new
    permutation test never shifts the folds of an existing stage).

Discovery never touches numpy's or torch's *global* RNG state.
"""
from __future__ import annotations

import hashlib
from typing import Union

import numpy as np

Key = Union[str, int]


def _key_words(parts) -> list:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return [int.from_bytes(h[i:i + 4], "little") for i in range(0, 16, 4)]


def seed_sequence(seed: int, *purpose: Key) -> np.random.SeedSequence:
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key_words(purpose)))


def rng_for(seed: int, *purpose: Key) -> np.random.Generator:
    """Independent, reproducible generator for one named purpose."""
    return np.random.default_rng(seed_sequence(seed, *purpose))


def int_seed_for(seed: int, *purpose: Key) -> int:
    """A reproducible 31-bit int seed (for libraries that want an int, e.g. torch)."""
    return int(seed_sequence(seed, *purpose).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)

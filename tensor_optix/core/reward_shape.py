"""
Domain-agnostic "breadth" signal derived purely from an episode's reward
stream — no per-domain knowledge required.

Motivation
----------
A single scalar primary_score cannot distinguish a policy that got lucky on
one narrow axis from a policy that is robustly good across many moments in
the episode. Every RL episode has a step-wise reward stream by construction,
so tensor-optix can measure "was this outcome built from one dominant moment
or many similar ones" without any domain-specific rubric.

This measures magnitude concentration (Gini coefficient over |r_t|), never
sign. Sign conventions are entirely domain-defined — some envs are
bonus-style (positive = good), some are cost-style (negative = good, e.g.
penalties/drawdown), some are mixed-sign shaped reward. A step whose
magnitude dominates the episode is structurally "narrow" whether that step
was a huge one-off bonus or a huge one-off cost avoided; using |r_t| makes
the measurement meaningful regardless of which direction counts as "good"
in a given domain.

Typicality is always computed relative to THIS RUN's own historical
distribution of breadth values, never an absolute threshold — this is what
prevents false positives on legitimate sparse/terminal-reward domains (e.g.
a landing bonus that only ever arrives on the terminal step is this domain's
normal, not evidence of a fluke).
"""
from collections import deque
from typing import Deque, List

import numpy as np


def split_episode_returns(values: List[float], dones: List[bool]) -> List[List[float]]:
    """
    Split a window of per-step values into per-completed-episode segments,
    using terminated-OR-truncated boundaries (same convention as
    TorchEvaluator._mean_episode_return / TFEvaluator._mean_episode_return).

    The trailing incomplete segment (no terminal done yet) is dropped — it
    isn't a completed episode and its shape isn't yet meaningful.
    """
    segments: List[List[float]] = []
    current: List[float] = []
    for v, done in zip(values, dones):
        current.append(v)
        if done:
            segments.append(current)
            current = []
    return segments


def _gini(values: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative array. 0 = perfectly uniform
    (spread evenly across steps), 1 = fully concentrated in one step.
    Defined only for non-negative inputs — callers must pass magnitudes.
    """
    n = values.shape[0]
    if n == 0:
        return 0.0
    total = values.sum()
    if total <= 1e-12:
        # No magnitude anywhere in the episode — nothing was concentrated
        # or spread; treat as maximally uniform (no unicorn signal possible).
        return 0.0
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def episode_breadth(rewards: List[float], dones: List[bool]) -> List[float]:
    """
    Returns one breadth value in [0, 1] per completed episode in the window.
    breadth = 1 - Gini(|r_t|) — higher means the episode's outcome was built
    from many steps of similar magnitude; lower means it hinged on one (or a
    few) dominant steps.

    Falls back to a single value derived from the whole window (treated as
    one incomplete "episode") when no episode boundary (done=True) occurs —
    mirrors the same incomplete-window fallback used by
    TorchEvaluator._mean_episode_return.
    """
    segments = split_episode_returns(rewards, dones)
    if not segments:
        if not rewards:
            return []
        segments = [list(rewards)]

    breadths = []
    for seg in segments:
        magnitudes = np.abs(np.asarray(seg, dtype=np.float64))
        breadths.append(1.0 - _gini(magnitudes))
    return breadths


def breadth_typicality(
    current_breadth: float,
    history: Deque[float],
    noise_k: float = 2.0,
    min_samples: int = 5,
) -> float:
    """
    Returns a satisfied-fraction-style score in [0, 1] measuring how typical
    current_breadth is relative to this run's own historical breadth
    distribution — never an absolute threshold.

    Falls back to 1.0 (no penalty) until `min_samples` observations exist,
    matching BackoffScheduler._adaptive_floor()'s min-samples fallback —
    short runs must never be blocked from checkpointing on this signal.

    A breadth reading within noise_k standard deviations of this run's own
    mean breadth scores 1.0 (fully typical). Further below the mean scores
    proportionally lower, floored at 0.0. Breadth *above* the mean never
    incurs a penalty — only anomalously narrow (concentrated) episodes,
    relative to the run's own norm, are discounted.
    """
    if len(history) < min_samples:
        return 1.0

    hist = np.asarray(history, dtype=np.float64)
    mean = float(hist.mean())
    std = float(hist.std())
    if std < 1e-12:
        # No variation in this run's own breadth history — any deviation
        # is meaningful, but with no noise estimate to scale against, don't
        # manufacture a penalty out of numerical noise.
        return 1.0

    floor = noise_k * std
    deficit = mean - current_breadth
    if deficit <= 0:
        return 1.0
    return float(np.clip(1.0 - deficit / floor, 0.0, 1.0))

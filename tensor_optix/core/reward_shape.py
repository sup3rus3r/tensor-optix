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


class EpisodeReturnTracker:
    """
    Accumulates per-episode returns ACROSS windows.

    A pipeline yields fixed-size windows, so an episode usually starts in one
    window and ends in a later one. Summing rewards inside a single window
    only ever sees the tail of such an episode (and nothing at all when the
    episode is longer than the window). The tracker keeps one running sum per
    environment between calls, so every return it reports is the full return
    of an episode, however many windows it spanned.

    update() returns the returns of the episodes that FINISHED in this
    window — an empty list when none did.
    """

    def __init__(self, n_envs: int = 1) -> None:
        self._n_envs = max(1, int(n_envs))
        self._running = [0.0] * self._n_envs

    def update(self, rewards, dones) -> List[float]:
        finished: List[float] = []
        n = self._n_envs
        for t in range(len(rewards)):
            env_i = t % n
            self._running[env_i] += float(rewards[t])
            if dones[t]:
                finished.append(self._running[env_i])
                self._running[env_i] = 0.0
        return finished

    def reset(self) -> None:
        self._running = [0.0] * self._n_envs


EPISODE_RETURNS_KEY = "episode_returns"


def completed_episode_returns(episode_data, n_envs: int = 1):
    """
    Returns of the episodes that finished in this window.

    Pipelines that track episodes across windows publish them in
    EpisodeData.metadata["episode_returns"] (full returns, possibly an empty
    list). For any other pipeline this falls back to window-local sums.
    """
    metadata = getattr(episode_data, "metadata", None) or {}
    tracked = metadata.get(EPISODE_RETURNS_KEY)
    if tracked is not None:
        return [float(r) for r in tracked]
    return [
        float(np.sum(seg))
        for seg in split_episode_returns(episode_data.rewards, episode_data.dones, n_envs)
    ]


def split_episode_returns(values: List[float], dones: List[bool], n_envs: int = 1) -> List[List[float]]:
    """
    Split a window of per-step values into per-completed-episode segments,
    using terminated-OR-truncated boundaries (same convention as
    TorchEvaluator._mean_episode_return / TFEvaluator._mean_episode_return).

    The trailing incomplete segment (no terminal done yet) is dropped — it
    isn't a completed episode and its shape isn't yet meaningful.
    """
    # n_envs > 1: the window interleaves parallel envs as [t * n_envs + env]
    # (VectorBatchPipeline). Each env's steps are split on that env's own
    # dones — walking the flat array would splice different envs' rewards
    # into one "episode".
    n_envs = max(1, int(n_envs))
    segments: List[List[float]] = []
    for env_i in range(n_envs):
        current: List[float] = []
        for t in range(env_i, len(values), n_envs):
            current.append(values[t])
            if dones[t]:
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


def episode_breadth(rewards: List[float], dones: List[bool], n_envs: int = 1) -> List[float]:
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
    segments = split_episode_returns(rewards, dones, n_envs)
    if not segments:
        if not len(rewards):
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

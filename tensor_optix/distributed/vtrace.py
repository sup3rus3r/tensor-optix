"""
tensor_optix.distributed.vtrace — V-trace importance-sampling correction.

Reference: Espeholt et al. 2018 — IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures.
https://arxiv.org/abs/1802.01561  (Section 4)

The actor uses a behaviour policy μ (stale weights) to collect a trajectory
of length T.  The learner has the current policy θ.  The IS ratio
    ρ_t = π_θ(a_t | s_t) / π_μ(a_t | s_t)
corrects for the mismatch.  Two truncations prevent explosive variance:

    ρ̄_t = min(ρ̄, ρ_t)      ← clips IS weight on the TD error
    c̄_t  = min(c̄, ρ_t)      ← clips IS weight on the trace (controls bias/variance)

V-trace target (backward recursion):

    δ_t   = ρ̄_t · (r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t))
    v_T   = V(s_T)                         (boundary condition)
    v_t   = V(s_t) + δ_t + γ · c̄_t · (v_{t+1} − V(s_{t+1}))

Policy-gradient advantage:

    A_t   = ρ̄_t · (r_t + γ · v_{t+1} · (1 − done_t) − V(s_t))

Setting ρ̄ = c̄ = 1 and μ = θ (on-policy) recovers standard GAE with λ = c̄.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_vtrace_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    behaviour_log_probs: np.ndarray,
    current_log_probs: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute V-trace targets and policy-gradient advantages.

    Parameters
    ----------
    rewards              shape [T]    per-step rewards
    values               shape [T+1]  V(s_t) for t = 0..T; values[T] is the
                                      bootstrap value for the state after the
                                      last step in the trajectory
    behaviour_log_probs  shape [T]    log π_μ(a_t | s_t) recorded by the actor
    current_log_probs    shape [T]    log π_θ(a_t | s_t) from the current
                                      learner policy (same or newer than μ)
    dones                shape [T]    bool — True when episode ended at step t
    gamma                float        discount factor
    rho_bar              float        IS weight clip ρ̄ ≥ 1 (paper default 1.0)
    c_bar                float        trace coefficient clip c̄ ≤ ρ̄ (default 1.0)

    Returns
    -------
    vs          shape [T]  V-trace targets for the value regression loss
    advantages  shape [T]  IS-corrected policy-gradient advantages
    """
    T = len(rewards)
    not_done = 1.0 - dones.astype(np.float32)

    # IS ratios ρ_t = exp(log π_θ − log π_μ)
    # Clip log-ratio before exp to prevent overflow / underflow.
    log_rho = current_log_probs - behaviour_log_probs
    rho = np.exp(np.clip(log_rho, -10.0, 10.0))
    rho_clipped = np.minimum(rho_bar, rho)   # ρ̄_t
    c_clipped   = np.minimum(c_bar,   rho)   # c̄_t

    # TD errors: δ_t = ρ̄_t · (r_t + γ · V(s_{t+1}) · (1−done_t) − V(s_t))
    delta = rho_clipped * (
        rewards + gamma * not_done * values[1:] - values[:-1]
    )

    # Backward recursion for V-trace targets.
    # Initialise with acc = v_T − V(s_T) = 0.
    # v_t = V(s_t) + δ_t + γ · c̄_t · (v_{t+1} − V(s_{t+1}))
    #      = V(s_t) + δ_t + γ · c̄_t · acc_{t+1}
    vs = np.empty(T, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(T)):
        acc   = delta[t] + gamma * not_done[t] * c_clipped[t] * acc
        vs[t] = values[t] + acc

    # Policy-gradient advantages: ρ̄_t · (r_t + γ · v_{t+1} · (1−done_t) − V(s_t))
    # vs_next[t] = vs[t+1] for t < T−1; vs_next[T−1] = values[T] (bootstrap).
    vs_next        = np.empty(T, dtype=np.float32)
    vs_next[:-1]   = vs[1:]
    vs_next[-1]    = values[T]
    advantages = rho_clipped * (
        rewards + gamma * not_done * vs_next - values[:-1]
    )

    return vs.astype(np.float32), advantages.astype(np.float32)

"""
RL probes: what the environment is, measured with a random policy before any
agent is trained.

  * spaces            observation / action space types, shapes, bounds; whether
                      make_agent() can build an agent for them
  * random rollout    episode lengths and returns, termination vs truncation,
                      per-step reward distribution (scale, sparsity), observation
                      statistics
  * determinism       same seed + same actions twice -> same trajectory?
  * partial           does the previous observation / action improve prediction
    observability     of the next observation? (paired CV of ridge models with and
                      without history -- discovery's ML statistics)
  * exploration       how rarely a random policy sees a non-default reward

Every probe produces evidence; the rules in ``candidates`` turn it into
which algorithms and variants to try, and the trials decide.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from . import env as E

SPARSE_STEP_FRACTION = 0.01
OBS_SCALE_LIMIT = 10.0
OBS_SPREAD_RATIO = 100.0
REWARD_SCALE_LIMIT = 1000.0


def space_profile(space) -> Dict[str, Any]:
    import gymnasium as gym
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, gym.spaces.Box):
        low, high = np.asarray(space.low, dtype=np.float64), np.asarray(space.high, dtype=np.float64)
        return {"type": "Box", "shape": list(space.shape),
                "bounded": bool(np.all(np.isfinite(low)) and np.all(np.isfinite(high))),
                "low_min": float(np.min(low)) if low.size else None, "high_max": float(np.max(high)) if high.size else None}
    return {"type": type(space).__name__}


def random_rollout(env, n_steps: int, seed: int, max_episodes: int = 1000) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    try:
        env.action_space.seed(int(rng.integers(0, 2 ** 31 - 1)))
    except Exception:
        pass
    obs_list: List[np.ndarray] = []
    next_obs: List[np.ndarray] = []
    actions: List[Any] = []
    rewards: List[float] = []
    ep_returns, ep_lengths, ep_trunc = [], [], []
    first_of_episode: List[bool] = []
    steps = 0
    ep = 0
    while steps < n_steps and ep < max_episodes:
        o = E.reset(env, seed=int(rng.integers(0, 2 ** 31 - 1)))
        ret, length, done, trunc = 0.0, 0, False, False
        first = True
        while not done and steps < n_steps:
            a = env.action_space.sample()
            o2, r, term, tr = E.step(env, a)
            fo, fo2 = E.flat_obs(o), E.flat_obs(o2)
            if fo is not None and fo2 is not None:
                obs_list.append(fo)
                next_obs.append(fo2)
                actions.append(np.atleast_1d(np.asarray(a, dtype=np.float64)))
                first_of_episode.append(first)
            rewards.append(r)
            ret += r
            length += 1
            steps += 1
            first = False
            done = term or tr
            trunc = tr
            o = o2
        if done:
            ep_returns.append(ret)
            ep_lengths.append(length)
            ep_trunc.append(trunc)
        ep += 1
    return {"obs": np.array(obs_list) if obs_list else None, "next_obs": np.array(next_obs) if next_obs else None,
            "actions": np.array(actions) if actions else None, "first": np.array(first_of_episode, dtype=bool),
            "rewards": np.array(rewards, dtype=np.float64), "returns": np.array(ep_returns, dtype=np.float64),
            "lengths": np.array(ep_lengths, dtype=np.float64), "truncated": np.array(ep_trunc, dtype=bool),
            "steps": steps}


def determinism(factory, seed: int, n: int = 50) -> Optional[bool]:
    try:
        trajs = []
        env0 = factory()
        acts = [env0.action_space.sample() for _ in range(n)]
        for _ in range(2):
            env = factory()
            o = E.reset(env, seed=seed)
            traj = [np.asarray(E.flat_obs(o) if E.flat_obs(o) is not None else 0.0)]
            for a in acts:
                o, r, term, tr = E.step(env, a)
                fo = E.flat_obs(o)
                traj.append(np.append(fo if fo is not None else [], r))
                if term or tr:
                    break
            trajs.append(traj)
        a, b = trajs
        return len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    except Exception:
        return None


HISTORY_GAIN_MIN = 0.10


def _quadratic(X: np.ndarray) -> np.ndarray:
    """Columns of X plus all pairwise products and squares (constant columns dropped)."""
    X = X[:, X.std(axis=0) > 1e-12]
    i, j = np.triu_indices(X.shape[1])
    return np.column_stack([X, X[:, i] * X[:, j]])


def history_gain(ro: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    """
    Partial-observability test. Target: the *change* of the observation,
    o_{t+1} - o_t. Models: quadratic-basis ridge on (o_t, a_t) vs on
    (o_t, a_t, o_{t-1}, a_{t-1}), compared with paired 5x2 CV. Statistic per
    fold: the largest R^2 gain over observation dimensions.

    Why the change, not the next observation: positions move little per step,
    so o_{t+1} is predicted almost perfectly from o_t whether or not a velocity
    is observed. The change is a function of the full state; when part of the
    state (e.g. a velocity) is hidden, the change is poorly predicted from o_t
    and the previous observation recovers it. In fully observed environments
    (CartPole, Pendulum, Acrobot, MountainCar, LunarLander) the gain stays
    below ~0.05; in masked CartPole it is ~0.57. Flag when the lower 95% bound
    exceeds ``HISTORY_GAIN_MIN`` (0.10).
    """
    from ..ml.splits import KFold, RepeatedSplitter
    from ..stats.intervals import corrected_resampled_t
    O, O2, A, first = ro["obs"], ro["next_obs"], ro["actions"], ro["first"]
    if O is None or len(O) < 400:
        return None
    keep = np.flatnonzero(~first)                      # rows that have a previous step in the same episode
    if len(keep) < 300:
        return None
    Y = (O2 - O)[keep]
    dims = np.flatnonzero(Y.var(axis=0) > 1e-12)
    if dims.size == 0:
        return None
    Y = Y[:, dims]
    cur = _quadratic(np.column_stack([O[keep], A[keep]]))
    hist = _quadratic(np.column_stack([O[keep], A[keep], O[keep - 1], A[keep - 1]]))
    folds = RepeatedSplitter(KFold(2, seed), 5).folds(len(keep))

    def r2(Xtr, Ytr, Xte, Yte):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xa = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        Xb = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
        W = np.linalg.solve(Xa.T @ Xa + 1e-3 * len(Xtr) * np.eye(Xa.shape[1]), Xa.T @ Ytr)
        return 1.0 - ((Yte - Xb @ W) ** 2).mean(0) / np.maximum(Yte.var(0), 1e-12)

    deltas, base = [], []
    for _, tr, te in folds:
        b = r2(cur[tr], Y[tr], cur[te], Y[te])
        h = r2(hist[tr], Y[tr], hist[te], Y[te])
        deltas.append(float(np.max(h - b)))
        base.append(b)
    iv = corrected_resampled_t(deltas, len(keep) / 2, len(keep) / 2)
    return {"gain": iv.mean, "interval": (iv.lo, iv.hi), "p": iv.p_two_sided,
            "r2_change_current_only": np.round(np.mean(base, axis=0), 4).tolist(), "rows": int(len(keep))}


def probe_env(factory, probe_env_inst, *, steps: int, seed: int) -> Tuple[Dict[str, Any], List[Finding], Dict[str, Any]]:
    """Returns (profile sections, findings, facts the rules need)."""
    env = probe_env_inst
    findings: List[Finding] = []
    obs_p = space_profile(env.observation_space)
    act_p = space_profile(env.action_space)
    facts: Dict[str, Any] = {"obs": obs_p, "action": act_p}
    supported_act = act_p["type"] in ("Discrete",) or (act_p["type"] == "Box" and len(act_p.get("shape", [])) == 1)
    flat = obs_p["type"] == "Box" and len(obs_p.get("shape", [])) == 1
    facts["supported"] = supported_act
    facts["flat_obs"] = flat
    findings.append(Finding(
        id="env.spaces", category=Category.ENV, severity=Severity.INFO,
        statement=(f"Observations: {obs_p['type']}{tuple(obs_p.get('shape', []))}; actions: {act_p['type']}"
                   + (f" with {act_p['n']} choices" if act_p["type"] == "Discrete" else
                      f"{tuple(act_p.get('shape', []))}") + "."),
        evidence=(Evidence(test="space_inspection", details={"observation": obs_p, "action": act_p}),),
        confidence="high", confidence_basis="space objects"))
    if not supported_act:
        findings.append(Finding(
            id="env.unsupported_actions", category=Category.LIMITATION, severity=Severity.WARNING,
            statement=f"No built-in tensor-optix agent supports {act_p['type']} action spaces; discovery can "
                      "profile the environment but not compare algorithms on it.",
            evidence=(Evidence(test="space_inspection", details={"action": act_p}),),
            confidence="high", confidence_basis="make_agent() routing"))

    ro = random_rollout(env, steps, seed)
    facts["rollout"] = ro
    rew = ro["rewards"]
    rets = ro["returns"]
    lens = ro["lengths"]
    rollout_sec = {"steps": int(ro["steps"]), "episodes": int(len(rets)),
                   "random_return_mean": float(rets.mean()) if len(rets) else None,
                   "random_return_std": float(rets.std(ddof=1)) if len(rets) > 1 else None,
                   "episode_length_mean": float(lens.mean()) if len(lens) else None,
                   "truncated_fraction": float(ro["truncated"].mean()) if len(rets) else None,
                   "reward_mean": float(rew.mean()) if len(rew) else None,
                   "reward_std": float(rew.std()) if len(rew) else None}
    # sparsity: share of steps whose reward differs from the most common value
    if len(rew):
        vals, cnt = np.unique(np.round(rew, 12), return_counts=True)
        mode = vals[np.argmax(cnt)]
        informative = float(np.mean(np.abs(rew - mode) > 1e-12))
        # a constant per-step reward (e.g. +1 for surviving) is dense when episode lengths vary: the
        # learning signal is in the return. Sparse = almost no informative steps *and* returns that
        # barely differ between random episodes.
        if len(rets) >= 2:
            rvals, rcnt = np.unique(np.round(rets, 9), return_counts=True)
            ep_with_signal = float(1.0 - rcnt.max() / len(rets))
        else:
            ep_with_signal = 0.0
        rollout_sec.update(reward_mode=float(mode), informative_step_fraction=informative,
                           episodes_with_distinct_return=ep_with_signal)
        facts["sparse"] = informative < SPARSE_STEP_FRACTION and ep_with_signal < 0.05
        if facts["sparse"]:
            findings.append(Finding(
                id="env.sparse_reward", category=Category.ENV, severity=Severity.ADVISORY,
                statement=(f"Only {informative:.2%} of random-policy steps receive a reward other than {mode:g}: "
                           "exploration is hard. An exploration bonus (RND) is tried alongside the best "
                           "algorithm."),
                evidence=(Evidence(test="reward_sparsity", statistic=informative, n=int(len(rew)),
                                   details={"mode": float(mode), "episodes_with_signal": ep_with_signal}),),
                confidence="moderate", confidence_basis=f"{len(rew)} random-policy steps"))
    if len(rets) and (abs(float(rets.mean())) > REWARD_SCALE_LIMIT or float(rets.std()) > REWARD_SCALE_LIMIT):
        findings.append(Finding(
            id="env.reward_scale", category=Category.ENV, severity=Severity.INFO,
            statement=(f"Random-policy returns are large (mean {rets.mean():.1f}, std {rets.std():.1f}). Value "
                       "targets of this size slow learning for value-based methods; reward scaling or "
                       "normalisation usually helps."),
            evidence=(Evidence(test="return_scale", statistic=float(rets.mean()), details=rollout_sec),),
            confidence="moderate", confidence_basis="random-policy returns"))
    # observation scale
    if flat and ro["obs"] is not None and len(ro["obs"]) >= 100:
        O = ro["obs"]
        mean, std = O.mean(0), O.std(0)
        scale = float(np.max(np.abs(mean) + std))
        live = std[std > 1e-8]
        ratio = float(live.max() / live.min()) if live.size >= 2 else 1.0
        facts["obs_stats"] = {"mean": mean.tolist(), "std": std.tolist()}
        rollout_sec.update(obs_scale=scale, obs_spread_ratio=ratio)
        if scale > OBS_SCALE_LIMIT or ratio > OBS_SPREAD_RATIO:
            facts["obs_badly_scaled"] = True
            findings.append(Finding(
                id="env.obs_scale", category=Category.ENV, severity=Severity.ADVISORY,
                statement=(f"Observation features are on very different or large scales (largest |mean|+std "
                           f"{scale:.3g}, largest/smallest spread {ratio:.3g}). Normalising observations is "
                           "tried with the best algorithm."),
                evidence=(Evidence(test="obs_statistics", statistic=scale, n=int(len(O)),
                                   details={"spread_ratio": ratio}),),
                confidence="moderate", confidence_basis="random-policy observations"))
    # action bounds (tanh-squashed agents emit [-1, 1])
    if act_p["type"] == "Box" and supported_act:
        sp = env.action_space
        low, high = np.asarray(sp.low, dtype=np.float64), np.asarray(sp.high, dtype=np.float64)
        if not act_p["bounded"]:
            findings.append(Finding(
                id="env.unbounded_actions", category=Category.ENV, severity=Severity.WARNING,
                statement="The action space is unbounded; SAC / TD3 emit actions in [-1, 1], so their actions "
                          "cover only that range. Bound the actions, or rescale them in the environment.",
                evidence=(Evidence(test="space_inspection", details=act_p),),
                confidence="high", confidence_basis="action space bounds"))
        elif not (np.allclose(low, -1.0) and np.allclose(high, 1.0)):
            facts["action_rescale"] = {"low": low.tolist(), "high": high.tolist()}
            findings.append(Finding(
                id="env.action_rescale", category=Category.ENV, severity=Severity.WARNING,
                statement=(f"Actions are bounded to [{low.min():g}, {high.max():g}] (per dimension: low "
                           f"{np.round(low, 3).tolist()}, high {np.round(high, 3).tolist()}), but tanh-squashed "
                           "policies emit [-1, 1]. Discovery rescales actions in its trials; keep the rescaling "
                           "when you train."),
                evidence=(Evidence(test="space_inspection", details={"low": low.tolist(), "high": high.tolist()}),),
                confidence="high", confidence_basis="action space bounds",
                suggested_action=Action("wrap_action_rescale", {"low": low.tolist(), "high": high.tolist()}),
                default_applied=True))
    # determinism
    det = determinism(factory, seed)
    rollout_sec["deterministic"] = det
    if det is not None:
        findings.append(Finding(
            id="env.determinism", category=Category.ENV, severity=Severity.INFO,
            statement=("The environment is deterministic given a seed and actions: evaluation differences "
                       "between seeds come from the agents." if det else
                       "The environment is stochastic: the same seed and actions give different trajectories, so "
                       "evaluation uses several episodes per trial."),
            evidence=(Evidence(test="replay_same_seed", statistic=float(det)),),
            confidence="high", confidence_basis="two replays of the same seeded action sequence"))
    # partial observability
    if flat and ro["obs"] is not None:
        hg = history_gain(ro, seed)
        rollout_sec["history_gain"] = hg
        if hg is not None and hg["interval"][0] > HISTORY_GAIN_MIN and hg["p"] < 0.01:
            facts["pomdp"] = True
            findings.append(Finding(
                id="env.partial_observability", category=Category.ENV, severity=Severity.ADVISORY,
                statement=(f"The previous observation and action improve prediction of how the observation "
                           f"changes (R² +{hg['gain']:.3f}, 95% CI {hg['interval'][0]:+.3f} to "
                           f"{hg['interval'][1]:+.3f}): "
                           "the observation alone is not a complete state. A recurrent policy (or stacking "
                           "recent observations) is tried."),
                evidence=(Evidence(test="history_gain_ridge_5x2cv", statistic=hg["gain"], interval=hg["interval"],
                                   p_value=hg["p"], details=hg, seed=seed),),
                confidence="moderate", confidence_basis="paired corrected-t on the R² of the observation change"))
    return {"spaces": {"observation": obs_p, "action": act_p}, "rollout": rollout_sec}, findings, facts


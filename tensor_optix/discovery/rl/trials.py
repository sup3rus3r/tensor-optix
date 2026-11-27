"""
Seeded short RL trials.

Each trial builds a fresh (wrapped) environment and a fresh agent, trains it
for a fixed number of environment steps with tensor-optix's own loop
(``RLOptimizer``; best checkpoint restored at the end), then evaluates the
policy on ``eval_episodes`` episodes with fixed evaluation seeds. Online
hyperparameter tuning is off (a no-op optimizer), so a trial measures the
algorithm with its defaults, not the tuner.

Determinism: torch runs inside ``fork_rng`` on the CPU; numpy's and the stdlib's
global RNGs are seeded for the trial (agents draw from them) and restored
afterwards, so the caller's random state is untouched.
"""
from __future__ import annotations

import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import env as E

TRIAL_HIDDEN = (64, 64)
MAX_EVAL_STEPS = 10_000


@dataclass(frozen=True)
class Variant:
    algorithm: str                       # make_agent name: PPO, DQN, RAINBOW, SAC, TD3; or RECURRENT_PPO
    obs_normalize: bool = False
    rnd: bool = False

    @property
    def key(self) -> str:
        s = self.algorithm
        if self.obs_normalize:
            s += "+obs_norm"
        if self.rnd:
            s += "+rnd"
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {"algorithm": self.algorithm, "obs_normalize": self.obs_normalize, "rnd": self.rnd}


@dataclass
class TrialResult:
    variant: Variant
    seed: int
    score: Optional[float]
    eval_returns: List[float] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    status: str = "ok"
    error: Optional[str] = None


def _recurrent_agent(env, hidden: int = 64):
    import torch
    import torch.nn as nn
    from ...algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent
    from ...core.types import HyperparamSet
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    rnn = nn.GRU(obs_dim, hidden, batch_first=True)
    actor, critic = nn.Linear(hidden, n_actions), nn.Linear(hidden, 1)
    params = list(rnn.parameters()) + list(actor.parameters()) + list(critic.parameters())
    hp = HyperparamSet(params={"learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01, "vf_coef": 0.5,
                               "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 4, "bptt_len": 16,
                               "max_grad_norm": 0.5}, episode_id=0)
    return TorchRecurrentPPOAgent(rnn=rnn, actor_head=actor, critic_head=critic, n_actions=n_actions,
                                  optimizer=torch.optim.Adam(params, lr=3e-4), hyperparams=hp, device="cpu")


VALID = {"Discrete": ("PPO", "DQN", "RAINBOW", "RECURRENT_PPO"), "Box": ("SAC", "TD3", "PPO")}


def make_trial_agent(variant: Variant, env, hidden=TRIAL_HIDDEN):
    # make_agent() falls back to its default for names it does not route; a trial must
    # never run a different algorithm than the one it is labelled with
    kind = type(env.action_space).__name__
    if variant.algorithm not in VALID.get(kind, ()):
        raise ValueError(f"algorithm {variant.algorithm!r} does not apply to {kind} actions; "
                         f"options: {VALID.get(kind, ())}")
    if variant.algorithm == "RECURRENT_PPO":
        return _recurrent_agent(env, hidden[0])
    from ...factory import make_agent
    return make_agent(variant.algorithm, env, hidden_sizes=tuple(hidden), device="cpu")


def evaluate(agent, env, episodes: int, seed_base: int) -> List[float]:
    out = []
    for i in range(episodes):
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden()
        o = E.reset(env, seed=seed_base + i)
        ret, done, n = 0.0, False, 0
        while not done and n < MAX_EVAL_STEPS:
            o, r, term, tr = E.step(env, agent.act(o))
            ret += r
            n += 1
            done = term or tr
        out.append(ret)
    return out


def run_trial(factory: Callable[[], Any], variant: Variant, *, wrappers: Dict[str, Any], seed: int, steps: int,
              window: int, eval_episodes: int, obs_stats: Optional[Dict[str, Any]] = None,
              eval_seed_base: int = 1_000_000) -> TrialResult:
    import torch
    from ...optimizer import RLOptimizer
    from ...pipeline.batch_pipeline import BatchPipeline
    from ...strategy.worker import _build_noop_optimizer
    t0 = time.perf_counter()
    np_state, py_state = np.random.get_state(), random.getstate()
    w = dict(wrappers)
    if variant.obs_normalize and obs_stats:
        w["obs_normalize"] = obs_stats
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            random.seed(seed)
            env = E.make_wrapped(factory, w)
            E.reset(env, seed=seed)
            try:
                env.action_space.seed(seed)
            except Exception:
                pass
            agent = make_trial_agent(variant, env)
            pipe = BatchPipeline(env=env, agent=agent, window_size=window)
            if variant.rnd:
                from ...exploration.rnd import RNDPipeline
                pipe = RNDPipeline(pipe, obs_dim=int(np.prod(env.observation_space.shape)))
            with tempfile.TemporaryDirectory(prefix="optix_discovery_rl_") as ckpt:
                RLOptimizer(agent=agent, pipeline=pipe, optimizer=_build_noop_optimizer(),
                            max_episodes=max(1, steps // window), checkpoint_dir=ckpt, verbose=False).run()
            eval_env = E.make_wrapped(factory, w)
            rets = evaluate(agent, eval_env, eval_episodes, eval_seed_base)
        return TrialResult(variant, seed, float(np.mean(rets)), rets, steps, time.perf_counter() - t0)
    except Exception as exc:                                  # a failing algorithm is a result, not a crash
        return TrialResult(variant, seed, None, [], steps, time.perf_counter() - t0, "failed",
                           f"{type(exc).__name__}: {exc}")
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)

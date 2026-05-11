#!/usr/bin/env python3
"""
tensor-optix Real-World Benchmark
==================================
Compares tensor-optix's autonomous training loop against an equivalent
baseline loop across multiple environments and algorithms.

  Baseline  — fixed step budget, no convergence detection, no auto-tuning.
              Same agent, same architecture, same starting hyperparameters.

  tensor-optix — make_agent() + Optimizer() (new API):
                  • AdaptiveOptimizer / SPSA online hyperparameter tuning
                  • Exponential-backoff convergence detection (ACTIVE→COOLING→DORMANT)
                  • PolicyManager: automatic rollback + policy spawning
                  • Stops early when converged (saves steps vs fixed budget)

The contrast in code size tells its own story:
  Baseline per env   → ~60 lines of boilerplate
  tensor-optix       → ~15 lines using make_agent + Optimizer

Environments:
  1. CartPole-v1                — DQN, discrete, classic balance task
  2. LunarLander-v3             — PPO, discrete, risk of local-optima collapse
  3. Acrobot-v1                 — PPO, discrete, sparse rewards
  4. LunarLanderContinuous-v3   — SAC, continuous
  5. BipedalWalker-v3           — SAC, continuous locomotion (hardest standard benchmark)
  6. CartPole Rainbow           — Rainbow DQN (all 6 improvements)
  7. Pendulum TD3               — TD3, continuous, twin-delayed
  8. POMDP CartPole             — Recurrent PPO, partial observability
  9. CartPole Neuroevo          — NeuronGraph + TopologyController (policy grows at runtime)

Usage:
    python benchmarks/benchmark.py                         # all envs, 3 seeds
    python benchmarks/benchmark.py --envs cartpole acrobot
    python benchmarks/benchmark.py --seeds 0 1
    python benchmarks/benchmark.py --no-baseline           # optix + neuroevo only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU-only for reproducible timing

import gymnasium as gym
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── tensor-optix imports ──────────────────────────────────────────────────────
import tensor_optix as optix
from tensor_optix import (
    make_agent,
    BatchPipeline,
    EpisodeData,
    EvalMetrics,
    HyperparamSet,
    LoopCallback,
    PolicyManager,
)
from tensor_optix.simple import Optimizer
from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer
from tensor_optix.optimizers.adaptive_optimizer import AdaptiveOptimizer
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.optimizer import RLOptimizer
from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator

# Algorithm imports (for baselines only — optix side uses make_agent)
from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.algorithms.torch_sac import TorchSACAgent
from tensor_optix.algorithms.torch_td3 import TorchTD3Agent
from tensor_optix.algorithms.torch_rainbow_dqn import TorchRainbowDQNAgent, RainbowQNetwork
from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared architecture builders  (baseline only — optix uses make_agent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mlp(in_dim: int, hidden: int, out_dim: int, act=nn.Tanh) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim),
    )

def build_ppo_nets(obs_dim, n_actions, hidden=64):
    return _mlp(obs_dim, hidden, n_actions), _mlp(obs_dim, hidden, 1)

def build_dqn_net(obs_dim, n_actions, hidden=128):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),  nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )

def build_sac_nets(obs_dim, action_dim, hidden=256):
    actor   = _mlp(obs_dim, hidden, action_dim * 2)
    critic1 = _mlp(obs_dim + action_dim, hidden, 1, act=nn.ReLU)
    critic2 = _mlp(obs_dim + action_dim, hidden, 1, act=nn.ReLU)
    return actor, critic1, critic2

def build_td3_nets(obs_dim, action_dim, hidden=256):
    actor = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),  nn.ReLU(),
        nn.Linear(hidden, action_dim), nn.Tanh(),
    )
    c1 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    c2 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    return actor, c1, c2

def build_rppo_nets(obs_dim, n_actions, hidden=64):
    return (nn.LSTM(obs_dim, hidden, batch_first=True),
            nn.Linear(hidden, n_actions),
            nn.Linear(hidden, 1))


class _POMDPCartPoleWrapper(gym.ObservationWrapper):
    """Mask velocity components — obs_dim 4 → 2."""
    def __init__(self, env):
        super().__init__(env)
        import gymnasium.spaces as _s
        low, high = env.observation_space.low[[0, 2]], env.observation_space.high[[0, 2]]
        self.observation_space = _s.Box(low=low, high=high, dtype=env.observation_space.dtype)
    def observation(self, obs):
        return obs[[0, 2]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Evaluation helpers  (deterministic, cache-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _dev(net):
    try:    return next(net.parameters()).device
    except: return torch.device("cpu")

def eval_dqn(q_net, env_id, n_eps=10, seed=9000):
    dev = _dev(q_net); env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(dev)).argmax(-1).item())
            obs, r, t, tr, _ = env.step(action); total += r; done = t or tr
        totals.append(total)
    env.close(); return float(np.mean(totals))

def eval_ppo(actor, env_id, n_eps=10, seed=9000):
    dev = _dev(actor); env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action = int(torch.argmax(actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(dev)), dim=-1).item())
            obs, r, t, tr, _ = env.step(action); total += r; done = t or tr
        totals.append(total)
    env.close(); return float(np.mean(totals))

def eval_sac(actor, env_id, action_scale=1.0, n_eps=10, seed=9000):
    dev = _dev(actor); env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        while not done:
            with torch.no_grad():
                out = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(dev))
                action = (torch.tanh(out.chunk(2, dim=-1)[0]).cpu().numpy()[0] * action_scale)
            obs, r, t, tr, _ = env.step(action); total += r; done = t or tr
        totals.append(total)
    env.close(); return float(np.mean(totals))

def eval_td3(actor, env_id, action_scale=1.0, n_eps=10, seed=9000):
    dev = _dev(actor); env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action = (actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(dev)).cpu().numpy()[0] * action_scale)
            obs, r, t, tr, _ = env.step(action); total += r; done = t or tr
        totals.append(total)
    env.close(); return float(np.mean(totals))

def eval_rainbow(q_net, env_id, n_eps=10, seed=9000):
    dev = _dev(q_net); q_net.eval(); env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        while not done:
            with torch.no_grad():
                lp = q_net(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(dev))
                action = int(lp.exp().sum(-1).argmax(-1).item())
            obs, r, t, tr, _ = env.step(action); total += r; done = t or tr
        totals.append(total)
    env.close(); q_net.train(); return float(np.mean(totals))

def eval_recurrent_ppo(agent, env_factory, n_eps=10, seed=9000):
    totals = []
    for ep in range(n_eps):
        env = env_factory(); obs, _ = env.reset(seed=seed + ep)
        agent.reset_hidden(); total, done = 0.0, False
        while not done:
            action = agent.act(obs); obs, r, t, tr, _ = env.step(action)
            total += r; done = t or tr
        env.close(); totals.append(total)
    agent._cache_obs.clear(); agent._cache_lp.clear()
    agent._cache_values.clear(); agent._cache_hidden.clear()
    agent.reset_hidden()
    return float(np.mean(totals))

def eval_neuroevo(agent, env_id, n_eps=10, seed=9000):
    env = gym.make(env_id); totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep); total, done = 0.0, False
        agent.graph.reset_state()
        while not done:
            action = agent.act(obs); obs, r, t, tr, _ = env.step(action)
            total += r; done = t or tr
        totals.append(total)
    env.close(); return float(np.mean(totals))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared pipeline helper for recurrent BPTT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from tensor_optix.core.base_pipeline import BasePipeline as _BP

class _RecurrentBatchPipeline(_BP):
    """Window pipeline that resets LSTM hidden state at episode boundaries."""
    def __init__(self, env, agent=None, window_size=1024):
        self._env = env; self._agent = agent; self._window_size = window_size
        self._window_id = 0; self._obs = None; self._needs_reset = True
    @property
    def is_live(self): return False
    def set_agent(self, agent): self._agent = agent
    def setup(self): self._needs_reset = True
    def episodes(self):
        while True:
            if self._needs_reset:
                self._obs, _ = self._env.reset()
                if hasattr(self._agent, "reset_hidden"): self._agent.reset_hidden()
                self._needs_reset = False
            obs_list, act_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], [], []
            starts = [0]
            for i in range(self._window_size):
                obs_list.append(self._obs)
                action = self._agent.act(self._obs)
                act_list.append(action)
                nobs, r, term, trunc, info = self._env.step(action)
                rew_list.append(float(r)); term_list.append(bool(term))
                trunc_list.append(bool(trunc)); info_list.append(info)
                if term or trunc:
                    self._obs, _ = self._env.reset()
                    if hasattr(self._agent, "reset_hidden"): self._agent.reset_hidden()
                    if i + 1 < self._window_size: starts.append(i + 1)
                else:
                    self._obs = nobs
            last_done = term_list[-1] or trunc_list[-1]
            yield EpisodeData(
                observations=np.array(obs_list), actions=np.array(act_list),
                rewards=rew_list, terminated=term_list, truncated=trunc_list,
                infos=info_list, episode_id=self._window_id,
                episode_starts=starts, final_obs=None if last_done else self._obs,
            )
            self._window_id += 1
    def teardown(self): self._env.close()
    @property
    def window_size(self): return self._window_size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Callbacks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Tracker(LoopCallback):
    """Records step-indexed eval scores; shared by baseline + optix."""
    def __init__(self, cfg, eval_fn, seed):
        self._cfg = cfg; self._eval_fn = eval_fn; self._seed = seed
        self.total_steps = 0; self.steps_to_solve = None
        self.history: list[tuple[int, float]] = []
        self._since_eval = 0; self._converged = False

    def on_episode_end(self, episode_id, eval_metrics):
        self.total_steps += self._cfg["window_size"]
        self._since_eval += self._cfg["window_size"]
        if self._since_eval >= self._cfg["eval_every"]:
            self._since_eval = 0
            score = self._eval_fn(seed=self._seed + 10_000)
            self.history.append((self.total_steps, score))
            print(
                f"  [{self._cfg['tag']:10s}| {self._cfg['label']:20s}| seed={self._seed}]"
                f"  steps={self.total_steps:>7,d}  score={score:>8.1f}", flush=True,
            )
            if self.steps_to_solve is None and score >= self._cfg["solve_threshold"]:
                self.steps_to_solve = self.total_steps

    def on_dormant(self, episode_id):
        if not self._converged:
            self._converged = True
            print(
                f"  [{self._cfg['tag']:10s}| {self._cfg['label']:20s}| seed={self._seed}]"
                f"  *** CONVERGED at {self.total_steps:,d} steps ***", flush=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vanilla (baseline) training — fixed budget, no loop features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _collect_window(env, agent, window_size, carry_obs):
    obs = carry_obs; obs_l, act_l, rew_l, term_l, trunc_l = [], [], [], [], []
    for _ in range(window_size):
        obs_l.append(obs); act_l.append(agent.act(obs))
        nobs, r, t, tr, _ = env.step(act_l[-1])
        rew_l.append(float(r)); term_l.append(bool(t)); trunc_l.append(bool(tr))
        obs = nobs if not (t or tr) else env.reset()[0]
    return EpisodeData(
        observations=np.array(obs_l), actions=np.array(act_l),
        rewards=rew_l, terminated=term_l, truncated=trunc_l,
        infos=[{}] * window_size, episode_id=0,
    ), obs


def _run_vanilla(cfg, seed, agent, eval_fn):
    """Generic fixed-budget loop. Works for all algorithm types."""
    env = gym.make(cfg["env_id"])
    carry, _ = env.reset(seed=seed)
    tracker = _Tracker(cfg={**cfg, "tag": "baseline "}, eval_fn=eval_fn, seed=seed)
    total, wid = 0, 0
    while total < cfg["max_steps"]:
        ep, carry = _collect_window(env, agent, cfg["window_size"], carry)
        ep.episode_id = wid; agent.learn(ep)
        total += cfg["window_size"]; wid += 1
        tracker.on_episode_end(wid, None)
    env.close()
    return tracker


def train_vanilla(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.perf_counter()
    agent = cfg["baseline_agent_factory"](seed)
    tracker = _run_vanilla(cfg, seed, agent, lambda seed: cfg["eval_fn"](agent, cfg["env_id"], seed=seed))
    final = cfg["eval_fn"](agent, cfg["env_id"], seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final))
    if tracker.steps_to_solve is None and final >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    return {
        "method": "Baseline", "seed": seed,
        "total_steps": tracker.total_steps, "steps_to_solve": tracker.steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
    }


def train_vanilla_recurrent(cfg: dict, seed: int) -> dict:
    """Baseline for POMDP RecurrentPPO — episode-based collection."""
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.perf_counter()
    agent = cfg["baseline_agent_factory"](seed)
    env_factory = cfg["env_factory"]
    env = env_factory()
    total, ep_id, since_eval = 0, 0, 0
    history: list[tuple[int, float]] = []
    steps_to_solve = None

    while total < cfg["max_steps"]:
        obs, _ = env.reset(seed=seed + ep_id); agent.reset_hidden()
        obs_l, act_l, rew_l, term_l, trunc_l = [], [], [], [], []
        done = False
        while not done:
            obs_l.append(obs.copy()); action = agent.act(obs); act_l.append(action)
            obs, r, t, tr, _ = env.step(action)
            rew_l.append(float(r)); term_l.append(bool(t)); trunc_l.append(bool(tr))
            done = t or tr
        ep_data = EpisodeData(
            observations=np.array(obs_l), actions=act_l, rewards=rew_l,
            terminated=term_l, truncated=trunc_l, infos=[{}] * len(rew_l), episode_id=ep_id,
        )
        agent.learn(ep_data); n = len(rew_l)
        total += n; since_eval += n; ep_id += 1
        if since_eval >= cfg["eval_every"]:
            since_eval = 0
            score = eval_recurrent_ppo(agent, env_factory, seed=seed + 10_000)
            history.append((total, score))
            print(f"  [baseline  | {cfg['label']:20s}| seed={seed}]"
                  f"  steps={total:>7,d}  score={score:>8.1f}", flush=True)
            if steps_to_solve is None and score >= cfg["solve_threshold"]:
                steps_to_solve = total
    env.close()
    final = history[-1][1] if history else 0.0
    return {
        "method": "Baseline", "seed": seed, "total_steps": total,
        "steps_to_solve": steps_to_solve, "final_score": final,
        "elapsed": time.perf_counter() - t0, "history": history,
        "solved": steps_to_solve is not None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  tensor-optix training — NEW API  (make_agent + Optimizer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_optix(cfg: dict, seed: int) -> dict:
    """
    tensor-optix autonomous training using the new API.

    Old API (per-algorithm boilerplate):   ~80 lines each × 6 algorithms = 480 lines
    New API (this function, all algorithms):  ~30 lines total

    make_agent()  auto-selects the right algorithm from the action space and the
                  optional algorithm= name.  Optimizer() wires window_size,
                  SPSA, and neuroevo callbacks automatically.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    ckpt_dir = f"./benchmarks/.ckpts/{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    t0 = time.perf_counter()

    # ── 1. Build agent — one line ─────────────────────────────────────────────
    env_ref = gym.make(cfg["env_id"])
    agent = make_agent(env_ref, **cfg.get("make_agent_kwargs", {}))
    env_ref.close()

    # ── 2. PolicyManager for rollback + policy spawning ───────────────────────
    registry   = CheckpointRegistry(ckpt_dir)
    pm         = PolicyManager(registry, max_spawns=3)
    agent_holder = {"agent": agent}

    def _agent_factory():
        e = gym.make(cfg["env_id"])
        a = make_agent(e, **cfg.get("make_agent_kwargs", {}))
        e.close()
        agent_holder["agent"] = a
        return a

    pm_cb = pm.as_callback(agent, agent_factory=_agent_factory)

    # ── 3. Eval tracker ───────────────────────────────────────────────────────
    tracker = _Tracker(
        cfg={**cfg, "tag": "tensor-optix"},
        eval_fn=lambda seed: cfg["eval_fn"](agent_holder["agent"], cfg["env_id"], seed=seed),
        seed=seed,
    )

    # ── 4. Optimizer — one line ───────────────────────────────────────────────
    opt = Optimizer(
        agent=agent,
        env=gym.make(cfg["env_id"]),
        window_size=cfg["window_size"],
        optimizer=SPSAOptimizer(**cfg["spsa_kwargs"]) if cfg.get("spsa_kwargs") else None,
        callbacks=[tracker, pm_cb],
        rollback_on_degradation=True,
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        checkpoint_score_fn=lambda a: cfg["eval_fn"](
            agent_holder["agent"], cfg["env_id"], seed=seed + 10_000
        ),
        dormant_threshold=cfg.get("dormant_threshold", 8),
        max_interval_episodes=cfg.get("max_interval_episodes", 8),
        min_episodes_before_dormant=cfg.get("min_episodes_before_dormant", 50),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
    )
    pm_cb.set_stop_fn(opt._rl_optimizer.stop)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    opt.run()

    final = cfg["eval_fn"](agent_holder["agent"], cfg["env_id"], seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final))
    if tracker.steps_to_solve is None and final >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    print(
        f"  [tensor-optix| {cfg['label']:20s}| seed={seed}]"
        f"  steps={tracker.total_steps:>7,d}  score={final:>8.1f}  [final]", flush=True,
    )
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "tensor-optix", "seed": seed,
        "total_steps": tracker.total_steps, "steps_to_solve": tracker.steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
    }


def train_optix_recurrent(cfg: dict, seed: int) -> dict:
    """
    tensor-optix Recurrent PPO (POMDP).
    Uses _RecurrentBatchPipeline directly since Optimizer uses BatchPipeline internally.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    ckpt_dir = f"./benchmarks/.ckpts/{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    t0 = time.perf_counter()

    env_factory = cfg["env_factory"]
    env = env_factory()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    rnn, ah, ch = build_rppo_nets(obs_dim, n_actions)
    agent = TorchRecurrentPPOAgent(
        rnn=rnn, actor_head=ah, critic_head=ch, n_actions=n_actions,
        optimizer=torch.optim.Adam(
            list(rnn.parameters()) + list(ah.parameters()) + list(ch.parameters()), lr=3e-4,
        ),
        hyperparams=HyperparamSet(params=RPPO_HP.copy(), episode_id=0),
    )

    registry = CheckpointRegistry(ckpt_dir)
    pm       = PolicyManager(registry, max_spawns=3)
    agent_holder = {"agent": agent}

    def _agent_factory():
        r, a, c = build_rppo_nets(obs_dim, n_actions)
        na = TorchRecurrentPPOAgent(
            rnn=r, actor_head=a, critic_head=c, n_actions=n_actions,
            optimizer=torch.optim.Adam(
                list(r.parameters()) + list(a.parameters()) + list(c.parameters()), lr=3e-4,
            ),
            hyperparams=HyperparamSet(params=RPPO_HP.copy(), episode_id=0),
        )
        agent_holder["agent"] = na; return na

    pm_cb = pm.as_callback(agent, agent_factory=_agent_factory)
    tracker = _Tracker(
        cfg={**cfg, "tag": "tensor-optix"},
        eval_fn=lambda seed: eval_recurrent_ppo(agent_holder["agent"], env_factory, seed=seed),
        seed=seed,
    )

    pipeline = _RecurrentBatchPipeline(env=env_factory(), agent=agent, window_size=cfg["window_size"])

    rl_opt = RLOptimizer(
        agent=agent, pipeline=pipeline,
        evaluator=TorchEvaluator(),
        optimizer=SPSAOptimizer(
            param_bounds={"learning_rate": (1e-4, 3e-3), "clip_ratio": (0.1, 0.3),
                          "entropy_coef": (0.0, 0.05)},
            log_params=["learning_rate"], warmup_episodes=30,
        ),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8, dormant_threshold=8, min_episodes_before_dormant=30,
        checkpoint_score_fn=lambda a: eval_recurrent_ppo(
            agent_holder["agent"], env_factory, seed=seed + 10_000
        ),
        target_score=cfg["solve_threshold"], convergence_patience=5,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
    rl_opt.add_callback(tracker); rl_opt.add_callback(pm_cb)
    rl_opt.run()

    final = eval_recurrent_ppo(agent_holder["agent"], env_factory, seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final))
    if tracker.steps_to_solve is None and final >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "tensor-optix", "seed": seed,
        "total_steps": tracker.total_steps, "steps_to_solve": tracker.steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Neuroevo benchmark  (make_agent neuroevo=True + Optimizer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _NeuroevoTracker(LoopCallback):
    """Tracks topology events alongside scores."""
    def __init__(self, cfg, agent, seed):
        self._cfg = cfg; self._agent = agent; self._seed = seed
        self.total_steps = 0; self.steps_to_solve = None
        self.history: list[tuple[int, float]] = []
        self.topology_events: list[tuple[int, int, int]] = []
        self._since_eval = 0

    def on_episode_end(self, episode_id, eval_metrics):
        self.total_steps += self._cfg["window_size"]
        self._since_eval += self._cfg["window_size"]
        n = self._agent.graph.n_neurons(); e = self._agent.graph.n_edges()
        self.topology_events.append((self.total_steps, n, e))
        if self._since_eval >= self._cfg["eval_every"]:
            self._since_eval = 0
            score = eval_neuroevo(self._agent, self._cfg["env_id"], seed=self._seed + 10_000)
            self.history.append((self.total_steps, score))
            print(
                f"  [neuroevo  | {self._cfg['label']:20s}| seed={self._seed}]"
                f"  steps={self.total_steps:>7,d}  score={score:>8.1f}"
                f"  neurons={n}  edges={e}", flush=True,
            )
            if self.steps_to_solve is None and score >= self._cfg["solve_threshold"]:
                self.steps_to_solve = self.total_steps


def train_neuroevo(cfg: dict, seed: int) -> dict:
    """
    Neuroevo: make_agent(env, neuroevo=True) + Optimizer.
    The graph starts small and grows topology on plateau via TopologyController.

    3 lines to set up (make_agent + Optimizer) vs the previous ~100-line manual build.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    ckpt_dir = f"./benchmarks/.ckpts/neuroevo_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    t0 = time.perf_counter()

    # ── make_agent with neuroevo=True ─────────────────────────────────────────
    env_ref = gym.make(cfg["env_id"])
    agent   = make_agent(env_ref, neuroevo=True, graph_hidden=cfg.get("n_hidden", 4),
                         device="cpu")
    env_ref.close()

    tracker = _NeuroevoTracker(cfg=cfg, agent=agent, seed=seed)

    # ── Optimizer auto-wires HebbianHook + TopologyController ─────────────────
    opt = Optimizer(
        agent=agent,
        env=gym.make(cfg["env_id"]),
        window_size=cfg["window_size"],
        callbacks=[tracker],
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        dormant_threshold=cfg.get("dormant_threshold", 8),
        target_score=cfg["solve_threshold"],
    )
    opt.run()

    final = eval_neuroevo(agent, cfg["env_id"], seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final))
    if tracker.steps_to_solve is None and final >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps

    # Topology growth summary
    if tracker.topology_events:
        n_start = tracker.topology_events[0][1]
        n_end   = tracker.topology_events[-1][1]
        e_end   = tracker.topology_events[-1][2]
        print(
            f"  [neuroevo  | {cfg['label']:20s}| seed={seed}]"
            f"  neurons {n_start}→{n_end}  edges→{e_end}  score={final:.1f}  [final]",
            flush=True,
        )
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "neuroevo", "seed": seed,
        "total_steps": tracker.total_steps, "steps_to_solve": tracker.steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
        "topology_events": tracker.topology_events,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Environment configs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PPO_HP = {
    "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
    "vf_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95,
    "n_epochs": 10, "minibatch_size": 64, "max_grad_norm": 0.5,
}
SAC_HP = {
    "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005,
    "batch_size": 256, "updates_per_step": 1, "replay_capacity": 100_000,
}
DQN_HP = {
    "learning_rate": 1e-3, "gamma": 0.99, "epsilon": 1.0,
    "epsilon_min": 0.05, "epsilon_decay": 0.995, "batch_size": 64,
    "target_update_freq": 10, "replay_capacity": 10_000,
}
RAINBOW_HP = {
    "learning_rate": 6.25e-5, "gamma": 0.99, "batch_size": 32,
    "target_update_freq": 200, "replay_capacity": 100_000,
    "per_alpha": 0.5, "per_beta": 0.4, "n_step": 3,
    "v_min": 0.0, "v_max": 500.0, "n_atoms": 51,
}
TD3_HP = {
    "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005,
    "batch_size": 256, "updates_per_step": 1, "replay_capacity": 200_000,
    "policy_delay": 2, "target_noise": 0.2, "target_noise_clip": 0.5,
    "per_alpha": 0.0, "per_beta": 0.4,
}
RPPO_HP = {
    "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
    "vf_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95,
    "n_epochs": 4, "bptt_len": 16, "max_grad_norm": 0.5,
}


def _dqn_factory(seed):
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    env.close(); q = build_dqn_net(obs_dim, n_actions)
    return TorchDQNAgent(
        q_network=q, n_actions=n_actions,
        optimizer=torch.optim.Adam(q.parameters(), lr=1e-3),
        hyperparams=HyperparamSet(params=DQN_HP.copy(), episode_id=0),
    )

def _ppo_factory(env_id, seed):
    env = gym.make(env_id); obs_dim = env.observation_space.shape[0]; n_actions = env.action_space.n; env.close()
    a, c = build_ppo_nets(obs_dim, n_actions)
    return TorchPPOAgent(
        actor=a, critic=c,
        optimizer=torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=3e-4),
        hyperparams=HyperparamSet(params=PPO_HP.copy(), episode_id=0), device="auto",
    )

def _sac_factory(env_id, seed, action_scale=1.0):
    env = gym.make(env_id); obs_dim = env.observation_space.shape[0]; action_dim = env.action_space.shape[0]; env.close()
    a, c1, c2 = build_sac_nets(obs_dim, action_dim); la = torch.tensor(0.0, requires_grad=True)
    return TorchSACAgent(
        actor=a, critic1=c1, critic2=c2, action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(a.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=3e-4),
        alpha_optimizer=torch.optim.Adam([la], lr=3e-4),
        hyperparams=HyperparamSet(params=SAC_HP.copy(), episode_id=0), device="auto",
    )

def _rainbow_factory(seed):
    env = gym.make("CartPole-v1"); obs_dim = env.observation_space.shape[0]; n_actions = env.action_space.n; env.close()
    q = RainbowQNetwork.build(obs_dim, n_actions)
    return TorchRainbowDQNAgent(
        q_network=q, n_actions=n_actions, obs_dim=obs_dim,
        optimizer=torch.optim.Adam(q.parameters(), lr=6.25e-5),
        hyperparams=HyperparamSet(params=RAINBOW_HP.copy(), episode_id=0), device="cpu",
    )

def _td3_factory(env_id, seed):
    env = gym.make(env_id); obs_dim = env.observation_space.shape[0]; action_dim = env.action_space.shape[0]; env.close()
    a, c1, c2 = build_td3_nets(obs_dim, action_dim)
    return TorchTD3Agent(
        actor=a, critic1=c1, critic2=c2, action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(a.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=3e-4),
        hyperparams=HyperparamSet(params=TD3_HP.copy(), episode_id=0), device="auto",
    )

def _rppo_factory(obs_dim, n_actions, seed):
    rnn, ah, ch = build_rppo_nets(obs_dim, n_actions)
    return TorchRecurrentPPOAgent(
        rnn=rnn, actor_head=ah, critic_head=ch, n_actions=n_actions,
        optimizer=torch.optim.Adam(
            list(rnn.parameters()) + list(ah.parameters()) + list(ch.parameters()), lr=3e-4,
        ),
        hyperparams=HyperparamSet(params=RPPO_HP.copy(), episode_id=0),
    )


ENV_CONFIGS = {
    "cartpole": {
        "key": "cartpole", "label": "CartPole-v1", "env_id": "CartPole-v1",
        "algo": "DQN (discrete)",
        "solve_threshold": 475.0, "max_steps": 200_000, "window_size": 512,
        "eval_every": 5_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_dqn(agent._q, env_id, seed=seed),
        "make_agent_kwargs": {"algorithm": "DQN"},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 1e-3), "gamma": (0.95, 0.999)},
                        "log_params": ["learning_rate"], "warmup_episodes": 30},
        "baseline_agent_factory": _dqn_factory,
        "train_vanilla": train_vanilla, "train_optix": train_optix,
    },
    "lunarlander": {
        "key": "lunarlander", "label": "LunarLander-v3", "env_id": "LunarLander-v3",
        "algo": "PPO (discrete)",
        "solve_threshold": 200.0, "max_steps": 500_000, "window_size": 2_048,
        "eval_every": 20_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_ppo(agent._actor, env_id, seed=seed),
        "make_agent_kwargs": {},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 3e-3), "clip_ratio": (0.1, 0.3),
                                         "entropy_coef": (0.0, 0.05)},
                        "log_params": ["learning_rate"], "warmup_episodes": 40},
        "baseline_agent_factory": lambda seed: _ppo_factory("LunarLander-v3", seed),
        "train_vanilla": train_vanilla, "train_optix": train_optix,
        "dormant_threshold": 6, "max_interval_episodes": 8, "min_episodes_before_dormant": 50,
    },
    "lunarlander_continuous": {
        "key": "lunarlander_continuous", "label": "LunarLanderContinuous-v3",
        "env_id": "LunarLanderContinuous-v3", "algo": "SAC (continuous)",
        "solve_threshold": 200.0, "max_steps": 600_000, "window_size": 2_048,
        "eval_every": 20_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_sac(agent._actor, env_id, 1.0, seed=seed),
        "make_agent_kwargs": {},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 3e-3), "gamma": (0.97, 0.999),
                                         "tau": (1e-3, 1e-1)},
                        "log_params": ["learning_rate", "tau"], "warmup_episodes": 30},
        "baseline_agent_factory": lambda seed: _sac_factory("LunarLanderContinuous-v3", seed),
        "train_vanilla": train_vanilla, "train_optix": train_optix,
    },
    "acrobot": {
        "key": "acrobot", "label": "Acrobot-v1", "env_id": "Acrobot-v1",
        "algo": "PPO (discrete, sparse)",
        "solve_threshold": -100.0, "max_steps": 400_000, "window_size": 2_048,
        "eval_every": 20_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_ppo(agent._actor, env_id, seed=seed),
        "make_agent_kwargs": {},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 3e-3), "clip_ratio": (0.1, 0.3),
                                         "entropy_coef": (0.0, 0.05)},
                        "log_params": ["learning_rate"], "warmup_episodes": 40},
        "baseline_agent_factory": lambda seed: _ppo_factory("Acrobot-v1", seed),
        "train_vanilla": train_vanilla, "train_optix": train_optix,
    },
    "bipedalwalker": {
        "key": "bipedalwalker", "label": "BipedalWalker-v3", "env_id": "BipedalWalker-v3",
        "algo": "SAC (continuous locomotion)",
        "solve_threshold": 300.0, "max_steps": 1_500_000, "window_size": 2_048,
        "eval_every": 50_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_sac(agent._actor, env_id, 1.0, seed=seed),
        "make_agent_kwargs": {},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 3e-3), "gamma": (0.97, 0.999),
                                         "tau": (1e-3, 1e-1)},
                        "log_params": ["learning_rate", "tau"], "warmup_episodes": 30},
        "baseline_agent_factory": lambda seed: _sac_factory("BipedalWalker-v3", seed),
        "train_vanilla": train_vanilla, "train_optix": train_optix,
        "dormant_threshold": 8,
    },
    "cartpole_rainbow": {
        "key": "cartpole_rainbow", "label": "CartPole Rainbow", "env_id": "CartPole-v1",
        "algo": "Rainbow DQN (all 6 improvements)",
        "solve_threshold": 475.0, "max_steps": 200_000, "window_size": 512,
        "eval_every": 5_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_rainbow(agent._q, env_id, seed=seed),
        "make_agent_kwargs": {"algorithm": "RAINBOW"},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-5, 1e-4), "gamma": (0.97, 0.999)},
                        "log_params": ["learning_rate"], "warmup_episodes": 30},
        "baseline_agent_factory": _rainbow_factory,
        "train_vanilla": train_vanilla, "train_optix": train_optix,
    },
    "pendulum_td3": {
        "key": "pendulum_td3", "label": "Pendulum TD3", "env_id": "Pendulum-v1",
        "algo": "TD3 (continuous, twin-delayed)",
        "solve_threshold": -150.0, "max_steps": 300_000, "window_size": 1_000,
        "eval_every": 10_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_td3(agent._actor, env_id, 2.0, seed=seed),
        "make_agent_kwargs": {"deterministic": True},
        "spsa_kwargs": {"param_bounds": {"learning_rate": (1e-4, 1e-3), "gamma": (0.97, 0.999),
                                         "tau": (1e-3, 1e-1)},
                        "log_params": ["learning_rate", "tau"], "warmup_episodes": 30},
        "baseline_agent_factory": lambda seed: _td3_factory("Pendulum-v1", seed),
        "train_vanilla": train_vanilla, "train_optix": train_optix,
    },
    "pomdp_cartpole": {
        "key": "pomdp_cartpole", "label": "POMDP CartPole", "env_id": "CartPole-v1",
        "env_factory": lambda: _POMDPCartPoleWrapper(gym.make("CartPole-v1")),
        "algo": "Recurrent PPO (LSTM, partial observability)",
        "solve_threshold": 350.0, "max_steps": 300_000, "window_size": 1_024,
        "eval_every": 10_000,
        "eval_fn": lambda agent, env_id, seed=9000: eval_recurrent_ppo(
            agent, lambda: _POMDPCartPoleWrapper(gym.make("CartPole-v1")), seed=seed,
        ),
        "baseline_agent_factory": lambda seed: _rppo_factory(2, 2, seed),
        "train_vanilla": train_vanilla_recurrent, "train_optix": train_optix_recurrent,
    },
    "cartpole_neuroevo": {
        "key": "cartpole_neuroevo", "label": "CartPole Neuroevo", "env_id": "CartPole-v1",
        "algo": "NeuronGraph + TopologyController",
        "solve_threshold": 400.0, "max_steps": 300_000, "window_size": 512,
        "eval_every": 10_000, "n_hidden": 4,
        "train_vanilla": None, "train_optix": train_neuroevo,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Results table + charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt_k(vals):
    v = [x for x in vals if x is not None]
    if not v: return "    -      "
    m, s = np.mean(v), np.std(v)
    return f"{m/1000:>6.1f}k ±{s/1000:.1f}k"

def _fmt_f(vals):
    v = [x for x in vals if x is not None]
    if not v: return "    -      "
    m, s = np.mean(v), np.std(v)
    return f"{m:>7.1f} ±{s:.1f}"

def _fmt_s(vals):
    m, s = np.mean(vals), np.std(vals)
    return f"{m:>6.1f}s ±{s:.1f}s"

def _delta(a_list, b_list):
    a = [x for x in a_list if x is not None]
    b = [x for x in b_list if x is not None]
    if not a or not b or abs(np.mean(a)) < 1e-9: return "  -  "
    return f"{(np.mean(b) - np.mean(a)) / abs(np.mean(a)) * 100:>+.0f}%"

def print_table(cfg, vanilla, optix):
    W = 72; sep = "=" * W; line = "-" * W; n = len(vanilla) if vanilla else 0
    vt = [r["total_steps"] for r in vanilla] if vanilla else []
    ot = [r["total_steps"] for r in optix]
    vs = [r["final_score"] for r in vanilla] if vanilla else []
    os_ = [r["final_score"] for r in optix]
    vtime = [r["elapsed"] for r in vanilla] if vanilla else []
    otime = [r["elapsed"] for r in optix]
    vnsol = sum(r["solved"] for r in vanilla) if vanilla else 0
    onsol = sum(r["solved"] for r in optix)
    no    = len(optix)

    print(f"\n{sep}")
    print(f"  {cfg['label']}   |   {cfg['algo']}   |   {n} seed{'s' if n != 1 else ''}")
    print(sep)
    print(f"  {'Metric':<30} {'Baseline':>14}  {'tensor-optix':>14}  {'Δ':>5}")
    print(f"  {line}")
    print(f"  {'Total steps used':<30} {_fmt_k(vt):>14}  {_fmt_k(ot):>14}  {_delta(vt, ot):>5}")
    print(f"  {'Final eval score':<30} {_fmt_f(vs):>14}  {_fmt_f(os_):>14}  {_delta(vs, os_):>5}")
    if vtime: print(f"  {'Wall time':<30} {_fmt_s(vtime):>14}  {_fmt_s(otime):>14}  {_delta(vtime, otime):>5}")
    print(f"  {'Solved':<30} {f'{vnsol}/{n}':>14}  {f'{onsol}/{no}':>14}")
    print(sep)
    saved = np.mean(vt) - np.mean(ot) if vt else 0
    if saved > 0:
        print(f"\n  -> tensor-optix used {saved/np.mean(vt)*100:.0f}% fewer steps "
              f"({saved/1000:.0f}k saved). Baseline kept running after convergence.")
    if vs and np.mean(os_) > np.mean(vs):
        print(f"  -> Auto-tuning improved final score by {np.mean(os_) - np.mean(vs):.1f} points.")
    print()


VANILLA_COLOR = "#e06c75"
OPTIX_COLOR   = "#61afef"
NEURO_COLOR   = "#c678dd"


def _interp(history, max_step, n=200):
    if not history: return np.linspace(0, max_step, n), np.zeros(n)
    steps  = np.array([h[0] for h in history], dtype=float)
    scores = np.array([h[1] for h in history], dtype=float)
    grid   = np.linspace(steps[0], min(steps[-1], max_step), n)
    return grid, np.interp(grid, steps, scores)


def plot_results(all_results, env_configs, out_path="benchmarks/benchmark_results.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping chart. pip install matplotlib"); return

    keys   = [k for k in env_configs if k in all_results]
    n_envs = len(keys)
    if not n_envs: return

    fig, axes = plt.subplots(2, n_envs, figsize=(6 * n_envs, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    if n_envs == 1: axes = axes.reshape(2, 1)

    fig.patch.set_facecolor("#1e2127")
    for ax in axes.flat:
        ax.set_facecolor("#282c34")
        ax.tick_params(colors="#abb2bf", labelsize=9)
        ax.xaxis.label.set_color("#abb2bf"); ax.yaxis.label.set_color("#abb2bf")
        ax.title.set_color("#e5c07b")
        for sp in ax.spines.values(): sp.set_edgecolor("#3e4452")

    for col, key in enumerate(keys):
        cfg    = env_configs[key]
        res    = all_results[key]
        vanilla = res.get("vanilla", [])
        optix   = res.get("tensor_optix", [])
        neuro   = res.get("neuroevo", [])
        max_s   = cfg["max_steps"]
        ax      = axes[0, col]

        def _plot_runs(runs, color, label):
            if not runs: return
            grids, scores = zip(*[_interp(r["history"], max_s) for r in runs])
            mat  = np.vstack(scores)
            mean = mat.mean(axis=0)
            for g, s in zip(grids, scores):
                ax.plot(g / 1000, s, color=color, lw=0.7, alpha=0.3)
            ax.plot(grids[0] / 1000, mean, color=color, lw=2.2, label=label)
            # convergence marker for optix/neuro
            if label != "Baseline":
                stop = np.mean([r["total_steps"] for r in runs])
                if stop < max_s * 0.98:
                    ax.axvline(stop / 1000, color=color, ls="--", lw=1.2, alpha=0.7)
                    ax.text(stop / 1000, ax.get_ylim()[0], "converged",
                            color=color, fontsize=7, va="bottom", alpha=0.85)

        _plot_runs(vanilla, VANILLA_COLOR, "Baseline")
        _plot_runs(optix,   OPTIX_COLOR,   "tensor-optix")
        _plot_runs(neuro,   NEURO_COLOR,   "neuroevo")

        ax.axhline(cfg["solve_threshold"], color="#98c379", ls=":", lw=1, alpha=0.6)
        ax.text(0, cfg["solve_threshold"], f" solved={cfg['solve_threshold']:.0f}",
                color="#98c379", fontsize=7, va="bottom")
        ax.set_title(f"{cfg['label']}\n{cfg['algo']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Steps (thousands)", fontsize=9)
        ax.set_ylabel("Eval Reward", fontsize=9)
        ax.legend(fontsize=8, facecolor="#282c34", edgecolor="#3e4452", labelcolor="#abb2bf")
        ax.grid(True, color="#3e4452", lw=0.5, alpha=0.7)

        # Bar comparison
        ax2 = axes[1, col]
        cats = ["Steps Used\n(thousands)", "Final Score"]
        x = np.arange(len(cats)); w = 0.25; offset = -w
        for runs, color, label in [(vanilla, VANILLA_COLOR, "Baseline"),
                                   (optix, OPTIX_COLOR, "tensor-optix"),
                                   (neuro, NEURO_COLOR, "neuroevo")]:
            if not runs: offset += w; continue
            vals = [np.mean([r["total_steps"] for r in runs]) / 1000,
                    np.mean([r["final_score"]  for r in runs])]
            errs = [np.std([r["total_steps"] for r in runs]) / 1000,
                    np.std([r["final_score"]  for r in runs])]
            ax2.bar(x + offset, vals, w, yerr=errs, capsize=4, color=color, alpha=0.85,
                    label=label, error_kw={"ecolor": "#abb2bf", "lw": 1})
            offset += w

        ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=9)
        ax2.set_title("Key Metrics Comparison", fontsize=10)
        ax2.legend(fontsize=7, facecolor="#282c34", edgecolor="#3e4452", labelcolor="#abb2bf")
        ax2.grid(True, axis="y", color="#3e4452", lw=0.5, alpha=0.7)

    fig.suptitle("tensor-optix vs Fixed Training Loop",
                 fontsize=13, fontweight="bold", color="#e5c07b", y=1.01)
    plt.tight_layout(pad=1.5)
    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Chart saved → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="tensor-optix benchmark")
    parser.add_argument("--envs", nargs="+", default=list(ENV_CONFIGS.keys()),
                        choices=list(ENV_CONFIGS.keys()),
                        help="Which environments to benchmark (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline runs (optix + neuroevo only)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip matplotlib chart generation")
    parser.add_argument("--out", default="benchmarks/benchmark_results.json",
                        help="JSON results output path")
    args = parser.parse_args()

    print("\n" + "━" * 72)
    print("  tensor-optix Benchmark")
    print(f"  Envs:  {args.envs}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Baseline: {'disabled' if args.no_baseline else 'enabled'}")
    print("━" * 72 + "\n")

    all_results = {}

    for key in args.envs:
        cfg = ENV_CONFIGS[key]
        print(f"\n{'━'*72}")
        print(f"  {cfg['label']}  —  {cfg['algo']}")
        print("━" * 72)

        vanilla_results = []
        optix_results   = []
        neuro_results   = []

        # Baseline
        if not args.no_baseline and cfg.get("train_vanilla"):
            print("\n  [Baseline]")
            for seed in args.seeds:
                r = cfg["train_vanilla"](cfg, seed)
                vanilla_results.append(r)

        # tensor-optix
        if cfg.get("train_optix"):
            print("\n  [tensor-optix]")
            for seed in args.seeds:
                r = cfg["train_optix"](cfg, seed)
                optix_results.append(r)

        # Neuroevo (only for neuroevo configs)
        if key == "cartpole_neuroevo":
            print("\n  [Neuroevo]")
            for seed in args.seeds:
                r = train_neuroevo(cfg, seed)
                neuro_results.append(r)

        all_results[key] = {
            "vanilla": vanilla_results,
            "tensor_optix": optix_results,
            "neuroevo": neuro_results,
        }

        if vanilla_results or optix_results:
            print_table(cfg, vanilla_results or None, optix_results or neuro_results)

    # Save JSON
    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        def _clean(obj):
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_clean(x) for x in obj]
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            return obj
        json.dump(_clean(all_results), f, indent=2)
    print(f"\n  Results saved → {args.out}")

    if not args.no_chart:
        plot_results(all_results, ENV_CONFIGS)

    # Grand summary
    print("\n" + "━" * 72)
    print("  SUMMARY")
    print("━" * 72)
    for key in args.envs:
        res = all_results.get(key, {})
        v   = res.get("vanilla", [])
        o   = res.get("tensor_optix", [])
        n   = res.get("neuroevo", [])
        cfg = ENV_CONFIGS[key]
        if v and o:
            saved_pct = (np.mean([r["total_steps"] for r in v]) -
                         np.mean([r["total_steps"] for r in o])) / np.mean(
                         [r["total_steps"] for r in v]) * 100
            score_gain = np.mean([r["final_score"] for r in o]) - np.mean([r["final_score"] for r in v])
            o_solved = sum(r["solved"] for r in o); v_solved = sum(r["solved"] for r in v)
            print(f"  {cfg['label']:<28}  steps saved: {saved_pct:>+5.0f}%  "
                  f"score Δ: {score_gain:>+6.1f}  solved: {v_solved}→{o_solved}/{len(o)}")
        elif o:
            o_solved = sum(r["solved"] for r in o)
            print(f"  {cfg['label']:<28}  score: {np.mean([r['final_score'] for r in o]):>7.1f}  "
                  f"solved: {o_solved}/{len(o)}")
    print("━" * 72 + "\n")


if __name__ == "__main__":
    main()

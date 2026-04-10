#!/usr/bin/env python3
"""
tensor-optix Real-World Benchmark
==================================
Compares tensor-optix's autonomous training loop against an equivalent
baseline loop across four distinct problem types:

  1. Acrobot-v1                  - DQN, discrete, sparse rewards — natural DQN use case
  2. LunarLander-v3              - PPO, discrete, risk of local-optima collapse
  3. Acrobot-v1 (PPO)            - PPO, discrete, sparse rewards, hard exploration
  4. LunarLanderContinuous-v3    - SAC, continuous, same domain different paradigm
  5. BipedalWalker-v3            - SAC, continuous, complex locomotion (24-dim obs,
                                   4-dim actions, genuinely hard — reference benchmark)

The baseline loop uses the exact same algorithm, architecture, and starting
hyperparameters as tensor-optix — only the loop infrastructure differs.

  Baseline: fixed step budget, no convergence detection, no auto-tuning
  tensor-optix: autonomous loop, BackoffOptimizer, PolicyManager (rollback +
                policy spawning), stops when spawn budget exhausted

Usage:
    uv run python benchmarks/benchmark.py                    # all 4 envs, 2 seeds
    uv run python benchmarks/benchmark.py --envs lunarlander acrobot
    uv run python benchmarks/benchmark.py --seeds 0 1 2
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

import gymnasium as gym
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor_optix import (
    BackoffOptimizer,
    SPSAOptimizer,
    MomentumOptimizer,
    BatchPipeline,
    EpisodeData,
    EvalMetrics,
    HyperparamSet,
    LoopCallback,
    PolicyManager,
    TrialOrchestrator,
)
from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.algorithms.torch_sac import TorchSACAgent
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.optimizer import RLOptimizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared architecture builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mlp(in_dim: int, hidden: int, out_dim: int, act=nn.Tanh) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim),
    )


def build_ppo_nets(obs_dim: int, n_actions: int, hidden: int = 64):
    actor  = _mlp(obs_dim, hidden, n_actions)
    critic = _mlp(obs_dim, hidden, 1)
    return actor, critic


def build_dqn_net(obs_dim: int, n_actions: int, hidden: int = 128) -> nn.Sequential:
    """Q-network for DQN. ReLU activations — standard for value functions."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),  nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def build_sac_nets(obs_dim: int, action_dim: int, hidden: int = 256):
    # actor: obs -> [mean || log_std]  (2 * action_dim outputs)
    actor   = _mlp(obs_dim, hidden, action_dim * 2)
    # critics: [obs || action] -> Q-value
    critic1 = _mlp(obs_dim + action_dim, hidden, 1, act=nn.ReLU)
    critic2 = _mlp(obs_dim + action_dim, hidden, 1, act=nn.ReLU)
    return actor, critic1, critic2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Evaluation helpers  (bypass agent caches - safe to call any time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _net_device(net: nn.Module):
    """Return the device of the first parameter in a network."""
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def eval_dqn(q_net: nn.Module, env_id: str, n_eps: int = 10, seed: int = 9000) -> float:
    """Greedy (argmax) evaluation of a DQN Q-network. Cache-safe."""
    device = _net_device(q_net)
    env = gym.make(env_id)
    totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                action = int(q_net(obs_t).argmax(dim=-1).item())
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def eval_ppo(actor: nn.Module, env_id: str, n_eps: int = 10, seed: int = 9000) -> float:
    """Deterministic evaluation of a discrete PPO actor. Cache-safe."""
    device = _net_device(actor)
    env = gym.make(env_id)
    totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                action = int(torch.argmax(actor(obs_t), dim=-1).item())
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def eval_sac(actor: nn.Module, env_id: str, action_scale: float,
             n_eps: int = 10, seed: int = 9000) -> float:
    """Deterministic (mean) evaluation of a SAC actor. Cache-safe."""
    device = _net_device(actor)
    env = gym.make(env_id)
    totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                out = actor(obs_t)
                mean, _ = out.chunk(2, dim=-1)
                action = (torch.tanh(mean).cpu().numpy()[0] * action_scale)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vanilla training loop  (no tensor-optix loop features)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _collect_window(env, agent, window_size: int, carry_obs, needs_reset: bool):
    """Collect exactly window_size steps. Returns (EpisodeData, carry_obs)."""
    if needs_reset:
        carry_obs, _ = env.reset()
    obs = carry_obs
    observations, actions, rewards, terminated, truncated = [], [], [], [], []
    for _ in range(window_size):
        observations.append(obs)
        action = agent.act(obs)
        actions.append(action)
        next_obs, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        obs = (next_obs if not (term or trunc)
               else env.reset()[0])
    return EpisodeData(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        infos=[{}] * window_size,
        episode_id=0,
    ), obs


def train_vanilla_ppo(cfg: dict, seed: int) -> dict:
    """
    Baseline PPO. Same agent, same arch, same hyperparams as tensor-optix.
    Runs until budget exhausted with no early stopping or hyperparam changes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    actor, critic = build_ppo_nets(obs_dim, n_actions)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=cfg["lr"],
    )
    agent = TorchPPOAgent(
        actor=actor, critic=critic, optimizer=opt,
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
    )

    carry_obs, _ = env.reset(seed=seed)
    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    window_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        ep_data, carry_obs = _collect_window(
            env, agent, cfg["window_size"], carry_obs, needs_reset=False
        )
        ep_data.episode_id = window_id
        agent.learn(ep_data)
        total_steps += cfg["window_size"]
        since_eval  += cfg["window_size"]
        window_id   += 1

        if since_eval >= cfg["eval_every"] or total_steps >= cfg["max_steps"]:
            since_eval = 0
            score = cfg["eval_fn"](actor, cfg["env_id"], seed=seed + 10_000)
            history.append((total_steps, score))
            print(
                f"  [baseline  | {cfg['label']:16s} | seed={seed}]"
                f"  steps={total_steps:>7,d}  score={score:>8.1f}",
                flush=True,
            )
            if steps_to_solve is None and score >= cfg["solve_threshold"]:
                steps_to_solve = total_steps

    env.close()
    final = history[-1][1] if history else 0.0
    return {
        "method": "Baseline", "seed": seed,
        "total_steps": total_steps, "steps_to_solve": steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": history, "solved": steps_to_solve is not None,
    }


def train_vanilla_sac(cfg: dict, seed: int) -> dict:
    """Baseline SAC - same structure as PPO vanilla."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_seed = seed
    import random; random.seed(random_seed)

    env = gym.make(cfg["env_id"])
    obs_dim   = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    actor, c1, c2 = build_sac_nets(obs_dim, action_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    agent = TorchSACAgent(
        actor=actor, critic1=c1, critic2=c2,
        action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=cfg["lr"]),
        critic_optimizer=torch.optim.Adam(
            list(c1.parameters()) + list(c2.parameters()), lr=cfg["lr"]
        ),
        alpha_optimizer=torch.optim.Adam([log_alpha], lr=cfg["lr"]),
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
    )

    carry_obs, _ = env.reset(seed=seed)
    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    window_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        # SAC needs per-step next_obs for the replay buffer.
        # We store obs[t] and obs[t+1] inside EpisodeData
        # TorchSACAgent.learn() indexes obs[t] and obs[t+1].
        observations, actions, rewards, terminated, truncated = [], [], [], [], []
        obs = carry_obs
        for _ in range(cfg["window_size"]):
            observations.append(obs)
            # Scale SAC action to env range
            raw_action = agent.act(obs)          # in [-1, 1]
            action = raw_action * action_scale   # scale to env range
            next_obs, reward, term, trunc, _ = env.step(action)
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            actions.append(raw_action)           # store unscaled (what SAC sees)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        carry_obs = obs
        observations.append(carry_obs)  # one extra obs so SAC has next_obs for last step

        ep_data = EpisodeData(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=[{}] * cfg["window_size"],
            episode_id=window_id,
        )
        agent.learn(ep_data)
        total_steps += cfg["window_size"]
        since_eval  += cfg["window_size"]
        window_id   += 1

        if since_eval >= cfg["eval_every"] or total_steps >= cfg["max_steps"]:
            since_eval = 0
            score = eval_sac(actor, cfg["env_id"], action_scale,
                             seed=seed + 10_000)
            history.append((total_steps, score))
            print(
                f"  [baseline  | {cfg['label']:16s} | seed={seed}]"
                f"  steps={total_steps:>7,d}  score={score:>8.1f}",
                flush=True,
            )
            if steps_to_solve is None and score >= cfg["solve_threshold"]:
                steps_to_solve = total_steps

    env.close()
    final = history[-1][1] if history else 0.0
    return {
        "method": "Baseline", "seed": seed,
        "total_steps": total_steps, "steps_to_solve": steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": history, "solved": steps_to_solve is not None,
    }


def train_vanilla_dqn(cfg: dict, seed: int) -> dict:
    """
    Baseline DQN. Same agent, same arch, same hyperparams as tensor-optix.
    Off-policy: windows of experience feed the replay buffer continuously.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = build_dqn_net(obs_dim, n_actions)
    agent = TorchDQNAgent(
        q_network=q_net, n_actions=n_actions,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=cfg["lr"]),
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
    )

    carry_obs, _ = env.reset(seed=seed)
    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    window_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        # Collect window. DQN uses obs[t] and obs[t+1] pairs internally.
        # BatchPipeline gives window_size obs; agent processes window_size-1
        # transitions (last step's next_obs is carry_obs from next window).
        ep_data, carry_obs = _collect_window(
            env, agent, cfg["window_size"], carry_obs, needs_reset=False
        )
        ep_data.episode_id = window_id
        agent.learn(ep_data)
        total_steps += cfg["window_size"]
        since_eval  += cfg["window_size"]
        window_id   += 1

        if since_eval >= cfg["eval_every"] or total_steps >= cfg["max_steps"]:
            since_eval = 0
            score = cfg["eval_fn"](q_net, cfg["env_id"], seed=seed + 10_000)
            history.append((total_steps, score))
            print(
                f"  [baseline  | {cfg['label']:16s} | seed={seed}]"
                f"  steps={total_steps:>7,d}  score={score:>8.1f}",
                flush=True,
            )
            if steps_to_solve is None and score >= cfg["solve_threshold"]:
                steps_to_solve = total_steps

    env.close()
    final = history[-1][1] if history else 0.0
    return {
        "method": "Baseline", "seed": seed,
        "total_steps": total_steps, "steps_to_solve": steps_to_solve,
        "final_score": final, "elapsed": time.perf_counter() - t0,
        "history": history, "solved": steps_to_solve is not None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  tensor-optix callback  (shared for all envs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Tracker(LoopCallback):
    def __init__(self, cfg: dict, actor: nn.Module, seed: int):
        self._cfg   = cfg
        self._actor = actor
        self._seed  = seed
        self.total_steps      = 0
        self.steps_to_solve: Optional[int] = None
        self.history: list[tuple[int, float]] = []
        self._since_eval = 0
        self._converged  = False

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        self.total_steps  += self._cfg["window_size"]
        self._since_eval  += self._cfg["window_size"]
        if self._since_eval >= self._cfg["eval_every"]:
            self._since_eval = 0
            score = self._cfg["eval_fn"](self._actor, self._cfg["env_id"],
                                         seed=self._seed + 10_000)
            self.history.append((self.total_steps, score))
            print(
                f"  [optix     | {self._cfg['label']:16s} | seed={self._seed}]"
                f"  steps={self.total_steps:>7,d}  score={score:>8.1f}",
                flush=True,
            )
            if self.steps_to_solve is None and score >= self._cfg["solve_threshold"]:
                self.steps_to_solve = self.total_steps

    def on_dormant(self, episode_id: int) -> None:
        if not self._converged:
            self._converged = True
            print(
                f"  [optix     | {self._cfg['label']:16s} | seed={self._seed}]"
                f"  *** CONVERGED at {self.total_steps:,d} steps ***",
                flush=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  tensor-optix training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_optix_ppo(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous PPO — the full feature set:
      - TrialOrchestrator (TPE): searches learning_rate, clip_ratio, entropy_coef
      - SPSA: online adaptation during the full run
      - Exponential-backoff convergence detection (ACTIVE -> COOLING -> DORMANT)
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    env.close()

    def agent_factory(params):
        merged = {**cfg["hp"], **params}
        a, c = build_ppo_nets(obs_dim, n_actions)
        return TorchPPOAgent(
            actor=a, critic=c,
            optimizer=torch.optim.Adam(
                list(a.parameters()) + list(c.parameters()),
                lr=params.get("learning_rate", cfg["lr"]),
            ),
            hyperparams=HyperparamSet(params=merged, episode_id=0),
        )

    def pipeline_factory():
        e = gym.make(cfg["env_id"])
        a, c = build_ppo_nets(obs_dim, n_actions)
        ag = TorchPPOAgent(
            actor=a, critic=c,
            optimizer=torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=cfg["lr"]),
            hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
        )
        return BatchPipeline(env=e, agent=ag, window_size=cfg["window_size"])

    ckpt_dir = f"./benchmarks/.ckpts/ppo_{cfg['key']}_{seed}"
    evaluator = TorchEvaluator(primary_score_fn=cfg.get("internal_score_fn"))
    pm_state: dict = {}
    tracker_holder: dict = {}

    def agent_factory_full(params):
        agent = agent_factory(params)
        registry = CheckpointRegistry(ckpt_dir)
        pm = PolicyManager(registry, max_spawns=3)
        pm_cb = pm.as_callback(agent, agent_factory=lambda: agent_factory(params))
        tracker = _Tracker(cfg=cfg, actor=agent._actor, seed=seed)
        pm_state.update({"pm_cb": pm_cb, "actor": agent._actor})
        tracker_holder["tracker"] = tracker
        rl_opt.add_callback(tracker)
        rl_opt.add_callback(pm_cb)
        return agent

    t0 = time.perf_counter()
    rl_opt = RLOptimizer(
        agent_factory=agent_factory_full,
        trial_agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space={
            "learning_rate": ("log_float", 1e-4, 3e-3),
            "clip_ratio":    ("float",     0.1,  0.3),
            "entropy_coef":  ("float",     0.0,  0.05),
        },
        n_trials=10,
        evaluator=evaluator,
        optimizer=SPSAOptimizer(param_bounds={
            "learning_rate": (1e-4, 3e-3),
            "clip_ratio":    (0.1,  0.3),
            "entropy_coef":  (0.0,  0.05),
        }, log_params=["learning_rate"]),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        dormant_threshold=6,
        max_interval_episodes=8,
        min_episodes_before_dormant=50,
        checkpoint_score_fn=lambda agent: cfg["eval_fn"](pm_state.get("actor", agent._actor), cfg["env_id"], seed=seed + 10_000),
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_state["rl_opt"] = rl_opt  # allow pm_cb stop_fn to reference rl_opt after construction
    rl_opt.run()

    tracker = tracker_holder["tracker"]
    actor   = pm_state["actor"]
    final_score = cfg["eval_fn"](actor, cfg["env_id"], seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final_score))
    if tracker.steps_to_solve is None and final_score >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    print(
        f"  [optix     | {cfg['label']:16s} | seed={seed}]"
        f"  steps={tracker.total_steps:>7,d}  score={final_score:>8.1f}  [final]",
        flush=True,
    )

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "tensor-optix", "seed": seed,
        "total_steps": tracker.total_steps,
        "steps_to_solve": tracker.steps_to_solve,
        "final_score": final_score, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
    }


def train_optix_sac(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous SAC:
      - No TrialOrchestrator: same problem as DQN — off-policy replay buffer
        warmup means short trials can't differentiate lrs. cfg["hp"] defaults
        are well-tuned; SPSA handles all online adaptation.
      - SPSA: online adaptation (learning_rate, gamma, tau)
      - Exponential-backoff convergence detection
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random; random.seed(seed)

    env_tmp = gym.make(cfg["env_id"])
    obs_dim    = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    action_scale = float(env_tmp.action_space.high[0])
    env_tmp.close()

    import shutil
    ckpt_dir = f"./benchmarks/.ckpts/sac_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    cfg_for_tracker = dict(cfg)
    cfg_for_tracker["eval_fn"] = lambda a, env_id, seed: eval_sac(a, env_id, action_scale, seed=seed)

    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        a, c1, c2 = build_sac_nets(obs_dim, action_dim)
        la = torch.tensor(0.0, requires_grad=True)
        lr = hp.get("learning_rate", cfg["lr"])
        return TorchSACAgent(
            actor=a, critic1=c1, critic2=c2, action_dim=action_dim,
            actor_optimizer=torch.optim.Adam(a.parameters(), lr=lr),
            critic_optimizer=torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr),
            alpha_optimizer=torch.optim.Adam([la], lr=lr),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
        )

    t0 = time.perf_counter()
    agent = make_agent()
    registry = CheckpointRegistry(ckpt_dir)
    pm = PolicyManager(registry, max_spawns=3)
    pm_cb = pm.as_callback(agent, agent_factory=make_agent)
    tracker = _Tracker(cfg=cfg_for_tracker, actor=agent._actor, seed=seed)
    pm_state.update({"pm_cb": pm_cb, "actor": agent._actor})
    tracker_holder["tracker"] = tracker

    env = gym.make(cfg["env_id"])
    wrapped = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    pipeline = BatchPipeline(env=wrapped, agent=agent, window_size=cfg["window_size"])

    rl_opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        evaluator=TorchEvaluator(),
        optimizer=SPSAOptimizer(param_bounds={
            "learning_rate": (1e-4, 3e-3),
            "gamma":         (0.97, 0.999),
            "tau":           (1e-3, 1e-1),
        }, log_params=["learning_rate", "tau"]),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8,
        dormant_threshold=8,  # recover faster from off-policy collapse
        checkpoint_score_fn=lambda a: eval_sac(pm_state.get("actor", a._actor), cfg["env_id"], action_scale, seed=seed + 10_000),
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    rl_opt.add_callback(tracker)
    rl_opt.add_callback(pm_cb)
    rl_opt.run()

    tracker = tracker_holder["tracker"]
    actor   = pm_state["actor"]
    final_score = eval_sac(actor, cfg["env_id"], action_scale, seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final_score))
    if tracker.steps_to_solve is None and final_score >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    print(
        f"  [optix     | {cfg['label']:16s} | seed={seed}]"
        f"  steps={tracker.total_steps:>7,d}  score={final_score:>8.1f}  [final]",
        flush=True,
    )

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "tensor-optix", "seed": seed,
        "total_steps": tracker.total_steps,
        "steps_to_solve": tracker.steps_to_solve,
        "final_score": final_score, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
    }


def train_optix_dqn(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous DQN:
      - No TrialOrchestrator: DQN needs 150k+ steps before lr performance
        differentiates. Short trials give TPE no signal — it picks randomly
        and consistently lands near the lr floor, causing the full run to
        stall at 9.2 for the entire budget. cfg["hp"] defaults are well-tuned.
      - SPSA: online adaptation (learning_rate, gamma, epsilon_decay)
      - Exponential-backoff convergence detection (ACTIVE -> COOLING -> DORMANT)
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_tmp = gym.make(cfg["env_id"])
    obs_dim   = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    import shutil
    ckpt_dir = f"./benchmarks/.ckpts/dqn_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # fresh start — no stale checkpoint pollution
    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        q = build_dqn_net(obs_dim, n_actions)
        return TorchDQNAgent(
            q_network=q, n_actions=n_actions,
            optimizer=torch.optim.Adam(q.parameters(), lr=hp.get("learning_rate", cfg["lr"])),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
        )

    t0 = time.perf_counter()
    agent = make_agent()
    registry = CheckpointRegistry(ckpt_dir)
    pm = PolicyManager(registry, max_spawns=3)
    pm_cb = pm.as_callback(agent, agent_factory=make_agent)
    tracker = _Tracker(cfg=cfg, actor=agent._q, seed=seed)
    pm_state.update({"pm_cb": pm_cb, "q_net": agent._q})
    tracker_holder["tracker"] = tracker

    env = gym.make(cfg["env_id"])
    pipeline = BatchPipeline(env=env, agent=agent, window_size=cfg["window_size"])

    rl_opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        evaluator=TorchEvaluator(),
        optimizer=SPSAOptimizer(param_bounds={
            "learning_rate": (1e-4, 1e-3),
            "gamma":         (0.95, 0.999),
        }, log_params=["learning_rate"]),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8,
        dormant_threshold=8,
        min_episodes_before_dormant=60,  # epsilon_decay=0.95 reaches floor at window~58; don't quit before then
        checkpoint_score_fn=lambda a: cfg["eval_fn"](pm_state.get("q_net", a._q), cfg["env_id"], seed=seed + 10_000),
        diag_min_episodes=50,  # DQN early losses are noisy; don't act on DIAG until buffer is warm
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    rl_opt.add_callback(tracker)
    rl_opt.add_callback(pm_cb)
    rl_opt.run()

    tracker = tracker_holder["tracker"]
    q_net   = pm_state["q_net"]
    final_score = cfg["eval_fn"](q_net, cfg["env_id"], seed=seed + 10_000)
    tracker.history.append((tracker.total_steps, final_score))
    if tracker.steps_to_solve is None and final_score >= cfg["solve_threshold"]:
        tracker.steps_to_solve = tracker.total_steps
    print(
        f"  [optix     | {cfg['label']:16s} | seed={seed}]"
        f"  steps={tracker.total_steps:>7,d}  score={final_score:>8.1f}  [final]",
        flush=True,
    )

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {
        "method": "tensor-optix", "seed": seed,
        "total_steps": tracker.total_steps,
        "steps_to_solve": tracker.steps_to_solve,
        "final_score": final_score, "elapsed": time.perf_counter() - t0,
        "history": tracker.history, "solved": tracker.steps_to_solve is not None,
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
    # epsilon_decay=0.95 per window (512 steps): hits epsilon_min=0.01
    # after ~85 windows (~43K steps). DQN starts exploiting at 43K steps,
    # leaving the remainder for greedy learning within the 500K budget.
    "learning_rate":      5e-4,
    "gamma":              0.99,
    "epsilon":            1.0,
    "epsilon_min":        0.01,
    "epsilon_decay":      0.95,
    "batch_size":         64,
    "target_update_freq": 25,
    "replay_capacity":    20_000,
}

ENV_CONFIGS = {
    "cartpole": {
        # CartPole-v1 with DQN — demonstrates framework is algorithm-agnostic
        # (off-policy, experience replay, epsilon-greedy). DQN learns CartPole
        # but with noisy oscillating scores — a good stress test for the loop.
        # Solve threshold: 475 (Gymnasium standard).
        "key":             "cartpole",
        "label":           "CartPole-v1",
        "env_id":          "CartPole-v1",
        "algo":            "DQN (discrete, off-policy)",
        "solve_threshold":  475.0,
        "max_steps":        500_000,
        "window_size":      512,
        "eval_every":       10_000,
        "lr":               1e-3,
        "hp":               DQN_HP,
        "eval_fn": lambda net, env_id, seed=9000: eval_dqn(net, env_id, seed=seed),
        "train_vanilla":    train_vanilla_dqn,
        "train_optix":      train_optix_dqn,
    },
    "lunarlander": {
        "key":             "lunarlander",
        "label":           "LunarLander-v3",
        "env_id":          "LunarLander-v3",
        "algo":            "PPO (discrete)",
        "solve_threshold":  200.0,
        "max_steps":        500_000,
        "window_size":      2_048,
        "eval_every":       20_000,
        "lr":               3e-4,
        "hp":               PPO_HP,
        "eval_fn":          lambda actor, env_id, seed=9000: eval_ppo(actor, env_id, seed=seed),
        "train_vanilla":    train_vanilla_ppo,
        "train_optix":      train_optix_ppo,
    },
    "lunarlander_continuous": {
        "key":             "lunarlander_continuous",
        "label":           "LunarLanderContinuous-v3",
        "env_id":          "LunarLanderContinuous-v3",
        "algo":            "SAC (continuous)",
        "solve_threshold":  200.0,      # same target as discrete version
        "max_steps":        600_000,
        "window_size":      2_048,
        "eval_every":       20_000,
        "lr":               3e-4,
        "hp":               SAC_HP,
        # action space is already Box(-1, 1) — action_scale = 1.0
        "eval_fn": lambda actor, env_id, seed=9000: eval_sac(actor, env_id, 1.0, seed=seed),
        "train_vanilla":    train_vanilla_sac,
        "train_optix":      train_optix_sac,
    },
    "acrobot": {
        # Acrobot-v1 with PPO — sparse rewards, hard exploration.
        # The agent must swing the end-effector above the bar; reward is -1
        # per step until solved. PPO with entropy bonus handles the sparse
        # signal well. Solve threshold: -100 (Gymnasium standard).
        "key":             "acrobot",
        "label":           "Acrobot-v1",
        "env_id":          "Acrobot-v1",
        "algo":            "PPO (discrete)",
        "solve_threshold":  -100.0,
        "max_steps":        400_000,
        "window_size":      2_048,
        "eval_every":       20_000,
        "lr":               3e-4,
        "hp":               PPO_HP,
        "eval_fn":          lambda actor, env_id, seed=9000: eval_ppo(actor, env_id, seed=seed),
        "train_vanilla":    train_vanilla_ppo,
        "train_optix":      train_optix_ppo,
    },
    "bipedalwalker": {
        # Complex locomotion: 24-dim continuous obs, 4-dim continuous actions.
        # One of the hardest standard continuous control benchmarks.
        # SAC is state-of-the-art for this. Vanilla SAC often collapses mid-training.
        # tensor-optix rollback + policy spawning is the exact mechanism needed.
        "key":             "bipedalwalker",
        "label":           "BipedalWalker-v3",
        "env_id":          "BipedalWalker-v3",
        "algo":            "SAC (continuous, complex locomotion)",
        "solve_threshold":  300.0,      # Gymnasium solve threshold
        "max_steps":        1_500_000,
        "window_size":      2_048,
        "eval_every":       50_000,
        "lr":               3e-4,
        "hp":               {**SAC_HP, "replay_capacity": 300_000, "batch_size": 256},
        "eval_fn": lambda actor, env_id, seed=9000: eval_sac(actor, env_id, 1.0, seed=seed),
        "train_vanilla":    train_vanilla_sac,
        "train_optix":      train_optix_sac,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Results table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt_k(vals: list) -> str:
    valid = [v for v in vals if v is not None]
    if not valid:
        return "    -      "
    m, s = np.mean(valid), np.std(valid)
    return f"{m/1000:>6.1f}k ±{s/1000:.1f}k"

def _fmt_f(vals: list) -> str:
    valid = [v for v in vals if v is not None]
    if not valid:
        return "    -      "
    m, s = np.mean(valid), np.std(valid)
    return f"{m:>7.1f} ±{s:.1f}"

def _fmt_s(vals: list) -> str:
    m, s = np.mean(vals), np.std(vals)
    return f"{m:>6.1f}s ±{s:.1f}s"

def _delta(a_list, b_list) -> str:
    a = [v for v in a_list if v is not None]
    b = [v for v in b_list if v is not None]
    if not a or not b:
        return "  -  "
    pct = (np.mean(b) - np.mean(a)) / abs(np.mean(a)) * 100
    return f"{pct:>+.0f}%"

def print_table(cfg: dict, vanilla: list[dict], optix: list[dict]) -> None:
    W = 70
    sep  = "=" * W
    line = "-" * W
    n    = len(vanilla)

    v_solve = [r["steps_to_solve"] for r in vanilla]
    o_solve = [r["steps_to_solve"] for r in optix]
    v_total = [r["total_steps"]    for r in vanilla]
    o_total = [r["total_steps"]    for r in optix]
    v_score = [r["final_score"]    for r in vanilla]
    o_score = [r["final_score"]    for r in optix]
    v_time  = [r["elapsed"]        for r in vanilla]
    o_time  = [r["elapsed"]        for r in optix]
    v_nsol  = sum(r["solved"] for r in vanilla)
    o_nsol  = sum(r["solved"] for r in optix)

    print(f"\n{sep}")
    print(f"  {cfg['label']}   |   {cfg['algo']}   |   {n} seed{'s' if n > 1 else ''}")
    print(sep)
    print(f"  {'Metric':<28} {'Baseline':>16}  {'tensor-optix':>16}  {'D':>5}")
    print(f"  {line}")
    print(f"  {'Steps to reach threshold':<28} {_fmt_k(v_solve):>16}  {_fmt_k(o_solve):>16}  {_delta(v_solve,o_solve):>5}")
    print(f"  {'Total steps used':<28} {_fmt_k(v_total):>16}  {_fmt_k(o_total):>16}  {_delta(v_total,o_total):>5}")
    print(f"  {'Final eval score':<28} {_fmt_f(v_score):>16}  {_fmt_f(o_score):>16}  {_delta(v_score,o_score):>5}")
    print(f"  {'Wall time':<28} {_fmt_s(v_time):>16}  {_fmt_s(o_time):>16}  {_delta(v_time,o_time):>5}")
    print(f"  {'Solved':<28} {f'{v_nsol}/{n}':>16}  {f'{o_nsol}/{n}':>16}")
    print(sep)

    # Plain-English summary
    o_total_mean = np.mean(o_total)
    v_total_mean = np.mean(v_total)
    saved = v_total_mean - o_total_mean
    if saved > 0:
        pct = saved / v_total_mean * 100
        print(f"\n  -> tensor-optix used {pct:.0f}% fewer steps ({saved/1000:.0f}k saved per run).")
        print(f"    The baseline loop kept running after the task was already solved.")

    o_s = [v for v in o_solve if v is not None]
    v_s = [v for v in v_solve if v is not None]
    if o_s and v_s and np.mean(o_s) < np.mean(v_s):
        ratio = np.mean(v_s) / np.mean(o_s)
        print(f"  -> tensor-optix reached the solve threshold {ratio:.1f}x faster.")

    score_gain = np.mean(o_score) - np.mean(v_score)
    if score_gain > 0:
        print(f"  -> Auto-tuning improved final score by {score_gain:.1f} points.")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Matplotlib charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VANILLA_COLOR = "#e06c75"   # red
OPTIX_COLOR   = "#61afef"   # blue


def _interp_history(history: list[tuple[int, float]], max_step: int, n: int = 200):
    """Linearly interpolate score history onto a uniform step grid."""
    if not history:
        return np.linspace(0, max_step, n), np.zeros(n)
    steps  = np.array([h[0] for h in history], dtype=float)
    scores = np.array([h[1] for h in history], dtype=float)
    grid   = np.linspace(steps[0], min(steps[-1], max_step), n)
    return grid, np.interp(grid, steps, scores)


def plot_results(all_results: dict, env_configs: dict, out_path: str = "benchmarks/benchmark_results.png") -> None:
    """
    Generate a 2-row figure:
      Row 1: Learning curves (reward vs steps) for each environment.
      Row 2: Bar comparison of total steps used and final score.
    Saves to out_path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except ImportError:
        print("  matplotlib not installed - skipping chart generation.")
        print("  Install with: pip install matplotlib")
        return

    env_keys = [k for k in env_configs if k in all_results]
    n_envs   = len(env_keys)
    if n_envs == 0:
        return

    fig, axes = plt.subplots(
        2, n_envs,
        figsize=(6 * n_envs, 9),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if n_envs == 1:
        axes = axes.reshape(2, 1)

    fig.patch.set_facecolor("#1e2127")
    for ax in axes.flat:
        ax.set_facecolor("#282c34")
        ax.tick_params(colors="#abb2bf", labelsize=9)
        ax.xaxis.label.set_color("#abb2bf")
        ax.yaxis.label.set_color("#abb2bf")
        ax.title.set_color("#e5c07b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3e4452")

    for col, key in enumerate(env_keys):
        cfg     = env_configs[key]
        res     = all_results[key]
        vanilla = res["vanilla"]
        optix   = res["tensor_optix"]
        max_s   = cfg["max_steps"]

        # ── Row 0: learning curves ──────────────────────────────────────
        ax = axes[0, col]

        # Vanilla
        if vanilla:
            v_grids, v_scores = zip(*[
                _interp_history(r["history"], max_s) for r in vanilla
            ])
            v_grid  = v_grids[0]
            v_mat   = np.vstack(v_scores)
            v_mean  = v_mat.mean(axis=0)
            for grid, scores in zip(v_grids, v_scores):
                ax.plot(grid / 1000, scores, color=VANILLA_COLOR, lw=0.8, alpha=0.35)
            ax.plot(v_grid / 1000, v_mean, color=VANILLA_COLOR, lw=2, label="Baseline")

        # tensor-optix
        o_grids, o_scores = zip(*[
            _interp_history(r["history"], max_s) for r in optix
        ])
        o_grid = o_grids[0]
        o_mat  = np.vstack(o_scores)
        o_mean = o_mat.mean(axis=0)
        for grid, scores in zip(o_grids, o_scores):
            ax.plot(grid / 1000, scores, color=OPTIX_COLOR, lw=0.8, alpha=0.35)
        ax.plot(o_grid / 1000, o_mean, color=OPTIX_COLOR, lw=2, label="tensor-optix")

        # Mark where tensor-optix stopped (mean convergence point)
        o_totals = [r["total_steps"] for r in optix]
        o_stop   = np.mean(o_totals)
        if o_stop < max_s * 0.98:
            ax.axvline(o_stop / 1000, color=OPTIX_COLOR, linestyle="--",
                       linewidth=1.2, alpha=0.7)
            ax.text(o_stop / 1000 + max_s / 1000 * 0.01,
                    ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else v_mean.min(),
                    "converged", color=OPTIX_COLOR, fontsize=7, va="bottom", alpha=0.9)

        # Solve threshold
        ax.axhline(cfg["solve_threshold"], color="#98c379", linestyle=":",
                   linewidth=1, alpha=0.6)
        ax.text(0, cfg["solve_threshold"], f" solved={cfg['solve_threshold']:.0f}",
                color="#98c379", fontsize=7, va="bottom")

        ax.set_title(f"{cfg['label']}\n{cfg['algo']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Steps (thousands)", fontsize=9)
        ax.set_ylabel("Eval Reward", fontsize=9)
        ax.legend(fontsize=8, facecolor="#282c34", edgecolor="#3e4452",
                  labelcolor="#abb2bf")
        ax.grid(True, color="#3e4452", linewidth=0.5, alpha=0.7)

        # ── Row 1: bar comparison ────────────────────────────────────────
        ax2 = axes[1, col]

        categories = ["Steps Used\n(thousands)", "Final Score"]
        o_vals = [np.mean(o_totals) / 1000, np.mean([r["final_score"] for r in optix])]
        o_errs = [np.std(o_totals) / 1000,  np.std([r["final_score"] for r in optix])]

        x = np.arange(len(categories))
        w = 0.35
        if vanilla:
            v_step_totals = [r["total_steps"] for r in vanilla]
            v_vals = [np.mean(v_step_totals) / 1000, np.mean([r["final_score"] for r in vanilla])]
            v_errs = [np.std(v_step_totals) / 1000,  np.std([r["final_score"] for r in vanilla])]
            ax2.bar(x - w/2, v_vals, w, yerr=v_errs, capsize=4,
                    color=VANILLA_COLOR, alpha=0.85, label="Baseline",
                    error_kw={"ecolor": "#abb2bf", "linewidth": 1})
        ax2.bar(x + w/2, o_vals, w, yerr=o_errs, capsize=4,
                color=OPTIX_COLOR,   alpha=0.85, label="tensor-optix",
                error_kw={"ecolor": "#abb2bf", "linewidth": 1})

        # Annotate % change (only when vanilla data available)
        if vanilla:
            for i, (vv, ov) in enumerate(zip(v_vals, o_vals)):
                if vv != 0:
                    pct = (ov - vv) / abs(vv) * 100
                    col_txt = OPTIX_COLOR if pct < 0 else "#98c379"
                    ax2.text(x[i], max(vv, ov) * 1.05 + max(v_errs[i], o_errs[i]),
                             f"{pct:+.0f}%", ha="center", va="bottom",
                             color=col_txt, fontsize=8, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=9)
        ax2.set_title("Key Metrics Comparison", fontsize=10)
        ax2.legend(fontsize=7, facecolor="#282c34", edgecolor="#3e4452",
                   labelcolor="#abb2bf")
        ax2.grid(True, axis="y", color="#3e4452", linewidth=0.5, alpha=0.7)

    # Main title
    fig.suptitle(
        "tensor-optix vs. Fixed Training Loop",
        fontsize=13, fontweight="bold", color="#e5c07b", y=1.01,
    )

    plt.tight_layout(pad=1.5)
    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved -> {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(envs: list[str], seeds: list[int], plot: bool = True, optix_only: bool = False, verbose: bool = False, verbose_log_file: Optional[str] = None) -> None:
    all_results: dict = {}
    grand_start = time.perf_counter()

    for key in envs:
        cfg = dict(ENV_CONFIGS[key])
        print("\n" + "=" * 70)
        print(f"  {cfg['label']}  |  {cfg['algo']}")
        print(f"  Budget: {cfg['max_steps']:,d} steps  |  Window: {cfg['window_size']}  |  Seeds: {seeds}")
        print("=" * 70)

        vanilla_runs, optix_runs = [], []

        for seed in seeds:
            if not optix_only:
                print(f"\n-- Baseline  seed={seed} " + "-" * 46)
                vanilla_runs.append(cfg["train_vanilla"](cfg, seed))

            print(f"\n-- tensor-optix  seed={seed} " + "-" * 46)
            optix_runs.append(cfg["train_optix"](cfg, seed, verbose=verbose, verbose_log_file=verbose_log_file))

        if not optix_only:
            print_table(cfg, vanilla_runs, optix_runs)
        all_results[key] = {"vanilla": vanilla_runs, "tensor_optix": optix_runs}

    total_elapsed = time.perf_counter() - grand_start
    print(f"  Total benchmark time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    out = Path("benchmarks/results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Raw data -> {out}")

    if plot:
        # Pass only the configs that were actually benchmarked
        run_cfgs = {k: ENV_CONFIGS[k] for k in envs}
        suffix = "_".join(envs) if len(envs) > 1 else envs[0]
        plot_results(all_results, run_cfgs, out_path=f"benchmarks/benchmark_results_{suffix}.png")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tensor-optix vs Baseline training loop benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--envs", nargs="+",
        choices=list(ENV_CONFIGS.keys()),
        default=["cartpole", "lunarlander", "acrobot", "lunarlander_continuous", "bipedalwalker"],
        metavar="ENV",
        help="Environments (default: all four)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib chart generation",
    )
    parser.add_argument(
        "--optix-only", action="store_true",
        help="Run tensor-optix training only (skip baseline)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode eval output during tensor-optix training",
    )
    parser.add_argument(
        "--verbose-log-file", default=None,
        help="Write verbose output to this file instead of stdout",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  torch {torch.__version__}  |  device: {device}")
    print(f"  gymnasium {gym.__version__}")
    if device == "cpu":
        print("  Note: CPU device detected. Install torch+cuda for faster runs.")

    run(envs=args.envs, seeds=args.seeds, plot=not args.no_plot, optix_only=args.optix_only, verbose=args.verbose, verbose_log_file=args.verbose_log_file)


if __name__ == "__main__":
    main()

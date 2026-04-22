#!/usr/bin/env python3
"""
tensor-optix Real-World Benchmark
==================================
Compares tensor-optix's autonomous training loop against an equivalent
baseline loop across five distinct problem types:

  1. CartPole-v1                 - DQN, discrete, classic balance task
  2. LunarLander-v3              - PPO, discrete, risk of local-optima collapse
  3. Acrobot-v1                  - PPO, discrete, sparse rewards, hard exploration
  4. LunarLanderContinuous-v3    - SAC, continuous, same domain different paradigm
  5. BipedalWalker-v3            - SAC, continuous, complex locomotion (24-dim obs,
                                   4-dim actions, genuinely hard — reference benchmark)

The baseline loop uses the exact same algorithm, architecture, and starting
hyperparameters as tensor-optix — only the loop infrastructure differs.

  Baseline: fixed step budget, no convergence detection, no auto-tuning
  tensor-optix: autonomous loop, BackoffOptimizer, PolicyManager (rollback +
                policy spawning), stops when spawn budget exhausted

Usage:
    uv run python benchmarks/benchmark.py                    # all 5 envs, 3 seeds
    uv run python benchmarks/benchmark.py --envs cartpole lunarlander
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
    AdaptiveOptimizer,
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
from tensor_optix.algorithms.torch_td3 import TorchTD3Agent
from tensor_optix.algorithms.torch_rainbow_dqn import TorchRainbowDQNAgent, RainbowQNetwork
from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent
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


def build_td3_nets(obs_dim: int, action_dim: int, hidden: int = 256):
    actor = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),  nn.ReLU(),
        nn.Linear(hidden, action_dim), nn.Tanh(),
    )
    critic1 = nn.Sequential(
        nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    critic2 = nn.Sequential(
        nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    return actor, critic1, critic2


def build_rppo_nets(obs_dim: int, n_actions: int, hidden: int = 64):
    rnn         = nn.LSTM(obs_dim, hidden, batch_first=True)
    actor_head  = nn.Linear(hidden, n_actions)
    critic_head = nn.Linear(hidden, 1)
    return rnn, actor_head, critic_head


class _POMDPCartPoleWrapper(gym.ObservationWrapper):
    """Mask velocity components (indices 1, 3) — obs_dim shrinks from 4 to 2."""
    def __init__(self, env):
        super().__init__(env)
        import gymnasium.spaces as _spaces
        low  = env.observation_space.low[[0, 2]]
        high = env.observation_space.high[[0, 2]]
        self.observation_space = _spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs[[0, 2]]


from tensor_optix.core.base_pipeline import BasePipeline as _BasePipeline


class _RecurrentBatchPipeline(_BasePipeline):
    """Window pipeline that calls agent.reset_hidden() at episode boundaries."""

    def __init__(self, env, agent=None, window_size: int = 1024):
        self._env          = env
        self._agent        = agent
        self._window_size  = window_size
        self._window_id    = 0
        self._obs          = None
        self._needs_reset  = True

    @property
    def is_live(self) -> bool:
        return False

    def set_agent(self, agent) -> None:
        self._agent = agent

    def setup(self) -> None:
        self._needs_reset = True

    def episodes(self):
        while True:
            if self._needs_reset:
                self._obs, _ = self._env.reset()
                if hasattr(self._agent, "reset_hidden"):
                    self._agent.reset_hidden()
                self._needs_reset = False

            observations, actions, rewards = [], [], []
            terminated_flags, truncated_flags, infos = [], [], []
            episode_starts = [0]

            for i in range(self._window_size):
                obs = self._obs
                observations.append(obs)
                action = self._agent.act(obs)
                actions.append(action)
                next_obs, reward, term, trunc, info = self._env.step(action)
                rewards.append(float(reward))
                terminated_flags.append(bool(term))
                truncated_flags.append(bool(trunc))
                infos.append(info)
                if term or trunc:
                    self._obs, _ = self._env.reset()
                    if hasattr(self._agent, "reset_hidden"):
                        self._agent.reset_hidden()
                    if i + 1 < self._window_size:
                        episode_starts.append(i + 1)
                else:
                    self._obs = next_obs

            last_done = terminated_flags[-1] or truncated_flags[-1]
            yield EpisodeData(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=rewards,
                terminated=terminated_flags,
                truncated=truncated_flags,
                infos=infos,
                episode_id=self._window_id,
                episode_starts=episode_starts,
                final_obs=None if last_done else self._obs,
            )
            self._window_id += 1

    def teardown(self) -> None:
        self._env.close()

    @property
    def window_size(self) -> int:
        return self._window_size


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
    """Greedy (argmax) evaluation of a DQN Q-network."""
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


def eval_rainbow(q_net: nn.Module, env_id: str, n_eps: int = 10, seed: int = 9000) -> float:
    """Greedy eval of Rainbow Q-network (eval mode disables noisy exploration)."""
    device = _net_device(q_net)
    q_net.eval()
    env = gym.make(env_id)
    totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                log_probs = q_net(obs_t)                     # (1, n_actions, n_atoms)
                q_vals    = log_probs.exp().sum(-1)           # (1, n_actions)
                action    = int(q_vals.argmax(dim=-1).item())
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    env.close()
    q_net.train()
    return float(np.mean(totals))


def eval_td3(actor: nn.Module, env_id: str, action_scale: float,
             n_eps: int = 10, seed: int = 9000) -> float:
    """Deterministic eval of TD3 actor (already tanh-bounded, scaled to env range)."""
    device = _net_device(actor)
    env = gym.make(env_id)
    totals = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                action = (actor(obs_t).cpu().numpy()[0] * action_scale)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def eval_recurrent_ppo(agent, env_factory, n_eps: int = 10, seed: int = 9000) -> float:
    """Eval TorchRecurrentPPOAgent; resets hidden state per episode."""
    totals = []
    for ep in range(n_eps):
        env = env_factory()
        obs, _ = env.reset(seed=seed + ep)
        agent.reset_hidden()
        total, done = 0.0, False
        while not done:
            action = agent.act(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        env.close()
        totals.append(total)
    # Clear caches filled by act() during eval
    agent._cache_obs.clear()
    agent._cache_lp.clear()
    agent._cache_values.clear()
    agent._cache_hidden.clear()
    agent.reset_hidden()
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


def _collect_episode_recurrent(env, agent, seed: int) -> EpisodeData:
    """Collect one full episode, resetting hidden state at the start."""
    obs, _ = env.reset(seed=seed)
    agent.reset_hidden()
    obs_list, act_list, rew_list, term_list, trunc_list = [], [], [], [], []
    done = False
    while not done:
        obs_list.append(obs.copy())
        action = agent.act(obs)
        act_list.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rew_list.append(float(reward))
        term_list.append(bool(terminated))
        trunc_list.append(bool(truncated))
        done = terminated or truncated
    return EpisodeData(
        observations=np.array(obs_list),
        actions=act_list,
        rewards=rew_list,
        terminated=term_list,
        truncated=trunc_list,
        infos=[{}] * len(rew_list),
        episode_id=0,
    )


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
        device="auto",
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
        device="auto",
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
    """Baseline DQN — same agent, arch, and hyperparams as tensor-optix, no loop features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
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


def train_vanilla_rainbow(cfg: dict, seed: int) -> dict:
    """Baseline Rainbow DQN — all 6 improvements, no tensor-optix loop features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q_net = RainbowQNetwork.build(obs_dim, n_actions)
    agent = TorchRainbowDQNAgent(
        q_network=q_net, n_actions=n_actions, obs_dim=obs_dim,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=cfg["lr"]),
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
        device="cpu",
    )

    carry_obs, _ = env.reset(seed=seed)
    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    window_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        ep_data, carry_obs = _collect_window(env, agent, cfg["window_size"], carry_obs, needs_reset=False)
        ep_data.episode_id = window_id
        agent.learn(ep_data)
        total_steps += cfg["window_size"]
        since_eval  += cfg["window_size"]
        window_id   += 1

        if since_eval >= cfg["eval_every"] or total_steps >= cfg["max_steps"]:
            since_eval = 0
            score = eval_rainbow(q_net, cfg["env_id"], seed=seed + 10_000)
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


def train_vanilla_td3(cfg: dict, seed: int) -> dict:
    """Baseline TD3 — fixed hyperparams, no tensor-optix loop features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_id"])
    obs_dim   = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    actor, c1, c2 = build_td3_nets(obs_dim, action_dim)
    agent = TorchTD3Agent(
        actor=actor, critic1=c1, critic2=c2, action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=cfg["lr"]),
        critic_optimizer=torch.optim.Adam(
            list(c1.parameters()) + list(c2.parameters()), lr=cfg["lr"]
        ),
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
        device="auto",
    )

    carry_obs, _ = env.reset(seed=seed)
    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    window_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        observations, actions, rewards, terminated, truncated = [], [], [], [], []
        obs = carry_obs
        for _ in range(cfg["window_size"]):
            observations.append(obs)
            raw_action = agent.act(obs)                     # in [-1, 1]
            action = raw_action * action_scale
            next_obs, reward, term, trunc, _ = env.step(action)
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            actions.append(raw_action)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        carry_obs = obs
        observations.append(carry_obs)

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
            score = eval_td3(actor, cfg["env_id"], action_scale, seed=seed + 10_000)
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


def train_vanilla_recurrent_ppo(cfg: dict, seed: int) -> dict:
    """Baseline Recurrent PPO on POMDP — fixed HP, episode-based collection."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_factory = cfg["env_factory"]
    env = env_factory()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    rnn, actor_head, critic_head = build_rppo_nets(obs_dim, n_actions)
    agent = TorchRecurrentPPOAgent(
        rnn=rnn, actor_head=actor_head, critic_head=critic_head,
        n_actions=n_actions,
        optimizer=torch.optim.Adam(
            list(rnn.parameters()) + list(actor_head.parameters()) + list(critic_head.parameters()),
            lr=cfg["lr"],
        ),
        hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
    )

    total_steps, since_eval = 0, 0
    steps_to_solve: Optional[int] = None
    history: list[tuple[int, float]] = []
    ep_id = 0
    t0 = time.perf_counter()

    while total_steps < cfg["max_steps"]:
        ep_data = _collect_episode_recurrent(env, agent, seed + ep_id)
        ep_data.episode_id = ep_id
        agent.learn(ep_data)
        n_steps = len(ep_data.rewards)
        total_steps += n_steps
        since_eval  += n_steps
        ep_id += 1

        if since_eval >= cfg["eval_every"] or total_steps >= cfg["max_steps"]:
            since_eval = 0
            score = eval_recurrent_ppo(agent, env_factory, seed=seed + 10_000)
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


def train_optix_dqn(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous DQN:
      - SPSA: online adaptation (learning_rate, gamma)
      - Exponential-backoff convergence detection
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_tmp = gym.make(cfg["env_id"])
    obs_dim, n_actions = env_tmp.observation_space.shape[0], env_tmp.action_space.n
    env_tmp.close()

    ckpt_dir = f"./benchmarks/.ckpts/dqn_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        q = build_dqn_net(obs_dim, n_actions)
        return TorchDQNAgent(
            q_network=q, n_actions=n_actions,
            optimizer=torch.optim.Adam(q.parameters(), lr=hp.get("learning_rate", cfg["lr"])),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
            device="auto",
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
        optimizer=AdaptiveOptimizer(param_bounds={
            "learning_rate": (1e-4, 1e-3),
            "gamma":         (0.95, 0.999),
        }, log_params=["learning_rate"], spsa_warmup_episodes=30),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=10,
        dormant_threshold=10,
        min_episodes_before_dormant=60,
        checkpoint_score_fn=lambda a: cfg["eval_fn"](pm_state.get("q_net", a._q), cfg["env_id"], seed=seed + 10_000),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
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
            device="auto",
        )

    def pipeline_factory():
        e = gym.make(cfg["env_id"])
        a, c = build_ppo_nets(obs_dim, n_actions)
        ag = TorchPPOAgent(
            actor=a, critic=c,
            optimizer=torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=cfg["lr"]),
            hyperparams=HyperparamSet(params=cfg["hp"].copy(), episode_id=0),
            device="auto",
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
        pm_cb.set_stop_fn(rl_opt.stop)
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
        }, log_params=["learning_rate"], warmup_episodes=40),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        dormant_threshold=6,
        max_interval_episodes=8,
        min_episodes_before_dormant=50,
        checkpoint_score_fn=lambda agent: cfg["eval_fn"](pm_state.get("actor", agent._actor), cfg["env_id"], seed=seed + 10_000),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
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
            device="auto",
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
        }, log_params=["learning_rate", "tau"], warmup_episodes=30),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8,
        dormant_threshold=8,  # recover faster from off-policy collapse
        checkpoint_score_fn=lambda a: eval_sac(pm_state.get("actor", a._actor), cfg["env_id"], action_scale, seed=seed + 10_000),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
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


def train_optix_rainbow(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous Rainbow DQN:
      - SPSA: online adaptation (learning_rate, gamma)
      - Exponential-backoff convergence detection
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_tmp = gym.make(cfg["env_id"])
    obs_dim, n_actions = env_tmp.observation_space.shape[0], env_tmp.action_space.n
    env_tmp.close()

    ckpt_dir = f"./benchmarks/.ckpts/rainbow_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        q = RainbowQNetwork.build(obs_dim, n_actions)
        return TorchRainbowDQNAgent(
            q_network=q, n_actions=n_actions, obs_dim=obs_dim,
            optimizer=torch.optim.Adam(q.parameters(), lr=hp.get("learning_rate", cfg["lr"])),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
            device="auto",
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
            "learning_rate": (1e-5, 1e-4),
            "gamma":         (0.97, 0.999),
        }, log_params=["learning_rate"], warmup_episodes=30),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=10,
        dormant_threshold=10,
        min_episodes_before_dormant=60,
        checkpoint_score_fn=lambda a: eval_rainbow(
            pm_state.get("q_net", a._q), cfg["env_id"], seed=seed + 10_000
        ),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
    rl_opt.add_callback(tracker)
    rl_opt.add_callback(pm_cb)
    rl_opt.run()

    tracker   = tracker_holder["tracker"]
    q_net     = pm_state["q_net"]
    final_score = eval_rainbow(q_net, cfg["env_id"], seed=seed + 10_000)
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


def train_optix_td3(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous TD3:
      - SPSA: online adaptation (learning_rate, gamma, tau)
      - Exponential-backoff convergence detection
      - PolicyManager: rollback + policy spawning (max_spawns=3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_tmp = gym.make(cfg["env_id"])
    obs_dim    = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    action_scale = float(env_tmp.action_space.high[0])
    env_tmp.close()

    ckpt_dir = f"./benchmarks/.ckpts/td3_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    cfg_for_tracker = dict(cfg)
    cfg_for_tracker["eval_fn"] = lambda a, env_id, seed=9000: eval_td3(a, env_id, action_scale, seed=seed)

    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        a, c1, c2 = build_td3_nets(obs_dim, action_dim)
        lr = hp.get("learning_rate", cfg["lr"])
        return TorchTD3Agent(
            actor=a, critic1=c1, critic2=c2, action_dim=action_dim,
            actor_optimizer=torch.optim.Adam(a.parameters(), lr=lr),
            critic_optimizer=torch.optim.Adam(
                list(c1.parameters()) + list(c2.parameters()), lr=lr
            ),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
            device="auto",
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
            "learning_rate": (1e-4, 1e-3),
            "gamma":         (0.97, 0.999),
            "tau":           (1e-3, 1e-1),
        }, log_params=["learning_rate", "tau"], warmup_episodes=30),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8,
        dormant_threshold=8,
        checkpoint_score_fn=lambda a: eval_td3(
            pm_state.get("actor", a._actor), cfg["env_id"], action_scale, seed=seed + 10_000
        ),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
    rl_opt.add_callback(tracker)
    rl_opt.add_callback(pm_cb)
    rl_opt.run()

    tracker = tracker_holder["tracker"]
    actor   = pm_state["actor"]
    final_score = eval_td3(actor, cfg["env_id"], action_scale, seed=seed + 10_000)
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


def train_optix_recurrent_ppo(cfg: dict, seed: int, verbose: bool = False, verbose_log_file: Optional[str] = None) -> dict:
    """
    tensor-optix autonomous Recurrent PPO (POMDP):
      - SPSA: online adaptation (learning_rate, clip_ratio, entropy_coef)
      - Exponential-backoff convergence detection
      - PolicyManager: rollback + policy spawning (max_spawns=3)
      - _RecurrentBatchPipeline: resets hidden state at episode boundaries
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_factory = cfg["env_factory"]
    env_tmp = env_factory()
    obs_dim   = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    ckpt_dir = f"./benchmarks/.ckpts/rppo_{cfg['key']}_{seed}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    cfg_for_tracker = dict(cfg)
    cfg_for_tracker["eval_fn"] = (
        lambda agent, env_id, seed=9000:
        eval_recurrent_ppo(agent, env_factory, seed=seed)
    )

    pm_state: dict = {}
    tracker_holder: dict = {}

    def make_agent(params=None):
        hp = {**cfg["hp"], **(params or {})}
        rnn, ah, ch = build_rppo_nets(obs_dim, n_actions)
        return TorchRecurrentPPOAgent(
            rnn=rnn, actor_head=ah, critic_head=ch, n_actions=n_actions,
            optimizer=torch.optim.Adam(
                list(rnn.parameters()) + list(ah.parameters()) + list(ch.parameters()),
                lr=hp.get("learning_rate", cfg["lr"]),
            ),
            hyperparams=HyperparamSet(params=hp, episode_id=0),
        )

    t0 = time.perf_counter()
    agent = make_agent()
    registry = CheckpointRegistry(ckpt_dir)
    pm = PolicyManager(registry, max_spawns=3)
    pm_cb = pm.as_callback(agent, agent_factory=make_agent)
    tracker = _Tracker(cfg=cfg_for_tracker, actor=agent, seed=seed)
    pm_state.update({"pm_cb": pm_cb, "agent": agent})
    tracker_holder["tracker"] = tracker

    env = env_factory()
    pipeline = _RecurrentBatchPipeline(env=env, agent=agent, window_size=cfg["window_size"])

    rl_opt = RLOptimizer(
        agent=agent,
        pipeline=pipeline,
        evaluator=TorchEvaluator(),
        optimizer=SPSAOptimizer(param_bounds={
            "learning_rate": (1e-4, 3e-3),
            "clip_ratio":    (0.1,  0.3),
            "entropy_coef":  (0.0,  0.05),
        }, log_params=["learning_rate"], warmup_episodes=30),
        checkpoint_dir=ckpt_dir,
        max_episodes=cfg["max_steps"] // cfg["window_size"],
        rollback_on_degradation=True,
        max_interval_episodes=8,
        dormant_threshold=8,
        min_episodes_before_dormant=30,
        checkpoint_score_fn=lambda a: eval_recurrent_ppo(
            pm_state.get("agent", a), env_factory, seed=seed + 10_000
        ),
        target_score=cfg["solve_threshold"],
        convergence_patience=5,
        verbose=verbose,
        verbose_log_file=verbose_log_file,
    )
    pm_cb.set_stop_fn(rl_opt.stop)
    rl_opt.add_callback(tracker)
    rl_opt.add_callback(pm_cb)
    rl_opt.run()

    tracker = tracker_holder["tracker"]
    ag      = pm_state["agent"]
    final_score = eval_recurrent_ppo(ag, env_factory, seed=seed + 10_000)
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
    "learning_rate":      1e-3,
    "gamma":              0.99,
    "epsilon":            1.0,
    "epsilon_min":        0.05,
    "epsilon_decay":      0.995,
    "batch_size":         64,
    "target_update_freq": 10,
    "replay_capacity":    10_000,
}
RAINBOW_HP = {
    "learning_rate":      6.25e-5,
    "gamma":              0.99,
    "batch_size":         32,
    "target_update_freq": 200,
    "replay_capacity":    100_000,
    "per_alpha":          0.5,
    "per_beta":           0.4,
    "n_step":             3,
    "v_min":              0.0,
    "v_max":              500.0,
    "n_atoms":            51,
}
TD3_HP = {
    "learning_rate":      3e-4,
    "gamma":              0.99,
    "tau":                0.005,
    "batch_size":         256,
    "updates_per_step":   1,
    "replay_capacity":    200_000,
    "policy_delay":       2,
    "target_noise":       0.2,
    "target_noise_clip":  0.5,
    "per_alpha":          0.0,
    "per_beta":           0.4,
}
RPPO_HP = {
    "learning_rate":  3e-4,
    "clip_ratio":     0.2,
    "entropy_coef":   0.01,
    "vf_coef":        0.5,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "n_epochs":       4,
    "bptt_len":       16,
    "max_grad_norm":  0.5,
}

ENV_CONFIGS = {
    "cartpole": {
        # CartPole-v1 with DQN — classic discrete control benchmark.
        # The agent must balance a pole on a cart. Solve threshold: 475.
        # DQN with replay buffer handles the short episode structure well.
        "key":             "cartpole",
        "label":           "CartPole-v1",
        "env_id":          "CartPole-v1",
        "algo":            "DQN (discrete)",
        "solve_threshold":  475.0,
        "max_steps":        200_000,
        "window_size":      512,
        "eval_every":       5_000,
        "lr":               1e-3,
        "hp":               DQN_HP,
        "eval_fn":          lambda net, env_id, seed=9000: eval_dqn(net, env_id, seed=seed),
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
    "cartpole_rainbow": {
        # CartPole-v1 with Rainbow DQN — all 6 improvements to DQN:
        # double-Q, PER, n-step returns, dueling networks, noisy nets, C51.
        # Demonstrates that the full Rainbow stack converges faster and more
        # reliably than vanilla DQN on the same environment.
        "key":             "cartpole_rainbow",
        "label":           "CartPole Rainbow",
        "env_id":          "CartPole-v1",
        "algo":            "Rainbow DQN (all 6 improvements)",
        "solve_threshold":  475.0,
        "max_steps":        200_000,
        "window_size":      512,
        "eval_every":       5_000,
        "lr":               6.25e-5,
        "hp":               RAINBOW_HP,
        "eval_fn":          lambda net, env_id, seed=9000: eval_rainbow(net, env_id, seed=seed),
        "train_vanilla":    train_vanilla_rainbow,
        "train_optix":      train_optix_rainbow,
    },
    "pendulum_td3": {
        # Pendulum-v1 with TD3 — the canonical continuous control benchmark.
        # TD3's three fixes (twin critics, delayed updates, target smoothing)
        # address overestimation bias that causes DDPG to diverge here.
        # Solve threshold: −150 (from Gymnasium leaderboard).
        "key":             "pendulum_td3",
        "label":           "Pendulum TD3",
        "env_id":          "Pendulum-v1",
        "algo":            "TD3 (continuous, twin-delayed)",
        "solve_threshold":  -150.0,
        "max_steps":        300_000,
        "window_size":      1_000,
        "eval_every":       10_000,
        "lr":               3e-4,
        "hp":               TD3_HP,
        "eval_fn":          lambda actor, env_id, seed=9000: eval_td3(actor, env_id, 2.0, seed=seed),
        "train_vanilla":    train_vanilla_td3,
        "train_optix":      train_optix_td3,
    },
    "pomdp_cartpole": {
        # CartPole-v1 with velocity observations masked (indices 1, 3 removed).
        # The feedforward PPO cannot infer velocity from position alone;
        # the LSTM in RecurrentPPO integrates past observations into a belief
        # state, giving it a decisive advantage in this POMDP setting.
        "key":             "pomdp_cartpole",
        "label":           "POMDP CartPole",
        "env_id":          "CartPole-v1",
        "env_factory":     lambda: _POMDPCartPoleWrapper(gym.make("CartPole-v1")),
        "algo":            "Recurrent PPO (LSTM, partial observability)",
        "solve_threshold":  350.0,
        "max_steps":        300_000,
        "window_size":      1_024,
        "eval_every":       10_000,
        "lr":               3e-4,
        "hp":               RPPO_HP,
        "eval_fn": (
            lambda agent, env_id, seed=9000:
            eval_recurrent_ppo(
                agent,
                lambda: _POMDPCartPoleWrapper(gym.make("CartPole-v1")),
                seed=seed,
            )
        ),
        "train_vanilla":    train_vanilla_recurrent_ppo,
        "train_optix":      train_optix_recurrent_ppo,
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
    mean_a = np.mean(a)
    if abs(mean_a) < 1e-9:
        return "  -  "
    pct = (np.mean(b) - mean_a) / abs(mean_a) * 100
    return f"{pct:>+.0f}%"

def print_table(cfg: dict, vanilla: list[dict], optix: list[dict]) -> None:
    W = 70
    sep  = "=" * W
    line = "-" * W
    n    = len(vanilla)

    # Unsolved seeds count as full budget (DNF penalty) so the mean is honest
    # when solve rates differ between methods.
    v_solve = [r["steps_to_solve"] if r["steps_to_solve"] is not None else r["total_steps"] for r in vanilla]
    o_solve = [r["steps_to_solve"] if r["steps_to_solve"] is not None else r["total_steps"] for r in optix]
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature showcase  (--demo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _demo_jax(seed: int = 0) -> None:
    """Demo: FlaxPPOAgent vs TorchPPOAgent on CartPole-v1 (200 episodes each)."""
    try:
        import jax  # noqa: F401
        from flax import nnx  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        print("  [demo-jax] skipped — install tensor-optix[jax] to enable")
        return

    from tensor_optix.algorithms.flax_ppo import FlaxPPOAgent
    from tensor_optix.core.types import EpisodeData, HyperparamSet

    hp = HyperparamSet(params=dict(
        learning_rate=3e-4, clip_ratio=0.2, entropy_coef=0.01,
        vf_coef=0.5, gamma=0.99, gae_lambda=0.95,
        n_epochs=4, minibatch_size=32,
    ), episode_id=0)

    W = 64

    print("\n" + "=" * W)
    print("  JAX/Flax Adapter  |  FlaxPPOAgent vs TorchPPOAgent")
    print("  Environment: CartPole-v1  |  200 training episodes  |  5 eval episodes")
    print("=" * W)

    # ── Flax ──
    import gymnasium as gym
    agent_flax = FlaxPPOAgent(obs_dim=4, n_actions=2, hyperparams=hp, seed=seed)
    env = gym.make("CartPole-v1")
    t0 = time.perf_counter()
    for ep_i in range(200):
        obs_t, _ = env.reset(seed=seed)
        obs_list, act_list, rew_list = [], [], []
        done = False
        while not done:
            action = agent_flax.act(obs_t)
            obs_list.append(obs_t.copy())
            act_list.append(action)
            obs_t, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            done = terminated or truncated
        ep = EpisodeData(
            observations=np.array(obs_list, dtype=np.float32),
            actions=act_list, rewards=rew_list,
            terminated=[False] * (len(rew_list) - 1) + [bool(terminated)],
            truncated=[False] * (len(rew_list) - 1) + [bool(truncated)],
            infos=[{}] * len(rew_list), episode_id=ep_i,
        )
        agent_flax.learn(ep)
    env.close()
    flax_train_t = time.perf_counter() - t0

    eval_env = gym.make("CartPole-v1")
    flax_rewards = []
    for _ in range(5):
        obs_e, _ = eval_env.reset(seed=seed + 1000)
        total, done = 0.0, False
        while not done:
            action = agent_flax.act(obs_e)
            obs_e, r, term, trunc, _ = eval_env.step(action)
            total += r; done = term or trunc
        agent_flax.reset_cache()
        flax_rewards.append(total)
    eval_env.close()
    flax_score = float(np.mean(flax_rewards))

    # ── Torch ──
    torch.manual_seed(seed)
    actor  = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
    from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
    agent_torch = TorchPPOAgent(
        actor=actor, critic=critic,
        optimizer=torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=3e-4
        ),
        hyperparams=hp, device="auto",
    )
    env = gym.make("CartPole-v1")
    t0 = time.perf_counter()
    for ep_i in range(200):
        obs_t, _ = env.reset(seed=seed)
        obs_list, act_list, rew_list = [], [], []
        done = False
        while not done:
            action = agent_torch.act(obs_t)
            obs_list.append(obs_t.copy())
            act_list.append(action)
            obs_t, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            done = terminated or truncated
        ep = EpisodeData(
            observations=np.array(obs_list, dtype=np.float32),
            actions=act_list, rewards=rew_list,
            terminated=[False] * (len(rew_list) - 1) + [bool(terminated)],
            truncated=[False] * (len(rew_list) - 1) + [bool(truncated)],
            infos=[{}] * len(rew_list), episode_id=ep_i,
        )
        agent_torch.learn(ep)
    env.close()
    torch_train_t = time.perf_counter() - t0

    eval_env = gym.make("CartPole-v1")
    torch_rewards = []
    for _ in range(5):
        obs_e, _ = eval_env.reset(seed=seed + 1000)
        total, done = 0.0, False
        while not done:
            action = agent_torch.act(obs_e)
            obs_e, r, term, trunc, _ = eval_env.step(action)
            total += r; done = term or trunc
        agent_torch.reset_cache()
        torch_rewards.append(total)
    eval_env.close()
    torch_score = float(np.mean(torch_rewards))

    gap = abs(flax_score - torch_score) / max(torch_score, 1.0) * 100
    print(f"  {'':30s} {'FlaxPPO':>12}  {'TorchPPO':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Mean eval reward (5 ep)':30s} {flax_score:>12.1f}  {torch_score:>12.1f}")
    print(f"  {'Training time (200 ep)':30s} {flax_train_t:>11.1f}s  {torch_train_t:>11.1f}s")
    print(f"  {'Relative score gap':30s} {gap:>11.1f}%")
    if gap < 10:
        print(f"\n  -> Parity confirmed: gap {gap:.1f}% < 10% threshold.")
    else:
        print(f"\n  -> Gap {gap:.1f}% — may need more episodes for full convergence.")
    print()


def _demo_async(seed: int = 0) -> None:
    """Demo: AsyncActorLearner throughput on CartPole-v1 (4 actors vs 1)."""
    import gymnasium as gym
    from tensor_optix.distributed import AsyncActorLearner

    W = 64
    print("\n" + "=" * W)
    print("  Distributed Async Actor-Learner  |  IMPALA + V-trace")
    print("  Environment: CartPole-v1  |  30k steps  |  1 actor vs 4 actors")
    print("=" * W)

    def make_nets():
        actor  = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
        critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
        opt = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=3e-4
        )
        return actor, critic, opt

    results = {}
    for n_actors in (1, 4):
        actor, critic, opt = make_nets()
        learner = AsyncActorLearner(
            actor=actor, critic=critic, optimizer=opt,
            env_factory=lambda: gym.make("CartPole-v1"),
            n_actors=n_actors, trajectory_len=64,
            max_queue_size=500, seed=seed,
        )
        stats = learner.run(max_steps=30_000)
        results[n_actors] = stats
        print(
            f"  n_actors={n_actors}  steps={stats['total_steps']:>7,d}"
            f"  updates={stats['total_updates']:>5,d}"
            f"  {stats['steps_per_second']:>7.0f} steps/s"
            f"  ({stats['elapsed']:.1f}s)"
        )

    ratio = results[4]["steps_per_second"] / max(results[1]["steps_per_second"], 1.0)
    print(f"\n  -> 4-actor throughput: {ratio:.2f}× single-actor")
    if ratio >= 2.0:
        print(f"  -> ≥ 2.0× threshold achieved.")
    print()


def run_demo(seed: int = 0) -> None:
    """Run a fast multi-feature showcase (~5 minutes on CPU)."""
    print("\n" + "━" * 64)
    print("  tensor-optix  Feature Showcase  (--demo)")
    print("  Demonstrates: JAX/Flax adapter + Async actor-learner")
    print("━" * 64)
    _demo_jax(seed=seed)
    _demo_async(seed=seed)
    print("━" * 64)
    print("  Demo complete.")
    print("  Full benchmark: uv run python benchmarks/benchmark.py --envs cartpole lunarlander --seeds 0")
    print("━" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tensor-optix vs Baseline training loop benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a fast feature showcase (~5 min): JAX/Flax parity + async throughput",
    )
    parser.add_argument(
        "--demo-seed", type=int, default=0,
        help="Random seed for --demo (default: 0)",
    )
    parser.add_argument(
        "--envs", nargs="+",
        choices=list(ENV_CONFIGS.keys()),
        default=["cartpole", "lunarlander", "acrobot", "lunarlander_continuous", "bipedalwalker"],
        metavar="ENV",
        help=(
            "Environments to benchmark (default: original 5). "
            "New: cartpole_rainbow, pendulum_td3, pomdp_cartpole"
        ),
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

    if args.demo:
        run_demo(seed=args.demo_seed)
        return

    run(envs=args.envs, seeds=args.seeds, plot=not args.no_plot, optix_only=args.optix_only, verbose=args.verbose, verbose_log_file=args.verbose_log_file)


if __name__ == "__main__":
    main()

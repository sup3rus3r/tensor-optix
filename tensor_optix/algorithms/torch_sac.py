import os
import copy
import random
from collections import deque

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self._buf.append((
            np.array(obs,    dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.float32),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


class TorchSACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) agent for PyTorch continuous action spaces.

    Entropy-regularized actor-critic (Haarnoja et al. 2018) with:
    - Squashed Gaussian actor (tanh), reparameterization trick
    - Twin Q-critics (clipped double-Q)
    - Soft Polyak target updates
    - Automatic entropy temperature (learnable log_alpha)

    Architecture:
        actor:   nn.Module, obs → [mean || log_std], shape [batch, 2*action_dim]
        critic1: nn.Module, [obs || action] → Q-value, shape [batch, 1]
        critic2: nn.Module, same as critic1

    Usage:
        import torch
        import torch.nn as nn
        from tensor_optix.algorithms.torch_sac import TorchSACAgent

        actor   = build_actor(obs_dim, action_dim)
        critic1 = build_critic(obs_dim + action_dim, 1)
        critic2 = build_critic(obs_dim + action_dim, 1)
        agent = TorchSACAgent(
            actor=actor, critic1=critic1, critic2=critic2,
            action_dim=action_dim,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
            critic_optimizer=torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
            ),
            alpha_optimizer=torch.optim.Adam([log_alpha], lr=3e-4),
            hyperparams=HyperparamSet(params={
                "learning_rate":   3e-4,
                "gamma":           0.99,
                "tau":             0.005,
                "batch_size":      256,
                "updates_per_step": 1,
                "replay_capacity": 1_000_000,
            }, episode_id=0),
        )

    Actions are in (-1, 1) via tanh. Wrap the env or scale actions in a subclass.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        actor,
        critic1,
        critic2,
        action_dim: int,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
        hyperparams: HyperparamSet,
    ):
        import torch
        self._torch    = torch
        self._actor    = actor
        self._c1       = critic1
        self._c2       = critic2
        self._c1_tgt   = copy.deepcopy(critic1)
        self._c2_tgt   = copy.deepcopy(critic2)
        self._action_dim    = action_dim
        self._actor_opt     = actor_optimizer
        self._critic_opt    = critic_optimizer
        self._alpha_opt     = alpha_optimizer
        self._hyperparams   = hyperparams.copy()

        log_alpha_init = float(hyperparams.params.get("log_alpha_init", 0.0))
        self._log_alpha = torch.tensor(log_alpha_init, dtype=torch.float32,
                                       requires_grad=True)
        # Replace alpha_optimizer param group with our log_alpha tensor
        self._alpha_opt = type(alpha_optimizer)(
            [self._log_alpha], **{k: v for k, v in alpha_optimizer.defaults.items()}
        )
        self._target_entropy = -float(action_dim)

        capacity = int(hyperparams.params.get("replay_capacity", 1_000_000))
        self._buffer = _ReplayBuffer(capacity)

    def act(self, observation) -> np.ndarray:
        import torch
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32)
        with torch.no_grad():
            action, _ = self._sample_action(obs)
        return action.numpy()[0]

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        gamma            = float(hp.get("gamma",            0.99))
        tau              = float(hp.get("tau",              0.005))
        batch_size       = int(hp.get("batch_size",         256))
        updates_per_step = int(hp.get("updates_per_step",  1))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = np.array(episode_data.actions,      dtype=np.float32)
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        for t in range(T - 1):
            self._buffer.push(obs_arr[t], act_arr[t], float(rew_arr[t]),
                               obs_arr[t + 1], float(done_arr[t]))

        if len(self._buffer) < batch_size:
            return {
                "actor_loss": 0.0, "critic_loss": 0.0,
                "alpha": float(self._log_alpha.exp().item()),
                "buffer_size": len(self._buffer),
            }

        n_updates = T * updates_per_step
        total_al = 0.0
        total_cl = 0.0
        for _ in range(n_updates):
            al, cl = self._update_step(batch_size, gamma, tau)
            total_al += al
            total_cl += cl

        return {
            "actor_loss":  total_al / n_updates,
            "critic_loss": total_cl / n_updates,
            "alpha":       float(self._log_alpha.exp().item()),
            "buffer_size": len(self._buffer),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._actor_opt.param_groups[0]["lr"]
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            for opt in (self._actor_opt, self._critic_opt, self._alpha_opt):
                for pg in opt.param_groups:
                    pg["lr"] = lr

    def save_weights(self, path: str) -> None:
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self._actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self._c1.state_dict(),    os.path.join(path, "critic1.pt"))
        torch.save(self._c2.state_dict(),    os.path.join(path, "critic2.pt"))
        np.save(os.path.join(path, "log_alpha.npy"), self._log_alpha.item())

    def load_weights(self, path: str) -> None:
        import torch
        self._actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"),   map_location="cpu"))
        self._c1.load_state_dict(   torch.load(os.path.join(path, "critic1.pt"), map_location="cpu"))
        self._c2.load_state_dict(   torch.load(os.path.join(path, "critic2.pt"), map_location="cpu"))
        la_path = os.path.join(path, "log_alpha.npy")
        if os.path.exists(la_path):
            with torch.no_grad():
                self._log_alpha.fill_(float(np.load(la_path)))
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_action(self, obs):
        import torch
        out     = self._actor(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        eps     = torch.randn_like(mean)
        u       = mean + std * eps
        action  = torch.tanh(u)

        gaussian_lp = -0.5 * (((u - mean) / (std + 1e-8)) ** 2
                               + 2.0 * log_std
                               + np.log(2.0 * np.pi))
        log_prob = gaussian_lp.sum(dim=-1)
        log_prob -= torch.log(1.0 - action ** 2 + 1e-6).sum(dim=-1)
        return action, log_prob

    def _update_step(self, batch_size: int, gamma: float, tau: float):
        import torch
        import torch.nn.functional as F

        obs_b, act_b, rew_b, next_b, done_b = self._buffer.sample(batch_size)
        obs_b  = torch.as_tensor(obs_b,  dtype=torch.float32)
        act_b  = torch.as_tensor(act_b,  dtype=torch.float32)
        rew_b  = torch.as_tensor(rew_b,  dtype=torch.float32)
        next_b = torch.as_tensor(next_b, dtype=torch.float32)
        done_b = torch.as_tensor(done_b, dtype=torch.float32)
        alpha  = self._log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            na, nlp = self._sample_action(next_b)
            ci_next = torch.cat([next_b, na], dim=-1)
            q1t = self._c1_tgt(ci_next).squeeze(-1)
            q2t = self._c2_tgt(ci_next).squeeze(-1)
            q_t = rew_b + gamma * (1 - done_b) * (torch.minimum(q1t, q2t) - alpha * nlp)

        ci = torch.cat([obs_b, act_b], dim=-1)
        q1 = self._c1(ci).squeeze(-1)
        q2 = self._c2(ci).squeeze(-1)
        c_loss = F.mse_loss(q1, q_t) + F.mse_loss(q2, q_t)
        self._critic_opt.zero_grad()
        c_loss.backward()
        self._critic_opt.step()

        # Actor update
        new_a, new_lp = self._sample_action(obs_b)
        ci_new = torch.cat([obs_b, new_a], dim=-1)
        q1n = self._c1(ci_new).squeeze(-1)
        q2n = self._c2(ci_new).squeeze(-1)
        a_loss = (alpha * new_lp - torch.minimum(q1n, q2n)).mean()
        self._actor_opt.zero_grad()
        a_loss.backward()
        self._actor_opt.step()

        # Alpha update
        alpha_loss = -(self._log_alpha * (new_lp.detach() + self._target_entropy)).mean()
        self._alpha_opt.zero_grad()
        alpha_loss.backward()
        self._alpha_opt.step()

        # Soft target update
        for sv, tv in zip(self._c1.parameters(), self._c1_tgt.parameters()):
            tv.data.copy_(tau * sv.data + (1 - tau) * tv.data)
        for sv, tv in zip(self._c2.parameters(), self._c2_tgt.parameters()):
            tv.data.copy_(tau * sv.data + (1 - tau) * tv.data)

        return float(a_loss.item()), float(c_loss.item())

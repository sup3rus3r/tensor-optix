from __future__ import annotations

"""
GraphAgent — a BaseAgent backed by a free-form NeuronGraph.

Weight learning is PPO-style: the graph acts as both actor and critic.
The actor head reads output neurons [0:-1], the critic head reads the last
output neuron. For discrete actions the actor outputs logits; for continuous
actions it outputs means (std is a learned parameter).

The agent is intentionally minimal — it wires the NeuronGraph into the
tensor-optix contract. Topology mutations are performed externally by
TopologyController, which holds a reference to the graph.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae

from ..graph.neuron_graph import NeuronGraph
from ..graph.topology_ops import add_input_neuron


class GraphAgent(BaseAgent):
    """
    RL agent whose policy network is a mutable NeuronGraph.

    Parameters
    ----------
    graph:
        A NeuronGraph that has already been configured with input/output neurons.
        The last output neuron is the value head; all others are action logits
        (discrete) or action means (continuous).
    obs_dim:
        Expected observation dimensionality. If a new obs arrives with more
        dimensions, the agent automatically grows new input neurons.
    n_actions:
        Number of discrete actions, or dimension of continuous action space.
    continuous:
        If True, actions are sampled from a Gaussian. If False, from Categorical.
    hyperparams:
        Initial HyperparamSet. Keys used: learning_rate, clip_ratio,
        entropy_coef, vf_coef, gamma, gae_lambda, n_epochs, minibatch_size,
        max_grad_norm.
    """

    is_on_policy = True

    default_hyperparams = {
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       4,
        "minibatch_size": 64,
        "max_grad_norm":  0.5,
    }

    def __init__(
        self,
        graph: NeuronGraph,
        obs_dim: int,
        n_actions: int,
        continuous: bool = False,
        hyperparams: Optional[HyperparamSet] = None,
        device: str = "cpu",
    ) -> None:
        self.graph = graph
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.continuous = continuous
        self.device = torch.device(device)

        self.graph.to(self.device)

        if continuous:
            # Learnable log-std, one per action dimension
            self.log_std = nn.Parameter(torch.zeros(n_actions, device=self.device))

        params = list(self.graph.parameters())
        if continuous:
            params.append(self.log_std)

        _hp = dict(self.default_hyperparams)
        if hyperparams is not None:
            _hp.update(hyperparams.params)
        self._hyperparams = HyperparamSet(params=_hp, episode_id=0)

        self.optimizer = torch.optim.Adam(params, lr=_hp["learning_rate"])
        self._episode_count = 0

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    def act(self, observation) -> any:
        """
        Given a numpy observation, return an action (and store log_prob).
        Dynamically grows input neurons if obs_dim has expanded.
        """
        obs = self._to_tensor(observation)
        self._maybe_grow_inputs(obs)

        self.graph.reset_state()
        with torch.no_grad():
            out = self.graph(obs)

        action, log_prob = self._sample_action(out)
        self._last_log_prob = log_prob
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        obs_t = torch.tensor(
            np.array(episode_data.observations), dtype=torch.float32, device=self.device
        )
        act_t = torch.tensor(
            np.array(episode_data.actions), dtype=torch.long if not self.continuous else torch.float32,
            device=self.device,
        )
        rew_t = torch.tensor(episode_data.rewards, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(
            episode_data.log_probs if episode_data.log_probs else [0.0] * len(episode_data.rewards),
            dtype=torch.float32, device=self.device,
        )

        # Compute values and advantages
        with torch.no_grad():
            values = self._batch_values(obs_t)
        advantages, returns = compute_gae(
            rewards=rew_t.cpu().numpy().tolist(),
            values=values.cpu().numpy().tolist(),
            dones=episode_data.dones,
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
        )
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = obs_t.shape[0]
        total_loss = total_pg = total_vf = total_ent = 0.0
        n_updates = 0

        for _ in range(hp["n_epochs"]):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, hp["minibatch_size"]):
                idx = perm[start : start + hp["minibatch_size"]]
                if idx.numel() == 0:
                    continue
                ob = obs_t[idx]
                ac = act_t[idx]
                old_lp_b = old_lp[idx]
                adv_b = adv_t[idx]
                ret_b = ret_t[idx]

                out_b = self._batch_forward(ob)
                new_lp, entropy, values_b = self._evaluate_actions(out_b, ac)

                ratio = torch.exp(new_lp - old_lp_b)
                pg1 = ratio * adv_b
                pg2 = torch.clamp(ratio, 1 - hp["clip_ratio"], 1 + hp["clip_ratio"]) * adv_b
                pg_loss = -torch.min(pg1, pg2).mean()

                vf_loss = F.mse_loss(values_b, ret_b)
                ent_loss = -entropy.mean()

                loss = pg_loss + hp["vf_coef"] * vf_loss + hp["entropy_coef"] * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.graph.parameters(), hp["max_grad_norm"])
                self.optimizer.step()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                n_updates += 1

        self._episode_count += 1
        denom = max(n_updates, 1)
        return {
            "loss": total_loss / denom,
            "pg_loss": total_pg / denom,
            "vf_loss": total_vf / denom,
            "entropy": -total_ent / denom,
            "n_neurons": self.graph.n_neurons(),
            "n_edges": self.graph.n_edges(),
        }

    def get_hyperparams(self) -> HyperparamSet:
        return self._hyperparams

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams
        for pg in self.optimizer.param_groups:
            pg["lr"] = hyperparams.params.get("learning_rate", pg["lr"])

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {"graph": self.graph.state_dict(), "episode": self._episode_count}
        if self.continuous:
            state["log_std"] = self.log_std.data
        torch.save(state, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.graph.load_state_dict(state["graph"])
        self._episode_count = state.get("episode", 0)
        if self.continuous and "log_std" in state:
            self.log_std.data.copy_(state["log_std"])

    def perturb_weights(self, noise_scale: float) -> None:
        with torch.no_grad():
            for p in self.graph.parameters():
                p.mul_(1 + noise_scale * torch.randn_like(p))

    def average_weights(self, paths: list) -> None:
        if not paths:
            return
        states = [torch.load(p, map_location=self.device)["graph"] for p in paths]
        avg = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
        self.graph.load_state_dict(avg)

    def teardown(self) -> None:
        self.graph.cpu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.float().to(self.device)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).flatten()

    def _maybe_grow_inputs(self, obs: torch.Tensor) -> None:
        current_inputs = len(self.graph.input_ids)
        needed = obs.shape[0]
        if needed > current_inputs:
            for _ in range(needed - current_inputs):
                add_input_neuron(self.graph, activation="linear")
            self.obs_dim = needed
            # Rebuild optimizer to include new parameters
            params = list(self.graph.parameters())
            if self.continuous:
                params.append(self.log_std)
            lr = self._hyperparams.params.get("learning_rate", 3e-4)
            self.optimizer = torch.optim.Adam(params, lr=lr)

    def _batch_forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Run graph forward for each obs in batch, return stacked outputs."""
        outs = []
        for i in range(obs_batch.shape[0]):
            self.graph.reset_state()
            out = self.graph(obs_batch[i])
            outs.append(out)
        return torch.stack(outs)

    def _batch_values(self, obs_batch: torch.Tensor) -> torch.Tensor:
        out = self._batch_forward(obs_batch)
        return out[:, -1]  # last output neuron = value

    def _sample_action(self, out: torch.Tensor):
        logits_or_means = out[:-1]  # all but last = actor
        if self.continuous:
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(logits_or_means, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            return action.cpu().numpy(), log_prob
        else:
            dist = Categorical(logits=logits_or_means)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    def _evaluate_actions(self, out_batch: torch.Tensor, actions: torch.Tensor):
        logits_or_means = out_batch[:, :-1]
        values = out_batch[:, -1]
        if self.continuous:
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(logits_or_means, std.unsqueeze(0).expand_as(logits_or_means))
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=logits_or_means)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        return log_prob, entropy, values

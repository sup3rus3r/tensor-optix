import os
import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches


class TorchPPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization) agent for PyTorch discrete action spaces.

    Implements the clipped surrogate objective (Schulman et al. 2017) with:
    - GAE-λ advantage estimation
    - Entropy bonus for exploration regularization
    - Value function loss (MSE)
    - Multiple epochs of minibatch gradient descent per rollout
    - Global gradient norm clipping

    Architecture: separate actor and critic networks.
        actor:  nn.Module, obs → logits [batch, n_actions]
        critic: nn.Module, obs → value  [batch, 1] or [batch]

    Usage:
        import torch
        import torch.nn as nn
        from tensor_optix.algorithms.torch_ppo import TorchPPOAgent

        actor  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, n_actions))
        critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        agent  = TorchPPOAgent(
            actor=actor, critic=critic,
            optimizer=torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()), lr=3e-4
            ),
            hyperparams=HyperparamSet(params={
                "learning_rate":  3e-4,
                "clip_ratio":     0.2,
                "entropy_coef":   0.01,
                "vf_coef":        0.5,
                "gamma":          0.99,
                "gae_lambda":     0.95,
                "n_epochs":       10,
                "minibatch_size": 64,
                "max_grad_norm":  0.5,
            }, episode_id=0),
        )
    """

    def __init__(self, actor, critic, optimizer, hyperparams: HyperparamSet):
        import torch
        self._torch  = torch
        self._actor  = actor
        self._critic = critic
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

        self._cache_obs: list      = []
        self._cache_log_probs: list = []
        self._cache_values: list   = []

    def act(self, observation) -> int:
        """Sample action; cache log_prob and V(obs) for learn()."""
        import torch
        import torch.nn.functional as F
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32)
        with torch.no_grad():
            logits = self._actor(obs)                       # [1, n_actions]
            lp_all = F.log_softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            lp     = lp_all[0, action.item()]
            value  = self._critic(obs).squeeze()

        self._cache_obs.append(np.squeeze(observation))
        self._cache_log_probs.append(float(lp.item()))
        self._cache_values.append(float(value.item()))
        return int(action.item())

    def learn(self, episode_data: EpisodeData) -> dict:
        import torch
        import torch.nn.functional as F

        hp = self._hyperparams.params
        clip_ratio    = float(hp.get("clip_ratio",    0.2))
        entropy_coef  = float(hp.get("entropy_coef",  0.01))
        vf_coef       = float(hp.get("vf_coef",       0.5))
        gamma         = float(hp.get("gamma",          0.99))
        gae_lambda    = float(hp.get("gae_lambda",     0.95))
        n_epochs      = int(hp.get("n_epochs",         10))
        mb_size       = int(hp.get("minibatch_size",   64))
        max_grad_norm = float(hp.get("max_grad_norm",  0.5))

        T = len(episode_data.rewards)
        obs_arr    = np.array(self._cache_obs[:T],       dtype=np.float32)
        old_lp_arr = np.array(self._cache_log_probs[:T], dtype=np.float32)
        val_arr    = np.array(self._cache_values[:T],    dtype=np.float32)
        rewards    = episode_data.rewards
        dones      = episode_data.dones

        advantages, returns = compute_gae(rewards, val_arr, dones, gamma, gae_lambda)
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout = {
            "obs":        obs_arr,
            "actions":    np.array(episode_data.actions, dtype=np.int64),
            "old_lp":     old_lp_arr,
            "advantages": advantages,
            "returns":    returns,
        }

        all_params = list(self._actor.parameters()) + list(self._critic.parameters())
        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        total_kl       = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for batch in make_minibatches(rollout, mb_size):
                obs_b   = torch.as_tensor(batch["obs"],        dtype=torch.float32)
                act_b   = torch.as_tensor(batch["actions"],    dtype=torch.long)
                old_b   = torch.as_tensor(batch["old_lp"],     dtype=torch.float32)
                adv_b   = torch.as_tensor(batch["advantages"], dtype=torch.float32)
                ret_b   = torch.as_tensor(batch["returns"],    dtype=torch.float32)

                logits   = self._actor(obs_b)
                lp_all   = F.log_softmax(logits, dim=-1)
                new_lp   = lp_all[range(len(act_b)), act_b]

                ratio    = torch.exp(new_lp - old_b)
                s1       = ratio * adv_b
                s2       = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_b
                pol_loss = -torch.minimum(s1, s2).mean()

                new_val  = self._critic(obs_b).squeeze(-1)
                val_loss = F.mse_loss(new_val, ret_b)

                probs   = torch.exp(lp_all)
                entropy = -(probs * lp_all).sum(dim=-1).mean()

                loss = pol_loss + vf_coef * val_loss - entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self._optimizer.step()

                total_pol_loss += float(pol_loss.item())
                total_val_loss += float(val_loss.item())
                total_entropy  += float(entropy.item())
                total_kl       += float((old_b - new_lp).mean().item())
                n_updates += 1

        self._cache_obs.clear()
        self._cache_log_probs.clear()
        self._cache_values.clear()

        n = max(n_updates, 1)
        ev = self._explained_variance(val_arr, returns)
        return {
            "policy_loss":   total_pol_loss / n,
            "value_loss":    total_val_loss  / n,
            "entropy":       total_entropy   / n,
            "approx_kl":     total_kl        / n,
            "explained_var": ev,
            "n_updates":     n_updates,
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._optimizer.param_groups[0]["lr"]
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr

    def save_weights(self, path: str) -> None:
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self._actor.state_dict(),  os.path.join(path, "actor.pt"))
        torch.save(self._critic.state_dict(), os.path.join(path, "critic.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        self._actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"),  map_location="cpu")
        )
        self._critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pt"), map_location="cpu")
        )

    @staticmethod
    def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return float("nan")
        return float(1.0 - np.var(returns - values) / var_returns)

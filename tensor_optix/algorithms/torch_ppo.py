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

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
        # entropy_coef lo=0.001: prevents SPSA from zeroing entropy and collapsing the policy.
        # gamma included: PPO advantage estimation is sensitive to discount horizon;
        # SPSA can adapt it per-environment consistently with DQN/SAC.
    }
    default_log_params = ["learning_rate"]

    def __init__(self, actor, critic, optimizer, hyperparams: HyperparamSet, device: str = "auto", reward_normalizer=None):
        import torch
        self._torch  = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._actor  = actor.to(self._device)
        self._critic = critic.to(self._device)
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()
        self._reward_normalizer = reward_normalizer

        self._cache_obs: list      = []
        self._cache_log_probs: list = []
        self._cache_values: list   = []

    def act(self, observation) -> int:
        """Sample action; cache log_prob and V(obs) for learn()."""
        import torch
        import torch.nn.functional as F
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(self._device)
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
        if len(self._cache_obs) < T:
            raise RuntimeError(
                f"Cache underflow: expected >= {T} entries but got {len(self._cache_obs)}. "
                "Ensure act() is called exactly once per environment step."
            )
        obs_arr    = np.array(self._cache_obs[:T],       dtype=np.float32)
        old_lp_arr = np.array(self._cache_log_probs[:T], dtype=np.float32)
        val_arr    = np.array(self._cache_values[:T],    dtype=np.float32)
        rewards    = list(episode_data.rewards)
        dones      = episode_data.dones

        # Reward normalizer: update running stats and reset at episode boundaries.
        if self._reward_normalizer is not None:
            for r, done in zip(rewards, dones):
                self._reward_normalizer.step(r)
                if done:
                    self._reward_normalizer.reset()
            rewards = list(self._reward_normalizer.normalize(np.array(rewards, dtype=np.float32)))

        # Bootstrap: if the window ended mid-episode, V(s_T) from the critic
        # corrects the TD target at the last step. Zero when terminal.
        last_value = 0.0
        if not dones[-1] and episode_data.final_obs is not None:
            with torch.no_grad():
                final_obs_t = torch.as_tensor(
                    np.atleast_2d(episode_data.final_obs), dtype=torch.float32
                ).to(self._device)
                last_value = float(self._critic(final_obs_t).squeeze().item())

        advantages, returns = compute_gae(rewards, val_arr, dones, gamma, gae_lambda, last_value)
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
                obs_b   = torch.as_tensor(batch["obs"],        dtype=torch.float32).to(self._device)
                act_b   = torch.as_tensor(batch["actions"],    dtype=torch.long).to(self._device)
                old_b   = torch.as_tensor(batch["old_lp"],     dtype=torch.float32).to(self._device)
                adv_b   = torch.as_tensor(batch["advantages"], dtype=torch.float32).to(self._device)
                ret_b   = torch.as_tensor(batch["returns"],    dtype=torch.float32).to(self._device)

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

        del self._cache_obs[:T]
        del self._cache_log_probs[:T]
        del self._cache_values[:T]

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

    def action_probs(self, observation) -> np.ndarray:
        """
        Return softmax action probabilities without sampling.
        Used by PolicyManager.ensemble_action() to average distributions
        across ensemble members rather than averaging sampled actions.
        """
        import torch
        import torch.nn.functional as F
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._actor(obs)
            probs = F.softmax(logits, dim=-1)
        return probs[0].cpu().numpy()

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
            torch.load(os.path.join(path, "actor.pt"),  map_location=self._device)
        )
        self._critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pt"), map_location=self._device)
        )

    def average_weights(self, paths: list) -> None:
        import torch
        modules = ("actor", "critic")
        files   = ("actor.pt", "critic.pt")
        nets    = (self._actor, self._critic)
        n = len(paths)
        with torch.no_grad():
            for net, fname in zip(nets, files):
                avg = None
                for path in paths:
                    sd = torch.load(os.path.join(path, fname), map_location=self._device)
                    if avg is None:
                        avg = {k: v.clone().float() for k, v in sd.items()}
                    else:
                        for k in avg:
                            avg[k] += sd[k].float()
                for k in avg:
                    avg[k] /= n
                net.load_state_dict({k: v.to(next(net.parameters()).dtype) for k, v in avg.items()})

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        with torch.no_grad():
            for module in (self._actor, self._critic):
                for param in module.parameters():
                    param.mul_(1.0 + noise_scale * torch.randn_like(param))

    @staticmethod
    def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return float("nan")
        return float(1.0 - np.var(returns - values) / var_returns)

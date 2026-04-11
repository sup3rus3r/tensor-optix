import os
import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class TorchGaussianPPOAgent(BaseAgent):
    """
    PPO for continuous action spaces (PyTorch) using a squashed Gaussian policy.

    Actions are bounded to (-1, 1)^action_dim via tanh. Suitable for continuous
    control tasks such as position sizing in trading (negative = short, positive = long).

    Actor network: obs → [batch, 2 * action_dim]
        First  action_dim outputs = pre-tanh mean
        Second action_dim outputs = log_std  (clamped to [LOG_STD_MIN, LOG_STD_MAX])

    Critic network: obs → [batch, 1] or [batch]  (scalar value estimate)

    Typical usage (position sizing, action_dim=1):

        import torch.nn as nn
        from tensor_optix.algorithms.torch_ppo_continuous import TorchGaussianPPOAgent

        obs_dim    = 13
        action_dim = 1    # scalar position in (-1, 1)

        actor  = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, 2 * action_dim),   # outputs mean || log_std
        )
        critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, 1),
        )
        agent = TorchGaussianPPOAgent(
            actor=actor,
            critic=critic,
            optimizer=torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()), lr=3e-4
            ),
            action_dim=action_dim,
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

    Hyperparams tuned by the framework (all optional, defaults shown above):
        learning_rate, clip_ratio, entropy_coef, vf_coef, gamma,
        gae_lambda, n_epochs, minibatch_size, max_grad_norm
    """

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
        # entropy_coef lo=0.001: continuous PPO std can collapse to zero when entropy_coef=0.
        # gamma included: SPSA can adapt the discount horizon per-environment.
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        actor,
        critic,
        optimizer,
        action_dim: int,
        hyperparams: HyperparamSet,
        device: str = "auto",
        reward_normalizer=None,
    ):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._actor = actor.to(self._device)
        self._critic = critic.to(self._device)
        self._optimizer = optimizer
        self._action_dim = action_dim
        self._hyperparams = hyperparams.copy()
        self._reward_normalizer = reward_normalizer

        # Rollout cache — populated by act(), consumed by learn()
        self._cache_obs: list = []
        self._cache_log_probs: list = []
        self._cache_values: list = []

    # ------------------------------------------------------------------
    # Gaussian log prob with tanh squashing correction
    # ------------------------------------------------------------------

    def _gaussian_log_prob(self, pre_tanh, mean, log_std):
        """
        Log probability of a squashed Gaussian action.

        log π(a|s) = Σᵢ [ log N(uᵢ; μᵢ, σᵢ) − log(1 − tanh²(uᵢ) + ε) ]

        where a = tanh(u), u ~ N(μ, σ).
        Returns a scalar per sample: shape [batch].
        """
        import torch
        std = log_std.exp()
        # Normal log prob: -0.5 * ((u - μ)/σ)² - log σ - 0.5 log(2π)
        log_prob_normal = (
            -0.5 * ((pre_tanh - mean) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        )
        # Tanh correction: subtract Σ log(1 - tanh²(u) + ε)
        tanh_correction = torch.log(1 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        return (log_prob_normal - tanh_correction).sum(dim=-1)  # [batch]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation) -> np.ndarray:
        """
        Sample action ∈ (-1, 1)^action_dim; cache log_prob and V(obs) for learn().
        Returns a float numpy array of shape [action_dim].
        """
        import torch
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(self._device)

        with torch.no_grad():
            out     = self._actor(obs)                                # [1, 2*action_dim]
            mean    = out[:, :self._action_dim]
            log_std = out[:, self._action_dim:].clamp(LOG_STD_MIN, LOG_STD_MAX)
            std     = log_std.exp()

            dist     = torch.distributions.Normal(mean, std)
            pre_tanh = dist.rsample()                                 # [1, action_dim]
            action   = torch.tanh(pre_tanh)                          # [1, action_dim]

            log_prob = self._gaussian_log_prob(pre_tanh, mean, log_std)  # [1]
            value    = self._critic(obs).squeeze()

        self._cache_obs.append(np.squeeze(observation))
        self._cache_log_probs.append(float(log_prob.item()))
        self._cache_values.append(float(value.item()))

        return action.squeeze(0).cpu().numpy()  # [action_dim]

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Run PPO update on the collected rollout.

        Uses the cached log_probs and values from act() calls.
        Recomputes log_prob under the new policy by recovering pre_tanh
        from the stored actions via atanh (numerically stable).

        1. Recover pre_tanh = atanh(actions.clamp(-1+ε, 1-ε))
        2. Compute GAE advantages
        3. Run n_epochs of minibatch clipped surrogate updates
        4. Partially clear cache (del [:T] preserves any residual entries)
        """
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
        # actions are float arrays of shape [action_dim] per step
        act_arr    = np.array(episode_data.actions,      dtype=np.float32)
        rewards    = list(episode_data.rewards)
        dones      = episode_data.dones

        if self._reward_normalizer is not None:
            for r, done in zip(rewards, dones):
                self._reward_normalizer.step(r)
                if done:
                    self._reward_normalizer.reset()
            rewards = list(self._reward_normalizer.normalize(np.array(rewards, dtype=np.float32)))

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
            "actions":    act_arr,
            "old_lp":     old_lp_arr,
            "advantages": advantages,
            "returns":    returns,
        }

        all_params = list(self._actor.parameters()) + list(self._critic.parameters())
        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        total_kl       = 0.0
        n_updates      = 0

        for _ in range(n_epochs):
            for batch in make_minibatches(rollout, mb_size):
                obs_b = torch.as_tensor(batch["obs"],        dtype=torch.float32).to(self._device)
                act_b = torch.as_tensor(batch["actions"],    dtype=torch.float32).to(self._device)
                old_b = torch.as_tensor(batch["old_lp"],     dtype=torch.float32).to(self._device)
                adv_b = torch.as_tensor(batch["advantages"], dtype=torch.float32).to(self._device)
                ret_b = torch.as_tensor(batch["returns"],    dtype=torch.float32).to(self._device)

                # Recover pre_tanh from stored actions (atanh, numerically safe)
                pre_tanh_b = torch.atanh(act_b.clamp(-1 + 1e-6, 1 - 1e-6))

                out_b    = self._actor(obs_b)
                mean_b   = out_b[:, :self._action_dim]
                log_std_b = out_b[:, self._action_dim:].clamp(LOG_STD_MIN, LOG_STD_MAX)

                new_lp   = self._gaussian_log_prob(pre_tanh_b, mean_b, log_std_b)

                ratio    = torch.exp(new_lp - old_b)
                s1       = ratio * adv_b
                s2       = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_b
                pol_loss = -torch.minimum(s1, s2).mean()

                new_val  = self._critic(obs_b).squeeze(-1)
                val_loss = F.mse_loss(new_val, ret_b)

                # Differential entropy of Gaussian (before tanh): H = 0.5 Σ log(2πe σ²)
                std_b   = log_std_b.exp()
                entropy = (0.5 * (1.0 + (2.0 * np.pi * std_b.pow(2)).log())).sum(dim=-1).mean()

                loss = pol_loss + vf_coef * val_loss - entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self._optimizer.step()

                total_pol_loss += float(pol_loss.item())
                total_val_loss += float(val_loss.item())
                total_entropy  += float(entropy.item())
                total_kl       += float((old_b - new_lp).mean().item())
                n_updates      += 1

        # Partial clear: preserve residual entries beyond T
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
        n = len(paths)
        with torch.no_grad():
            for net, fname in ((self._actor, "actor.pt"), (self._critic, "critic.pt")):
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

    def reset_cache(self) -> None:
        """Discard all rollout cache entries without learning from them."""
        self._cache_obs.clear()
        self._cache_log_probs.clear()
        self._cache_values.clear()

    @staticmethod
    def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return float("nan")
        return float(1.0 - np.var(returns - values) / var_returns)

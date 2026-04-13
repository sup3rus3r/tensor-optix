import os
import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class TorchAgent(BaseAgent):
    """
    Base agent for PyTorch models. Mirrors TFAgent's role but for PyTorch.

    The user passes in:
    - model:       their torch.nn.Module (the policy network)
    - optimizer:   torch.optim.Optimizer instance
    - hyperparams: initial HyperparamSet

    For algorithm-specific learning logic (PPO, SAC, DQN), subclass TorchAgent
    and override learn(). The base learn() performs a generic policy gradient
    update (REINFORCE with optional A2C advantage baseline).

    Automatically applied hyperparam keys in set_hyperparams():
    - "learning_rate": float → applied to all optimizer param groups

    Usage:
        import torch
        import torch.nn as nn
        from tensor_optix.adapters.pytorch import TorchAgent

        model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
        agent = TorchAgent(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
            hyperparams=HyperparamSet(params={"learning_rate": 3e-4, "gamma": 0.99}, episode_id=0),
            device="auto",  # "cuda" if available, else "cpu"
        )
    """

    def __init__(self, model, optimizer, hyperparams: HyperparamSet, compute_loss_fn=None, device: str = "auto"):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self.model = model.to(self._device)
        self.optimizer = optimizer
        self._hyperparams = hyperparams.copy()
        self._compute_loss_fn = compute_loss_fn

    def act(self, observation) -> np.ndarray:
        """
        Forward pass. Returns a sampled action index (discrete action spaces).
        Override for continuous actions or custom sampling strategies.
        """
        import torch
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self.model(obs)
        action = int(torch.argmax(logits, dim=-1).item())
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Policy gradient update (REINFORCE or A2C advantage baseline).

        If episode_data.values is provided, computes A2C advantages:
            A_t = G_t - V(s_t)
        Otherwise falls back to normalized REINFORCE returns.

        Override in a subclass for PPO, SAC, DQN, etc.
        """
        import torch
        gamma = float(self._hyperparams.params.get("gamma", 0.99))

        rewards = episode_data.rewards
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_arr = np.array(returns, dtype=np.float32)

        if episode_data.values is not None:
            values_arr  = np.array(episode_data.values, dtype=np.float32)
            advantages  = returns_arr - values_arr
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            targets = torch.as_tensor(advantages, dtype=torch.float32)
            var_returns  = float(np.var(returns_arr))
            explained_var = float(
                1.0 - np.var(returns_arr - values_arr) / (var_returns + 1e-8)
            )
            baseline_type = "a2c"
        else:
            returns_t = torch.as_tensor(returns_arr, dtype=torch.float32)
            if len(returns_arr) > 1:
                targets = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            else:
                targets = returns_t
            explained_var = 0.0
            baseline_type = "reinforce"

        obs = torch.as_tensor(np.array(episode_data.observations), dtype=torch.float32)
        actions = torch.as_tensor(np.array(episode_data.actions), dtype=torch.long)

        if self._compute_loss_fn is not None:
            loss = self._compute_loss_fn(self.model, episode_data, targets)
        else:
            logits   = self.model(obs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            act_lp   = log_probs[range(len(actions)), actions]
            loss     = -(act_lp * targets).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
        )
        self.optimizer.step()

        return {
            "loss":             float(loss.item()),
            "grad_norm":        grad_norm,
            "baseline_type":    baseline_type,
            "explained_variance": explained_var,
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self.optimizer.param_groups[0]["lr"]
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def save_weights(self, path: str) -> None:
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        state = torch.load(os.path.join(path, "model.pt"), map_location=self._device)
        self.model.load_state_dict(state)

    def teardown(self) -> None:
        """Move model to CPU and free CUDA memory."""
        import torch
        self.model.cpu()
        torch.cuda.empty_cache()

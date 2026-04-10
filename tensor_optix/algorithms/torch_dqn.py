import os
import copy

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer


class TorchDQNAgent(BaseAgent):
    """
    DQN agent for PyTorch with discrete action spaces.

    Experience replay with Prioritized Experience Replay (PER) and n-step returns.
    Hard target network updates. Epsilon-greedy exploration.

    Architecture: q_network maps obs → Q-values [n_actions].

    PER and n-step params are exposed in hyperparams so SPSA can adapt them
    automatically during training:
        per_alpha      — prioritization strength (0=uniform, 1=full PER)
        per_beta       — IS correction exponent (annealed toward 1)
        n_step         — multi-step TD target length

    Usage:
        import torch.nn as nn
        q_net = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, n_actions))
        agent = TorchDQNAgent(
            q_network=q_net, n_actions=n_actions,
            optimizer=torch.optim.Adam(q_net.parameters(), lr=1e-3),
            hyperparams=HyperparamSet(params={
                "learning_rate":      1e-3,
                "gamma":              0.99,
                "epsilon":            1.0,
                "epsilon_min":        0.05,
                "epsilon_decay":      0.995,
                "batch_size":         64,
                "target_update_freq": 100,
                "replay_capacity":    100_000,
                "per_alpha":          0.0,
                "per_beta":           0.4,
                "n_step":             1,
            }, episode_id=0),
        )
    """

    def __init__(self, q_network, n_actions: int, optimizer, hyperparams: HyperparamSet, device: str = "auto"):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._q        = q_network.to(self._device)
        self._q_target = copy.deepcopy(q_network).to(self._device)
        self._n_actions = n_actions
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

        hp = hyperparams.params
        capacity  = int(hp.get("replay_capacity", 100_000))
        per_alpha = float(hp.get("per_alpha", 0.0))
        per_beta  = float(hp.get("per_beta",  0.4))
        n_step    = int(hp.get("n_step",    1))
        gamma     = float(hp.get("gamma",   0.99))

        self._buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=per_alpha,
            beta=per_beta,
            n_step=n_step,
            gamma=gamma,
        )
        self._learn_calls = 0

    # Recommended SPSA/TrialOrchestrator bounds for this algorithm.
    # Lower-bound on lr is 1e-4 — below this DQN updates are too small to
    # overcome CartPole-scale noise before the replay buffer warms up.
    default_param_bounds = {
        "learning_rate": (1e-4, 1e-3),
        "gamma":         (0.95, 0.999),
        # epsilon_decay intentionally excluded: it's a schedule, not an optimisable
        # hyperparameter.  SPSA has no signal to differentiate epsilon_decay values
        # within a single episode window, so it random-walks toward 0.999, which
        # prevents epsilon from ever decaying and leaves the agent permanently random.
    }
    default_log_params = ["learning_rate"]

    # With default epsilon_decay=0.95, epsilon reaches floor at episode ~58.
    # DORMANT must not fire before then — a stalled agent before the floor is reached
    # is still exploring, not genuinely stuck.
    default_min_episodes_before_dormant = 60

    @property
    def is_on_policy(self) -> bool:
        return False

    def act(self, observation) -> int:
        import torch
        epsilon = float(self._hyperparams.params.get("epsilon", 0.1))
        if np.random.random() < epsilon:
            return np.random.randint(self._n_actions)
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            q = self._q(obs)
        return int(q.argmax(dim=-1).item())

    def learn(self, episode_data: EpisodeData) -> dict:
        import torch
        import torch.nn.functional as F

        hp = self._hyperparams.params
        gamma              = float(hp.get("gamma",              0.99))
        batch_size         = int(hp.get("batch_size",           64))
        target_update_freq = int(hp.get("target_update_freq",   100))
        epsilon_decay      = float(hp.get("epsilon_decay",      0.995))
        epsilon_min        = float(hp.get("epsilon_min",        0.05))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = episode_data.actions
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        for t in range(T - 1):
            self._buffer.push(obs_arr[t], int(act_arr[t]), float(rew_arr[t]),
                              obs_arr[t + 1], bool(done_arr[t]))
        self._buffer.flush_episode()

        self._learn_calls += 1
        new_eps = max(epsilon_min, float(hp.get("epsilon", 1.0)) * epsilon_decay)
        self._hyperparams.params["epsilon"] = new_eps

        if len(self._buffer) < batch_size:
            return {"loss": 0.0, "epsilon": new_eps, "buffer_size": len(self._buffer)}

        # Gradient updates proportional to steps collected: 1 update per batch_size steps.
        # Standard DQN does ~1 update per 4 steps; BatchPipeline gives us a window of
        # T steps in one learn() call.  Without this loop the Q-network is trained
        # 512× too rarely relative to environment interaction, which prevents convergence.
        n_updates = max(1, (T - 1) // batch_size)

        total_loss     = 0.0
        total_td_error = 0.0
        target_updated = False
        last_indices   = None
        last_td_errors = None

        for _ in range(n_updates):
            obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = self._buffer.sample(batch_size)

            obs_b   = torch.as_tensor(obs_b,    dtype=torch.float32).to(self._device)
            act_b   = torch.as_tensor(act_b,    dtype=torch.long).to(self._device)
            rew_b   = torch.as_tensor(rew_b,    dtype=torch.float32).to(self._device)
            next_b  = torch.as_tensor(next_b,   dtype=torch.float32).to(self._device)
            done_b  = torch.as_tensor(done_b,   dtype=torch.float32).to(self._device)
            weights = torch.as_tensor(weights,  dtype=torch.float32).to(self._device)
            gammas_n = torch.as_tensor(
                gamma ** n_steps.astype(np.float32), dtype=torch.float32
            ).to(self._device)

            with torch.no_grad():
                max_next = self._q_target(next_b).max(dim=-1).values
                targets  = rew_b + gammas_n * max_next * (1.0 - done_b)

            q_vals  = self._q(obs_b)
            q_taken = q_vals.gather(1, act_b.unsqueeze(1)).squeeze(1)

            td_errors = (q_taken - targets).detach().cpu().numpy()
            loss = (weights * F.mse_loss(q_taken, targets, reduction="none")).mean()

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._q.parameters(), 10.0)
            self._optimizer.step()

            self._buffer.update_priorities(indices, np.abs(td_errors))

            total_loss     += float(loss.item())
            total_td_error += float(np.abs(td_errors).mean())
            last_indices    = indices
            last_td_errors  = td_errors

        # Target network update counted per window (not per gradient step)
        if self._learn_calls % target_update_freq == 0:
            self._q_target.load_state_dict(self._q.state_dict())
            target_updated = True

        episode_reward = float(np.sum(episode_data.rewards)) if episode_data.rewards is not None else None
        return {
            "loss":           total_loss / n_updates,
            "epsilon":        new_eps,
            "buffer_size":    len(self._buffer),
            "target_updated": int(target_updated),
            "td_error_mean":  total_td_error / n_updates,
            "episode_reward": episode_reward,
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._optimizer.param_groups[0]["lr"]
        )
        self._hyperparams.params["per_alpha"] = self._buffer._alpha
        self._hyperparams.params["per_beta"]  = self._buffer._beta
        self._hyperparams.params["n_step"]    = self._buffer._n_step
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        hp = hyperparams.params
        if "learning_rate" in hp:
            lr = float(hp["learning_rate"])
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr
        self._buffer.set_params(
            alpha=hp.get("per_alpha"),
            beta=hp.get("per_beta"),
            n_step=hp.get("n_step"),
            gamma=hp.get("gamma"),
        )

    def save_weights(self, path: str) -> None:
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self._q.state_dict(), os.path.join(path, "q_network.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        self._q.load_state_dict(
            torch.load(os.path.join(path, "q_network.pt"), map_location=self._device)
        )
        self._q_target.load_state_dict(self._q.state_dict())

    def average_weights(self, paths: list) -> None:
        import torch
        n = len(paths)
        with torch.no_grad():
            avg = None
            for path in paths:
                sd = torch.load(os.path.join(path, "q_network.pt"), map_location=self._device)
                if avg is None:
                    avg = {k: v.clone().float() for k, v in sd.items()}
                else:
                    for k in avg:
                        avg[k] += sd[k].float()
            for k in avg:
                avg[k] /= n
            self._q.load_state_dict({k: v.to(next(self._q.parameters()).dtype) for k, v in avg.items()})
            self._q_target.load_state_dict(self._q.state_dict())

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        with torch.no_grad():
            for param in self._q.parameters():
                param.mul_(1.0 + noise_scale * torch.randn_like(param))
        self._q_target.load_state_dict(self._q.state_dict())

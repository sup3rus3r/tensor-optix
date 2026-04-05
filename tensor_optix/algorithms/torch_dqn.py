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
            np.array(obs, dtype=np.float32), int(action),
            float(reward), np.array(next_obs, dtype=np.float32), float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


class TorchDQNAgent(BaseAgent):
    """
    DQN agent for PyTorch with discrete action spaces.

    Experience replay, hard target network updates, epsilon-greedy exploration.

    Architecture: q_network maps obs → Q-values [n_actions].

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
            }, episode_id=0),
        )
    """

    def __init__(self, q_network, n_actions: int, optimizer, hyperparams: HyperparamSet):
        import torch
        self._torch    = torch
        self._q        = q_network
        self._q_target = copy.deepcopy(q_network)
        self._n_actions = n_actions
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

        capacity = int(hyperparams.params.get("replay_capacity", 100_000))
        self._buffer = _ReplayBuffer(capacity)
        self._learn_calls = 0

    def act(self, observation) -> int:
        import torch
        epsilon = float(self._hyperparams.params.get("epsilon", 0.1))
        if np.random.random() < epsilon:
            return np.random.randint(self._n_actions)
        obs = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32)
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

        self._learn_calls += 1
        new_eps = max(epsilon_min, float(hp.get("epsilon", 1.0)) * epsilon_decay)
        self._hyperparams.params["epsilon"] = new_eps

        if len(self._buffer) < batch_size:
            return {"loss": 0.0, "epsilon": new_eps, "buffer_size": len(self._buffer)}

        obs_b, act_b, rew_b, next_b, done_b = self._buffer.sample(batch_size)
        obs_b   = torch.as_tensor(obs_b,   dtype=torch.float32)
        act_b   = torch.as_tensor(act_b,   dtype=torch.long)
        rew_b   = torch.as_tensor(rew_b,   dtype=torch.float32)
        next_b  = torch.as_tensor(next_b,  dtype=torch.float32)
        done_b  = torch.as_tensor(done_b,  dtype=torch.float32)

        with torch.no_grad():
            max_next = self._q_target(next_b).max(dim=-1).values
            targets  = rew_b + gamma * max_next * (1.0 - done_b)

        q_vals  = self._q(obs_b)
        q_taken = q_vals.gather(1, act_b.unsqueeze(1)).squeeze(1)
        loss    = F.mse_loss(q_taken, targets)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        target_updated = False
        if self._learn_calls % target_update_freq == 0:
            self._q_target.load_state_dict(self._q.state_dict())
            target_updated = True

        return {
            "loss":           float(loss.item()),
            "epsilon":        new_eps,
            "buffer_size":    len(self._buffer),
            "target_updated": int(target_updated),
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
        torch.save(self._q.state_dict(), os.path.join(path, "q_network.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        self._q.load_state_dict(
            torch.load(os.path.join(path, "q_network.pt"), map_location="cpu")
        )
        self._q_target.load_state_dict(self._q.state_dict())

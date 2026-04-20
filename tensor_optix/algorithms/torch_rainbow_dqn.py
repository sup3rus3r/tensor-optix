"""
tensor_optix.algorithms.torch_rainbow_dqn — Rainbow DQN for PyTorch.

Rainbow (Hessel et al. 2017) combines six improvements to DQN.
This agent implements all six:

| Component          | Status | Where                                   |
|--------------------|--------|-----------------------------------------|
| Double Q-learning  | [x]    | online net selects action, target net   |
|                    |        | evaluates value in Bellman target       |
| PER                | [x]    | PrioritizedReplayBuffer (shared core)   |
| n-step returns     | [x]    | PrioritizedReplayBuffer (shared core)   |
| Dueling networks   | [x]    | RainbowQNetwork (V + A head)            |
| Noisy nets         | [x]    | NoisyLinear replaces Linear             |
| Distributional RL  | [x]    | C51: 51-atom return distribution        |

**Architecture:**

    Input → NoisyLinear(hidden) → ReLU
          → [Value stream: NoisyLinear → V(s)]
          → [Advantage stream: NoisyLinear → A(s, ·)]
          → Q_θ(s, a) = V(s) + A(s,a) - mean_a A(s,a)     [dueling aggregation]
          → softmax over n_atoms for each action            [C51 distributional]

**C51 distributional Bellman projection (Bellemare et al. 2017):**

Support z = {V_min + i·Δz | i=0,...,N-1},  Δz = (V_max - V_min) / (N-1)

For a batch with n-step returns:

    T̂z_j = clip(r + γⁿ·z_j, V_min, V_max)     for each atom j
    b_j   = (T̂z_j - V_min) / Δz               ← fractional position
    l_j   = floor(b_j),  u_j = ceil(b_j)

    m_l += p_j(s', a*) · (u_j - b_j)           ← lower bin weight
    m_u += p_j(s', a*) · (b_j - l_j)           ← upper bin weight

Loss: L = -Σ_i m_i · log p_i(s, a)             (cross-entropy)

**Noisy nets exploration:**

    ε-greedy is completely removed. No epsilon_decay to tune.
    Exploration is driven by σ parameters learned by gradient descent.
    σ→0 on confident states (exploit), σ→large on uncertain states (explore).

Usage:
    import torch.nn as nn
    from tensor_optix.algorithms.torch_rainbow_dqn import TorchRainbowDQNAgent, RainbowQNetwork
    from tensor_optix.core.types import HyperparamSet

    net = RainbowQNetwork(obs_dim=4, n_actions=2, hidden_size=128)
    agent = TorchRainbowDQNAgent(
        q_network=net, n_actions=2, obs_dim=4,
        optimizer=torch.optim.Adam(net.parameters(), lr=6.25e-5),
        hyperparams=HyperparamSet(params={
            "learning_rate":      6.25e-5,
            "gamma":              0.99,
            "batch_size":         32,
            "target_update_freq": 200,
            "replay_capacity":    100_000,
            "per_alpha":          0.5,
            "per_beta":           0.4,
            "n_step":             3,
            "v_min":             -10.0,
            "v_max":              10.0,
            "n_atoms":            51,
        }, episode_id=0),
    )
"""

import os
import copy
import math

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer
from tensor_optix.core.noisy_linear import NoisyLinear


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class RainbowQNetwork:
    """
    Factory helper — builds the dueling + noisy + distributional Q-network
    as a plain nn.Module.

    Call RainbowQNetwork.build(obs_dim, n_actions, hidden_size, n_atoms)
    to get the nn.Module.
    """

    @staticmethod
    def build(obs_dim: int, n_actions: int, hidden_size: int = 128, n_atoms: int = 51):
        """
        Returns an nn.Module: obs → (n_actions, n_atoms) log-probabilities.

        Architecture (dueling + noisy):
            shared:    NoisyLinear(obs_dim, hidden_size) → ReLU
            value:     NoisyLinear(hidden_size, n_atoms)
            advantage: NoisyLinear(hidden_size, n_actions * n_atoms)

        Dueling aggregation over atoms:
            Q_i(s,a) = V_i(s) + A_i(s,a) - mean_a A_i(s,a)
            then log_softmax over atoms.
        """
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self._obs_dim    = obs_dim
                self._n_actions  = n_actions
                self._n_atoms    = n_atoms
                self._hidden     = hidden_size

                self.shared   = nn.Sequential(
                    NoisyLinear(obs_dim, hidden_size),
                    nn.ReLU(),
                )
                self.val_head = NoisyLinear(hidden_size, n_atoms)
                self.adv_head = NoisyLinear(hidden_size, n_actions * n_atoms)

            def forward(self, x):
                feat = self.shared(x)                                   # (B, hidden)
                val  = self.val_head(feat)                              # (B, n_atoms)
                adv  = self.adv_head(feat).view(-1, self._n_actions,
                                                 self._n_atoms)         # (B, A, n_atoms)
                # Dueling: Q = V + A - mean(A)
                q = (val.unsqueeze(1)
                     + adv
                     - adv.mean(dim=1, keepdim=True))                  # (B, A, n_atoms)
                return q.log_softmax(dim=-1)                            # log P(Z|s,a)

            def reset_noise(self):
                """Resample factorized noise in all NoisyLinear layers."""
                for m in self.modules():
                    if isinstance(m, NoisyLinear):
                        m.reset_noise()

        return _Net()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TorchRainbowDQNAgent(BaseAgent):
    """
    Rainbow DQN (Hessel et al. 2017) for PyTorch discrete action spaces.

    See module docstring for the full component list and math.

    Parameters
    ----------
    q_network : nn.Module
        A RainbowQNetwork (or compatible module) that returns
        log P(Z|s, a) of shape (batch, n_actions, n_atoms).
        Build with RainbowQNetwork.build(obs_dim, n_actions).
    n_actions : int
    obs_dim   : int   — needed to create dummy inputs for ONNX export.
    optimizer : torch.optim.Optimizer
    hyperparams : HyperparamSet

        Required params:
            gamma, batch_size, replay_capacity
        Rainbow-specific (with defaults):
            v_min   — lower bound of support  (default -10)
            v_max   — upper bound of support  (default +10)
            n_atoms — support size            (default  51)
            per_alpha, per_beta, n_step       (defaults: 0.5, 0.4, 3)
            target_update_freq               (default: 200)
    device : str
    """

    def __init__(
        self,
        q_network,
        n_actions: int,
        obs_dim: int,
        optimizer,
        hyperparams: HyperparamSet,
        device: str = "auto",
    ):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device    = torch.device(device)
        self._q         = q_network.to(self._device)
        self._q_target  = copy.deepcopy(q_network).to(self._device)
        self._n_actions = n_actions
        self._obs_dim   = obs_dim
        self._optimizer = optimizer
        self._hyperparams = hyperparams.copy()

        hp = hyperparams.params
        self._v_min    = float(hp.get("v_min",   -10.0))
        self._v_max    = float(hp.get("v_max",    10.0))
        self._n_atoms  = int(hp.get("n_atoms",     51))
        # Support tensor: shape (n_atoms,)
        self._support  = torch.linspace(
            self._v_min, self._v_max, self._n_atoms,
            dtype=torch.float32, device=self._device
        )
        self._delta_z  = (self._v_max - self._v_min) / (self._n_atoms - 1)

        capacity  = int(hp.get("replay_capacity", 100_000))
        per_alpha = float(hp.get("per_alpha", 0.5))
        per_beta  = float(hp.get("per_beta",  0.4))
        n_step    = int(hp.get("n_step",    3))
        gamma     = float(hp.get("gamma",   0.99))

        self._buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=per_alpha,
            beta=per_beta,
            n_step=n_step,
            gamma=gamma,
        )
        self._learn_calls = 0

    @property
    def is_on_policy(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    def act(self, observation) -> int:
        """
        Greedy action: argmax_a E[Z(s,a)] = argmax_a Σ_i z_i · P_i(s,a)

        No ε-greedy.  Exploration is entirely driven by noise in σ parameters.
        At eval time (model.eval()), σ is ignored and only μ weights are used.
        """
        import torch
        obs = torch.as_tensor(np.atleast_2d(observation),
                              dtype=torch.float32).to(self._device)
        with torch.no_grad():
            log_probs = self._q(obs)                        # (1, n_actions, n_atoms)
            q_mean    = (log_probs.exp() * self._support).sum(dim=-1)  # (1, n_actions)
        return int(q_mean.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # learn
    # ------------------------------------------------------------------

    def learn(self, episode_data: EpisodeData) -> dict:
        import torch
        import torch.nn.functional as F

        hp = self._hyperparams.params
        gamma              = float(hp.get("gamma",              0.99))
        batch_size         = int(hp.get("batch_size",           32))
        target_update_freq = int(hp.get("target_update_freq",   200))

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

        if len(self._buffer) < batch_size:
            return {"loss": 0.0, "buffer_size": len(self._buffer)}

        n_updates  = max(1, (T - 1) // batch_size)
        total_loss = 0.0

        for _ in range(n_updates):
            obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = \
                self._buffer.sample(batch_size)

            obs_b   = torch.as_tensor(obs_b,   dtype=torch.float32).to(self._device)
            act_b   = torch.as_tensor(act_b,   dtype=torch.long).to(self._device)
            rew_b   = torch.as_tensor(rew_b,   dtype=torch.float32).to(self._device)
            next_b  = torch.as_tensor(next_b,  dtype=torch.float32).to(self._device)
            done_b  = torch.as_tensor(done_b,  dtype=torch.float32).to(self._device)
            weights = torch.as_tensor(weights, dtype=torch.float32).to(self._device)
            gammas_n = torch.as_tensor(
                gamma ** n_steps.astype(np.float32), dtype=torch.float32
            ).to(self._device)                                          # (B,)

            B = obs_b.shape[0]

            # ----------------------------------------------------------
            # C51 distributional Bellman projection
            # ----------------------------------------------------------
            with torch.no_grad():
                # Double-Q: online net selects best action
                self._q.reset_noise()
                log_p_next_online = self._q(next_b)                     # (B, A, N)
                q_next_mean       = (log_p_next_online.exp()
                                     * self._support).sum(dim=-1)       # (B, A)
                best_actions      = q_next_mean.argmax(dim=-1)          # (B,)

                # Target net evaluates the value at the chosen action
                self._q_target.reset_noise()
                log_p_next_target = self._q_target(next_b)              # (B, A, N)
                # p_j(s', a*) — probabilities of best action under target net
                p_next = log_p_next_target.exp()[
                    torch.arange(B, device=self._device), best_actions
                ]                                                        # (B, N)

                # Project onto support
                # T̂z_j = clip(r + γⁿ · z_j, V_min, V_max)
                # shape broadcasts: rew_b (B,1), gammas_n (B,1), support (N,)
                tz = (rew_b.unsqueeze(1)
                      + gammas_n.unsqueeze(1) * self._support.unsqueeze(0)
                      * (1.0 - done_b.unsqueeze(1)))                    # (B, N)
                tz = tz.clamp(self._v_min, self._v_max)

                b  = (tz - self._v_min) / self._delta_z                # (B, N) float
                lo = b.floor().long().clamp(0, self._n_atoms - 1)      # (B, N)
                hi = b.ceil().long().clamp(0, self._n_atoms - 1)       # (B, N)

                # Distribute probability mass m (B, N_atoms)
                # Use fractional-offset form to handle exact-integer b correctly.
                # b_lo = b - floor(b) ∈ [0, 1).  When b is exactly integer, b_lo = 0
                # and the full mass goes to lo; (hi.float() - b) would be 0 in that
                # case and lose all mass.  The fraction form avoids that edge case:
                #   m[lo] += p * (1 - b_lo),  m[hi] += p * b_lo
                # sum = p for all b, including when lo == hi.
                m    = torch.zeros(B, self._n_atoms, device=self._device)
                b_lo = b - lo.float()          # fractional offset from lower atom
                m.scatter_add_(1, lo, p_next * (1.0 - b_lo))
                m.scatter_add_(1, hi, p_next * b_lo)

            # ----------------------------------------------------------
            # Loss: cross-entropy between target m and predicted log_p
            # ----------------------------------------------------------
            self._q.reset_noise()
            log_p = self._q(obs_b)                                      # (B, A, N)
            log_p_taken = log_p[torch.arange(B, device=self._device),
                                act_b]                                  # (B, N)

            # Per-sample loss: -Σ_i m_i · log p_i
            elem_loss = -(m * log_p_taken).sum(dim=-1)                  # (B,)
            loss      = (weights * elem_loss).mean()

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._q.parameters(), 10.0)
            self._optimizer.step()

            # PER priorities: use cross-entropy per sample as proxy for TD error
            self._buffer.update_priorities(indices, elem_loss.detach().cpu().numpy())
            total_loss += float(loss.item())

        # Hard target update
        target_updated = False
        if self._learn_calls % target_update_freq == 0:
            self._q_target.load_state_dict(self._q.state_dict())
            target_updated = True

        episode_reward = float(np.sum(episode_data.rewards)) if episode_data.rewards is not None else None
        return {
            "loss":           total_loss / n_updates,
            "buffer_size":    len(self._buffer),
            "target_updated": int(target_updated),
            "episode_reward": episode_reward,
        }

    # ------------------------------------------------------------------
    # Hyperparams
    # ------------------------------------------------------------------

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
            for pg in self._optimizer.param_groups:
                pg["lr"] = float(hp["learning_rate"])
        self._buffer.set_params(
            alpha=hp.get("per_alpha"),
            beta=hp.get("per_beta"),
            n_step=hp.get("n_step"),
            gamma=hp.get("gamma"),
        )
        if "v_min" in hp or "v_max" in hp or "n_atoms" in hp:
            import torch
            self._v_min   = float(hp.get("v_min",   self._v_min))
            self._v_max   = float(hp.get("v_max",   self._v_max))
            self._n_atoms = int(hp.get("n_atoms",   self._n_atoms))
            self._support = torch.linspace(
                self._v_min, self._v_max, self._n_atoms,
                dtype=torch.float32, device=self._device,
            )
            self._delta_z = (self._v_max - self._v_min) / (self._n_atoms - 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
            dtype = next(self._q.parameters()).dtype
            self._q.load_state_dict({k: v.to(dtype) for k, v in avg.items()})
            self._q_target.load_state_dict(self._q.state_dict())

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        with torch.no_grad():
            for p in self._q.parameters():
                p.mul_(1.0 + noise_scale * torch.randn_like(p))
        self._q_target.load_state_dict(self._q.state_dict())

    def export_onnx(self, path: str) -> None:
        """
        Export the distributional Q-network (mean Q-values) to ONNX.

        Input  — "observation": (batch_size, obs_dim)    float32
        Output — "q_mean":      (batch_size, n_actions)  float32
                 Expected return E[Z(s,a)] = Σ_i z_i · P_i(s,a)

        At inference: argmax q_mean for the greedy action.

        Requires the ``onnx`` optional dependency:
            pip install tensor-optix[onnx]
        """
        import torch
        import torch.nn as nn

        # Wrapper that converts log-probabilities to expected Q-values
        class _QMeanWrapper(nn.Module):
            def __init__(self, net, support):
                super().__init__()
                self.net     = net
                self.support = nn.Parameter(support, requires_grad=False)

            def forward(self, obs):
                log_p = self.net(obs)          # (B, A, N)
                return (log_p.exp() * self.support).sum(dim=-1)  # (B, A)

        was_training = self._q.training
        self._q.eval().cpu()
        support_cpu = self._support.cpu()
        wrapper     = _QMeanWrapper(self._q, support_cpu)

        dummy = torch.zeros(1, self._obs_dim, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["observation"],
            output_names=["q_mean"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "q_mean":      {0: "batch_size"},
            },
            opset_version=17,
        )
        self._q.train(was_training).to(self._device)

    def teardown(self) -> None:
        """Move networks to CPU and free CUDA memory."""
        import torch
        self._q.cpu()
        self._q_target.cpu()
        self._support = self._support.cpu()
        torch.cuda.empty_cache()

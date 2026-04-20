"""
TorchTD3Agent — Twin Delayed DDPG for PyTorch continuous action spaces.

Reference: Fujimoto et al. (2018). Addressing Function Approximation Error
in Actor-Critic Methods. ICML.

TD3 extends DDPG with three targeted fixes to the deterministic policy gradient:

Fix 1 — Twin critics (clipped double-Q):
    Q_target = r + γ · min(Q_φ1'(s', ã), Q_φ2'(s', ã))
    Taking the minimum over two independently-initialized critics reduces
    the overestimation bias that arises when a single critic is used as both
    the policy improvement target and the value estimator.

Fix 2 — Delayed policy updates (variance reduction):
    The actor gradient ∇_θ J = E[∇_a Q_φ1(s,a)|_{a=π_θ(s)} · ∇_θ π_θ(s)]
    flows through the critic. When the critic is noisy (early training), the
    actor gradient is also noisy. Updating the actor every d critic steps
    (default d=2) allows the critic to partially stabilize before propagating
    error into the policy, reducing the variance of the policy gradient.

Fix 3 — Target policy smoothing (Q-function regularization):
    ã = clip(π_θ'(s') + clip(ε, -c, c), a_low, a_high),  ε ~ N(0, σ)
    Without smoothing, the critic can learn spurious Q-value spikes around
    specific target actions because no replay experience exists in their
    neighbourhood. Smoothing enforces Q-function consistency in a local
    neighbourhood around the target action, eliminating these exploitable
    singularities.

Architecture:
    actor:   nn.Module, obs → tanh(action), shape [batch, action_dim]
             Actions are deterministic — no sampling, no reparameterization.
    critic1: nn.Module, [obs || action] → Q-value, shape [batch, 1]
    critic2: nn.Module, same architecture, independent weights (twin-Q)

Usage::

    import torch
    import torch.nn as nn
    from tensor_optix.algorithms.torch_td3 import TorchTD3Agent

    obs_dim, act_dim = 8, 2

    actor   = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                             nn.Linear(256, 256),    nn.ReLU(),
                             nn.Linear(256, act_dim), nn.Tanh())
    critic1 = nn.Sequential(nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, 1))
    critic2 = nn.Sequential(nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, 1))

    agent = TorchTD3Agent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
        ),
        hyperparams=HyperparamSet(params={
            "learning_rate":     3e-4,
            "gamma":             0.99,
            "tau":               0.005,
            "batch_size":        256,
            "updates_per_step":  1,
            "replay_capacity":   1_000_000,
            "policy_delay":      2,
            "target_noise":      0.2,
            "target_noise_clip": 0.5,
            "per_alpha":         0.0,
            "per_beta":          0.4,
        }, episode_id=0),
    )

Actions are in (-1, 1) via tanh. Scale to your env's action range via a
wrapper or subclass if the environment expects a different range.
"""

import copy
import os

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer


class TorchTD3Agent(BaseAgent):
    """TD3 agent for PyTorch continuous action spaces. See module docstring."""

    default_param_bounds = {
        "learning_rate": (1e-4, 1e-3),
        "gamma":         (0.97, 0.999),
        "tau":           (1e-3, 1e-1),
    }
    default_log_params = ["learning_rate", "tau"]

    # Twin Q-networks need ~30 episodes to produce stable value estimates
    # before DORMANT-triggered interventions are meaningful.
    default_min_episodes_before_dormant = 30

    def __init__(
        self,
        actor,
        critic1,
        critic2,
        action_dim: int,
        actor_optimizer,
        critic_optimizer,
        hyperparams: HyperparamSet,
        device: str = "auto",
    ):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._actor   = actor.to(self._device)
        self._c1      = critic1.to(self._device)
        self._c2      = critic2.to(self._device)

        # Target networks: separate copies that lag the online networks via
        # Polyak averaging.  Initialized to exact copies of the online nets.
        self._actor_tgt = copy.deepcopy(actor).to(self._device)
        self._c1_tgt    = copy.deepcopy(critic1).to(self._device)
        self._c2_tgt    = copy.deepcopy(critic2).to(self._device)

        self._action_dim  = action_dim
        self._actor_opt   = actor_optimizer
        self._critic_opt  = critic_optimizer
        self._hyperparams = hyperparams.copy()

        # Total critic update count — drives the policy_delay gate.
        # The actor is updated when _update_count % policy_delay == 0.
        self._update_count: int = 0

        hp = hyperparams.params
        self._buffer = PrioritizedReplayBuffer(
            capacity=int(hp.get("replay_capacity", 1_000_000)),
            alpha=float(hp.get("per_alpha", 0.0)),
            beta=float(hp.get("per_beta", 0.4)),
            n_step=int(hp.get("n_step", 1)),
            gamma=float(hp.get("gamma", 0.99)),
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        return False  # replay buffer — rollback without buffer clear is harmful

    def act(self, observation) -> np.ndarray:
        """
        Deterministic action: a = π_θ(s) = tanh(actor(s)).
        No sampling — the actor network is fully deterministic at inference.
        """
        import torch
        obs = torch.as_tensor(
            np.atleast_2d(observation), dtype=torch.float32
        ).to(self._device)
        with torch.no_grad():
            action = self._actor(obs)
        return action.cpu().numpy()[0]

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        gamma            = float(hp.get("gamma",            0.99))
        tau              = float(hp.get("tau",              0.005))
        batch_size       = int(hp.get("batch_size",         256))
        updates_per_step = int(hp.get("updates_per_step",   1))
        policy_delay     = int(hp.get("policy_delay",       2))

        obs_arr  = np.array(episode_data.observations, dtype=np.float32)
        act_arr  = np.array(episode_data.actions,      dtype=np.float32)
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        for t in range(T - 1):
            self._buffer.push(
                obs_arr[t], act_arr[t], float(rew_arr[t]),
                obs_arr[t + 1], float(done_arr[t]),
            )
        self._buffer.flush_episode()

        if len(self._buffer) < batch_size:
            return {
                "actor_loss":    0.0,
                "critic_loss":   0.0,
                "buffer_size":   len(self._buffer),
                "policy_update": 0,
            }

        n_updates = T * updates_per_step
        total_al = 0.0
        total_cl = 0.0
        n_actor_updates = 0

        for _ in range(n_updates):
            self._update_count += 1
            update_actor = (self._update_count % policy_delay == 0)
            al, cl = self._update_step(batch_size, gamma, tau, update_actor)
            total_cl += cl
            if update_actor:
                total_al += al
                n_actor_updates += 1

        return {
            "actor_loss":    total_al / max(n_actor_updates, 1),
            "critic_loss":   total_cl / n_updates,
            "buffer_size":   len(self._buffer),
            "policy_update": int(n_actor_updates > 0),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._actor_opt.param_groups[0]["lr"]
        )
        self._hyperparams.params["per_alpha"] = self._buffer._alpha
        self._hyperparams.params["per_beta"]  = self._buffer._beta
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()
        hp = hyperparams.params
        if "learning_rate" in hp:
            lr = float(hp["learning_rate"])
            for opt in (self._actor_opt, self._critic_opt):
                for pg in opt.param_groups:
                    pg["lr"] = lr
        self._buffer.set_params(
            alpha=hp.get("per_alpha"),
            beta=hp.get("per_beta"),
            gamma=hp.get("gamma"),
        )

    def save_weights(self, path: str) -> None:
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self._actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self._c1.state_dict(),    os.path.join(path, "critic1.pt"))
        torch.save(self._c2.state_dict(),    os.path.join(path, "critic2.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        self._actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"),    map_location=self._device)
        )
        self._c1.load_state_dict(
            torch.load(os.path.join(path, "critic1.pt"),  map_location=self._device)
        )
        self._c2.load_state_dict(
            torch.load(os.path.join(path, "critic2.pt"),  map_location=self._device)
        )
        # Sync target networks from loaded weights
        self._actor_tgt.load_state_dict(self._actor.state_dict())
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    def average_weights(self, paths: list) -> None:
        import torch
        n = len(paths)
        nets_files = [
            (self._actor, "actor.pt"),
            (self._c1,    "critic1.pt"),
            (self._c2,    "critic2.pt"),
        ]
        with torch.no_grad():
            for net, fname in nets_files:
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
                net.load_state_dict(
                    {k: v.to(next(net.parameters()).dtype) for k, v in avg.items()}
                )
        # Sync target networks
        self._actor_tgt.load_state_dict(self._actor.state_dict())
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        with torch.no_grad():
            for module in (self._actor, self._c1, self._c2):
                for param in module.parameters():
                    param.mul_(1.0 + noise_scale * torch.randn_like(param))
        # Sync target networks to perturbed online networks
        self._actor_tgt.load_state_dict(self._actor.state_dict())
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    def export_onnx(self, path: str) -> None:
        """
        Export the deterministic continuous actor to ONNX.

        Input  — "observation": (batch_size, obs_dim)    float32
        Output — "action":      (batch_size, action_dim) float32

        The actor applies tanh internally, so outputs are clipped to (-1, 1).
        Rescale to your environment's action bounds at deployment if required.

        Requires the ``onnx`` optional dependency:
            pip install tensor-optix[onnx]
        """
        import torch
        obs_dim = None
        for m in self._actor.modules():
            if hasattr(m, "in_features"):
                obs_dim = m.in_features
                break
        if obs_dim is None:
            raise RuntimeError(
                "export_onnx: cannot infer obs_dim — no nn.Linear found in actor."
            )
        was_training = self._actor.training
        self._actor.eval().cpu()
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        torch.onnx.export(
            self._actor,
            dummy,
            str(path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action":      {0: "batch_size"},
            },
            opset_version=17,
        )
        self._actor.train(was_training).to(self._device)

    def teardown(self) -> None:
        """Move all networks to CPU and free CUDA memory."""
        import torch
        for module in (self._actor, self._c1, self._c2,
                       self._actor_tgt, self._c1_tgt, self._c2_tgt):
            module.cpu()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_step(
        self,
        batch_size: int,
        gamma: float,
        tau: float,
        update_actor: bool,
    ):
        """
        One TD3 gradient step.

        Critic:  always updated (every step).
        Actor:   updated only when update_actor=True (every policy_delay steps).
        Targets: Polyak-averaged toward online networks after every step.
        """
        import torch
        import torch.nn.functional as F

        hp = self._hyperparams.params
        target_noise      = float(hp.get("target_noise",      0.2))
        target_noise_clip = float(hp.get("target_noise_clip", 0.5))

        obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = \
            self._buffer.sample(batch_size)

        obs_b   = torch.as_tensor(obs_b,   dtype=torch.float32).to(self._device)
        act_b   = torch.as_tensor(act_b,   dtype=torch.float32).to(self._device)
        rew_b   = torch.as_tensor(rew_b,   dtype=torch.float32).to(self._device)
        next_b  = torch.as_tensor(next_b,  dtype=torch.float32).to(self._device)
        done_b  = torch.as_tensor(done_b,  dtype=torch.float32).to(self._device)
        weights = torch.as_tensor(weights, dtype=torch.float32).to(self._device)
        gammas_n = torch.as_tensor(
            gamma ** n_steps.astype(np.float32), dtype=torch.float32
        ).to(self._device)

        # ---- Compute TD targets ----
        with torch.no_grad():
            # Fix 3: target policy smoothing.
            # Noise ε ~ N(0, target_noise²) clipped to [-c, c], added to the
            # deterministic target action. This prevents the critic from learning
            # a narrow Q-value spike around the current policy action.
            noise = (
                torch.randn_like(act_b) * target_noise
            ).clamp(-target_noise_clip, target_noise_clip)

            next_action = (self._actor_tgt(next_b) + noise).clamp(-1.0, 1.0)

            # Fix 1: twin critics — take the minimum to reduce overestimation.
            ci_next = torch.cat([next_b, next_action], dim=-1)
            q1_tgt = self._c1_tgt(ci_next).squeeze(-1)
            q2_tgt = self._c2_tgt(ci_next).squeeze(-1)
            q_tgt  = rew_b + gammas_n * (1.0 - done_b) * torch.minimum(q1_tgt, q2_tgt)

        ci = torch.cat([obs_b, act_b], dim=-1)
        q1 = self._c1(ci).squeeze(-1)
        q2 = self._c2(ci).squeeze(-1)

        td_errors = ((q1 + q2) / 2.0 - q_tgt).detach().cpu().numpy()
        self._buffer.update_priorities(indices, np.abs(td_errors))

        # ---- Critic update (every step) ----
        c_loss = (weights * (
            F.mse_loss(q1, q_tgt, reduction="none") +
            F.mse_loss(q2, q_tgt, reduction="none")
        )).mean()

        self._critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._c1.parameters()) + list(self._c2.parameters()), 10.0
        )
        self._critic_opt.step()

        # ---- Actor update (Fix 2: delayed, every policy_delay steps) ----
        a_loss = 0.0
        if update_actor:
            # Deterministic policy gradient:
            # ∇_θ J = E[∇_a Q_φ1(s, a)|_{a=π_θ(s)} · ∇_θ π_θ(s)]
            # Implemented as -mean(Q1(s, π_θ(s))) and backpropagating through
            # the actor into Q1 (no entropy term — TD3 is fully deterministic).
            new_a  = self._actor(obs_b)
            ci_new = torch.cat([obs_b, new_a], dim=-1)
            a_loss = -self._c1(ci_new).mean()

            self._actor_opt.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 10.0)
            self._actor_opt.step()
            a_loss = float(a_loss.item())

            # Target networks only updated when actor is updated (Fix 2 coupling)
            self._soft_update(self._actor, self._actor_tgt, tau)
            self._c1_tgt_updated = True
        else:
            self._c1_tgt_updated = False

        # Critics' target networks always updated (decoupled from actor delay)
        self._soft_update(self._c1, self._c1_tgt, tau)
        self._soft_update(self._c2, self._c2_tgt, tau)

        return a_loss, float(c_loss.item())

    @staticmethod
    def _soft_update(source, target, tau: float) -> None:
        """θ_target ← τ·θ_source + (1−τ)·θ_target  (Polyak averaging)"""
        import torch
        with torch.no_grad():
            for sv, tv in zip(source.parameters(), target.parameters()):
                tv.data.copy_(tau * sv.data + (1.0 - tau) * tv.data)

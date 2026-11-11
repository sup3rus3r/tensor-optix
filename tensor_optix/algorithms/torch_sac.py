import os
import copy

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.resume_state import save_torch_resume_state, load_torch_resume_state
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer, episode_n_envs, window_transition_pairs
from tensor_optix.core.obs_utils import to_torch_batch, already_batched_to_torch


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

    PER and n-step params are exposed in hyperparams so SPSA can adapt them:
        per_alpha      — prioritization strength (0=uniform, 1=full PER)
        per_beta       — IS correction exponent
        n_step         — multi-step TD target length

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
                "learning_rate":    3e-4,
                "gamma":            0.99,
                "tau":              0.005,
                "batch_size":       256,
                "updates_per_step": 1,
                "replay_capacity":  1_000_000,
                "per_alpha":        0.0,
                "per_beta":         0.4,
                "n_step":           1,
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
        device: str = "auto",
    ):
        import torch
        self._torch    = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._actor    = actor.to(self._device)
        self._c1       = critic1.to(self._device)
        self._c2       = critic2.to(self._device)
        self._c1_tgt   = copy.deepcopy(critic1).to(self._device)
        self._c2_tgt   = copy.deepcopy(critic2).to(self._device)
        self._action_dim    = action_dim
        self._actor_opt     = actor_optimizer
        self._critic_opt    = critic_optimizer
        self._alpha_opt     = alpha_optimizer
        self._hyperparams   = hyperparams.copy()

        log_alpha_init = float(hyperparams.params.get("log_alpha_init", 0.0))
        self._log_alpha = torch.tensor(log_alpha_init, dtype=torch.float32,
                                       requires_grad=True, device=self._device)
        # Replace alpha_optimizer param group with our log_alpha tensor
        self._alpha_opt = type(alpha_optimizer)(
            [self._log_alpha], **{k: v for k, v in alpha_optimizer.defaults.items()}
        )
        self._target_entropy = -float(action_dim)

        hp = hyperparams.params
        capacity  = int(hp.get("replay_capacity", 1_000_000))
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

    default_param_bounds = {
        "learning_rate": (1e-4, 1e-3),
        "gamma":         (0.97, 0.999),
        "tau":           (1e-3, 1e-1),
        # lr upper bound 1e-3: gradient spikes at early training (large td_errors on random
        # Q-networks) cause instability at lr=3e-3 even with clip_by_norm.  1e-3 keeps the
        # maximum parameter step ≤ 0.01 (lr × clip_norm=10), same reasoning as DQN.
    }
    default_log_params = ["learning_rate", "tau"]

    # SAC has no epsilon decay but twin Q-functions need ~20 episodes to stabilize
    # their value estimates before DORMANT intervention is meaningful.
    default_min_episodes_before_dormant = 30

    @property
    def is_on_policy(self) -> bool:
        return False  # replay buffer — rollback without buffer clear is harmful

    def _encode(self, obs_tensor):
        """Run the actor's encoder alone, if it has one (EncodedHead for
        non-flat observations); identity pass-through for the flat-Box
        case, exactly matching today's behavior. Used to pre-encode
        observations before they enter the replay buffer -- see
        tensor_optix.encoders' module docstring for why SAC/TD3's critic
        only ever sees already-encoded flat vectors, never raw observations."""
        encoder = getattr(self._actor, "encoder", None)
        if encoder is None:
            return obs_tensor
        with self._torch.no_grad():
            return encoder(obs_tensor)

    def act(self, observation) -> np.ndarray:
        import torch
        obs = to_torch_batch(observation, self._device)
        with torch.no_grad():
            action, _ = self._sample_action(obs)
        return action.cpu().numpy()[0]

    def learn(self, episode_data: EpisodeData) -> dict:
        hp = self._hyperparams.params
        gamma            = float(hp.get("gamma",            0.99))
        tau              = float(hp.get("tau",              0.005))
        batch_size       = int(hp.get("batch_size",         256))
        updates_per_step = float(hp.get("updates_per_step", 1))

        obs_batched = to_torch_batch(list(episode_data.observations), self._device)
        obs_arr  = self._encode(obs_batched).cpu().numpy()
        act_arr  = np.array(episode_data.actions,      dtype=np.float32)
        rew_arr  = episode_data.rewards
        done_arr = episode_data.dones
        T = len(rew_arr)

        # Pair each step with the NEXT step of the same environment. With a
        # VectorBatchPipeline window the flat arrays interleave n_envs envs
        # ([t * n_envs + i]), so index t + 1 belongs to a different env.
        for env_pairs in window_transition_pairs(T, episode_n_envs(episode_data)):
            for t, t_next in env_pairs:
                self._buffer.push(obs_arr[t], act_arr[t], float(rew_arr[t]),
                                   obs_arr[t_next], float(done_arr[t]))
            self._buffer.flush_episode()

        if len(self._buffer) < batch_size:
            return {
                "actor_loss": 0.0, "critic_loss": 0.0,
                "alpha": float(self._log_alpha.exp().item()),
                "buffer_size": len(self._buffer),
            }

        # Truncate the product, not updates_per_step itself -- truncating
        # a fractional updates_per_step (e.g. 0.5) to int first would zero
        # it out regardless of T.
        n_updates = int(T * updates_per_step)
        total_al = 0.0
        total_cl = 0.0
        for _ in range(n_updates):
            al, cl = self._update_step(batch_size, gamma, tau)
            total_al += al
            total_cl += cl

        return {
            "actor_loss":  total_al / max(n_updates, 1),
            "critic_loss": total_cl / max(n_updates, 1),
            "alpha":       float(self._log_alpha.exp().item()),
            "buffer_size": len(self._buffer),
        }

    def get_hyperparams(self) -> HyperparamSet:
        self._hyperparams.params["learning_rate"] = float(
            self._actor_opt.param_groups[0]["lr"]
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
            for opt in (self._actor_opt, self._critic_opt, self._alpha_opt):
                for pg in opt.param_groups:
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
        torch.save(self._actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self._c1.state_dict(),    os.path.join(path, "critic1.pt"))
        torch.save(self._c2.state_dict(),    os.path.join(path, "critic2.pt"))
        np.save(os.path.join(path, "log_alpha.npy"), self._log_alpha.item())

    def load_weights(self, path: str) -> None:
        import torch
        self._actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"),   map_location=self._device))
        self._c1.load_state_dict(   torch.load(os.path.join(path, "critic1.pt"), map_location=self._device))
        self._c2.load_state_dict(   torch.load(os.path.join(path, "critic2.pt"), map_location=self._device))
        la_path = os.path.join(path, "log_alpha.npy")
        if os.path.exists(la_path):
            with torch.no_grad():
                self._log_alpha.fill_(float(np.load(la_path)))
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    @property
    def resumable_components(self) -> frozenset:
        # Deliberately does NOT include "replay_buffer" — PrioritizedReplayBuffer
        # has no state_dict()/load_state_dict(), so resuming from a saved
        # checkpoint restarts with an EMPTY buffer. For an off-policy algorithm
        # this is a real gap (see BaseAgent.is_on_policy docstring): the first
        # batch_size steps after resume have no replay data to sample from.
        # Claiming "replay_buffer" here would be dishonest about what's
        # actually restored — see resume_state.py's module docstring.
        return frozenset({"model", "optimizer", "rng"})

    def save_full_state(self, path: str) -> None:
        self.save_weights(path)
        save_torch_resume_state(
            path,
            actor_opt=self._actor_opt,
            critic_opt=self._critic_opt,
            alpha_opt=self._alpha_opt,
        )

    def load_full_state(self, path: str) -> None:
        self.load_weights(path)
        load_torch_resume_state(
            path,
            actor_opt=self._actor_opt,
            critic_opt=self._critic_opt,
            alpha_opt=self._alpha_opt,
        )

    def get_weights(self) -> dict:
        return {
            "actor":   {k: v.detach().clone() for k, v in self._actor.state_dict().items()},
            "critic1": {k: v.detach().clone() for k, v in self._c1.state_dict().items()},
            "critic2": {k: v.detach().clone() for k, v in self._c2.state_dict().items()},
        }

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
                net.load_state_dict({k: v.to(next(net.parameters()).dtype) for k, v in avg.items()})
            # Sync target networks from averaged online networks
            self._c1_tgt.load_state_dict(self._c1.state_dict())
            self._c2_tgt.load_state_dict(self._c2.state_dict())

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        # Perturb actor and online critics only. Target networks sync via
        # soft update (τ) during learning — don't perturb them directly.
        with torch.no_grad():
            for module in (self._actor, self._c1, self._c2):
                for param in module.parameters():
                    param.mul_(1.0 + noise_scale * torch.randn_like(param))
        # Sync targets to match perturbed critics so they don't diverge immediately.
        self._c1_tgt.load_state_dict(self._c1.state_dict())
        self._c2_tgt.load_state_dict(self._c2.state_dict())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def export_onnx(self, path: str) -> None:
        """
        Export the continuous stochastic actor to ONNX.

        Input  — "observation":  (batch_size, obs_dim)          float32
        Output — "mean_logstd":  (batch_size, 2 * action_dim)   float32

        The first ``action_dim`` elements are the Gaussian means (μ);
        the last ``action_dim`` elements are the log standard deviations (log σ).

        Deterministic deployment policy:   a = tanh(μ)
        Stochastic deployment policy:      a = tanh(μ + exp(log σ) · ε),  ε ~ N(0, I)

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
            output_names=["mean_logstd"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "mean_logstd": {0: "batch_size"},
            },
            opset_version=17,
        )
        self._actor.train(was_training).to(self._device)

    def teardown(self) -> None:
        """Move all networks to CPU and free CUDA memory."""
        import torch
        for module in (self._actor, self._c1, self._c2, self._c1_tgt, self._c2_tgt):
            module.cpu()
        torch.cuda.empty_cache()

    def _actor_head(self):
        """The actor's downstream head only, skipping its encoder (if any).
        Buffer-sampled data (obs_b/next_b in _update_step) is already a
        flat, pre-encoded vector by the time it reaches here -- see
        tensor_optix.encoders' module docstring -- so running it back
        through the actor's encoder (EncodedHead's full forward) a second
        time would be wrong (and, for a CNN/Dict encoder, shape-incompatible
        outright). For the flat-Box case (self._actor has no .head
        attribute, it's a plain nn.Sequential), the actor itself IS the
        head -- this falls back to calling it directly, identical to
        today's behavior."""
        return getattr(self._actor, "head", self._actor)

    def _sample_action(self, obs, pre_encoded: bool = False):
        """pre_encoded=True: obs is already a flat, encoder-output vector
        (buffer-sampled data) -- route through the actor's head only.
        pre_encoded=False (default): obs is a raw observation (from act())
        -- route through the full actor (encoder + head)."""
        import torch
        net = self._actor_head() if pre_encoded else self._actor
        out     = net(obs)
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

        obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = self._buffer.sample(batch_size)
        obs_b   = already_batched_to_torch(obs_b, self._device)
        act_b   = torch.as_tensor(act_b,    dtype=torch.float32).to(self._device)
        rew_b   = torch.as_tensor(rew_b,    dtype=torch.float32).to(self._device)
        next_b  = already_batched_to_torch(next_b, self._device)
        done_b  = torch.as_tensor(done_b,   dtype=torch.float32).to(self._device)
        weights = torch.as_tensor(weights,  dtype=torch.float32).to(self._device)
        gammas_n = torch.as_tensor(
            gamma ** n_steps.astype(np.float32), dtype=torch.float32
        ).to(self._device)
        alpha  = self._log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            na, nlp = self._sample_action(next_b, pre_encoded=True)
            ci_next = torch.cat([next_b, na], dim=-1)
            q1t = self._c1_tgt(ci_next).squeeze(-1)
            q2t = self._c2_tgt(ci_next).squeeze(-1)
            q_t = rew_b + gammas_n * (1 - done_b) * (torch.minimum(q1t, q2t) - alpha * nlp)

        ci = torch.cat([obs_b, act_b], dim=-1)
        q1 = self._c1(ci).squeeze(-1)
        q2 = self._c2(ci).squeeze(-1)

        td_errors = ((q1 + q2) / 2 - q_t).detach().cpu().numpy()
        self._buffer.update_priorities(indices, np.abs(td_errors))

        c_loss = (weights * (F.mse_loss(q1, q_t, reduction="none") +
                             F.mse_loss(q2, q_t, reduction="none"))).mean()
        self._critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._c1.parameters()) + list(self._c2.parameters()), 10.0
        )
        self._critic_opt.step()

        # Actor update
        new_a, new_lp = self._sample_action(obs_b, pre_encoded=True)
        ci_new = torch.cat([obs_b, new_a], dim=-1)
        q1n = self._c1(ci_new).squeeze(-1)
        q2n = self._c2(ci_new).squeeze(-1)
        a_loss = (alpha * new_lp - torch.minimum(q1n, q2n)).mean()
        self._actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 10.0)
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

"""
TorchRecurrentPPOAgent — PPO with LSTM/GRU policy for POMDPs.

Mathematical motivation
-----------------------
Standard PPO assumes the Markov property:
    P(s_{t+1} | s_0:t, a_0:t) = P(s_{t+1} | s_t, a_t)

In a POMDP the agent sees observation o_t ≠ s_t.  The belief state
b_t = P(s_t | o_1:t, a_0:t-1) is the sufficient statistic for optimal
action selection, but is intractable.  An LSTM approximates it:

    h_t = LSTM(h_{t-1}, o_t)
    π(a_t | o_1:t) ≈ π_θ(a_t | h_t)
    V(o_1:t)       ≈ V_φ(h_t)

For MDP environments the LSTM degenerates to a feedforward network
(gradient forces h_t ≈ f(o_t)), so the recurrent agent strictly
subsumes the feedforward agent.

BPTT correctness
----------------
PPO re-uses a rollout for n_epochs.  During each epoch the LSTM is
re-run from scratch on sequences of length bptt_len.  The initial
hidden state of each sequence is the stored h_start from rollout
(detached — not re-computed).  Episode boundaries (done=True) reset
the hidden state to zeros inside the sequence.

The clipped surrogate objective over a BPTT window of length T:

    L_clip = (1/T) Σ_t min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)

    r_t = π_θ(a_t | h_t^{new}) / π_θ_old(a_t | h_t^{old})

h_t^{old} is from rollout (detached).  h_t^{new} is re-computed with
gradients flowing back at most bptt_len steps.

Architecture
------------
    rnn:         nn.LSTM or nn.GRU  (obs_dim → hidden_size)
    actor_head:  nn.Linear(hidden_size → n_actions)
    critic_head: nn.Linear(hidden_size → 1)

Usage::

    import torch.nn as nn
    from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent

    rnn         = nn.LSTM(obs_dim, 128, batch_first=True)
    actor_head  = nn.Linear(128, n_actions)
    critic_head = nn.Linear(128, 1)
    agent = TorchRecurrentPPOAgent(
        rnn=rnn, actor_head=actor_head, critic_head=critic_head,
        n_actions=n_actions,
        optimizer=torch.optim.Adam(
            list(rnn.parameters()) +
            list(actor_head.parameters()) +
            list(critic_head.parameters()),
            lr=3e-4,
        ),
        hyperparams=HyperparamSet(params={
            "learning_rate":  3e-4,
            "clip_ratio":     0.2,
            "entropy_coef":   0.01,
            "vf_coef":        0.5,
            "gamma":          0.99,
            "gae_lambda":     0.95,
            "n_epochs":       4,
            "bptt_len":       16,
            "max_grad_norm":  0.5,
        }, episode_id=0),
    )
"""

import os
from typing import Optional, Tuple

import numpy as np

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae


class TorchRecurrentPPOAgent(BaseAgent):
    """PPO with LSTM/GRU policy for partially-observable environments."""

    default_param_bounds = {
        "learning_rate": (1e-4, 3e-3),
        "gamma":         (0.95, 0.999),
        "clip_ratio":    (0.1,  0.3),
        "entropy_coef":  (0.001, 0.05),
    }
    default_log_params = ["learning_rate"]
    default_min_episodes_before_dormant = 20

    def __init__(
        self,
        rnn,                      # nn.LSTM or nn.GRU (batch_first=True)
        actor_head,               # nn.Linear(hidden_size → n_actions)
        critic_head,              # nn.Linear(hidden_size → 1)
        n_actions: int,
        optimizer,
        hyperparams: HyperparamSet,
        device: str = "auto",
    ):
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._rnn         = rnn.to(self._device)
        self._actor_head  = actor_head.to(self._device)
        self._critic_head = critic_head.to(self._device)
        self._n_actions   = n_actions
        self._optimizer   = optimizer
        self._hyperparams = hyperparams.copy()

        # Detect RNN type (LSTM has (h, c), GRU has h only)
        self._is_lstm = hasattr(rnn, "hidden_size") and hasattr(rnn, "proj_size")
        try:
            import torch.nn as nn
            self._is_lstm = isinstance(rnn, nn.LSTM)
        except ImportError:
            pass

        # Per-episode rollout caches
        self._cache_obs:    list = []
        self._cache_lp:     list = []
        self._cache_values: list = []
        self._cache_hidden: list = []   # h_t BEFORE step t (what was passed in)

        # Running hidden state (updated by act(), reset at episode start)
        self._h = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        return True

    def reset_hidden(self) -> None:
        """Reset hidden state to zeros.  Call at episode start."""
        self._h = None

    def act(self, observation) -> int:
        """
        Sample action; update and cache hidden state.

        The hidden state h_t-1 is cached BEFORE the LSTM step so learn()
        can re-initialise each BPTT chunk from the correct starting h.
        """
        import torch
        import torch.nn.functional as F

        obs = torch.as_tensor(
            np.atleast_2d(observation), dtype=torch.float32
        ).unsqueeze(0).to(self._device)   # [1, 1, obs_dim]

        with torch.no_grad():
            # Cache the hidden state that will be passed to the LSTM
            # (h BEFORE this step — used by learn() to re-init BPTT chunks)
            if self._h is None:
                h_cached = self._zero_hidden()
            else:
                h_cached = self._detach_hidden(self._h)
            self._cache_hidden.append(self._hidden_to_numpy(h_cached))

            out, new_h = self._rnn(obs, self._h)   # out: [1, 1, hidden_size]
            feat = out.squeeze(0)                   # [1, hidden_size]

            logits = self._actor_head(feat)         # [1, n_actions]
            value  = self._critic_head(feat).squeeze()

            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            lp     = dist.log_prob(action)

        self._cache_obs.append(np.squeeze(np.atleast_1d(observation)))
        self._cache_lp.append(float(lp.item()))
        self._cache_values.append(float(value.item()))

        self._h = new_h
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
        n_epochs      = int(hp.get("n_epochs",         4))
        bptt_len      = int(hp.get("bptt_len",         16))
        max_grad_norm = float(hp.get("max_grad_norm",  0.5))

        T = len(episode_data.rewards)
        if len(self._cache_obs) < T:
            raise RuntimeError(
                f"Hidden state cache underflow: expected >= {T}, "
                f"got {len(self._cache_obs)}. Call act() exactly once per step."
            )

        # Pull rollout data from caches
        obs_arr    = np.array(self._cache_obs[:T],     dtype=np.float32)
        old_lp_arr = np.array(self._cache_lp[:T],      dtype=np.float32)
        val_arr    = np.array(self._cache_values[:T],  dtype=np.float32)
        hidden_arr = self._cache_hidden[:T]             # list of T numpy hidden states
        rewards    = list(episode_data.rewards)
        dones      = episode_data.dones

        # Clear caches (episode is done)
        self._cache_obs    = self._cache_obs[T:]
        self._cache_lp     = self._cache_lp[T:]
        self._cache_values = self._cache_values[T:]
        self._cache_hidden = self._cache_hidden[T:]
        self._h = None   # reset for next episode

        # GAE advantage estimation
        advantages, returns = compute_gae(rewards, val_arr, dones, gamma, gae_lambda, 0.0)
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensors
        obs_t    = torch.as_tensor(obs_arr,    dtype=torch.float32).to(self._device)
        acts_t   = torch.as_tensor(
            np.array(episode_data.actions, dtype=np.int64)[:T], dtype=torch.long
        ).to(self._device)
        old_lp_t = torch.as_tensor(old_lp_arr, dtype=torch.float32).to(self._device)
        adv_t    = torch.as_tensor(advantages, dtype=torch.float32).to(self._device)
        ret_t    = torch.as_tensor(returns,    dtype=torch.float32).to(self._device)
        dones_t  = torch.as_tensor(
            np.array(dones, dtype=np.float32), dtype=torch.float32
        ).to(self._device)

        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy  = 0.0
        total_kl       = 0.0
        n_updates      = 0

        for _ in range(n_epochs):
            # Iterate over BPTT chunks
            chunk_starts = range(0, T, bptt_len)
            for c_start in chunk_starts:
                c_end = min(c_start + bptt_len, T)

                # Retrieve the stored initial hidden state for this chunk
                h_init = self._numpy_to_hidden(hidden_arr[c_start])

                # Slice this chunk
                obs_c  = obs_t[c_start:c_end].unsqueeze(0)    # [1, L, obs_dim]
                acts_c = acts_t[c_start:c_end]
                old_lp_c = old_lp_t[c_start:c_end]
                adv_c  = adv_t[c_start:c_end]
                ret_c  = ret_t[c_start:c_end]
                done_c = dones_t[c_start:c_end]                # [L]

                # Re-run LSTM on the chunk with gradient flow
                # Apply episode-boundary resets inside the chunk
                feats = self._run_rnn_chunk(obs_c, h_init, done_c)  # [L, hidden]

                logits = self._actor_head(feats)             # [L, n_actions]
                values = self._critic_head(feats).squeeze(-1) # [L]

                dist    = torch.distributions.Categorical(logits=logits)
                new_lp  = dist.log_prob(acts_c)
                entropy = dist.entropy().mean()

                ratio    = torch.exp(new_lp - old_lp_c)
                surr1    = ratio * adv_c
                surr2    = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_c
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(values, ret_c)
                loss     = pol_loss + vf_coef * val_loss - entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self._rnn.parameters()) +
                    list(self._actor_head.parameters()) +
                    list(self._critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self._optimizer.step()

                with torch.no_grad():
                    kl = (old_lp_c - new_lp).mean()

                total_pol_loss += float(pol_loss.item())
                total_val_loss += float(val_loss.item())
                total_entropy  += float(entropy.item())
                total_kl       += float(kl.item())
                n_updates      += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss":  total_pol_loss / denom,
            "value_loss":   total_val_loss / denom,
            "entropy":      total_entropy  / denom,
            "approx_kl":    total_kl       / denom,
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
        torch.save(self._rnn.state_dict(),         os.path.join(path, "rnn.pt"))
        torch.save(self._actor_head.state_dict(),  os.path.join(path, "actor_head.pt"))
        torch.save(self._critic_head.state_dict(), os.path.join(path, "critic_head.pt"))

    def load_weights(self, path: str) -> None:
        import torch
        self._rnn.load_state_dict(
            torch.load(os.path.join(path, "rnn.pt"),         map_location=self._device))
        self._actor_head.load_state_dict(
            torch.load(os.path.join(path, "actor_head.pt"),  map_location=self._device))
        self._critic_head.load_state_dict(
            torch.load(os.path.join(path, "critic_head.pt"), map_location=self._device))

    def average_weights(self, paths: list) -> None:
        import torch
        nets = [
            (self._rnn,         "rnn.pt"),
            (self._actor_head,  "actor_head.pt"),
            (self._critic_head, "critic_head.pt"),
        ]
        with torch.no_grad():
            for net, fname in nets:
                avg = None
                for p in paths:
                    sd = torch.load(os.path.join(p, fname), map_location=self._device)
                    if avg is None:
                        avg = {k: v.clone().float() for k, v in sd.items()}
                    else:
                        for k in avg:
                            avg[k] += sd[k].float()
                for k in avg:
                    avg[k] /= len(paths)
                net.load_state_dict({
                    k: v.to(next(net.parameters()).dtype) for k, v in avg.items()
                })

    def perturb_weights(self, noise_scale: float) -> None:
        import torch
        with torch.no_grad():
            for module in (self._rnn, self._actor_head, self._critic_head):
                for p in module.parameters():
                    p.mul_(1.0 + noise_scale * torch.randn_like(p))

    def export_onnx(self, path: str) -> None:
        """
        Export the recurrent actor (LSTM + actor head) to ONNX.

        Inputs
        ------
        "observation" : (1, 1, obs_dim)              float32   — single step
        "h_0"         : (num_layers, 1, hidden_size) float32   — LSTM h state
        "c_0"         : (num_layers, 1, hidden_size) float32   — LSTM c state

        Outputs
        -------
        "logits"      : (1, n_actions)               float32
        "h_n"         : (num_layers, 1, hidden_size) float32   — updated h state
        "c_n"         : (num_layers, 1, hidden_size) float32   — updated c state

        Feed h_n / c_n back as h_0 / c_0 on the next call to maintain
        episode continuity.  Reset to zeros at each episode boundary.

        Requires the ``onnx`` optional dependency:
            pip install tensor-optix[onnx]
        """
        import torch
        import torch.nn as nn

        class _RecurrentActorWrapper(nn.Module):
            def __init__(self, rnn, actor_head):
                super().__init__()
                self.rnn        = rnn
                self.actor_head = actor_head

            def forward(self, obs, h_0, c_0):
                # obs: (1, 1, obs_dim); h_0, c_0: (num_layers, 1, hidden_size)
                out, (h_n, c_n) = self.rnn(obs, (h_0, c_0))
                feat   = out.squeeze(1)          # (1, hidden_size)
                logits = self.actor_head(feat)   # (1, n_actions)
                return logits, h_n, c_n

        obs_dim     = self._rnn.input_size
        hidden_size = self._rnn.hidden_size
        num_layers  = self._rnn.num_layers

        was_rnn_train  = self._rnn.training
        was_head_train = self._actor_head.training
        self._rnn.eval().cpu()
        self._actor_head.eval().cpu()

        wrapper = _RecurrentActorWrapper(self._rnn, self._actor_head)

        dummy_obs = torch.zeros(1, 1, obs_dim,     dtype=torch.float32)
        dummy_h   = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
        dummy_c   = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_h, dummy_c),
            str(path),
            input_names=["observation", "h_0", "c_0"],
            output_names=["logits", "h_n", "c_n"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "h_0":         {1: "batch_size"},
                "c_0":         {1: "batch_size"},
                "logits":      {0: "batch_size"},
                "h_n":         {1: "batch_size"},
                "c_n":         {1: "batch_size"},
            },
            opset_version=17,
        )
        self._rnn.train(was_rnn_train).to(self._device)
        self._actor_head.train(was_head_train).to(self._device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_rnn_chunk(self, obs_seq, h_init, done_seq):
        """
        Run the RNN over obs_seq [1, L, obs_dim] with episode-boundary resets.

        When done[t-1] is True, reset hidden to zeros before step t.
        Returns features [L, hidden_size].
        """
        import torch
        L = obs_seq.shape[1]
        feats = []
        h = h_init

        for t in range(L):
            # Reset hidden at episode boundaries (after a done at previous step)
            if t > 0 and float(done_seq[t - 1].item()) > 0.5:
                h = self._zero_hidden()

            obs_t = obs_seq[:, t:t+1, :]          # [1, 1, obs_dim]
            out, h = self._rnn(obs_t, h)           # out: [1, 1, hidden_size]
            feats.append(out.squeeze(0).squeeze(0))  # [hidden_size]

        return torch.stack(feats, dim=0)           # [L, hidden_size]

    def _zero_hidden(self):
        """Return a zero initial hidden state on the correct device."""
        import torch
        num_layers  = self._rnn.num_layers
        hidden_size = self._rnn.hidden_size
        h = torch.zeros(num_layers, 1, hidden_size, device=self._device)
        if self._is_lstm:
            c = torch.zeros(num_layers, 1, hidden_size, device=self._device)
            return (h, c)
        return h

    def _detach_hidden(self, h):
        """Detach hidden state from computation graph."""
        if self._is_lstm:
            return (h[0].detach(), h[1].detach())
        return h.detach()

    def _hidden_to_numpy(self, h) -> tuple:
        """Convert hidden state tensor(s) to numpy for caching."""
        if self._is_lstm:
            return (h[0].cpu().numpy(), h[1].cpu().numpy())
        return (h.cpu().numpy(),)

    def _numpy_to_hidden(self, h_np: tuple):
        """Restore hidden state tensor(s) from cached numpy arrays."""
        import torch
        if self._is_lstm:
            h = torch.as_tensor(h_np[0], dtype=torch.float32).to(self._device)
            c = torch.as_tensor(h_np[1], dtype=torch.float32).to(self._device)
            return (h, c)
        h = torch.as_tensor(h_np[0], dtype=torch.float32).to(self._device)
        return h

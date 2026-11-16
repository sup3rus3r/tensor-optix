from __future__ import annotations

"""
GraphAgent — a BaseAgent backed by a free-form NeuronGraph.

Weight learning is PPO-style: the graph acts as both actor and critic.
The actor head reads output neurons [0:-1], the critic head reads the last
output neuron. For discrete actions the actor outputs logits; for continuous
actions it outputs means (std is a learned parameter).

The agent is intentionally minimal — it wires the NeuronGraph into the
tensor-optix contract. Topology mutations are performed externally by
TopologyController, which holds a reference to the graph.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet
from tensor_optix.core.trajectory_buffer import compute_gae
from tensor_optix.core.device import get_device

from ..graph.neuron_graph import NeuronGraph
from ..graph.topology_ops import add_input_neuron
from ..optimizer.topology_aware_adam import TopologyAwareAdam


class GraphAgent(BaseAgent):
    """
    RL agent whose policy network is a mutable NeuronGraph.

    Parameters
    ----------
    graph:
        A NeuronGraph that has already been configured with input/output neurons.
        The last output neuron is the value head; all others are action logits
        (discrete) or action means (continuous).
    obs_dim:
        Expected observation dimensionality. If a new obs arrives with more
        dimensions, the agent automatically grows new input neurons.
    n_actions:
        Number of discrete actions, or dimension of continuous action space.
    continuous:
        If True, actions are sampled from a Gaussian. If False, from Categorical.
    hyperparams:
        Initial HyperparamSet. Keys used: learning_rate, clip_ratio,
        entropy_coef, vf_coef, gamma, gae_lambda, n_epochs, minibatch_size,
        max_grad_norm.
    """

    is_on_policy = True

    default_hyperparams = {
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       4,
        "minibatch_size": 64,
        "max_grad_norm":  0.5,
    }

    def __init__(
        self,
        graph: NeuronGraph,
        obs_dim: int,
        n_actions: int,
        continuous: bool = False,
        hyperparams: Optional[HyperparamSet] = None,
        device=None,
    ) -> None:
        self.graph = graph
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.continuous = continuous
        self.device = torch.device(device) if device is not None else get_device()

        self.graph.to(self.device)
        # Force the packed-parameter rebuild now, before we capture
        # graph.parameters() below — otherwise the optimizer would grab
        # individual per-edge Parameters that the first forward pass then
        # discards in favor of a combined _ff_weight_vec.
        self.graph.ensure_built()

        if continuous:
            # Learnable log-std, one per action dimension
            self.log_std = nn.Parameter(torch.zeros(n_actions, device=self.device))

        params = list(self.graph.parameters())
        if continuous:
            params.append(self.log_std)

        _hp = dict(self.default_hyperparams)
        if hyperparams is not None:
            _hp.update(hyperparams.params)
        self._hyperparams = HyperparamSet(params=_hp, episode_id=0)

        self.optimizer = TopologyAwareAdam(params, lr=_hp["learning_rate"])
        self._graph_struct_version = self.graph._struct_version
        self._opt_ff_vec = self.graph._ff_weight_vec
        self._opt_ff_ids = self.graph._ff_edge_ids
        self._episode_count = 0

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    def act(self, observation) -> any:
        """
        Given a numpy observation, return an action (and store log_prob).
        Dynamically grows input neurons if obs_dim has expanded.
        """
        obs = self._to_tensor(observation)
        self._maybe_grow_inputs(obs)
        self._sync_optimizer()

        self.graph.reset_state()
        with torch.no_grad():
            out = self.graph(obs)

        action, log_prob = self._sample_action(out)
        self._last_log_prob = log_prob
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        self._sync_optimizer()
        hp = self._hyperparams.params
        obs_t = torch.tensor(
            np.array(episode_data.observations), dtype=torch.float32, device=self.device
        )
        act_t = torch.tensor(
            np.array(episode_data.actions), dtype=torch.long if not self.continuous else torch.float32,
            device=self.device,
        )
        rew_t = torch.tensor(episode_data.rewards, dtype=torch.float32, device=self.device)
        # log pi_old(a_t|s_t). Pipelines do not record log-probs, so when the
        # window carries none, evaluate the actions under the current policy
        # before any update: no gradient step has happened since collection,
        # so this IS the behaviour policy. (Falling back to zeros would turn
        # the PPO ratio exp(new - old) into the raw action probability.)
        if episode_data.log_probs:
            old_lp = torch.tensor(episode_data.log_probs, dtype=torch.float32, device=self.device)
        else:
            with torch.no_grad():
                old_lp, _, _ = self._evaluate_actions(self._batch_forward(obs_t), act_t)

        # Compute values and advantages
        with torch.no_grad():
            values = self._batch_values(obs_t)
        advantages, returns = compute_gae(
            rewards=rew_t.cpu().numpy().tolist(),
            values=values.cpu().numpy().tolist(),
            dones=episode_data.dones,
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            last_value=self._bootstrap_value(episode_data),
        )
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = obs_t.shape[0]
        total_loss = total_pg = total_vf = total_ent = 0.0
        n_updates = 0

        for _ in range(hp["n_epochs"]):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, hp["minibatch_size"]):
                idx = perm[start : start + hp["minibatch_size"]]
                if idx.numel() == 0:
                    continue
                ob = obs_t[idx]
                ac = act_t[idx]
                old_lp_b = old_lp[idx]
                adv_b = adv_t[idx]
                ret_b = ret_t[idx]

                out_b = self._batch_forward(ob)
                new_lp, entropy, values_b = self._evaluate_actions(out_b, ac)

                ratio = torch.exp(new_lp - old_lp_b)
                pg1 = ratio * adv_b
                pg2 = torch.clamp(ratio, 1 - hp["clip_ratio"], 1 + hp["clip_ratio"]) * adv_b
                pg_loss = -torch.min(pg1, pg2).mean()

                vf_loss = F.mse_loss(values_b, ret_b)
                ent_loss = -entropy.mean()

                loss = pg_loss + hp["vf_coef"] * vf_loss + hp["entropy_coef"] * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.graph.parameters(), hp["max_grad_norm"])
                self.optimizer.step()
                self.graph.enforce_dale()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                n_updates += 1

        self._episode_count += 1
        denom = max(n_updates, 1)
        return {
            "loss": total_loss / denom,
            "pg_loss": total_pg / denom,
            "vf_loss": total_vf / denom,
            "entropy": -total_ent / denom,
            "n_neurons": self.graph.n_neurons(),
            "n_edges": self.graph.n_edges(),
        }

    def get_hyperparams(self) -> HyperparamSet:
        return self._hyperparams

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams
        for pg in self.optimizer.param_groups:
            pg["lr"] = hyperparams.params.get("learning_rate", pg["lr"])

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "topology": self.graph.to_dict(include_weights=False),
            "graph":    self.graph.state_dict(),
            "episode":  self._episode_count,
        }
        if self.continuous:
            state["log_std"] = self.log_std.data
        torch.save(state, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "topology" in state:
            # Reconstruct topology so state_dict keys match exactly — handles
            # evolved graphs where neurons were added or pruned since construction.
            # Rebuilt IN PLACE: TopologyController / HebbianHook hold references
            # to this graph object and must keep following it after a restore.
            self.graph.restore_from_dict(state["topology"])
            self.graph.to(self.device)
        self.graph.load_state_dict(state["graph"])
        self._episode_count = state.get("episode", 0)
        if self.continuous and "log_std" in state:
            self.log_std.data.copy_(state["log_std"])
        # self.graph may now be a different object entirely (topology branch
        # above), so force a resync regardless of version-number coincidence.
        self._graph_struct_version = -1
        self._sync_optimizer(carry_state=False)   # weights jumped; old momentum is meaningless

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "GraphAgent":
        """
        Load a fully-evolved GraphAgent from a checkpoint saved by save_weights().

        Unlike load_weights(), this does not require a pre-constructed agent —
        it reconstructs the topology from the checkpoint and returns a ready
        agent.

            agent = GraphAgent.from_checkpoint("checkpoints/best.pt")
            agent.act(obs)
        """
        from ..graph.neuron_graph import NeuronGraph

        state = torch.load(path, map_location="cpu", weights_only=False)
        if "topology" not in state:
            raise ValueError(
                f"Checkpoint at '{path}' has no topology key. "
                "It was saved before topology-aware checkpointing was introduced. "
                "Construct the agent manually and call load_weights() instead."
            )
        graph = NeuronGraph.from_dict(state["topology"])
        n_inputs   = len(graph.input_ids)
        n_outputs  = len(graph.output_ids)
        continuous = "log_std" in state
        # Last output neuron is always the value head; remaining are action outputs.
        n_actions  = n_outputs - 1

        _device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        agent = cls(
            graph=graph,
            obs_dim=n_inputs,
            n_actions=n_actions,
            continuous=continuous,
            device=_device,
        )
        agent.graph.load_state_dict(state["graph"])
        agent._episode_count = state.get("episode", 0)
        if continuous and "log_std" in state:
            agent.log_std.data.copy_(state["log_std"].to(agent.device))
        return agent

    def perturb_weights(self, noise_scale: float) -> None:
        with torch.no_grad():
            for p in self.graph.parameters():
                p.mul_(1 + noise_scale * torch.randn_like(p))

    def average_weights(self, paths: list) -> None:
        if not paths:
            return
        states = [torch.load(p, map_location=self.device)["graph"] for p in paths]
        avg = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
        self.graph.load_state_dict(avg)

    def teardown(self) -> None:
        self.graph.cpu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.float().to(self.device)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).flatten()

    def _maybe_grow_inputs(self, obs: torch.Tensor) -> None:
        current_inputs = len(self.graph.input_ids)
        needed = obs.shape[0]
        if needed > current_inputs:
            for _ in range(needed - current_inputs):
                add_input_neuron(self.graph, activation="linear")
            self.obs_dim = needed
            # Optimizer resync happens in _sync_optimizer(), called right
            # after this in act()/learn() — rebuilding here would be premature
            # since the graph is still dirty and hasn't repacked its weights yet.

    def _sync_optimizer(self, carry_state: bool = True) -> None:
        """
        Re-fetch graph.parameters() into the optimizer whenever the graph's
        packed weight structure has been rebuilt since the optimizer was last
        (re)built — e.g. after TopologyController grow/prune/merge, after
        add_input_neuron, or after any other structural mutation. Cheap no-op
        when nothing has changed.

        A rebuild replaces the packed edge-weight vector with a new Parameter,
        so a fresh optimizer is needed, but Adam's momentum is carried over:
        per edge (matched by edge id) for the packed vector, and as-is for
        every Parameter that survived (neuron biases and gates, log_std,
        delayed-edge weights). Edges that are new start from zero moments.
        carry_state=False starts from scratch (used after a checkpoint load).
        """
        self.graph.ensure_built()
        if self.graph._struct_version == self._graph_struct_version:
            return
        params = list(self.graph.parameters())
        if self.continuous:
            params.append(self.log_std)
        lr = self.optimizer.param_groups[0]["lr"]
        new_opt = TopologyAwareAdam(params, lr=lr)
        if carry_state:
            new_opt.carry_over(
                self.optimizer,
                old_packed=self._opt_ff_vec, old_ids=self._opt_ff_ids,
                new_packed=self.graph._ff_weight_vec, new_ids=self.graph._ff_edge_ids,
            )
        self.optimizer = new_opt
        self._graph_struct_version = self.graph._struct_version
        self._opt_ff_vec = self.graph._ff_weight_vec
        self._opt_ff_ids = self.graph._ff_edge_ids

    def _batch_forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Batch forward over the graph from a zero state, which is exactly what
        act() computes (it resets the graph before every step). Runs through
        NeuronGraph.forward_batch, so it handles every neuron type and reads
        the packed weight vector, which is where edge gradients accumulate.
        """
        n_in = len(self.graph.input_ids)
        if obs_batch.shape[1] > n_in:
            obs_batch = obs_batch[:, :n_in]
        out, _ = self.graph.forward_batch(obs_batch)
        return out

    def _bootstrap_value(self, episode_data: EpisodeData) -> float:
        """V(s_T) for a window that ended mid-episode; 0.0 when it ended on a
        terminal step or the pipeline supplied no single final observation."""
        final_obs = getattr(episode_data, "final_obs", None)
        dones = episode_data.dones
        if final_obs is None or isinstance(final_obs, (list, tuple)) or not len(dones) or dones[-1]:
            return 0.0
        with torch.no_grad():
            obs = self._to_tensor(final_obs).unsqueeze(0)
            if obs.shape[1] != len(self.graph.input_ids):
                return 0.0
            return float(self._batch_values(obs)[0].item())

    def _batch_values(self, obs_batch: torch.Tensor) -> torch.Tensor:
        out = self._batch_forward(obs_batch)
        return out[:, -1]  # last output neuron = value

    def _sample_action(self, out: torch.Tensor):
        logits_or_means = out[:-1]  # all but last = actor
        if self.continuous:
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(logits_or_means, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            return action.cpu().numpy(), log_prob
        else:
            dist = Categorical(logits=logits_or_means)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    def _evaluate_actions(self, out_batch: torch.Tensor, actions: torch.Tensor):
        logits_or_means = out_batch[:, :-1]
        values = out_batch[:, -1]
        if self.continuous:
            std = self.log_std.exp().clamp(1e-4, 2.0)
            dist = Normal(logits_or_means, std.unsqueeze(0).expand_as(logits_or_means))
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=logits_or_means)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        return log_prob, entropy, values


# ──────────────────────────────────────────────────────────────────────────────
# RecurrentGraphAgent
# ──────────────────────────────────────────────────────────────────────────────

class RecurrentGraphAgent(GraphAgent):
    """
    GraphAgent whose policy keeps memory across timesteps: GRU/LSTM hidden
    and cell states and delayed-edge history carry over from one act() call
    to the next, and learn() trains that same recurrent policy by truncated
    BPTT.

    act() and learn() compute the same function. act() keeps a GraphState
    and steps it with NeuronGraph.forward_batch; the state resets at episode
    start when the pipeline calls reset_hidden() (BatchPipeline does, as for
    TorchRecurrentPPOAgent). learn() replays the window from the state act()
    started it with, resetting wherever act() did, so the log-probabilities it
    computes before updating are exactly the behaviour policy's. Each chunk of
    chunk_len steps then trains from the state act() had at the chunk's start.

    If learn() receives a window that act() did not produce (e.g. data from
    elsewhere), it starts from zeros and resets after every done.

    When the graph holds no state at all (only point neurons, no delayed
    edges) the recurrent and stateless policies are the same function, and
    learn() uses GraphAgent's shuffled-minibatch PPO.

    Extra hyperparams:
      chunk_len (int, default 64): BPTT truncation length in timesteps.
    """

    default_hyperparams = {
        **GraphAgent.default_hyperparams,
        "chunk_len": 64,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state = None              # GraphState act() is carrying
        self._state_version = -1        # graph._struct_version the state belongs to
        self._window_start = None       # state before the first act() of the window
        self._reset_flags: list = []    # per act(): did the state start from zero?

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """Zero the recurrent state. Pipelines call this at episode start."""
        self._state = None

    def act(self, observation) -> any:
        obs = self._to_tensor(observation)
        self._maybe_grow_inputs(obs)
        self._sync_optimizer()
        self.graph.ensure_built()
        if self._state is not None and self._state_version != self.graph._struct_version:
            self._state = None          # topology changed: old state has the wrong layout
        if not self._reset_flags:
            self._window_start = self._state
        self._reset_flags.append(self._state is None)
        with torch.no_grad():
            out, self._state = self.graph.forward_batch(obs.unsqueeze(0), self._state)
        self._state_version = self.graph._struct_version
        action, log_prob = self._sample_action(out[0])
        self._last_log_prob = log_prob
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        if not self._graph_has_state():
            self._consume_window(len(episode_data.rewards), None)
            return super().learn(episode_data)
        return self._recurrent_learn(episode_data)

    # ------------------------------------------------------------------
    # Sequential / recurrent forward
    # ------------------------------------------------------------------

    def recurrent_forward(self, obs_sequence: torch.Tensor, chunk_len: int = 64) -> torch.Tensor:
        """
        Run T timesteps in order from a zero state, carrying the graph's
        recurrent state (GRU/LSTM hidden and cell states, delayed-edge history)
        from step to step, with truncated BPTT every chunk_len steps.

        This is NeuronGraph.forward_batch unrolled, so it computes exactly what
        forward() computes step by step: delayed edges read the source's
        activation from `delay` steps earlier.

        Returns [T, n_outputs].
        """
        T = obs_sequence.shape[0]
        n_in = len(self.graph.input_ids)
        if obs_sequence.shape[1] > n_in:
            obs_sequence = obs_sequence[:, :n_in]
        state = None
        outputs = []
        for t in range(T):
            if t > 0 and t % chunk_len == 0:
                state = state.detach()
            out, state = self.graph.forward_batch(obs_sequence[t:t + 1], state)
            outputs.append(out[0])
        return torch.stack(outputs)  # [T, n_outputs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph_has_state(self) -> bool:
        """True if anything carries information across timesteps: a GRU/LSTM
        neuron (any variant) or a delayed edge."""
        from ..graph.neuron import Neuron
        self.graph.ensure_built()
        if self.graph._max_rec_delay > 0:
            return True
        return any(type(n) is not Neuron for n in self.graph._neurons.values())

    def _window_plan(self, T: int, dones):
        """(reset flags, start state) for a window of T steps: what act() did
        if act() produced this window, else zeros with a reset after each done."""
        if len(self._reset_flags) >= T:
            start = self._window_start
            if start is not None and self._state_version != self.graph._struct_version:
                start = None
            return list(self._reset_flags[:T]), start
        return [t == 0 or bool(dones[t - 1]) for t in range(T)], None

    def _consume_window(self, T: int, state_after) -> None:
        """Drop the act() bookkeeping for the T steps just learned from."""
        self._reset_flags = self._reset_flags[T:]
        self._window_start = state_after if self._reset_flags else None

    def _replay(self, obs_t, flags, start, begin, end, chunk_len, chunk_starts=None):
        """Run steps [begin, end) of the window through forward_batch, starting
        from `start`, zeroing the state wherever flags says act() did, and
        detaching it every chunk_len steps. Returns (outputs [end-begin, n_out],
        state after the last step). Records the state at every chunk start in
        chunk_starts when given."""
        state = start
        outs = []
        for t in range(begin, end):
            if flags[t]:
                state = None
            elif t > begin and (t - begin) % chunk_len == 0 and state is not None:
                state = state.detach()
            if chunk_starts is not None and (t - begin) % chunk_len == 0:
                chunk_starts.append(None if state is None else state.detach())
            out, state = self.graph.forward_batch(obs_t[t:t + 1], state)
            outs.append(out[0])
        return torch.stack(outs), state

    def _recurrent_learn(self, episode_data: EpisodeData) -> dict:
        self._sync_optimizer()
        hp = self._hyperparams.params
        chunk_len = int(hp.get("chunk_len", 64))

        obs_t = torch.tensor(
            np.array(episode_data.observations), dtype=torch.float32, device=self.device
        )
        n_in = len(self.graph.input_ids)
        if obs_t.shape[1] > n_in:
            obs_t = obs_t[:, :n_in]
        act_t = torch.tensor(
            np.array(episode_data.actions),
            dtype=torch.long if not self.continuous else torch.float32,
            device=self.device,
        )
        rew_t = torch.tensor(episode_data.rewards, dtype=torch.float32, device=self.device)
        T = obs_t.shape[0]
        flags, start = self._window_plan(T, episode_data.dones)

        # Replay the window exactly as act() ran it, before any update: this
        # gives the behaviour policy's log-probs and values, and the state
        # each training chunk has to start from.
        chunk_starts: list = []
        with torch.no_grad():
            out_old, state_T = self._replay(obs_t, flags, start, 0, T, chunk_len, chunk_starts)
            old_lp_eval, _, values = self._evaluate_actions(out_old, act_t)
            last_value = 0.0
            final_obs = getattr(episode_data, "final_obs", None)
            dones = episode_data.dones
            if (
                final_obs is not None and not isinstance(final_obs, (list, tuple))
                and len(dones) and not dones[-1]
            ):
                f = self._to_tensor(final_obs).unsqueeze(0)
                if f.shape[1] == n_in:
                    out_f, _ = self.graph.forward_batch(f, state_T)
                    last_value = float(out_f[0, -1].item())
        self._consume_window(T, None if state_T is None else state_T.detach())

        if episode_data.log_probs:
            old_lp = torch.tensor(episode_data.log_probs, dtype=torch.float32, device=self.device)
        else:
            old_lp = old_lp_eval

        advantages, returns = compute_gae(
            rewards=rew_t.cpu().numpy().tolist(),
            values=values.cpu().numpy().tolist(),
            dones=episode_data.dones,
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            last_value=last_value,
        )
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = total_pg = total_vf = total_ent = 0.0
        n_updates = 0

        for _ in range(hp["n_epochs"]):
            # Sequential chunk windows — no shuffling
            for k, begin in enumerate(range(0, T, chunk_len)):
                end = min(begin + chunk_len, T)
                if end - begin < 2:
                    continue

                ac_chunk = act_t[begin:end]
                old_lp_chunk = old_lp[begin:end]
                adv_chunk = adv_t[begin:end]
                ret_chunk = ret_t[begin:end]

                out_chunk, _ = self._replay(obs_t, flags, chunk_starts[k], begin, end, chunk_len)
                new_lp, entropy, values_chunk = self._evaluate_actions(out_chunk, ac_chunk)

                ratio = torch.exp(new_lp - old_lp_chunk)
                pg1 = ratio * adv_chunk
                pg2 = torch.clamp(ratio, 1 - hp["clip_ratio"], 1 + hp["clip_ratio"]) * adv_chunk
                pg_loss = -torch.min(pg1, pg2).mean()

                vf_loss = F.mse_loss(values_chunk, ret_chunk)
                ent_loss = -entropy.mean()

                loss = pg_loss + hp["vf_coef"] * vf_loss + hp["entropy_coef"] * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.graph.parameters(), hp["max_grad_norm"])
                self.optimizer.step()
                self.graph.enforce_dale()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                n_updates += 1

        self._episode_count += 1
        denom = max(n_updates, 1)
        return {
            "loss": total_loss / denom,
            "pg_loss": total_pg / denom,
            "vf_loss": total_vf / denom,
            "entropy": -total_ent / denom,
            "n_neurons": self.graph.n_neurons(),
            "n_edges": self.graph.n_edges(),
        }

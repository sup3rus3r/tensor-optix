"""
tensor_optix.distributed.async_learner — IMPALA-style async actor-learner.

Architecture
------------
N actor processes run environment episodes in parallel, each using the current
policy weights read directly from shared memory.  A single learner
(main process) dequeues trajectories, applies V-trace IS correction, and
performs gradient updates.  Because actor and learner share the same physical
memory pages (via ``torch.Tensor.share_memory_()``),
weight updates are immediately visible to all actors — no explicit weight
broadcast queue, no serialization overhead.

                ┌──────────────────────────────────────────┐
  shared mem    │  actor  nn.Module  (shared_memory=True)  │
                │  critic nn.Module  (shared_memory=True)  │
                └──────────────────────────────────────────┘
                        ↑ reads (lock-free)    ↑ writes (optimizer.step)
        ┌───────────┐   │               ┌──────────────────────────┐
        │  Actor 0  │───┘               │  Learner (main process)  │
        │  Actor 1  │──── traj_queue ──►│  V-trace correction      │
        │  ...      │                   │  gradient update         │
        │  Actor N  │                   └──────────────────────────┘
        └───────────┘

Off-policy correction
---------------------
Actors run a policy μ that may lag behind the current learner policy θ.
The V-trace importance-sampling correction (Espeholt et al. 2018) adjusts
the value targets and policy-gradient advantages for this staleness:

    ρ̄_t = min(ρ̄, π_θ(a_t|s_t) / π_μ(a_t|s_t))   ← clipped IS weight
    v_t  = V(s_t) + δ_t + γ c̄_t (v_{t+1} − V(s_{t+1}))

With ρ̄ = c̄ = 1 and synchronous actors (μ = θ), this reduces to standard
on-policy advantage estimation.

Platform note
-------------
Automatically selects ``fork`` on Linux and ``spawn`` on Windows / macOS.
With ``fork`` (Linux), all objects passed to the actor subprocess are inherited
via ``os.fork()`` — no pickling of ``nn.Module`` instances is required.
With ``spawn`` (Windows / macOS), ``env_factory`` must be picklable and the
actor / critic models must have ``share_memory()`` called before launching.
"""

from __future__ import annotations

import sys
import time
import logging
import multiprocessing as mp
import warnings
import numpy as np
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actor subprocess — module-level so it is picklable for spawn/forkserver.
# ---------------------------------------------------------------------------

def _actor_loop(
    actor_id: int,
    actor,                  # nn.Module in shared memory — read only
    critic,                 # nn.Module in shared memory — read only
    env_factory: Callable,  # () → gym.Env
    trajectory_len: int,
    traj_queue,             # mp.Queue
    stop_event,             # mp.Event
    step_counter,           # mp.Value('l', 0)
    seed: int,
) -> None:
    """
    Actor process body.

    Steps the environment indefinitely, accumulating observations, actions,
    rewards, dones, log-probs, and values into a rolling buffer.  Every
    ``trajectory_len`` steps the buffer is serialised and pushed to
    ``traj_queue``.  The process polls ``stop_event`` before each step and
    exits cleanly when it is set.
    """
    import torch
    import torch.nn.functional as F

    # Limit actor to a single intra-op thread.  Actor processes run simple
    # small-batch forward passes that get no benefit from multi-threading but
    # do cause BLAS contention when multiple actors share CPU cores.
    torch.set_num_threads(1)

    torch.manual_seed(seed + actor_id * 7919)
    np.random.seed(seed + actor_id * 7919)

    env = env_factory()
    reset_result = env.reset(seed=seed + actor_id)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    obs_buf:  List = []
    act_buf:  List = []
    rew_buf:  List = []
    done_buf: List = []
    logp_buf: List = []
    val_buf:  List = []

    try:
        while not stop_event.is_set():
            with torch.no_grad():
                obs_t  = torch.as_tensor(
                    np.atleast_1d(obs).astype(np.float32)
                ).unsqueeze(0)
                logits = actor(obs_t)                          # [1, n_actions]
                lp_all = F.log_softmax(logits, dim=-1)
                dist   = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                logp   = float(lp_all[0, action].item())
                value  = float(critic(obs_t).squeeze().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_np = np.atleast_1d(obs).astype(np.float32)
            obs_buf.append(obs_np)
            act_buf.append(action)
            rew_buf.append(float(reward))
            done_buf.append(done)
            logp_buf.append(logp)
            val_buf.append(value)

            if done:
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            else:
                obs = next_obs

            if len(obs_buf) >= trajectory_len:
                # Bootstrap value for state immediately after this trajectory.
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        np.atleast_1d(obs).astype(np.float32)
                    ).unsqueeze(0)
                    bootstrap_val = float(critic(obs_t).squeeze().item())

                traj = {
                    "actor_id":            actor_id,
                    "observations":        np.array(obs_buf,  dtype=np.float32),
                    "actions":             np.array(act_buf,  dtype=np.int64),
                    "rewards":             np.array(rew_buf,  dtype=np.float32),
                    "dones":               np.array(done_buf, dtype=bool),
                    "behaviour_log_probs": np.array(logp_buf, dtype=np.float32),
                    # values[T+1]: V(s_0)..V(s_{T-1}), bootstrap_val at index T
                    "values":              np.array(
                        val_buf + [bootstrap_val], dtype=np.float32
                    ),
                }
                # Batch-increment the step counter once per trajectory
                # instead of per step — reduces lock contention by trajectory_len×.
                with step_counter.get_lock():
                    step_counter.value += trajectory_len

                try:
                    traj_queue.put(traj, timeout=0.1)
                except Exception:
                    pass  # Queue full — discard trajectory and continue

                obs_buf.clear()
                act_buf.clear()
                rew_buf.clear()
                done_buf.clear()
                logp_buf.clear()
                val_buf.clear()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class AsyncActorLearner:
    """
    IMPALA-style async actor-learner for PyTorch discrete policies.

    Parameters
    ----------
    actor            nn.Module — policy network (obs → logits over actions)
    critic           nn.Module — value network (obs → scalar)
    optimizer        torch.optim.Optimizer — over all actor+critic parameters
    env_factory      Callable → gym.Env — called once per actor process.
                     With fork (Linux default) this may be any callable,
                     including lambdas.  With spawn (Windows / macOS), it must
                     be picklable (module-level function or functools.partial).
    n_actors         int — parallel actor processes (default 4)
    trajectory_len   int — environment steps per trajectory batch (default 64)
    max_queue_size   int — max pending trajectories (0 = 8 × n_actors)
    gamma            float — discount factor
    rho_bar          float — V-trace IS weight clip ρ̄ (default 1.0)
    c_bar            float — V-trace trace clip c̄ (default 1.0)
    entropy_coef     float — entropy bonus coefficient
    vf_coef          float — value function loss coefficient
    max_grad_norm    float — gradient norm clipping threshold
    seed             int — base random seed; actor i uses seed + i×7919
    """

    def __init__(
        self,
        actor,
        critic,
        optimizer,
        env_factory: Callable,
        n_actors: int = 4,
        trajectory_len: int = 64,
        max_queue_size: int = 0,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 0,
    ):
        self._actor         = actor
        self._critic        = critic
        self._optimizer     = optimizer
        self._env_factory   = env_factory
        self._n_actors      = n_actors
        self._trajectory_len = trajectory_len
        self._max_queue     = max_queue_size or (8 * n_actors)
        self._gamma         = gamma
        self._rho_bar       = rho_bar
        self._c_bar         = c_bar
        self._entropy_coef  = entropy_coef
        self._vf_coef       = vf_coef
        self._max_grad_norm = max_grad_norm
        self._seed          = seed

        self._actor_procs: List[mp.Process] = []
        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: int) -> dict:
        """
        Launch actor processes and run the learner loop.

        Blocks until ``max_steps`` total environment steps have been
        collected across all actors, then signals all actors to stop
        and waits for them to exit.

        Parameters
        ----------
        max_steps  int — total env steps to collect (summed over all actors)

        Returns
        -------
        dict with:
            total_steps       — actual env steps collected (≥ max_steps)
            total_updates     — learner gradient updates performed
            steps_per_second  — env throughput (steps / wall-clock seconds)
            elapsed           — wall-clock seconds
        """
        import torch
        import torch.nn.functional as F
        from .vtrace import compute_vtrace_targets

        # Platform-aware multiprocessing context.
        # fork is unavailable on Windows and discouraged on macOS.
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        if method == "spawn":
            warnings.warn(
                f"AsyncActorLearner is using multiprocessing 'spawn' on {sys.platform}. "
                "POSIX fork zero-copy memory inheritance is not available; "
                "ensure actor/critic models have share_memory() called and that "
                "env_factory is picklable.",
                RuntimeWarning,
                stacklevel=2,
            )
        ctx          = mp.get_context(method)
        traj_queue   = ctx.Queue(maxsize=self._max_queue)
        stop_event   = ctx.Event()
        step_counter = ctx.Value("l", 0)

        # Move parameters to shared memory so all actor processes see
        # weight updates immediately without explicit IPC broadcast.
        # On Linux with fork this is zero-copy; on Windows/macOS with spawn
        # PyTorch implements this via file-backed shared memory.
        self._actor.share_memory()
        self._critic.share_memory()

        self._actor_procs = []
        for i in range(self._n_actors):
            p = ctx.Process(
                target=_actor_loop,
                args=(
                    i, self._actor, self._critic,
                    self._env_factory, self._trajectory_len,
                    traj_queue, stop_event, step_counter, self._seed,
                ),
                daemon=True,
                name=f"impala-actor-{i}",
            )
            p.start()
            self._actor_procs.append(p)
            logger.debug("Started actor process %d (pid=%d)", i, p.pid)

        t_start = time.monotonic()
        self._total_updates = 0
        all_params = (
            list(self._actor.parameters()) + list(self._critic.parameters())
        )

        try:
            while True:
                with step_counter.get_lock():
                    steps_so_far = step_counter.value
                if steps_so_far >= max_steps:
                    break

                # Block until a trajectory arrives (1 s timeout to re-check stop).
                try:
                    traj = traj_queue.get(timeout=1.0)
                except Exception:
                    continue  # timeout — re-check max_steps

                obs     = torch.as_tensor(traj["observations"],        dtype=torch.float32)
                actions = torch.as_tensor(traj["actions"],             dtype=torch.long)
                rewards = traj["rewards"]
                dones   = traj["dones"]
                behav_lp = traj["behaviour_log_probs"]
                values   = traj["values"]   # shape [T+1]

                # ── Learner forward pass (with grad) ──────────────────────
                self._actor.train()
                logits   = self._actor(obs)              # [T, n_actions]
                lp_all   = F.log_softmax(logits, dim=-1)
                lp_taken = lp_all[range(len(actions)), actions]   # [T]

                # Current log-probs for V-trace IS ratios (no grad needed here)
                curr_lp = lp_taken.detach().cpu().numpy()

                # ── V-trace targets ───────────────────────────────────────
                self._critic.train()
                vs, advantages = compute_vtrace_targets(
                    rewards=rewards,
                    values=values,
                    behaviour_log_probs=behav_lp,
                    current_log_probs=curr_lp,
                    dones=dones,
                    gamma=self._gamma,
                    rho_bar=self._rho_bar,
                    c_bar=self._c_bar,
                )

                # ── Losses ────────────────────────────────────────────────
                adv_t    = torch.as_tensor(advantages, dtype=torch.float32)
                vs_t     = torch.as_tensor(vs,         dtype=torch.float32)

                # Policy gradient: ∇_θ E[A_t · log π_θ(a_t|s_t)]
                pol_loss = -(lp_taken * adv_t).mean()

                # Value regression: E[(V(s_t) − v_t)²]
                new_val  = self._critic(obs).squeeze(-1)
                val_loss = F.mse_loss(new_val, vs_t)

                # Entropy bonus
                probs   = torch.exp(lp_all)
                entropy = -(probs * lp_all).sum(dim=-1).mean()

                loss = pol_loss + self._vf_coef * val_loss - self._entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, self._max_grad_norm)
                self._optimizer.step()

                self._total_updates += 1

        finally:
            stop_event.set()
            for p in self._actor_procs:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    logger.warning("Actor process %s did not exit cleanly — terminated", p.name)
            self._actor_procs = []

        elapsed = max(time.monotonic() - t_start, 1e-9)
        total_steps = step_counter.value
        return {
            "total_steps":      total_steps,
            "total_updates":    self._total_updates,
            "steps_per_second": total_steps / elapsed,
            "elapsed":          elapsed,
        }

    @property
    def total_updates(self) -> int:
        return self._total_updates

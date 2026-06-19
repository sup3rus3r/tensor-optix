# tensor_optix.distributed

IMPALA-style asynchronous actor-learner training with V-trace off-policy correction (Espeholt et al. 2018, [arXiv:1802.01561](https://arxiv.org/abs/1802.01561)).

## AsyncActorLearner

```python
class AsyncActorLearner:
    """
    IMPALA-style async actor-learner for PyTorch discrete policies.

    N actor processes run environment episodes in parallel, each using the
    current policy weights read directly from shared memory (via
    torch.Tensor.share_memory_()). A single learner (main process) dequeues
    trajectories, applies V-trace IS correction, and performs gradient
    updates. Weight updates are immediately visible to all actors - no
    explicit weight broadcast queue, no serialization overhead.

    Platform note: 'fork' is used automatically on Linux (no pickling
    required), 'spawn' on Windows/macOS (env_factory must be picklable -
    a module-level function or functools.partial, not a lambda with a
    non-picklable closure).

    Parameters
    ----------
    actor            nn.Module - policy network (obs → logits over actions)
    critic           nn.Module - value network (obs → scalar)
    optimizer        torch.optim.Optimizer - over all actor+critic parameters
    env_factory      Callable → gym.Env - called once per actor process.
    n_actors         int - parallel actor processes (default 4)
    trajectory_len   int - environment steps per trajectory batch (default 64)
    max_queue_size   int - max pending trajectories (0 = 8 × n_actors)
    gamma            float - discount factor
    rho_bar          float - V-trace IS weight clip ρ̄ (default 1.0)
    c_bar            float - V-trace trace clip c̄ (default 1.0)
    entropy_coef     float - entropy bonus coefficient
    vf_coef          float - value function loss coefficient
    max_grad_norm    float - gradient norm clipping threshold
    seed             int - base random seed; actor i uses seed + i×7919
    """

    def run(self, max_steps: int) -> dict:
        """
        Launch actor processes and run the learner loop.

        Blocks until max_steps total environment steps have been collected
        across all actors, then signals all actors to stop and waits for
        them to exit.

        Returns
        -------
        dict with: total_steps, total_updates, steps_per_second, elapsed
        """
```

## compute_vtrace_targets

```python
def compute_vtrace_targets(
    rewards,              # shape [T]    per-step rewards
    values,                # shape [T+1]  V(s_t) for t=0..T; values[T] is bootstrap
    behaviour_log_probs,   # shape [T]    log π_μ(a_t|s_t) recorded by the actor
    current_log_probs,     # shape [T]    log π_θ(a_t|s_t) from current learner policy
    dones,                 # shape [T]    bool - True when episode ended at step t
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
):
    """
    Compute V-trace targets and policy-gradient advantages.

    IS ratio: ρ_t = π_θ(a_t|s_t) / π_μ(a_t|s_t)
    Clipped:  ρ̄_t = min(ρ̄, ρ_t)   c̄_t = min(c̄, ρ_t)

    V-trace target (backward recursion):
        δ_t = ρ̄_t · (r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t))
        v_T = V(s_T)
        v_t = V(s_t) + δ_t + γ·c̄_t·(v_{t+1} − V(s_{t+1}))

    Policy-gradient advantage:
        A_t = ρ̄_t · (r_t + γ·v_{t+1}·(1−done_t) − V(s_t))

    Setting ρ̄ = c̄ = 1 and μ = θ (on-policy) recovers standard GAE with λ = c̄.

    Returns
    -------
    vs          shape [T]  V-trace targets for the value regression loss
    advantages  shape [T]  IS-corrected policy-gradient advantages
    """
```

See [Distributed training (IMPALA + V-trace)](../guides/distributed-training.md) for usage guidance.

# tensor_optix.core.diagnostic_controller

```
DiagnosticController - episode-level internal signal monitoring.

Role: Read train_diagnostics returned by agent.learn() every episode and
apply immediate targeted hyperparam corrections when specific thresholds
are crossed. Fires every episode, no eval cycle needed.

Why this exists alongside SPSA:
    SPSA estimates a score gradient over 2 episodes and applies a small
    nudge in the right direction. It is deliberately slow and blind to
    internals - it only sees the score. This is fine for global search
    (finding the right lr range, gamma, etc.) but cannot respond to acute
    internal failures:

        - Loss explosion:    lr is too high RIGHT NOW. SPSA won't notice
                             for 10+ episodes of compounding damage.
        - Entropy collapse:  policy stopped exploring. Score may look fine
                             temporarily (exploiting a local optimum) while
                             the policy quietly degenerates.
        - KL too high:       PPO update was too aggressive. Next episode
                             will be trained on a broken policy.
        - Epsilon exhausted: DQN ran out of exploration budget without
                             learning. Score is stuck at ~9, will never
                             recover without a reset.

    A human watching verbose logs would intervene on all of these within
    one episode. DiagnosticController does the same thing autonomously.

Design principles:
    - Each rule has a clear cause, threshold, and single targeted action.
    - Rules are additive - multiple can fire in the same episode.
    - Every firing is logged so it's visible and auditable.
    - Rules only fire when enough history exists to avoid false positives
      in the first few noisy episodes.
    - No rule modifies more than one param at a time (keeps cause/effect clear).
```

## DiagnosticController

```python
class DiagnosticController:
    """
    Monitors agent train_diagnostics every episode and applies immediate
    targeted hyperparam corrections when thresholds are crossed.

    Attach to LoopController. Called after agent.learn() returns, before
    the eval cycle.
    """

    def __init__(
        self,
        loss_spike_factor: float = 5.0,
        loss_window: int = 10,
        entropy_floor: Optional[float] = 0.05,
        target_kl: Optional[float] = 0.02,
        epsilon_patience: int = 20,
        epsilon_reset_value: float = 0.3,
        epsilon_score_threshold: float = 20.0,
        min_episodes: int = 5,
        verbose: bool = False,
    ):
        """
        Args:
            loss_spike_factor:    Fire loss-explosion rule when loss > N × rolling mean.
                                  Default 5.0 - catches genuine explosions, not noise.
            loss_window:          Rolling window for computing mean loss. Default 10.
            entropy_floor:        PPO: fire entropy-collapse rule below this value.
                                  Default 0.05 (nats). Env-specific - set to None to disable.
            target_kl:            PPO: fire KL-too-high rule above 2 × this value.
                                  Default 0.02. Set to None to disable.
            epsilon_patience:         DQN: episodes at epsilon_min before reset fires.
                                      Default 20. Set to 0 to disable.
            epsilon_reset_value:      DQN: value to reset epsilon to. Default 0.3.
            epsilon_score_threshold:  DQN: only reset epsilon if score is below this.
                                      Prevents resetting during active exploitation.
                                      Default 20.0 (well above random CartPole ~9).
            min_episodes:         Minimum episodes before any rule can fire.
                                  Default 5 - lets the loss mean stabilise first.
            verbose:              Print a line whenever a rule fires.
        """

    def step(self, episode_id: int, agent: BaseAgent, train_diagnostics: dict) -> list:
        """
        Evaluate all rules against train_diagnostics. Apply any triggered
        corrections directly via agent.set_hyperparams().

        Returns list of (rule_name, action_str) for anything that fired,
        so LoopController can include them in verbose output.
        """
```

## Rules

| Rule | Trigger | Action |
|---|---|---|
| `LOSS_SPIKE` | `loss > loss_spike_factor × rolling_mean(loss)` (≥3 samples) | `learning_rate × 0.5` |
| `ENTROPY_COLLAPSE` | `entropy < entropy_floor` | `entropy_coef × 2.0` (capped at 0.5) |
| `KL_TOO_HIGH` | `approx_kl > 2 × target_kl` | `learning_rate × 0.5` - reduces the update size rather than tightening `clip_ratio`, since high KL means the policy has already diverged from the data-collection policy |
| `EPSILON_RESET` | DQN `epsilon` stuck at `epsilon_min` for `epsilon_patience` episodes **and** score below `epsilon_score_threshold` | `epsilon → epsilon_reset_value` |

`EPSILON_RESET` is gated on score specifically so it doesn't fire while the agent is successfully exploiting a learned policy - only when it's stuck at the floor *and* failing.

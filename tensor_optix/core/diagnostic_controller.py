"""
DiagnosticController — episode-level internal signal monitoring.

Role: Read train_diagnostics returned by agent.learn() every episode and
apply immediate targeted hyperparam corrections when specific thresholds
are crossed. Fires every episode, no eval cycle needed.

Why this exists alongside SPSA:
    SPSA estimates a score gradient over 2 episodes and applies a small
    nudge in the right direction. It is deliberately slow and blind to
    internals — it only sees the score. This is fine for global search
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
    - Rules are additive — multiple can fire in the same episode.
    - Every firing is logged so it's visible and auditable.
    - Rules only fire when enough history exists to avoid false positives
      in the first few noisy episodes.
    - No rule modifies more than one param at a time (keeps cause/effect clear).
"""

from collections import deque
from typing import Optional
import numpy as np

from .base_agent import BaseAgent


class DiagnosticController:
    """
    Monitors agent train_diagnostics every episode and applies immediate
    targeted hyperparam corrections when thresholds are crossed.

    Attach to LoopController. Called after agent.learn() returns, before
    the eval cycle.

    Args:
        loss_spike_factor:    Fire loss-explosion rule when loss > N × rolling mean.
                              Default 5.0 — catches genuine explosions, not noise.
        loss_window:          Rolling window for computing mean loss. Default 10.
        entropy_floor:        PPO: fire entropy-collapse rule below this value.
                              Default 0.05 (nats). Env-specific — set to None to disable.
        target_kl:            PPO: fire KL-too-high rule above 2 × this value.
                              Default 0.02. Set to None to disable.
        epsilon_patience:         DQN: episodes at epsilon_min before reset fires.
                                  Default 20. Set to 0 to disable.
        epsilon_reset_value:      DQN: value to reset epsilon to. Default 0.3.
        epsilon_score_threshold:  DQN: only reset epsilon if score is below this.
                                  Prevents resetting during active exploitation.
                                  Default 20.0 (well above random CartPole ~9).
        min_episodes:         Minimum episodes before any rule can fire.
                              Default 5 — lets the loss mean stabilise first.
        verbose:              Print a line whenever a rule fires.
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
        self._loss_spike_factor = loss_spike_factor
        self._loss_window = deque(maxlen=loss_window)
        self._entropy_floor = entropy_floor
        self._target_kl = target_kl
        self._epsilon_patience = epsilon_patience
        self._epsilon_reset_value = epsilon_reset_value
        self._epsilon_score_threshold = epsilon_score_threshold
        self._min_episodes = min_episodes
        self._verbose = verbose

        self._episode_count = 0
        self._eps_at_floor_count = 0  # consecutive episodes where epsilon == epsilon_min

        # Diagnostic history for verbose summary
        self.firings: list = []  # list of (episode_id, rule, action_str)

    def step(self, episode_id: int, agent: BaseAgent, train_diagnostics: dict) -> list:
        """
        Evaluate all rules against train_diagnostics. Apply any triggered
        corrections directly via agent.set_hyperparams().

        Returns list of (rule_name, action_str) for anything that fired,
        so LoopController can include them in verbose output.
        """
        self._episode_count += 1
        fired = []

        if self._episode_count < self._min_episodes:
            return fired

        hp = agent.get_hyperparams()
        params = dict(hp.params)
        changed = False

        # ── Rule 1: Loss explosion ────────────────────────────────────────
        loss = train_diagnostics.get("loss") or train_diagnostics.get("actor_loss")
        if loss is not None and float(loss) > 0:
            loss = float(loss)
            self._loss_window.append(loss)
            if len(self._loss_window) >= 3:
                mean_loss = float(np.mean(list(self._loss_window)[:-1]))  # exclude current
                if mean_loss > 0 and loss > self._loss_spike_factor * mean_loss:
                    old_lr = float(params.get("learning_rate", 0))
                    if old_lr > 0:
                        new_lr = old_lr * 0.5
                        params["learning_rate"] = new_lr
                        action = f"lr {old_lr:.5g}→{new_lr:.5g} (-50%)  [loss spike: {loss:.3g} > {self._loss_spike_factor:.0f}× mean {mean_loss:.3g}]"
                        fired.append(("LOSS_SPIKE", action))
                        changed = True
        elif loss is not None:
            self._loss_window.append(float(loss))

        # ── Rule 2: PPO entropy collapse ──────────────────────────────────
        entropy = train_diagnostics.get("entropy")
        if entropy is not None and self._entropy_floor is not None:
            entropy = float(entropy)
            if entropy < self._entropy_floor:
                old_ec = float(params.get("entropy_coef", 0))
                if old_ec > 0:
                    new_ec = min(old_ec * 2.0, 0.5)
                    params["entropy_coef"] = new_ec
                    action = f"entropy_coef {old_ec:.5g}→{new_ec:.5g} (+100%)  [entropy={entropy:.4f} < floor={self._entropy_floor}]"
                    fired.append(("ENTROPY_COLLAPSE", action))
                    changed = True

        # ── Rule 3: PPO KL too high ───────────────────────────────────────
        kl = train_diagnostics.get("kl_div") or train_diagnostics.get("approx_kl")
        if kl is not None and self._target_kl is not None:
            kl = float(kl)
            if kl > 2.0 * self._target_kl:
                old_cr = float(params.get("clip_ratio", 0))
                if old_cr > 0:
                    new_cr = max(old_cr * 0.8, 0.05)
                    params["clip_ratio"] = new_cr
                    action = f"clip_ratio {old_cr:.5g}→{new_cr:.5g} (-20%)  [KL={kl:.4f} > 2×target={2*self._target_kl:.4f}]"
                    fired.append(("KL_TOO_HIGH", action))
                    changed = True

        # ── Rule 4: DQN epsilon exhausted without learning ────────────────
        # Only reset if the agent has FAILED to learn — not when epsilon is at
        # floor during active exploitation. Gate on score: if the agent is scoring
        # well above random play (epsilon_min_score × 3 as a rough heuristic),
        # it is exploiting correctly and resetting epsilon would sabotage it.
        epsilon = train_diagnostics.get("epsilon")
        epsilon_min = float(params.get("epsilon_min", 0.05))
        score = train_diagnostics.get("score") or train_diagnostics.get("episode_reward")
        if epsilon is not None:
            epsilon = float(epsilon)
            if abs(epsilon - epsilon_min) < 1e-6:
                self._eps_at_floor_count += 1
            else:
                self._eps_at_floor_count = 0

            score_is_low = score is None or float(score) < self._epsilon_score_threshold
            if (
                self._epsilon_patience > 0
                and self._eps_at_floor_count >= self._epsilon_patience
                and score_is_low
            ):
                new_eps = self._epsilon_reset_value
                params["epsilon"] = new_eps
                action = f"epsilon {epsilon:.4f}→{new_eps:.4f}  [stuck at floor for {self._eps_at_floor_count} episodes, score low]"
                fired.append(("EPSILON_RESET", action))
                self._eps_at_floor_count = 0  # reset counter after firing
                changed = True
            elif self._eps_at_floor_count >= self._epsilon_patience and not score_is_low:
                # Agent is exploiting — just reset the counter silently
                self._eps_at_floor_count = 0

        if changed:
            hp.params.update(params)
            agent.set_hyperparams(hp)

        if fired and self._verbose:
            for rule, action in fired:
                print(f"  DIAG     [{rule}] {action}", flush=True)

        self.firings.extend([(episode_id, r, a) for r, a in fired])
        return fired

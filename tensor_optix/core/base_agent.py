from abc import ABC, abstractmethod
from .types import EpisodeData, HyperparamSet


class BaseAgent(ABC):
    """
    The only contract between the loop controller and any agent.

    The loop controller calls ONLY these six methods. It never inspects
    internals, never assumes a network architecture, never assumes gradient-based
    learning, never assumes discrete or continuous actions.

    Any algorithm — PPO, DQN, SAC, CMA-ES, a custom evolutionary algorithm,
    or one that doesn't exist yet — must be wrappable by implementing these
    six methods. Nothing more is required.
    """

    @abstractmethod
    def act(self, observation) -> any:
        """
        Given an observation, return an action.
        Called at every step during episode interaction.
        Must be fast — this is in the hot path.
        """

    @abstractmethod
    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Update internal weights given a completed episode's data.

        Returns a dict of training diagnostics (loss, entropy, grad_norm, etc.).
        These diagnostics are forwarded to the evaluator's score() method.
        The dict may be empty if the agent has no diagnostics to report.
        """

    @abstractmethod
    def get_hyperparams(self) -> HyperparamSet:
        """
        Return the current hyperparameter set.
        Called before and after each optimizer suggestion cycle.
        """

    @abstractmethod
    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        """
        Apply a new hyperparameter set.
        The agent is responsible for knowing how to apply each param
        (e.g. updating optimizer learning rate, entropy coefficient, etc.)
        """

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Persist model weights to the given path."""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Restore model weights from the given path."""

    @property
    def is_on_policy(self) -> bool:
        """
        True  → on-policy (PPO, A2C, REINFORCE).
                 Each window is fresh data. Rollback to a previous checkpoint
                 is safe because the next window will be generated entirely by
                 the restored policy.

        False → off-policy (SAC, DQN, TD3).
                 A replay buffer accumulates experience from ALL past policies.
                 Rolling back weights WITHOUT clearing the buffer means the
                 agent immediately trains on stale, mismatched experience —
                 corrupted Bellman targets drag the policy back down.
                 LoopController skips weight rollback for off-policy agents
                 even when rollback_on_degradation=True.

        Default: True. Off-policy agents MUST override this to False.
        """
        return True

    def average_weights(self, paths: list) -> None:
        """
        Replace current weights with the element-wise mean of weights
        loaded from each path in `paths`.

        Math: θ_avg = (1/N) × Σᵢ θᵢ

        This implements Stochastic Weight Averaging (SWA). Averaging weights
        from multiple high-scoring checkpoints tends to land in a flatter,
        wider region of the loss landscape, improving generalisation and
        robustness without any inference cost.

        Only checkpoints within a score band of the best should be averaged —
        checkpoints from very different training stages (e.g. pre/post collapse)
        will produce a broken policy when averaged.

        Default: no-op. Framework-specific agents override this.
        Called by CheckpointRegistry.load_ensemble().
        """

    def perturb_weights(self, noise_scale: float) -> None:
        """
        Apply multiplicative Gaussian noise to all network parameters.

        Math: θ_new = θ × (1 + noise_scale × ε),  ε ~ N(0, I)

        Multiplicative noise is scale-invariant: a parameter of magnitude
        0.001 receives the same *relative* perturbation as one of magnitude
        10.0.  This matches the hyperparam perturbation in spawn_variant()
        and is the standard weight-space mutation used in ES and PBT.

        Default: no-op.  Framework-specific agents (TorchPPOAgent, etc.)
        override this to perturb their actual network parameters.

        Called by PolicyManagerCallback._do_spawn() after spawn_variant()
        restores the best checkpoint — so perturbation is always relative
        to the best known weights, not the current (possibly degraded) ones.
        """

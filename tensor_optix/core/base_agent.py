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

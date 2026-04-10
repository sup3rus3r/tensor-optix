from abc import ABC, abstractmethod
from typing import List
from .types import EvalMetrics, HyperparamSet


class BaseOptimizer(ABC):
    """
    Suggests hyperparameter adjustments based on performance history.

    The optimizer never knows which RL algorithm is running. It sees only:
    - A HyperparamSet (opaque dict of param_name → value)
    - A history of EvalMetrics (scores over time)

    It returns a new HyperparamSet. What those params mean, and how the
    agent applies them, is entirely the agent's responsibility.

    The optimizer must treat HyperparamSet.params as an opaque dict.
    It may perturb values numerically, but must never hardcode key names
    or assume which params exist. Use param_bounds config for constraints.
    """

    @abstractmethod
    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        """
        Propose new hyperparameters.

        Args:
            current_hyperparams: What the agent is currently using.
            metrics_history: Full history of EvalMetrics, oldest first.
                             Use this to detect trends, plateaus, oscillation.

        Returns:
            A new HyperparamSet to apply before the next episode.
            May return current_hyperparams unchanged if no adjustment is warranted.
        """

    def on_improvement(self, metrics: EvalMetrics) -> None:
        """Called when a new best score is achieved. Override to react."""

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        """Called when a plateau is detected. Override to react."""

    @property
    def is_probing(self) -> bool:
        """
        Returns True when the optimizer has just applied a probe perturbation
        and is waiting to measure the effect. During this episode the score
        drop (or gain) is caused by the probe itself — not a genuine policy
        collapse — so LoopController suppresses degradation detection.

        Default: False (no probing). Override in optimizers that use
        finite-difference probing (BackoffOptimizer, SPSA, etc.).
        """
        return False

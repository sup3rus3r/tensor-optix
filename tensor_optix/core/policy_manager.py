import logging
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import numpy as np

from .base_agent import BaseAgent
from .checkpoint_registry import CheckpointRegistry
from .loop_controller import LoopCallback
from .types import EvalMetrics, PolicySnapshot

logger = logging.getLogger(__name__)


class PolicyManager:
    """
    Manages model evolution and ensemble logic.

    Separation of concerns:
        Optimizer  → tunes hyperparameters
        PolicyManager → evolves models (swap, variant spawn)

    Core responsibilities:
    1. evolve(): On DORMANT, compare current score vs registry best.
                 If current < best: rollback agent to best checkpoint.
                 If current >= best: no-op (system is at its best known state).
    2. spawn_variant(): Clone best checkpoint into a new agent shell and apply
                        hyperparam mutation. Produces a candidate for the ensemble.
    3. ensemble_action(): Combine actions from multiple registered agents
                          via weighted averaging.
    4. update_weights() / auto_update_weights(): Adjust agent weights based on
                          externally provided or internally recorded score history.

    Minimal usage (evolution only):
        pm = PolicyManager(registry)
        cb = pm.as_callback(agent)
        optimizer = RLOptimizer(..., callbacks=[cb])

    Ensemble with spawning:
        pm = PolicyManager(registry)
        pm.add_agent(primary_agent, weight=1.0)
        variant = pm.spawn_variant(SecondAgent(...), noise_scale=0.05)
        pm.add_agent(variant, weight=1.0)
        ensemble = EnsembleAgent(pm, primary_agent=primary_agent)
    """

    def __init__(self, registry: CheckpointRegistry, score_window: int = 10):
        self._registry = registry
        self._ensemble: List[Tuple[BaseAgent, float]] = []
        self._score_history: Dict[int, Deque[float]] = {}
        self._score_window = score_window

    # ------------------------------------------------------------------
    # Ensemble management
    # ------------------------------------------------------------------

    def add_agent(self, agent: BaseAgent, weight: float = 1.0) -> None:
        """Add an agent to the ensemble pool."""
        self._ensemble.append((agent, weight))

    def ensemble_action(self, obs: Any) -> Any:
        """
        Return a combined action from all registered agents.

        Single agent: equivalent to agent.act(obs).
        Multiple agents: weighted average — action = Σ(w_i * a_i) / Σ(w_i).

        Works for both continuous (direct average) and discrete action spaces
        (returns a soft value; caller decides whether to argmax).
        """
        if not self._ensemble:
            raise RuntimeError("No agents registered in PolicyManager")

        if len(self._ensemble) == 1:
            return self._ensemble[0][0].act(obs)

        total_weight = sum(w for _, w in self._ensemble)
        if total_weight == 0:
            total_weight = float(len(self._ensemble))
            weights = [1.0] * len(self._ensemble)
        else:
            weights = [w for _, w in self._ensemble]

        combined = sum(
            np.asarray(agent.act(obs), dtype=float) * w
            for (agent, _), w in zip(self._ensemble, weights)
        ) / total_weight

        return combined

    def update_weights(self, agent_scores: Dict[int, float]) -> None:
        """
        Update ensemble weights based on recent performance.

        agent_scores: {index_in_ensemble: score}
        Weights are set proportional to shifted scores (ensuring positivity).
        Agents not in agent_scores keep their current weight.
        """
        if not agent_scores:
            return

        scores = list(agent_scores.values())
        shift = max(0.0, -min(scores))  # shift so minimum score becomes 0

        new_ensemble = []
        for i, (agent, old_weight) in enumerate(self._ensemble):
            if i in agent_scores:
                new_weight = agent_scores[i] + shift + 1e-8
            else:
                new_weight = old_weight
            new_ensemble.append((agent, new_weight))

        self._ensemble = new_ensemble

    def record_agent_score(self, agent_idx: int, score: float) -> None:
        """
        Record a performance score for a specific agent.

        agent_idx: index in the ensemble (order of add_agent calls).
        score: comparable scalar — higher is better.

        Scores are stored in a rolling window of size score_window.
        Call auto_update_weights() to apply the recorded history to weights.
        """
        if agent_idx not in self._score_history:
            self._score_history[agent_idx] = deque(maxlen=self._score_window)
        self._score_history[agent_idx].append(score)

    def auto_update_weights(self) -> None:
        """
        Recompute ensemble weights from internally recorded score history.

        For each agent with recorded scores, weight = mean(recent_scores).
        Agents without recorded scores keep their current weight.
        No-op if no scores have been recorded via record_agent_score().
        """
        if not self._score_history:
            return
        agent_scores = {
            idx: float(np.mean(list(scores)))
            for idx, scores in self._score_history.items()
            if scores
        }
        if agent_scores:
            self.update_weights(agent_scores)

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self, agent: BaseAgent, current_score: float) -> bool:
        """
        Called when the loop enters DORMANT state.

        Compares current_score against the best checkpoint in the registry.
        - If current < best: loads best weights into agent. Returns True.
        - If current >= best: no-op. Returns False.

        When no checkpoint exists yet, returns False (nothing to roll back to).
        """
        best: Optional[PolicySnapshot] = self._registry.best
        if best is None:
            # Read metadata only — do NOT load weights yet.
            manifest = self._registry._load_manifest()
            if not manifest:
                logger.debug("PolicyManager.evolve: registry empty — skipping")
                return False
            best_entry = max(manifest, key=lambda e: e["primary_score"])
            best = self._registry._load_snapshot_from_dir(best_entry["snapshot_dir"])
            if best is None:
                logger.debug("PolicyManager.evolve: could not load snapshot metadata — skipping")
                return False

        if current_score < best.eval_metrics.primary_score:
            logger.info(
                "PolicyManager.evolve: current=%.4f < best=%.4f — rolling back to %s",
                current_score,
                best.eval_metrics.primary_score,
                best.snapshot_id,
            )
            self._registry.load_best(agent)
            return True

        logger.debug(
            "PolicyManager.evolve: current=%.4f >= best=%.4f — no rollback",
            current_score,
            best.eval_metrics.primary_score,
        )
        return False

    def spawn_variant(
        self,
        agent_shell: BaseAgent,
        noise_scale: float = 0.01,
        mutation_fn: Optional[Callable[[BaseAgent], None]] = None,
    ) -> BaseAgent:
        """
        Clone best checkpoint into agent_shell and apply mutation.

        Loads the best known weights into agent_shell, then perturbs its
        hyperparameters with multiplicative Gaussian noise. If mutation_fn
        is provided, it is called after weight loading for custom weight
        perturbation (e.g. adding noise directly to network parameters).

        agent_shell: pre-instantiated, compatible agent. Its weights will
                     be overwritten with the best-known checkpoint weights.
        noise_scale: std dev for multiplicative Gaussian noise on each
                     numeric hyperparam. Default 0.01 (1% perturbation).
        mutation_fn: optional callable(agent) -> None for weight-space
                     perturbation. Called after hyperparams are applied.

        Returns agent_shell (mutated in place, also returned for chaining).

        Example:
            variant = pm.spawn_variant(MyAgent(...), noise_scale=0.05)
            pm.add_agent(variant, weight=0.5)
        """
        snapshot = self._registry.load_best(agent_shell)
        if snapshot is None:
            logger.warning(
                "PolicyManager.spawn_variant: registry empty — variant starts from scratch"
            )
            return agent_shell

        hp = snapshot.hyperparams.copy()
        rng = np.random.default_rng()
        for key, val in list(hp.params.items()):
            if isinstance(val, float):
                hp.params[key] = float(val * (1.0 + rng.normal(0.0, noise_scale)))
            elif isinstance(val, int):
                hp.params[key] = max(1, round(val * (1.0 + rng.normal(0.0, noise_scale))))
        agent_shell.set_hyperparams(hp)

        if mutation_fn is not None:
            mutation_fn(agent_shell)

        logger.info(
            "PolicyManager.spawn_variant: cloned %s with noise_scale=%.4f",
            snapshot.snapshot_id,
            noise_scale,
        )
        return agent_shell

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_callback(self, agent: BaseAgent) -> "PolicyManagerCallback":
        """
        Returns a LoopCallback that auto-triggers evolve() on DORMANT events.
        Pass the result to RLOptimizer or LoopController as a callback.
        """
        return PolicyManagerCallback(self, agent)

    @property
    def ranked_snapshots(self) -> List[dict]:
        """
        All registry snapshots ranked by primary_score (highest first).
        Returns raw manifest dicts: {snapshot_id, primary_score, episode_id, ...}
        """
        manifest = self._registry._load_manifest()
        return sorted(manifest, key=lambda e: e["primary_score"], reverse=True)

    @property
    def ensemble_size(self) -> int:
        return len(self._ensemble)


class PolicyManagerCallback(LoopCallback):
    """
    Wires PolicyManager into the LoopController event system.

    Pass to RLOptimizer / LoopController via the callbacks list:
        cb = pm.as_callback(agent)
        RLOptimizer(..., callbacks=[cb])

    Tracks the most recent eval score.
    On DORMANT: auto-rebalances ensemble weights (if scores were recorded
    via record_agent_score()), then calls evolve() for rollback check.
    """

    def __init__(self, policy_manager: PolicyManager, agent: BaseAgent):
        self._pm = policy_manager
        self._agent = agent
        self._last_score: Optional[float] = None

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is not None:
            self._last_score = eval_metrics.primary_score

    def on_dormant(self, episode_id: int) -> None:
        if self._last_score is not None:
            self._pm.auto_update_weights()
            rolled_back = self._pm.evolve(self._agent, self._last_score)
            if rolled_back:
                logger.info(
                    "Episode %d: DORMANT — PolicyManager rolled back to best checkpoint",
                    episode_id,
                )

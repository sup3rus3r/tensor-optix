import logging
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import numpy as np

from .meta_controller import MetaAction

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
    3. prune(): Remove the lowest-performing agents to keep the ensemble lean.
    4. boost(): Multiply a specific agent's weight — use after regime detection.
    5. ensemble_action(): Combine actions from multiple registered agents
                          via weighted averaging.
    6. update_weights() / auto_update_weights(): Adjust agent weights based on
                          externally provided or internally recorded score history.
    7. adaptive_noise_scale(): Compute a dynamic noise_scale for spawn_variant
                          based on recent improvement — high noise on plateau,
                          low noise when improving.
    8. status(): Structured snapshot of all ensemble state for observability.

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

    def __init__(
        self,
        registry: CheckpointRegistry,
        score_window: int = 10,
        max_spawns: Optional[int] = None,
        max_ensemble_size: Optional[int] = None,
    ):
        self._registry = registry
        self._ensemble: List[Tuple[BaseAgent, float]] = []
        self._score_history: Dict[int, Deque[float]] = {}
        self._score_window = score_window
        self._max_spawns = max_spawns
        self._max_ensemble_size = max_ensemble_size
        self._spawn_count: int = 0
        self._prune_count: int = 0
        self._current_regime: Optional[str] = None

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

    def prune(self, bottom_k: int = 1) -> List[BaseAgent]:
        """
        Remove the bottom_k agents by current weight from the ensemble.

        Keeps the ensemble lean — call after spawn_variant() if the
        population has grown beyond a desired size.

        Returns the list of removed agents (in ascending weight order).
        No-op and returns [] if ensemble has bottom_k or fewer agents.
        """
        if len(self._ensemble) <= bottom_k:
            logger.debug("PolicyManager.prune: ensemble too small to prune (%d agents)", len(self._ensemble))
            return []

        order = sorted(range(len(self._ensemble)), key=lambda i: self._ensemble[i][1])
        to_remove = set(order[:bottom_k])

        removed_agents = [self._ensemble[i][0] for i in sorted(to_remove)]

        new_ensemble: List[Tuple[BaseAgent, float]] = []
        new_score_history: Dict[int, Deque[float]] = {}
        new_idx = 0
        for old_idx, (agent, weight) in enumerate(self._ensemble):
            if old_idx not in to_remove:
                new_ensemble.append((agent, weight))
                if old_idx in self._score_history:
                    new_score_history[new_idx] = self._score_history[old_idx]
                new_idx += 1

        self._ensemble = new_ensemble
        self._score_history = new_score_history
        self._prune_count += bottom_k

        logger.info(
            "PolicyManager.prune: removed %d agent(s), ensemble size now %d",
            len(removed_agents),
            len(self._ensemble),
        )
        return removed_agents

    def boost(self, agent: BaseAgent, factor: float = 2.0) -> None:
        """
        Multiply the weight of a specific agent by factor.

        Use after regime detection to shift action weight toward the most
        relevant policy without zeroing out the others.

        ensemble_action() normalises by total weight, so boosting one
        agent proportionally reduces the influence of all others.

        Example:
            regime = detector.detect(metrics_history)
            if regime == "volatile":
                pm.boost(agent_volatile, factor=2.0)
        """
        for i, (a, w) in enumerate(self._ensemble):
            if a is agent:
                self._ensemble[i] = (a, w * factor)
                logger.info(
                    "PolicyManager.boost: agent[%d] weight %.4f → %.4f (factor=%.2f)",
                    i, w, w * factor, factor,
                )
                return
        logger.warning("PolicyManager.boost: agent not found in ensemble")

    def set_regime(self, regime: str) -> None:
        """
        Record the current regime label for observability.

        Call this after RegimeDetector.detect() so that status() reflects
        the active regime. Logs a message when the regime changes.

        Example:
            regime = detector.detect(metrics_history)
            pm.set_regime(regime)
            pm.boost(regime_agents[regime], factor=2.0)
        """
        if regime != self._current_regime:
            logger.info(
                "PolicyManager.set_regime: %s → %s",
                self._current_regime or "none",
                regime,
            )
            self._current_regime = regime

    def adaptive_noise_scale(
        self,
        metrics_history: List[EvalMetrics],
        min_scale: float = 0.001,
        max_scale: float = 0.1,
        window: int = 10,
    ) -> float:
        """
        Compute a dynamic noise_scale for spawn_variant().

        Driven by three signals, not one:

        1. val_score slope: improving val → less noise (don't disrupt)
        2. generalization_gap: large train-val gap → more noise (overfitting,
           need to explore different solutions)
        3. train/val correlation: low correlation → more noise (train is lying —
           val is not following the training signal)

        Without val data (no val_pipeline), falls back to slope-only mode.

        Pass directly to spawn_variant():
            scale = pm.adaptive_noise_scale(metrics_history)
            pm.spawn_variant(MyAgent(...), noise_scale=scale)
        """
        if len(metrics_history) < 3:
            return max_scale

        recent = metrics_history[-window:]
        scores = np.array([m.primary_score for m in recent], dtype=float)
        mean = float(np.mean(scores))
        if mean == 0.0:
            return max_scale

        # --- Signal 1: val (or train) slope ---
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        normalized_slope = slope / abs(mean)
        # t=1 → strongly improving, t=0 → plateau/declining
        t = min(1.0, max(0.0, normalized_slope / 0.05))

        # --- Signals 2 & 3: gap and correlation (only when val data present) ---
        train_scores = [m.metrics.get("train_score") for m in recent]
        val_scores = [m.metrics.get("val_score") for m in recent]
        has_val = all(v is not None for v in train_scores + val_scores)

        gap_penalty = 0.0
        corr_penalty = 0.0

        if has_val:
            train_arr = np.array(train_scores, dtype=float)
            val_arr = np.array(val_scores, dtype=float)

            # Signal 2: mean generalization gap, normalized by |mean val|
            mean_val = float(np.mean(val_arr))
            if mean_val != 0.0:
                mean_gap = float(np.mean(train_arr - val_arr))
                gap_penalty = min(1.0, max(0.0, mean_gap / abs(mean_val)))

            # Signal 3: Pearson correlation between train and val trends
            # corr=1 → moving together (healthy), corr≈0 or negative → diverging
            if len(train_arr) >= 3 and np.std(train_arr) > 0 and np.std(val_arr) > 0:
                corr = float(np.corrcoef(train_arr, val_arr)[0, 1])
                # corr_penalty: 0 when perfectly correlated, 1 when fully decorrelated
                corr_penalty = min(1.0, max(0.0, 1.0 - corr))

        # Combine: t reduces noise (improvement), gap and corr penalties increase it
        # effective_t is reduced by overfitting and decorrelation signals
        effective_t = t * (1.0 - 0.5 * gap_penalty) * (1.0 - 0.5 * corr_penalty)
        scale = float(max_scale - effective_t * (max_scale - min_scale))

        logger.debug(
            "PolicyManager.adaptive_noise_scale: slope_t=%.2f gap_penalty=%.2f "
            "corr_penalty=%.2f effective_t=%.2f scale=%.4f",
            t, gap_penalty, corr_penalty, effective_t, scale,
        )
        return scale

    def status(self) -> dict:
        """
        Structured snapshot of current ensemble state.

        Returns a dict with:
        - ensemble_size: number of active agents
        - agents: list of {index, weight, mean_score, recent_scores}
        - regime: current regime label (if set via set_regime())
        - spawn_count: total spawns since construction
        - prune_count: total agents pruned since construction

        Use for logging, dashboards, or debugging:
            import json; print(json.dumps(pm.status(), indent=2))
        """
        agents_info = []
        for i, (_, weight) in enumerate(self._ensemble):
            history = list(self._score_history.get(i, []))
            agents_info.append({
                "index": i,
                "weight": round(weight, 6),
                "mean_score": round(float(np.mean(history)), 4) if history else None,
                "recent_scores": [round(s, 4) for s in history],
            })
        return {
            "ensemble_size": len(self._ensemble),
            "max_ensemble_size": self._max_ensemble_size,
            "agents": agents_info,
            "regime": self._current_regime,
            "spawn_count": self._spawn_count,
            "max_spawns": self._max_spawns,
            "spawns_remaining": self.spawns_remaining,
            "budget_exhausted": self.budget_exhausted,
            "prune_count": self._prune_count,
        }

    def training_report(self) -> dict:
        """
        Structured training report for display after training completes.

        Returns a superset of status() enriched with:
        - best_score: primary_score from the best registry checkpoint
        - best_val_score: val_score from that checkpoint (if a val pipeline was used)
        - best_generalization_gap: train - val from that checkpoint (if available)
        - termination_reason: why training stopped ("budget_exhausted", "meta_stop", "natural_dormant")

        Use pm.training_report() after opt.run() returns, or read it from the
        formatted stdout block that PolicyManagerCallback prints automatically.
        """
        report = self.status()

        # Enrich with best checkpoint info
        best_score = None
        best_val_score = None
        best_gap = None
        if self._registry is not None:
            manifest = self._registry._load_manifest()
            if manifest:
                best_entry = max(manifest, key=lambda e: e["primary_score"])
                best_score = best_entry.get("primary_score")
                # Load full snapshot to access the complete metrics dict
                snap = self._registry._load_snapshot_from_dir(best_entry["snapshot_dir"])
                if snap is not None:
                    m = snap.eval_metrics.metrics or {}
                    best_val_score = m.get("val_score")
                    best_gap = m.get("generalization_gap")

        report["best_score"] = best_score
        report["best_val_score"] = best_val_score
        report["best_generalization_gap"] = best_gap
        return report

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

        self._spawn_count += 1
        logger.info(
            "PolicyManager.spawn_variant: cloned %s with noise_scale=%.4f (total spawns: %d)",
            snapshot.snapshot_id,
            noise_scale,
            self._spawn_count,
        )
        return agent_shell

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_callback(
        self,
        agent: BaseAgent,
        agent_factory: Optional[Callable[[], BaseAgent]] = None,
        meta_controller: Optional[Any] = None,
    ) -> "PolicyManagerCallback":
        """
        Returns a LoopCallback that wires PolicyManager into the loop.

        agent_factory: callable that returns a fresh agent shell for spawning.
            If provided, the callback spawns variants autonomously on DORMANT.
            Required for autonomous operation.

        meta_controller: a MetaController (or any object with a
            decide(metrics_history, pm_status) -> MetaAction interface).
            If provided, all DORMANT decisions (spawn/prune/stop/no-op) are
            delegated to it. If omitted, spawning always fires when budget allows.

        Wiring:
            pm = PolicyManager(registry, max_spawns=3, max_ensemble_size=5)
            cb = pm.as_callback(agent, agent_factory=lambda: MyAgent(...),
                                meta_controller=MetaController())
            opt = RLOptimizer(..., callbacks=[cb])
            cb.set_stop_fn(opt.stop)
        """
        return PolicyManagerCallback(self, agent, agent_factory, meta_controller)

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

    @property
    def budget_exhausted(self) -> bool:
        """True when max_spawns is set and the spawn budget has been fully used."""
        return self._max_spawns is not None and self._spawn_count >= self._max_spawns

    @property
    def spawns_remaining(self) -> Optional[int]:
        """Remaining spawns in budget. None if no budget was set."""
        if self._max_spawns is None:
            return None
        return max(0, self._max_spawns - self._spawn_count)


class PolicyManagerCallback(LoopCallback):
    """
    Wires PolicyManager into the LoopController event system.

    Minimal wiring (rollback + budget termination only):
        cb = pm.as_callback(agent)
        opt = RLOptimizer(..., callbacks=[cb])
        cb.set_stop_fn(opt.stop)

    Fully autonomous (rollback + spawn + prune + MetaController decisions):
        mc = MetaController(gap_threshold=0.2, corr_threshold=0.5)
        cb = pm.as_callback(
            agent,
            agent_factory=lambda: MyAgent(...),
            meta_controller=mc,
        )
        opt = RLOptimizer(..., callbacks=[cb])
        cb.set_stop_fn(opt.stop)

    On every DORMANT event:
    1. Auto-rebalance ensemble weights from score history
    2. Rollback to best checkpoint if current score < best
    3. If agent_factory provided:
       - If MetaController present: delegate SPAWN/PRUNE/STOP/NO_OP decision to it
       - If no MetaController: spawn whenever budget allows (default behaviour)
    4. If budget exhausted: call stop_fn → opt.run() returns cleanly
    """

    def __init__(
        self,
        policy_manager: PolicyManager,
        agent: BaseAgent,
        agent_factory: Optional[Callable[[], BaseAgent]] = None,
        meta_controller: Optional[Any] = None,
    ):
        self._pm = policy_manager
        self._agent = agent
        self._agent_factory = agent_factory
        self._meta = meta_controller
        self._last_score: Optional[float] = None
        self._stop_fn: Optional[Callable[[], None]] = None
        self._metrics_history: List[EvalMetrics] = []
        self._reported: bool = False

    def set_stop_fn(self, fn: Callable[[], None]) -> None:
        """
        Register a callable invoked when the spawn budget is exhausted.
        Typically: cb.set_stop_fn(optimizer.stop)
        """
        self._stop_fn = fn

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is not None:
            self._last_score = eval_metrics.primary_score
            self._metrics_history.append(eval_metrics)

    def on_dormant(self, episode_id: int) -> None:
        # Step 1: rebalance weights
        self._pm.auto_update_weights()

        # Step 2: rollback if degraded
        if self._last_score is not None:
            rolled_back = self._pm.evolve(self._agent, self._last_score)
            if rolled_back:
                logger.info(
                    "Episode %d: DORMANT — rolled back to best checkpoint",
                    episode_id,
                )

        # Step 3: autonomous decisions (only when factory is available)
        # When no factory, this is natural convergence — print report once
        if self._agent_factory is None and not self._reported:
            self._reported = True
            self._print_training_report("natural_dormant")

        if self._agent_factory is not None:
            if self._meta is not None:
                action = self._meta.decide(self._metrics_history, self._pm.status())
                self._execute(action, episode_id)
            elif not self._pm.budget_exhausted:
                self._do_spawn(episode_id)

        # Step 4: terminate if budget exhausted
        if self._pm.budget_exhausted:
            logger.info(
                "Episode %d: spawn budget exhausted (%d/%d) — signalling stop",
                episode_id,
                self._pm._spawn_count,
                self._pm._max_spawns,
            )
            self._print_training_report("budget_exhausted")
            if self._stop_fn is not None:
                self._stop_fn()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_spawn(self, episode_id: int) -> None:
        """Clone best checkpoint into a new variant and add to ensemble."""
        shell = self._agent_factory()
        scale = self._pm.adaptive_noise_scale(self._metrics_history)
        variant = self._pm.spawn_variant(shell, noise_scale=scale)
        self._pm.add_agent(variant, weight=1.0)
        max_size = self._pm._max_ensemble_size
        if max_size is not None and self._pm.ensemble_size > max_size:
            self._pm.prune(bottom_k=1)
        logger.info(
            "Episode %d: spawned variant (noise=%.4f ensemble_size=%d spawns=%d/%s)",
            episode_id, scale, self._pm.ensemble_size,
            self._pm._spawn_count,
            str(self._pm._max_spawns) if self._pm._max_spawns is not None else "∞",
        )

    def _execute(self, action: MetaAction, episode_id: int) -> None:
        """Execute a MetaController decision."""
        if action == MetaAction.STOP:
            logger.info("Episode %d: MetaController → STOP", episode_id)
            self._print_training_report("meta_stop")
            if self._stop_fn is not None:
                self._stop_fn()
        elif action == MetaAction.SPAWN:
            if not self._pm.budget_exhausted:
                self._do_spawn(episode_id)
            else:
                logger.debug("Episode %d: MetaController → SPAWN but budget exhausted", episode_id)
        elif action == MetaAction.PRUNE:
            if self._pm.ensemble_size > 1:
                removed = self._pm.prune(bottom_k=1)
                logger.info(
                    "Episode %d: MetaController → PRUNE (removed %d agent)",
                    episode_id, len(removed),
                )
        elif action == MetaAction.NO_OP:
            logger.debug("Episode %d: MetaController → NO_OP", episode_id)

    def _print_training_report(self, reason: str) -> None:
        """Print a human-readable training summary to stdout."""
        report = self._pm.training_report()
        sep = "━" * 54

        reason_label = {
            "budget_exhausted": "Spawn budget exhausted",
            "meta_stop": "MetaController decision",
            "natural_dormant": "Plateau — model converged",
        }.get(reason, reason)

        lines = [
            "",
            sep,
            "  Training Complete",
            f"  Reason           : {reason_label}",
            sep,
        ]

        best = report.get("best_score")
        if best is not None:
            lines.append(f"  Best score       : {best:.4f}")

        val = report.get("best_val_score")
        if val is not None:
            lines.append(f"  Val score        : {val:.4f}")

        gap = report.get("best_generalization_gap")
        if gap is not None:
            lines.append(f"  Generalization   : {gap:.4f}  (train − val)")

        max_sp = report.get("max_spawns")
        sp = report.get("spawn_count", 0)
        if max_sp is not None:
            lines.append(f"  Spawns used      : {sp} / {max_sp}")
        else:
            lines.append(f"  Spawns used      : {sp}")

        lines.append(f"  Pruned agents    : {report.get('prune_count', 0)}")
        lines.append(f"  Ensemble size    : {report.get('ensemble_size', 0)}")

        regime = report.get("regime")
        if regime:
            lines.append(f"  Regime           : {regime}")

        agents = report.get("agents", [])
        if agents:
            lines.append("  Agents           :")
            for a in agents:
                ms = f"{a['mean_score']:.4f}" if a["mean_score"] is not None else "n/a"
                lines.append(f"    [{a['index']}] weight={a['weight']:.4f}  mean_score={ms}")

        lines += [sep, ""]
        print("\n".join(lines))

from __future__ import annotations

"""
NeuromodulatorSignal — translates training regime into global parameter changes
across the neuroevo stack, analogous to dopamine / norepinephrine / acetylcholine.

    RegimeDetector
          │
          ▼  (regime: "trending" | "ranging" | "volatile")
    NeuromodulatorSignal
          │
          ├── HebbianHook.hebbian_lr      (local plasticity)
          ├── GraphAgent entropy_coef     (exploration breadth)
          └── TopologyController         (grow / prune aggressiveness)

Biological analogues
--------------------
Regime      Neuromodulator   Effect
─────────── ──────────────── ──────────────────────────────────────────────
trending    Dopamine ↑       Reward signal strong → consolidate, exploit.
                             Lower hebbian_lr (don't overwrite good patterns).
                             Lower entropy (exploit known policy).
                             Raise prune threshold (trim redundant neurons).

ranging     Acetylcholine ↑  Plateau → shift attention to new structure.
            Dopamine ↓       Raise hebbian_lr (explore local correlations).
                             Raise entropy (broaden action distribution).
                             Lower grow threshold (allow topology to expand).

volatile    Norepinephrine ↑ High noise → arousal, cautious plasticity.
                             Lower hebbian_lr (don't lock in noisy patterns).
                             Raise entropy (explore to escape noise).
                             Raise prune threshold (don't prune during instability).

Usage::

    from tensor_optix.neuroevo.neuromodulator import NeuromodulatorSignal
    from tensor_optix.core.regime_detector import RegimeDetector

    signal = NeuromodulatorSignal(
        detector=RegimeDetector(),
        hebbian_hook=hook,          # optional
        agent=agent,                # optional — modulates entropy_coef
        topology_controller=tc,     # optional
    )

    # In your training loop, after each episode:
    signal.step(metrics_history)

    # Inspect current regime and active levels:
    print(signal.state)
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tensor_optix.core.regime_detector import RegimeDetector
    from tensor_optix.core.types import EvalMetrics
    from .hebbian import HebbianHook
    from .agent.graph_agent import GraphAgent
    from .controller.topology_controller import TopologyController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-regime multiplier presets
# Each preset maps regime -> scale factor applied to the *base* parameter value.
# ---------------------------------------------------------------------------

_DEFAULT_HEBBIAN_LR_SCALE = {
    "trending":  0.5,   # consolidate — slow down local plasticity
    "ranging":   2.0,   # plateau — speed up local structure exploration
    "volatile":  0.3,   # noisy — dampen plasticity to avoid locking in noise
}

_DEFAULT_ENTROPY_SCALE = {
    "trending":  0.5,   # exploit — tighten the policy
    "ranging":   2.0,   # explore — broaden the action distribution
    "volatile":  1.5,   # arousal — moderate exploration boost
}

_DEFAULT_GROW_GAP_SCALE = {
    "trending":  1.5,   # harder to trigger grow (less gap tolerance)
    "ranging":   0.5,   # easier to trigger grow (allow topology to expand)
    "volatile":  1.2,   # slightly suppress grow during instability
}

_DEFAULT_PRUNE_THRESHOLD_SCALE = {
    "trending":  1.5,   # prune more aggressively (trim redundant neurons)
    "ranging":   0.8,   # prune less (keep capacity for growth)
    "volatile":  0.3,   # barely prune (don't discard neurons mid-instability)
}


class NeuromodulatorSignal:
    """
    Reads the current training regime and modulates learning parameters
    across HebbianHook, GraphAgent, and TopologyController.

    Parameters
    ----------
    detector : RegimeDetector
        Detects the current regime from EvalMetrics history.
    hebbian_hook : HebbianHook, optional
        If provided, hebbian_lr is scaled per regime.
    agent : GraphAgent, optional
        If provided, the entropy_coef hyperparameter is scaled per regime.
    topology_controller : TopologyController, optional
        If provided, grow_gap_threshold and prune_neuron_threshold are scaled.
    hebbian_lr_scale : dict, optional
        Override per-regime scale factors for hebbian_lr.
        Keys: "trending", "ranging", "volatile". Values: float multipliers.
    entropy_scale : dict, optional
        Override per-regime scale factors for entropy_coef.
    grow_gap_scale : dict, optional
        Override per-regime scale factors for grow_gap_threshold.
    prune_threshold_scale : dict, optional
        Override per-regime scale factors for prune_neuron_threshold.
    """

    def __init__(
        self,
        detector: "RegimeDetector",
        hebbian_hook: Optional["HebbianHook"] = None,
        agent: Optional["GraphAgent"] = None,
        topology_controller: Optional["TopologyController"] = None,
        hebbian_lr_scale: Optional[dict] = None,
        entropy_scale: Optional[dict] = None,
        grow_gap_scale: Optional[dict] = None,
        prune_threshold_scale: Optional[dict] = None,
    ) -> None:
        self.detector = detector
        self.hebbian_hook = hebbian_hook
        self.agent = agent
        self.topology_controller = topology_controller

        self._hebbian_lr_scale = {**_DEFAULT_HEBBIAN_LR_SCALE, **(hebbian_lr_scale or {})}
        self._entropy_scale = {**_DEFAULT_ENTROPY_SCALE, **(entropy_scale or {})}
        self._grow_gap_scale = {**_DEFAULT_GROW_GAP_SCALE, **(grow_gap_scale or {})}
        self._prune_threshold_scale = {**_DEFAULT_PRUNE_THRESHOLD_SCALE, **(prune_threshold_scale or {})}

        # Snapshot base values on construction so scaling is always relative
        # to the original configuration, not the last modulated value.
        self._base_hebbian_lr: Optional[float] = (
            hebbian_hook.hebbian_lr if hebbian_hook is not None else None
        )
        self._base_entropy_coef: Optional[float] = (
            agent.get_hyperparams().params.get("entropy_coef")
            if agent is not None else None
        )
        self._base_grow_gap: Optional[float] = (
            topology_controller.grow_gap_threshold
            if topology_controller is not None else None
        )
        self._base_prune_threshold: Optional[float] = (
            topology_controller.prune_neuron_threshold
            if topology_controller is not None else None
        )

        self._current_regime: str = "ranging"  # neutral default
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def step(self, metrics_history: "List[EvalMetrics]") -> str:
        """
        Detect the current regime and apply parameter modulation.

        Call once per episode after metrics are appended.
        Returns the detected regime string.
        """
        regime = self.detector.detect(metrics_history)
        prev = self._current_regime
        self._current_regime = regime
        self._step_count += 1

        self._modulate_hebbian(regime)
        self._modulate_entropy(regime)
        self._modulate_topology(regime)

        if regime != prev:
            logger.info(
                "NeuromodulatorSignal: regime %s → %s (step %d)",
                prev, regime, self._step_count,
            )

        return regime

    # ------------------------------------------------------------------
    # Modulation targets
    # ------------------------------------------------------------------

    def _modulate_hebbian(self, regime: str) -> None:
        if self.hebbian_hook is None or self._base_hebbian_lr is None:
            return
        scale = self._hebbian_lr_scale.get(regime, 1.0)
        self.hebbian_hook.hebbian_lr = self._base_hebbian_lr * scale
        logger.debug(
            "NeuromodulatorSignal: hebbian_lr → %.6f (regime=%s, scale=%.2f)",
            self.hebbian_hook.hebbian_lr, regime, scale,
        )

    def _modulate_entropy(self, regime: str) -> None:
        if self.agent is None or self._base_entropy_coef is None:
            return
        scale = self._entropy_scale.get(regime, 1.0)
        hp = self.agent.get_hyperparams()
        new_coef = self._base_entropy_coef * scale
        hp.params["entropy_coef"] = new_coef
        self.agent.set_hyperparams(hp)
        logger.debug(
            "NeuromodulatorSignal: entropy_coef → %.6f (regime=%s, scale=%.2f)",
            new_coef, regime, scale,
        )

    def _modulate_topology(self, regime: str) -> None:
        if self.topology_controller is None:
            return
        if self._base_grow_gap is not None:
            gap_scale = self._grow_gap_scale.get(regime, 1.0)
            self.topology_controller.grow_gap_threshold = self._base_grow_gap * gap_scale
        if self._base_prune_threshold is not None:
            prune_scale = self._prune_threshold_scale.get(regime, 1.0)
            self.topology_controller.prune_neuron_threshold = (
                self._base_prune_threshold * prune_scale
            )
        logger.debug(
            "NeuromodulatorSignal: topology grow_gap=%.4f prune_thresh=%.2e (regime=%s)",
            self.topology_controller.grow_gap_threshold,
            self.topology_controller.prune_neuron_threshold,
            regime,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> str:
        """The most recently detected regime."""
        return self._current_regime

    @property
    def state(self) -> dict:
        """Snapshot of all currently active modulated values."""
        s: dict = {
            "regime": self._current_regime,
            "step": self._step_count,
        }
        if self.hebbian_hook is not None:
            s["hebbian_lr"] = self.hebbian_hook.hebbian_lr
        if self.agent is not None:
            s["entropy_coef"] = self.agent.get_hyperparams().params.get("entropy_coef")
        if self.topology_controller is not None:
            s["grow_gap_threshold"] = self.topology_controller.grow_gap_threshold
            s["prune_neuron_threshold"] = self.topology_controller.prune_neuron_threshold
        return s

    def reset_to_base(self) -> None:
        """Restore all modulated parameters to their original base values."""
        if self.hebbian_hook is not None and self._base_hebbian_lr is not None:
            self.hebbian_hook.hebbian_lr = self._base_hebbian_lr
        if self.agent is not None and self._base_entropy_coef is not None:
            hp = self.agent.get_hyperparams()
            hp.params["entropy_coef"] = self._base_entropy_coef
            self.agent.set_hyperparams(hp)
        if self.topology_controller is not None:
            if self._base_grow_gap is not None:
                self.topology_controller.grow_gap_threshold = self._base_grow_gap
            if self._base_prune_threshold is not None:
                self.topology_controller.prune_neuron_threshold = self._base_prune_threshold

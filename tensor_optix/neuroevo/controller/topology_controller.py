from __future__ import annotations

"""
TopologyController — a LoopCallback that evolves the NeuronGraph topology.

Grow / prune / merge decisions are driven by three independent statistical
signals computed from the live training stream.  The scheduler state is no
longer the grow trigger — topology changes only when the math says so.

Grow fires when ALL of:
  1. Improvement test   — slope β is NOT significantly positive (corrected t-test)
  2. Structure test     — lag-1 autocorrelation of residuals r1 > 2/√n (adaptive)
  3. Capacity test      — gradient utilisation > grow_grad_threshold

Prune (neuron) fires when:
  - avg importance < prune_neuron_threshold  (accumulated over min_prune_observations)
  - avg gradient magnitude < grad_eps        (neuron is truly dead, not just waiting)
  - neuron age > maturation_window           (prevents pruning freshly added neurons)

Prune (edge) fires when:
  - |w| < prune_edge_threshold for prune_edge_patience consecutive episodes

Merge fires when:
  - Pearson correlation of activation histories > merge_similarity_threshold
  - Both neurons have importance > prune_neuron_threshold (not dead — redundant)
"""

import logging
import random
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, LoopState, PolicySnapshot

from ..graph.neuron_graph import NeuronGraph
from ..graph.topology_ops import (
    add_free_edge,
    cosine_similarity_neurons,
    edge_importance,
    insert_neuron_on_edge,
    merge_neurons,
    neuron_importance,
    prune_edge,
    prune_neuron,
    split_neuron,
)

if TYPE_CHECKING:
    from tensor_optix.core.backoff_scheduler import BackoffScheduler

logger = logging.getLogger(__name__)


class TopologyController(LoopCallback):
    """
    Evolves a NeuronGraph's topology using statistical signals from the
    live training stream.  See module docstring for full signal definitions.

    Parameters
    ----------
    graph:
        The NeuronGraph owned by the GraphAgent.
    scheduler:
        The BackoffScheduler inside LoopController.
    grow_op:
        'insert_edge' | 'split_neuron' | 'add_edge'
    grow_grad_threshold:
        Minimum fraction of hidden neurons with |∇| > grad_eps to allow grow.
    grow_cooldown:
        Minimum episodes between successive GROW events.
    backoff_reset_factor:
        After GROW, multiply current backoff interval by this factor.
    grad_eps:
        Gradient magnitude below which a neuron is considered inactive.
    maturation_window:
        Episodes a neuron must exist before it is eligible for pruning.
    prune_edge_threshold:
        Edges with |w| below this are candidates for pruning.
    prune_edge_patience:
        Episodes an edge must remain below threshold before being pruned.
    prune_neuron_threshold:
        Neurons with avg importance below this are pruned.
    min_prune_observations:
        Minimum episodes of importance data required before neuron pruning.
    merge_similarity_threshold:
        Pearson correlation of activation histories above which neurons merge.
    merge_check_interval:
        Check for merge candidates every N episodes.
    act_history_len:
        Length of per-neuron activation history window used for merge detection.
    min_score_buffer:
        Minimum number of eval scores required before grow check runs.
    max_neurons:
        Hard cap on total neurons.
    min_hidden_neurons:
        Minimum hidden neurons kept during pruning.
    """

    def __init__(
        self,
        graph: NeuronGraph,
        scheduler: Optional["BackoffScheduler"] = None,
        grow_op: str = "insert_edge",
        grow_grad_threshold: float = 0.70,
        grow_cooldown: int = 20,
        backoff_reset_factor: float = 0.5,
        grad_eps: float = 1e-6,
        maturation_window: int = 30,
        prune_edge_threshold: float = 1e-3,
        prune_edge_patience: int = 10,
        prune_neuron_threshold: float = 1e-4,
        min_prune_observations: int = 10,
        merge_similarity_threshold: float = 0.95,
        merge_check_interval: int = 50,
        act_history_len: int = 100,
        min_score_buffer: int = 30,
        max_neurons: int = 256,
        min_hidden_neurons: int = 1,
    ) -> None:
        self.graph = graph
        self.scheduler = scheduler

        self.grow_op = grow_op
        self.grow_grad_threshold = grow_grad_threshold
        self.grow_cooldown = grow_cooldown
        self.backoff_reset_factor = backoff_reset_factor
        self.grad_eps = grad_eps
        self.maturation_window = maturation_window

        self.prune_edge_threshold = prune_edge_threshold
        self.prune_edge_patience = prune_edge_patience
        self.prune_neuron_threshold = prune_neuron_threshold
        self.min_prune_observations = min_prune_observations
        self.merge_similarity_threshold = merge_similarity_threshold
        self.merge_check_interval = merge_check_interval
        self.act_history_len = act_history_len
        self.min_score_buffer = min_score_buffer
        self.max_neurons = max_neurons
        self.min_hidden_neurons = min_hidden_neurons

        # ── Internal state ─────────────────────────────────────────────────
        self._episodes_since_grow: int = grow_cooldown   # ready immediately
        self._current_episode: int = 0

        # Score buffer — only populated on eval episodes (eval_metrics != None)
        self._score_buffer: deque = deque(maxlen=200)

        # Rolling gradient utilisation (one value per episode)
        self._grad_util_buffer: deque = deque(maxlen=100)

        # Per-neuron activation history for merge detection
        self._act_history: Dict[str, deque] = {}

        # Neuron birth episode — guards maturation window
        self._neuron_birth: Dict[str, int] = {}

        # Importance accumulator for prune decisions
        self._neuron_importance_accum: Dict[str, float] = defaultdict(float)
        self._accum_steps: int = 0

        # Edge patience counter
        self._edge_below_threshold: Dict[str, int] = defaultdict(int)

        # Diagnostic counters
        self._plateau_count: int = 0
        self._grow_count: int = 0
        self._prune_count: int = 0
        self._merge_count: int = 0

    def set_scheduler(self, scheduler: "BackoffScheduler") -> None:
        """Late-bind scheduler after LoopController is constructed."""
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # LoopCallback hooks
    # ------------------------------------------------------------------

    def on_episode_end(
        self, episode_id: int, eval_metrics: Optional[EvalMetrics]
    ) -> None:
        self._current_episode = episode_id
        self._episodes_since_grow += 1

        # Append score only when this episode was evaluated
        if eval_metrics is not None:
            self._score_buffer.append(eval_metrics.primary_score)

        # Gradient utilisation from the last backward pass (grads are live)
        self._grad_util_buffer.append(self._compute_grad_util())

        # Per-neuron activation history for merge detection
        self._accumulate_activations()

        # Importance accumulator for prune
        self._accumulate_neuron_importance()

        # ── Grow check ────────────────────────────────────────────────────
        if self._grow_signal_check():
            self._do_grow(episode_id)

        # ── Continuous edge pruning ───────────────────────────────────────
        self._check_edge_pruning()

        # ── Neuron pruning (fires when accumulator is full) ───────────────
        self._check_neuron_pruning(episode_id)

        # ── Periodic merge check ─────────────────────────────────────────
        if episode_id % self.merge_check_interval == 0:
            self._check_merge()

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        """
        Scheduler entered COOLING.  Log it and run an immediate grow check —
        a plateau is a hint, not a command.  The three-signal test still gates
        whether topology actually changes.
        """
        self._plateau_count += 1
        logger.debug(
            "TopologyController: plateau #%d at episode %d — running signal check",
            self._plateau_count, episode_id,
        )
        if self._grow_signal_check():
            self._do_grow(episode_id)

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        self._plateau_count = 0

    # ------------------------------------------------------------------
    # Grow — signal-gated
    # ------------------------------------------------------------------

    def _grow_signal_check(self) -> bool:
        """
        Returns True only when all three statistical conditions hold:
          1. Score trend is not significantly improving  (corrected t-test)
          2. Residual autocorrelation is significant     (adaptive threshold)
          3. Gradient utilisation is high                (network at capacity)
        """
        if self._episodes_since_grow < self.grow_cooldown:
            return False
        if self.graph.n_neurons() >= self.max_neurons:
            return False
        if len(self._score_buffer) < self.min_score_buffer:
            return False

        scores = np.array(self._score_buffer, dtype=float)
        n = len(scores)
        t = np.arange(n, dtype=float)

        # 1. Linear trend fit
        beta = float(np.cov(t, scores)[0, 1] / (np.var(t) + 1e-10))
        intercept = float(np.mean(scores) - beta * np.mean(t))
        residuals = scores - (beta * t + intercept)

        # 2. Lag-1 autocorrelation of residuals
        e = residuals - np.mean(residuals)
        r1 = float(np.clip(
            np.sum(e[:-1] * e[1:]) / (np.sum(e ** 2) + 1e-10),
            -1.0, 1.0,
        ))

        # 3. Autocorrelation-corrected effective sample size
        n_eff = max(2.0, n * (1.0 - abs(r1)) / (1.0 + abs(r1) + 1e-10))

        # Slope significance: β > 2 * SE_corrected → improving
        se_beta = float(np.std(residuals)) / (float(np.std(t)) * float(np.sqrt(n_eff)) + 1e-10)
        improving = beta > 2.0 * se_beta

        # Residual structure: r1 > 2/√n  (adaptive significance threshold)
        structured = r1 > (2.0 / (float(np.sqrt(n)) + 1e-10))

        # Gradient utilisation: mean over recent window
        grad_util = float(np.mean(self._grad_util_buffer)) if self._grad_util_buffer else 0.0
        saturated = grad_util > self.grow_grad_threshold

        should = (not improving) and structured and saturated
        if should:
            logger.info(
                "TopologyController: grow signal — r1=%.3f n_eff=%.1f "
                "β=%.5f se=%.5f grad_util=%.2f",
                r1, n_eff, beta, se_beta, grad_util,
            )
        return should

    def _do_grow(self, episode_id: int) -> None:
        op = self.grow_op
        new_nid: Optional[str] = None

        if op == "insert_edge":
            new_nid = self._grow_insert_edge()
        elif op == "split_neuron":
            new_nid = self._grow_split_neuron()
        elif op == "add_edge":
            self._grow_add_edge()
        else:
            logger.warning("TopologyController: unknown grow_op '%s'", op)
            return

        self._grow_count += 1
        self._episodes_since_grow = 0

        if new_nid is not None:
            self._neuron_birth[new_nid] = episode_id

        # Clear importance accumulators so new neurons start with a clean slate
        self._neuron_importance_accum.clear()
        self._accum_steps = 0

        logger.info(
            "TopologyController: GROW (%s) #%d at episode %d — "
            "%d neurons, %d edges",
            op, self._grow_count, episode_id,
            self.graph.n_neurons(), self.graph.n_edges(),
        )
        self._reset_scheduler()

    def _grow_insert_edge(self) -> Optional[str]:
        edges = self.graph.all_edges()
        if not edges:
            return None
        best = max(edges, key=lambda e: abs(e.weight.item()))
        return insert_neuron_on_edge(self.graph, best.edge_id)

    def _grow_split_neuron(self) -> Optional[str]:
        hidden = self.graph.hidden_ids
        if not hidden:
            return None
        best_id = max(
            hidden,
            key=lambda nid: abs(self.graph.get_neuron(nid)._current.item()),
        )
        _, new_id = split_neuron(self.graph, best_id)
        return new_id

    def _grow_add_edge(self) -> None:
        all_ids = self.graph.all_neuron_ids()
        if len(all_ids) < 2:
            return
        src = random.choice(all_ids)
        dst = random.choice([n for n in all_ids if n != src])
        add_free_edge(self.graph, src=src, dst=dst, delay=random.randint(1, 3))

    def _reset_scheduler(self) -> None:
        if self.scheduler is None:
            return
        if self.backoff_reset_factor == 0.0:
            self.scheduler.record_restart()
        else:
            new_interval = max(
                1,
                int(self.scheduler.current_interval * self.backoff_reset_factor),
            )
            self.scheduler._current_interval = new_interval
            from tensor_optix.core.types import LoopState as LS
            self.scheduler._state = LS.ACTIVE
            self.scheduler._consecutive_non_improvements = 0

    # ------------------------------------------------------------------
    # Gradient utilisation
    # ------------------------------------------------------------------

    def _compute_grad_util(self) -> float:
        """
        Fraction of hidden neurons whose total gradient magnitude exceeds
        grad_eps.  Read from .grad after the most recent backward pass.
        """
        hidden = self.graph.hidden_ids
        if not hidden:
            return 0.0
        saturated = 0
        for nid in hidden:
            neuron = self.graph.get_neuron(nid)
            mag = 0.0
            if neuron.bias.grad is not None:
                mag += float(neuron.bias.grad.abs().item())
            for edge in self.graph.edges_into(nid) + self.graph.edges_from(nid):
                if edge.weight.grad is not None:
                    mag += float(edge.weight.grad.abs().item())
            if mag > self.grad_eps:
                saturated += 1
        return saturated / len(hidden)

    # ------------------------------------------------------------------
    # Prune — edge
    # ------------------------------------------------------------------

    def _check_edge_pruning(self) -> None:
        to_prune = []
        for edge in self.graph.all_edges():
            if abs(edge.weight.item()) < self.prune_edge_threshold:
                self._edge_below_threshold[edge.edge_id] += 1
                if self._edge_below_threshold[edge.edge_id] >= self.prune_edge_patience:
                    to_prune.append(edge.edge_id)
            else:
                self._edge_below_threshold[edge.edge_id] = 0

        for eid in to_prune:
            logger.debug(
                "TopologyController: pruning edge %s (below threshold for %d eps)",
                eid[:8], self.prune_edge_patience,
            )
            prune_edge(self.graph, eid)
            self._edge_below_threshold.pop(eid, None)
            self._prune_count += 1

    # ------------------------------------------------------------------
    # Prune — neuron
    # ------------------------------------------------------------------

    def _check_neuron_pruning(self, episode_id: int) -> bool:
        if self._accum_steps < self.min_prune_observations:
            return False
        hidden = list(self.graph.hidden_ids)
        if len(hidden) <= self.min_hidden_neurons:
            return False

        pruned_any = False
        for nid in hidden:
            if len(self.graph.hidden_ids) <= self.min_hidden_neurons:
                break

            # Maturation guard — never prune a freshly added neuron
            birth = self._neuron_birth.get(nid, 0)
            if episode_id - birth < self.maturation_window:
                continue

            avg_importance = self._neuron_importance_accum[nid] / self._accum_steps

            # Also require near-zero gradient (not just low current importance)
            neuron = self.graph.get_neuron(nid)
            grad_mag = 0.0
            if neuron.bias.grad is not None:
                grad_mag += float(neuron.bias.grad.abs().item())

            if avg_importance < self.prune_neuron_threshold and grad_mag < self.grad_eps:
                logger.info(
                    "TopologyController: pruning neuron %s "
                    "(importance=%.2e grad=%.2e age=%d)",
                    nid[:8], avg_importance, grad_mag, episode_id - birth,
                )
                prune_neuron(self.graph, nid, redistribute=True)
                self._neuron_importance_accum.pop(nid, None)
                self._neuron_birth.pop(nid, None)
                self._act_history.pop(nid, None)
                self._prune_count += 1
                pruned_any = True

        self._neuron_importance_accum.clear()
        self._accum_steps = 0
        return pruned_any

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _check_merge(self) -> None:
        hidden = self.graph.hidden_ids
        if len(hidden) < 2:
            return

        for i, nid_a in enumerate(hidden):
            for nid_b in hidden[i + 1:]:
                hist_a = self._act_history.get(nid_a)
                hist_b = self._act_history.get(nid_b)
                if not hist_a or not hist_b or len(hist_a) < 10 or len(hist_b) < 10:
                    continue

                n = min(len(hist_a), len(hist_b))
                a = np.array(list(hist_a)[-n:])
                b = np.array(list(hist_b)[-n:])

                # Pearson correlation over activation history window
                if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])

                if corr > self.merge_similarity_threshold:
                    # Both neurons must be non-trivially important — not dead
                    imp_a = self._neuron_importance_accum.get(nid_a, 0.0) / max(self._accum_steps, 1)
                    imp_b = self._neuron_importance_accum.get(nid_b, 0.0) / max(self._accum_steps, 1)
                    if imp_a > self.prune_neuron_threshold and imp_b > self.prune_neuron_threshold:
                        logger.info(
                            "TopologyController: merging %s + %s (corr=%.3f)",
                            nid_a[:8], nid_b[:8], corr,
                        )
                        merge_neurons(self.graph, nid_a, nid_b)
                        self._act_history.pop(nid_b, None)
                        self._merge_count += 1
                        return  # one merge per check to keep graph stable

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    def _accumulate_neuron_importance(self) -> None:
        for nid in self.graph.hidden_ids:
            self._neuron_importance_accum[nid] += neuron_importance(self.graph, nid)
        self._accum_steps += 1

    def _accumulate_activations(self) -> None:
        for nid in self.graph.hidden_ids:
            if nid not in self._act_history:
                self._act_history[nid] = deque(maxlen=self.act_history_len)
            neuron = self.graph.get_neuron(nid)
            val = (
                float(neuron._current.item())
                if neuron._current.numel() == 1
                else float(neuron._current.abs().mean())
            )
            self._act_history[nid].append(val)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        grad_util = float(np.mean(self._grad_util_buffer)) if self._grad_util_buffer else 0.0
        return {
            "grow_count":   self._grow_count,
            "prune_count":  self._prune_count,
            "merge_count":  self._merge_count,
            "plateau_count": self._plateau_count,
            "n_neurons":    self.graph.n_neurons(),
            "n_edges":      self.graph.n_edges(),
            "score_buffer": len(self._score_buffer),
            "grad_util":    round(grad_util, 3),
        }

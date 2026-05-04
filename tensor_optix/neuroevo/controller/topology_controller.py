from __future__ import annotations

"""
TopologyController — a LoopCallback that evolves the NeuronGraph topology.

Hooks:
  on_plateau  (COOLING state) — consider GROW or PRUNE based on metrics
  on_episode_end              — accumulate neuron activation stats for importance scoring
  on_improvement              — reset plateau counter

Grow resets the BackoffScheduler to ACTIVE via record_restart() so the loop
immediately treats the grown network as a fresh start. A partial-reset factor
(backoff_reset_factor) decays the current interval rather than zeroing it,
preserving memory that this region has been hard.

Prune fires when:
  - neuron importance < prune_neuron_threshold (dead neurons)
  - edge |w| < prune_edge_threshold for prune_edge_patience consecutive evals

Grow fires when:
  - COOLING and no overfitting (gap < grow_gap_threshold)
  - Respects grow_cooldown to prevent infinite grow loops

Merge fires when:
  - cosine similarity between two hidden neurons > merge_similarity_threshold
  - Only checked every merge_check_interval episodes (expensive)
"""

import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
    Evolves a NeuronGraph's topology in response to training dynamics.

    Parameters
    ----------
    graph:
        The NeuronGraph owned by the GraphAgent.
    scheduler:
        The BackoffScheduler inside LoopController. Pass via
        ``loop_controller.scheduler`` after constructing LoopController.
    grow_op:
        Which grow operation to use on plateau.
        'insert_edge'  — insert a relay neuron on the highest-weight edge
        'split_neuron' — split the most active hidden neuron
        'add_edge'     — add a free recurrent edge between two random neurons
    grow_gap_threshold:
        Max generalization gap allowed before GROW is suppressed (overfitting).
    grow_cooldown:
        Minimum episodes between successive GROW events.
    backoff_reset_factor:
        After GROW, multiply current backoff interval by this factor (0.5 = halve).
        0.0 = full reset (record_restart). 1.0 = no reset.
    prune_edge_threshold:
        Edges with |w| below this are candidates for pruning.
    prune_edge_patience:
        Episodes an edge must remain below threshold before being pruned.
    prune_neuron_threshold:
        Neurons with importance below this are pruned.
    min_prune_observations:
        Minimum episodes of importance data required before neuron pruning fires.
        Prevents newly inserted neurons from being immediately pruned.
    merge_similarity_threshold:
        Cosine similarity above which two neurons are merged.
    merge_check_interval:
        Check for merge candidates every N episodes.
    max_neurons:
        Hard cap on total neurons. GROW suppressed above this.
    min_hidden_neurons:
        Minimum hidden neurons. PRUNE suppressed below this.
    """

    def __init__(
        self,
        graph: NeuronGraph,
        scheduler: Optional["BackoffScheduler"] = None,
        grow_op: str = "insert_edge",
        grow_gap_threshold: float = 0.15,
        grow_cooldown: int = 20,
        backoff_reset_factor: float = 0.5,
        prune_edge_threshold: float = 1e-3,
        prune_edge_patience: int = 10,
        prune_neuron_threshold: float = 1e-4,
        min_prune_observations: int = 10,
        merge_similarity_threshold: float = 0.95,
        merge_check_interval: int = 50,
        max_neurons: int = 256,
        min_hidden_neurons: int = 1,
    ) -> None:
        self.graph = graph
        self.scheduler = scheduler

        self.grow_op = grow_op
        self.grow_gap_threshold = grow_gap_threshold
        self.grow_cooldown = grow_cooldown
        self.backoff_reset_factor = backoff_reset_factor

        self.prune_edge_threshold = prune_edge_threshold
        self.prune_edge_patience = prune_edge_patience
        self.prune_neuron_threshold = prune_neuron_threshold
        self.min_prune_observations = min_prune_observations
        self.merge_similarity_threshold = merge_similarity_threshold
        self.merge_check_interval = merge_check_interval
        self.max_neurons = max_neurons
        self.min_hidden_neurons = min_hidden_neurons

        # Internal state
        self._episodes_since_grow: int = grow_cooldown  # ready immediately
        self._edge_below_threshold: Dict[str, int] = defaultdict(int)  # edge_id -> count
        self._neuron_importance_accum: Dict[str, float] = defaultdict(float)
        self._accum_steps: int = 0
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
        self._episodes_since_grow += 1
        self._accumulate_neuron_importance()

        # Continuous edge pruning check
        self._check_edge_pruning()

        # Periodic merge check
        if episode_id % self.merge_check_interval == 0:
            self._check_merge()

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        """Fires on COOLING — the primary topology decision point."""
        self._plateau_count += 1
        logger.info(
            "TopologyController: plateau #%d at episode %d (state=%s)",
            self._plateau_count, episode_id, state.name,
        )

        # Check overfitting via last eval metrics — prune if overfitting
        # We don't have direct access to metrics_history here, so we rely
        # on the accumulated importance signal to drive pruning.
        pruned = self._check_neuron_pruning()

        if not pruned and self._should_grow():
            self._do_grow(episode_id)

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        self._plateau_count = 0

    # ------------------------------------------------------------------
    # Grow
    # ------------------------------------------------------------------

    def _should_grow(self) -> bool:
        if self._episodes_since_grow < self.grow_cooldown:
            logger.debug(
                "TopologyController: grow suppressed (cooldown %d/%d)",
                self._episodes_since_grow, self.grow_cooldown,
            )
            return False
        if self.graph.n_neurons() >= self.max_neurons:
            logger.debug("TopologyController: grow suppressed (max_neurons=%d)", self.max_neurons)
            return False
        return True

    def _do_grow(self, episode_id: int) -> None:
        op = self.grow_op
        success = False

        if op == "insert_edge":
            success = self._grow_insert_edge()
        elif op == "split_neuron":
            success = self._grow_split_neuron()
        elif op == "add_edge":
            success = self._grow_add_edge()
        else:
            logger.warning("TopologyController: unknown grow_op '%s'", op)

        if success:
            self._grow_count += 1
            self._episodes_since_grow = 0
            # Reset importance accumulators so new neurons get a clean slate
            # before the pruner can evaluate them.
            self._neuron_importance_accum.clear()
            self._accum_steps = 0
            logger.info(
                "TopologyController: GROW (%s) #%d at episode %d — "
                "graph now has %d neurons, %d edges",
                op, self._grow_count, episode_id,
                self.graph.n_neurons(), self.graph.n_edges(),
            )
            self._reset_scheduler()

    def _grow_insert_edge(self) -> bool:
        edges = self.graph.all_edges()
        if not edges:
            return False
        # Pick highest-weight non-zero edge
        best = max(edges, key=lambda e: abs(e.weight.item()))
        insert_neuron_on_edge(self.graph, best.edge_id)
        return True

    def _grow_split_neuron(self) -> bool:
        hidden = self.graph.hidden_ids
        if not hidden:
            return False
        # Split most active hidden neuron
        best_id = max(
            hidden,
            key=lambda nid: abs(self.graph.get_neuron(nid)._current.item()),
        )
        split_neuron(self.graph, best_id)
        return True

    def _grow_add_edge(self) -> bool:
        all_ids = self.graph.all_neuron_ids()
        if len(all_ids) < 2:
            return False
        src = random.choice(all_ids)
        dst = random.choice([n for n in all_ids if n != src])
        delay = random.randint(1, 3)
        add_free_edge(self.graph, src=src, dst=dst, delay=delay)
        return True

    def _reset_scheduler(self) -> None:
        if self.scheduler is None:
            return
        if self.backoff_reset_factor == 0.0:
            self.scheduler.record_restart()
        else:
            # Partial reset: decay current interval
            new_interval = max(
                1,
                int(self.scheduler.current_interval * self.backoff_reset_factor),
            )
            self.scheduler._current_interval = new_interval
            from tensor_optix.core.types import LoopState as LS
            self.scheduler._state = LS.ACTIVE
            self.scheduler._consecutive_non_improvements = 0

    # ------------------------------------------------------------------
    # Prune
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
            logger.debug("TopologyController: pruning edge %s (below threshold for %d episodes)",
                         eid[:8], self.prune_edge_patience)
            prune_edge(self.graph, eid)
            self._edge_below_threshold.pop(eid, None)
            self._prune_count += 1

    def _check_neuron_pruning(self) -> bool:
        if self._accum_steps < self.min_prune_observations:
            return False
        hidden = list(self.graph.hidden_ids)
        if len(hidden) <= self.min_hidden_neurons:
            return False

        pruned_any = False
        for nid in hidden:
            if len(self.graph.hidden_ids) <= self.min_hidden_neurons:
                break
            avg_importance = self._neuron_importance_accum[nid] / self._accum_steps
            if avg_importance < self.prune_neuron_threshold:
                logger.info(
                    "TopologyController: pruning neuron %s (importance=%.2e)",
                    nid[:8], avg_importance,
                )
                prune_neuron(self.graph, nid, redistribute=True)
                self._neuron_importance_accum.pop(nid, None)
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
        # O(n^2) scan — only on hidden neurons, bounded by max_neurons
        for i, nid_a in enumerate(hidden):
            for nid_b in hidden[i + 1:]:
                sim = cosine_similarity_neurons(self.graph, nid_a, nid_b)
                if sim > self.merge_similarity_threshold:
                    logger.info(
                        "TopologyController: merging neurons %s + %s (cos_sim=%.3f)",
                        nid_a[:8], nid_b[:8], sim,
                    )
                    merge_neurons(self.graph, nid_a, nid_b)
                    self._merge_count += 1
                    return  # one merge per check to keep graph stable

    # ------------------------------------------------------------------
    # Accumulate importance over episodes
    # ------------------------------------------------------------------

    def _accumulate_neuron_importance(self) -> None:
        for nid in self.graph.hidden_ids:
            self._neuron_importance_accum[nid] += neuron_importance(self.graph, nid)
        self._accum_steps += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "grow_count": self._grow_count,
            "prune_count": self._prune_count,
            "merge_count": self._merge_count,
            "plateau_count": self._plateau_count,
            "n_neurons": self.graph.n_neurons(),
            "n_edges": self.graph.n_edges(),
        }

from __future__ import annotations

"""
TopologyController — a LoopCallback that evolves NeuronGraph topology.

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
  - neuron_a.can_merge_with(neuron_b) — same type, verified
  - Pearson correlation of signed activation histories > merge_similarity_threshold
  - Both neurons have importance > prune_neuron_threshold (not dead — redundant)

Multi-region support:
  The controller operates on a Dict[str, NeuronGraph].  Signal buffers and
  cooldowns are per-region; score buffer and grow/prune/merge signal math are
  shared (eval score reflects the whole agent).  Each region evolves
  independently — a saturated encoder region can grow without triggering
  growth in the decision region.

  Use TopologyController.for_graph(graph) for a single NeuronGraph.
  Use TopologyController.for_brain(brain) for a BrainNetwork.
"""

import logging
import random
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, LoopState, PolicySnapshot

from .topology_callback import TopologyCallback
from ..graph.neuron_graph import NeuronGraph
from ..graph.topology_ops import (
    add_free_edge,
    edge_importance,
    insert_neuron_on_edge,
    merge_neurons,
    prune_edge,
    prune_neuron,
    split_neuron,
)

if TYPE_CHECKING:
    from tensor_optix.core.backoff_scheduler import BackoffScheduler
    from tensor_optix.neuroevo.brain_network import BrainNetwork

logger = logging.getLogger(__name__)


class TopologyController(LoopCallback):
    """
    Evolves one or more NeuronGraphs using statistical signals from the live
    training stream.  See module docstring for signal definitions.

    Construct via:
        TopologyController.for_graph(graph, scheduler, **kwargs)
        TopologyController.for_brain(brain, scheduler, **kwargs)

    Parameters
    ----------
    regions:
        Dict mapping region name → NeuronGraph.  All regions share the same
        score buffer and statistical thresholds; cooldowns and grad-util
        buffers are per-region.
    scheduler:
        The LoopController's BackoffScheduler (forwards restarts to its
        ConvergenceMonitor) or the monitor itself (controller.monitor).
    grow_op:
        'insert_edge' | 'split_neuron' | 'add_edge'
    grow_grad_threshold:
        Minimum fraction of hidden neurons with |∇| > grad_eps to allow grow.
    grow_cooldown:
        Minimum episodes between successive GROW events (per region).
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
        Pearson correlation of signed activation histories above which neurons merge.
    merge_check_interval:
        Check for merge candidates every N episodes.
    act_history_len:
        Length of per-neuron activation history window used for merge detection.
    min_score_buffer:
        Minimum number of eval scores required before grow check runs.
    max_neurons:
        Hard cap on total neurons per region, 256 by default. Pass None for
        no ceiling.
    min_hidden_neurons:
        Minimum hidden neurons kept during pruning per region.
    """

    def __init__(
        self,
        regions: Dict[str, NeuronGraph],
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
        max_neurons: Optional[int] = 256,
        min_hidden_neurons: int = 1,
        topology_callbacks: Optional[List[TopologyCallback]] = None,
    ) -> None:
        # Accept a single NeuronGraph as a convenience shorthand for {"default": graph}
        if isinstance(regions, NeuronGraph):
            regions = {"default": regions}
        self._regions = regions
        self._brain: Optional["BrainNetwork"] = None
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

        # ── Shared state (whole-agent) ─────────────────────────────────────
        self._current_episode: int = 0
        self._score_buffer: deque = deque(maxlen=200)

        # ── Per-region state ───────────────────────────────────────────────
        # Cooldown counter and grad-util buffer are per-region so that a
        # saturated encoder doesn't trigger growth in the decision region.
        self._episodes_since_grow: Dict[str, int] = {
            name: grow_cooldown for name in regions
        }
        self._grad_util_buffers: Dict[str, deque] = {
            name: deque(maxlen=100) for name in regions
        }
        # Every forward the graph runs (forward() or forward_batch()) is
        # recorded, so merge detection sees many activation samples per call
        # instead of one read from neuron state.
        for graph in regions.values():
            graph.enable_activation_trace(act_history_len)

        # The remaining accumulators are keyed by neuron UUID so they work
        # across all regions without collision.
        self._act_history: Dict[str, deque] = {}
        self._neuron_birth: Dict[str, int] = {}
        self._neuron_importance_accum: Dict[str, float] = defaultdict(float)
        self._accum_steps: int = 0
        self._edge_below_threshold: Dict[str, int] = defaultdict(int)

        # ── Diagnostic counters ────────────────────────────────────────────
        self._plateau_count: int = 0
        self._grow_count: int = 0
        self._prune_count: int = 0
        self._merge_count: int = 0

        self._topology_callbacks: List[TopologyCallback] = list(topology_callbacks or [])

    @property
    def grow_gap_threshold(self) -> float:
        """Alias of grow_grad_threshold (the name NeuromodulatorSignal
        modulates). A property rather than a copied attribute, so writing it
        actually changes the threshold the grow test uses."""
        return self.grow_grad_threshold

    @grow_gap_threshold.setter
    def grow_gap_threshold(self, value: float) -> None:
        self.grow_grad_threshold = value

    def _fire_topology(self, event: str, *args) -> None:
        """Same per-callback exception isolation as LoopController._fire —
        a broken observer must never interrupt training."""
        for cb in self._topology_callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception as e:
                logger.warning("TopologyCallback %s raised during %s: %s", cb, event, e)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_graph(
        cls,
        graph: NeuronGraph,
        scheduler: Optional["BackoffScheduler"] = None,
        **kwargs,
    ) -> "TopologyController":
        """Single-graph convenience constructor."""
        return cls({"main": graph}, scheduler=scheduler, **kwargs)

    @classmethod
    def for_brain(
        cls,
        brain: "BrainNetwork",
        scheduler: Optional["BackoffScheduler"] = None,
        **kwargs,
    ) -> "TopologyController":
        """BrainNetwork convenience constructor — one region per brain region."""
        tc = cls(brain.regions, scheduler=scheduler, **kwargs)
        tc._brain = brain
        return tc

    # ------------------------------------------------------------------
    # Backward-compatibility: expose .graph for single-region users
    # ------------------------------------------------------------------

    @property
    def graph(self) -> NeuronGraph:
        """Returns the 'main' region graph.  Raises if multi-region."""
        if "main" not in self._regions:
            raise AttributeError(
                "TopologyController has multiple regions — use ._regions[name] directly."
            )
        return self._regions["main"]

    def set_scheduler(self, scheduler: "BackoffScheduler") -> None:
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # LoopCallback hooks
    # ------------------------------------------------------------------

    def on_episode_end(
        self, episode_id: int, eval_metrics: Optional[EvalMetrics]
    ) -> None:
        self._current_episode = episode_id

        if eval_metrics is not None:
            self._score_buffer.append(eval_metrics.primary_score)

        for region_name, graph in self._regions.items():
            self._episodes_since_grow[region_name] += 1
            hidden = graph.hidden_ids

            # Single shared edge pass for both grad-util and importance
            grad_mag, total_w = self._episode_neuron_stats(graph, hidden)

            self._grad_util_buffers[region_name].append(
                self._compute_grad_util(hidden, grad_mag)
            )
            samples = self._collect_activations(graph, hidden)
            self._accumulate_activations(hidden, graph, samples)
            self._accumulate_neuron_importance(hidden, total_w, graph, samples)

            if self._grow_signal_check(region_name):
                self._do_grow(episode_id, region_name, graph)

            self._check_edge_pruning(episode_id, region_name, graph)
            self._check_neuron_pruning(episode_id, graph, region_name)

        if episode_id % self.merge_check_interval == 0:
            for region_name, graph in self._regions.items():
                self._check_merge(episode_id, region_name, graph)

    def on_plateau(self, episode_id: int, state: LoopState) -> None:
        self._plateau_count += 1
        logger.debug(
            "TopologyController: plateau #%d at episode %d — running signal check",
            self._plateau_count, episode_id,
        )
        for region_name, graph in self._regions.items():
            if self._grow_signal_check(region_name):
                self._do_grow(episode_id, region_name, graph)

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        self._plateau_count = 0

    # ------------------------------------------------------------------
    # Grow — signal-gated
    # ------------------------------------------------------------------

    def _grow_signal_check(self, region_name: str) -> bool:
        """
        Returns True only when all three statistical conditions hold:
          1. Score trend is not significantly improving  (corrected t-test)
          2. Residual autocorrelation is significant     (adaptive threshold)
          3. Gradient utilisation is high                (region at capacity)
        """
        graph = self._regions[region_name]

        if self._episodes_since_grow[region_name] < self.grow_cooldown:
            remaining = self.grow_cooldown - self._episodes_since_grow[region_name]
            if remaining % 5 == 0 and remaining > 0:
                logger.info(
                    "EVO warming up — grow/prune eligible in %d episodes (cooldown)", remaining
                )
            return False
        if self.max_neurons is not None and graph.n_neurons() >= self.max_neurons:
            return False
        if len(self._score_buffer) < self.min_score_buffer:
            have = len(self._score_buffer)
            if have % 5 == 0:
                logger.info(
                    "EVO: collecting baseline scores (%d/%d)", have, self.min_score_buffer
                )
            return False

        scores = np.array(self._score_buffer, dtype=float)
        n = len(scores)

        # With fewer than 3 scores, statistical tests are meaningless — allow grow.
        if n < 3:
            return True

        t = np.arange(n, dtype=float)

        beta = float(np.cov(t, scores)[0, 1] / (np.var(t) + 1e-10))
        intercept = float(np.mean(scores) - beta * np.mean(t))
        residuals = scores - (beta * t + intercept)

        e = residuals - np.mean(residuals)
        r1 = float(np.clip(
            np.sum(e[:-1] * e[1:]) / (np.sum(e ** 2) + 1e-10),
            -1.0, 1.0,
        ))

        n_eff = max(2.0, n * (1.0 - abs(r1)) / (1.0 + abs(r1) + 1e-10))
        se_beta = float(np.std(residuals)) / (float(np.std(t)) * float(np.sqrt(n_eff)) + 1e-10)
        improving = beta > 2.0 * se_beta
        structured = r1 > (2.0 / (float(np.sqrt(n)) + 1e-10))

        util_buf = self._grad_util_buffers[region_name]
        grad_util = float(np.mean(util_buf)) if util_buf else 0.0
        saturated = grad_util > self.grow_grad_threshold

        should = (not improving) and structured and saturated
        if should:
            logger.info(
                "TopologyController[%s]: grow signal — r1=%.3f n_eff=%.1f "
                "β=%.5f se=%.5f grad_util=%.2f",
                region_name, r1, n_eff, beta, se_beta, grad_util,
            )
        return should

    def _do_grow(self, episode_id: int, region_name: str, graph: NeuronGraph) -> None:
        op = self.grow_op
        new_nid: Optional[str] = None

        if op == "insert_edge":
            new_nid = self._grow_insert_edge(graph)
        elif op == "split_neuron":
            new_nid = self._grow_split_neuron(graph)
        elif op == "add_edge":
            self._grow_add_edge(graph)
        else:
            logger.warning("TopologyController: unknown grow_op '%s'", op)
            return

        self._grow_count += 1
        self._episodes_since_grow[region_name] = 0

        if new_nid is not None:
            self._neuron_birth[new_nid] = episode_id

        self._neuron_importance_accum.clear()
        self._accum_steps = 0

        logger.info(
            "TopologyController[%s]: GROW (%s) #%d at episode %d — "
            "%d neurons, %d edges",
            region_name, op, self._grow_count, episode_id,
            graph.n_neurons(), graph.n_edges(),
        )
        graph.invalidate_compile()
        self._reset_scheduler()
        self._fire_topology(
            "on_grow", region_name, episode_id, op, new_nid,
            graph.n_neurons(), graph.n_edges(),
        )

    def _grow_insert_edge(self, graph: NeuronGraph) -> Optional[str]:
        table = graph.edge_table()
        if not table.edge_ids:
            return None
        best_id = table.edge_ids[int(table.weight.abs().argmax())]
        return insert_neuron_on_edge(graph, best_id)

    def _grow_split_neuron(self, graph: NeuronGraph) -> Optional[str]:
        hidden = graph.hidden_ids
        if not hidden:
            return None
        pos = {nid: i for i, nid in enumerate(graph.neuron_index)}
        cols = torch.tensor([pos[nid] for nid in hidden], device=graph._device)
        act = graph.activation_snapshot().abs().index_select(0, cols)
        best_id = hidden[int(act.argmax())]
        _, new_id = split_neuron(graph, best_id)
        return new_id

    def _grow_add_edge(self, graph: NeuronGraph) -> None:
        all_ids = graph.all_neuron_ids()
        if len(all_ids) < 2:
            return
        src = random.choice(all_ids)
        dst = random.choice([n for n in all_ids if n != src])
        add_free_edge(graph, src=src, dst=dst, delay=random.randint(1, 3))

    def _reset_scheduler(self) -> None:
        """
        A topology change makes a new network: give it fresh convergence
        patience. Accepts a BackoffScheduler (the one a LoopController was
        built with -- it forwards the restart to the loop's monitor) or the
        loop's ConvergenceMonitor itself (``controller.monitor``).
        """
        if self.scheduler is None:
            return
        if not hasattr(self.scheduler, "record_restart"):
            # ConvergenceMonitor: keeps its peak, restarts trends and patience.
            self.scheduler.restart()
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
            monitor = getattr(self.scheduler, "_monitor", None)
            if monitor is not None:
                monitor.restart()

    # ------------------------------------------------------------------
    # Gradient utilisation
    # ------------------------------------------------------------------

    def _compute_grad_util(
        self, hidden: List[str], grad_mag: Dict[str, float]
    ) -> float:
        """Fraction of hidden neurons with total |grad| > grad_eps."""
        if not hidden:
            return 0.0
        saturated = sum(1 for m in grad_mag.values() if m > self.grad_eps)
        return saturated / len(hidden)

    # ------------------------------------------------------------------
    # Prune — edge
    # ------------------------------------------------------------------

    def _check_edge_pruning(self, episode_id: int, region_name: str, graph: NeuronGraph) -> None:
        # Effective weights, all edges in one tensor op: in dale_mode="softplus"
        # the raw parameter of a near-zero edge is a large negative number, not
        # a small one. One host transfer for the whole graph.
        table = graph.edge_table()
        below = (table.weight.abs() < self.prune_edge_threshold).nonzero().flatten().tolist()
        below_ids = [table.edge_ids[i] for i in below]
        below_set = set(below_ids)
        to_prune = []
        for eid in below_ids:
            self._edge_below_threshold[eid] += 1
            if self._edge_below_threshold[eid] >= self.prune_edge_patience:
                to_prune.append(eid)
        for eid in [e for e in self._edge_below_threshold if e not in below_set]:
            del self._edge_below_threshold[eid]

        for eid in to_prune:
            logger.debug(
                "TopologyController: pruning edge %s (below threshold for %d eps)",
                eid[:8], self.prune_edge_patience,
            )
            prune_edge(graph, eid)
            self._edge_below_threshold.pop(eid, None)
            self._prune_count += 1
            self._fire_topology(
                "on_prune_edge", region_name, episode_id, eid,
                graph.n_neurons(), graph.n_edges(),
            )

        if to_prune:
            graph.invalidate_compile()

    # ------------------------------------------------------------------
    # Prune — neuron
    # ------------------------------------------------------------------

    def _check_neuron_pruning(self, episode_id: int, graph: NeuronGraph, region_name: str = "default") -> bool:
        if self._accum_steps < self.min_prune_observations:
            return False
        hidden = list(graph.hidden_ids)
        if len(hidden) <= self.min_hidden_neurons:
            return False

        # Gradient on ALL of a neuron's own parameters. Reading only .bias
        # made GRU/LSTM neurons (which don't use the bias) look permanently dead.
        own_grad = dict(zip(hidden, self._own_grad_magnitude(graph, hidden).tolist()))

        pruned_any = False
        for nid in hidden:
            if len(graph.hidden_ids) <= self.min_hidden_neurons:
                break

            birth = self._neuron_birth.get(nid, 0)
            if episode_id - birth < self.maturation_window:
                continue

            avg_importance = self._neuron_importance_accum[nid] / self._accum_steps
            grad_mag = own_grad[nid]

            if avg_importance < self.prune_neuron_threshold and grad_mag < self.grad_eps:
                logger.info(
                    "TopologyController: pruning neuron %s "
                    "(importance=%.2e grad=%.2e age=%d)",
                    nid[:8], avg_importance, grad_mag, episode_id - birth,
                )
                prune_neuron(graph, nid, redistribute=True)
                if self._brain is not None:
                    self._brain.notify_neuron_pruned(region_name, nid)
                self._neuron_importance_accum.pop(nid, None)
                self._neuron_birth.pop(nid, None)
                self._act_history.pop(nid, None)
                self._prune_count += 1
                pruned_any = True
                self._fire_topology(
                    "on_prune_neuron", region_name, episode_id, nid,
                    graph.n_neurons(), graph.n_edges(),
                )

        self._neuron_importance_accum.clear()
        self._accum_steps = 0
        if pruned_any:
            graph.invalidate_compile()
        return pruned_any

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _check_merge(self, episode_id: int, region_name: str, graph: NeuronGraph) -> None:
        hidden = graph.hidden_ids
        if len(hidden) < 2:
            return

        for i, nid_a in enumerate(hidden):
            neuron_a = graph.get_neuron(nid_a)
            for nid_b in hidden[i + 1:]:
                neuron_b = graph.get_neuron(nid_b)

                # Protocol check — same type required
                if not neuron_a.can_merge_with(neuron_b):
                    continue

                hist_a = self._act_history.get(nid_a)
                hist_b = self._act_history.get(nid_b)
                if not hist_a or not hist_b or len(hist_a) < 10 or len(hist_b) < 10:
                    continue

                n = min(len(hist_a), len(hist_b))
                a = np.array(list(hist_a)[-n:])
                b = np.array(list(hist_b)[-n:])

                # Signed Pearson correlation (Option B — validated by math tests)
                if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])

                if corr > self.merge_similarity_threshold:
                    imp_a = self._neuron_importance_accum.get(nid_a, 0.0) / max(self._accum_steps, 1)
                    imp_b = self._neuron_importance_accum.get(nid_b, 0.0) / max(self._accum_steps, 1)
                    if imp_a > self.prune_neuron_threshold and imp_b > self.prune_neuron_threshold:
                        logger.info(
                            "TopologyController: merging %s + %s (corr=%.3f)",
                            nid_a[:8], nid_b[:8], corr,
                        )
                        merge_neurons(graph, nid_a, nid_b)
                        self._act_history.pop(nid_b, None)
                        self._merge_count += 1
                        graph.invalidate_compile()
                        self._fire_topology(
                            "on_merge", region_name, episode_id, nid_a, nid_b,
                            graph.n_neurons(), graph.n_edges(),
                        )
                        return

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    def _collect_activations(
        self, graph: NeuronGraph, hidden: List[str]
    ) -> Dict[str, List[float]]:
        """
        {hidden neuron id: every activation recorded since the last call}.

        The graph records each forward it runs (forward() and forward_batch(),
        or your own via graph.record_activations), so one call yields as many
        samples as there were steps, not one. If nothing ran since the last
        call, the latest activation of each neuron is used. One host transfer.
        """
        if not hidden:
            return {}
        trace = graph.drain_activation_trace()
        if trace is None:
            trace = graph.activation_snapshot().unsqueeze(0)
        pos = {nid: i for i, nid in enumerate(graph.neuron_index)}
        cols = torch.tensor([pos[nid] for nid in hidden], device=trace.device)
        rows = trace.index_select(1, cols)[-self.act_history_len:]      # [S, H]
        return dict(zip(hidden, rows.t().tolist()))

    @staticmethod
    def _own_grad_magnitude(graph: NeuronGraph, nids: List[str]) -> torch.Tensor:
        """Σ|grad| over every parameter of each neuron (GRU/LSTM gates included),
        as one [len(nids)] tensor. No host sync."""
        zero = torch.zeros((), device=graph._device)
        parts = []
        for nid in nids:
            total = zero
            for p in graph.get_neuron(nid).parameters():
                if p.grad is not None:
                    total = total + p.grad.abs().sum()
            parts.append(total)
        if not parts:
            return torch.zeros(0, device=graph._device)
        return torch.stack(parts)

    def _accumulate_neuron_importance(
        self,
        hidden: List[str],
        total_w: Dict[str, float],
        graph: NeuronGraph,
        samples: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """I(v) = Σ|w_e| * (mean|h|/d + ε), mean taken over this call's samples."""
        if not hidden:
            self._accum_steps += 1
            return
        if samples is None:
            samples = self._collect_activations(graph, hidden)
        for nid in hidden:
            vals = samples[nid]
            mean_abs = sum(abs(v) for v in vals) / max(len(vals), 1)
            dim = graph.get_neuron(nid)._hidden_dim
            self._neuron_importance_accum[nid] += total_w[nid] * (mean_abs / dim + 1e-8)
        self._accum_steps += 1

    def _accumulate_activations(
        self,
        hidden: List[str],
        graph: NeuronGraph,
        samples: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Append every sampled signed activation to each hidden neuron's history deque."""
        if not hidden:
            return
        if samples is None:
            samples = self._collect_activations(graph, hidden)
        for nid in hidden:
            if nid not in self._act_history:
                self._act_history[nid] = deque(maxlen=self.act_history_len)
            # Signed values — sign matters for merge detection (Option B)
            self._act_history[nid].extend(samples[nid])

    # ------------------------------------------------------------------
    # Shared per-episode edge pass (one pass serves both grad-util + importance)
    # ------------------------------------------------------------------

    def _episode_neuron_stats(
        self,
        graph: NeuronGraph,
        hidden: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Two dicts keyed by hidden neuron id:
          grad_mag:  Σ|grad| over the neuron's own parameters + Σ|edge grad|
                     over incident edges
          total_w:   Σ|effective weight| over incident edges

        Computed on the whole edge table at once (scatter-add per neuron) and
        moved to the host in a single transfer, so the cost in GPU round trips
        does not grow with the number of edges.
        """
        if not hidden:
            return {}, {}
        table = graph.edge_table()
        n = graph.n_neurons()
        abs_w, abs_g = table.weight.abs(), table.grad.abs()
        per_neuron_w = torch.zeros(n, device=abs_w.device)
        per_neuron_g = torch.zeros(n, device=abs_w.device)
        for ends in (table.src, table.dst):
            per_neuron_w = per_neuron_w.index_add(0, ends, abs_w)
            per_neuron_g = per_neuron_g.index_add(0, ends, abs_g)

        pos = {nid: i for i, nid in enumerate(graph.neuron_index)}
        cols = torch.tensor([pos[nid] for nid in hidden], device=abs_w.device)
        grad = per_neuron_g.index_select(0, cols) + self._own_grad_magnitude(graph, hidden)
        weight = per_neuron_w.index_select(0, cols)
        grad_l, weight_l = torch.stack([grad, weight]).tolist()

        return dict(zip(hidden, grad_l)), dict(zip(hidden, weight_l))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        total_neurons = sum(g.n_neurons() for g in self._regions.values())
        total_edges   = sum(g.n_edges()   for g in self._regions.values())
        mean_util = float(np.mean([
            float(np.mean(buf)) if buf else 0.0
            for buf in self._grad_util_buffers.values()
        ]))
        return {
            "grow_count":    self._grow_count,
            "prune_count":   self._prune_count,
            "merge_count":   self._merge_count,
            "plateau_count": self._plateau_count,
            "n_neurons":     total_neurons,
            "n_edges":       total_edges,
            "n_regions":     len(self._regions),
            "score_buffer":  len(self._score_buffer),
            "grad_util":     round(mean_util, 3),
        }

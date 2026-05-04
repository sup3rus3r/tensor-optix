"""
Tests for TopologyController — verifies trigger logic and grow/prune behaviour.
"""
import pytest
import torch

from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.graph.topology_ops import add_free_edge
from tensor_optix.neuroevo.controller.topology_controller import TopologyController
from tensor_optix.core.types import EvalMetrics, LoopState, PolicySnapshot, HyperparamSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph_with_hidden():
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    h0 = g.add_neuron(role="hidden", activation="tanh")
    o0 = g.add_neuron(role="output", activation="linear")
    e0 = g.add_edge(i0, h0, weight=1.0, delay=0)
    e1 = g.add_edge(h0, o0, weight=1.0, delay=0)
    # Warm up the graph so neurons have non-zero activations
    g(torch.tensor([1.0]))
    return g, i0, h0, o0, e0, e1


def fake_snapshot() -> PolicySnapshot:
    return PolicySnapshot(
        snapshot_id="test",
        eval_metrics=EvalMetrics(primary_score=1.0, metrics={}, episode_id=0),
        hyperparams=HyperparamSet(params={}, episode_id=0),
        weights_path="/tmp/test.pt",
        episode_id=0,
    )


# ---------------------------------------------------------------------------
# Grow on plateau
# ---------------------------------------------------------------------------

class TestGrowOnPlateau:

    def test_insert_edge_grow_increases_neurons(self):
        g, *_ = make_graph_with_hidden()
        n_before = g.n_neurons()
        ctrl = TopologyController(g, grow_op="insert_edge", grow_cooldown=0)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        assert g.n_neurons() > n_before

    def test_split_neuron_grow_increases_neurons(self):
        g, *_ = make_graph_with_hidden()
        n_before = g.n_neurons()
        ctrl = TopologyController(g, grow_op="split_neuron", grow_cooldown=0)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        assert g.n_neurons() > n_before

    def test_add_edge_grow_increases_edges(self):
        g, *_ = make_graph_with_hidden()
        e_before = g.n_edges()
        ctrl = TopologyController(g, grow_op="add_edge", grow_cooldown=0)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        assert g.n_edges() > e_before

    def test_grow_cooldown_suppresses_second_grow(self):
        g, *_ = make_graph_with_hidden()
        ctrl = TopologyController(g, grow_op="insert_edge", grow_cooldown=5)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        n_after_first = g.n_neurons()
        # Simulate 2 episodes (< cooldown of 5)
        ctrl.on_episode_end(11, None)
        ctrl.on_episode_end(12, None)
        ctrl.on_plateau(episode_id=12, state=LoopState.COOLING)
        assert g.n_neurons() == n_after_first  # no second grow

    def test_grow_fires_after_cooldown_expires(self):
        g, *_ = make_graph_with_hidden()
        ctrl = TopologyController(g, grow_op="insert_edge", grow_cooldown=3)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        n_after_first = g.n_neurons()
        for ep in range(11, 15):  # 4 episodes > cooldown
            ctrl.on_episode_end(ep, None)
        # Replenish activations
        g(torch.tensor([1.0]))
        ctrl.on_plateau(episode_id=15, state=LoopState.COOLING)
        assert g.n_neurons() > n_after_first

    def test_max_neurons_suppresses_grow(self):
        g, *_ = make_graph_with_hidden()
        ctrl = TopologyController(g, grow_op="split_neuron", grow_cooldown=0, max_neurons=3)
        n_before = g.n_neurons()  # already 3
        ctrl.on_plateau(episode_id=1, state=LoopState.COOLING)
        assert g.n_neurons() == n_before


# ---------------------------------------------------------------------------
# Scheduler reset after grow
# ---------------------------------------------------------------------------

class TestSchedulerReset:

    def test_partial_reset_halves_interval(self):
        g, *_ = make_graph_with_hidden()

        class FakeScheduler:
            _current_interval = 16
            current_interval = 16
            _consecutive_non_improvements = 10
            _state = None
            def record_restart(self): pass

        fs = FakeScheduler()
        ctrl = TopologyController(
            g, grow_op="insert_edge", grow_cooldown=0, backoff_reset_factor=0.5
        )
        ctrl.set_scheduler(fs)
        ctrl.on_plateau(episode_id=5, state=LoopState.COOLING)
        assert fs._current_interval == 8

    def test_full_reset_calls_record_restart(self):
        g, *_ = make_graph_with_hidden()

        class FakeScheduler:
            current_interval = 16
            _consecutive_non_improvements = 10
            restarted = False
            def record_restart(self):
                self.restarted = True

        fs = FakeScheduler()
        ctrl = TopologyController(
            g, grow_op="insert_edge", grow_cooldown=0, backoff_reset_factor=0.0
        )
        ctrl.set_scheduler(fs)
        ctrl.on_plateau(episode_id=5, state=LoopState.COOLING)
        assert fs.restarted


# ---------------------------------------------------------------------------
# Edge pruning
# ---------------------------------------------------------------------------

class TestEdgePruning:

    def test_dead_edge_pruned_after_patience(self):
        g, i0, h0, o0, e0, e1 = make_graph_with_hidden()
        # Add a near-zero edge
        dead_eid = g.add_edge(i0, o0, weight=1e-9, delay=0)
        ctrl = TopologyController(
            g, prune_edge_threshold=1e-4, prune_edge_patience=3, grow_cooldown=999
        )
        for ep in range(4):
            ctrl.on_episode_end(ep, None)
        assert dead_eid not in [e.edge_id for e in g.all_edges()]

    def test_healthy_edge_not_pruned(self):
        g, i0, h0, o0, e0, e1 = make_graph_with_hidden()
        ctrl = TopologyController(
            g, prune_edge_threshold=1e-4, prune_edge_patience=3, grow_cooldown=999
        )
        for ep in range(10):
            ctrl.on_episode_end(ep, None)
        assert e0 in [e.edge_id for e in g.all_edges()]


# ---------------------------------------------------------------------------
# Neuron pruning
# ---------------------------------------------------------------------------

class TestNeuronPruning:

    def test_dead_neuron_pruned_on_plateau(self):
        g, i0, h0, o0, e0, e1 = make_graph_with_hidden()
        dead_id = g.add_neuron(role="hidden", activation="linear")
        # Dead neuron has no edges and zero activation → importance = 0
        ctrl = TopologyController(
            g,
            prune_neuron_threshold=1e-6,
            min_prune_observations=5,
            grow_cooldown=999,
        )
        # Accumulate importance over enough episodes to pass min_prune_observations
        for ep in range(10):
            g(torch.tensor([1.0]))
            ctrl.on_episode_end(ep, None)
        ctrl.on_plateau(episode_id=10, state=LoopState.COOLING)
        assert dead_id not in g.all_neuron_ids()

    def test_min_hidden_neurons_prevents_all_pruning(self):
        g, i0, h0, o0, e0, e1 = make_graph_with_hidden()
        ctrl = TopologyController(
            g,
            prune_neuron_threshold=1.0,  # prune everything
            min_hidden_neurons=1,
            min_prune_observations=3,
            grow_cooldown=999,
        )
        for ep in range(10):
            ctrl.on_episode_end(ep, None)
        ctrl.on_plateau(episode_id=5, state=LoopState.COOLING)
        assert len(g.hidden_ids) >= 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_dict_has_expected_keys():
    g, *_ = make_graph_with_hidden()
    ctrl = TopologyController(g, grow_cooldown=0)
    ctrl.on_plateau(episode_id=1, state=LoopState.COOLING)
    stats = ctrl.stats
    for key in ("grow_count", "prune_count", "merge_count", "plateau_count", "n_neurons", "n_edges"):
        assert key in stats


def test_improvement_resets_plateau_count():
    g, *_ = make_graph_with_hidden()
    ctrl = TopologyController(g, grow_cooldown=0)
    ctrl.on_plateau(episode_id=1, state=LoopState.COOLING)
    ctrl.on_plateau(episode_id=2, state=LoopState.COOLING)
    assert ctrl._plateau_count == 2
    ctrl.on_improvement(fake_snapshot())
    assert ctrl._plateau_count == 0

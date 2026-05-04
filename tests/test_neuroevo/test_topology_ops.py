"""
Tests for topology_ops — focus on function-preserving guarantees.

Each grow operation is tested by:
1. Running the graph on a fixed input before mutation
2. Applying the operation
3. Running the same input again
4. Asserting output is approximately identical (atol=1e-5)
"""
import pytest
import torch
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.graph.topology_ops import (
    add_free_edge,
    add_input_neuron,
    cosine_similarity_neurons,
    edge_importance,
    insert_neuron_on_edge,
    merge_neurons,
    neuron_importance,
    prune_edge,
    prune_neuron,
    split_neuron,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_graph_2in_1out():
    """2 inputs -> 1 output, all linear zero-bias."""
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    i1 = g.add_neuron(role="input", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    e0 = g.add_edge(i0, o0, weight=2.0, delay=0)
    e1 = g.add_edge(i1, o0, weight=3.0, delay=0)
    return g, i0, i1, o0, e0, e1


def linear_graph_chain():
    """input -> hidden -> output, linear, zero-bias, weights=1."""
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    h0 = g.add_neuron(role="hidden", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    e0 = g.add_edge(i0, h0, weight=1.0, delay=0)
    e1 = g.add_edge(h0, o0, weight=1.0, delay=0)
    return g, i0, h0, o0, e0, e1


def run(g, obs):
    g.reset_state()
    return g(obs).detach().clone()


# ---------------------------------------------------------------------------
# insert_neuron_on_edge
# ---------------------------------------------------------------------------

class TestInsertNeuronOnEdge:

    def test_function_preserving_feedforward(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        obs = torch.tensor([1.5, 2.5])
        before = run(g, obs)
        insert_neuron_on_edge(g, e0)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-5), f"{before} vs {after}"

    def test_neuron_count_increases_by_one(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        n_before = g.n_neurons()
        insert_neuron_on_edge(g, e0)
        assert g.n_neurons() == n_before + 1

    def test_edge_count_net_increases_by_one(self):
        # Removes 1 edge, adds 2 → net +1
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        e_before = g.n_edges()
        insert_neuron_on_edge(g, e0)
        assert g.n_edges() == e_before + 1

    def test_function_preserving_with_weight(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        obs = torch.tensor([4.0])
        before = run(g, obs)
        insert_neuron_on_edge(g, e1)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-5)

    def test_delay_split_on_recurrent_edge(self):
        """Inserting on a delay=2 edge splits delay into 1+1."""
        g = NeuronGraph()
        i0 = g.add_neuron(role="input", activation="linear")
        o0 = g.add_neuron(role="output", activation="linear")
        g.add_edge(i0, o0, weight=1.0, delay=0)
        eid = g.add_edge(o0, o0, weight=0.5, delay=2)
        insert_neuron_on_edge(g, eid)
        # Verify graph still runs without error
        g.reset_state()
        out = g(torch.tensor([1.0]))
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# split_neuron
# ---------------------------------------------------------------------------

class TestSplitNeuron:

    def test_function_preserving(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        obs = torch.tensor([3.0])
        before = run(g, obs)
        split_neuron(g, h0)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-5), f"{before} vs {after}"

    def test_neuron_count_increases_by_one(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        n_before = g.n_neurons()
        split_neuron(g, h0)
        assert g.n_neurons() == n_before + 1

    def test_both_copies_same_output_initially(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        obs = torch.tensor([2.0])
        original_id, new_id = split_neuron(g, h0)
        g.reset_state()
        g(obs)
        h_orig = g.get_neuron(original_id)._current.item()
        h_new = g.get_neuron(new_id)._current.item()
        assert abs(h_orig - h_new) < 1e-5


# ---------------------------------------------------------------------------
# add_input_neuron
# ---------------------------------------------------------------------------

class TestAddInputNeuron:

    def test_output_unchanged_for_existing_inputs(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        obs = torch.tensor([1.0, 2.0])
        before = run(g, obs)
        add_input_neuron(g)
        # Must pad obs with zero for new neuron
        obs_padded = torch.tensor([1.0, 2.0, 0.0])
        after = run(g, obs_padded)
        assert torch.allclose(before, after, atol=1e-5)

    def test_input_count_increases(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        n_in = len(g.input_ids)
        add_input_neuron(g)
        assert len(g.input_ids) == n_in + 1

    def test_new_input_edges_are_zero_weight(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        new_id = add_input_neuron(g)
        for e in g.edges_from(new_id):
            assert abs(e.weight.item()) < 1e-9


# ---------------------------------------------------------------------------
# add_free_edge
# ---------------------------------------------------------------------------

class TestAddFreeEdge:

    def test_output_unchanged_at_insertion(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        obs = torch.tensor([5.0])
        before = run(g, obs)
        add_free_edge(g, src=i0, dst=o0, delay=1)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-5)

    def test_edge_weight_is_zero(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        eid = add_free_edge(g, src=i0, dst=o0, delay=1)
        assert abs(g.get_edge(eid).weight.item()) < 1e-9


# ---------------------------------------------------------------------------
# prune_edge
# ---------------------------------------------------------------------------

class TestPruneEdge:

    def test_edge_removed(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        n_before = g.n_edges()
        prune_edge(g, e0)
        assert g.n_edges() == n_before - 1

    def test_graph_still_runs_after_prune(self):
        g, i0, i1, o0, e0, e1 = linear_graph_2in_1out()
        prune_edge(g, e0)
        out = g(torch.tensor([1.0, 1.0]))
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# prune_neuron
# ---------------------------------------------------------------------------

class TestPruneNeuron:

    def test_neuron_removed(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        n_before = g.n_neurons()
        prune_neuron(g, h0, redistribute=True)
        assert g.n_neurons() == n_before - 1

    def test_output_approximately_preserved_linear(self):
        """With linear activation, redistribution is exact."""
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        with torch.no_grad():
            g.get_neuron(h0).bias.fill_(0.0)
        obs = torch.tensor([7.0])
        before = run(g, obs)
        prune_neuron(g, h0, redistribute=True)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-4), f"{before} vs {after}"

    def test_cannot_prune_input_neuron(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        with pytest.raises(ValueError, match="input"):
            prune_neuron(g, i0)

    def test_cannot_prune_output_neuron(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        with pytest.raises(ValueError, match="output"):
            prune_neuron(g, o0)


# ---------------------------------------------------------------------------
# merge_neurons
# ---------------------------------------------------------------------------

class TestMergeNeurons:

    def _make_twin_graph(self):
        """Two identical hidden neurons with same weights — classic merge candidate."""
        g = NeuronGraph()
        i0 = g.add_neuron(role="input", activation="linear")
        h0 = g.add_neuron(role="hidden", activation="linear")
        h1 = g.add_neuron(role="hidden", activation="linear")
        o0 = g.add_neuron(role="output", activation="linear")
        g.add_edge(i0, h0, weight=1.0, delay=0)
        g.add_edge(i0, h1, weight=1.0, delay=0)
        g.add_edge(h0, o0, weight=0.5, delay=0)
        g.add_edge(h1, o0, weight=0.5, delay=0)
        with torch.no_grad():
            g.get_neuron(h0).bias.fill_(0.0)
            g.get_neuron(h1).bias.fill_(0.0)
        return g, i0, h0, h1, o0

    def test_neuron_count_decreases(self):
        g, i0, h0, h1, o0 = self._make_twin_graph()
        n_before = g.n_neurons()
        merge_neurons(g, h0, h1)
        assert g.n_neurons() == n_before - 1

    def test_surviving_neuron_is_a(self):
        g, i0, h0, h1, o0 = self._make_twin_graph()
        result = merge_neurons(g, h0, h1)
        assert result == h0
        assert h1 not in g.all_neuron_ids()

    def test_output_approximately_preserved(self):
        g, i0, h0, h1, o0 = self._make_twin_graph()
        obs = torch.tensor([4.0])
        before = run(g, obs)
        merge_neurons(g, h0, h1)
        after = run(g, obs)
        assert torch.allclose(before, after, atol=1e-4), f"{before} vs {after}"

    def test_merge_same_neuron_noop(self):
        g, i0, h0, h1, o0 = self._make_twin_graph()
        n_before = g.n_neurons()
        merge_neurons(g, h0, h0)
        assert g.n_neurons() == n_before

    def test_cannot_merge_input_neuron(self):
        g, i0, h0, h1, o0 = self._make_twin_graph()
        with pytest.raises(ValueError):
            merge_neurons(g, i0, h0)


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

class TestImportanceScoring:

    def test_neuron_importance_positive(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        g.reset_state()
        g(torch.tensor([1.0]))
        imp = neuron_importance(g, h0)
        assert imp >= 0.0

    def test_edge_importance_absolute_weight(self):
        g, i0, h0, o0, e0, e1 = linear_graph_chain()
        assert abs(edge_importance(g, e0) - 1.0) < 1e-6

    def test_cosine_similarity_identical(self):
        g, i0, h0, h1, o0 = TestMergeNeurons()._make_twin_graph()
        g.reset_state()
        g(torch.tensor([2.0]))
        sim = cosine_similarity_neurons(g, h0, h1)
        assert abs(sim - 1.0) < 1e-5

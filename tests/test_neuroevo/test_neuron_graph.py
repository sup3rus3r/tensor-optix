import pytest
import torch
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph


def make_simple_graph():
    """2 inputs -> 1 hidden -> 1 output, all linear, zero bias."""
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    i1 = g.add_neuron(role="input", activation="linear")
    h0 = g.add_neuron(role="hidden", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    g.add_edge(i0, h0, weight=1.0, delay=0)
    g.add_edge(i1, h0, weight=1.0, delay=0)
    g.add_edge(h0, o0, weight=1.0, delay=0)
    return g, i0, i1, h0, o0


def test_forward_feedforward():
    g, *_ = make_simple_graph()
    obs = torch.tensor([2.0, 3.0])
    out = g(obs)
    assert out.shape == (1,)
    assert torch.isclose(out[0], torch.tensor(5.0))


def test_wrong_obs_dim_raises():
    g, *_ = make_simple_graph()
    with pytest.raises(ValueError, match="obs dim"):
        g(torch.tensor([1.0, 2.0, 3.0]))


def test_add_remove_edge():
    g, i0, i1, h0, o0 = make_simple_graph()
    initial_edges = g.n_edges()
    eid = g.add_edge(i0, o0, weight=0.5, delay=0)
    assert g.n_edges() == initial_edges + 1
    g.remove_edge(eid)
    assert g.n_edges() == initial_edges


def test_remove_nonexistent_edge_noop():
    g, *_ = make_simple_graph()
    g.remove_edge("nonexistent-edge-id")  # should not raise


def test_add_remove_neuron():
    g, *_ = make_simple_graph()
    n_before = g.n_neurons()
    nid = g.add_neuron(role="hidden", activation="tanh")
    assert g.n_neurons() == n_before + 1
    g.remove_neuron(nid)
    assert g.n_neurons() == n_before


def test_remove_neuron_cleans_edges():
    g, i0, i1, h0, o0 = make_simple_graph()
    extra = g.add_neuron(role="hidden", activation="linear")
    g.add_edge(h0, extra, weight=1.0, delay=0)
    g.add_edge(extra, o0, weight=1.0, delay=0)
    edges_before = g.n_edges()
    g.remove_neuron(extra)
    assert g.n_edges() == edges_before - 2


def test_recurrent_edge_delay():
    """Recurrent edge reads from history (delay=1), not current timestep."""
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    g.add_edge(i0, o0, weight=1.0, delay=0)
    # Self-recurrent with delay=1
    g.add_edge(o0, o0, weight=0.5, delay=1)

    # t=0: input=2, recurrent history=0 → output = 1*2 + 0.5*0 = 2
    out0 = g(torch.tensor([2.0]))
    assert torch.isclose(out0[0], torch.tensor(2.0), atol=1e-5)

    # t=1: input=2, recurrent history=2 → output = 1*2 + 0.5*2 = 3
    out1 = g(torch.tensor([2.0]))
    assert torch.isclose(out1[0], torch.tensor(3.0), atol=1e-5)


def test_reset_state_clears_recurrent():
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    g.add_edge(i0, o0, weight=1.0, delay=0)
    g.add_edge(o0, o0, weight=0.5, delay=1)

    g(torch.tensor([2.0]))  # t=0
    g.reset_state()
    out = g(torch.tensor([2.0]))  # history reset → same as t=0
    assert torch.isclose(out[0], torch.tensor(2.0), atol=1e-5)


def test_topological_order_respects_feedforward():
    """Neurons connected in sequence should execute in correct order."""
    g = NeuronGraph()
    i0 = g.add_neuron(role="input", activation="linear")
    h1 = g.add_neuron(role="hidden", activation="linear")
    h2 = g.add_neuron(role="hidden", activation="linear")
    o0 = g.add_neuron(role="output", activation="linear")
    g.add_edge(i0, h1, weight=2.0, delay=0)
    g.add_edge(h1, h2, weight=3.0, delay=0)
    g.add_edge(h2, o0, weight=1.0, delay=0)
    # output = 2*3*1 * input = 6 * input
    out = g(torch.tensor([1.0]))
    assert torch.isclose(out[0], torch.tensor(6.0), atol=1e-5)


def test_parameter_count():
    g, *_ = make_simple_graph()
    params = list(g.parameters())
    # 4 biases + 3 edge weights = 7 parameters
    assert len(params) == 7


def test_edges_into_and_from():
    g, i0, i1, h0, o0 = make_simple_graph()
    assert len(g.edges_into(h0)) == 2
    assert len(g.edges_from(h0)) == 1

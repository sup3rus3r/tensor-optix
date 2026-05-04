"""Tests for HebbianHook — local Hebbian learning."""
import pytest
import torch
from tensor_optix.neuroevo.hebbian import HebbianHook
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.brain_network import BrainNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_two_neuron_graph(w: float = 0.5) -> tuple[NeuronGraph, str, str, str]:
    """
    One input -> one output, single edge with weight w.
    Returns (graph, input_id, output_id, edge_id).
    """
    g = NeuronGraph()
    inp = g.add_neuron(role="input", activation="linear")
    out = g.add_neuron(role="output", activation="linear")
    eid = g.add_edge(src=inp, dst=out, weight=w)
    return g, inp, out, eid


def _fire(graph: NeuronGraph, obs: torch.Tensor) -> None:
    """Run one forward pass on the graph."""
    graph(obs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_single_graph():
    g, *_ = make_two_neuron_graph()
    hook = HebbianHook(g, hebbian_lr=1e-3, weight_decay=0.0)
    assert len(hook.graphs) == 1


def test_construction_list_of_graphs():
    g1, *_ = make_two_neuron_graph()
    g2, *_ = make_two_neuron_graph()
    hook = HebbianHook([g1, g2])
    assert len(hook.graphs) == 2


def test_from_brain_covers_all_regions():
    brain = BrainNetwork()
    for name in ["sensory", "memory", "motor"]:
        g, *_ = make_two_neuron_graph()
        brain.add_region(name, g)
    hook = HebbianHook.from_brain(brain)
    assert len(hook.graphs) == 3


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------

def test_record_increments_steps():
    g, *_ = make_two_neuron_graph()
    hook = HebbianHook(g)
    _fire(g, torch.tensor([1.0]))
    hook.record()
    assert hook.n_steps_recorded == 1
    _fire(g, torch.tensor([1.0]))
    hook.record()
    assert hook.n_steps_recorded == 2


def test_record_accumulates_coactivation():
    g, inp, out, eid = make_two_neuron_graph(w=1.0)
    hook = HebbianHook(g, hebbian_lr=0.0, weight_decay=0.0)
    _fire(g, torch.tensor([2.0]))
    hook.record()
    coact = hook.mean_coactivation()
    assert eid in coact
    # h_pre=2, h_post=2 (weight=1, bias=0, linear): product = 4
    assert coact[eid] == pytest.approx(4.0)


def test_record_no_steps_coactivation_empty():
    g, *_ = make_two_neuron_graph()
    hook = HebbianHook(g)
    assert hook.mean_coactivation() == {}


# ---------------------------------------------------------------------------
# apply() — weight update direction
# ---------------------------------------------------------------------------

def test_apply_strengthens_correlated_edge():
    """
    Both neurons fire positively → co-activation > 0 → weight should increase.
    """
    g, inp, out, eid = make_two_neuron_graph(w=0.0)
    hook = HebbianHook(g, hebbian_lr=0.1, weight_decay=0.0)

    # Force both neurons to have positive _current
    g.get_neuron(inp)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([1.0])
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() > 0.0


def test_apply_weakens_anticorrelated_edge():
    """
    Pre fires positive, post fires negative → co-activation < 0 → weight decreases.
    """
    g, inp, out, eid = make_two_neuron_graph(w=0.5)
    hook = HebbianHook(g, hebbian_lr=0.1, weight_decay=0.0)

    g.get_neuron(inp)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([-1.0])
    hook.record()
    w_before = g.get_edge(eid).weight.item()
    hook.apply()
    assert g.get_edge(eid).weight.item() < w_before


def test_apply_weight_decay_shrinks_weight():
    """With zero co-activation, weight decay alone should shrink the weight."""
    g, inp, out, eid = make_two_neuron_graph(w=1.0)
    hook = HebbianHook(g, hebbian_lr=0.0, weight_decay=0.1)

    # Zero activations → zero co-activation
    g.get_neuron(inp)._current = torch.tensor([0.0])
    g.get_neuron(out)._current = torch.tensor([0.0])
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() < 1.0


def test_apply_no_records_is_noop():
    g, inp, out, eid = make_two_neuron_graph(w=0.5)
    hook = HebbianHook(g, hebbian_lr=1.0, weight_decay=0.0)
    hook.apply()  # no record() called — should be a no-op
    assert g.get_edge(eid).weight.item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# clip_weight
# ---------------------------------------------------------------------------

def test_clip_weight_upper():
    g, inp, out, eid = make_two_neuron_graph(w=0.9)
    hook = HebbianHook(g, hebbian_lr=1.0, weight_decay=0.0, clip_weight=1.0)

    g.get_neuron(inp)._current = torch.tensor([10.0])
    g.get_neuron(out)._current = torch.tensor([10.0])
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() <= 1.0


def test_clip_weight_lower():
    g, inp, out, eid = make_two_neuron_graph(w=-0.9)
    hook = HebbianHook(g, hebbian_lr=1.0, weight_decay=0.0, clip_weight=1.0)

    g.get_neuron(inp)._current = torch.tensor([10.0])
    g.get_neuron(out)._current = torch.tensor([-10.0])
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() >= -1.0


# ---------------------------------------------------------------------------
# Dale's Law respected
# ---------------------------------------------------------------------------

def test_hebbian_respects_dale_excitatory():
    """Excitatory neuron's outgoing weight must remain >= 0 after Hebbian update."""
    g = NeuronGraph()
    exc = g.add_neuron(role="input", activation="linear", cell_type="excitatory")
    out = g.add_neuron(role="output", activation="linear")
    eid = g.add_edge(src=exc, dst=out, weight=0.1)

    hook = HebbianHook(g, hebbian_lr=1.0, weight_decay=0.0, respect_dale=True)
    g.get_neuron(exc)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([-5.0])  # anti-correlated, would push negative
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() >= 0.0


def test_hebbian_no_dale_allows_sign_flip():
    """With respect_dale=False, the weight may cross zero."""
    g = NeuronGraph()
    exc = g.add_neuron(role="input", activation="linear", cell_type="excitatory")
    out = g.add_neuron(role="output", activation="linear")
    eid = g.add_edge(src=exc, dst=out, weight=0.01)

    hook = HebbianHook(g, hebbian_lr=10.0, weight_decay=0.0, respect_dale=False)
    g.get_neuron(exc)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([-1.0])
    hook.record()
    hook.apply()

    assert g.get_edge(eid).weight.item() < 0.0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_accumulators():
    g, inp, out, eid = make_two_neuron_graph()
    hook = HebbianHook(g)
    g.get_neuron(inp)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([1.0])
    hook.record()
    hook.reset()
    assert hook.n_steps_recorded == 0
    assert hook.mean_coactivation() == {}


def test_apply_and_reset_combined():
    g, inp, out, eid = make_two_neuron_graph(w=0.0)
    hook = HebbianHook(g, hebbian_lr=0.1, weight_decay=0.0)
    g.get_neuron(inp)._current = torch.tensor([1.0])
    g.get_neuron(out)._current = torch.tensor([1.0])
    hook.record()
    hook.apply_and_reset()
    assert hook.n_steps_recorded == 0
    assert g.get_edge(eid).weight.item() > 0.0


# ---------------------------------------------------------------------------
# Multi-step accumulation
# ---------------------------------------------------------------------------

def test_multi_step_mean_coactivation():
    """Mean across multiple timesteps should equal the arithmetic mean."""
    g, inp, out, eid = make_two_neuron_graph(w=0.0)
    hook = HebbianHook(g, hebbian_lr=0.0, weight_decay=0.0)

    products = [2.0, 4.0, 6.0]
    for p in products:
        v = p ** 0.5
        g.get_neuron(inp)._current = torch.tensor([v])
        g.get_neuron(out)._current = torch.tensor([v])
        hook.record()

    coact = hook.mean_coactivation()
    assert coact[eid] == pytest.approx(sum(products) / len(products), rel=1e-5)


# ---------------------------------------------------------------------------
# Import from top level
# ---------------------------------------------------------------------------

def test_hebbian_importable_from_top_level():
    from tensor_optix import HebbianHook as HH
    assert HH is HebbianHook

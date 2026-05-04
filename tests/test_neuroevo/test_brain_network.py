"""Tests for BrainNetwork — modular brain regions with inter-region pathways."""
import pytest
import torch
from tensor_optix.neuroevo.brain_network import BrainNetwork, Pathway, InterRegionEdge
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_linear_region(n_in: int, n_out: int) -> NeuronGraph:
    """Minimal graph: n_in input neurons -> n_out output neurons, zero weights."""
    g = NeuronGraph()
    inputs = [g.add_neuron(role="input", activation="linear") for _ in range(n_in)]
    outputs = [g.add_neuron(role="output", activation="linear") for _ in range(n_out)]
    for inp in inputs:
        for out in outputs:
            g.add_edge(src=inp, dst=out, weight=1.0)
    return g


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_add_region():
    brain = BrainNetwork()
    g = make_linear_region(2, 1)
    brain.add_region("sensory", g)
    assert "sensory" in brain.region_names


def test_duplicate_region_raises():
    brain = BrainNetwork()
    g = make_linear_region(1, 1)
    brain.add_region("sensory", g)
    with pytest.raises(ValueError, match="already exists"):
        brain.add_region("sensory", make_linear_region(1, 1))


def test_get_region_returns_correct_graph():
    brain = BrainNetwork()
    g = make_linear_region(2, 1)
    brain.add_region("sensory", g)
    assert brain.get_region("sensory") is g


def test_region_names_order_preserved():
    brain = BrainNetwork()
    for name in ["sensory", "memory", "executive", "motor"]:
        brain.add_region(name, make_linear_region(1, 1))
    assert brain.region_names == ["sensory", "memory", "executive", "motor"]


# ---------------------------------------------------------------------------
# Pathway construction
# ---------------------------------------------------------------------------

def test_add_pathway_creates_edges():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(2, 2))
    brain.add_region("b", make_linear_region(2, 1))
    pid = brain.add_pathway("a", "b", n_connections=4, delay=1)
    pathway = brain.get_pathway(pid)
    assert len(pathway.edge_ids) == 4


def test_add_pathway_wrong_region_raises():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(1, 1))
    with pytest.raises(ValueError, match="not found"):
        brain.add_pathway("a", "nonexistent", n_connections=2)


def test_inter_region_edges_tracked():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(1, 2))
    brain.add_region("b", make_linear_region(1, 1))
    brain.add_pathway("a", "b", n_connections=3)
    assert len(brain.all_inter_region_edges()) == 3


def test_remove_inter_region_edge():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(1, 2))
    brain.add_region("b", make_linear_region(1, 1))
    pid = brain.add_pathway("a", "b", n_connections=2)
    edges_before = len(brain.all_inter_region_edges())
    eid = brain.get_pathway(pid).edge_ids[0]
    brain.remove_inter_region_edge(eid)
    assert len(brain.all_inter_region_edges()) == edges_before - 1


# ---------------------------------------------------------------------------
# Forward pass — shape and basic behaviour
# ---------------------------------------------------------------------------

def test_forward_output_shape_single_region():
    brain = BrainNetwork()
    brain.add_region("motor", make_linear_region(2, 3))
    obs = torch.ones(2)
    out = brain({"motor": obs})
    assert out.shape == (3,)


def test_forward_output_shape_two_regions():
    brain = BrainNetwork()
    brain.add_region("sensory", make_linear_region(2, 2))
    brain.add_region("motor", make_linear_region(1, 3))
    brain.add_pathway("sensory", "motor", n_connections=2, delay=1)
    out = brain({"sensory": torch.ones(2), "motor": torch.ones(1)})
    # both regions contribute to output: 2 + 3
    assert out.shape == (5,)


def test_forward_output_regions_filter():
    brain = BrainNetwork(output_regions=["motor"])
    brain.add_region("sensory", make_linear_region(2, 2))
    brain.add_region("motor", make_linear_region(1, 3))
    brain.add_pathway("sensory", "motor", n_connections=2, delay=1)
    out = brain({"sensory": torch.ones(2), "motor": torch.ones(1)})
    assert out.shape == (3,)


def test_forward_missing_region_input_uses_zeros():
    """Regions not in region_inputs get a zero observation — should not crash."""
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(2, 1))
    brain.add_region("b", make_linear_region(2, 1))
    brain.add_pathway("a", "b", n_connections=1, delay=1)
    out = brain({"a": torch.ones(2)})  # "b" obs omitted intentionally
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

def test_reset_state_clears_history():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(1, 1))
    brain.add_region("b", make_linear_region(1, 1))
    brain.add_pathway("a", "b", n_connections=1, delay=1)
    brain({"a": torch.ones(1), "b": torch.ones(1)})
    assert len(brain._region_output_history["a"]) > 0
    brain.reset_state()
    assert brain._region_output_history["a"] == []


# ---------------------------------------------------------------------------
# summary / repr
# ---------------------------------------------------------------------------

def test_summary_keys():
    brain = BrainNetwork()
    brain.add_region("sensory", make_linear_region(2, 2))
    brain.add_region("motor", make_linear_region(2, 1))
    brain.add_pathway("sensory", "motor", n_connections=3)
    s = brain.summary()
    assert "inter_region_edges" in s
    assert "regions" in s
    assert s["inter_region_edges"] == 3
    assert "sensory" in s["regions"]
    assert "motor" in s["regions"]


def test_repr_contains_region_names():
    brain = BrainNetwork(name="test_brain")
    brain.add_region("sensory", make_linear_region(1, 1))
    r = repr(brain)
    assert "test_brain" in r
    assert "sensory" in r


# ---------------------------------------------------------------------------
# enforce_dale propagates to all regions
# ---------------------------------------------------------------------------

def test_enforce_dale_across_regions():
    brain = BrainNetwork()
    g = NeuronGraph()
    exc = g.add_neuron(role="input", activation="linear", cell_type="excitatory")
    out = g.add_neuron(role="output", activation="linear")
    eid = g.add_edge(src=exc, dst=out, weight=-5.0)
    brain.add_region("sensory", g)
    brain.enforce_dale()
    assert g.get_edge(eid).weight.item() >= 0.0


# ---------------------------------------------------------------------------
# Parameters visible to PyTorch
# ---------------------------------------------------------------------------

def test_brain_network_parameters_include_pathway_weights():
    brain = BrainNetwork()
    brain.add_region("a", make_linear_region(2, 2))
    brain.add_region("b", make_linear_region(2, 1))
    brain.add_pathway("a", "b", n_connections=4, weight_init=0.5)
    params = list(brain.parameters())
    # Should include region params + 4 pathway weights
    assert len(params) > 4


def test_brain_importable_from_top_level():
    from tensor_optix import BrainNetwork as BN
    assert BN is BrainNetwork

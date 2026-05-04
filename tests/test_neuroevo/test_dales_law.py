"""Tests for Dale's Law: excitatory/inhibitory cell types."""
import pytest
import torch
from tensor_optix.neuroevo.graph.neuron import Neuron, CELL_TYPES
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph


# ---------------------------------------------------------------------------
# Neuron cell_type attribute
# ---------------------------------------------------------------------------

def test_neuron_default_cell_type():
    n = Neuron()
    assert n.cell_type == "any"


def test_neuron_excitatory():
    n = Neuron(cell_type="excitatory")
    assert n.cell_type == "excitatory"


def test_neuron_inhibitory():
    n = Neuron(cell_type="inhibitory")
    assert n.cell_type == "inhibitory"


def test_neuron_invalid_cell_type():
    with pytest.raises(ValueError, match="cell_type"):
        Neuron(cell_type="glial")


def test_cell_types_constant():
    assert CELL_TYPES == {"excitatory", "inhibitory", "any"}


# ---------------------------------------------------------------------------
# NeuronGraph.add_neuron passes cell_type through
# ---------------------------------------------------------------------------

def test_graph_add_neuron_cell_type():
    g = NeuronGraph()
    eid = g.add_neuron(role="hidden", cell_type="excitatory")
    assert g.cell_type_of(eid) == "excitatory"

    iid = g.add_neuron(role="hidden", cell_type="inhibitory")
    assert g.cell_type_of(iid) == "inhibitory"

    aid = g.add_neuron(role="hidden", cell_type="any")
    assert g.cell_type_of(aid) == "any"


# ---------------------------------------------------------------------------
# enforce_dale clamps weights correctly
# ---------------------------------------------------------------------------

def _make_dale_graph():
    """
    3 neurons:
      src_exc  (excitatory) -> dst
      src_inh  (inhibitory) -> dst
      src_any  (any)        -> dst
    """
    g = NeuronGraph()
    exc = g.add_neuron(role="input", activation="linear", cell_type="excitatory")
    inh = g.add_neuron(role="input", activation="linear", cell_type="inhibitory")
    any_ = g.add_neuron(role="input", activation="linear", cell_type="any")
    dst = g.add_neuron(role="output", activation="linear", cell_type="any")

    e_exc = g.add_edge(src=exc, dst=dst, weight=-5.0)   # will be clamped to 0
    e_inh = g.add_edge(src=inh, dst=dst, weight=+5.0)   # will be clamped to 0
    e_any = g.add_edge(src=any_, dst=dst, weight=-3.0)   # left unchanged
    return g, e_exc, e_inh, e_any


def test_enforce_dale_clamps_excitatory():
    g, e_exc, e_inh, e_any = _make_dale_graph()
    g.enforce_dale()
    exc_w = g.get_edge(e_exc).weight.item()
    assert exc_w >= 0.0, f"Excitatory weight should be >= 0, got {exc_w}"


def test_enforce_dale_clamps_inhibitory():
    g, e_exc, e_inh, e_any = _make_dale_graph()
    g.enforce_dale()
    inh_w = g.get_edge(e_inh).weight.item()
    assert inh_w <= 0.0, f"Inhibitory weight should be <= 0, got {inh_w}"


def test_enforce_dale_leaves_any_unchanged():
    g, e_exc, e_inh, e_any = _make_dale_graph()
    g.enforce_dale()
    any_w = g.get_edge(e_any).weight.item()
    assert any_w == pytest.approx(-3.0), f"'any' weight should be unchanged, got {any_w}"


def test_enforce_dale_already_valid_unchanged():
    """Weights that already comply should not be altered."""
    g = NeuronGraph()
    src = g.add_neuron(role="input", cell_type="excitatory")
    dst = g.add_neuron(role="output")
    eid = g.add_edge(src=src, dst=dst, weight=2.5)
    g.enforce_dale()
    assert g.get_edge(eid).weight.item() == pytest.approx(2.5)


def test_enforce_dale_idempotent():
    """Calling enforce_dale twice produces the same result as once."""
    g, e_exc, e_inh, e_any = _make_dale_graph()
    g.enforce_dale()
    w_exc_1 = g.get_edge(e_exc).weight.item()
    w_inh_1 = g.get_edge(e_inh).weight.item()
    g.enforce_dale()
    assert g.get_edge(e_exc).weight.item() == pytest.approx(w_exc_1)
    assert g.get_edge(e_inh).weight.item() == pytest.approx(w_inh_1)


# ---------------------------------------------------------------------------
# cell_type_of query
# ---------------------------------------------------------------------------

def test_cell_type_of_all_roles():
    g = NeuronGraph()
    inp = g.add_neuron(role="input", cell_type="excitatory")
    hid = g.add_neuron(role="hidden", cell_type="inhibitory")
    out = g.add_neuron(role="output", cell_type="any")
    assert g.cell_type_of(inp) == "excitatory"
    assert g.cell_type_of(hid) == "inhibitory"
    assert g.cell_type_of(out) == "any"


# ---------------------------------------------------------------------------
# GraphAgent.learn() calls enforce_dale after optimizer step
# ---------------------------------------------------------------------------

def test_graph_agent_enforces_dale_after_learn():
    """After a learn() call, excitatory outgoing weights must be >= 0."""
    import numpy as np
    from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
    from tensor_optix.neuroevo.agent.graph_agent import GraphAgent
    from tensor_optix.core.types import EpisodeData

    g = NeuronGraph()
    inp = g.add_neuron(role="input", activation="linear", cell_type="excitatory")
    out = g.add_neuron(role="output", activation="linear")
    act_out = g.add_neuron(role="output", activation="linear")
    eid = g.add_edge(src=inp, dst=out, weight=-9.0)   # violating Dale
    g.add_edge(src=inp, dst=act_out, weight=0.1)

    agent = GraphAgent(graph=g, obs_dim=1, n_actions=1, continuous=False)

    T = 10
    episode = EpisodeData(
        observations=[[float(i)] for i in range(T)],
        actions=[0] * T,
        rewards=[1.0] * T,
        terminated=[False] * (T - 1) + [True],
        truncated=[False] * T,
        infos=[{}] * T,
        episode_id=0,
        log_probs=[-0.5] * T,
    )
    agent.learn(episode)

    # The excitatory neuron's outgoing weight must now be >= 0
    w = g.get_edge(eid).weight.item()
    assert w >= 0.0, f"Expected excitatory weight >= 0 after learn(), got {w}"

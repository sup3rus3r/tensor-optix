"""
Regression tests for the neuroevo fixes: split_neuron bias, optimizer state
across topology changes, edge gradients, host-sync cost of the controller,
batched forward with recurrent neurons, max_neurons=None, activation sampling,
to_dict weights, and delayed edges between different neurons.
"""
import json

import numpy as np
import pytest
import torch

from tensor_optix.core.types import EpisodeData
from tensor_optix.neuroevo.agent.graph_agent import GraphAgent
from tensor_optix.neuroevo.controller.topology_controller import TopologyController
from tensor_optix.neuroevo.graph.neuron import (
    GRUNeuron, LSTMNeuron, Neuron, TrainableGRUNeuron, TrainableLSTMNeuron,
)
from tensor_optix.neuroevo.graph.neuron_graph import GraphState, NeuronGraph
from tensor_optix.neuroevo.graph.topology_ops import insert_neuron_on_edge, split_neuron
from tensor_optix.neuroevo.optimizer import TopologyAwareAdam


def _dense_graph(n_in=3, n_hid=4, n_out=2, seed=0, dale_mode="clamp") -> NeuronGraph:
    torch.manual_seed(seed)
    g = NeuronGraph(dale_mode=dale_mode)
    ins = [g.add_neuron(role="input", activation="linear") for _ in range(n_in)]
    hid = [g.add_neuron(role="hidden", activation="tanh") for _ in range(n_hid)]
    outs = [g.add_neuron(role="output", activation="linear") for _ in range(n_out)]
    for a in ins:
        for b in hid:
            g.add_edge(a, b, weight=float(torch.randn(()) * 0.5), delay=0)
    for a in hid:
        for b in outs:
            g.add_edge(a, b, weight=float(torch.randn(()) * 0.5), delay=0)
    return g


def _episode(obs_dim, n_actions, length=24) -> EpisodeData:
    return EpisodeData(
        observations=np.random.randn(length, obs_dim).astype(np.float32),
        actions=np.random.randint(0, n_actions, size=length),
        rewards=np.random.randn(length).tolist(),
        terminated=[False] * (length - 1) + [True],
        truncated=[False] * length,
        infos=[{}] * length,
        episode_id=0,
        log_probs=np.random.randn(length).tolist(),
    )


# ---------------------------------------------------------------------------
# 1. split_neuron keeps the output, including with a nonzero bias
# ---------------------------------------------------------------------------

class TestSplitNeuronBias:

    def _graph(self, bias):
        g = NeuronGraph()
        i = g.add_neuron(role="input", activation="linear")
        h = g.add_neuron(role="hidden", activation="tanh")
        o = g.add_neuron(role="output", activation="linear")
        g.add_edge(i, h, weight=0.9, delay=0)
        g.add_edge(h, o, weight=1.3, delay=0)
        with torch.no_grad():
            g.get_neuron(h).bias.fill_(bias)
        return g, h

    @pytest.mark.parametrize("bias", [0.0, 0.8, -1.5])
    def test_output_unchanged(self, bias):
        g, h = self._graph(bias)
        obs = torch.tensor([0.7])
        g.reset_state()
        before = g(obs).detach().clone()
        split_neuron(g, h)
        g.reset_state()
        after = g(obs).detach()
        assert torch.allclose(before, after, atol=1e-6)

    def test_both_copies_keep_the_full_bias(self):
        g, h = self._graph(0.8)
        _, new_id = split_neuron(g, h)
        assert g.get_neuron(h).bias.item() == pytest.approx(0.8)
        assert g.get_neuron(new_id).bias.item() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 2. Optimizer state survives a topology change
# ---------------------------------------------------------------------------

class TestOptimizerStateCarried:

    def _agent(self):
        g = _dense_graph(n_in=4, n_hid=3, n_out=3)
        return GraphAgent(graph=g, obs_dim=4, n_actions=2, continuous=False)

    def test_agent_uses_topology_aware_adam(self):
        assert isinstance(self._agent().optimizer, TopologyAwareAdam)

    def test_surviving_edges_keep_momentum_and_new_edges_start_at_zero(self):
        agent = self._agent()
        agent.learn(_episode(4, 2))

        old_vec = agent.graph._ff_weight_vec
        old_ids = list(agent.graph._ff_edge_ids)
        old_exp_avg = agent.optimizer.state[old_vec]["exp_avg"].clone()
        assert old_exp_avg.abs().sum() > 0

        victim = old_ids[0]
        bias_param = next(agent.graph.get_neuron(agent.graph.hidden_ids[0]).parameters())
        old_bias_state = agent.optimizer.state[bias_param]["exp_avg"].clone()

        new_nid = insert_neuron_on_edge(agent.graph, victim)
        agent.graph.invalidate_compile()
        agent._sync_optimizer()

        new_vec = agent.graph._ff_weight_vec
        new_ids = list(agent.graph._ff_edge_ids)
        assert new_vec is not old_vec
        assert id(new_vec) in {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
        new_exp_avg = agent.optimizer.state[new_vec]["exp_avg"]
        assert new_exp_avg.shape == new_vec.shape

        survivors = [e for e in old_ids if e != victim]
        assert survivors
        for eid in survivors:
            assert new_exp_avg[new_ids.index(eid)].item() == pytest.approx(
                old_exp_avg[old_ids.index(eid)].item())
        created = [e for e in new_ids if e not in old_ids]
        assert created
        for eid in created:
            assert new_exp_avg[new_ids.index(eid)].item() == 0.0

        # An untouched neuron parameter is the same object and keeps its state.
        assert torch.equal(agent.optimizer.state[bias_param]["exp_avg"], old_bias_state)

    def test_load_weights_starts_from_fresh_state(self, tmp_path):
        agent = self._agent()
        agent.learn(_episode(4, 2))
        path = str(tmp_path / "w.pt")
        agent.save_weights(path)
        agent.load_weights(path)
        assert len(agent.optimizer.state) == 0

    def test_learning_still_moves_weights_after_carry(self):
        agent = self._agent()
        agent.learn(_episode(4, 2))
        insert_neuron_on_edge(agent.graph, agent.graph.all_edges()[0].edge_id)
        agent.graph.invalidate_compile()
        before = agent.graph._ff_weight_vec.detach().clone()
        agent.learn(_episode(4, 2))
        assert not torch.equal(before, agent.graph._ff_weight_vec.detach())


# ---------------------------------------------------------------------------
# 3. Edge gradients reach the controller; recurrent neurons are not "dead"
# ---------------------------------------------------------------------------

class TestEdgeGradients:

    def test_edge_table_has_nonzero_gradients_after_backward(self):
        g = _dense_graph()
        g.reset_state()
        g(torch.randn(3)).sum().backward()
        table = g.edge_table()
        assert len(table.edge_ids) == g.n_edges()
        assert (table.grad.abs() > 0).sum().item() > 0

    def test_edge_table_grad_is_zero_before_any_backward(self):
        table = _dense_graph().edge_table()
        assert torch.count_nonzero(table.grad) == 0

    def test_gru_hidden_neuron_reads_as_active_not_dead(self):
        g = NeuronGraph()
        i = g.add_neuron(role="input", activation="linear")
        h = g.add_neuron(role="hidden", neuron=GRUNeuron())
        o = g.add_neuron(role="output", activation="linear")
        g.add_edge(i, h, weight=1.0, delay=0)
        g.add_edge(h, o, weight=1.0, delay=0)
        out, _ = g.forward_batch(torch.randn(5, 1))
        out.sum().backward()
        assert g.get_neuron(h).bias.grad is None   # bias is unused by GRU

        ctrl = TopologyController.for_graph(g)
        grad_mag, _ = ctrl._episode_neuron_stats(g, [h])
        assert grad_mag[h] > 0
        assert ctrl._compute_grad_util([h], grad_mag) == 1.0


# ---------------------------------------------------------------------------
# 4. Controller cost in host syncs does not grow with the number of edges
# ---------------------------------------------------------------------------

class TestHostSyncs:

    def _count_syncs(self, monkeypatch, graph):
        counts = {"item": 0}
        real_item = torch.Tensor.item

        def counting_item(self):
            counts["item"] += 1
            return real_item(self)

        monkeypatch.setattr(torch.Tensor, "item", counting_item)
        ctrl = TopologyController.for_graph(graph, grow_cooldown=10**6)
        graph.reset_state()
        graph(torch.randn(len(graph.input_ids))).sum().backward()
        counts["item"] = 0
        for ep in range(3):
            ctrl.on_episode_end(ep, None)
        return counts["item"]

    def test_item_calls_independent_of_edge_count(self, monkeypatch):
        small = self._count_syncs(monkeypatch, _dense_graph(n_in=3, n_hid=4, n_out=2))
        big_graph = _dense_graph(n_in=12, n_hid=24, n_out=6)
        assert big_graph.n_edges() > 400
        big = self._count_syncs(monkeypatch, big_graph)
        assert big == small
        assert big < 20


# ---------------------------------------------------------------------------
# 5. forward_batch matches step-by-step forward(), recurrent neurons included
# ---------------------------------------------------------------------------

def _recurrent_graph(dale_mode="clamp", trainable=False) -> NeuronGraph:
    torch.manual_seed(3)
    g = NeuronGraph(dale_mode=dale_mode)
    ins = [g.add_neuron(role="input", activation="linear") for _ in range(2)]
    gru = g.add_neuron(role="hidden", neuron=(TrainableGRUNeuron() if trainable else GRUNeuron()))
    lstm = g.add_neuron(role="hidden", neuron=(TrainableLSTMNeuron() if trainable else LSTMNeuron()))
    relu = g.add_neuron(role="hidden", activation="relu")
    tnh = g.add_neuron(role="hidden", activation="tanh")
    outs = [g.add_neuron(role="output", activation="linear") for _ in range(2)]
    for a in ins:
        for b in (gru, lstm, relu):
            g.add_edge(a, b, weight=float(torch.randn(())), delay=0)
    g.add_edge(relu, tnh, weight=0.8, delay=0)
    g.add_edge(gru, tnh, weight=-0.6, delay=0)
    for a in (gru, lstm, tnh):
        for b in outs:
            g.add_edge(a, b, weight=float(torch.randn(())), delay=0)
    # delayed edges between DIFFERENT neurons, a self loop, and a delay of 2
    g.add_edge(ins[0], tnh, weight=0.7, delay=1)
    g.add_edge(tnh, relu, weight=0.5, delay=1)
    g.add_edge(gru, gru, weight=0.4, delay=1)
    g.add_edge(lstm, outs[1], weight=0.9, delay=2)
    for nid in g.hidden_ids:
        with torch.no_grad():
            g.get_neuron(nid).bias.add_(0.1 * torch.randn(1))
    return g


@pytest.mark.parametrize("dale_mode", ["clamp", "softplus"])
@pytest.mark.parametrize("trainable", [False, True])
def test_forward_batch_matches_sequential_forward(dale_mode, trainable):
    g = _recurrent_graph(dale_mode, trainable)
    B, T = 3, 6
    seq = torch.randn(T, B, 2)

    # reference: every batch row is an independent sequence through forward()
    ref = torch.zeros(T, B, 2)
    with torch.no_grad():
        for b in range(B):
            g.reset_state()
            for t in range(T):
                ref[t, b] = g(seq[t, b])

    state = None
    got = torch.zeros(T, B, 2)
    with torch.no_grad():
        for t in range(T):
            out, state = g.forward_batch(seq[t], state)
            got[t] = out
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_forward_batch_leaves_neuron_state_alone():
    g = _recurrent_graph()
    g.reset_state()
    gru = next(n for n in g._neurons.values() if isinstance(n, GRUNeuron))
    h_before = gru._h.clone()
    g.forward_batch(torch.randn(4, 2))
    assert torch.equal(gru._h, h_before)


def test_forward_batch_backprop_through_time_and_detach():
    g = _recurrent_graph(trainable=True)
    state = g.initial_state(2)
    outs = []
    for _ in range(4):
        out, state = g.forward_batch(torch.randn(2, 2), state)
        outs.append(out)
    torch.stack(outs).sum().backward()
    gru = next(n for n in g._neurons.values() if isinstance(n, TrainableGRUNeuron))
    assert gru.wz.grad is not None and gru.wz.grad.abs().sum() > 0

    detached = state.detach()
    assert not detached.h.requires_grad and not detached.hist.requires_grad


def test_forward_batch_rejects_wrong_shapes():
    g = _recurrent_graph()
    with pytest.raises(ValueError):
        g.forward_batch(torch.randn(3, 5))
    state = g.initial_state(2)
    with pytest.raises(ValueError):
        g.forward_batch(torch.randn(3, 2), state)


def test_graph_agent_batch_forward_matches_stateless_act_path():
    g = _dense_graph(n_in=4, n_hid=3, n_out=3)
    agent = GraphAgent(graph=g, obs_dim=4, n_actions=2, continuous=False)
    obs = torch.randn(5, 4)
    batched = agent._batch_forward(obs)
    rows = []
    for o in obs:
        agent.graph.reset_state()
        rows.append(agent.graph(o))
    assert torch.allclose(batched, torch.stack(rows), atol=1e-6)


# ---------------------------------------------------------------------------
# 6. max_neurons=None means no ceiling
# ---------------------------------------------------------------------------

class TestMaxNeurons:

    def _big(self):
        g = NeuronGraph()
        g.add_neuron(role="input", activation="linear")
        g.add_neuron(role="output", activation="linear")
        for _ in range(300):
            g.add_neuron(role="hidden", activation="tanh")
        return g

    def test_default_ceiling_still_applies(self):
        ctrl = TopologyController(self._big(), grow_cooldown=0, min_score_buffer=0)
        assert ctrl.max_neurons == 256
        assert ctrl._grow_signal_check("default") is False

    def test_none_removes_the_ceiling(self):
        ctrl = TopologyController(
            self._big(), grow_cooldown=0, min_score_buffer=0, max_neurons=None)
        assert ctrl._grow_signal_check("default") is True


# ---------------------------------------------------------------------------
# 7. Merge detection sees many activation samples per call
# ---------------------------------------------------------------------------

class TestActivationSampling:

    def test_batched_forward_gives_one_sample_per_row(self):
        g = _dense_graph(n_in=3, n_hid=4, n_out=2)
        ctrl = TopologyController.for_graph(g, act_history_len=100)
        g.forward_batch(torch.randn(40, 3))
        ctrl.on_episode_end(0, None)
        for nid in g.hidden_ids:
            assert len(ctrl._act_history[nid]) == 40

    def test_samples_come_from_a_forward_that_never_touches_neuron_state(self):
        g = _dense_graph()
        ctrl = TopologyController.for_graph(g)
        g.forward_batch(torch.randn(8, 3) * 3)
        hid = g.hidden_ids[0]
        assert g.get_neuron(hid)._current.abs().sum() == 0   # untouched
        ctrl.on_episode_end(0, None)
        assert any(abs(v) > 0 for v in ctrl._act_history[hid])

    def test_external_forward_can_report_activations(self):
        g = _dense_graph()
        ctrl = TopologyController.for_graph(g)
        h = torch.randn(25, g.n_neurons())
        g.record_activations(h)
        ctrl.on_episode_end(0, None)
        assert len(ctrl._act_history[g.hidden_ids[0]]) == 25

    def test_single_step_forward_still_records_one_sample_each(self):
        g = _dense_graph()
        ctrl = TopologyController.for_graph(g)
        for _ in range(5):
            g(torch.randn(3))
        ctrl.on_episode_end(0, None)
        assert len(ctrl._act_history[g.hidden_ids[0]]) == 5

    def test_history_is_capped(self):
        g = _dense_graph()
        ctrl = TopologyController.for_graph(g, act_history_len=30)
        g.forward_batch(torch.randn(200, 3))
        ctrl.on_episode_end(0, None)
        assert len(ctrl._act_history[g.hidden_ids[0]]) == 30


# ---------------------------------------------------------------------------
# to_dict keeps the weights by default; device is settable
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_round_trip_through_json_preserves_function(self):
        g = _recurrent_graph()
        x = torch.randn(2)
        g.reset_state()
        want = [g(x).detach().clone() for _ in range(3)]

        blob = json.loads(json.dumps(g.to_dict()))
        clone = NeuronGraph.from_dict(blob)
        clone.reset_state()
        got = [clone(x).detach().clone() for _ in range(3)]
        for a, b in zip(want, got):
            assert torch.allclose(a, b, atol=1e-6)

    def test_structure_only_dict_has_no_state(self):
        d = _dense_graph().to_dict(include_weights=False)
        assert "state" not in d
        assert NeuronGraph.from_dict(d).n_edges() == d["edges"].__len__()

    def test_restore_from_dict_loads_weights_in_place(self):
        g = _dense_graph(seed=1)
        d = g.to_dict()
        other = _dense_graph(seed=2)
        same_object = other.restore_from_dict(d)
        assert same_object is other
        x = torch.randn(3)
        g.reset_state(), other.reset_state()
        assert torch.allclose(g(x), other(x), atol=1e-6)

    def test_device_argument(self):
        assert NeuronGraph(device="cpu")._device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Delayed edges between different neurons deliver the source's history
# ---------------------------------------------------------------------------

def test_delayed_edge_between_different_neurons_delivers_history():
    g = NeuronGraph()
    a = g.add_neuron(role="input", activation="linear")
    b = g.add_neuron(role="output", activation="linear")
    g.add_edge(a, b, weight=1.0, delay=1)
    outs = [g(torch.tensor([float(t)])).item() for t in (1, 2, 3, 4)]
    assert outs == [0.0, 1.0, 2.0, 3.0]


def test_delayed_edge_of_two_steps():
    g = NeuronGraph()
    a = g.add_neuron(role="input", activation="linear")
    b = g.add_neuron(role="output", activation="linear")
    g.add_edge(a, b, weight=1.0, delay=2)
    outs = [g(torch.tensor([float(t)])).item() for t in (1, 2, 3, 4)]
    assert outs == [0.0, 0.0, 1.0, 2.0]

"""
HebbianHook works on whole tensors (no per-edge host syncs) and computes the
same update as the per-edge rule; RecurrentGraphAgent's act() and learn()
compute the same recurrent function.
"""
import numpy as np
import pytest
import torch

from tensor_optix.core.types import EpisodeData
from tensor_optix.neuroevo.agent.graph_agent import RecurrentGraphAgent
from tensor_optix.neuroevo.graph.neuron import GRUNeuron, TrainableGRUNeuron, TrainableLSTMNeuron
from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.graph.topology_ops import insert_neuron_on_edge
from tensor_optix.neuroevo.hebbian import HebbianHook


def _graph(dale_mode="clamp", seed=0, n_hid=4):
    torch.manual_seed(seed)
    g = NeuronGraph(dale_mode=dale_mode)
    ins = [g.add_neuron(role="input", activation="linear") for _ in range(3)]
    cells = ["excitatory", "inhibitory", "any", "excitatory"]
    hid = [g.add_neuron(role="hidden", activation="tanh", cell_type=cells[i % 4])
           for i in range(n_hid)]
    out = g.add_neuron(role="output", activation="linear")
    for a in ins:
        for b in hid:
            g.add_edge(a, b, weight=float(torch.randn(()).abs() * 0.5), delay=0)
    for i, a in enumerate(hid):
        sign = {"excitatory": 1, "inhibitory": -1}.get(cells[i % 4], 1)
        g.add_edge(a, out, weight=sign * float(torch.rand(())) * 0.5, delay=0)
    g.add_edge(hid[0], hid[1], weight=0.3, delay=1)
    return g


def _reference_update(graph, samples, lr, decay, clip):
    """The per-edge Hebbian rule, written out edge by edge."""
    expected = {}
    for e in graph.all_edges():
        vals = samples[e.edge_id]
        mean = sum(vals) / len(vals)
        w = graph.effective_weight(e.edge_id)
        new_w = w + lr * mean - decay * w
        if clip is not None:
            new_w = max(-clip, min(clip, new_w))
        ct = graph.get_neuron(e.src).cell_type
        if ct == "excitatory":
            new_w = max(new_w, 0.0)
        elif ct == "inhibitory":
            new_w = min(new_w, 0.0)
        expected[e.edge_id] = new_w
    return expected


@pytest.mark.parametrize("dale_mode", ["clamp", "softplus"])
@pytest.mark.parametrize("clip", [None, 0.4])
def test_hebbian_update_matches_per_edge_rule(dale_mode, clip):
    g = _graph(dale_mode)
    hook = HebbianHook(g, hebbian_lr=0.05, weight_decay=0.01, clip_weight=clip)
    samples = {e.edge_id: [] for e in g.all_edges()}
    for _ in range(5):
        g(torch.randn(3))
        for e in g.all_edges():
            samples[e.edge_id].append(
                g.get_neuron(e.src)._current.item() * g.get_neuron(e.dst)._current.item())
        hook.record()
    expected = _reference_update(g, samples, 0.05, 0.01, clip)
    hook.apply()
    for eid, w in expected.items():
        assert g.effective_weight(eid) == pytest.approx(w, abs=1e-5)


def test_hebbian_mean_coactivation_matches_samples():
    g = _graph()
    hook = HebbianHook(g)
    samples = {e.edge_id: [] for e in g.all_edges()}
    for _ in range(4):
        g(torch.randn(3))
        for e in g.all_edges():
            samples[e.edge_id].append(
                g.get_neuron(e.src)._current.item() * g.get_neuron(e.dst)._current.item())
        hook.record()
    got = hook.mean_coactivation()
    for eid, vals in samples.items():
        assert got[eid] == pytest.approx(sum(vals) / len(vals), abs=1e-6)


def test_hebbian_survives_topology_change_mid_episode():
    g = _graph()
    hook = HebbianHook(g)
    g(torch.randn(3))
    hook.record()
    victim = g.all_edges()[0].edge_id
    keep = g.all_edges()[1].edge_id
    first = g.get_neuron(g.get_edge(keep).src)._current.item() * \
        g.get_neuron(g.get_edge(keep).dst)._current.item()
    insert_neuron_on_edge(g, victim)
    g.invalidate_compile()
    g(torch.randn(3))
    second = g.get_neuron(g.get_edge(keep).src)._current.item() * \
        g.get_neuron(g.get_edge(keep).dst)._current.item()
    hook.record()
    assert hook.mean_coactivation()[keep] == pytest.approx((first + second) / 2, abs=1e-6)
    hook.apply()   # must not fail on edges that appeared or vanished


def test_hebbian_host_syncs_do_not_grow_with_edges(monkeypatch):
    def count(graph):
        calls = {"n": 0}
        real = torch.Tensor.item

        def counting(self):
            calls["n"] += 1
            return real(self)

        hook = HebbianHook(graph)
        graph(torch.randn(3))
        monkeypatch.setattr(torch.Tensor, "item", counting)
        for _ in range(10):
            hook.record()
        hook.apply()
        monkeypatch.setattr(torch.Tensor, "item", real)
        return calls["n"]

    small = count(_graph(n_hid=4))
    big = _graph(n_hid=120)
    assert big.n_edges() > 400
    assert count(big) == small


# ---------------------------------------------------------------------------
# RecurrentGraphAgent
# ---------------------------------------------------------------------------

def _recurrent_agent(seed=0, chunk_len=64):
    torch.manual_seed(seed)
    g = NeuronGraph()
    ins = [g.add_neuron(role="input", activation="linear") for _ in range(2)]
    gru = g.add_neuron(role="hidden", neuron=TrainableGRUNeuron())
    lstm = g.add_neuron(role="hidden", neuron=TrainableLSTMNeuron())
    tnh = g.add_neuron(role="hidden", activation="tanh")
    outs = [g.add_neuron(role="output", activation="linear") for _ in range(3)]
    for a in ins:
        for b in (gru, lstm, tnh):
            g.add_edge(a, b, weight=float(torch.randn(())), delay=0)
    for a in (gru, lstm, tnh):
        for b in outs:
            g.add_edge(a, b, weight=float(torch.randn(())), delay=0)
    g.add_edge(ins[0], tnh, weight=0.8, delay=1)
    g.add_edge(tnh, outs[0], weight=-0.7, delay=2)
    from tensor_optix.core.types import HyperparamSet
    hp = HyperparamSet(params={"chunk_len": chunk_len}, episode_id=0)
    return RecurrentGraphAgent(graph=g, obs_dim=2, n_actions=2, hyperparams=hp)


def _rollout(agent, T, done_at=()):
    obs, acts, lps, dones = [], [], [], []
    agent.reset_hidden()
    for t in range(T):
        o = np.random.randn(2).astype(np.float32)
        a = agent.act(o)
        obs.append(o)
        acts.append(a)
        lps.append(float(agent._last_log_prob))
        done = t in done_at
        dones.append(done)
        if done:
            agent.reset_hidden()
    ep = EpisodeData(
        observations=np.array(obs), actions=np.array(acts),
        rewards=np.random.randn(T).tolist(),
        terminated=dones, truncated=[False] * T, infos=[{}] * T, episode_id=0,
    )
    return ep, lps


def test_act_carries_state_between_steps():
    agent = _recurrent_agent()
    agent.reset_hidden()
    x = np.array([1.0, -1.0], dtype=np.float32)
    agent.act(x)
    with torch.no_grad():
        second, _ = agent.graph.forward_batch(torch.tensor(x).unsqueeze(0), agent._state)
        fresh, _ = agent.graph.forward_batch(torch.tensor(x).unsqueeze(0))
    assert not torch.allclose(second, fresh)


@pytest.mark.parametrize("chunk_len", [64, 5])
def test_learn_replays_exactly_what_act_computed(chunk_len):
    agent = _recurrent_agent(chunk_len=chunk_len)
    ep, act_lps = _rollout(agent, T=20, done_at=(8,))
    T = len(ep.rewards)
    flags, start = agent._window_plan(T, ep.dones)
    assert flags[0] and flags[9] and sum(flags) == 2
    obs_t = torch.tensor(ep.observations)
    act_t = torch.tensor(ep.actions, dtype=torch.long)
    with torch.no_grad():
        out, _ = agent._replay(obs_t, flags, start, 0, T, chunk_len)
        lp, _, _ = agent._evaluate_actions(out, act_t)
    assert np.allclose(lp.numpy(), act_lps, atol=1e-5)


def test_learn_updates_weights_and_consumes_window():
    agent = _recurrent_agent(chunk_len=8)
    ep, _ = _rollout(agent, T=24, done_at=(10,))
    before = agent.graph._ff_weight_vec.detach().clone()
    stats = agent.learn(ep)
    assert {"loss", "pg_loss", "vf_loss", "entropy"} <= set(stats)
    assert not torch.equal(before, agent.graph._ff_weight_vec.detach())
    assert agent._reset_flags == []


def test_learn_on_data_act_did_not_produce():
    agent = _recurrent_agent()
    T = 12
    ep = EpisodeData(
        observations=np.random.randn(T, 2).astype(np.float32),
        actions=np.random.randint(0, 2, size=T),
        rewards=np.random.randn(T).tolist(),
        terminated=[False] * 5 + [True] + [False] * 6,
        truncated=[False] * T, infos=[{}] * T, episode_id=0,
    )
    flags, start = agent._window_plan(T, ep.dones)
    assert start is None and flags[0] and flags[6] and sum(flags) == 2
    agent.learn(ep)


def test_stateless_graph_falls_back_to_minibatch_ppo(monkeypatch):
    g = NeuronGraph()
    i = g.add_neuron(role="input", activation="linear")
    h = g.add_neuron(role="hidden", activation="tanh")
    outs = [g.add_neuron(role="output", activation="linear") for _ in range(3)]
    g.add_edge(i, h, weight=0.5)
    for o in outs:
        g.add_edge(h, o, weight=0.5)
    agent = RecurrentGraphAgent(graph=g, obs_dim=1, n_actions=2)
    assert agent._graph_has_state() is False
    called = {"recurrent": False}
    monkeypatch.setattr(agent, "_recurrent_learn", lambda ep: called.__setitem__("recurrent", True))
    T = 10
    agent.learn(EpisodeData(
        observations=np.random.randn(T, 1).astype(np.float32),
        actions=np.random.randint(0, 2, size=T), rewards=[0.0] * T,
        terminated=[False] * T, truncated=[False] * T, infos=[{}] * T, episode_id=0,
    ))
    assert called["recurrent"] is False


def test_plain_gru_counts_as_state():
    g = NeuronGraph()
    i = g.add_neuron(role="input", activation="linear")
    h = g.add_neuron(role="hidden", neuron=GRUNeuron())
    outs = [g.add_neuron(role="output", activation="linear") for _ in range(3)]
    g.add_edge(i, h, weight=0.5)
    for o in outs:
        g.add_edge(h, o, weight=0.5)
    assert RecurrentGraphAgent(graph=g, obs_dim=1, n_actions=2)._graph_has_state() is True

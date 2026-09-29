"""
Tests for GraphAgent — verifies the BaseAgent contract and key behaviours.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.agent.graph_agent import GraphAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_graph(obs_dim: int, n_actions: int) -> NeuronGraph:
    g = NeuronGraph()
    for _ in range(obs_dim):
        g.add_neuron(role="input", activation="linear")
    for _ in range(n_actions):
        g.add_neuron(role="hidden", activation="tanh")
    # action outputs + 1 value output
    for _ in range(n_actions + 1):
        g.add_neuron(role="output", activation="linear")
    # wire inputs -> hidden -> outputs
    for inp in g.input_ids:
        for hid in g.hidden_ids:
            g.add_edge(inp, hid, weight=0.1, delay=0)
    for hid in g.hidden_ids:
        for out in g.output_ids:
            g.add_edge(hid, out, weight=0.1, delay=0)
    return g


def make_agent(obs_dim=4, n_actions=2) -> GraphAgent:
    g = make_graph(obs_dim, n_actions)
    return GraphAgent(graph=g, obs_dim=obs_dim, n_actions=n_actions, continuous=False)


def fake_episode(obs_dim=4, n_actions=2, length=20) -> EpisodeData:
    obs = np.random.randn(length, obs_dim).astype(np.float32)
    acts = np.random.randint(0, n_actions, size=length)
    rews = np.random.randn(length).tolist()
    terms = [False] * (length - 1) + [True]
    truncs = [False] * length
    log_probs = np.random.randn(length).tolist()
    return EpisodeData(
        observations=obs,
        actions=acts,
        rewards=rews,
        terminated=terms,
        truncated=truncs,
        infos=[{}] * length,
        episode_id=0,
        log_probs=log_probs,
    )


# ---------------------------------------------------------------------------
# act()
# ---------------------------------------------------------------------------

class TestAct:

    def test_returns_int_for_discrete(self):
        agent = make_agent(obs_dim=4, n_actions=2)
        obs = np.random.randn(4).astype(np.float32)
        action = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < 2

    def test_returns_array_for_continuous(self):
        g = make_graph(4, 2)
        agent = GraphAgent(g, obs_dim=4, n_actions=2, continuous=True)
        obs = np.random.randn(4).astype(np.float32)
        action = agent.act(obs)
        assert hasattr(action, '__len__')
        assert len(action) == 2

    def test_auto_grows_inputs_on_larger_obs(self):
        agent = make_agent(obs_dim=4, n_actions=2)
        obs = np.random.randn(6).astype(np.float32)  # 2 extra dims
        action = agent.act(obs)  # should not raise
        assert agent.obs_dim == 6
        assert len(agent.graph.input_ids) == 6


# ---------------------------------------------------------------------------
# learn()
# ---------------------------------------------------------------------------

class TestLearn:

    def test_returns_dict_with_expected_keys(self):
        agent = make_agent()
        ep = fake_episode()
        diag = agent.learn(ep)
        for key in ("loss", "pg_loss", "vf_loss", "entropy", "n_neurons", "n_edges"):
            assert key in diag, f"Missing key: {key}"

    def test_loss_is_finite(self):
        agent = make_agent()
        ep = fake_episode()
        diag = agent.learn(ep)
        assert np.isfinite(diag["loss"])

    def test_parameters_change_after_learn(self):
        agent = make_agent()
        params_before = [p.data.clone() for p in agent.graph.parameters()]
        ep = fake_episode()
        agent.learn(ep)
        params_after = [p.data.clone() for p in agent.graph.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert changed, "No parameters changed after learn()"


# ---------------------------------------------------------------------------
# get/set hyperparams
# ---------------------------------------------------------------------------

class TestHyperparams:

    def test_get_hyperparams_returns_hyperparamset(self):
        agent = make_agent()
        hp = agent.get_hyperparams()
        assert isinstance(hp, HyperparamSet)
        assert "learning_rate" in hp.params

    def test_set_hyperparams_updates_lr(self):
        agent = make_agent()
        new_hp = HyperparamSet(params={"learning_rate": 1e-5}, episode_id=0)
        agent.set_hyperparams(new_hp)
        for pg in agent.optimizer.param_groups:
            assert abs(pg["lr"] - 1e-5) < 1e-9


# ---------------------------------------------------------------------------
# save / load weights
# ---------------------------------------------------------------------------

class TestWeights:

    def test_save_load_roundtrip(self):
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pt")
            agent.save_weights(path)
            assert os.path.exists(path)
            params_before = {k: v.clone() for k, v in agent.graph.state_dict().items()}
            # Perturb weights
            agent.perturb_weights(0.5)
            # Restore
            agent.load_weights(path)
            params_after = agent.graph.state_dict()
            for k in params_before:
                assert torch.allclose(params_before[k], params_after[k], atol=1e-6)

    def test_perturb_weights_changes_params(self):
        agent = make_agent()
        before = [p.data.clone() for p in agent.graph.parameters()]
        agent.perturb_weights(0.1)
        after = [p.data.clone() for p in agent.graph.parameters()]
        assert any(not torch.equal(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# is_on_policy
# ---------------------------------------------------------------------------

def test_is_on_policy():
    agent = make_agent()
    assert agent.is_on_policy is True


# ---------------------------------------------------------------------------
# Optimizer / graph structural-rebuild identity (regression coverage for the
# "optimizer captures dead pre-rebuild Parameters" bug class)
# ---------------------------------------------------------------------------

class TestOptimizerResync:

    def test_optimizer_tracks_live_ff_weight_vec_after_init(self):
        agent = make_agent()
        live_ids = {id(p) for p in agent.graph.parameters()}
        opt_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
        assert opt_ids <= live_ids or opt_ids == live_ids
        # The combined packed vector must be the one the optimizer holds.
        assert id(agent.graph._ff_weight_vec) in opt_ids

    def test_optimizer_resyncs_after_external_structural_rebuild(self):
        # Simulates what TopologyController does after grow/prune/merge:
        # it calls graph.invalidate_compile() with no knowledge of any
        # optimizer, replacing _ff_weight_vec with a new object.
        agent = make_agent()
        old_ff_vec_id = id(agent.graph._ff_weight_vec)

        agent.graph.invalidate_compile()
        new_ff_vec_id = id(agent.graph._ff_weight_vec)
        assert new_ff_vec_id != old_ff_vec_id  # sanity: rebuild actually replaced it

        # Before the next act()/learn() call, the optimizer is stale.
        opt_ids_before = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
        assert old_ff_vec_id in opt_ids_before
        assert new_ff_vec_id not in opt_ids_before

        agent._sync_optimizer()

        opt_ids_after = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
        assert new_ff_vec_id in opt_ids_after
        assert old_ff_vec_id not in opt_ids_after

    def test_learn_actually_updates_live_weights_after_grow(self):
        agent = make_agent(obs_dim=4, n_actions=2)
        episode = fake_episode(obs_dim=4, n_actions=2, length=32)

        # Trigger an input-growth mutation mid-episode, mimicking act()
        # observing a larger obs than the graph currently has inputs for.
        agent.act(np.random.randn(6).astype(np.float32))
        assert agent.obs_dim == 6

        episode6 = fake_episode(obs_dim=6, n_actions=2, length=32)
        before = agent.graph._ff_weight_vec.data.clone()
        agent.learn(episode6)
        after = agent.graph._ff_weight_vec.data.clone()

        # The live packed weight vector must actually have moved — proving
        # the optimizer stepped the parameters the forward pass reads from,
        # not a stale, orphaned copy.
        assert not torch.equal(before, after)

    def test_learn_updates_weights_after_manual_topology_rebuild(self):
        agent = make_agent()
        episode = fake_episode()

        agent.graph.invalidate_compile()  # stand-in for a controller-driven rebuild
        before = agent.graph._ff_weight_vec.data.clone()
        agent.learn(episode)
        after = agent.graph._ff_weight_vec.data.clone()
        assert not torch.equal(before, after)

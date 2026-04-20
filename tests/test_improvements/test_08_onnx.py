"""
tests/test_improvements/test_08_onnx.py

Tests for agent.export_onnx(path) on all PyTorch agents.

Correctness claims:

1. NUMERICAL PARITY
   The ONNX model produces numerically identical outputs to the original
   PyTorch network on the same observations (atol ≤ 1e-5).  This is the
   primary correctness guarantee: ONNX export is a lossless serialisation
   of the inference graph.

2. FILE PRODUCED
   export_onnx() writes a valid ONNX file to the given path.

3. ONNX VALIDITY
   onnx.checker.check_model() passes — the graph is well-formed.

4. DYNAMIC BATCH SIZE
   The exported model accepts batches of size 1 and 16 without graph
   recompilation.

5. EVAL-MODE INVARIANCE
   export_onnx() puts the network in eval mode for export and restores the
   original training mode afterwards.  The agent is fully functional after
   export_onnx() returns.

6. RECURRENT ACTOR STATE THREADING
   The RecurrentPPO ONNX model accepts (obs, h0, c0) and returns
   (logits, h_n, c_n).  Threading h_n / c_n back as inputs on the next
   call replicates the hidden-state update in TorchRecurrentPPOAgent.act().

7. UNSUPPORTED AGENT
   Calling export_onnx() on a plain BaseAgent subclass raises
   NotImplementedError.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers — minimal agent builders
# ---------------------------------------------------------------------------

def _ppo_agent(obs_dim=8, n_actions=4, device="cpu"):
    from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
    from tensor_optix.core.types import HyperparamSet

    actor  = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, n_actions))
    critic = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, 1))
    opt    = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    hp     = HyperparamSet(params={"learning_rate": 1e-3, "gamma": 0.99, "clip_ratio": 0.2,
                                    "n_epochs": 1, "minibatch_size": 4}, episode_id=0)
    return TorchPPOAgent(actor=actor, critic=critic, optimizer=opt, hyperparams=hp, device=device)


def _sac_agent(obs_dim=6, action_dim=2, device="cpu"):
    from tensor_optix.algorithms.torch_sac import TorchSACAgent
    from tensor_optix.core.types import HyperparamSet

    def _mlp(in_d, out_d):
        return nn.Sequential(nn.Linear(in_d, 32), nn.ReLU(), nn.Linear(32, out_d))

    actor = _mlp(obs_dim, action_dim * 2)   # mean || log_std
    c1    = _mlp(obs_dim + action_dim, 1)
    c2    = _mlp(obs_dim + action_dim, 1)
    # Dummy log_alpha param for alpha_optimizer — SAC replaces it internally
    log_alpha_param = torch.zeros(1, requires_grad=True)
    hp = HyperparamSet(params={"learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005,
                                "batch_size": 8, "replay_capacity": 100},
                       episode_id=0)
    return TorchSACAgent(
        actor=actor, critic1=c1, critic2=c2, action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=3e-4),
        alpha_optimizer=torch.optim.Adam([log_alpha_param], lr=3e-4),
        hyperparams=hp, device=device,
    )


def _td3_agent(obs_dim=6, action_dim=2, device="cpu"):
    from tensor_optix.algorithms.torch_td3 import TorchTD3Agent
    from tensor_optix.core.types import HyperparamSet

    def _mlp(in_d, out_d, out_act=None):
        layers = [nn.Linear(in_d, 32), nn.ReLU(), nn.Linear(32, out_d)]
        if out_act:
            layers.append(out_act())
        return nn.Sequential(*layers)

    actor = _mlp(obs_dim, action_dim, nn.Tanh)
    c1    = _mlp(obs_dim + action_dim, 1)
    c2    = _mlp(obs_dim + action_dim, 1)
    hp    = HyperparamSet(params={"learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005,
                                   "batch_size": 8, "replay_capacity": 100},
                          episode_id=0)
    return TorchTD3Agent(
        actor=actor, critic1=c1, critic2=c2, action_dim=action_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=3e-4),
        hyperparams=hp, device=device,
    )


def _dqn_agent(obs_dim=8, n_actions=4, device="cpu"):
    from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
    from tensor_optix.core.types import HyperparamSet

    q_net = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, n_actions))
    hp    = HyperparamSet(params={"learning_rate": 1e-3, "gamma": 0.99, "epsilon": 1.0,
                                   "epsilon_min": 0.1, "epsilon_decay": 0.99,
                                   "batch_size": 8, "replay_capacity": 100},
                          episode_id=0)
    return TorchDQNAgent(
        q_network=q_net, n_actions=n_actions,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=1e-3),
        hyperparams=hp, device=device,
    )


def _recurrent_ppo_agent(obs_dim=8, n_actions=4, hidden_size=16, device="cpu"):
    from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent
    from tensor_optix.core.types import HyperparamSet

    rnn         = nn.LSTM(obs_dim, hidden_size, num_layers=1, batch_first=True)
    actor_head  = nn.Sequential(nn.Linear(hidden_size, n_actions))
    critic_head = nn.Sequential(nn.Linear(hidden_size, 1))
    params      = (list(rnn.parameters()) + list(actor_head.parameters())
                   + list(critic_head.parameters()))
    hp = HyperparamSet(params={"learning_rate": 1e-3, "gamma": 0.99, "clip_ratio": 0.2,
                                "n_epochs": 1, "minibatch_size": 4, "bptt_len": 8},
                       episode_id=0)
    return TorchRecurrentPPOAgent(
        rnn=rnn, actor_head=actor_head, critic_head=critic_head,
        n_actions=n_actions, optimizer=torch.optim.Adam(params, lr=1e-3),
        hyperparams=hp, device=device,
    )


# ---------------------------------------------------------------------------
# 1. Numerical parity — PPO
# ---------------------------------------------------------------------------

class TestONNXParityPPO:
    OBS_DIM   = 8
    N_ACTIONS = 4

    def test_ppo_onnx_file_created(self, tmp_path):
        agent = _ppo_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "ppo_actor.onnx")
        agent.export_onnx(path)
        assert os.path.isfile(path)

    def test_ppo_onnx_model_valid(self, tmp_path):
        import onnx
        agent = _ppo_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "ppo_actor.onnx")
        agent.export_onnx(path)
        model = onnx.load(path)
        onnx.checker.check_model(model)   # raises if graph is malformed

    def test_ppo_onnx_parity_single(self, tmp_path):
        """Single observation: ONNX logits == torch logits (atol 1e-5)."""
        import onnxruntime as ort
        agent = _ppo_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "ppo_actor.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(1, self.OBS_DIM).astype(np.float32)

        with torch.no_grad():
            logits_torch = agent._actor(torch.from_numpy(obs)).numpy()

        sess = ort.InferenceSession(path)
        logits_onnx = sess.run(["logits"], {"observation": obs})[0]

        np.testing.assert_allclose(logits_torch, logits_onnx, atol=1e-5)

    def test_ppo_onnx_parity_batch(self, tmp_path):
        """Batch of 16 observations: parity holds."""
        import onnxruntime as ort
        agent = _ppo_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "ppo_actor.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(16, self.OBS_DIM).astype(np.float32)
        with torch.no_grad():
            logits_torch = agent._actor(torch.from_numpy(obs)).numpy()
        sess = ort.InferenceSession(path)
        logits_onnx = sess.run(["logits"], {"observation": obs})[0]
        np.testing.assert_allclose(logits_torch, logits_onnx, atol=1e-5)

    def test_ppo_export_restores_training_mode(self, tmp_path):
        """After export_onnx(), actor is back in its original mode."""
        agent = _ppo_agent(self.OBS_DIM, self.N_ACTIONS)
        agent._actor.train()   # ensure it starts in train mode
        path = str(tmp_path / "ppo_actor.onnx")
        agent.export_onnx(path)
        assert agent._actor.training, "actor should be training=True after export"


# ---------------------------------------------------------------------------
# 2. Numerical parity — SAC
# ---------------------------------------------------------------------------

class TestONNXParitySAC:
    OBS_DIM    = 6
    ACTION_DIM = 2

    def test_sac_onnx_parity(self, tmp_path):
        """mean_logstd output matches PyTorch actor forward."""
        import onnxruntime as ort
        agent = _sac_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "sac_actor.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(8, self.OBS_DIM).astype(np.float32)
        with torch.no_grad():
            out_torch = agent._actor(torch.from_numpy(obs)).numpy()
        sess = ort.InferenceSession(path)
        out_onnx = sess.run(["mean_logstd"], {"observation": obs})[0]

        np.testing.assert_allclose(out_torch, out_onnx, atol=1e-5)

    def test_sac_onnx_output_shape(self, tmp_path):
        """Output is (batch, 2*action_dim)."""
        import onnxruntime as ort
        agent = _sac_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "sac_actor.onnx")
        agent.export_onnx(path)

        obs  = np.zeros((3, self.OBS_DIM), dtype=np.float32)
        sess = ort.InferenceSession(path)
        out  = sess.run(["mean_logstd"], {"observation": obs})[0]
        assert out.shape == (3, self.ACTION_DIM * 2)

    def test_sac_onnx_model_valid(self, tmp_path):
        import onnx
        agent = _sac_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "sac_actor.onnx")
        agent.export_onnx(path)
        onnx.checker.check_model(onnx.load(path))


# ---------------------------------------------------------------------------
# 3. Numerical parity — TD3
# ---------------------------------------------------------------------------

class TestONNXParityTD3:
    OBS_DIM    = 6
    ACTION_DIM = 2

    def test_td3_onnx_parity(self, tmp_path):
        """action output matches PyTorch actor forward."""
        import onnxruntime as ort
        agent = _td3_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "td3_actor.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(8, self.OBS_DIM).astype(np.float32)
        with torch.no_grad():
            act_torch = agent._actor(torch.from_numpy(obs)).numpy()
        sess = ort.InferenceSession(path)
        act_onnx = sess.run(["action"], {"observation": obs})[0]

        np.testing.assert_allclose(act_torch, act_onnx, atol=1e-5)

    def test_td3_onnx_action_bounded(self, tmp_path):
        """TD3 actor applies tanh internally — outputs are in (-1, 1)."""
        import onnxruntime as ort
        agent = _td3_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "td3_actor.onnx")
        agent.export_onnx(path)

        obs  = np.random.randn(64, self.OBS_DIM).astype(np.float32)
        sess = ort.InferenceSession(path)
        act  = sess.run(["action"], {"observation": obs})[0]

        assert np.all(act > -1.0) and np.all(act < 1.0), \
            "TD3 actor (tanh) should produce actions strictly in (-1, 1)"

    def test_td3_onnx_model_valid(self, tmp_path):
        import onnx
        agent = _td3_agent(self.OBS_DIM, self.ACTION_DIM)
        path  = str(tmp_path / "td3_actor.onnx")
        agent.export_onnx(path)
        onnx.checker.check_model(onnx.load(path))


# ---------------------------------------------------------------------------
# 4. Numerical parity — DQN
# ---------------------------------------------------------------------------

class TestONNXParityDQN:
    OBS_DIM   = 8
    N_ACTIONS = 4

    def test_dqn_onnx_parity(self, tmp_path):
        """q_values output matches PyTorch Q-network forward."""
        import onnxruntime as ort
        agent = _dqn_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "dqn_q.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(8, self.OBS_DIM).astype(np.float32)
        with torch.no_grad():
            q_torch = agent._q(torch.from_numpy(obs)).numpy()
        sess = ort.InferenceSession(path)
        q_onnx = sess.run(["q_values"], {"observation": obs})[0]

        np.testing.assert_allclose(q_torch, q_onnx, atol=1e-5)

    def test_dqn_onnx_output_shape(self, tmp_path):
        """Output is (batch, n_actions)."""
        import onnxruntime as ort
        agent = _dqn_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "dqn_q.onnx")
        agent.export_onnx(path)

        obs  = np.zeros((5, self.OBS_DIM), dtype=np.float32)
        sess = ort.InferenceSession(path)
        out  = sess.run(["q_values"], {"observation": obs})[0]
        assert out.shape == (5, self.N_ACTIONS)

    def test_dqn_onnx_model_valid(self, tmp_path):
        import onnx
        agent = _dqn_agent(self.OBS_DIM, self.N_ACTIONS)
        path  = str(tmp_path / "dqn_q.onnx")
        agent.export_onnx(path)
        onnx.checker.check_model(onnx.load(path))


# ---------------------------------------------------------------------------
# 5. Numerical parity — RecurrentPPO (LSTM)
# ---------------------------------------------------------------------------

class TestONNXParityRecurrentPPO:
    OBS_DIM     = 8
    N_ACTIONS   = 4
    HIDDEN_SIZE = 16

    def test_recurrent_ppo_onnx_parity(self, tmp_path):
        """
        Math:  h_t, c_t = LSTM(obs_t, h_{t-1}, c_{t-1})
               logits_t  = actor_head(h_t)

        The ONNX model replicates this step exactly: threading the output
        (h_n, c_n) back as inputs (h_0, c_0) must produce the same logits
        as running the PyTorch modules directly.
        """
        import onnxruntime as ort
        agent = _recurrent_ppo_agent(self.OBS_DIM, self.N_ACTIONS, self.HIDDEN_SIZE)
        path  = str(tmp_path / "rppo_actor.onnx")
        agent.export_onnx(path)

        obs = np.random.randn(1, 1, self.OBS_DIM).astype(np.float32)
        h0  = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)  # (num_layers, 1, hidden)
        c0  = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)

        # PyTorch reference
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)
            ht    = torch.from_numpy(h0)
            ct    = torch.from_numpy(c0)
            out, (h_n, c_n) = agent._rnn(obs_t, (ht, ct))
            feat        = out.squeeze(1)
            logits_torch = agent._actor_head(feat).numpy()

        # ONNX
        sess = ort.InferenceSession(path)
        logits_onnx, h_n_onnx, c_n_onnx = sess.run(
            ["logits", "h_n", "c_n"],
            {"observation": obs, "h_0": h0, "c_0": c0},
        )

        np.testing.assert_allclose(logits_torch, logits_onnx, atol=1e-5)

    def test_recurrent_ppo_onnx_state_threading(self, tmp_path):
        """
        Running the ONNX model twice with state threading must match
        two sequential PyTorch forward passes.

        This proves h_n / c_n threading is lossless — the deployment
        stateful loop is identical to the PyTorch training loop.
        """
        import onnxruntime as ort
        agent = _recurrent_ppo_agent(self.OBS_DIM, self.N_ACTIONS, self.HIDDEN_SIZE)
        path  = str(tmp_path / "rppo_actor.onnx")
        agent.export_onnx(path)
        sess = ort.InferenceSession(path)

        h0 = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)
        c0 = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)

        obs_seq = np.random.randn(2, 1, 1, self.OBS_DIM).astype(np.float32)

        # ONNX: two steps with state threading
        logits1, h1, c1 = sess.run(
            ["logits", "h_n", "c_n"],
            {"observation": obs_seq[0], "h_0": h0, "c_0": c0},
        )
        logits2, _, _ = sess.run(
            ["logits", "h_n", "c_n"],
            {"observation": obs_seq[1], "h_0": h1, "c_0": c1},
        )

        # PyTorch: two sequential steps
        with torch.no_grad():
            ht, ct = torch.from_numpy(h0), torch.from_numpy(c0)
            out1, (ht, ct) = agent._rnn(torch.from_numpy(obs_seq[0]), (ht, ct))
            f1 = out1.squeeze(1)
            ref_logits1 = agent._actor_head(f1).numpy()

            out2, _ = agent._rnn(torch.from_numpy(obs_seq[1]), (ht, ct))
            f2 = out2.squeeze(1)
            ref_logits2 = agent._actor_head(f2).numpy()

        np.testing.assert_allclose(ref_logits1, logits1, atol=1e-5)
        np.testing.assert_allclose(ref_logits2, logits2, atol=1e-5)

    def test_recurrent_ppo_onnx_model_valid(self, tmp_path):
        import onnx
        agent = _recurrent_ppo_agent(self.OBS_DIM, self.N_ACTIONS, self.HIDDEN_SIZE)
        path  = str(tmp_path / "rppo_actor.onnx")
        agent.export_onnx(path)
        onnx.checker.check_model(onnx.load(path))


# ---------------------------------------------------------------------------
# 6. Training mode preserved after export
# ---------------------------------------------------------------------------

class TestTrainingModePreservation:

    def test_sac_eval_mode_preserved(self, tmp_path):
        """If actor was in eval mode before export, it stays in eval mode."""
        agent = _sac_agent()
        agent._actor.eval()
        agent.export_onnx(str(tmp_path / "sac.onnx"))
        assert not agent._actor.training

    def test_dqn_train_mode_preserved(self, tmp_path):
        agent = _dqn_agent()
        agent._q.train()
        agent.export_onnx(str(tmp_path / "dqn.onnx"))
        assert agent._q.training

    def test_agent_still_acts_after_export(self, tmp_path):
        """export_onnx() must not break the agent's act() method."""
        import gymnasium as gym
        agent = _ppo_agent()
        agent.export_onnx(str(tmp_path / "ppo.onnx"))

        obs = np.random.randn(8).astype(np.float32)
        action = agent.act(obs)
        assert 0 <= action < 4


# ---------------------------------------------------------------------------
# 7. Unsupported base agent raises NotImplementedError
# ---------------------------------------------------------------------------

class TestUnsupportedAgentExport:

    def test_base_agent_raises_not_implemented(self, tmp_path):
        from tensor_optix.core.base_agent import BaseAgent
        from tensor_optix.core.types import EpisodeData, HyperparamSet

        class _Stub(BaseAgent):
            def act(self, obs): return 0
            def learn(self, ep): return {}
            def get_hyperparams(self): return HyperparamSet(params={}, episode_id=0)
            def set_hyperparams(self, hp): pass
            def save_weights(self, path): pass
            def load_weights(self, path): pass

        stub = _Stub()
        with pytest.raises(NotImplementedError, match="ONNX export"):
            stub.export_onnx(str(tmp_path / "stub.onnx"))

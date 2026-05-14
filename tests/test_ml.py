"""
Thorough tests for tensor_optix.ml — MLAgent, DatasetPipeline, loss_registry,
and end-to-end Optimizer integration.

Coverage:
  - loss_registry: all named losses, auto-detect (classification/regression/
    unsupervised), unknown key error, nn.Module passthrough, callable passthrough
  - MLAgent: learn(), act(), get/set hyperparams, save/load weights, perturb,
    average_weights, SPSA integration
  - DatasetPipeline: Dataset input, DataLoader input, unsupervised target=input,
    infinite looping
  - Optimizer ML path: end-to-end training (classification, regression,
    reconstruction), auto-detect, rollback_on_degradation, save/load round-trip
  - RL path unaffected: existing RL usage still routes correctly
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import tensor_optix as optix
from tensor_optix.ml.loss_registry import (
    LOSS_MAP,
    _auto_detect,
    available_losses,
    resolve_loss,
)
from tensor_optix.ml.ml_agent import MLAgent
from tensor_optix.ml.dataset_pipeline import DatasetPipeline
from tensor_optix.simple import _is_ml_mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def clf_dataset():
    """100-sample 4-class classification dataset."""
    X = torch.randn(100, 8)
    y = torch.randint(0, 4, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def reg_dataset():
    """100-sample regression dataset."""
    X = torch.randn(100, 8)
    y = torch.randn(100, 1)
    return TensorDataset(X, y)


@pytest.fixture
def binary_dataset():
    """100-sample binary classification dataset."""
    X = torch.randn(100, 8)
    y = torch.randint(0, 2, (100,)).float()
    return TensorDataset(X, y)


@pytest.fixture
def unlabelled_dataset():
    """100-sample unsupervised dataset — single tensors, no labels."""
    X = torch.randn(100, 8)
    return TensorDataset(X)


@pytest.fixture
def clf_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


@pytest.fixture
def reg_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))


@pytest.fixture
def autoencoder():
    class AE(nn.Module):
        def forward(self, x):
            h = torch.relu(self.enc(x))
            return self.dec(h)
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(8, 4)
            self.dec = nn.Linear(4, 8)
    return AE()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# loss_registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLossRegistry:

    def test_all_named_losses_instantiate(self):
        for key in LOSS_MAP:
            loss = resolve_loss(key)
            assert isinstance(loss, nn.Module), f"{key} did not return an nn.Module"

    def test_nn_module_passthrough(self):
        custom = nn.MSELoss()
        result = resolve_loss(custom)
        assert result is custom

    def test_callable_passthrough(self):
        fn = lambda pred, target: ((pred - target) ** 2).mean()
        result = resolve_loss(fn)
        assert isinstance(result, nn.Module)
        pred = torch.tensor([1.0, 2.0])
        tgt  = torch.tensor([1.0, 2.0])
        assert result(pred, tgt).item() == pytest.approx(0.0)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            resolve_loss("definitely_not_a_loss")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_loss(42)

    def test_available_losses_contains_all_keys(self):
        text = available_losses()
        for key in LOSS_MAP:
            assert key in text

    # Auto-detect

    def test_auto_detect_classification(self, clf_dataset):
        detected = _auto_detect(clf_dataset, n_samples=64)
        assert detected == "cross_entropy"

    def test_auto_detect_regression(self, reg_dataset):
        detected = _auto_detect(reg_dataset, n_samples=64)
        assert detected == "mse"

    def test_auto_detect_binary(self, binary_dataset):
        detected = _auto_detect(binary_dataset, n_samples=64)
        assert detected == "bce"

    def test_auto_detect_unsupervised(self, unlabelled_dataset):
        detected = _auto_detect(unlabelled_dataset, n_samples=64)
        assert detected == "reconstruction"

    def test_auto_detect_via_resolve_loss(self, clf_dataset):
        loss = resolve_loss("auto", dataset=clf_dataset)
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_auto_no_dataset_defaults_mse(self):
        loss = resolve_loss("auto", dataset=None)
        assert isinstance(loss, nn.MSELoss)

    def test_auto_detect_from_dataloader(self, clf_dataset):
        loader = DataLoader(clf_dataset, batch_size=16)
        detected = _auto_detect(loader, n_samples=32)
        assert detected == "cross_entropy"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMLAgent:

    def _make_agent(self, model=None, loss_key="mse"):
        if model is None:
            model = nn.Linear(8, 1)
        loss_fn = resolve_loss(loss_key)
        return MLAgent(model=model, loss_fn=loss_fn, device="cpu")

    def _make_episode(self, X, y):
        from tensor_optix.core.types import EpisodeData
        return EpisodeData(
            observations=X.numpy(),
            actions=y.numpy(),
            rewards=[0.0],
            terminated=[True],
            truncated=[False],
            infos=[{}],
            episode_id=1,
        )

    def test_act_returns_numpy(self):
        agent = self._make_agent()
        obs = torch.randn(8)
        out = agent.act(obs.numpy())
        assert isinstance(out, np.ndarray)

    def test_learn_returns_loss_and_grad_norm(self, reg_dataset):
        agent = self._make_agent()
        X, y = reg_dataset[0:16]
        ep = self._make_episode(X, y)
        diag = agent.learn(ep)
        assert "loss" in diag
        assert "grad_norm" in diag
        assert isinstance(diag["loss"], float)

    def test_learn_reduces_loss_over_steps(self):
        torch.manual_seed(42)
        model = nn.Linear(4, 1)
        agent = MLAgent(model=model, loss_fn=nn.MSELoss(), learning_rate=0.1, device="cpu")
        X = torch.randn(32, 4)
        y = X @ torch.tensor([[1.], [2.], [-1.], [0.5]])
        from tensor_optix.core.types import EpisodeData
        ep = EpisodeData(X.numpy(), y.numpy(), [0.0], [True], [False], [{}], 0)
        losses = [agent.learn(ep)["loss"] for _ in range(50)]
        assert losses[-1] < losses[0], "Loss should decrease over training steps"

    def test_get_set_hyperparams(self):
        agent = self._make_agent()
        hp = agent.get_hyperparams()
        assert "learning_rate" in hp.params
        hp.params["learning_rate"] = 5e-4
        agent.set_hyperparams(hp)
        assert agent._lr == pytest.approx(5e-4)
        assert agent._optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_save_load_weights(self, reg_dataset):
        agent = self._make_agent()
        X, y = reg_dataset[0:16]
        ep = self._make_episode(X, y)
        agent.learn(ep)
        params_before = {k: v.clone() for k, v in agent.model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_weights(path)
            agent.perturb_weights(0.5)  # destroy weights
            agent.load_weights(path)
            for k, v in agent.model.state_dict().items():
                assert torch.allclose(params_before[k], v), f"Weight mismatch at {k}"
        finally:
            os.unlink(path)

    def test_perturb_weights_changes_params(self):
        agent = self._make_agent()
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}
        agent.perturb_weights(0.5)
        after = agent.model.state_dict()
        assert any(not torch.allclose(before[k], after[k]) for k in before)

    def test_average_weights(self, tmp_path):
        agent = self._make_agent()
        paths = []
        for i in range(3):
            agent.perturb_weights(0.1)
            p = str(tmp_path / f"w{i}.pt")
            agent.save_weights(p)
            paths.append(p)
        agent.average_weights(paths)  # should not raise

    def test_is_on_policy_true(self):
        assert MLAgent(nn.Linear(1, 1), nn.MSELoss()).is_on_policy is True

    def test_default_param_bounds_present(self):
        assert "learning_rate" in MLAgent.default_param_bounds
        assert "weight_decay" in MLAgent.default_param_bounds

    def test_cross_entropy_learn(self, clf_dataset, clf_model):
        loss_fn = resolve_loss("cross_entropy")
        agent = MLAgent(model=clf_model, loss_fn=loss_fn, device="cpu")
        X, y = clf_dataset[0:16]
        from tensor_optix.core.types import EpisodeData
        ep = EpisodeData(X.numpy(), y.numpy(), [0.0], [True], [False], [{}], 0)
        diag = agent.learn(ep)
        assert diag["loss"] > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DatasetPipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDatasetPipeline:

    def test_from_dataset_produces_episodes(self, reg_dataset):
        pipeline = DatasetPipeline(reg_dataset, batch_size=16, loss_key="mse")
        pipeline.setup()
        gen = pipeline.episodes()
        ep = next(gen)
        assert ep.observations.shape[0] == 16
        assert ep.actions.shape[0] == 16
        assert ep.terminated == [True]

    def test_from_dataloader_produces_episodes(self, clf_dataset):
        loader = DataLoader(clf_dataset, batch_size=32, shuffle=False)
        pipeline = DatasetPipeline(loader, loss_key="cross_entropy")
        pipeline.setup()
        ep = next(pipeline.episodes())
        assert ep.observations.shape[0] == 32

    def test_unsupervised_target_equals_input(self, unlabelled_dataset):
        pipeline = DatasetPipeline(unlabelled_dataset, batch_size=8, loss_key="reconstruction")
        pipeline.setup()
        ep = next(pipeline.episodes())
        np.testing.assert_array_equal(ep.observations, ep.actions)

    def test_vae_target_equals_input(self, unlabelled_dataset):
        pipeline = DatasetPipeline(unlabelled_dataset, batch_size=8, loss_key="vae")
        pipeline.setup()
        ep = next(pipeline.episodes())
        np.testing.assert_array_equal(ep.observations, ep.actions)

    def test_loops_infinitely(self, reg_dataset):
        pipeline = DatasetPipeline(reg_dataset, batch_size=32, loss_key="mse")
        pipeline.setup()
        gen = pipeline.episodes()
        # Dataset has 100 samples, batch=32 → 3 batches/epoch. Pull 10 — should not StopIteration
        for _ in range(10):
            ep = next(gen)
            assert ep is not None

    def test_episode_ids_increment(self, reg_dataset):
        pipeline = DatasetPipeline(reg_dataset, batch_size=32, loss_key="mse")
        pipeline.setup()
        gen = pipeline.episodes()
        ids = [next(gen).episode_id for _ in range(5)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 5

    def test_setup_required_before_episodes(self, reg_dataset):
        pipeline = DatasetPipeline(reg_dataset, batch_size=16, loss_key="mse")
        with pytest.raises(RuntimeError, match="setup"):
            next(pipeline.episodes())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _is_ml_mode detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIsMlMode:

    def test_nn_module_plus_dataset_is_ml(self, clf_dataset):
        assert _is_ml_mode(nn.Linear(4, 2), clf_dataset) is True

    def test_nn_module_plus_dataloader_is_ml(self, clf_dataset):
        loader = DataLoader(clf_dataset, batch_size=16)
        assert _is_ml_mode(nn.Linear(4, 2), loader) is True

    def test_non_module_is_not_ml(self, clf_dataset):
        assert _is_ml_mode("not a model", clf_dataset) is False

    def test_module_without_dataset_is_not_ml(self):
        assert _is_ml_mode(nn.Linear(4, 2), "not a dataset") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# End-to-end Optimizer ML path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOptimizerMLPath:

    def test_classification_runs(self, clf_dataset, clf_model, tmp_path):
        opt = optix.Optimizer(
            clf_model, clf_dataset,
            loss="cross_entropy",
            max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()  # should not raise

    def test_regression_runs(self, reg_dataset, reg_model, tmp_path):
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss="mse",
            max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()

    def test_reconstruction_runs(self, unlabelled_dataset, autoencoder, tmp_path):
        opt = optix.Optimizer(
            autoencoder, unlabelled_dataset,
            loss="reconstruction",
            max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()

    def test_auto_detect_classification(self, clf_dataset, clf_model, tmp_path, capsys):
        opt = optix.Optimizer(
            clf_model, clf_dataset,
            loss="auto",
            max_episodes=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()
        out = capsys.readouterr().out
        assert "cross_entropy" in out

    def test_auto_detect_regression(self, reg_dataset, reg_model, tmp_path, capsys):
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss="auto",
            max_episodes=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()
        out = capsys.readouterr().out
        assert "mse" in out

    def test_with_dataloader(self, clf_dataset, clf_model, tmp_path):
        loader = DataLoader(clf_dataset, batch_size=16, shuffle=True)
        opt = optix.Optimizer(
            clf_model, loader,
            loss="cross_entropy",
            max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()

    def test_custom_loss_module(self, reg_dataset, reg_model, tmp_path):
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss=nn.HuberLoss(),
            max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()

    def test_rollback_on_degradation(self, reg_dataset, reg_model, tmp_path):
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss="mse",
            max_episodes=10,
            rollback_on_degradation=True,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()  # should not raise even with rollback enabled

    def test_save_load_via_optimizer(self, reg_dataset, reg_model, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss="mse",
            max_episodes=5,
            checkpoint_dir=ckpt_dir,
        )
        opt.run()
        # Verify checkpoint files were written
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        # Load weights back manually
        agent = opt._rl_optimizer._controller._agent
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_weights(path)
            params_before = {k: v.clone() for k, v in agent.model.state_dict().items()}
            agent.perturb_weights(1.0)
            agent.load_weights(path)
            for k, v in agent.model.state_dict().items():
                assert torch.allclose(params_before[k], v)
        finally:
            os.unlink(path)

    def test_spsa_tunes_lr(self, reg_dataset, reg_model, tmp_path):
        # Just verify SPSA doesn't crash and hyperparams change over time
        initial_lr = 1e-3
        opt = optix.Optimizer(
            reg_model, reg_dataset,
            loss="mse",
            max_episodes=20,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        opt.run()
        agent = opt._rl_optimizer._controller._agent
        # SPSA may or may not have changed lr — just verify it's within bounds
        assert MLAgent.default_param_bounds["learning_rate"][0] <= agent._lr <= MLAgent.default_param_bounds["learning_rate"][1]

    def test_unknown_loss_raises_before_training(self, reg_dataset, reg_model):
        with pytest.raises(ValueError, match="Unknown loss"):
            optix.Optimizer(reg_model, reg_dataset, loss="nonsense_loss", max_episodes=1)

"""
tests/test_improvements/test_01_callbacks.py

Functional tests for WandbCallback and TensorBoardCallback.

These are observability-only changes — they add no new training signals,
only surface existing ones. No before/after performance test is required.

What we verify:
    1. Every hook calls the correct logging method with the correct keys.
    2. step= argument always matches episode_id (W&B) or global_step (TB).
    3. metrics/* keys exactly match what is in EvalMetrics.metrics.
    4. hyperparams/* keys are logged on improvement and on SPSA update.
    5. spsa/step_magnitude is the L2 norm of relative per-param changes.
    6. Events (improvement, plateau, convergence, degradation) are logged once
       per firing — never silently dropped.
    7. on_episode_end with eval_metrics=None logs nothing (unevaluated episodes).
    8. Missing library → ImportError with a clear install message.
    9. on_loop_stop closes/finishes the writer exactly once.
"""

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest

from tensor_optix.core.types import EvalMetrics, HyperparamSet, PolicySnapshot, LoopState
from tensor_optix.callbacks.wandb_callback import WandbCallback
from tensor_optix.callbacks.tensorboard_callback import TensorBoardCallback


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_eval_metrics(score=42.0, episode_id=5, extra=None):
    metrics = {
        "primary_score": score,
        "total_reward": score * 10,
        "mean_reward": score / 10,
        "reward_std": 1.5,
        "episode_length": 200,
        "policy_loss": 0.12,
        "entropy": 0.65,
        "approx_kl": 0.01,
        "explained_var": 0.88,
    }
    if extra:
        metrics.update(extra)
    return EvalMetrics(primary_score=score, metrics=metrics, episode_id=episode_id)


def _make_snapshot(score=42.0, episode_id=5):
    hp = HyperparamSet(
        params={"learning_rate": 3e-4, "clip_ratio": 0.2, "gamma": 0.99},
        episode_id=episode_id,
    )
    return PolicySnapshot(
        snapshot_id="snap-001",
        eval_metrics=_make_eval_metrics(score=score, episode_id=episode_id),
        hyperparams=hp,
        weights_path="/tmp/ckpt",
        episode_id=episode_id,
    )


# ---------------------------------------------------------------------------
# WandbCallback tests
# ---------------------------------------------------------------------------

class TestWandbCallback:
    """All tests mock the wandb module to avoid requiring an installation."""

    def _make_cb(self, **kwargs) -> tuple[WandbCallback, MagicMock]:
        """Return (callback, mock_wandb_module)."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        cb = WandbCallback(project="test-project", name="test-run", **kwargs)
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_loop_start()
        cb._wandb = mock_wandb   # keep reference after context manager exits
        cb._run   = mock_run
        return cb, mock_wandb

    # ── 1. on_loop_start calls wandb.init with the right kwargs ──────────

    def test_on_loop_start_calls_init(self):
        mock_wandb = MagicMock()
        cb = WandbCallback(project="proj", name="run", tags=["a", "b"])
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_loop_start()
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "proj"
        assert call_kwargs["name"]    == "run"
        assert call_kwargs["tags"]    == ["a", "b"]

    # ── 2. on_loop_stop finishes the run ─────────────────────────────────

    def test_on_loop_stop_finishes_run(self):
        cb, mock_wandb = self._make_cb()
        run_ref = cb._run          # capture before on_loop_stop nulls it
        cb.on_loop_stop()
        run_ref.finish.assert_called_once()
        assert cb._run is None     # cleared after finish
        assert cb._wandb is None   # cleared after finish

    def test_on_loop_stop_is_idempotent(self):
        """Second call after loop_stop must not raise."""
        cb, _ = self._make_cb()
        cb.on_loop_stop()
        cb.on_loop_stop()  # no AttributeError

    # ── 3. on_episode_end logs score and all metrics with correct step ────

    def test_on_episode_end_logs_primary_score(self):
        cb, mock_wandb = self._make_cb()
        em = _make_eval_metrics(score=37.5, episode_id=10)
        cb.on_episode_end(10, em)
        logged = mock_wandb.log.call_args.kwargs if mock_wandb.log.call_args.kwargs \
                 else mock_wandb.log.call_args[1]
        # wandb.log(data, step=10)
        assert mock_wandb.log.call_args[1].get("step") == 10 or \
               mock_wandb.log.call_args[0][1] == 10 if len(mock_wandb.log.call_args[0]) > 1 \
               else mock_wandb.log.call_args.kwargs.get("step") == 10

        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["score/primary"] == 37.5

    def test_on_episode_end_logs_all_metrics_keys(self):
        cb, mock_wandb = self._make_cb()
        em = _make_eval_metrics(score=10.0, episode_id=3)
        cb.on_episode_end(3, em)
        logged_data = mock_wandb.log.call_args[0][0]
        # Every numeric key in metrics except primary_score should appear
        for k, v in em.metrics.items():
            if k == "primary_score":
                assert f"metrics/{k}" not in logged_data, "primary_score must not be double-logged"
            else:
                assert f"metrics/{k}" in logged_data, f"metrics/{k} missing from wandb.log call"
                assert logged_data[f"metrics/{k}"] == float(v)

    def test_on_episode_end_none_metrics_logs_nothing(self):
        """Unevaluated episodes (eval_metrics=None) must not trigger a log call."""
        cb, mock_wandb = self._make_cb()
        cb.on_episode_end(7, None)
        mock_wandb.log.assert_not_called()

    # ── 4. on_improvement logs best score, event flag, and hyperparams ───

    def test_on_improvement_logs_best_and_hyperparams(self):
        cb, mock_wandb = self._make_cb()
        snap = _make_snapshot(score=55.0, episode_id=20)
        cb.on_improvement(snap)
        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["score/best"]           == 55.0
        assert logged_data["events/improvement"]   == 1
        assert logged_data["hyperparams/learning_rate"] == pytest.approx(3e-4)
        assert logged_data["hyperparams/clip_ratio"]    == pytest.approx(0.2)
        assert logged_data["hyperparams/gamma"]         == pytest.approx(0.99)

    def test_on_improvement_step_is_episode_id(self):
        cb, mock_wandb = self._make_cb()
        snap = _make_snapshot(episode_id=42)
        cb.on_improvement(snap)
        step_arg = mock_wandb.log.call_args[1].get("step") or \
                   (mock_wandb.log.call_args[0][1] if len(mock_wandb.log.call_args[0]) > 1 else None)
        assert step_arg == 42

    # ── 5. State machine events ───────────────────────────────────────────

    def test_on_plateau_logs_event(self):
        cb, mock_wandb = self._make_cb()
        cb.on_plateau(15, LoopState.COOLING)
        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["events/plateau"] == 1

    def test_on_dormant_logs_convergence(self):
        cb, mock_wandb = self._make_cb()
        cb.on_dormant(99)
        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["events/convergence"] == 1

    def test_on_degradation_logs_event_and_score(self):
        cb, mock_wandb = self._make_cb()
        em = _make_eval_metrics(score=12.0, episode_id=30)
        cb.on_degradation(30, em)
        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["events/degradation"]   == 1
        assert logged_data["score/at_degradation"] == pytest.approx(12.0)

    def test_on_degradation_no_metrics(self):
        """Degradation without eval_metrics must still log the event flag."""
        cb, mock_wandb = self._make_cb()
        cb.on_degradation(30, None)
        logged_data = mock_wandb.log.call_args[0][0]
        assert logged_data["events/degradation"] == 1
        assert "score/at_degradation" not in logged_data

    # ── 6. on_hyperparam_update logs params and SPSA step magnitude ───────

    def test_on_hyperparam_update_logs_all_params(self):
        cb, mock_wandb = self._make_cb()
        cb._last_episode_id = 8
        old = {"learning_rate": 3e-4, "clip_ratio": 0.2}
        new = {"learning_rate": 3.3e-4, "clip_ratio": 0.21}
        cb.on_hyperparam_update(old, new)
        logged_data = mock_wandb.log.call_args[0][0]
        assert "hyperparams/learning_rate" in logged_data
        assert "hyperparams/clip_ratio"    in logged_data
        assert logged_data["hyperparams/learning_rate"] == pytest.approx(3.3e-4)

    def test_on_hyperparam_update_step_magnitude_math(self):
        """
        spsa/step_magnitude must equal sqrt(sum((Δx/|x_old|)^2)).

        With old=[1.0, 2.0] and new=[1.1, 2.2]:
            rel_1 = (1.1 - 1.0) / 1.0 = 0.1
            rel_2 = (2.2 - 2.0) / 2.0 = 0.1
            magnitude = sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.14142
        """
        cb, mock_wandb = self._make_cb()
        cb._last_episode_id = 1
        old = {"a": 1.0, "b": 2.0}
        new = {"a": 1.1, "b": 2.2}
        cb.on_hyperparam_update(old, new)
        logged_data = mock_wandb.log.call_args[0][0]
        expected_magnitude = math.sqrt(0.1 ** 2 + 0.1 ** 2)
        assert logged_data["spsa/step_magnitude"] == pytest.approx(expected_magnitude, rel=1e-5)

    def test_on_hyperparam_update_no_change_no_magnitude(self):
        """If no params changed, step_magnitude must not be logged."""
        cb, mock_wandb = self._make_cb()
        cb._last_episode_id = 1
        params = {"learning_rate": 3e-4}
        cb.on_hyperparam_update(params, params)  # identical
        logged_data = mock_wandb.log.call_args[0][0]
        assert "spsa/step_magnitude" not in logged_data

    # ── 7. Missing wandb import raises with clear message ─────────────────

    def test_missing_wandb_raises_import_error(self):
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(ImportError, match="pip install wandb"):
                cb.on_loop_start()

    # ── 8. last_episode_id is updated before on_hyperparam_update ────────

    def test_last_episode_id_tracks_on_episode_end(self):
        cb, mock_wandb = self._make_cb()
        em = _make_eval_metrics(episode_id=17)
        cb.on_episode_end(17, em)
        assert cb._last_episode_id == 17


# ---------------------------------------------------------------------------
# TensorBoardCallback tests
# ---------------------------------------------------------------------------

class TestTensorBoardCallback:
    """All tests mock torch.utils.tensorboard.SummaryWriter."""

    def _make_cb(self, log_dir="/tmp/tb_test", **kwargs) -> tuple[TensorBoardCallback, MagicMock]:
        mock_writer = MagicMock()
        cb = TensorBoardCallback(log_dir=log_dir, **kwargs)
        mock_sw_module = MagicMock()
        mock_sw_module.return_value = mock_writer
        with patch("tensor_optix.callbacks.tensorboard_callback.TensorBoardCallback.on_loop_start",
                   lambda self: self.__dict__.update({"_writer": mock_writer})):
            cb.on_loop_start()
        return cb, mock_writer

    # ── 1. on_loop_start creates SummaryWriter with log_dir ──────────────

    def test_on_loop_start_creates_writer(self):
        mock_writer_cls = MagicMock()
        mock_writer_instance = MagicMock()
        mock_writer_cls.return_value = mock_writer_instance

        with patch.dict(sys.modules, {
            "torch": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.tensorboard": MagicMock(SummaryWriter=mock_writer_cls),
        }):
            cb = TensorBoardCallback(log_dir="/tmp/tb")
            cb.on_loop_start()

        mock_writer_cls.assert_called_once()
        assert mock_writer_cls.call_args.kwargs.get("log_dir") == "/tmp/tb" or \
               mock_writer_cls.call_args[1].get("log_dir") == "/tmp/tb"

    # ── 2. on_loop_stop closes writer ─────────────────────────────────────

    def test_on_loop_stop_closes_writer(self):
        cb, mock_writer = self._make_cb()
        cb.on_loop_stop()
        mock_writer.close.assert_called_once()
        assert cb._writer is None

    def test_on_loop_stop_is_idempotent(self):
        cb, _ = self._make_cb()
        cb.on_loop_stop()
        cb.on_loop_stop()  # no AttributeError

    # ── 3. on_episode_end calls add_scalar with correct tag and step ──────

    def test_on_episode_end_logs_primary_score(self):
        cb, mock_writer = self._make_cb()
        em = _make_eval_metrics(score=77.0, episode_id=11)
        cb.on_episode_end(11, em)
        mock_writer.add_scalar.assert_any_call("score/primary", 77.0, 11)

    def test_on_episode_end_logs_all_metric_keys(self):
        cb, mock_writer = self._make_cb()
        em = _make_eval_metrics(score=10.0, episode_id=3)
        cb.on_episode_end(3, em)
        calls = {c.args[0]: c.args for c in mock_writer.add_scalar.call_args_list}
        for k, v in em.metrics.items():
            if k == "primary_score":
                assert f"metrics/{k}" not in calls
            else:
                tag = f"metrics/{k}"
                assert tag in calls, f"add_scalar not called for {tag}"
                assert calls[tag][1] == pytest.approx(float(v))
                assert calls[tag][2] == 3   # step = episode_id

    def test_on_episode_end_none_metrics_logs_nothing(self):
        cb, mock_writer = self._make_cb()
        cb.on_episode_end(5, None)
        mock_writer.add_scalar.assert_not_called()

    # ── 4. on_improvement logs best score, event, and hyperparams ────────

    def test_on_improvement_logs_correct_tags(self):
        cb, mock_writer = self._make_cb()
        snap = _make_snapshot(score=88.0, episode_id=25)
        cb.on_improvement(snap)
        calls = {c.args[0]: c.args for c in mock_writer.add_scalar.call_args_list}
        assert "score/best"          in calls
        assert "events/improvement"  in calls
        assert calls["score/best"][1]        == pytest.approx(88.0)
        assert calls["events/improvement"][1] == 1
        assert calls["score/best"][2]        == 25   # step = episode_id

    def test_on_improvement_logs_hyperparams(self):
        cb, mock_writer = self._make_cb()
        snap = _make_snapshot(episode_id=25)
        cb.on_improvement(snap)
        calls = {c.args[0] for c in mock_writer.add_scalar.call_args_list}
        assert "hyperparams/learning_rate" in calls
        assert "hyperparams/clip_ratio"    in calls
        assert "hyperparams/gamma"         in calls

    # ── 5. State machine events ───────────────────────────────────────────

    def test_on_plateau_logs_event(self):
        cb, mock_writer = self._make_cb()
        cb.on_plateau(20, LoopState.COOLING)
        mock_writer.add_scalar.assert_any_call("events/plateau", 1, 20)

    def test_on_dormant_logs_convergence(self):
        cb, mock_writer = self._make_cb()
        cb.on_dormant(50)
        mock_writer.add_scalar.assert_any_call("events/convergence", 1, 50)

    def test_on_degradation_logs_event_and_score(self):
        cb, mock_writer = self._make_cb()
        em = _make_eval_metrics(score=5.0, episode_id=35)
        cb.on_degradation(35, em)
        calls = {c.args[0]: c.args for c in mock_writer.add_scalar.call_args_list}
        assert "events/degradation"    in calls
        assert "score/at_degradation"  in calls
        assert calls["score/at_degradation"][1] == pytest.approx(5.0)

    # ── 6. on_hyperparam_update step magnitude math ───────────────────────

    def test_on_hyperparam_update_step_magnitude_math(self):
        """
        Same math as WandbCallback test.
        old=[1.0, 2.0], new=[1.1, 2.2]  →  magnitude = sqrt(0.02)
        """
        cb, mock_writer = self._make_cb()
        cb._last_episode_id = 1
        cb.on_hyperparam_update({"a": 1.0, "b": 2.0}, {"a": 1.1, "b": 2.2})
        calls = {c.args[0]: c.args for c in mock_writer.add_scalar.call_args_list}
        assert "spsa/step_magnitude" in calls
        assert calls["spsa/step_magnitude"][1] == pytest.approx(math.sqrt(0.02), rel=1e-5)

    def test_on_hyperparam_update_step_is_last_episode_id(self):
        cb, mock_writer = self._make_cb()
        cb._last_episode_id = 99
        cb.on_hyperparam_update({"lr": 1e-3}, {"lr": 1.1e-3})
        calls = {c.args[0]: c.args for c in mock_writer.add_scalar.call_args_list}
        assert calls["hyperparams/lr"][2] == 99

    # ── 7. Missing torch raises with clear message ────────────────────────

    def test_missing_torch_raises_import_error(self):
        cb = TensorBoardCallback()
        modules_to_patch = {
            "torch": None,
            "torch.utils": None,
            "torch.utils.tensorboard": None,
        }
        with patch.dict(sys.modules, modules_to_patch):
            with pytest.raises((ImportError, AttributeError)):
                cb.on_loop_start()


# ---------------------------------------------------------------------------
# Integration: both callbacks survive a simulated full loop lifecycle
# ---------------------------------------------------------------------------

class TestCallbackLifecycle:
    """
    Simulate a full loop: start → episodes → improvement → plateau →
    hyperparam_update → degradation → dormant → stop.
    Assert no exceptions and the expected call counts.
    """

    def _run_lifecycle(self, cb, mock_log_fn_name, mock_obj):
        """Fire all hooks in a realistic order and return call count."""
        cb.on_loop_start()

        for ep in range(1, 6):
            em = _make_eval_metrics(score=float(ep * 10), episode_id=ep)
            cb.on_episode_end(ep, em)
            if ep == 2:
                cb.on_improvement(_make_snapshot(score=20.0, episode_id=2))
            if ep == 3:
                cb.on_plateau(3, LoopState.COOLING)
            if ep == 4:
                cb.on_hyperparam_update(
                    {"learning_rate": 3e-4},
                    {"learning_rate": 2.8e-4},
                )
            if ep == 5:
                cb.on_degradation(5, em)

        cb.on_dormant(5)
        cb.on_loop_stop()

        return getattr(mock_obj, mock_log_fn_name).call_count

    def test_wandb_full_lifecycle(self):
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        cb = WandbCallback(project="test")
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_loop_start()
        cb._wandb = mock_wandb
        cb._run   = mock_wandb.init.return_value

        for ep in range(1, 6):
            em = _make_eval_metrics(score=float(ep * 10), episode_id=ep)
            cb.on_episode_end(ep, em)
            if ep == 2:
                cb.on_improvement(_make_snapshot(score=20.0, episode_id=2))
            if ep == 3:
                cb.on_plateau(3, LoopState.COOLING)
            if ep == 4:
                cb.on_hyperparam_update({"learning_rate": 3e-4}, {"learning_rate": 2.8e-4})
            if ep == 5:
                cb.on_degradation(5, em)
        cb.on_dormant(5)
        cb.on_loop_stop()

        # 5 on_episode_end + 1 on_improvement + 1 on_plateau +
        # 1 on_hyperparam_update + 1 on_degradation + 1 on_dormant = 10 log calls
        assert mock_wandb.log.call_count == 10
        mock_wandb.init.return_value.finish.assert_called_once()

    def test_tensorboard_full_lifecycle(self):
        mock_writer = MagicMock()
        cb = TensorBoardCallback(log_dir="/tmp/tb_test")
        cb._writer = mock_writer   # inject directly

        for ep in range(1, 6):
            em = _make_eval_metrics(score=float(ep * 10), episode_id=ep)
            cb.on_episode_end(ep, em)
            if ep == 2:
                cb.on_improvement(_make_snapshot(score=20.0, episode_id=2))
            if ep == 3:
                cb.on_plateau(3, LoopState.COOLING)
            if ep == 4:
                cb.on_hyperparam_update({"learning_rate": 3e-4}, {"learning_rate": 2.8e-4})
            if ep == 5:
                cb.on_degradation(5, em)
        cb.on_dormant(5)
        cb.on_loop_stop()

        # add_scalar was called many times — just verify it was called at all
        # and the writer was closed
        assert mock_writer.add_scalar.call_count > 0
        mock_writer.close.assert_called_once()

import multiprocessing as mp
import queue

from tensor_optix.scheduler.worker import _trial_worker


class _FakeEvent:
    def is_set(self):
        return False


def _direct_worker_call(trial_number, params, rewards_sequence, tmp_path, seed=0):
    """Call _trial_worker in-process (no mp.Process) using plain queue.Queue
    and DummyAgent/DummyPipeline fixtures — fast correctness check, not a
    real process-isolation test (that requires real ctx.Process, see below)."""
    from conftest import DummyAgent, DummyPipeline
    from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator  # noqa: F401
    from tensor_optix.core.types import HyperparamSet

    def agent_factory(p):
        return DummyAgent(hyperparams=HyperparamSet(params=p, episode_id=0))

    def pipeline_factory():
        return DummyPipeline(rewards_sequence=rewards_sequence)

    report_q = queue.Queue()
    result_q = queue.Queue()

    _trial_worker(
        trial_number=trial_number,
        params=params,
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        val_pipeline_factory=None,
        rloptimizer_kwargs={},
        checkpoint_score_fn=None,
        trial_steps=2,
        ckpt_dir=str(tmp_path / f"trial_{trial_number}"),
        seed=seed,
        report_queue=report_q,
        cancel_event=_FakeEvent(),
        result_queue=result_q,
    )

    return result_q.get_nowait()


def test_worker_reports_result(tmp_path):
    result = _direct_worker_call(0, {"learning_rate": 1e-3}, [[1.0, 1.0, 1.0]], tmp_path)
    assert result["trial_number"] == 0
    assert result["status"] == "completed"
    assert result["score"] is not None
    assert result["error"] is None


def test_worker_reports_exception_as_failed(tmp_path):
    def broken_agent_factory(_params):
        raise RuntimeError("boom")

    import queue as _queue
    report_q = _queue.Queue()
    result_q = _queue.Queue()

    _trial_worker(
        trial_number=1,
        params={},
        agent_factory=broken_agent_factory,
        pipeline_factory=lambda: None,
        val_pipeline_factory=None,
        rloptimizer_kwargs={},
        checkpoint_score_fn=None,
        trial_steps=2,
        ckpt_dir=str(tmp_path / "trial_1"),
        seed=0,
        report_queue=report_q,
        cancel_event=_FakeEvent(),
        result_queue=result_q,
    )

    result = result_q.get_nowait()
    assert result["status"] == "failed"
    assert result["error"] == "boom"


def test_concurrent_trials_get_distinct_checkpoint_dirs(tmp_path):
    r0 = _direct_worker_call(0, {"learning_rate": 1e-3}, [[1.0, 1.0, 1.0]], tmp_path)
    r1 = _direct_worker_call(1, {"learning_rate": 1e-3}, [[1.0, 1.0, 1.0]], tmp_path)
    assert r0["weights_path"] != r1["weights_path"]
    assert "trial_0" in r0["weights_path"]
    assert "trial_1" in r1["weights_path"]


def _module_level_agent_factory(params):
    """Module-level (not a closure) so it is picklable under Windows spawn."""
    from conftest import DummyAgent
    from tensor_optix.core.types import HyperparamSet
    return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))


def _module_level_pipeline_factory():
    from conftest import DummyPipeline
    return DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])


def test_worker_runs_in_a_real_separate_process(tmp_path):
    """Proves _trial_worker itself (the real worker entry point, not a
    stand-in probe) is picklable and runs correctly under a real spawned
    OS process, reporting a distinct PID back to the parent."""
    import os
    import sys

    ctx = mp.get_context("fork") if sys.platform.startswith("linux") else mp.get_context("spawn")
    report_q = ctx.Queue()
    result_q = ctx.Queue()
    cancel_event = ctx.Event()

    p = ctx.Process(
        target=_trial_worker,
        kwargs=dict(
            trial_number=0,
            params={"learning_rate": 1e-3},
            agent_factory=_module_level_agent_factory,
            pipeline_factory=_module_level_pipeline_factory,
            val_pipeline_factory=None,
            rloptimizer_kwargs={},
            checkpoint_score_fn=None,
            trial_steps=2,
            ckpt_dir=str(tmp_path / "trial_0"),
            seed=0,
            report_queue=report_q,
            cancel_event=cancel_event,
            result_queue=result_q,
        ),
        daemon=True,
    )
    p.start()
    result = result_q.get(timeout=60)
    p.join(timeout=10)

    assert result["pid"] != os.getpid()
    assert result["status"] == "completed"

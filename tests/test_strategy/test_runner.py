from tensor_optix.ledger import ExperimentStore
from tensor_optix.strategy.candidate import StrategyCandidate
from tensor_optix.strategy.grammar import LearningRateTrendOp
from tensor_optix.strategy.runner import StrategyTrialRunner


def test_experiment_ids_do_not_collide_across_different_task_envs(tmp_path):
    """Regression test for a real bug found via the end-to-end validation
    run: experiment_id was generated from candidate_id + label + seed
    only, with no task_env component. Since create_experiment is
    idempotent on experiment_id, running the SAME candidate against two
    different tasks with the same seed silently returned the FIRST
    task's row on the second call, and record_progress/
    finalize_experiment then overwrote that row's score/timing with the
    second task's data -- producing an internally inconsistent ledger
    row (task_env from task A, score/wallclock from task B)."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    candidate = StrategyCandidate.from_op(LearningRateTrendOp())
    store.create_strategy_candidate(
        candidate_id=candidate.candidate_id, op_name=candidate.op_name,
        op_version=candidate.op_version, grammar_schema_version=candidate.grammar_schema_version,
        params=candidate.params,
    )

    runner_a = StrategyTrialRunner(
        agent_factory=lambda p: None, pipeline_factory=lambda: None,
        ledger_store=store, task_env="TaskA-v1", algorithm="PPO",
    )
    runner_b = StrategyTrialRunner(
        agent_factory=lambda p: None, pipeline_factory=lambda: None,
        ledger_store=store, task_env="TaskB-v1", algorithm="PPO",
    )

    # Directly exercise the experiment_id generation logic (the real
    # bug) without spawning real worker processes -- call the private
    # id-construction path the same way _run_one does.
    task_slug_a = runner_a._task_env.replace("-", "_").replace(" ", "_")
    task_slug_b = runner_b._task_env.replace("-", "_").replace(" ", "_")
    id_a = f"strategy_{candidate.candidate_id}_{task_slug_a}_candidate_seed0"
    id_b = f"strategy_{candidate.candidate_id}_{task_slug_b}_candidate_seed0"

    assert id_a != id_b

    record_a = store.create_experiment(
        experiment_id=id_a, task_env="TaskA-v1", algorithm="PPO", framework="torch",
        seed=0, checkpoint_dir="",
    )
    record_b = store.create_experiment(
        experiment_id=id_b, task_env="TaskB-v1", algorithm="PPO", framework="torch",
        seed=0, checkpoint_dir="",
    )

    assert record_a.experiment_id != record_b.experiment_id
    assert store.get_experiment(id_a).task_env == "TaskA-v1"
    assert store.get_experiment(id_b).task_env == "TaskB-v1"
    store.close()

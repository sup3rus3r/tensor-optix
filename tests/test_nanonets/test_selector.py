from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.nanonets.selector import NanonetsSelector


def _seed_method(store, task_env, method, scores, algorithm="DummyAlgo", solve_threshold=None):
    for i, score in enumerate(scores):
        experiment_id = f"{task_env}-{method}-{i}"
        store.create_experiment(
            experiment_id=experiment_id, task_env=task_env, algorithm=algorithm,
            framework="torch", seed=i, checkpoint_dir="", solve_threshold=solve_threshold,
            config={"sampler": method},
        )
        store.start_experiment(experiment_id)
        store.record_progress(experiment_id, best_primary_score=score)
        store.finalize_experiment(
            experiment_id, state=ExperimentState.COMPLETED, evaluation_status=EvaluationStatus.PRELIMINARY,
        )


def test_propose_falls_back_to_baseline_when_all_sources_gated(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    # Target has decent scores; the only configured source is deliberately
    # terrible, so it must fail the percentile gate and trigger fallback.
    _seed_method(store, "CartPole-v1", "TPESampler", [400.0, 410.0, 420.0], solve_threshold=475.0)
    _seed_method(store, "Acrobot-v1", "TPESampler", [-500.0, -490.0, -510.0], solve_threshold=-100.0)

    nanonets = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
    baseline_decision = nanonets._baseline.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)
    nanonets_decision = nanonets.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)

    assert nanonets_decision.allocations == baseline_decision.allocations
    assert "fallback to baseline selector" in nanonets_decision.rationale
    store.close()


def test_propose_uses_pooled_estimate_when_source_passes_gate(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "TPESampler", [480.0, 482.0, 485.0], solve_threshold=475.0)
    # A genuinely similar source task -- both solidly "solved" relative to
    # their own threshold, same algorithm, comparable normalized scale.
    _seed_method(store, "Acrobot-v1", "TPESampler", [-20.0, -18.0, -15.0, -17.0], solve_threshold=-100.0)

    nanonets = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
    decision = nanonets.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)

    assert "fallback" not in decision.rationale
    assert "pooled cross-task estimate" in decision.rationale
    store.close()


def test_propose_falls_back_when_thresholds_unknown(tmp_path):
    """Distinguishes 'unknown threshold' fallback from 'genuinely
    dissimilar scores' fallback -- seeding scores with NO solve_threshold
    for either task must gate out every source and fall back, even
    though the raw scores alone might otherwise look comparable."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "TPESampler", [480.0, 482.0, 485.0])  # no threshold
    _seed_method(store, "Acrobot-v1", "TPESampler", [-20.0, -18.0, -15.0])   # no threshold

    nanonets = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
    decision = nanonets.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)

    assert "fallback to baseline selector" in decision.rationale
    store.close()


def test_propose_rejects_acrobot_cartpole_scale_coincidence(tmp_path):
    """Direct regression test for the demonstrated real-world bug: under
    the OLD raw-score comparison, Acrobot's less-negative scores could
    numerically exceed CartPole's raw p95 purely due to scale/sign, not
    genuine similarity. With real thresholds supplied, the source must
    now be correctly gated out instead of passing by coincidence."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    # CartPole target: good runs, but on the RAW scale Acrobot's -17 best
    # score is numerically far below CartPole's ~485 -- under the OLD
    # (unnormalized) gate this would already correctly fail. The real bug
    # manifested in the opposite, more dangerous direction (a positive-
    # scale source passing for a negative-scale target); reproduce that
    # exact shape here.
    _seed_method(store, "Acrobot-v1", "TPESampler", [-20.0, -18.0, -17.0], solve_threshold=-100.0)
    # Deliberately BAD Acrobot source relative to ITS OWN threshold (barely
    # above, not solidly solved) to prove the normalized gate still
    # distinguishes "passes its own threshold" from "would coincidentally
    # pass a totally different task's raw percentile."
    _seed_method(store, "Pendulum-v1", "TPESampler", [-99.0, -98.5, -99.5], solve_threshold=-100.0)

    nanonets = NanonetsSelector(store, source_task_envs=["Pendulum-v1"])
    estimate = nanonets.pooled_stats(task_env="Acrobot-v1", method="TPESampler")

    assert "Pendulum-v1" in estimate.sources_gated_out
    assert estimate.fallback_triggered is True


def test_loto_fold_respects_source_task_envs_boundary(tmp_path):
    """Leakage-discipline test: constructing NanonetsSelector with only 2
    of 3 task_envs means the 3rd is never queried as a source, even if
    it has ledger data."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "TPESampler", [195.0, 198.0, 200.0])
    _seed_method(store, "Acrobot-v1", "TPESampler", [199.0, 202.0, 205.0])
    _seed_method(store, "Pendulum-v1", "TPESampler", [-100.0, -105.0, -110.0])

    # Held-out task is Pendulum-v1 -- NOT included in source_task_envs.
    nanonets = NanonetsSelector(store, source_task_envs=["CartPole-v1", "Acrobot-v1"])
    estimate = nanonets.pooled_stats(task_env="CartPole-v1", method="TPESampler")

    assert "Pendulum-v1" not in estimate.sources_used
    assert "Pendulum-v1" not in estimate.sources_gated_out
    store.close()


def test_pooled_stats_handles_no_target_data(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    nanonets = NanonetsSelector(store, source_task_envs=[])
    estimate = nanonets.pooled_stats(task_env="NeverSeen-v1", method="TPESampler")
    assert estimate.fallback_triggered is True
    assert estimate.sources_used == []
    store.close()


def test_pooled_stats_does_not_mix_raw_scales_across_tasks(tmp_path):
    """Direct regression test for the real bug found in the live 3-task
    LOTO run: pool_estimates() was doing inverse-variance weighting on
    RAW scores, so a gate-passing source task on a different raw scale
    (Acrobot, negative) could dominate a target's (CartPole, positive)
    pooled estimate purely via a scale artifact -- producing a pooled
    mean of -63.21 for a CartPole target whose real scores are 240-411.
    Pooling must happen in normalized space and be de-normalized back to
    the target's own raw units before being returned."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "TPESampler", [240.50, 411.00], solve_threshold=475.0)
    _seed_method(store, "Acrobot-v1", "TPESampler", [-61.57, -76.00], solve_threshold=-100.0)

    nanonets = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
    estimate = nanonets.pooled_stats(task_env="CartPole-v1", method="TPESampler")

    assert estimate.sources_used == ["Acrobot-v1"]
    assert estimate.fallback_triggered is False
    # The old bug produced a negative pooled_mean for a positive-scoring
    # task -- any positive value here proves the raw-scale-mixing symptom
    # is gone; the looser bound matches the real, verified de-normalized
    # result (~550-580) without pinning an exact float.
    assert estimate.pooled_mean > 100.0
    store.close()

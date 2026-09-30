"""
End-to-end regression coverage for the "unicorn" checkpoint bug: a single
raw-scalar-comparison lucky episode (high reward concentrated on one step,
i.e. low breadth) permanently blocking a later, genuinely-better, broader
policy whose raw score happens to be lower.
"""
import pytest

from conftest import DummyAgent, DummyEvaluator, DummyPipeline
from tensor_optix.core.loop_controller import LoopController
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer


def make_unicorn_sequence():
    """
    ep0-4: baseline, broad reward (breadth ~1), sum=20 each — warms up
           breadth_history past min_samples so typicality is active by ep5.
    ep5:   the "unicorn" — reward concentrated entirely on one step, sum=95.
           Higher raw score than anything that follows, but structurally
           narrow (low breadth).
    ep6+:  a genuinely, durably better policy — broad reward (breadth ~1),
           sum=60 each. Raw score is LOWER than the unicorn's 95, so a naive
           raw-point comparison can never displace it.
    """
    seq = [[5.0, 5.0, 5.0, 5.0]] * 5          # ep0 (cold start) .. ep4
    seq += [[0.0, 0.0, 0.0, 95.0]]            # ep5 — unicorn
    seq += [[15.0, 15.0, 15.0, 15.0]] * 10    # ep6+ — genuinely better, broad
    return seq


def make_controller(criteria_mode, tmp_path, **kwargs):
    agent = DummyAgent()
    evaluator = DummyEvaluator()
    pipeline = DummyPipeline(rewards_sequence=make_unicorn_sequence())
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=20)
    scheduler = BackoffScheduler(base_interval=1, plateau_threshold=50, dormant_threshold=100)
    optimizer = BackoffOptimizer()
    controller = LoopController(
        agent=agent,
        evaluator=evaluator,
        optimizer=optimizer,
        pipeline=pipeline,
        checkpoint_registry=registry,
        backoff_scheduler=scheduler,
        max_episodes=16,
        criteria_mode=criteria_mode,
        **kwargs,
    )
    return controller


class TestLegacyBehaviorUnchangedByDefault:

    def test_default_criteria_mode_reproduces_unicorn_bug(self, tmp_path):
        # Proves the default (criteria_mode="none") is bit-identical to the
        # pre-existing behavior: the unicorn permanently wins, exactly the
        # bug this whole mechanism exists to fix when opted into.
        controller = make_controller(criteria_mode="none", tmp_path=tmp_path)
        controller.run()
        assert controller.best_snapshot.eval_metrics.primary_score == 95.0
        assert controller.best_snapshot.eval_metrics.episode_id == 5


class TestAutoCriteriaModeFixesUnicorn:

    def test_auto_mode_lets_broad_policy_displace_the_unicorn(self, tmp_path):
        controller = make_controller(
            criteria_mode="auto",
            tmp_path=tmp_path,
            checkpoint_confirm_window=2,
        )
        controller.run()
        # The broad, durably-better policy (raw sum=60) must win despite
        # scoring lower raw reward than the narrow unicorn (raw sum=95).
        assert controller.best_snapshot.eval_metrics.episode_id != 5
        assert controller.best_snapshot.eval_metrics.primary_score == 60.0


class TestGracefulDegradation:

    def test_short_run_still_saves_first_checkpoint(self, tmp_path):
        controller = make_controller(
            criteria_mode="auto", tmp_path=tmp_path,
        )
        controller._max_episodes = 1  # cold start only
        controller.run()
        assert controller.best_snapshot is not None

    def test_manual_mode_with_no_criteria_override_is_a_no_op(self, tmp_path):
        # BaseEvaluator.criteria() defaults to {} — manual mode with no
        # override must behave exactly like "none".
        controller_none = make_controller(criteria_mode="none", tmp_path=tmp_path)
        controller_none.run()
        controller_manual = make_controller(criteria_mode="manual", tmp_path=tmp_path)
        controller_manual.run()
        assert (
            controller_none.best_snapshot.eval_metrics.primary_score
            == controller_manual.best_snapshot.eval_metrics.primary_score
        )


class TestCompareOverrideRespected:

    def test_user_compare_override_is_actually_invoked(self, tmp_path):
        class NeverImprove(DummyEvaluator):
            def compare(self, candidate, baseline, confirmation=None):
                return False

        agent = DummyAgent()
        evaluator = NeverImprove()
        pipeline = DummyPipeline(rewards_sequence=[[float(i)] * 3 for i in range(1, 20)])
        registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=20)
        scheduler = BackoffScheduler(base_interval=1, plateau_threshold=50, dormant_threshold=100)
        optimizer = BackoffOptimizer()
        controller = LoopController(
            agent=agent, evaluator=evaluator, optimizer=optimizer, pipeline=pipeline,
            checkpoint_registry=registry, backoff_scheduler=scheduler,
            max_episodes=10, criteria_mode="auto",
        )
        controller.run()
        # Only the cold-start checkpoint (episode 0) should ever be saved —
        # every later episode's compare() unconditionally returns False.
        assert controller.best_snapshot.eval_metrics.episode_id == 0


class TestSkipMessageDistinguishesReasons:
    """
    Regression coverage: the verbose skip message must not claim a magnitude
    comparison ("ckpt_score < best") when the actual rejection reason was
    "unconfirmed" (candidate exceeded best but hasn't been corroborated yet).
    That would state something false about why the candidate was rejected.
    """

    def test_unconfirmed_skip_message_does_not_claim_below_best(self, tmp_path, capsys):
        controller = make_controller(
            criteria_mode="auto",
            tmp_path=tmp_path,
            checkpoint_confirm_window=5,  # deliberately hard to satisfy quickly
            verbose=True,
        )
        controller.run()
        out = capsys.readouterr().out
        unconfirmed_lines = [l for l in out.splitlines() if "unconfirmed" in l]
        for line in unconfirmed_lines:
            assert "<" not in line  # must not claim a magnitude comparison
            assert ">=" in line

    def test_below_best_skip_message_unchanged_for_legacy_mode(self, tmp_path, capsys):
        controller = make_controller(criteria_mode="none", tmp_path=tmp_path, verbose=True)
        controller.run()
        out = capsys.readouterr().out
        skip_lines = [l for l in out.splitlines() if "CKPT" in l and "skipped" in l]
        assert skip_lines  # sanity: some episodes were skipped
        for line in skip_lines:
            assert "skipped (below best)" in line
            assert "<" in line


class TestCheckpointScoreFnInteraction:

    def test_checkpoint_score_fn_skips_composite_but_keeps_confirmation(self, tmp_path):
        scores = iter([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 95.0, 20.0, 20.0, 20.0])

        def ckpt_fn(agent):
            return next(scores, 20.0)

        agent = DummyAgent()
        evaluator = DummyEvaluator()
        pipeline = DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]] * 30)
        registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=20)
        scheduler = BackoffScheduler(base_interval=1, plateau_threshold=50, dormant_threshold=100)
        optimizer = BackoffOptimizer()
        controller = LoopController(
            agent=agent, evaluator=evaluator, optimizer=optimizer, pipeline=pipeline,
            checkpoint_registry=registry, backoff_scheduler=scheduler,
            max_episodes=10, criteria_mode="auto", checkpoint_score_fn=ckpt_fn,
            checkpoint_confirm_window=2,
        )
        controller.run()
        # composite adjustment skipped -> ckpt_score used as-is; the isolated
        # 95.0 spike still needs confirmation (not applied blindly), so the
        # eventual best should reflect the sustained 20.0 level, not the
        # one-off 95.0 spike.
        assert controller.best_snapshot.eval_metrics.primary_score != 95.0

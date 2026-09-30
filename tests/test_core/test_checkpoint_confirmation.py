from tensor_optix.core.checkpoint_confirmation import CheckpointConfirmationTracker


class TestCheckpointConfirmationTracker:

    def test_first_checkpoint_always_accepted(self):
        tracker = CheckpointConfirmationTracker()
        tracker.observe(10.0)
        accepted, reason = tracker.evaluate_candidate(10.0, best_score=None)
        assert accepted
        assert reason == "first_checkpoint"

    def test_unconfirmed_isolated_spike_is_rejected(self):
        # The "unicorn" scenario at the confirmation layer: one episode above
        # best, but not far enough outside the run's own measured noise to
        # be self-evidently real — must wait for corroboration.
        tracker = CheckpointConfirmationTracker(confirm_window=3, min_samples_for_floor=5)
        best = 100.0
        # High-variance seed history (std ~8.9): floor = min(2*8.9, 0.5*100) ≈ 17.9
        for s in [90.0, 110.0, 90.0, 110.0, 100.0]:
            tracker.observe(s)
        tracker.observe(112.0)  # jump of 12, below the ~17.9 floor
        accepted, reason = tracker.evaluate_candidate(112.0, best_score=best)
        assert not accepted
        assert reason == "unconfirmed"

    def test_spike_that_immediately_regresses_is_not_confirmed_by_streak(self):
        tracker = CheckpointConfirmationTracker(confirm_window=3, min_samples_for_floor=5)
        best = 100.0
        for s in [90.0, 110.0, 90.0, 110.0, 100.0]:
            tracker.observe(s)

        tracker.observe(112.0)
        accepted, _ = tracker.evaluate_candidate(112.0, best_score=best)
        assert not accepted  # streak = 1

        # Regresses back to baseline (at-or-below best) — streak resets to 0.
        tracker.observe(100.0)
        accepted2, reason2 = tracker.evaluate_candidate(100.0, best_score=best)
        assert not accepted2
        assert reason2 == "below_best"

        # Even a later spike at the same unconfirmed level starts from streak=1.
        tracker.observe(112.0)
        accepted3, _ = tracker.evaluate_candidate(112.0, best_score=best)
        assert not accepted3

    def test_sustained_improvement_gets_confirmed_by_streak(self):
        tracker = CheckpointConfirmationTracker(confirm_window=3, min_samples_for_floor=5)
        best = 100.0
        for s in [90.0, 110.0, 90.0, 110.0, 100.0]:
            tracker.observe(s)

        # A durably better policy: score stays consistently above best
        # (same magnitude as the isolated-spike test above), corroborated
        # across confirm_window consecutive evals this time.
        tracker.observe(112.0)
        accepted1, _ = tracker.evaluate_candidate(112.0, best_score=best)
        assert not accepted1  # streak = 1

        tracker.observe(113.0)
        accepted2, _ = tracker.evaluate_candidate(113.0, best_score=best)
        assert not accepted2  # streak = 2

        tracker.observe(111.0)
        accepted3, reason3 = tracker.evaluate_candidate(111.0, best_score=best)
        assert accepted3  # streak = 3 == confirm_window
        assert reason3 == "confirmed_by_streak"

    def test_low_noise_regime_confirms_immediately(self):
        # Deterministic/near-zero-variance history (e.g. checkpoint_score_fn
        # with a fixed seed) — a single clear improvement should not have to
        # wait for corroboration.
        tracker = CheckpointConfirmationTracker(noise_k=2.0, min_samples_for_floor=5)
        for s in [10.0, 10.0, 10.0, 10.0, 10.0]:
            tracker.observe(s)
        tracker.observe(20.0)
        accepted, reason = tracker.evaluate_candidate(20.0, best_score=10.0)
        assert accepted
        assert reason == "confirmed_by_floor"

    def test_candidate_not_exceeding_best_is_rejected(self):
        tracker = CheckpointConfirmationTracker()
        tracker.observe(10.0)
        tracker.observe(5.0)
        accepted, reason = tracker.evaluate_candidate(5.0, best_score=10.0)
        assert not accepted
        assert reason == "below_best"

    def test_last_reason_mirrors_returned_reason(self):
        tracker = CheckpointConfirmationTracker()
        tracker.observe(10.0)
        _, reason = tracker.evaluate_candidate(5.0, best_score=10.0)
        assert tracker.last_reason == reason == "below_best"

    def test_oscillating_noise_never_builds_a_false_streak(self):
        # Regression guard for the original streak-logic bug: noisy
        # oscillation that never actually clears best must never accumulate
        # into a false "confirmed_by_streak", regardless of how long the run
        # goes on.
        tracker = CheckpointConfirmationTracker(confirm_window=3, min_samples_for_floor=5)
        best = 100.0
        for _ in range(20):
            tracker.observe(90.0)
            accepted, _ = tracker.evaluate_candidate(90.0, best_score=best)
            assert not accepted
            tracker.observe(95.0)
            accepted, _ = tracker.evaluate_candidate(95.0, best_score=best)
            assert not accepted

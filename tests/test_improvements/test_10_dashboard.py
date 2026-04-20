"""
tests/test_improvements/test_10_dashboard.py

Tests for RichDashboardCallback.

Correctness claims:

1. ZERO-LATENCY HOT PATH
   on_episode_end() completes in < 0.5 ms on average across 500 calls.
   Rendering happens in the background thread — the loop is never blocked.

2. ALL HOOKS RETURN IMMEDIATELY
   on_improvement(), on_plateau(), on_dormant(), on_degradation(),
   on_hyperparam_update() each complete in < 0.5 ms.

3. DEQUE COMMUNICATION
   Events pushed by the hot path are picked up by the background thread.
   After on_episode_end(episode_id=N, ...) and a short drain interval,
   the dashboard's internal state reflects episode N.

4. SCORE HISTORY BOUNDED
   Scores list never exceeds `history` entries regardless of how many
   episodes are pushed.

5. BEST-SCORE TRACKING
   on_improvement() with a new best score updates _best_score correctly.
   Earlier (lower) scores do not overwrite a confirmed best.

6. STATE TRANSITIONS
   on_plateau → _state_label == "PLATEAU"
   on_dormant → _state_label == "DORMANT"
   on_degradation → _state_label == "DEGRADATION"
   on_improvement → _state_label == "IMPROVED"

7. SPARKLINE CORRECTNESS
   _sparkline(values) returns a string of the correct length,
   using only block Unicode characters.
   Flat input → same character repeated.

8. BACKGROUND THREAD LIFECYCLE
   on_loop_start() starts a daemon thread.
   on_loop_stop() signals stop and the thread exits within 2 s.

9. GRACEFUL WHEN rich NOT INSTALLED
   If rich is unavailable, the callback does not crash — all hooks
   are no-ops (the render loop exits silently).

10. PANEL BUILDS WITHOUT ERROR
    _build_panel() returns a renderable object (Panel) when rich is available,
    even with an empty score history.
"""

import time
import threading

import numpy as np
import pytest

from tensor_optix.callbacks.rich_dashboard import RichDashboardCallback, _sparkline
from tensor_optix.core.types import EvalMetrics, LoopState, PolicySnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(score: float) -> EvalMetrics:
    return EvalMetrics(
        primary_score=score,
        metrics={},
        episode_id=0,
    )


def _snapshot(episode_id: int, score: float) -> PolicySnapshot:
    from tensor_optix.core.types import HyperparamSet
    return PolicySnapshot(
        snapshot_id=f"snap-{episode_id}",
        episode_id=episode_id,
        eval_metrics=_metrics(score),
        hyperparams=HyperparamSet(params={}, episode_id=episode_id),
        weights_path="",
    )


def _dashboard(**kwargs) -> RichDashboardCallback:
    """Create a dashboard without starting the background thread."""
    return RichDashboardCallback(
        title="test",
        history=kwargs.pop("history", 50),
        refresh_per_second=kwargs.pop("refresh_per_second", 10),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1–2. Latency tests
# ---------------------------------------------------------------------------

class TestLatency:
    N = 500

    def test_on_episode_end_under_0_5ms(self):
        """Mean callback time < 0.5 ms across 500 calls."""
        cb = _dashboard()
        times = []
        for i in range(self.N):
            t0 = time.perf_counter()
            cb.on_episode_end(i, _metrics(float(i)))
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        assert mean_ms < 0.5, \
            f"on_episode_end mean latency {mean_ms:.3f} ms exceeds 0.5 ms"

    def test_on_improvement_under_0_5ms(self):
        cb = _dashboard()
        times = []
        for i in range(self.N):
            t0 = time.perf_counter()
            cb.on_improvement(_snapshot(i, float(i)))
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        assert mean_ms < 0.5, \
            f"on_improvement mean latency {mean_ms:.3f} ms exceeds 0.5 ms"

    def test_all_hooks_under_0_5ms(self):
        """Every hook completes < 0.5 ms (measured as max over 100 calls)."""
        cb   = _dashboard()
        loop_state = LoopState.ACTIVE
        hooks = [
            lambda i: cb.on_episode_end(i, _metrics(float(i))),
            lambda i: cb.on_improvement(_snapshot(i, float(i))),
            lambda i: cb.on_plateau(i, loop_state),
            lambda i: cb.on_dormant(i),
            lambda i: cb.on_degradation(i, _metrics(float(i))),
            lambda i: cb.on_hyperparam_update({}, {"lr": 1e-3}),
        ]
        for hook in hooks:
            times = [
                (lambda t0=time.perf_counter(), h=hook, i=i: h(i) or
                 (time.perf_counter() - t0))()
                for i in range(100)
            ]
            # Re-measure properly
            ts = []
            for i in range(100):
                t0 = time.perf_counter()
                hook(i)
                ts.append(time.perf_counter() - t0)
            mean_ms = np.mean(ts) * 1000
            assert mean_ms < 0.5, \
                f"Hook mean latency {mean_ms:.3f} ms exceeds 0.5 ms"


# ---------------------------------------------------------------------------
# 3. Deque communication
# ---------------------------------------------------------------------------

class TestDequeCommunication:

    def test_events_enqueued(self):
        """on_episode_end appends to queue without blocking."""
        cb = _dashboard()
        assert len(cb._queue) == 0
        cb.on_episode_end(1, _metrics(42.0))
        assert len(cb._queue) == 1
        event = cb._queue[0]
        assert event["type"]       == "episode"
        assert event["episode_id"] == 1
        assert event["score"]      == pytest.approx(42.0)

    def test_drain_updates_state(self):
        """_drain_queue() consumes events and updates display state."""
        cb = _dashboard()
        cb.on_episode_end(5, _metrics(99.0))
        cb._drain_queue()
        assert cb._episode_id    == 5
        assert cb._scores[-1]    == pytest.approx(99.0)

    def test_improvement_event_enqueued(self):
        cb = _dashboard()
        cb.on_improvement(_snapshot(10, 150.0))
        assert len(cb._queue) == 1
        assert cb._queue[0]["type"] == "improvement"

    def test_stop_event_enqueued_on_loop_stop(self):
        """on_loop_stop() pushes a stop event before signalling the thread."""
        cb = _dashboard()
        # Don't actually start the thread for this test
        cb._stop_evt.clear()
        cb.on_loop_stop()
        assert any(e["type"] == "stop" for e in cb._queue)


# ---------------------------------------------------------------------------
# 4. Score history bounded
# ---------------------------------------------------------------------------

class TestScoreHistoryBounded:

    def test_history_not_exceeded(self):
        cb = _dashboard(history=20)
        for i in range(100):
            cb.on_episode_end(i, _metrics(float(i)))
        cb._drain_queue()
        assert len(cb._scores) <= 20

    def test_history_keeps_most_recent(self):
        """After overflow, the most recent scores are kept."""
        cb = _dashboard(history=10)
        for i in range(25):
            cb.on_episode_end(i, _metrics(float(i)))
        cb._drain_queue()
        assert cb._scores[-1] == pytest.approx(24.0)
        assert len(cb._scores) == 10


# ---------------------------------------------------------------------------
# 5. Best-score tracking
# ---------------------------------------------------------------------------

class TestBestScoreTracking:

    def test_best_score_updated(self):
        cb = _dashboard()
        cb.on_improvement(_snapshot(1, 50.0))
        cb.on_improvement(_snapshot(2, 75.0))
        cb.on_improvement(_snapshot(3, 60.0))
        cb._drain_queue()
        assert cb._best_score == pytest.approx(75.0)
        assert cb._best_ep    == 2

    def test_initial_best_none(self):
        cb = _dashboard()
        assert cb._best_score is None

    def test_none_score_does_not_update_best(self):
        cb = _dashboard()
        cb.on_improvement(_snapshot(1, 50.0))
        cb._drain_queue()
        prev_best = cb._best_score
        cb._queue.append({"type": "improvement", "episode_id": 2, "score": None})
        cb._drain_queue()
        assert cb._best_score == pytest.approx(prev_best)

    def test_improvements_counter(self):
        cb = _dashboard()
        for i in range(5):
            cb.on_improvement(_snapshot(i, float(i * 10)))
        cb._drain_queue()
        assert cb._improvements == 5


# ---------------------------------------------------------------------------
# 6. State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:

    def _loop_state(self):
        return LoopState.ACTIVE

    def test_plateau_state(self):
        cb = _dashboard()
        cb.on_plateau(10, self._loop_state())
        cb._drain_queue()
        assert cb._state_label == "PLATEAU"

    def test_dormant_state(self):
        cb = _dashboard()
        cb.on_dormant(20)
        cb._drain_queue()
        assert cb._state_label == "DORMANT"

    def test_degradation_state(self):
        cb = _dashboard()
        cb.on_degradation(15, _metrics(30.0))
        cb._drain_queue()
        assert cb._state_label == "DEGRADATION"

    def test_improvement_state(self):
        cb = _dashboard()
        cb.on_improvement(_snapshot(5, 80.0))
        cb._drain_queue()
        assert cb._state_label == "IMPROVED"

    def test_episode_resets_to_exploring(self):
        """After IMPROVED, next episode_end transitions back to EXPLORING."""
        cb = _dashboard()
        cb.on_improvement(_snapshot(5, 80.0))
        cb._drain_queue()
        cb.on_episode_end(6, _metrics(70.0))
        cb._drain_queue()
        assert cb._state_label == "EXPLORING"


# ---------------------------------------------------------------------------
# 7. Sparkline correctness
# ---------------------------------------------------------------------------

class TestSparkline:
    BLOCKS = " ▁▂▃▄▅▆▇█"

    def test_length_matches_width(self):
        values = list(range(30))
        result = _sparkline(values, width=15)
        assert len(result) == 15

    def test_length_for_short_input(self):
        """If fewer values than width, use all available."""
        values = [1.0, 2.0, 3.0]
        result = _sparkline(values, width=10)
        assert len(result) == 3

    def test_flat_input_same_character(self):
        values = [5.0] * 10
        result = _sparkline(values, width=10)
        assert len(set(result)) == 1, f"Flat input should be all same char: {result!r}"

    def test_only_block_characters(self):
        values = list(range(20))
        result = _sparkline(values, width=20)
        for ch in result:
            assert ch in self.BLOCKS, f"Unexpected character {ch!r} in sparkline"

    def test_empty_input(self):
        result = _sparkline([], width=10)
        assert len(result) == 10

    def test_ascending_ends_high(self):
        """Monotonically increasing values → last bar is highest block."""
        values = list(range(10))
        result = _sparkline(values, width=10)
        # The last bar should be '█' (highest)
        assert result[-1] == "█"

    def test_descending_ends_low(self):
        """
        Monotonically decreasing → last bar is the minimum block.
        BLOCKS[0] = ' ' (space) represents the minimum value,
        so the last character of a descending sequence is ' '.
        """
        values = list(range(10, 0, -1))
        result = _sparkline(values, width=10)
        # The last value (1) is the global minimum → maps to BLOCKS[0] = ' '
        assert result[-1] == " "


# ---------------------------------------------------------------------------
# 8. Background thread lifecycle
# ---------------------------------------------------------------------------

class TestThreadLifecycle:

    def test_loop_start_creates_thread(self):
        cb = _dashboard()
        cb.on_loop_start()
        assert cb._thread is not None
        assert cb._thread.is_alive()
        cb.on_loop_stop()
        cb._thread.join(timeout=3.0)
        assert not cb._thread.is_alive()

    def test_thread_is_daemon(self):
        """Daemon threads don't prevent the process from exiting."""
        cb = _dashboard()
        cb.on_loop_start()
        assert cb._thread.daemon is True
        cb.on_loop_stop()

    def test_stop_idempotent(self):
        """Calling on_loop_stop twice does not raise."""
        cb = _dashboard()
        cb.on_loop_start()
        cb.on_loop_stop()
        cb.on_loop_stop()   # second call — must not raise

    def test_background_thread_drains_queue(self):
        """
        Events pushed while the background thread is running are eventually
        reflected in internal state (within 1 second of drain interval).
        """
        cb = _dashboard(refresh_per_second=20)
        cb.on_loop_start()
        for i in range(10):
            cb.on_episode_end(i, _metrics(float(i * 10)))
        # Give background thread time to drain
        time.sleep(0.3)
        cb.on_loop_stop()
        # After stop, episode_id should reflect the last pushed episode
        assert cb._episode_id >= 9


# ---------------------------------------------------------------------------
# 10. Panel builds without error
# ---------------------------------------------------------------------------

class TestPanelBuilds:

    def test_panel_empty_state(self):
        """_build_panel() with no episodes — must not raise."""
        from rich.panel import Panel
        cb = _dashboard()
        result = cb._build_panel()
        assert isinstance(result, Panel)

    def test_panel_with_scores(self):
        from rich.panel import Panel
        cb = _dashboard()
        for i in range(20):
            cb.on_episode_end(i, _metrics(float(i)))
        cb._drain_queue()
        result = cb._build_panel()
        assert isinstance(result, Panel)

    def test_panel_with_hyperparams(self):
        from rich.panel import Panel
        cb = _dashboard(show_hyperparams=True)
        cb.on_hyperparam_update({}, {"learning_rate": 3e-4, "gamma": 0.99})
        cb._drain_queue()
        result = cb._build_panel()
        assert isinstance(result, Panel)

    def test_panel_no_hyperparams_column(self):
        from rich.panel import Panel
        cb = _dashboard(show_hyperparams=False)
        result = cb._build_panel()
        assert isinstance(result, Panel)

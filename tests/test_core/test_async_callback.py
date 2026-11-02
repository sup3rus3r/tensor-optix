import threading
import time

from tensor_optix.core.async_callback import AsyncCallbackAdapter
from tensor_optix.core.loop_controller import LoopCallback


class RecordingCallback(LoopCallback):
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def on_episode_end(self, episode_id, eval_metrics):
        with self.lock:
            self.events.append(("on_episode_end", episode_id, eval_metrics))

    def on_improvement(self, snapshot):
        with self.lock:
            self.events.append(("on_improvement", snapshot))


class BlockingCallback(LoopCallback):
    """Simulates a slow/blocking consumer (e.g. a GUI update over the network)."""

    def __init__(self, block_seconds: float):
        self.block_seconds = block_seconds
        self.calls = 0
        self.lock = threading.Lock()

    def on_episode_end(self, episode_id, eval_metrics):
        time.sleep(self.block_seconds)
        with self.lock:
            self.calls += 1


class BrokenCallback(LoopCallback):
    def on_episode_end(self, episode_id, eval_metrics):
        raise RuntimeError("boom")


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestAsyncCallbackAdapter:

    def test_events_eventually_delivered_to_inner_callback(self):
        inner = RecordingCallback()
        adapter = AsyncCallbackAdapter(inner)
        try:
            adapter.on_episode_end(1, "metrics_1")
            adapter.on_episode_end(2, "metrics_2")
            assert _wait_until(lambda: len(inner.events) == 2)
            assert inner.events == [
                ("on_episode_end", 1, "metrics_1"),
                ("on_episode_end", 2, "metrics_2"),
            ]
        finally:
            adapter.close(timeout=2.0)

    def test_enqueue_never_blocks_even_with_a_slow_inner_callback(self):
        # The whole point: _enqueue() must return immediately regardless of
        # how slow the wrapped callback is -- this is what protects the
        # training thread from stalling.
        inner = BlockingCallback(block_seconds=0.5)
        adapter = AsyncCallbackAdapter(inner, maxsize=8)
        try:
            start = time.time()
            for i in range(5):
                adapter.on_episode_end(i, None)
            elapsed = time.time() - start
            assert elapsed < 0.1, f"enqueue blocked for {elapsed:.3f}s"
        finally:
            adapter.close(timeout=5.0)

    def test_overflow_drops_oldest_and_counts_drops(self):
        inner = BlockingCallback(block_seconds=1.0)  # never drains in time
        adapter = AsyncCallbackAdapter(inner, maxsize=2)
        try:
            # First event is immediately picked up by the drain thread and
            # blocks it for 1s, so the queue fills up behind it.
            for i in range(10):
                adapter.on_episode_end(i, None)
            assert adapter.dropped_count > 0
        finally:
            adapter.close(timeout=5.0)

    def test_broken_inner_callback_does_not_crash_drain_thread(self):
        inner = BrokenCallback()
        adapter = AsyncCallbackAdapter(inner)
        try:
            adapter.on_episode_end(1, None)
            adapter.on_episode_end(2, None)
            # If the drain thread died on the first exception, this second
            # enqueue + a subsequent close() would hang or the thread would
            # already be dead; close() must still complete cleanly.
            time.sleep(0.2)
            assert adapter._thread.is_alive()
        finally:
            adapter.close(timeout=2.0)

    def test_close_stops_background_thread(self):
        inner = RecordingCallback()
        adapter = AsyncCallbackAdapter(inner)
        adapter.close(timeout=2.0)
        assert not adapter._thread.is_alive()

    def test_close_drains_remaining_queue_before_stopping(self):
        inner = RecordingCallback()
        adapter = AsyncCallbackAdapter(inner, maxsize=64)
        for i in range(20):
            adapter.on_episode_end(i, None)
        adapter.close(timeout=5.0)
        assert len(inner.events) == 20

    def test_all_loop_callback_hooks_are_overridden(self):
        # Every LoopCallback hook must be wrapped -- an un-overridden hook
        # would silently call the base no-op instead of enqueueing, which
        # would make that one event type stay synchronous by accident.
        inner = RecordingCallback()
        adapter = AsyncCallbackAdapter(inner)
        try:
            hooks = [
                "on_loop_start", "on_loop_stop", "on_episode_end",
                "on_improvement", "on_plateau", "on_dormant",
                "on_degradation", "on_hyperparam_update",
            ]
            for hook in hooks:
                assert type(adapter).__dict__.get(hook) is not None, f"{hook} not overridden"
        finally:
            adapter.close(timeout=2.0)

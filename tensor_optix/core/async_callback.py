"""
AsyncCallbackAdapter — a non-blocking wrapper around any LoopCallback.

LoopController fires callbacks synchronously, inline, on the training
thread (see LoopController._fire) -- a slow callback (GUI update, network
I/O, an unbuffered disk write) stalls training for as long as it takes to
return. This adapter enqueues each event onto a bounded queue and lets a
single background thread drain it into the wrapped callback, so the
training thread's call into _fire() always returns immediately regardless
of how slow the wrapped callback is.

This does NOT change LoopController's own contract -- callbacks are still
fired synchronously by design (so a broken callback can't silently corrupt
shared state via unsynchronized concurrent access to the agent/graph). Use
this adapter only around callbacks that are safe to run on a separate
thread and don't need to observe training state synchronously.
"""
import logging
import queue
import threading
from typing import Optional

from .loop_controller import LoopCallback

logger = logging.getLogger(__name__)


class AsyncCallbackAdapter(LoopCallback):
    """
    Wraps `inner` so its hooks run on a background thread via a bounded
    queue, never on the training thread.

    Overflow policy: when the queue is full, the OLDEST pending event is
    dropped to make room for the newest one -- a consumer showing live
    state (e.g. a GUI) cares about keeping up with "now," not replaying
    every historical event. Dropped-event counts are tracked and exposed
    via `dropped_count` so a consumer can detect and report that it's
    falling behind.

    Usage:
        adapter = AsyncCallbackAdapter(MyGuiCallback())
        controller = LoopController(..., callbacks=[adapter])
        ...
        controller.run()
        adapter.close()  # stop the background thread once training is done
    """

    def __init__(self, inner: LoopCallback, maxsize: int = 256):
        self._inner = inner
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._dropped_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def _drain_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event, args = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                getattr(self._inner, event)(*args)
            except Exception as e:
                logger.warning(
                    "AsyncCallbackAdapter: inner callback %s raised during %s: %s",
                    self._inner, event, e,
                )

    def _enqueue(self, event: str, *args) -> None:
        try:
            self._queue.put_nowait((event, args))
            return
        except queue.Full:
            pass
        # Drop the oldest pending event to make room for this one.
        try:
            self._queue.get_nowait()
            with self._lock:
                self._dropped_count += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait((event, args))
        except queue.Full:
            with self._lock:
                self._dropped_count += 1

    def on_loop_start(self) -> None:
        self._enqueue("on_loop_start")

    def on_loop_stop(self) -> None:
        self._enqueue("on_loop_stop")

    def on_episode_end(self, episode_id, eval_metrics) -> None:
        self._enqueue("on_episode_end", episode_id, eval_metrics)

    def on_improvement(self, snapshot) -> None:
        self._enqueue("on_improvement", snapshot)

    def on_plateau(self, episode_id, state) -> None:
        self._enqueue("on_plateau", episode_id, state)

    def on_dormant(self, episode_id) -> None:
        self._enqueue("on_dormant", episode_id)

    def on_degradation(self, episode_id, eval_metrics) -> None:
        self._enqueue("on_degradation", episode_id, eval_metrics)

    def on_hyperparam_update(self, old_params, new_params) -> None:
        self._enqueue("on_hyperparam_update", old_params, new_params)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def close(self, timeout: Optional[float] = None) -> None:
        """
        Stop the background thread after draining whatever is already
        queued. Call once the owning LoopController.run() has returned
        (e.g. from a different, synchronous callback's on_loop_stop, or
        by the caller directly) to avoid leaking the thread indefinitely.
        """
        self._stop.set()
        self._thread.join(timeout=timeout)

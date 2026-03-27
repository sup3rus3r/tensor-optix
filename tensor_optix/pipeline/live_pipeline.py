import time
import threading
import queue
import logging
import numpy as np
from typing import Any, Generator, Optional
from tensor_optix.core.base_pipeline import BasePipeline, EpisodeBoundaryFn
from tensor_optix.core.types import EpisodeData

logger = logging.getLogger(__name__)

_SENTINEL = object()


class LivePipeline(BasePipeline):
    """
    Streams data from a real-time source.

    Use case: live trading, real-world robotics, online environments.

    The user provides a data_source with a stream() generator that yields
    (observation, reward, terminated, truncated, info) tuples per step.

    The data source runs in a background thread with a bounded queue.
    The main loop consumes from the queue safely.

    On disconnect (StopIteration or exception from stream()):
    - If reconnect_on_disconnect=True: calls data_source.stream() again
    - Otherwise: signals end of pipeline

    Episode boundary is determined by episode_boundary_fn.
    Preset factories:
    - LivePipeline.every_n_steps(n)
    - LivePipeline.every_n_seconds(n)
    - LivePipeline.on_done_signal()   (default)
    """

    def __init__(
        self,
        data_source: Any,
        agent=None,
        episode_boundary_fn: Optional[EpisodeBoundaryFn] = None,
        buffer_size: int = 1000,
        reconnect_on_disconnect: bool = True,
    ):
        self._data_source = data_source
        self._agent = agent
        self._episode_boundary_fn = episode_boundary_fn or LivePipeline.on_done_signal()
        self._buffer_size = buffer_size
        self._reconnect = reconnect_on_disconnect
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._episode_counter = 0

    def set_agent(self, agent) -> None:
        self._agent = agent

    def setup(self) -> None:
        self._stop_event.clear()
        self._start_producer()

    def _start_producer(self) -> None:
        self._producer_thread = threading.Thread(
            target=self._produce, daemon=True, name="LivePipeline-producer"
        )
        self._producer_thread.start()

    def _produce(self) -> None:
        while not self._stop_event.is_set():
            try:
                for item in self._data_source.stream():
                    if self._stop_event.is_set():
                        break
                    self._queue.put(item, timeout=5.0)
                if not self._reconnect:
                    self._queue.put(_SENTINEL)
                    return
                logger.info("LivePipeline: stream exhausted, reconnecting")
            except Exception as e:
                logger.warning("LivePipeline producer error: %s", e)
                if not self._reconnect:
                    self._queue.put(_SENTINEL)
                    return
                time.sleep(1.0)

    def episodes(self) -> Generator[EpisodeData, None, None]:
        while True:
            observations = []
            actions = []
            rewards = []
            terminated_flags = []
            truncated_flags = []
            infos = []

            step_count = 0
            episode_start = time.time()

            while True:
                try:
                    item = self._queue.get(timeout=30.0)
                except queue.Empty:
                    logger.warning("LivePipeline: queue empty for 30s, ending episode")
                    break

                if item is _SENTINEL:
                    return

                obs, reward, terminated, truncated, info = item
                action = self._agent.act(obs)

                observations.append(obs)
                actions.append(action)
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                infos.append(info)
                step_count += 1
                elapsed = time.time() - episode_start

                natural_end = terminated or truncated
                boundary_end = self._episode_boundary_fn(step_count, elapsed, obs)

                if natural_end or boundary_end:
                    break

            if len(rewards) == 0:
                continue

            yield EpisodeData(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=rewards,
                terminated=terminated_flags,
                truncated=truncated_flags,
                infos=infos,
                episode_id=self._episode_counter,
            )
            self._episode_counter += 1

    def teardown(self) -> None:
        self._stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5.0)

    @property
    def is_live(self) -> bool:
        return True

    @staticmethod
    def every_n_steps(n: int) -> EpisodeBoundaryFn:
        return lambda step_count, elapsed, obs: step_count >= n

    @staticmethod
    def every_n_seconds(n: float) -> EpisodeBoundaryFn:
        return lambda step_count, elapsed, obs: elapsed >= n

    @staticmethod
    def on_done_signal() -> EpisodeBoundaryFn:
        # Relies on terminated/truncated flags from the data source
        return lambda step_count, elapsed, obs: False

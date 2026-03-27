from abc import ABC, abstractmethod
from typing import Callable, Generator
from .types import EpisodeData

# Signature: (step_count, elapsed_seconds, last_observation) -> bool (True = end episode)
EpisodeBoundaryFn = Callable[[int, float, any], bool]


class BasePipeline(ABC):
    """
    Data source abstraction. The loop controller uses this to get episode data.

    Two concrete implementations:
    - BatchPipeline: Gymnasium env, episodic/batch training
    - LivePipeline: real-time streaming source

    Both produce EpisodeData. The loop controller cannot tell the difference.
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize the data source. Called once before the loop starts."""

    @abstractmethod
    def episodes(self) -> Generator[EpisodeData, None, None]:
        """
        Infinite generator of episodes. Each yield produces one complete EpisodeData.

        BatchPipeline: runs the Gymnasium env until terminated|truncated, cycles forever.
        LivePipeline: streams real-time data, uses boundary_fn to end episodes.

        The generator MUST be infinite — the loop runs forever.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources. Called on loop stop or exception."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for LivePipeline, False for BatchPipeline."""

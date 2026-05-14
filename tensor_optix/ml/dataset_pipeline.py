from __future__ import annotations

"""
DatasetPipeline — adapts a PyTorch Dataset or DataLoader to the BasePipeline
contract so the tensor-optix loop controller can drive ML training exactly as
it drives RL training.

Each "episode" is one mini-batch. The pipeline loops through the dataset
indefinitely — the loop controller decides when to stop (max_episodes,
convergence, KeyboardInterrupt), never the pipeline.

Mapping to EpisodeData
----------------------
observations  → input tensor X (shape [batch, ...])
actions       → target tensor y  — OR X again for unsupervised tasks
                (reconstruction, VAE, contrastive) so MLAgent.learn()
                receives the original input as both input and target.
rewards       → [-loss] placeholder filled by MLAgent.learn(); loop
                controller sees higher-is-better signal automatically.
terminated    → [True] — treated as one "episode" per batch.
truncated     → [False]
"""

import logging
from typing import Generator, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.types import EpisodeData

logger = logging.getLogger(__name__)

# Loss keys that need target = input (unsupervised)
_SELF_SUPERVISED_LOSSES = {"reconstruction", "vae", "contrastive"}


class DatasetPipeline(BasePipeline):
    """
    Wraps a PyTorch Dataset or DataLoader as a tensor-optix pipeline.

    Parameters
    ----------
    dataset:
        A torch.utils.data.Dataset or DataLoader. When a Dataset is passed,
        a DataLoader is built automatically using batch_size and shuffle.
    agent:
        Set by the loop controller via set_agent(). Not used directly by the
        pipeline — stored for interface compatibility.
    batch_size:
        Batch size when building a DataLoader from a Dataset. Ignored when
        a DataLoader is passed directly.
    shuffle:
        Whether to shuffle each epoch. Default True.
    loss_key:
        The loss string used by MLAgent. When it is an unsupervised key
        ("reconstruction", "vae", "contrastive"), the pipeline sets
        actions = observations so MLAgent.learn() receives input as target.
    num_workers:
        DataLoader worker processes. Default 0 (main process).
    pin_memory:
        DataLoader pin_memory flag. Default False.
    """

    is_live = False

    def __init__(
        self,
        dataset: Union[Dataset, DataLoader],
        agent=None,
        batch_size: int = 64,
        shuffle: bool = True,
        loss_key: str = "mse",
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self._dataset = dataset
        self._agent = agent
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._loss_key = loss_key.lower() if isinstance(loss_key, str) else "mse"
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._loader: Optional[DataLoader] = None
        self._episode_id = 0

    def set_agent(self, agent) -> None:
        self._agent = agent

    # ------------------------------------------------------------------
    # BasePipeline contract
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if isinstance(self._dataset, DataLoader):
            self._loader = self._dataset
            logger.info(
                "DatasetPipeline: using provided DataLoader (batch_size=%s)",
                getattr(self._loader, "batch_size", "?"),
            )
        else:
            self._loader = DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                drop_last=True,
            )
            logger.info(
                "DatasetPipeline: built DataLoader from Dataset "
                "(n=%d, batch_size=%d, shuffle=%s)",
                len(self._dataset), self._batch_size, self._shuffle,
            )

    def episodes(self) -> Generator[EpisodeData, None, None]:
        """Infinite generator — loops through the dataset epoch by epoch."""
        if self._loader is None:
            raise RuntimeError("DatasetPipeline.setup() must be called before episodes().")

        epoch = 0
        while True:
            epoch += 1
            for batch in self._loader:
                obs, actions = self._unpack_batch(batch)
                self._episode_id += 1
                yield EpisodeData(
                    observations=obs,
                    actions=actions,
                    rewards=[0.0],         # filled by MLAgent.learn() diagnostics
                    terminated=[True],
                    truncated=[False],
                    infos=[{"epoch": epoch}],
                    episode_id=self._episode_id,
                )

    def teardown(self) -> None:
        self._loader = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """
        Extract (X, y) from a batch.

        Handles:
          (X, y)  tuples  — supervised
          (X,)    tuples  — unsupervised single-tensor datasets
          X       tensor  — bare tensor datasets
          (X1,X2) pairs   — contrastive (both views stacked as obs)
        """
        unsupervised = self._loss_key in _SELF_SUPERVISED_LOSSES

        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                X = self._to_numpy(batch[0])
                return X, X  # unsupervised: target = input

            X = self._to_numpy(batch[0])
            y_raw = batch[1]

            if unsupervised:
                # For reconstruction/VAE: ignore label, target = input
                return X, X

            # Contrastive: both views already packed; stack as obs, y = X
            if self._loss_key == "contrastive":
                return X, X

            y = self._to_numpy(y_raw)
            return X, y

        # Bare tensor
        X = self._to_numpy(batch)
        return X, X

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.numpy() if not t.requires_grad else t.detach().numpy()
        return np.asarray(t)

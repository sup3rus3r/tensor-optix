"""Typed records for the experiment ledger.

These dataclasses are the in-memory mirror of ledger database rows. They
exist so callers deal with typed, versioned objects rather than raw dicts,
per the Learning-to-Learn proposal's guidance (Sec 13).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class ExperimentState(Enum):
    """Execution lifecycle state of an experiment."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class EvaluationStatus(Enum):
    """Scientific evaluation state of an experiment, separate from execution state."""

    PENDING = "pending"
    PRELIMINARY = "preliminary"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


SCHEMA_VERSION = 2


@dataclass
class ExperimentRecord:
    """In-memory mirror of one `experiments` table row.

    Mutable fields (state, evaluation_status, progress/outcome columns) are
    updated in place by ExperimentStore methods and re-persisted on each
    call — this is not a write-once value object.
    """

    experiment_id: str
    task_env: str
    algorithm: str
    framework: str
    seed: int
    checkpoint_dir: str
    solve_threshold: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    parent_experiment_id: Optional[str] = None
    study_id: Optional[str] = None
    tensor_optix_version: Optional[str] = None
    source_revision: Optional[str] = None
    library_versions: Dict[str, str] = field(default_factory=dict)
    hostname: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    state: ExperimentState = ExperimentState.CREATED
    evaluation_status: EvaluationStatus = EvaluationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    best_primary_score: Optional[float] = None
    best_train_score: Optional[float] = None
    best_val_score: Optional[float] = None
    generalization_gap: Optional[float] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class EventRecord:
    """One row in the `events` table — an immutable, ordered fact about an experiment."""

    event_id: str
    experiment_id: str
    sequence: int
    event_type: str
    source: str = "loop_controller"
    step: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION


@dataclass
class ArtifactRef:
    """A reference to a checkpoint artifact already written by CheckpointRegistry.

    The ledger never duplicates weight bytes — it stores paths and an
    optional checksum pointing at files CheckpointRegistry owns.
    """

    experiment_id: str
    snapshot_id: str
    checkpoint_dir: str
    snapshot_dir: str
    weights_path: str
    primary_score: float
    episode_id: int
    checksum: Optional[str] = None

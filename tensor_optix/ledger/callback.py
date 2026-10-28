"""ExperimentLedgerCallback — records a LoopController run into an ExperimentStore.

Follows the exact lazy-init / defensive pattern established by
TensorBoardCallback and WandbCallback: no I/O in __init__ beyond storing
configuration, real setup happens in on_loop_start(), and on_loop_stop()
tears down / finalizes idempotently.
"""

import hashlib
import os
import uuid
from typing import Any, Dict, Optional

from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics, PolicySnapshot

from .records import ArtifactRef, EventRecord, ExperimentState, EvaluationStatus
from .store import ExperimentStore


def _checksum_weights(weights_path: str) -> Optional[str]:
    """Best-effort sha256 over every file under weights_path, sorted for determinism.

    Weights layout is framework-dependent (a single file, a directory of
    shards, etc.) so this never assumes structure and never raises.
    """
    try:
        if os.path.isfile(weights_path):
            files = [weights_path]
        elif os.path.isdir(weights_path):
            files = sorted(
                os.path.join(root, name)
                for root, _dirs, names in os.walk(weights_path)
                for name in names
            )
        else:
            return None

        digest = hashlib.sha256()
        for file_path in files:
            with open(file_path, "rb") as f:
                digest.update(f.read())
        return digest.hexdigest()
    except Exception:
        return None


class ExperimentLedgerCallback(LoopCallback):
    def __init__(
        self,
        store: ExperimentStore,
        *,
        task_env: str,
        algorithm: str,
        framework: str = "torch",
        seed: int = 0,
        checkpoint_dir: str,
        config: Optional[Dict[str, Any]] = None,
        parent_experiment_id: Optional[str] = None,
        study_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        checkpoint_registry: Optional[Any] = None,
    ):
        self._store = store
        self._task_env = task_env
        self._algorithm = algorithm
        self._framework = framework
        self._seed = seed
        self._checkpoint_dir = checkpoint_dir
        self._config = config
        self._parent_experiment_id = parent_experiment_id
        self._study_id = study_id
        self._checkpoint_registry = checkpoint_registry

        self._experiment_id = experiment_id or str(uuid.uuid4())[:8]
        self._sequence = 0
        self._last_metrics: Dict[str, float] = {}
        self._finalized = False

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _append(self, event_type: str, *, step: Optional[int] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        self._store.append_event(
            self._experiment_id,
            EventRecord(
                event_id=str(uuid.uuid4()),
                experiment_id=self._experiment_id,
                sequence=self._next_sequence(),
                event_type=event_type,
                step=step,
                payload=payload or {},
            ),
        )

    # ------------------------------------------------------------------
    # LoopCallback hooks
    # ------------------------------------------------------------------

    def on_loop_start(self) -> None:
        self._store.create_experiment(
            task_env=self._task_env,
            algorithm=self._algorithm,
            framework=self._framework,
            seed=self._seed,
            checkpoint_dir=self._checkpoint_dir,
            config=self._config,
            parent_experiment_id=self._parent_experiment_id,
            study_id=self._study_id,
            experiment_id=self._experiment_id,
        )
        self._store.start_experiment(self._experiment_id)
        self._append("loop_start")

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is None:
            return
        self._last_metrics = dict(eval_metrics.metrics)
        self._append(
            "episode_end",
            step=episode_id,
            payload={"primary_score": eval_metrics.primary_score, "metrics": eval_metrics.metrics},
        )

    def on_improvement(self, snapshot: PolicySnapshot) -> None:
        metrics = snapshot.eval_metrics.metrics
        self._append(
            "improvement",
            step=snapshot.episode_id,
            payload={
                "snapshot_id": snapshot.snapshot_id,
                "primary_score": snapshot.eval_metrics.primary_score,
                "weights_path": snapshot.weights_path,
            },
        )
        self._store.record_progress(
            self._experiment_id,
            best_primary_score=snapshot.eval_metrics.primary_score,
            best_train_score=metrics.get("train_score"),
            best_val_score=metrics.get("val_score"),
            generalization_gap=snapshot.eval_metrics.generalization_gap,
        )

        snapshot_dir = os.path.dirname(snapshot.weights_path.rstrip(os.sep)) or snapshot.weights_path
        self._store.record_artifact(
            ArtifactRef(
                experiment_id=self._experiment_id,
                snapshot_id=snapshot.snapshot_id,
                checkpoint_dir=self._checkpoint_dir,
                snapshot_dir=snapshot_dir,
                weights_path=snapshot.weights_path,
                primary_score=snapshot.eval_metrics.primary_score,
                episode_id=snapshot.episode_id,
                checksum=_checksum_weights(snapshot.weights_path),
            )
        )

    def on_plateau(self, episode_id: int, state) -> None:
        self._append("plateau", step=episode_id, payload={"loop_state": state.name})

    def on_dormant(self, episode_id: int) -> None:
        self._append("dormant", step=episode_id)

    def on_degradation(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        payload = {}
        if eval_metrics is not None:
            payload["primary_score"] = eval_metrics.primary_score
        self._append("degradation", step=episode_id, payload=payload)

    def on_hyperparam_update(self, old_params: dict, new_params: dict) -> None:
        self._append("hyperparam_update", payload={"old": old_params, "new": new_params})

    def on_loop_stop(self) -> None:
        self._append("loop_stop")
        if self._finalized:
            return
        self._finalized = True
        self._store.finalize_experiment(
            self._experiment_id,
            state=ExperimentState.COMPLETED,
            evaluation_status=EvaluationStatus.PRELIMINARY,
            final_metrics=self._last_metrics,
        )

    def mark_failed(self, exc: Exception) -> None:
        """Not a LoopCallback hook — call this from the caller's own except block.

        LoopController._fire() swallows exceptions raised inside callbacks,
        and there is no on_error hook today, so a callback cannot observe an
        exception raised by the training loop itself. Callers that run
        opt.run() in a try/except must call this explicitly to record the
        FAILED state.
        """
        if self._finalized:
            return
        self._finalized = True
        self._store.finalize_experiment(
            self._experiment_id,
            state=ExperimentState.FAILED,
            evaluation_status=EvaluationStatus.INCONCLUSIVE,
            error_message=str(exc),
        )

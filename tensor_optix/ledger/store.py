"""ExperimentStore — SQLite-backed experiment ledger.

Every public method opens a short, single-statement transaction via
`with self._conn:` (sqlite3's context-manager commit/rollback). No method
spans more than one logical operation, and nothing here ever holds a
transaction open for the duration of model training — the caller (a
LoopCallback or a hand-rolled training loop) calls these methods between
episodes, not around them.
"""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from .provenance import capture_environment
from .records import (
    ArtifactRef, EventRecord, ExperimentRecord, ExperimentState, EvaluationStatus,
    SCHEMA_VERSION, StrategyCandidateRecord, StrategyEvaluationRecord,
    StrategyRegistryRecord, StrategyStatus,
)
from .schema import ensure_schema


def _row_to_experiment(row: sqlite3.Row) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=row["experiment_id"],
        parent_experiment_id=row["parent_experiment_id"],
        study_id=row["study_id"],
        task_env=row["task_env"],
        algorithm=row["algorithm"],
        framework=row["framework"],
        seed=row["seed"],
        checkpoint_dir=row["checkpoint_dir"],
        solve_threshold=row["solve_threshold"],
        config=json.loads(row["config_json"]),
        tensor_optix_version=row["tensor_optix_version"],
        source_revision=row["source_revision"],
        library_versions=json.loads(row["library_versions_json"]),
        hostname=row["hostname"],
        schema_version=row["schema_version"],
        state=ExperimentState(row["state"]),
        evaluation_status=EvaluationStatus(row["evaluation_status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        best_primary_score=row["best_primary_score"],
        best_train_score=row["best_train_score"],
        best_val_score=row["best_val_score"],
        generalization_gap=row["generalization_gap"],
        final_metrics=json.loads(row["final_metrics_json"]),
        error_message=row["error_message"],
    )


def _row_to_event(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        event_id=row["event_id"],
        experiment_id=row["experiment_id"],
        sequence=row["sequence"],
        event_type=row["event_type"],
        source=row["source"],
        step=row["step"],
        payload=json.loads(row["payload_json"]),
        timestamp=row["timestamp"],
        schema_version=row["schema_version"],
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRef:
    return ArtifactRef(
        experiment_id=row["experiment_id"],
        snapshot_id=row["snapshot_id"],
        checkpoint_dir=row["checkpoint_dir"],
        snapshot_dir=row["snapshot_dir"],
        weights_path=row["weights_path"],
        primary_score=row["primary_score"],
        episode_id=row["episode_id"],
        checksum=row["checksum"],
    )


def _row_to_strategy_candidate(row: sqlite3.Row) -> StrategyCandidateRecord:
    return StrategyCandidateRecord(
        candidate_id=row["candidate_id"],
        op_name=row["op_name"],
        op_version=row["op_version"],
        grammar_schema_version=row["grammar_schema_version"],
        params=json.loads(row["params_json"]),
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


def _row_to_strategy_evaluation(row: sqlite3.Row) -> StrategyEvaluationRecord:
    return StrategyEvaluationRecord(
        evaluation_id=row["evaluation_id"],
        candidate_id=row["candidate_id"],
        experiment_id=row["experiment_id"],
        baseline_experiment_id=row["baseline_experiment_id"],
        task_env=row["task_env"],
        algorithm=row["algorithm"],
        framework=row["framework"],
        seed=row["seed"],
        outcome_delta=row["outcome_delta"],
        created_at=row["created_at"],
    )


def _row_to_strategy_registry(row: sqlite3.Row) -> StrategyRegistryRecord:
    return StrategyRegistryRecord(
        candidate_id=row["candidate_id"],
        status=StrategyStatus(row["status"]),
        approved_at=row["approved_at"],
        demoted_at=row["demoted_at"],
        demotion_reason=row["demotion_reason"],
        scope=json.loads(row["scope_json"]),
        reviewer=row["reviewer"],
    )


class ExperimentStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        ensure_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Experiments
    # ------------------------------------------------------------------ #

    def create_experiment(
        self,
        *,
        task_env: str,
        algorithm: str,
        framework: str,
        seed: int,
        checkpoint_dir: str,
        solve_threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        parent_experiment_id: Optional[str] = None,
        study_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """Insert a new experiment row in CREATED state.

        Idempotent: calling twice with the same explicit experiment_id
        returns the existing row rather than raising or duplicating it.
        """
        existing = self.get_experiment(experiment_id) if experiment_id else None
        if existing is not None:
            return existing

        eid = experiment_id or str(uuid.uuid4())[:8]
        provenance = capture_environment()
        now = time.time()

        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO experiments (
                    experiment_id, parent_experiment_id, study_id, task_env,
                    algorithm, framework, seed, checkpoint_dir, solve_threshold,
                    config_json, tensor_optix_version, source_revision,
                    library_versions_json, hostname, schema_version, state,
                    evaluation_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eid,
                    parent_experiment_id,
                    study_id,
                    task_env,
                    algorithm,
                    framework,
                    seed,
                    checkpoint_dir,
                    solve_threshold,
                    json.dumps(config or {}),
                    provenance["tensor_optix_version"],
                    provenance["source_revision"],
                    json.dumps(provenance["library_versions"]),
                    provenance["hostname"],
                    SCHEMA_VERSION,
                    ExperimentState.CREATED.value,
                    EvaluationStatus.PENDING.value,
                    now,
                ),
            )

        record = self.get_experiment(eid)
        assert record is not None
        return record

    def start_experiment(self, experiment_id: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE experiments SET state = ?, started_at = ?
                WHERE experiment_id = ? AND state = ?
                """,
                (
                    ExperimentState.RUNNING.value,
                    time.time(),
                    experiment_id,
                    ExperimentState.CREATED.value,
                ),
            )

    def record_progress(
        self,
        experiment_id: str,
        *,
        best_primary_score: Optional[float] = None,
        best_train_score: Optional[float] = None,
        best_val_score: Optional[float] = None,
        generalization_gap: Optional[float] = None,
    ) -> None:
        sets, params = [], []
        if best_primary_score is not None:
            sets.append("best_primary_score = ?")
            params.append(best_primary_score)
        if best_train_score is not None:
            sets.append("best_train_score = ?")
            params.append(best_train_score)
        if best_val_score is not None:
            sets.append("best_val_score = ?")
            params.append(best_val_score)
        if generalization_gap is not None:
            sets.append("generalization_gap = ?")
            params.append(generalization_gap)
        if not sets:
            return
        params.append(experiment_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE experiments SET {', '.join(sets)} WHERE experiment_id = ?",
                params,
            )

    def finalize_experiment(
        self,
        experiment_id: str,
        *,
        state: ExperimentState,
        evaluation_status: EvaluationStatus = EvaluationStatus.PRELIMINARY,
        final_metrics: Optional[Dict[str, float]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark an experiment as terminal. Idempotent — a second call is a no-op.

        The WHERE finished_at IS NULL guard prevents a later call (e.g. a
        stray on_loop_stop firing after mark_failed already finalized the
        row) from overwriting an already-recorded terminal outcome.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE experiments
                SET state = ?, evaluation_status = ?, finished_at = ?,
                    final_metrics_json = ?, error_message = ?
                WHERE experiment_id = ? AND finished_at IS NULL
                """,
                (
                    state.value,
                    evaluation_status.value,
                    time.time(),
                    json.dumps(final_metrics or {}),
                    error_message,
                    experiment_id,
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return _row_to_experiment(row) if row is not None else None

    def query_experiments(
        self,
        *,
        task_env: Optional[str] = None,
        algorithm: Optional[str] = None,
        framework: Optional[str] = None,
        state: Optional[ExperimentState] = None,
        evaluation_status: Optional[EvaluationStatus] = None,
        study_id: Optional[str] = None,
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
        min_best_primary_score: Optional[float] = None,
        limit: int = 100,
    ) -> List[ExperimentRecord]:
        clauses: List[str] = []
        params: List[Any] = []

        for column, value in (
            ("task_env", task_env),
            ("algorithm", algorithm),
            ("framework", framework),
            ("study_id", study_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)

        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if evaluation_status is not None:
            clauses.append("evaluation_status = ?")
            params.append(evaluation_status.value)
        if created_after is not None:
            clauses.append("created_at >= ?")
            params.append(created_after)
        if created_before is not None:
            clauses.append("created_at <= ?")
            params.append(created_before)
        if min_best_primary_score is not None:
            clauses.append("best_primary_score >= ?")
            params.append(min_best_primary_score)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_experiment(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def append_event(self, experiment_id: str, event: EventRecord) -> bool:
        """Insert an event idempotently. Returns True iff a new row was inserted.

        A duplicate event_id or a duplicate (experiment_id, sequence) pair
        is silently ignored (INSERT OR IGNORE against the PRIMARY KEY and
        the UNIQUE constraint respectively).
        """
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_id, experiment_id, sequence, event_type, source,
                    step, payload_json, timestamp, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    experiment_id,
                    event.sequence,
                    event.event_type,
                    event.source,
                    event.step,
                    json.dumps(event.payload),
                    event.timestamp,
                    event.schema_version,
                ),
            )
            return cursor.rowcount == 1

    def get_events(self, experiment_id: str) -> List[EventRecord]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE experiment_id = ? ORDER BY sequence ASC",
            (experiment_id,),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #

    def record_artifact(self, artifact: ArtifactRef) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    experiment_id, snapshot_id, checkpoint_dir, snapshot_dir,
                    weights_path, primary_score, episode_id, checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.experiment_id,
                    artifact.snapshot_id,
                    artifact.checkpoint_dir,
                    artifact.snapshot_dir,
                    artifact.weights_path,
                    artifact.primary_score,
                    artifact.episode_id,
                    artifact.checksum,
                    time.time(),
                ),
            )

    def get_artifacts(self, experiment_id: str) -> List[ArtifactRef]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE experiment_id = ? ORDER BY primary_score DESC",
            (experiment_id,),
        ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Strategy candidates / evaluations / registry (Phase 5)
    # ------------------------------------------------------------------ #

    def create_strategy_candidate(
        self,
        *,
        candidate_id: str,
        op_name: str,
        op_version: int,
        grammar_schema_version: int,
        params: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> StrategyCandidateRecord:
        """Insert a new strategy candidate row. Idempotent on candidate_id
        (content-addressed -- see tensor_optix/strategy/candidate.py)."""
        existing = self.get_strategy_candidate(candidate_id)
        if existing is not None:
            return existing

        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO strategy_candidates (
                    candidate_id, op_name, op_version, grammar_schema_version,
                    params_json, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id, op_name, op_version, grammar_schema_version,
                    json.dumps(params or {}), time.time(), created_by,
                ),
            )

        record = self.get_strategy_candidate(candidate_id)
        assert record is not None
        return record

    def get_strategy_candidate(self, candidate_id: str) -> Optional[StrategyCandidateRecord]:
        row = self._conn.execute(
            "SELECT * FROM strategy_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return _row_to_strategy_candidate(row) if row is not None else None

    def record_strategy_evaluation(
        self,
        *,
        candidate_id: str,
        experiment_id: str,
        task_env: str,
        algorithm: str,
        framework: str,
        seed: int,
        baseline_experiment_id: Optional[str] = None,
        outcome_delta: Optional[float] = None,
        evaluation_id: Optional[str] = None,
    ) -> StrategyEvaluationRecord:
        eid = evaluation_id or str(uuid.uuid4())[:8]
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO strategy_evaluations (
                    evaluation_id, candidate_id, experiment_id, baseline_experiment_id,
                    task_env, algorithm, framework, seed, outcome_delta, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eid, candidate_id, experiment_id, baseline_experiment_id,
                    task_env, algorithm, framework, seed, outcome_delta, time.time(),
                ),
            )
        row = self._conn.execute(
            "SELECT * FROM strategy_evaluations WHERE evaluation_id = ?", (eid,)
        ).fetchone()
        assert row is not None
        return _row_to_strategy_evaluation(row)

    def query_strategy_evaluations(
        self,
        *,
        candidate_id: Optional[str] = None,
        task_env: Optional[str] = None,
        limit: int = 1000,
    ) -> List[StrategyEvaluationRecord]:
        clauses: List[str] = []
        params: List[Any] = []
        if candidate_id is not None:
            clauses.append("candidate_id = ?")
            params.append(candidate_id)
        if task_env is not None:
            clauses.append("task_env = ?")
            params.append(task_env)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM strategy_evaluations {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_strategy_evaluation(row) for row in rows]

    def get_strategy_registry_entry(self, candidate_id: str) -> Optional[StrategyRegistryRecord]:
        row = self._conn.execute(
            "SELECT * FROM strategy_registry WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return _row_to_strategy_registry(row) if row is not None else None

    def approve_strategy(
        self, candidate_id: str, *, reviewer: str, scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Human-only promotion write path. PromotionPolicy.evaluate() never
        calls this itself -- a human calls it after inspecting the
        automatable criteria (margin, repeat, generalization, compute,
        reproducibility)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO strategy_registry (candidate_id, status, approved_at, scope_json, reviewer)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    status = excluded.status, approved_at = excluded.approved_at,
                    scope_json = excluded.scope_json, reviewer = excluded.reviewer,
                    demoted_at = NULL, demotion_reason = NULL
                """,
                (candidate_id, StrategyStatus.APPROVED.value, time.time(),
                 json.dumps(scope or {}), reviewer),
            )

    def demote_strategy(self, candidate_id: str, reason: str) -> None:
        """Rollback mechanism: any consumer reading the registry must
        filter status='approved' live on every read, never cache -- a
        demotion takes effect immediately for all future callers."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO strategy_registry (candidate_id, status, demoted_at, demotion_reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    status = excluded.status, demoted_at = excluded.demoted_at,
                    demotion_reason = excluded.demotion_reason
                """,
                (candidate_id, StrategyStatus.DEMOTED.value, time.time(), reason),
            )

    def query_approved_strategies(self, *, limit: int = 100) -> List[StrategyRegistryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM strategy_registry WHERE status = ? ORDER BY approved_at DESC LIMIT ?",
            (StrategyStatus.APPROVED.value, limit),
        ).fetchall()
        return [_row_to_strategy_registry(row) for row in rows]

    def record_strategy_scope(self, candidate_id: str, scope: Dict[str, Any]) -> None:
        """Stage a candidate's validated scope ahead of any approval
        decision -- creates a 'proposed' registry row if none exists
        yet, or updates scope_json in place without disturbing an
        existing approved/demoted status."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO strategy_registry (candidate_id, status, scope_json)
                VALUES (?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET scope_json = excluded.scope_json
                """,
                (candidate_id, StrategyStatus.PROPOSED.value, json.dumps(scope)),
            )

    def get_strategy_scope(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Raw scope dict for a candidate, as recorded in
        strategy_registry.scope_json -- callers reconstruct a
        StrategyScope via StrategyScope.from_dict(...) if a typed view
        is needed. Returns None if no registry row exists yet."""
        entry = self.get_strategy_registry_entry(candidate_id)
        return entry.scope if entry is not None else None

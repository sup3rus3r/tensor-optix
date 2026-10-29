"""SQLite schema definition and migration guard for the experiment ledger."""

import sqlite3

from .records import SCHEMA_VERSION

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id          TEXT PRIMARY KEY,
    parent_experiment_id   TEXT,
    study_id               TEXT,
    task_env               TEXT NOT NULL,
    algorithm               TEXT NOT NULL,
    framework               TEXT NOT NULL,
    seed                    INTEGER NOT NULL,
    checkpoint_dir          TEXT NOT NULL,
    solve_threshold         REAL,
    config_json             TEXT NOT NULL DEFAULT '{}',
    tensor_optix_version    TEXT,
    source_revision         TEXT,
    library_versions_json   TEXT NOT NULL DEFAULT '{}',
    hostname                TEXT,
    schema_version          INTEGER NOT NULL DEFAULT 2,
    state                   TEXT NOT NULL DEFAULT 'created',
    evaluation_status       TEXT NOT NULL DEFAULT 'pending',
    created_at              REAL NOT NULL,
    started_at              REAL,
    finished_at             REAL,
    best_primary_score      REAL,
    best_train_score        REAL,
    best_val_score          REAL,
    generalization_gap      REAL,
    final_metrics_json      TEXT NOT NULL DEFAULT '{}',
    error_message           TEXT,
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_task_env    ON experiments(task_env);
CREATE INDEX IF NOT EXISTS idx_experiments_algorithm   ON experiments(algorithm);
CREATE INDEX IF NOT EXISTS idx_experiments_state       ON experiments(state);
CREATE INDEX IF NOT EXISTS idx_experiments_eval_status ON experiments(evaluation_status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at  ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_study_id    ON experiments(study_id);

CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT NOT NULL,
    experiment_id  TEXT NOT NULL,
    sequence       INTEGER NOT NULL,
    event_type     TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'loop_controller',
    step           INTEGER,
    payload_json   TEXT NOT NULL DEFAULT '{}',
    timestamp      REAL NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    PRIMARY KEY (event_id),
    UNIQUE (experiment_id, sequence),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_events_experiment_id ON events(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type    ON events(event_type);

CREATE TABLE IF NOT EXISTS artifacts (
    experiment_id  TEXT NOT NULL,
    snapshot_id    TEXT NOT NULL,
    checkpoint_dir TEXT NOT NULL,
    snapshot_dir   TEXT NOT NULL,
    weights_path   TEXT NOT NULL,
    primary_score  REAL NOT NULL,
    episode_id     INTEGER NOT NULL,
    checksum       TEXT,
    created_at     REAL NOT NULL,
    PRIMARY KEY (experiment_id, snapshot_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_experiment_id ON artifacts(experiment_id);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create ledger tables/indexes if absent, and guard against a newer schema.

    Raises RuntimeError if the database already reports a schema_version
    higher than this code knows about (fail loud rather than silently
    operate on an incompatible schema).
    """
    conn.executescript(DDL)

    cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        return

    existing_version = int(row[0])
    if existing_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Ledger database schema_version={existing_version} is newer than this "
            f"code's SCHEMA_VERSION={SCHEMA_VERSION}. Refusing to open it to avoid "
            f"silently corrupting data written by a newer version of tensor_optix."
        )

    if existing_version < SCHEMA_VERSION:
        _migrate(conn, from_version=existing_version, to_version=SCHEMA_VERSION)


def _migrate(conn: sqlite3.Connection, *, from_version: int, to_version: int) -> None:
    """Apply in-place schema upgrades.

    Each 'if' below is one real, shipped migration step. Add a new
    'if from_version == N:' block here when the next real migration is
    needed -- do not pre-build a generic chain for hops that don't exist
    yet; there is exactly one hop today (1 -> 2).
    """
    if from_version == 1:
        conn.execute("ALTER TABLE experiments ADD COLUMN solve_threshold REAL")
        from_version = 2

    conn.execute(
        "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
        (str(to_version),),
    )
    conn.commit()

import sqlite3

from tensor_optix.ledger.store import ExperimentStore


def _create_v1_db(db_path: str) -> None:
    """Hand-construct a v1-schema database (pre-solve_threshold, pre-SCHEMA_VERSION-2)
    using a frozen snapshot of the old DDL -- this is deliberate so the test
    exercises a real upgrade, not a no-op against code already at v2."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE experiments (
            experiment_id          TEXT PRIMARY KEY,
            parent_experiment_id   TEXT,
            study_id               TEXT,
            task_env               TEXT NOT NULL,
            algorithm               TEXT NOT NULL,
            framework               TEXT NOT NULL,
            seed                    INTEGER NOT NULL,
            checkpoint_dir          TEXT NOT NULL,
            config_json             TEXT NOT NULL DEFAULT '{}',
            tensor_optix_version    TEXT,
            source_revision         TEXT,
            library_versions_json   TEXT NOT NULL DEFAULT '{}',
            hostname                TEXT,
            schema_version          INTEGER NOT NULL DEFAULT 1,
            state                   TEXT NOT NULL DEFAULT 'created',
            evaluation_status       TEXT NOT NULL DEFAULT 'pending',
            created_at              REAL NOT NULL,
            started_at              REAL,
            finished_at              REAL,
            best_primary_score      REAL,
            best_train_score        REAL,
            best_val_score          REAL,
            generalization_gap      REAL,
            final_metrics_json      TEXT NOT NULL DEFAULT '{}',
            error_message           TEXT
        );

        CREATE TABLE events (
            event_id       TEXT NOT NULL PRIMARY KEY,
            experiment_id  TEXT NOT NULL,
            sequence       INTEGER NOT NULL,
            event_type     TEXT NOT NULL,
            source         TEXT NOT NULL DEFAULT 'loop_controller',
            step           INTEGER,
            payload_json   TEXT NOT NULL DEFAULT '{}',
            timestamp      REAL NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE (experiment_id, sequence)
        );

        CREATE TABLE artifacts (
            experiment_id  TEXT NOT NULL,
            snapshot_id    TEXT NOT NULL,
            checkpoint_dir TEXT NOT NULL,
            snapshot_dir   TEXT NOT NULL,
            weights_path   TEXT NOT NULL,
            primary_score  REAL NOT NULL,
            episode_id     INTEGER NOT NULL,
            checksum       TEXT,
            created_at     REAL NOT NULL,
            PRIMARY KEY (experiment_id, snapshot_id)
        );

        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1');

        INSERT INTO experiments (
            experiment_id, task_env, algorithm, framework, seed, checkpoint_dir,
            schema_version, created_at
        ) VALUES ('pre-migration-exp', 'CartPole-v1', 'DQN', 'torch', 0, '/tmp/ckpt', 1, 1000.0);
        """
    )
    conn.commit()
    conn.close()


def test_migration_adds_solve_threshold_column_and_bumps_version(tmp_path):
    db_path = str(tmp_path / "v1_ledger.sqlite3")
    _create_v1_db(db_path)

    store = ExperimentStore(db_path)

    columns = [row[1] for row in store._conn.execute("PRAGMA table_info(experiments)").fetchall()]
    assert "solve_threshold" in columns

    version_row = store._conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert version_row[0] == "2"

    pre_migration_record = store.get_experiment("pre-migration-exp")
    assert pre_migration_record is not None
    assert pre_migration_record.solve_threshold is None

    record = store.create_experiment(
        task_env="Acrobot-v1", algorithm="PPO", framework="torch", seed=0,
        checkpoint_dir="/tmp/ckpt2", solve_threshold=123.0,
    )
    round_tripped = store.get_experiment(record.experiment_id)
    assert round_tripped.solve_threshold == 123.0

    store.close()


def test_migration_is_idempotent_on_reopen(tmp_path):
    db_path = str(tmp_path / "v1_ledger.sqlite3")
    _create_v1_db(db_path)

    store1 = ExperimentStore(db_path)
    store1.close()

    # Second open must not attempt the ALTER TABLE again (would raise
    # "duplicate column name" if existing_version < SCHEMA_VERSION were
    # still true after the first migration).
    store2 = ExperimentStore(db_path)
    version_row = store2._conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert version_row[0] == "2"
    store2.close()

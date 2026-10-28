import sqlite3

import pytest

from tensor_optix.ledger.records import SCHEMA_VERSION
from tensor_optix.ledger.store import ExperimentStore


def test_schema_creates_all_tables(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"schema_meta", "experiments", "events", "artifacts"} <= tables
    store.close()


def test_schema_is_idempotent_on_reopen(tmp_path):
    db_path = str(tmp_path / "ledger.sqlite3")
    store1 = ExperimentStore(db_path)
    store1.close()
    # Reopening must not raise and must not duplicate tables.
    store2 = ExperimentStore(db_path)
    count = store2._conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'experiments'"
    ).fetchone()[0]
    assert count == 1
    store2.close()


def test_schema_version_recorded(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    row = store._conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert row is not None
    assert int(row[0]) == SCHEMA_VERSION
    store.close()


def test_schema_rejects_future_version(tmp_path):
    db_path = str(tmp_path / "ledger.sqlite3")
    store = ExperimentStore(db_path)
    store._conn.execute(
        "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    store._conn.commit()
    store.close()

    with pytest.raises(RuntimeError):
        ExperimentStore(db_path)

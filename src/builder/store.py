"""Builder — result store.

Every sanitized analysis payload is persisted to a SQLite database in the
researcher's working directory (`<cwd>/.builder/results.db`). Claude
carries only the ID + one-liner label in context; when it needs the full
payload, it calls `expand_result(id)` which hits this store.

Why SQLite, here:
- Single-file, no server, part of the Python stdlib. Zero ops burden for
  the researcher.
- Survives across invocations. A session's results are available the next
  time Builder is launched in that directory.
- Queryable. Future session-state features (step 7) extend the same
  schema.
- Atomic writes. A crash mid-insert doesn't corrupt the store.

Things this deliberately does NOT handle at v0:
- Multi-session separation. All results for a cwd share one table.
  Sessions are step 7.
- Full-text search. Not needed until long analyses appear.
- Compression. Sanitized payloads are small JSON; no need yet.
- Raw-log archival. Step 4's executor will persist raw logs (in the
  researcher's TUI-visible channel only — NEVER to Claude). This store
  holds the sanitized payload and a pointer to the raw log on disk.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# Where results land, relative to the cwd. The `.builder` prefix keeps
# the directory out of most project listings (and matches conventions
# like `.git`, `.venv`).
STORE_SUBDIR = ".builder"
DB_FILENAME = "results.db"


@dataclass
class StoredResult:
    """One row of the results table, hydrated."""
    id: str
    label: str
    analysis_type: str
    sanitized_payload: dict[str, Any]
    language: str                        # "R" or "Stata"
    script_code: str                     # original source for audit
    transformations: list[str]           # what the sanitizer did
    raw_log_path: str | None             # filesystem pointer, not content
    created_at: str                      # ISO 8601 UTC


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ResultStore:
    """Thin wrapper over a SQLite file.

    Methods are synchronous — SQLite's own locking is enough for our
    single-writer-single-reader pattern (the MCP tool serializes tool
    calls per session anyway).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS results (
        id                TEXT PRIMARY KEY,
        label             TEXT NOT NULL,
        analysis_type     TEXT NOT NULL,
        sanitized_payload TEXT NOT NULL,  -- JSON
        language          TEXT NOT NULL,
        script_code       TEXT NOT NULL,
        transformations   TEXT NOT NULL,  -- JSON array
        raw_log_path      TEXT,
        created_at        TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_results_created_at ON results (created_at);
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit; we manage transactions with
        # explicit BEGIN/COMMIT blocks.
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # -- Insert -------------------------------------------------------------

    def insert(
        self,
        *,
        label: str,
        analysis_type: str,
        sanitized_payload: dict[str, Any],
        language: str,
        script_code: str,
        transformations: list[str],
        raw_log_path: Path | None = None,
    ) -> StoredResult:
        """Add a new result; return the hydrated row (including assigned ID)."""
        result_id = self._next_id()
        now = datetime.now(timezone.utc).isoformat()
        row = StoredResult(
            id=result_id,
            label=label,
            analysis_type=analysis_type,
            sanitized_payload=sanitized_payload,
            language=language,
            script_code=script_code,
            transformations=list(transformations),
            raw_log_path=str(raw_log_path) if raw_log_path else None,
            created_at=now,
        )
        with self._txn():
            self._conn.execute(
                "INSERT INTO results (id, label, analysis_type, "
                "sanitized_payload, language, script_code, transformations, "
                "raw_log_path, created_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.id,
                    row.label,
                    row.analysis_type,
                    json.dumps(row.sanitized_payload, ensure_ascii=False),
                    row.language,
                    row.script_code,
                    json.dumps(row.transformations, ensure_ascii=False),
                    row.raw_log_path,
                    row.created_at,
                ),
            )
        return row

    # -- Read ---------------------------------------------------------------

    def get(self, result_id: str) -> StoredResult | None:
        cur = self._conn.execute(
            "SELECT * FROM results WHERE id = ?", (result_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._hydrate(row)

    def list_all(self) -> list[StoredResult]:
        cur = self._conn.execute(
            "SELECT * FROM results ORDER BY created_at ASC"
        )
        return [self._hydrate(r) for r in cur.fetchall()]

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM results")
        return int(cur.fetchone()["c"])

    # -- Internals ----------------------------------------------------------

    def _next_id(self) -> str:
        """Assign the next sequential ID of the form `M1`, `M2`, ...

        We compute it from the current row count rather than using SQLite's
        autoincrement because the IDs are human-facing (Claude types them)
        and should be readable / predictable. Concurrent inserts aren't a
        concern; see class docstring.
        """
        return f"M{self.count() + 1}"

    def _hydrate(self, row: sqlite3.Row) -> StoredResult:
        return StoredResult(
            id=row["id"],
            label=row["label"],
            analysis_type=row["analysis_type"],
            sanitized_payload=json.loads(row["sanitized_payload"]),
            language=row["language"],
            script_code=row["script_code"],
            transformations=json.loads(row["transformations"]),
            raw_log_path=row["raw_log_path"],
            created_at=row["created_at"],
        )

    # Minimal transaction helper — we don't have complex write patterns yet.
    def _txn(self) -> "_Txn":
        return _Txn(self._conn)


class _Txn:
    """Tiny context manager for atomic writes.

    sqlite3's default transactions are implicit and surprising; this makes
    it explicit. BEGIN IMMEDIATE avoids the deferred-lock upgrade that
    can cause busy errors under contention.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> "_Txn":
        self._conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# Convenience: single-process store pinned to the cwd
# ---------------------------------------------------------------------------

_store: ResultStore | None = None


def get_store(cwd: Path) -> ResultStore:
    """Return the process-wide store for `cwd`, creating it on first call."""
    global _store
    if _store is None:
        _store = ResultStore(cwd / STORE_SUBDIR / DB_FILENAME)
    return _store


def reset_store_for_tests() -> None:
    """Test-only hook: drop the cached process-wide store."""
    global _store
    if _store is not None:
        _store.close()
    _store = None

"""Nora — result store.

Every sanitized analysis payload is persisted to a SQLite database in the
researcher's working directory (`<cwd>/.nora/results.db`). Claude
carries only the ID + one-liner label in context; when it needs the full
payload, it calls `expand_result(id)` which hits this store.

Why SQLite, here:
- Single-file, no server, part of the Python stdlib. Zero ops burden for
  the researcher.
- Survives across invocations. A session's results are available the next
  time Nora is launched in that directory.
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


# Where results land, relative to the cwd. The `.nora` prefix keeps
# the directory out of most project listings (and matches conventions
# like `.git`, `.venv`).
STORE_SUBDIR = ".nora"
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
    # Groups results that came from the same submit_script call. NULL on
    # rows from before the multi-result wire format (one row per call).
    # Today's submit_script generates one script_run_id per invocation
    # and tags every row produced by that invocation with it.
    script_run_id: str | None = None


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
        created_at        TEXT NOT NULL,
        script_run_id     TEXT             -- groups multi-result submit_script calls; NULL on legacy rows
    );
    CREATE INDEX IF NOT EXISTS idx_results_created_at ON results (created_at);
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit; we manage transactions with
        # explicit BEGIN/COMMIT blocks.
        #
        # check_same_thread=False: Nora's bridge runs in pywebview's
        # webview thread while tool calls (submit_script, list_results,
        # expand_result) run on the asyncio runner thread. Whichever
        # thread first calls ``get_store`` opens the connection; the
        # other thread reusing the cached store would otherwise get
        # ``ProgrammingError: SQLite objects created in a thread can
        # only be used in that same thread``. Single-writer-single-
        # reader serialization (this class's docstring) plus SQLite's
        # own locking and the GIL is enough; we don't need Python's
        # default thread-affinity check on top.
        self._conn = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Apply additive migrations to an existing DB.

        SQLite's ``CREATE TABLE IF NOT EXISTS`` skips the body when the
        table is already there, so a fresh column declared in SCHEMA
        won't reach a pre-existing DB on its own. Each migration step
        is idempotent: check ``PRAGMA table_info`` before issuing the
        ``ALTER``, so re-running on an already-migrated DB is a no-op.
        Indexes that depend on migrated columns are created here too,
        not in SCHEMA: the SCHEMA's CREATE INDEX runs before _migrate
        on legacy DBs and would fail referencing a column that doesn't
        exist yet.
        """
        cols = {row["name"] for row in self._conn.execute(
            "PRAGMA table_info(results)"
        ).fetchall()}
        if "script_run_id" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN script_run_id TEXT"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_script_run_id "
            "ON results (script_run_id)"
        )

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
        script_run_id: str | None = None,
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
            script_run_id=script_run_id,
        )
        with self._txn():
            self._conn.execute(
                "INSERT INTO results (id, label, analysis_type, "
                "sanitized_payload, language, script_code, transformations, "
                "raw_log_path, created_at, script_run_id) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    row.script_run_id,
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

    def list_by_script_run(self, script_run_id: str) -> list[StoredResult]:
        """All rows produced by one ``submit_script`` invocation, in
        emission order. Returns ``[]`` for unknown ids or for legacy
        rows where the field was never set."""
        cur = self._conn.execute(
            "SELECT * FROM results WHERE script_run_id = ? "
            "ORDER BY created_at ASC, id ASC",
            (script_run_id,),
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
        keys = row.keys() if hasattr(row, "keys") else None
        run_id = row["script_run_id"] if (
            keys is None or "script_run_id" in keys
        ) else None
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
            script_run_id=run_id,
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
# Convenience: per-cwd store cache
# ---------------------------------------------------------------------------
#
# Earlier versions cached exactly one ``ResultStore`` instance and
# returned it regardless of the requested cwd — so after a session
# switch, tool calls against the new cwd were writing to the OLD
# session's ``results.db``. Catastrophic: Project A could see
# Project B's stored sanitized results by calling ``list_results``
# in the same app process.
#
# We now key the cache by the resolved cwd. Switching sessions gets
# a fresh store that points at the new session's DB; re-opening the
# same session reuses the existing one (SQLite connection open is
# cheap but not free — reuse is a modest win, and the file handle
# limit is finite).

_stores: dict[Path, ResultStore] = {}


def get_store(cwd: Path) -> ResultStore:
    """Return the store pinned to ``cwd`` — NOT a process-wide
    singleton.

    Resolves the cwd before keying the cache so two paths that
    normalize to the same directory (one with symlinks or ``./``,
    one without) share a store rather than racing on the same
    sqlite file through two different handles.
    """
    key = cwd.resolve()
    existing = _stores.get(key)
    if existing is not None:
        return existing
    store = ResultStore(key / STORE_SUBDIR / DB_FILENAME)
    _stores[key] = store
    return store


def close_store(cwd: Path) -> None:
    """Close and drop the cached store for ``cwd``.

    Called from the UI bridge when the session switches so the new
    cwd's store isn't shadowed by a stale handle. Safe to call when
    no store exists for this cwd — it's a no-op in that case.
    """
    key = cwd.resolve()
    existing = _stores.pop(key, None)
    if existing is not None:
        try:
            existing.close()
        except Exception:  # noqa: BLE001 — closing a dead handle shouldn't crash the switch
            pass


def reset_store_for_tests() -> None:
    """Test-only hook: close and drop every cached store. Used to
    clear process-wide state between tests so one test's store
    can't leak into the next."""
    global _stores
    for store in list(_stores.values()):
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
    _stores = {}

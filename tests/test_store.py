"""Tests for the result store.

Scope: insert round-trips, ID sequencing, list/get retrieval, persistence
across connections. Kept focused — the store is simple enough that
exhaustive property testing isn't warranted; the sanitizer is where the
real guarantee lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora.store import (
    ResultStore,
    close_store,
    get_store,
    reset_store_for_tests,
)


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / ".nora" / "results.db")


def _sample_payload() -> dict:
    return {
        "type": "linear_regression",
        "n": 1000,
        "coefficients": {"x": 0.5},
        "standard_errors": {"x": 0.05},
        "r_squared": 0.42,
        "response_variable": "y",
        "predictor_variables": ["x"],
    }


def test_empty_store_has_zero_count(store: ResultStore):
    assert store.count() == 0
    assert store.list_all() == []


def test_store_enables_secure_delete(store: ResultStore):
    """``purge_script_code`` blanks the ``script_code`` column to drop
    any credentials / PII the researcher may have pasted into a
    script that has since been hidden by rewind. Without
    ``PRAGMA secure_delete = ON`` the previous bytes survive on
    SQLite's freelist until the page is overwritten by an unrelated
    insert, and a forensic tool reading raw pages (undark, hexdump,
    sqlite3_analyzer) can recover the supposedly-purged content
    even when the row has been re-saved as empty.

    The fix sets the PRAGMA at connection time. Test reads it back
    to confirm the connection state actually carries the property —
    a future regression that drops the line in __init__ would
    re-open the recovery channel silently.
    """
    cur = store._conn.execute("PRAGMA secure_delete")
    value = cur.fetchone()[0]
    # SQLite reports it as ``1`` (boolean ON) on macOS' sqlite3
    # build. The pragma's accepted boolean inputs are 0/1/ON/OFF;
    # the read-back is the numeric form.
    assert value == 1, (
        f"secure_delete must be ON so purged script_code bytes are "
        f"zeroed on the freelist; got value={value!r}"
    )


def test_store_creates_nora_dir_with_owner_only_mode(tmp_path: Path):
    """Every Nora-owned file lives under ``<cwd>/.nora``. Default
    umask leaves files inside at 0o644 — readable by other users on
    any filesystem where the home directory's mode permits
    traversal (HPC, NFS, university research servers). Gating the
    parent directory at 0o700 makes every descendant unreachable
    via the no-execute-on-parent rule, regardless of per-file
    modes. The store's __init__ routes through
    ``ensure_private_nora_dir`` to enforce this whenever it lazily
    creates the directory.
    """
    import os
    import stat

    db_path = tmp_path / ".nora" / "results.db"
    ResultStore(db_path)
    mode = stat.S_IMODE(os.stat(tmp_path / ".nora").st_mode)
    assert mode == 0o700, (
        f"expected .nora at 0o700, got {oct(mode)}; lower modes "
        f"expose chat history / raw logs / pre-SDC result.json on "
        f"shared filesystems"
    )


def test_concurrent_inserts_get_distinct_ids(store: ResultStore):
    """The runner thread (submit_script → insert) and the bridge
    thread (rewind → hide_results_not_in / unhide_results, sidebar
    render → list_all) share one cached ResultStore per cwd.
    ``check_same_thread=False`` allows the cross-thread reuse but
    the sqlite3 module relies on Python-side serialization. The
    pre-fix ``_next_id`` read ``count()`` OUTSIDE the transaction,
    so two concurrent inserters could both observe N, both compute
    M(N+1), and one of them would crash on UNIQUE-constraint
    violation when it tried to commit.

    With the per-store lock + id allocation inside the BEGIN
    IMMEDIATE transaction, concurrent inserts serialize cleanly:
    each gets a distinct sequential id, no collisions.
    """
    import threading

    payload = _sample_payload()
    barrier = threading.Barrier(8)

    def insert_one(label: str) -> str:
        # Wait for all threads to be ready before racing, so the
        # window where two inserters could read the same count is
        # maximized.
        barrier.wait()
        result = store.insert(
            label=label, analysis_type="linear_regression",
            sanitized_payload=payload,
            language="R", script_code="x <- 1", transformations=[],
        )
        return result.id

    ids: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        rid = insert_one(f"row {i}")
        with lock:
            ids.append(rid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All ids must be unique.
    assert len(set(ids)) == 8, (
        f"concurrent inserts collided on the same id: {sorted(ids)}"
    )
    # And contiguous M1..M8 — the lock + in-txn count delivers
    # monotone sequential ids even under racing inserters.
    assert sorted(ids, key=lambda s: int(s[1:])) == [f"M{i}" for i in range(1, 9)]
    assert store.count() == 8


def test_insert_returns_sequential_ids(store: ResultStore):
    r1 = store.insert(
        label="first", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R", script_code="x <- 1", transformations=[],
    )
    r2 = store.insert(
        label="second", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R", script_code="x <- 2", transformations=[],
    )
    r3 = store.insert(
        label="third", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="Stata", script_code="reg y x", transformations=[],
    )
    assert r1.id == "M1"
    assert r2.id == "M2"
    assert r3.id == "M3"
    assert store.count() == 3


def test_get_returns_inserted_row(store: ResultStore):
    inserted = store.insert(
        label="my regression",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="library(stats); lm(y ~ x)",
        transformations=["clamped coefficients to 4 sig figs"],
    )
    fetched = store.get(inserted.id)
    assert fetched is not None
    assert fetched.id == inserted.id
    assert fetched.label == "my regression"
    assert fetched.analysis_type == "linear_regression"
    assert fetched.sanitized_payload == _sample_payload()
    assert fetched.language == "R"
    assert fetched.script_code == "library(stats); lm(y ~ x)"
    assert fetched.transformations == ["clamped coefficients to 4 sig figs"]


def test_get_missing_returns_none(store: ResultStore):
    assert store.get("M999") is None
    assert store.get("") is None


def test_list_all_orders_by_creation(store: ResultStore):
    for i in range(5):
        store.insert(
            label=f"label-{i}",
            analysis_type="descriptive",
            sanitized_payload={"type": "descriptive", "variable": f"v{i}"},
            language="R",
            script_code=f"# script {i}",
            transformations=[],
        )
    rows = store.list_all()
    assert [r.label for r in rows] == [f"label-{i}" for i in range(5)]


def test_list_by_script_run_orders_by_insertion_not_lexical_id(
    store: ResultStore,
):
    """Multiple helpers in one ``submit_script`` call insert N rows in
    a tight loop where ``created_at`` can plausibly tie at sub-
    microsecond resolution. The previous tiebreaker, ``ORDER BY id
    ASC`` on a string id, sorted lexically (M1, M10, M11, ..., M2)
    instead of in insertion order. Switch to ``rowid`` (always
    monotone in insert order) and pin the property under a forced
    same-timestamp scenario.
    """
    run_id = "R-deadbeef"
    forced_ts = "2026-01-01T00:00:00.000000+00:00"
    for i in range(12):
        row = store.insert(
            label=f"label-{i}",
            analysis_type="descriptive",
            sanitized_payload={"type": "descriptive", "variable": f"v{i}"},
            language="R",
            script_code="x",
            transformations=[],
            script_run_id=run_id,
        )
        # Force the timestamp tie that motivates this fix.
        store._conn.execute(
            "UPDATE results SET created_at = ? WHERE id = ?",
            (forced_ts, row.id),
        )

    rows = store.list_by_script_run(run_id)
    assert [r.label for r in rows] == [f"label-{i}" for i in range(12)]
    # Cross-check: the previous lexical-id ordering would have put
    # M10 before M2. Confirm we don't see that.
    ids = [r.id for r in rows]
    assert ids.index("M2") < ids.index("M10")


def test_persistence_across_connections(tmp_path: Path):
    db = tmp_path / ".nora" / "results.db"
    s1 = ResultStore(db)
    s1.insert(
        label="persistent",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="x",
        transformations=["a"],
    )
    s1.close()

    s2 = ResultStore(db)
    assert s2.count() == 1
    row = s2.get("M1")
    assert row is not None
    assert row.label == "persistent"
    assert row.transformations == ["a"]
    s2.close()


def test_unicode_roundtrip(store: ResultStore):
    """Labels, code, and transformations with non-ASCII survive the JSON round-trip."""
    store.insert(
        label="régression — n=200",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="# résidus… émission",
        transformations=["clamped — 4 sig figs"],
    )
    row = store.get("M1")
    assert row is not None
    assert row.label == "régression — n=200"
    assert row.script_code == "# résidus… émission"
    assert row.transformations == ["clamped — 4 sig figs"]


# ---------------------------------------------------------------------------
# Cross-session isolation — the get_store cache
# ---------------------------------------------------------------------------
#
# Earlier versions cached exactly one ResultStore process-wide, so
# after a session switch Project A could see Project B's sanitized
# results in the same app process. These tests lock in the fix:
# get_store is now keyed by resolved cwd, and close_store drops
# the cached handle so the UI switch path can force a clean state.


@pytest.fixture(autouse=True)
def _reset_store_cache():
    """Every test starts with an empty cache. Without this, state
    from an earlier test in the same run can mask a real bug in
    the cache logic under test."""
    reset_store_for_tests()
    yield
    reset_store_for_tests()


def test_get_store_is_per_cwd(tmp_path: Path):
    """Two different cwds must get two different stores pointing at
    two different DBs — NOT a shared singleton."""
    session_a = tmp_path / "session-a"
    session_b = tmp_path / "session-b"
    session_a.mkdir()
    session_b.mkdir()

    store_a = get_store(session_a)
    store_b = get_store(session_b)

    assert store_a is not store_b
    assert store_a.db_path.parent.parent == session_a
    assert store_b.db_path.parent.parent == session_b


def test_get_store_same_cwd_returns_same_instance(tmp_path: Path):
    """Repeated calls for the same cwd reuse the handle — sqlite
    connections aren't free, and the UI tool calls hit get_store
    on every invocation."""
    (tmp_path / "s").mkdir()
    first = get_store(tmp_path / "s")
    second = get_store(tmp_path / "s")
    assert first is second


def test_get_store_normalizes_path(tmp_path: Path):
    """Two cwd paths that resolve to the same directory share one
    store. Without this, a cwd passed as './data' could race with
    the same cwd passed as its absolute form on the same sqlite
    file through two different handles."""
    session = tmp_path / "s"
    session.mkdir()
    via_absolute = get_store(session)
    via_relative = get_store(session / "." / "nested" / "..")
    assert via_absolute is via_relative


def test_insert_in_one_cwd_is_invisible_from_another(tmp_path: Path):
    """The core cross-session-leak regression test. Insert a row
    into Project A's store; a fresh get_store for Project B must
    see zero rows. Previously Project B got Project A's store
    back and saw all its results."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    get_store(tmp_path / "a").insert(
        label="project-a secret result",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="x",
        transformations=["a"],
    )

    b = get_store(tmp_path / "b")
    assert b.count() == 0
    assert b.list_all() == []


def test_close_store_drops_cache_entry(tmp_path: Path):
    """After close_store(cwd), the next get_store(cwd) must open a
    fresh handle rather than returning the closed one."""
    session = tmp_path / "s"
    session.mkdir()
    first = get_store(session)
    close_store(session)
    second = get_store(session)
    assert first is not second


def test_close_store_is_safe_with_no_cached_entry(tmp_path: Path):
    """close_store for a cwd that was never opened must be a
    no-op, not an error — the UI calls it defensively on every
    session switch."""
    close_store(tmp_path / "never-opened")  # must not raise


def test_store_can_be_used_across_threads(tmp_path: Path) -> None:
    """The bridge thread opens the store via ``_build_context_prefix``
    on session resume. The asyncio runner thread reuses the cached
    store via ``submit_script`` / ``list_results`` / ``expand_result``.
    Without ``check_same_thread=False`` on the SQLite connection, the
    second thread blows up with ``ProgrammingError: SQLite objects
    created in a thread can only be used in that same thread.`` The
    store's docstring already promises single-writer-single-reader
    serialization, so SQLite's locking plus the GIL is enough; we
    don't need Python's thread-affinity check on top.
    """
    import threading

    session = tmp_path / "session"
    session.mkdir()

    store = get_store(session)
    store.insert(
        label="opened-on-bridge-thread",
        analysis_type="ttest",
        sanitized_payload={"x": 1},
        language="R",
        script_code="t.test(1:5)",
        transformations=[],
    )

    captured: dict[str, object] = {}

    def _use_from_other_thread() -> None:
        try:
            again = get_store(session)
            captured["count"] = again.count()
            captured["rows_visible"] = len(again.list_all())
            again.insert(
                label="written-on-runner-thread",
                analysis_type="lm",
                sanitized_payload={"y": 2},
                language="Python",
                script_code="ols(...)",
                transformations=[],
            )
        except Exception as exc:  # pragma: no cover — only fires on regression
            captured["error"] = exc

    t = threading.Thread(target=_use_from_other_thread)
    t.start()
    t.join(timeout=5.0)

    assert "error" not in captured, (
        f"cross-thread store access raised: {captured.get('error')!r}. "
        "Restore check_same_thread=False on sqlite3.connect()."
    )
    assert captured["count"] == 1, "thread B couldn't read thread A's row"
    assert len(store.list_all()) == 2, "thread B's write didn't land in shared store"

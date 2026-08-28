"""Tests for the env-gated cross-session result recall.

The bridge keeps per-session result stores (a SQLite DB under each
session's ``.nora/`` dir). By default the model only ever reads the
current session's store — researcher-side project separation. With
``NORA_ALLOW_CROSS_SESSION_RECALL=1`` set the model gains two
extensions:

  - ``list_results_global(query?)`` walks ``~/.nora-sessions/`` and
    returns rows tagged with their ``session_path``.
  - ``expand_result(result_id, session_path=...)`` looks up the
    payload in the named session's store.

Stored payloads are pre-sanitized so the privacy boundary is
preserved either way; the gate exists because researchers may want
explicit project separation regardless of payload safety.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import use_cwd
from nora.store import StoredResult, get_store, reset_store_for_tests
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    return json.loads(payload["content"][0]["text"])


def _insert_fake_result(
    cwd: Path, *, label: str, analysis_type: str = "linear_regression",
) -> StoredResult:
    """Plant a stored result in the session's results.db using the
    same insert path the real submit_script handler uses."""
    store = get_store(cwd)
    return store.insert(
        label=label,
        analysis_type=analysis_type,
        sanitized_payload={"type": analysis_type, "n": 100},
        language="R",
        script_code="lm(y ~ x, data=df)",
        transformations=[],
        raw_log_path=None,
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset the store-cache and the cross-session env var between
    tests so one test's state doesn't leak into the next."""
    import os
    reset_store_for_tests()
    prior = os.environ.pop("NORA_ALLOW_CROSS_SESSION_RECALL", None)
    yield
    reset_store_for_tests()
    if prior is not None:
        os.environ["NORA_ALLOW_CROSS_SESSION_RECALL"] = prior
    else:
        os.environ.pop("NORA_ALLOW_CROSS_SESSION_RECALL", None)


def _patch_sessions_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """list_results_global iterates ``ui.SESSIONS_ROOT``; pivot it
    onto the test's temp dir so we don't read the dev's real
    sessions during tests."""
    import nora.ui as ui_mod
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", root, raising=True)


# ---------------------------------------------------------------------------
# list_results_global — gating + listing
# ---------------------------------------------------------------------------

def test_list_results_global_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var, the tool returns a clear 'disabled'
    message naming the env var to set. Stored payloads remain
    untouched; the model just sees a denial."""
    _patch_sessions_root(monkeypatch, tmp_path)
    other = tmp_path / "20260101T000000Z_aaa"
    other.mkdir()
    _insert_fake_result(other, label="prior project")

    res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    assert body["status"] == "denied"
    assert "NORA_ALLOW_CROSS_SESSION_RECALL" in body["reason"]


def test_list_results_global_enabled_lists_other_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the env var on, the tool walks every session under the
    sessions root and returns rows tagged with session_path."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)

    current = tmp_path / "20260101T000000Z_current"
    other_a = tmp_path / "20260101T000000Z_alpha"
    other_b = tmp_path / "20260101T000000Z_beta"
    for d in (current, other_a, other_b):
        d.mkdir()

    _insert_fake_result(current, label="current's regression")
    _insert_fake_result(other_a, label="alpha's regression")
    _insert_fake_result(other_b, label="beta's regression")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    # The CURRENT session's row is excluded from the global list —
    # the model already has list_results for that. Cross-session is
    # the value-add; double-listing wastes tokens.
    labels = [r["label"] for r in body["results"]]
    assert "alpha's regression" in labels
    assert "beta's regression" in labels
    assert "current's regression" not in labels
    # Every row carries a session_path so expand_result can use it.
    for r in body["results"]:
        assert r["session_path"]
        assert r["session_name"]


def test_list_results_global_caps_at_limit_and_reports_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user with many sessions × many results would otherwise ship
    megabytes of metadata into the model context on a single call.
    Same default and hard cap as ``list_results``; truncation is
    surfaced via ``total`` and ``truncated``.
    """
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)

    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()

    # Insert more rows than the explicit ``limit`` we ask for.
    for i in range(7):
        _insert_fake_result(other, label=f"row {i:02d}")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"limit": 3}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["total"] == 7
    assert body["count"] == 3
    assert body["limit"] == 3
    assert body["truncated"] is True
    assert len(body["results"]) == 3


def test_list_results_global_query_filters_by_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty query string filters results by case-insensitive
    substring on label or analysis_type."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)

    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()

    _insert_fake_result(other, label="H1: ln_rev_total event-study")
    _insert_fake_result(other, label="size split: small orgs")
    _insert_fake_result(other, label="bcov_lo program coverage")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": "size"}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    labels = [r["label"] for r in body["results"]]
    assert labels == ["size split: small orgs"]


# ---------------------------------------------------------------------------
# expand_result with session_path — gating + cross-session lookup
# ---------------------------------------------------------------------------

def test_expand_result_cross_session_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is off, passing session_path is denied with
    a clear message naming the env var. The current-session lookup
    (no session_path) still works."""
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()

    other_row = _insert_fake_result(other, label="other's analysis")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": other_row.id,
            "session_path": str(other),
        }))
    body = _mcp_text(res)
    assert body["status"] == "denied"
    assert "NORA_ALLOW_CROSS_SESSION_RECALL" in body["reason"]


def test_expand_result_cross_session_enabled_fetches_from_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the env var on and a valid session_path under
    ~/.nora-sessions/, expand_result fetches the payload from the
    other session's store."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()

    other_row = _insert_fake_result(other, label="other's analysis")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": other_row.id,
            "session_path": str(other),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["label"] == "other's analysis"
    assert body["session_path"] == str(other.resolve())


def test_expand_result_rejects_session_path_outside_sessions_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session_path is path-confined to ~/.nora-sessions/ so a
    prompt-injected request can't direct the store-loader at
    arbitrary paths on the machine."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    current.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": "M1",
            "session_path": str(outside),
        }))
    body = _mcp_text(res)
    assert body["status"] == "denied"
    assert "~/.nora-sessions/" in body["reason"]


def test_expand_result_rejects_sessions_root_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sessions root itself is not a session — accepting it
    would let ``get_store(target_cwd)`` create
    ``~/.nora-sessions/.nora/results.db`` at the root, polluting
    the directory the bridge uses to enumerate sessions. The
    narrow gate matches the equivalent fix in
    ``ui.switch_session`` / ``ui.delete_session``."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)
    current = sessions_root / "current"
    current.mkdir()

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": "M1",
            "session_path": str(sessions_root),
        }))
    body = _mcp_text(res)
    assert body["status"] == "denied"
    # Crucial: no .nora/ was planted at the root.
    assert not (sessions_root / ".nora").exists()


def test_expand_result_rejects_nested_path_under_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subdirectory inside a session is not itself a session.
    Accepting it would create a parallel ``<subdir>/.nora/results.db``
    that the bridge's session-management paths don't see."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)
    current = sessions_root / "current"
    current.mkdir()
    other = sessions_root / "other"
    other.mkdir()
    nested = other / "subdir"
    nested.mkdir()

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": "M1",
            "session_path": str(nested),
        }))
    body = _mcp_text(res)
    assert body["status"] == "denied"
    # And no .nora/ planted at the nested location either.
    assert not (nested / ".nora").exists()


# ---------------------------------------------------------------------------
# Folder-backed sessions — opened via ``choose_folder``, registered in
# ``external_sessions``. The cross-session recall surface must include
# these or researchers using "Choose folder" get a silent "result
# doesn't exist" the next time they ask for an older row.
# ---------------------------------------------------------------------------


def test_list_results_global_includes_folder_backed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A folder-backed session registered through ``choose_folder``
    lives outside ``~/.nora-sessions/`` but is a real Nora session.
    ``list_results_global`` must surface its rows; otherwise an
    older result the researcher genuinely stored shows up as
    "doesn't exist" when they ask for it cross-session."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    current.mkdir()

    # A registered project folder, outside SESSIONS_ROOT, with a
    # stored result.
    project_folder = tmp_path / "research_project"
    project_folder.mkdir()
    _insert_fake_result(project_folder, label="folder-backed row")
    from nora import external_sessions
    external_sessions.register(sessions_root, project_folder)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    labels = [r["label"] for r in body["results"]]
    assert "folder-backed row" in labels
    # And the session_path the row publishes points at the folder
    # itself — that's the same path the model will pass back via
    # ``expand_result(session_path=...)``.
    folder_rows = [
        r for r in body["results"] if r["label"] == "folder-backed row"
    ]
    assert len(folder_rows) == 1
    assert Path(folder_rows[0]["session_path"]) == project_folder.resolve()


def test_expand_result_accepts_registered_folder_backed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``expand_result(session_path=<folder>)`` must succeed when
    ``<folder>`` is a registered folder-backed session, even
    though it isn't a direct child of SESSIONS_ROOT."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    current.mkdir()
    project_folder = tmp_path / "research_project"
    project_folder.mkdir()
    row = _insert_fake_result(project_folder, label="folder-backed row")
    from nora import external_sessions
    external_sessions.register(sessions_root, project_folder)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": row.id,
            "session_path": str(project_folder),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["label"] == "folder-backed row"


def test_expand_result_rejects_unregistered_external_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path OUTSIDE SESSIONS_ROOT that ISN'T in the
    ``external_sessions`` registry must still be denied — the gate
    only opens for paths the researcher explicitly opened via the
    picker. Otherwise the registry-aware fix would silently widen
    the surface to any arbitrary directory on the machine."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    current.mkdir()
    # An external folder that LOOKS like a session (has a results
    # db planted) but was never registered.
    unregistered = tmp_path / "unregistered_folder"
    unregistered.mkdir()
    _insert_fake_result(unregistered, label="should be invisible")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": "M1",
            "session_path": str(unregistered),
        }))
    body = _mcp_text(res)
    assert body["status"] == "denied"


def test_list_results_global_omits_unregistered_external_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``list_results_global`` must NOT pick up an external folder
    just because it has a ``.nora/results.db``; only registered
    folder-backed sessions qualify. Symmetric to the
    expand_result rejection above."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    current.mkdir()
    unregistered = tmp_path / "unregistered_folder"
    unregistered.mkdir()
    _insert_fake_result(unregistered, label="ghost row")

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    labels = [r["label"] for r in body["results"]]
    assert "ghost row" not in labels


def test_list_results_global_skips_folder_backed_when_path_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered folder-backed session whose path no longer
    exists on disk (researcher deleted the directory, transient
    unmount, etc.) must be skipped without crashing the scan.
    ``external_sessions.list_entries`` already filters stale
    paths, so the scan never even visits them."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    current = sessions_root / "current"
    legit = sessions_root / "20260101T000000Z_legit"
    for d in (current, legit):
        d.mkdir()
    _insert_fake_result(legit, label="staged row")

    # Register a folder, then delete it.
    transient = tmp_path / "popped_drive_project"
    transient.mkdir()
    from nora import external_sessions
    external_sessions.register(sessions_root, transient)
    import shutil
    shutil.rmtree(transient)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    labels = [r["label"] for r in body["results"]]
    assert "staged row" in labels


def test_list_results_global_skips_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink under ~/.nora-sessions/ pointing at an arbitrary
    directory outside the sessions root must not be followed by
    the global scan. ``is_dir()`` follows symlinks; the fix is to
    resolve the entry and re-check that the target is still a
    direct child of SESSIONS_ROOT — the same discipline the
    bridge enforces on other paths."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    _patch_sessions_root(monkeypatch, sessions_root)

    # A legitimate session with a stored row — we should see this one.
    current = sessions_root / "current"
    legit = sessions_root / "20260101T000000Z_legit"
    for d in (current, legit):
        d.mkdir()
    _insert_fake_result(legit, label="legit row")

    # Plant a results.db OUTSIDE the sessions root and symlink to
    # the *containing directory* from inside the root. Pre-fix the
    # scan would follow the symlink and read that DB.
    outside = tmp_path / "outside_session"
    outside.mkdir()
    (outside / ".nora").mkdir()
    _insert_fake_result(outside, label="LEAKED through symlink")
    (sessions_root / "20260101T000000Z_evil").symlink_to(outside)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["list_results_global"]({"query": ""}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    labels = [r["label"] for r in body["results"]]
    assert "legit row" in labels
    # The symlinked-out content must NOT appear.
    assert "LEAKED through symlink" not in labels


# ---------------------------------------------------------------------------
# compose_results — cross-session rows
# ---------------------------------------------------------------------------
#
# Every session's store numbers results from M1, so "compare this
# session's M1 with that session's M1" is the NORMAL cross-session
# composition, not an edge case. The renderer keys cells by
# result_id alone; compose_results therefore re-keys cross-session
# rows onto call-private aliases before rendering (see the handler).
# These tests pin the observable contract: colliding ids compose
# cleanly, a denied foreign row can't borrow the local payload, and
# the alias never leaks into researcher-facing output.


def _insert_regression_result(
    cwd: Path, *, label: str, estimate: float,
) -> StoredResult:
    """Plant a regression-shaped result whose composed cell carries a
    recognizable estimate, so tests can tell WHICH session's payload
    rendered into a row."""
    store = get_store(cwd)
    return store.insert(
        label=label,
        analysis_type="linear_regression",
        sanitized_payload={
            "type": "linear_regression",
            "coefficients": {"x": estimate},
            "standard_errors": {"x": 0.05},
            "p_values": {"x": 0.01},
            "n": 100,
        },
        language="R",
        script_code="lm(y ~ x, data=df)",
        transformations=[],
        raw_log_path=None,
    )


def _compose_spec(rows: list) -> dict:
    return {
        "columns": [{"id": "x", "label": "x"}],
        "groups": [{"rows": rows}],
    }


def test_compose_results_local_rows_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: two local results compose without any cross-session
    machinery involved."""
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    current.mkdir()
    _insert_regression_result(current, label="ln_revenue", estimate=0.111)
    _insert_regression_result(current, label="ln_expenses", estimate=0.333)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["compose_results"]({
            "spec": _compose_spec(["M1", "M2"]),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["rows_rendered"] == 2
    md = body["markdown"]
    assert "ln_revenue" in md and "ln_expenses" in md
    assert "0.111" in md and "0.333" in md
    assert "missing_result_ids" not in body


def test_compose_results_same_result_id_across_sessions_renders_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline cross-session case: this session's M1 next to
    another session's M1 in one table. Both sessions' stores assign
    M1 independently; the composite must render BOTH payloads, each
    under its own stored label, with no collision error and no
    internal alias leaking into the markdown."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()
    local = _insert_regression_result(
        current, label="this project", estimate=0.111,
    )
    foreign = _insert_regression_result(
        other, label="prior project", estimate=0.222,
    )
    # The collision under test: independent stores, identical ids.
    assert local.id == foreign.id == "M1"

    with use_cwd(current):
        res = asyncio.run(HANDLERS["compose_results"]({
            "spec": _compose_spec([
                "M1",
                {"result_id": "M1", "session_path": str(other)},
            ]),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok", body.get("reason")
    assert body["rows_rendered"] == 2
    md = body["markdown"]
    assert "this project" in md and "prior project" in md
    assert "0.111" in md and "0.222" in md
    # The re-keying alias is an internal render key only.
    assert "\x1f" not in md
    # Researcher-facing id list keeps the original ids.
    assert body["result_ids_referenced"] == ["M1"]


def test_compose_results_denied_foreign_row_cannot_borrow_local_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the env gate OFF, a row naming another session renders as
    a missing-result em-dash — and specifically must NOT pick up the
    local store's payload for the same id through the shared render
    key. (Pre-fix, the renderer keyed by bare rid, so a denied
    foreign M1 silently rendered the local M1's numbers.)"""
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()
    _insert_regression_result(current, label="this project", estimate=0.111)
    _insert_regression_result(other, label="prior project", estimate=0.222)

    with use_cwd(current):
        res = asyncio.run(HANDLERS["compose_results"]({
            "spec": _compose_spec([
                "M1",
                {
                    "result_id": "M1",
                    "session_path": str(other),
                    "label": "prior project",
                },
            ]),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    md = body["markdown"]
    # Local row renders its own numbers; denied row renders em-dash.
    assert "0.111" in md
    assert "0.222" not in md
    assert "—" in md
    assert body["denied_result_ids"] == ["M1"]
    assert "NORA_ALLOW_CROSS_SESSION_RECALL" in body["hint"]


def test_compose_results_missing_foreign_id_keeps_readable_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-session row whose id isn't in the target store renders
    with the ORIGINAL rid as its row label (never the internal
    alias), em-dash cells, and lands in missing_result_ids."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()

    with use_cwd(current):
        res = asyncio.run(HANDLERS["compose_results"]({
            "spec": _compose_spec([
                {"result_id": "M9", "session_path": str(other)},
            ]),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["missing_result_ids"] == ["M9"]
    md = body["markdown"]
    assert "M9" in md
    assert "\x1f" not in md
    assert "—" in md


def test_compose_results_two_spellings_of_same_session_both_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two rows naming the same foreign session through different
    path spellings mint two aliases but resolve to one store — both
    rows must render the payload (the lookup cache fills every alias,
    it doesn't skip repeats)."""
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    _patch_sessions_root(monkeypatch, tmp_path)
    current = tmp_path / "20260101T000000Z_current"
    other = tmp_path / "20260101T000000Z_other"
    for d in (current, other):
        d.mkdir()
    _insert_regression_result(other, label="prior project", estimate=0.222)

    spelling_a = str(other)
    spelling_b = str(other) + "/"
    with use_cwd(current):
        res = asyncio.run(HANDLERS["compose_results"]({
            "spec": _compose_spec([
                {"result_id": "M1", "session_path": spelling_a,
                 "label": "row one"},
                {"result_id": "M1", "session_path": spelling_b,
                 "label": "row two"},
            ]),
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    md = body["markdown"]
    assert md.count("0.222") == 2, md
    assert "row one" in md and "row two" in md

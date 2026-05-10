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

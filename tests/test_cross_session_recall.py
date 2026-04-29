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

"""Lifecycle tests for the multi-runner :class:`NoraBridge`.

The bridge holds a ``dict[str, SessionRunner]`` keyed by cwd. Each
runner owns its own provider session, lock, and turn task. Switching
the visible session is a pure UI focus change — it does NOT close any
runner. These tests pin the contract so a future "helpfully tear down
on switch" regression gets caught.

Persistence is keyed by event ``session_cwd`` (falling back to the
bridge focus), so a turn streaming in session A persists to A's log
even while the UI is showing B.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nora.chat_history import build_context_prefix
from nora.runner import SessionRunner
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Construction / initial state
# ---------------------------------------------------------------------------

def test_bridge_starts_with_no_runners(tmp_path: Path):
    """A freshly-constructed bridge with a cwd has exactly one
    lazy-created runner for that cwd; ``needs_context_prefix`` on
    that runner is True (set by the constructor since the cwd was
    handed in eagerly)."""
    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None, "constructor with cwd should create a runner"
    assert runner.cwd == tmp_path.resolve()
    # The runner's own session is opened lazily on first send.
    assert runner._session is None


def test_bridge_without_cwd_has_no_runners():
    """Without a cwd (landing screen), no runners exist yet. The
    first focus event lazy-creates one."""
    bridge = NoraBridge(cwd=None)
    assert bridge._active_runner() is None
    assert bridge._runners == {}


# ---------------------------------------------------------------------------
# _persist_event — routes by event session_cwd, falls back to bridge focus
# ---------------------------------------------------------------------------

def test_persist_event_adds_iso_timestamp(tmp_path: Path):
    """Persisted events get stamped with a UTC ISO timestamp so the
    Turn reader can order them and the session_state file can show
    'last active' times."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "user_message", "text": "hello"})

    log = tmp_path / ".nora" / "chat_history.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text().splitlines()[0])
    assert "timestamp" in rec
    from datetime import datetime
    parsed = datetime.fromisoformat(rec["timestamp"])
    assert parsed.tzinfo is not None, "timestamp must carry timezone info"


def test_persist_event_skips_non_persist_types(tmp_path: Path):
    """Transient events (turn_done, auth_failure, ready, etc.) must
    not pollute the chat log — otherwise replay reconstructs phantom
    turns."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "turn_done", "input_tokens": 100})
    bridge._persist_event({"type": "ready"})

    log = tmp_path / ".nora" / "chat_history.jsonl"
    assert not log.exists()


def test_persist_event_preserves_caller_timestamp(tmp_path: Path):
    """If the caller already supplied a timestamp (replay, import
    from external log), we don't overwrite it."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({
        "type": "user_message",
        "text": "x",
        "timestamp": "2024-01-01T00:00:00+00:00",
    })
    rec = json.loads(
        (tmp_path / ".nora" / "chat_history.jsonl").read_text().splitlines()[0]
    )
    assert rec["timestamp"] == "2024-01-01T00:00:00+00:00"


def test_persist_event_routes_by_session_cwd(tmp_path: Path):
    """An event carrying ``session_cwd`` lands in THAT session's
    log, not the bridge-focused one. This is the rule that makes
    background sessions safe: a runner whose turn is mid-stream
    persists to ITS cwd even when the UI is focused elsewhere."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    # Bridge focused on A, but the event carries B's cwd — should
    # land in B's log, not A's.
    bridge = NoraBridge(cwd=a)
    bridge._persist_event({
        "type": "user_message",
        "text": "from B's runner",
        "session_cwd": str(b),
    })

    a_log = a / ".nora" / "chat_history.jsonl"
    b_log = b / ".nora" / "chat_history.jsonl"
    assert not a_log.exists(), "A's log must be untouched"
    assert b_log.exists(), "B's log must receive the event"
    rec = json.loads(b_log.read_text().splitlines()[0])
    # session_cwd is a routing annotation — stripped before write.
    assert "session_cwd" not in rec
    assert rec["text"] == "from B's runner"


def test_persist_event_falls_back_to_bridge_focus(tmp_path: Path):
    """Events without ``session_cwd`` (legacy / direct test calls)
    persist to the bridge's focused cwd."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "user_message", "text": "hi"})
    log = tmp_path / ".nora" / "chat_history.jsonl"
    assert log.exists()


def test_record_user_message_replaces_trailing_orphan_turn(tmp_path: Path):
    """A failed send can leave a lone persisted ``user_message`` at
    the tail. The next real send replaces that stale attempt."""
    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None
    bridge._persist_event({
        "type": "user_message",
        "text": "stuck send",
        "session_cwd": str(runner.cwd),
    })

    bridge._record_user_message(runner, "retry")

    log = tmp_path / ".nora" / "chat_history.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["text"] for row in rows if row.get("type") == "user_message"] == [
        "retry"
    ]


def test_record_user_message_keeps_completed_prior_turn(tmp_path: Path):
    """Only orphaned tail user turns are disposable. A completed turn
    with an assistant reply must stay in the log."""
    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None
    bridge._persist_event({
        "type": "user_message",
        "text": "first",
        "session_cwd": str(runner.cwd),
    })
    bridge._persist_event({
        "type": "assistant_text",
        "text": "reply",
        "session_cwd": str(runner.cwd),
    })

    bridge._record_user_message(runner, "second")

    log = tmp_path / ".nora" / "chat_history.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["type"] for row in rows] == [
        "user_message",
        "assistant_text",
        "user_message",
    ]
    assert [row["text"] for row in rows if row["type"] == "user_message"] == [
        "first",
        "second",
    ]


# ---------------------------------------------------------------------------
# switch_session — DOES NOT close any runner (the multi-session fix)
# ---------------------------------------------------------------------------

def test_switch_session_does_not_close_other_runners(tmp_path: Path):
    """The bug we fixed: switching focus used to tear down the
    previous session's SDK client mid-stream. The new contract:
    switching is a pure UI focus change. Both runners' sessions
    stay alive so a turn in flight in A keeps streaming after the
    user clicks B in the sidebar."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    # Force both runners under ~/.nora-sessions/ — switch_session
    # refuses paths outside SESSIONS_ROOT for safety.
    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        runner_a = bridge._active_runner()
        assert runner_a is not None
        # Pretend A has an open session.
        runner_a._session = MagicMock()
        sentinel_a = runner_a._session

        # Switch focus to B.
        bridge.switch_session(str(b))

        assert bridge.cwd == b.resolve()
        # A's session is UNTOUCHED — that's the fix.
        assert runner_a._session is sentinel_a, (
            "switching focus must NOT close runner A's session"
        )
        # B has its own runner with no session yet.
        runner_b = bridge._active_runner()
        assert runner_b is not None
        assert runner_b is not runner_a
        assert runner_b._session is None
    finally:
        ui_mod.SESSIONS_ROOT = real_root


def test_switch_session_returns_to_existing_runner(tmp_path: Path):
    """Re-focusing a session you've visited before returns the
    SAME runner (not a new one). Memory, model preference, and any
    open SDK client all carry over."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        runner_a_first = bridge._active_runner()

        bridge.switch_session(str(b))
        bridge.switch_session(str(a))
        runner_a_second = bridge._active_runner()

        assert runner_a_first is runner_a_second, (
            "returning to a session must reuse its existing runner"
        )
    finally:
        ui_mod.SESSIONS_ROOT = real_root


# ---------------------------------------------------------------------------
# Cold start prefix — unchanged (memory still per-cwd)
# ---------------------------------------------------------------------------

def test_cold_start_prefix_contains_prior_exchange(tmp_path: Path):
    """``build_context_prefix`` renders a block containing prior
    exchanges so Claude picks up where the researcher left off."""
    (tmp_path / ".nora").mkdir()
    log = tmp_path / ".nora" / "chat_history.jsonl"
    log.write_text(
        json.dumps({"type": "user_message", "text": "what does the gate do?"}) + "\n"
        + json.dumps({"type": "assistant_text",
                      "text": "It flags revolving-door entries."}) + "\n"
    )

    prefix = build_context_prefix(tmp_path, results=[])
    assert "Prior conversation context" in prefix
    assert "what does the gate do?" in prefix
    assert "revolving-door" in prefix
    assert "End of prior context" in prefix


def test_cold_start_brand_new_session_has_no_prefix(tmp_path: Path):
    """A session with no chat_history and no results must produce
    an empty prefix."""
    assert build_context_prefix(tmp_path, results=[]) == ""


def test_cold_start_prefix_across_multiple_sessions(tmp_path: Path):
    """History is per-cwd, not global — two different session dirs
    produce two different prefixes."""
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    (session_a / ".nora").mkdir(parents=True)
    (session_b / ".nora").mkdir(parents=True)

    (session_a / ".nora" / "chat_history.jsonl").write_text(
        json.dumps({"type": "user_message", "text": "about dataset A"}) + "\n"
    )
    (session_b / ".nora" / "chat_history.jsonl").write_text(
        json.dumps({"type": "user_message", "text": "about dataset B"}) + "\n"
    )

    prefix_a = build_context_prefix(session_a, results=[])
    prefix_b = build_context_prefix(session_b, results=[])

    assert "about dataset A" in prefix_a
    assert "about dataset B" not in prefix_a
    assert "about dataset B" in prefix_b
    assert "about dataset A" not in prefix_b


# ---------------------------------------------------------------------------
# Stop button — affects the active runner only
# ---------------------------------------------------------------------------

def test_interrupt_turn_no_running_turn(tmp_path: Path):
    """Stop with nothing in flight: surfaces a clean error rather
    than raising."""
    bridge = NoraBridge(cwd=tmp_path)

    bridge.start_loop()
    try:
        result = bridge.interrupt_turn()
    finally:
        bridge.stop_loop()

    assert result["ok"] is False
    assert "no turn in flight" in result["reason"]


# ---------------------------------------------------------------------------
# Out-of-scope for these tests (require live SDK or live Claude)
# ---------------------------------------------------------------------------
#
# Verifying that the runner's ``_run_turn`` actually streams events
# from the SDK requires a mocked client. The runner-level tests in
# ``test_concurrent_sessions.py`` cover the cross-session
# non-trampling invariant directly with a fake provider session.

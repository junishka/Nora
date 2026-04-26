"""Restart- and lifecycle-focused tests for NoraBridge memory.

The memory stack relies on a small state machine on the bridge:

- ``_needs_context_prefix`` gets flipped True whenever a fresh
  provider session is opened (first turn, session switch, interrupted
  turn, app reopen). The next turn consumes the flag by prepending a
  warm-start prefix; on cancel / error the flag is restored so the
  prefix isn't lost to a failed first turn.
- ``_close_session_blocking`` is the canonical teardown entry point
  used by session switches, cwd changes, and the Stop button; it
  must clear ``_session`` reliably so the next turn opens fresh.
- ``_set_cwd`` must tear down the session when the cwd actually
  changes, but NOT when the same cwd is re-set.
- Persisted events must carry an ISO timestamp so the turn reader
  and session_state writer can order and display them honestly.

These tests exercise those behaviors directly on the bridge,
without running a real turn (which would require a live provider
connection).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nora.chat_history import build_context_prefix
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Construction / initial state
# ---------------------------------------------------------------------------

def test_bridge_starts_with_clean_memory_state(tmp_path: Path):
    """A freshly-constructed bridge has no provider session and no
    pending context prefix — memory injection is opt-in, triggered
    only when a session is actually opened."""
    bridge = NoraBridge(cwd=tmp_path)
    assert bridge._session is None
    assert bridge._needs_context_prefix is False


# ---------------------------------------------------------------------------
# _persist_event — timestamp + type filtering
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
    # ISO 8601 with timezone — simplest sanity check is that
    # datetime can parse it round-trip.
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
    # No file written at all — persistence only happens for log-worthy
    # types.
    assert not log.exists()


def test_persist_event_preserves_caller_timestamp(tmp_path: Path):
    """If the caller already supplied a timestamp (replay, import
    from external log), we don't overwrite it — otherwise importing
    a log would rewrite all its times to 'now'."""
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


def test_record_user_message_replaces_trailing_orphan_turn(tmp_path: Path):
    """A failed send can leave a lone persisted ``user_message`` at
    the tail of chat_history. The next real send should replace that
    stale attempt rather than keep both bubbles forever."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "user_message", "text": "stuck send"})

    bridge._record_user_message("retry")

    log = tmp_path / ".nora" / "chat_history.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["text"] for row in rows if row.get("type") == "user_message"] == [
        "retry"
    ]


def test_record_user_message_keeps_completed_prior_turn(tmp_path: Path):
    """Only orphaned tail user turns are disposable. A completed turn
    with an assistant reply must stay in the log when the next user
    message is recorded."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "user_message", "text": "first"})
    bridge._persist_event({"type": "assistant_text", "text": "reply"})

    bridge._record_user_message("second")

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
# _close_session_blocking — handles all corner cases without raising
# ---------------------------------------------------------------------------

def test_close_client_blocking_when_no_client(tmp_path: Path):
    """No session to close: no-op, no error. Safe to call defensively
    from Stop / switch / teardown paths."""
    bridge = NoraBridge(cwd=tmp_path)
    assert bridge._session is None
    bridge._close_session_blocking()  # must not raise
    assert bridge._session is None


def test_close_client_blocking_without_loop(tmp_path: Path):
    """Session is set but worker loop isn't running: we can't run
    the async close, but we still need to drop the reference so the
    next turn opens fresh. Otherwise Stop after a startup failure
    would leak the half-initialized session."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._session = MagicMock()  # pretend there's a session
    bridge._loop = None

    bridge._close_session_blocking()
    assert bridge._session is None


# ---------------------------------------------------------------------------
# _set_cwd — session switch should tear down session, same-cwd should not
# ---------------------------------------------------------------------------

def test_set_cwd_closes_client_on_cwd_change(tmp_path: Path):
    """Switching to a different session dir must close the provider
    session so the next turn starts a fresh conversation against
    the new cwd. Without this, conversation state can leak across
    sessions."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    bridge = NoraBridge(cwd=tmp_path / "a")
    # Simulate a live session. _close_session_blocking handles the
    # no-loop case by just clearing the reference, so we don't need
    # a working event loop here.
    bridge._session = MagicMock()

    bridge._set_cwd(tmp_path / "b")

    assert bridge.cwd == tmp_path / "b"
    assert bridge._session is None, "session must be closed on cwd change"


def test_set_cwd_same_path_keeps_client(tmp_path: Path):
    """Re-setting the same cwd (e.g. after a harmless state refresh)
    must NOT close the session — otherwise every idempotent _set_cwd
    call would waste a conversation."""
    (tmp_path / "a").mkdir()

    bridge = NoraBridge(cwd=tmp_path / "a")
    sentinel = MagicMock()
    bridge._session = sentinel

    bridge._set_cwd(tmp_path / "a")

    assert bridge._session is sentinel, (
        "same-cwd _set_cwd must not tear down the session"
    )


# ---------------------------------------------------------------------------
# Cold start (app reopen) — prefix contains what we expect
# ---------------------------------------------------------------------------

def test_cold_start_prefix_contains_prior_exchange(tmp_path: Path):
    """Simulate an app reopen: chat_history.jsonl exists on disk
    from a previous run. ``build_context_prefix`` must render a
    block that contains the prior user + assistant exchange so
    Claude picks up where the researcher left off."""
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
    an empty prefix — we don't want to inject a phantom header on
    the very first turn of a fresh session."""
    assert build_context_prefix(tmp_path, results=[]) == ""


def test_cold_start_prefix_across_multiple_sessions(tmp_path: Path):
    """Two different session dirs produce two different prefixes —
    history is per-cwd, not global. This guards against a subtle
    bug where a cached reader could serve the wrong session's log
    after a switch."""
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
# Stop button — interrupt_turn tears down the client
# ---------------------------------------------------------------------------

def test_interrupt_turn_no_running_turn(tmp_path: Path):
    """Stop with nothing in flight: surfaces a clean error rather
    than raising. The UI can then re-enable the composer without
    special-casing."""
    bridge = NoraBridge(cwd=tmp_path)

    # Need a running loop for the interrupt_turn code path that
    # uses call_soon_threadsafe. We spin up a tiny one for the test.
    bridge.start_loop()
    try:
        result = bridge.interrupt_turn()
    finally:
        bridge.stop_loop()

    assert result["ok"] is False
    assert "no turn in flight" in result["reason"]


# ---------------------------------------------------------------------------
# Flag state machine — carrying a prefix across an error
# ---------------------------------------------------------------------------

def test_needs_context_prefix_survives_restart_of_bridge(tmp_path: Path):
    """A fresh NoraBridge starts with the flag False. That's
    correct — the flag means 'the NEXT turn must inject a prefix'.
    On cold boot, the flag flips True inside ``_ensure_session`` when
    it opens the first session of the process, not at construction.

    This test documents the invariant so a future refactor that
    'helpfully' pre-sets the flag to True in __init__ (and thereby
    double-injects the prefix on the first-ever turn of a brand-new
    session) gets caught."""
    bridge1 = NoraBridge(cwd=tmp_path)
    assert bridge1._needs_context_prefix is False

    # Simulate an app reopen: a second bridge against the same cwd.
    bridge2 = NoraBridge(cwd=tmp_path)
    assert bridge2._needs_context_prefix is False


# ---------------------------------------------------------------------------
# Out-of-scope for these tests (require live SDK or live Claude)
# ---------------------------------------------------------------------------
#
# A few reviewer-suggested paths can't be exercised at the unit
# level because they require either a live Claude Agent SDK client
# or the frontier model itself. They're listed here so a reader
# knows they're intentionally left to manual / smoke testing:
#
# 1. First-turn prefix consumption. The flag transition
#    False → set True by ``_ensure_client`` → consumed by
#    ``_run_turn`` → reset to False is simple, but verifying the
#    full loop needs a mocked async client context manager.
#    Smoke-test: restart Nora in a session with prior history,
#    send a message, confirm Claude references earlier turns.
#
# 2. Carried-prefix restoration on cancel / error. The
#    ``_run_turn`` exception handlers set ``_needs_context_prefix``
#    back to True when a carried prefix was in flight. Unit-testing
#    requires a mocked client that raises from ``query`` or
#    ``receive_response``. Smoke-test: hit Stop on the first turn
#    after a reopen, then send a new message, confirm the prefix
#    still lands.
#
# 3. "What were we doing?" semantic response. Whether Claude
#    actually uses the injected prefix correctly (summarizes from
#    the enclosed turns, calls expand_result on the recent-results
#    listing, doesn't re-run old analyses) is a model-behavior
#    question, not a code-behavior one. Smoke-test: ask the
#    question after a reopen and eyeball the reply.
#
# 4. Session-state write after turn_done. The
#    ``write_session_state`` hook fires inside the successful-turn
#    branch of ``_run_turn``. Covered by the session_state tests
#    for the writer itself; the hook wiring is verified by
#    inspection and a smoke test (send a message, confirm
#    ``.nora/session_state.json`` timestamps refresh).

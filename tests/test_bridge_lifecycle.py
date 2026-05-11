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
# Staging cleanup — partial-session orphans
# ---------------------------------------------------------------------------
#
# The fresh session dir is created BEFORE files are copied. If a copy
# raises mid-way, an unguarded path leaves the dir (with whatever was
# already copied) under ~/.nora-sessions/, where the global session
# listing later surfaces it as if it were a real session. The all-
# or-nothing contract requires tearing the dir down on any failure.


def test_stage_session_removes_dir_on_copy_failure(
    tmp_path: Path, monkeypatch,
):
    """A mid-staging OSError must remove the freshly-created session
    dir AND any files already copied into it."""
    import shutil as _shutil
    import nora.ui as ui_mod

    bridge = NoraBridge.__new__(NoraBridge)
    bridge._set_cwd = lambda p: {"ok": True, "state": "ready"}

    session_dir = tmp_path / "session_under_test"
    session_dir.mkdir()
    monkeypatch.setattr(ui_mod, "_new_session_dir", lambda: session_dir)

    real_copy = _shutil.copy2
    call_count = {"n": 0}

    def flaky_copy(src, dst, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk full")
        return real_copy(src, dst, **kw)

    monkeypatch.setattr(ui_mod.shutil, "copy2", flaky_copy)

    src_a = tmp_path / "a.csv"
    src_a.write_text("a")
    src_b = tmp_path / "b.csv"
    src_b.write_text("b")

    result = bridge._stage_session([str(src_a), str(src_b)])
    assert result["ok"] is False
    assert "copy failed" in result["reason"]
    # Cleanup invariant: the dir is gone, including the partial
    # first-file copy.
    assert not session_dir.exists()


def test_stage_session_from_blobs_removes_dir_on_write_failure(
    tmp_path: Path, monkeypatch,
):
    """Same all-or-nothing contract for the drag-drop path."""
    import nora.ui as ui_mod

    bridge = NoraBridge.__new__(NoraBridge)
    bridge._set_cwd = lambda p: {"ok": True, "state": "ready"}

    session_dir = tmp_path / "session_blobs"
    session_dir.mkdir()
    monkeypatch.setattr(ui_mod, "_new_session_dir", lambda: session_dir)

    # Patch Path.write_bytes to fail on the second call.
    real_write_bytes = Path.write_bytes
    call_count = {"n": 0}

    def flaky_write(self, data):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk full")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky_write)

    result = bridge._stage_session_from_blobs([
        ("a.csv", b"a"),
        ("b.csv", b"b"),
    ])
    assert result["ok"] is False
    assert not session_dir.exists()


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
    """Lifecycle events (``ready``, ``auth_failure``, …) must not
    pollute the chat log — replay would reconstruct phantom turns
    or surface stale auth banners."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({"type": "ready"})
    bridge._persist_event({"type": "auth_failure", "reason": "stale token"})

    log = tmp_path / ".nora" / "chat_history.jsonl"
    assert not log.exists()


def test_persist_event_keeps_turn_done_for_diagnostics(tmp_path: Path):
    """``turn_done`` carries the per-turn token usage (input, output,
    cache_read, cache_creation, cost) and is persisted so post-hoc
    inspection of cache hit rate / cost trends is possible. The
    transcript readers (``read_turns``, ``replayEvent``) ignore
    unknown event types, so persisting it does not introduce phantom
    turns."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._persist_event({
        "type": "turn_done",
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    })

    log = tmp_path / ".nora" / "chat_history.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["type"] == "turn_done"
    assert rec["input_tokens"] == 100


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

def test_runner_close_clears_pending_close_flag(tmp_path: Path) -> None:
    """``mark_close_after_turn`` arms the flag; ``close`` is the
    chokepoint that clears it. A future direct ``close`` (e.g., a
    model swap that tears down the session) MUST drop the flag too,
    otherwise the next time the runner is reopened and bound to a
    NEW session, ``run_turn``'s finally block would still see
    ``_pending_close = True`` and close the freshly opened session
    after one turn — a memory-leak-of-state bug.
    """
    runner = SessionRunner(
        cwd=tmp_path, provider="openai", model="gpt-x",
    )
    # Pretend a session exists so ``mark_close_after_turn`` arms.
    runner._session = MagicMock()
    runner.mark_close_after_turn()
    assert runner._pending_close is True
    asyncio.run(runner.close())
    assert runner._pending_close is False, (
        "close must clear pending_close so a future reopen doesn't "
        "inherit a stale flag"
    )


def test_runner_mark_close_noop_without_session(tmp_path: Path) -> None:
    """``mark_close_after_turn`` on a runner that never opened a
    session is a no-op: there's nothing to close, and arming the
    flag would cause the FIRST turn after a future open to
    immediately close — exactly wrong for the typical "user added
    a key, deleted it, then added it again" flow.
    """
    runner = SessionRunner(
        cwd=tmp_path, provider="openai", model="gpt-x",
    )
    assert runner._session is None
    runner.mark_close_after_turn()
    assert runner._pending_close is False


def test_turn_error_carries_context_reset_flag() -> None:
    """SDC + continuity closure: ``TurnError.context_reset`` defaults
    to False (preserving the prior single-arg call sites) and gets
    set to True only when the provider's server-side memory has
    been lost. The OpenAI provider sets it on
    ``previous_response_id`` expiry. The runner reads it in the
    failure-restoration branch and re-arms ``needs_context_prefix``
    so the next turn re-injects the warm-start context — without
    this flag, an established session that hits chain expiry would
    silently start fresh with no recoverable context.
    """
    from nora.provider.base import TurnError
    plain = TurnError(message="generic failure")
    assert plain.context_reset is False
    reset = TurnError(
        message="chain expired", context_reset=True,
    )
    assert reset.context_reset is True


def test_delete_credential_closes_idle_and_marks_busy_runner_for_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDC + credential-hygiene closure: deleting an API key while a
    runner of that provider is mid-turn must NOT leak the deleted
    credential into subsequent sends.

    Mechanism. Both provider SDKs (``AsyncOpenAI``,
    ``AsyncAnthropic``) capture ``api_key`` at client construction
    and reuse it until the client is closed. If
    ``delete_credential`` skipped busy runners entirely, the cached
    client would happily keep authenticating with the now-deleted
    key for every subsequent send in the same process.

    The fix: idle runners are closed immediately (as before), but
    busy runners are marked for close-after-turn via
    ``mark_close_after_turn``. The next turn's finally block honours
    the flag and closes the session, evicting the cached client.
    The send after that opens a fresh session, which fails cleanly
    at ``_resolve_api_key`` (no key left in keychain).

    This test exercises the bridge-side dispatch — that idle and
    busy runners are routed to the right path. The runner-side
    contract (the flag actually closes after the turn finishes) is
    pinned by ``test_runner_lifecycle.test_pending_close_runs_after_turn``.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        idle_runner = bridge._active_runner()
        assert idle_runner is not None
        idle_runner.provider = "openai"
        idle_runner._session = MagicMock()
        # Switch focus to B to lazy-create a second runner. Both
        # runners are on the OpenAI provider for this test.
        bridge.switch_session(str(b))
        busy_runner = bridge._active_runner()
        assert busy_runner is not None
        assert busy_runner is not idle_runner
        busy_runner.provider = "openai"
        busy_runner._session = MagicMock()
        # busy_runner is_busy() returns True; idle_runner stays idle.
        monkeypatch.setattr(busy_runner, "is_busy", lambda: True)
        monkeypatch.setattr(idle_runner, "is_busy", lambda: False)

        # Patch the auth call so we don't touch the real keychain;
        # match the success shape ``delete_credential`` expects.
        import nora.auth as _auth
        monkeypatch.setattr(
            _auth, "delete_credential",
            lambda provider: {"ok": True, "provider": provider},
        )

        # Track close + mark_close calls on each runner.
        idle_close_calls: list[bool] = []
        busy_close_calls: list[bool] = []
        busy_mark_calls: list[bool] = []

        async def _idle_close():
            idle_close_calls.append(True)
        async def _busy_close():
            busy_close_calls.append(True)
        monkeypatch.setattr(idle_runner, "close", _idle_close)
        monkeypatch.setattr(busy_runner, "close", _busy_close)
        monkeypatch.setattr(
            busy_runner, "mark_close_after_turn",
            lambda: busy_mark_calls.append(True),
        )
        # Also stub _run_on_loop so the test doesn't need an event
        # loop. The schedulers it would invoke are already covered
        # by their own tests; here we only care that the bridge
        # picked the right path per runner.
        monkeypatch.setattr(
            bridge, "_run_on_loop", lambda coro: asyncio.run(coro),
        )

        result = bridge.delete_credential("openai")
        assert result["ok"] is True

        # Idle runner: close ran.
        assert idle_close_calls == [True]
        # Busy runner: NOT closed mid-turn, but marked for close-
        # after-turn so the in-flight stream finishes naturally and
        # the cached client gets evicted before the next send.
        assert busy_close_calls == []
        assert busy_mark_calls == [True]
    finally:
        ui_mod.SESSIONS_ROOT = real_root


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


def test_switch_session_rejects_root_and_nested_paths(tmp_path: Path):
    """``switch_session`` must accept only direct children of
    SESSIONS_ROOT. Accepting the root itself would point cwd at the
    directory that contains every session, breaking the cross-session
    isolation gate (every other session becomes a child of cwd).
    Accepting a nested path inside a session would spawn a runner
    whose cwd doesn't see the session's ``.nora/`` state.
    """
    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        a = tmp_path / "a"
        a.mkdir()
        nested = a / "subdir"
        nested.mkdir()

        bridge = NoraBridge(cwd=a)

        # The sessions root itself is NOT a session.
        result = bridge.switch_session(str(tmp_path))
        assert not result["ok"]
        assert "session directory" in result["reason"]
        assert bridge.cwd == a.resolve(), (
            "rejected switch must not change focus"
        )

        # A nested directory inside a session is also NOT a session.
        result = bridge.switch_session(str(nested))
        assert not result["ok"]
        assert "session directory" in result["reason"]
        assert bridge.cwd == a.resolve()

        # Sanity: a real direct child still works.
        b = tmp_path / "b"
        b.mkdir()
        result = bridge.switch_session(str(b))
        assert result["ok"], result
        assert bridge.cwd == b.resolve()
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
    assert "Session state at resume" in prefix
    assert "what does the gate do?" in prefix
    assert "revolving-door" in prefix
    assert "End of session state" in prefix


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


def test_interrupt_pending_turn_marks_cancelled_before_run(tmp_path: Path):
    """The fast-Stop race: if the researcher hits Stop in the gap
    between ``send_message`` returning and ``run_turn`` actually
    starting on the worker loop, the runner must mark the pending id
    cancelled so the eventual ``run_turn`` call bails before opening
    a session or hitting the API.

    Pre-fix: ``interrupt_turn`` saw ``is_busy() is False`` (no
    current task yet) and returned "no turn in flight" while the
    queued coroutine went on to execute the cancelled turn.
    """
    bridge = NoraBridge(cwd=tmp_path)
    bridge.start_loop()
    try:
        runner = bridge._active_runner()
        assert runner is not None

        # Simulate what ``_send_to_active`` does synchronously: register
        # the pending id BEFORE the coroutine has been picked up by the
        # worker loop.
        runner.register_pending_turn("t-pending")

        # Bridge sees a turn in flight via the pending list.
        assert runner.is_busy() is True

        res = bridge.interrupt_turn()
        assert res["ok"] is True
        assert res["turn_id"] == "t-pending"
        # The id is now in ``_cancelled_turn_ids``; ``run_turn`` checks
        # this on entry and bails before doing any LLM work.
        assert runner.is_turn_cancelled("t-pending") is True
        # Pending list drained so a second Stop doesn't re-cancel the
        # same id.
        assert "t-pending" not in runner._pending_turn_ids
    finally:
        bridge.stop_loop()


def test_cancel_turn_does_not_cancel_running_task_when_pending_targeted(
    tmp_path: Path,
):
    """A pending turn cancellation must NOT cancel the already-
    running turn's asyncio task. Without the equality check in
    ``cancel_turn``, falling back to the latest pending id would
    still call ``task.cancel`` on whatever task happened to be in
    ``_current_turn_task`` — the previous, still-running turn.
    """
    bridge = NoraBridge(cwd=tmp_path)
    bridge.start_loop()
    try:
        runner = bridge._active_runner()
        assert runner is not None

        running_cancelled = False

        class _FakeTask:
            def done(self) -> bool: return False
            def cancel(self) -> None:
                nonlocal running_cancelled
                running_cancelled = True
            def get_loop(self):
                return bridge._loop

        # Plant a fake "currently running" task A and pend a separate
        # turn B.
        runner._current_turn_task = _FakeTask()  # type: ignore[assignment]
        runner._current_turn_id = "t-running"
        runner.register_pending_turn("t-pending")

        # Cancel without an explicit id. Resolution order is:
        # _current_turn_id → 't-running'. (cancel_turn falls back to
        # pending only when nothing is current.) The running task
        # SHOULD get cancelled in this case.
        result = runner.cancel_turn()
        assert result == "t-running"
        # Cancellation hops via ``call_soon_threadsafe``; let it land.
        import time
        time.sleep(0.05)
        assert running_cancelled is True

        # Now the pending one: explicit id, no current task left.
        running_cancelled = False
        runner._current_turn_task = None
        runner._current_turn_id = None
        # Re-register since cancel_turn drained it.
        runner.register_pending_turn("t-pending")
        # Plant ANOTHER running task to make sure it ISN'T cancelled
        # when we target the pending id.
        other_cancelled = False

        class _OtherTask:
            def done(self) -> bool: return False
            def cancel(self) -> None:
                nonlocal other_cancelled
                other_cancelled = True
            def get_loop(self):
                return bridge._loop

        runner._current_turn_task = _OtherTask()  # type: ignore[assignment]
        runner._current_turn_id = "t-other-running"

        result = runner.cancel_turn("t-pending")
        assert result == "t-pending"
        time.sleep(0.05)
        # The other task is NOT cancelled because its id doesn't
        # match the cancellation target.
        assert other_cancelled is False
    finally:
        # Clear before stop_loop awaits runner.close().
        for r in bridge._runners.values():
            r._current_turn_task = None
            r._current_turn_id = None
        bridge.stop_loop()


# ---------------------------------------------------------------------------
# send_message_to_session — explicit-target send for the queue-flush path
# ---------------------------------------------------------------------------

def test_send_message_to_session_routes_to_target_not_focus(tmp_path: Path):
    """``fireQueuedMessage`` calls the targeted variant after a
    background turn finishes. If session A queued a follow-up and
    the user has since switched the focus to session B, the queued
    send MUST land on A's runner, not B's. Pre-fix: the bridge had
    no targeted variant; ``send_message`` always routed to
    ``self.cwd`` (the focused session), so A's queued message would
    persist and execute against B's working directory.
    """
    a = (tmp_path / "session-a").resolve()
    b = (tmp_path / "session-b").resolve()
    a.mkdir()
    b.mkdir()

    bridge = NoraBridge(cwd=a)
    bridge.start_loop()
    try:
        # Lazy-create both runners so the targeted send has someone
        # to route to.
        runner_a = bridge._ensure_runner_for_cwd(a)
        runner_b = bridge._ensure_runner_for_cwd(b)
        assert runner_a is not runner_b
        # Simulate the user switching focus to B without going through
        # ``switch_session`` (which enforces SESSIONS_ROOT containment
        # — irrelevant to what this test pins).
        bridge.cwd = b
        assert bridge.cwd == b

        # Patch run_turn on both runners to a no-op coroutine that
        # records which runner got the call. We don't want the test
        # to actually open a provider session.
        called_on: list[str] = []

        async def _spy_a(*args, **kwargs):
            called_on.append("a")

        async def _spy_b(*args, **kwargs):
            called_on.append("b")

        runner_a.run_turn = _spy_a  # type: ignore[assignment]
        runner_b.run_turn = _spy_b  # type: ignore[assignment]

        turn_id = bridge.send_message_to_session(str(a), "hi from A's queue")
        assert turn_id is not None, (
            "targeted send should schedule even when focus is on B"
        )
        # Give the worker loop a beat to dispatch.
        import time
        time.sleep(0.05)
        assert called_on == ["a"], (
            f"queued message must run on session A, got {called_on}"
        )
    finally:
        bridge.stop_loop()


def test_send_message_to_session_rejects_unknown_cwd(tmp_path: Path):
    """A targeted send must NOT lazy-create runners for arbitrary
    caller-supplied paths — that would let a stale queue resurrect
    a session the researcher has since deleted. Unknown targets
    should produce a turn_error event and return None instead.
    """
    bridge = NoraBridge(cwd=tmp_path)
    bridge.start_loop()
    try:
        events: list[dict] = []
        bridge._dispatch_event = events.append  # type: ignore[assignment]

        result = bridge.send_message_to_session(
            str(tmp_path / "does-not-exist"), "hi",
        )
        assert result is None
        assert any(
            e.get("type") == "turn_error"
            and "no longer open" in (e.get("message") or "")
            for e in events
        ), f"expected turn_error event, got {events}"
    finally:
        bridge.stop_loop()


# ---------------------------------------------------------------------------
# delete_session on the currently-active session
# ---------------------------------------------------------------------------

def test_delete_session_on_active_clears_cwd_and_signals_landing(tmp_path: Path):
    """Deleting the focused session must:
       1. rmtree the session directory
       2. close the active runner and drop it from ``_runners``
       3. set ``self.cwd = None`` so subsequent bridge calls don't
          read from a vanished path
       4. return ``was_active=True`` so the page knows to navigate
          to the landing screen.

    Without (3) the bridge would keep handing back a stale Path to
    ``policy_summary``, ``get_chat_history``, etc. — every subsequent
    call would crash on a missing directory."""
    import nora.ui as ui_mod
    from nora.ui import NoraBridge

    session = tmp_path / "active_session"
    session.mkdir()

    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=session)
        # Sanity: bridge is focused on this session.
        assert bridge.cwd == session.resolve()
        assert str(session.resolve()) in bridge._runners

        res = bridge.delete_session(str(session))

        assert res["ok"] is True
        assert res.get("was_active") is True
        assert res["path"] == str(session.resolve())
        assert not session.exists(), "the session directory must be gone"
        assert bridge.cwd is None, "active cwd must be cleared after delete"
        assert str(session.resolve()) not in bridge._runners, (
            "the runner must be dropped along with the directory"
        )
    finally:
        ui_mod.SESSIONS_ROOT = real_root


def test_delete_session_on_inactive_keeps_focus(tmp_path: Path):
    """Deleting a non-focused session does NOT clear the bridge's
    active cwd — only the targeted runner is removed. ``was_active``
    is False so the page stays on the current chat."""
    import nora.ui as ui_mod
    from nora.ui import NoraBridge

    active = tmp_path / "active"
    other = tmp_path / "other"
    active.mkdir()
    other.mkdir()

    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=active)
        # Touch ``other`` enough that it has a runner entry so we can
        # confirm only that one gets popped (not the active one).
        bridge._ensure_runner_for_cwd(other)
        assert str(other.resolve()) in bridge._runners
        assert str(active.resolve()) in bridge._runners

        res = bridge.delete_session(str(other))

        assert res["ok"] is True
        assert res.get("was_active") is False
        assert not other.exists()
        assert active.exists(), "the focused session must be untouched"
        assert bridge.cwd == active.resolve()
        assert str(active.resolve()) in bridge._runners
        assert str(other.resolve()) not in bridge._runners
    finally:
        ui_mod.SESSIONS_ROOT = real_root


# ---------------------------------------------------------------------------
# Out-of-scope for these tests (require live SDK or live Claude)
# ---------------------------------------------------------------------------
#
# Verifying that the runner's ``_run_turn`` actually streams events
# from the SDK requires a mocked client. The runner-level tests in
# ``test_concurrent_sessions.py`` cover the cross-session
# non-trampling invariant directly with a fake provider session.

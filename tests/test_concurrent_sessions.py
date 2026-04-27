"""Concurrent-session execution invariants.

This is the regression home for the headline fix: switching the
visible session in the sidebar must NOT kill the in-flight turn in
the previous session, and tool execution in two sessions running
simultaneously must NOT trample each other's working directory.

The trampling vector being closed is ``nora.config.get_cwd``: tool
handlers (``get_schema``, ``submit_script``, …) used to read a
process-global cwd that the bridge mutated on every focus change. A
fast switch could land a tool call from session A's still-streaming
turn against session B's cwd, blowing up sandbox path resolution and
results-store routing.

The fix is two-layered:

1. ``SessionRunner`` carries its own cwd, lock, session, and turn
   task. Each runner runs concurrently as a sister asyncio task.
2. ``run_turn`` enters ``nora.config.use_cwd(self.cwd)``, which
   sets a ContextVar bound to the runner's asyncio task. Sister
   tasks see their own cwds — ``get_cwd()`` is now task-local.

These tests pin both layers. The first asserts ContextVar
isolation directly; the second drives two runners against fake
sessions that record what ``get_cwd()`` returned during their tool
phase.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from nora.config import get_cwd, set_cwd, use_cwd
from nora.runner import SessionRunner
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# ContextVar isolation
# ---------------------------------------------------------------------------

def test_use_cwd_overrides_get_cwd_in_scope(tmp_path: Path) -> None:
    """Inside ``use_cwd(p)``, ``get_cwd()`` returns ``p``. Outside,
    it falls back to the process default."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    set_cwd(a)
    assert get_cwd() == a.resolve()

    with use_cwd(b):
        assert get_cwd() == b.resolve()

    # Restored on exit.
    assert get_cwd() == a.resolve()


def test_use_cwd_isolates_concurrent_asyncio_tasks(tmp_path: Path) -> None:
    """Two concurrent asyncio tasks running ``use_cwd`` against
    different paths see their own bindings — neither task observes
    the other's cwd. This is what makes per-runner tool execution
    safe under simultaneous turns."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    observed_a: list[Path] = []
    observed_b: list[Path] = []

    async def runner(cwd: Path, observed: list[Path]) -> None:
        with use_cwd(cwd):
            # Yield several times so the two tasks interleave.
            for _ in range(5):
                observed.append(get_cwd())
                await asyncio.sleep(0)

    async def driver() -> None:
        await asyncio.gather(
            runner(a, observed_a),
            runner(b, observed_b),
        )

    asyncio.run(driver())

    assert observed_a == [a.resolve()] * 5, (
        "task A must see only A's cwd despite interleaving with B"
    )
    assert observed_b == [b.resolve()] * 5, (
        "task B must see only B's cwd despite interleaving with A"
    )


# ---------------------------------------------------------------------------
# Runner-level cwd binding inside run_turn
# ---------------------------------------------------------------------------

class _CwdProbingSession:
    """Fake provider session that records what ``get_cwd()`` returns
    while ``send()`` is iterating. Stands in for a real SDK client
    when we want to probe what the tool handlers would have seen."""

    def __init__(self) -> None:
        from nora.provider import AssistantText, TurnDone

        self._AssistantText = AssistantText
        self._TurnDone = TurnDone
        self.observed_cwds: list[Path] = []

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, prompt: str, images: Any = None) -> AsyncIterator[Any]:
        # Each yield interleaves with sister tasks. If the runner
        # forgot to bind its cwd via use_cwd, sister runs would
        # mutate get_cwd() out from under us between yields.
        for _ in range(3):
            self.observed_cwds.append(get_cwd())
            await asyncio.sleep(0)
            yield self._AssistantText(text="ping")
        yield self._TurnDone()


def test_two_runners_observe_their_own_cwd_under_concurrent_send(
    tmp_path: Path,
) -> None:
    """The headline guarantee: two runners, two cwds, sending in
    parallel — each runner's session sees ITS cwd via ``get_cwd``
    throughout, regardless of how their async iterations interleave.

    Without ``use_cwd`` in ``run_turn`` (the trampling fix), this
    test is racy: each task would see whichever cwd was last set on
    the process-global, which is determined by interleaving order.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    # Set a clearly-wrong default so the test fails loudly if either
    # runner accidentally falls through to it.
    other = tmp_path / "wrong"
    other.mkdir()
    set_cwd(other)

    runner_a = SessionRunner(cwd=a, provider="anthropic", model="claude-sonnet-4-6[1m]")
    runner_b = SessionRunner(cwd=b, provider="anthropic", model="claude-sonnet-4-6[1m]")

    session_a = _CwdProbingSession()
    session_b = _CwdProbingSession()
    runner_a._session = session_a
    runner_b._session = session_b

    async def drive(runner: SessionRunner) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        await runner.run_turn(
            "go",
            images=None,
            on_event=events.append,
            build_context_prefix=lambda cwd: "",
            build_script_prefix=lambda atts, cwd: "",
        )
        return events

    async def both() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return await asyncio.gather(drive(runner_a), drive(runner_b))

    events_a, events_b = asyncio.run(both())

    # Each session observed ONLY its runner's cwd, despite the two
    # turns running concurrently and yielding back and forth.
    assert session_a.observed_cwds == [a.resolve()] * 3, (
        f"runner A's session saw {session_a.observed_cwds} — expected "
        f"{[a.resolve()] * 3}. The cwd ContextVar didn't isolate "
        f"per-task; tool handlers in concurrent turns would trample."
    )
    assert session_b.observed_cwds == [b.resolve()] * 3, (
        f"runner B's session saw {session_b.observed_cwds}"
    )

    # Both runs produced their events — neither was killed.
    assert any(e.get("type") == "assistant_text" for e in events_a)
    assert any(e.get("type") == "assistant_text" for e in events_b)
    # And every event is stamped with its runner's cwd so the bridge
    # can route persistence correctly.
    assert all(e.get("session_cwd") == str(a.resolve()) for e in events_a)
    assert all(e.get("session_cwd") == str(b.resolve()) for e in events_b)


# ---------------------------------------------------------------------------
# Bridge-level: switching focus does NOT close the previous runner
# ---------------------------------------------------------------------------

def test_switch_session_keeps_previous_runner_session_alive(tmp_path: Path) -> None:
    """A focus change must leave the previous runner's SDK client
    untouched — the in-flight turn keeps streaming. (The bug was a
    ``_close_session_blocking`` call inside ``switch_session`` that
    tore the SDK client down mid-stream, which raised inside the
    receive loop and surfaced as a ``turn_error`` "fail" message in
    the new session.)"""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        runner_a = bridge._active_runner()
        assert runner_a is not None
        sentinel = MagicMock()
        runner_a._session = sentinel

        bridge.switch_session(str(b))

        # A's session is still alive. The bug was that this assertion
        # would have failed because switch_session called close on it.
        assert runner_a._session is sentinel
        # And A's runner is still in the bridge's dict — the runner
        # outlives the focus switch.
        assert str(a.resolve()) in bridge._runners
        assert str(b.resolve()) in bridge._runners
    finally:
        ui_mod.SESSIONS_ROOT = real_root


def test_event_persists_to_event_session_cwd_not_focus(tmp_path: Path) -> None:
    """A streaming event from runner A must persist to A's
    ``chat_history.jsonl`` even when the bridge has switched focus
    to B. Otherwise mid-flight tool results from background sessions
    would land in whichever transcript happens to be on screen — a
    cross-session leak."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    bridge = NoraBridge(cwd=b)  # focused on B
    bridge._persist_event({
        "type": "assistant_text",
        "text": "from A's mid-flight turn",
        "session_cwd": str(a),
    })

    a_log = a / ".nora" / "chat_history.jsonl"
    b_log = b / ".nora" / "chat_history.jsonl"
    assert a_log.exists(), "A's log must receive A's event"
    assert not b_log.exists(), "B's log must not be polluted"
    rec = json.loads(a_log.read_text().splitlines()[0])
    assert rec["text"] == "from A's mid-flight turn"


def test_interrupt_only_cancels_active_runner(tmp_path: Path) -> None:
    """Stop button cancels ONLY the focused runner's turn. Other
    runners' in-flight turns continue."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    import nora.ui as ui_mod
    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        bridge.start_loop()
        try:
            # Pretend B has an in-flight turn (mocked task that
            # doesn't actually do anything).
            runner_a = bridge._active_runner()
            assert runner_a is not None
            bridge.switch_session(str(b))
            runner_b = bridge._active_runner()
            assert runner_b is not None
            assert runner_a is not runner_b

            # Plant a fake task on each.
            cancelled_a = False
            cancelled_b = False

            class _FakeTask:
                def __init__(self, on_cancel):
                    self._on_cancel = on_cancel
                def done(self) -> bool: return False
                def cancel(self) -> None: self._on_cancel()

            def cancel_a():
                nonlocal cancelled_a
                cancelled_a = True

            def cancel_b():
                nonlocal cancelled_b
                cancelled_b = True

            runner_a._current_turn_task = _FakeTask(cancel_a)  # type: ignore[assignment]
            runner_b._current_turn_task = _FakeTask(cancel_b)  # type: ignore[assignment]

            # Bridge is focused on B → interrupt should hit B only.
            res = bridge.interrupt_turn()
            # call_soon_threadsafe runs on the worker loop, give it a
            # tiny moment to land.
            import time
            time.sleep(0.05)

            assert res["ok"] is True
            assert cancelled_b is True
            assert cancelled_a is False, (
                "interrupt_turn must NOT cancel the background runner"
            )
        finally:
            # Clear fake tasks before stop_loop tries to await runner.close().
            for r in bridge._runners.values():
                r._current_turn_task = None
            bridge.stop_loop()
    finally:
        ui_mod.SESSIONS_ROOT = real_root

"""Regression tests for the four correctness bugs flagged in review.

Each test pins a behaviour that broke in production at least once:

1. ``ui_ready()`` returns a payload with ``state == 'ready'`` when
   the bridge already has a cwd. The frontend's startup branch
   keys off ``state`` and was bouncing researchers to the landing
   screen even when chat was ready to open.

2. ``NoraBridge(cwd=...)`` (the launcher path that opens straight
   into chat) restores the per-session model preference. Without
   this, a session whose state file says ``active_model =
   "claude-opus-4-7[1m]"`` came up on the default Sonnet.

3. ``unstage_attachment`` removes a staged script from
   ``_pending_script_attachments`` so the chip × in the JS UI is
   actually backed by a Python-side removal — not a privacy
   mismatch where the chip vanishes but the prompt-prefix still
   includes the script.

4. The bridge surfaces datasets that arrive AFTER session open in
   the next turn's prompt prefix. Without this, dropping a
   ``panel.parquet`` ten messages in left the model unaware of
   it until the session was reopened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nora.session_state import write_session_state
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Fix 1 — ui_ready 'ready' branch carries the state key
# ---------------------------------------------------------------------------

def test_ui_ready_with_cwd_returns_state_ready(tmp_path: Path) -> None:
    """``nora-ui /path/to/session`` constructs the bridge with a cwd
    set. The first ``ui_ready`` call must return ``state ==
    'ready'`` so the JS startup branch routes to chat instead of
    falling through to ``showLanding``."""
    bridge = NoraBridge(cwd=tmp_path)
    payload = bridge.ui_ready()
    assert payload.get("state") == "ready", (
        f"expected state='ready', got {payload!r}"
    )


def test_ui_ready_without_cwd_signals_needs_session(tmp_path: Path) -> None:
    """Counterpart: when there's no cwd, the state must be
    ``needs_session`` (or ``needs_auth`` if no provider is set up)
    — never silently fall through to a missing key. Both branches
    must always populate ``state``."""
    bridge = NoraBridge(cwd=None)
    payload = bridge.ui_ready()
    assert payload.get("state") in ("needs_session", "needs_auth"), (
        f"expected a non-ready state, got {payload!r}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — constructor path runs per-session model restore
# ---------------------------------------------------------------------------

def test_constructor_with_cwd_restores_recorded_model(
    tmp_path: Path,
) -> None:
    """A session whose ``.nora/session_state.json`` records
    ``active_model = "claude-opus-4-7[1m]"`` must come up on Opus
    when the launcher passes the cwd directly. Skipping the
    restore step (the previous behaviour) silently dropped
    researchers onto the catalog default."""
    session_dir = tmp_path / "saved-on-opus"
    session_dir.mkdir()
    write_session_state(session_dir, model="claude-opus-4-7[1m]")

    bridge = NoraBridge(cwd=session_dir)

    assert bridge._model == "claude-opus-4-7[1m]"
    assert bridge._provider == "anthropic"


def test_constructor_without_cwd_keeps_default_model(
    tmp_path: Path,
) -> None:
    """No cwd = no session state to restore from = stay on the
    catalog default. Documents the boundary so a future
    refactor that always-runs-restore doesn't accidentally read
    something stale."""
    bridge = NoraBridge(cwd=None)
    from nora.provider.catalog import PROVIDER_DEFAULTS
    assert bridge._model == PROVIDER_DEFAULTS["anthropic"]


# ---------------------------------------------------------------------------
# Fix 3 — chip × actually unstages on the Python side
# ---------------------------------------------------------------------------

def test_unstage_attachment_removes_from_pending(tmp_path: Path) -> None:
    """The composer chip × must remove the corresponding entry from
    ``_pending_script_attachments`` — otherwise the next message's
    prefix block silently re-includes the script. Privacy mismatch:
    the user thinks they unstaged it, but the prompt still carries
    the content."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._pending_script_attachments.append({
        "name": "regression.py", "ext": ".py",
        "content": "import pandas\n", "bytes": 14,
    })
    bridge._pending_script_attachments.append({
        "name": "robustness.do", "ext": ".do",
        "content": "regress y x\n", "bytes": 12,
    })

    res = bridge.unstage_attachment("regression.py")

    assert res["ok"] is True
    assert res["removed"] == 1
    names = [a["name"] for a in bridge._pending_script_attachments]
    assert names == ["robustness.do"]


def test_unstage_attachment_idempotent(tmp_path: Path) -> None:
    """Removing a name that isn't staged is a no-op success — the
    JS chip × can be double-clicked without the bridge throwing."""
    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.unstage_attachment("never-staged.py")
    assert res["ok"] is True
    assert res["removed"] == 0


def test_unstage_attachment_basenames_input(tmp_path: Path) -> None:
    """Defensive — refuse path traversal in the name. A staged
    ``regression.py`` is removed when the JS sends ``regression.py``
    OR a basename-stripped traversal attempt; nothing gets confused
    into removing the wrong staged entry."""
    bridge = NoraBridge(cwd=tmp_path)
    bridge._pending_script_attachments.append({
        "name": "regression.py", "ext": ".py",
        "content": "x", "bytes": 1,
    })

    bridge.unstage_attachment("../../regression.py")

    assert bridge._pending_script_attachments == []


# ---------------------------------------------------------------------------
# Fix 4 — mid-chat dataset uploads surface in the next turn's prefix
# ---------------------------------------------------------------------------

def test_known_datasets_initialises_empty() -> None:
    """A bridge with no cwd has no datasets to know about. Documents
    the empty-snapshot starting state so the diff in ``_run_turn``
    has a stable baseline."""
    bridge = NoraBridge(cwd=None)
    assert bridge._known_datasets == frozenset()


# ---------------------------------------------------------------------------
# Stuck-UI guard — provider stream closes without a terminal event
# ---------------------------------------------------------------------------

def test_run_turn_synthesises_terminal_event_when_stream_silent(
    tmp_path: Path,
) -> None:
    """If the provider's send() generator closes without yielding
    TurnDone / TurnError / AuthFailure, ``_run_turn`` must push a
    synthetic ``turn_error`` so the JS composer flips back to
    enabled. Without this the UI stayed stuck on "sending"
    forever and Stop reported "no turn in flight" because the
    task had already finished.
    """
    import asyncio
    from nora.provider import AssistantText
    from nora.ui import NoraBridge

    # Stub session whose send() yields ONLY a non-terminal event
    # and then returns — exactly the pathological shape we're
    # guarding against.
    class _SilentSession:
        async def open(self) -> None: ...
        async def close(self) -> None: ...

        async def send(self, prompt, images=None):
            yield AssistantText(text="hello")

    bridge = NoraBridge(cwd=tmp_path)
    bridge._send_lock = asyncio.Lock()
    bridge._session = _SilentSession()

    pushed: list[dict] = []
    bridge._push_event = pushed.append  # type: ignore[assignment]

    asyncio.run(bridge._run_turn("test"))

    types = [p.get("type") for p in pushed]
    assert "assistant_text" in types
    # The synthesised terminal event lands as turn_error (the JS
    # state machine treats turn_error as a session-recover signal —
    # any terminal event flips setSending(false)).
    assert "turn_error" in types, (
        "_run_turn must push a terminal event even when the "
        "provider stream closes silently — otherwise the JS UI "
        "stays stuck on 'sending'"
    )


def test_dataset_diff_detects_new_uploads(tmp_path: Path) -> None:
    """End-to-end of the diff logic: snapshot a small set of
    datasets, drop a new one, recompute the diff — the new name
    surfaces. We exercise the scan_datasets helper the bridge
    uses, not the full _run_turn (that would need a live
    provider session)."""
    from nora.system_prompt import scan_datasets

    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "trial.csv").write_text("a,b\n1,2\n")

    initial = frozenset(p.name for p in scan_datasets(cwd))
    assert initial == {"trial.csv"}

    # Researcher drops a parquet mid-chat.
    (cwd / "panel.parquet").write_text("placeholder")

    current = frozenset(p.name for p in scan_datasets(cwd))
    new_datasets = current - initial
    assert new_datasets == {"panel.parquet"}, (
        "the diff is the data the bridge prepends to the next turn"
    )

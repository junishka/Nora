"""Regression test for per-session model memory (P2).

Before the fix, ``active_model`` was written to
``.nora/session_state.json`` after every successful turn but never
read back when the session was opened — the JS frontend instead
applied a global ``localStorage`` value, so switching between two
sessions ignored the model used in each.

These tests pin the new behaviour: opening a session whose state file
records ``active_model`` restores that model and provider; sessions
without a recorded model fall back to whatever the bridge is already
set to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nora.session_state import SessionState, write_session_state
from nora.ui import NoraBridge


def _seed_session_state(cwd: Path, model: str) -> None:
    """Write a session_state.json with the given active_model."""
    cwd.mkdir(parents=True, exist_ok=True)
    write_session_state(cwd, model=model)


def test_set_cwd_restores_recorded_model(tmp_path: Path) -> None:
    """A session whose state file says ``active_model = "claude-opus-..."``
    should open with Opus selected, not the global default."""
    session_dir = tmp_path / "session-a"
    session_dir.mkdir()
    _seed_session_state(session_dir, "claude-opus-4-7[1m]")

    bridge = NoraBridge(cwd=None)
    # Default model at construction is the catalog default. Confirm we
    # actually swap away from it.
    assert bridge._model != "claude-opus-4-7[1m]"

    bridge._set_cwd(session_dir)

    assert bridge._model == "claude-opus-4-7[1m]"
    assert bridge._provider == "anthropic"


def test_set_cwd_with_no_state_file_keeps_default(tmp_path: Path) -> None:
    """A fresh session dir with no state file should leave the bridge
    on its current model (the constructor's catalog default)."""
    session_dir = tmp_path / "session-fresh"
    session_dir.mkdir()

    bridge = NoraBridge(cwd=None)
    original_model = bridge._model
    original_provider = bridge._provider

    bridge._set_cwd(session_dir)

    assert bridge._model == original_model
    assert bridge._provider == original_provider


def test_set_cwd_ignores_unknown_model(tmp_path: Path) -> None:
    """A state file pointing at a model that's been removed from the
    catalog (renamed, deprecated) should silently fall back to the
    current value rather than wedge the bridge with a bad id."""
    session_dir = tmp_path / "session-stale"
    session_dir.mkdir()
    _seed_session_state(session_dir, "claude-haiku-removed-2099")

    bridge = NoraBridge(cwd=None)
    original_model = bridge._model

    bridge._set_cwd(session_dir)

    assert bridge._model == original_model


def test_set_cwd_ignores_unauthed_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded OpenAI model should NOT be restored if the
    researcher has no OpenAI credential — opening a session against a
    provider we can't auth would just fail at first turn."""
    session_dir = tmp_path / "session-needs-openai"
    session_dir.mkdir()
    _seed_session_state(session_dir, "gpt-5.5")

    # Force ``_authed_providers`` to report only Anthropic.
    monkeypatch.setattr(
        NoraBridge, "_authed_providers", lambda self: {"anthropic"}
    )

    bridge = NoraBridge(cwd=None)
    original_model = bridge._model

    bridge._set_cwd(session_dir)

    assert bridge._model == original_model
    assert bridge._provider == "anthropic"


def test_switching_back_and_forth_remembers_per_session(
    tmp_path: Path,
) -> None:
    """End-to-end: two sessions, each remembers its own model. The
    point of the per-session feature."""
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    session_a.mkdir()
    session_b.mkdir()
    _seed_session_state(session_a, "claude-sonnet-4-6[1m]")
    _seed_session_state(session_b, "claude-opus-4-7[1m]")

    bridge = NoraBridge(cwd=None)

    bridge._set_cwd(session_a)
    assert bridge._model == "claude-sonnet-4-6[1m]"

    bridge._set_cwd(session_b)
    assert bridge._model == "claude-opus-4-7[1m]"

    bridge._set_cwd(session_a)
    assert bridge._model == "claude-sonnet-4-6[1m]"


def test_set_model_persists_choice_immediately(tmp_path: Path) -> None:
    """A successful set_model should write active_model to the state
    file BEFORE the first turn lands, so swap-then-quit-then-reopen
    comes back to the chosen model rather than the prior recorded
    one."""
    session_dir = tmp_path / "session-swap"
    session_dir.mkdir()
    _seed_session_state(session_dir, "claude-sonnet-4-6[1m]")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(session_dir)
    assert bridge._model == "claude-sonnet-4-6[1m]"

    # Swap to Opus. Bridge has no session open yet so set_model is the
    # no-session branch — just stashes the choice. Without our
    # _persist_active_model() call, the next read_session_state would
    # still return Sonnet.
    res = bridge.set_model("claude-opus-4-7[1m]")
    assert res["ok"] is True

    # Confirm the on-disk file reflects the swap.
    state_path = session_dir / ".nora" / "session_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["active_model"] == "claude-opus-4-7[1m]"

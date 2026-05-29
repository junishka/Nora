"""Regression test for the failed-model-switch rollback bug (P1).

Before the fix, ``NoraBridge.set_model`` assigned ``self._model`` to
the new id BEFORE calling ``session.set_model``, then on either the
exception branch or the ``res.ok=False`` branch returned without
restoring it. The next turn's ``_ensure_session`` would reopen with
the rejected id and fail again, while the JS chip still showed the
old name — researcher saw a chat that "stopped working" with no
recovery short of restart.

These tests pin the rollback behaviour: after any failure path,
``self._model`` and ``self._provider`` must equal what they were
before the switch attempt.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from nora.ui import NoraBridge


def _spin_loop(bridge: NoraBridge) -> None:
    """Start a worker loop on a background thread, the way pywebview
    drives the bridge in production. Tests use the loop to schedule
    the async ``session.set_model`` swap; we tear it down at the end
    of the test."""
    bridge.start_loop()


def _stop_loop(bridge: NoraBridge) -> None:
    bridge.stop_loop()


class _FakeSession:
    """Stand-in for a ``ProviderSession``. The test injects one of
    these as ``bridge._session`` and asserts what ``bridge.set_model``
    does in the face of various session responses."""

    def __init__(self, *, swap_returns: dict[str, Any] | None = None,
                 swap_raises: Exception | None = None) -> None:
        self._swap_returns = swap_returns
        self._swap_raises = swap_raises
        self.set_model_called_with: list[str] = []
        self.close_count = 0

    async def set_model(self, model_id: str) -> dict[str, Any]:
        self.set_model_called_with.append(model_id)
        if self._swap_raises is not None:
            raise self._swap_raises
        return self._swap_returns or {"ok": True, "model": model_id}

    async def close(self) -> None:
        self.close_count += 1

    async def open(self) -> None:
        return None


def test_failed_swap_via_exception_restores_model(tmp_path: Path) -> None:
    """SDK raises mid-swap → bridge must restore the previous model
    id so the next turn reopens with the WORKING model, not the
    rejected one."""
    bridge = NoraBridge(cwd=tmp_path)
    _spin_loop(bridge)
    try:
        original_model = bridge._model
        original_provider = bridge._provider
        bridge._session = _FakeSession(swap_raises=RuntimeError("nope"))

        # Pick a different Anthropic model to trigger the swap path.
        target = "claude-opus-4-8[1m]"
        assert target != original_model

        res = bridge.set_model(target)

        assert res["ok"] is False, "swap reporting failure"
        assert bridge._model == original_model, (
            "model id must be restored to its pre-swap value"
        )
        assert bridge._provider == original_provider
        # Session was torn down so the next turn starts fresh.
        assert bridge._session is None
    finally:
        _stop_loop(bridge)


def test_failed_swap_via_ok_false_restores_model(tmp_path: Path) -> None:
    """Session refuses the swap (returns ok=False) → same restoration."""
    bridge = NoraBridge(cwd=tmp_path)
    _spin_loop(bridge)
    try:
        original_model = bridge._model
        original_provider = bridge._provider
        bridge._session = _FakeSession(
            swap_returns={"ok": False, "reason": "unknown id at SDK level"}
        )

        target = "claude-opus-4-8[1m]"
        res = bridge.set_model(target)

        assert res["ok"] is False
        assert "unknown id" in res["reason"]
        assert bridge._model == original_model
        assert bridge._provider == original_provider
    finally:
        _stop_loop(bridge)


def test_successful_swap_keeps_new_model(tmp_path: Path) -> None:
    """Sanity: when the session accepts the swap, the new id sticks."""
    bridge = NoraBridge(cwd=tmp_path)
    _spin_loop(bridge)
    try:
        original_model = bridge._model
        bridge._session = _FakeSession(swap_returns={"ok": True})

        target = "claude-opus-4-8[1m]"
        assert target != original_model
        res = bridge.set_model(target)

        assert res["ok"] is True
        assert bridge._model == target
    finally:
        _stop_loop(bridge)


def test_failed_swap_with_no_open_session_still_assigns(
    tmp_path: Path,
) -> None:
    """When no session is open yet, set_model has nothing to fail at —
    the new id should stick because the next ``_ensure_session``
    will build with it. This documents the "swap before first turn"
    behaviour so a future refactor doesn't accidentally rollback in
    that case too."""
    bridge = NoraBridge(cwd=tmp_path)
    # No loop, no session — just a fresh bridge.
    original_model = bridge._model
    target = "claude-opus-4-8[1m]"
    assert target != original_model
    res = bridge.set_model(target)
    assert res["ok"] is True
    assert bridge._model == target

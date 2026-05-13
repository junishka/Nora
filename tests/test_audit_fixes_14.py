"""Regression tests for the fourteenth batch of reviewer-flagged
fixes.

The behaviors pinned here:

1. ``NoraBridge._emit_install_confirmation_request`` propagates
   ``evaluate_js`` failures instead of swallowing them. The prior
   code swallowed the exception with a comment claiming
   ``request_confirmation`` would catch it; but the swallowing
   meant ``request_confirmation`` saw a successful emit, then
   blocked on the awaiting future for the full
   ``DEFAULT_TIMEOUT_SECONDS=300`` because no JS-side handler
   would ever call back. Re-raising hands control to
   ``request_confirmation``'s emitter-exception path, which
   resolves to deny immediately.

2. ``file_provenance.initialize`` does NOT fingerprint files whose
   extensions can't be recalled as bytes by any tool. Hashing a
   multi-GB ``.dta`` / ``.parquet`` / ``.csv`` at session-open
   blocked folder-open with no security gain — the recall tools
   (``read_attached_file``, ``submit_script_file``,
   ``search_in_session_files``) all reject data extensions
   regardless of provenance state. The skip is gated on the
   union of script / log / graph extensions from
   ``session_files``.

3. ``packaging/launcher.sh`` extends the hard-coded PATH bridge
   to include the two common shell-shim managers (asdf, mise).
   The prior comment claimed ``~/.zshenv`` was a working
   workaround for paths outside the hard-coded list; that's
   wrong because the launcher is a ``#!/bin/bash`` script
   launched by launchd, neither of which sources zsh init
   files. Removed the misleading claim and added concrete
   alternatives (``launchctl setenv``, terminal launch).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Install confirmation emitter propagates evaluate_js failures
# ---------------------------------------------------------------------------


class _FakeWindow:
    """Stand-in for a pywebview window whose ``evaluate_js`` raises.

    Mirrors the runtime shape: the bridge calls
    ``self._window.evaluate_js(...)``; any RuntimeError there must
    propagate to the emitter caller (and ultimately to
    ``request_confirmation``)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def evaluate_js(self, _js: str) -> None:
        raise self._exc


def _make_bridge_with_window(window: object) -> object:
    """Construct a minimally-wired NoraBridge so we can exercise
    the emitter without setting up the full UI stack."""
    from nora.ui import NoraBridge
    bridge = NoraBridge.__new__(NoraBridge)
    bridge._window = window  # type: ignore[attr-defined]
    return bridge


def test_install_confirmation_emitter_propagates_evaluate_js_failure() -> None:
    """The emitter must re-raise so ``request_confirmation`` can
    treat the failure as deny. The prior swallow-and-pass meant
    the awaiting future timed out at 300 seconds instead."""
    bridge = _make_bridge_with_window(
        _FakeWindow(RuntimeError("webview shutting down")),
    )

    with pytest.raises(RuntimeError, match="webview shutting down"):
        bridge._emit_install_confirmation_request(  # type: ignore[attr-defined]
            "token-abc", "Python", ["pandas"], "install",
        )


def test_request_confirmation_denies_promptly_when_emitter_raises() -> None:
    """End-to-end: register an emitter that fails immediately, call
    ``request_confirmation``, and confirm it returns False without
    waiting for the timeout. The test uses a 5-second timeout so a
    regression (waiting on the future) would show up as a
    ``TimeoutError`` from ``asyncio.wait_for`` here, not as the
    function returning False."""
    from nora.install_confirmation import (
        clear_request_emitter,
        request_confirmation,
        set_request_emitter,
    )

    def failing_emitter(
        token: str, language: str, packages: list[str], action: str,
    ) -> None:
        raise RuntimeError("simulated webview-closed")

    set_request_emitter(failing_emitter)
    try:
        async def _run() -> bool:
            return await asyncio.wait_for(
                request_confirmation(
                    "Python", ["pandas"], "install", timeout=5.0,
                ),
                timeout=5.0,
            )

        result = asyncio.run(_run())
        assert result is False
    finally:
        clear_request_emitter()


# ---------------------------------------------------------------------------
# 2. file_provenance skips non-recallable extensions
# ---------------------------------------------------------------------------


def test_initialize_skips_data_extensions_no_disk_read(
    tmp_path: Path,
) -> None:
    """Confirm the optimization at the I/O boundary: data files
    are never opened for hashing during initialize. A 3 GB
    ``.dta`` in cwd would otherwise block folder-open."""
    from nora.file_provenance import _fingerprint, initialize
    import nora.file_provenance as fp_mod

    (tmp_path / "panel.dta").write_bytes(b"<dta>")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "results.parquet").write_bytes(b"<parquet>")
    (tmp_path / "analysis.py").write_text("import pandas\n")

    opened: list[str] = []

    def _trace(path: Path):
        opened.append(path.name)
        return _fingerprint(path)

    fp_mod._fingerprint = _trace  # type: ignore[assignment]
    try:
        names = initialize(tmp_path)
    finally:
        fp_mod._fingerprint = _fingerprint  # type: ignore[assignment]

    # Recallable extensions get hashed.
    assert "analysis.py" in opened
    # Non-recallable extensions are never opened.
    assert "panel.dta" not in opened
    assert "data.csv" not in opened
    assert "results.parquet" not in opened
    # Manifest reflects the same partition.
    assert names == {"analysis.py"}


def test_read_attached_file_dataset_rejection_message_preserved(
    tmp_path: Path,
) -> None:
    """The provenance gate should not mask the dataset-specific
    rejection. A ``.dta`` in cwd that's not in the manifest must
    still be rejected with a ``get_schema`` hint (the helpful
    error), not with a generic "not staged" reason. The fix
    gates the provenance check on recallable-extension membership;
    data extensions skip provenance and hit the extension
    fallback's get_schema hint instead."""
    from nora.config import set_cwd
    from nora.tools import HANDLERS
    set_cwd(tmp_path)
    (tmp_path / "panel.dta").write_bytes(b"<stata bytes>")

    payload = asyncio.run(
        HANDLERS["read_attached_file"]({"name": "panel.dta"}),
    )
    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "rejected"
    assert "get_schema" in body["reason"], (
        f"expected the dataset-specific get_schema hint; got: "
        f"{body['reason']!r}"
    )


def test_submit_script_file_unknown_ext_returns_error_status(
    tmp_path: Path,
) -> None:
    """Files with unknown extensions hit the script-language check
    and return ``status="error"`` with a "supported extensions"
    list — not the provenance "rejected: not staged" message that
    would mask which extensions are even allowed."""
    from nora.config import set_cwd
    from nora.tools import HANDLERS
    set_cwd(tmp_path)
    (tmp_path / "notes.txt").write_text("not a script\n")

    payload = asyncio.run(
        HANDLERS["submit_script_file"]({"name": "notes.txt"}),
    )
    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "error"
    assert "recognised script file" in body["reason"]


# ---------------------------------------------------------------------------
# 3. launcher.sh covers asdf/mise and drops the bad ~/.zshenv claim
# ---------------------------------------------------------------------------


def test_launcher_path_bridge_includes_shim_managers() -> None:
    """``packaging/launcher.sh`` must include shim-manager
    directories for asdf and mise. Without these, users whose
    ``node`` / ``claude`` live behind a shim get the same
    ``env: node: No such file or directory`` failure the original
    hard-coded list was supposed to fix."""
    launcher = (
        Path(__file__).resolve().parents[1] / "packaging" / "launcher.sh"
    )
    src = launcher.read_text(encoding="utf-8")
    # Match the assignment line to keep the assertion stable
    # across reorderings of other path entries.
    match = re.search(r'_NORA_EXTRA_PATH="([^"]*)"', src)
    assert match is not None, "expected _NORA_EXTRA_PATH assignment"
    path_value = match.group(1)
    assert "$HOME/.asdf/shims" in path_value, (
        "asdf shim dir missing from launcher PATH bridge"
    )
    assert "$HOME/.local/share/mise/shims" in path_value, (
        "mise shim dir missing from launcher PATH bridge"
    )


def test_launcher_drops_misleading_zshenv_workaround_claim() -> None:
    """The earlier comment told users they could extend PATH by
    exporting it from ``~/.zshenv``. That's wrong for a bash
    launcher started by launchd; the comment must not promise it
    works. (A reference to zshenv in another context, eg a
    troubleshooting bullet, can remain — what we're guarding
    against is the specific claim that an export there flows
    through.)"""
    launcher = (
        Path(__file__).resolve().parents[1] / "packaging" / "launcher.sh"
    )
    src = launcher.read_text(encoding="utf-8")
    # The old promise was a sentence claiming an export in
    # ~/.zshenv "flows through" to the launcher.
    assert "flows through" not in src or "zshenv" not in src, (
        "launcher.sh still claims ~/.zshenv export 'flows through' "
        "to the launcher; that's not true for a bash script "
        "started by launchd"
    )

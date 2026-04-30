"""Regression tests for two server-side audit fixes:

1. ``read_attached_file`` plot-fallback used ``str.startswith`` for
   containment, which allows a sibling whose path-prefix collides
   with cwd to escape (``/sessions/foo`` "contains"
   ``/sessions/foo-bar``). Switched to ``Path.is_relative_to``.

2. Cancelling or erroring a turn used to silently re-prepend
   mentioned files / images to the bridge's pending lists. The
   composer chip already cleared on send, so the next user message
   carried attachments the researcher no longer saw. Now the cancel
   / error branches drop the carry-back; researcher re-attaches
   explicitly if they want.

Two corresponding JS-side fixes (Send guard for attachments-only,
composer image dual-tracking) live in ``app.js`` and aren't
exercised by Python tests; they're contained changes in handlers
that don't have a JS test harness.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import use_cwd
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    text_block = next(
        b for b in payload["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


def test_plot_fallback_refuses_path_prefix_collision(tmp_path: Path) -> None:
    """Construct a sibling directory whose absolute path string
    starts with cwd's path string (``/.../foo`` vs ``/.../foo-bar``).
    The plot-fallback path used to admit it via ``str.startswith``;
    with the fix, ``Path.is_relative_to`` correctly refuses."""
    # Two sibling session dirs whose names share a prefix.
    cwd = tmp_path / "session-foo"
    sibling = tmp_path / "session-foo-bar"
    cwd.mkdir()
    sibling.mkdir()

    # Plant a file in the sibling that the fallback might mistakenly
    # claim is "inside" cwd if the check is string-prefix.
    leak = sibling / "leak.do"
    leak.write_text("* secret content from a different session\n")

    # Set up a runs/_nora_plots dir under cwd so the fallback runs.
    plots_dir = cwd / ".nora" / "runs" / "r1" / "_nora_plots"
    plots_dir.mkdir(parents=True)

    # Place a SYMLINK with the same basename pointing at the leak.
    # candidate.resolve() will follow it to the sibling path —
    # whose absolute string starts with cwd's absolute string when
    # the cwd is a strict prefix of the sibling.
    symlink = plots_dir / "leak.do"
    symlink.symlink_to(leak)

    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["read_attached_file"]({"name": "leak.do"}))
    body = _mcp_text(res)
    # The fallback must NOT have admitted the symlink target. Either
    # status is "not_found" (correct refusal) or the body's content
    # field doesn't carry the leak content. Either way: no escape.
    if body.get("status") == "ok":
        content = body.get("content", "")
        assert "secret content" not in content, (
            "plot fallback admitted a sibling-prefix-collision file"
        )
    else:
        assert body["status"] == "not_found"


def test_plot_fallback_still_admits_legitimate_run_dir_files(
    tmp_path: Path,
) -> None:
    """Negative regression: the stricter ``is_relative_to`` check
    must still let through actual files inside cwd's run dirs.
    Otherwise we'd block the researcher's own plots."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    plots_dir = cwd / ".nora" / "runs" / "r1" / "_nora_plots"
    plots_dir.mkdir(parents=True)
    plot = plots_dir / "residuals.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    with use_cwd(cwd):
        res = asyncio.run(
            HANDLERS["read_attached_file"]({"name": "residuals.png"}),
        )
    # Either we get a valid image envelope or, for the placeholder
    # PNG, an error from the rasterizer. The point is we did NOT
    # get "not_found" — the fallback still finds files in cwd.
    body = _mcp_text(res)
    assert body.get("status") != "not_found", body


# ---------------------------------------------------------------------------
# Carry-back fix: mentioned files / images no longer survive a
# cancel or error.
# ---------------------------------------------------------------------------

def test_runner_does_not_re_prepend_mentioned_files_on_cancel() -> None:
    """The cancel branch in ``SessionRunner._run_turn`` must NOT
    re-prepend ``carried_mentioned_files`` / ``attached_mentioned_images``
    onto the pending lists. The composer cleared the chip on send;
    silently sneaking the attachment into the next message violates
    what-you-see-is-what-you-send."""
    import inspect

    from nora.runner import SessionRunner
    src = inspect.getsource(SessionRunner)
    # Locate the cancel branch and the exception branch. Neither
    # may contain a re-prepend of the mentioned-files / images
    # state.
    cancel_idx = src.index("except asyncio.CancelledError:")
    except_idx = src.index("except Exception as e:", cancel_idx)
    finally_idx = src.index("finally:", except_idx)
    cancel_block = src[cancel_idx:except_idx]
    except_block = src[except_idx:finally_idx]
    for label, block in (("cancel", cancel_block), ("error", except_block)):
        assert "self.pending_mentioned_files" not in block, (
            f"{label} branch still re-prepends pending_mentioned_files"
        )
        assert "self.pending_mentioned_images" not in block, (
            f"{label} branch still re-prepends pending_mentioned_images"
        )
        # Plots and prefix carry still belong; just confirm the
        # branch isn't accidentally empty (would mean a refactor
        # blew it away).
        assert "pending_plot_images" in block or "needs_context_prefix" in block

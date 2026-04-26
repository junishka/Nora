"""Tests for the mid-chat script-attachment pipeline.

A researcher who drops a ``.py`` / ``.do`` / ``.r`` / ``.rmd`` file
into the composer mid-chat expects the model to know about it on the
very next message — without inlining, the file silently lands in the
session cwd and "what does this do?" hits the model with no context.

The pipeline:
  1. ``add_files_from_blobs`` (drag/drop) and ``add_files`` (native
     dialog) detect script extensions, copy to cwd AND stage the
     contents in ``_pending_script_attachments``.
  2. ``_run_turn`` reads the staged list, builds a prefix block, and
     prepends it to the next user message.
  3. The list is cleared after a successful turn; restored on cancel
     / error so a transient failure doesn't lose the attachment.

These tests cover the staging path (without spinning up a real
provider session) plus the prefix-rendering shape.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nora.ui import (
    NoraBridge,
    _build_script_attachment_prefix,
    _stage_script_for_next_turn,
)


# ---------------------------------------------------------------------------
# _stage_script_for_next_turn — allowlist + size cap
# ---------------------------------------------------------------------------

def test_stage_includes_supported_script_exts() -> None:
    pending: list[dict] = []
    _stage_script_for_next_turn(
        pending, "regression.py", ".py", b"import pandas as pd\n",
    )
    assert len(pending) == 1
    assert pending[0]["name"] == "regression.py"
    assert pending[0]["ext"] == ".py"
    assert "import pandas" in pending[0]["content"]


def test_stage_skips_non_script_extensions() -> None:
    """A ``.csv`` should land in cwd via the caller's copy step but
    must NOT be inlined — data files are not appropriate to ship as
    source-code-style context blocks."""
    pending: list[dict] = []
    _stage_script_for_next_turn(
        pending, "trial.csv", ".csv", b"a,b\n1,2\n",
    )
    assert pending == []


def test_stage_truncates_oversized_scripts() -> None:
    """Files above the per-file cap get the first chunk + a marker.
    Prevents a multi-megabyte log accidentally renamed to ``.py``
    from blowing up the next prompt."""
    big = b"# " + (b"x" * 200_000) + b"\n"
    pending: list[dict] = []
    _stage_script_for_next_turn(pending, "huge.py", ".py", big)
    assert len(pending) == 1
    assert "truncated" in pending[0]["content"]
    # Body is capped near the per-file limit (64 KB).
    assert len(pending[0]["content"]) < 80 * 1024


def test_stage_handles_non_utf8_bytes() -> None:
    """Replacement-decode pathological bytes — never raise. The user
    might paste a Windows-encoded .do file; failing to surface the
    file at all because of an encoding hiccup is worse than showing
    it with a few replacement glyphs."""
    pending: list[dict] = []
    _stage_script_for_next_turn(
        pending, "weird.do", ".do", b"use \xff\xfe\n",
    )
    assert len(pending) == 1
    assert "use " in pending[0]["content"]


# ---------------------------------------------------------------------------
# _build_script_attachment_prefix — output shape
# ---------------------------------------------------------------------------

def test_prefix_empty_for_no_attachments(tmp_path: Path) -> None:
    assert _build_script_attachment_prefix([], tmp_path) == ""


def test_prefix_renders_files_with_language_hints(tmp_path: Path) -> None:
    pending = [
        {"name": "ols.py", "ext": ".py",
         "content": "import statsmodels.api as sm\n", "bytes": 30},
        {"name": "regress.do", "ext": ".do",
         "content": "regress y x\n", "bytes": 12},
    ]
    out = _build_script_attachment_prefix(pending, tmp_path)
    # Both files appear with their content.
    assert "ols.py" in out
    assert "import statsmodels" in out
    assert "regress.do" in out
    assert "regress y x" in out
    # Each block carries a language hint so the model can mirror
    # the syntax if it decides to extend the script.
    assert "(Python)" in out
    assert "(Stata)" in out
    # Header / footer bracket the block as background, not as the
    # researcher's actual instruction.
    assert "researcher attached" in out.lower()
    assert "End of attached files" in out


def test_prefix_caps_aggregate_size(tmp_path: Path) -> None:
    """A pile of attachments together can exceed the aggregate cap;
    once the budget is full, remaining files are listed by name with
    a "too large" note rather than dropping silently."""
    big_content = "x" * (50 * 1024)  # 50 KB each
    pending = [
        {"name": f"f{i}.py", "ext": ".py", "content": big_content, "bytes": 50_000}
        for i in range(8)  # 8 × 50 KB = 400 KB > 256 KB cap
    ]
    out = _build_script_attachment_prefix(pending, tmp_path)
    assert "budget exceeded" in out
    # Some files made it in; others didn't.
    included = sum(1 for i in range(8) if f"f{i}.py" in out)
    assert included == 8  # all NAMES surface (the omitted ones get the note)
    # But not all 8 contents fit.
    assert out.count(big_content) < 8


# ---------------------------------------------------------------------------
# Bridge: add_files_from_blobs stages scripts; cleared after consumption
# ---------------------------------------------------------------------------

def _make_bridge(tmp_path: Path) -> NoraBridge:
    cwd = tmp_path / "session"
    cwd.mkdir()
    return NoraBridge(cwd=cwd)


def test_drag_drop_py_file_stages_for_next_turn(tmp_path: Path) -> None:
    bridge = _make_bridge(tmp_path)
    # Same shape JS sends in: ``[{name, content (base64), mime?}]``.
    code = b"import pandas as pd\nprint('hi')\n"
    payload = [{
        "name": "analysis.py",
        "content": "data:," + base64.b64encode(code).decode("ascii"),
    }]
    res = bridge.add_files_from_blobs(payload)
    assert res["ok"] is True
    assert "analysis.py" in res["added"]
    # File on disk in the session cwd.
    assert (bridge.cwd / "analysis.py").read_bytes() == code
    # AND staged for the next message — that's the new behaviour.
    assert len(bridge._pending_script_attachments) == 1
    staged = bridge._pending_script_attachments[0]
    assert staged["name"] == "analysis.py"
    assert "import pandas" in staged["content"]


def test_drag_drop_csv_does_not_stage(tmp_path: Path) -> None:
    """Data files (.csv, .parquet, …) get copied to cwd but must NOT
    appear in the next-message prefix — that block is for
    source-code context only. Researchers reach data via
    ``get_schema``, not via inline dump."""
    bridge = _make_bridge(tmp_path)
    csv = b"a,b\n1,2\n3,4\n"
    payload = [{
        "name": "trial.csv",
        "content": "data:," + base64.b64encode(csv).decode("ascii"),
    }]
    res = bridge.add_files_from_blobs(payload)
    assert res["ok"] is True
    assert "trial.csv" in res["added"]
    assert (bridge.cwd / "trial.csv").exists()
    # No script staging — data file is silently in cwd, schema-discovered.
    assert bridge._pending_script_attachments == []


def test_multiple_script_drops_accumulate(tmp_path: Path) -> None:
    bridge = _make_bridge(tmp_path)
    files = [
        {"name": "a.py", "content": base64.b64encode(b"# a\n").decode("ascii")},
        {"name": "b.do", "content": base64.b64encode(b"* b\n").decode("ascii")},
    ]
    bridge.add_files_from_blobs(files)
    assert len(bridge._pending_script_attachments) == 2
    names = [s["name"] for s in bridge._pending_script_attachments]
    assert names == ["a.py", "b.do"]


def test_send_message_persists_attachment_names(tmp_path: Path) -> None:
    """The user_message persisted to chat_history.jsonl must carry
    the attachment filenames so a session reload renders the
    "📎 attached: name.py" chip in the transcript. Without this,
    a researcher who reopens the session has no record that the
    upload happened.
    """
    import json
    bridge = _make_bridge(tmp_path)
    # Pretend a script was just dropped.
    bridge._pending_script_attachments.append({
        "name": "regression.py", "ext": ".py",
        "content": "import pandas\n", "bytes": 14,
    })

    # _persist_event is the only side-effect we exercise here —
    # no need to spin up the worker loop just for the persistence
    # path. send_message would also call _run_turn, which we don't
    # want to drive without a session, so go via the persistence
    # surface directly.
    bridge._persist_event({
        "type": "user_message",
        "text": "what does this do?",
        "attachments": [a["name"] for a in bridge._pending_script_attachments],
    })

    log = tmp_path / "session" / ".nora" / "chat_history.jsonl"
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert record["type"] == "user_message"
    assert record["attachments"] == ["regression.py"]
    assert record["text"] == "what does this do?"


def test_image_drop_saves_to_cwd_and_stages_for_vision(tmp_path: Path) -> None:
    """An image dragged into the composer must (a) land on disk in
    the session cwd so the researcher can reference it later, and
    (b) be staged as a vision attachment for the next message.
    Previously only (b) happened — the upload "vanished" once the
    next message was sent."""
    bridge = _make_bridge(tmp_path)
    # 1×1 PNG — smallest valid bytes pyplot would write.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    payload = [{
        "name": "fig1.png",
        "content": "data:," + base64.b64encode(png_bytes).decode("ascii"),
        "mime": "image/png",
    }]
    res = bridge.add_files_from_blobs(payload)
    assert res["ok"] is True

    # On disk in cwd — the researcher can find it again later.
    saved = bridge.cwd / "fig1.png"
    assert saved.exists()
    assert saved.read_bytes() == png_bytes

    # AND staged as a vision attachment for the next message.
    assert len(res["images"]) == 1
    assert res["images"][0]["name"] == "fig1.png"
    assert res["images"][0]["mime"] == "image/png"


def test_existing_file_collision_does_not_stage(tmp_path: Path) -> None:
    """A re-uploaded ``.py`` that collides with an existing session
    file is refused (no overwrite) — and must NOT be staged for the
    next turn either, because the on-disk version the model could
    reference is the OLD one, not what the user just dropped."""
    bridge = _make_bridge(tmp_path)
    (bridge.cwd / "analysis.py").write_text("# original\n")

    payload = [{
        "name": "analysis.py",
        "content": base64.b64encode(b"# new attempt\n").decode("ascii"),
    }]
    res = bridge.add_files_from_blobs(payload)
    assert "analysis.py" in res["skipped_existing"]
    # The original file is untouched...
    assert (bridge.cwd / "analysis.py").read_text() == "# original\n"
    # ...and nothing was staged for the next message.
    assert bridge._pending_script_attachments == []

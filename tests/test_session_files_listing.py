"""Tests for ``NoraBridge.list_session_files``.

The Files chip in the topbar reads from this endpoint. Without it,
only data files showed up (the chip was previously fed from
``policy.datasets``, which only covers schemas) — researchers who
dropped a ``.py`` or ``.gph`` saw the count stay flat and silently
worried the upload had failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora.ui import NoraBridge


def _bridge_with_files(tmp_path: Path, names: list[str]) -> NoraBridge:
    cwd = tmp_path / "session"
    cwd.mkdir()
    for n in names:
        (cwd / n).write_text("placeholder", encoding="utf-8")
    return NoraBridge(cwd=cwd)


def test_list_session_files_groups_by_kind(tmp_path: Path) -> None:
    bridge = _bridge_with_files(tmp_path, [
        "data.csv",
        "panel.parquet",
        "regression.py",
        "robustness.do",
        "fig1.gph",
        "stata.log",
    ])
    res = bridge.list_session_files()
    assert res["ok"] is True
    by_name = {f["name"]: f for f in res["files"]}
    assert by_name["data.csv"]["kind"] == "data"
    assert by_name["panel.parquet"]["kind"] == "data"
    assert by_name["regression.py"]["kind"] == "script"
    assert by_name["robustness.do"]["kind"] == "script"
    assert by_name["fig1.gph"]["kind"] == "graph"
    assert by_name["stata.log"]["kind"] == "log"


def test_list_session_files_orders_kinds_data_first(tmp_path: Path) -> None:
    """The Files popup renders in priority order. Data first because
    it's what the analysis is about; scripts, graphs, logs follow."""
    bridge = _bridge_with_files(tmp_path, [
        "fig.gph", "x.py", "data.csv", "run.log",
    ])
    res = bridge.list_session_files()
    kinds_in_order = [f["kind"] for f in res["files"]]
    assert kinds_in_order == ["data", "script", "graph", "log"]


def test_list_session_files_skips_unknown_extensions(tmp_path: Path) -> None:
    """A stray ``.txt`` or ``.tex`` doesn't crash the listing nor
    pollute it. Only Nora-recognised extensions appear."""
    bridge = _bridge_with_files(tmp_path, [
        "data.csv", "notes.txt", "paper.tex",
    ])
    res = bridge.list_session_files()
    names = [f["name"] for f in res["files"]]
    assert names == ["data.csv"]


def test_list_session_files_returns_empty_when_no_cwd() -> None:
    bridge = NoraBridge(cwd=None)
    res = bridge.list_session_files()
    assert res == {"ok": True, "files": []}


def test_list_session_files_reports_size(tmp_path: Path) -> None:
    cwd = tmp_path / "s"
    cwd.mkdir()
    (cwd / "small.py").write_text("x" * 50, encoding="utf-8")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    assert res["files"][0]["size"] == 50


# ---------------------------------------------------------------------------
# attach_session_file — bring an already-uploaded script into the prompt
# ---------------------------------------------------------------------------

def test_attach_session_file_stages_script_for_next_turn(tmp_path: Path) -> None:
    """Clicking a script row in the Files popup must stage that
    file's contents for the next message — same effect as
    drag-dropping it again from Finder. Without this, a researcher
    who uploaded a script earlier in the session has no in-app way
    to surface it to the model."""
    bridge = _bridge_with_files(tmp_path, ["regression.py"])
    # Real content so the staged copy is meaningful.
    (bridge.cwd / "regression.py").write_text(
        "import pandas as pd\nprint(\"ols\")\n", encoding="utf-8"
    )

    res = bridge.attach_session_file("regression.py")

    assert res["ok"] is True
    assert res.get("already_attached") is not True
    assert len(bridge._pending_script_attachments) == 1
    staged = bridge._pending_script_attachments[0]
    assert staged["name"] == "regression.py"
    assert "import pandas" in staged["content"]


def test_attach_session_file_is_idempotent(tmp_path: Path) -> None:
    """Clicking the same row twice should not double-attach. The
    model would otherwise see two copies of the same script in
    the prefix, and the composer chip count would be wrong."""
    bridge = _bridge_with_files(tmp_path, ["analysis.py"])
    (bridge.cwd / "analysis.py").write_text("# code\n", encoding="utf-8")

    bridge.attach_session_file("analysis.py")
    res2 = bridge.attach_session_file("analysis.py")

    assert res2["ok"] is True
    assert res2["already_attached"] is True
    assert len(bridge._pending_script_attachments) == 1


def test_attach_session_file_refuses_data_extensions(tmp_path: Path) -> None:
    """Data files reach the model via get_schema; inlining a
    multi-MB CSV would just blow up the prompt. The endpoint
    refuses anything outside the script allowlist with a clear
    message."""
    bridge = _bridge_with_files(tmp_path, ["panel.parquet"])

    res = bridge.attach_session_file("panel.parquet")
    assert res["ok"] is False
    assert "script files" in res["reason"]
    assert bridge._pending_script_attachments == []


def test_attach_session_file_refuses_path_traversal(tmp_path: Path) -> None:
    """A malicious caller passing ``../../../etc/hosts`` must be
    refused. The bridge basenames the input and resolves against
    cwd before reading."""
    bridge = _bridge_with_files(tmp_path, ["data.csv"])
    # Plant a file outside cwd at the path traversal would point to.
    outside = tmp_path / "secret.py"
    outside.write_text("# secrets\n", encoding="utf-8")

    res = bridge.attach_session_file("../secret.py")
    # Basename strips the ../ ; bridge then looks for "secret.py" inside cwd,
    # which doesn't exist.
    assert res["ok"] is False
    assert "not found" in res["reason"].lower()
    assert bridge._pending_script_attachments == []


def test_attach_session_file_no_cwd_returns_clean_error() -> None:
    bridge = NoraBridge(cwd=None)
    res = bridge.attach_session_file("anything.py")
    assert res["ok"] is False
    assert "no active session" in res["reason"].lower()

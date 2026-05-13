"""Tests for the @-mention file-resolve flow.

When a researcher types ``@filename`` in the composer, the JS
dropdown calls two new bridge endpoints:

  - :meth:`NoraBridge.list_mentionable_files` for the search list
  - :meth:`NoraBridge.attach_session_file` to stage the chosen file
    without re-uploading

These tests pin the contract on the Python side. The JS side is
covered by ad-hoc smoke tests on the desktop app - there's no
headless web harness here.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A 1×1 transparent PNG - enough to exercise the vision-staging path
# without depending on an external image file.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAAS"
    "sJTYQAAAAASUVORK5CYII="
)


def _bridge(tmp_path: Path, names: list[str]) -> NoraBridge:
    cwd = tmp_path / "session"
    cwd.mkdir()
    for n in names:
        (cwd / n).write_text("placeholder", encoding="utf-8")
    return NoraBridge(cwd=cwd)


# ---------------------------------------------------------------------------
# list_mentionable_files
# ---------------------------------------------------------------------------

def test_list_mentionable_files_returns_all_kinds(tmp_path: Path) -> None:
    """The dropdown needs ALL session files to filter against -
    scripts, datasets, logs, graphs alike. Without a single source
    of truth a researcher can drop a CSV, see it in the Files panel,
    then have @-autocomplete come back empty."""
    bridge = _bridge(tmp_path, [
        "data.csv", "regression.py", "robustness.do",
        "fig.gph", "stata.log",
    ])
    res = bridge.list_mentionable_files()
    assert res["ok"] is True
    names = [f["name"] for f in res["files"]]
    assert set(names) == {
        "data.csv", "regression.py", "robustness.do",
        "fig.gph", "stata.log",
    }


def test_list_mentionable_files_strips_inline_image_bytes(tmp_path: Path) -> None:
    """The Files panel ships base64 PNG thumbs through the bridge
    so it can render an in-popup preview. The mention dropdown
    doesn't render images - shipping those bytes on every "@" would
    waste IPC. The lighter endpoint should drop ``data`` / ``mime``
    fields."""
    cwd = tmp_path / "s"
    cwd.mkdir()
    (cwd / "fig.png").write_bytes(_TINY_PNG)
    bridge = NoraBridge(cwd=cwd)

    full = bridge.list_session_files()
    light = bridge.list_mentionable_files()

    full_row = next(f for f in full["files"] if f["name"] == "fig.png")
    light_row = next(f for f in light["files"] if f["name"] == "fig.png")

    # The full listing carries thumbnail bytes for in-panel preview.
    assert "data" in full_row
    # The mention listing must NOT - it's a search index, not a gallery.
    assert "data" not in light_row
    assert "mime" not in light_row
    # But the descriptive fields the dropdown filters on still ride.
    assert light_row["kind"] == "graph"
    assert "size" in light_row


def test_list_mentionable_files_no_cwd_returns_empty() -> None:
    bridge = NoraBridge(cwd=None)
    res = bridge.list_mentionable_files()
    assert res == {"ok": True, "files": []}


# ---------------------------------------------------------------------------
# attach_session_file - script kind (preserved behaviour)
# ---------------------------------------------------------------------------

def test_attach_script_via_mention_inlines_content(tmp_path: Path) -> None:
    """Mentioning a script file should still inline its contents
    (same as the Files-panel attach button or a fresh drag-drop).
    The mention path must reuse the existing ``pending_script_attachments``
    list - otherwise the renderer's prefix builder won't see it."""
    bridge = _bridge(tmp_path, ["analysis.py"])
    (bridge.cwd / "analysis.py").write_text(
        "import pandas as pd\nols()\n", encoding="utf-8"
    )

    res = bridge.attach_session_file("analysis.py")

    assert res["ok"] is True
    assert res["kind"] == "script"
    assert any(
        a.get("name") == "analysis.py"
        and "import pandas" in a.get("content", "")
        for a in bridge._pending_script_attachments
    )
    # And NOT in the announcement-only list - scripts get the heavier
    # treatment (full content), not a name-only notice.
    assert "analysis.py" not in bridge._pending_mentioned_files


# ---------------------------------------------------------------------------
# attach_session_file - data / log / .gph announcement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kind", [
    ("panel.csv", "data"),
    ("survey.dta", "data"),
    ("snapshot.parquet", "data"),
    ("results.tsv", "data"),
    ("trace.log", "log"),
    ("session.smcl", "log"),
    ("fig.gph", "graph"),
])
def test_attach_data_log_and_gph_are_announcement_only(
    tmp_path: Path, name: str, kind: str,
) -> None:
    """Anything that ISN'T a script source file or a vision-eligible
    image should be announced by name only - the file is on disk and
    reachable via get_schema / expand_result. Inlining a 5M-row CSV
    or a 200K-line .log would torch the prompt budget."""
    bridge = _bridge(tmp_path, [name])
    res = bridge.attach_session_file(name)
    assert res["ok"] is True
    assert res["kind"] == kind
    assert bridge._pending_script_attachments == []
    assert bridge._pending_mentioned_images == []
    assert name in bridge._pending_mentioned_files


def test_attach_announcement_is_idempotent(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, ["panel.csv"])
    bridge.attach_session_file("panel.csv")
    res2 = bridge.attach_session_file("panel.csv")
    assert res2["ok"] is True
    assert res2["already_attached"] is True
    # Same name only appears once.
    assert bridge._pending_mentioned_files.count("panel.csv") == 1


# ---------------------------------------------------------------------------
# attach_session_file - image kind (vision)
# ---------------------------------------------------------------------------

def test_attach_png_stages_as_vision(tmp_path: Path) -> None:
    """Mentioning a PNG should ride the next turn as vision so the
    model can SEE the plot - not just be told a file with that name
    exists. Without this, the researcher saying "look at residuals.png"
    is empty unless they re-upload."""
    cwd = tmp_path / "s"
    cwd.mkdir()
    (cwd / "residuals.png").write_bytes(_TINY_PNG)
    bridge = NoraBridge(cwd=cwd)

    res = bridge.attach_session_file("residuals.png")

    assert res["ok"] is True
    assert res["kind"] == "image"
    images = bridge._pending_mentioned_images
    assert len(images) == 1
    img = images[0]
    assert img["name"] == "residuals.png"
    assert img["mime"] == "image/png"
    # Round-trips back to the original bytes.
    assert base64.b64decode(img["data"]) == _TINY_PNG
    # And the name also rides in the announcement notice so the
    # model has a textual handle.
    assert "residuals.png" in bridge._pending_mentioned_files


def test_attach_image_is_idempotent_by_name(tmp_path: Path) -> None:
    """Two clicks on the same row mustn't double-stage the image -
    the model would see the plot twice and the chip count would
    be wrong."""
    cwd = tmp_path / "s"
    cwd.mkdir()
    (cwd / "fig.png").write_bytes(_TINY_PNG)
    bridge = NoraBridge(cwd=cwd)

    bridge.attach_session_file("fig.png")
    res2 = bridge.attach_session_file("fig.png")
    assert res2["ok"] is True
    assert res2["already_attached"] is True
    assert len(bridge._pending_mentioned_images) == 1


def test_attach_image_refuses_oversized(tmp_path: Path) -> None:
    """5 MB cap matches the composer-drop limit - the on-disk copy
    is left alone so the researcher can still open it themselves."""
    cwd = tmp_path / "s"
    cwd.mkdir()
    huge = cwd / "huge.png"
    huge.write_bytes(b"\0" * (6 * 1024 * 1024))
    bridge = NoraBridge(cwd=cwd)

    res = bridge.attach_session_file("huge.png")
    assert res["ok"] is False
    assert "5 MB" in res["reason"]
    assert bridge._pending_mentioned_images == []


# ---------------------------------------------------------------------------
# attach_session_file - path safety preserved across the new dispatch
# ---------------------------------------------------------------------------

def test_attach_basenames_path_traversal_attempts(tmp_path: Path) -> None:
    """``../../../etc/hosts`` must not resolve outside the session,
    even though the dispatch is now wider (covers data + image + log
    + graph). The safety check used to live next to the script-only
    branch - it has to apply to the new branches too."""
    bridge = _bridge(tmp_path, ["legit.csv"])
    outside = tmp_path / "secret.csv"
    outside.write_text("secret", encoding="utf-8")

    res = bridge.attach_session_file("../secret.csv")
    assert res["ok"] is False
    assert "not found" in res["reason"].lower()
    assert bridge._pending_mentioned_files == []


# ---------------------------------------------------------------------------
# unstage_attachment cleans every pending list
# ---------------------------------------------------------------------------

def test_unstage_clears_mentioned_files_and_images(tmp_path: Path) -> None:
    """Dismissing a chip in the composer (× button) calls
    unstage_attachment. Before the mention surface, that only
    looked at script attachments. The new pending lists need the
    same hygiene - otherwise the researcher's "I removed this"
    gesture leaves the file silently rolling along."""
    cwd = tmp_path / "s"
    cwd.mkdir()
    (cwd / "panel.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (cwd / "fig.png").write_bytes(_TINY_PNG)
    bridge = NoraBridge(cwd=cwd)

    bridge.attach_session_file("panel.csv")
    bridge.attach_session_file("fig.png")
    assert "panel.csv" in bridge._pending_mentioned_files
    assert "fig.png" in bridge._pending_mentioned_files
    assert len(bridge._pending_mentioned_images) == 1

    bridge.unstage_attachment("panel.csv")
    assert "panel.csv" not in bridge._pending_mentioned_files
    # The image's announcement entry follows when the image goes.
    bridge.unstage_attachment("fig.png")
    assert bridge._pending_mentioned_images == []
    assert "fig.png" not in bridge._pending_mentioned_files


# ---------------------------------------------------------------------------
# @-mention extends the provenance manifest (folder-backed workflow)
# ---------------------------------------------------------------------------

def test_attach_session_file_marks_cwd_file_as_known(tmp_path: Path) -> None:
    """Folder-backed sessions are explicitly meant to be edited
    outside Nora: the researcher opens a project dir, closes Nora,
    adds ``analysis_v2.py`` in their editor, reopens Nora. The
    file_provenance manifest snapshots cwd ONLY on the first open
    (so sandbox-output written between sessions can't be silently
    promoted), which means externally-added files end up unknown to
    ``is_known`` even though they appear in the @-mention dropdown.

    Fix: @-mention IS an explicit researcher action — clicking a row
    in the dropdown vouches for the file. The bridge's
    ``attach_session_file`` now folds the cwd top-level target into
    the manifest, so a subsequent ``read_attached_file`` /
    ``submit_script_file`` against the same name passes the
    provenance gate. The model can't trigger @-mention, so this
    expansion can't be abused to launder sandbox-written files.
    """
    from nora.file_provenance import initialize, is_known

    cwd = tmp_path / "project"
    cwd.mkdir()
    # First-open snapshot: only this file exists; manifest pins it.
    (cwd / "original.py").write_text("print(1)", encoding="utf-8")
    initialize(cwd)

    # Between sessions: researcher adds a new file in their editor.
    new_file = cwd / "analysis_v2.py"
    new_file.write_text("import pandas\n", encoding="utf-8")

    # Pre-fix invariant: the new file is NOT in the manifest because
    # initialize is first-open-only. Without the @-mention hook, the
    # next read_attached_file would refuse it.
    assert not is_known(cwd, "analysis_v2.py"), (
        "test premise: a between-session add is not yet known"
    )

    bridge = NoraBridge(cwd=cwd)
    res = bridge.attach_session_file("analysis_v2.py")
    assert res["ok"] is True

    # After @-mention: the file IS known, so the provenance gate
    # downstream accepts it.
    assert is_known(cwd, "analysis_v2.py"), (
        "@-mention must mark the cwd target as known so a subsequent "
        "read_attached_file / submit_script_file isn't refused — that "
        "was the folder-backed-session breakage"
    )


def test_attach_session_file_does_not_mark_run_dir_files(tmp_path: Path) -> None:
    """The provenance manifest tracks cwd top-level only. Helper
    plots / run-dir scripts live under ``<cwd>/.nora/runs/<id>/`` and
    are out of scope for ``is_known`` (it's basename-keyed on cwd
    top-level). Don't pollute the manifest with run-dir filenames —
    a top-level file later created with the same basename would
    then be accepted without a researcher having vouched for it.
    """
    from nora.file_provenance import initialize, is_known, known_names

    cwd = tmp_path / "project"
    cwd.mkdir()
    initialize(cwd)

    # Stage a helper plot under a run dir, simulating a prior submit_script.
    run_dir = cwd / ".nora" / "runs" / "r0001"
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir(parents=True)
    plot = plots_dir / "residuals.png"
    plot.write_bytes(_TINY_PNG)

    bridge = NoraBridge(cwd=cwd)
    bridge.attach_session_file("residuals.png", path=str(plot))

    # The run-dir plot's basename must NOT enter the cwd manifest —
    # a future cwd-level write of ``residuals.png`` should NOT inherit
    # known-status from a same-named run-dir plot the researcher
    # @-mentioned.
    assert "residuals.png" not in known_names(cwd), (
        "@-mention of a run-dir file must not extend the cwd-top-level "
        "manifest — run-dir basenames overlap is common and would "
        "launder unknown cwd files"
    )
    assert not is_known(cwd, "residuals.png")

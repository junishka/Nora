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


# ---------------------------------------------------------------------------
# Plot files accumulate in the Files panel — both session-cwd writes
# (Stata `graph export`, ggsave/savefig with bare filenames) AND the
# manifest-allowlisted `<run>/_nora_plots/` outputs.
# ---------------------------------------------------------------------------

def test_list_session_files_includes_pngs_in_session_cwd(tmp_path: Path) -> None:
    """A `.png` written into the session cwd (Stata `graph export`,
    direct `plt.savefig("foo.png")`, etc.) should surface in the
    Files panel as a graph row."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "female_gap.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    names = [f["name"] for f in res["files"]]
    assert "female_gap.png" in names
    row = next(f for f in res["files"] if f["name"] == "female_gap.png")
    assert row["kind"] == "graph"


def test_list_session_files_walks_run_dir_nora_plots(tmp_path: Path) -> None:
    """Helper-produced plots live in
    ``<cwd>/.nora/runs/<id>/_nora_plots/`` — outside the session-cwd
    top-level scan. The Files panel walks those subdirs too so the
    panel is the persistent gallery for every plot the analysis ever
    produced."""
    cwd = tmp_path / "session"
    plots_dir = cwd / ".nora" / "runs" / "r0001" / "_nora_plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "residuals.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    names = [f["name"] for f in res["files"]]
    assert "residuals.png" in names


def test_list_session_files_inlines_thumbnail_for_small_images(
    tmp_path: Path,
) -> None:
    """Small image files carry inline base64 ``data`` so the panel
    can render thumbnails without a second round-trip. Above the
    cap, the row exists with ``path`` only — currently the cap is
    3 MB so a sharp 1600px Stata PNG lands inline. Anything truly
    huge falls through to the path-only branch."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    small = cwd / "small.png"
    small.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024)
    huge = cwd / "huge.png"
    # 5 MB > 3 MB cap, so this should fall through to path-only.
    huge.write_bytes(b"\x89PNG" + b"x" * (5 * 1024 * 1024))
    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    by_name = {f["name"]: f for f in res["files"]}
    assert "data" in by_name["small.png"]
    assert by_name["small.png"]["mime"] == "image/png"
    assert "data" not in by_name["huge.png"]
    assert by_name["huge.png"]["path"]


def test_list_session_files_sorts_graphs_newest_first(tmp_path: Path) -> None:
    """Within the graph kind, newer plots come first so the most
    recent output sits at the top of the panel — matching how
    researchers iterate (last plot is the one they care about)."""
    import os
    cwd = tmp_path / "session"
    cwd.mkdir()
    older = cwd / "older.png"
    newer = cwd / "newer.png"
    older.write_bytes(b"\x89PNG" + b"\x00" * 200)
    newer.write_bytes(b"\x89PNG" + b"\x00" * 200)
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    graph_names = [f["name"] for f in res["files"] if f["kind"] == "graph"]
    assert graph_names == ["newer.png", "older.png"]


# ---------------------------------------------------------------------------
# delete_session_file — Files-panel trash icon
# ---------------------------------------------------------------------------

def test_delete_session_file_unlinks_top_level_file(tmp_path: Path) -> None:
    """The simple case: a file in session_cwd is unlinked when the
    researcher clicks the trash icon. The path is verified to be
    inside session_cwd before any unlink — outside paths refused."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    target = cwd / "coef_plot.pdf"
    target.write_bytes(b"%PDF-1.4")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.delete_session_file(str(target))
    assert res["ok"] is True
    assert res["name"] == "coef_plot.pdf"
    assert not target.exists()


def test_delete_session_file_unlinks_run_dir_plot(tmp_path: Path) -> None:
    """Helper-produced plots live under
    ``<cwd>/.nora/runs/<id>/_nora_plots/``. The Files panel surfaces
    them with full paths; delete must accept those paths and unlink
    them too. Without this the trash icon would silently no-op for
    every helper-produced graph."""
    cwd = tmp_path / "session"
    plots = cwd / ".nora" / "runs" / "r0001" / "_nora_plots"
    plots.mkdir(parents=True)
    target = plots / "residuals.png"
    target.write_bytes(b"\x89PNG fake")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.delete_session_file(str(target))
    assert res["ok"] is True
    assert not target.exists()


def test_delete_session_file_refuses_path_outside_cwd(tmp_path: Path) -> None:
    """Defense in depth: a path outside the session is refused
    with no unlink. Prevents a malformed JS caller (or a future
    rendering bug) from deleting arbitrary files."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("sensitive")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.delete_session_file(str(outside))
    assert res["ok"] is False
    assert "outside" in res["reason"]
    assert outside.exists()


def test_delete_session_file_drops_pending_attachment(tmp_path: Path) -> None:
    """If the researcher staged a script for attachment and then
    deletes the file, the pending-attachment chip must vanish too —
    otherwise the next send would silently skip the inline content
    and the chip would lie about what's about to be sent."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    script = cwd / "regression.do"
    script.write_text("regress y x")
    bridge = NoraBridge(cwd=cwd)
    runner = bridge._active_runner()
    assert runner is not None
    runner.pending_script_attachments = [
        {"name": "regression.do", "ext": ".do", "content": "regress y x"},
    ]
    res = bridge.delete_session_file(str(script))
    assert res["ok"] is True
    assert runner.pending_script_attachments == []


def test_delete_session_file_also_removes_pdf_png_sidecar(tmp_path: Path) -> None:
    """When a PDF is deleted, its cached ``.nora.png`` sidecar from
    the sips conversion path must also be removed — otherwise the
    Files panel would still show an orphan PNG with no source."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    pdf = cwd / "fig.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    sidecar = cwd / "fig.nora.png"
    sidecar.write_bytes(b"\x89PNG cached")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.delete_session_file(str(pdf))
    assert res["ok"] is True
    assert not pdf.exists()
    assert not sidecar.exists()


def test_delete_session_file_no_active_session_returns_clean_error() -> None:
    bridge = NoraBridge(cwd=None)
    res = bridge.delete_session_file("/anywhere")
    assert res["ok"] is False
    assert "no active session" in res["reason"]


# ---------------------------------------------------------------------------
# Files panel renders graphs first
# ---------------------------------------------------------------------------

def test_files_panel_groups_graphs_first() -> None:
    """The render order is ['graph', 'script', 'log'] so plots —
    the most-clicked output — sit at the top instead of being
    buried under script rows. Pin the order in source so a future
    refactor doesn't quietly bury graphs again."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent
           / "src" / "nora" / "web" / "app.js").read_text(encoding="utf-8")
    # The forEach lists the order. Anchor on the literal array.
    needle = "['graph', 'script', 'log']"
    assert needle in src, (
        "Files panel must render graphs first; the render-order "
        "literal in app.js is the single source of truth"
    )


def test_files_popup_has_taller_max_height() -> None:
    """The Files dropdown was clipped at the original popup height
    (a few rows). After 10+ iterations the researcher had to
    inner-scroll to see anything past the first few. Pin the new
    explicit max-height so a future style refactor doesn't shrink
    it back."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent
           / "src" / "nora" / "web" / "style.css").read_text(encoding="utf-8")
    assert ".files-popup" in src
    # 70vh keeps it tall but not full-screen.
    assert "max-height: 70vh" in src

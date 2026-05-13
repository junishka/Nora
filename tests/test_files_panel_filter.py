"""Tests for the Files-panel filter that hides ``submit_script``-
produced clutter from the researcher's view.

The contract: the panel shows files the researcher would expect to
find in their working directory — uploads, pre-existing files, and
files they explicitly placed there. Files a script wrote in passing
(``ggsave("debug.png")``, ``write.csv("tmp.csv")``,
``<run_dir>/script.do``, ``<run_dir>/_nora_plots/*.png``) are
intentionally hidden because they're already represented inline on
the script's result card.

Two mechanisms cooperate:
- ``include_run_scripts=False`` + ``include_run_plots=False`` skip
  the run-dir traversal entirely.
- ``exclude_script_writes=True`` drops cwd top-level files tagged
  by the executor's ``cwd_writes.json`` manifest.

A tag de-applies when the on-disk file's (mtime, size) no longer
matches the manifest — so a researcher who overwrites or replaces
a script-written file makes it visible again.
"""

from __future__ import annotations

import json
from pathlib import Path

from nora.session_files import (
    enumerate_session_files,
    script_written_cwd_files,
)


def _make_cwd_writes_manifest(
    cwd: Path,
    run_id: str,
    entries: list[tuple[str, float, int]],
) -> Path:
    """Write a fake ``cwd_writes.json`` under a synthetic run dir.
    Each entry is ``(name, mtime, size)`` — the same shape the
    executor's ``_write_cwd_writes_manifest`` produces.
    """
    run_dir = cwd / ".nora" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"name": n, "mtime": m, "size": s} for n, m, s in entries]
    (run_dir / "cwd_writes.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    return run_dir


def test_script_written_filter_hides_matching_cwd_file(tmp_path: Path) -> None:
    """A file in cwd that matches a ``cwd_writes.json`` entry's
    (name, mtime, size) is excluded from the panel listing."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    plot = cwd / "debug.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    st = plot.stat()
    _make_cwd_writes_manifest(
        cwd, "r1", [("debug.png", st.st_mtime, st.st_size)],
    )

    names_full = [
        f["name"] for f in enumerate_session_files(cwd, include_data=True)
    ]
    assert "debug.png" in names_full, (
        "without exclude_script_writes the file is visible"
    )

    names_panel = [
        f["name"] for f in enumerate_session_files(
            cwd,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
            exclude_script_writes=True,
        )
    ]
    assert "debug.png" not in names_panel, (
        "with exclude_script_writes the script-produced plot is hidden"
    )


def test_script_written_filter_releases_on_mtime_change(tmp_path: Path) -> None:
    """If the researcher overwrites a script-written file, the on-disk
    (mtime, size) no longer matches the manifest and the file
    reappears in the panel. This is the recovery path for "Nora wrote
    a file but I want to claim it as my deliverable."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    artifact = cwd / "model_output.csv"
    artifact.write_text("orig", encoding="utf-8")
    st_old = artifact.stat()
    _make_cwd_writes_manifest(
        cwd, "r1", [("model_output.csv", st_old.st_mtime, st_old.st_size)],
    )
    # Researcher overwrites with their own content (different size →
    # different (mtime, size) tuple).
    artifact.write_text("researcher's curated version", encoding="utf-8")

    names = [
        f["name"] for f in enumerate_session_files(
            cwd,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
            exclude_script_writes=True,
        )
    ]
    assert "model_output.csv" in names, (
        "overwriting de-tags the file — it should reappear in the panel"
    )


def test_script_written_filter_off_by_default(tmp_path: Path) -> None:
    """``exclude_script_writes=False`` is the default. Model-facing
    callers and any caller that doesn't opt in get the full picture.
    This protects the model-facing tool from being silently
    narrowed."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    f = cwd / "out.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    st = f.stat()
    _make_cwd_writes_manifest(
        cwd, "r1", [("out.png", st.st_mtime, st.st_size)],
    )

    names = [f["name"] for f in enumerate_session_files(cwd, include_data=True)]
    assert "out.png" in names


def test_include_run_plots_false_skips_nora_plots_dir(tmp_path: Path) -> None:
    """Files panel mode (``include_run_plots=False``) does not walk
    the per-run ``_nora_plots/`` directories. Those plots already
    render inline in their result cards."""
    cwd = tmp_path / "session"
    plots = cwd / ".nora" / "runs" / "r0001" / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "residuals.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    panel = [
        f["name"] for f in enumerate_session_files(
            cwd,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
        )
    ]
    assert "residuals.png" not in panel

    # Full view still surfaces it.
    full = [
        f["name"] for f in enumerate_session_files(
            cwd, include_data=True, include_run_plots=True,
        )
    ]
    assert "residuals.png" in full


def test_include_run_scripts_false_skips_run_dir_scripts(tmp_path: Path) -> None:
    """Files panel mode (``include_run_scripts=False``) does not walk
    ``<run_dir>/script.{do,R,py,ipynb}`` — those scripts are already
    accessible from their result card's "Open in Stata/R/Python"
    button."""
    cwd = tmp_path / "session"
    run_dir = cwd / ".nora" / "runs" / "20260507T120000Z_aaaaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// hi", encoding="utf-8")

    panel = [
        f["name"] for f in enumerate_session_files(
            cwd,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
        )
    ]
    assert not any(n.endswith(".do") for n in panel), panel


def test_researcher_uploads_pass_through(tmp_path: Path) -> None:
    """Files the researcher placed in cwd directly (composer uploads,
    pre-existing work) are not tagged in any ``cwd_writes.json`` and
    so always appear in the panel. The whole point of the filter is
    to leave these visible."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "panel.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (cwd / "intro.py").write_text("# hi", encoding="utf-8")
    # An unrelated submit_script run that wrote a different file.
    other = cwd / "scratch.png"
    other.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    st = other.stat()
    _make_cwd_writes_manifest(
        cwd, "r1", [("scratch.png", st.st_mtime, st.st_size)],
    )

    names = [
        f["name"] for f in enumerate_session_files(
            cwd,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
            exclude_script_writes=True,
        )
    ]
    assert "panel.csv" in names
    assert "intro.py" in names
    assert "scratch.png" not in names


def test_script_written_helper_unions_across_multiple_runs(
    tmp_path: Path,
) -> None:
    """The cwd-writes manifests from every run get unioned. A
    researcher's panel reflects every script's output history, not
    just the latest run."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    a = cwd / "a.png"
    a.write_bytes(b"a" * 50)
    b = cwd / "b.csv"
    b.write_text("x,y\n", encoding="utf-8")
    sa = a.stat()
    sb = b.stat()
    _make_cwd_writes_manifest(cwd, "r1", [("a.png", sa.st_mtime, sa.st_size)])
    _make_cwd_writes_manifest(cwd, "r2", [("b.csv", sb.st_mtime, sb.st_size)])

    written = script_written_cwd_files(cwd)
    assert written == {"a.png", "b.csv"}


def test_script_written_helper_ignores_bad_manifest(tmp_path: Path) -> None:
    """Malformed or unparseable ``cwd_writes.json`` files don't crash
    the panel — they're skipped, the rest of the union proceeds, and
    files from those runs default to visible (fail-open: more
    visible, not fewer)."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    good = cwd / "good.csv"
    good.write_text("x\n", encoding="utf-8")
    st = good.stat()
    _make_cwd_writes_manifest(cwd, "r1", [("good.csv", st.st_mtime, st.st_size)])

    bad_run = cwd / ".nora" / "runs" / "r2"
    bad_run.mkdir(parents=True)
    (bad_run / "cwd_writes.json").write_text("not json {{{", encoding="utf-8")

    written = script_written_cwd_files(cwd)
    assert written == {"good.csv"}, (
        "the malformed manifest is skipped; the good one still tags its files"
    )


def test_executor_snapshot_excludes_nora_subtree_and_dotfiles(
    tmp_path: Path,
) -> None:
    """The cwd snapshot must not include ``.nora/`` (the per-session
    store and run dirs) or other dotfiles — otherwise every run's
    diff would include phantom 'changes' to internal state."""
    from nora.executor import _snapshot_cwd_top_level

    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "data.csv").write_text("x\n", encoding="utf-8")
    (cwd / ".DS_Store").write_bytes(b"\x00\x01")
    (cwd / ".nora").mkdir()
    (cwd / ".nora" / "results.db").write_text("placeholder", encoding="utf-8")

    snap = _snapshot_cwd_top_level(cwd)
    assert "data.csv" in snap
    assert ".DS_Store" not in snap
    assert ".nora" not in snap


def test_executor_diff_writes_manifest_for_new_files(tmp_path: Path) -> None:
    """The diff helper writes a manifest only when files were added or
    modified; an unchanged cwd produces no manifest."""
    from nora.executor import _snapshot_cwd_top_level, _write_cwd_writes_manifest

    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "existing.csv").write_text("x\n", encoding="utf-8")
    pre = _snapshot_cwd_top_level(cwd)

    run_dir = cwd / ".nora" / "runs" / "r1"
    run_dir.mkdir(parents=True)

    # Script writes a new file.
    (cwd / "fresh.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
    _write_cwd_writes_manifest(cwd, run_dir, pre)

    manifest_path = run_dir / "cwd_writes.json"
    assert manifest_path.is_file()
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {r["name"] for r in rows}
    assert "fresh.png" in names
    assert "existing.csv" not in names, (
        "unchanged files don't enter the manifest"
    )


def test_executor_diff_no_writes_no_manifest(tmp_path: Path) -> None:
    """If the script wrote nothing to cwd, no manifest gets created.
    Empty manifests would clutter ``.nora/runs/`` for no benefit."""
    from nora.executor import _snapshot_cwd_top_level, _write_cwd_writes_manifest

    cwd = tmp_path / "session"
    cwd.mkdir()
    (cwd / "in.csv").write_text("x\n", encoding="utf-8")
    pre = _snapshot_cwd_top_level(cwd)

    run_dir = cwd / ".nora" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    # Script ran but didn't touch cwd.
    _write_cwd_writes_manifest(cwd, run_dir, pre)

    assert not (run_dir / "cwd_writes.json").exists()

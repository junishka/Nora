"""Tests for ``NoraBridge.list_session_files`` and the
``enumerate_session_files`` walk it delegates to.

The Files chip in the topbar reads from the bridge endpoint. The
bridge runs in researcher-mode: it hides files that already render
on a result card (run-dir scripts, ``_nora_plots/`` helper outputs)
and files that a ``submit_script`` run produced in cwd (per its
``cwd_writes.json`` manifest). The underlying walk preserves a
full-view mode for the model-facing tool and for tests that pin
labeling / traversal behavior — those call ``enumerate_session_files``
directly with ``include_run_scripts=True, include_run_plots=True``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora.session_files import enumerate_session_files
from nora.ui import NoraBridge


def _full_view(cwd: Path) -> list[dict]:
    """Listing with run-dir scripts and helper-plot traversal —
    matches what the model sees and what the bridge USED to show
    before the panel filter. Tests that pin the labeling / traversal
    contract use this so a tighter panel default doesn't break them.
    """
    return enumerate_session_files(
        cwd,
        include_data=True,
        include_run_scripts=True,
        include_run_plots=True,
    )


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


def test_attach_session_file_announces_data_extensions(tmp_path: Path) -> None:
    """Data files don't get inlined as text. That would blow up the
    prompt for a multi-MB CSV. They DO get added to the @-mention
    notice for the next turn, so the model knows the researcher is
    pointing at this specific file (vs the generic dataset listing
    in the system prompt)."""
    bridge = _bridge_with_files(tmp_path, ["panel.parquet"])

    res = bridge.attach_session_file("panel.parquet")
    assert res["ok"] is True
    assert res["kind"] == "data"
    assert bridge._pending_script_attachments == []
    assert "panel.parquet" in bridge._pending_mentioned_files


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


def test_enumerate_walks_run_dir_nora_plots_in_full_view(tmp_path: Path) -> None:
    """Helper-produced plots live in
    ``<cwd>/.nora/runs/<id>/_nora_plots/`` — outside the session-cwd
    top-level scan. The full-view enumeration (model-facing tool,
    audit paths) walks those subdirs so the analysis-wide plot
    gallery is reachable. The Files panel itself sets
    ``include_run_plots=False`` because those plots already render
    on their result cards."""
    cwd = tmp_path / "session"
    plots_dir = cwd / ".nora" / "runs" / "r0001" / "_nora_plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "residuals.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    names = [f["name"] for f in _full_view(cwd)]
    assert "residuals.png" in names


def test_list_session_files_surfaces_script_with_label(tmp_path: Path) -> None:
    """``submit_script`` writes ``<cwd>/.nora/runs/<id>/script.do`` and
    inserts a result row whose ``label`` describes the analytic intent
    ("reg_v12 2v common sample"). The Files panel surfaces the script
    under that label so the researcher can identify which script is
    which without opening each one. Falls back to ``script_<short_id>``
    when no row exists for the run dir (script crashed before any
    helper fired)."""
    cwd = tmp_path / "session"
    runs = cwd / ".nora" / "runs"
    labeled_run = runs / "20260507T120000Z_aaaaaaaa"
    labeled_run.mkdir(parents=True)
    (labeled_run / "script.do").write_text("// labeled", encoding="utf-8")
    bare_run = runs / "20260507T120100Z_bbbbbbbb"
    bare_run.mkdir(parents=True)
    (bare_run / "script.do").write_text("// no label", encoding="utf-8")

    from nora.store import get_store
    get_store(cwd).insert(
        label="reg_v12 2v common sample: op_margin + ln_employees only",
        analysis_type="linear_regression",
        sanitized_payload={"type": "linear_regression"},
        language="Stata",
        script_code="// labeled",
        transformations=[],
        raw_log_path=str(labeled_run),
        script_run_id="run-aaaaaaaa",
    )

    names = [f["name"] for f in _full_view(cwd)]
    assert (
        "reg_v12 2v common sample: op_margin + ln_employees only.do" in names
    ), names
    assert "script_bbbbbbbb.do" in names, names


def test_list_session_files_label_lookup_survives_resolved_cwd(
    tmp_path: Path,
) -> None:
    """The store records ``raw_log_path`` as ``str(run_dir)`` from
    whatever cwd representation the executor held. If the bridge is
    later constructed with a resolved (or unresolved) cwd that differs
    by a symlink prefix, a full-path key would silently miss every
    row. Keying by the run dir's basename insulates the panel from
    that drift — exercise the case explicitly."""
    real_cwd = (tmp_path / "real").resolve()
    real_cwd.mkdir()
    link_cwd = tmp_path / "session_link"
    link_cwd.symlink_to(real_cwd)

    runs = real_cwd / ".nora" / "runs"
    run_dir = runs / "20260507T120000Z_cccccccc"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// via real path", encoding="utf-8")

    # Insert via the resolved path...
    from nora.store import get_store
    get_store(real_cwd).insert(
        label="H2 panel main",
        analysis_type="linear_regression",
        sanitized_payload={"type": "linear_regression"},
        language="Stata",
        script_code="// real",
        transformations=[],
        raw_log_path=str(run_dir),  # resolved cwd
        script_run_id="run-cccccccc",
    )
    # ...but list via the symlinked cwd. A naive walk would otherwise
    # see "/tmp/.../session_link/.nora/runs/<id>" which doesn't match
    # the stored "/tmp/.../real/.nora/runs/<id>".
    names = [f["name"] for f in _full_view(link_cwd)]
    assert "H2 panel main.do" in names, names


def test_list_session_files_prefers_run_dir_label_txt_over_per_helper(
    tmp_path: Path,
) -> None:
    """A multi-result ``submit_script`` writes umbrella + per-helper
    labels — the umbrella to ``<run_dir>/label.txt``, each per-helper
    label as the row's ``label`` in ``results.db``. The Files panel
    must surface the umbrella, not the first per-helper label.

    Without this, a 20-regression script labeled
    ``reg_v16: H1/H2/H3, Path A and Path B`` shows up as
    ``Path A H1: operating_margin.do`` (the first cell), which
    misleads the researcher about what the script does."""
    cwd = tmp_path / "session"
    runs = cwd / ".nora" / "runs"
    run_dir = runs / "20260507T120000Z_eeeeeeee"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// 20 specs", encoding="utf-8")
    (run_dir / "label.txt").write_text(
        "reg_v16: H1/H2/H3, Path A and Path B", encoding="utf-8",
    )

    from nora.store import get_store
    store = get_store(cwd)
    # Per-helper labels — what each cell would carry. None of these
    # should land in the file panel as the script name.
    for label in (
        "Path A H1: operating_margin",
        "Path A H1: asset_turnover",
        "Path A H2: ln_employees",
    ):
        store.insert(
            label=label,
            analysis_type="linear_regression",
            sanitized_payload={"type": "linear_regression"},
            language="Stata",
            script_code="// 20 specs",
            transformations=[],
            raw_log_path=str(run_dir),
            script_run_id="run-eeeeeeee",
        )

    names = [f["name"] for f in _full_view(cwd)]
    # Forward slashes in the umbrella label collapse to spaces under
    # ``label_to_filename_stem`` (path-character hygiene) — that's
    # expected. The umbrella structure survives, which is what the
    # researcher actually reads.
    assert "reg_v16: H1 H2 H3, Path A and Path B.do" in names, names
    # And the per-helper labels must NOT be the surfaced file name.
    for cell_label in (
        "Path A H1: operating_margin.do",
        "Path A H1: asset_turnover.do",
    ):
        assert cell_label not in names, (
            f"per-helper label {cell_label!r} surfaced as script name"
        )


def test_list_session_files_falls_back_to_store_when_no_label_txt(
    tmp_path: Path,
) -> None:
    """Backwards compat: runs created before label.txt was written
    have no umbrella file, so the panel still uses the first
    per-helper row label as the name. Without this fallback, every
    pre-existing session would show only ``script_<short_id>.do``
    after a Nora upgrade."""
    cwd = tmp_path / "session"
    runs = cwd / ".nora" / "runs"
    run_dir = runs / "20260507T120100Z_ffffffff"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// legacy", encoding="utf-8")
    # No label.txt — the legacy path.

    from nora.store import get_store
    get_store(cwd).insert(
        label="OLS log salary on female",
        analysis_type="linear_regression",
        sanitized_payload={"type": "linear_regression"},
        language="Stata",
        script_code="// legacy",
        transformations=[],
        raw_log_path=str(run_dir),
        script_run_id="run-ffffffff",
    )

    names = [f["name"] for f in _full_view(cwd)]
    assert "OLS log salary on female.do" in names, names


def test_list_session_files_skips_only_truly_empty_labels_for_first_pick(
    tmp_path: Path,
) -> None:
    """A run can produce an ``(unlabeled)`` row before a real helper
    label lands; the cleaning step strips that placeholder to empty,
    so the map-build walks past it and picks up the next row's label.
    Diagnostic prefixes that carry real content (``[rejected] X``)
    are kept as-is — the bracket tag is informative and stripping
    both would leave a crashed run indistinguishable from any other.
    This test pins both behaviors at once: ``(unlabeled)`` is skipped,
    a real label following it surfaces."""
    cwd = tmp_path / "session"
    runs = cwd / ".nora" / "runs"
    run_dir = runs / "20260507T120000Z_dddddddd"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// mixed", encoding="utf-8")

    from nora.store import get_store
    store = get_store(cwd)
    # First row: placeholder (model omitted ``label`` and the helper
    # didn't pass its own). Cleans to empty, so it's skipped.
    # Second row: a real helper label. This is what should show.
    for label in ("(unlabeled)", "H1a Path A: op margin, FP-only"):
        store.insert(
            label=label,
            analysis_type="linear_regression",
            sanitized_payload={"type": "linear_regression"},
            language="Stata",
            script_code="// mixed",
            transformations=[],
            raw_log_path=str(run_dir),
            script_run_id="run-dddddddd",
        )

    names = [f["name"] for f in _full_view(cwd)]
    assert "H1a Path A: op margin, FP-only.do" in names, names


def test_list_session_files_disambiguates_label_collisions(
    tmp_path: Path,
) -> None:
    """Two runs sharing a label still need to be distinguishable in
    the panel. Collisions get the run's short_id appended in
    parens before the extension."""
    cwd = tmp_path / "session"
    runs = cwd / ".nora" / "runs"
    run_a = runs / "20260507T120000Z_aaaaaaaa"
    run_b = runs / "20260507T120100Z_bbbbbbbb"
    for r in (run_a, run_b):
        r.mkdir(parents=True)
        (r / "script.do").write_text("// dup label", encoding="utf-8")

    from nora.store import get_store
    store = get_store(cwd)
    for r, sid in ((run_a, "run-aaaaaaaa"), (run_b, "run-bbbbbbbb")):
        store.insert(
            label="H1 panel",
            analysis_type="linear_regression",
            sanitized_payload={"type": "linear_regression"},
            language="Stata",
            script_code="// dup",
            transformations=[],
            raw_log_path=str(r),
            script_run_id=sid,
        )

    names = sorted(f["name"] for f in _full_view(cwd))
    assert names == ["H1 panel (aaaaaaaa).do", "H1 panel (bbbbbbbb).do"], names


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


def test_delete_session_file_refuses_run_dir_plot(tmp_path: Path) -> None:
    """Helper-produced plots under ``<cwd>/.nora/runs/<id>/_nora_plots/``
    are NOT in the Files panel listing (the panel calls
    ``enumerate_session_files`` with ``include_run_plots=False`` —
    those plots already render on the result card). The bridge
    gate mirrors the panel: a delete call against a run-dir plot
    has no legitimate UI origin (the only JS caller is the panel
    trash icon, which can't fire for a row that isn't rendered)
    and is refused. The disk file stays put.

    Regression for the gap where ``read_session_file_text`` /
    ``delete_session_file`` used the broader ``include_run_plots=True``
    enumeration and ended up reachable through the bridge even
    though no panel row could trigger them."""
    cwd = tmp_path / "session"
    plots = cwd / ".nora" / "runs" / "r0001" / "_nora_plots"
    plots.mkdir(parents=True)
    target = plots / "residuals.png"
    target.write_bytes(b"\x89PNG fake")
    bridge = NoraBridge(cwd=cwd)
    res = bridge.delete_session_file(str(target))
    assert res["ok"] is False
    assert "panel listing" in res["reason"]
    assert target.exists()


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
# read_session_file_text — Files-panel "copy" button
# ---------------------------------------------------------------------------

def test_read_session_file_text_refuses_unlisted_run_log(tmp_path: Path) -> None:
    """``stdout.log`` / ``stderr.log`` live under ``.nora/runs/<id>/``
    but the Files panel intentionally never lists them — the
    executor's contract is that raw subprocess transcripts never
    cross to the model. Containment in cwd alone is not enough to
    let a page-rendered JS caller pull them onto the page (same
    threat model that ``delete_session_file`` defends against).
    ``read_session_file_text`` must mirror that gate: only files
    the Files panel surfaces are readable through the bridge.
    """
    cwd = tmp_path / "session"
    run_dir = cwd / ".nora" / "runs" / "r0001"
    run_dir.mkdir(parents=True)
    raw_log = run_dir / "stdout.log"
    raw_log.write_text("subject 17 PII leak in raw output", encoding="utf-8")

    bridge = NoraBridge(cwd=cwd)
    res = bridge.read_session_file_text(str(raw_log))
    assert res["ok"] is False
    # The reason should point at the listing gate, not the
    # extension gate — .log IS in the allowed-extension set, so a
    # caller catching "wrong extension" would miss this regression.
    assert "Files panel listing" in res["reason"]


def test_delete_session_file_refuses_run_dir_script(
    tmp_path: Path,
) -> None:
    """Run-dir scripts (under ``<cwd>/.nora/runs/<id>/script.<ext>``)
    are not in the Files panel listing — the panel calls
    ``enumerate_session_files`` with ``include_run_scripts=False``
    so the script row only renders on its result card, not in the
    panel. The bridge gate mirrors the panel: a delete call
    against a run-dir script path is refused.

    For un-staging the script from a composer chip (the
    legitimate UI flow), JS calls ``unstage_attachment`` instead;
    that path is unaffected by the file-copy gate. See
    ``test_unstage_attachment_*`` for that contract.
    """
    cwd = tmp_path / "session"
    run_dir = cwd / ".nora" / "runs" / "20260511T120000Z_abcdef01"
    run_dir.mkdir(parents=True)
    script_on_disk = run_dir / "script.py"
    script_on_disk.write_text("print('hi')\n", encoding="utf-8")
    bridge = NoraBridge(cwd=cwd)

    res = bridge.delete_session_file(str(script_on_disk))
    assert res["ok"] is False
    assert "panel listing" in res["reason"]
    # File is untouched.
    assert script_on_disk.exists()


def test_delete_session_file_unstaged_handles_top_level_basename(
    tmp_path: Path,
) -> None:
    """Top-level scripts have name == on-disk basename, so the
    pre-fix behaviour was already correct for that case. Pin it:
    delete must still drop the staged entry and surface the name
    in ``unstaged``."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    script = cwd / "regression.py"
    script.write_text("import pandas as pd\n", encoding="utf-8")

    bridge = NoraBridge(cwd=cwd)
    bridge.attach_session_file("regression.py")
    assert len(bridge._pending_script_attachments) == 1

    res = bridge.delete_session_file(str(script))
    assert res["ok"] is True
    assert bridge._pending_script_attachments == []
    assert res.get("unstaged") == ["regression.py"]


def test_delete_session_file_refuses_run_dir_mentioned_image(
    tmp_path: Path,
) -> None:
    """@-mentioned helper plots in ``_nora_plots/`` are not in the
    Files panel listing. A delete call against one is refused by
    the bridge gate even when the path is staged for attachment;
    un-staging from the composer chip uses ``unstage_attachment``
    (by name), which is a separate code path.

    The previous behaviour accepted such deletes via the broader
    enumeration that the bridge gate used. Tightening the gate to
    mirror the panel closes that path. The trade-off: un-staging
    plot A specifically (when two plots share a basename) isn't
    expressible through ``unstage_attachment`` alone — that's a
    separate composer-chip identity issue, not a file-copy-gate
    issue.
    """
    cwd = tmp_path / "session"
    run_a = cwd / ".nora" / "runs" / "20260511T100000Z_aaaaaaaa" / "_nora_plots"
    run_a.mkdir(parents=True)
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    plot_a = run_a / "coefficients.png"
    plot_a.write_bytes(png_bytes)

    bridge = NoraBridge(cwd=cwd)
    bridge.attach_session_file("coefficients.png", str(plot_a))
    assert len(bridge._active_runner().pending_mentioned_images) == 1

    res = bridge.delete_session_file(str(plot_a))
    assert res["ok"] is False
    assert "panel listing" in res["reason"]
    # Staged entry is untouched (un-staging uses unstage_attachment).
    assert len(bridge._active_runner().pending_mentioned_images) == 1
    # And the disk file is untouched too.
    assert plot_a.exists()


def test_attach_session_file_path_resolves_exact_helper_plot(
    tmp_path: Path,
) -> None:
    """The mention dropdown surfaces two rows when two run dirs
    produce the same plot basename (``coefficients.png``). The JS
    side now passes ``item.path`` to ``attach_session_file``;
    without it the backend's ``iterdir`` walk staged whichever run
    dir the filesystem returned first, not the row the user
    clicked. Verify both that the path picks the right plot AND
    that the name-only fallback still resolves *something* for
    back-compat."""
    cwd = tmp_path / "session"
    run_a = cwd / ".nora" / "runs" / "20260511T100000Z_aaaaaaaa" / "_nora_plots"
    run_b = cwd / ".nora" / "runs" / "20260511T110000Z_bbbbbbbb" / "_nora_plots"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    plot_a = run_a / "coefficients.png"
    plot_b = run_b / "coefficients.png"
    plot_a.write_bytes(b"\x89PNG run A bytes")
    plot_b.write_bytes(b"\x89PNG run B bytes")

    bridge = NoraBridge(cwd=cwd)
    res = bridge.attach_session_file("coefficients.png", str(plot_b))
    assert res["ok"] is True
    staged = bridge._active_runner().pending_mentioned_images
    assert len(staged) == 1
    assert staged[0].get("path") == str(plot_b.resolve()), (
        "explicit path must override the iterdir walk and pin "
        "the staged entry to the exact row clicked"
    )


def test_attach_session_file_path_refuses_outside_cwd(
    tmp_path: Path,
) -> None:
    """A caller-supplied path must still pass containment. A
    malicious or buggy page-rendered JS that hands the bridge a
    path outside cwd must be refused — same posture as
    ``delete_session_file``."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("# secrets\n", encoding="utf-8")

    bridge = NoraBridge(cwd=cwd)
    # ``name`` is a basename that doesn't exist in cwd; ``path``
    # points outside. Both gates must combine to reject.
    res = bridge.attach_session_file("secret.py", str(outside))
    assert res["ok"] is False
    assert bridge._pending_script_attachments == []


def test_record_user_message_persists_mentioned_files_and_images(
    tmp_path: Path,
) -> None:
    """Before the fix, ``_record_user_message`` only persisted
    ``pending_script_attachments`` plus direct composer images. A
    researcher who @-mentioned a CSV (notice ride-along) and a
    PNG (vision ride-along) saw chips and thumbnails in the live
    UI, but a reload or session switch replayed a bare text
    bubble — the durable transcript no longer matched what the
    model actually saw.

    Pin the unified persistence: ``attachments`` carries every
    chip name the live UI showed (scripts + mention notices +
    mention-image names) and ``images`` carries direct composer
    blobs PLUS the @-mention vision blobs."""
    cwd = tmp_path / "session"
    cwd.mkdir()

    bridge = NoraBridge(cwd=cwd)
    runner = bridge._active_runner()
    assert runner is not None

    # Simulate a turn's worth of staged state, the way attach_*
    # paths would have populated it before send. (We bypass the
    # actual attach so the test stays focused on the persistence
    # contract, not the staging plumbing.)
    runner.pending_script_attachments = [
        {
            "name": "regression.py",
            "ext": ".py",
            "content": "import pandas as pd\n",
            "bytes": 22,
            "path": str((cwd / "regression.py").resolve()),
        },
    ]
    runner.pending_mentioned_files = ["panel.csv"]
    runner.pending_mentioned_images = [
        {
            "name": "coefficients.png",
            "data": "ZmFrZQ==",  # base64 "fake"
            "mime": "image/png",
            "path": str((cwd / ".nora" / "runs" / "r1" / "_nora_plots"
                         / "coefficients.png").resolve()),
        },
    ]

    composer_image = {
        "name": "screenshot.png",
        "data": "ZHJvcA==",  # base64 "drop"
        "mime": "image/png",
    }
    bridge._record_user_message(
        runner, "explain these", images=[composer_image],
    )

    # Read what landed on disk.
    import json
    hist = (cwd / ".nora" / "chat_history.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in hist.splitlines() if line.strip()]
    user_records = [r for r in records if r.get("type") == "user_message"]
    assert len(user_records) == 1
    rec = user_records[0]

    assert rec.get("text") == "explain these"
    # ``attachments`` is the unified chip list.
    attachments = rec.get("attachments") or []
    assert "regression.py" in attachments
    assert "panel.csv" in attachments, (
        "@-mentioned non-image files must be persisted so the "
        "transcript reflects what the model was pointed at"
    )
    assert "coefficients.png" in attachments

    # ``images`` carries both the direct composer image AND the
    # mentioned-image vision blob.
    images = rec.get("images") or []
    datas = {i.get("data") for i in images}
    assert "ZHJvcA==" in datas, "direct composer image must persist"
    assert "ZmFrZQ==" in datas, (
        "@-mentioned vision blob must persist — without it a "
        "reload replays the bubble with no thumbnail even "
        "though the model received vision input"
    )
    assert rec.get("image_count") == 2


def test_read_session_file_text_allows_listed_top_level_script(tmp_path: Path) -> None:
    """A researcher-uploaded top-level script remains readable
    through the bridge — that's the legitimate copy-button flow.
    Pin the positive case alongside the negative-case tests so a
    future tightening doesn't accidentally break the actual UI
    surface."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    script = cwd / "regression.do"
    script.write_text("regress y x\n", encoding="utf-8")

    bridge = NoraBridge(cwd=cwd)
    res = bridge.read_session_file_text(str(script))
    assert res["ok"] is True, res.get("reason")
    assert "regress y x" in res["text"]


def test_read_session_file_text_refuses_run_dir_script(tmp_path: Path) -> None:
    """Run-dir scripts (under ``<cwd>/.nora/runs/<id>/script.<ext>``)
    are not in the Files panel listing — the panel hides them
    because they already render on the result card. The bridge
    gate mirrors the panel; the read call is refused.

    Regression for the gap where ``read_session_file_text`` used
    the broader ``include_run_scripts=True`` enumeration and so
    could be invoked via page-rendered JS to surface bytes from
    files the panel intentionally hid."""
    cwd = tmp_path / "session"
    run_dir = cwd / ".nora" / "runs" / "r0001"
    run_dir.mkdir(parents=True)
    script = run_dir / "script.do"
    script.write_text("regress y x\n", encoding="utf-8")

    bridge = NoraBridge(cwd=cwd)
    res = bridge.read_session_file_text(str(script))
    assert res["ok"] is False
    assert "panel listing" in res["reason"]


def test_read_session_file_text_refuses_script_written_cwd_file(tmp_path: Path) -> None:
    """The load-bearing case: a cwd top-level file that was created
    or modified by a ``submit_script`` run (recorded in
    ``cwd_writes.json``) is hidden from the Files panel by design
    — those bytes are already represented on the script's result
    card and may contain script-derived row-level data. The bridge
    gate must refuse to surface them through ``read_session_file_text``
    too; otherwise a compromised result that escaped sanitization
    could call this method with the script-output filename and
    pull the bytes back onto the page, bypassing the SDC posture
    that gates submit_script result payloads.
    """
    cwd = tmp_path / "session"
    run_dir = cwd / ".nora" / "runs" / "r0001"
    run_dir.mkdir(parents=True)
    # The cwd-resident file that the script "wrote" — its bytes
    # would otherwise reach the page through read_session_file_text.
    written = cwd / "script_output.log"
    written.write_text("name,ssn\nAlice,000-00-0000\n", encoding="utf-8")
    # Record the file in the run's cwd_writes manifest under the
    # exact (mtime, size) it currently has — that's the tagging the
    # executor would normally apply.
    import json
    stat = written.stat()
    (run_dir / "cwd_writes.json").write_text(
        json.dumps([{
            "name": "script_output.log",
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }]),
        encoding="utf-8",
    )

    bridge = NoraBridge(cwd=cwd)
    res = bridge.read_session_file_text(str(written))
    assert res["ok"] is False
    assert "panel listing" in res["reason"]


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

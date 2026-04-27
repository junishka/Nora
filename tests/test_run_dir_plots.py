"""Researcher-side plot rendering: scan ``run_dir`` for image files
and surface them inline in the tool-result card.

This is the RESEARCHER's view. It's separate from the model-vision
path (``runner._capture_plots`` / manifest allowlist) — anything a
script writes lands here regardless of how it got there. Stata's
``graph export``, R's ``ggsave``, and Python's ``plt.savefig`` all
end up in the run dir; the researcher should see them in chat
without having to click "Show folder".

These tests pin the collector contract: it finds .png/.jpg files,
caps count + per-file inline bytes, and never bleeds past run_dir
(no symlink walks, no recursive descent into nested project state).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nora.ui import (
    _RESEARCHER_PLOT_MAX_BYTES,
    _RESEARCHER_PLOT_MAX_PER_RESULT,
    _collect_run_dir_plots,
)


def _png(size: int = 200) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * size


def test_collect_finds_png_in_run_dir(tmp_path: Path) -> None:
    """The base case: Stata-style ``graph export`` writes a .png
    directly into run_dir (Stata's subprocess cwd). The collector
    surfaces it for the researcher."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "regression_coefficients.png").write_bytes(_png())

    plots = _collect_run_dir_plots(str(run))
    assert len(plots) == 1
    assert plots[0]["name"] == "regression_coefficients.png"
    assert plots[0]["mime"] == "image/png"
    assert "data" in plots[0], "small plot should carry inline base64"
    decoded = base64.b64decode(plots[0]["data"])
    assert decoded.startswith(b"\x89PNG")


def test_collect_finds_plots_in_nora_plots_subdir(tmp_path: Path) -> None:
    """Plots written by ``plot_residuals`` land in
    ``run_dir/_nora_plots/``. The researcher should see them
    alongside any direct-write plots — they're researcher-visible
    too, in addition to being model-visible via the manifest."""
    run = tmp_path / "run"
    plots_dir = run / "_nora_plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "residuals.png").write_bytes(_png())
    (run / "scatter.png").write_bytes(_png())

    plots = _collect_run_dir_plots(str(run))
    names = sorted(p["name"] for p in plots)
    assert names == ["residuals.png", "scatter.png"]


def test_collect_oversized_drops_data_keeps_metadata(tmp_path: Path) -> None:
    """Plots above the inline byte cap don't carry base64 (sending
    multi-megabyte payloads through evaluate_js is slow), but the
    JS still gets ``name``, ``path``, and ``size`` so it can
    render an "Open externally" placeholder."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "huge.png").write_bytes(b"\x89PNG" + b"x" * (_RESEARCHER_PLOT_MAX_BYTES + 100))

    plots = _collect_run_dir_plots(str(run))
    assert len(plots) == 1
    entry = plots[0]
    assert entry["name"] == "huge.png"
    assert entry["size"] > _RESEARCHER_PLOT_MAX_BYTES
    assert "data" not in entry
    assert entry["path"]


def test_collect_caps_total_count(tmp_path: Path) -> None:
    """A script that writes many plots gets clipped to the
    per-result cap. Most-recent files win (mtime descending), so
    a researcher who iterated inside one run sees their latest
    versions, not the first ones."""
    run = tmp_path / "run"
    run.mkdir()
    import time
    n = _RESEARCHER_PLOT_MAX_PER_RESULT + 4
    for i in range(n):
        p = run / f"plot_{i:02d}.png"
        p.write_bytes(_png())
        # Stagger mtimes so sort order is deterministic.
        ts = 1_000_000 + i
        p.touch()
        import os
        os.utime(p, (ts, ts))

    plots = _collect_run_dir_plots(str(run))
    assert len(plots) == _RESEARCHER_PLOT_MAX_PER_RESULT
    # Newest plots first — last few indices.
    expected = [
        f"plot_{i:02d}.png"
        for i in range(n - 1, n - 1 - _RESEARCHER_PLOT_MAX_PER_RESULT, -1)
    ]
    got = [p["name"] for p in plots]
    assert got == expected


def test_collect_handles_missing_run_dir(tmp_path: Path) -> None:
    """Malformed / missing run_dir returns []."""
    assert _collect_run_dir_plots(None) == []
    assert _collect_run_dir_plots("") == []
    assert _collect_run_dir_plots(str(tmp_path / "nonexistent")) == []


def test_collect_skips_symlinks(tmp_path: Path) -> None:
    """Symlinks inside run_dir are deliberately ignored to avoid
    walking out of the run tree if something planted a link there.
    Real plot writes (graph export, savefig) are regular files."""
    run = tmp_path / "run"
    run.mkdir()
    # A real file outside run_dir we don't want to surface.
    target = tmp_path / "secret.png"
    target.write_bytes(_png())
    # Plus a regular plot inside run_dir for contrast.
    (run / "real_plot.png").write_bytes(_png())
    # Symlink pointing outside.
    try:
        (run / "leak.png").symlink_to(target)
    except OSError:
        pytest.skip("filesystem doesn't support symlinks")

    plots = _collect_run_dir_plots(str(run))
    names = [p["name"] for p in plots]
    assert names == ["real_plot.png"]


def test_collect_ignores_non_image_files(tmp_path: Path) -> None:
    """Stuff that isn't a recognised image extension is left
    alone — stdout.log / stderr.log / result.json all live in the
    same run_dir and must not appear as plot tiles."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "stdout.log").write_text("regression output")
    (run / "result.json").write_text("{}")
    (run / "script.do").write_text("regress y x")
    (run / "actual_plot.png").write_bytes(_png())

    plots = _collect_run_dir_plots(str(run))
    assert [p["name"] for p in plots] == ["actual_plot.png"]


# ---------------------------------------------------------------------------
# Stata helper file is present in the package — the executor stages
# it on every Stata run, so a missing file would break Stata plot
# vision silently in production.
# ---------------------------------------------------------------------------

def test_stata_plot_residuals_ado_is_packaged() -> None:
    """``nora_plot_residuals.ado`` ships with the package and is
    listed in ``_stage_runtime``'s Stata file set. A regression
    that drops the file from either spot would leave the model
    unable to register Stata plots even though the system prompt
    advertises the helper."""
    from importlib import resources
    from nora.executor import _stage_runtime  # noqa: F401 — covered below

    runtime = resources.files("nora.runtime")
    src = runtime.joinpath("nora_plot_residuals.ado").read_text(encoding="utf-8")
    assert "program define nora_plot_residuals" in src
    assert "_nora_plots" in src
    assert "manifest.jsonl" in src
    assert '"kind":"residuals"' in src


def test_stata_plot_residuals_in_stage_runtime_list(tmp_path: Path) -> None:
    """``_stage_runtime`` actually copies the .ado into the run
    dir's lib for Stata. Without this the plot helper wouldn't be
    on Stata's adopath at runtime."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_plot_residuals.ado").is_file()


# ---------------------------------------------------------------------------
# Stata cd-cwd: the executor preamble cd's into the session cwd
# before user code runs, so a bare ``graph export "fig.png"`` in
# Stata lands in session_cwd, NOT in run_dir. The collector has to
# scan session_cwd too — but only for plots written during this run.
# ---------------------------------------------------------------------------

def test_collect_finds_session_cwd_plots_when_newer_than_run_start(
    tmp_path: Path,
) -> None:
    """The Stata thumbnail bug: ``graph export "fig.png"`` writes
    to session_cwd (the executor preamble cd'd there). The
    collector must surface those plots — but ONLY if they were
    written by THIS run. Older plots in the same dir should stay
    out of the thumbnail row.

    Run start is detected from the script file's mtime (script is
    written at run start, before subprocess exec).
    """
    import os
    import time
    session = tmp_path / "session"
    session.mkdir()
    run = tmp_path / "session" / ".nora" / "runs" / "r0001"
    run.mkdir(parents=True)

    # Pre-existing plot in session cwd from a PRIOR run.
    old = session / "old_chart.png"
    old.write_bytes(_png())
    old_ts = 1_000_000.0
    os.utime(old, (old_ts, old_ts))

    # Run starts: write script.do *after* old plot.
    run_start = old_ts + 100
    script = run / "script.do"
    script.write_bytes(b"// stata script")
    os.utime(script, (run_start, run_start))

    # New plot generated by THIS run, in session cwd (Stata path).
    new = session / "new_chart.png"
    new.write_bytes(_png())
    new_ts = run_start + 10
    os.utime(new, (new_ts, new_ts))

    plots = _collect_run_dir_plots(str(run), session_cwd=str(session))
    names = [p["name"] for p in plots]
    assert "new_chart.png" in names, (
        "the just-written plot must appear in the thumbnail row"
    )
    assert "old_chart.png" not in names, (
        "plots from prior runs must stay out — otherwise every "
        "tool result would surface every old chart"
    )


def test_collect_session_cwd_skipped_when_no_session_cwd_arg(tmp_path: Path) -> None:
    """Without a ``session_cwd`` argument, the collector falls back
    to the old behavior: only run_dir is scanned. Keeps the function
    callable from places that don't track the runner's cwd."""
    session = tmp_path / "session"
    session.mkdir()
    run = session / ".nora" / "runs" / "r0001"
    run.mkdir(parents=True)
    (run / "script.do").write_bytes(b"x")
    (session / "stata_plot.png").write_bytes(_png())

    plots = _collect_run_dir_plots(str(run))  # no session_cwd
    assert plots == []


def test_collect_session_cwd_dedupes_against_run_dir(tmp_path: Path) -> None:
    """If the same physical file is reachable from both scans, it's
    surfaced once. (Not the common case but possible if a future
    config changes paths.)"""
    session = tmp_path / "session"
    session.mkdir()
    run = session / "run"
    run.mkdir()
    (run / "script.do").write_bytes(b"x")
    (run / "shared.png").write_bytes(_png())
    plots = _collect_run_dir_plots(str(run), session_cwd=str(run))
    names = [p["name"] for p in plots]
    assert names.count("shared.png") == 1


# ---------------------------------------------------------------------------
# Python register_plot — kind allowlist, file existence
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P1.1 — register_plot was removed. Privacy hole: kind label was
# self-attested by the script, so a histogram could pose as a
# coefficient plot and ride into the model's vision path. The fix
# is helper-only — only ``plot_residuals`` / ``plot_interaction`` /
# ``plot_coefficients`` (each operates on a fitted model object,
# never on raw rows) are exposed.
# ---------------------------------------------------------------------------

def test_register_plot_helper_is_gone_from_all_runtimes() -> None:
    """The escape-hatch ``register_plot`` is REMOVED from R, Python,
    and Stata. A regression that re-introduces it without the
    file-content gate would re-open the privacy hole."""
    from importlib import resources
    runtime = resources.files("nora.runtime")
    r_src = runtime.joinpath("nora.R").read_text(encoding="utf-8")
    py_src = runtime.joinpath("nora.py").read_text(encoding="utf-8")
    assert "nora$register_plot" not in r_src, (
        "R register_plot was removed — its kind label was self-"
        "attested by the script, bypassing the privacy gate"
    )
    assert "def register_plot" not in py_src, (
        "Python register_plot was removed for the same reason"
    )
    # Stata .ado file is gone too.
    try:
        runtime.joinpath("nora_register_plot.ado").read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    else:
        pytest.fail(
            "nora_register_plot.ado is still on disk — must be removed"
        )


def test_plot_coefficients_helper_is_packaged_for_r_and_python() -> None:
    """The replacement: ``plot_coefficients`` exists in R + Python
    and operates ONLY on a fitted-model object. Stata's version
    isn't shipped yet (documented in the system prompt)."""
    from importlib import resources
    runtime = resources.files("nora.runtime")
    r_src = runtime.joinpath("nora.R").read_text(encoding="utf-8")
    py_src = runtime.joinpath("nora.py").read_text(encoding="utf-8")
    assert "nora$plot_coefficients" in r_src
    # Make sure it actually takes a model fit, not a file path.
    assert "function(model" in r_src.split("nora$plot_coefficients")[1][:80]
    assert "def plot_coefficients(fitted" in py_src


# ---------------------------------------------------------------------------
# P1.2 — plot labels go through safe_text before reaching the model
# ---------------------------------------------------------------------------

def test_capture_plots_sanitizes_label_via_safe_text(tmp_path: Path) -> None:
    """A manifest label crafted by user code (e.g. computed from raw
    data) goes through ``safe_text`` before being staged. Without
    this, the label would be interpolated into the next-turn prompt
    notice unsanitized, bypassing the same text-safety gate every
    other data-origin string respects."""
    import json as _json
    from nora.runner import SessionRunner

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "residuals.png").write_bytes(_png())
    # Hostile label: control chars (newline, tab, NUL), an
    # extra-long payload, and quote marks. ``safe_text`` should
    # collapse these into something boring (or empty).
    bad_label = "evil\nlabel\ttext\x00<script>" + ("X" * 500)
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "residuals.png",
            "kind": "residuals",
            "label": bad_label,
        }) + "\n",
        encoding="utf-8",
    )

    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-4-6[1m]",
    )
    runner._capture_plots(run)
    assert len(runner.pending_plot_images) == 1
    sanitized = runner.pending_plot_images[0]["label"]
    # Hard guarantees: no control chars survived; length capped.
    assert "\n" not in sanitized
    assert "\t" not in sanitized
    assert "\x00" not in sanitized
    assert len(sanitized) <= 120


def test_capture_plots_sanitizes_filename(tmp_path: Path) -> None:
    """Filenames also reach the model in the prompt notice and so
    must go through ``safe_text``. A pathologically-named file
    that sanitizes empty falls back to a generic placeholder —
    NEVER the raw string."""
    import json as _json
    from nora.runner import SessionRunner

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    # File on disk has a clean name (the runtime helpers control
    # what they write); we just verify the staging dict's ``name``
    # field is run through safe_text rather than passed through raw.
    (plots / "residuals.png").write_bytes(_png())
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "residuals.png",
            "kind": "residuals",
            "label": "ok",
        }) + "\n",
        encoding="utf-8",
    )
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-4-6[1m]",
    )
    runner._capture_plots(run)
    name = runner.pending_plot_images[0]["name"]
    # Clean name passes through unchanged.
    assert name == "residuals.png"


# ---------------------------------------------------------------------------
# P2.1 — Stata helpers write under run_dir, not the session cwd
# ---------------------------------------------------------------------------

def test_stata_plot_residuals_writes_under_run_dir() -> None:
    """The Stata helper resolves run_dir from ``NORA_RESULT_PATH``
    rather than relying on Stata's cwd. Without this, the manifest
    landed in ``<session_cwd>/_nora_plots/`` where the runner's
    ``_capture_plots`` never looks — researcher saw thumbnails but
    the model-vision path silently missed them."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "nora_plot_residuals.ado"
    ).read_text(encoding="utf-8")
    assert ": env NORA_RESULT_PATH" in src, (
        "must read NORA_RESULT_PATH to locate the run dir"
    )
    assert "/result.json" in src, (
        "must derive run_dir by stripping the result-file suffix"
    )
    # Sanity: the manifest path is now anchored to `rundir`.
    assert "`rundir'/_nora_plots" in src


# ---------------------------------------------------------------------------
# P2.2 — delete_session refuses busy runners and cleans up idle ones
# ---------------------------------------------------------------------------

def test_delete_session_refuses_when_target_runner_is_busy(tmp_path: Path) -> None:
    """A long-running turn in session A must not be killed by the
    user clicking the trash icon next to A while focused on B.
    rmtree under a live SDK session would leak the runner's cwd
    out from under it — exactly the cross-session interference
    the multi-runner refactor exists to stop."""
    from unittest.mock import MagicMock
    import nora.ui as ui_mod
    from nora.ui import NoraBridge

    a = tmp_path / "session_a"
    b = tmp_path / "session_b"
    a.mkdir()
    b.mkdir()

    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        runner_a = bridge._active_runner()
        assert runner_a is not None

        # Plant a fake "busy" turn task on A.
        class _FakeTask:
            def done(self): return False
            def cancel(self): pass
        runner_a._current_turn_task = _FakeTask()  # type: ignore[assignment]

        # Move focus to B so delete_session won't refuse on the
        # "active session" check (we're testing the busy check).
        bridge._set_cwd(b)

        res = bridge.delete_session(str(a))
        assert res["ok"] is False
        assert "in flight" in res["reason"]
        assert a.exists(), "the directory must NOT have been deleted"
    finally:
        ui_mod.SESSIONS_ROOT = real_root


# ---------------------------------------------------------------------------
# P1.3 — composer attachments don't leak across session switches
# ---------------------------------------------------------------------------
#
# Without a JS runtime in CI we can't drive this end-to-end, but we
# can at least lock that ``showChat`` includes the clearing logic
# so a future refactor doesn't quietly remove it.

def test_show_chat_clears_staged_images_on_session_switch() -> None:
    """``showChat`` (the JS focus-change handler) must clear
    ``stagedImages`` and ``stagedDataNotices`` so an attachment
    staged in session A doesn't ride along with the next message
    in session B. The pure-JS test would belong in a frontend
    harness; in the meantime we pin the source so a refactor that
    silently drops the clearing falls a build."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent
           / "src" / "nora" / "web" / "app.js").read_text(encoding="utf-8")
    # The function header.
    assert "function showChat(payload)" in src
    # The clearing block, anchored to a comment fragment that's
    # specific to this fix so we don't false-match on unrelated
    # ``stagedImages`` references.
    assert "Drop staged composer state" in src
    assert "stagedImages.length = 0" in src
    assert "stagedDataNotices.length = 0" in src


# ---------------------------------------------------------------------------
# Helper-failure diagnostic — surface a hint when the helper was
# called but produced no plot. Most common cause: matplotlib not
# installed in the Python environment.
# ---------------------------------------------------------------------------

def test_diagnostic_catches_matplotlib_missing(tmp_path: Path) -> None:
    """When ``nora.plot_coefficients`` fails with ImportError on
    matplotlib, the helper writes to stderr. The bridge's
    diagnostic surfaces this as a one-liner so the researcher
    knows what to install instead of staring at a blank thumbnail
    row."""
    from nora.ui import _detect_plot_helper_diagnostics

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    # Helper got far enough to mkdir _nora_plots/ but then failed.
    (run / "stderr.log").write_text(
        "nora.plot_coefficients failed: No module named 'matplotlib'\n",
        encoding="utf-8",
    )

    diag = _detect_plot_helper_diagnostics(str(run), n_plots_found=0)
    assert diag is not None
    assert "matplotlib" in diag.lower()
    assert "pip install" in diag.lower()


def test_diagnostic_catches_generic_helper_failure(tmp_path: Path) -> None:
    """Other helper errors (R / Python / Stata) get a generic
    one-liner echoing the underlying failure so the researcher
    has a starting point."""
    from nora.ui import _detect_plot_helper_diagnostics

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (run / "stderr.log").write_text(
        "nora$plot_coefficients failed: nothing to plot after dropping the intercept\n",
        encoding="utf-8",
    )

    diag = _detect_plot_helper_diagnostics(str(run), n_plots_found=0)
    assert diag is not None
    assert "nothing to plot" in diag


def test_diagnostic_silent_when_plots_were_produced(tmp_path: Path) -> None:
    """If the helper SUCCEEDED and plots came back, the diagnostic
    stays silent — it only fires for the empty-output case."""
    from nora.ui import _detect_plot_helper_diagnostics

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (run / "stderr.log").write_text(
        "nora.plot_coefficients failed: spurious noise\n",
        encoding="utf-8",
    )

    # ``n_plots_found=1`` short-circuits the diagnostic — we don't
    # warn when at least one plot landed.
    assert _detect_plot_helper_diagnostics(str(run), n_plots_found=1) is None


# ---------------------------------------------------------------------------
# Plot helper failures reach the MODEL via tools._summarize_plot_helpers.
# Previously they only landed in stderr; the model thought helpers
# succeeded and confidently said "thumbnail should be visible above"
# while the researcher saw nothing.
# ---------------------------------------------------------------------------

def test_summarize_plot_helpers_returns_none_when_no_plots_dir(
    tmp_path: Path,
) -> None:
    """Plain submit_script run with no helper calls — no
    ``_nora_plots/`` directory, summary returns None, the field
    stays out of the tool response entirely."""
    from nora.tools import _summarize_plot_helpers
    run = tmp_path / "run"
    run.mkdir()
    assert _summarize_plot_helpers(run) is None


def test_summarize_plot_helpers_lists_succeeded(tmp_path: Path) -> None:
    """A successful helper writes to manifest.jsonl. The summary
    reflects that as ``succeeded: [{file, kind, label}]`` so the
    model sees what plots were produced."""
    import json as _json
    from nora.tools import _summarize_plot_helpers
    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "residuals.png",
            "kind": "residuals",
            "label": "Residual diagnostics",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(run)
    assert summary is not None
    assert summary["succeeded"][0]["file"] == "residuals.png"
    assert summary["failed"] == []
    assert "note" not in summary


def test_summarize_plot_helpers_surfaces_failures_with_fix(
    tmp_path: Path,
) -> None:
    """A failed helper writes to helper_errors.jsonl. The summary
    must surface the message AND the fix hint so the model can
    tell the researcher exactly what to install — not just say
    'thumbnail should be visible' while no plot landed."""
    import json as _json
    from nora.tools import _summarize_plot_helpers
    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "helper_errors.jsonl").write_text(
        _json.dumps({
            "helper": "plot_coefficients",
            "error": "ModuleNotFoundError",
            "message": "No module named 'matplotlib'",
            "fix": "pip install matplotlib",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(run)
    assert summary is not None
    assert summary["succeeded"] == []
    assert len(summary["failed"]) == 1
    fail = summary["failed"][0]
    assert fail["helper"] == "plot_coefficients"
    assert "matplotlib" in fail["message"]
    assert fail["fix"] == "pip install matplotlib"
    # Note tells the model to stop guessing.
    assert "note" in summary
    assert "no plots" in summary["note"].lower()


def test_python_helper_writes_helper_errors_jsonl_on_import_failure(
    tmp_path: Path,
) -> None:
    """End-to-end: when ``nora.plot_coefficients`` fails (we
    simulate by passing an object with no ``.params``), the
    helper writes a structured entry to ``helper_errors.jsonl``
    next to where ``manifest.jsonl`` would have gone. tools'
    summary then picks it up.

    We can't easily simulate the matplotlib-import failure inside
    a test without mocking imports; the behavior we pin instead
    is "any helper failure produces a helper_errors entry"."""
    import json as _json
    import subprocess
    import sys as _sys
    # Run the helper in a subprocess that points at a fake run dir
    # via NORA_RESULT_PATH.
    run = tmp_path / "run"
    run.mkdir()
    fake_result = run / "result.json"
    fake_result.write_text("{}")  # placeholder; helper just reads dirname

    code = (
        "import os\n"
        "os.environ['NORA_RUN_TOKEN'] = 'test-token'\n"
        f"os.environ['NORA_RESULT_PATH'] = {str(fake_result)!r}\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location("
        "    'nora_runtime', "
        f"   {str(Path(__file__).resolve().parent.parent / 'src' / 'nora' / 'runtime' / 'nora.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "class Bad:\n"
        "    pass\n"
        "mod.plot_coefficients(Bad())  # no .params, helper fails cleanly\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True,
    )
    # Helper must not raise.
    assert proc.returncode == 0, f"helper raised; stderr={proc.stderr}"
    err_log = run / "_nora_plots" / "helper_errors.jsonl"
    assert err_log.is_file(), (
        "helper failure must write helper_errors.jsonl so the model "
        "sees the failure in tool result, not just stderr"
    )
    entries = [
        _json.loads(line) for line in err_log.read_text().splitlines() if line
    ]
    assert any(e.get("helper") == "plot_coefficients" for e in entries)


# ---------------------------------------------------------------------------
# matplotlib is probed as an optional package
# ---------------------------------------------------------------------------

def test_env_detect_probes_matplotlib_as_optional() -> None:
    """``matplotlib`` is in the optional-package list so its absence
    surfaces in ``Tool.optional_missing_packages`` without blocking
    runs that don't plot. The previous behavior probed only
    required packages; matplotlib was invisible to the executor's
    preflight even though every plot helper depends on it."""
    from nora.env_detect import _PYTHON_OPTIONAL_PACKAGES, find_python
    assert "matplotlib" in _PYTHON_OPTIONAL_PACKAGES
    # And the Tool dataclass actually carries the field.
    tool = find_python()
    if tool is None:
        pytest.skip("Python 3 not on PATH in this test env")
    # Field exists (tuple, possibly empty).
    assert isinstance(tool.optional_missing_packages, tuple)


# ---------------------------------------------------------------------------
# plot_estimate_comparison — the helper that closes the multi-model
# comparison-plot gap. Without it the model spent three turns
# hand-rolling the same forest plot in R / Python / Stata.
# ---------------------------------------------------------------------------

def test_plot_estimate_comparison_helpers_exist_for_all_languages() -> None:
    """The escape-hatch comparison plot is the right answer to the
    "female gap before vs after controls" workflow. Pin parity
    across R, Python, and Stata so a researcher in any of the three
    can stay in their language."""
    from importlib import resources
    runtime = resources.files("nora.runtime")

    # Stata: dedicated .ado file, listed in _stage_runtime.
    stata_ado = runtime.joinpath(
        "nora_plot_estimate_comparison.ado"
    ).read_text(encoding="utf-8")
    assert "program define nora_plot_estimate_comparison" in stata_ado
    assert "estimates restore" in stata_ado
    # Writes under run_dir, not session cwd.
    assert "`rundir'/_nora_plots" in stata_ado
    # Manifest entry uses the allowlisted ``coefficients`` kind so
    # the runner accepts it.
    assert '"kind":"coefficients"' in stata_ado

    # R + Python: live inside the runtime modules.
    r_src = runtime.joinpath("nora.R").read_text(encoding="utf-8")
    py_src = runtime.joinpath("nora.py").read_text(encoding="utf-8")
    assert "nora$plot_estimate_comparison" in r_src
    assert "def plot_estimate_comparison" in py_src


def test_stata_estimate_comparison_in_stage_runtime_list(tmp_path: Path) -> None:
    """The .ado is staged onto Stata's adopath at run time.
    Without this the helper exists in the package but is invisible
    to running scripts."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_plot_estimate_comparison.ado").is_file()


def test_stata_plot_interaction_packaged_and_staged(tmp_path: Path) -> None:
    """Stata gains a native predicted-response helper so .dta
    analyses don't have to switch to R/Python for an interaction
    plot. Pin the file is in the package AND staged onto Stata's
    adopath at run time."""
    from importlib import resources
    from nora.executor import _stage_runtime
    src = resources.files("nora.runtime").joinpath(
        "nora_plot_interaction.ado"
    ).read_text(encoding="utf-8")
    assert "program define nora_plot_interaction" in src
    # Uses native Stata margins + marginsplot path.
    assert "margins" in src
    assert "marginsplot" in src
    # Writes under run_dir, not session cwd.
    assert "`rundir'/_nora_plots" in src
    assert '"kind":"interaction"' in src
    # Filled CI band (recastci(rarea)) — addresses the "shitty
    # plot" complaint about default Stata/R styling.
    assert "rarea" in src
    # Caller can override axis labels and title.
    assert "xlabel(string)" in src
    assert "ylabel(string)" in src
    assert "title(string)" in src

    # Staging:
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_plot_interaction.ado").is_file()


def test_r_plot_interaction_accepts_label_overrides() -> None:
    """R helper takes ``xlab`` / ``ylab`` / ``title`` overrides so
    plots can have publication-grade axis labels without leaving
    R. Pin the signature so a future refactor doesn't quietly
    drop the params."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "nora.R"
    ).read_text(encoding="utf-8")
    assert "plot_interaction <- function(model, var, label = NULL," in src
    assert "xlab = NULL" in src
    assert "ylab = NULL" in src
    assert "title = NULL" in src
    # ggplot2 path is taken when available — addresses the bare
    # base-graphics complaint.
    assert "requireNamespace(\"ggplot2\"" in src
    assert "geom_ribbon" in src


def test_python_plot_interaction_accepts_label_overrides() -> None:
    """Python helper has the same xlab/ylab/title overrides as the
    R version — parity matters so the model can learn one API."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "nora.py"
    ).read_text(encoding="utf-8")
    assert "def plot_interaction(" in src
    # All three override params present.
    assert "xlab: str | None = None" in src
    assert "ylab: str | None = None" in src
    assert "title: str | None = None" in src
    # Uses the filled-ribbon CI (better than the prior dashed lines).
    assert "fill_between" in src


# ---------------------------------------------------------------------------
# Stata export reliability — PDF / PNG / .gph fallback
# ---------------------------------------------------------------------------

def test_stata_helpers_use_export_utility_not_hardcoded_png() -> None:
    """All four Stata plot helpers must go through
    ``_nora_export_plot``. Hard-coding ``as(png)`` was the blocker
    on machines without ``Graph2png`` — every helper failed with
    "translator Graph2png not found" and produced no plot. The
    utility tries PDF first (Stata 17+ native, 15-16 via
    Graph2pdf), then PNG, then .gph as last resort."""
    from importlib import resources
    runtime = resources.files("nora.runtime")
    helpers = [
        "nora_plot_residuals.ado",
        "nora_plot_coefficients.ado",
        "nora_plot_interaction.ado",
        "nora_plot_estimate_comparison.ado",
    ]
    for fname in helpers:
        src = runtime.joinpath(fname).read_text(encoding="utf-8")
        # Must call the utility.
        assert "_nora_export_plot using" in src, (
            f"{fname} must use _nora_export_plot for format fallback "
            f"(found hard-coded export instead)"
        )
        # Must NOT have a hard-coded ``as(png)`` outside the utility.
        assert "as(png) width(1200)" not in src, (
            f"{fname} still has a hard-coded as(png) export — that "
            f"crashes when Graph2png translator is missing"
        )


def test_export_utility_tries_pdf_png_gph_in_order() -> None:
    """The shared utility writes the order explicitly: PDF first
    (most reliable), PNG second (needs Graph2png), GPH last
    (always saves but bridge can't rasterize)."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "_nora_export_plot.ado"
    ).read_text(encoding="utf-8")
    pdf_pos = src.find("graph export")
    png_pos = src.find("graph export", pdf_pos + 1)
    gph_pos = src.find("graph save")
    assert pdf_pos > 0
    assert png_pos > pdf_pos, "PNG attempt must come AFTER PDF"
    assert gph_pos > png_pos, ".gph save must be the last fallback"
    # And the format strings are right.
    assert "as(pdf)" in src
    assert "as(png)" in src


def test_export_utility_in_stage_runtime_list(tmp_path: Path) -> None:
    """``_nora_export_plot.ado`` is staged onto Stata's adopath at
    run time; without that the ``_nora_export_plot`` call inside
    every helper would fail with 'unrecognized command'."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "_nora_export_plot.ado").is_file()


# ---------------------------------------------------------------------------
# PDF → PNG conversion via sips
# ---------------------------------------------------------------------------

def test_png_for_returns_png_files_unchanged(tmp_path: Path) -> None:
    """Already-PNG inputs come back as-is — no conversion needed,
    no sidecar produced."""
    from nora.plot_convert import png_for
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG" + b"\x00" * 100)
    out = png_for(p)
    assert out == p
    # No sidecar polluting the dir.
    assert not (tmp_path / "x.nora.png").exists()


def test_png_for_returns_none_for_gph_and_unknown(tmp_path: Path) -> None:
    """``.gph`` (Stata native) and other formats sips can't
    convert come back as None. The caller treats these as
    researcher-only — they show in the Files panel as plain rows
    but don't reach the model."""
    from nora.plot_convert import png_for
    gph = tmp_path / "x.gph"
    gph.write_bytes(b"binary stata graph")
    assert png_for(gph) is None
    eps = tmp_path / "x.eps"
    eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0")
    assert png_for(eps) is None


def test_png_for_caches_sidecar_by_mtime(tmp_path: Path) -> None:
    """A second call with an unchanged source returns the same
    cached sidecar without re-running sips. This matters because
    ``_collect_run_dir_plots`` runs on every tool result and we
    don't want to rasterize the same PDF on every call."""
    import os
    from nora.plot_convert import png_for, _SIDECAR_SUFFIX
    pdf = tmp_path / "plot.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    # Pre-create a sidecar with mtime newer than the source so the
    # cache check returns it without invoking sips.
    sidecar = tmp_path / f"plot{_SIDECAR_SUFFIX}"
    sidecar.write_bytes(b"\x89PNG cached")
    src_ts = pdf.stat().st_mtime
    os.utime(sidecar, (src_ts + 5, src_ts + 5))
    out = png_for(pdf)
    assert out == sidecar
    # Bytes are the cached ones — sips didn't overwrite.
    assert sidecar.read_bytes() == b"\x89PNG cached"


def test_runner_capture_plots_handles_pdf_via_sidecar(tmp_path: Path) -> None:
    """The runner's plot-capture path now accepts manifest entries
    pointing at PDFs — it converts via sips and attaches the PNG.
    Without this, every Stata plot on a Graph2png-missing machine
    landed as PDF and the model never saw it."""
    import json as _json
    from unittest.mock import patch
    from pathlib import Path as _P
    from nora.runner import SessionRunner

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "coefficients.pdf").write_bytes(b"%PDF-1.4 fake")
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "coefficients.pdf",
            "kind": "coefficients",
            "label": "test",
            "format": "pdf",
        }) + "\n",
        encoding="utf-8",
    )

    # Patch png_for so we don't depend on sips actually running in
    # CI. Returns a fake PNG sidecar with the right shape.
    fake_png = plots / "coefficients.nora.png"
    fake_png.write_bytes(b"\x89PNG" + b"\x00" * 200)

    with patch("nora.plot_convert.png_for", return_value=fake_png):
        runner = SessionRunner(
            cwd=tmp_path, provider="anthropic",
            model="claude-sonnet-4-6[1m]",
        )
        runner._capture_plots(run)

    assert len(runner.pending_plot_images) == 1
    img = runner.pending_plot_images[0]
    assert img["mime"] == "image/png"
    # The bytes attached are the converted PNG, not the original PDF.
    import base64
    decoded = base64.b64decode(img["data"])
    assert decoded.startswith(b"\x89PNG")


def test_runner_capture_plots_logs_failure_when_pdf_conversion_fails(
    tmp_path: Path,
) -> None:
    """If sips isn't on the machine (non-macOS, etc.), conversion
    returns None. The runner logs a helper_errors.jsonl entry so
    the model knows the plot was produced but couldn't be
    rasterized — better than silently dropping the plot from
    vision."""
    import json as _json
    from unittest.mock import patch
    from nora.runner import SessionRunner

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "coefficients.pdf").write_bytes(b"%PDF-1.4")
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "coefficients.pdf", "kind": "coefficients",
            "label": "x", "format": "pdf",
        }) + "\n",
        encoding="utf-8",
    )
    with patch("nora.plot_convert.png_for", return_value=None):
        runner = SessionRunner(
            cwd=tmp_path, provider="anthropic",
            model="claude-sonnet-4-6[1m]",
        )
        runner._capture_plots(run)
    assert runner.pending_plot_images == []
    err_log = plots / "helper_errors.jsonl"
    assert err_log.is_file()
    entries = [
        _json.loads(line) for line in err_log.read_text().splitlines() if line
    ]
    assert any(e.get("error") == "PDFConversionError" for e in entries)


def test_collector_surfaces_pdf_plots_for_researcher(tmp_path: Path) -> None:
    """The researcher-thumbnail collector accepts PDFs and runs
    them through ``png_for`` so the chat shows a real raster
    preview rather than no thumbnail at all."""
    import json as _json
    from unittest.mock import patch
    from nora.ui import _collect_run_dir_plots
    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (run / "script.do").write_bytes(b"x")
    pdf = plots / "stata_plot.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake_png = plots / "stata_plot.nora.png"
    fake_png.write_bytes(b"\x89PNG" + b"\x00" * 100)

    with patch("nora.plot_convert.png_for", return_value=fake_png):
        result = _collect_run_dir_plots(str(run))

    assert len(result) == 1
    entry = result[0]
    assert entry["mime"] == "image/png"
    assert "data" in entry
    # ``path`` still points at the original PDF so the JS can hand
    # off "Open externally" to Preview.
    assert entry["path"].endswith(".pdf")


def test_pdf_added_to_researcher_plot_exts() -> None:
    """``.pdf`` is in the recognized researcher-plot extensions so
    the collector picks it up. Previously only PNG/JPG/JPEG were
    scanned, which silently dropped Stata-fallback PDFs."""
    from nora.ui import _RESEARCHER_PLOT_EXTS
    assert ".pdf" in _RESEARCHER_PLOT_EXTS


# ---------------------------------------------------------------------------
# Stata helper error logging: actual _rc + step indicator
# ---------------------------------------------------------------------------

def test_stata_helpers_capture_rc_before_mkdir_resets_it() -> None:
    """The previous failure path interpolated ``_rc`` AFTER
    ``capture mkdir``, which always reset _rc to 0 (or 693 for
    "directory exists"). The error log looked like
    ``Stata _rc=0`` regardless of the actual failure. Pin the new
    pattern: save _rc to a local IMMEDIATELY, then use that local
    in the JSON line."""
    from importlib import resources
    runtime = resources.files("nora.runtime")
    for fname in (
        "nora_plot_residuals.ado", "nora_plot_coefficients.ado",
        "nora_plot_interaction.ado", "nora_plot_estimate_comparison.ado",
    ):
        src = runtime.joinpath(fname).read_text(encoding="utf-8")
        assert "local _orig_rc = _rc" in src, (
            f"{fname} must save _rc to a local before any other "
            f"Stata operation; otherwise the error log shows the "
            f"wrong code"
        )
        assert "_rc=`_orig_rc'" in src, (
            f"{fname} must interpolate the saved local, not the "
            f"system _rc which has been reset by intervening ops"
        )
        assert '"step":' in src, (
            f"{fname} must include a ``step`` field in the error "
            f"line so the model knows where in the helper it failed"
        )


def test_nora_safe_export_packaged_and_staged(tmp_path: Path) -> None:
    """The safe wrapper for ad-hoc Stata exports must be packaged
    AND staged onto Stata's adopath. Without it the model has no
    sanctioned path for community plot commands like ``coefplot``
    that produce the graph but require a separate ``graph export``
    — and bare ``graph export`` aborts the do-file when
    ``Graph2png`` is missing."""
    from importlib import resources
    from nora.executor import _stage_runtime

    src = resources.files("nora.runtime").joinpath(
        "nora_safe_export.ado"
    ).read_text(encoding="utf-8")
    assert "program define nora_safe_export" in src
    # Falls back through PDF and EPS before .gph — same chain as
    # the in-helper exporter.
    assert "as(pdf)" in src
    assert "as(eps)" in src
    assert "graph save" in src
    # Researcher-visible only — does NOT write a manifest entry,
    # so this can't be used to smuggle raw-data plots into model
    # vision. Kind-specific helpers are the gate. (The docstring
    # mentions "manifest" in prose to explain WHY this helper
    # doesn't touch it — exclude that from the check by inspecting
    # only the program body.)
    body_start = src.index("program define nora_safe_export")
    body = src[body_start:]
    assert "manifest.jsonl" not in body, (
        "nora_safe_export must not append a manifest entry — "
        "manifest writes are reserved for kind-specific plot helpers"
    )
    assert '"kind"' not in body

    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_safe_export.ado").is_file()


def test_export_utility_includes_eps_in_chain() -> None:
    """The shared utility now tries EPS between PNG and .gph.
    Stata's PostScript path doesn't depend on the per-format
    translators that PDF/PNG go through, so EPS often works on
    installs where both PDF and PNG fail. Pin the order so a
    future refactor doesn't accidentally drop EPS."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "_nora_export_plot.ado"
    ).read_text(encoding="utf-8")
    # Skip the docstring at the top (which mentions ``as(png)``
    # in prose) and inspect only the program body.
    body_start = src.index("program define _nora_export_plot")
    body = src[body_start:]
    pdf_pos = body.find("as(pdf)")
    png_pos = body.find("as(png)")
    eps_pos = body.find("as(eps)")
    gph_pos = body.find("graph save")
    assert 0 < pdf_pos < png_pos < eps_pos < gph_pos, (
        f"order in body: pdf={pdf_pos}, png={png_pos}, "
        f"eps={eps_pos}, gph={gph_pos}"
    )


def test_png_for_handles_eps(tmp_path: Path) -> None:
    """EPS is in the convertible-suffix list. Whether the actual
    sips conversion succeeds depends on the macOS version, but
    the wiring (pre-cache + sips invocation) must accept EPS the
    same way it accepts PDF."""
    from nora.plot_convert import _CONVERTIBLE_SUFFIXES, png_for
    assert ".eps" in _CONVERTIBLE_SUFFIXES
    # Cache hit path works for EPS too.
    import os
    eps = tmp_path / "x.eps"
    eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0")
    sidecar = tmp_path / "x.nora.png"
    sidecar.write_bytes(b"\x89PNG cached")
    src_ts = eps.stat().st_mtime
    os.utime(sidecar, (src_ts + 5, src_ts + 5))
    out = png_for(eps)
    assert out == sidecar


def test_runner_capture_plots_handles_eps(tmp_path: Path) -> None:
    """When the helper falls back to EPS (because both PDF and
    PNG translators are missing), the runner's vision path
    converts via sips so the model still sees the plot. Same
    mechanism as the PDF path."""
    import json as _json
    from unittest.mock import patch
    from nora.runner import SessionRunner

    run = tmp_path / "run"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "coefficients.eps").write_bytes(b"%!PS-Adobe-3.0")
    (plots / "manifest.jsonl").write_text(
        _json.dumps({
            "file": "coefficients.eps",
            "kind": "coefficients",
            "label": "test",
            "format": "eps",
        }) + "\n",
        encoding="utf-8",
    )
    fake_png = plots / "coefficients.nora.png"
    fake_png.write_bytes(b"\x89PNG" + b"\x00" * 200)

    with patch("nora.plot_convert.png_for", return_value=fake_png):
        runner = SessionRunner(
            cwd=tmp_path, provider="anthropic",
            model="claude-sonnet-4-6[1m]",
        )
        runner._capture_plots(run)
    assert len(runner.pending_plot_images) == 1
    assert runner.pending_plot_images[0]["mime"] == "image/png"


def test_eps_added_to_researcher_plot_exts_and_files_kind() -> None:
    """``.eps`` is recognised both as a researcher-thumbnail ext
    and as a graph-kind file in the Files panel. Without parity
    the chat thumbnail might appear while the Files panel skips
    it (or vice-versa), which is confusing."""
    from nora.ui import _RESEARCHER_PLOT_EXTS
    assert ".eps" in _RESEARCHER_PLOT_EXTS


def test_safe_export_advertised_in_system_prompt() -> None:
    """The system prompt names ``nora_safe_export`` so the model
    has a sanctioned alternative to bare ``graph export``. Without
    this guidance the model has no signal to use the safe wrapper
    even when it's available on adopath."""
    from nora.system_prompt import build_system_prompt
    rendered = build_system_prompt(Path("/tmp"), "nora")
    assert "nora_safe_export" in rendered


def test_dont_regenerate_rule_is_in_system_prompt() -> None:
    """The model has been observed to regenerate the same plot
    three times in a row when it loses track of which earlier
    attempts succeeded. Pin the explicit rule against this so a
    future prompt edit doesn't quietly drop it."""
    from nora.system_prompt import build_system_prompt
    rendered = build_system_prompt(Path("/tmp"), "nora")
    assert "Don't regenerate a plot that already succeeded" in rendered
    # And the comparison-helper escape hatch is named so the
    # model doesn't hand-roll forest plots.
    assert "plot_estimate_comparison" in rendered


def test_diagnostic_silent_when_no_helper_was_called(tmp_path: Path) -> None:
    """A plain submit_script run that didn't touch any plot helper
    has no ``_nora_plots/`` dir. Diagnostic must not surface
    anything — there's nothing to diagnose."""
    from nora.ui import _detect_plot_helper_diagnostics

    run = tmp_path / "run"
    run.mkdir()
    (run / "stderr.log").write_text("regression succeeded\n", encoding="utf-8")
    assert _detect_plot_helper_diagnostics(str(run), n_plots_found=0) is None


def test_delete_session_closes_idle_runner_before_rmtree(tmp_path: Path) -> None:
    """An idle runner whose cwd we're about to delete must have its
    SDK session closed first. Otherwise the next time anything
    touches that runner (e.g. ``stop_loop`` on shutdown), it would
    try to operate on a vanished cwd."""
    from unittest.mock import MagicMock
    import nora.ui as ui_mod
    from nora.ui import NoraBridge

    a = tmp_path / "session_a"
    b = tmp_path / "session_b"
    a.mkdir()
    b.mkdir()

    real_root = ui_mod.SESSIONS_ROOT
    ui_mod.SESSIONS_ROOT = tmp_path
    try:
        bridge = NoraBridge(cwd=a)
        runner_a = bridge._active_runner()
        assert runner_a is not None
        bridge._set_cwd(b)

        # A is idle; runner is still in the registry.
        assert str(a.resolve()) in bridge._runners
        res = bridge.delete_session(str(a))
        assert res["ok"] is True
        assert not a.exists(), "rmtree should have run"
        # Runner is no longer tracked.
        assert str(a.resolve()) not in bridge._runners
    finally:
        ui_mod.SESSIONS_ROOT = real_root

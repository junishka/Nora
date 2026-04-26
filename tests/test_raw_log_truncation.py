"""Regression tests for the raw-log truncation in ``ui._read_raw_logs``.

Original behaviour kept only the LAST 32 KB of stdout, which dropped
the column names and dtypes for any reasonably wide ``df.head()``
exploratory output. Researchers saw two trailing rows of values
prefixed by ``NaN, None, None, NaN, …`` (the wrapped header of
pandas' last continuation block) and the actual rows that mattered
were already off-screen.

New behaviour: keep both ends with a marker between, so the start
(column names, dtypes, first head rows) and the end (any
helper-printed summary) both survive.
"""

from __future__ import annotations

from pathlib import Path

from nora.ui import _read_raw_logs


def _write_run_dir(tmp_path: Path, stdout: str, stderr: str = "") -> Path:
    """Create a fake run dir with the given log contents."""
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    return run_dir


def test_short_log_passes_through_untouched(tmp_path: Path) -> None:
    """Anything inside the 32 KB cap is returned verbatim — no
    truncation marker, no slicing."""
    run_dir = _write_run_dir(tmp_path, "small output\n")
    out, err = _read_raw_logs(str(run_dir))
    assert out == "small output\n"
    assert err == ""


def test_long_log_keeps_both_head_and_tail(tmp_path: Path) -> None:
    """The exact regression: previously only the tail survived, so a
    df.head() on a wide table dropped its column names and dtypes
    block. Now the start of the log (where the useful exploratory
    output lives) AND the end (where helper-printed summaries live)
    are both visible."""
    head_marker = "FIRST_LINE_HAS_COLUMN_NAMES_AND_DTYPES\n"
    tail_marker = "LAST_LINE_IS_THE_HELPER_SUMMARY\n"
    middle_filler = ("a" * 1024 + "\n") * 64  # ~64 KB of fluff
    content = head_marker + middle_filler + tail_marker

    run_dir = _write_run_dir(tmp_path, content)
    out, _ = _read_raw_logs(str(run_dir))

    # Both ends survive.
    assert head_marker.strip() in out, (
        "head of the log should survive truncation — researchers "
        "need column names from df.head()"
    )
    assert tail_marker.strip() in out, (
        "tail of the log should survive truncation — runtime "
        "helpers print their summary right before result()"
    )

    # Truncation marker explains the gap.
    assert "truncated from the middle" in out


def test_truncation_marker_points_at_full_log_path(tmp_path: Path) -> None:
    """The truncation note should give the researcher the on-disk
    path so they can open the full log when something interesting
    fell into the dropped middle."""
    content = "x" * (40 * 1024)
    run_dir = _write_run_dir(tmp_path, content)
    out, _ = _read_raw_logs(str(run_dir))
    assert str(run_dir) in out
    assert "stdout.log" in out


def test_total_size_under_cap_plus_marker(tmp_path: Path) -> None:
    """Truncated output must not blow past the cap by much. The
    marker adds a small fixed-size string; total stays well under
    2× the per-stream cap so the WebView doesn't choke on a
    pathologically large stdout."""
    content = "y" * (5 * 1024 * 1024)  # 5 MB of fluff
    run_dir = _write_run_dir(tmp_path, content)
    out, _ = _read_raw_logs(str(run_dir))
    # Cap is 32 KB; marker is tiny. Output must stay below ~36 KB.
    assert len(out) < 36 * 1024


def test_missing_run_dir_returns_empty(tmp_path: Path) -> None:
    """Defensive: a missing or empty run_dir hint must not raise."""
    assert _read_raw_logs(None) == ("", "")
    assert _read_raw_logs(str(tmp_path / "does-not-exist")) == ("", "")


def test_stderr_truncation_independent_of_stdout(tmp_path: Path) -> None:
    """stdout and stderr are capped separately — one stream can be
    huge while the other passes through untouched."""
    big = "z" * (40 * 1024)
    run_dir = _write_run_dir(tmp_path, "small\n", big)
    out, err = _read_raw_logs(str(run_dir))
    assert out == "small\n"
    assert "truncated from the middle" in err

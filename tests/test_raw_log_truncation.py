"""Tests for ``ui._read_raw_logs`` and ``ui._filter_stderr_boilerplate``.

The function used to apply a 32 KB head+tail cap per stream — too
tight for any moderately long regression trace, and forced the
researcher to open ``run_dir/stderr.log`` directly to cross-examine
a card. The cap is now 1 MB per stream: normal R / Stata / Python
output rides through untouched, and the cap is a safety belt for
pathological runaway logs (where head+tail truncation kicks in with
a marker pointing at the full on-disk log).

``stderr`` is also filtered to drop R package-load boilerplate
(``Attaching package`` banners, masked-objects blocks, tidyverse
decorations) without losing warnings or errors.
"""

from __future__ import annotations

from pathlib import Path

from nora.ui import (
    _RAW_LOG_STREAM_CAP,
    _REPLAY_RAW_OUTPUT_CAP,
    _REPLAY_TEXT_ENVELOPE_CAP,
    _filter_stderr_boilerplate,
    _read_raw_logs,
    _trim_event_for_replay,
)


def _write_run_dir(tmp_path: Path, stdout: str, stderr: str = "") -> Path:
    """Create a fake run dir with the given log contents."""
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# _read_raw_logs: full-fidelity stdout, filtered stderr
# ---------------------------------------------------------------------------


def test_short_log_passes_through_untouched(tmp_path: Path) -> None:
    """Anything short and free of boilerplate is returned verbatim."""
    run_dir = _write_run_dir(tmp_path, "small output\n")
    out, err = _read_raw_logs(str(run_dir))
    assert out == "small output\n"
    assert err == ""


def test_normal_regression_trace_under_cap_passes_through(
    tmp_path: Path,
) -> None:
    """A 200 KB regression trace (well under the 1 MB cap) rides
    through untouched. This is the common case — researchers see
    the entire trace inline, not a head+tail window."""
    head_marker = "FIRST_LINE_HAS_COLUMN_NAMES_AND_DTYPES"
    tail_marker = "LAST_LINE_IS_THE_HELPER_SUMMARY"
    middle_filler = ("a" * 1024 + "\n") * 200  # ~200 KB of fluff
    content = f"{head_marker}\n{middle_filler}{tail_marker}\n"
    assert len(content) < _RAW_LOG_STREAM_CAP, "test sized wrong"

    run_dir = _write_run_dir(tmp_path, content)
    out, _ = _read_raw_logs(str(run_dir))

    assert out == content, "stdout must ride to the WebView untouched"
    assert "truncated from the middle" not in out


def test_pathological_stdout_keeps_head_and_tail(tmp_path: Path) -> None:
    """A runaway log that blows past the 1 MB cap falls back to
    head+tail truncation with a marker pointing at the full on-disk
    log. Both ends survive so the start (call lines, data shape)
    and the end (final warnings, errors) stay visible."""
    head_marker = "FIRST_LINE_HAS_COLUMN_NAMES_AND_DTYPES"
    tail_marker = "LAST_LINE_IS_THE_HELPER_SUMMARY"
    # 3 MB of fluff — comfortably over the 1 MB cap so head+tail
    # truncation must engage.
    middle_filler = ("a" * 1024 + "\n") * 3000
    content = f"{head_marker}\n{middle_filler}{tail_marker}\n"

    run_dir = _write_run_dir(tmp_path, content)
    out, _ = _read_raw_logs(str(run_dir))

    assert head_marker in out, "head of long log must survive"
    assert tail_marker in out, "tail of long log must survive"
    assert "truncated from the middle" in out
    assert str(run_dir) in out, "marker must point at on-disk log"
    # Output stays bounded — not 3 MB.
    assert len(out) < _RAW_LOG_STREAM_CAP + 1024


def test_missing_run_dir_returns_empty(tmp_path: Path) -> None:
    """Defensive: a missing or empty run_dir hint must not raise."""
    assert _read_raw_logs(None) == ("", "")
    assert _read_raw_logs(str(tmp_path / "does-not-exist")) == ("", "")


def test_stderr_passes_through_when_no_boilerplate(tmp_path: Path) -> None:
    """A stderr stream with only warnings (no R package chatter) is
    returned verbatim — the filter must not touch real diagnostics."""
    stderr = (
        "Warning message:\n"
        "In glm.fit(x, y, ...) : algorithm did not converge\n"
    )
    run_dir = _write_run_dir(tmp_path, "ok\n", stderr)
    _, err = _read_raw_logs(str(run_dir))
    assert err == stderr


# ---------------------------------------------------------------------------
# _trim_event_for_replay: envelope cap must exceed sanitized payload max
# ---------------------------------------------------------------------------


def test_replay_envelope_cap_exceeds_sanitized_payload_max() -> None:
    """The sanitized ``submit_script`` envelope can carry up to
    ``_INLINE_PAYLOAD_BUDGET + _INLINE_MARKDOWN_BUDGET`` ≈ 42 KB
    (see ``tools.py``). The replay envelope cap MUST be larger,
    or moderately complex regression cards lose their JSON on
    warm-start replay and render blank.

    Regression: an earlier cap of 16 KB silently truncated the
    envelope for any regression with verbose markdown, making
    result cards "disappear" after subsequent script submits."""
    from nora.tools import _INLINE_MARKDOWN_BUDGET, _INLINE_PAYLOAD_BUDGET

    max_envelope = _INLINE_PAYLOAD_BUDGET + _INLINE_MARKDOWN_BUDGET
    assert _REPLAY_TEXT_ENVELOPE_CAP > max_envelope, (
        f"replay envelope cap ({_REPLAY_TEXT_ENVELOPE_CAP}) must "
        f"exceed sanitized envelope max ({max_envelope}) or "
        f"result cards lose their JSON on warm-start replay"
    )


def test_replay_preserves_realistic_regression_envelope() -> None:
    """A 40 KB JSON envelope (representative of a multi-regression
    batch with verbose markdown) rides through replay untouched, so
    ``renderCanonicalResultTables`` can still parse it on warm
    start."""
    import json

    payload = {
        "results": [
            {
                "result_id": f"r{i}",
                "status": "ok",
                "type": "linear_regression",
                "n": 1000,
                "markdown": "| coef | est | se |\n|---|---|---|\n" + (
                    "| beta | 0.5 | 0.1 |\n" * 500
                ),
            }
            for i in range(3)
        ],
    }
    envelope = json.dumps(payload)
    assert 30_000 < len(envelope) < _REPLAY_TEXT_ENVELOPE_CAP, (
        "test fixture sized wrong"
    )

    evt = {"type": "tool_result", "call_id": "c1", "text": envelope}
    trimmed = _trim_event_for_replay(evt)
    # Envelope rides through unchanged — parseable, full content.
    assert trimmed["text"] == envelope
    assert json.loads(trimmed["text"]) == payload


def test_replay_caps_raw_stdout_at_safety_ceiling() -> None:
    """``raw_stdout`` / ``raw_stderr`` are bounded at 1 MB on the
    replay path so a single malformed event can't drown the
    WebView. Normal output (well under the cap) passes through."""
    short = "regression output\n" * 100  # ~2 KB
    evt = {
        "type": "tool_result",
        "call_id": "c1",
        "text": "{}",
        "raw_stdout": short,
        "raw_stderr": "",
    }
    trimmed = _trim_event_for_replay(evt)
    assert trimmed["raw_stdout"] == short, "normal-size stdout untouched"

    # Pathological case: a 2 MB log must be bounded.
    huge = "x" * (2 * 1024 * 1024)
    evt["raw_stdout"] = huge
    trimmed = _trim_event_for_replay(evt)
    assert len(trimmed["raw_stdout"]) <= _REPLAY_RAW_OUTPUT_CAP + 256, (
        "replay cap must engage on pathologically large raw_stdout"
    )


# ---------------------------------------------------------------------------
# _filter_stderr_boilerplate: drop R chatter, keep warnings & errors
# ---------------------------------------------------------------------------


def test_filter_drops_loading_required_package() -> None:
    """``Loading required package: <name>`` is pure boilerplate."""
    raw = (
        "Loading required package: Matrix\n"
        "Loading required package: lme4\n"
        "Warning: matrix is singular\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert "Loading required package" not in out
    assert "Warning: matrix is singular" in out


def test_filter_drops_attaching_package_block() -> None:
    """``Attaching package: 'dplyr'`` and the masked-objects block
    that follows it (indented symbol list) both get dropped, while
    the warning afterwards survives."""
    raw = (
        "Attaching package: 'dplyr'\n"
        "\n"
        "The following objects are masked from 'package:stats':\n"
        "\n"
        "    filter, lag\n"
        "\n"
        "The following objects are masked from 'package:base':\n"
        "\n"
        "    intersect, setdiff, setequal, union\n"
        "\n"
        "Warning message:\n"
        "In model$fit() : convergence not reached\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert "Attaching package" not in out
    assert "masked from" not in out
    assert "filter, lag" not in out
    assert "intersect, setdiff" not in out
    # Warnings ride through untouched.
    assert "Warning message:" in out
    assert "convergence not reached" in out


def test_filter_drops_tidyverse_banner() -> None:
    """Tidyverse uses Unicode horizontal-line decorations + ``✔`` /
    ``✖`` ticks. All boilerplate, all dropped."""
    raw = (
        "── Attaching core tidyverse packages ───────── tidyverse 2.0.0 ──\n"
        "✔ dplyr     1.1.4     ✔ readr     2.1.5\n"
        "✔ forcats   1.0.0     ✔ stringr   1.5.1\n"
        "── Conflicts ─────────────────────── tidyverse_conflicts() ──\n"
        "✖ dplyr::filter() masks stats::filter()\n"
        "✖ dplyr::lag()    masks stats::lag()\n"
        "Warning: model failed to converge\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert "tidyverse" not in out.lower() or "Warning" in out
    assert "✔" not in out
    assert "✖" not in out
    assert "── Attaching" not in out
    assert "── Conflicts" not in out
    assert "Warning: model failed to converge" in out


def test_filter_preserves_errors() -> None:
    """``Error in`` and ``Error:`` lines are diagnostic gold — must
    pass through even when surrounded by boilerplate."""
    raw = (
        "Loading required package: Matrix\n"
        "Error in lm.fit(x, y) : NA/NaN/Inf in 'x'\n"
        "Calls: lm -> lm.fit\n"
        "Execution halted\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert "Loading required package" not in out
    assert "Error in lm.fit" in out
    assert "Calls: lm -> lm.fit" in out
    assert "Execution halted" in out


def test_filter_handles_empty_input() -> None:
    """Empty stderr is the common case and must short-circuit."""
    assert _filter_stderr_boilerplate("") == ""


def test_filter_preserves_python_stderr() -> None:
    """Python stderr (``DeprecationWarning``, traceback frames) does
    not match any R-specific pattern and must ride through whole."""
    raw = (
        "DeprecationWarning: np.float is deprecated\n"
        "Traceback (most recent call last):\n"
        '  File "fit.py", line 42, in <module>\n'
        "    model.fit(X, y)\n"
        "ValueError: shapes (10,3) and (4,) not aligned\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert out == raw


def test_filter_drops_leading_blank_lines_left_by_filtering() -> None:
    """If the filter removes the opening lines, the card shouldn't
    begin with whitespace."""
    raw = (
        "Loading required package: Matrix\n"
        "\n"
        "\n"
        "Warning: fit is rank-deficient\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert out.startswith("Warning:")


def test_filter_handles_adjacent_masked_blocks() -> None:
    """Two ``Attaching package`` blocks back-to-back: the filter must
    re-enter masked-block mode for the second one after exiting the
    first, instead of leaking the second package's symbol list."""
    raw = (
        "Attaching package: 'dplyr'\n"
        "\n"
        "The following objects are masked from 'package:base':\n"
        "\n"
        "    filter, lag\n"
        "Attaching package: 'tidyr'\n"
        "\n"
        "The following objects are masked from 'package:dplyr':\n"
        "\n"
        "    extract, nest\n"
        "\n"
        "Warning message:\n"
        "In fit() : convergence not reached\n"
    )
    out = _filter_stderr_boilerplate(raw)
    assert "filter, lag" not in out
    assert "extract, nest" not in out
    assert "Warning message:" in out

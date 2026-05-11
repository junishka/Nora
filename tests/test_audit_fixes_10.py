"""Regression tests for the tenth batch of reviewer-flagged fixes.

The behaviors pinned here:

1. Cox PH ``n_failures`` / ``n_subjects`` get small-cell coarsening.
   The sanitizer admits both via ``_OLS_ALLOWED_INT_FIELDS`` so a Cox
   model with n=200 records, n_failures=1 would otherwise publish that
   1 exactly — the same disclosure shape ``cell_suppression_threshold``
   guards against on every other subgroup count. Below threshold the
   field is coarsened to ``<10`` (the suppression marker) and the
   transformation is logged.

2. ``_summarize_plot_helpers`` honors row and byte caps. Per-entry
   field lengths were already capped, but a script writing thousands
   of JSONL rows under ``_nora_plots/`` could ship a megabyte-scale
   ``plots.succeeded`` / ``plots.failed`` payload that bypassed
   ``_INLINE_PAYLOAD_BUDGET`` trimming (that logic only inspects
   ``payload`` / ``markdown`` on result entries).

3. ``list_session_files`` honors a global thumbnail byte budget.
   Per-image cap (3 MB) was enforced, but a session with hundreds of
   plot files could base64-inline hundreds of MB through
   ``evaluate_js`` and stall the WebView. Once the global budget is
   exhausted, the remaining rows still appear in the panel but ship
   without ``data`` — same fallback path used for >3 MB singles.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Cox PH n_failures / n_subjects small-cell suppression
# ---------------------------------------------------------------------------


def _cox_payload(**overrides: object) -> dict[str, object]:
    """Minimal stcox-shaped payload that flows through the OLS sanitizer."""
    base: dict[str, object] = {
        "type": "linear_regression",
        "n": 200,
        "response_variable": "time",
        "predictor_variables": ["treatment"],
        "coefficients": {"treatment": 0.42},
        "standard_errors": {"treatment": 0.05},
        "n_subjects": 200,
        "n_failures": 50,
    }
    base.update(overrides)
    return base


def test_cox_small_n_failures_coarsened() -> None:
    """``n_failures < cell_suppression_threshold`` must be replaced by
    the suppression marker so a rare-outcome event count never crosses
    the sanitizer boundary as an exact small integer."""
    from nora.sanitizer import sanitize
    from nora.sdc import suppression_marker

    result = sanitize(_cox_payload(n_failures=1))
    assert result.ok
    assert result.sanitized["n_failures"] == suppression_marker(10)
    assert any("n_failures" in t for t in result.transformations)


def test_cox_small_n_subjects_coarsened() -> None:
    """Subject counts can be smaller than ``n`` (records) when stset
    splits one subject across multiple episodes — the records gate
    doesn't catch that. ``n_subjects=3, n=200`` must still coarsen."""
    from nora.sanitizer import sanitize
    from nora.sdc import suppression_marker

    result = sanitize(_cox_payload(n=200, n_subjects=3, n_failures=20))
    assert result.ok
    assert result.sanitized["n_subjects"] == suppression_marker(10)
    # n_failures above threshold survives unchanged.
    assert result.sanitized["n_failures"] == 20


def test_cox_at_threshold_passes_through() -> None:
    """The threshold is strict-less-than. ``n_failures == 10`` is on
    the boundary and must survive as an integer; only values in
    ``(0, threshold)`` get coarsened."""
    from nora.sanitizer import sanitize

    result = sanitize(_cox_payload(n_failures=10, n_subjects=10))
    assert result.ok
    assert result.sanitized["n_failures"] == 10
    assert result.sanitized["n_subjects"] == 10


def test_cox_zero_failures_passes_through() -> None:
    """Zero is "no events / no subjects" and not itself disclosive
    (it's the empty case, not a rare-individual case). Coarsening
    must skip zero so a no-event Cox card still reads naturally."""
    from nora.sanitizer import sanitize

    result = sanitize(_cox_payload(n_failures=0, n_subjects=0))
    assert result.ok
    assert result.sanitized["n_failures"] == 0
    assert result.sanitized["n_subjects"] == 0


# ---------------------------------------------------------------------------
# 2. _summarize_plot_helpers row and byte caps
# ---------------------------------------------------------------------------


def test_plot_helper_summary_caps_row_count(tmp_path: Path) -> None:
    """A script that loops the plot helpers (or writes manifest.jsonl
    directly) thousands of times must not be able to ship thousands
    of entries through ``plots.succeeded`` — the cap is enforced on
    a per-call total so this surface can't bypass the inline payload
    budget."""
    from nora.tools import _PLOT_HELPER_MAX_ROWS, _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    manifest = plots_dir / "manifest.jsonl"
    overflow_factor = 4
    lines = [
        json.dumps({"file": f"p{i:04d}.png", "kind": "scatter", "label": ""})
        for i in range(_PLOT_HELPER_MAX_ROWS * overflow_factor)
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    assert len(summary["succeeded"]) <= _PLOT_HELPER_MAX_ROWS
    assert summary.get("truncated_succeeded", 0) >= 1


def test_plot_helper_summary_caps_byte_budget(tmp_path: Path) -> None:
    """The byte budget guards against a smaller number of bulky rows
    bypassing the row count cap. Total serialized size of the summary
    must stay within the byte budget (plus a small envelope overhead
    for the surrounding keys)."""
    from nora.tools import _PLOT_HELPER_MAX_BYTES, _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    errors = plots_dir / "helper_errors.jsonl"
    lines = []
    for i in range(40):
        lines.append(json.dumps({
            "helper": f"plot_residuals_{i:02d}",
            "message": (
                "No module named 'matplotlib'"
                if i % 2 == 0
                else "boom"
            ),
        }))
    errors.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    payload_bytes = len(json.dumps(summary, separators=(",", ":")))
    # The cap applies to the per-row JSON cost; the envelope (the
    # surrounding ``{succeeded, failed, ...}`` keys) adds a small
    # constant. Allow a generous overhead so the test isn't brittle
    # against future cap-adjacent fields.
    assert payload_bytes <= _PLOT_HELPER_MAX_BYTES + 4_000


def test_plot_helper_summary_under_caps_unchanged(tmp_path: Path) -> None:
    """A normal three-plot run must not see the truncation markers —
    the caps only kick in for pathological volume."""
    from nora.tools import _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    manifest = plots_dir / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"file": f"{name}.png", "kind": name, "label": ""})
            for name in ("residuals", "coefficients", "interaction")
        ) + "\n",
        encoding="utf-8",
    )

    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    assert len(summary["succeeded"]) == 3
    assert "truncated_succeeded" not in summary
    assert "truncated_failed" not in summary


# ---------------------------------------------------------------------------
# 3. Files panel global thumbnail byte budget
# ---------------------------------------------------------------------------


_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _make_png(path: Path, payload_size: int) -> None:
    """Write a file with a PNG magic header. Content is otherwise
    arbitrary — the listing path classifies by extension, not by
    actual decode."""
    path.write_bytes(_PNG_HEADER + b"\x00" * payload_size)


def test_files_panel_thumbnail_global_budget(tmp_path: Path) -> None:
    """Many under-cap images that collectively exceed the global
    budget must stop receiving inline ``data`` once the budget is
    exhausted. Rows still appear in the panel (researcher can
    click-to-open) but their bytes don't ride through ``evaluate_js``."""
    from nora.ui import NoraBridge

    cwd = tmp_path / "session"
    cwd.mkdir()
    # Each PNG is ~2 MB raw; 32 MB budget fits ~12 of them after
    # base64 expansion (~2.67 MB encoded each). 30 files reliably
    # exceed the budget.
    for i in range(30):
        _make_png(cwd / f"plot_{i:02d}.png", payload_size=2 * 1024 * 1024)

    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    rows = [r for r in res["files"] if r["name"].startswith("plot_")]
    assert len(rows) == 30, "every PNG must surface in the listing"
    with_data = [r for r in rows if "data" in r]
    without_data = [r for r in rows if "data" not in r]
    assert with_data, "at least the first few rows must carry thumbnail data"
    assert without_data, (
        "the global budget must skip ``data`` on the tail rows; "
        "otherwise the bridge can ship hundreds of MB to the WebView"
    )


def test_files_panel_thumbnail_small_session_unchanged(tmp_path: Path) -> None:
    """A normal session well under the budget must not see any rows
    stripped — the cap is a defense for pathological volume, not a
    routine downgrade."""
    from nora.ui import NoraBridge

    cwd = tmp_path / "session"
    cwd.mkdir()
    # Three small PNGs — well under both per-row and global caps.
    for i in range(3):
        _make_png(cwd / f"plot_{i}.png", payload_size=10_000)

    bridge = NoraBridge(cwd=cwd)
    res = bridge.list_session_files()
    rows = [r for r in res["files"] if r["name"].startswith("plot_")]
    assert len(rows) == 3
    for r in rows:
        assert "data" in r, (
            f"small-session rows must carry inline thumbnail bytes; "
            f"missing on {r['name']}"
        )
        assert r["mime"] == "image/png"

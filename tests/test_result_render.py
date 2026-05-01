"""Tests for canonical sanitized-payload rendering.

Covers ``nora.result_render.render_table`` for every analysis type
the sanitizer accepts. The bar is "the markdown comes back, has the
expected header columns, includes one row per real entry, and
preserves suppression markers."

We also exercise the new ``expand_result(view="markdown")`` view
end-to-end so the wire path is pinned alongside the renderer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nora.config import use_cwd
from nora.result_render import render_table
from nora.store import StoredResult, get_store, reset_store_for_tests
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    text_block = next(
        b for b in payload["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


# ---------------------------------------------------------------------------
# render_table — per-type
# ---------------------------------------------------------------------------


def test_render_unknown_type_returns_none() -> None:
    assert render_table({"type": "totally_unknown"}) is None
    assert render_table({}) is None
    assert render_table({"type": None}) is None


def test_fmt_num_never_uses_scientific_notation() -> None:
    """Estimate / Std. Error formatting stays fixed-point even for
    very small or very large magnitudes — 4 sig figs. Pin against
    the previous ``e-04`` / ``e+06`` hop the user explicitly asked
    to remove from the regression card."""
    from nora.result_render import _fmt_num

    # Very small: 4 sig figs in fixed notation, no e-notation.
    # Trailing zeros trimmed (the input ``0.000383`` carries 3 sig
    # figs, so showing ``0.0003830`` would falsely advertise 4).
    assert _fmt_num(0.000383) == "0.000383"
    assert _fmt_num(1.8e-06) == "0.0000018"
    assert _fmt_num(-0.000383) == "-0.000383"
    # Common regression-coefficient range.
    assert _fmt_num(0.02158) == "0.02158"
    assert _fmt_num(0.004531) == "0.004531"
    assert _fmt_num(-0.009297) == "-0.009297"
    assert _fmt_num(13.58) == "13.58"
    # Trailing zeros trimmed.
    assert _fmt_num(0.5) == "0.5"
    assert _fmt_num(10.0) == "10"
    # Large magnitudes round to 4 sig figs but stay fixed-point.
    assert _fmt_num(1234567) == "1235000"
    assert _fmt_num(-1234567) == "-1235000"
    # Edge cases.
    assert _fmt_num(0) == "0"
    assert _fmt_num(None) == ""
    assert _fmt_num(float("nan")) == ""
    assert _fmt_num("<10") == "<10"
    # No string from this formatter contains an exponent marker.
    for v in (1.8e-06, 0.000383, 0.02158, 13.58, 1234567, -1234567, 1e-15):
        assert "e" not in _fmt_num(v).lower(), f"scientific leaked for {v}"


def test_fmt_pvalue_publication_style() -> None:
    """P-values render in publication style: 3 decimals, ``<0.001``
    floor for very small values (a bare ``0.000`` reads as exactly
    zero, which it isn't), ``>0.999`` ceiling on the high end. Pin
    against the previous use of ``_fmt_num`` which switched to
    scientific notation below 1e-3 (``1.800e-06``) — too noisy for a
    card the researcher reads."""
    from nora.result_render import _fmt_pvalue

    # Floor: anything < 0.001 collapses to "<0.001".
    assert _fmt_pvalue(1.8e-06) == "<0.001"
    assert _fmt_pvalue(0.0009) == "<0.001"
    assert _fmt_pvalue(0.0) == "<0.001"
    # Three-decimal band.
    assert _fmt_pvalue(0.001) == "0.001"
    assert _fmt_pvalue(0.0023) == "0.002"
    assert _fmt_pvalue(0.05) == "0.050"
    assert _fmt_pvalue(0.222) == "0.222"
    assert _fmt_pvalue(0.999) == "0.999"
    # Ceiling: anything > 0.999 collapses to ">0.999".
    assert _fmt_pvalue(0.9995) == ">0.999"
    assert _fmt_pvalue(1.0) == ">0.999"
    # Missing / None / non-finite / out-of-range render empty (matches
    # the rest of the renderer's "blank cell, not a fake zero" rule).
    assert _fmt_pvalue(None) == ""
    assert _fmt_pvalue(float("nan")) == ""
    assert _fmt_pvalue(-0.1) == ""  # invalid range
    assert _fmt_pvalue(1.5) == ""
    # Suppression markers pass through unchanged.
    assert _fmt_pvalue("<10") == "<10"


def test_render_linear_regression_pvalues_use_publication_format() -> None:
    """The regression card's p-value column uses the 3-decimal /
    ``<0.001`` formatter, NOT scientific notation. Pin against the
    user-visible formatting the researcher reads off the card."""
    payload = {
        "type": "linear_regression",
        "n": 561758,
        "coefficients": {"a_ym2": 0.02158, "a_yp1": 0.01348, "fp_yp3": -0.009297},
        "standard_errors": {"a_ym2": 0.004531, "a_yp1": 0.003795, "fp_yp3": 0.007613},
        # First two were the noisy-scientific cases under _fmt_num
        # (1.8e-06, 3.83e-04); the third is a plain mid-range value.
        "p_values": {"a_ym2": 1.8e-06, "a_yp1": 0.000383, "fp_yp3": 0.222},
        "response_variable": "y",
        "predictor_variables": ["a_ym2", "a_yp1", "fp_yp3"],
    }
    md = render_table(payload)
    assert md is not None
    assert "<0.001" in md, f"expected <0.001 floor, got:\n{md}"
    assert "0.222" in md
    # Scientific notation must not leak into the p-value column.
    assert "e-06" not in md and "e-04" not in md, (
        f"scientific notation leaked into p-value column:\n{md}"
    )


def test_render_linear_regression_minimal() -> None:
    payload = {
        "type": "linear_regression",
        "n": 100,
        "coefficients": {"x1": 0.42, "x2": -0.13},
        "standard_errors": {"x1": 0.05, "x2": 0.04},
        "p_values": {"x1": 0.001, "x2": 0.06},
        "r_squared": 0.31,
        "response_variable": "y",
        "predictor_variables": ["x1", "x2"],
    }
    md = render_table(payload)
    assert md is not None
    # Header columns present.
    assert "Term" in md
    assert "Estimate" in md
    assert "Std. Error" in md
    assert "p-value" in md
    # One row per coefficient.
    assert "x1" in md and "x2" in md
    # Footer carries n and R².
    assert "n = 100" in md
    assert "R²" in md


def test_render_linear_regression_drops_p_value_column_when_payload_missing_it(
) -> None:
    """When the payload doesn't carry ``p_values`` (legacy script,
    custom emitter, robust-SE path that didn't compute t-stats), the
    p-value column is omitted rather than rendered as a column of
    blanks. Term, Estimate, and Std. Error remain. The model's
    inline composite tables enforce the p-value contract via the
    prompt; this is the canonical-renderer fallback for malformed
    payloads."""
    payload = {
        "type": "linear_regression",
        "n": 50,
        "coefficients": {"x1": 0.5},
        "standard_errors": {"x1": 0.1},
        "response_variable": "y",
        "predictor_variables": ["x1"],
    }
    md = render_table(payload)
    assert md is not None
    assert "p-value" not in md
    assert "Std. Error" in md


def test_render_t_test_two_sample() -> None:
    payload = {
        "type": "t_test",
        "test_type": "two_sample",
        "n1": 100, "n2": 80,
        "mean1": 1.5, "mean2": 1.2,
        "sd1": 0.4, "sd2": 0.3,
        "mean_difference": 0.3,
        "t_statistic": 5.4,
        "p_value": 0.0001,
    }
    md = render_table(payload)
    assert md is not None
    assert "Group" in md
    assert "Mean diff: 0.3" in md
    assert "p = " in md
    # Both groups.
    lines = md.splitlines()
    body_lines = [ln for ln in lines if ln.startswith("|") and "---" not in ln]
    assert len(body_lines) == 3  # header + 2 group rows


def test_render_descriptive_with_optional_min_max() -> None:
    payload = {
        "type": "descriptive",
        "variable": "age",
        "n": 200, "mean": 35.4, "sd": 12.1,
        "missing_count": 5,
        "min_value": 18, "max_value": 80,
    }
    md = render_table(payload)
    assert md is not None
    assert "Min" in md and "Max" in md
    assert "age" in md
    assert "200" in md


def test_render_descriptive_without_min_max() -> None:
    payload = {
        "type": "descriptive",
        "variable": "salary",
        "n": 100, "mean": 50000, "sd": 12000,
        "missing_count": 0,
    }
    md = render_table(payload)
    assert md is not None
    assert "Min" not in md
    assert "Max" not in md


def test_render_frequency_table_preserves_suppression_markers() -> None:
    payload = {
        "type": "frequency_table",
        "variable": "region",
        "n": 500,
        "missing_count": 0,
        "counts": {"north": 200, "south": 290, "rare": "<10"},
    }
    md = render_table(payload)
    assert md is not None
    assert "Level" in md and "Count" in md and "Proportion" in md
    # Suppression marker preserved verbatim.
    assert "<10" in md
    # Proportion rendered for normal rows.
    assert "0.400" in md or "0.4" in md  # 200/500


def test_render_crosstab_2d() -> None:
    payload = {
        "type": "crosstab",
        "row_variable": "region",
        "col_variable": "gender",
        "missing_count": 0,
        "counts": {
            "north": {"F": 90, "M": 110},
            "south": {"F": 140, "M": 150},
        },
    }
    md = render_table(payload)
    assert md is not None
    assert "region" in md
    assert "F" in md and "M" in md
    assert "north" in md and "south" in md


def test_render_magnitude_table_sum() -> None:
    payload = {
        "type": "magnitude_table",
        "row_variable": "region",
        "value_variable": "revenue",
        "aggregation": "sum",
        "cells": {
            "north": {"value": 1234567.0, "n": 100, "max_share": 0.05},
            "south": {"value": "<suppressed>", "n": 50, "max_share": 0.95},
        },
    }
    md = render_table(payload)
    assert md is not None
    # Aggregation column reflects sum.
    assert "Sum" in md
    # Suppression marker passes through.
    assert "<suppressed>" in md


def test_render_correlation_matrix_has_diagonal_ones() -> None:
    payload = {
        "type": "correlation_matrix",
        "n": 200,
        "method": "pearson",
        "variables": ["x", "y", "z"],
        "correlations": {
            "x": {"y": 0.4, "z": -0.1},
            "y": {"x": 0.4, "z": 0.2},
            "z": {"x": -0.1, "y": 0.2},
        },
    }
    md = render_table(payload)
    assert md is not None
    # Diagonals filled with 1.000.
    assert "1.000" in md
    # All three variables on both axes.
    for v in ("x", "y", "z"):
        assert v in md
    assert "method: pearson" in md


# ---------------------------------------------------------------------------
# expand_result(view="markdown") end-to-end
# ---------------------------------------------------------------------------


def test_expand_result_markdown_view_returns_canonical_table(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        store = get_store(cwd)
        row: StoredResult = store.insert(
            label="OLS y ~ x1 + x2",
            analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression",
                "n": 200,
                "coefficients": {"x1": 0.4, "x2": -0.1},
                "standard_errors": {"x1": 0.05, "x2": 0.04},
                "p_values": {"x1": 0.001, "x2": 0.05},
                "r_squared": 0.25,
                "response_variable": "y",
                "predictor_variables": ["x1", "x2"],
            },
            language="R",
            script_code="lm(y ~ x1 + x2, data=df)",
            transformations=[],
        )
        with use_cwd(cwd):
            res = asyncio.run(HANDLERS["expand_result"]({
                "result_id": row.id, "view": "markdown",
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        assert body.get("view") == "markdown"
        md = body.get("markdown")
        assert isinstance(md, str)
        # Canonical table shape.
        assert "Term" in md and "Estimate" in md
        assert "x1" in md and "x2" in md
        assert "n = 200" in md
        # Payload still ships in full (the markdown view ADDS
        # the rendered field; doesn't replace the JSON).
        assert body["payload"]["coefficients"] == {"x1": 0.4, "x2": -0.1}
    finally:
        reset_store_for_tests()


def test_expand_result_markdown_view_unknown_payload_omits_field(
    tmp_path: Path,
) -> None:
    """If the stored payload type isn't one the renderer knows, the
    response should still come back ok — just without the ``markdown``
    field. Callers fall back to the JSON ``payload``."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        store = get_store(cwd)
        row = store.insert(
            label="exotic",
            analysis_type="totally_unknown",
            sanitized_payload={"type": "totally_unknown", "value": 42},
            language="Python",
            script_code="",
            transformations=[],
        )
        with use_cwd(cwd):
            res = asyncio.run(HANDLERS["expand_result"]({
                "result_id": row.id, "view": "markdown",
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        assert "markdown" not in body
    finally:
        reset_store_for_tests()


def test_expand_result_unknown_view_rejected(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        store = get_store(cwd)
        row = store.insert(
            label="x", analysis_type="descriptive",
            sanitized_payload={
                "type": "descriptive", "variable": "x",
                "n": 10, "mean": 1.0, "sd": 0.5, "missing_count": 0,
            },
            language="Python", script_code="", transformations=[],
        )
        with use_cwd(cwd):
            res = asyncio.run(HANDLERS["expand_result"]({
                "result_id": row.id, "view": "garbage",
            }))
        body = _mcp_text(res)
        assert body["status"] == "error"
        assert "view must be" in body["reason"]
        assert "markdown" in body["reason"]
    finally:
        reset_store_for_tests()

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


def test_render_linear_regression_without_p_values_drops_column() -> None:
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

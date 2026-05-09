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
from nora.result_render import compose_layout, render_table
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


def test_render_linear_regression_caption_includes_pseudo_r_squared() -> None:
    """Logit / probit / Poisson payloads carry ``pseudo_r_squared``
    (McFadden) but no ``r_squared``. The caption must expose pseudo
    R² rather than dropping the fit metric silently."""
    payload = {
        "type": "linear_regression",
        "n": 800,
        "coefficients": {"x1": 0.42, "(Intercept)": -0.5},
        "standard_errors": {"x1": 0.05, "(Intercept)": 0.1},
        "p_values": {"x1": 0.001, "(Intercept)": 0.001},
        "pseudo_r_squared": 0.142,
        "log_likelihood": -312.4,
        "chi_squared": 88.7,
        "chi_squared_p_value": 1e-18,
        "response_variable": "y",
        "predictor_variables": ["x1"],
    }
    md = render_table(payload)
    assert md is not None
    assert "pseudo R²" in md, md
    assert "0.142" in md
    assert "log-lik" in md
    assert "χ²" in md
    # Omnibus χ² renders alongside its p-value when present.
    assert "<0.001" in md


def test_render_linear_regression_caption_cox_subjects_and_concordance(
) -> None:
    """Cox PH payloads expose ``n_subjects`` / ``n_failures`` and
    ``concordance``. The caption must surface those because for survival
    models the OLS-style ``n = ...`` line alone is the wrong sample
    metric — researchers read off "S subjects, E events". Records ``n``
    is shown only when it diverges from subjects (split-episode data)."""
    payload = {
        "type": "linear_regression",
        "n": 412,
        "n_subjects": 324,
        "n_failures": 178,
        "coefficients": {"treatment": -0.31},
        "standard_errors": {"treatment": 0.08},
        "p_values": {"treatment": 0.0001},
        "log_likelihood": -921.5,
        "concordance": 0.74,
        "response_variable": "_t",
        "predictor_variables": ["treatment"],
    }
    md = render_table(payload)
    assert md is not None
    assert "subjects = 324" in md
    assert "events = 178" in md
    # Records leg appears because n != n_subjects (split episodes).
    assert "records = 412" in md
    assert "C = 0.74" in md
    assert "log-lik" in md
    # OLS R² line must NOT appear — payload didn't carry it.
    assert "R² = " not in md or "pseudo R² = " in md


def test_render_linear_regression_caption_ols_unchanged() -> None:
    """The OLS caption must keep its original ``n · R² · κ(X)`` form so
    existing renderings don't shift when the new optional metrics are
    absent."""
    payload = {
        "type": "linear_regression",
        "n": 1000,
        "coefficients": {"x1": 0.5},
        "standard_errors": {"x1": 0.05},
        "r_squared": 0.27,
        "condition_number": 18.4,
        "response_variable": "y",
        "predictor_variables": ["x1"],
    }
    md = render_table(payload)
    assert md is not None
    caption_line = md.split("\n\n")[-1]
    assert caption_line == "n = 1,000 · R² = 0.27 · κ(X) = 18.4"


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


def test_render_escapes_pipe_in_data_origin_cell_keys() -> None:
    """A coefficient key from the researcher's data may legitimately
    contain ``|`` (categorical level "A | B" for "A or B"). Without
    escaping, the pipe table grows an extra column and every
    following row reads as garbage. Sanitised cell text must escape
    delimiter characters at the renderer boundary."""
    payload = {
        "type": "linear_regression",
        "n": 100,
        "coefficients": {"A | B": 0.42, "x1": 0.13},
        "standard_errors": {"A | B": 0.05, "x1": 0.04},
        "p_values": {"A | B": 0.001, "x1": 0.06},
        "response_variable": "y",
        "predictor_variables": ["A | B", "x1"],
    }
    md = render_table(payload)
    assert md is not None
    # Each row must keep exactly the right number of cell delimiters
    # (4 columns → 5 unescaped pipes per row when surrounded by
    # ``|`` … ``|``). The escaped pipe inside the cell shows as
    # ``\|`` and does NOT count as a column boundary.
    body_rows = [
        ln for ln in md.splitlines()
        if ln.startswith("|") and "---" not in ln
    ]
    assert body_rows, md
    for row in body_rows:
        # Strip escaped pipes before counting unescaped delimiters.
        unescaped = row.replace("\\|", "")
        assert unescaped.count("|") == 5, (
            f"row has wrong delimiter count after pipe escape:\n{row}"
        )
    # The escaped form is what reaches the markdown source.
    assert "A \\| B" in md


def test_render_escapes_backslash_in_data_origin_cell_keys() -> None:
    """A literal backslash in a category label must round-trip through
    the table without being interpreted as an escape character of
    something else (e.g., another pipe)."""
    payload = {
        "type": "linear_regression",
        "n": 50,
        "coefficients": {"path\\A": 0.5},
        "standard_errors": {"path\\A": 0.05},
        "response_variable": "y",
        "predictor_variables": ["path\\A"],
    }
    md = render_table(payload)
    assert md is not None
    # Backslash is doubled in the markdown source so it renders as a
    # literal backslash in the visible output.
    assert "path\\\\A" in md


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
        # When markdown renders successfully, the JSON ``payload``
        # is dropped from the response — shipping both is real
        # cost duplication (kilobytes for a wide regression) and
        # the model's only reason to ask for ``view="markdown"`` is
        # the rendered table, not the raw arrays. ``view="full"``
        # remains the way to get the JSON back.
        assert "payload" not in body
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
        # Renderer didn't know this payload type, so the JSON
        # falls back in. Without this the model would get an empty
        # response on a markdown view of an unrecognised payload —
        # worse than just returning the raw fields.
        assert body["payload"] == {"type": "totally_unknown", "value": 42}
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


# ---------------------------------------------------------------------------
# compose_layout — multi-result composite tables
# ---------------------------------------------------------------------------


def _payload(coefs, ses, pvals=None):
    """Compact constructor for a fake regression payload."""
    out = {
        "type": "linear_regression",
        "coefficients": dict(coefs),
        "standard_errors": dict(ses),
    }
    if pvals is not None:
        out["p_values"] = dict(pvals)
    return out


def test_compose_layout_happy_path() -> None:
    """A single-group spec with two rows and two columns produces a
    table whose cells are looked up from the payload store. Pin the
    cell shape (``estimate (SE) [p-value]``) and the column-label /
    row-label rendering."""
    payloads = {
        "M1": _payload(
            {"x": 0.020, "y": -0.010},
            {"x": 0.005, "y": 0.004},
            {"x": 0.001, "y": 0.022},
        ),
        "M2": _payload(
            {"x": 0.015, "y": 0.005},
            {"x": 0.003, "y": 0.006},
            {"x": 0.0001, "y": 0.4},
        ),
    }
    spec = {
        "title": "Mechanism A",
        "columns": [{"id": "x", "label": "treat × t0"},
                    {"id": "y", "label": "treat × t+1"}],
        "groups": [
            {"label": None, "rows": [
                {"result_id": "M1", "label": "ln_rev"},
                {"result_id": "M2", "label": "ln_exp"},
            ]},
        ],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    assert "**Mechanism A**" in md
    assert "treat × t0" in md and "treat × t+1" in md
    assert "ln_rev" in md and "ln_exp" in md
    # Cell shape: estimate (SE) [p-value], no scientific notation,
    # publication-style p-value formatting (3 decimals or <0.001).
    assert "0.02 (0.005) [0.001]" in md or "0.02 (0.005) [<0.001]" in md
    assert "[<0.001]" in md  # M2 row x has p=0.0001


def test_compose_layout_hallucinated_result_id_renders_em_dash() -> None:
    """The structural guarantee: a result_id the model invented (or
    typed wrong) renders as a row of ``—`` cells, NOT a hallucinated
    coefficient. Grouping is fallible; the numbers aren't. Pin this
    against any future "fall back to a guess" change."""
    payloads = {
        "M1": _payload({"x": 0.5}, {"x": 0.05}, {"x": 0.001}),
    }
    spec = {
        "columns": [{"id": "x", "label": "x"}],
        "groups": [{"label": None, "rows": [
            {"result_id": "M1", "label": "real"},
            {"result_id": "M_BOGUS", "label": "made up"},
        ]}],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    lines = md.splitlines()
    bogus_line = next(ln for ln in lines if "made up" in ln)
    # Every cell on the bogus row should be the dash, not a number.
    cells = [c.strip() for c in bogus_line.split("|") if c.strip()]
    assert cells[0] == "made up"
    for c in cells[1:]:
        assert c == "—", f"hallucinated row leaked a non-dash cell: {c!r}"


def test_compose_layout_missing_term_id_renders_em_dash() -> None:
    """Term IDs not in the payload's coefficients dict render as
    ``—`` for that cell (the result existed, but didn't carry the
    requested coefficient — e.g., a robust-SE estimator that didn't
    populate p_values, or a column the spec asked for that wasn't
    in the regression)."""
    payloads = {
        "M1": _payload({"x": 0.5}, {"x": 0.05}, {"x": 0.001}),
    }
    spec = {
        "columns": [
            {"id": "x", "label": "x"},
            {"id": "z_not_in_model", "label": "z"},
        ],
        "groups": [{"label": None, "rows": [
            {"result_id": "M1", "label": "row1"},
        ]}],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    line = next(ln for ln in md.splitlines() if "row1" in ln)
    cells = [c.strip() for c in line.split("|") if c.strip()]
    assert cells[0] == "row1"
    assert cells[1].startswith("0.5")  # x cell has values
    assert cells[2] == "—", f"missing-term cell should be '—', got {cells[2]!r}"


def test_compose_layout_multi_group_has_bold_header_rows() -> None:
    """Group labels render as bold first-cell header rows above their
    member rows. Markdown pipe tables don't support row spans, so the
    blank-cells convention is what we pin."""
    payloads = {
        "M1": _payload({"x": 0.1}, {"x": 0.01}),
        "M2": _payload({"x": 0.2}, {"x": 0.02}),
    }
    spec = {
        "columns": [{"id": "x", "label": "x"}],
        "groups": [
            {"label": "H1: direct", "rows": [
                {"result_id": "M1", "label": "outcome A"}]},
            {"label": "H2: indirect", "rows": [
                {"result_id": "M2", "label": "outcome B"}]},
        ],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    assert "**H1: direct**" in md
    assert "**H2: indirect**" in md
    # Headers come BEFORE their member rows.
    pos_h1 = md.find("**H1: direct**")
    pos_a = md.find("outcome A")
    pos_h2 = md.find("**H2: indirect**")
    pos_b = md.find("outcome B")
    assert 0 < pos_h1 < pos_a < pos_h2 < pos_b


def test_compose_layout_returns_none_for_malformed_specs() -> None:
    """Malformed specs return None instead of raising or rendering
    garbage. Caller falls back to its default error handling."""
    payloads = {"M1": _payload({"x": 0.5}, {"x": 0.05})}
    # Not a dict.
    assert compose_layout("not a spec", payloads) is None  # type: ignore[arg-type]
    # Missing columns.
    assert compose_layout({"groups": [{"rows": [{"result_id": "M1"}]}]}, payloads) is None
    # Empty columns.
    assert compose_layout(
        {"columns": [], "groups": [{"rows": [{"result_id": "M1"}]}]},
        payloads,
    ) is None
    # Missing groups.
    assert compose_layout({"columns": [{"id": "x", "label": "x"}]}, payloads) is None
    # Group missing rows.
    assert compose_layout(
        {"columns": [{"id": "x", "label": "x"}], "groups": [{}]},
        payloads,
    ) is None
    # Row missing result_id.
    assert compose_layout(
        {
            "columns": [{"id": "x", "label": "x"}],
            "groups": [{"rows": [{"label": "no id"}]}],
        },
        payloads,
    ) is None


def test_compose_layout_trichotomy_distinguishes_failure_modes() -> None:
    """Three distinct failure glyphs let a researcher tell why a cell
    is empty:

    - ``—``: result_id wasn't found (model hallucinated, or deleted).
    - ``·``: term IS in this model's predictors but no estimate came
      back. In OLS this almost always means perfect collinearity.
    - ``n/a``: term isn't part of this model at all (different model
      family, or the spec asked for a column some models don't have).

    Pinning all three requires a payload that DOES carry
    ``predictor_variables``; without that list we can't tell case 2
    from case 3 and the renderer falls back to ``—`` (covered by
    ``test_compose_layout_missing_term_id_renders_em_dash``).
    """
    # M1 declares x1 + x2 but only x1 has an estimate (x2 was perfect-
    # collinearity-dropped). M2 declares x3 only.
    payloads = {
        "M1": {
            "type": "linear_regression",
            "coefficients": {"x1": 0.5},
            "standard_errors": {"x1": 0.05},
            "predictor_variables": ["x1", "x2"],
        },
        "M2": {
            "type": "linear_regression",
            "coefficients": {"x3": 0.7},
            "standard_errors": {"x3": 0.06},
            "predictor_variables": ["x3"],
        },
    }
    spec = {
        "columns": [
            {"id": "x1", "label": "x1"},
            {"id": "x2", "label": "x2"},
            {"id": "x3", "label": "x3"},
        ],
        "groups": [{"label": None, "rows": [
            {"result_id": "M1", "label": "row1"},
            {"result_id": "M2", "label": "row2"},
            {"result_id": "M_BOGUS", "label": "row3"},
        ]}],
    }
    md = compose_layout(spec, payloads)
    assert md is not None

    def _cells(line_substr: str) -> list[str]:
        line = next(ln for ln in md.splitlines() if line_substr in ln)
        return [c.strip() for c in line.split("|") if c.strip()]

    row1 = _cells("row1")
    # x1 has data; x2 is declared but no estimate (collinear);
    # x3 isn't in M1's model at all.
    assert row1[1].startswith("0.5"), f"x1 cell should have data: {row1[1]!r}"
    assert row1[2] == "·", f"x2 should be middle dot (collinear): {row1[2]!r}"
    assert row1[3] == "n/a", f"x3 should be n/a (not in model): {row1[3]!r}"

    row2 = _cells("row2")
    assert row2[1] == "n/a", f"x1 not in M2's model: {row2[1]!r}"
    assert row2[2] == "n/a", f"x2 not in M2's model: {row2[2]!r}"
    assert row2[3].startswith("0.7"), f"x3 cell should have data: {row2[3]!r}"

    row3 = _cells("row3")
    # Hallucinated row: every cell is em-dash regardless of predictors.
    for cell in row3[1:]:
        assert cell == "—", f"hallucinated row leaked: {cell!r}"

    # Legend should appear because at least one cell rendered as a
    # failure glyph.
    assert "Legend:" in md
    assert "result not found" in md
    assert "perfect" in md  # collinearity description
    assert "not part of this model" in md


def test_compose_layout_legend_persists_across_groups() -> None:
    """A failure glyph in an EARLIER group must still trigger the
    legend even when the FINAL group renders cleanly.

    Earlier shape reset ``any_unresolved`` inside the per-group loop,
    so a clean last group silently dropped the legend — leaving the
    table with unexplained ``—`` / ``·`` / ``n/a`` glyphs from
    earlier groups."""
    payloads = {
        "M_OK": {
            "type": "linear_regression",
            "coefficients": {"x": 0.5},
            "standard_errors": {"x": 0.05},
            "p_values": {"x": 0.001},
            "predictor_variables": ["x"],
        },
        # M_BAD is intentionally missing from payloads → renders ``—``
    }
    spec = {
        "columns": [{"id": "x", "label": "x"}],
        "groups": [
            {"label": "first", "rows": [
                {"result_id": "M_BAD", "label": "missing-row"},
            ]},
            {"label": "second", "rows": [
                {"result_id": "M_OK", "label": "clean-row"},
            ]},
        ],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    # The em-dash from the first group must explain via legend even
    # though the second (final) group has no failure glyphs.
    assert "—" in md
    assert "Legend:" in md, (
        "Legend was dropped because final group resolved cleanly — "
        "any_unresolved must persist across groups"
    )


def test_compose_layout_missing_pvalues_render_without_dot() -> None:
    """A payload with coefficients + SEs but no p-values must NOT
    render cells like ``2 (0.2) [·]`` — the ``·`` glyph would be
    misread as the legend's collinearity marker even though the term
    IS estimated. The format degrades to ``2 (0.2)`` instead.
    Same posture for missing SEs."""
    payloads = {
        "M_NO_P": {
            "type": "linear_regression",
            "coefficients": {"x": 0.5},
            "standard_errors": {"x": 0.05},
            # no p_values dict at all
            "predictor_variables": ["x"],
        },
        "M_NO_SE": {
            "type": "linear_regression",
            "coefficients": {"y": 0.7},
            # no standard_errors dict
            "p_values": {"y": 0.001},
            "predictor_variables": ["y"],
        },
    }
    spec = {
        "columns": [
            {"id": "x", "label": "x"},
            {"id": "y", "label": "y"},
        ],
        "groups": [{"label": None, "rows": [
            {"result_id": "M_NO_P", "label": "row-no-p"},
            {"result_id": "M_NO_SE", "label": "row-no-se"},
        ]}],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    # Locate the rendered cells.
    def _cells(line_substr: str) -> list[str]:
        line = next(ln for ln in md.splitlines() if line_substr in ln)
        return [c.strip() for c in line.split("|") if c.strip()]

    no_p = _cells("row-no-p")
    # x cell: coef + SE present, p missing → ``0.5 (0.05)`` with no
    # ``[·]`` glyph appended.
    assert no_p[1].startswith("0.5")
    assert "(0.05)" in no_p[1]
    assert "·" not in no_p[1], f"missing-p cell shouldn't carry ·: {no_p[1]!r}"
    assert "[" not in no_p[1], f"empty p-brackets leaked: {no_p[1]!r}"

    no_se = _cells("row-no-se")
    # y cell: coef + p present, SE missing → ``0.7 [0.001]`` (or
    # similar) with no ``(·)`` glyph.
    assert no_se[2].startswith("0.7")
    assert "·" not in no_se[2], f"missing-SE cell shouldn't carry ·: {no_se[2]!r}"
    assert "(" not in no_se[2], f"empty SE parens leaked: {no_se[2]!r}"


def test_compose_layout_no_legend_when_all_cells_resolve() -> None:
    """When every cell resolves to a data value, the legend is
    suppressed — its only purpose is to disambiguate failure glyphs
    that didn't appear here. Keeps clean composites uncluttered."""
    payloads = {
        "M1": {
            "type": "linear_regression",
            "coefficients": {"x": 0.5},
            "standard_errors": {"x": 0.05},
            "p_values": {"x": 0.001},
            "predictor_variables": ["x"],
        },
    }
    spec = {
        "columns": [{"id": "x", "label": "x"}],
        "groups": [{"label": None, "rows": [
            {"result_id": "M1", "label": "row1"},
        ]}],
    }
    md = compose_layout(spec, payloads)
    assert md is not None
    assert "Legend:" not in md


def test_compose_results_tool_renders_from_store(tmp_path: Path) -> None:
    """End-to-end: insert two regressions in the session store, call
    the ``compose_results`` tool with a spec referencing them, assert
    the response carries a markdown table whose cells came out of
    the store. Pin the wire shape (``status``, ``markdown``,
    ``rows_rendered``, ``result_ids_referenced``)."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        store = get_store(cwd)
        m1 = store.insert(
            label="H1 ln_rev",
            analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression",
                "n": 1000,
                "coefficients": {"fp_y0": 0.020, "fp_yp1": 0.015},
                "standard_errors": {"fp_y0": 0.005, "fp_yp1": 0.004},
                "p_values": {"fp_y0": 0.001, "fp_yp1": 0.002},
                "response_variable": "ln_rev",
                "predictor_variables": ["fp_y0", "fp_yp1"],
            },
            language="Stata", script_code="", transformations=[],
        )
        m2 = store.insert(
            label="H1 ln_exp",
            analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression",
                "n": 1000,
                "coefficients": {"fp_y0": 0.010, "fp_yp1": 0.005},
                "standard_errors": {"fp_y0": 0.003, "fp_yp1": 0.002},
                "p_values": {"fp_y0": 0.0001, "fp_yp1": 0.05},
                "response_variable": "ln_exp",
                "predictor_variables": ["fp_y0", "fp_yp1"],
            },
            language="Stata", script_code="", transformations=[],
        )
        with use_cwd(cwd):
            res = asyncio.run(HANDLERS["compose_results"]({
                "spec": {
                    "title": "H1 mechanism",
                    "columns": [
                        {"id": "fp_y0",  "label": "year 0"},
                        {"id": "fp_yp1", "label": "year +1"},
                    ],
                    "groups": [
                        {"label": "Direct effect", "rows": [
                            {"result_id": m1.id, "label": "ln_rev"},
                            {"result_id": m2.id, "label": "ln_exp"},
                        ]},
                    ],
                },
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        md = body["markdown"]
        assert "**H1 mechanism**" in md
        assert "**Direct effect**" in md
        assert "ln_rev" in md and "ln_exp" in md
        assert "0.02 (0.005) [0.001]" in md  # m1 fp_y0 cell
        assert "[<0.001]" in md              # m2 fp_y0 cell
        assert body["rows_rendered"] == 2
        assert sorted(body["result_ids_referenced"]) == sorted([m1.id, m2.id])
        assert "missing_result_ids" not in body
    finally:
        reset_store_for_tests()


def test_compose_results_tool_flags_missing_ids(tmp_path: Path) -> None:
    """A spec that references a result_id not in the store gets a
    ``missing_result_ids`` array back so the model can correct
    without inventing a coefficient. The rendered markdown still
    comes back, with ``—`` cells in the missing rows."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        store = get_store(cwd)
        real = store.insert(
            label="real",
            analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression",
                "n": 100,
                "coefficients": {"x": 0.5},
                "standard_errors": {"x": 0.05},
                "response_variable": "y",
                "predictor_variables": ["x"],
            },
            language="Python", script_code="", transformations=[],
        )
        with use_cwd(cwd):
            res = asyncio.run(HANDLERS["compose_results"]({
                "spec": {
                    "columns": [{"id": "x", "label": "x"}],
                    "groups": [{"rows": [
                        {"result_id": real.id, "label": "real"},
                        {"result_id": "M_BOGUS", "label": "made up"},
                    ]}],
                },
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        assert body["missing_result_ids"] == ["M_BOGUS"]
        assert "hint" in body and "list_results" in body["hint"]
        # Bogus row renders with em-dash, real row has the real value.
        md = body["markdown"]
        assert "made up" in md and "—" in md
        assert "0.5" in md
        # rows_rendered counts data rows in the layout — missing-id
        # rows still render as placeholder cells, so a 2-row spec with
        # one bogus id should report 2, not 1.
        assert body["rows_rendered"] == 2
    finally:
        reset_store_for_tests()


def test_compose_results_tool_cross_session_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row that carries ``session_path`` looks up its ``result_id``
    in THAT session's store rather than the current cwd's. Mirrors
    ``expand_result(session_path=...)`` and the symmetry the reviewer
    flagged. Gated by ``NORA_ALLOW_CROSS_SESSION_RECALL=1`` for
    parity with the rest of the cross-session surface."""
    # Two sessions, both under a fake SESSIONS_ROOT.
    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    sess_a = sessions_root / "20260101T000000Z_aaa"
    sess_a.mkdir()
    sess_b = sessions_root / "20260101T000001Z_bbb"
    sess_b.mkdir()
    monkeypatch.setattr("nora.ui.SESSIONS_ROOT", sessions_root)
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")

    reset_store_for_tests()
    try:
        # Result IDs are auto-generated (M1, M2, …) per store, so
        # both fresh stores would assign M1 to their first insert and
        # collide. Insert a placeholder in B first so its real
        # regression lands as M2, distinct from A's M1.
        store_b = get_store(sess_b)
        store_b.insert(
            label="placeholder", analysis_type="linear_regression",
            sanitized_payload={"type": "linear_regression", "n": 1},
            language="R", script_code="", transformations=[],
        )
        m_b = store_b.insert(
            label="B", analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression", "n": 75,
                "coefficients": {"x": 0.7},
                "standard_errors": {"x": 0.07},
                "response_variable": "y", "predictor_variables": ["x"],
            },
            language="R", script_code="", transformations=[],
        )
        store_a = get_store(sess_a)
        m_a = store_a.insert(
            label="A", analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression", "n": 50,
                "coefficients": {"x": 0.5},
                "standard_errors": {"x": 0.05},
                "response_variable": "y", "predictor_variables": ["x"],
            },
            language="R", script_code="", transformations=[],
        )
        assert m_a.id != m_b.id  # otherwise the collision-detection
        # would obscure the real lookup behaviour we're testing here

        with use_cwd(sess_b):
            res = asyncio.run(HANDLERS["compose_results"]({
                "spec": {
                    "columns": [{"id": "x", "label": "x"}],
                    "groups": [{"rows": [
                        {"result_id": m_b.id, "label": "in B"},
                        {
                            "result_id": m_a.id,
                            "session_path": str(sess_a),
                            "label": "in A",
                        },
                    ]}],
                },
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok", body
        # Both rows should have data cells (neither is missing or
        # denied).
        assert "missing_result_ids" not in body
        assert "denied_result_ids" not in body
        md = body["markdown"]
        # B's coefficient (0.7) and A's coefficient (0.5) both
        # appeared, proving the cross-session lookup actually
        # resolved against sess_a.
        assert "0.7" in md
        assert "0.5" in md
    finally:
        reset_store_for_tests()


def test_compose_results_tool_flags_cross_session_rid_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When two rows reference the same ``result_id`` in different
    sessions, the layout's payload dict is keyed by rid alone — one
    payload silently overwrites the other. The tool flags this in
    ``rid_collisions_across_sessions`` so the model can rename one
    of them and re-emit instead of trusting a confused render."""
    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    sess_a = sessions_root / "20260101T000000Z_aaa"
    sess_a.mkdir()
    sess_b = sessions_root / "20260101T000001Z_bbb"
    sess_b.mkdir()
    monkeypatch.setattr("nora.ui.SESSIONS_ROOT", sessions_root)
    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")

    reset_store_for_tests()
    try:
        # Both stores assign M1 to their first insert; the rid
        # collides on purpose.
        m_a = get_store(sess_a).insert(
            label="A", analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression", "n": 50,
                "coefficients": {"x": 0.5},
                "standard_errors": {"x": 0.05},
                "response_variable": "y", "predictor_variables": ["x"],
            },
            language="R", script_code="", transformations=[],
        )
        m_b = get_store(sess_b).insert(
            label="B", analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression", "n": 75,
                "coefficients": {"x": 0.7},
                "standard_errors": {"x": 0.07},
                "response_variable": "y", "predictor_variables": ["x"],
            },
            language="R", script_code="", transformations=[],
        )
        assert m_a.id == m_b.id  # the collision case under test

        with use_cwd(sess_b):
            res = asyncio.run(HANDLERS["compose_results"]({
                "spec": {
                    "columns": [{"id": "x", "label": "x"}],
                    "groups": [{"rows": [
                        {"result_id": m_b.id, "label": "in B"},
                        {
                            "result_id": m_a.id,
                            "session_path": str(sess_a),
                            "label": "in A",
                        },
                    ]}],
                },
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        assert m_a.id in body["rid_collisions_across_sessions"]
        assert "distinct" in body["hint"] or "split" in body["hint"]
    finally:
        reset_store_for_tests()


def test_compose_results_tool_denies_cross_session_when_gate_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env gate, a row with ``session_path`` is rejected:
    the row renders as missing (em-dash), and ``denied_result_ids``
    flags it."""
    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    sess_a = sessions_root / "20260101T000000Z_aaa"
    sess_a.mkdir()
    sess_b = sessions_root / "20260101T000001Z_bbb"
    sess_b.mkdir()
    monkeypatch.setattr("nora.ui.SESSIONS_ROOT", sessions_root)
    monkeypatch.delenv("NORA_ALLOW_CROSS_SESSION_RECALL", raising=False)

    reset_store_for_tests()
    try:
        store_a = get_store(sess_a)
        m_a = store_a.insert(
            label="A", analysis_type="linear_regression",
            sanitized_payload={
                "type": "linear_regression", "n": 50,
                "coefficients": {"x": 0.5},
                "standard_errors": {"x": 0.05},
                "response_variable": "y", "predictor_variables": ["x"],
            },
            language="R", script_code="", transformations=[],
        )
        with use_cwd(sess_b):
            res = asyncio.run(HANDLERS["compose_results"]({
                "spec": {
                    "columns": [{"id": "x", "label": "x"}],
                    "groups": [{"rows": [
                        {
                            "result_id": m_a.id,
                            "session_path": str(sess_a),
                            "label": "in A",
                        },
                    ]}],
                },
            }))
        body = _mcp_text(res)
        assert body["status"] == "ok"
        assert m_a.id in body.get("denied_result_ids", [])
        assert "NORA_ALLOW_CROSS_SESSION_RECALL" in body.get("hint", "")
    finally:
        reset_store_for_tests()


def test_compose_results_tool_rejects_malformed_spec(tmp_path: Path) -> None:
    """The tool returns a structured error (not a crash) on a
    malformed spec, with a hint pointing at the required shape."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        with use_cwd(cwd):
            # Missing ``spec`` argument.
            res = asyncio.run(HANDLERS["compose_results"]({}))
            assert _mcp_text(res)["status"] == "error"
            # spec not a dict.
            res = asyncio.run(HANDLERS["compose_results"]({"spec": "no"}))
            assert _mcp_text(res)["status"] == "error"
            # spec missing required keys.
            res = asyncio.run(HANDLERS["compose_results"]({"spec": {}}))
            body = _mcp_text(res)
            assert body["status"] == "error"
            assert "columns" in body["reason"]
            assert "groups" in body["reason"]
    finally:
        reset_store_for_tests()


def test_compose_results_tool_rejects_oversized_spec(tmp_path: Path) -> None:
    """A spec that exceeds any layout cap (columns, groups,
    rows-per-group, total rows, label length) must be rejected
    BEFORE rendering. The whole point of compose_results is
    context-economy — a runaway spec rendered into one tool result
    defeats the feature.
    """
    cwd = tmp_path / "session"
    cwd.mkdir()
    reset_store_for_tests()
    try:
        with use_cwd(cwd):
            # Too many columns (cap is 25).
            big_columns = {
                "columns": [{"id": f"c{i}", "label": f"col {i}"}
                            for i in range(40)],
                "groups": [
                    {"rows": [{"result_id": "X", "label": "row"}]},
                ],
            }
            body = _mcp_text(asyncio.run(
                HANDLERS["compose_results"]({"spec": big_columns})
            ))
            assert body["status"] == "error"
            assert "columns" in body["reason"]

            # Too many groups (cap is 25).
            big_groups = {
                "columns": [{"id": "c0", "label": "c"}],
                "groups": [
                    {"rows": [{"result_id": "X", "label": "row"}]}
                    for _ in range(60)
                ],
            }
            body = _mcp_text(asyncio.run(
                HANDLERS["compose_results"]({"spec": big_groups})
            ))
            assert body["status"] == "error"
            assert "groups" in body["reason"]

            # Total rows over the budget (cap is 250 across all groups).
            wide_rows = {
                "columns": [{"id": "c0", "label": "c"}],
                "groups": [
                    {"rows": [{"result_id": "X", "label": "r"}
                              for _ in range(80)]}
                    for _ in range(20)  # 20 × 80 = 1600 total rows
                ],
            }
            body = _mcp_text(asyncio.run(
                HANDLERS["compose_results"]({"spec": wide_rows})
            ))
            assert body["status"] == "error"
            assert "row" in body["reason"].lower()

            # Pathological row label (cap is 200 chars).
            long_label_spec = {
                "columns": [{"id": "c0", "label": "c"}],
                "groups": [
                    {"rows": [
                        {"result_id": "X", "label": "y" * 1000},
                    ]},
                ],
            }
            body = _mcp_text(asyncio.run(
                HANDLERS["compose_results"]({"spec": long_label_spec})
            ))
            assert body["status"] == "error"
            assert "label" in body["reason"].lower()
    finally:
        reset_store_for_tests()


def test_compose_layout_against_real_24_result_run() -> None:
    """Smoke test against the user's actual reg_v11.do output: 24
    valid sanitized linear_regression payloads. Compose a layout
    grouped by hypothesis panel (H1a / H1b / H2a / H2b), verify the
    markdown comes back with one header row per group, one row per
    outcome, and no scientific-notation leakage in cells. This is
    the regression pin for the design proposal that motivated the
    feature."""
    fixture = Path(
        "/Users/bb/.nora-sessions/20260501T032851Z_be5f2a77/.nora/runs/"
        "20260501T032915Z_69914fb1/result.json"
    )
    if not fixture.exists():
        # Fixture not present in CI; skip silently.
        import pytest
        pytest.skip("real-data fixture not available on this machine")

    from nora.sanitizer import DEFAULT_CONFIG, sanitize

    payloads_by_id: dict[str, dict] = {}
    labels: list[tuple[str, str]] = []
    with fixture.open() as f:
        for i, line in enumerate(f, start=1):
            raw = json.loads(line)
            result = sanitize(raw, DEFAULT_CONFIG)
            assert result.ok, f"line {i}: {result.rejection_reason}"
            rid = f"M{i}"
            payloads_by_id[rid] = result.sanitized or {}
            labels.append((rid, raw.get("label", rid)))
    assert len(payloads_by_id) == 24

    # Group by hypothesis prefix (H1a / H1b / H2a / H2b → 6 outcomes each).
    def _group_for(label: str) -> str:
        for tag in ("H1a", "H1b", "H2a", "H2b"):
            if label.startswith(tag):
                return tag
        return "other"

    groups: dict[str, list] = {"H1a": [], "H1b": [], "H2a": [], "H2b": []}
    for rid, label in labels:
        outcome = label.split(" ", 1)[1] if " " in label else label
        groups[_group_for(label)].append({"result_id": rid, "label": outcome})

    spec = {
        "title": "reg_v11: event-study coefficients across panels",
        "columns": [
            {"id": "fp_ym2", "label": "year -2"},
            {"id": "fp_y0",  "label": "year 0"},
            {"id": "fp_yp1", "label": "year +1"},
            {"id": "fp_yp2", "label": "year +2"},
            {"id": "fp_yp3", "label": "year +3"},
        ],
        "groups": [
            {"label": tag, "rows": rows}
            for tag, rows in groups.items() if rows
        ],
    }
    md = compose_layout(spec, payloads_by_id)
    assert md is not None
    # Title + four group headers + 24 outcome rows + table header line.
    assert "**reg_v11" in md
    for tag in ("**H1a**", "**H1b**", "**H2a**", "**H2b**"):
        assert tag in md, f"missing group header {tag} in:\n{md}"
    # 24 outcome rows: each label appears once.
    for _, label in labels:
        outcome = label.split(" ", 1)[1] if " " in label else label
        assert outcome in md
    # No scientific notation leak in cells.
    assert "e-0" not in md and "e+0" not in md
    # Cell format ``est (SE) [p-value]`` present somewhere.
    import re
    assert re.search(r"-?\d+\.\d+ \(\d+\.\d+\) \[", md), (
        f"cell shape ``est (SE) [p]`` not found in:\n{md[:1500]}"
    )

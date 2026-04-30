"""Canonical markdown-table rendering for sanitized payloads.

A pure-function renderer that turns a sanitized analysis payload
(linear_regression / t_test / descriptive / frequency_table /
crosstab / magnitude_table / correlation_matrix) into a markdown
pipe-table the model can drop directly into a response — or that
the UI can render as a result card without going through the
model at all.

The motivation is consistency. Prompt rules tell the model what
columns to use, but prompts drift on long contexts. The model also
has no reason to format the same regression payload identically on
two recalls. A canonical renderer gives the SAME table for the
SAME payload every time.

Scope notes:

- This module formats. It does NOT validate, sanitize, or interpret.
  Inputs are already-sanitized payloads from the disclosure-control
  layer; outputs are markdown bytes. No SDC decisions live here.
- Single-payload only. Composite tables (24 specs in one matrix,
  before/after-controls forest plots) require choices about row
  order, column emphasis, and which results to highlight — those
  are model judgment, not deterministic formatting.
- Suppressed cells (the ``"<10"`` / similar markers the sanitizer
  inserts) pass through verbatim. Never silently drop them.
- Fields the sanitizer has dropped (e.g., ``vif`` not present)
  simply don't appear in the rendered table; we never invent
  zeros or None placeholders.

Public surface:

- ``render_table(payload)`` — top-level dispatch by ``payload["type"]``.
  Returns ``None`` if the type is unknown or the payload is malformed
  beyond what we can render. Callers fall back to whatever they were
  doing before.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def render_table(payload: dict[str, Any]) -> str | None:
    """Render a sanitized payload as a canonical markdown pipe-table.

    Returns ``None`` on unknown analysis type or unparseable payload —
    callers can fall back to whatever they had (model-formatted prose
    or a generic stringification). Never raises on shape problems.
    """
    if not isinstance(payload, dict):
        return None
    handler = _HANDLERS.get(payload.get("type"))
    if handler is None:
        return None
    try:
        return handler(payload)
    except Exception:  # noqa: BLE001 — formatting must never crash callers
        return None


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------


def _render_linear_regression(p: dict[str, Any]) -> str | None:
    """One row per term. Columns: Term, Estimate, Std. Error, p-value.

    The p-value column is part of Nora's public reporting contract, so
    it stays present even when a malformed/custom payload omitted
    ``p_values``. Missing cells render blank; the UI/model must not
    silently change the table shape.
    """
    coefs = p.get("coefficients") or {}
    if not isinstance(coefs, dict) or not coefs:
        return None
    ses = p.get("standard_errors") or {}
    pvals = p.get("p_values") or {}

    header = ["Term", "Estimate", "Std. Error", "p-value"]
    rows: list[list[str]] = []
    for term, est in coefs.items():
        row = [str(term), _fmt_num(est), _fmt_num(ses.get(term))]
        row.append(_fmt_num(pvals.get(term) if isinstance(pvals, dict) else None))
        rows.append(row)
    table = _markdown_table(header, rows)

    n = p.get("n")
    r2 = p.get("r_squared")
    cap_parts: list[str] = []
    if isinstance(n, int):
        cap_parts.append(f"n = {n:,}")
    if isinstance(r2, (int, float)) and math.isfinite(float(r2)):
        cap_parts.append(f"R² = {_fmt_num(r2)}")
    cond = p.get("condition_number")
    if isinstance(cond, (int, float)) and math.isfinite(float(cond)):
        cap_parts.append(f"κ(X) = {_fmt_num(cond)}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_t_test(p: dict[str, Any]) -> str | None:
    """Per-group rows + a difference row.

    Two-sample / Welch: rows for group 1 and group 2, then a Mean
    diff row. One-sample: a single row with the sample, then a Mean
    diff row (which is mean - hypothesised_mean if available, or
    just the t/p line). Paired: a single pairs row, then the Mean
    diff row.
    """
    test_type = str(p.get("test_type") or "").lower()
    n1 = p.get("n1")
    n2 = p.get("n2")
    m1 = p.get("mean1")
    m2 = p.get("mean2")
    sd1 = p.get("sd1")
    sd2 = p.get("sd2")
    md = p.get("mean_difference")
    tstat = p.get("t_statistic")
    pval = p.get("p_value")

    header = ["Group", "n", "Mean", "SD"]
    rows: list[list[str]] = []
    if test_type in ("two_sample", "welch"):
        rows.append(["1", _fmt_int(n1), _fmt_num(m1), _fmt_num(sd1)])
        rows.append(["2", _fmt_int(n2), _fmt_num(m2), _fmt_num(sd2)])
    elif test_type == "paired":
        rows.append(["pairs", _fmt_int(n1), _fmt_num(m1), _fmt_num(sd1)])
    else:
        # one_sample (default).
        rows.append(["sample", _fmt_int(n1), _fmt_num(m1), _fmt_num(sd1)])
    table = _markdown_table(header, rows)

    diff_lines = []
    if md is not None:
        diff_lines.append(f"Mean diff: {_fmt_num(md)}")
    if tstat is not None:
        diff_lines.append(f"t = {_fmt_num(tstat)}")
    if pval is not None:
        diff_lines.append(f"p = {_fmt_num(pval)}")
    diff = " · ".join(diff_lines)
    label = test_type.replace("_", "-") or "t-test"
    return f"{table}\n\n{label}: {diff}" if diff else f"{table}\n\n{label}"


def _render_descriptive(p: dict[str, Any]) -> str | None:
    """One row. Columns: Variable, n, Mean, SD, Missing (and Min/Max
    when the researcher's policy opted them in)."""
    has_min = "min_value" in p and p["min_value"] is not None
    has_max = "max_value" in p and p["max_value"] is not None
    header = ["Variable", "n", "Mean", "SD"]
    if has_min:
        header.append("Min")
    if has_max:
        header.append("Max")
    header.append("Missing")
    row = [
        str(p.get("variable") or ""),
        _fmt_int(p.get("n")),
        _fmt_num(p.get("mean")),
        _fmt_num(p.get("sd")),
    ]
    if has_min:
        row.append(_fmt_num(p["min_value"]))
    if has_max:
        row.append(_fmt_num(p["max_value"]))
    row.append(_fmt_int(p.get("missing_count", 0)))
    return _markdown_table(header, [row])


def _render_frequency_table(p: dict[str, Any]) -> str | None:
    """One row per level. Columns: Level, Count, Proportion.
    Suppressed cells (``"<10"`` / similar) pass through verbatim."""
    counts = p.get("counts") or {}
    if not isinstance(counts, dict) or not counts:
        return None
    n = p.get("n")
    total_int = n if isinstance(n, int) and n > 0 else None
    rows: list[list[str]] = []
    for level, count in counts.items():
        c_str = str(count) if isinstance(count, str) else _fmt_int(count)
        if isinstance(count, int) and total_int:
            prop = f"{count / total_int:.3f}"
        else:
            prop = ""
        rows.append([str(level), c_str, prop])
    table = _markdown_table(["Level", "Count", "Proportion"], rows)
    var = p.get("variable")
    miss = p.get("missing_count")
    cap_parts: list[str] = []
    if isinstance(var, str):
        cap_parts.append(f"variable: {var}")
    if isinstance(n, int):
        cap_parts.append(f"n = {n:,}")
    if isinstance(miss, int) and miss > 0:
        cap_parts.append(f"missing = {miss:,}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_crosstab(p: dict[str, Any]) -> str | None:
    """2D table: row-variable values down the left, column-variable
    values across the top, counts in cells. Suppression markers pass
    through. Margins (row totals, col totals) deliberately NOT
    rendered — the sanitizer doesn't expose them, and the renderer
    must not synthesize them.
    """
    counts = p.get("counts") or {}
    if not isinstance(counts, dict) or not counts:
        return None
    # Collect all column keys in first-seen order.
    col_keys: list[str] = []
    seen_cols: set[str] = set()
    for row in counts.values():
        if not isinstance(row, dict):
            continue
        for col in row.keys():
            if col not in seen_cols:
                seen_cols.add(col)
                col_keys.append(col)

    row_var = p.get("row_variable") or "row"
    header = [str(row_var)] + [str(c) for c in col_keys]
    rows: list[list[str]] = []
    for row_key, row_value in counts.items():
        if not isinstance(row_value, dict):
            continue
        line = [str(row_key)]
        for col in col_keys:
            v = row_value.get(col)
            if v is None:
                line.append("")
            elif isinstance(v, str):
                line.append(v)  # suppression marker
            else:
                line.append(_fmt_int(v))
        rows.append(line)
    table = _markdown_table(header, rows)
    col_var = p.get("col_variable")
    miss = p.get("missing_count")
    cap_parts: list[str] = []
    if isinstance(col_var, str):
        cap_parts.append(f"columns: {col_var}")
    if isinstance(miss, int) and miss > 0:
        cap_parts.append(f"missing = {miss:,}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_magnitude_table(p: dict[str, Any]) -> str | None:
    """One row per group. Columns: Cell, n, <aggregation>.
    ``aggregation`` is 'sum' or 'mean'; the column header reflects it.
    Suppressed cells (``value`` is a string marker) render verbatim.
    """
    cells = p.get("cells") or {}
    if not isinstance(cells, dict) or not cells:
        return None
    agg = str(p.get("aggregation") or "value").lower()
    header_agg = agg.capitalize() if agg in ("sum", "mean") else "Value"
    row_var = p.get("row_variable") or "Cell"
    header = [str(row_var), "n", header_agg]
    rows: list[list[str]] = []
    for cell_key, cell in cells.items():
        if not isinstance(cell, dict):
            continue
        n = cell.get("n")
        value = cell.get("value")
        v_str = (
            value if isinstance(value, str) else _fmt_num(value)
        )
        rows.append([str(cell_key), _fmt_int(n), v_str])
    table = _markdown_table(header, rows)
    value_var = p.get("value_variable")
    cap = (
        f"{agg} of {value_var} by {row_var}"
        if isinstance(value_var, str) else None
    )
    return f"{table}\n\n{cap}" if cap else table


def _render_correlation_matrix(p: dict[str, Any]) -> str | None:
    """Variable × Variable matrix. Rows and columns are the same set
    of variables (the canonical pairwise layout). Diagonals are 1
    by definition; render them so the table reads as expected."""
    correlations = p.get("correlations") or {}
    if not isinstance(correlations, dict) or not correlations:
        return None
    variables = p.get("variables") or list(correlations.keys())
    if not variables:
        return None
    header = [""] + [str(v) for v in variables]
    rows: list[list[str]] = []
    for v in variables:
        row_value = correlations.get(v) or {}
        line = [str(v)]
        for w in variables:
            if v == w:
                line.append("1.000")
                continue
            val = row_value.get(w) if isinstance(row_value, dict) else None
            line.append(_fmt_num(val) if val is not None else "")
        rows.append(line)
    table = _markdown_table(header, rows)
    method = p.get("method")
    n = p.get("n")
    cap_parts: list[str] = []
    if isinstance(method, str):
        cap_parts.append(f"method: {method}")
    if isinstance(n, int):
        cap_parts.append(f"n = {n:,}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


_HANDLERS: dict[str, Any] = {
    "linear_regression": _render_linear_regression,
    "t_test": _render_t_test,
    "descriptive": _render_descriptive,
    "frequency_table": _render_frequency_table,
    "crosstab": _render_crosstab,
    "magnitude_table": _render_magnitude_table,
    "correlation_matrix": _render_correlation_matrix,
}


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored pipe table.

    Pads cells to the column max so the source markdown stays
    readable when copy-pasted. Renderers that don't care about
    raw-source alignment (most chat clients) ignore the padding.
    """
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        padded = [c.ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [_row(header), sep]
    for row in rows:
        out.append(_row(row))
    return "\n".join(out)


def _fmt_num(x: Any) -> str:
    """Render a number for table display.

    Suppression markers (string values like ``"<10"``) pass through.
    Non-finite values render empty so a NaN doesn't read as a
    suspicious zero. Otherwise: 4 sig figs at the magnitude boundary,
    scientific notation only for very small / very large absolute
    values.
    """
    if isinstance(x, str):
        return x  # suppression marker or pre-formatted
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    abs_v = abs(v)
    if abs_v == 0:
        return "0"
    if abs_v >= 1e6 or abs_v < 1e-3:
        return f"{v:.3e}"
    # 4 sig figs, trimmed of trailing zeros where natural.
    s = f"{v:.4g}"
    return s


def _fmt_int(x: Any) -> str:
    """Render an integer-typed field. Strings (suppression markers)
    pass through; non-numeric or None becomes empty."""
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return ""

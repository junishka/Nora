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
- Suppressed cells (the ``"<10"`` / similar markers the sanitizer
  inserts) pass through verbatim. Never silently drop them.
- Fields the sanitizer has dropped (e.g., ``vif`` not present)
  simply don't appear in the rendered table; we never invent
  zeros or None placeholders.

Public surface:

- ``render_table(payload)`` — single-payload dispatch by
  ``payload["type"]``. One result per call.
- ``compose_layout(spec, payloads_by_id)`` — multi-result composite
  table from a model-emitted layout spec. Cell values are looked up
  from the payload store by the spec's result IDs; the model never
  types a coefficient. Hallucinated IDs render as ``—`` so grouping
  errors are recoverable but number errors are structurally
  impossible.

Both return ``None`` if the input is malformed beyond what we can
render; callers fall back to whatever they had.
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
# Layout-driven composite table
# ---------------------------------------------------------------------------


def compose_layout(
    spec: dict[str, Any],
    payloads_by_id: dict[str, dict[str, Any]],
) -> str | None:
    """Compose a multi-result comparison table from a model-emitted
    layout spec.

    The spec carries the model's judgment about which results to
    surface together, how to group them, and which terms to put in
    columns. Cell values are looked up in ``payloads_by_id`` — the
    model never types a coefficient. A ``result_id`` not in the
    store, or a ``term_id`` not in a payload's coefficients, renders
    as ``—``. This separation matches where each kind of error is
    recoverable: grouping is fallible (the user can re-prompt or
    edit), the numbers aren't (they come from the sanitized store).

    Spec shape::

        {
            "title": "Mechanism A: revenue effects",   # optional
            "columns": [
                {"id": "fp_y0",  "label": "year 0"},
                {"id": "fp_yp1", "label": "year +1"},
                ...
            ],
            "groups": [
                {
                    "label": "H1: direct effect",      # optional row header
                    "rows": [
                        {"result_id": "M1", "label": "ln_rev_total"},
                        ...
                    ]
                },
                ...
            ]
        }

    Cells render as ``estimate (SE) [p-value]``. Each piece falls
    back to ``—`` independently when the underlying payload doesn't
    carry it (a Stata robust-SE path with no t-stats produces
    ``coef (SE) [—]``; a fully-omitted collinear term produces
    ``—``). Group labels render as bold header rows above their
    members; missing labels just skip the header row.

    Returns ``None`` when the spec is malformed (not a dict, missing
    or empty ``columns`` / ``groups``, wrong inner shapes); callers
    fall back to their default error handling. Never raises.
    """
    try:
        return _compose_layout_inner(spec, payloads_by_id)
    except Exception:  # noqa: BLE001 — formatting must never crash callers
        return None


def _compose_layout_inner(
    spec: dict[str, Any],
    payloads_by_id: dict[str, dict[str, Any]],
) -> str | None:
    if not isinstance(spec, dict):
        return None
    columns = spec.get("columns")
    groups = spec.get("groups")
    if not isinstance(columns, list) or not columns:
        return None
    if not isinstance(groups, list) or not groups:
        return None

    col_ids: list[str] = []
    col_labels: list[str] = []
    for c in columns:
        if not isinstance(c, dict):
            return None
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            return None
        col_ids.append(cid)
        clabel = c.get("label", cid)
        col_labels.append(str(clabel) if clabel is not None else cid)

    header = ["Outcome", *col_labels]
    rows: list[list[str]] = []

    for group in groups:
        if not isinstance(group, dict):
            return None
        group_rows = group.get("rows")
        if not isinstance(group_rows, list) or not group_rows:
            return None
        group_label = group.get("label")
        if isinstance(group_label, str) and group_label.strip():
            # Header row: bold label in first cell, blanks elsewhere.
            # Markdown pipe tables don't support row spans, so a
            # blank-cells header row is the conventional shape.
            rows.append([f"**{group_label.strip()}**", *([""] * len(col_ids))])
        for row in group_rows:
            if not isinstance(row, dict):
                return None
            rid = row.get("result_id")
            if not isinstance(rid, str) or not rid:
                return None
            rlabel = row.get("label", rid)
            payload = payloads_by_id.get(rid) or {}
            coefs = payload.get("coefficients") if isinstance(payload, dict) else None
            ses = payload.get("standard_errors") if isinstance(payload, dict) else None
            pvals = payload.get("p_values") if isinstance(payload, dict) else None
            cells = [
                _compose_cell(coefs, ses, pvals, col_id) for col_id in col_ids
            ]
            rows.append([str(rlabel) if rlabel is not None else rid, *cells])

    table = _markdown_table(header, rows)
    title = spec.get("title")
    if isinstance(title, str) and title.strip():
        return f"**{title.strip()}**\n\n{table}"
    return table


def _compose_cell(
    coefs: Any, ses: Any, pvals: Any, term_id: str,
) -> str:
    """Render one ``estimate (SE) [p-value]`` cell.

    Each component falls back to ``—`` independently. If all three
    are absent (the result_id missed entirely, or the term isn't in
    any of coefficients / SEs / p_values), collapse to a single
    ``—`` rather than ``— (—) [—]`` which is just noise.
    """
    e = coefs.get(term_id) if isinstance(coefs, dict) else None
    s = ses.get(term_id) if isinstance(ses, dict) else None
    p = pvals.get(term_id) if isinstance(pvals, dict) else None
    if e is None and s is None and p is None:
        return "—"
    e_str = (_fmt_num(e) if e is not None else "") or "—"
    s_str = (_fmt_num(s) if s is not None else "") or "—"
    p_str = (_fmt_pvalue(p) if p is not None else "") or "—"
    return f"{e_str} ({s_str}) [{p_str}]"


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------


def _render_linear_regression(p: dict[str, Any]) -> str | None:
    """One row per term. Columns: Term, Estimate, Std. Error, p-value.

    Optional p-values are omitted when the sanitizer didn't pass
    ``p_values`` through (e.g., scripts that emitted coefficients +
    SEs without t-stats). The intercept is rendered as ``(Intercept)``
    if it appears in coefficients.
    """
    coefs = p.get("coefficients") or {}
    if not isinstance(coefs, dict) or not coefs:
        return None
    ses = p.get("standard_errors") or {}
    pvals = p.get("p_values") or {}

    has_p = bool(isinstance(pvals, dict) and pvals)
    header = ["Term", "Estimate", "Std. Error"]
    if has_p:
        header.append("p-value")
    rows: list[list[str]] = []
    for term, est in coefs.items():
        row = [str(term), _fmt_num(est), _fmt_num(ses.get(term))]
        if has_p:
            row.append(_fmt_pvalue(pvals.get(term)))
        rows.append(row)
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []

    # Sample-size leg. For Cox PH / discrete-time survival, ``n`` is
    # records (post-stset, can include split episodes per subject) and
    # the researcher reads off "S subjects, E events" — so prefer the
    # subjects/events pair when present, with records in parens. For
    # everything else (OLS, logit, probit, Poisson, ...) show ``n``.
    n = p.get("n")
    n_subj = p.get("n_subjects")
    n_fail = p.get("n_failures")
    if isinstance(n_subj, int):
        leg = f"subjects = {n_subj:,}"
        if isinstance(n_fail, int):
            leg += f" · events = {n_fail:,}"
        if isinstance(n, int) and n != n_subj:
            leg += f" (records = {n:,})"
        cap_parts.append(leg)
    elif isinstance(n, int):
        cap_parts.append(f"n = {n:,}")

    # Fit metric. The sanitiser admits both R² (OLS) and pseudo R²
    # (McFadden, for logit/probit/Poisson); a typical script sets one
    # or the other. Showing both verbatim when both are present keeps
    # the renderer dumb — we don't infer the model family from
    # ``type`` alone, since the sanitiser canonicalises everything as
    # ``linear_regression``.
    r2 = p.get("r_squared")
    if isinstance(r2, (int, float)) and math.isfinite(float(r2)):
        cap_parts.append(f"R² = {_fmt_num(r2)}")
    pseudo = p.get("pseudo_r_squared")
    if isinstance(pseudo, (int, float)) and math.isfinite(float(pseudo)):
        cap_parts.append(f"pseudo R² = {_fmt_num(pseudo)}")

    # Survival-specific discrimination metric.
    concordance = p.get("concordance")
    if isinstance(concordance, (int, float)) and math.isfinite(
        float(concordance),
    ):
        cap_parts.append(f"C = {_fmt_num(concordance)}")

    # Likelihood-based diagnostics. Either the omnibus χ² (with
    # p-value when present), or the log-likelihood. AIC/BIC are
    # rendered when the model picker / cross-spec comparison cards
    # need them — they aren't useful in the per-result caption alone
    # and would crowd it.
    chi2 = p.get("chi_squared")
    chi2_p = p.get("chi_squared_p_value")
    if isinstance(chi2, (int, float)) and math.isfinite(float(chi2)):
        leg = f"χ² = {_fmt_num(chi2)}"
        if isinstance(chi2_p, (int, float)) and math.isfinite(float(chi2_p)):
            leg += f" (p = {_fmt_pvalue(chi2_p)})"
        cap_parts.append(leg)
    loglik = p.get("log_likelihood")
    if isinstance(loglik, (int, float)) and math.isfinite(float(loglik)):
        cap_parts.append(f"log-lik = {_fmt_num(loglik)}")

    # OLS-specific design diagnostic. Tail position keeps the OLS
    # caption identical to its previous form (n · R² · κ), modulo the
    # new optional middle slots.
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
        diff_lines.append(f"p = {_fmt_pvalue(pval)}")
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


def _escape_table_cell(s: str) -> str:
    """Escape characters that would break a GitHub-flavored pipe table.

    The renderer writes data-origin strings (variable names,
    coefficient keys, category labels) into cells verbatim, and
    ``safe_text`` / ``safe_key`` only neutralise control chars,
    bidi tricks, and over-length — they leave ``|``, ``\\``, and
    backticks alone because those are valid characters in research
    identifiers. So a category labelled ``A | B`` (a legitimate
    "A or B" ordinal) would otherwise emit an extra column and
    derail every following row.

    Escapes:
      - ``\\`` → ``\\\\`` first (must precede pipe escape so the
        backslash we add for ``|`` isn't itself escaped twice).
      - ``|``  → ``\\|``  (the column delimiter).
      - any residual ``\\n`` / ``\\r`` → space (defensive — text
        sanitisers should have flattened these already, but fail
        closed if a caller fed unsanitised text in).
    """
    return (
        s.replace("\\", "\\\\")
         .replace("|", "\\|")
         .replace("\n", " ")
         .replace("\r", " ")
    )


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored pipe table.

    Pads cells to the column max so the source markdown stays
    readable when copy-pasted. Renderers that don't care about
    raw-source alignment (most chat clients) ignore the padding.

    Cells are escaped via ``_escape_table_cell`` so a data-origin
    label containing ``|`` doesn't break the table structure.
    """
    esc_header = [_escape_table_cell(h) for h in header]
    esc_rows = [[_escape_table_cell(c) for c in row] for row in rows]

    widths = [len(h) for h in esc_header]
    for row in esc_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        padded = [c.ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [_row(esc_header), sep]
    for row in esc_rows:
        out.append(_row(row))
    return "\n".join(out)


def _fmt_num(x: Any) -> str:
    """Render a number for table display.

    4 significant figures, fixed-point notation only — never
    scientific. Coefficients in a regression often span many
    magnitudes within one table (continuous slope ≈ 0.0004,
    dummy ≈ 0.5, intercept / year FE ≈ 13.6); a mid-row hop into
    ``e-04`` reads worse than a slightly wider column. Trailing
    zeros after the decimal are trimmed so columns stay tight,
    but precision is preserved (``0.0021`` keeps two sig figs;
    ``0.002100`` would falsely advertise four).

    Suppression markers (string values like ``"<10"``) pass
    through. Non-finite or missing values render empty so a NaN
    doesn't read as a suspicious zero.
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
    # Decimal places needed to show 4 sig figs in fixed notation.
    # ``place`` is the index of the leading sig fig (e.g. 0.0021
    # has ``place = -3``; 13.58 has ``place = 1``). ``decimals``
    # can go negative for very large magnitudes — Python's
    # ``round(v, ndigits)`` accepts negative ndigits to round
    # left of the decimal point, which is exactly what we want
    # for e.g. 1234567 → 1235000.
    place = math.floor(math.log10(abs_v))
    decimals = 4 - 1 - place
    if decimals >= 0:
        s = f"{v:.{decimals}f}"
    else:
        s = f"{round(v, decimals):.0f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
        if s in ("", "-"):
            s = "0"
    return s


def _fmt_pvalue(x: Any) -> str:
    """Render a p-value for table display in publication style.

    Three decimals; ``<0.001`` floor (the standard convention — a
    bare ``0.000`` reads as exactly zero, which it isn't); ``>0.999``
    ceiling for symmetric honesty on near-1 values. Suppression
    markers and non-finite/missing values follow ``_fmt_num``'s
    conventions: pass through / render empty.
    """
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v) or v < 0 or v > 1:
        return ""
    if v < 0.001:
        return "<0.001"
    if v > 0.999:
        return ">0.999"
    return f"{v:.3f}"


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

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
import re
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
    labels_by_id: dict[str, str] | None = None,
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

    Row shapes (both are accepted in the same ``rows`` list):

      * ``"M37"`` — bare result_id. Row label is auto-resolved from
        ``labels_by_id`` (the store's helper-call label) when the
        caller passed that dict, otherwise falls back to the rid.
        This is the minimal form for the common case ("decide the
        groups, hand me result_ids, let the store provide labels").
      * ``{"result_id": "M37", "label": "ln_revenue"}`` — explicit
        label override. Use when the stored label is too verbose or
        the row needs renaming for the comparison context.

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
                    "rows": ["M1", "M2", ...]          # bare ids OR row dicts
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

    ``labels_by_id`` is the store-resolved label map (rid → helper
    call label). Wired up by ``compose_results`` in tools.py from
    each ``store.get(rid).label``. Passing ``None`` keeps the
    legacy "label defaults to rid" behaviour — used by tests that
    drive the renderer directly without a store.

    Returns ``None`` when the spec is malformed (not a dict, missing
    or empty ``columns`` / ``groups``, wrong inner shapes); callers
    fall back to their default error handling. Never raises.
    """
    try:
        return _compose_layout_inner(spec, payloads_by_id, labels_by_id or {})
    except Exception:  # noqa: BLE001 — formatting must never crash callers
        return None


# Recognised group-tag separator for the auto-consolidation pass. The
# canonical shape the model emits when it bakes a hypothesis tag into
# each helper call's ``label(...)`` arg is ``"<TAG> :: <variable>"``
# (double colon with surrounding whitespace). Matching only that
# separator avoids false positives on legitimate prose labels that
# happen to contain a single colon (e.g. ``"H1: direct effect"`` is
# itself a valid group.label and must not be split).
_GROUP_PREFIX_RE = re.compile(
    r"^(?P<tag>\S[^:]*?)\s*::\s*(?P<rest>.+?)\s*$"
)


def _split_group_prefix(label: str) -> tuple[str, str] | None:
    """Split ``"H1 :: outcome_a"`` into
    ``("H1", "outcome_a")``. Returns ``None`` when the
    ``::`` separator isn't present or the tag/rest would be empty.
    """
    m = _GROUP_PREFIX_RE.match(label)
    if not m:
        return None
    tag = m.group("tag").strip()
    rest = m.group("rest").strip()
    if not tag or not rest:
        return None
    return tag, rest


def _consolidate_group_prefix(
    group_label: str | None,
    row_labels: list[str],
) -> tuple[str | None, list[str]]:
    """Hoist or strip a ``<TAG> :: `` prefix shared by every row label.

    Three outcomes, all conservative:

      * Every row label carries the same ``<TAG> :: `` prefix and
        ``group_label`` is unset (or whitespace) → hoist ``TAG`` to
        ``group_label`` and strip the prefix from each row label.
        This is the common case where the script baked the
        hypothesis tag into each helper call's ``label("H1 ::
        outcome_a")`` arg and the compose spec passed bare
        result_ids without a group.label.
      * Every row label carries the same ``<TAG> :: `` prefix AND
        the existing ``group_label`` (stripped) equals ``TAG`` →
        just strip the prefix. The model set group.label correctly
        but ALSO baked the prefix into row labels; drop the
        duplication.
      * Anything else (partial prefix, mixed prefixes, group.label
        already set to something different, no prefix at all) →
        leave both inputs unchanged. The model's explicit
        choice wins over our heuristic.

    Returns ``(group_label, row_labels)`` (possibly rewritten).
    """
    if not row_labels:
        return group_label, row_labels
    splits = [_split_group_prefix(lbl) for lbl in row_labels]
    # All-or-nothing: a single row without the prefix means the group
    # isn't uniformly tagged and we leave it alone. Partial-prefix
    # rewriting would lose information.
    if any(s is None for s in splits):
        return group_label, row_labels
    tags = {s[0] for s in splits if s is not None}
    if len(tags) != 1:
        return group_label, row_labels
    common_tag = next(iter(tags))
    stripped_group = group_label.strip() if isinstance(group_label, str) else ""
    if stripped_group and stripped_group != common_tag:
        # Model set group.label to something specific that doesn't
        # match the common prefix. Honor the explicit choice.
        return group_label, row_labels
    new_group_label = common_tag if not stripped_group else group_label
    new_row_labels = [s[1] for s in splits if s is not None]
    return new_group_label, new_row_labels


def _compose_layout_inner(
    spec: dict[str, Any],
    payloads_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, str],
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

    # ``any_unresolved`` must persist across ALL groups: the legend
    # describes what the failure glyphs (``—`` / ``·`` / ``n/a``)
    # mean, and once any group emits one, the legend is needed
    # regardless of whether later groups happen to render cleanly.
    # Resetting this inside the per-group loop (the prior shape)
    # silently dropped the legend whenever the FINAL group resolved
    # cleanly even though earlier groups had failures — leaving the
    # rendered table with unexplained glyphs.
    any_unresolved = False

    for group in groups:
        if not isinstance(group, dict):
            return None
        group_rows = group.get("rows")
        if not isinstance(group_rows, list) or not group_rows:
            return None
        group_label_raw = group.get("label")
        group_label: str | None = (
            group_label_raw if isinstance(group_label_raw, str) else None
        )

        # First pass: validate row shapes and resolve each to
        # (rid, rlabel, payload). Two row shapes accepted: a bare
        # result_id string (auto-label from ``labels_by_id``, fallback
        # to rid) and the explicit ``{"result_id": ..., "label": ...}``
        # dict. The bare-string form is the minimal-friction path that
        # matters for big multi-result batches — the model picks the
        # groups, hands us result_ids, the store provides labels — so
        # it doesn't have to type both the rid and a re-derived label
        # for each row.
        resolved: list[tuple[str, str, dict[str, Any] | None]] = []
        for row in group_rows:
            if isinstance(row, str):
                rid = row
                if not rid:
                    return None
                rlabel_override: Any = None
            elif isinstance(row, dict):
                rid = row.get("result_id")
                if not isinstance(rid, str) or not rid:
                    return None
                rlabel_override = row.get("label")
            else:
                return None
            # Label precedence: explicit row override > stored helper
            # label (``labels_by_id``) > raw result_id. The store
            # fallback turns "M37" into "ln_revenue" (the
            # ``nora.result(label=...)`` the script authored) without
            # the model having to re-type it.
            if rlabel_override is not None:
                rlabel = rlabel_override
            else:
                rlabel = labels_by_id.get(rid, rid)
            rlabel_str = str(rlabel) if rlabel is not None else rid
            resolved.append((rid, rlabel_str, payloads_by_id.get(rid)))

        # Consolidate a common "<TAG> :: " prefix shared by every row
        # label in this group. Addresses the common shape where the
        # script bakes the hypothesis tag into each helper call's
        # ``label("H1 :: outcome_a")`` arg and the compose
        # call passes bare result_ids without a group.label. Without
        # this pass the rendered table reads as a flat ungrouped list
        # with the hypothesis tag pasted into every row's first cell,
        # defeating the bold-header row that group.label is meant to
        # produce. See module docstring for the broader contract.
        consolidated_group_label, consolidated_row_labels = (
            _consolidate_group_prefix(group_label, [r[1] for r in resolved])
        )

        if consolidated_group_label and consolidated_group_label.strip():
            # Header row: bold label in first cell, blanks elsewhere.
            # Markdown pipe tables don't support row spans, so a
            # blank-cells header row is the conventional shape.
            rows.append([
                f"**{consolidated_group_label.strip()}**",
                *([""] * len(col_ids)),
            ])

        for (rid, _orig_label, payload), new_label in zip(
            resolved, consolidated_row_labels
        ):
            # Pass the raw lookup result (None when the result_id
            # isn't in the store) so ``_compose_cell`` can distinguish
            # "result missing" from "result present but term missing"
            # — three failure modes get three distinct glyphs.
            cells = [_compose_cell(payload, col_id) for col_id in col_ids]
            rows.append([new_label, *cells])
            if payload is None or _has_unresolved_term(payload, col_ids):
                any_unresolved = True

    table = _markdown_table(header, rows)
    title = spec.get("title")
    parts: list[str] = []
    if isinstance(title, str) and title.strip():
        parts.append(f"**{title.strip()}**")
    parts.append(table)
    # Legend: only emit when at least one cell rendered as something
    # other than the data-bearing form. A clean composite has no need
    # for the legend; a researcher who hits one of the three failure
    # glyphs needs to know which means what. Keeping the legend
    # conditional avoids polluting tidy outputs.
    if any_unresolved:
        parts.append(
            "Legend: ``—`` result not found · "
            "``·`` term in model but no estimate (often perfect "
            "collinearity) · ``n/a`` term not part of this model."
        )
    return "\n\n".join(parts)


# Intercept aliases each runtime emits. R's ``lm()`` reports
# "(Intercept)"; statsmodels formula fits report "Intercept";
# statsmodels ``add_constant(X)`` reports "const"; Stata reports
# "_cons". The lowercase "intercept" form is a permissive fallback in
# case a future runtime normalizes naming. Mirrors the set in
# ``nora.sanitizer`` — kept duplicated here rather than imported
# because result_render is on the cold render path and depending on
# the sanitizer module from it would invert the natural import order.
_INTERCEPT_ALIASES = frozenset({
    "(Intercept)", "_cons", "intercept", "Intercept", "const",
})


def _has_unresolved_term(
    payload: dict[str, Any], term_ids: list[str],
) -> bool:
    """Whether any ``term_id`` in ``term_ids`` will render as a
    failure glyph (``·`` or ``n/a``) rather than a data cell. Used to
    decide whether to emit the legend below the composite table.
    """
    coefs = payload.get("coefficients") if isinstance(payload, dict) else None
    if not isinstance(coefs, dict):
        return True
    for tid in term_ids:
        if coefs.get(tid) is None:
            return True
    return False


def _compose_cell(
    payload: dict[str, Any] | None, term_id: str,
) -> str:
    """Render one ``estimate (SE) [p-value]`` cell.

    Three distinct failure glyphs let a researcher tell why a cell is
    empty — the previous single ``—`` collapsed three meaningfully
    different cases into one shape:

    - ``—`` (em-dash): the result_id wasn't in the store at all. The
      model invented or mistyped the id, or the result was deleted.
    - ``·`` (middle dot): the term IS declared as a predictor for
      this model (it's in ``predictor_variables``) but no coefficient
      came back. In OLS this almost always means perfect collinearity
      with another term — the estimator silently dropped it.
    - ``n/a``: the term isn't part of this model at all (it lives in
      a different column's model, or the model type structurally
      doesn't produce that quantity).

    The trichotomy only kicks in when the payload carries
    ``predictor_variables`` — without it we can't tell case 2 from
    case 3, so we fall back to ``—`` for the missing-term case
    (matches the old behavior).
    """
    if payload is None:
        return "—"
    if not isinstance(payload, dict):
        return "—"
    coefs = payload.get("coefficients")
    ses = payload.get("standard_errors")
    pvals = payload.get("p_values")
    declared = payload.get("predictor_variables")

    e = coefs.get(term_id) if isinstance(coefs, dict) else None
    s = ses.get(term_id) if isinstance(ses, dict) else None
    p = pvals.get(term_id) if isinstance(pvals, dict) else None

    if e is None and s is None and p is None:
        # No data on this term. Distinguish "in this model but dropped"
        # from "not in this model" using ``predictor_variables``. If
        # the payload didn't carry that list we can't tell, so we keep
        # the conservative ``—``.
        if isinstance(declared, list):
            in_model = (term_id in declared) or (term_id in _INTERCEPT_ALIASES)
            return "·" if in_model else "n/a"
        return "—"

    # Coefficient is present but a within-cell component (SE or
    # p-value) may be missing. Earlier code rendered every missing
    # component as ``·`` (the same glyph the legend defines as
    # "term in model but no estimate (often collinearity)"), which
    # made cells like ``2 (0.2) [·]`` ambiguous: the ``·`` looked
    # like a per-term collinearity marker even when the term IS
    # estimated and the sanitizer simply dropped p-values from the
    # payload (e.g., scripts that emit coef + SE without t-stats).
    # Drop the missing component's punctuation entirely instead —
    # the format gracefully degrades from ``est (SE) [p]`` through
    # ``est (SE)`` / ``est [p]`` to bare ``est``, with no glyph
    # collisions. ``_has_unresolved_term`` already fires only when
    # the coefficient itself is missing, which is consistent with
    # this rendering.
    e_str = _fmt_num(e) if e is not None else None
    s_str = _fmt_num(s) if s is not None else None
    p_str = _fmt_pvalue(p) if p is not None else None
    parts: list[str] = []
    # If the coefficient itself didn't format (rare — ``_fmt_num`` of
    # a non-numeric), fall back to the in-model glyph so the cell
    # isn't empty.
    parts.append(e_str if e_str else "·")
    if s_str:
        parts.append(f"({s_str})")
    if p_str:
        parts.append(f"[{p_str}]")
    return " ".join(parts)


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
    # ``n_subjects`` / ``n_failures`` may arrive as a suppression-marker
    # string ("<10") when small-cell coarsening fired in the sanitizer.
    # Render those verbatim so the researcher sees the suppression
    # rather than silently falling back to the records-only leg.
    if isinstance(n_subj, (int, str)) and n_subj != "":
        subj_str = f"{n_subj:,}" if isinstance(n_subj, int) else n_subj
        leg = f"subjects = {subj_str}"
        if isinstance(n_fail, (int, str)) and n_fail != "":
            fail_str = f"{n_fail:,}" if isinstance(n_fail, int) else n_fail
            leg += f" · events = {fail_str}"
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
    # ``p.get("missing_count")`` (no default) — when the sanitizer
    # has DROPPED the field entirely (cross-query back-calc case in
    # ``_render_crosstab`` analogues, single-suppressed-cell paths,
    # etc.), defaulting to 0 would render "missing = 0" and claim no
    # missingness — actively misleading. ``_fmt_int(None)`` returns
    # "" so the column renders blank instead.
    row.append(_fmt_int(p.get("missing_count")))
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


def _render_did_event_study(p: dict[str, Any]) -> str | None:
    """ATT(g, t) panel with one row per surviving cohort.

    Renders the cohort × event-time matrix as a wide markdown table:
    one column for the cohort label, one column for each event time,
    cells carry ATT estimates. SE / p-values are summarised in the
    caption rather than crowded into the matrix. The aggregate ATT
    appears in the caption when present.

    Pre-fix, payloads of this shape rendered as raw JSON in the
    tool-result card; the model could read them but couldn't drop
    a clean table directly into its reply.
    """
    att = p.get("att") or {}
    if not isinstance(att, dict) or not att:
        return None
    event_times = p.get("event_times") or []
    if not isinstance(event_times, list):
        return None
    # Normalize event-time keys to their str form for matrix lookup
    # (the sanitizer stores them as ints when integer-valued).
    et_keys = [str(t) for t in event_times]
    header = ["Cohort"] + et_keys
    rows: list[list[str]] = []
    for cohort, cells in att.items():
        if not isinstance(cells, dict):
            continue
        row = [str(cohort)]
        for et in et_keys:
            row.append(_fmt_num(cells.get(et)))
        rows.append(row)
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []
    n_treated = p.get("n_treated_per_group") or {}
    if isinstance(n_treated, dict) and n_treated:
        total = sum(v for v in n_treated.values() if isinstance(v, int))
        if total > 0:
            cap_parts.append(f"treated N = {total:,} across {len(n_treated)} cohorts")
    est = p.get("estimator")
    if isinstance(est, str):
        cap_parts.append(est.replace("_", "-"))
    agg_att = p.get("aggregate_att")
    if isinstance(agg_att, (int, float)) and math.isfinite(float(agg_att)):
        leg = f"aggregate ATT = {_fmt_num(agg_att)}"
        agg_p = p.get("aggregate_p_value")
        if isinstance(agg_p, (int, float)) and math.isfinite(float(agg_p)):
            leg += f" (p = {_fmt_pvalue(agg_p)})"
        cap_parts.append(leg)
    agg_method = p.get("aggregation_method")
    if isinstance(agg_method, str):
        cap_parts.append(f"aggregation: {agg_method}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_rdd(p: dict[str, Any]) -> str | None:
    """Three-flavor τ table (conventional / bias-corrected / robust)
    with bandwidth and effective-N diagnostics in the caption.

    Calonico-Cattaneo-Titiunik's ``rdrobust`` reports three flavors
    of the same parameter at a single fit. The model needs all three
    to write the standard "robust τ = X (BC: Y, conv: Z)" line; this
    renderer lays them out as one row per flavor.
    """
    flavors = (
        ("conventional", "tau_conventional", "se_conventional",
         "p_conventional", "ci_lower_conventional", "ci_upper_conventional"),
        ("bias-corrected", "tau_bias_corrected", "se_bias_corrected",
         "p_bias_corrected", "ci_lower_bias_corrected", "ci_upper_bias_corrected"),
        ("robust", "tau_robust", "se_robust",
         "p_robust", "ci_lower_robust", "ci_upper_robust"),
    )
    has_p = any(isinstance(p.get(f[3]), (int, float)) for f in flavors)
    has_ci = any(isinstance(p.get(f[4]), (int, float)) for f in flavors)
    header = ["Estimator", "τ", "SE"]
    if has_p:
        header.append("p-value")
    if has_ci:
        header.append("95% CI")
    rows: list[list[str]] = []
    for label, tau_k, se_k, p_k, lo_k, hi_k in flavors:
        tau = p.get(tau_k)
        if tau is None:
            continue
        row = [label, _fmt_num(tau), _fmt_num(p.get(se_k))]
        if has_p:
            row.append(_fmt_pvalue(p.get(p_k)))
        if has_ci:
            lo, hi = p.get(lo_k), p.get(hi_k)
            row.append(
                f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"
                if lo is not None and hi is not None else ""
            )
        rows.append(row)
    if not rows:
        return None
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []
    n_left = p.get("effective_n_left")
    n_right = p.get("effective_n_right")
    if isinstance(n_left, int) and isinstance(n_right, int):
        cap_parts.append(f"effective N: {n_left:,} left · {n_right:,} right")
    bw_l = p.get("bandwidth_left")
    bw_r = p.get("bandwidth_right")
    if isinstance(bw_l, (int, float)) and isinstance(bw_r, (int, float)):
        if bw_l == bw_r:
            cap_parts.append(f"bandwidth = {_fmt_num(bw_l)}")
        else:
            cap_parts.append(
                f"bandwidth: {_fmt_num(bw_l)} left · {_fmt_num(bw_r)} right"
            )
    kernel = p.get("kernel")
    po = p.get("polynomial_order")
    if isinstance(kernel, str):
        leg = kernel
        if isinstance(po, int):
            leg += f", deg {po}"
        cap_parts.append(leg)
    cutoff = p.get("cutoff")
    rv = p.get("running_variable")
    if cutoff is not None and isinstance(rv, str):
        cap_parts.append(f"cutoff: {rv} = {_fmt_num(cutoff)}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_kaplan_meier(p: dict[str, Any]) -> str | None:
    """Median survival + S(t) at preset horizons. The full step
    function isn't in the payload by design (privacy carve-out);
    this renderer shows the horizon-scalar safe form."""
    rows: list[list[str]] = []
    median = p.get("median_survival_time")
    if isinstance(median, (int, float)) and math.isfinite(float(median)):
        ci_lo = p.get("median_survival_ci_lower")
        ci_hi = p.get("median_survival_ci_upper")
        ci_str = (
            f"[{_fmt_num(ci_lo)}, {_fmt_num(ci_hi)}]"
            if ci_lo is not None and ci_hi is not None else ""
        )
        rows.append(["Median", _fmt_num(median), "", ci_str])
    for h in ("1y", "3y", "5y", "10y"):
        s = p.get(f"survival_at_{h}")
        if not isinstance(s, (int, float)):
            continue
        n_risk = p.get(f"n_at_risk_{h}")
        ci_lo = p.get(f"survival_at_{h}_ci_lower")
        ci_hi = p.get(f"survival_at_{h}_ci_upper")
        ci_str = (
            f"[{_fmt_num(ci_lo)}, {_fmt_num(ci_hi)}]"
            if ci_lo is not None and ci_hi is not None else ""
        )
        rows.append([
            f"S({h})", _fmt_num(s),
            _fmt_int(n_risk) if n_risk is not None else "",
            ci_str,
        ])
    if not rows:
        return None
    header = ["Quantity", "Estimate", "N at risk", "95% CI"]
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []
    n_subj = p.get("n_subjects")
    n_fail = p.get("n_failures")
    if isinstance(n_subj, int):
        leg = f"subjects = {n_subj:,}"
        if isinstance(n_fail, int):
            leg += f" · events = {n_fail:,}"
        cap_parts.append(leg)
    lr_chi = p.get("logrank_chi_squared")
    lr_p = p.get("logrank_p_value")
    n_groups = p.get("n_groups")
    if isinstance(lr_chi, (int, float)) and math.isfinite(float(lr_chi)):
        leg = f"log-rank χ² = {_fmt_num(lr_chi)}"
        if isinstance(lr_p, (int, float)) and math.isfinite(float(lr_p)):
            leg += f" (p = {_fmt_pvalue(lr_p)})"
        if isinstance(n_groups, int):
            leg += f" across {n_groups} groups"
        cap_parts.append(leg)
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _render_factor_decomposition(p: dict[str, Any]) -> str | None:
    """Loadings matrix (variable × component) with explained-variance
    in the caption.

    Renders one row per variable, one column per component, with
    loadings as cell values. The explained-variance / cumulative-
    variance / eigenvalues live in the caption since they're per-
    component aggregates rather than per-variable. KMO + Bartlett +
    chi² also land in the caption when present.
    """
    loadings = p.get("loadings") or {}
    if not isinstance(loadings, dict) or not loadings:
        return None
    variables = p.get("variables") or sorted(loadings.keys())
    components = p.get("components") or []
    if not isinstance(variables, list) or not isinstance(components, list):
        return None
    header = ["Variable"] + list(components)
    rows: list[list[str]] = []
    for v in variables:
        row = [str(v)]
        v_loadings = loadings.get(v, {}) if isinstance(loadings.get(v), dict) else {}
        for c in components:
            row.append(_fmt_num(v_loadings.get(c)))
        rows.append(row)
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []
    method = p.get("method")
    if isinstance(method, str):
        cap_parts.append(method.replace("_", " "))
    rot = p.get("rotation")
    if isinstance(rot, str) and rot != "none":
        cap_parts.append(f"rotation: {rot}")
    n_obs = p.get("n_observations")
    if isinstance(n_obs, int):
        cap_parts.append(f"n = {n_obs:,}")
    # Variance summary per component.
    evr = p.get("explained_variance_ratio") or {}
    if isinstance(evr, dict) and evr:
        parts = []
        for c in components:
            if c in evr:
                parts.append(f"{c}: {evr[c]*100:.1f}%")
        if parts:
            cap_parts.append("variance: " + ", ".join(parts))
    cum = p.get("cumulative_variance") or {}
    if isinstance(cum, dict) and cum and components:
        # Show only the final cumulative (most informative).
        last = components[-1]
        if last in cum:
            cap_parts.append(f"cumulative: {cum[last]*100:.1f}%")
    # Goodness-of-fit summary for ML-FA.
    kmo = p.get("kmo")
    if isinstance(kmo, (int, float)) and math.isfinite(float(kmo)):
        cap_parts.append(f"KMO = {_fmt_num(kmo)}")
    chi2 = p.get("chi_squared")
    chi2_p = p.get("chi_squared_p_value")
    if isinstance(chi2, (int, float)) and math.isfinite(float(chi2)):
        leg = f"χ² = {_fmt_num(chi2)}"
        if isinstance(chi2_p, (int, float)) and math.isfinite(float(chi2_p)):
            leg += f" (p = {_fmt_pvalue(chi2_p)})"
        cap_parts.append(leg)
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


def _cluster_caption(p: dict[str, Any]) -> str:
    """Shared caption builder for cluster_analysis payloads.

    Pulled out so the centroids and no-centroids (DBSCAN / HDBSCAN)
    render paths share the same caption — keeps the model's view
    consistent across cluster methods.
    """
    cap_parts: list[str] = []
    method = p.get("method")
    if isinstance(method, str):
        cap_parts.append(method.replace("_", "-"))
    n_obs = p.get("n_observations")
    if isinstance(n_obs, int):
        cap_parts.append(f"n = {n_obs:,}")
    n_cl = p.get("n_clusters")
    if isinstance(n_cl, int):
        cap_parts.append(f"k = {n_cl}")
    n_noise = p.get("n_noise_points")
    if isinstance(n_noise, int):
        # DBSCAN / HDBSCAN-specific diagnostic: points that didn't fit
        # any cluster. Shown alongside k so the reader can see the
        # noise fraction at a glance.
        cap_parts.append(f"noise = {n_noise:,}")
    ss_ratio = p.get("ss_ratio")
    if isinstance(ss_ratio, (int, float)) and math.isfinite(float(ss_ratio)):
        cap_parts.append(f"between/total = {ss_ratio*100:.1f}%")
    sil = p.get("silhouette_score")
    if isinstance(sil, (int, float)) and math.isfinite(float(sil)):
        cap_parts.append(f"silhouette = {_fmt_num(sil)}")
    n_iter = p.get("n_iterations")
    if isinstance(n_iter, int):
        cap_parts.append(f"iter = {n_iter}")
    return " · ".join(cap_parts)


def _render_cluster_analysis(p: dict[str, Any]) -> str | None:
    """Centroids matrix (cluster × variable) with cluster sizes
    column.

    The "Size" column is the SDC-relevant info — small clusters
    were already suppressed by the sanitizer, but the surviving
    cluster sizes are still load-bearing context for the model.
    Centroid values are precision-clamped per-cluster by the
    sanitizer; the renderer just formats whatever survived.

    DBSCAN / HDBSCAN payloads carry no centroids by construction
    (the sanitizer accepts these absent — see
    ``_CLUSTER_METHODS_WITHOUT_CENTROIDS``); for those, fall through
    to a sizes-only table so the model still sees a rendered result
    instead of silently nothing.
    """
    centroids = p.get("centroids") or {}
    cluster_labels = p.get("cluster_labels") or []
    sizes = p.get("cluster_sizes") or {}
    if not isinstance(cluster_labels, list) or not isinstance(sizes, dict):
        return None

    # No-centroids fallback (DBSCAN / HDBSCAN). Render a sizes-only
    # table when there's at least a label list to drive it. Without
    # this branch, a successful density-based cluster run produced
    # no rendered output and the result was invisible in chat — the
    # tool card still existed, but the conversational surface that
    # the model embeds (this markdown) was empty.
    if not isinstance(centroids, dict) or not centroids:
        if not cluster_labels:
            return None
        header = ["Cluster", "Size"]
        rows: list[list[str]] = [
            [str(cl), _fmt_int(sizes.get(cl))] for cl in cluster_labels
        ]
        table = _markdown_table(header, rows)
        caption = _cluster_caption(p)
        return f"{table}\n\n{caption}" if caption else table

    if not cluster_labels:
        cluster_labels = sorted(centroids.keys())
    variables = p.get("variables") or []
    if not isinstance(variables, list):
        return None
    header = ["Cluster", "Size"] + list(variables)
    rows = []
    for cl in cluster_labels:
        row = [str(cl)]
        row.append(_fmt_int(sizes.get(cl)))
        cl_centroid = centroids.get(cl, {}) if isinstance(centroids.get(cl), dict) else {}
        for v in variables:
            row.append(_fmt_num(cl_centroid.get(v)))
        rows.append(row)
    table = _markdown_table(header, rows)

    caption = _cluster_caption(p)
    return f"{table}\n\n{caption}" if caption else table


def _render_marginal_effects(p: dict[str, Any]) -> str | None:
    """One row per focal variable. Effects + SE + p + CI in the
    standard regression-row format, plus the method and the
    conditioning point in the caption.

    Marginal effects from a non-linear estimator (logit, probit,
    Poisson) carry units the raw coefficient doesn't (probability
    change, count change); the caption surfaces ``model_family``
    so the model can interpret scale without re-deriving it.
    """
    variables = p.get("variables") or []
    if not isinstance(variables, list) or not variables:
        return None
    effects = p.get("effects") or {}
    if not isinstance(effects, dict) or not effects:
        return None
    ses = p.get("standard_errors") or {}
    pvs = p.get("p_values") or {}
    lows = p.get("ci_lower") or {}
    highs = p.get("ci_upper") or {}
    has_se = isinstance(ses, dict) and any(
        isinstance(ses.get(v), (int, float)) for v in variables
    )
    has_p = isinstance(pvs, dict) and any(
        isinstance(pvs.get(v), (int, float)) for v in variables
    )
    has_ci = (
        isinstance(lows, dict) and isinstance(highs, dict)
        and any(
            isinstance(lows.get(v), (int, float))
            and isinstance(highs.get(v), (int, float))
            for v in variables
        )
    )
    header = ["Variable", "Effect"]
    if has_se:
        header.append("SE")
    if has_p:
        header.append("p-value")
    if has_ci:
        header.append("95% CI")
    rows: list[list[str]] = []
    for v in variables:
        eff = effects.get(v)
        if eff is None:
            continue
        row = [str(v), _fmt_num(eff)]
        if has_se:
            row.append(_fmt_num(ses.get(v)))
        if has_p:
            row.append(_fmt_pvalue(pvs.get(v)))
        if has_ci:
            lo, hi = lows.get(v), highs.get(v)
            row.append(
                f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"
                if lo is not None and hi is not None else ""
            )
        rows.append(row)
    if not rows:
        return None
    table = _markdown_table(header, rows)

    cap_parts: list[str] = []
    method = p.get("method")
    if isinstance(method, str):
        cap_parts.append(method.replace("_", " "))
    family = p.get("model_family")
    if isinstance(family, str):
        cap_parts.append(family)
    outcome = p.get("outcome_variable")
    if isinstance(outcome, str):
        cap_parts.append(f"outcome: {outcome}")
    n = p.get("n")
    if isinstance(n, int):
        cap_parts.append(f"n = {n:,}")
    # Conditioning point: only meaningful for at_representative.
    at = p.get("at_values") or {}
    if isinstance(at, dict) and at:
        items = ", ".join(f"{k}={_fmt_num(at[k])}" for k in sorted(at))
        cap_parts.append(f"at: {items}")
    caption = " · ".join(cap_parts)
    return f"{table}\n\n{caption}" if caption else table


_HANDLERS: dict[str, Any] = {
    # Regression bucket — canonical descriptive name and legacy alias
    # both render through the same function. Older stored results
    # carry ``linear_regression``; new emissions carry the
    # ``coefficient_table_with_fit_stats`` form.
    "coefficient_table_with_fit_stats": _render_linear_regression,
    "linear_regression": _render_linear_regression,
    "t_test": _render_t_test,
    "descriptive": _render_descriptive,
    "frequency_table": _render_frequency_table,
    "crosstab": _render_crosstab,
    "magnitude_table": _render_magnitude_table,
    "correlation_matrix": _render_correlation_matrix,
    "did_event_study": _render_did_event_study,
    "rdd": _render_rdd,
    "kaplan_meier": _render_kaplan_meier,
    "factor_decomposition": _render_factor_decomposition,
    "cluster_analysis": _render_cluster_analysis,
    "marginal_effects": _render_marginal_effects,
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

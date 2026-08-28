"""Nora — MCP tool surface.

This module defines the *exhaustive* interface through which the frontier
model is allowed to reach the researcher's local machine. The canonical
tool list lives in ``nora.provider.tool_schemas.TOOL_SPECS``; this module
registers each spec with the Claude Agent SDK and supplies the handler
bodies. The Claude Agent SDK's built-in tools (Bash, Read, Write, Edit,
Glob, Grep, WebFetch, WebSearch, etc.) are disabled at the
provider layer (``provider/anthropic.py``) via ``disallowed_tools``
+ a ``can_use_tool`` catch-all.

All tools return structured payloads (JSON-encoded). Raw stdout never
crosses the boundary; the executor + sanitizer pipeline reduces script
output to typed result entries with SDC rules applied before the model
ever sees them.

Invariants enforced here:
- Values never cross the boundary. Mocked payloads never include simulated
  observation-level data.
- Every tool returns a structured payload (JSON-encoded). No raw stdout.
- Result IDs are opaque to Claude — it references them by label + ID.

See also: `project_builder_mcp_surface.md` (user memory) for the full spec.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool as _sdk_tool

from nora import data_request, executor, policy as policy_module, schema
from nora.config import PathEscapeError, get_cwd, resolve_in_cwd
from nora.policy import (
    depth_allowed,
    get_max_depth,
    has_explicit_policy,
    load_policy,
)
from nora import sanitizer
from nora.provider.tool_schemas import build_tool_specs
from nora.sanitizer import sanitize
from nora.store import get_store, open_store_uncached
from nora.text_safety import safe_text


# Caps for plot-helper manifest fields. Match the runner's
# ``_PLOT_LABEL_MAX_LEN`` / ``_PLOT_NAME_MAX_LEN`` so the chars cap on
# the same data crossing through both surfaces is consistent.
_PLOT_HELPER_LABEL_MAX_LEN = 120
_PLOT_HELPER_NAME_MAX_LEN = 80
# Helper-error prose: tight cap. The previous 400-char value was a
# leak surface — pandas / numpy exceptions can carry raw cell values
# in their formatted message bodies, and a script can deliberately
# raise with a row excerpt as the message. Bound here is short
# enough that even a hostile message can't smuggle a meaningful
# data slice through; the text-pattern allowlist below is what
# actually decides whether to forward the message at all.
_PLOT_HELPER_MESSAGE_MAX_LEN = 160

# Structural caps on the plot-helper summary payload returned by
# ``_summarize_plot_helpers``. Per-entry fields are already bounded by
# the length caps above (~280 bytes/row), but without entry-count caps
# a script can loop over helper calls (or write directly to the JSONL
# files in its run_dir) and force a megabyte-scale ``plots.succeeded``
# / ``plots.failed`` payload that bypasses ``_INLINE_PAYLOAD_BUDGET``
# trimming — the trim logic only inspects ``payload`` / ``markdown``
# on result entries, not the ``plots`` envelope. The numbers below
# comfortably accommodate legitimate research output (a single
# script rarely produces more than a handful of plots, and helper
# failures stop being useful past the first few) and refuse anything
# that looks engineered.
_PLOT_HELPER_MAX_ROWS = 50
_PLOT_HELPER_MAX_BYTES = 16_000

# Anchored full-string regexes for canonical import / dependency
# error shapes the model can act on. Each pattern matches the
# ENTIRE cleaned message; a partial match is not enough. The
# package-name character class is restricted to identifier
# characters so a hostile exception like
# ``ModuleNotFoundError: No module named 'matplotlib' (secret=foo)``
# does not match.
#
# Why exact-anchored shapes and not substring tokens: prior versions
# allowlisted bare words like ``params`` / ``pandas`` / ``numpy``,
# which let any row-derived exception body containing those words
# (trivial — pandas formatters routinely embed them) bypass the
# gate verbatim. The gate is defense-in-depth around the SDC
# boundary; it has to refuse anything it can't structurally vouch
# for, even at the cost of redacting some legitimate diagnostic
# text.
_PLOT_HELPER_IMPORT_REGEXES: tuple[re.Pattern[str], ...] = (
    # Python: ``ModuleNotFoundError: No module named 'matplotlib'``
    # and the bare ``No module named '...'`` form (R / generic).
    re.compile(r"(?:ModuleNotFoundError: )?No module named ['\"][A-Za-z_][\w.\-]*['\"]"),
    # Python: ``ImportError: cannot import name 'X' from 'Y'``
    re.compile(
        r"(?:ImportError: )?cannot import name ['\"][A-Za-z_]\w*['\"]"
        r"(?: from ['\"][A-Za-z_][\w.\-]*['\"])?"
    ),
    # Python: ``could not import statsmodels.api`` (helper-emitted
    # phrasing, identifier-only).
    re.compile(r"could not import [A-Za-z_][\w.]*"),
    # R: ``could not find function "read_dta"``.
    re.compile(r"could not find function ['\"][A-Za-z_][\w.]*['\"]"),
    # R: ``there is no package called 'haven'``.
    re.compile(r"there is no package called ['\"][A-Za-z_][\w.]*['\"]"),
    # Stata: helper-emitted, fully structural — no row data path.
    # See ``nora_plot_residuals.ado`` and siblings.
    re.compile(
        r"plot_\w+ failed at step \w+ with _rc=\d+; "
        r"check stderr\.log for the underlying error"
    ),
)

# Exact-match Nora-authored helper-failure messages. These come
# from ``_helper_failed`` call sites that pass a fully-static
# string (no f-string interpolation of caller-controlled values).
# Stems with interpolated values are deliberately excluded — the
# interpolated coef / column / dict-key can carry row data.
_PLOT_HELPER_NORA_AUTHORED_EXACT: frozenset[str] = frozenset({
    "fitted object has no .resid / .fittedvalues; skipping",
    "numpy missing",
    "fitted object has no .params; need a statsmodels-style fit",
    "nothing to plot after dropping intercept term",
    "`models` must be a dict of at least 2 fits keyed by label",
    "`coef` must be a coefficient name string",
    # R-side equivalent of the var-must-be-string guard.
    "nora$plot_interaction: `var` must be a single name string",
})

_PLOT_HELPER_REDACTED_PLACEHOLDER = (
    "(error message redacted; full text in researcher's run log)"
)


def _safe_helper_error_message(raw: object) -> str:
    """Forward only structurally-recognizable import / dependency
    errors and an exact-match set of Nora-authored helper messages;
    redact anything else.

    Helper exceptions are written by user-authored scripts, and
    Python / R exception formatters (pandas, numpy, base R) routinely
    embed raw cell values in their message bodies. ``safe_text``
    alone caps length and strips control characters, but a 160-char
    excerpt of cell values would still leak. The gate is therefore
    structural: the message must FULLY match one of the anchored
    import-error regexes (whose character classes are restricted to
    identifier characters) or be exactly equal to a known Nora-
    authored string. Any other input — including messages that
    happen to contain a recognized word like ``pandas`` or
    ``params`` alongside row data — is replaced with a redacted
    placeholder.
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    cleaned = safe_text(raw, max_len=_PLOT_HELPER_MESSAGE_MAX_LEN)
    if not cleaned:
        return ""
    if cleaned in _PLOT_HELPER_NORA_AUTHORED_EXACT:
        return cleaned
    for pattern in _PLOT_HELPER_IMPORT_REGEXES:
        if pattern.fullmatch(cleaned):
            return cleaned
    return _PLOT_HELPER_REDACTED_PLACEHOLDER


# Single source of truth for tool name + description + arg shape lives
# in ``nora.provider.tool_schemas``. The decorator below pulls every
# field from there so the @tool registration cannot drift from the
# canonical spec — and so the model sees identical guidance regardless
# of provider. Cached at module load: ``build_tool_specs()`` resolves
# ``request_data``'s description from
# ``data_request.SUPPORTED_REQUEST_TYPES``, which is already imported
# transitively above via ``from nora import data_request``.
_TOOL_SPECS_BY_NAME = {spec.name: spec for spec in build_tool_specs()}


def tool(name: str):
    """Apply ``claude_agent_sdk.tool`` with description and arg-types
    pulled from ``nora.provider.tool_schemas.TOOL_SPECS``. Editing a
    tool's description means editing the spec; the @tool registration
    follows. Drift is caught by ``test_tool_schema_consistency``."""
    spec = _TOOL_SPECS_BY_NAME[name]
    return _sdk_tool(name, spec.description, spec.as_sdk_args())


def _effective_n(payload: dict[str, Any]) -> int | None:
    """Return the number of observations an analysis *used*, per payload type.

    This is the field we compare against the source dataset's row count
    to catch silent filtering. Different analysis types encode "rows
    used" differently — the regression's ``n`` is post-NA-drop; a
    frequency table's ``n`` is total rows including missing; a
    crosstab has no explicit total but can be summed from cells, etc.

    Returns ``None`` when we can't confidently compute it (types whose
    payload omits the relevant field, or whose cells are fully
    suppressed so summation would underestimate).
    """
    t = payload.get("type")
    # ``coefficient_table_with_fit_stats`` is the canonical name for
    # the regression bucket as emitted by the R / Python / Stata
    # helpers; ``linear_regression`` is the legacy alias kept on read
    # for older stored payloads. See ``sanitizer.py`` for the alias
    # definition. ``result_render.py`` handles the same pair inline
    # in its dispatch table.
    if t in ("linear_regression", "coefficient_table_with_fit_stats"):
        n = payload.get("n")
        return n if isinstance(n, int) else None
    if t == "t_test":
        n1 = payload.get("n1")
        n2 = payload.get("n2")
        if not isinstance(n1, int):
            return None
        if n2 is None:
            return n1  # one_sample / paired
        if not isinstance(n2, int):
            return None
        return n1 + n2
    if t == "descriptive":
        n = payload.get("n")
        m = payload.get("missing_count")
        if isinstance(n, int) and isinstance(m, int):
            return n + m
        return None
    if t == "frequency_table":
        n = payload.get("n")
        return n if isinstance(n, int) else None
    if t == "crosstab":
        # Sum over cells + missing_count. Suppressed cells are strings,
        # which we skip; if any cell is suppressed, the sum is a lower
        # bound — the check is therefore conservative (won't false-
        # flag row-count changes that are really suppression artefacts).
        counts = payload.get("counts", {})
        total = payload.get("missing_count", 0) or 0
        if not isinstance(total, int):
            return None
        any_suppressed = False
        for inner in counts.values():
            if not isinstance(inner, dict):
                return None
            for v in inner.values():
                if isinstance(v, int):
                    total += v
                else:
                    any_suppressed = True
        return None if any_suppressed else total
    if t == "magnitude_table":
        cells = payload.get("cells", {})
        total = 0
        any_suppressed = False
        for cell in cells.values():
            if not isinstance(cell, dict):
                return None
            n = cell.get("n")
            if isinstance(n, int):
                total += n
            else:
                any_suppressed = True
        return None if any_suppressed else total
    if t == "correlation_matrix":
        # n = complete-cases count; missing_count = rows dropped by
        # listwise NA-removal on the chosen variables. Sum gives the
        # rows the DataFrame had when the correlation was computed,
        # which is what we want to compare to source_n. Same pattern
        # as ``descriptive`` above.
        n = payload.get("n")
        m = payload.get("missing_count")
        if isinstance(n, int) and isinstance(m, int):
            return n + m
        return None
    if t == "marginal_effects":
        n = payload.get("n")
        return n if isinstance(n, int) else None
    if t == "kaplan_meier":
        # Subject-level count, not row-level. The KM helper consumes
        # wide-form survival data (one row per subject), so for that
        # use case n_subjects == source row count. Long-form / split-
        # episode data isn't the helper's intended input and would
        # legitimately false-flag here; KM users should ship wide-form.
        n = payload.get("n_subjects")
        return n if isinstance(n, int) else None
    if t in ("factor_decomposition", "cluster_analysis"):
        n = payload.get("n_observations")
        return n if isinstance(n, int) else None
    # Deliberately NOT handled — the absence is intentional, not a
    # forgotten branch:
    # - ``rdd``: only emits ``effective_n_left`` / ``effective_n_right``
    #   / ``effective_n_total``, all bandwidth-restricted by
    #   construction. Comparing to source_n would false-flag every
    #   run since narrowing to the bandwidth IS the analysis.
    # - ``did_event_study``: only emits ``n_treated_per_group`` (per-
    #   cohort treated-unit counts). Not rows, and not even total
    #   units (untreated controls aren't counted) — units-vs-rows
    #   mismatch against source_n.
    # Revisit if either schema gains a "rows the analysis ingested"
    # field.
    return None


def _resolve_source_row_count(source_dataset: str | None) -> int | None:
    """Look up the row count of ``source_dataset`` for the row-count
    audit. Returns ``None`` when the dataset can't be read, isn't
    inside cwd, or the format has no fast row-count path.

    Used once per ``submit_script`` invocation, with the result threaded
    into every per-payload ``_check_row_count`` call. Calling this once
    per result instead — the previous behavior — re-read the entire
    dataset on every iteration; on a 3 GB .dta with 24 emitted results
    that meant a ~20 minute post-execution lag.
    """
    if not source_dataset:
        return None
    try:
        path = resolve_in_cwd(source_dataset)
    except PathEscapeError:
        return None
    if not path.is_file():
        return None
    return schema.row_count(path)


def _check_row_count(
    sanitized_payload: dict[str, Any],
    source_dataset: str | None,
    source_n: int | None,
) -> str | None:
    """If ``source_dataset`` and ``source_n`` are given, compare
    analysis N to dataset N.

    Returns a transformation-log string describing the row-count change,
    or ``None`` if no check could be made or no discrepancy exists.
    All error paths are silent — this is a best-effort audit signal,
    not a gate.

    ``source_n`` is computed ONCE per submit_script call (in
    ``_resolve_source_row_count``) and threaded in so the per-payload
    loop doesn't re-read the dataset on every iteration.
    """
    if not source_dataset or source_n is None:
        return None

    analysis_n = _effective_n(sanitized_payload)
    if analysis_n is None:
        # We couldn't compute the effective N — don't claim a change.
        return None

    if analysis_n == source_n:
        return None
    if analysis_n > source_n:
        # Reverse case: analysis used more rows than the dataset has.
        # Unlikely (suggests the script read a different file), but
        # worth surfacing so it doesn't go unnoticed.
        return (
            f"ROW COUNT ANOMALY: analysis used n={analysis_n} but source "
            f"dataset {source_dataset!r} has only {source_n} rows. The "
            f"script may have read a different file, merged in another, "
            f"or bootstrapped. Verify the intent."
        )

    diff = source_n - analysis_n
    pct = diff * 100.0 / source_n if source_n else 0.0
    return (
        f"ROW COUNT CHANGE: analysis used n={analysis_n} rows but source "
        f"dataset {source_dataset!r} has {source_n}; {diff} row(s) "
        f"excluded ({pct:.1f}%). Common causes: NA-drop by the analysis "
        f"command, an `if` / `subset(...)` / `filter(...)` in the script, "
        f"or a listwise-deletion from complete.cases. Verify the "
        f"exclusion was intentional."
    )


# Text constants returned by mocked tools so it is unambiguous to the
# researcher (and the frontier model) that this is a placeholder interface.
_MOCK_NOTE = "MOCKED. Step 2 of the build ladder. Returns placeholder data."


def _summarize(payload: dict[str, Any]) -> str:
    """Produce a compact one-liner for a sanitized payload.

    This is what Claude carries in context; the full payload is accessible
    via `expand_result(id)`. Keeping summaries terse matters — by the third
    or fourth analysis, the context is where most of the token pressure
    lives, not the individual tool results.
    """
    t = payload.get("type")
    # Same alias pair as ``_effective_n`` above — match both so the
    # regression-bucket one-liner fires regardless of which name the
    # emitting helper used.
    if t in ("linear_regression", "coefficient_table_with_fit_stats"):
        n = payload.get("n")
        r2 = payload.get("r_squared")
        k = len(payload.get("predictor_variables", []))
        return f"OLS, n={n}, R²={r2}, {k} predictor(s)"
    if t == "t_test":
        sub = payload.get("test_type")
        n1 = payload.get("n1")
        n2 = payload.get("n2")
        tstat = payload.get("t_statistic")
        p = payload.get("p_value")
        n_part = f"n1={n1}" + (f", n2={n2}" if n2 is not None else "")
        return f"{sub} t-test, {n_part}, t={tstat}, p={p}"
    if t == "descriptive":
        v = payload.get("variable")
        n = payload.get("n")
        m = payload.get("mean")
        sd = payload.get("sd")
        # ``distinct_count`` is the headline for a unique-count query and
        # is the field most at risk of being lost to context trimming
        # (the summary is the last representation to survive the staged
        # inline-budget trim). Carry it when present — handles both the
        # exact integer and the ``"<10"`` suppression marker.
        dc = payload.get("distinct_count")
        distinct_part = f", distinct={dc}" if dc is not None else ""
        return f"descriptive for {v!r}, n={n}, mean={m}, sd={sd}{distinct_part}"
    if t == "frequency_table":
        v = payload.get("variable")
        counts = payload.get("counts", {})
        suppressed = sum(1 for x in counts.values() if isinstance(x, str))
        return (
            f"frequency table for {v!r}, {len(counts)} levels "
            f"({suppressed} suppressed)"
        )
    if t == "crosstab":
        rv = payload.get("row_variable")
        cv = payload.get("col_variable")
        counts = payload.get("counts", {})
        n_rows = len(counts)
        n_cols = max((len(v) for v in counts.values()), default=0)
        suppressed = sum(
            1 for inner in counts.values() for x in inner.values()
            if isinstance(x, str)
        )
        return (
            f"crosstab {rv!r} × {cv!r}, {n_rows}×{n_cols} cells "
            f"({suppressed} suppressed)"
        )
    if t == "magnitude_table":
        rv = payload.get("row_variable")
        vv = payload.get("value_variable")
        agg = payload.get("aggregation")
        cells = payload.get("cells", {})
        suppressed = sum(
            1 for cell in cells.values()
            if isinstance(cell.get("value"), str)
        )
        return (
            f"magnitude table: {agg} of {vv!r} by {rv!r}, "
            f"{len(cells)} groups ({suppressed} suppressed)"
        )
    return f"result of type {t!r}"

# Per-analysis-type drop registry for the ``view="coefficients"``
# trim. The same map is consulted by ``_compact_payload`` (inline
# per-result trim on submit_script responses) and by
# ``expand_result(view="coefficients")``, so a new diagnostic field
# (e.g. a Wald-test side table on logit) only needs one entry here
# to be hidden from the compact view across BOTH callers.
#
# Keys are ``payload["type"]`` strings as emitted by the runtime
# helpers. Values are tuples of payload field names to drop. An
# analysis_type absent from this map is "view-coefficients-clean"
# already (no expensive diagnostics to trim) and the view is a
# no-op — see ``_VIEW_COEFFICIENTS_NOOP`` below for the explicit
# signal to the model.
#
# The previous code had ``("vcov", "vif")`` written twice in two
# different locations; adding a third diagnostic to the runtime
# (condition_number, partial_residuals, …) silently leaked into the
# compact view because only one of the two copies got updated.
_VIEW_COEFFICIENTS_DROP_FIELDS: dict[str, tuple[str, ...]] = {
    # Both the legacy ``linear_regression`` and the canonical
    # ``coefficient_table_with_fit_stats`` map to the same drop set —
    # the trim follows the payload shape, not the type name. Same
    # alias treatment as ``_HANDLERS`` in ``result_render.py``.
    "linear_regression": ("vcov", "vif"),
    "coefficient_table_with_fit_stats": ("vcov", "vif"),
}


def _apply_view_coefficients_trim(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str], bool]:
    """Apply the ``view="coefficients"`` drop registry to a payload.

    Returns ``(trimmed_payload, dropped_fields, view_applied)``:

    - ``trimmed_payload`` is a shallow copy with the registered
      fields removed (or the original payload when no trim applies).
    - ``dropped_fields`` lists the keys actually removed (always a
      subset of the registry entry — fields that weren't present in
      the payload aren't reported as dropped).
    - ``view_applied`` is ``True`` when a registry entry matched the
      payload's analysis type, regardless of whether anything was
      actually removed. Callers use this to surface
      ``view_ignored_for_type`` when a model passed
      ``view="coefficients"`` against a payload type that doesn't
      participate in the trim — without that signal the response's
      ``view="coefficients"`` field would lie-by-omission about what
      the trim did.
    """
    if not isinstance(payload, dict):
        return payload, [], False
    drop_fields = _VIEW_COEFFICIENTS_DROP_FIELDS.get(payload.get("type", ""))
    if drop_fields is None:
        return payload, [], False
    trimmed = dict(payload)
    dropped: list[str] = []
    for key in drop_fields:
        if key in trimmed:
            trimmed.pop(key)
            dropped.append(key)
    return trimmed, dropped, True


def _compact_payload(sanitized: dict[str, Any]) -> dict[str, Any]:
    """Inline-trimmed version of a sanitized payload for the per-result
    response entry. Same shape as ``expand_result(view="coefficients")``
    for regressions: full coefficient pattern, R^2, condition number,
    n, df, etc., minus the variance-covariance matrix and per-
    predictor VIF table. Other analysis types pass through unchanged
    (their payloads are already small).

    The motivation is the parameterized-batch case: a 24-spec script
    used to force the model into 24 ``expand_result`` calls just to
    render the headline coefficient tables, since the per-result
    ``summary`` is a one-line ("OLS, n=…, R²=…, K predictor(s)")
    that doesn't carry coefficients. Including the trimmed payload
    inline turns those 24 round-trips into zero — the model has the
    data it needs to render tables directly from the submit_script
    response. ``expand_result`` is still there for the cases where
    full ``vcov``/``vif`` matter (collinearity audits, joint tests).

    Note: a second-stage trim in ``submit_script`` may strip this
    field entirely from each entry when the assembled envelope would
    exceed ``_INLINE_PAYLOAD_BUDGET`` (the SDK persists oversize tool
    results to a file the model can't read). When that fires, the
    inline ``markdown`` table remains and the model can call
    ``expand_result(view="full")`` on specific result_ids for raw
    numbers.
    """
    if not isinstance(sanitized, dict):
        return {}
    trimmed, _dropped, _applied = _apply_view_coefficients_trim(sanitized)
    return trimmed if trimmed is not sanitized else dict(sanitized)


# Two-stage inline budget for the assembled ``submit_script``
# envelope. Earlier behavior set a single threshold at 35k chars
# (just below the Claude Agent SDK's ~56k tool-result cap) — that
# only fired in the rare worst case, leaving every "moderate" multi-
# regression turn shipping its full payload inline and bloating the
# conversation faster than necessary.
#
# Stage 1 — payload trim, fires at ``_INLINE_PAYLOAD_BUDGET``: drop
# the heavy ``payload`` (raw coefficient arrays, vcov, vif) from
# each ok-status result. ``markdown`` (the canonical table) and the
# small fields (label, type, n, summary, result_id) stay. This is
# the right default for context economy — markdown carries the
# numbers the model actually reasons over; payload is for cases the
# model needs vcov / vif / raw arrays, which it can pull via
# ``expand_result(view="full")`` per result_id. 12k is below most
# multi-regression batches' total inline cost, so this fires often
# enough to noticeably slow context growth without depriving the
# model of the table view.
#
# Stage 2 — markdown summarization, fires at ``_INLINE_MARKDOWN_BUDGET``:
# even after stripping payloads, very heavy turns (e.g., 20+
# regression batch with verbose markdown) can still ship 45k+ chars
# of tables inline. At this point we replace each entry's
# ``markdown`` with a one-line summary noting the result_id and
# instruction to call ``expand_result`` for the table. This is the
# "pure handoff" mode — the model only sees handles, has to call
# ``expand_result`` to see anything substantive. Used sparingly.
#
# Both stages still preserve every result's ``result_id``,
# ``label``, ``type``, ``n``, ``summary``, and any error fields —
# enough for the model to compare batches and decide which to
# inspect deeper.
#
# Budget tuning: ``_INLINE_MARKDOWN_BUDGET`` was 30k originally,
# bumped to 45k so a typical 10–15-regression batch (common when
# the model runs the same spec across an asset-category panel) fits
# inline without the model having to expand each result manually.
# At ~3 chars/token for table markdown that's ~15k tokens per
# submit_script call — small fraction of even the 200k context
# window and well under 2% on 1M-context configurations.
_INLINE_PAYLOAD_BUDGET = 12_000
_INLINE_MARKDOWN_BUDGET = 45_000


def _trim_oversize_inline_payloads(results: list[dict[str, Any]]) -> dict[str, bool]:
    """Two-stage trim. Mutates ``results`` in place. Returns a dict
    of which stages fired:
        {"payload_omitted": bool, "markdown_omitted": bool}

    Stage 1 fires when the assembled envelope (payload + markdown
    cost across ok results) exceeds ``_INLINE_PAYLOAD_BUDGET``: the
    ``payload`` field is dropped from every ok entry. Stage 2 fires
    when the markdown alone still exceeds ``_INLINE_MARKDOWN_BUDGET``
    after stage 1: per-entry ``markdown`` is replaced with a single
    line pointing at the result_id.

    The caller surfaces these in the response envelope as
    ``_inline_payload_omitted`` / ``_inline_markdown_omitted`` so
    the model knows the shape changed and can call
    ``expand_result`` for the trimmed content.
    """
    payload_cost = sum(
        len(json.dumps(r.get("payload"), ensure_ascii=False))
        for r in results
        if r.get("status") == "ok" and "payload" in r
    )
    markdown_cost = sum(
        len(r.get("markdown", "")) for r in results
        if r.get("status") == "ok"
    )
    flags = {"payload_omitted": False, "markdown_omitted": False}

    # Stage 1: drop payloads if the combined inline body crosses the
    # budget. We could be cleverer (drop payloads only from the
    # heaviest entries) but uniform-drop keeps the contract simple
    # — model sees either "all payloads inline" or "all payloads
    # behind expand_result" for a given turn, never a mix.
    if payload_cost + markdown_cost > _INLINE_PAYLOAD_BUDGET:
        for entry in results:
            if entry.get("status") == "ok" and "payload" in entry:
                del entry["payload"]
                flags["payload_omitted"] = True

    # Stage 2: even with payloads dropped, markdown alone may still
    # be heavy on big regression batches. Replace each ``markdown``
    # with a stub. Recompute the markdown cost rather than reuse
    # the pre-stage-1 number (markdown didn't change in stage 1, so
    # the value is identical, but reading it explicitly here makes
    # the staging contract clearer for future edits).
    md_cost_post_s1 = sum(
        len(r.get("markdown", "")) for r in results
        if r.get("status") == "ok"
    )
    if md_cost_post_s1 > _INLINE_MARKDOWN_BUDGET:
        for entry in results:
            if entry.get("status") != "ok" or "markdown" not in entry:
                continue
            rid = entry.get("result_id", "?")
            label = entry.get("label", "")
            label_part = f" ({label})" if label else ""
            entry["markdown"] = (
                f"[Heavy result trimmed for context. result_id={rid}{label_part}; "
                f"call expand_result(\"{rid}\", view=\"full\") to fetch the table.]"
            )
            flags["markdown_omitted"] = True

    return flags


def _shared_transformations(results: list[dict[str, Any]]) -> list[str]:
    """Return transformation entries common to every status="ok" result,
    in first-seen order.

    Used by submit_script to hoist sanitizer transformations that
    repeat across a multi-result response (e.g. "clamped coefficient
    SEs to 3 sig figs at N=…" repeated 24 times in a 24-spec script).
    Entries that don't appear on every ok result stay per-result.

    Returns ``[]`` when there are fewer than 2 ok results (nothing
    to dedupe), or when the intersection is empty.
    """
    ok_lists = [
        entry.get("transformations", [])
        for entry in results
        if entry.get("status") == "ok"
    ]
    if len(ok_lists) < 2:
        return []
    ok_lists = [
        list(lst) if isinstance(lst, list) else []
        for lst in ok_lists
    ]
    if any(not lst for lst in ok_lists):
        return []
    intersection = set(ok_lists[0])
    for lst in ok_lists[1:]:
        intersection &= set(lst)
    if not intersection:
        return []
    # Preserve first-seen order from the first list.
    seen: set[str] = set()
    ordered: list[str] = []
    for t in ok_lists[0]:
        if t in intersection and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _summarize_plot_helpers(run_dir: Any) -> dict[str, Any] | None:
    """Summarize what the script's plot helpers actually did.

    Reads two files the runtime libraries write into
    ``<run_dir>/_nora_plots/``:

    - ``manifest.jsonl`` — one JSON line per SUCCESSFUL plot, with
      ``file``, ``kind``, optional ``label``.
    - ``helper_errors.jsonl`` — one JSON line per FAILED helper
      call, with ``helper``, ``error``, ``message``, optional ``fix``.

    Returns a dict the tool result includes as ``plots: ...`` so the
    MODEL sees what actually happened with each helper call.
    Without this surface, helper failures (matplotlib not installed,
    ``library(haven)`` error, etc.) only land in stderr — the model
    confidently says "thumbnail should be visible above" while the
    researcher sees nothing. Returns ``None`` when no helper calls
    were made (no ``_nora_plots/`` directory) so the field stays out
    of the response on plain analysis runs.
    """
    if run_dir is None:
        return None
    plots_dir = Path(run_dir) / "_nora_plots"
    if not plots_dir.is_dir():
        return None

    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    manifest = plots_dir / "manifest.jsonl"
    errors = plots_dir / "helper_errors.jsonl"

    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out

    # Every string in these JSONL files originates from a user-
    # authored script (the helper libraries write whatever the
    # researcher's code passes them). Without sanitization these go
    # straight into the model-visible tool result, so a script could
    # compute a label or filename from raw data values and inject
    # prompt instructions that bypass the analysis-payload sanitizer.
    # The runner's ``_capture_plots`` already runs the same data
    # through ``safe_text`` before storing pending plot images;
    # apply the same boundary here.
    #
    # Researcher-only kinds (e.g., residuals diagnostics) get a
    # ``researcher_only: true`` marker so the model knows the plot
    # was made on the disk side without seeing the image — the
    # image side channel around SDC is closed in the runner's
    # ``_PLOT_KIND_ALLOWLIST``; the surface here just keeps the
    # signal so the model doesn't loop calling ``plot_residuals``.
    from nora.runner import _PLOT_KIND_RESEARCHER_ONLY

    # Running totals enforced across both lists. Once we hit either the
    # row count cap or the byte budget, remaining entries are counted in
    # ``truncated_succeeded`` / ``truncated_failed`` (surfaced as
    # ``_truncated`` markers below) rather than appended. The byte cost
    # is measured on the per-row JSON encoding so the cap reflects the
    # actual size shipped through the bridge.
    bytes_used = 0
    truncated_succeeded = 0
    truncated_failed = 0

    def _row_bytes(r: dict[str, Any]) -> int:
        return len(json.dumps(r, ensure_ascii=False, separators=(",", ":")))

    def _budget_exhausted(extra: int) -> bool:
        total_rows = len(succeeded) + len(failed)
        return (
            total_rows >= _PLOT_HELPER_MAX_ROWS
            or bytes_used + extra > _PLOT_HELPER_MAX_BYTES
        )

    # Authenticity gate. Same posture as the runner's
    # ``_capture_plots``: each entry must carry a ``_token`` that
    # matches the executor-registered per-run token. The on-disk
    # rewrite in ``_filter_plot_manifest`` is best-effort (the
    # manifest lives in script-writable territory and the rewrite
    # CAN fail), so this surface re-validates rather than implicitly
    # trusting the on-disk file. Forged entries are dropped silently
    # so the model never sees "I plotted evil.png succeeded" for an
    # entry the runner refused to attach as vision.
    from nora.executor import get_run_token, RESULT_TOKEN_FIELD
    import secrets as _secrets
    expected_token = get_run_token(Path(run_dir))

    def _entry_authentic(entry: dict[str, Any]) -> bool:
        if expected_token is None:
            return False
        got = entry.get(RESULT_TOKEN_FIELD)
        return (
            isinstance(got, str)
            and _secrets.compare_digest(got, expected_token)
        )

    if manifest.is_file():
        for entry in _read_jsonl(manifest):
            if not _entry_authentic(entry):
                continue
            raw_file = entry.get("file", "?")
            raw_kind = entry.get("kind", "?")
            raw_label = entry.get("label", "")
            kind_str = (
                raw_kind if isinstance(raw_kind, str) else str(raw_kind)
            )
            row: dict[str, Any] = {
                "file": safe_text(
                    raw_file if isinstance(raw_file, str) else str(raw_file),
                    max_len=_PLOT_HELPER_NAME_MAX_LEN,
                ) or "?",
                "kind": safe_text(
                    kind_str, max_len=_PLOT_HELPER_NAME_MAX_LEN,
                ) or "?",
                "label": safe_text(
                    raw_label if isinstance(raw_label, str) else str(raw_label),
                    max_len=_PLOT_HELPER_LABEL_MAX_LEN,
                ),
            }
            if kind_str in _PLOT_KIND_RESEARCHER_ONLY:
                row["researcher_only"] = True
            cost = _row_bytes(row)
            if _budget_exhausted(cost):
                truncated_succeeded += 1
                continue
            succeeded.append(row)
            bytes_used += cost
    if errors.is_file():
        for entry in _read_jsonl(errors):
            raw_helper = entry.get("helper", "?")
            row: dict[str, Any] = {
                "helper": safe_text(
                    raw_helper if isinstance(raw_helper, str)
                    else str(raw_helper),
                    max_len=_PLOT_HELPER_NAME_MAX_LEN,
                ) or "?",
                "message": _safe_helper_error_message(
                    entry.get("message", ""),
                ),
            }
            # Surface ``fix`` only when we kept the message verbatim
            # — i.e. the message matched the import/dependency
            # allowlist. A redacted message has no actionable
            # context, so a "fix" hint that came from the same
            # exception body would be at best confusing and at
            # worst smuggle an attacker-controlled instruction
            # through a separate field.
            if row["message"] != _PLOT_HELPER_REDACTED_PLACEHOLDER:
                raw_fix = entry.get("fix")
                if raw_fix:
                    cleaned_fix = safe_text(
                        raw_fix if isinstance(raw_fix, str) else str(raw_fix),
                        max_len=_PLOT_HELPER_MESSAGE_MAX_LEN,
                    )
                    if cleaned_fix:
                        row["fix"] = cleaned_fix
            cost = _row_bytes(row)
            if _budget_exhausted(cost):
                truncated_failed += 1
                continue
            failed.append(row)
            bytes_used += cost

    if not succeeded and not failed:
        return None
    summary: dict[str, Any] = {
        "succeeded": succeeded,
        "failed": failed,
    }
    if truncated_succeeded:
        summary["truncated_succeeded"] = truncated_succeeded
    if truncated_failed:
        summary["truncated_failed"] = truncated_failed
    if failed and not succeeded:
        # Make the failure mode obvious in the model's reading of
        # the response. The model has been observed to say
        # "thumbnail should be visible above" when no plot landed;
        # this hint short-circuits that.
        summary["note"] = (
            "Plot helpers were called but produced no plots. "
            "Check failed[].message; common cause is a missing "
            "package (matplotlib / haven / scipy). The researcher "
            "won't see anything. Interpret with the numerical "
            "payload only, or ask them to install the missing "
            "package and re-run."
        )
    return summary


def _as_mcp_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON-serializable dict as an MCP text-content response.

    MCP content is a list of typed blocks; our convention is a single text
    block containing JSON. Keeping the payload structured (not prose) makes
    the downstream sanitizer job clean and keeps the contract testable.

    JSON is emitted minified (no indentation, tight separators) because
    the model consuming this is the only audience: the UI never renders
    the JSON body to the researcher, and the persisted log is a
    diagnostic artifact, not a human-reading surface. Minifying saves
    roughly 25-35% on every tool result, which compounds fast on
    sessions with wide-dataset get_schema calls.
    """
    return {
        "content": [
            {"type": "text", "text": json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False,
            )}
        ]
    }


# ---------------------------------------------------------------------------
# Tool: get_schema
# ---------------------------------------------------------------------------

@tool("get_schema")
async def get_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Step-3 implementation: real structural extraction, never values.

    Dispatches to `nora.schema.extract()` which handles .dta / .rds /
    .csv. The `dataset` argument is treated as a path relative to the
    researcher's working directory (or absolute, as long as it stays
    inside the cwd). Escape attempts get a policy-shaped denial.
    """
    dataset = args.get("dataset", "")
    depth = args.get("depth", policy_module.DEFAULT_MAX_DEPTH)

    if not dataset:
        return _as_mcp_text({
            "status": "error",
            "reason": "missing required argument: dataset",
        })

    # Path sandbox: reject anything outside cwd.
    try:
        path = resolve_in_cwd(dataset)
    except PathEscapeError as e:
        return _as_mcp_text({
            "status": "denied",
            "reason": str(e),
            "dataset": dataset,
        })

    if not path.exists():
        return _as_mcp_text({
            "status": "error",
            "reason": f"file not found: {dataset!r}. Check the path is correct.",
            "dataset": dataset,
        })
    if not path.is_file():
        return _as_mcp_text({
            "status": "error",
            "reason": f"{dataset!r} is not a file.",
            "dataset": dataset,
        })

    # Researcher consent policy: compare requested depth against the
    # ceiling set in `<cwd>/.nora/policy.json`. A missing policy
    # file or a missing per-dataset entry uses the app default
    # (``policy.DEFAULT_MAX_DEPTH``). The policy is a *ceiling* —
    # Claude can still request something narrower than the ceiling
    # if that's enough for the task.
    policy_doc = load_policy(get_cwd())
    ceiling = get_max_depth(policy_doc, path.name)
    if depth in policy_module.VALID_DEPTHS and not depth_allowed(depth, ceiling):
        explicit = has_explicit_policy(policy_doc, path.name)
        return _as_mcp_text({
            "status": "denied",
            "reason": (
                f"schema depth {depth!r} exceeds the researcher's "
                f"policy ceiling for {path.name!r} "
                f"({ceiling!r}{'; explicit' if explicit else '; default'}"
                f"). Ask for a narrower depth, or ask the researcher "
                f"to raise the ceiling for this dataset in "
                f".nora/policy.json."
            ),
            "dataset": dataset,
            "requested_depth": depth,
            "policy_max_depth": ceiling,
        })

    try:
        payload = schema.extract(path, depth)
    except schema.SchemaExtractError as e:
        # Parser-owned validation errors (unsupported format, invalid
        # depth, .rds-without-dataframe). Their messages are crafted
        # by ``schema.extract`` itself and are safe to forward
        # verbatim. Note: ``SchemaExtractError`` is a subclass of
        # ``ValueError``; we catch it BEFORE the generic ValueError
        # path so pandas-style ``ParserError`` (also a ``ValueError``
        # subclass, but with row content in its message) doesn't
        # take this branch and leak data.
        return _as_mcp_text({
            "status": "error",
            "reason": str(e),
            "dataset": dataset,
            "depth": depth,
        })
    except Exception as e:  # broad: lib-specific parse errors
        # ``e`` here is whatever the underlying reader raised
        # (pandas ParserError, pyreadstat ReadstatError, json
        # decode errors, ...). Those messages routinely embed the
        # offending row text or column value — surfacing them in
        # the model-visible reason contradicts the schema tool's
        # promise of never returning individual observation values.
        # We log the full detail server-side and return only the
        # exception class name to the model, which is enough to
        # tell pandas-CSV-malformed from pyreadstat-Stata-corrupted
        # without quoting any data.
        logging.getLogger(__name__).warning(
            "schema.extract failed for %s: %s", dataset, e, exc_info=True,
        )
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"failed to read {dataset!r} ({e.__class__.__name__}). "
                f"The dataset may be malformed or corrupted; researcher "
                f"logs have the underlying parser error."
            ),
            "dataset": dataset,
        })

    # Annotate the response with the policy ceiling so Claude knows
    # the max depth this dataset allows for future calls, without
    # needing to hit a denial to learn it.
    payload["policy_max_depth"] = ceiling
    return _as_mcp_text(payload)


# ---------------------------------------------------------------------------
# Tool: search_schema
# ---------------------------------------------------------------------------

# Cap on how many matches a single search_schema call returns. The model
# can refine the query if the cap is hit. Higher caps trade context size
# for fewer follow-ups; 50 is enough to surface every salary-related
# column on a typical wide research dataset without burning context.
_SEARCH_SCHEMA_DEFAULT_LIMIT = 50
_SEARCH_SCHEMA_HARD_CAP = 200


@tool("search_schema")
async def search_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Filter a dataset's schema by a case-insensitive name/label query.

    Same path-sandbox and policy ceiling as ``get_schema``. Returns a
    schema-shaped payload whose ``variables`` list is filtered to just
    the matches, plus a ``total_matches`` count and the original
    ``query`` so the response is self-describing.
    """
    dataset = args.get("dataset", "")
    query = args.get("query", "")
    requested_limit = args.get("limit", 0)

    if not dataset:
        return _as_mcp_text({
            "status": "error",
            "reason": "missing required argument: dataset",
        })
    if not isinstance(query, str) or not query.strip():
        return _as_mcp_text({
            "status": "error",
            "reason": (
                "missing required argument: query (case-insensitive "
                "substring; use get_schema for the full variable list)"
            ),
        })

    needle = query.strip().lower()

    # Path sandbox: same logic as get_schema. A common branch would be
    # nice to share but the divergence is small and inlining keeps
    # each tool's error path self-contained.
    try:
        path = resolve_in_cwd(dataset)
    except PathEscapeError as e:
        return _as_mcp_text({
            "status": "denied", "reason": str(e), "dataset": dataset,
        })
    if not path.exists():
        return _as_mcp_text({
            "status": "error",
            "reason": f"file not found: {dataset!r}. Check the path is correct.",
            "dataset": dataset,
        })
    if not path.is_file():
        return _as_mcp_text({
            "status": "error",
            "reason": f"{dataset!r} is not a file.",
            "dataset": dataset,
        })

    # Resolve the search depth: cap at names_types_labels (no point
    # loading summary stats for a name/label search). Honor the
    # researcher's policy ceiling — if they've restricted this
    # dataset to names_only, the search runs at names_only and only
    # name matches will land.
    policy_doc = load_policy(get_cwd())
    ceiling = get_max_depth(policy_doc, path.name)
    search_target = "names_types_labels"
    extract_depth = (
        ceiling if not policy_module.depth_allowed(search_target, ceiling)
        else search_target
    )

    try:
        payload = schema.extract(path, extract_depth)
    except schema.SchemaExtractError as e:
        # See get_schema for why this is the FIRST except clause —
        # SchemaExtractError extends ValueError, but we must not
        # blanket-catch ValueError here because pandas ParserError
        # (also a ValueError) would leak row content.
        return _as_mcp_text({
            "status": "error",
            "reason": str(e),
            "dataset": dataset,
            "depth": extract_depth,
        })
    except Exception as e:  # noqa: BLE001 — broad: lib-specific parse errors
        # See get_schema for why we drop ``str(e)`` from the
        # model-visible reason — parser exceptions can quote the
        # offending row content, which would defeat the schema
        # tool's no-individual-observation guarantee.
        logging.getLogger(__name__).warning(
            "search_schema extract failed for %s: %s",
            dataset, e, exc_info=True,
        )
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"failed to read {dataset!r} ({e.__class__.__name__}). "
                f"The dataset may be malformed or corrupted; researcher "
                f"logs have the underlying parser error."
            ),
            "dataset": dataset,
        })

    all_vars = payload.get("variables") or []
    matches: list[dict[str, Any]] = []
    for var in all_vars:
        if not isinstance(var, dict):
            continue
        name = str(var.get("name", "")).lower()
        label = str(var.get("label", "")).lower()
        if needle in name or (label and needle in label):
            matches.append(var)
            continue
        # Match against value_labels content too, when present —
        # useful for "find columns whose levels include 'private'".
        vls = var.get("value_labels")
        if isinstance(vls, dict):
            for k, v in vls.items():
                if needle in str(k).lower() or needle in str(v).lower():
                    matches.append(var)
                    break

    total_matches = len(matches)

    # Resolve limit: 0 / negative / non-int → default. Above hard cap → cap.
    if not isinstance(requested_limit, int) or requested_limit <= 0:
        limit = _SEARCH_SCHEMA_DEFAULT_LIMIT
    else:
        limit = min(requested_limit, _SEARCH_SCHEMA_HARD_CAP)
    truncated = total_matches > limit
    matches = matches[:limit]

    return _as_mcp_text({
        "status": "ok",
        "dataset": payload.get("dataset", dataset),
        "file_type": payload.get("file_type"),
        "depth": extract_depth,
        "policy_max_depth": ceiling,
        "observation_count": payload.get("observation_count"),
        "variable_count": len(all_vars),
        "query": query,
        "total_matches": total_matches,
        "limit": limit,
        "truncated": truncated,
        "variables": matches,
    })


# ---------------------------------------------------------------------------
# Tool: request_data
# ---------------------------------------------------------------------------

@tool("request_data")
async def request_data(args: dict[str, Any]) -> dict[str, Any]:
    """Step-5 implementation: real, SDC-gated bounded data queries.

    Each request type is a pre-approved computation with its own
    disclosure-control rule. See ``nora.data_request`` for the
    per-type logic.
    """
    dataset = args.get("dataset", "")
    request_type = args.get("request_type", "")
    variable = args.get("variable", "")
    # Optional second variable used by multi-variable types (e.g.,
    # correlation_pair). Single-variable types ignore it; passing it
    # to one is silently OK.
    variable2 = args.get("variable2") or None

    if not dataset:
        return _as_mcp_text({
            "status": "error",
            "reason": "missing required argument: dataset",
        })
    if not request_type:
        return _as_mcp_text({
            "status": "error",
            "reason": "missing required argument: request_type",
        })
    if not variable:
        return _as_mcp_text({
            "status": "error",
            "reason": "missing required argument: variable",
        })

    # Path sandbox.
    try:
        path = resolve_in_cwd(dataset)
    except PathEscapeError as e:
        return _as_mcp_text({
            "status": "denied",
            "reason": str(e),
            "dataset": dataset,
        })
    if not path.is_file():
        return _as_mcp_text({
            "status": "error",
            "reason": f"file not found: {dataset!r}",
            "dataset": dataset,
        })

    result = data_request.handle(
        path, request_type, variable, variable2=variable2,
    )
    payload: dict[str, Any] = {
        "status": result.status,
        "dataset": dataset,
        "request_type": request_type,
        "variable": variable,
    }
    if variable2:
        payload["variable2"] = variable2
    if result.answer is not None:
        payload["answer"] = result.answer
    if result.reason is not None:
        payload["reason"] = result.reason
    return _as_mcp_text(payload)


# ---------------------------------------------------------------------------
# Tool: submit_script
# ---------------------------------------------------------------------------
#
# ``submit_script`` is a pipeline. The body below is a thin coordinator
# over five helpers, each handling one phase. The split mirrors the
# data flow: execute → resolve SDC + source-row count → sanitize +
# store → build the base envelope → attach status / debug / plot
# metadata. Same observable behaviour as the prior single-function
# implementation; the helpers exist to make each phase readable and
# testable in isolation.


async def _execute_script_for_submit(
    language: str, code: str, cwd: Path,
) -> Any:
    """Run the executor in a worker thread, with cancellation handling.

    Returns the ``ExecutionResult`` on completion. Raises
    ``asyncio.CancelledError`` when the turn was cancelled mid-run —
    callers do nothing; the exception propagates through the SDK's
    tool-dispatch path (the MCP server wrapper only catches plain
    ``Exception``) and lands in the runner's outer ``except
    asyncio.CancelledError`` branch, which handles teardown.
    """
    import asyncio as _asyncio
    from nora.runtime.turn_context import (
        is_current_turn_cancelled,
        register_turn_process,
    )

    # Run the executor (synchronous Popen) in a worker thread while
    # registering the spawned process into the runner's per-turn
    # registry. If Stop fires mid-run, the runner has already marked
    # the turn cancelled and either:
    #   (a) ``register_turn_process`` saw the cancel flag and killed
    #       the subprocess on the spot (closes the race where the
    #       prior local ``proc_box`` saw a ``None`` because Stop fired
    #       between Popen returning and the register call landing), or
    #   (b) the registration completed first, in which case the
    #       runner's ``cancel_turn`` walked the registry under its
    #       lock and killed the proc.
    # Either way the subprocess actually halts when Stop fires; "Stop"
    # never feels like a no-op the way it did with the prior pattern.
    exec_result = await _asyncio.to_thread(
        executor.run_script,
        language, code, cwd,
        proc_register=register_turn_process,
    )

    if is_current_turn_cancelled():
        # Drop the result entirely if the turn was cancelled while the
        # subprocess was still running. Earlier this path returned a
        # ``{"status": "cancelled", ...}`` early payload, which the
        # caller wrapped as an MCP text response — the provider then
        # yielded a ``ToolCallResult`` carrying it, and even though
        # the dispatcher drops events from cancelled turns, the
        # payload still briefly entered the provider's outgoing state
        # (OpenAI's ``function_call_output`` list, Anthropic's CLI
        # session) before the asyncio cancel propagated. Raising
        # CancelledError here ensures no ``ToolCallResult`` is ever
        # emitted for the cancelled run: the MCP server wrapper
        # catches ``Exception`` only, so ``CancelledError`` (a
        # ``BaseException``) flows through the SDK's tool-dispatch
        # path uninterrupted and lands in the runner's outer
        # ``except asyncio.CancelledError`` branch where session
        # teardown happens. The raw run_dir stays on disk for the
        # researcher; no sanitize / store / response work runs.
        raise _asyncio.CancelledError()
    return exec_result


def _resolve_sdc_and_source_n(
    cwd: Path, source_dataset: str | None,
) -> tuple[sanitizer.SDCConfig, int | None, float]:
    """Load the dataset's SDC config and count its rows once per call.

    Returns ``(sdc_cfg, source_n, audit_seconds)``. ``source_n`` is
    used downstream by ``_check_row_count`` to flag silent filtering
    (NA-drops, subset conditions) and is computed once outside the
    sanitize loop — the file doesn't change between iterations, and
    the previous behaviour (re-reading the dataset every iteration)
    cost ~60 s per loop pass on a 3 GB ``.dta``. Policy-load failures
    are non-fatal: SDC degrades to ``DEFAULT_CONFIG``.
    """
    import time as _time
    from dataclasses import replace

    sdc_cfg = sanitizer.DEFAULT_CONFIG
    if source_dataset:
        try:
            policy_obj = policy_module.load_policy(cwd)
            # Policy keys are dataset basenames (see policy.py:134-137:
            # "datasets keys are dataset filenames (not full paths)").
            # ``get_schema`` honours this contract by passing
            # ``path.name``; here we must do the same — otherwise the
            # model passing ``./data.csv`` or ``sub/data.csv`` as
            # ``source_dataset`` silently misses a policy entry keyed
            # ``data.csv`` and a researcher's explicit
            # ``non_disclosive_variables`` opt-in is ignored. Path
            # resolution failures fall back to the raw string; the
            # lookup will just miss, which is no worse than the prior
            # behaviour.
            try:
                dataset_key = resolve_in_cwd(source_dataset).name
            except PathEscapeError:
                dataset_key = source_dataset
            non_disclosive = policy_module.non_disclosive_for(
                policy_obj, dataset_key,
            )
        except Exception:  # noqa: BLE001 — policy load must never block sanitization
            non_disclosive = frozenset()
        if non_disclosive:
            sdc_cfg = replace(
                sanitizer.DEFAULT_CONFIG,
                non_disclosive_variables=non_disclosive,
            )

    audit_t0 = _time.monotonic()
    source_n = _resolve_source_row_count(source_dataset or None)
    audit_seconds = _time.monotonic() - audit_t0
    return sdc_cfg, source_n, audit_seconds


def _sanitize_and_store_payloads(
    raw_payloads: list[Any],
    *,
    cwd: Path,
    label: str,
    language: str,
    code: str,
    source_dataset: str | None,
    source_n: int | None,
    sdc_cfg: sanitizer.SDCConfig,
    run_dir: Any,
    script_run_id: str,
    store: Any,
) -> tuple[list[dict[str, Any]], bool, float, float]:
    """Run sanitize + store for each emitted payload.

    Returns ``(results, any_ok, sanitize_seconds, store_seconds)``.
    ``results`` carries one entry per raw payload, with two shapes:
    successful entries (``status="ok"``) include the inline compact
    payload, the markdown table, and the per-result transformations;
    rejected entries (``status="rejected_by_sanitizer"``) carry the
    rejection reason and the diagnostic-row id. Both shapes are
    stored — even rejections keep an audit trail tagged with
    ``script_run_id``.
    """
    import time as _time

    results: list[dict[str, Any]] = []
    any_ok = False
    sanitize_seconds = 0.0
    store_seconds = 0.0

    from nora.text_safety import safe_text

    for raw_payload in raw_payloads:
        # Prefer the per-helper label (each ``nora_result_*`` takes
        # its own ``label("...")`` argument and embeds it in the
        # payload). Fall back to the script-level label when a helper
        # didn't pass one. The label is data-origin text — a script
        # can compute it from raw dataset values
        # (``label=f"income={df.income.iloc[0]}"``) — so it MUST go
        # through the same boundary check as the rest of the payload
        # before being persisted on the row or echoed back. ``label``
        # (the fallback) is already sanitized at the submit_script
        # entry; ``safe_text`` here covers the per-helper path.
        raw_helper_label = (
            raw_payload.get("label")
            if isinstance(raw_payload, dict) else None
        )
        helper_label = safe_text(raw_helper_label) if raw_helper_label else ""
        if not helper_label:
            helper_label = label

        s0 = _time.monotonic()
        sanitized = sanitize(raw_payload, sdc_cfg)
        sanitize_seconds += _time.monotonic() - s0
        if not sanitized.ok:
            # SDC bounced this payload. Still store it so the researcher
            # can audit, and surface the rejection inline so the model
            # sees which one failed without losing the others.
            i0 = _time.monotonic()
            diag_row = store.insert(
                label=f"[rejected] {helper_label}",
                analysis_type=sanitized.analysis_type or "unknown",
                sanitized_payload={
                    "type": "sanitizer_rejection",
                    "reason": sanitized.rejection_reason,
                    "analysis_type": sanitized.analysis_type,
                },
                language=language,
                script_code=code,
                transformations=[],
                raw_log_path=run_dir,
                script_run_id=script_run_id,
            )
            store_seconds += _time.monotonic() - i0
            results.append({
                "status": "rejected_by_sanitizer",
                "result_id": diag_row.id,
                "label": diag_row.label,
                "analysis_type": sanitized.analysis_type,
                "reason": sanitized.rejection_reason,
            })
            continue

        # Row-count check runs AFTER sanitize (operates on sanitized
        # structure; row counts themselves aren't disclosive). Uses
        # the ``source_n`` resolved once by the caller so the
        # per-payload loop never re-reads the dataset.
        row_count_msg = _check_row_count(
            sanitized.sanitized or {}, source_dataset or None, source_n,
        )
        transformations = list(sanitized.transformations)
        if row_count_msg:
            transformations.append(row_count_msg)

        i0 = _time.monotonic()
        row = store.insert(
            label=helper_label,
            analysis_type=sanitized.analysis_type or "unknown",
            sanitized_payload=sanitized.sanitized or {},
            language=language,
            script_code=code,
            transformations=transformations,
            raw_log_path=run_dir,
            script_run_id=script_run_id,
        )
        store_seconds += _time.monotonic() - i0
        any_ok = True
        # Render the canonical markdown table once per result and
        # ship it inline. Same source as
        # ``expand_result(view="markdown")``. The model can drop the
        # table directly into a reply rather than re-deriving column
        # choice and number precision per call; the UI can render it
        # on the tool-result card without going through the model.
        # Falls back to None when the renderer doesn't recognise the
        # type — callers fall back to the JSON payload.
        try:
            from nora.result_render import render_table
            md_table = render_table(sanitized.sanitized or {})
        except Exception:  # noqa: BLE001 — formatting must never block storage
            md_table = None
        result_entry: dict[str, Any] = {
            "status": "ok",
            "result_id": row.id,
            "label": row.label,
            "analysis_type": row.analysis_type,
            "summary": _summarize(sanitized.sanitized or {}),
            "transformations": transformations,
            # Inline compact payload — same trim as
            # ``expand_result(view="coefficients")``. Lets the model
            # render coefficient tables from the submit_script
            # response directly, instead of N separate
            # ``expand_result`` round-trips on a parameterized
            # batch. Full ``vcov`` / ``vif`` is still reachable via
            # ``expand_result(view="full")`` when collinearity
            # diagnostics matter.
            "payload": _compact_payload(sanitized.sanitized or {}),
        }
        if md_table is not None:
            result_entry["markdown"] = md_table
        results.append(result_entry)

    return results, any_ok, sanitize_seconds, store_seconds


def _build_response_envelope(
    *,
    overall_status: str,
    script_run_id: str,
    results: list[dict[str, Any]],
    exec_result: Any,
    language: str,
    sanitize_seconds: float,
    store_seconds: float,
    row_count_audit_seconds: float,
) -> dict[str, Any]:
    """Assemble the base response envelope (pre status-specific fields).

    Three things happen here that all mutate the visible response:
    transformation hoisting (dedupe shared SDC notes across multi-
    result responses into one envelope-level field), inline-payload
    trimming (two-stage cap so a wide multi-spec response doesn't
    blow the tool-result size budget), and phase timings (subprocess
    vs post-execution audit work, kept under ``_phase_timings`` so
    the model can read or ignore it). Returned envelope still needs
    ``_attach_status_metadata`` for hint / debug_excerpt / plots
    before going on the wire.
    """
    # Dedupe transformations that repeat across multi-result responses.
    # A 24-spec script typically generates 24 identical SDC entries
    # ("clamped coefficient SEs to 3 sig figs at N=…"); hoisting the
    # shared set into one envelope-level field saves a lot of context
    # while preserving audit transparency: per-result entries that
    # actually differ (e.g. row-count messages with N specific to that
    # spec) stay where they are. The store keeps the un-deduped lists
    # per row regardless, so ``expand_result`` still surfaces every
    # transformation a row received.
    shared_transformations = _shared_transformations(results)
    if shared_transformations:
        shared_set = set(shared_transformations)
        for entry in results:
            if entry.get("status") != "ok":
                continue
            entry["transformations"] = [
                t for t in entry.get("transformations", [])
                if t not in shared_set
            ]

    # Envelope-size guard — see ``_trim_oversize_inline_payloads``.
    # Two-stage now: payload-strip at the first threshold, markdown-
    # summarize at the second. The flags propagate into the envelope
    # so the model knows whether to reach for ``expand_result``.
    trim_flags = _trim_oversize_inline_payloads(results)
    inline_payload_omitted = trim_flags["payload_omitted"]
    inline_markdown_omitted = trim_flags["markdown_omitted"]

    # Phase timings: subprocess vs post-execution audit work. The
    # default ``duration_seconds`` only reports the subprocess, which
    # used to hide a real bug — the post-execution row-count audit
    # was re-reading a 3 GB .dta on every iteration of the multi-
    # result loop, costing 20+ minutes after Stata had already
    # finished in seconds. ``_phase_timings`` makes that visible
    # without bloating the default response.
    phase_timings = {
        "executor_seconds": round(exec_result.duration_seconds, 3),
        "row_count_audit_seconds": round(row_count_audit_seconds, 3),
        "sanitize_seconds": round(sanitize_seconds, 3),
        "store_seconds": round(store_seconds, 3),
    }

    response: dict[str, Any] = {
        "status": overall_status,
        "script_run_id": script_run_id,
        "results": results,
        "duration_seconds": round(exec_result.duration_seconds, 3),
        "_phase_timings": phase_timings,
        # Path for the TUI to read raw R/Stata output from. Claude
        # seeing the path is not a leak (directory names are
        # structural, not data), but Claude should not try to read
        # files from it — there are no tools for that.
        "_run_dir": str(exec_result.run_dir),
        # Language the script was written in. The web UI uses this
        # to label the "Open in R / Stata" button and pick the right
        # invocation when launching the native app.
        "_language": language,
    }
    # Surface the script's process-level exit code on EVERY envelope
    # where the subprocess actually ran. Earlier code only set
    # ``exit_code`` on the failure branches (execution_failed /
    # execution_failed_partial), which left the success path silently
    # without an exit code — useful information for the model
    # (confirms the script exited cleanly, not killed by sandbox /
    # timeout) and required by tests that assert
    # ``body["exit_code"] == 0`` on the clean-exit-with-warnings
    # path. ``exit_code`` is None when the executor short-circuited
    # before running the subprocess at all (sandbox preflight
    # failure, missing interpreter); we omit the field then so the
    # model isn't misled by a fake zero.
    if exec_result.exit_code is not None:
        response["exit_code"] = exec_result.exit_code
    # Structured runtime-environment snapshot from the executor.
    # Surfaced on every response (success and failure) so the model
    # can self-diagnose environment-shaped failures without
    # speculating: which interpreter Nora picked, which required
    # packages it provided, whether the sandbox-health probe
    # rejected any python3 candidates, etc. Phase-safe by
    # construction — none of these fields read researcher data.
    env_metadata = getattr(exec_result, "environment", None)
    if env_metadata:
        response["_environment"] = env_metadata
    if shared_transformations:
        response["transformations_summary"] = shared_transformations
    if inline_payload_omitted:
        response["_inline_payload_omitted"] = True
    if inline_markdown_omitted:
        response["_inline_markdown_omitted"] = True
    # Non-fatal advisories from the executor (currently: malformed
    # JSONL lines that were skipped while other lines parsed cleanly).
    # Surfacing these on a status="ok" response lets the model see "the
    # run succeeded, but spec #5's helper emitted a bogus line" without
    # demoting the whole run to execution_failed_partial.
    exec_warnings = getattr(exec_result, "warnings", None) or []
    if exec_warnings:
        response["warnings"] = list(exec_warnings)
    return response


def _attach_status_metadata(
    response: dict[str, Any],
    *,
    overall_status: str,
    exec_result: Any,
    language: str,
    label: str,
    code: str,
    script_run_id: str,
    results: list[dict[str, Any]],
    store: Any,
) -> None:
    """Attach status-specific fields to ``response`` in place.

    Three branches: ``rejected_by_sanitizer`` gets a fix-up hint;
    ``execution_failed`` and ``execution_failed_partial`` get a
    debug excerpt + reason + exit code, plus a status-specific
    hint; the bare ``execution_failed`` branch additionally inserts
    a diagnostic row tagged with the same ``script_run_id`` so the
    researcher's audit path always finds the run dir from the store
    even when ``results`` carries only rejection rows. Plot-helper
    summary is attached last regardless of status — plots produced
    on a partial-success run are still useful.
    """
    if overall_status == "rejected_by_sanitizer":
        response["hint"] = (
            "Every payload this script emitted was rejected by the "
            "disclosure-control layer. Inspect each result's reason "
            "(e.g., n too small, forbidden field, type mismatch) and "
            "resubmit a corrected analysis."
        )

    if overall_status in ("execution_failed", "execution_failed_partial"):
        # Add the script-level error context: reason and a bounded
        # debug_excerpt of stdout/stderr so the model can diagnose
        # the abort. ``exit_code`` is already on the envelope from
        # ``_build_response_envelope`` for every run that reached
        # the subprocess; setting it again here would be redundant.
        # The partial-success branch carries reason+excerpt
        # ALONGSIDE the partial results; the bare-failure branch
        # carries it alone (or alongside rejection-only rows).
        response["reason"] = exec_result.error
        from nora.error_summary import extract_debug_excerpt
        excerpt = extract_debug_excerpt(
            exec_result.raw_stdout,
            exec_result.raw_stderr,
            exec_result.exit_code,
            language,
            run_dir=exec_result.run_dir,
            pre_user_stderr=getattr(exec_result, "pre_user_stderr", None),
            user_stderr=getattr(exec_result, "user_stderr", None),
        )
        if not excerpt:
            excerpt = (
                f"script failed (exit code {exec_result.exit_code}); "
                f"inspect raw log in UI"
            )
        # Localize the crash inside a labelled multi-spec run by
        # prepending the most-recent ok payload's label. The
        # extractor's idiom names the failing operation but not WHICH
        # iteration of a loop produced it — without this, "object 'x'
        # not found" inside a 12-spec sweep leaves the model guessing
        # which spec hit the missing column. Skip the prefix when no
        # ok payload landed (no localization to add) or when the
        # payload's label matches the script-level label (same info,
        # no value).
        last_ok_label: str | None = None
        for r in results:
            if r.get("status") == "ok":
                lbl = r.get("label") or ""
                if lbl and lbl != label:
                    last_ok_label = lbl
        if last_ok_label:
            excerpt = f"[crashed after spec: {last_ok_label}]\n{excerpt}"
        response["debug_excerpt"] = excerpt

        if overall_status == "execution_failed":
            # No payloads survived sanitization. Persist a diagnostic
            # row tagged with the same script_run_id so the
            # researcher's audit path always finds the run dir from
            # the store, even when results carries only rejections.
            diag_row = store.insert(
                label=f"[error] {label}",
                analysis_type="script_error",
                sanitized_payload={
                    "type": "script_error",
                    "reason": exec_result.error,
                    "exit_code": exec_result.exit_code,
                    "run_dir": str(exec_result.run_dir),
                },
                language=language,
                script_code=code,
                transformations=[],
                raw_log_path=exec_result.run_dir,
                script_run_id=script_run_id,
            )
            response["result_id"] = diag_row.id
            # Hint depends on whether the script emitted anything at
            # all. With rejection rows present, the model needs to
            # read both the per-result rejection reasons AND the
            # abort cause — these are independent failure modes.
            if results:
                response["hint"] = (
                    f"{len(results)} payload(s) reached the sanitizer "
                    f"but every one was rejected (see per-result "
                    f"reasons). The script also aborted afterward "
                    f"(see debug_excerpt). Both failures are "
                    f"independent and both need addressing on "
                    f"resubmit."
                )
            else:
                response["hint"] = (
                    "The full raw stdout/stderr stays in the run "
                    "directory for the researcher (no model-side "
                    "file-read tool). The debug_excerpt above is a "
                    "short slice of the language's error output. "
                    "Read it before resubmitting."
                )
        else:
            # Partial success: at least one payload reached the model.
            # Rejection rows may also be present; the count below is
            # ``any_ok`` payloads only, not the full results length.
            ok_count = sum(1 for r in results if r.get("status") == "ok")
            response["hint"] = (
                f"{ok_count} payload(s) reached you cleanly before "
                "the script aborted. Read them as you would any "
                "other result; the abort cause is in debug_excerpt. "
                "If the abort was a known data condition (thin cell "
                "at one spec, missing variable on one outcome), "
                "guard that case and re-emit only the missing "
                "payloads in a follow-up."
            )

    plot_summary = _summarize_plot_helpers(exec_result.run_dir)
    if plot_summary is not None:
        response["plots"] = plot_summary


def _with_zero_phase_metadata(
    response: dict[str, Any],
    *,
    language: str = "",
) -> dict[str, Any]:
    """Stamp the envelope-shape monitoring fields onto an early-error
    response so callers see the same shape regardless of whether the
    submit_script call reached the executor or not.

    Without this, monitoring code that introspects ``_phase_timings``
    has to handle "field absent" alongside "field present with zero
    values" — two shapes for "nothing happened in this phase."
    Stamping zeros makes the shape always-present, with the values
    making it obvious that no real work ran (every phase clocks at
    0.000 seconds).

    Caller doesn't need to remove existing keys: when the early
    payload already populates one of these (e.g. a cancellation
    path that captured a non-zero ``duration_seconds`` for the
    process that did launch), we leave it. ``setdefault`` protects
    that.
    """
    response.setdefault("duration_seconds", 0.0)
    response.setdefault("_phase_timings", {
        "executor_seconds": 0.0,
        "row_count_audit_seconds": 0.0,
        "sanitize_seconds": 0.0,
        "store_seconds": 0.0,
    })
    if language:
        response.setdefault("_language", language)
    return response


@tool("submit_script")
async def submit_script(args: dict[str, Any]) -> dict[str, Any]:
    """Run an R / Stata / Python script end-to-end: execute → sanitize → store.

    The researcher's raw stdout / stderr is captured and stashed on the
    stored row (available via ``expand_result``). Claude only ever sees
    the sanitizer's output — never the raw log.

    Pipeline (each phase in its own helper above):
      1. ``_execute_script_for_submit`` — run the subprocess with
         per-turn cancellation handling.
      2. ``_resolve_sdc_and_source_n`` — load the dataset's SDC
         config and count its rows once per call.
      3. ``_sanitize_and_store_payloads`` — sanitize and persist each
         emitted payload, recording rejections alongside successes.
      4. ``_build_response_envelope`` — assemble the base response
         (status decision, transformation hoisting, size trimming,
         phase timings).
      5. ``_attach_status_metadata`` — attach hint / debug_excerpt /
         diagnostic row / plot summary.

    Behaviour is identical to the prior single-function implementation;
    extracting these phases makes each one independently readable and
    testable.
    """
    language = args.get("language", "")
    code = args.get("code", "")
    # Sanitize the script-level label here, once. Same threat model as
    # the per-helper label below (data-derived strings as injection
    # vectors / SDC bypass) — without this, a label like
    # f"income={df.income.iloc[0]}" leaks raw values through label.txt
    # and the response envelope even when the sanitized payload is
    # compliant or rejected.
    from nora.text_safety import safe_text
    raw_label = args.get("label") or "(unlabeled)"
    label = safe_text(raw_label) or "(unlabeled)"
    source_dataset = args.get("source_dataset", "") or ""

    if language not in {"R", "Stata", "Python"}:
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": (
                f"unsupported language: {language!r}. Nora runs R "
                f"(via Rscript), Stata, and Python (3.x with pandas)."
            ),
        }, language=language))
    if not code.strip():
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": "code argument is empty",
        }, language=language))

    cwd = get_cwd()

    # 1. Execute. Cancellation surfaces here as ``CancelledError``,
    # which propagates through the SDK's tool-dispatch path (the MCP
    # server wrapper catches ``Exception`` only) and lands in the
    # runner's outer cancel branch where session teardown happens.
    # No ``ToolCallResult`` ever gets yielded for a cancelled run.
    exec_result = await _execute_script_for_submit(language, code, cwd)

    # 2a. Persist the script-level label to the run dir so the Files
    # panel can name the SCRIPT after what the model called the whole
    # invocation, not after the first per-helper label that happens
    # to land in the store. For a 20-regression script labeled
    # "reg_v16: H1/H2/H3, Path A and Path B", the per-helper rows
    # carry per-cell names like "Path A H1: operating_margin" — fine
    # for individual result lookup, wrong as the file name. Writing
    # the umbrella here keeps the panel pointed at the script's
    # purpose. Best-effort: a write failure leaves the panel falling
    # back to the per-helper-label path it used before this change.
    try:
        if exec_result.run_dir is not None:
            (exec_result.run_dir / "label.txt").write_text(
                label, encoding="utf-8",
            )
    except OSError:
        pass

    # 2. SDC config + source-dataset row count, both resolved once.
    sdc_cfg, source_n, row_count_audit_seconds = _resolve_sdc_and_source_n(
        cwd, source_dataset or None,
    )

    # One id per submit_script call; every row produced is tagged with
    # it so an audit can pull them together. ``run-`` prefix (not
    # ``R-``) avoids the misread as the R language in Stata / Python
    # sessions.
    script_run_id = "run-" + secrets.token_hex(4)

    # 3. Sanitize + store every emitted payload (rejections kept).
    store = get_store(cwd)
    results, any_ok, sanitize_seconds, store_seconds = (
        _sanitize_and_store_payloads(
            exec_result.result_payloads,
            cwd=cwd,
            label=label,
            language=language,
            code=code,
            source_dataset=source_dataset or None,
            source_n=source_n,
            sdc_cfg=sdc_cfg,
            run_dir=exec_result.run_dir,
            script_run_id=script_run_id,
            store=store,
        )
    )

    # Envelope-status decision. Five outcomes; the "all-rejected then
    # aborted" case is the one that's easy to get wrong:
    #
    #   exec ok | raw payloads | any_ok | envelope status
    #   --------+--------------+--------+-----------------------------
    #     yes   |    any       |  yes   | "ok"
    #     yes   |    any       |  no    | "rejected_by_sanitizer"
    #     no    |    any       |  yes   | "execution_failed_partial"
    #     no    |    any       |  no    | "execution_failed"  (rejection rows visible in results)
    #     no    |    none      |   -    | "execution_failed"  (no rows, diag row only)
    #
    # "execution_failed_partial" is reserved for partial SUCCESS: at
    # least one payload made it through SDC despite the abort. When
    # every emitted payload was rejected AND the script also aborted,
    # status is "execution_failed" — rejection rows still appear in
    # ``results`` alongside the abort context.
    if exec_result.ok:
        overall_status = "ok" if any_ok else "rejected_by_sanitizer"
    else:
        overall_status = (
            "execution_failed_partial" if any_ok else "execution_failed"
        )

    # 4. Build the base envelope (transformations hoist, size trim,
    # phase timings, base response dict).
    response = _build_response_envelope(
        overall_status=overall_status,
        script_run_id=script_run_id,
        results=results,
        exec_result=exec_result,
        language=language,
        sanitize_seconds=sanitize_seconds,
        store_seconds=store_seconds,
        row_count_audit_seconds=row_count_audit_seconds,
    )

    # 5. Attach status-specific fields (hint, debug excerpt, diagnostic
    # row for bare-failure, plot summary).
    _attach_status_metadata(
        response,
        overall_status=overall_status,
        exec_result=exec_result,
        language=language,
        label=label,
        code=code,
        script_run_id=script_run_id,
        results=results,
        store=store,
    )
    return _as_mcp_text(response)


# ---------------------------------------------------------------------------
# Tool: submit_script_file
# ---------------------------------------------------------------------------

# Mapping from file suffix to ``submit_script``'s expected language
# string. Kept narrow so a researcher who attaches an unrelated text
# file (.txt, .md) gets a clear refusal instead of an ambiguous run
# attempt.
_SCRIPT_FILE_LANGUAGES: dict[str, str] = {
    ".do": "Stata",
    ".r": "R",
    ".rmd": "R",
    ".py": "Python",
}


@tool("submit_script_file")
async def submit_script_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read a script from cwd by basename and forward to submit_script.

    Path safety mirrors ``read_attached_file``: the ``name`` argument
    is treated as a basename (any directory component is stripped),
    resolved against the session cwd, and refused if it escapes. The
    extension allowlist (``_SCRIPT_FILE_LANGUAGES``) bounds what gets
    treated as a runnable script.
    """
    raw_name = args.get("name", "")
    if not raw_name or not isinstance(raw_name, str):
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": "name argument is required (basename of a script file)",
        }))
    safe_name = Path(raw_name).name
    if not safe_name:
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": f"could not parse a basename from {raw_name!r}",
        }))

    target: Path | None
    try:
        target = resolve_in_cwd(safe_name)
        if not target.is_file():
            target = None
    except (PathEscapeError, OSError):
        target = None

    # Run-dir scripts: ``list_session_files`` advertises Nora-written
    # ``script.{do,R,py}`` files under label-derived display names
    # (e.g. "Linear Regression Run.do"). Without this fallback, the
    # purpose-built run-this-file tool would refuse them with
    # ``not_found``, and the model had to round-trip the bytes
    # through ``read_attached_file`` + ``submit_script`` — defeating
    # the feature and failing on over-cap scripts. Mirrors the
    # third fallback ``read_attached_file`` already does.
    if target is None:
        try:
            from nora.run_files import find_run_dir_script_by_name
            from nora.session_files import visible_run_dir_names
            cwd = get_cwd()
            candidate = find_run_dir_script_by_name(
                cwd, safe_name,
                visible_run_dirs=visible_run_dir_names(cwd),
            )
        except Exception:  # noqa: BLE001
            candidate = None
        if candidate is not None and candidate.is_file():
            target = candidate

    if target is None:
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "not_found",
            "reason": f"no script named {safe_name!r} in this session",
        }))

    # SDC provenance gate. Same rationale as ``read_attached_file``:
    # the script sandbox lets a run write to cwd, and a script that
    # wrote raw rows into a ``.R`` / ``.do`` / ``.py`` file would
    # otherwise be re-runnable through here, with the file's bytes
    # crossing into the executor's run-dir staging copy AND the
    # store's ``script_code`` column verbatim. ``<cwd>/.nora/runs/``
    # is exempt — Nora-written wrapper scripts are safe by
    # construction (and ``find_run_dir_script_by_name`` above is the
    # only way ``submit_script_file`` reaches them).
    try:
        cwd_for_check = get_cwd()
        resolved_target = target.resolve()
        cwd_resolved = cwd_for_check.resolve()
    except OSError:
        cwd_for_check = get_cwd()
        resolved_target = target
        cwd_resolved = cwd_for_check
    nora_subdir = (cwd_resolved / ".nora").resolve() if cwd_resolved.exists() else cwd_resolved / ".nora"
    is_under_nora = False
    try:
        is_under_nora = resolved_target.is_relative_to(nora_subdir)
    except (ValueError, OSError):
        is_under_nora = False
    ext = target.suffix.lower()
    inferred_language = _SCRIPT_FILE_LANGUAGES.get(ext)
    # Provenance only matters for files this tool would actually
    # run. Non-script extensions are rejected below by the
    # extension check with a more useful error than "not staged";
    # checking provenance first would mask that message and force
    # the user through a re-staging dance for a file that wouldn't
    # have been runnable either way.
    if not is_under_nora and inferred_language is not None:
        # Fail CLOSED: a manifest read that raises (corrupt JSON,
        # permission-blocked path, FS error) used to flip
        # ``staged_ok`` to True and let the call through. That
        # behavior turns the safety gate into a no-op exactly where
        # raw file bytes can cross into execution — the failure mode
        # most likely to be exploited via deliberate manifest
        # corruption. Treat manifest-unreadable as "not staged" and
        # tell the caller to re-stage.
        try:
            from nora.file_provenance import is_known, known_names
            staged_ok = is_known(cwd_for_check, resolved_target.name)
            name_in_manifest = (
                resolved_target.name in known_names(cwd_for_check)
            ) if not staged_ok else False
        except Exception:  # noqa: BLE001 — fail closed (see above)
            staged_ok = False
            name_in_manifest = False
        if not staged_ok:
            if name_in_manifest:
                reason = (
                    f"{safe_name!r} appears in this session's staged-"
                    f"files manifest, but its current on-disk content "
                    f"does not match what was staged. A script may "
                    f"have rewritten it (the sandbox permits writes "
                    f"to your cwd) or you edited it outside Nora. "
                    f"Re-attach it via the chat composer to authorise "
                    f"the new content, or paste its contents inline "
                    f"with submit_script."
                )
            else:
                reason = (
                    f"{safe_name!r} is not in this session's staged-"
                    f"files manifest, so I cannot run it. The "
                    f"analysis sandbox intentionally lets scripts "
                    f"write to your cwd; a script-shaped file that "
                    f"appeared via that path may carry data the "
                    f"researcher never authorised me to re-execute. "
                    f"Re-attach it via the chat composer (drop or "
                    f"paste it into the message box) to mark it as "
                    f"researcher-staged, or paste its contents "
                    f"inline with submit_script."
                )
            return _as_mcp_text(_with_zero_phase_metadata({
                "status": "rejected",
                "reason": reason,
            }, language=None))
    if inferred_language is None:
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": (
                f"{safe_name!r} is not a recognised script file. "
                f"Supported extensions: "
                f"{sorted(_SCRIPT_FILE_LANGUAGES.keys())}"
            ),
        }))

    explicit_language = (args.get("language") or "").strip()
    language = explicit_language or inferred_language
    if explicit_language and explicit_language != inferred_language:
        # The model overrode the extension-based inference.
        # Two ways this goes wrong:
        #   1. Override isn't a language we support at all → reject
        #      with the same shape the downstream submit_script would.
        #   2. Override IS supported, but doesn't match the file
        #      extension. Earlier behavior silently honored the
        #      override, so a researcher who attached ``script.do``
        #      and got ``language="Python"`` from the model would
        #      hand a Stata-syntax script to the Python interpreter.
        #      Reject loudly instead — the model can either drop
        #      the override (and use the extension-inferred language)
        #      or rename the file. Silent mis-routing is the worst
        #      failure mode.
        if explicit_language not in {"R", "Stata", "Python"}:
            return _as_mcp_text(_with_zero_phase_metadata({
                "status": "error",
                "reason": (
                    f"language must be one of R / Stata / Python; "
                    f"got {explicit_language!r}"
                ),
            }, language=explicit_language))
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": (
                f"language override {explicit_language!r} conflicts "
                f"with the file extension {ext!r} (which infers "
                f"{inferred_language!r}). Drop the language argument "
                f"to use the extension-inferred language, or rename "
                f"the file to match the language you want to run."
            ),
        }, language=explicit_language))

    try:
        code = target.read_text(encoding="utf-8")
    except OSError as e:
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": f"could not read {safe_name}: {e}",
        }, language=language))
    except UnicodeDecodeError:
        # Fall back to replace-mode so a stray non-UTF-8 byte doesn't
        # block the run; the script is the researcher's, they can fix
        # it if encoding matters.
        code = target.read_bytes().decode("utf-8", errors="replace")

    if not code.strip():
        return _as_mcp_text(_with_zero_phase_metadata({
            "status": "error",
            "reason": f"{safe_name!r} is empty",
        }, language=language))

    return await submit_script.handler({
        "language": language,
        "code": code,
        "label": args.get("label") or safe_name,
        "source_dataset": args.get("source_dataset") or "",
    })


# ---------------------------------------------------------------------------
# Tool: expand_result
# ---------------------------------------------------------------------------

# Env-gated opt-in for cross-session result recall. Default OFF —
# matches the historic per-session isolation that researcher mental
# models depend on. Setting ``NORA_ALLOW_CROSS_SESSION_RECALL=1``
# lets ``expand_result(result_id, session_path=...)`` and the
# ``list_results_global`` tool reach into other sessions' stores.
# Stored payloads are pre-sanitized so the privacy boundary is
# preserved; the gate exists because researchers may want explicit
# project separation regardless of payload safety.
_CROSS_SESSION_ENV_VAR = "NORA_ALLOW_CROSS_SESSION_RECALL"


def _cross_session_enabled() -> bool:
    """Whether the env-gated cross-session lookup is on. Truthy
    values: ``1`` / ``true`` / ``yes`` (case-insensitive)."""
    val = os.environ.get(_CROSS_SESSION_ENV_VAR, "").strip().lower()
    return val in ("1", "true", "yes")


def _resolve_cross_session_cwd(session_path: str) -> Path | None:
    """Validate a researcher-supplied session_path and return its
    resolved Path, or None if it isn't a concrete Nora session
    the researcher has opened.

    Two acceptable shapes:

      1. Direct child of SESSIONS_ROOT — staged sessions created
         through Nora's own session-open path. The narrow "direct
         child" rule (not the root itself, not an arbitrary
         descendant) avoids the failure mode where the caller
         points the recall at a sub-path inside another session —
         ``get_store(target_cwd)`` would then *create* a
         ``.nora/results.db`` inside that sub-path, turning the
         read-side recall path into an arbitrary-directory write
         under the sessions root.

      2. A path registered via ``external_sessions`` — folder-
         backed sessions opened through ``choose_folder``. The
         registry only contains paths the researcher explicitly
         opened via the picker, so accepting them here mirrors
         the trust the rest of the UI already extends. Without
         this branch, every folder-backed session is silently
         excluded from cross-session recall even though
         ``list_sessions`` surfaces it and the model can read its
         current ``list_results`` while the session is active.

    Same gate matches the discipline in ``ui.switch_session`` /
    ``ui.delete_session`` (which also accept either shape).
    """
    from nora.ui import SESSIONS_ROOT
    from nora import external_sessions

    try:
        target = Path(session_path).expanduser().resolve()
    except OSError:
        return None
    sessions_root = SESSIONS_ROOT.resolve()
    if target == sessions_root:
        return None
    if not target.is_dir():
        return None
    # Staged-session path: direct child of SESSIONS_ROOT.
    if target.parent == sessions_root:
        return target
    # Folder-backed path: registered via choose_folder. The registry
    # is researcher-curated state (picker writes only), so trusting
    # it here is the same trust the active-session UI already extends.
    try:
        if external_sessions.is_registered(sessions_root, target):
            return target
    except Exception:  # noqa: BLE001 — registry is best-effort
        pass
    return None


@tool("expand_result")
async def expand_result(args: dict[str, Any]) -> dict[str, Any]:
    """Return the full stored sanitized payload for a given ID.

    Defaults to the current session's store. With ``session_path``
    set and the cross-session env gate on, looks up in another
    session's store under ``~/.nora-sessions/``. The optional
    ``view`` argument trims the payload to a regression-coefficient
    slice when set to ``"coefficients"``.
    """
    result_id = args.get("result_id", "")
    if not result_id:
        return _as_mcp_text({
            "status": "error",
            "reason": "result_id argument is required",
        })
    view = (args.get("view") or "").strip().lower()
    if view not in ("", "full", "coefficients", "markdown"):
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"view must be '' / 'full' / 'coefficients' / "
                f"'markdown', got {view!r}"
            ),
        })
    raw_session_path = (args.get("session_path") or "").strip()
    if raw_session_path:
        if not _cross_session_enabled():
            return _as_mcp_text({
                "status": "denied",
                "reason": (
                    f"cross-session expand is disabled in this "
                    f"configuration. Set {_CROSS_SESSION_ENV_VAR}=1 "
                    f"in the environment to enable, or omit "
                    f"session_path to look up in the current session."
                ),
            })
        target_cwd = _resolve_cross_session_cwd(raw_session_path)
        if target_cwd is None:
            return _as_mcp_text({
                "status": "denied",
                "reason": (
                    "session_path must be either a directory inside "
                    "~/.nora-sessions/ or a folder-backed session "
                    "previously opened via the 'Choose folder' picker"
                ),
            })
    else:
        target_cwd = get_cwd()
    store = get_store(target_cwd)
    row = store.get(result_id)
    if row is None:
        return _as_mcp_text({
            "status": "not_found",
            "reason": f"no stored result with id {result_id!r}",
        })
    payload = row.sanitized_payload
    view_dropped: list[str] = []
    view_ignored_for_type = False
    if view == "coefficients" and isinstance(payload, dict):
        # Trim the regression-collinearity diagnostics through the
        # shared ``_VIEW_COEFFICIENTS_DROP_FIELDS`` registry — the
        # same map ``_compact_payload`` consults for the inline
        # submit_script response, so adding a new diagnostic field
        # to the runtime only needs one entry in the registry to be
        # hidden from the compact view across both call sites.
        #
        # When the payload's analysis_type isn't in the registry,
        # the view is a no-op for that type (a t-test or descriptive
        # has no expensive diagnostics to trim). The previous code
        # silently fell through, leaving the response's
        # ``view="coefficients"`` field claiming a trim that didn't
        # happen. We now flag that with ``view_ignored_for_type``
        # so the model can either accept the full payload or
        # request ``view="full"`` explicitly.
        payload, view_dropped, applied = _apply_view_coefficients_trim(payload)
        if not applied:
            view_ignored_for_type = True

    # ``view="markdown"`` returns a canonical markdown pipe-table
    # rendered from the sanitized payload. Same source as the JSON
    # payload, but pre-formatted so the model can drop it into a
    # response without re-deriving columns / precision per-call (the
    # source of inconsistent renders across recalls). When the
    # markdown render succeeds we DROP the JSON ``payload`` from
    # the response — shipping both is double-cost (kilobytes for a
    # wide regression) and the model's only reason to call
    # ``view="markdown"`` is when it wants the rendered table, not
    # the raw arrays. If the payload type isn't one the renderer
    # knows, ``markdown`` is omitted and ``payload`` falls back in
    # so the call still has usable content.
    markdown: str | None = None
    if view == "markdown" and isinstance(payload, dict):
        from nora.result_render import render_table
        markdown = render_table(payload)

    response: dict[str, Any] = {
        "status": "ok",
        "result_id": row.id,
        "label": row.label,
        "analysis_type": row.analysis_type,
        "language": row.language,
        "transformations": row.transformations,
        "created_at": row.created_at,
    }
    if markdown is not None:
        response["markdown"] = markdown
    else:
        response["payload"] = payload
    if view:
        response["view"] = view
    if view_dropped:
        response["view_dropped_fields"] = view_dropped
    if view_ignored_for_type:
        response["view_ignored_for_type"] = True
    if raw_session_path:
        response["session_path"] = str(target_cwd)
    # Surface the run_dir so the TUI can re-render the raw R/Stata
    # output alongside the (possibly dense) sanitized payload. Without
    # this, re-expanding a stored regression gives the researcher
    # nothing but rows of coefficients / SEs / t-stats / p-values,
    # with no trace of the conventional R or Stata output they'd
    # recognize.
    if row.raw_log_path:
        response["_run_dir"] = row.raw_log_path
    if row.language:
        response["_language"] = row.language
    return _as_mcp_text(response)


# ---------------------------------------------------------------------------
# Tool: compose_results
# ---------------------------------------------------------------------------

# Hard caps on the layout spec. The model emits this object, and the
# whole point of compose_results is to keep the comparison table
# *out* of the conversation as raw numbers — instead the model names
# IDs, and we render the table from sanitized payloads. Without caps
# a malformed (or runaway) spec can produce a multi-MB markdown blob
# that defeats the context-economy goal of the feature.
#
# Picked from operator experience: research papers rarely report a
# composite table wider than ~12 model columns, and the longest
# academic regression-table run we've seen had ~40 outcome rows
# spread across ~6 groups. The caps below give 2× headroom on each
# axis and a strict total-rows budget that stops a Cartesian
# explosion (50 groups × 50 rows = 2,500 rendered rows). Spec
# rejection is loud — a structured "too_large" denial — so the model
# can split the call into pages instead of silently truncating.
_COMPOSE_MAX_COLUMNS = 25
_COMPOSE_MAX_GROUPS = 25
_COMPOSE_MAX_ROWS_PER_GROUP = 100
_COMPOSE_MAX_TOTAL_ROWS = 250
_COMPOSE_MAX_LABEL_LEN = 200


def _validate_compose_spec(spec: dict[str, Any]) -> str | None:
    """Pre-render hard-cap validator. Returns an error reason on
    over-size specs, or ``None`` when the spec passes.

    Doesn't validate shape correctness — ``compose_layout`` already
    rejects malformed specs with ``None``. This guard runs first so
    a 10,000-row spec gets rejected with bytes-saved rather than
    rendered into the model's context.
    """
    columns = spec.get("columns")
    groups = spec.get("groups")
    if isinstance(columns, list) and len(columns) > _COMPOSE_MAX_COLUMNS:
        return (
            f"spec.columns has {len(columns)} entries, over the "
            f"{_COMPOSE_MAX_COLUMNS}-column cap. A wider table won't "
            f"render usefully in the response — split into multiple "
            f"compose_results calls grouped by topic."
        )
    if isinstance(groups, list) and len(groups) > _COMPOSE_MAX_GROUPS:
        return (
            f"spec.groups has {len(groups)} entries, over the "
            f"{_COMPOSE_MAX_GROUPS}-group cap. Combine related groups "
            f"or paginate into multiple compose_results calls."
        )
    total_rows = 0
    if isinstance(groups, list):
        for idx, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            rows = group.get("rows")
            if not isinstance(rows, list):
                continue
            if len(rows) > _COMPOSE_MAX_ROWS_PER_GROUP:
                return (
                    f"spec.groups[{idx}].rows has {len(rows)} entries, "
                    f"over the {_COMPOSE_MAX_ROWS_PER_GROUP}-row "
                    f"per-group cap. Split this group into smaller "
                    f"groups."
                )
            total_rows += len(rows)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rlabel = row.get("label")
                if (isinstance(rlabel, str)
                        and len(rlabel) > _COMPOSE_MAX_LABEL_LEN):
                    return (
                        f"a row label is {len(rlabel)} chars, over "
                        f"the {_COMPOSE_MAX_LABEL_LEN}-char label cap. "
                        f"Shorten the label."
                    )
    if total_rows > _COMPOSE_MAX_TOTAL_ROWS:
        return (
            f"spec has {total_rows} total rows across all groups, "
            f"over the {_COMPOSE_MAX_TOTAL_ROWS}-row total cap. "
            f"Paginate into multiple compose_results calls."
        )
    if isinstance(columns, list):
        for c in columns:
            if not isinstance(c, dict):
                continue
            clabel = c.get("label")
            if (isinstance(clabel, str)
                    and len(clabel) > _COMPOSE_MAX_LABEL_LEN):
                return (
                    f"a column label is {len(clabel)} chars, over "
                    f"the {_COMPOSE_MAX_LABEL_LEN}-char label cap. "
                    f"Shorten the label."
                )
    title = spec.get("title")
    if (isinstance(title, str)
            and len(title) > _COMPOSE_MAX_LABEL_LEN):
        return (
            f"spec.title is {len(title)} chars, over the "
            f"{_COMPOSE_MAX_LABEL_LEN}-char label cap. Shorten "
            f"the title."
        )
    return None


@tool("compose_results")
async def compose_results(args: dict[str, Any]) -> dict[str, Any]:
    """Render a layout spec into a composite comparison table.

    Each row in ``spec.groups[*].rows`` may carry an optional
    ``session_path`` to look the row's ``result_id`` up in another
    session's store, mirroring the per-call ``session_path`` argument
    on ``expand_result``. Cross-session lookups are gated by the
    ``NORA_ALLOW_CROSS_SESSION_RECALL=1`` environment variable; when
    the flag is off, any row that supplies a ``session_path`` outside
    the current cwd is rejected and rendered as a missing-result em-
    dash (``—``). The SDC posture is identical to expand_result —
    payloads are pre-sanitized at write time, so cross-session reads
    don't leak unsanitized data — the env gate exists because some
    researchers want explicit project separation regardless.

    Cross-session rows are re-keyed before rendering. The renderer
    (``compose_layout``) keys payload cells by ``result_id`` alone,
    but every session's store numbers results from ``M1`` — so the
    canonical cross-session composition ("this run vs. the previous
    project's run") collides on the bare id almost by construction,
    and neither the researcher nor the model can avoid it: ids are
    store-assigned, and ``session_path`` is a lookup hint, not a
    render key. We therefore hand the renderer a deep copy of the
    spec in which each cross-session row's ``result_id`` is replaced
    by a call-private alias, with the label maps guaranteeing the
    alias never surfaces in the rendered table. All researcher-facing
    fields of the response (``missing_result_ids``,
    ``result_ids_referenced``, hints) keep the original ids.
    """
    spec = args.get("spec")
    if not isinstance(spec, dict):
        return _as_mcp_text({
            "status": "error",
            "reason": "spec argument is required and must be a JSON object",
        })

    # Hard-cap guard: reject over-size specs BEFORE walking groups or
    # touching the store. Without this gate, a multi-MB rendered table
    # defeats the context-economy goal of compose_results — the whole
    # point of the tool is that the model names IDs and we render
    # numbers, not that the model can stuff an arbitrary slab of
    # markdown into the response.
    too_large = _validate_compose_spec(spec)
    if too_large is not None:
        return _as_mcp_text({
            "status": "error",
            "reason": f"spec exceeds the layout caps: {too_large}",
        })

    cwd = get_cwd()
    cross_enabled = _cross_session_enabled()

    # Walk a DEEP COPY of the spec, resolving every row to a payload
    # and re-keying cross-session rows as we go. Two row shapes
    # accepted (mirroring ``compose_layout``): a bare result_id
    # string (minimal-friction form, store provides the label; can't
    # carry a session_path) OR an explicit dict with ``result_id``
    # and optional ``label`` / ``session_path`` overrides.
    #
    # Re-keying is what makes cross-session composition work at all:
    # the renderer looks payloads up by result_id alone, and every
    # session's store numbers from M1, so "compare this session's M1
    # with that session's M1" is the NORMAL cross-session shape, not
    # an edge case. Each row that names another session gets a call-
    # private alias (original rid + an unrenderable unit-separator
    # suffix) as its render key. The alias never reaches the
    # researcher: ``labels_by_id`` always carries an entry for it
    # (the stored label, or the original rid as fallback), so the
    # rendered row label stays human-readable even for a missing or
    # denied cross-session id. The deep copy is bounded by the
    # layout caps validated above.
    import copy
    spec_render = copy.deepcopy(spec)
    # (raw session_path, rid) → alias. Keyed on the raw string the
    # model typed (not the resolved path) so the rewrite below can
    # look aliases up without re-resolving; two spellings of the same
    # session simply mint two aliases that resolve to the same
    # payload, which renders identically.
    alias_by_key: dict[tuple[str, str], str] = {}
    # (resolved session dir, rid) → stored row or None. Cache so a
    # rid referenced by several rows (or via several session_path
    # spellings) costs one disk read, while every alias still gets
    # its payload — the previous shape skipped repeat lookups
    # entirely, which the per-alias keying can no longer afford.
    lookup_cache: dict[tuple[str, str], Any] = {}
    referenced: list[tuple[str, str | None]] = []
    payloads_by_id: dict[str, dict[str, Any]] = {}
    # Store-resolved helper-call label per RENDER key (bare rid for
    # local rows, alias for cross-session rows), used by
    # ``compose_layout`` to auto-label rows when the spec uses the
    # bare-string row shape. Keeps the model from having to re-type
    # labels it already named at script time.
    labels_by_id: dict[str, str] = {}
    missing: list[str] = []
    denied: list[str] = []

    groups = spec_render.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            rows = group.get("rows")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, str):
                    rid = row
                    sp: str | None = None
                    if not rid:
                        continue
                elif isinstance(row, dict):
                    rid_val = row.get("result_id")
                    if not isinstance(rid_val, str) or not rid_val:
                        continue
                    rid = rid_val
                    sp_val = row.get("session_path")
                    sp = sp_val if isinstance(sp_val, str) and sp_val else None
                else:
                    continue
                referenced.append((rid, sp))

                # Resolve the row's store. ``None`` target ⇒ the row
                # stays payload-less and renders as ``—``.
                if sp is None:
                    target_cwd: Path | None = cwd
                elif not cross_enabled:
                    denied.append(rid)
                    target_cwd = None
                else:
                    target_cwd = _resolve_cross_session_cwd(sp)
                    if target_cwd is None:
                        denied.append(rid)

                # Re-key any row that names a session other than the
                # current one — including denied / unresolvable rows,
                # so a foreign ``M1`` can never borrow the local
                # ``M1``'s numbers through the shared render key.
                if sp is not None and target_cwd != cwd:
                    akey = (sp, rid)
                    alias = alias_by_key.get(akey)
                    if alias is None:
                        alias = f"{rid}\x1f{len(alias_by_key)}"
                        alias_by_key[akey] = alias
                    row["result_id"] = alias  # dict row by construction
                    render_key = alias
                    # Fallback label first; the stored label below
                    # overrides it when the lookup succeeds.
                    labels_by_id.setdefault(alias, rid)
                else:
                    render_key = rid

                if target_cwd is None:
                    continue
                ckey = (str(target_cwd), rid)
                if ckey not in lookup_cache:
                    lookup_cache[ckey] = get_store(target_cwd).get(rid)
                row_obj = lookup_cache[ckey]
                if row_obj is None:
                    missing.append(rid)
                    continue
                if isinstance(row_obj.sanitized_payload, dict):
                    payloads_by_id[render_key] = row_obj.sanitized_payload
                if isinstance(row_obj.label, str) and row_obj.label:
                    labels_by_id[render_key] = row_obj.label

    from nora.result_render import compose_layout
    markdown = compose_layout(spec_render, payloads_by_id, labels_by_id)
    if markdown is None:
        return _as_mcp_text({
            "status": "error",
            "reason": (
                "spec is malformed. Required shape: an object with "
                "non-empty ``columns`` (list of {id, label}) and "
                "non-empty ``groups`` (list of {rows: [...]}). Each "
                "row must carry a string ``result_id``."
            ),
        })

    response: dict[str, Any] = {
        "status": "ok",
        "markdown": markdown,
        # Count actual rendered data rows, not resolved-payload uniques:
        # a missing result_id still produces a row in the layout (cells
        # render as ``—``), and the same id appearing in two groups
        # renders as two distinct rows. Tying the count to the layout
        # the model will see keeps "rows_rendered" honest when callers
        # reconcile their spec against the response.
        "rows_rendered": len(referenced),
        "result_ids_referenced": sorted({rid for rid, _ in referenced}),
    }
    if missing:
        response["missing_result_ids"] = sorted(set(missing))
        response["hint"] = (
            f"{len(set(missing))} referenced result_id(s) not in "
            f"the resolved store; cells for those rows rendered as "
            f"'—'. Use list_results (or list_results_global for "
            f"cross-session) to get the canonical IDs and re-emit."
        )
    if denied:
        response["denied_result_ids"] = sorted(set(denied))
        response.setdefault("hint", "")
        gate_msg = (
            f"{len(set(denied))} row(s) carried a session_path "
            f"outside the current session"
            + (
                "; that session_path didn't resolve under "
                "~/.nora-sessions/ so it was rejected."
                if cross_enabled else
                "; cross-session lookup is gated by "
                "NORA_ALLOW_CROSS_SESSION_RECALL=1."
            )
        )
        response["hint"] = (
            f"{response['hint']}\n{gate_msg}".strip()
            if response["hint"] else gate_msg
        )
    return _as_mcp_text(response)


# ---------------------------------------------------------------------------
# Tool: list_results
# ---------------------------------------------------------------------------

_LIST_RESULTS_DEFAULT_LIMIT = 50
_LIST_RESULTS_HARD_CAP = 500


@tool("list_results")
async def list_results(args: dict[str, Any]) -> dict[str, Any]:
    """Return the most recent stored results, capped at ``limit``.

    Newest-first because the model's typical follow-up is "what did
    we just run", not "what did we run six hours ago." The ``ASC``
    chronological order from ``list_all()`` was the worst possible
    layout for that.
    """
    requested_limit = args.get("limit", 0)
    if not isinstance(requested_limit, int) or requested_limit <= 0:
        limit = _LIST_RESULTS_DEFAULT_LIMIT
    else:
        limit = min(requested_limit, _LIST_RESULTS_HARD_CAP)

    store = get_store(get_cwd())
    all_rows = store.list_all()
    total = len(all_rows)
    # Newest first: list_all returns chronological ASC, so reverse.
    newest_first = list(reversed(all_rows))[:limit]
    truncated = total > limit
    # Re-sanitize ``label`` and ``analysis_type`` at READ time. New
    # rows are sanitized at insert; legacy rows from older Nora
    # binaries (pre-sanitization) or partially-corrupted writes
    # could otherwise carry raw bidi/zero-width/control chars or
    # ``[system] override:`` text into the listing the model sees.
    # Mirrors the parallel guard in ``chat_history.build_context_prefix``
    # for warm-start replay.
    from nora.text_safety import safe_text, safe_key
    return _as_mcp_text({
        "status": "ok",
        "total": total,
        "count": len(newest_first),
        "limit": limit,
        "truncated": truncated,
        "results": [
            {
                "id": r.id,
                "label": safe_text(r.label or ""),
                "analysis_type": (
                    safe_key(r.analysis_type) if r.analysis_type else ""
                ),
                "created_at": r.created_at,
            }
            for r in newest_first
        ],
    })


# ---------------------------------------------------------------------------
# Tool: list_results_global
# ---------------------------------------------------------------------------

@tool("list_results_global")
async def list_results_global(args: dict[str, Any]) -> dict[str, Any]:
    """Walk ``~/.nora-sessions/*/.nora/results.db`` and return one
    row per stored result across all sessions, optionally filtered
    by ``query``.

    Gated by ``NORA_ALLOW_CROSS_SESSION_RECALL``. The stored
    payloads are pre-sanitized so the privacy boundary is preserved
    regardless of which session they came from; the gate exists for
    researcher-side project separation, not as a privacy property.

    Capped at ``_LIST_RESULTS_HARD_CAP`` rows newest-first. A user
    with hundreds of sessions × dozens of results per session would
    otherwise ship megabytes of session metadata into the model
    context on a single call. Same default and hard cap as
    ``list_results`` for symmetry; the model treats the two tools
    interchangeably aside from scope.
    """
    if not _cross_session_enabled():
        return _as_mcp_text({
            "status": "denied",
            "reason": (
                f"cross-session listing is disabled in this "
                f"configuration. Set {_CROSS_SESSION_ENV_VAR}=1 in "
                f"the environment to enable. Stored payloads are "
                f"pre-sanitized — the gate exists for project "
                f"separation, not privacy."
            ),
        })
    query = (args.get("query") or "").strip().lower()

    requested_limit = args.get("limit", 0)
    if not isinstance(requested_limit, int) or requested_limit <= 0:
        limit = _LIST_RESULTS_DEFAULT_LIMIT
    else:
        limit = min(requested_limit, _LIST_RESULTS_HARD_CAP)

    from nora.ui import SESSIONS_ROOT

    if not SESSIONS_ROOT.exists():
        return _as_mcp_text({
            "status": "ok",
            "total": 0,
            "count": 0,
            "limit": limit,
            "truncated": False,
            "results": [],
        })

    rows_out: list[dict[str, Any]] = []
    current_cwd = get_cwd().resolve()
    sessions_root_resolved = SESSIONS_ROOT.resolve()

    # Build the list of session dirs to scan. Two sources, deduped
    # by resolved path:
    #
    #   1. Direct children of SESSIONS_ROOT — staged sessions.
    #   2. Folder-backed sessions registered via choose_folder.
    #      Without this, ``list_results_global`` silently omitted
    #      every session opened through the picker; researchers
    #      using folder-backed sessions would ask for an older
    #      result and Nora would act like it doesn't exist.
    #
    # Symlink escape defence: for the SESSIONS_ROOT walk, resolve
    # each entry and re-check that the resolved target's parent is
    # still SESSIONS_ROOT — a symlink in ``~/.nora-sessions/``
    # pointing at an arbitrary directory would otherwise let this
    # scan open a results.db under attacker-controlled paths.
    # Folder-backed entries don't need that re-check because the
    # registry only contains paths the researcher explicitly
    # opened, and ``external_sessions.list_entries`` already
    # filters paths whose target is no longer a directory.
    session_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for child in sorted(SESSIONS_ROOT.iterdir()):
        try:
            resolved_child = child.resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved_child.is_dir():
            continue
        if resolved_child.parent != sessions_root_resolved:
            continue
        if resolved_child in seen_dirs:
            continue
        seen_dirs.add(resolved_child)
        session_dirs.append(resolved_child)
    try:
        from nora import external_sessions
        for entry in external_sessions.list_entries(sessions_root_resolved):
            raw = entry.get("path")
            if not isinstance(raw, str):
                continue
            try:
                ext_resolved = Path(raw).resolve()
            except (OSError, RuntimeError):
                continue
            if not ext_resolved.is_dir():
                continue
            if ext_resolved in seen_dirs:
                continue
            seen_dirs.add(ext_resolved)
            session_dirs.append(ext_resolved)
    except Exception:  # noqa: BLE001 — registry is best-effort
        pass

    # Skip the current session below: the model already has the
    # per-session ``list_results`` for it — cross-session is the
    # value-add. Including it would double-list and waste tokens.
    for resolved_child in session_dirs:
        if resolved_child == current_cwd:
            continue
        db_path = resolved_child / ".nora" / "results.db"
        if not db_path.is_file():
            continue
        # Open uncached + close after reading. ``get_store`` keeps a
        # SQLite connection per cwd in a process-wide dict so the
        # interactive tool calls reuse one handle per session, but
        # this scan touches every other session on the machine —
        # using the cached path would permanently retain a
        # connection per visited session even though the scan needs
        # the rows once. Long-lived UI processes that call this tool
        # would slowly burn through file descriptors and hold stale
        # handles to sessions the researcher had since deleted.
        # ``open_store_uncached`` reuses the cache when an entry
        # already exists (so the active session's own handle isn't
        # double-opened) but doesn't insert new entries — the
        # ``finally`` closes whatever this iteration opened.
        store = None
        try:
            store = open_store_uncached(resolved_child)
            rows = store.list_all()
        except Exception:  # noqa: BLE001 — never let one bad db kill the listing
            from nora.store import _stores
            if (
                store is not None
                and _stores.get(resolved_child) is not store
            ):
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
            continue
        try:
            # Re-sanitize ``label`` and ``analysis_type`` at READ
            # time. New rows are sanitized at insert; legacy rows
            # from older Nora binaries (pre-sanitization) or
            # partially-corrupted writes could otherwise carry raw
            # bidi/zero-width/control chars or ``[system] override:``
            # text into the cross-session listing the model sees.
            # Same guard as ``list_results`` and the warm-start
            # prefix.
            from nora.text_safety import safe_text, safe_key
            for r in rows:
                raw_label = r.label or ""
                raw_atype = r.analysis_type or ""
                # Run the query match against the sanitized values
                # too, so a pre-sanitization legacy row can't be
                # surfaced (or hidden) by control chars in its
                # original label.
                label = safe_text(raw_label)
                atype = safe_key(raw_atype) if raw_atype else ""
                if (
                    query
                    and query not in label.lower()
                    and query not in atype.lower()
                ):
                    continue
                rows_out.append({
                    # Publish the resolved path so the model's
                    # ``expand_result(session_path=...)`` call also
                    # passes the tightened direct-child gate in
                    # ``_resolve_cross_session_cwd``. The symlink
                    # name is the entry point; the resolved target
                    # is the session.
                    "session_path": str(resolved_child),
                    "session_name": resolved_child.name,
                    "id": r.id,
                    "label": label,
                    "analysis_type": atype,
                    "created_at": r.created_at,
                })
        finally:
            # Always close: the cached path returns the same
            # singleton, which has its own lifecycle and shouldn't
            # be torn down here. ``open_store_uncached`` only
            # returns the cached store when one already exists, so
            # closing in that case would surprise other callers.
            # Tell them apart by re-checking the cache.
            from nora.store import _stores
            if store is not None and _stores.get(resolved_child) is not store:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
    total = len(rows_out)
    # Newest-first so the model sees recent sessions before old ones.
    rows_out.sort(
        key=lambda r: r.get("created_at") or "", reverse=True,
    )
    truncated_rows = rows_out[:limit]
    return _as_mcp_text({
        "status": "ok",
        "total": total,
        "count": len(truncated_rows),
        "limit": limit,
        "truncated": total > limit,
        "results": truncated_rows,
        "query": query if query else None,
    })


# ---------------------------------------------------------------------------
# Tool: recall_conversation
# ---------------------------------------------------------------------------


def _render_tool_use(use: Any) -> dict[str, Any]:
    """Render a ``ToolUse`` into the recall response shape.

    Carries the call's name, human label, and any ``result_id``s the
    call produced. submit_script's multi-result wire format means one
    tool call can yield N stored rows; for single-id tools (or pre-
    multi-result rows) we emit a flat ``result_id`` field so casual
    recalls stay compact. The older singular ``result_id`` attribute
    on ``ToolUse`` was renamed to ``result_ids`` in commit f70a4e1;
    this renderer used to read the stale name and AttributeError on
    every recall path that included a tool call.

    ``label`` is derived from script-controlled tool inputs
    (submit_script.label, filenames, recall queries) so it must pass
    ``safe_text`` before reaching the model — without it, a label
    like ``"...\n[system] override: ..."`` would be forwarded
    verbatim through the recall_conversation response.
    """
    label = use.label or ""
    out: dict[str, Any] = {
        "name": use.name,
        "label": safe_text(label, max_len=200) if label else "",
    }
    rids = list(use.result_ids or [])
    if len(rids) == 1:
        out["result_id"] = rids[0]
    elif rids:
        out["result_ids"] = rids
    if use.is_error:
        out["is_error"] = True
    return out


@tool("recall_conversation")
async def recall_conversation(args: dict[str, Any]) -> dict[str, Any]:
    """Search or tail the persisted chat log for the active session.

    Returns grouped Turn records (not loose event snippets), with the
    option to include ±N neighboring turns around each query match so
    Claude sees the conversation flow around the hit rather than a
    context-free line.
    """
    from nora.chat_history import read_turns as _read_turns

    query = (args.get("query") or "").strip()
    tail_raw = args.get("tail")
    context_raw = args.get("context")
    max_chars_raw = args.get("max_chars")

    # Hard ceilings on what the model can request. Without these, a
    # ``tail=1_000_000`` or ``max_chars=10_000_000`` call (typo or
    # otherwise) would dump essentially the whole persisted
    # conversation in a single response — both an obvious DoS path
    # and an unintended surface for re-exposing earlier sanitized
    # content. The ceilings stay well above the working defaults
    # so legitimate "show me a wide sweep" requests aren't pinched.
    MAX_TAIL = 200
    MAX_CHARS_CEILING = 64 * 1024  # 64 KB

    # Defaults: no args → last 10 turns. Query-only → all matches.
    if not query and (tail_raw is None or tail_raw == 0):
        tail = 10
    else:
        try:
            tail = int(tail_raw) if tail_raw is not None else 0
        except (TypeError, ValueError):
            tail = 0
    tail = max(0, min(tail, MAX_TAIL))
    try:
        context_n = int(context_raw) if context_raw is not None else 2
    except (TypeError, ValueError):
        context_n = 2
    context_n = max(0, min(context_n, 5))  # clamp — don't let a typo blow the budget
    try:
        max_chars = int(max_chars_raw) if max_chars_raw is not None else 8000
    except (TypeError, ValueError):
        max_chars = 8000
    max_chars = max(256, min(max_chars, MAX_CHARS_CEILING))

    turns = _read_turns(get_cwd())
    if not turns:
        return _as_mcp_text({
            "status": "ok",
            "turn_count": 0,
            "turns": [],
            "note": "No chat history yet for this session.",
        })

    turn_count = len(turns)

    # Pick which turns to return.
    #   - query: matching turns + `context_n` neighbors on each side,
    #     most-recent first.
    #   - tail > 0: last N turns, chronological.
    #   - default (shouldn't hit here — we set tail=10 above): all
    #     turns, chronological.
    if query:
        q = query.lower()

        def _matches(t: Turn) -> bool:
            if q in (t.user or "").lower():
                return True
            if q in (t.assistant or "").lower():
                return True
            for use in t.tools:
                if q in (use.label or "").lower():
                    return True
            return False

        matching_indices = [i for i, t in enumerate(turns) if _matches(t)]
        # Expand each match with ±context_n neighbors, dedup via set,
        # then sort. Most-recent first at the end so newest matches
        # appear at the top of the response.
        keep_set: set[int] = set()
        for idx in matching_indices:
            lo = max(0, idx - context_n)
            hi = min(turn_count - 1, idx + context_n)
            keep_set.update(range(lo, hi + 1))
        picked_indices = sorted(keep_set, reverse=True)
        picked = [turns[i] for i in picked_indices]
        chronological_output = False
    elif tail > 0:
        picked = turns[-tail:]
        chronological_output = True
    else:
        picked = list(turns)
        chronological_output = True

    # Budget: render newest-first so the newest survive if we hit
    # the cap, then flip to chronological for tail/default output.
    PER_FIELD_CAP = 1200
    FRAMING_PER_TURN = 80  # rough overhead per turn in the payload dict

    def _cap(s: str) -> str:
        return s if len(s) <= PER_FIELD_CAP else s[:PER_FIELD_CAP] + "…[truncated]"

    rendered: list[dict[str, Any]] = []
    running = 0
    budget_order = list(picked) if not chronological_output else list(reversed(picked))
    for t in budget_order:
        entry: dict[str, Any] = {"index": t.index}
        if t.user:
            entry["user"] = _cap(t.user)
        if t.assistant:
            entry["assistant"] = _cap(t.assistant)
        if t.tools:
            entry["tools"] = [
                _render_tool_use(use)
                for use in t.tools
            ]
        if t.result_ids:
            entry["result_ids"] = list(t.result_ids)
        if t.timestamp:
            entry["timestamp"] = t.timestamp
        # Cost: framing overhead plus the JSON-rendered size of every
        # field in the entry. Earlier versions counted only user +
        # assistant text, which under-budgeted result-heavy turns
        # (a 24-tool turn could add ~1.5 KB of `tools` and
        # `result_ids` payload outside the cap). Using the actual
        # serialized length keeps the soft limit honest regardless of
        # how the turn skews between prose and structured fields.
        cost = FRAMING_PER_TURN + len(json.dumps(entry, ensure_ascii=False))
        if running + cost > max_chars and rendered:
            break
        running += cost
        rendered.append(entry)
    if chronological_output:
        rendered.reverse()

    response: dict[str, Any] = {
        "status": "ok",
        "turn_count": turn_count,
        "returned": len(rendered),
        "turns": rendered,
    }
    # Tell the model when we hit a clamp ceiling on either knob so it
    # pages deliberately rather than silently re-asking for the same
    # over-cap value.
    if tail_raw is not None:
        try:
            asked_tail = int(tail_raw)
        except (TypeError, ValueError):
            asked_tail = None
        if asked_tail is not None and asked_tail > MAX_TAIL:
            response["tail_clamped_to"] = MAX_TAIL
    if max_chars_raw is not None:
        try:
            asked_chars = int(max_chars_raw)
        except (TypeError, ValueError):
            asked_chars = None
        if asked_chars is not None and asked_chars > MAX_CHARS_CEILING:
            response["max_chars_clamped_to"] = MAX_CHARS_CEILING
    return _as_mcp_text(response)


# ---------------------------------------------------------------------------
# Tool: read_attached_file
# ---------------------------------------------------------------------------

# Files the model may recall on demand. Scripts are returned inline as
# text; images are returned as an MCP image content block plus a text
# metadata sibling so non-vision providers degrade gracefully.
_RECALL_SCRIPT_EXTS: frozenset[str] = frozenset({
    ".py", ".do", ".r", ".rmd",
})
# Notebooks are JSON envelopes with a mix of code cells (safe) and
# output cells (NOT safe — outputs may carry raw DataFrame prints,
# ``list``/``summarize`` rows, etc., that the SDC sanitizer normally
# strips). We extract the *source* of code and markdown cells, drop
# all outputs, and surface the result as a script-like text payload.
# Listed separately from ``_RECALL_SCRIPT_EXTS`` because the codepath
# is different (parse → extract → assemble).
_RECALL_NOTEBOOK_EXTS: frozenset[str] = frozenset({".ipynb"})
_RECALL_IMAGE_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    # WebP and GIF are accepted by the composer (drop / paste / +
    # button); without them here ``read_attached_file`` rejected
    # the same files the UI promised the researcher could re-mention.
    # Anthropic and OpenAI vision both accept these MIME types.
    ".webp": "image/webp",
    ".gif": "image/gif",
}
# PDF / EPS are graphs the researcher might mention. We rasterise via
# the existing sips-backed sidecar (same path the Files panel uses)
# rather than shipping the original PDF — vision wants raster.
_RECALL_RASTERIZE_EXTS: frozenset[str] = frozenset({".pdf", ".eps"})


def _manifest_allowed_plot_kinds(plots_dir: Path, basename: str) -> str | None:
    """Return the manifest-recorded ``kind`` for ``basename`` in
    ``plots_dir``, or ``None`` if the file isn't manifest-listed.

    The manifest is the SDC chokepoint: ``_capture_plots`` only
    surfaces files whose entry has a ``kind`` in the allowlist
    (``interaction`` / ``coefficients`` / ``marginal_effects``).
    Files written into ``_nora_plots/`` without a manifest entry
    (e.g. a ``residuals.png`` from ``nora.plot_residuals`` — kept
    researcher-only because residual values are individual
    observations) MUST NOT be recallable as image bytes either,
    or the disclosure-control posture is undone via this side door.

    No re-validation of ``_token`` here: the executor's
    ``_filter_plot_manifest`` has already sanitized the manifest
    on-disk by the time this function runs. Either the file
    contains only validated entries (filter succeeded) or the
    file has been neutralized (unlink / rename) so the
    ``manifest.is_file()`` check below returns False. The token
    validation lives at the in-session paths (``_capture_plots`` /
    ``_summarize_plot_helpers``); the recall path runs later and
    only needs the post-sanitization state.
    """
    manifest = plots_dir / "manifest.jsonl"
    if not manifest.is_file():
        return None
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        file = entry.get("file")
        kind = entry.get("kind")
        if (isinstance(file, str) and file == basename
                and isinstance(kind, str)):
            return kind
    return None


def _is_disclosure_safe_image(target: Path, cwd: Path) -> bool:
    """Whether ``target`` is an image we may return through the recall
    path without bypassing the SDC posture.

    Two paths are safe:

    1. The target lives under ``<cwd>/.nora/runs/<id>/_nora_plots/``
       AND that run's ``manifest.jsonl`` lists the file with a kind
       in the helper allowlist. These files were already eligible
       to ride the next turn through ``_capture_plots`` — recall
       just lets the model fetch one explicitly later.

    2. (No other path is safe.) Arbitrary cwd PNGs (a researcher's
       ``plt.savefig("scatter.png")`` from a non-helper script,
       exported tables in PDF, etc.) are NOT vetted by any helper
       and would be a vision side channel around the JSON
       sanitizer. Same for ``.gph`` Stata files (rasterising those
       reveals raw observations the JSON sanitizer would have
       suppressed).

    A researcher who wants the model to see an arbitrary plot can
    drag-drop or @-mention it into the next message; that path runs
    through the composer's vision-input flow and is the documented
    "researcher-explicit" channel.
    """
    runs_root = (cwd / ".nora" / "runs").resolve() if cwd is not None else None
    if runs_root is None:
        return False
    try:
        target_resolved = target.resolve()
    except OSError:
        return False
    if not target_resolved.is_relative_to(runs_root):
        return False
    # ``<runs_root>/<run_id>/_nora_plots/<basename>`` — the parent
    # must be a ``_nora_plots`` directory directly under a run dir.
    parent = target_resolved.parent
    if parent.name != "_nora_plots":
        return False
    try:
        parent.parent.relative_to(runs_root)
    except ValueError:
        return False
    kind = _manifest_allowed_plot_kinds(parent, target_resolved.name)
    if kind is None:
        return False
    # Import the runner's allowlist to keep a single source of
    # truth for what kinds are disclosure-safe; avoids drift if
    # the allowlist ever grows or shrinks.
    from nora.runner import _PLOT_KIND_ALLOWLIST
    return kind in _PLOT_KIND_ALLOWLIST
# Per-file caps. Scripts: 96 KB. Most analysis scripts (Stata do-files,
# Python pipelines, R scripts) fit whole. Over-cap files come back
# head+tail-truncated (see below) so the imports up top AND the
# save / write call at the bottom are both visible — the head-only
# truncation we used to do hid the tail, which is exactly where the
# question "did this script write the dataset out" gets answered.
# Images: 5 MB matches the Anthropic vision ballpark and the
# composer's drop limit.
_RECALL_SCRIPT_MAX_BYTES = 96 * 1024
_RECALL_IMAGE_MAX_BYTES = 5 * 1024 * 1024


def _match_dir_by_display_name(directory: Path, displayed: str) -> Path | None:
    """Find a file in ``directory`` whose ``safe_text(name)`` matches
    the displayed name the model passed in.

    ``list_session_files`` and the search tools surface filenames
    through ``safe_text``; long names get a ``[TRUNCATED]`` marker,
    embedded whitespace is flattened, and control chars are stripped.
    The displayed name therefore does not always equal the on-disk
    basename, so a direct path lookup fails. Re-scan and match by the
    same sanitisation so the model can pass back exactly what it saw.

    Returns ``None`` on any read error or if no on-disk file maps to
    the displayed name.
    """
    from nora.text_safety import safe_text
    try:
        children = list(directory.iterdir())
    except OSError:
        return None
    for child in children:
        try:
            # ``is_file`` follows symlinks; ``is_symlink`` doesn't. A
            # researcher-uploaded symlink to ``/etc/passwd`` or a
            # neighbouring session's directory would otherwise let the
            # model recall its target via display name. ``read_attached_file``
            # is the only path that can return file BYTES to the model;
            # excluding symlinks at every match site is the chokepoint.
            if not child.is_file() or child.is_symlink():
                continue
        except OSError:
            continue
        if safe_text(child.name) == displayed:
            return child
    return None


def _match_cwd_by_display_name(cwd: Path, displayed: str) -> Path | None:
    """``_match_dir_by_display_name`` over cwd, with the empty-cwd
    guard ``read_attached_file`` would otherwise need inline."""
    if cwd is None or not cwd.is_dir():
        return None
    return _match_dir_by_display_name(cwd, displayed)


@tool("read_attached_file")
async def read_attached_file(args: dict[str, Any]) -> dict[str, Any]:
    """Return the file's contents as inline text (scripts) or an MCP
    image content block (images). See the tool description above for
    the why; here we focus on the path resolution + safety dance.
    """
    raw_name = args.get("name", "")
    if not raw_name or not isinstance(raw_name, str):
        return _as_mcp_text({
            "status": "error",
            "reason": "name argument is required (basename of an attached file)",
        })
    cwd = get_cwd()
    safe_name = Path(raw_name).name
    if not safe_name:
        return _as_mcp_text({
            "status": "error",
            "reason": f"could not parse a basename from {raw_name!r}",
        })

    try:
        target = resolve_in_cwd(safe_name)
    except (PathEscapeError, OSError):
        target = None
    # Round-trip fallback: ``list_session_files`` surfaces filenames
    # through ``safe_text`` (control-char strip, whitespace flatten,
    # 120-char truncation marker). If the on-disk name was modified
    # by that pass — long autogenerated filename, embedded whitespace,
    # extreme edge cases like control chars — direct ``resolve_in_cwd``
    # against the displayed name fails. Re-scan cwd top-level and
    # match by the SAME sanitisation the listing applied, so the
    # model can pass back exactly what it saw.
    if target is None or not target.is_file():
        target = _match_cwd_by_display_name(cwd, safe_name)
    if target is None or not target.is_file():
        target = None
        runs_root = cwd / ".nora" / "runs"
        if runs_root.is_dir():
            try:
                cwd_resolved = cwd.resolve()
            except OSError:
                cwd_resolved = None
            # Rewind enforcement: only walk run dirs whose run is
            # still associated with a visible (non-hidden) result.
            # ``None`` from the helper means "store unavailable";
            # treat that as "no rewind to enforce" so a fresh
            # session before any results still works.
            from nora.session_files import visible_run_dir_names
            visible_runs = visible_run_dir_names(cwd)
            if cwd_resolved is not None:
                try:
                    for run_dir in runs_root.iterdir():
                        if (visible_runs is not None
                                and run_dir.name not in visible_runs):
                            continue
                        plots_dir = run_dir / "_nora_plots"
                        if not plots_dir.is_dir():
                            continue
                        candidate = _match_dir_by_display_name(
                            plots_dir, safe_name,
                        )
                        if candidate is None or not candidate.is_file():
                            continue
                        try:
                            resolved = candidate.resolve()
                        except OSError:
                            continue
                        # ``is_relative_to`` is the path-aware
                        # containment check; ``str.startswith`` (the
                        # earlier behavior) treats ``/sessions/foo``
                        # as containing ``/sessions/foobar/...`` —
                        # path-prefix collision opens an escape
                        # vector for sibling sessions whose names
                        # start with this session's name.
                        if (resolved == cwd_resolved
                                or resolved.is_relative_to(cwd_resolved)):
                            target = candidate
                            break
                except OSError:
                    pass
    # Third fallback: scripts Nora wrote on prior ``submit_script``
    # calls. Each lives at ``<cwd>/.nora/runs/<id>/script.{do,R,py}``;
    # the panel surfaces them under labeled or ``script_<short_id>``
    # display names. Resolve by exactly that display name so the
    # model can pass back what it saw in ``list_session_files``
    # output. Containment back into cwd is implicit — the helper
    # only walks ``<cwd>/.nora/runs``.
    #
    # Same rewind enforcement as the plot-dir walk above: pass the
    # visible-run-dir set so a script from a discarded branch can't
    # be re-fetched by name.
    if target is None or not target.is_file():
        try:
            from nora.run_files import find_run_dir_script_by_name
            from nora.session_files import visible_run_dir_names
            candidate = find_run_dir_script_by_name(
                cwd, safe_name,
                visible_run_dirs=visible_run_dir_names(cwd),
            )
        except Exception:  # noqa: BLE001
            candidate = None
        if candidate is not None and candidate.is_file():
            target = candidate
    if target is None or not target.is_file():
        return _as_mcp_text({
            "status": "not_found",
            "reason": f"no file named {safe_name!r} in this session",
        })

    # SDC provenance gate for cwd top-level files. The script
    # sandbox at ``executor.py`` allows writes to cwd by design
    # (``saveRDS``, ``df.to_csv``, ``save "panel.dta"`` are part of
    # the normal workflow), but a model-authored script can also
    # write raw row values into a script- or notebook-shaped file
    # and recall it through this tool — bypassing every other SDC
    # gate. The ``file_provenance`` manifest tracks every cwd
    # top-level file the BRIDGE knows the researcher staged
    # (initial cwd snapshot at session-open + each ``add_files`` /
    # ``add_files_from_blobs`` / ``upload_files`` event). Anything
    # in cwd top-level that isn't in the manifest is presumed
    # sandbox-output and refused. Files under ``<cwd>/.nora/runs/``
    # are exempt — those are Nora-controlled (the wrapper script
    # the executor wrote and the manifest-allowlisted helper plots),
    # and the runs-dir fallbacks above already enforce their own
    # rewind / manifest checks.
    try:
        resolved_target = target.resolve()
        cwd_resolved = cwd.resolve()
    except OSError:
        resolved_target = target
        cwd_resolved = cwd
    nora_subdir = (cwd_resolved / ".nora").resolve() if cwd_resolved.exists() else cwd_resolved / ".nora"
    is_under_nora = False
    try:
        is_under_nora = resolved_target.is_relative_to(nora_subdir)
    except (ValueError, OSError):
        is_under_nora = False
    ext = target.suffix.lower()
    # Provenance only matters for extensions this tool would
    # actually return bytes for. Non-recallable extensions
    # (datasets, unknown types) are rejected below with extension-
    # specific guidance ("use get_schema", "unsupported type"); the
    # provenance gate would mask those messages with a less useful
    # "not staged" reason and force the user to re-attach a file
    # the tool wouldn't have returned anyway.
    _recallable_exts = (
        _RECALL_SCRIPT_EXTS
        | _RECALL_NOTEBOOK_EXTS
        | _RECALL_RASTERIZE_EXTS
        | frozenset(_RECALL_IMAGE_MIMES.keys())
    )
    if not is_under_nora and ext in _recallable_exts:
        # Fail CLOSED — same posture as ``submit_script_file``. A
        # manifest read that raises (corrupt JSON, permission-blocked
        # path, FS error) used to flip ``staged_ok`` to True, turning
        # the recall safety gate into a no-op on exactly the failure
        # mode most likely to be deliberately corrupted. Treat
        # manifest-unreadable as "not staged" and tell the caller to
        # re-stage.
        try:
            from nora.file_provenance import is_known, known_names
            staged_ok = is_known(cwd, resolved_target.name)
            # Disambiguate "never staged" vs. "staged-but-content-
            # changed" so the rejection message points at the right
            # fix. The two cases need the same recovery action
            # (re-stage via the bridge) but the cause is different,
            # and the content-changed case is also the legitimate
            # "I edited the file outside Nora" flow.
            name_in_manifest = (
                resolved_target.name in known_names(cwd)
            ) if not staged_ok else False
        except Exception:  # noqa: BLE001 — fail closed (see above)
            staged_ok = False
            name_in_manifest = False
        if not staged_ok:
            if name_in_manifest:
                reason = (
                    f"{safe_name!r} appears in this session's staged-"
                    f"files manifest, but its current on-disk content "
                    f"does not match what was staged. Either the file "
                    f"was overwritten by an earlier script (the "
                    f"sandbox permits writes to your cwd), or you "
                    f"edited it outside Nora. Re-attach it via the "
                    f"chat composer to authorise the new content."
                )
            else:
                reason = (
                    f"{safe_name!r} is not in this session's staged-"
                    f"files manifest, so I (the model) cannot read it. "
                    f"This guard exists because the analysis sandbox "
                    f"intentionally lets scripts write to your cwd, "
                    f"and a script that wrote raw rows into a "
                    f"script-shaped file would otherwise round-trip "
                    f"the data back through this tool — bypassing the "
                    f"SDC sanitizer. To make this file readable, "
                    f"please re-attach it via the chat composer "
                    f"(drop or paste it into the message box) so the "
                    f"bridge marks it as researcher-staged."
                )
            return _as_mcp_text({
                "status": "rejected",
                "reason": reason,
            })

    # ----- notebook branch ---------------------------------------------
    # ``list_session_files`` advertises ``.ipynb`` as a script-kind
    # file, so the model expects to be able to recall it. Earlier the
    # model got a "rejected" — the file existed and was advertised but
    # not retrievable, a discoverability/recovery mismatch. We now
    # extract the notebook's code + markdown cell source (dropping
    # outputs, which can carry raw DataFrame rows that the JSON
    # sanitizer would normally strip from a result) and return it as
    # script-like text. Outputs are NOT included — that's the SDC line.
    if ext in _RECALL_NOTEBOOK_EXTS:
        try:
            blob = target.read_bytes()
        except OSError as e:
            return _as_mcp_text({
                "status": "error",
                "reason": f"could not read {safe_name}: {e}",
            })
        original_size = len(blob)
        text, code_cells, markdown_cells = _extract_notebook_source(blob)
        if not text:
            return _as_mcp_text({
                "status": "error",
                "reason": (
                    f"{safe_name} parsed as a notebook but has no "
                    f"recognisable cells (malformed JSON, or empty)"
                ),
            })
        encoded = text.encode("utf-8")
        truncated = False
        if len(encoded) > _RECALL_SCRIPT_MAX_BYTES:
            half = _RECALL_SCRIPT_MAX_BYTES // 2
            head = encoded[:half]
            tail = encoded[-half:]
            elided = len(encoded) - len(head) - len(tail)
            marker = (
                f"\n\n# [... {elided} bytes elided by Nora's "
                f"read_attached_file head+tail truncation ...]\n\n"
            ).encode("utf-8")
            text = (head + marker + tail).decode("utf-8", errors="replace")
            truncated = True
        return _as_mcp_text({
            "status": "ok",
            "name": safe_name,
            "kind": "notebook",
            "ext": ext,
            "language": _ext_to_language(ext),
            "size": original_size,
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
            "truncated": truncated,
            "content": text,
            "note": (
                "Notebook outputs are stripped to keep raw DataFrame "
                "prints / list rows out of context — only cell source "
                "is returned. To run the notebook's code, paste it "
                "into submit_script with language='Python'."
            ),
        })

    # ----- script branch -----------------------------------------------
    if ext in _RECALL_SCRIPT_EXTS:
        try:
            blob = target.read_bytes()
        except OSError as e:
            return _as_mcp_text({
                "status": "error",
                "reason": f"could not read {safe_name}: {e}",
            })
        # ``script.do`` / ``script.py`` are written clean by the
        # executor (preamble lives in a sibling ``_nora_wrapper.*``
        # file invoked by the runner). No additional stripping is
        # needed here — what's on disk is exactly the researcher's
        # code.
        original_size = len(blob)
        truncated = False
        if original_size > _RECALL_SCRIPT_MAX_BYTES:
            # Head + tail truncation. Splits the byte budget evenly:
            # the first half shows imports and setup; the second half
            # shows the bottom of the script (save calls, main block).
            # The middle is elided with a marker that names how many
            # bytes were dropped, so the model knows the truncation
            # exists and roughly how big the gap is.
            #
            # Why not head-only: scripts of interest almost always
            # have load-bearing content at the END (df.to_parquet,
            # save, write_dta, the main entry). Head-only truncation
            # hides that and forces the model to either guess or ask
            # the researcher.
            half = _RECALL_SCRIPT_MAX_BYTES // 2
            head = blob[:half]
            tail = blob[-half:]
            elided = original_size - len(head) - len(tail)
            marker = (
                f"\n\n# [... {elided} bytes elided by Nora's "
                f"read_attached_file head+tail truncation ...]\n\n"
            ).encode("utf-8")
            blob = head + marker + tail
            truncated = True
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            text = blob.decode("utf-8", errors="replace")
        return _as_mcp_text({
            "status": "ok",
            "name": safe_name,
            "kind": "script",
            "ext": ext,
            "language": _ext_to_language(ext),
            "size": original_size,
            "truncated": truncated,
            "content": text,
        })

    # ----- image branch ------------------------------------------------
    if ext in _RECALL_IMAGE_MIMES or ext in _RECALL_RASTERIZE_EXTS:
        # Disclosure-control gate: only manifest-allowlisted helper
        # plots may cross to the model as vision bytes. Without this,
        # a ``residuals.png`` from ``nora.plot_residuals`` (kind
        # deliberately excluded from the per-turn capture path
        # because residuals are individual observations) or an
        # arbitrary ``plt.savefig`` PNG in cwd would slip past the
        # JSON sanitizer via this side channel. See
        # ``_is_disclosure_safe_image`` for the full rationale.
        if not _is_disclosure_safe_image(target, cwd):
            return _as_mcp_text({
                "status": "rejected",
                "reason": (
                    f"{safe_name} is an image, but it isn't a helper-"
                    f"sanitized plot (no manifest entry with an "
                    f"SDC-allowed kind). Image recall is restricted to "
                    f"plots produced by Nora's plot helpers "
                    f"(plot_interaction / plot_coefficients / "
                    f"plot_marginal_effects). Ask the researcher to "
                    f"re-attach the file in their next message if you "
                    f"need to see it again."
                ),
            })
        blob_path = target
        mime = _RECALL_IMAGE_MIMES.get(ext)
        if ext in _RECALL_RASTERIZE_EXTS:
            try:
                from nora.plot_convert import png_for
                sidecar = png_for(target)
            except Exception:  # noqa: BLE001 — conversion is best-effort
                sidecar = None
            if sidecar is None or not sidecar.is_file():
                return _as_mcp_text({
                    "status": "error",
                    "reason": (
                        f"could not rasterise {safe_name} for vision; "
                        f"open it directly in the UI instead"
                    ),
                })
            blob_path = sidecar
            mime = "image/png"
        try:
            size = blob_path.stat().st_size
        except OSError as e:
            return _as_mcp_text({
                "status": "error",
                "reason": f"stat failed: {e}",
            })
        if size > _RECALL_IMAGE_MAX_BYTES:
            return _as_mcp_text({
                "status": "error",
                "reason": (
                    f"{safe_name} is {size // (1024 * 1024)} MB, over "
                    f"the 5 MB vision limit. Ask the researcher to "
                    f"export a smaller version or open it themselves."
                ),
            })
        try:
            data_bytes = blob_path.read_bytes()
        except OSError as e:
            return _as_mcp_text({
                "status": "error",
                "reason": f"could not read {safe_name}: {e}",
            })
        import base64 as _b64
        data_b64 = _b64.b64encode(data_bytes).decode("ascii")
        descriptor = json.dumps({
            "status": "ok",
            "name": safe_name,
            "kind": "image",
            "ext": ext,
            "mime": mime or "image/png",
            "size": size,
            "note": (
                "The image is attached as an inline content block. "
                "If your provider doesn't support image tool results, "
                "ask the researcher to re-@mention the file in their "
                "next message."
            ),
        }, separators=(",", ":"), ensure_ascii=False)
        return {
            "content": [
                {
                    "type": "image",
                    "data": data_b64,
                    "mimeType": mime or "image/png",
                },
                {"type": "text", "text": descriptor},
            ]
        }

    # ----- other extensions: refused with a clear hint ------------------
    return _as_mcp_text({
        "status": "rejected",
        "reason": (
            f"{safe_name} is a {ext or 'unknown'} file; only scripts "
            f"(.py / .do / .r / .rmd), notebooks (.ipynb — code + "
            f"markdown cells only), and images (.png / .jpg / .jpeg / "
            f".webp / .gif / .pdf / .eps) can be recalled through "
            f"this tool. For datasets use get_schema; for stored "
            f"results use expand_result."
        ),
    })


def _ext_to_language(ext: str) -> str:
    """Map a script extension to the corresponding submit_script
    language label so the model knows which interpreter to ask for."""
    return {
        ".py": "Python",
        ".do": "Stata",
        ".r": "R",
        ".rmd": "R Markdown",
        ".ipynb": "Python",
    }.get(ext, "unknown")


def _extract_notebook_source(blob: bytes) -> tuple[str, int, int]:
    """Extract the source of a Jupyter notebook's code + markdown cells,
    discarding outputs.

    Returns ``(text, code_cells, markdown_cells)``.

    Output cells are dropped because they may carry raw DataFrame
    prints, ``list``/``summarize`` rows, regression tables, etc.,
    that the JSON sanitizer normally strips out of result payloads.
    Re-surfacing them through the recall path would be a side
    channel around SDC. The source (cell.source) IS the model's
    legitimate target — same character of content as a ``.py``
    script.

    Markdown cells are kept as comment blocks (``# ...``) so the
    researcher's narrative survives the round-trip; non-Python
    content can't accidentally execute when the model later treats
    the recall output as Python source.
    """
    try:
        nb = json.loads(blob.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "", 0, 0
    cells = nb.get("cells")
    if not isinstance(cells, list):
        return "", 0, 0
    parts: list[str] = []
    code_cells = 0
    markdown_cells = 0
    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        source = cell.get("source")
        # nbformat stores source as either a list of lines or a single
        # string. Normalize to one text blob.
        if isinstance(source, list):
            text = "".join(s for s in source if isinstance(s, str))
        elif isinstance(source, str):
            text = source
        else:
            continue
        if not text:
            continue
        if cell_type == "code":
            code_cells += 1
            parts.append(f"# --- cell {idx} (code) ---\n{text.rstrip()}\n")
        elif cell_type == "markdown":
            markdown_cells += 1
            commented = "\n".join(
                f"# {line}" if line else "#"
                for line in text.rstrip().splitlines()
            )
            parts.append(f"# --- cell {idx} (markdown) ---\n{commented}\n")
        # raw cells: skip — not source the model can usefully read.
    return "\n".join(parts), code_cells, markdown_cells


# ---------------------------------------------------------------------------
# Tool: list_session_files
# ---------------------------------------------------------------------------

# Caps mirror list_results: a busy project directory or a script that
# emits dozens of plot files would otherwise ship the entire enumeration
# into one tool result. Names are bounded data-origin strings (each
# goes through ``safe_text``), so the listing is also a small but real
# context channel — capping limits how much of it the model can pull
# in a single round-trip.
_LIST_SESSION_FILES_DEFAULT_LIMIT = 100
_LIST_SESSION_FILES_HARD_CAP = 500


@tool("list_session_files")
async def list_session_files(args: dict[str, Any]) -> dict[str, Any]:
    """Enumerate non-data files in the session cwd, grouped by kind.

    Datasets are intentionally NOT included: they're already
    enumerated in the system prompt's cwd listing AND gated by the
    SDC schema-depth policy. Listing them through this tool would
    create a second discovery path that bypasses the policy story.
    The shared taxonomy lives in :mod:`nora.session_files`.

    Capped at ``_LIST_SESSION_FILES_HARD_CAP`` rows mtime-desc. Same
    posture as ``list_results``: ``total`` and ``truncated`` fields
    let the model know when the listing was clipped, so it can
    refine via ``search_in_session_files`` rather than scrolling
    through hundreds of names.
    """
    from datetime import datetime, timezone
    from nora.session_files import NON_DATA_KINDS, classify_ext
    from nora.text_safety import safe_text

    raw_kinds = args.get("kinds") or []
    if not isinstance(raw_kinds, list):
        return _as_mcp_text({
            "status": "error",
            "reason": "kinds must be a list of strings",
        })
    requested = {str(k).lower() for k in raw_kinds if isinstance(k, (str, int))}
    if requested and not requested.issubset(NON_DATA_KINDS):
        bad = requested - NON_DATA_KINDS
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"unknown kinds: {sorted(bad)!r}; "
                f"valid: {sorted(NON_DATA_KINDS)!r}"
            ),
        })
    keep_kinds = requested or NON_DATA_KINDS

    cwd = get_cwd()
    if cwd is None or not cwd.is_dir():
        return _as_mcp_text({
            "status": "error",
            "reason": "no active session cwd",
        })

    rows: list[dict[str, Any]] = []
    try:
        children = list(cwd.iterdir())
    except OSError as e:
        return _as_mcp_text({
            "status": "error",
            "reason": f"could not list session cwd: {e}",
        })
    for child in children:
        try:
            # Skip symlinks: ``is_file`` follows them, so a symlink to
            # ``/etc/passwd`` or another session's run dir would
            # otherwise be listed and (via ``read_attached_file``)
            # readable. The shared session_files helper already does
            # this; the inline tool path missed it.
            if not child.is_file() or child.is_symlink():
                continue
        except OSError:
            continue
        ext = child.suffix.lower()
        kind = classify_ext(ext)
        if kind is None or kind not in keep_kinds:
            continue
        try:
            st = child.stat()
        except OSError:
            continue
        # Sanitize filenames before they land in the model's context —
        # a dropped file with embedded "System:" markers / bidi
        # overrides / newlines is exactly the prompt-injection vector
        # the system prompt's dataset listing already guards against.
        name = safe_text(child.name)
        if not name:
            continue
        rows.append({
            "name": name,
            "kind": kind,
            "size_bytes": st.st_size,
            "mtime": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc,
            ).isoformat(timespec="seconds"),
        })
    # Surface the scripts Nora wrote on prior ``submit_script`` calls.
    # They live at ``<cwd>/.nora/runs/<id>/script.{do,R,py,ipynb}``,
    # outside the cwd top-level scan above. Without this, the model
    # has no way to find a script she wrote earlier in the session
    # — the chat history may have scrolled away or been rewound, and
    # the Files panel surfaces them but the model's own tool view
    # didn't. The display name matches what the panel shows.
    #
    # Rewind-aware: ``visible_run_dir_names`` returns the basenames
    # of run dirs that still have a non-hidden result. After a
    # rewind, the on-disk run dir for a discarded branch is still
    # there but its results are hidden — without this filter the
    # model could still discover and read the script through this
    # listing.
    if "script" in keep_kinds:
        from datetime import datetime as _dt, timezone as _tz
        from nora.run_files import enumerate_run_dir_scripts
        from nora.session_files import visible_run_dir_names
        visible_runs = visible_run_dir_names(cwd)
        # Reserved names: any top-level row already emitted above. We
        # pass them into the enumeration so a colliding run-dir script
        # gets a disambiguating ``(short_id)`` suffix appended to its
        # display name rather than being silently dropped (which left
        # the model unable to fetch a prior script of the same name)
        # or shadowed by ``read_attached_file`` (which resolves
        # top-level cwd files first). Lookup uses the same reserved
        # set to reproduce the suffix.
        reserved_names = frozenset(
            r["name"] for r in rows if r.get("name")
        )
        seen_paths = {r.get("name") for r in rows}
        for entry in enumerate_run_dir_scripts(
            cwd, visible_run_dirs=visible_runs,
            reserved_names=reserved_names,
        ):
            name = safe_text(entry.display_name)
            if not name or name in seen_paths:
                continue
            rows.append({
                "name": name,
                "kind": "script",
                "size_bytes": entry.size_bytes,
                "mtime": _dt.fromtimestamp(
                    entry.mtime, tz=_tz.utc,
                ).isoformat(timespec="seconds"),
            })
            seen_paths.add(name)
    # Flat mtime-desc — newest first, what the model expects to see.
    # The previous code did two ``rows.sort()`` calls (first by
    # ``(kind, -size_bytes)``, then by ``mtime`` desc); Python's
    # stable sort meant the second call's key was the only effective
    # one, so the first sort was dead weight. Single sort makes the
    # contract explicit. Ties on the mtime ISO string fall back to
    # whatever order the tuple comparator gave us (rare in practice
    # — sub-second filesystem timestamps).
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    # Counts are computed from the FULL set so the model still sees
    # honest totals when truncation fires. Then we cap the rendered
    # ``files`` list at the hard cap; ``total`` and ``truncated``
    # advertise the cut.
    counts = {k: 0 for k in NON_DATA_KINDS}
    for r in rows:
        counts[r["kind"]] += 1
    requested_limit = args.get("limit", 0)
    if not isinstance(requested_limit, int) or requested_limit <= 0:
        limit = _LIST_SESSION_FILES_DEFAULT_LIMIT
    else:
        limit = min(requested_limit, _LIST_SESSION_FILES_HARD_CAP)
    total = len(rows)
    truncated = total > limit
    listed = rows[:limit]
    return _as_mcp_text({
        "status": "ok",
        "files": listed,
        "counts": counts,
        "total": total,
        "count": len(listed),
        "limit": limit,
        "truncated": truncated,
    })


# ---------------------------------------------------------------------------
# Tool: search_in_session_files
# ---------------------------------------------------------------------------

# Bound the per-file work so a giant log can't dominate the response.
_SEARCH_FILES_MATCH_DEFAULT = 10
_SEARCH_FILES_MATCH_HARD_CAP = 50
# Don't ingest huge files into memory just to grep — anything past the
# cap returns a "skipped: too large" entry so the model knows to read
# it directly via read_attached_file if it really needs to.
_SEARCH_FILES_FILE_BYTE_CAP = 256 * 1024
# Per-line excerpt cap so a 5000-char line in a generated log doesn't
# blow up the response payload.
_SEARCH_FILES_LINE_EXCERPT_CAP = 240
# Whole-response budgets. The per-file caps above are necessary but
# not sufficient: a broad query (a common verb, an analysis variable
# name) over a session with dozens of small scripts can satisfy the
# per-file ceiling on every file and still ship a megabyte of
# match payload back to the model. These global ceilings stop the
# scan early — the response shape includes ``total`` /
# ``truncated`` so the model knows it didn't see everything and can
# narrow the query.
_SEARCH_FILES_TOTAL_MATCHES_CAP = 200
_SEARCH_FILES_TOTAL_FILES_CAP = 50
# Approximate char cap on the rendered ``matches`` payload across
# all files. Sized so the JSON-encoded result stays under ~64 KiB of
# text before the MCP wrapping. Hit when many files each carry near-
# excerpt-cap matches; stops early with a truncation flag rather
# than streaming an oversized tool result.
_SEARCH_FILES_TOTAL_CHARS_CAP = 60_000
# Extensions whose lines can be returned verbatim to the model. These
# are plain source files: the bytes ARE the model's mental model of
# what the script does, and nothing in them was computed from the
# dataset rows. Anything else (run logs, notebook outputs) gets line-
# number-only matches because those files routinely contain raw
# observations / regression rows from `list`, `summarize, detail`,
# `print(df)`, notebook ``outputs[*].text`` blocks, etc. — content
# that the SDC sanitizer would normally strip out of a result, and
# that should not reach the model through a sibling search path.
_SEARCH_FILES_EXCERPT_EXTS: frozenset[str] = frozenset({
    ".py", ".do", ".r", ".rmd",
})


@tool("search_in_session_files")
async def search_in_session_files(args: dict[str, Any]) -> dict[str, Any]:
    """Substring search across session script + log files."""
    from nora.session_files import classify_ext
    from nora.text_safety import safe_text

    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return _as_mcp_text({
            "status": "error",
            "reason": (
                "missing required argument: query (case-insensitive "
                "substring; use list_session_files for an unfiltered "
                "file list)"
            ),
        })
    needle = query.strip().lower()

    raw_kinds = args.get("kinds") or ["script", "log"]
    if not isinstance(raw_kinds, list):
        return _as_mcp_text({
            "status": "error",
            "reason": "kinds must be a list of strings",
        })
    allowed = {"script", "log"}
    requested = {str(k).lower() for k in raw_kinds if isinstance(k, (str, int))}
    if not requested.issubset(allowed):
        bad = requested - allowed
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"unsupported kinds: {sorted(bad)!r}; valid: "
                f"{sorted(allowed)!r} (datasets and graphs aren't "
                f"text-searchable here)"
            ),
        })
    keep_kinds = requested or allowed

    requested_max = args.get("max_matches_per_file", 0) or _SEARCH_FILES_MATCH_DEFAULT
    if not isinstance(requested_max, int) or requested_max <= 0:
        max_per_file = _SEARCH_FILES_MATCH_DEFAULT
    else:
        max_per_file = min(requested_max, _SEARCH_FILES_MATCH_HARD_CAP)

    cwd = get_cwd()
    if cwd is None or not cwd.is_dir():
        return _as_mcp_text({
            "status": "error",
            "reason": "no active session cwd",
        })

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    files_searched = 0
    total_matches = 0
    try:
        children = list(cwd.iterdir())
    except OSError as e:
        return _as_mcp_text({
            "status": "error",
            "reason": f"could not list session cwd: {e}",
        })

    # Build the search set: cwd top-level files first, then the
    # Nora-written run-dir scripts when "script" is requested. The
    # run-dir scripts live under ``<cwd>/.nora/runs/<id>/`` and don't
    # appear in ``cwd.iterdir()``, but ``list_session_files`` and
    # ``read_attached_file`` both surface them — search must too,
    # otherwise the model can list a prior labeled spec, recall it,
    # but not grep across recent runs to find which one set a given
    # variable. That breaks the recovery path after a rewind.
    #
    # Each entry is ``(display_name, path, kind)``. Display names
    # come from ``safe_text`` for top-level (matches the listing
    # output) and from ``run_files`` for run-dir scripts (already
    # cleaned by ``label_to_filename_stem``). Same-name de-dup
    # prefers the top-level cwd entry (researcher's file) over the
    # run-dir copy.
    # Provenance gate (same posture as ``read_attached_file`` and
    # ``submit_script_file``): top-level cwd files that aren't in the
    # researcher-staged manifest don't get their bytes returned here.
    # The sandbox at ``executor.py`` intentionally allows scripts to
    # write to cwd, and ``search_in_session_files`` returns verbatim
    # line excerpts for ``.py`` / ``.do`` / ``.r`` / ``.rmd`` — so a
    # model script that writes raw rows into ``leak.py`` could
    # otherwise round-trip those bytes through this tool's match
    # excerpts, bypassing the SDC sanitizer that gates submit_script.
    # Manifest reads that raise are treated as "not staged" (fail
    # closed; matches the recall-path posture). Run-dir scripts
    # (handled below) don't need this gate — those are Nora-written
    # copies of the model's own submissions, not arbitrary cwd files.
    try:
        from nora.file_provenance import is_known as _is_known_provenance
    except Exception:  # noqa: BLE001
        _is_known_provenance = None  # type: ignore[assignment]
    search_entries: list[tuple[str, Path, str]] = []
    seen_names: set[str] = set()
    for child in sorted(children, key=lambda p: p.name):
        try:
            # Skip symlinks (see list_session_files for the same guard
            # rationale): a symlink in cwd would otherwise let
            # search_in_session_files read its target's bytes,
            # bypassing the SDC line for the file's actual location.
            if not child.is_file() or child.is_symlink():
                continue
        except OSError:
            continue
        ext = child.suffix.lower()
        kind = classify_ext(ext)
        if kind not in keep_kinds:
            continue
        name = safe_text(child.name)
        if not name or name in seen_names:
            continue
        # Drop unstaged top-level files silently rather than emitting
        # a "skipped: not staged" entry. The denial reason would echo
        # the model-authored basename back through the response,
        # giving the model a free directory-listing channel into
        # sandbox-output it isn't allowed to read. ``list_session_files``
        # already advertises which files are listable.
        staged_ok = True
        if _is_known_provenance is not None:
            try:
                staged_ok = _is_known_provenance(cwd, child.name)
            except Exception:  # noqa: BLE001 — fail closed
                staged_ok = False
        if not staged_ok:
            continue
        seen_names.add(name)
        search_entries.append((name, child, kind))
    if "script" in keep_kinds:
        from nora.run_files import enumerate_run_dir_scripts
        from nora.session_files import visible_run_dir_names
        # Rewind-aware: a discarded chat branch leaves its run dirs
        # on disk but its results are hidden, and ``list_session_files``
        # /``read_attached_file`` already filter on ``visible_run_dirs``
        # so a hidden-branch script is invisible to the model. Without
        # the same filter here, ``search_in_session_files`` would still
        # return excerpts from those scripts — a sibling discovery
        # path that bypasses the rewind visibility contract.
        visible_runs = visible_run_dir_names(cwd)
        # Same disambiguation contract as list_session_files: pass
        # the already-emitted top-level names as reserved so a
        # colliding run-dir script gets a ``(short_id)`` suffix
        # rather than being silently dropped here.
        reserved_names = frozenset(seen_names)
        for entry in enumerate_run_dir_scripts(
            cwd, visible_run_dirs=visible_runs,
            reserved_names=reserved_names,
        ):
            name = safe_text(entry.display_name)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            search_entries.append((name, entry.path, "script"))

    # Track when a global cap fires so the response can flag truncation.
    response_truncated = False
    response_truncated_reason: str | None = None
    rendered_chars = 0

    for name, child, kind in search_entries:
        # Global response budgets. Each one stops the scan immediately,
        # records the reason, and lets the loop fall through to the
        # response build below — the partial results we already have
        # are still useful, and the truncation flag tells the model
        # to narrow the query.
        if len(results) >= _SEARCH_FILES_TOTAL_FILES_CAP:
            response_truncated = True
            response_truncated_reason = (
                f"hit the {_SEARCH_FILES_TOTAL_FILES_CAP}-file response "
                f"cap; narrow the query or filter by ``kinds``"
            )
            break
        if total_matches >= _SEARCH_FILES_TOTAL_MATCHES_CAP:
            response_truncated = True
            response_truncated_reason = (
                f"hit the {_SEARCH_FILES_TOTAL_MATCHES_CAP}-match "
                f"response cap; narrow the query"
            )
            break
        if rendered_chars >= _SEARCH_FILES_TOTAL_CHARS_CAP:
            response_truncated = True
            response_truncated_reason = (
                f"hit the {_SEARCH_FILES_TOTAL_CHARS_CAP}-char response "
                f"payload cap; narrow the query"
            )
            break
        ext = child.suffix.lower()
        try:
            st = child.stat()
        except OSError:
            continue
        if st.st_size > _SEARCH_FILES_FILE_BYTE_CAP:
            # Recovery hint depends on whether read_attached_file
            # actually accepts this file type. The earlier message
            # said "use read_attached_file" universally, but that
            # tool refuses .log / .smcl (their bytes can carry raw
            # rows that the SDC sanitizer normally strips, so they're
            # outside the recall contract). Telling the model to call
            # read_attached_file on a 256 KB+ log produced a
            # guaranteed failed follow-up in a common path. For
            # those, the right move is to ask the researcher for the
            # relevant snippet directly. Notebooks (.ipynb) ARE
            # recallable now (code + markdown cells, outputs
            # stripped) so they fall into the script-recovery branch.
            if ext in _RECALL_SCRIPT_EXTS or ext in _RECALL_NOTEBOOK_EXTS:
                recover_hint = "use read_attached_file to fetch it"
            else:
                recover_hint = (
                    "ask the researcher to paste the relevant snippet "
                    "(read_attached_file refuses .log / .smcl to keep "
                    "raw rows out of context)"
                )
            skipped.append({
                "name": name,
                "kind": kind,
                "reason": (
                    f"file too large for inline search "
                    f"({st.st_size} bytes > "
                    f"{_SEARCH_FILES_FILE_BYTE_CAP} cap); "
                    f"{recover_hint}"
                ),
            })
            continue
        try:
            text = child.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            skipped.append({
                "name": name,
                "kind": kind,
                "reason": f"read failed: {e}",
            })
            continue
        files_searched += 1
        # Only plain-source extensions return verbatim line excerpts.
        # Logs and notebooks return ``{line: N}`` entries so the model
        # can locate matches without seeing raw rows / cell outputs.
        # See the disclosure-control note in the tool docstring.
        excerpts_allowed = ext in _SEARCH_FILES_EXCERPT_EXTS
        # Per-file budget: stop scanning the moment a global cap
        # would be breached by appending the next match. The pre-
        # loop ``total_matches >= cap`` / ``rendered_chars >= cap``
        # gates above test the budgets BEFORE this file is touched,
        # but with ``max_per_file=50`` and the 200-match global
        # cap, the previous file could leave us at 199; appending
        # 50 more from THIS file would push us to 249. Enforce
        # both caps as hard caps by checking remaining budget per
        # match rather than per file.
        remaining_matches = (
            _SEARCH_FILES_TOTAL_MATCHES_CAP - total_matches
        )
        remaining_chars = _SEARCH_FILES_TOTAL_CHARS_CAP - rendered_chars
        matches: list[dict[str, Any]] = []
        added_chars = 0
        file_truncated = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle not in line.lower():
                continue
            if excerpts_allowed:
                excerpt = line.strip()
                if len(excerpt) > _SEARCH_FILES_LINE_EXCERPT_CAP:
                    excerpt = excerpt[:_SEARCH_FILES_LINE_EXCERPT_CAP] + "…"
                match_entry: dict[str, Any] = {
                    "line": lineno, "text": safe_text(excerpt),
                }
                match_cost = 12 + len(match_entry["text"])
            else:
                match_entry = {"line": lineno}
                match_cost = 12
            # Refuse the match if either global budget would be
            # blown by adding it. ``response_truncated`` gets set
            # below if the per-file cap or a global cap halts the
            # scan early.
            if len(matches) >= remaining_matches:
                file_truncated = True
                break
            if added_chars + match_cost > remaining_chars:
                file_truncated = True
                break
            matches.append(match_entry)
            added_chars += match_cost
            if len(matches) >= max_per_file:
                file_truncated = True
                break
        if matches:
            total_matches += len(matches)
            rendered_chars += added_chars
            row = {
                "name": name,
                "kind": kind,
                "excerpts": excerpts_allowed,
                "matches": matches,
                "truncated": file_truncated,
            }
            results.append(row)
        if total_matches >= _SEARCH_FILES_TOTAL_MATCHES_CAP:
            response_truncated = True
            response_truncated_reason = (
                f"hit the {_SEARCH_FILES_TOTAL_MATCHES_CAP}-match "
                f"response cap; narrow the query"
            )
            break
        if rendered_chars >= _SEARCH_FILES_TOTAL_CHARS_CAP:
            response_truncated = True
            response_truncated_reason = (
                f"hit the {_SEARCH_FILES_TOTAL_CHARS_CAP}-char response "
                f"payload cap; narrow the query"
            )
            break

    response: dict[str, Any] = {
        "status": "ok",
        "query": query,
        "files_searched": files_searched,
        "total_matches": total_matches,
        "results": results,
        "skipped": skipped,
        "truncated": response_truncated,
    }
    if response_truncated_reason is not None:
        response["truncated_reason"] = response_truncated_reason
    return _as_mcp_text(response)


# ---------------------------------------------------------------------------
# Tool: install_packages
# ---------------------------------------------------------------------------

@tool("install_packages")
async def install_packages(args: dict[str, Any]) -> dict[str, Any]:
    """Install / remove / reinstall language packages out-of-band.

    Network + library-write happens here, not in submit_script. The
    system prompt instructs the model to ask the researcher in chat
    first; this handler is the HARD gate behind that prompt-only
    request. Before running the underlying installer we surface a
    modal in the UI via ``install_confirmation.request_confirmation``
    and only proceed on an explicit approval. Without a UI attached
    (headless / test), the gate fails closed — denying the install
    is the safe default for a tool that mutates the researcher's
    machine.
    """
    from nora.package_installer import install_packages as _do_install
    from nora.install_confirmation import request_confirmation

    language = args.get("language", "")
    packages_arg = args.get("packages") or []
    action = args.get("action") or "install"

    if not isinstance(language, str):
        return _as_mcp_text({
            "status": "error",
            "reason": "language must be a string ('R', 'Python', or 'Stata')",
        })
    if not isinstance(packages_arg, list):
        return _as_mcp_text({
            "status": "error",
            "reason": "packages must be a list of names",
        })
    if not isinstance(action, str):
        return _as_mcp_text({
            "status": "error",
            "reason": "action must be a string ('install', 'remove', or 'reinstall')",
        })

    # Validate before opening the modal — pointless to ask the
    # researcher to approve an install we'd reject downstream anyway.
    # The package_installer module re-validates internally; the
    # checks here are just to fail fast on obvious malformed input.
    if not packages_arg:
        return _as_mcp_text({
            "status": "error",
            "reason": "packages list is empty",
        })

    approved = await request_confirmation(
        language=language,
        packages=list(packages_arg),
        action=action,
    )
    if not approved:
        return _as_mcp_text({
            "status": "rejected",
            "reason": (
                "researcher declined the install (or did not respond "
                "within the confirmation window). Ask the researcher "
                "directly in chat what they'd like to do; do not "
                "re-call this tool without their explicit go-ahead."
            ),
            "language": language,
            "action": action,
            # Package names echoed back so the model can confirm what
            # was proposed, but no install was performed.
            "packages": list(packages_arg),
        })

    # Register the installer subprocess with the runner's per-turn
    # registry so a Stop fired during install kills the whole
    # process group (pip's compile-step grandchildren, R's
    # ``configure``, Stata's ado-update children). Without this,
    # cancel only stopped the asyncio task while the install kept
    # running in the background, continuing to mutate the
    # researcher's environment after Stop.
    from nora.runtime.turn_context import register_turn_process
    result = await _do_install(
        language, list(packages_arg), action,
        proc_register=register_turn_process,
    )

    statuses = [
        {"name": s.name, "status": s.status, "detail": s.detail}
        for s in result.statuses
    ]
    if result.error:
        # Scrub the raw subprocess output before forwarding. The
        # invariant in ``docs/direction.md`` ("Raw stderr / stdout
        # never reach the model") is preserved for ``submit_script``
        # by ``error_summary.extract_debug_excerpt`` (language-
        # anchored extraction + credential / path scrub). The
        # install path skips the language anchor (pip / R / Stata
        # install logs aren't tracebacks to extract from) but it
        # MUST NOT skip the credential and path normalisation —
        # otherwise a private pip index URL with embedded
        # ``user:token@`` (echoed from pip.conf on every install),
        # an AWS-style key dumped via ``echo $TOKEN``, or an
        # internal absolute path would round-trip through this
        # tool's error excerpt unredacted. The tail-slice cap
        # bytes are picked first (stderr is more informative on
        # failure) and the scrubber re-caps to the same budget
        # after redactions.
        from nora.error_summary import scrub_raw_output
        return _as_mcp_text({
            "status": "error",
            "language": result.language,
            "action": result.action,
            "reason": result.error,
            "statuses": statuses,
            "duration_seconds": round(result.duration_seconds, 2),
            # Scrub both stdout and stderr through ``scrub_raw_output``
            # — the credential / path scrubber documented above. This
            # supersedes the earlier "drop stdout, keep stderr" posture
            # because the scrubber handles the exact concrete leaks
            # that motivated dropping stdout (pip's ``user:token@`` in
            # index URLs, absolute filesystem paths a malicious
            # ``setup.py`` could echo) while preserving the diagnostic
            # value of stdout for legitimate install failures.
            "raw_stdout_excerpt": scrub_raw_output(
                (result.raw_stdout or "")[-1500:], cap_bytes=1500,
            ),
            "raw_stderr_excerpt": scrub_raw_output(
                (result.raw_stderr or "")[-3000:], cap_bytes=3000,
            ),
        })
    # No raw output on success. pip's progress output echoes the full
    # index URL — including any token-bearing
    # ``index-url = https://USER:TOKEN@private-pypi.acme.com/simple``
    # the researcher configured in pip.conf or PIP_INDEX_URL — and
    # that excerpt was being forwarded into the model's transcript on
    # every successful install. The ``statuses`` list already tells
    # the model which packages were touched and at what version;
    # pip's chatty output adds no information beyond that.
    return _as_mcp_text({
        "status": "ok",
        "language": result.language,
        "action": result.action,
        "statuses": statuses,
        "duration_seconds": round(result.duration_seconds, 2),
    })


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------

SERVER_NAME = "nora"

REGISTERED_TOOLS: tuple[Any, ...] = (
    get_schema,
    search_schema,
    request_data,
    submit_script,
    submit_script_file,
    expand_result,
    compose_results,
    list_results,
    list_results_global,
    recall_conversation,
    read_attached_file,
    list_session_files,
    search_in_session_files,
    install_packages,
)

# Tool names Claude will see are prefixed: mcp__<server>__<tool>.
# Keep this list in sync with the @tool-decorated functions above.
ALLOWED_TOOL_NAMES: tuple[str, ...] = (
    f"mcp__{SERVER_NAME}__get_schema",
    f"mcp__{SERVER_NAME}__search_schema",
    f"mcp__{SERVER_NAME}__request_data",
    f"mcp__{SERVER_NAME}__submit_script",
    f"mcp__{SERVER_NAME}__submit_script_file",
    f"mcp__{SERVER_NAME}__expand_result",
    f"mcp__{SERVER_NAME}__compose_results",
    f"mcp__{SERVER_NAME}__list_results",
    f"mcp__{SERVER_NAME}__list_results_global",
    f"mcp__{SERVER_NAME}__recall_conversation",
    f"mcp__{SERVER_NAME}__read_attached_file",
    f"mcp__{SERVER_NAME}__list_session_files",
    f"mcp__{SERVER_NAME}__search_in_session_files",
    f"mcp__{SERVER_NAME}__install_packages",
)


def friendly_tool_names(prefixed: bool = True) -> tuple[str, ...]:
    """Return tool names for human-facing messages (denial hints, etc.).

    Derived from ``ALLOWED_TOOL_NAMES`` so any new tool added to the
    registry shows up automatically in recovery hints. Hard-coded
    copies of this list previously drifted — the catch-all permission
    deny in ``provider/anthropic.py`` got stuck at six names while
    the registry grew to thirteen — and the denial message is exactly
    the recovery path the model needs to discover new tools like
    ``list_session_files`` and ``search_in_session_files``, so any
    drift is user-visible.

    ``prefixed=True`` keeps the ``mcp__<server>__`` prefix that Claude
    actually sees in its tool list. ``prefixed=False`` strips it for
    display contexts (terminal banners) where the prefix is noise.
    """
    if prefixed:
        return ALLOWED_TOOL_NAMES
    cut = len(f"mcp__{SERVER_NAME}__")
    return tuple(name[cut:] for name in ALLOWED_TOOL_NAMES)


# Provider-neutral dispatch table. The Anthropic path goes through the
# in-process MCP server; the OpenAI path calls the bare async handlers
# directly from this map. Built from the SDK-decorated tool objects'
# ``.handler`` attribute so both paths invoke the *same* function — no
# risk of drift, no duplication of handler bodies.
#
# Tool names here are FLAT (``"get_schema"``, not the
# ``"mcp__nora__get_schema"`` MCP prefix). OpenAI function-tool names
# are flat by API; the Anthropic path doesn't consult this map.
HANDLERS: dict[str, Any] = {t.name: t.handler for t in REGISTERED_TOOLS}


def build_server() -> dict[str, Any]:
    """Construct the in-process MCP server with all Nora tools registered."""
    return create_sdk_mcp_server(
        name=SERVER_NAME,
        version="0.9.1",
        tools=list(REGISTERED_TOOLS),
    )

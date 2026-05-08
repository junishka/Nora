"""Nora — MCP tool surface.

This module defines the *exhaustive* interface through which the frontier
model is allowed to reach the researcher's local machine. The canonical
tool list lives in ``nora.provider.tool_schemas.TOOL_SPECS``; this module
registers each spec with the Claude Agent SDK and supplies the handler
bodies. The Claude Agent SDK's built-in tools (Bash, Read, Write, Edit,
Glob, Grep, WebFetch, WebSearch, etc.) are disabled at the `app.py` layer
via `disallowed_tools` + a `can_use_tool` catch-all.

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
import os
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
from nora.store import get_store


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
    if t == "linear_regression":
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
    if t == "linear_regression":
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
        return f"descriptive for {v!r}, n={n}, mean={m}, sd={sd}"
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
    if sanitized.get("type") == "linear_regression":
        return {k: v for k, v in sanitized.items() if k not in ("vcov", "vif")}
    return dict(sanitized)


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
# regression batch with verbose markdown) can still ship 30k+ chars
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
_INLINE_PAYLOAD_BUDGET = 12_000
_INLINE_MARKDOWN_BUDGET = 30_000


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

    if manifest.is_file():
        for entry in _read_jsonl(manifest):
            succeeded.append({
                "file": str(entry.get("file", "?")),
                "kind": str(entry.get("kind", "?")),
                "label": str(entry.get("label", "")),
            })
    if errors.is_file():
        for entry in _read_jsonl(errors):
            row: dict[str, Any] = {
                "helper": str(entry.get("helper", "?")),
                "message": str(entry.get("message", "")),
            }
            if entry.get("fix"):
                row["fix"] = str(entry["fix"])
            failed.append(row)

    if not succeeded and not failed:
        return None
    summary: dict[str, Any] = {
        "succeeded": succeeded,
        "failed": failed,
    }
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
    except ValueError as e:
        # Unsupported format, invalid depth, or a file-shape problem
        # we can express as a user-facing reason.
        return _as_mcp_text({
            "status": "error",
            "reason": str(e),
            "dataset": dataset,
            "depth": depth,
        })
    except Exception as e:  # broad: lib-specific parse errors
        return _as_mcp_text({
            "status": "error",
            "reason": f"failed to read {dataset!r}: {e.__class__.__name__}: {e}",
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
    except ValueError as e:
        return _as_mcp_text({
            "status": "error",
            "reason": str(e),
            "dataset": dataset,
            "depth": extract_depth,
        })
    except Exception as e:  # noqa: BLE001 — broad: lib-specific parse errors
        return _as_mcp_text({
            "status": "error",
            "reason": f"failed to read {dataset!r}: {e.__class__.__name__}: {e}",
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
) -> tuple[Any, dict[str, Any] | None]:
    """Run the executor in a worker thread, with cancellation handling.

    Returns ``(exec_result, None)`` on completion, or
    ``(exec_result, early_payload)`` when the turn was cancelled
    mid-run — the caller short-circuits with ``early_payload`` and
    skips sanitize / store / response assembly. ``CancelledError`` is
    re-raised so the runner's outer cancel branch handles teardown.
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
        # subprocess was still running. The Popen finished naturally
        # (or got killed) and we now hold an ExecutionResult, but
        # persisting it as a chat-visible result would surface a tool
        # answer for a turn the researcher cancelled — exactly the
        # leak the turn-identity contract is designed to prevent. The
        # raw run_dir stays on disk for debugging; we just don't
        # sanitize, store, or return a structured payload.
        early = {
            "status": "cancelled",
            "reason": (
                "turn was cancelled while the script was running; "
                "raw stdout/stderr remain on disk in the run_dir but "
                "are not surfaced as a result"
            ),
            "run_dir": str(exec_result.run_dir),
        }
        return exec_result, early
    return exec_result, None


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
            non_disclosive = policy_module.non_disclosive_for(
                policy_obj, source_dataset,
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
    if shared_transformations:
        response["transformations_summary"] = shared_transformations
    if inline_payload_omitted:
        response["_inline_payload_omitted"] = True
    if inline_markdown_omitted:
        response["_inline_markdown_omitted"] = True
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
        # Add the script-level error context: reason, exit code, and
        # a bounded debug_excerpt of stdout/stderr so the model can
        # diagnose the abort. The partial-success branch carries this
        # ALONGSIDE the partial results; the bare-failure branch
        # carries it alone (or alongside rejection-only rows).
        response["reason"] = exec_result.error
        response["exit_code"] = exec_result.exit_code
        from nora.error_summary import extract_debug_excerpt
        excerpt = extract_debug_excerpt(
            exec_result.raw_stdout,
            exec_result.raw_stderr,
            exec_result.exit_code,
            language,
        )
        if not excerpt:
            excerpt = (
                f"script failed (exit code {exec_result.exit_code}); "
                f"inspect raw log in UI"
            )
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
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"unsupported language: {language!r}. Nora runs R "
                f"(via Rscript), Stata, and Python (3.x with pandas)."
            ),
        })
    if not code.strip():
        return _as_mcp_text({
            "status": "error",
            "reason": "code argument is empty",
        })

    cwd = get_cwd()

    # 1. Execute. Cancellation surfaces here as either ``CancelledError``
    # (re-raised so the runner's outer cancel branch handles teardown)
    # or as ``early_payload`` set when the turn was cancelled mid-run.
    exec_result, early_payload = await _execute_script_for_submit(
        language, code, cwd,
    )
    if early_payload is not None:
        return _as_mcp_text(early_payload)

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
        return _as_mcp_text({
            "status": "error",
            "reason": "name argument is required (basename of a script file)",
        })
    safe_name = Path(raw_name).name
    if not safe_name:
        return _as_mcp_text({
            "status": "error",
            "reason": f"could not parse a basename from {raw_name!r}",
        })

    try:
        target = resolve_in_cwd(safe_name)
    except (PathEscapeError, OSError):
        return _as_mcp_text({
            "status": "error",
            "reason": f"path {raw_name!r} is outside the session cwd",
        })
    if not target.is_file():
        return _as_mcp_text({
            "status": "not_found",
            "reason": f"no script named {safe_name!r} in this session",
        })

    ext = target.suffix.lower()
    inferred_language = _SCRIPT_FILE_LANGUAGES.get(ext)
    if inferred_language is None:
        return _as_mcp_text({
            "status": "error",
            "reason": (
                f"{safe_name!r} is not a recognised script file. "
                f"Supported extensions: "
                f"{sorted(_SCRIPT_FILE_LANGUAGES.keys())}"
            ),
        })

    explicit_language = (args.get("language") or "").strip()
    language = explicit_language or inferred_language
    if explicit_language and explicit_language != inferred_language:
        # The model overrode the extension-based inference. Honor it,
        # but only if the override is one of the supported languages —
        # otherwise the downstream submit_script would reject anyway.
        if explicit_language not in {"R", "Stata", "Python"}:
            return _as_mcp_text({
                "status": "error",
                "reason": (
                    f"language must be one of R / Stata / Python; "
                    f"got {explicit_language!r}"
                ),
            })

    try:
        code = target.read_text(encoding="utf-8")
    except OSError as e:
        return _as_mcp_text({
            "status": "error",
            "reason": f"could not read {safe_name}: {e}",
        })
    except UnicodeDecodeError:
        # Fall back to replace-mode so a stray non-UTF-8 byte doesn't
        # block the run; the script is the researcher's, they can fix
        # it if encoding matters.
        code = target.read_bytes().decode("utf-8", errors="replace")

    if not code.strip():
        return _as_mcp_text({
            "status": "error",
            "reason": f"{safe_name!r} is empty",
        })

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
    resolved Path, or None if it isn't a session under
    ``~/.nora-sessions/``. Path-confined to that root so the model
    can't direct the store-loader at arbitrary paths on the
    machine."""
    from nora.ui import SESSIONS_ROOT, _is_within

    try:
        target = Path(session_path).expanduser().resolve()
    except OSError:
        return None
    if not _is_within(target, SESSIONS_ROOT.resolve()):
        return None
    if not target.is_dir():
        return None
    return target


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
                    "session_path must be a directory inside "
                    "~/.nora-sessions/"
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
    if view == "coefficients" and isinstance(payload, dict):
        # Trim the regression-collinearity diagnostics. ``vcov`` is the
        # full variance-covariance matrix (NxN dict-of-dict, biggest
        # field on a wide regression) and ``vif`` is the per-predictor
        # VIF table. Coefficients, SEs, p-values, n, R², condition
        # number, and degrees of freedom all stay; the model can
        # re-fetch the trimmed fields with view="full" if needed.
        if payload.get("type") == "linear_regression":
            payload = dict(payload)
            for key in ("vcov", "vif"):
                if key in payload:
                    payload.pop(key)
                    view_dropped.append(key)

    # ``view="markdown"`` returns a canonical markdown pipe-table
    # rendered from the sanitized payload. Same source as the JSON
    # payload, but pre-formatted so the model can drop it into a
    # response without re-deriving columns / precision per-call (the
    # source of inconsistent renders across recalls).
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
        "payload": payload,
        "transformations": row.transformations,
        "created_at": row.created_at,
    }
    if view:
        response["view"] = view
    if view_dropped:
        response["view_dropped_fields"] = view_dropped
    if markdown is not None:
        response["markdown"] = markdown
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


@tool("compose_results")
async def compose_results(args: dict[str, Any]) -> dict[str, Any]:
    """Render a layout spec into a composite comparison table."""
    spec = args.get("spec")
    if not isinstance(spec, dict):
        return _as_mcp_text({
            "status": "error",
            "reason": "spec argument is required and must be a JSON object",
        })

    cwd = get_cwd()
    store = get_store(cwd)

    # Walk the spec and collect referenced result_ids; fetch each from
    # the store. Missing IDs are flagged separately so the model can
    # see exactly which references it got wrong.
    referenced_ids: list[str] = []
    groups = spec.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            rows = group.get("rows")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    rid = row.get("result_id")
                    if isinstance(rid, str) and rid:
                        referenced_ids.append(rid)

    payloads_by_id: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for rid in referenced_ids:
        if rid in seen:
            continue
        seen.add(rid)
        row_obj = store.get(rid)
        if row_obj is None:
            missing.append(rid)
            continue
        if isinstance(row_obj.sanitized_payload, dict):
            payloads_by_id[rid] = row_obj.sanitized_payload

    from nora.result_render import compose_layout
    markdown = compose_layout(spec, payloads_by_id)
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
        "rows_rendered": len(payloads_by_id),
        "result_ids_referenced": sorted(seen),
    }
    if missing:
        response["missing_result_ids"] = sorted(missing)
        response["hint"] = (
            f"{len(missing)} referenced result_id(s) not in this "
            f"session's store; cells for those rows rendered as '—'. "
            f"Use list_results to get the canonical IDs and re-emit."
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
    return _as_mcp_text({
        "status": "ok",
        "total": total,
        "count": len(newest_first),
        "limit": limit,
        "truncated": truncated,
        "results": [
            {
                "id": r.id,
                "label": r.label,
                "analysis_type": r.analysis_type,
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

    from nora.ui import SESSIONS_ROOT

    if not SESSIONS_ROOT.exists():
        return _as_mcp_text({
            "status": "ok",
            "count": 0,
            "results": [],
        })

    rows_out: list[dict[str, Any]] = []
    # Iterate every session dir under ~/.nora-sessions/. Skip the
    # current session here because the model already has list_results
    # for that — cross-session is the value-add. Including it would
    # double-list and waste tokens.
    current_cwd = get_cwd().resolve()
    for child in sorted(SESSIONS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() == current_cwd:
            continue
        db_path = child / ".nora" / "results.db"
        if not db_path.is_file():
            continue
        try:
            store = get_store(child)
            rows = store.list_all()
        except Exception:  # noqa: BLE001 — never let one bad db kill the listing
            continue
        for r in rows:
            label = r.label or ""
            atype = r.analysis_type or ""
            if query and query not in label.lower() and query not in atype.lower():
                continue
            rows_out.append({
                "session_path": str(child),
                "session_name": child.name,
                "id": r.id,
                "label": label,
                "analysis_type": atype,
                "created_at": r.created_at,
            })
    # Newest-first so the model sees recent sessions before old ones.
    rows_out.sort(
        key=lambda r: r.get("created_at") or "", reverse=True,
    )
    return _as_mcp_text({
        "status": "ok",
        "count": len(rows_out),
        "results": rows_out,
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
    """
    out: dict[str, Any] = {"name": use.name, "label": use.label}
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

    # Defaults: no args → last 10 turns. Query-only → all matches.
    if not query and (tail_raw is None or tail_raw == 0):
        tail = 10
    else:
        try:
            tail = int(tail_raw) if tail_raw is not None else 0
        except (TypeError, ValueError):
            tail = 0
    try:
        context_n = int(context_raw) if context_raw is not None else 2
    except (TypeError, ValueError):
        context_n = 2
    context_n = max(0, min(context_n, 5))  # clamp — don't let a typo blow the budget
    try:
        max_chars = int(max_chars_raw) if max_chars_raw is not None else 8000
    except (TypeError, ValueError):
        max_chars = 8000

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

    return _as_mcp_text({
        "status": "ok",
        "turn_count": turn_count,
        "returned": len(rendered),
        "turns": rendered,
    })


# ---------------------------------------------------------------------------
# Tool: read_attached_file
# ---------------------------------------------------------------------------

# Files the model may recall on demand. Scripts are returned inline as
# text; images are returned as an MCP image content block plus a text
# metadata sibling so non-vision providers degrade gracefully.
_RECALL_SCRIPT_EXTS: frozenset[str] = frozenset({
    ".py", ".do", ".r", ".rmd",
})
_RECALL_IMAGE_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
# PDF / EPS are graphs the researcher might mention. We rasterise via
# the existing sips-backed sidecar (same path the Files panel uses)
# rather than shipping the original PDF — vision wants raster.
_RECALL_RASTERIZE_EXTS: frozenset[str] = frozenset({".pdf", ".eps"})
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
            if not child.is_file():
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
            if cwd_resolved is not None:
                try:
                    for run_dir in runs_root.iterdir():
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
    if target is None or not target.is_file():
        try:
            from nora.run_files import find_run_dir_script_by_name
            candidate = find_run_dir_script_by_name(cwd, safe_name)
        except Exception:  # noqa: BLE001
            candidate = None
        if candidate is not None and candidate.is_file():
            target = candidate
    if target is None or not target.is_file():
        return _as_mcp_text({
            "status": "not_found",
            "reason": f"no file named {safe_name!r} in this session",
        })

    ext = target.suffix.lower()

    # ----- script branch -----------------------------------------------
    if ext in _RECALL_SCRIPT_EXTS:
        try:
            blob = target.read_bytes()
        except OSError as e:
            return _as_mcp_text({
                "status": "error",
                "reason": f"could not read {safe_name}: {e}",
            })
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
            f"(.py / .do / .r / .rmd) and images (.png / .jpg / .jpeg / "
            f".pdf / .eps) can be recalled through this tool. For "
            f"datasets use get_schema; for stored results use "
            f"expand_result."
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
    }.get(ext, "unknown")


# ---------------------------------------------------------------------------
# Tool: list_session_files
# ---------------------------------------------------------------------------

@tool("list_session_files")
async def list_session_files(args: dict[str, Any]) -> dict[str, Any]:
    """Enumerate non-data files in the session cwd, grouped by kind.

    Datasets are intentionally NOT included: they're already
    enumerated in the system prompt's cwd listing AND gated by the
    SDC schema-depth policy. Listing them through this tool would
    create a second discovery path that bypasses the policy story.
    The shared taxonomy lives in :mod:`nora.session_files`.
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
            if not child.is_file():
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
    if "script" in keep_kinds:
        from datetime import datetime as _dt, timezone as _tz
        from nora.run_files import enumerate_run_dir_scripts
        seen_paths = {r.get("name") for r in rows}
        for entry in enumerate_run_dir_scripts(cwd):
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
    rows.sort(key=lambda r: (r["kind"], -r["size_bytes"]))
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    counts = {k: 0 for k in NON_DATA_KINDS}
    for r in rows:
        counts[r["kind"]] += 1
    return _as_mcp_text({
        "status": "ok",
        "files": rows,
        "counts": counts,
        "total": len(rows),
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
    search_entries: list[tuple[str, Path, str]] = []
    seen_names: set[str] = set()
    for child in sorted(children, key=lambda p: p.name):
        try:
            if not child.is_file():
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
        seen_names.add(name)
        search_entries.append((name, child, kind))
    if "script" in keep_kinds:
        from nora.run_files import enumerate_run_dir_scripts
        for entry in enumerate_run_dir_scripts(cwd):
            name = safe_text(entry.display_name)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            search_entries.append((name, entry.path, "script"))

    for name, child, kind in search_entries:
        ext = child.suffix.lower()
        try:
            st = child.stat()
        except OSError:
            continue
        if st.st_size > _SEARCH_FILES_FILE_BYTE_CAP:
            # Recovery hint depends on whether read_attached_file
            # actually accepts this file type. The earlier message
            # said "use read_attached_file" universally, but that
            # tool refuses .log / .smcl / .ipynb (their bytes can
            # carry raw rows or cell outputs the SDC sanitizer
            # normally strips, so they're outside the recall
            # contract). Telling the model to call read_attached_file
            # on a 256 KB+ log produced a guaranteed failed follow-up
            # in a common path. For those, the right move is to ask
            # the researcher for the relevant snippet directly.
            if ext in _RECALL_SCRIPT_EXTS:
                recover_hint = "use read_attached_file to fetch it"
            else:
                recover_hint = (
                    "ask the researcher to paste the relevant snippet "
                    "(read_attached_file refuses .log / .smcl / .ipynb "
                    "to keep raw rows and cell outputs out of context)"
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
        matches: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                if excerpts_allowed:
                    excerpt = line.strip()
                    if len(excerpt) > _SEARCH_FILES_LINE_EXCERPT_CAP:
                        excerpt = excerpt[:_SEARCH_FILES_LINE_EXCERPT_CAP] + "…"
                    matches.append({"line": lineno, "text": safe_text(excerpt)})
                else:
                    matches.append({"line": lineno})
                if len(matches) >= max_per_file:
                    break
        if matches:
            total_matches += len(matches)
            results.append({
                "name": name,
                "kind": kind,
                "excerpts": excerpts_allowed,
                "matches": matches,
                "truncated": len(matches) >= max_per_file,
            })

    return _as_mcp_text({
        "status": "ok",
        "query": query,
        "files_searched": files_searched,
        "total_matches": total_matches,
        "results": results,
        "skipped": skipped,
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
)


def friendly_tool_names(prefixed: bool = True) -> tuple[str, ...]:
    """Return tool names for human-facing messages (denial hints, etc.).

    Derived from ``ALLOWED_TOOL_NAMES`` so any new tool added to the
    registry shows up automatically in recovery hints. Prior versions
    hardcoded a comma-separated list in two places (the catch-all
    permission deny in ``provider/anthropic.py`` and the terminal
    catch-all in ``app.py``). Both drifted — the Anthropic copy got
    stuck at six names while the registry grew to thirteen, and the
    terminal copy stalled at ten. Drift is bad here because the
    denial message is exactly the recovery path the model needs to
    discover new tools like ``list_session_files`` and
    ``search_in_session_files``.

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
        version="0.0.1",
        tools=list(REGISTERED_TOOLS),
    )

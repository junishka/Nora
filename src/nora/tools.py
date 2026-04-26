"""Nora — MCP tool surface (step 2 of the build ladder, mocked stage).

This module defines the *exhaustive* interface through which the frontier
model is allowed to reach the researcher's local machine. Six tools, no
others. The Claude Agent SDK's built-in tools (Bash, Read, Write, Edit,
Glob, Grep, WebFetch, WebSearch, etc.) are disabled at the `app.py` layer
via `disallowed_tools` + a `can_use_tool` catch-all.

All tools return structured payloads. Earlier steps returned mocked values — the goal of
step 2 is to define and enforce the interface, not to actually execute
scripts or read data. Step 3 wires in real schema extraction; step 4 wires
in the executor + sanitizer choke point; step 5 turns the pass-through
sanitizer into a real allowlist with SDC rules.

Invariants enforced here:
- Values never cross the boundary. Mocked payloads never include simulated
  observation-level data.
- Every tool returns a structured payload (JSON-encoded). No raw stdout.
- Result IDs are opaque to Claude — it references them by label + ID.

See also: `project_builder_mcp_surface.md` (user memory) for the full spec.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from nora import data_request, executor, policy as policy_module, schema
from nora.config import PathEscapeError, get_cwd, resolve_in_cwd
from nora.data_request import SUPPORTED_REQUEST_TYPES
from nora.policy import (
    depth_allowed,
    get_max_depth,
    has_explicit_policy,
    load_policy,
)
from nora.sanitizer import sanitize
from nora.store import get_store


# Build the request_type enumeration string from the canonical list in
# data_request so the tool's help text cannot drift from the actual
# implementation. Previously the help listed `numeric_range` and
# `missingness_pattern` — neither supported by the runtime — so Claude
# would call them and get "denied: request_type not in the allowlist".
_REQUEST_TYPE_LIST_STR = ", ".join(f"'{t}'" for t in SUPPORTED_REQUEST_TYPES)


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


def _check_row_count(
    sanitized_payload: dict[str, Any], source_dataset: str | None
) -> str | None:
    """If ``source_dataset`` was given, compare analysis N to dataset N.

    Returns a transformation-log string describing the row-count change,
    or ``None`` if no check could be made or no discrepancy exists.
    All error paths are silent — this is a best-effort audit signal,
    not a gate.
    """
    if not source_dataset:
        return None
    try:
        path = resolve_in_cwd(source_dataset)
    except PathEscapeError:
        return None
    if not path.is_file():
        return None

    try:
        df = schema.load_data(path)
        source_n = int(len(df))
    except Exception:
        # Any load problem → skip silently. The main submit_script
        # result is unaffected.
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
            f"or bootstrapped — verify the intent."
        )

    diff = source_n - analysis_n
    pct = diff * 100.0 / source_n if source_n else 0.0
    return (
        f"ROW COUNT CHANGE: analysis used n={analysis_n} rows but source "
        f"dataset {source_dataset!r} has {source_n} — {diff} row(s) "
        f"excluded ({pct:.1f}%). Common causes: NA-drop by the analysis "
        f"command, an `if` / `subset(...)` / `filter(...)` in the script, "
        f"or a listwise-deletion from complete.cases. Verify the "
        f"exclusion was intentional."
    )


# Text constants returned by mocked tools so it is unambiguous to the
# researcher (and the frontier model) that this is a placeholder interface.
_MOCK_NOTE = "MOCKED — step 2 of the build ladder. Returns placeholder data."


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

def _as_mcp_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON-serializable dict as an MCP text-content response.

    MCP content is a list of typed blocks; our convention is a single text
    block containing JSON. Keeping the payload structured (not prose) makes
    the downstream sanitizer job clean and keeps the contract testable.
    """
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}
        ]
    }


# ---------------------------------------------------------------------------
# Tool: get_schema
# ---------------------------------------------------------------------------

@tool(
    "get_schema",
    (
        "Return the structural summary of a dataset — variable names, types, "
        "labels, value labels, observation count. Never returns individual "
        "observation values. Use this before writing any analysis script so "
        "you know what variables exist and their types.\n\n"
        "Supported file types: .dta (Stata), .rds (R), .csv, .tsv, .parquet, "
        ".jsonl / .ndjson.\n\n"
        "Arguments:\n"
        "  dataset: path to the dataset file, relative to the researcher's "
        "working directory (or absolute path within it).\n"
        "  depth: one of:\n"
        "    - 'names_only': variable names only.\n"
        "    - 'names_types': + type of each variable.\n"
        "    - 'names_types_labels': + variable labels and value labels.\n"
        "    - 'names_types_labels_summary': + NA counts and distinct counts "
        "for categoricals.\n"
        "  Default: 'names_types_labels_summary'. Each successful response "
        "includes a 'policy_max_depth' field showing the ceiling the "
        "researcher has set for this dataset — you cannot exceed it. "
        "Requests above the ceiling are denied with the current ceiling "
        "named in the reason."
    ),
    {"dataset": str, "depth": str},
)
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
                f"({ceiling!r}{' — explicit' if explicit else ' — default'}"
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
# Tool: request_data
# ---------------------------------------------------------------------------

@tool(
    "request_data",
    (
        "Ask the layer for a specific, bounded piece of information about "
        "the data that is NOT in the default schema. The layer evaluates "
        "the request against disclosure policy and returns either a "
        "sanitized answer or a denial with a reason. Use this instead of "
        "writing an exploratory probe script.\n\n"
        "Arguments:\n"
        "  dataset: identifier for the dataset.\n"
        f"  request_type: one of {_REQUEST_TYPE_LIST_STR}.\n"
        "  variable: name of the variable the request is about."
    ),
    {"dataset": str, "request_type": str, "variable": str},
)
async def request_data(args: dict[str, Any]) -> dict[str, Any]:
    """Step-5 implementation: real, SDC-gated bounded data queries.

    Each request type is a pre-approved computation with its own
    disclosure-control rule. See ``nora.data_request`` for the
    per-type logic.
    """
    dataset = args.get("dataset", "")
    request_type = args.get("request_type", "")
    variable = args.get("variable", "")

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

    result = data_request.handle(path, request_type, variable)
    payload: dict[str, Any] = {
        "status": result.status,
        "dataset": dataset,
        "request_type": request_type,
        "variable": variable,
    }
    if result.answer is not None:
        payload["answer"] = result.answer
    if result.reason is not None:
        payload["reason"] = result.reason
    return _as_mcp_text(payload)


# ---------------------------------------------------------------------------
# Tool: submit_script
# ---------------------------------------------------------------------------

@tool(
    "submit_script",
    (
        "Run an R, Stata, or Python analysis script against the researcher's "
        "data. The script must emit structured results via the nora runtime "
        "library (nora$result(...) in R, nora_result_* in Stata, "
        "nora.result(...) / nora.from_lm(...) in Python). Raw stdout/stderr "
        "is shown to the researcher in their TUI but is not returned to you "
        "— you receive only the sanitized structured payload. Returns a "
        "result ID and a one-line label.\n\n"
        "Arguments:\n"
        "  language: 'R', 'Stata', or 'Python'.\n"
        "  code: the full script source as a single string.\n"
        "  label: short description of what the script is doing (e.g., "
        "'OLS of outcome on predictors').\n"
        "  source_dataset: path (relative to cwd) of the dataset the "
        "script reads. When set, Nora compares the analysis's "
        "effective N to the dataset's row count and flags silent "
        "filtering (NA-drops, subset conditions, listwise deletion) "
        "in the transformations log. PASS THIS whenever the script "
        "reads a known file — this is how researchers catch analyses "
        "that quietly ran on a subset. Empty string is fine if the "
        "script generates its own data or touches multiple files."
    ),
    {"language": str, "code": str, "label": str, "source_dataset": str},
)
async def submit_script(args: dict[str, Any]) -> dict[str, Any]:
    """Run an R / Stata script end-to-end: execute → sanitize → store.

    The researcher's raw stdout / stderr is captured and stashed on the
    stored row (available via expand_result). Claude only ever sees the
    sanitizer's output — never the raw log.
    """
    language = args.get("language", "")
    code = args.get("code", "")
    label = args.get("label", "(unlabeled)")
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

    # --- Execution ---------------------------------------------------------
    cwd = get_cwd()
    exec_result = executor.run_script(language, code, cwd)

    # Execution-level failures (interpreter missing, timeout, no structured
    # output, bad JSON) come back to Claude as policy-shaped errors. The
    # researcher still sees the raw log in the scratch dir.
    if not exec_result.ok:
        # Persist a diagnostic row anyway so the researcher can find the
        # run dir from the store. Analysis type is "script_error" to keep
        # it distinguishable from successful results.
        store = get_store(cwd)
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
        )
        return _as_mcp_text({
            "status": "execution_failed",
            "reason": exec_result.error,
            "exit_code": exec_result.exit_code,
            "result_id": diag_row.id,
            "duration_seconds": round(exec_result.duration_seconds, 3),
            "hint": (
                "The raw stdout/stderr is preserved in the run directory "
                "(see result via expand_result). Adjust the script and "
                "resubmit."
            ),
            "_run_dir": str(exec_result.run_dir),
            # Language hint must travel on error paths too: the UI
            # uses it to strip the Stata preamble from raw stdout
            # and to pick the right "Open in Stata" / "Open in R"
            # button. Without it, errored Stata runs leaked the
            # preamble into the visible output.
            "_language": language,
        })

    # --- Sanitizer ---------------------------------------------------------
    raw_payload = exec_result.result_payload or {}
    sanitized = sanitize(raw_payload)
    if not sanitized.ok:
        # The script produced a result, but SDC rules / schema mismatch
        # bounced it. Still store it so the researcher can audit.
        store = get_store(cwd)
        diag_row = store.insert(
            label=f"[rejected] {label}",
            analysis_type=sanitized.analysis_type or "unknown",
            sanitized_payload={
                "type": "sanitizer_rejection",
                "reason": sanitized.rejection_reason,
                "analysis_type": sanitized.analysis_type,
            },
            language=language,
            script_code=code,
            transformations=[],
            raw_log_path=exec_result.run_dir,
        )
        return _as_mcp_text({
            "status": "rejected_by_sanitizer",
            "reason": sanitized.rejection_reason,
            "analysis_type": sanitized.analysis_type,
            "result_id": diag_row.id,
            "hint": (
                "The script ran successfully but its output violates a "
                "disclosure-control rule (e.g., n too small, forbidden "
                "field, type mismatch). Adjust the analysis (e.g., "
                "larger sample) and resubmit."
            ),
            "_run_dir": str(exec_result.run_dir),
            "_language": language,
        })

    # --- Row-count change check --------------------------------------------
    # Compare the analysis's effective N to the source dataset's N. A
    # shortfall means rows were silently excluded — typically NA-drop
    # from `lm()` / `ttest`, or a filter / subset / `if` in the script.
    # Runs AFTER sanitization because the check operates on the
    # sanitized payload structure; the row-counts themselves aren't
    # disclosive.
    row_count_msg = _check_row_count(
        sanitized.sanitized or {}, source_dataset or None
    )
    transformations = list(sanitized.transformations)
    if row_count_msg:
        transformations.append(row_count_msg)

    # --- Store -------------------------------------------------------------
    store = get_store(cwd)
    row = store.insert(
        label=label,
        analysis_type=sanitized.analysis_type or "unknown",
        sanitized_payload=sanitized.sanitized or {},
        language=language,
        script_code=code,
        transformations=transformations,
        raw_log_path=exec_result.run_dir,
    )

    return _as_mcp_text({
        "status": "ok",
        "result_id": row.id,
        "label": row.label,
        "analysis_type": row.analysis_type,
        "summary": _summarize(sanitized.sanitized or {}),
        "transformations": transformations,
        "duration_seconds": round(exec_result.duration_seconds, 3),
        # Path for the TUI to read raw R/Stata output from. Claude
        # seeing the path is not a leak (directory names are
        # structural, not data), but Claude should not try to read
        # files from it — there are no tools for that. The researcher
        # UI uses this to render the raw log alongside the sanitized
        # result.
        "_run_dir": str(exec_result.run_dir),
        # Language the script was written in. The web UI uses this
        # to label the "Open in R / Stata" button and pick the right
        # invocation when launching the native app.
        "_language": language,
    })


# ---------------------------------------------------------------------------
# Tool: expand_result
# ---------------------------------------------------------------------------

@tool(
    "expand_result",
    (
        "Retrieve the full sanitized payload for a previously stored result "
        "by its ID. Use this when you need to reference details of an "
        "earlier result — e.g., coefficients from a prior regression — "
        "without carrying the whole payload in context.\n\n"
        "Arguments:\n"
        "  result_id: the ID returned by a previous submit_script call."
    ),
    {"result_id": str},
)
async def expand_result(args: dict[str, Any]) -> dict[str, Any]:
    """Return the full stored sanitized payload for a given ID."""
    result_id = args.get("result_id", "")
    if not result_id:
        return _as_mcp_text({
            "status": "error",
            "reason": "result_id argument is required",
        })
    store = get_store(get_cwd())
    row = store.get(result_id)
    if row is None:
        return _as_mcp_text({
            "status": "not_found",
            "reason": f"no stored result with id {result_id!r}",
        })
    response: dict[str, Any] = {
        "status": "ok",
        "result_id": row.id,
        "label": row.label,
        "analysis_type": row.analysis_type,
        "language": row.language,
        "payload": row.sanitized_payload,
        "transformations": row.transformations,
        "created_at": row.created_at,
    }
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
# Tool: list_results
# ---------------------------------------------------------------------------

@tool(
    "list_results",
    (
        "List stored sanitized results from this session as a table of "
        "(id, label). Use to remind yourself what analyses you've run so "
        "far without pulling full payloads into context."
    ),
    {},  # no arguments
)
async def list_results(args: dict[str, Any]) -> dict[str, Any]:
    """Return a table of (id, label, type, created_at) for this project's store."""
    store = get_store(get_cwd())
    rows = store.list_all()
    return _as_mcp_text({
        "status": "ok",
        "count": len(rows),
        "results": [
            {
                "id": r.id,
                "label": r.label,
                "analysis_type": r.analysis_type,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    })


# ---------------------------------------------------------------------------
# Tool: recall_conversation
# ---------------------------------------------------------------------------

@tool(
    "recall_conversation",
    (
        "Search this session's archived chat log for turns NOT already "
        "in your context. The most recent ~20 turns are auto-loaded "
        "when a session opens, so you already have short-term memory; "
        "use this tool for DEEPER lookups into older history.\n\n"
        "When to call this:\n"
        "- The researcher references an analysis or exchange from "
        "earlier in a long session that's no longer in your context "
        "window (\"the regression we ran at the start\", \"what did "
        "I ask yesterday about the gate variable\").\n"
        "- You need the exact wording of something older — quote it "
        "back verbatim rather than paraphrasing.\n"
        "- The auto-injected history starts with "
        "\"N earlier turns omitted\" and the researcher's question "
        "clearly points at those omitted turns.\n\n"
        "Do NOT call this for content already visible to you in the "
        "current conversation — answer from context. The tool is a "
        "disk read; use it when context genuinely can't answer the "
        "question.\n\n"
        "Arguments (all optional):\n"
        "  query: case-insensitive substring matched against user + "
        "assistant text and tool labels. Returns matching turns with "
        "±2 neighboring turns for context, most-recent first.\n"
        "  tail: last N turns regardless of query. Useful with a "
        "large N to page further back than the auto-injected window.\n"
        "  context: neighbor turns to include around each match "
        "(default 2). Only applies when ``query`` is set.\n"
        "  max_chars: soft cap on total text returned (default ~8000).\n\n"
        "Returns {turn_count (total in archive), turns (list of "
        "{index, user, assistant, tools: [{name,label,result_id?}], "
        "result_ids, timestamp?})}. Thinking traces and raw tool-"
        "result bodies are excluded — use list_results / expand_result "
        "for stored sanitized payloads."
    ),
    {"query": str, "tail": int, "context": int, "max_chars": int},
)
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
                {
                    "name": use.name,
                    "label": use.label,
                    **({"result_id": use.result_id} if use.result_id else {}),
                    **({"is_error": True} if use.is_error else {}),
                }
                for use in t.tools
            ]
        if t.result_ids:
            entry["result_ids"] = list(t.result_ids)
        if t.timestamp:
            entry["timestamp"] = t.timestamp
        cost = FRAMING_PER_TURN + len(entry.get("user", "")) + len(entry.get("assistant", ""))
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
# Server registration
# ---------------------------------------------------------------------------

SERVER_NAME = "nora"

REGISTERED_TOOLS: tuple[Any, ...] = (
    get_schema,
    request_data,
    submit_script,
    expand_result,
    list_results,
    recall_conversation,
)

# Tool names Claude will see are prefixed: mcp__<server>__<tool>.
# Keep this list in sync with the @tool-decorated functions above.
ALLOWED_TOOL_NAMES: tuple[str, ...] = (
    f"mcp__{SERVER_NAME}__get_schema",
    f"mcp__{SERVER_NAME}__request_data",
    f"mcp__{SERVER_NAME}__submit_script",
    f"mcp__{SERVER_NAME}__expand_result",
    f"mcp__{SERVER_NAME}__list_results",
    f"mcp__{SERVER_NAME}__recall_conversation",
)


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

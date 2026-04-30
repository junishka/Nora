"""Canonical, provider-neutral schemas for the six Nora MCP tools.

This module is the single source of truth for tool name, description,
and input schema. Both the Claude Agent SDK registration in
``nora.tools`` and the OpenAI function-tool builder in
``provider/openai.py`` derive from here, so the two providers cannot
drift on what the model is told a tool does.

Adding or modifying a tool means editing ``TOOL_SPECS`` here and the
matching handler body in ``nora.tools``. The
``test_tool_schema_consistency`` test enforces that the SDK
decorator's view of each tool matches this module's view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral description of a single Nora tool.

    ``input_schema`` is a JSON-Schema fragment for the tool's input
    object. It is consumed unchanged by the OpenAI Responses API
    (``parameters`` field on a function tool) and reduced to the
    Anthropic SDK's lighter ``{name: python_type}`` shape via
    ``as_sdk_args()`` for the in-process MCP server.

    ``description`` is the canonical text used by the Anthropic SDK
    (verified by ``test_tool_descriptions_match`` against the
    ``@tool`` decorator in ``nora.tools``).

    ``openai_description`` is an optional, leaner variant used only
    when the spec is rendered for OpenAI. OpenAI's Responses API
    sends ``tools`` on every ``responses.create()`` call gated by
    auto-cache at a 50% discount. Anthropic's CLI puts the same tools
    behind a 90% cache discount, so a 1-token cut on the OpenAI side
    is worth 5x what the same cut buys on Anthropic. When the OpenAI
    variant is omitted, ``as_openai_tool()`` falls back to
    ``description``. Short variants must convey *when to call* the
    tool; full args list lives in the JSON schema.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    required: tuple[str, ...] = field(default_factory=tuple)
    openai_description: str | None = None

    def as_sdk_args(self) -> dict[str, type]:
        """Return ``{param_name: python_type}`` for ``claude_agent_sdk.tool``.

        The SDK accepts a Python-type-mapping form rather than full
        JSON Schema; this strips the schema down. JSON-Schema ``string``
        → ``str``, ``integer`` → ``int``, ``number`` → ``float``,
        ``boolean`` → ``bool``, ``object`` → ``dict``, ``array`` → ``list``.
        """
        out: dict[str, type] = {}
        props = self.input_schema.get("properties", {})
        for pname, pschema in props.items():
            t = pschema.get("type", "string")
            out[pname] = _JSON_TO_PY.get(t, str)
        return out

    def as_openai_tool(self) -> dict[str, Any]:
        """Return the OpenAI Responses-API ``tool`` entry.

        Shape: ``{"type": "function", "name": ..., "description": ...,
        "parameters": {<JSON Schema>}, "strict": False}``. ``strict``
        stays off so optional fields work without forcing every caller
        to thread null defaults — the Nora handlers tolerate missing
        keys with explicit error messages.

        Uses ``openai_description`` when set; falls back to the
        canonical ``description`` otherwise. The leaner variant cuts
        per-call wire payload at OpenAI's 50% cache discount where
        each saved token is worth ~5x more than the same cut on
        Anthropic (90% discount).
        """
        params = dict(self.input_schema)
        # Required list goes inside the parameters schema per JSON Schema.
        if self.required:
            params = {**params, "required": list(self.required)}
        elif "required" not in params:
            params = {**params, "required": []}
        if "additionalProperties" not in params:
            params = {**params, "additionalProperties": False}
        return {
            "type": "function",
            "name": self.name,
            "description": self.openai_description or self.description,
            "parameters": params,
            "strict": False,
        }


_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------
#
# Descriptions copied verbatim from the @tool decorations in
# ``nora.tools`` so the model sees identical guidance regardless of
# provider. If you edit one, edit the other (the consistency test will
# fail loudly otherwise).

_GET_SCHEMA_DESC = (
    "Return the structural summary of a dataset. Variable names, types, "
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
    "researcher has set for this dataset. You cannot exceed it. "
    "Requests above the ceiling are denied with the current ceiling "
    "named in the reason."
)

_SEARCH_SCHEMA_DESC = (
    "Find variables in a dataset whose name or label matches a "
    "case-insensitive substring. Designed for wide datasets where "
    "``get_schema`` would return hundreds of variables; "
    "search_schema lets you ask 'which columns are salary-related' "
    "without pulling the full schema into context.\n\n"
    "Matches against variable ``name`` and (when policy allows) "
    "``label``. Results are capped at ``limit`` (default 50, hard "
    "max 200); the response includes ``total_matches`` so you "
    "know whether to refine the query.\n\n"
    "The search depth is the lower of (a) the dataset's policy "
    "ceiling and (b) names_types_labels (no need to load summary "
    "stats just to filter names). Returned variables carry the "
    "same fields ``get_schema`` would return at that depth.\n\n"
    "Arguments:\n"
    "  dataset: path to the dataset, relative to cwd.\n"
    "  query: case-insensitive substring to match against names "
    "and labels. Empty string is rejected — list-everything is "
    "what get_schema is for.\n"
    "  limit: optional cap on matches returned (default 50, max "
    "200). 0 or unset uses the default."
)

# ``request_data``'s description is built from
# ``data_request.SUPPORTED_REQUEST_TYPES`` so the tool's docstring
# can never claim a request type the runtime doesn't support. Same
# pattern as ``nora.tools`` uses.
def _request_data_desc() -> str:
    from nora.data_request import SUPPORTED_REQUEST_TYPES
    types_str = ", ".join(f"'{t}'" for t in SUPPORTED_REQUEST_TYPES)
    return (
        "Ask the layer for a specific, bounded piece of information about "
        "the data that is NOT in the default schema. The layer evaluates "
        "the request against disclosure policy and returns either a "
        "sanitized answer or a denial with a reason. Use this instead of "
        "writing an exploratory probe script.\n\n"
        "Arguments:\n"
        "  dataset: identifier for the dataset.\n"
        f"  request_type: one of {types_str}.\n"
        "  variable: name of the (first) variable the request is about.\n"
        "  variable2: optional second variable, only used by "
        "multi-variable types (correlation_pair). Single-variable "
        "types ignore it."
    )

_SUBMIT_SCRIPT_DESC = (
    "Run an R, Stata, or Python analysis script against the researcher's "
    "data. The script must emit structured results via the nora runtime "
    "library (nora$result(...) / nora$from_* in R, nora_result_* in "
    "Stata, nora.result(...) / nora.from_* in Python). Raw stdout/stderr "
    "is shown to the researcher in their TUI but is not returned to you "
    "; you receive only sanitized structured payloads.\n\n"
    "A script can call helpers more than once; each call appends a "
    "payload that comes back to you. The response carries a ``results`` "
    "list (one entry per helper call, in emission order, each with its "
    "own ``result_id``, ``label``, ``analysis_type``, and ``summary``) "
    "plus a shared ``script_run_id`` so the group can be retrieved "
    "together for audit. For parameterized batches (the same model "
    "across N specifications, subgroups, outcomes, or sensitivity "
    "perturbations), use ONE looping script over N separate scripts "
    "to avoid repeated data preparation and a fragmented audit. If "
    "the script aborts mid-loop, status becomes "
    "``execution_failed_partial`` and the helpers that emitted "
    "before the abort are returned in ``results`` alongside a "
    "``debug_excerpt`` of the abort cause.\n\n"
    "Arguments:\n"
    "  language: 'R', 'Stata', or 'Python'.\n"
    "  code: the full script source as a single string.\n"
    "  label: short description of what the script is doing (e.g., "
    "'OLS of outcome on predictors'). Used as the fallback row label "
    "for any helper call that didn't pass its own label(\"...\").\n"
    "  source_dataset: path (relative to cwd) of the dataset the "
    "script reads. When set, Nora compares each analysis's "
    "effective N to the dataset's row count and flags silent "
    "filtering (NA-drops, subset conditions, listwise deletion) "
    "in the transformations log. PASS THIS whenever the script "
    "reads a known file. This is how researchers catch analyses "
    "that quietly ran on a subset. Empty string is fine if the "
    "script generates its own data or touches multiple files."
)

_SUBMIT_SCRIPT_FILE_DESC = (
    "Run a script from a file the researcher attached, instead of "
    "re-emitting the bytes through your tool input. Use this when "
    "the researcher @-mentioned or uploaded a .do / .R / .py file "
    "and wants it run as-is. For a 12 KB do-file, this skips a "
    "12 KB tool-input round-trip and the latency that comes with "
    "it.\n\n"
    "Same downstream behavior as submit_script (sanitizer, "
    "row-count audit, store, multi-result, partial-success). The "
    "response shape is identical.\n\n"
    "Arguments:\n"
    "  name: basename of the attached file (e.g., 'reg_v10.do'). "
    "Must exist in the session cwd. Path components are stripped "
    "(same posture as read_attached_file).\n"
    "  language: 'R', 'Stata', or 'Python'. Optional — when "
    "omitted, inferred from the file extension (.do→Stata, .r/"
    ".rmd→R, .py→Python).\n"
    "  label: short description (used as the fallback row label "
    "for any helper that didn't pass its own label).\n"
    "  source_dataset: same as submit_script."
)

_EXPAND_RESULT_DESC = (
    "Retrieve a stored sanitized payload by ID. Use this when you "
    "need details of an earlier result (e.g., coefficients from a "
    "prior regression) without carrying the whole payload in "
    "context.\n\n"
    "Arguments:\n"
    "  result_id: the ID returned by a previous submit_script call.\n"
    "  view: optional payload trim. ``\"\"`` (default) or "
    "``\"full\"`` returns the complete stored payload. "
    "``\"coefficients\"`` is a regression-specific shorthand that "
    "drops the variance-covariance matrix (``vcov``) and per-"
    "predictor VIF table — useful when you only need the headline "
    "coefficient pattern and not the collinearity diagnostics. "
    "Other analysis types ignore the option.\n"
    "  session_path: optional path to ANOTHER session under "
    "~/.nora-sessions/ to expand a result from. Requires the "
    "NORA_ALLOW_CROSS_SESSION_RECALL=1 env var to be set; otherwise "
    "returns 'cross-session disabled'."
)

_LIST_RESULTS_GLOBAL_DESC = (
    "List stored sanitized results from EVERY Nora session under "
    "~/.nora-sessions/. Use when the researcher refers to an analysis "
    "from a different project/session and you need to find it. "
    "Returns rows tagged with their session_path; pair with "
    "expand_result(result_id, session_path=...) to fetch the full "
    "payload.\n\n"
    "Requires the NORA_ALLOW_CROSS_SESSION_RECALL=1 env var to be set "
    "(default OFF — researchers may want explicit project "
    "separation regardless of payload safety). When disabled, "
    "returns 'cross-session disabled' with no results.\n\n"
    "Arguments:\n"
    "  query: optional case-insensitive substring filter on "
    "label / analysis_type. Omit to list everything."
)

_LIST_RESULTS_DESC = (
    "List stored sanitized results from this session as a table of "
    "(id, label). Use to remind yourself what analyses you've run so "
    "far without pulling full payloads into context."
)

_RECALL_CONVERSATION_DESC = (
    "Search this session's archived chat log for turns NOT already "
    "in your context. The most recent ~20 turns are auto-loaded "
    "when a session opens, so you already have short-term memory; "
    "use this tool for DEEPER lookups into older history.\n\n"
    "When to call this:\n"
    "- The researcher references an analysis or exchange from "
    "earlier in a long session that's no longer in your context "
    "window (\"the regression we ran at the start\", \"what did "
    "I ask yesterday about the gate variable\").\n"
    "- You need the exact wording of something older. Quote it "
    "back verbatim rather than paraphrasing.\n"
    "- The auto-injected history starts with "
    "\"N earlier turns omitted\" and the researcher's question "
    "clearly points at those omitted turns.\n\n"
    "Do NOT call this for content already visible to you in the "
    "current conversation. Answer from context. The tool is a "
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
    "result bodies are excluded. Use list_results / expand_result "
    "for stored sanitized payloads."
)

_READ_ATTACHED_FILE_DESC = (
    "Re-read a file the researcher attached to this session. "
    "Scripts (.py / .do / .r / .rmd) and images (.png / .jpg / "
    ".jpeg / .pdf / .eps). Use this when a file's content was in "
    "your context earlier (because the researcher @-mentioned or "
    "uploaded it) but has since scrolled out as the conversation "
    "grew. The bytes are still on disk in the session cwd; this "
    "tool fetches them again on demand so you don't have to ask "
    "the researcher to re-attach.\n\n"
    "Behaviour:\n"
    "  - Scripts: full text returned inline (capped at 96 KB; "
    "longer files come back head+tail-truncated with an explicit "
    "elision marker, so imports up top AND save / write calls at "
    "the bottom are both visible). Use this to "
    "recall a previously-attached do-file / .py before resubmitting "
    "or proposing edits.\n"
    "  - Images: returned as an MCP image content block so you can "
    "see the plot. PDF / EPS are rasterised first.\n\n"
    "Datasets (.csv / .dta / .parquet / .tsv / .jsonl / .ndjson / "
    ".rds) are NOT retrievable through this tool. That boundary "
    "is the SDC line. Use get_schema for column names / dtypes, "
    "or write a script that reads the dataset.\n\n"
    "Path safety: ``name`` is treated as a basename. Any directory "
    "component is stripped before resolving against cwd. Paths "
    "outside cwd are refused.\n\n"
    "Arguments:\n"
    "  name: basename of the file (e.g., 'reg_v9.do', "
    "'residuals.png'). Must exist in the session cwd or one of "
    "its plot subdirectories."
)


# ---------------------------------------------------------------------------
# The seven tools.
# ---------------------------------------------------------------------------
# Order matches the order in which they appear to the model in the
# system-prompt enumeration in ``app.py``.

def _spec(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...],
    openai_description: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
        required=required,
        openai_description=openai_description,
    )


# ---------------------------------------------------------------------------
# OpenAI-specific lean descriptions
# ---------------------------------------------------------------------------
#
# Sent on every ``responses.create()`` call as part of the ``tools``
# array. OpenAI auto-caches the array at a 50% discount; trimming
# tokens here is worth ~5x what the same trim buys on Anthropic
# (which caches at 90%). Each variant must convey *when to call*;
# full args are in the JSON schema below it.

_GET_SCHEMA_DESC_OAI = (
    "Return a dataset's structural summary (variable names, types, "
    "labels, observation count). Call this before writing any analysis "
    "script. Supported file types: .dta / .rds / .csv / .tsv / "
    ".parquet / .jsonl. The 'depth' argument controls detail "
    "(names_only, names_types, names_types_labels, "
    "names_types_labels_summary); the researcher's policy sets a "
    "ceiling and over-ceiling requests are denied."
)

_SUBMIT_SCRIPT_DESC_OAI = (
    "Run an R, Stata, or Python script against the researcher's data. "
    "The script must emit structured results via the nora runtime "
    "helpers (nora$result(...) / nora$from_lm(...) in R, "
    "nora_result_* in Stata, nora.result(...) / nora.from_lm(...) in "
    "Python). Sanitized payload + result ID return to you; raw "
    "stdout goes to the researcher only. Always pass 'source_dataset' "
    "when the script reads a known file so silent row drops "
    "(NA-drops, subset, listwise deletion) are flagged."
)

_RECALL_CONVERSATION_DESC_OAI = (
    "Search this session's older archived turns for content that has "
    "scrolled out of your context. The most recent ~20 turns are "
    "already in your context (auto-loaded on session open), so use "
    "this only for DEEPER lookups (older history, or keyword search "
    "like 'what did I say about X at the start'). Don't call for "
    "content already visible. Args: query (substring), tail (last N "
    "turns), context (neighbors per match, default 2), max_chars "
    "(default ~8000)."
)

_READ_ATTACHED_FILE_DESC_OAI = (
    "Re-fetch a file the researcher attached earlier (.py / .do / .r "
    "/ .rmd as inline text, .png / .jpg / .pdf / .eps as a vision "
    "content block; PDF/EPS are rasterised). Use when the file's "
    "content has scrolled out of your context but the file is still "
    "on disk in the session cwd. Datasets (.csv / .dta / .parquet / "
    "etc.) are NOT retrievable through this tool; use get_schema or "
    "write a script. 'name' is treated as a basename."
)


def build_tool_specs() -> tuple[ToolSpec, ...]:
    """Construct the canonical tool specs.

    Function-form (rather than module-level constant) so
    ``request_data``'s description can resolve at call time —
    ``data_request.SUPPORTED_REQUEST_TYPES`` may not be importable at
    module-load time depending on import order.
    """
    return (
        _spec(
            "get_schema",
            _GET_SCHEMA_DESC,
            properties={
                "dataset": {"type": "string"},
                "depth": {"type": "string"},
            },
            required=("dataset",),
            openai_description=_GET_SCHEMA_DESC_OAI,
        ),
        _spec(
            "search_schema",
            _SEARCH_SCHEMA_DESC,
            properties={
                "dataset": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            required=("dataset", "query"),
        ),
        _spec(
            "request_data",
            _request_data_desc(),
            properties={
                "dataset": {"type": "string"},
                "request_type": {"type": "string"},
                "variable": {"type": "string"},
                "variable2": {"type": "string"},
            },
            required=("dataset", "request_type", "variable"),
        ),
        _spec(
            "submit_script",
            _SUBMIT_SCRIPT_DESC,
            properties={
                "language": {"type": "string"},
                "code": {"type": "string"},
                "label": {"type": "string"},
                "source_dataset": {"type": "string"},
            },
            required=("language", "code", "label"),
            openai_description=_SUBMIT_SCRIPT_DESC_OAI,
        ),
        _spec(
            "submit_script_file",
            _SUBMIT_SCRIPT_FILE_DESC,
            properties={
                "name": {"type": "string"},
                "language": {"type": "string"},
                "label": {"type": "string"},
                "source_dataset": {"type": "string"},
            },
            required=("name",),
        ),
        _spec(
            "expand_result",
            _EXPAND_RESULT_DESC,
            properties={
                "result_id": {"type": "string"},
                "view": {"type": "string"},
                "session_path": {"type": "string"},
            },
            required=("result_id",),
        ),
        _spec(
            "list_results",
            _LIST_RESULTS_DESC,
            properties={},
            required=(),
        ),
        _spec(
            "list_results_global",
            _LIST_RESULTS_GLOBAL_DESC,
            properties={
                "query": {"type": "string"},
            },
            required=(),
        ),
        _spec(
            "recall_conversation",
            _RECALL_CONVERSATION_DESC,
            properties={
                "query": {"type": "string"},
                "tail": {"type": "integer"},
                "context": {"type": "integer"},
                "max_chars": {"type": "integer"},
            },
            required=(),
            openai_description=_RECALL_CONVERSATION_DESC_OAI,
        ),
        _spec(
            "read_attached_file",
            _READ_ATTACHED_FILE_DESC,
            properties={
                "name": {"type": "string"},
            },
            required=("name",),
            openai_description=_READ_ATTACHED_FILE_DESC_OAI,
        ),
    )


def tool_spec(name: str) -> ToolSpec:
    """Look up a single tool spec by name."""
    for spec in build_tool_specs():
        if spec.name == name:
            return spec
    raise KeyError(f"no tool spec named {name!r}")

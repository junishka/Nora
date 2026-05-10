"""Turn-grouped reader over ``.nora/chat_history.jsonl``.

The chat log is persisted as one typed event per line (``user_message``,
``assistant_text``, ``assistant_thinking``, ``tool_call``,
``tool_result``). That layout is fine for persistence but awkward for
anyone trying to reason about the conversation turn-by-turn — a single
turn is one ``user_message`` plus every assistant/tool event that
follows it until the next user message.

This module groups those loose events into ``Turn`` records with a
single-line summary of each tool call and any stored result IDs the
tool_result surfaced. Both the warm-start prefix in ``ui.py`` and
the ``recall_conversation`` tool in ``tools.py`` consume the same
helper so the two views of history stay in sync.

Event schema drift: we never-remove fields from persisted events, so
this reader tolerates missing timestamps / missing call_ids (older
sessions wrote neither). Newer sessions carry both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolUse:
    """One tool call inside a turn, paired with the result if any.

    ``label`` is a short human-readable summary (language + submit_script
    label, or dataset + depth for get_schema, etc.) suitable for a
    compact "what happened" listing — callers don't have to re-derive
    it from ``input``.
    """
    name: str                       # short name, e.g. "submit_script"
    input: dict[str, Any]           # raw tool input the model sent
    label: str                      # one-line human summary
    call_id: str | None = None
    is_error: bool = False
    # All sanitized-store ids the call produced. submit_script under
    # the multi-result wire format returns N ids per call (one per
    # nora_result_* helper that emitted); expand_result and other
    # single-result tools return one. Empty list when none.
    result_ids: list[str] = field(default_factory=list)


@dataclass
class Turn:
    """One user-message → assistant-response exchange.

    Assistant text and thinking traces come in multiple blocks when the
    turn involves tool use; we join them with ``\n\n`` so consumers get
    the whole side of the exchange as one string without caring about
    the block boundary.
    """
    index: int                      # 0-based position in the full log
    user: str
    assistant: str
    thinking: str                   # joined thinking blocks; "" if none
    tools: list[ToolUse] = field(default_factory=list)
    result_ids: list[str] = field(default_factory=list)
    attachments: int = 0            # images the user attached
    timestamp: str | None = None    # ISO 8601, when available


def read_turns(cwd: Path | None) -> list[Turn]:
    """Read and turn-group the session's persisted chat log.

    Returns an empty list when the log is missing or unreadable. Never
    raises — this is a best-effort reader; callers generally want to
    fall back gracefully rather than crash on a corrupted log line.
    """
    if cwd is None:
        return []
    path = cwd / ".nora" / "chat_history.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return []

    # First pass: collect raw events in order. Skips blank and
    # unparseable lines rather than failing the whole read.
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    # Second pass: group by turn. A turn starts at each user_message;
    # everything after it (until the next user_message) belongs to the
    # same turn. Events before the first user_message (possible in
    # older logs that started with a system banner) are dropped.
    turns: list[Turn] = []
    current_user: dict[str, Any] | None = None
    current_assistant: list[str] = []
    current_thinking: list[str] = []
    current_tools: list[ToolUse] = []
    current_timestamp: str | None = None
    # Track tool_call call_ids so tool_result can pair by id.
    tools_by_call_id: dict[str, ToolUse] = {}

    def _attachment_count(value: Any) -> int:
        """Normalize persisted ``attachments`` into a count.

        Older sessions stored an integer image count. Newer web-UI
        sessions may store a list of attached script filenames so
        replay can render the same chips on reload. The turn-grouped
        reader only needs a stable count and should never crash on
        either shape.
        """
        if isinstance(value, list):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _flush() -> None:
        if current_user is None:
            return
        result_ids = [
            rid
            for t in current_tools
            for rid in t.result_ids
        ]
        turns.append(Turn(
            index=len(turns),
            user=current_user.get("text", "") or "",
            assistant="\n\n".join(p for p in current_assistant if p).strip(),
            thinking="\n\n".join(p for p in current_thinking if p).strip(),
            tools=list(current_tools),
            result_ids=result_ids,
            attachments=_attachment_count(current_user.get("attachments", 0)),
            timestamp=current_timestamp,
        ))

    for rec in events:
        t = rec.get("type")
        if t == "user_message":
            _flush()
            current_user = rec
            current_assistant = []
            current_thinking = []
            current_tools = []
            tools_by_call_id = {}
            current_timestamp = rec.get("timestamp")
        elif current_user is None:
            # Orphan event before any user message — skip. Older
            # sessions sometimes have a system banner at the top of
            # the log; ignoring it keeps turn indices honest.
            continue
        elif t == "assistant_text":
            text = rec.get("text") or ""
            if text:
                current_assistant.append(text)
        elif t == "assistant_thinking":
            text = rec.get("text") or ""
            if text:
                current_thinking.append(text)
        elif t == "tool_call":
            name_raw = rec.get("name", "") or ""
            short = name_raw.split("__")[-1] if name_raw else ""
            input_args = rec.get("input") or {}
            if not isinstance(input_args, dict):
                input_args = {}
            use = ToolUse(
                name=short,
                input=input_args,
                label=summarize_tool_call(short, input_args),
                call_id=rec.get("call_id"),
            )
            current_tools.append(use)
            if use.call_id:
                tools_by_call_id[use.call_id] = use
        elif t == "tool_result":
            call_id = rec.get("call_id")
            use = tools_by_call_id.get(call_id) if call_id else None
            if use is not None:
                use.is_error = bool(rec.get("is_error", False))
                use.result_ids = _extract_result_ids(rec.get("text", ""))

    _flush()
    return turns


# Quick prefix-match for the JSONL ``"type"`` field. Every event the
# runner writes has ``"type"`` first because we serialise from a
# Python dict whose first key is ``type`` — the regex is a
# correctness-preserving shortcut, with the full ``json.loads`` path
# below as a fallback for any line where the prefix doesn't match.
_TYPE_PREFIX_RE = re.compile(rb'^\s*\{\s*"type"\s*:\s*"([^"]+)"')

# Event types that carry the heavy payloads (raw stdout/stderr, plot
# thumbnails, tool input echo) we don't need for a session snapshot.
# Skipping ``json.loads`` on these lines is the difference between an
# O(n) full parse of the whole UI replay log and an O(n) byte scan.
_HEAVY_EVENT_TYPES = frozenset((
    b"tool_call", b"tool_result", b"assistant_thinking",
))


def read_last_turn_summary(cwd: Path | None) -> tuple[int, str, str]:
    """Lightweight scan over ``chat_history.jsonl`` for session_state.

    Returns ``(turn_count, last_user, last_assistant)``. The pair is
    sourced from the most recent turn that has a ``user_message``;
    ``last_assistant`` defaults to ``""`` when the turn is in-flight.

    This avoids ``read_turns``'s full per-event ``json.loads`` —
    ``tool_result`` payloads in particular can carry tens of KB of
    plot data each, and a long session can hit that cost on every
    successful turn. We keep ``read_turns`` for callers that need
    grouped Turn objects (recall, warm-start prefix, tests); session
    state writes go through this lighter path.
    """
    if cwd is None:
        return (0, "", "")
    path = cwd / ".nora" / "chat_history.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return (0, "", "")

    turn_count = 0
    last_user = ""
    last_assistant_parts: list[str] = []
    try:
        with path.open("rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                m = _TYPE_PREFIX_RE.match(line)
                if m is not None and m.group(1) in _HEAVY_EVENT_TYPES:
                    # Skip parse — these don't contribute to the
                    # snapshot and may carry kilobytes of payload.
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(rec, dict):
                    continue
                t = rec.get("type")
                if t == "user_message":
                    turn_count += 1
                    last_user = rec.get("text", "") or ""
                    last_assistant_parts = []
                elif t == "assistant_text" and turn_count > 0:
                    text = rec.get("text", "")
                    if text:
                        last_assistant_parts.append(text)
    except OSError:
        return (0, "", "")

    last_assistant = "\n\n".join(p for p in last_assistant_parts if p).strip()
    return (turn_count, last_user, last_assistant)


def summarize_tool_call(short_name: str, input_args: dict[str, Any]) -> str:
    """One-line human-readable summary of a tool call's arguments.

    Used by both the warm-start prefix and the recall_conversation
    tool so the phrasing stays identical across the two views.
    Returns an empty string for unknown tool names; callers should
    fall back to a bare ``[name]`` tag in that case.
    """
    if not isinstance(input_args, dict):
        return ""
    if short_name == "submit_script":
        lang = input_args.get("language") or ""
        label = input_args.get("label") or "(no label)"
        return f"{lang}: {label}" if lang else str(label)
    if short_name == "submit_script_file":
        name = input_args.get("name") or "(unnamed)"
        label = input_args.get("label") or ""
        return f"{name} — {label}" if label else str(name)
    if short_name == "get_schema":
        return (
            f"{input_args.get('dataset', '')} at "
            f"{input_args.get('depth', '')}"
        )
    if short_name == "search_schema":
        q = input_args.get("query") or ""
        ds = input_args.get("dataset") or ""
        return f"{q!r} in {ds}" if q else ds
    if short_name == "request_data":
        return (
            f"{input_args.get('request_type', '')} on "
            f"{input_args.get('variable', '')} "
            f"({input_args.get('dataset', '')})"
        )
    if short_name == "expand_result":
        rid = str(input_args.get("result_id", ""))
        view = input_args.get("view") or ""
        return f"{rid} (view={view})" if view else rid
    if short_name == "list_results":
        limit = input_args.get("limit")
        return f"limit={limit}" if limit else ""
    if short_name == "list_results_global":
        q = input_args.get("query") or ""
        return f"query={q!r}" if q else ""
    if short_name == "read_attached_file":
        return str(input_args.get("name", ""))
    if short_name == "recall_conversation":
        bits: list[str] = []
        q = input_args.get("query")
        if q:
            bits.append(f"query={q!r}")
        tail = input_args.get("tail")
        if tail:
            bits.append(f"tail={tail}")
        return ", ".join(bits)
    if short_name == "compose_results":
        # Layout-spec composer — surface the dimensions so the recall
        # snippet conveys "this turn built a 3-column × 4-row table"
        # rather than just "[compose_results]".
        spec = input_args.get("spec")
        cols = spec.get("columns") if isinstance(spec, dict) else None
        groups = spec.get("groups") if isinstance(spec, dict) else None
        ncols = len(cols) if isinstance(cols, list) else 0
        nrows = 0
        if isinstance(groups, list):
            for g in groups:
                gr = g.get("rows") if isinstance(g, dict) else None
                if isinstance(gr, list):
                    nrows += len(gr)
        if ncols or nrows:
            return f"{ncols} cols × {nrows} rows"
        return ""
    if short_name == "list_session_files":
        # No required args; surface the optional ``kinds`` filter
        # when set so a recall like ``list_session_files [data]``
        # reads sensibly.
        kinds = input_args.get("kinds")
        if isinstance(kinds, list) and kinds:
            return ",".join(str(k) for k in kinds)
        return ""
    if short_name == "search_in_session_files":
        q = input_args.get("query") or ""
        kinds = input_args.get("kinds")
        kinds_part = ""
        if isinstance(kinds, list) and kinds:
            kinds_part = ",".join(str(k) for k in kinds)
        if q and kinds_part:
            return f"{q!r} ({kinds_part})"
        if q:
            return f"{q!r}"
        return kinds_part
    return ""


def build_context_prefix(
    cwd: Path | None,
    *,
    results: list[Any] | None = None,
) -> str:
    """Render the warm-start prefix that Nora prepends to the first
    user message on a fresh SDK client.

    The prefix is wrapped in a clearly-marked block so Claude parses
    it as background, not a new request. It has two sections:

    1. **Recent analytical results** — one line per stored result
       (id, label, analysis type). In this product "what happened" is
       often a regression or crosstab, not just prose; giving Claude
       concrete ``result_id``s to feed into ``expand_result`` means
       resume ties directly to the analytical work, not just the
       conversation around it.
    2. **Conversation turns** — the last N turns in chronological
       order, each rendered with user line, tool-call summaries
       (with ``result_id`` pointers where available), and the
       assistant reply.

    Returns an empty string when there's nothing to resume from
    (no chat history AND no stored results, or ``cwd`` is None).

    ``results`` is an injection point for tests and for the ui.py
    shim that reads from the real store — pass a list of objects
    exposing ``id`` / ``label`` / ``analysis_type`` / ``created_at``.
    """
    if cwd is None:
        return ""
    turns = read_turns(cwd)

    rows_with_ts = [r for r in (results or []) if getattr(r, "created_at", None)]
    rows_with_ts.sort(key=lambda r: r.created_at, reverse=True)

    if not turns and not rows_with_ts:
        return ""

    MAX_TURNS = 20
    MAX_RESULTS = 10
    # Per-side per-turn density cap. Lowered from 1500 → 1000 after
    # observing that real sessions hit TOTAL_CAP after ~8 turns at the
    # higher density; tighter per-turn truncation lets more turns fit
    # in the same budget. The model has list_results / expand_result
    # for full payloads, so a truncation marker on a long assistant
    # turn is recoverable rather than lossy.
    PER_FIELD_CAP = 1000
    # Tool labels come from raw tool inputs (submit_script.label,
    # filenames, query strings) — script-controllable text that
    # could carry instruction-shaped content or oversized blobs.
    # Cap them tighter than user/assistant prose; one line per tool
    # call is plenty for a "what happened" summary.
    TOOL_LABEL_CAP = 200
    # Total prefix budget, in characters. Lowered from 20_000 → 12_000.
    # At ~4 chars/token this is roughly 3,000 tokens on session resume
    # (down from ~5,000), recovered exactly once per resume. The cap
    # drops oldest turns when the budget runs out (newest-first
    # rendering); MAX_TURNS continues to be a hard ceiling.
    TOTAL_CAP = 12_000

    total_turns = len(turns)
    picked = turns[-MAX_TURNS:]
    omitted = total_turns - len(picked)

    def _cap(s: str) -> str:
        # User and assistant text get re-injected verbatim into the
        # next turn's prompt. A prior message containing a bidi
        # override or "\n###System: ..."-shaped content (either typed
        # by the researcher OR echoed by Claude from a tool result it
        # didn't filter) would otherwise survive into the new turn as
        # a fresh injection vector. Strip control/bidi characters and
        # flatten whitespace via the same chokepoint every other
        # data-origin string crosses, then cap to PER_FIELD_CAP. The
        # wider cap is intentional — turn prose can run long; tool
        # labels keep the tighter TOOL_LABEL_CAP.
        from nora.text_safety import safe_text
        return safe_text(s, max_len=PER_FIELD_CAP)

    def _cap_label(s: str) -> str:
        # Strip control chars and cap. Tool labels are derived from
        # script-controlled strings (filenames, recall queries,
        # submit_script.label) — without scrubbing, a label like
        # "...\n\n[system] override: ..." would land in the
        # warm-start prefix verbatim.
        from nora.text_safety import safe_text
        return safe_text(s, max_len=TOOL_LABEL_CAP)

    def _render_turn(t: Turn) -> str:
        # Render a turn as: header line, user line, tool summaries,
        # assistant reply. Tool lines carry result_id pointers so
        # Claude can pull full payloads via expand_result.
        parts: list[str] = [f"[turn {t.index}]"]
        if t.user:
            parts.append(f"user: {_cap(t.user)}")
        for use in t.tools:
            # ``use.name`` is the tool's registered short name (a
            # bounded identifier) so it doesn't need the safe_text
            # treatment; ``use.label`` and the result_id list both
            # do, since they originate in script-controlled strings.
            tag = f"tool: [{use.name}]"
            if use.label:
                tag += f" {_cap_label(use.label)}"
            if use.result_ids:
                # result_ids are produced by the store (M-prefixed),
                # so they're parser-controlled. Cap the joined list
                # defensively in case a future schema change widens
                # the field.
                if len(use.result_ids) == 1:
                    tag += f" → result_id={_cap_label(use.result_ids[0])}"
                else:
                    joined = ",".join(use.result_ids)
                    tag += f" → result_ids={_cap_label(joined)}"
            if use.is_error:
                tag += " (error)"
            parts.append(tag)
        if t.assistant:
            parts.append(f"assistant: {_cap(t.assistant)}")
        return "\n".join(parts)

    # Newest-first rendering so the total-cap budget drops oldest
    # turns if we overflow; flip back to chronological at the end.
    blocks: list[str] = []
    running = 0
    for t in reversed(picked):
        block = _render_turn(t)
        # The "always keep at least one block" rule used to admit a
        # single oversized first turn unbounded — a turn whose tool
        # labels alone exceeded TOTAL_CAP would still be shipped
        # whole. Guard the head-of-budget case by capping the block
        # itself to TOTAL_CAP rather than letting it through.
        if not blocks and len(block) > TOTAL_CAP:
            block = block[: TOTAL_CAP - len("…[turn truncated]")] + "…[turn truncated]"
        cost = len(block) + 2
        if running + cost > TOTAL_CAP and blocks:
            break
        running += cost
        blocks.append(block)
    blocks.reverse()

    # Recent-results block: the researcher's actual analytical work
    # catalogued for resume. Each line has id + label + type so the
    # model can match a question ("that OLS we ran") to an id and
    # call expand_result for the numbers.
    results_lines: list[str] = []
    # Re-sanitize ``label`` and ``analysis_type`` at READ time as
    # well as at insert time. Insert-time sanitization handles new
    # rows, but the warm-start prefix re-renders rows that may have
    # been written by an older Nora binary (pre-sanitization), or by
    # a partially-corrupted DB write. Without a read-time pass, a
    # legacy row carrying raw newlines / "[system] override:" /
    # bidi-overrides in its ``label`` would land verbatim in the
    # next-turn prompt prefix and either inject instructions or
    # smuggle data the model otherwise wouldn't see. Trust-on-read
    # symmetry with the in-memory tool-label path (``_cap_label``
    # above) keeps the prompt boundary consistent regardless of how
    # old the underlying row is.
    from nora.text_safety import safe_text, safe_key
    for r in rows_with_ts[:MAX_RESULTS]:
        rid = str(getattr(r, "id", "") or "")
        raw_label = str(getattr(r, "label", "") or "")
        raw_atype = str(getattr(r, "analysis_type", "") or "")
        label = safe_text(raw_label, max_len=TOOL_LABEL_CAP)
        # ``analysis_type`` is a parser-controlled identifier
        # (linear_regression, frequency_table, …) — narrow
        # ``safe_key`` cap is correct here.
        atype = safe_key(raw_atype) if raw_atype else ""
        atype_tag = f" [{atype}]" if atype else ""
        results_lines.append(f"  - {rid}: {label}{atype_tag}")
    if len(rows_with_ts) > MAX_RESULTS:
        older = len(rows_with_ts) - MAX_RESULTS
        results_lines.append(
            f"  (… {older} older result{'s' if older != 1 else ''}; "
            f"use list_results / expand_result for details)"
        )
    results_block = ""
    if results_lines:
        header_r = (
            f"[Analyses already produced in this session "
            f"({len(rows_with_ts)} stored, newest first; "
            f"call expand_result(id) for any payload):]"
        )
        results_block = header_r + "\n" + "\n".join(results_lines)

    # Reframed from "Prior conversation context" to "Session state".
    # The previous wording read as background chatter that PRECEDED
    # the current task; the model parsed it as history and would
    # silently re-derive or replace listed work on a fresh request
    # after reload (the "drift on resume" failure mode). The new
    # wording tells the model this is the current state of analytical
    # work, plus an explicit directive to build on it rather than
    # around it. The system prompt has a paired rule for the same
    # thing — both surfaces matter because the prefix sits with the
    # data and the system prompt sits with the instructions.
    header = (
        "[Session state at resume — analyses and turns already "
        "completed in this session. Treat this as the current state "
        "of the analytical work, not background. Build on what is "
        "here; use list_results / expand_result for any id you need "
        "to inspect."
    )
    if omitted > 0:
        header += f" {omitted} earlier turns omitted,"
    header += f" showing last {len(blocks)} of {total_turns} turns.]"
    footer = "[End of session state. Current message follows.]"

    sections = [header]
    if results_block:
        sections.append(results_block)
    if blocks:
        sections.append("\n\n".join(blocks))
    sections.append(footer)
    return "\n\n".join(sections)


def _extract_result_ids(tool_result_text: str) -> list[str]:
    """Pull every stored ``result_id`` from a tool_result payload.

    Two shapes occur in the wild:
    - ``expand_result`` and a few other tools carry ``result_id`` at
      the top level (single id).
    - ``submit_script`` under the multi-result wire format carries a
      ``results`` list, one entry per helper call, each with its own
      ``result_id``. All are returned in emission order so resume /
      recall context can point to every payload the call produced,
      not just the first.

    Returns ``[]`` on shape mismatch. Best-effort, never raises.
    """
    if not tool_result_text or not isinstance(tool_result_text, str):
        return []
    s = tool_result_text.strip()
    if not s or s[0] != "{":
        return []
    try:
        payload = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    out: list[str] = []
    # Top-level id (expand_result, single-result tools, plus the
    # error-diagnostic id submit_script attaches when status is
    # "execution_failed"). Collected first so the diag id leads in
    # bare-failure cases.
    rid = payload.get("result_id") or payload.get("id")
    if isinstance(rid, str) and rid:
        out.append(rid)

    results = payload.get("results")
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict):
                continue
            rid = entry.get("result_id")
            if isinstance(rid, str) and rid and rid not in out:
                out.append(rid)
    return out

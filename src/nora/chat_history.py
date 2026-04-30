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
        return s if len(s) <= PER_FIELD_CAP else s[:PER_FIELD_CAP] + "…[truncated]"

    def _render_turn(t: Turn) -> str:
        # Render a turn as: header line, user line, tool summaries,
        # assistant reply. Tool lines carry result_id pointers so
        # Claude can pull full payloads via expand_result.
        parts: list[str] = [f"[turn {t.index}]"]
        if t.user:
            parts.append(f"user: {_cap(t.user)}")
        for use in t.tools:
            tag = f"tool: [{use.name}]"
            if use.label:
                tag += f" {use.label}"
            if use.result_ids:
                if len(use.result_ids) == 1:
                    tag += f" → result_id={use.result_ids[0]}"
                else:
                    tag += f" → result_ids={','.join(use.result_ids)}"
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
    for r in rows_with_ts[:MAX_RESULTS]:
        rid = str(getattr(r, "id", "") or "")
        label = str(getattr(r, "label", "") or "")
        atype = str(getattr(r, "analysis_type", "") or "")
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
            f"[Recent analytical results in this session "
            f"({len(rows_with_ts)} stored, newest first; "
            f"call expand_result(id) to see any payload):]"
        )
        results_block = header_r + "\n" + "\n".join(results_lines)

    header = "[Prior conversation context — resuming this session:"
    if omitted > 0:
        header += f" {omitted} earlier turns omitted,"
    header += f" showing last {len(blocks)} of {total_turns} turns]"
    footer = "[End of prior context. Current message follows.]"

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

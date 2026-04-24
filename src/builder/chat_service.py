"""Typed event stream over the Claude SDK client.

Both the terminal UI (``app.py``) and the web UI (``ui.py``) need to
iterate a chat turn and render what comes back. This module provides a
small, UI-agnostic event layer: the SDK produces AssistantMessage /
UserMessage / ResultMessage; ``run_turn`` converts those into a flat
stream of typed events that either renderer can consume without
reaching into SDK internals.

The terminal renderer in ``app.py`` still calls ``receive_response()``
directly — no point changing what works. The web UI always goes
through ``run_turn`` here.

Events kept lightweight: no rendering logic, no UI-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class AssistantText:
    """A block of assistant text. The SDK already gives us complete
    blocks per turn — not token-by-token streaming — so this is the
    whole text block, not a delta."""
    text: str


@dataclass
class AssistantThinking:
    """Claude's reasoning trace, when thinking is enabled. Shown to the
    researcher but visually subdued (it's not the main response)."""
    text: str


@dataclass
class ToolCall:
    """Claude is calling one of the MCP tools. ``input`` is the JSON
    args Claude sent; ``call_id`` ties this to the matching
    ToolCallResult."""
    name: str
    input: dict[str, Any]
    call_id: str


@dataclass
class ToolCallResult:
    """The tool returned. ``text`` is the MCP text-content payload
    (JSON for our tools). ``is_error`` marks explicit failures.

    ``run_dir`` and ``language`` are hints ``tools.submit_script`` /
    ``tools.expand_result`` inject so the UI can render the native
    R/Stata output alongside the sanitized payload and offer the
    right "Open in R/Stata" action.
    """
    call_id: str
    text: str
    is_error: bool
    run_dir: str | None = None
    language: str | None = None


@dataclass
class TurnDone:
    """Turn completed cleanly. Token / cost fields are optional — the
    subscription-auth path doesn't carry cost_usd.

    The four token fields together describe the *full* conversation
    context, which the UI sums for the "Context X / Y" indicator:

    - ``input_tokens``: new tokens in this turn's prompt (not cached).
    - ``cache_read_input_tokens``: prior context served from the
      Anthropic prompt cache. Invisible to ``input_tokens`` but still
      occupies the model's context window.
    - ``cache_creation_input_tokens``: tokens written to the cache
      this turn (also in the window).
    - ``output_tokens``: what Claude just produced. Becomes part of
      next turn's context, so we count it now.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class AuthFailure:
    """The SDK or server rejected the request because auth is missing
    or invalid. UI layer decides how to surface this (terminal prints
    a hint, web UI shows an auth banner)."""
    reason: str


@dataclass
class TurnError:
    """Something else went wrong. ``message`` is human-readable."""
    message: str


Event = Union[
    AssistantText,
    AssistantThinking,
    ToolCall,
    ToolCallResult,
    TurnDone,
    AuthFailure,
    TurnError,
]


# ---------------------------------------------------------------------------
# Turn driver
# ---------------------------------------------------------------------------

# Sentinels for the SDK's signaled auth / billing failures. Kept in
# sync with app.py's AssistantMessage.error enum values.
_AUTH_FAILURE = "auth_failure"
_BILLING_FAILURE = "billing_failure"


async def run_turn(
    client: ClaudeSDKClient,
    prompt: str,
    images: list[dict[str, Any]] | None = None,
) -> AsyncIterator[Event]:
    """Send ``prompt`` to the SDK and yield a flat stream of events.

    The async generator terminates when the turn's ``ResultMessage``
    arrives (after yielding ``TurnDone``). Exceptions propagate as
    ``TurnError`` rather than raising.

    ``images`` is an optional list of ``{"data": <base64>, "mime": ...}``
    dicts. When present, the message is sent as a structured
    user-message with text + image content blocks so Claude's vision
    model can see the attachments.
    """
    try:
        if images:
            await client.query(_image_message_iter(prompt, images))
        else:
            await client.query(prompt)
    except Exception as e:  # noqa: BLE001 — SDK may raise various things
        yield TurnError(message=f"failed to send prompt: {e}")
        return

    # Track the largest prompt-side usage seen during the turn, rather
    # than yielding TurnDone on the first ResultMessage. Tool-use
    # turns can produce multiple internal round-trips, each with its
    # own ResultMessage: taking only the first one makes the
    # context-usage chip jump around (first round reports small input
    # before the tool_result is folded in). Max across all
    # ResultMessages gives the turn's actual peak prompt size.
    max_input = 0
    max_output = 0
    max_cache_read = 0
    max_cache_creation = 0
    max_prompt_total = -1   # sum of prompt-side counters; used to pick the peak
    last_cost: float | None = None
    saw_result = False

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            err = getattr(msg, "error", None)
            if err == _AUTH_FAILURE:
                yield AuthFailure(reason="auth failure from server")
                return
            if err == _BILLING_FAILURE:
                yield AuthFailure(reason="billing failure — check your account")
                return
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text.strip():
                        yield AssistantText(text=block.text)
                elif isinstance(block, ThinkingBlock):
                    if block.thinking.strip():
                        yield AssistantThinking(text=block.thinking)
                elif isinstance(block, ToolUseBlock):
                    yield ToolCall(
                        name=block.name,
                        input=dict(block.input or {}),
                        call_id=block.id,
                    )
                elif isinstance(block, ToolResultBlock):
                    # Unusual — tool results usually arrive in UserMessage.
                    yield _tool_result_event(block)
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        yield _tool_result_event(block)
        elif isinstance(msg, ResultMessage):
            saw_result = True
            usage = msg.usage or {}
            inp = _maybe_int(usage.get("input_tokens")) or 0
            outp = _maybe_int(usage.get("output_tokens")) or 0
            cr = _maybe_int(usage.get("cache_read_input_tokens")) or 0
            cc = _maybe_int(usage.get("cache_creation_input_tokens")) or 0
            prompt_total = inp + cr + cc
            if prompt_total >= max_prompt_total:
                # Peak prompt sub-call of this turn so far. Capture
                # ALL fields at that moment — including output — so
                # the reported snapshot is self-consistent.
                max_prompt_total = prompt_total
                max_input = inp
                max_output = outp
                max_cache_read = cr
                max_cache_creation = cc
            if msg.total_cost_usd is not None:
                last_cost = msg.total_cost_usd
            # Don't return: the SDK can follow a ResultMessage with
            # more rounds if tools are still running. Let the
            # iterator finish naturally.
        # SystemMessage / StreamEvent / RateLimitEvent: ignored — they
        # don't carry anything the researcher or the UI needs to render.

    if saw_result:
        yield TurnDone(
            input_tokens=max_input,
            output_tokens=max_output,
            cache_read_input_tokens=max_cache_read,
            cache_creation_input_tokens=max_cache_creation,
            cost_usd=last_cost,
        )


def _tool_result_event(block: ToolResultBlock) -> ToolCallResult:
    text = _extract_tool_result_text(block.content)
    run_dir, language = _extract_hints(text)
    return ToolCallResult(
        call_id=block.tool_use_id,
        text=text,
        is_error=bool(block.is_error),
        run_dir=run_dir,
        language=language,
    )


def _extract_tool_result_text(
    content: str | list[dict[str, Any]] | None,
) -> str:
    """Normalize ToolResultBlock.content (which the SDK allows to be
    str | list[{type, text}] | None) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(parts)


def _extract_hints(text: str) -> tuple[str | None, str | None]:
    """Peel off the ``_run_dir`` and ``_language`` hints
    ``tools.submit_script`` / ``tools.expand_result`` inject so the
    UI can show raw R/Stata output and pick the right "Open in …"
    button. Returns ``(None, None)`` when the tool result isn't a
    script-style response."""
    if not text.strip():
        return None, None
    try:
        import json
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    rd = parsed.get("_run_dir")
    lang = parsed.get("_language")
    return (
        rd if isinstance(rd, str) else None,
        lang if isinstance(lang, str) else None,
    )


async def _image_message_iter(
    text: str, images: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    """Async-yield one structured user message carrying text + image
    content blocks. The SDK's ``client.query`` accepts either a
    string (short-circuited to a plain text user message) or an
    async iterable of message dicts; for vision we need the latter
    so we can attach image blocks in the Anthropic API shape:

        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/png",
                                      "data": "<b64>"}}

    Only one message is yielded per call. Everything after that is
    streamed back by the SDK as model output.
    """
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("mime", "image/png"),
                "data": img.get("data", ""),
            },
        })
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def _maybe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

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
    subscription-auth path doesn't carry cost_usd."""
    input_tokens: int | None = None
    output_tokens: int | None = None
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
    client: ClaudeSDKClient, prompt: str
) -> AsyncIterator[Event]:
    """Send ``prompt`` to the SDK and yield a flat stream of events.

    The async generator terminates when the turn's ``ResultMessage``
    arrives (after yielding ``TurnDone``). Exceptions propagate as
    ``TurnError`` rather than raising.
    """
    try:
        await client.query(prompt)
    except Exception as e:  # noqa: BLE001 — SDK may raise various things
        yield TurnError(message=f"failed to send prompt: {e}")
        return

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
            yield TurnDone(
                input_tokens=_maybe_int(
                    (msg.usage or {}).get("input_tokens")
                ),
                output_tokens=_maybe_int(
                    (msg.usage or {}).get("output_tokens")
                ),
                cost_usd=msg.total_cost_usd,
            )
            return
        # SystemMessage / StreamEvent / RateLimitEvent: ignored — they
        # don't carry anything the researcher or the UI needs to render.


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


def _maybe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

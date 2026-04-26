"""Provider-neutral session interface and shared event types.

A ``ProviderSession`` wraps whatever per-provider machinery is needed
to drive a single chat turn (Claude Agent SDK client, OpenAI Responses
API streaming session, etc.) and exposes the same ``send()`` /
``set_model()`` / ``close()`` surface to the rest of Nora.

The Event dataclasses live here as the canonical home: every provider
yields the same shapes regardless of which underlying SDK it wraps.
``chat_service.py`` re-exports them for backward compatibility with
existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, Union, runtime_checkable


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------
#
# These are the contract every provider implements. Anything new a
# provider wants to surface should land here so neither frontend (web,
# terminal) needs provider-specific rendering branches.


@dataclass
class AssistantText:
    """A block of assistant text. Providers emit complete blocks per
    turn rather than token-by-token streaming, so this is the whole
    text block, not a delta."""
    text: str


@dataclass
class AssistantThinking:
    """The model's reasoning trace, when thinking is enabled. Shown to
    the researcher but visually subdued (it's not the main response).
    OpenAI's o-series reasoning text is not user-readable; the OpenAI
    provider therefore does not emit this event."""
    text: str


@dataclass
class ToolCall:
    """The model is calling one of the Nora MCP tools. ``input`` is
    the JSON args; ``call_id`` ties this to the matching
    ToolCallResult."""
    name: str
    input: dict[str, Any]
    call_id: str


@dataclass
class ToolCallResult:
    """The tool returned. ``text`` is the MCP text-content payload
    (JSON for Nora's tools). ``is_error`` marks explicit failures.

    ``run_dir`` and ``language`` are hints ``tools.submit_script`` /
    ``tools.expand_result`` inject so the UI can render the native
    R/Stata output alongside the sanitized payload and offer the right
    "Open in R/Stata" action.
    """
    call_id: str
    text: str
    is_error: bool
    run_dir: str | None = None
    language: str | None = None


@dataclass
class TurnDone:
    """Turn completed cleanly. Token / cost fields are optional — the
    Anthropic subscription path doesn't carry ``cost_usd``; the OpenAI
    path doesn't populate the cache fields (no equivalent concept).

    For Anthropic, the four token fields together describe the *full*
    conversation context, which the UI sums for the "Context X / Y"
    indicator:

    - ``input_tokens``: new tokens in this turn's prompt (not cached).
    - ``cache_read_input_tokens``: prior context served from the
      Anthropic prompt cache. Invisible to ``input_tokens`` but still
      occupies the model's context window.
    - ``cache_creation_input_tokens``: tokens written to the cache
      this turn (also in the window).
    - ``output_tokens``: what the model just produced. Becomes part of
      next turn's context, so we count it now.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class AuthFailure:
    """The provider rejected the request because auth is missing or
    invalid. UI layer decides how to surface this (terminal prints a
    hint, web UI shows an auth banner)."""
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
# Session protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderSession(Protocol):
    """The narrow surface every provider must implement.

    ``open()`` constructs whatever underlying client the provider
    needs. ``send()`` drives one chat turn and yields a flat stream of
    provider-neutral events. ``set_model()`` swaps the active model;
    implementations may do this in place (Anthropic SDK supports that)
    or by tearing down and reopening (OpenAI). ``close()`` releases
    resources.

    The session is bound to a single working directory and system
    prompt at construction. Switching cwd means constructing a new
    session — there is no ``set_cwd``.
    """

    async def open(self) -> None:
        """Acquire the underlying client. Idempotent."""
        ...

    async def close(self) -> None:
        """Release the underlying client. Idempotent."""
        ...

    def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one chat turn. Returns an async iterator of events.

        ``images`` is an optional list of ``{"data": <base64>, "mime":
        ...}`` dicts attached to the message as image content blocks.
        """
        ...

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch to a different model.

        Returns ``{"ok": True, "model": <id>, ...}`` on success or
        ``{"ok": False, "reason": <str>}`` on failure. The session
        remains usable in either case (failure leaves the previous
        model in effect; the conversation may have been reset).
        """
        ...


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------

# AuthMode values: ``"subscription"``, ``"api_key"``, ``"unknown"``.
# Each provider's ``detect_auth()`` returns one of these strings.
AuthMode = str

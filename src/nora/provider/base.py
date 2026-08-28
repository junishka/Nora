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

    ``post_turn_tokens`` is the canonical "context window occupied
    after this turn" the web chip displays. It is computed by each
    provider from its own usage fields, so the UI doesn't have to
    reconcile divergent provider semantics — see "field semantics"
    below for why a single sum across providers wouldn't have worked.
    Consumers that want the breakdown (cost attribution, cache
    diagnostics) read the granular fields directly.

    Field semantics:

    - ``input_tokens``: Anthropic emits "new tokens in this turn's
      prompt, not cached" (matching the SDK's usage object). OpenAI's
      Responses API instead reports the FULL prompt for the request
      including any cached prefix served via ``previous_response_id``;
      the OpenAI provider passes that through unchanged. The two
      providers' values are therefore not directly comparable — read
      ``post_turn_tokens`` for a comparable number.
    - ``cache_read_input_tokens``: Anthropic only. Prior context
      served from the prompt cache, invisible to ``input_tokens`` but
      still occupies the window. OpenAI leaves this ``None`` (its
      ``cached_tokens`` is a subset of ``input_tokens``, not
      additive).
    - ``cache_creation_input_tokens``: Anthropic only. Tokens written
      to the cache this turn (also in the window). OpenAI leaves
      this ``None``.
    - ``output_tokens``: what the model just produced. Folds back
      into the next turn's prompt accounting; included in
      ``post_turn_tokens`` so a long reply shows on the chip
      immediately rather than only after the next turn.
    - ``post_turn_tokens``: provider-canonical "context occupied
      after this turn." Anthropic computes ``input + cache_read +
      cache_creation + output``; OpenAI computes ``input + output``
      (its ``input_tokens`` already covers the full prompt). Older
      sessions persisted before this field existed leave it ``None``;
      consumers fall back to summing the granular fields above.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float | None = None
    post_turn_tokens: int | None = None


@dataclass
class AuthFailure:
    """The provider rejected the request because auth is missing or
    invalid. UI layer decides how to surface this (terminal prints a
    hint, web UI shows an auth banner)."""
    reason: str


@dataclass
class TurnError:
    """Something else went wrong. ``message`` is human-readable.

    ``context_reset``: True iff the failure means the provider's
    server-side conversation memory is gone and the next turn must
    re-prime via the warm-start prefix. The OpenAI provider sets
    this when ``previous_response_id`` expires; the Anthropic
    provider doesn't currently need it (the SDK rebuilds context
    from the on-disk transcript on every turn). The runner reads
    this flag in the failure-restoration branch and re-arms
    ``needs_context_prefix`` so the next turn doesn't silently
    start with no recoverable context. Defaults to False so existing
    error sites stay unchanged.

    ``context_preserved``: True iff the turn failed but the provider
    holds everything needed to deliver the turn's content with the
    next message (the OpenAI provider's mid-turn recovery stash —
    the failed turn's rounds live server-side and the unsent tool
    outputs ride along on the next send). The runner reads this in
    the failure-restoration branch and SKIPS the usual restoration
    (prefix re-arm, dataset-diff rollback, plot re-queue): the
    failed turn's prompt content reached the provider and is
    preserved, so re-sending it would duplicate context. Mutually
    exclusive with ``context_reset`` by construction. Defaults to
    False so existing error sites stay unchanged.
    """
    message: str
    context_reset: bool = False
    context_preserved: bool = False


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

    async def set_effort(self, effort: str) -> dict[str, Any]:
        """Switch the reasoning-effort level (``catalog.EFFORT_LEVELS``).

        Returns ``{"ok": True, "effort": <id>, ...}`` on success or
        ``{"ok": False, "reason": <str>}`` on failure. Providers that
        can only apply effort at client construction (the Anthropic
        Agent SDK passes it as a CLI flag at launch) report
        ``"requires_reopen": True`` when a live client exists — the
        caller (the runner) then closes the session so the next turn
        reopens with the new level and the warm-start context prefix
        carries the conversation. Providers that apply it per request
        (OpenAI) just take effect on the next message.
        """
        ...


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------

# AuthMode values: ``"subscription"``, ``"api_key"``, ``"unknown"``.
# Each provider's ``detect_auth()`` returns one of these strings.
AuthMode = str

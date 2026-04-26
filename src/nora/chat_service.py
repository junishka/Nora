"""Backward-compat re-exports of the provider-neutral event types.

The Event dataclasses (``AssistantText``, ``ToolCall``, ``TurnDone``,
…) used to live here together with the Anthropic-specific
``run_turn()`` driver. The driver moved into
``provider.anthropic.AnthropicSession.send`` and the Event types moved
into ``provider.base`` as the canonical home. This module remains so
existing imports (``from nora.chat_service import AssistantText``,
``from nora import chat_service``) keep working without churn.

New code should import from ``nora.provider`` instead.
"""

from __future__ import annotations

from nora.provider.base import (  # noqa: F401  (re-export)
    AssistantText,
    AssistantThinking,
    AuthFailure,
    Event,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)


__all__ = [
    "AssistantText",
    "AssistantThinking",
    "AuthFailure",
    "Event",
    "ToolCall",
    "ToolCallResult",
    "TurnDone",
    "TurnError",
]

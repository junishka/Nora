"""Nora provider package — multi-provider session interface.

Public surface (importable as ``from nora.provider import ...``):

- ``ProviderSession``: the protocol both Anthropic and OpenAI sessions
  satisfy. Wraps everything a chat turn needs (lifecycle, send,
  set_model) so the rest of Nora doesn't reach into provider SDKs
  directly.
- ``Event``-types (``AssistantText``, ``ToolCall``, …): the
  provider-neutral stream every session yields.
- ``ModelInfo`` + catalog helpers: which models exist, which provider
  owns each one, defaults.
- ``open_session(provider, cwd, model, system_prompt)``: factory that
  returns the right session class for the given provider id, with
  lazy import so missing optional deps (the OpenAI SDK) don't break
  Anthropic-only installs.
- ``detect_auth(provider)``: per-provider auth detection. Returns
  ``"subscription"``, ``"api_key"``, or ``"unknown"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nora.provider.base import (
    AssistantText,
    AssistantThinking,
    AuthFailure,
    AuthMode,
    Event,
    ProviderSession,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from nora.provider.catalog import (
    ALL_MODELS,
    ANTHROPIC_MODELS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    OPENAI_MODELS,
    PROVIDER_DEFAULTS,
    ModelInfo,
    get_model,
    models_for_provider,
    provider_for_model,
)


# Provider IDs the rest of the codebase recognises.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")


def open_session(
    provider: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    **kwargs: Any,
) -> ProviderSession:
    """Construct (but don't yet ``open()``) a provider session.

    Lazy-imports the per-provider module so an Anthropic-only install
    doesn't trip on a missing ``openai`` dep.
    """
    if provider == "anthropic":
        from nora.provider.anthropic import AnthropicSession
        return AnthropicSession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    if provider == "openai":
        # Lazy import so the openai SDK only loads when actually used.
        from nora.provider.openai import OpenAISession
        return OpenAISession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    raise ValueError(
        f"unknown provider: {provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def detect_auth(provider: str) -> AuthMode:
    """Per-provider auth detection.

    Anthropic checks ``ANTHROPIC_API_KEY`` env + ``~/.claude.json``
    OAuth account (Claude CLI subscription path). OpenAI checks the
    keyring for a stored ``OPENAI_API_KEY``-style credential.
    """
    if provider == "anthropic":
        from nora.provider.anthropic import detect_auth as _a
        return _a()
    if provider == "openai":
        from nora.provider.openai import detect_auth as _o
        return _o()
    raise ValueError(f"unknown provider: {provider!r}")


__all__ = [
    "ProviderSession",
    "Event",
    "AssistantText",
    "AssistantThinking",
    "AuthFailure",
    "AuthMode",
    "ToolCall",
    "ToolCallResult",
    "TurnDone",
    "TurnError",
    "ModelInfo",
    "ALL_MODELS",
    "ANTHROPIC_MODELS",
    "OPENAI_MODELS",
    "PROVIDER_DEFAULTS",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "SUPPORTED_PROVIDERS",
    "get_model",
    "models_for_provider",
    "provider_for_model",
    "open_session",
    "detect_auth",
]

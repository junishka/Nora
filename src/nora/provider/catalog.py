"""Cross-provider model catalog.

The model picker in both frontends reads from here. Each entry is a
``ModelInfo`` carrying everything the UI needs to render a row in the
picker (label, context window) plus the provider name needed to route
``set_model`` calls to the right session.

Adding a new model: extend the relevant per-provider tuple. The web
UI's ``list_models`` bridge call returns rows derived from this
catalog, filtered to providers the researcher has authenticated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model. ``provider`` matches the keys used by
    ``provider/__init__.py`` and ``auth.py`` (``"anthropic"``,
    ``"openai"``)."""

    id: str
    label: str
    context_window: int
    provider: str


# Pricing URLs surfaced as a "view pricing" link in the model picker.
# Per-provider — every model in a provider's catalog points at the
# same page; the page itself lists each variant. Centralised here so
# a future docs URL change is a one-line edit.
PROVIDER_PRICING_URLS: dict[str, str] = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openai": "https://openai.com/api/pricing/",
}


# ---------------------------------------------------------------------------
# Per-provider models
# ---------------------------------------------------------------------------

# Anthropic. The ``[1m]`` suffix on Sonnet/Opus requests the 1M-context
# beta via the Claude CLI / Agent SDK. Within the first 200k tokens,
# 1M costs the same as the standard tier; above 200k, the 1M tier is
# ~2x input and ~1.5x output. Labels are clean — context-window
# numbers live in the picker's right-side column, no need to repeat
# them in the name. Haiku is intentionally excluded for now: the
# Nora workload (multi-turn analysis with tool use) calls for the
# heavier models.
ANTHROPIC_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="claude-sonnet-4-6[1m]",
        label="Sonnet 4.6",
        context_window=1_000_000,
        provider="anthropic",
    ),
    ModelInfo(
        id="claude-opus-4-7[1m]",
        label="Opus 4.7",
        context_window=1_000_000,
        provider="anthropic",
    ),
)

# OpenAI. Two models: the flagship plus its extended-reasoning
# variant. ``gpt-5.5`` is the regular flagship; ``gpt-5.5-pro`` is the
# higher-reasoning model (slower, costlier, deeper). Names match the
# OpenAI Models API exactly so a researcher can cross-reference
# pricing and limits in OpenAI's own docs / billing dashboard.
# Context window: 1.05M tokens for both per OpenAI's published spec
# (the Models API itself doesn't expose this — it has to be hard-coded
# from OpenAI's docs and updated when they publish new variants).
OPENAI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gpt-5.5",
        label="GPT-5.5",
        context_window=1_050_000,
        provider="openai",
    ),
    ModelInfo(
        id="gpt-5.5-pro",
        label="GPT-5.5 Pro",
        context_window=1_050_000,
        provider="openai",
    ),
)


ALL_MODELS: tuple[ModelInfo, ...] = ANTHROPIC_MODELS + OPENAI_MODELS


# Default model per provider — what an "open a session for provider X"
# call uses when the researcher hasn't picked something explicitly.
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6[1m]",
    "openai": "gpt-5.5",
}


# Default provider when none is configured yet — only matters as a
# placeholder; the auth screen forces the researcher to pick before
# they reach the chat view.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]


def get_model(model_id: str) -> ModelInfo:
    """Look up a model by id. Raises ``KeyError`` if unknown."""
    for m in ALL_MODELS:
        if m.id == model_id:
            return m
    raise KeyError(f"unknown model id: {model_id!r}")


def models_for_provider(provider: str) -> tuple[ModelInfo, ...]:
    """All models exposed by a given provider."""
    return tuple(m for m in ALL_MODELS if m.provider == provider)


def provider_for_model(model_id: str) -> str:
    """Which provider owns a given model id."""
    return get_model(model_id).provider

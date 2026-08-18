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

# API-key creation pages surfaced as a "create / get a key" link on
# the auth screen's per-provider help text. A researcher arriving
# without a key clicks straight through to the provider's console.
PROVIDER_API_KEY_URLS: dict[str, str] = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
}


# ---------------------------------------------------------------------------
# Per-provider models
# ---------------------------------------------------------------------------

# Anthropic — the Claude 5 family, cheapest tier first. The ``[1m]``
# suffix is the Claude CLI / Agent SDK convention for the 1M-context
# opt-in (the CLI strips it and sets the beta header; ``fable[1m]``
# is a built-in CLI alias, so the suffix parses generically). On the
# Claude 5 models 1M is both the default and the maximum, so the
# suffix is belt-and-braces rather than load-bearing — kept so every
# Anthropic id in the catalog follows one convention and per-session
# model memory (which restores only catalog-known ids) stays stable.
# There's no pricing tier on context length: a 900k-token request
# costs the same per-token as a 9k-token one. Labels are clean —
# context-window numbers live in the picker's right-side column.
# Tiers (per Anthropic's pricing doc, Aug 2026): Sonnet 5 $3/$15 per
# MTok, Opus 5 $5/$25, Fable 5 $10/$50. Opus 5 and Fable 5 are
# listed in the picker but deliberately NOT the default (see
# PROVIDER_DEFAULTS below); a researcher opts in per session and
# per-session model memory keeps the choice.
# Haiku is intentionally excluded for now: the Nora workload
# (multi-turn analysis with tool use) calls for the heavier models.
ANTHROPIC_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="claude-sonnet-5[1m]",
        label="Sonnet 5",
        context_window=1_000_000,
        provider="anthropic",
    ),
    ModelInfo(
        id="claude-opus-5[1m]",
        label="Opus 5",
        context_window=1_000_000,
        provider="anthropic",
    ),
    ModelInfo(
        id="claude-fable-5[1m]",
        label="Fable 5",
        context_window=1_000_000,
        provider="anthropic",
    ),
)

# OpenAI — the GPT-5.6 family, cheapest tier first. ``gpt-5.6-terra``
# is the balanced / cost-tier model (roughly the "mini" slot of
# earlier GPT-5 families; $2/$12 per MTok); ``gpt-5.6-sol`` is the
# flagship ($5/$30 — same price point as the gpt-5.5 it replaces;
# the bare ``gpt-5.6`` alias routes to Sol). Both accept the full
# none/low/medium/high/xhigh/max reasoning range, so the provider's
# pinned ``effort="xhigh"`` is valid on either. Ids match the OpenAI
# Models API exactly so a researcher can cross-reference pricing and
# limits in OpenAI's own docs / billing dashboard.
# Context window: 1.05M tokens for both per OpenAI's published spec
# (the Models API itself doesn't expose this — it has to be hard-coded
# from OpenAI's docs and updated when they publish new variants).
OPENAI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gpt-5.6-terra",
        label="GPT-5.6 Terra",
        context_window=1_050_000,
        provider="openai",
    ),
    ModelInfo(
        id="gpt-5.6-sol",
        label="GPT-5.6 Sol",
        context_window=1_050_000,
        provider="openai",
    ),
)


ALL_MODELS: tuple[ModelInfo, ...] = ANTHROPIC_MODELS + OPENAI_MODELS


# Default model per provider — what an "open a session for provider X"
# call uses when the researcher hasn't picked something explicitly.
# Anthropic stays on Sonnet even though heavier tiers (Opus, Fable)
# are in the picker: the default is what a researcher gets without
# asking, and silently defaulting to a 2–3x-priced tier would change
# their bill, not just their model. OpenAI defaults to Sol — it is
# the direct successor to gpt-5.5 (the previous default) at the same
# $5/$30 price point, so a researcher's bill doesn't move on upgrade;
# Terra is the cheaper opt-in.
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-5[1m]",
    "openai": "gpt-5.6-sol",
}


# Default provider when none is configured yet — only matters as a
# placeholder; the auth screen forces the researcher to pick before
# they reach the chat view.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]


# ---------------------------------------------------------------------------
# Effort levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffortInfo:
    """One selectable reasoning-effort level. ``id`` is the wire value
    both providers accept (Anthropic ``output_config.effort`` via the
    Agent SDK's ``--effort`` flag; OpenAI ``reasoning.effort``)."""

    id: str
    label: str
    hint: str


# Reasoning effort, cheapest first. Every catalog model — the Claude
# 5 family and the GPT-5.6 family — accepts this exact ladder, so a
# single provider-neutral list is enough (OpenAI additionally has
# ``none``, deliberately excluded: Nora surfaces the reasoning trace,
# and Anthropic has no equivalent). Mirrors Claude Code's effort
# picker so a researcher who knows that dial finds the same one here.
# ``xhigh`` is the default: it's what the providers were pinned to
# before effort became selectable, and the recommended setting for
# multi-step tool-using analysis on both families.
EFFORT_OPTIONS: tuple[EffortInfo, ...] = (
    EffortInfo(
        id="low",
        label="Low",
        hint="Fastest, cheapest — quick lookups and small edits.",
    ),
    EffortInfo(
        id="medium",
        label="Medium",
        hint="Balanced — routine analysis where speed matters.",
    ),
    EffortInfo(
        id="high",
        label="High",
        hint="Deep reasoning — the provider default.",
    ),
    EffortInfo(
        id="xhigh",
        label="Extra high",
        hint="Extended reasoning — best for multi-step, tool-heavy work.",
    ),
    EffortInfo(
        id="max",
        label="Max",
        hint="Ceiling — most thorough, slowest, most tokens.",
    ),
)

EFFORT_LEVELS: tuple[str, ...] = tuple(e.id for e in EFFORT_OPTIONS)
DEFAULT_EFFORT = "xhigh"


def get_effort(effort_id: str) -> EffortInfo:
    """Look up an effort level by id. Raises ``KeyError`` if unknown."""
    for e in EFFORT_OPTIONS:
        if e.id == effort_id:
            return e
    raise KeyError(f"unknown effort level: {effort_id!r}")


def normalize_effort(effort_id: str | None) -> str:
    """Return ``effort_id`` when it's a known level, else the default.

    Used on the restore path (per-session memory) so a state file
    written by a future build with a level this build doesn't know
    falls back to the default rather than wedging the session."""
    if effort_id in EFFORT_LEVELS:
        return effort_id  # type: ignore[return-value]
    return DEFAULT_EFFORT


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

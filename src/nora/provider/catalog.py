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
# Tiers (per Anthropic's pricing doc, Sep 2026): Sonnet 5 $2/$10 per
# MTok (the launch price, since made permanent), Opus 5 $5/$25,
# Fable 5.1 $10/$50. Fable 5.1 replaced Fable 5 in the picker: same
# tier, same per-token price, so a researcher who opted into Fable
# pays the same. The id spells out ``claude-fable-5-1`` rather than
# the CLI's ``fable`` alias because an older installed CLI may still
# resolve that alias to Fable 5. Opus 5 and Fable 5.1 are listed in
# the picker but deliberately NOT the default (see PROVIDER_DEFAULTS
# below); a researcher opts in per session and per-session model
# memory keeps the choice.
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
        id="claude-fable-5-1[1m]",
        label="Fable 5.1",
        context_window=1_000_000,
        provider="anthropic",
    ),
)

# OpenAI — cheapest tier first: the GPT-5.6 pair, then GPT-6 Astra
# on top. ``gpt-5.6-terra`` is the balanced / cost-tier model
# (roughly the "mini" slot of earlier GPT-5 families; $2/$12 per
# MTok); ``gpt-5.6-sol`` is the GPT-5.6 flagship ($5/$30 — same
# price point as the gpt-5.5 it replaced; the bare ``gpt-5.6`` alias
# routes to Sol). ``gpt-6-astra`` (Sep 2026) is OpenAI's top tier at
# $10/$50 — the same price point as Fable 5.1 on the Anthropic side
# — and the one entry in this catalog with a length tier: prompts
# over 272k input tokens bill 2x on input and cache and 1.5x on
# output. All three take low/medium/high/xhigh and pro mode, so
# every rung of the OpenAI effort ladder below is valid on any of
# them (Astra drops ``none``, which the ladder never offered). Ids
# match the OpenAI Models API exactly so a researcher can
# cross-reference pricing and limits in OpenAI's own docs / billing
# dashboard.
# Context window: 1.05M tokens for all three per OpenAI's published
# spec (the Models API itself doesn't expose this — it has to be
# hard-coded from OpenAI's docs and updated when they publish new
# variants). Astra caps input at 922k of that and output at 128k.
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
    ModelInfo(
        id="gpt-6-astra",
        label="GPT-6 Astra",
        context_window=1_050_000,
        provider="openai",
    ),
)


ALL_MODELS: tuple[ModelInfo, ...] = ANTHROPIC_MODELS + OPENAI_MODELS


# Default model per provider — what an "open a session for provider X"
# call uses when the researcher hasn't picked something explicitly.
# Anthropic stays on Sonnet even though heavier tiers (Opus, Fable)
# are in the picker: the default is what a researcher gets without
# asking, and silently defaulting to a 2.5–5x-priced tier would
# change their bill, not just their model. OpenAI stays on Sol for
# the same reason: GPT-6 Astra is the newer flagship, but at $10/$50
# it costs double, so it is a per-session opt-in like Fable 5.1
# rather than the default. Sol itself was the direct successor to
# gpt-5.5 (the default before it) at the same $5/$30 price point;
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
    the provider accepts (Anthropic ``output_config.effort``, passed
    by the Agent SDK as the CLI's ``--effort`` flag; OpenAI
    ``reasoning.effort`` on the Responses request)."""

    id: str
    label: str


# Reasoning effort, cheapest first. **The ladders differ per provider**
# — the picker renders whichever one belongs to the selected model's
# provider, so a level that provider can't take is never offered:
#
#   Anthropic  low, medium, high, xhigh, max
#   OpenAI     low, medium, high, xhigh, pro
#
# The four lower rungs are the same dial on both sides
# (``output_config.effort`` on Anthropic, ``reasoning.effort`` on
# OpenAI). The top rung is where they part:
#
# - Anthropic's ceiling is ``max``. The Claude Agent SDK types
#   ``EffortLevel`` as low|medium|high|xhigh|max and the CLI lists
#   all five.
# - OpenAI has no ``max`` the client can express — the pinned SDK
#   (2.41.0) types ``ReasoningEffort`` as none|minimal|low|medium|
#   high|xhigh. (OpenAI's model pages do claim ``max``; that's the
#   SDK lagging the API, and we follow the SDK because it's what
#   ships in the bundle.) What OpenAI has instead is ``pro`` — a
#   *separate* knob, ``reasoning.mode``, which buys more model work
#   per turn at the same effort. It is genuinely orthogonal to
#   effort in the API, but for a researcher choosing "how hard
#   should this try" it is the rung above ``xhigh``, so that's where
#   the bar puts it. ``provider/openai.py`` translates it back into
#   the two real parameters on the way out.
#
# ``none`` / ``minimal`` (OpenAI-only) are deliberately excluded:
# Nora surfaces the reasoning trace in the thinking panel, and those
# levels suppress it. Anthropic has no equivalent rung either, so
# offering them would make the two panels diverge for no gain.
_LOW = EffortInfo(id="low", label="Low")
_MEDIUM = EffortInfo(id="medium", label="Medium")
_HIGH = EffortInfo(id="high", label="High")
_XHIGH = EffortInfo(id="xhigh", label="Extra high")
_MAX = EffortInfo(id="max", label="Max")
_PRO = EffortInfo(id="pro", label="Pro")

PROVIDER_EFFORTS: dict[str, tuple[EffortInfo, ...]] = {
    "anthropic": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX),
    "openai": (_LOW, _MEDIUM, _HIGH, _XHIGH, _PRO),
}

# Every level this build knows, in canonical cheapest-first order.
EFFORT_OPTIONS: tuple[EffortInfo, ...] = (
    _LOW, _MEDIUM, _HIGH, _XHIGH, _MAX, _PRO,
)
EFFORT_LEVELS: tuple[str, ...] = tuple(e.id for e in EFFORT_OPTIONS)

# Rank drives ``clamp_effort``. ``max`` and ``pro`` deliberately
# SHARE the top rank: they aren't the same parameter, but they
# occupy the same position on their provider's bar — each is that
# provider's "work as hard as you can". Tying them means a
# researcher at one ceiling who switches providers lands on the
# other ceiling instead of silently dropping a rung.
_EFFORT_RANK: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4, "pro": 4,
}

# What a session runs at when nobody has chosen: exactly what both
# providers were hard-pinned to before effort became selectable, so
# turning the dial on changed no existing behaviour. Supported on
# both ladders.
DEFAULT_EFFORT = "xhigh"


def get_effort(effort_id: str) -> EffortInfo:
    """Look up an effort level by id. Raises ``KeyError`` if unknown."""
    for e in EFFORT_OPTIONS:
        if e.id == effort_id:
            return e
    raise KeyError(f"unknown effort level: {effort_id!r}")


def efforts_for_provider(provider: str) -> tuple[EffortInfo, ...]:
    """The ladder a given provider actually accepts. Unknown provider
    falls back to the canonical list — callers validate the provider
    elsewhere, and an empty picker would be worse than a wrong one."""
    return PROVIDER_EFFORTS.get(provider, EFFORT_OPTIONS)


def effort_levels_for_provider(provider: str) -> tuple[str, ...]:
    """Just the ids — the validation surface for ``set_effort``."""
    return tuple(e.id for e in efforts_for_provider(provider))


def clamp_effort(effort_id: str | None, provider: str) -> str:
    """Return the closest level ``provider`` supports at or below
    ``effort_id``, else the provider's default.

    Two callers need this, both crossing a boundary where the ladder
    can change out from under a recorded choice:

    - a cross-provider model swap (a researcher on Anthropic ``max``
      switching to OpenAI, which has no ``max``), and
    - the per-session restore path (a state file recording ``max``
      against a session whose model is now an OpenAI one).

    Stepping *down* rather than resetting keeps the researcher's
    intent: someone who asked for the ceiling gets the new provider's
    ceiling, not a silent drop to the middle of the ladder. Never
    steps up — that would raise spend nobody asked for.
    """
    supported = effort_levels_for_provider(provider)
    if effort_id in supported:
        return effort_id  # type: ignore[return-value]
    default = DEFAULT_EFFORT if DEFAULT_EFFORT in supported else supported[-1]
    if effort_id not in _EFFORT_RANK:
        return default
    want = _EFFORT_RANK[effort_id]
    at_or_below = [e for e in supported if _EFFORT_RANK[e] <= want]
    # Ladders are cheapest-first, so the last match is the highest
    # supported rung that doesn't exceed what was asked for.
    return at_or_below[-1] if at_or_below else supported[0]


def normalize_effort(effort_id: str | None, provider: str | None = None) -> str:
    """Return ``effort_id`` when this build knows it, else the default.

    With ``provider``, this is :func:`clamp_effort` — the level is
    additionally held to that provider's ladder. Without one it only
    checks the canonical list, which is what the runner does before
    it knows which session it will open.
    """
    if provider is not None:
        return clamp_effort(effort_id, provider)
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

"""Regression tests for the model catalog.

The catalog is the single source of truth for the picker (the web
UI's ``list_models`` derives its rows from it) and for ``set_model``
validation in both the bridge and the provider sessions. These tests
pin the entries, the defaults, and the routing so a future catalog
edit that drops or renames a model trips loudly instead of silently
breaking per-session model memory (which restores only catalog-known
ids).

Current lineup (Aug 2026): the Claude 5 family (Sonnet 5 / Opus 5 /
Fable 5) alongside the GPT-5.6 family (Terra / Sol).
"""

from __future__ import annotations

import pytest

from nora.provider.catalog import (
    ALL_MODELS,
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
    PROVIDER_DEFAULTS,
    PROVIDER_PRICING_URLS,
    get_model,
    provider_for_model,
)
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Anthropic — Claude 5 family
# ---------------------------------------------------------------------------

def test_anthropic_catalog_is_the_claude_5_family() -> None:
    """Sonnet 5 / Opus 5 / Fable 5, cheapest tier first. Every id
    carries the ``[1m]`` suffix (the Claude CLI / Agent SDK 1M-context
    convention) so the ids pass through ``ClaudeAgentOptions(model=...)``
    unchanged and follow one convention across the provider."""
    ids = [m.id for m in ANTHROPIC_MODELS]
    assert ids == [
        "claude-sonnet-5[1m]",
        "claude-opus-5[1m]",
        "claude-fable-5[1m]",
    ]
    for m in ANTHROPIC_MODELS:
        assert m.provider == "anthropic"
        assert m.context_window == 1_000_000
        assert m.id.endswith("[1m]")


@pytest.mark.parametrize(
    "model_id, label",
    [
        ("claude-sonnet-5[1m]", "Sonnet 5"),
        ("claude-opus-5[1m]", "Opus 5"),
        ("claude-fable-5[1m]", "Fable 5"),
    ],
)
def test_anthropic_entries_resolve(model_id: str, label: str) -> None:
    info = get_model(model_id)
    assert info.label == label
    assert provider_for_model(model_id) == "anthropic"


def test_anthropic_default_stays_on_sonnet() -> None:
    """The default is what a researcher gets without asking. Opus 5
    bills $5/$25 per MTok and Fable 5 $10/$50 (vs Sonnet 5's $3/$15)
    — the heavier tiers must be an explicit per-session opt-in, never
    the silent default."""
    assert PROVIDER_DEFAULTS["anthropic"] == "claude-sonnet-5[1m]"


def test_pre_5_anthropic_ids_are_gone() -> None:
    """The 4.x ids were replaced, not kept alongside. Per-session
    model memory drops ids that fall out of the catalog, so a session
    saved on Sonnet 4.6 / Opus 4.8 falls back to the default rather
    than restoring a model the picker no longer offers."""
    for old in ("claude-sonnet-4-6[1m]", "claude-opus-4-8[1m]"):
        with pytest.raises(KeyError):
            get_model(old)


# ---------------------------------------------------------------------------
# OpenAI — GPT-5.6 family
# ---------------------------------------------------------------------------

def test_openai_catalog_is_the_gpt_5_6_family() -> None:
    """Terra (cost tier) then Sol (flagship). Ids match the OpenAI
    Models API exactly — no bare ``gpt-5.6`` alias, which routes to
    Sol server-side and would make the picker's row ambiguous."""
    ids = [m.id for m in OPENAI_MODELS]
    assert ids == ["gpt-5.6-terra", "gpt-5.6-sol"]
    for m in OPENAI_MODELS:
        assert m.provider == "openai"
        assert m.context_window == 1_050_000


@pytest.mark.parametrize(
    "model_id, label",
    [
        ("gpt-5.6-terra", "GPT-5.6 Terra"),
        ("gpt-5.6-sol", "GPT-5.6 Sol"),
    ],
)
def test_openai_entries_resolve(model_id: str, label: str) -> None:
    info = get_model(model_id)
    assert info.label == label
    assert provider_for_model(model_id) == "openai"


def test_openai_default_is_sol() -> None:
    """Sol is the direct successor to gpt-5.5 (the previous default)
    at the same $5/$30 price point, so a researcher's bill doesn't
    move on upgrade. Terra is the cheaper opt-in, not the default."""
    assert PROVIDER_DEFAULTS["openai"] == "gpt-5.6-sol"


def test_gpt_5_5_ids_are_gone() -> None:
    for old in ("gpt-5.5", "gpt-5.5-pro"):
        with pytest.raises(KeyError):
            get_model(old)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

def test_every_default_is_in_the_catalog() -> None:
    """A default that isn't a catalog entry would wedge every fresh
    session for that provider (``set_model`` validates against the
    catalog)."""
    for provider, model_id in PROVIDER_DEFAULTS.items():
        assert get_model(model_id).provider == provider


def test_every_catalog_entry_resolves_a_provider() -> None:
    """Sanity over the whole catalog: every id routes to a provider
    the auth layer knows about, and that provider has a pricing URL
    for the picker's ``$`` link. Guards against a typo'd ``provider``
    field wedging ``set_model`` for one row."""
    for m in ALL_MODELS:
        assert provider_for_model(m.id) in ("anthropic", "openai")
        assert PROVIDER_PRICING_URLS[m.provider]


def test_catalog_ids_are_unique() -> None:
    ids = [m.id for m in ALL_MODELS]
    assert len(ids) == len(set(ids))


def test_list_models_exposes_full_lineup(
    tmp_path, anthropic_authed: None,
) -> None:
    """The bridge's ``list_models`` (what the JS picker renders) must
    carry every catalog row with the fields the popup needs, and mark
    the Anthropic rows available when Anthropic is authed."""
    bridge = NoraBridge(cwd=None)
    payload = bridge.list_models()
    rows = {m["id"]: m for m in payload["models"]}
    assert set(rows) == {m.id for m in ALL_MODELS}
    for m in ALL_MODELS:
        row = rows[m.id]
        assert row["label"] == m.label
        assert row["provider"] == m.provider
        assert row["context_window"] == m.context_window
        assert row["pricing_url"], "picker's $ link needs a pricing URL"
    for m in ANTHROPIC_MODELS:
        assert rows[m.id]["available"] is True


@pytest.mark.parametrize(
    "model_id, provider",
    [
        ("claude-opus-5[1m]", "anthropic"),
        ("claude-fable-5[1m]", "anthropic"),
        ("gpt-5.6-terra", "openai"),
    ],
)
def test_set_model_accepts_new_entries(
    tmp_path, model_id: str, provider: str,
) -> None:
    """``set_model`` with no focused session stashes the choice in the
    bridge defaults — the validation path that rejects unknown ids
    must accept every new entry."""
    bridge = NoraBridge(cwd=None)
    res = bridge.set_model(model_id)
    assert res["ok"] is True
    assert bridge._default_model == model_id
    assert bridge._default_provider == provider

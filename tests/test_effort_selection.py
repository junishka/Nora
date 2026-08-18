"""Tests for per-session reasoning-effort selection.

Effort used to be pinned to ``xhigh`` at both provider boundaries.
It's now a researcher-facing dial in the model picker, so it has to
travel the same rails as the model: catalog → picker payload →
bridge → runner → provider session → ``.nora/session_state.json``
and back on the next open.

Two asymmetries drive most of these tests:

1. **Anthropic can only take effort at launch.** The Claude Agent SDK
   passes it as the CLI's ``--effort`` flag when the client starts;
   there's no in-place control request the way there is for
   ``set_model``. So a live Anthropic session reports
   ``requires_reopen`` and the RUNNER closes it — closing at the
   provider layer would let ``send()`` lazily reopen without the
   runner re-arming ``needs_context_prefix``, silently dropping the
   conversation. OpenAI sends effort per request, so it just applies
   to the next message.

2. **The ladders differ at the top.** Both providers share
   ``low``…``xhigh``, then part: Anthropic's ceiling is ``max``,
   OpenAI's is ``pro`` — which isn't an effort value at all but
   ``reasoning.mode``, presented as the rung above ``xhigh`` and
   unpacked back into two API parameters on the way out. So the
   picker renders the ladder belonging to the selected model's
   provider, validation is per-provider, and a level crossing a
   boundary it doesn't exist on maps by RANK — ceiling to ceiling,
   never silently down a rung. Effort still restores independently
   of the model, so a session whose recorded model left the catalog
   keeps its level.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nora.provider.catalog import (
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    EFFORT_OPTIONS,
    PROVIDER_EFFORTS,
    _EFFORT_RANK,
    clamp_effort,
    effort_levels_for_provider,
    efforts_for_provider,
    get_effort,
    normalize_effort,
)
from nora.runner import SessionRunner
from nora.session_state import read_session_state, write_session_state
from nora.ui import NoraBridge


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_effort_ladder_is_cheapest_first() -> None:
    """Order is load-bearing: the picker renders each bar in list
    order, so a shuffle would put a ceiling rung next to ``low`` and
    invite a mis-click that multiplies someone's bill."""
    assert EFFORT_LEVELS == (
        "low", "medium", "high", "xhigh", "max", "pro",
    )
    assert [e.id for e in EFFORT_OPTIONS] == list(EFFORT_LEVELS)


def test_provider_ladders_share_a_prefix_and_differ_at_the_top() -> None:
    """The whole reason the picker is provider-aware. The four lower
    rungs are the same dial on both sides; the ceiling is not.
    Anthropic's is ``max`` (the Agent SDK types EffortLevel as
    low..max). OpenAI has no ``max`` its client can express — the
    pinned SDK (2.41.0) types ReasoningEffort without it — so its
    ceiling is ``pro``, which is a different API knob entirely
    (``reasoning.mode``) presented as the rung above ``xhigh``."""
    assert effort_levels_for_provider("anthropic") == (
        "low", "medium", "high", "xhigh", "max",
    )
    assert effort_levels_for_provider("openai") == (
        "low", "medium", "high", "xhigh", "pro",
    )
    assert "max" not in effort_levels_for_provider("openai")
    assert "pro" not in effort_levels_for_provider("anthropic")


def test_ceilings_share_a_rank() -> None:
    """``max`` and ``pro`` aren't the same parameter, but they hold
    the same position on their provider's bar. Tying their rank is
    what makes a provider switch land ceiling-to-ceiling instead of
    silently dropping a rung."""
    assert _EFFORT_RANK["max"] == _EFFORT_RANK["pro"]
    assert _EFFORT_RANK["max"] > _EFFORT_RANK["xhigh"]


def test_openai_ladder_omits_none_and_minimal() -> None:
    """OpenAI additionally accepts ``none`` / ``minimal``, both
    deliberately excluded: they suppress the reasoning trace Nora
    renders in the thinking panel, and Anthropic has no equivalent
    rung, so offering them would split the panels for no gain."""
    ids = effort_levels_for_provider("openai")
    assert "none" not in ids
    assert "minimal" not in ids


def test_every_provider_ladder_is_ranked_and_ascending() -> None:
    """``clamp_effort`` steps down by rank, which only works if every
    rung is ranked and each ladder ascends."""
    for provider, ladder in PROVIDER_EFFORTS.items():
        ids = [e.id for e in ladder]
        assert set(ids) <= set(EFFORT_LEVELS), provider
        ranks = [_EFFORT_RANK[i] for i in ids]
        assert ranks == sorted(ranks), f"{provider} ladder is out of order"
        assert len(set(ranks)) == len(ranks), f"{provider} has a tied rung"


def test_default_effort_is_offered_by_every_provider() -> None:
    """A default that some provider doesn't offer would mean a fresh
    session on that provider starts on a clamped level nobody chose."""
    for provider in PROVIDER_EFFORTS:
        assert DEFAULT_EFFORT in effort_levels_for_provider(provider)


@pytest.mark.parametrize(
    "requested, provider, expected",
    [
        # Supported levels pass through untouched.
        ("max", "anthropic", "max"),
        ("low", "openai", "low"),
        ("xhigh", "openai", "xhigh"),
        # The real case: each provider's ceiling maps to the other's
        # ceiling, not down a rung. Someone who asked for "work as
        # hard as you can" keeps asking for it across a switch.
        ("max", "openai", "pro"),
        ("pro", "anthropic", "max"),
        ("pro", "openai", "pro"),
        # Unknown levels (future build, hand-edited state file) fall
        # back to the default rather than wedging the session.
        ("ultra", "openai", DEFAULT_EFFORT),
        ("ultra", "anthropic", DEFAULT_EFFORT),
        (None, "openai", DEFAULT_EFFORT),
    ],
)
def test_clamp_effort(requested, provider: str, expected: str) -> None:
    assert clamp_effort(requested, provider) == expected


def test_clamp_never_steps_up() -> None:
    """Clamping raises spend if it rounds upward. Every clamped
    result must sit at or below the requested RANK (ceilings tie, so
    ceiling-to-ceiling is level, never a step up)."""
    for provider in PROVIDER_EFFORTS:
        for level in EFFORT_LEVELS:
            got = clamp_effort(level, provider)
            assert _EFFORT_RANK[got] <= _EFFORT_RANK[level]


def test_default_effort_is_xhigh() -> None:
    """``xhigh`` is what both providers were hard-pinned to before
    effort became selectable — keeping it as the default means
    turning the dial ON changes nothing until a researcher moves it."""
    assert DEFAULT_EFFORT == "xhigh"
    assert DEFAULT_EFFORT in EFFORT_LEVELS


def test_every_level_has_a_label() -> None:
    """The label is what the toast and the button tooltip show."""
    for e in EFFORT_OPTIONS:
        assert e.label
        assert get_effort(e.id) is e


def test_get_effort_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        get_effort("ultra")


def test_normalize_effort_falls_back_to_default() -> None:
    """The restore path runs this against whatever a state file says.
    A level written by a future build (or a hand-edited file) must
    degrade to the default, not wedge the session."""
    assert normalize_effort("max") == "max"
    assert normalize_effort("ultra") == DEFAULT_EFFORT
    assert normalize_effort(None) == DEFAULT_EFFORT
    assert normalize_effort("") == DEFAULT_EFFORT


# ---------------------------------------------------------------------------
# session_state persistence
# ---------------------------------------------------------------------------

def test_active_effort_round_trips(tmp_path: Path) -> None:
    write_session_state(tmp_path, model="claude-opus-5[1m]", effort="max")
    state = read_session_state(tmp_path)
    assert state is not None
    assert state.active_effort == "max"
    raw = json.loads(
        (tmp_path / ".nora" / "session_state.json").read_text()
    )
    assert raw["active_effort"] == "max"


def test_writer_carries_prior_effort_when_not_supplied(tmp_path: Path) -> None:
    """``effort=None`` means "caller didn't say", not "clear it".

    The turn-end writer knows the effort, but other call sites
    (older code paths, tests) pass only the model — without the
    carry, every such write would silently reset a researcher's
    per-session effort back to unset."""
    write_session_state(tmp_path, model="claude-opus-5[1m]", effort="low")
    write_session_state(tmp_path, model="claude-opus-5[1m]")
    state = read_session_state(tmp_path)
    assert state is not None
    assert state.active_effort == "low"


def test_rename_and_pin_preserve_effort(tmp_path: Path) -> None:
    """``set_custom_name`` / ``set_pinned`` are targeted edits that
    rebuild the dataclass field-by-field — a missed field silently
    drops the researcher's effort choice on rename."""
    from nora.session_state import set_custom_name, set_pinned

    write_session_state(tmp_path, model="claude-opus-5[1m]", effort="medium")
    set_custom_name(tmp_path, "wage gap replication")
    assert read_session_state(tmp_path).active_effort == "medium"  # type: ignore[union-attr]
    set_pinned(tmp_path, True)
    assert read_session_state(tmp_path).active_effort == "medium"  # type: ignore[union-attr]


def test_state_file_without_effort_reads_as_none(tmp_path: Path) -> None:
    """Forward-compat in the other direction: a file written before
    this feature has no ``active_effort`` key at all."""
    nora_dir = tmp_path / ".nora"
    nora_dir.mkdir(parents=True)
    (nora_dir / "session_state.json").write_text(json.dumps({
        "version": 1,
        "last_active_at": "2026-08-18T00:00:00+00:00",
        "active_model": "claude-opus-5[1m]",
    }))
    state = read_session_state(tmp_path)
    assert state is not None
    assert state.active_effort is None


# ---------------------------------------------------------------------------
# Provider sessions
# ---------------------------------------------------------------------------

def test_anthropic_options_carry_the_selected_effort(tmp_path: Path) -> None:
    """The level reaches ``ClaudeAgentOptions`` — which is what the
    SDK turns into the CLI's ``--effort`` flag. Pinning this catches
    a refactor that drops the field back to a hard-coded literal."""
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]",
        system_prompt="you are nora", effort="low",
    )
    opts = sess._build_options()
    assert opts.effort == "low"
    # Thinking stays adaptive+summarized regardless of effort — the
    # trace panel must not go blank just because effort dropped.
    assert opts.thinking == {"type": "adaptive", "display": "summarized"}


def test_anthropic_defaults_to_catalog_effort(tmp_path: Path) -> None:
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]", system_prompt="x",
    )
    assert sess.effort == DEFAULT_EFFORT
    assert sess._build_options().effort == DEFAULT_EFFORT


def test_anthropic_set_effort_without_client_needs_no_reopen(
    tmp_path: Path,
) -> None:
    """No client yet = nothing to tear down; the next ``open()``
    launches the CLI with the new flag."""
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]", system_prompt="x",
    )
    res = asyncio.run(sess.set_effort("max"))
    assert res["ok"] is True
    assert res["requires_reopen"] is False
    assert sess.effort == "max"
    assert sess._build_options().effort == "max"


def test_anthropic_set_effort_with_live_client_requires_reopen(
    tmp_path: Path,
) -> None:
    """A live client can't be re-flagged in place, so the session
    reports ``requires_reopen`` — and deliberately does NOT close
    itself (see the module docstring for why that matters)."""
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]", system_prompt="x",
    )
    sess._client = object()  # stand-in for a live ClaudeSDKClient
    res = asyncio.run(sess.set_effort("medium"))
    assert res["ok"] is True
    assert res["requires_reopen"] is True
    assert sess.effort == "medium"
    assert sess._client is not None, "provider must not close itself"


def test_anthropic_set_effort_rejects_unknown_level(tmp_path: Path) -> None:
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]", system_prompt="x",
    )
    res = asyncio.run(sess.set_effort("ultra"))
    assert res["ok"] is False
    assert sess.effort == DEFAULT_EFFORT, "a rejected level must not stick"


def test_openai_sends_selected_effort_per_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """OpenAI carries effort in the request body, so the level the
    researcher picked has to show up in ``reasoning.effort`` on the
    wire — not just on the session object."""
    from nora.provider import openai as openai_provider
    from nora.provider.openai import OpenAISession
    from tests.test_openai_lockdown import _FakeAsyncOpenAI

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FakeAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol",
        system_prompt="you are nora", effort="low",
    )

    async def _drive() -> None:
        async for _ in sess.send("hello"):
            pass

    asyncio.run(_drive())
    call = sess._client.responses.calls[0]  # type: ignore[union-attr]
    assert call["reasoning"]["effort"] == "low"
    # Summaries stay on so the thinking panel keeps populating.
    assert call["reasoning"]["summary"] == "auto"


def test_openai_set_effort_needs_no_reopen(tmp_path: Path) -> None:
    """Per-request delivery means the ``previous_response_id`` chain
    survives — no ``requires_reopen``, no conversation reset."""
    from nora.provider.openai import OpenAISession

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="x",
    )
    res = asyncio.run(sess.set_effort("low"))
    assert res["ok"] is True
    assert not res.get("requires_reopen")
    assert sess.effort == "low"


def test_openai_session_rejects_max(tmp_path: Path) -> None:
    """``max`` isn't in the pinned OpenAI SDK's ReasoningEffort, so
    the session refuses it rather than putting it on the wire — even
    though ``max`` is a perfectly real level on the other provider."""
    from nora.provider.openai import OpenAISession

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="x",
    )
    res = asyncio.run(sess.set_effort("max"))
    assert res["ok"] is False
    assert "max" in res["reason"]
    assert sess.effort == DEFAULT_EFFORT


def test_anthropic_session_rejects_pro(tmp_path: Path) -> None:
    """The mirror image: ``pro`` is an OpenAI-only knob, so it must
    never reach the Agent SDK's ``--effort`` flag."""
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]", system_prompt="x",
    )
    res = asyncio.run(sess.set_effort("pro"))
    assert res["ok"] is False
    assert sess.effort == DEFAULT_EFFORT
    assert sess._build_options().effort != "pro"


def test_openai_session_maps_max_to_its_own_ceiling(tmp_path: Path) -> None:
    """A runner carrying Anthropic's ``max`` into a newly-opened
    OpenAI session lands on OpenAI's ceiling, not a rung below it."""
    from nora.provider.openai import OpenAISession

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="x", effort="max",
    )
    assert sess.effort == "pro"


@pytest.mark.parametrize(
    "effort, expected",
    [
        ("low", {"summary": "auto", "effort": "low"}),
        ("high", {"summary": "auto", "effort": "high"}),
        ("xhigh", {"summary": "auto", "effort": "xhigh"}),
        # The translation that makes the bar honest: ``pro`` is not an
        # effort value at all. It becomes reasoning.mode="pro", and it
        # carries the highest expressible effort so the top rung
        # doesn't reason LESS than the rung below it (effort would
        # otherwise default to medium in pro mode).
        ("pro", {"summary": "auto", "effort": "xhigh", "mode": "pro"}),
    ],
)
def test_openai_reasoning_params(
    tmp_path: Path, effort: str, expected: dict[str, Any],
) -> None:
    from nora.provider.openai import OpenAISession

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="x", effort=effort,
    )
    assert sess._reasoning_params() == expected


def test_pro_reaches_the_wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """End-to-end through a fake client: picking ``pro`` must put
    ``mode`` in the request body. ``mode`` is absent from the pinned
    SDK's Reasoning TypedDict, so this is the test that catches an
    SDK upgrade that starts stripping unknown keys."""
    from nora.provider import openai as openai_provider
    from nora.provider.openai import OpenAISession
    from tests.test_openai_lockdown import _FakeAsyncOpenAI

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FakeAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol",
        system_prompt="you are nora", effort="pro",
    )

    async def _drive() -> None:
        async for _ in sess.send("hello"):
            pass

    asyncio.run(_drive())
    call = sess._client.responses.calls[0]  # type: ignore[union-attr]
    assert call["reasoning"]["mode"] == "pro"
    assert call["reasoning"]["effort"] == "xhigh"


def test_anthropic_session_keeps_max(tmp_path: Path) -> None:
    """The other side of the same coin — Anthropic does offer it."""
    from nora.provider.anthropic import AnthropicSession

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5[1m]",
        system_prompt="x", effort="max",
    )
    assert sess.effort == "max"
    assert sess._build_options().effort == "max"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class _FakeSession:
    """Stand-in provider session recording effort swaps."""

    def __init__(self, *, requires_reopen: bool = False,
                 fail: bool = False, raises: Exception | None = None) -> None:
        self._requires_reopen = requires_reopen
        self._fail = fail
        self._raises = raises
        self.calls: list[str] = []
        self.close_count = 0

    async def set_effort(self, effort: str) -> dict[str, Any]:
        self.calls.append(effort)
        if self._raises is not None:
            raise self._raises
        if self._fail:
            return {"ok": False, "reason": "nope"}
        return {
            "ok": True, "effort": effort,
            "requires_reopen": self._requires_reopen,
        }

    async def close(self) -> None:
        self.close_count += 1


def test_runner_swap_effort_without_session(tmp_path: Path) -> None:
    """Nothing open yet — record it so the lazy open picks it up."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    res = asyncio.run(runner.swap_effort("low"))
    assert res["ok"] is True
    assert runner.effort == "low"


def test_runner_closes_session_when_provider_requires_reopen(
    tmp_path: Path,
) -> None:
    """The Anthropic path. Closing at the runner level is what
    re-arms ``needs_context_prefix`` on the next ``ensure_session``,
    so the conversation is re-warmed rather than lost."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    fake = _FakeSession(requires_reopen=True)
    runner._session = fake  # type: ignore[assignment]
    res = asyncio.run(runner.swap_effort("max"))
    assert res["ok"] is True
    assert res["conversation_rewarmed"] is True
    assert fake.close_count == 1
    assert runner._session is None
    assert runner.effort == "max"


def test_runner_keeps_session_when_provider_applies_in_place(
    tmp_path: Path,
) -> None:
    """The OpenAI path — no close, no re-warm flag."""
    runner = SessionRunner(
        cwd=tmp_path, provider="openai", model="gpt-5.6-sol",
    )
    fake = _FakeSession(requires_reopen=False)
    runner._session = fake  # type: ignore[assignment]
    res = asyncio.run(runner.swap_effort("medium"))
    assert res["ok"] is True
    assert not res.get("conversation_rewarmed")
    assert fake.close_count == 0
    assert runner._session is fake
    assert runner.effort == "medium"


@pytest.mark.parametrize(
    "fake",
    [
        _FakeSession(fail=True),
        _FakeSession(raises=RuntimeError("sdk blew up")),
    ],
)
def test_runner_rolls_back_effort_on_failure(
    tmp_path: Path, fake: _FakeSession,
) -> None:
    """Same rollback discipline as ``swap_model``: a failed swap must
    leave the runner on the level that actually works, or the next
    turn reopens with a rejected flag and fails again."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    original = runner.effort
    runner._session = fake  # type: ignore[assignment]
    res = asyncio.run(runner.swap_effort("low"))
    assert res["ok"] is False
    assert runner.effort == original


def test_runner_unknown_level_is_rejected(tmp_path: Path) -> None:
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    res = asyncio.run(runner.swap_effort("ultra"))
    assert res["ok"] is False
    assert runner.effort == DEFAULT_EFFORT


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

def test_list_models_carries_every_provider_ladder(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """The picker rebuilds the bar from the selected model's provider
    on every render, so it needs ALL the ladders up front — a model
    switch must repaint without another round-trip."""
    bridge = NoraBridge(cwd=None)
    payload = bridge.list_models()
    assert payload["current_effort"] == DEFAULT_EFFORT
    assert payload["default_effort"] == DEFAULT_EFFORT
    by_provider = payload["efforts_by_provider"]
    assert set(by_provider) == set(PROVIDER_EFFORTS)
    for provider, rows in by_provider.items():
        assert [r["id"] for r in rows] == list(
            effort_levels_for_provider(provider)
        )
        for row in rows:
            assert row["label"]
    # ``efforts`` is the current provider's ladder, so a caller that
    # only wants today's bar doesn't have to index the map.
    assert payload["efforts"] == by_provider[payload["current_provider"]]


def test_set_effort_rejects_a_level_the_provider_lacks(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Validation is per-provider, not against the canonical union —
    ``max`` is a real level, just not an OpenAI one."""
    bridge = NoraBridge(cwd=None)
    bridge._default_provider = "openai"
    res = bridge.set_effort("max")
    assert res["ok"] is False
    assert "openai" in res["reason"]
    assert bridge._default_effort == DEFAULT_EFFORT


def test_set_effort_without_session_updates_the_default(tmp_path: Path) -> None:
    """Landing screen: no runner to swap, so the pick becomes the
    default the next runner is built with."""
    bridge = NoraBridge(cwd=None)
    res = bridge.set_effort("medium")
    assert res["ok"] is True
    assert bridge._default_effort == "medium"


def test_set_effort_rejects_unknown_level(tmp_path: Path) -> None:
    bridge = NoraBridge(cwd=None)
    assert bridge.set_effort("ultra")["ok"] is False
    assert bridge._default_effort == DEFAULT_EFFORT


def test_set_effort_persists_to_session_state(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """A pick has to survive a restart even before the researcher
    sends their first message — same promise ``set_model`` makes."""
    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.set_effort("max")
    assert res["ok"] is True
    assert bridge._effort == "max"
    state = read_session_state(tmp_path)
    assert state is not None
    assert state.active_effort == "max"


def test_set_effort_is_refused_mid_turn(
    tmp_path: Path, anthropic_authed: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swapping the CLI launch flag under a streaming turn would
    tear down the client mid-stream."""
    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None
    monkeypatch.setattr(runner, "is_busy", lambda: True)
    res = bridge.set_effort("low")
    assert res["ok"] is False
    assert "in flight" in res["reason"]
    assert runner.effort == DEFAULT_EFFORT


def test_new_session_restores_recorded_effort(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Per-session memory: a session that ran at ``max`` comes back
    at ``max``, not at the global default."""
    session_dir = tmp_path / "saved-on-max"
    session_dir.mkdir()
    write_session_state(
        session_dir, model="claude-opus-5[1m]", effort="max",
    )
    bridge = NoraBridge(cwd=None)
    assert bridge._default_effort == DEFAULT_EFFORT
    bridge._set_cwd(session_dir)
    assert bridge._effort == "max"
    assert bridge._model == "claude-opus-5[1m]"


def test_effort_restores_even_when_the_model_does_not(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Effort is provider-neutral, so it's restored independently:
    a state file naming a model that fell out of the catalog still
    gets its effort back while the model falls to the default."""
    session_dir = tmp_path / "stale-model"
    session_dir.mkdir()
    write_session_state(
        session_dir, model="claude-sonnet-4-6[1m]", effort="low",
    )
    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(session_dir)
    assert bridge._model == bridge._default_model, "stale model must not restore"
    assert bridge._effort == "low"


def test_unknown_recorded_effort_falls_back(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """A file from a future build naming a level this build doesn't
    have must not wedge the session."""
    session_dir = tmp_path / "future-level"
    session_dir.mkdir()
    write_session_state(
        session_dir, model="claude-opus-5[1m]", effort="ultra",
    )
    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(session_dir)
    assert bridge._effort == DEFAULT_EFFORT


def test_two_sessions_keep_independent_effort(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Effort is per-session, like the model: a researcher can run a
    cheap exploratory session next to an expensive one."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_session_state(a, model="claude-sonnet-5[1m]", effort="low")
    write_session_state(b, model="claude-opus-5[1m]", effort="max")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    assert bridge._effort == "low"
    bridge._set_cwd(b)
    assert bridge._effort == "max"
    bridge._set_cwd(a)
    assert bridge._effort == "low"


def test_cross_provider_swap_clamps_effort(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Anthropic ``max`` has no OpenAI rung, so hopping providers
    lands on that provider's own ceiling (``pro``) — and reports
    where it landed so the UI can repaint the bar, whose top button
    is now a different level."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic",
        model="claude-sonnet-5[1m]", effort="max",
    )
    res = asyncio.run(runner.swap_model("gpt-5.6-sol", "openai"))
    assert res["ok"] is True
    assert runner.provider == "openai"
    assert runner.effort == "pro"
    assert res["effort"] == "pro"


def test_cross_provider_swap_keeps_a_shared_level(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """Only levels the target provider lacks get clamped — a shared
    rung carries across untouched."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic",
        model="claude-sonnet-5[1m]", effort="medium",
    )
    res = asyncio.run(runner.swap_model("gpt-5.6-sol", "openai"))
    assert res["ok"] is True
    assert runner.effort == "medium"


def test_restore_clamps_recorded_effort_to_the_session_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state file can record a level the restored model's provider
    doesn't offer — Anthropic ``max`` against a session that now
    opens on OpenAI. The runner must come up on a level the OpenAI
    client can actually express."""
    import nora.provider as provider_mod
    monkeypatch.setattr(provider_mod, "detect_auth", lambda p: "api_key")

    session_dir = tmp_path / "openai-at-max"
    session_dir.mkdir()
    write_session_state(session_dir, model="gpt-5.6-sol", effort="max")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(session_dir)
    assert bridge._provider == "openai"
    assert bridge._effort == "pro"

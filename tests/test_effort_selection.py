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

2. **Effort is provider-neutral.** Every catalog model accepts the
   same ladder, so effort survives a cross-provider model swap and
   restores independently of whether the recorded model is still
   selectable.
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
    """Order is load-bearing: the picker renders the segmented
    control in list order, so a shuffle would put ``max`` next to
    ``low`` and invite a mis-click that triples someone's bill."""
    assert EFFORT_LEVELS == ("low", "medium", "high", "xhigh", "max")
    assert [e.id for e in EFFORT_OPTIONS] == list(EFFORT_LEVELS)


def test_default_effort_is_xhigh() -> None:
    """``xhigh`` is what both providers were hard-pinned to before
    effort became selectable — keeping it as the default means
    turning the dial ON changes nothing until a researcher moves it."""
    assert DEFAULT_EFFORT == "xhigh"
    assert DEFAULT_EFFORT in EFFORT_LEVELS


def test_every_level_has_a_label_and_hint() -> None:
    """Both are rendered — the label in the toast and tooltip, the
    hint as the one-liner under the segmented control."""
    for e in EFFORT_OPTIONS:
        assert e.label and e.hint
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
    res = asyncio.run(sess.set_effort("max"))
    assert res["ok"] is True
    assert not res.get("requires_reopen")
    assert sess.effort == "max"


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

def test_list_models_carries_the_effort_ladder(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """The picker renders its Effort section straight from this
    payload, so every field the JS reads has to be present."""
    bridge = NoraBridge(cwd=None)
    payload = bridge.list_models()
    assert payload["current_effort"] == DEFAULT_EFFORT
    assert payload["default_effort"] == DEFAULT_EFFORT
    assert [e["id"] for e in payload["efforts"]] == list(EFFORT_LEVELS)
    for row in payload["efforts"]:
        assert row["label"] and row["hint"]


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


def test_effort_survives_a_cross_provider_model_swap(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """``swap_model`` across providers tears down the session, but
    effort is a runner-level, provider-neutral setting — a
    researcher on ``max`` stays on ``max`` when they hop to OpenAI."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic",
        model="claude-sonnet-5[1m]", effort="max",
    )
    res = asyncio.run(runner.swap_model("gpt-5.6-sol", "openai"))
    assert res["ok"] is True
    assert runner.provider == "openai"
    assert runner.effort == "max"

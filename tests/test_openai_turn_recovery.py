"""Mid-turn failure recovery for the OpenAI provider.

A turn that makes server-side progress (at least one successful
``responses.create`` round) and then fails must NOT be silently
dropped from the model's context. The UI and ``chat_history.jsonl``
show the turn's user message, tool calls, and tool results as they
stream — so a follow-up like "explain that result" has to reach a
model that actually saw them. Pre-fix, the provider rolled the chain
back to the last committed turn and the next send chained onto a
point BEFORE the failed turn: history and model context silently
diverged, and the model answered confidently about the wrong thing.

The fix: on a mid-turn failure the provider stashes the dangling
response head (which carries the turn's user message + function
calls server-side) plus the unsent ``function_call_output`` items,
and the next ``send()`` delivers both — chaining onto the dangling
head with the outputs prepended to the new user message. If the
recovery request itself fails before making progress, the provider
falls back to the chain-expiry contract (reset + ``context_reset``)
rather than looping on a possibly-poisoned payload.

These tests drive ``OpenAISession.send`` against scripted fakes of
the Responses API, mirroring the harness in test_openai_lockdown.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nora.provider.base import AuthFailure, TurnDone, TurnError
from nora.provider.openai import OpenAISession


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _ScriptedResponse:
    """Minimal Responses-API response: one function_call (loop
    continues) or one plain message (turn ends cleanly)."""

    def __init__(self, response_id: str, *, with_tool_call: bool) -> None:
        self.id = response_id
        self.usage = _FakeUsage()
        if with_tool_call:
            class _Call:
                type = "function_call"
                name = "list_results"
                call_id = "call_xyz"
                arguments = "{}"

            self.output = [_Call()]
        else:
            class _Block:
                type = "output_text"
                text = "ok"

            class _Msg:
                type = "message"
                content = [_Block()]

            self.output = [_Msg()]


class _ScriptedResponsesAPI:
    """Pops one step per ``create()`` call — a response is returned,
    an exception is raised — and records every call's kwargs."""

    def __init__(self, steps: list[Any]) -> None:
        self._steps = list(steps)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._steps:
            raise RuntimeError("test queue exhausted")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _ScriptedAsyncOpenAI:
    def __init__(self, api_key: str | None, steps: list[Any]) -> None:
        self.api_key = api_key
        self.responses = _ScriptedResponsesAPI(steps)

    async def close(self) -> None:
        return None


def _make_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, steps: list[Any],
) -> OpenAISession:
    from nora.provider import openai as openai_provider
    import openai as openai_pkg

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    # The few-shot prepend is covered elsewhere; disabling it keeps
    # these tests' input-shape assertions about recovery items exact.
    monkeypatch.setenv("NORA_DISABLE_FEWSHOT", "1")
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None: _ScriptedAsyncOpenAI(api_key, steps),
        raising=True,
    )
    return OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are nora",
    )


def _drain(sess: OpenAISession, prompt: str) -> list[Any]:
    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send(prompt):
            events.append(ev)

    asyncio.run(_drive())
    return events


def test_mid_turn_failure_stashes_dangling_head_and_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Round 1 succeeds (tool call dispatched, result shown to the
    researcher), round 2's request dies on transport. The provider
    must (a) keep the committed pointer untouched, (b) stash the
    dangling head + the built function_call_output, and (c) flag the
    TurnError as context-preserved so the runner skips restoration."""
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        RuntimeError("server returned 502: bad gateway"),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    events = _drain(sess, "analyse")

    terminal = events[-1]
    assert isinstance(terminal, TurnError)
    assert terminal.context_preserved is True
    assert terminal.context_reset is False

    # Committed pointer untouched (nothing was committed this session).
    assert sess._last_response_id is None
    # Stash: dangling head + the one unsent output.
    assert sess._pending_turn_recovery is not None
    head, outputs = sess._pending_turn_recovery
    assert head == "resp_1"
    assert len(outputs) == 1
    assert outputs[0]["type"] == "function_call_output"
    assert outputs[0]["call_id"] == "call_xyz"


def test_next_send_delivers_stash_then_new_user_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The recovery send chains onto the dangling head with the
    stashed outputs FIRST and the new user message after — the model
    sees the failed turn exactly as the UI showed it, then the new
    prompt. A clean end commits the full chain and clears the stash."""
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        RuntimeError("server returned 502: bad gateway"),
        _ScriptedResponse("resp_2", with_tool_call=False),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    _drain(sess, "analyse")
    events = _drain(sess, "explain that result")

    assert isinstance(events[-1], TurnDone)

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 3
    recovery_call = api.calls[2]
    # Chained onto the DANGLING head, not the pre-failure turn.
    assert recovery_call.get("previous_response_id") == "resp_1"
    # Outputs first, new user message last.
    assert recovery_call["input"][0]["type"] == "function_call_output"
    assert recovery_call["input"][0]["call_id"] == "call_xyz"
    assert recovery_call["input"][-1]["role"] == "user"
    assert (
        recovery_call["input"][-1]["content"][0]["text"]
        == "explain that result"
    )

    # Clean end: full chain committed, stash consumed.
    assert sess._last_response_id == "resp_2"
    assert sess._pending_turn_recovery is None


def test_failed_recovery_falls_back_to_context_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Single-shot recovery: if the recovery request ITSELF fails
    before making progress, don't re-stash (the payload may be what
    the server rejects, and a re-stash would double-deliver this
    turn's user message on retry). Reset the chain and signal
    ``context_reset`` so the next turn re-primes from on-disk
    history — the chain-expiry contract."""
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        RuntimeError("server returned 502: bad gateway"),
        RuntimeError("server returned 500: still down"),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    _drain(sess, "analyse")
    events = _drain(sess, "explain that result")

    terminal = events[-1]
    assert isinstance(terminal, TurnError)
    assert terminal.context_reset is True
    assert terminal.context_preserved is False
    assert sess._pending_turn_recovery is None
    assert sess._last_response_id is None


def test_round_one_failure_without_progress_keeps_prior_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A turn whose FIRST request fails made no server-side progress:
    nothing to stash, no reset — the committed chain stays valid and
    the researcher simply retries the message."""
    steps = [
        _ScriptedResponse("resp_good", with_tool_call=False),
        RuntimeError("server returned 502: bad gateway"),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    _drain(sess, "first")
    events = _drain(sess, "second")

    terminal = events[-1]
    assert isinstance(terminal, TurnError)
    assert terminal.context_preserved is False
    assert terminal.context_reset is False
    assert sess._pending_turn_recovery is None
    assert sess._last_response_id == "resp_good"


def test_exhausted_tool_rounds_stash_instead_of_abandoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When the tool loop hits MAX_TOOL_ROUNDS, the rounds so far —
    already shown to the researcher — must ride the stash rather
    than being abandoned server-side (the pre-fix behavior silently
    dropped up to 16 rounds of visible tool work from the model's
    context)."""
    monkeypatch.setenv("NORA_OPENAI_MAX_TOOL_ROUNDS", "1")
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    events = _drain(sess, "analyse")

    terminal = events[-1]
    assert isinstance(terminal, TurnError)
    assert "did not converge" in terminal.message
    assert terminal.context_preserved is True
    assert sess._pending_turn_recovery is not None
    head, outputs = sess._pending_turn_recovery
    assert head == "resp_1"
    assert outputs and outputs[0]["type"] == "function_call_output"
    assert sess._last_response_id is None


def test_mid_turn_auth_failure_stashes_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An auth rejection never consumed the request, so mid-turn
    progress is stashed the same way — after the researcher fixes
    the key, the next send resumes losslessly."""
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        RuntimeError("401 unauthorized: bad api key"),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    events = _drain(sess, "analyse")

    assert isinstance(events[-1], AuthFailure)
    assert sess._pending_turn_recovery is not None
    head, outputs = sess._pending_turn_recovery
    assert head == "resp_1"
    assert outputs[0]["call_id"] == "call_xyz"


def test_auth_failure_on_recovery_turn_restores_original_stash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If the recovery request bounces on auth before progressing,
    the ORIGINAL stash is put back untouched (the rejected request
    never consumed the outputs), so recovery survives a key fix.
    The new user message is NOT folded into the stash — the
    researcher retries it as their own message."""
    steps = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        RuntimeError("server returned 502: bad gateway"),
        RuntimeError("401 unauthorized: bad api key"),
    ]
    sess = _make_session(monkeypatch, tmp_path, steps)
    _drain(sess, "analyse")
    original_stash = sess._pending_turn_recovery
    assert original_stash is not None

    events = _drain(sess, "explain that result")
    assert isinstance(events[-1], AuthFailure)
    assert sess._pending_turn_recovery == original_stash

"""Lockdown tests for the OpenAI provider.

Mirrors the spirit of Nora's existing Claude SDK lockdown coverage:
no matter what changes upstream in the OpenAI Responses API, the
``tools`` field Nora sends must contain EXACTLY the six Nora function
tools and no built-in types (web_search, code_interpreter,
file_search, image_generation, mcp, computer_use_preview, …).

Without these guards, a future "let's enable web_search to help with
literature lookups" PR could silently punch a hole in the privacy
boundary — the model would gain a way to talk to the open internet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nora.provider.openai import (
    FORBIDDEN_BUILTIN_TYPES,
    OpenAISession,
    build_openai_tools,
    _verify_lockdown,
)
from nora.provider.tool_schemas import build_tool_specs


# ---------------------------------------------------------------------------
# Static checks on the tool list
# ---------------------------------------------------------------------------

def test_tool_list_contains_only_function_tools():
    tools = build_openai_tools()
    # Sanity floor: drop-out below this would mean the schema list
    # got truncated. We don't pin an exact count any more because
    # new tools land here naturally as the surface grows.
    assert len(tools) >= 6
    # Match the canonical specs exactly. That's the real invariant
    # (covered separately by test_tool_list_names_match_canonical_specs)
    # and stops a stray duplicate from sneaking in.
    assert len(tools) == len(build_tool_specs())
    assert all(t.get("type") == "function" for t in tools), (
        "every Nora tool sent to OpenAI must be a function tool"
    )


def test_tool_list_names_match_canonical_specs():
    tools = build_openai_tools()
    sent_names = {t["name"] for t in tools}
    expected_names = {s.name for s in build_tool_specs()}
    assert sent_names == expected_names


def test_tool_list_excludes_every_known_builtin():
    tools = build_openai_tools()
    sent_types = {t.get("type") for t in tools}
    # Every entry must be the "function" type — none of the built-in
    # types may ever appear.
    forbidden_seen = sent_types & FORBIDDEN_BUILTIN_TYPES
    assert not forbidden_seen, (
        f"built-in types leaked into the tool list: {forbidden_seen}"
    )


def test_lockdown_verifier_rejects_each_forbidden_type():
    """Synthetically inject every known built-in into the tool list
    and confirm the verifier raises. Mirrors the SDK lockdown test
    that ensures we know how to spot each kind of escape."""
    base = build_openai_tools()
    for builtin in FORBIDDEN_BUILTIN_TYPES:
        bad = base + [{"type": builtin}]
        with pytest.raises(RuntimeError, match="lockdown"):
            _verify_lockdown(bad)


def test_lockdown_verifier_rejects_unknown_function_name():
    """A non-Nora function tool name must also be rejected — covers
    the case of someone "helpfully" appending a custom helper."""
    base = build_openai_tools()
    bad = base + [{
        "type": "function",
        "name": "exfiltrate_data",
        "description": "evil",
        "parameters": {"type": "object", "properties": {}},
    }]
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(bad)


# ---------------------------------------------------------------------------
# Lockdown is checked on every request
# ---------------------------------------------------------------------------


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeResponse:
    """Minimal Responses-API-shaped object: an ``output`` list with
    one finished message, plus a usage block. Used to short-circuit
    the tool-loop in send() so the test only verifies what was sent,
    not what comes back."""

    def __init__(self) -> None:
        # One assistant text item, no function_calls — send() exits
        # the tool loop after the first round.
        class _Block:
            type = "output_text"
            text = "ok"

        class _Item:
            type = "message"
            content = [_Block()]

            def model_dump(self) -> dict[str, Any]:
                return {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}

        self.output = [_Item()]
        self.usage = _FakeUsage()


class _FakeResponsesAPI:
    """Captures every ``create()`` call's kwargs so the test can
    assert what was sent."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI``. The test substitutes one
    of these for the real client by monkey-patching the SDK import
    inside ``OpenAISession.open()``."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.responses = _FakeResponsesAPI()

    async def close(self) -> None:
        return None


def test_send_only_passes_function_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Drive a real send() against a fake OpenAI client and assert
    the ``tools`` kwarg contains exactly Nora's function tools, and
    nothing else. This is the test that catches a future PR adding
    a built-in (web_search, code_interpreter, file_search,
    image_generation, MCP) to the request."""
    import asyncio

    # Stub out the keyring resolver so we don't need a real OpenAI key.
    from nora.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    # Make AsyncOpenAI inside the lazy import resolve to our fake.
    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FakeAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="you are nora",
    )

    async def _drive() -> None:
        async for _ in sess.send("hello"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 1, "expected exactly one Responses-API call"
    call = api.calls[0]

    # Lockdown assertions. Floor of 6 (the original locked surface)
    # plus exact match against the canonical spec count, so a future
    # PR adding a tool flows through naturally but a duplicate or a
    # stray built-in does not.
    tools = call.get("tools")
    assert tools is not None and len(tools) >= 6
    assert len(tools) == len(build_tool_specs())
    assert all(t.get("type") == "function" for t in tools)
    sent_names = {t["name"] for t in tools}
    expected = {s.name for s in build_tool_specs()}
    assert sent_names == expected
    assert not any(t.get("type") in FORBIDDEN_BUILTIN_TYPES for t in tools)

    # store=True is required for reasoning models (gpt-5.5-pro and
    # the like): the model emits ``reasoning`` items with ``rs_…``
    # ids that the next round-trip references by id. server-side
    # persistence of response *objects* is retrievability only — it
    # doesn't change OpenAI's data-retention posture (that's set
    # independently in the account's data-handling settings).
    assert call.get("store") is True


# ---------------------------------------------------------------------------
# previous_response_id chaining (the token-saving refactor)
# ---------------------------------------------------------------------------
#
# The session must thread ``previous_response_id`` through every
# round-trip after the first. Without this the server holds the
# conversation but we keep paying full re-replay cost on every turn.
# The test below pins:
#   - Turn 1, round 1: no previous_response_id, input is just the user msg.
#   - Turn 1, round 2 (after a tool call): previous_response_id is the
#     id from round 1, input is just the function_call_output.
#   - Turn 2, round 1: previous_response_id is the id from the last
#     round of turn 1, input is just the new user msg (no replay).


class _ScriptedResponse:
    """Drop-in for ``_FakeResponse`` that lets the test specify the
    response id and whether the round emits a function_call (loop
    continues) vs. a plain message (loop exits)."""

    def __init__(self, response_id: str, *, with_tool_call: bool) -> None:
        self.id = response_id
        self.usage = _FakeUsage()
        if with_tool_call:
            class _Call:
                type = "function_call"
                name = "list_results"
                call_id = "call_xyz"
                arguments = "{}"

                def model_dump(self) -> dict[str, Any]:
                    return {
                        "type": "function_call",
                        "name": "list_results",
                        "call_id": "call_xyz",
                        "arguments": "{}",
                    }

            self.output = [_Call()]
        else:
            class _Block:
                type = "output_text"
                text = "ok"

            class _Msg:
                type = "message"
                content = [_Block()]

                def model_dump(self) -> dict[str, Any]:
                    return {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }

            self.output = [_Msg()]


class _ScriptedResponsesAPI:
    """Returns a queue of pre-built responses in order, capturing the
    kwargs of each call for assertions."""

    def __init__(self, responses: list[_ScriptedResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _ScriptedResponse:
        self.calls.append(kwargs)
        if not self._queue:
            raise RuntimeError("test queue exhausted")
        return self._queue.pop(0)


class _ScriptedAsyncOpenAI:
    def __init__(
        self, api_key: str | None = None, *, responses: list[_ScriptedResponse],
    ) -> None:
        self.api_key = api_key
        self.responses = _ScriptedResponsesAPI(responses)

    async def close(self) -> None:
        return None


def test_previous_response_id_chains_across_rounds_and_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two turns with one tool-loop iteration each in turn 1.

    Expected request sequence:
      call 1: turn 1 round 1 — user msg only, no previous_response_id
      call 2: turn 1 round 2 — function_call_output, prev=resp_1
      call 3: turn 2 round 1 — user msg only, prev=resp_2

    The server holds prior assistant / tool / reasoning content via
    the chain, so each request input shrinks to just the new content.
    """
    import asyncio

    from nora.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    # Three scripted responses driving two turns:
    scripted = [
        _ScriptedResponse("resp_1", with_tool_call=True),    # turn 1 r1
        _ScriptedResponse("resp_2", with_tool_call=False),   # turn 1 r2
        _ScriptedResponse("resp_3", with_tool_call=False),   # turn 2 r1
    ]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None: _ScriptedAsyncOpenAI(api_key, responses=scripted),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="you are nora",
    )

    async def _drive() -> None:
        async for _ in sess.send("turn one"):
            pass
        async for _ in sess.send("turn two"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 3, (
        f"expected 3 round-trips (2 in turn 1, 1 in turn 2); got {len(api.calls)}"
    )

    c1, c2, c3 = api.calls

    # Call 1: fresh chain, no prior id.
    assert "previous_response_id" not in c1, (
        "first call of a fresh session must NOT carry previous_response_id"
    )
    assert len(c1["input"]) == 1
    assert c1["input"][0]["role"] == "user"

    # Call 2: chained to resp_1, body is ONLY the function_call_output —
    # no replay of the prior assistant / function_call items.
    assert c2.get("previous_response_id") == "resp_1"
    assert len(c2["input"]) == 1
    assert c2["input"][0]["type"] == "function_call_output"
    assert c2["input"][0]["call_id"] == "call_xyz"

    # Call 3 (turn 2): chained to the LAST id of turn 1 (resp_2),
    # body is ONLY the new user message.
    assert c3.get("previous_response_id") == "resp_2"
    assert len(c3["input"]) == 1
    assert c3["input"][0]["role"] == "user"


def test_request_failure_does_not_advance_committed_response_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If a turn fails mid-flight, ``_last_response_id`` must stay at
    the last committed turn's id so the next user message threads onto
    a coherent point in the chain rather than a half-done response."""
    import asyncio

    from nora.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    class _FailingResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._call_count = 0
            self._first_response = _ScriptedResponse("resp_good", with_tool_call=False)

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return self._first_response
            # Second turn: fail. _last_response_id should remain
            # ``resp_good`` for any subsequent turn to chain on.
            raise RuntimeError("server returned 500: simulated failure")

    class _FlakyAsyncOpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.responses = _FailingResponsesAPI()

        async def close(self) -> None:
            return None

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FlakyAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="you are nora",
    )

    async def _drive() -> None:
        async for _ in sess.send("first"):
            pass
        # Second send fails; just drain the events.
        async for _ in sess.send("second"):
            pass

    asyncio.run(_drive())

    # The committed pointer is the id from the successful first turn.
    assert sess._last_response_id == "resp_good", (
        "a failed turn must not overwrite the committed response id"
    )


def test_send_yields_turnerror_when_tool_loop_does_not_converge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A model that emits a function_call on every one of
    MAX_TOOL_ROUNDS rounds must produce a TurnError, not TurnDone.

    Before the fix, the for-loop's natural exhaustion fell through to
    the same TurnDone branch as a clean exit, so a runaway tool-using
    model looked indistinguishable from a normal completion to the UI.
    """
    import asyncio

    from nora.provider import openai as openai_provider
    from nora.provider.base import TurnDone, TurnError
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    # Sixteen scripted responses, each requesting a function_call. This
    # is exactly the number range(MAX_TOOL_ROUNDS) gives us, so the
    # loop exhausts on iteration 16 without ever seeing a plain message
    # — the path the old code mishandled.
    scripted = [
        _ScriptedResponse(f"resp_{i}", with_tool_call=True)
        for i in range(16)
    ]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None: _ScriptedAsyncOpenAI(api_key, responses=scripted),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="you are nora",
    )

    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send("loop forever"):
            events.append(ev)

    asyncio.run(_drive())

    assert events, "send() yielded no events"
    assert not any(isinstance(e, TurnDone) for e in events), (
        "exhausted tool loop must NOT emit TurnDone; that's the silent "
        "truncation the fix prevents"
    )
    assert isinstance(events[-1], TurnError), (
        f"last event must be TurnError; got {type(events[-1]).__name__}"
    )
    # Loop ran exactly MAX_TOOL_ROUNDS times before giving up — no
    # short-circuit, no overrun.
    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 16
    # The chain head still advances through every successful round-trip
    # so a follow-up user message threads onto the last response we got.
    assert sess._last_response_id == "resp_15"

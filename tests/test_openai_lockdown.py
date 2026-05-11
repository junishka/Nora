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
    _mcp_payload_to_text,
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
# Image-bearing tool results — pixels can't ride function_call_output
# ---------------------------------------------------------------------------


def test_text_only_payload_passes_through_unchanged():
    """A normal tool result with only a text content block is
    forwarded verbatim — no rewrite, no provider-specific noise."""
    payload = {
        "content": [
            {"type": "text", "text": '{"status":"ok","result_id":"M1"}'},
        ]
    }
    assert _mcp_payload_to_text(payload) == (
        '{"status":"ok","result_id":"M1"}'
    )


def test_image_payload_descriptor_is_rewritten_for_openai():
    """``read_attached_file`` returns a hedged "if your provider
    doesn't support images" descriptor alongside an image block. On
    OpenAI the image bytes can't ride the function_call_output, so
    the descriptor must be rewritten to tell the model definitively
    that the image was dropped — and to point at the recovery path
    (re-@mention, which uses the user-message vision channel)."""
    descriptor = json.dumps({
        "status": "ok",
        "name": "residuals.png",
        "kind": "image",
        "ext": ".png",
        "mime": "image/png",
        "size": 12345,
        "note": (
            "The image is attached as an inline content block. "
            "If your provider doesn't support image tool results, "
            "ask the researcher to re-@mention the file in their "
            "next message."
        ),
    })
    payload = {
        "content": [
            {"type": "image", "data": "BASE64...", "mimeType": "image/png"},
            {"type": "text", "text": descriptor},
        ]
    }
    rewritten = _mcp_payload_to_text(payload)
    parsed = json.loads(rewritten)
    assert parsed["status"] == "image_not_supported_on_provider"
    assert parsed["name"] == "residuals.png"
    # The model is told what to ask the researcher to do, definitively.
    assert "re-@mention" in parsed["reason"]
    assert "residuals.png" in parsed["reason"]
    # The hedged "If your provider doesn't support" wording must NOT
    # leak through — that conditional was the whole problem.
    assert "If your provider" not in rewritten


def test_malformed_descriptor_falls_back_to_original_text():
    """If the descriptor isn't the JSON shape we expect (e.g. an
    older tool, a hand-written test fixture, an MCP server we don't
    own), the rewrite path must NOT raise — fall back to the
    original text. The rewrite is best-effort polish, not a load-
    bearing parse."""
    payload = {
        "content": [
            {"type": "image", "data": "BASE64...", "mimeType": "image/png"},
            {"type": "text", "text": "not json"},
        ]
    }
    out = _mcp_payload_to_text(payload)
    assert out == "not json"


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


def test_previous_response_id_expiry_yields_context_reset_turn_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Continuity closure: when OpenAI reports that the
    ``previous_response_id`` we tried to chain on has expired (the
    server-side retention window elapsed, or the response was
    deleted), the provider:

      1. Resets ``_last_response_id = None`` so the next user
         message starts a fresh chain.
      2. Yields a ``TurnError`` with ``context_reset=True``.

    The flag is the signal to the runner that the provider's
    server-side memory is gone and the next turn MUST re-prime via
    the warm-start context prefix. Without it, an established
    session that hits chain expiry would silently start a new chain
    with no ``previous_response_id`` AND no context prefix — the
    model would see the next user turn as the first message in a
    brand-new conversation.

    Earlier we set ``_last_response_id`` to a "stale" value to
    simulate the situation where Nora believes it has a usable
    chain pointer; the scripted server then refuses with a 404 /
    "response not found" shape.
    """
    import asyncio

    from nora.provider import openai as openai_provider
    from nora.provider.base import TurnError
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    class _ExpiringResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise RuntimeError(
                "openai.NotFoundError: previous_response_id "
                "'resp_expired' not found (404)"
            )

    class _ExpiringAsyncOpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.responses = _ExpiringResponsesAPI()

        async def close(self) -> None:
            return None

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI", _ExpiringAsyncOpenAI, raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="you are nora",
    )
    # Simulate an established session — there's a committed
    # response id from a prior turn that the server has since
    # expired.
    await_open = sess.open()
    asyncio.run(await_open)
    sess._last_response_id = "resp_expired"

    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send("continue the analysis"):
            events.append(ev)

    asyncio.run(_drive())

    # Provider must reset the committed pointer so the next turn
    # starts a fresh chain.
    assert sess._last_response_id is None, (
        "chain-expiry handling must clear _last_response_id so the "
        "next turn doesn't re-send the dead id"
    )
    # Locate the TurnError; it MUST carry context_reset=True so the
    # runner re-arms ``needs_context_prefix`` for the next turn.
    errors = [e for e in events if isinstance(e, TurnError)]
    assert len(errors) == 1, (
        f"expected exactly one TurnError; got {len(errors)}"
    )
    err = errors[0]
    assert err.context_reset is True, (
        "TurnError from chain expiry must set context_reset=True so "
        "the runner re-arms needs_context_prefix for the next turn"
    )


def test_handler_exception_does_not_leak_message_to_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When a tool handler raises, the OpenAI tool loop's catch-all
    fallback must NOT interpolate ``str(e)`` into the model-visible
    reason. The exception message can include parser excerpts,
    file paths, or raw data values that per-tool error handling
    would have redacted. Return only the exception class (a bounded
    identifier) plus a generic recovery hint; log full details
    locally.

    The sentinel text is constructed to look like the kind of
    content the per-tool redaction pass would scrub: a row of
    data with a name, dollar amount, and email. If any of those
    tokens reach the function_call_output the fix didn't hold.
    """
    import asyncio

    from nora.provider import openai as openai_provider
    from nora.tools import HANDLERS
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    SENSITIVE = (
        "row 42: name=Jane Doe income=$487192 email=jane.doe@example.com"
    )

    async def _raising_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(SENSITIVE)

    # The scripted tool call names ``list_results``; route THAT handler
    # to our raising one for the test only. monkeypatch undoes the
    # change at teardown so other tests aren't affected.
    monkeypatch.setitem(HANDLERS, "list_results", _raising_handler)

    # Two scripted responses: round 1 emits the function_call; round 2
    # (after our raising handler) emits a clean message so the loop
    # exits and we can inspect what the provider sent back in
    # ``function_call_output``.
    scripted = [
        _ScriptedResponse("resp_call", with_tool_call=True),
        _ScriptedResponse("resp_done", with_tool_call=False),
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
        async for _ in sess.send("trigger the failing tool"):
            pass
    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 2, "expected exactly two round-trips"
    # The function_call_output that travels back to the model on
    # round 2 carries the tool result.
    round2 = api.calls[1]
    func_outputs = [
        item for item in round2["input"]
        if item.get("type") == "function_call_output"
    ]
    assert func_outputs, "round 2 must carry the tool's output back"
    output_text = func_outputs[0]["output"]
    # Critical: the raw exception message must NOT appear.
    assert SENSITIVE not in output_text, (
        f"sensitive exception text leaked into model-visible tool "
        f"result: {output_text!r}"
    )
    assert "Jane Doe" not in output_text
    assert "jane.doe@example.com" not in output_text
    assert "$487192" not in output_text
    # The class name is a bounded identifier and stays — the model
    # can route on it without seeing the redacted detail.
    assert "RuntimeError" in output_text
    # Generic recovery hint surfaces.
    assert "Retry" in output_text or "fall back" in output_text


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
    # Pre-set the committed pointer so we can verify it is NOT
    # overwritten by the exhausted turn — chaining onto the last
    # in-turn response would point at an unsatisfied function_call.
    sess._last_response_id = "prior_clean_turn"

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
    # _last_response_id stays at the prior clean turn. Promoting to
    # ``resp_15`` would chain the NEXT user message onto a response
    # that still has an unsatisfied function_call (its
    # function_call_output items lived in our local pending_input,
    # never sent), so the server would either reject the chain or
    # continue with inconsistent context.
    assert sess._last_response_id == "prior_clean_turn", (
        "exhausted tool loop must NOT advance the committed response id "
        "onto a turn whose last response has a pending function_call"
    )

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

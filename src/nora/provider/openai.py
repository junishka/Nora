"""OpenAI-backed ``ProviderSession`` implementation.

Wraps the OpenAI Responses API behind the same ``ProviderSession``
contract Anthropic implements. Tool registration draws from
``provider.tool_schemas`` (the canonical, provider-neutral list);
tool dispatch routes back to the same handler functions in
``nora.tools.HANDLERS`` Anthropic uses, so the privacy semantics —
sandbox-wrapped execution, sanitizer-clamped results — are byte-for-
byte identical regardless of which model authored the call.

Lockdown discipline is the headline guarantee:

- The ``tools`` list sent to OpenAI contains EXACTLY the six Nora
  function tools and nothing else. No ``{"type": "web_search"}``,
  ``{"type": "code_interpreter"}``, ``{"type": "file_search"}``,
  no Agents-SDK built-ins. ``test_openai_lockdown.py`` mocks the
  client and asserts this on every request.
- ``parallel_tool_calls`` is on (the default) so the model can run
  ``get_schema`` and ``request_data`` in parallel during exploration,
  but every dispatch goes through the SAME ``HANDLERS`` map and the
  SAME sanitizer/sandbox boundary.
- The OpenAI client is constructed with an explicit ``api_key``
  pulled from the keyring (``nora.auth``). Env-var
  ``OPENAI_API_KEY`` is also honored as a fallback for power users
  who'd rather export their own.

Conversation state: the bridge prepends its own context prefix on
the first turn after open (same pattern as Anthropic), so
``previous_response_id`` is not used. A persistent ``_input`` list
on the session accumulates messages across turns within the
session's lifetime. New session = new conversation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from nora.provider.base import (
    AssistantText,
    AuthFailure,
    Event,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from nora.provider.tool_schemas import build_tool_specs
from nora.tools import HANDLERS


PROVIDER_ID = "openai"


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------


def detect_auth() -> str:
    """Return ``'api_key'`` if any OpenAI credential source is
    present, ``'unknown'`` otherwise. OpenAI has no Nora-supported
    subscription path — only API keys are recognised here."""
    if os.environ.get("OPENAI_API_KEY"):
        return "api_key"
    from nora import auth as _auth
    if _auth.has_credential("openai"):
        return "api_key"
    return "unknown"


def _resolve_api_key() -> str | None:
    """Pick the OpenAI API key from env first (researcher override),
    keyring second. Returns ``None`` if neither is set."""
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env
    from nora import auth as _auth
    return _auth.get_credential("openai")


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
#
# These three helpers are kept module-level (not session methods) so
# the lockdown test can call them directly without spinning up a real
# session — the test imports ``build_openai_tools`` and asserts the
# returned list matches the canonical six tools, no others.


def build_openai_tools() -> list[dict[str, Any]]:
    """Build the OpenAI Responses-API tool list from the canonical
    spec. ALWAYS returns exactly the six Nora tools — never any
    Responses-API built-ins (``web_search``, ``code_interpreter``,
    ``file_search``, …)."""
    return [spec.as_openai_tool() for spec in build_tool_specs()]


# Names of OpenAI Responses-API built-in tools the lockdown forbids.
# Any future built-in additions should be added here AND named in the
# lockdown test so a regression is caught at the API surface, not at
# runtime.
FORBIDDEN_BUILTIN_TYPES: frozenset[str] = frozenset({
    "web_search",
    "web_search_preview",
    "computer_use_preview",
    "code_interpreter",
    "file_search",
    "image_generation",
    "mcp",  # remote MCP servers; Nora's MCP runs in-process only
})


def _verify_lockdown(tools: list[dict[str, Any]]) -> None:
    """Last-line guard: raise if anything in ``tools`` is a built-in
    type or otherwise off-allowlist. This is checked on every
    request, not just at startup, because a tool list that round-trips
    through serialisation could in principle pick up extra entries."""
    expected_names = {s.name for s in build_tool_specs()}
    for t in tools:
        ttype = t.get("type")
        if ttype != "function":
            raise RuntimeError(
                f"Nora lockdown violation: tools list contains "
                f"non-function entry of type {ttype!r}"
            )
        if ttype in FORBIDDEN_BUILTIN_TYPES:
            raise RuntimeError(
                f"Nora lockdown violation: forbidden built-in {ttype!r}"
            )
        if t.get("name") not in expected_names:
            raise RuntimeError(
                f"Nora lockdown violation: unknown function tool "
                f"{t.get('name')!r}"
            )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class OpenAISession:
    """OpenAI Responses-API session.

    Maintains an in-memory ``_input`` list of messages for the
    duration of the session — this is what makes multi-turn
    conversation work (each ``send()`` appends to it, the next
    ``send()`` sends the whole history). The bridge's session
    lifecycle therefore IS the conversation lifecycle on this
    provider.
    """

    PROVIDER = PROVIDER_ID

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self._system_prompt = system_prompt
        # ``continue_conversation`` is Anthropic-specific (CLI session
        # store); accepted for interface symmetry but ignored here.
        del continue_conversation
        self._client: Any = None
        # Accumulated conversation in Responses-API ``input`` shape.
        # Each turn appends one user message + the assistant's text/
        # function_call / function_call_output items.
        self._input: list[dict[str, Any]] = []
        # Cached tool list — same six function tools for every call.
        # Built once at open() rather than per-send so the lockdown
        # check has a stable reference.
        self._tools: list[dict[str, Any]] = []

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        if self._client is not None:
            return
        # Lazy import: keeps the openai SDK off Anthropic-only code
        # paths and means a missing dep manifests as a clean
        # AuthFailure instead of an import error at startup.
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover — openai is a dep
            raise RuntimeError(
                f"openai SDK not installed: {e}. Reinstall Nora dependencies."
            )
        api_key = _resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "no OpenAI API key configured. Add one in the auth "
                "screen or set OPENAI_API_KEY in the environment."
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._tools = build_openai_tools()
        # Lockdown verified at session open AND at every send (defense
        # in depth — a future caller could in principle mutate
        # _tools).
        _verify_lockdown(self._tools)

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._input = []
        self._tools = []
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — close-time errors aren't useful
                pass

    # ---- model swap ------------------------------------------------------

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch active OpenAI model. The Responses API accepts the
        model id per request, so swapping is just a field change —
        no client rebuild. Conversation state is preserved."""
        from nora.provider.catalog import get_model

        try:
            info = get_model(model_id)
        except KeyError:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        if info.provider != self.PROVIDER:
            return {
                "ok": False,
                "reason": (
                    f"model {model_id!r} belongs to provider "
                    f"{info.provider!r}, not OpenAI"
                ),
            }
        if model_id == self.model:
            return {
                "ok": True,
                "model": model_id,
                "label": info.label,
                "context_window": info.context_window,
                "unchanged": True,
            }
        self.model = model_id
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
        }

    # ---- send ------------------------------------------------------------

    async def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one chat turn. Yields a flat stream of provider-neutral
        events terminated by ``TurnDone`` / ``TurnError`` /
        ``AuthFailure``.

        Internally drives the OpenAI tool-loop: each round-trip is one
        ``responses.create()`` call. If the response contains
        ``function_call`` items, dispatch them via ``HANDLERS``,
        append outputs to ``_input``, and loop. Loop exits when the
        response has no further function calls.
        """
        await self.open()
        client = self._client
        assert client is not None

        # Append the new user message.
        user_content = _build_user_content(prompt, images)
        self._input.append({"role": "user", "content": user_content})

        # Bound the tool-loop iterations so a runaway model can't pin
        # the loop forever. 16 is generous — most analyses use 1–4.
        MAX_TOOL_ROUNDS = 16
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                _verify_lockdown(self._tools)
                try:
                    resp = await client.responses.create(
                        model=self.model,
                        instructions=self._system_prompt,
                        input=self._input,
                        tools=self._tools,
                        tool_choice="auto",
                        # store=True (the default) is required for
                        # reasoning models: they emit ``reasoning``
                        # items with ids like ``rs_…`` that the next
                        # round-trip in the tool loop references by
                        # id. With ``store=False`` OpenAI throws
                        # ``Item with id 'rs_…' not found`` on the
                        # second round. Server-side persistence of
                        # response objects is just retrievability;
                        # it doesn't change OpenAI's data-retention
                        # posture, which is governed separately by
                        # the account's API data-handling settings.
                        store=True,
                    )
                except Exception as e:  # noqa: BLE001 — translate to event
                    msg = str(e)
                    lower = msg.lower()
                    if "auth" in lower or "api key" in lower or "401" in lower:
                        yield AuthFailure(reason=f"OpenAI auth failure: {msg}")
                        return
                    yield TurnError(message=f"OpenAI request failed: {msg}")
                    return

                # Track usage. Only the LAST round's output_tokens
                # counts as the user-visible "this turn produced N
                # output tokens", but input_tokens accumulates across
                # rounds.
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                    total_output_tokens = getattr(usage, "output_tokens", 0) or 0

                output = list(getattr(resp, "output", []) or [])
                # Translate items + decide whether to keep looping.
                pending_calls: list[tuple[str, str, str]] = []  # (call_id, name, args_json)
                for item in output:
                    itype = getattr(item, "type", None)
                    if itype == "message":
                        text = _extract_message_text(item)
                        if text and text.strip():
                            yield AssistantText(text=text)
                        # Append the assistant message to the
                        # conversation so subsequent turns see it.
                        self._input.append(_serialize_item(item))
                    elif itype == "function_call":
                        name = getattr(item, "name", "")
                        call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                        args_json = getattr(item, "arguments", "") or "{}"
                        pending_calls.append((call_id, name, args_json))
                        yield ToolCall(
                            name=name,
                            input=_safe_json(args_json),
                            call_id=call_id,
                        )
                        self._input.append(_serialize_item(item))
                    elif itype in ("reasoning", "reasoning_summary"):
                        # OpenAI reasoning is opaque text; don't
                        # surface as AssistantThinking (which Anthropic
                        # uses for visible chain-of-thought). Append
                        # to input so subsequent rounds see it.
                        self._input.append(_serialize_item(item))
                    else:
                        # Unknown item type — preserve to keep the
                        # conversation coherent.
                        self._input.append(_serialize_item(item))

                if not pending_calls:
                    break  # no more tool calls; turn is done

                # Dispatch every function_call in this round (parallel-
                # friendly but executed serially here; the underlying
                # handlers aren't expected to be concurrency-safe yet).
                for call_id, name, args_json in pending_calls:
                    handler = HANDLERS.get(name)
                    if handler is None:
                        out_text = json.dumps({
                            "status": "error",
                            "reason": f"unknown tool: {name!r}",
                        })
                        yield ToolCallResult(
                            call_id=call_id, text=out_text, is_error=True,
                        )
                        self._input.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": out_text,
                        })
                        continue
                    try:
                        args = _safe_json(args_json)
                        result = await handler(args)
                        out_text = _mcp_payload_to_text(result)
                        is_error = False
                    except Exception as e:  # noqa: BLE001
                        out_text = json.dumps({
                            "status": "error",
                            "reason": f"handler raised: {e.__class__.__name__}: {e}",
                        })
                        is_error = True
                    run_dir, language = _extract_hints(out_text)
                    yield ToolCallResult(
                        call_id=call_id,
                        text=out_text,
                        is_error=is_error,
                        run_dir=run_dir,
                        language=language,
                    )
                    self._input.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out_text,
                    })

            yield TurnDone(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                # Cache fields are Anthropic-specific; leave None.
                # ``cost_usd`` is also None — Nora doesn't compute
                # OpenAI costs locally.
            )
        except Exception as e:  # noqa: BLE001 — last-line catch
            yield TurnError(message=f"OpenAI session error: {e}")

    # ---- async-context-manager sugar ------------------------------------

    async def __aenter__(self) -> "OpenAISession":
        await self.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_content(
    prompt: str, images: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Build a Responses-API user message content list. Text is one
    ``input_text`` block; each image is one ``input_image`` block
    with a base64 data URL."""
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "input_text", "text": prompt})
    if images:
        for img in images:
            mime = img.get("mime", "image/png")
            data = img.get("data", "")
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{data}",
            })
    return content


def _extract_message_text(item: Any) -> str:
    """Pull the concatenated output text from a Responses-API message
    item. Items may carry multiple text blocks (e.g., when reasoning
    interleaves)."""
    parts: list[str] = []
    content = getattr(item, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None)
        if btype in ("output_text", "text"):
            t = getattr(block, "text", None) or ""
            if t:
                parts.append(t)
    return "".join(parts)


# Fields the Responses API includes on OUTPUT items but rejects when
# the same items are sent back as INPUT. Server-only metadata —
# ``status`` is the headline offender; pydantic ``model_dump()``
# emits it on every message / function_call / reasoning item, so
# round-tripping a turn back into the next request triggers a 400
# "Unknown parameter: 'input[N].status'" without this strip.
_INPUT_FORBIDDEN_FIELDS: frozenset[str] = frozenset({"status"})


def _serialize_item(item: Any) -> dict[str, Any]:
    """Convert a Responses-API output item back to a dict shape the
    next ``responses.create()`` call accepts as ``input``.

    The SDK's pydantic models support ``model_dump()`` which emits
    the full server-side shape, including fields like ``status`` that
    the input schema rejects. We dump, then strip the server-only
    fields and drop any None defaults (which the input schema also
    refuses for some types).
    """
    raw: dict[str, Any] = {}
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(item, attr, None)
        if callable(fn):
            try:
                raw = fn()
                break
            except Exception:  # noqa: BLE001
                pass
    if not raw:
        raw = {"type": getattr(item, "type", "unknown")}
    return {
        k: v for k, v in raw.items()
        if k not in _INPUT_FORBIDDEN_FIELDS and v is not None
    }


def _safe_json(s: str) -> dict[str, Any]:
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (ValueError, TypeError):
        return {}
    if not isinstance(out, dict):
        return {}
    return out


def _mcp_payload_to_text(payload: Any) -> str:
    """Nora handlers return MCP-shaped payloads:
    ``{"content": [{"type": "text", "text": "..."}]}``. Extract the
    JSON-text body for the OpenAI tool output."""
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        # Already-flat dict: serialise.
        return json.dumps(payload)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _extract_hints(text: str) -> tuple[str | None, str | None]:
    """Same hint-extraction as the Anthropic side: peel ``_run_dir``
    and ``_language`` from the tool-result JSON if present so the UI
    can render raw R/Stata output and the right "Open in …" button."""
    if not text.strip():
        return None, None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    rd = parsed.get("_run_dir")
    lang = parsed.get("_language")
    return (
        rd if isinstance(rd, str) else None,
        lang if isinstance(lang, str) else None,
    )

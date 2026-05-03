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

Conversation state: each ``responses.create()`` call passes
``previous_response_id`` so the OpenAI server holds the prior
turns of the conversation; the bridge only sends the new content
for the current round (the user message on a fresh turn, the
function-call outputs between tool-loop rounds). The first turn
after open carries the bridge's warm-start context prefix as the
user message body — same pattern as Anthropic. New session =
new ``previous_response_id`` chain (the bridge does not resume
across opens; the warm-start prefix re-establishes context).

Token effect: on a long session this avoids re-sending the entire
conversation array on every turn. On the wire, each turn carries
one new user message plus per-round function-call outputs, not
the full N-turn replay.
"""

from __future__ import annotations

import json
import os
import sys
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
        # Server-side conversation chain pointer. Updated only after a
        # turn completes cleanly (no pending tool calls); within a turn
        # we walk a local pointer through each round so a mid-turn
        # failure leaves the committed pointer at the last good turn.
        self._last_response_id: str | None = None
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
        self._last_response_id = None
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

        # Round-1 input is just the new user message. Subsequent
        # rounds inside this turn carry only the function-call outputs
        # we produce locally — the prior assistant / function_call /
        # reasoning items already live on the server, reachable via
        # the response-id chain.
        user_content = _build_user_content(prompt, images)
        pending_input: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]

        # Walk a local pointer through each round of the tool loop.
        # Initialised from the last committed turn so the new turn
        # threads onto the prior conversation. Only promoted to
        # ``self._last_response_id`` after a clean turn end so a
        # mid-turn failure doesn't strand the chain on a half-done
        # response.
        turn_response_id: str | None = self._last_response_id

        # Bound the tool-loop iterations so a runaway model can't pin
        # the loop forever. 16 is generous — most analyses use 1–4.
        MAX_TOOL_ROUNDS = 16
        # We capture the LAST round's prompt size (not a cross-round
        # sum) because the Responses API reports ``input_tokens`` for
        # the FULL prompt at each round — cached prefix from
        # ``previous_response_id`` plus any new content this round.
        # Within a tool-loop turn the prompt grows monotonically as
        # tool outputs join the chain, so the last round naturally
        # captures the peak prompt size for the turn. Accumulating
        # with ``+=`` (the previous behavior) multi-counted the cached
        # prefix once per round, inflating the chip into nonsense on
        # tool-heavy turns.
        last_input_tokens = 0
        last_output_tokens = 0

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                _verify_lockdown(self._tools)
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": self._system_prompt,
                    "input": pending_input,
                    "tools": self._tools,
                    "tool_choice": "auto",
                    # store=True is required for reasoning models AND
                    # for ``previous_response_id`` chaining: the
                    # server has to retain the prior response object
                    # for the next call to reference it. Server-side
                    # persistence of response objects is just
                    # retrievability; it doesn't change OpenAI's
                    # data-retention posture (governed separately by
                    # the account's API data-handling settings).
                    "store": True,
                    # Surface context-window overruns as errors instead
                    # of letting the server silently drop the oldest
                    # items in the chain. With ``truncation="auto"``
                    # (which is the API default in some SDK versions),
                    # ``usage.input_tokens`` reports the truncated
                    # prompt size — making the context chip read
                    # smaller than the actual conversation, then
                    # smaller still as more turns get truncated. The
                    # chip's whole point is honesty about how full the
                    # window is; silent truncation defeats it. If we
                    # ever want to allow truncation as a UX choice,
                    # surface it in settings rather than baking it in
                    # at the request boundary.
                    "truncation": "disabled",
                }
                if turn_response_id is not None:
                    request_kwargs["previous_response_id"] = turn_response_id
                try:
                    resp = await client.responses.create(**request_kwargs)
                except Exception as e:  # noqa: BLE001 — translate to event
                    msg = str(e)
                    lower = msg.lower()
                    if "auth" in lower or "api key" in lower or "401" in lower:
                        yield AuthFailure(reason=f"OpenAI auth failure: {msg}")
                        return
                    yield TurnError(message=f"OpenAI request failed: {msg}")
                    return

                # Advance the in-turn pointer immediately so the next
                # round's request chains onto THIS response, not the
                # prior turn's tail.
                new_id = getattr(resp, "id", None)
                if isinstance(new_id, str) and new_id:
                    turn_response_id = new_id

                # Track usage. ``=`` not ``+=`` (see the rationale
                # above where last_input_tokens is initialized).
                # ``input_tokens`` already includes the cached prefix
                # for this round, so the last round's value is the
                # full prompt size at end-of-turn — exactly what the
                # "context occupied" chip wants.
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    last_input_tokens = getattr(usage, "input_tokens", 0) or 0
                    last_output_tokens = getattr(usage, "output_tokens", 0) or 0
                # Diagnostic: gated by NORA_DEBUG_USAGE. Mirrors the
                # Anthropic provider's logging so a head-to-head
                # comparison of the two providers' token accounting is
                # possible from the same on-disk file. Writes to
                # stderr (visible if launched from a terminal) AND
                # appends to ``<cwd>/.nora-usage.log`` (always reachable
                # by the researcher regardless of launch method —
                # pywebview swallows stderr on a double-clicked app).
                if os.environ.get("NORA_DEBUG_USAGE") == "1" and usage is not None:
                    cached = (
                        getattr(getattr(usage, "input_tokens_details", None),
                                "cached_tokens", 0) or 0
                    )
                    line = (
                        f"[nora.usage.openai] round model={self.model} "
                        f"input_tokens={last_input_tokens} "
                        f"output_tokens={last_output_tokens} "
                        f"cached_tokens={cached} "
                        f"(cached is a SUBSET of input_tokens, not additive)"
                    )
                    print(line, file=sys.stderr, flush=True)
                    try:
                        with (self.cwd / ".nora-usage.log").open("a") as _f:
                            _f.write(line + "\n")
                    except Exception:  # noqa: BLE001 — diagnostic must never crash a turn
                        pass

                output = list(getattr(resp, "output", []) or [])
                # Translate items + decide whether to keep looping.
                # We do NOT accumulate output items locally — the
                # server already holds them via the response-id chain.
                pending_calls: list[tuple[str, str, str]] = []  # (call_id, name, args_json)
                for item in output:
                    itype = getattr(item, "type", None)
                    if itype == "message":
                        text = _extract_message_text(item)
                        if text and text.strip():
                            yield AssistantText(text=text)
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
                    # ``reasoning`` / ``reasoning_summary`` and any
                    # other item types are held server-side and
                    # carried forward implicitly by the chain — no
                    # local tracking needed.

                if not pending_calls:
                    # Clean turn end. Promote the in-turn pointer to the
                    # durable session field and emit TurnDone here —
                    # falling through to the post-loop branch would also
                    # emit TurnDone on the exhausted-rounds path, which
                    # silently truncates a model that's still requesting
                    # tools. That path is now treated as TurnError below.
                    self._last_response_id = turn_response_id
                    yield TurnDone(
                        input_tokens=last_input_tokens,
                        output_tokens=last_output_tokens,
                        # Cache fields stay None on this path. OpenAI's
                        # ``cached_tokens`` is a subset of ``input_tokens``
                        # (not additive), so emitting it through
                        # ``cache_read_input_tokens`` would double-count
                        # against any consumer that sums input + cache (the
                        # web context chip does). ``input_tokens`` already
                        # represents the full prompt size on the OpenAI
                        # path. ``cost_usd`` is also None — Nora doesn't
                        # compute OpenAI costs locally.
                    )
                    return

                # Dispatch every function_call in this round (parallel-
                # friendly but executed serially here; the underlying
                # handlers aren't expected to be concurrency-safe yet).
                # The next round's request will carry these outputs as
                # ``input`` plus ``previous_response_id`` pointing at
                # the response we just received.
                next_input: list[dict[str, Any]] = []
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
                        next_input.append({
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
                    next_input.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out_text,
                    })
                pending_input = next_input

            # Loop exhausted — the model kept requesting tools through
            # MAX_TOOL_ROUNDS without producing a final non-tool
            # response. Surface as TurnError so the caller sees the
            # truncation rather than a misleading "clean done".
            #
            # Do NOT promote ``turn_response_id`` here. The last response
            # in this turn carries an unsatisfied ``function_call`` and
            # we built ``function_call_output`` items in ``pending_input``
            # that we never sent back. Chaining the next user message
            # onto that response would either 400 server-side (open tool
            # call) or silently continue with inconsistent context. Roll
            # back to the prior committed turn instead — the next user
            # message threads onto the last clean state, and the
            # orphaned round-trips on the server are simply abandoned.
            yield TurnError(
                message=(
                    f"OpenAI tool loop did not converge within "
                    f"{MAX_TOOL_ROUNDS} rounds; last response still had "
                    f"pending function calls."
                ),
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

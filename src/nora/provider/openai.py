"""OpenAI-backed ``ProviderSession`` implementation.

Wraps the OpenAI Responses API behind the same ``ProviderSession``
contract Anthropic implements. Tool registration draws from
``provider.tool_schemas`` (the canonical, provider-neutral list);
tool dispatch routes back to the same handler functions in
``nora.tools.HANDLERS`` Anthropic uses, so the privacy semantics —
sandbox-wrapped execution, sanitizer-clamped results — are byte-for-
byte identical regardless of which model authored the call.

Lockdown discipline is the headline guarantee:

- The ``tools`` list sent to OpenAI contains EXACTLY the Nora
  function tools listed in ``build_tool_specs()`` and nothing else.
  No ``{"type": "web_search"}``, ``{"type": "code_interpreter"}``,
  ``{"type": "file_search"}``, no Agents-SDK built-ins.
  ``test_openai_lockdown.py`` mocks the client and asserts this on
  every request.
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
    AssistantThinking,
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
# returned list matches the canonical Nora tool set exactly, with no
# Responses-API built-ins mixed in.


def build_openai_tools() -> list[dict[str, Any]]:
    """Build the OpenAI Responses-API tool list from the canonical
    spec. Returns exactly the Nora tools defined in
    ``nora.provider.tool_schemas.TOOL_SPECS`` — never any Responses-API
    built-ins (``web_search``, ``code_interpreter``, ``file_search``,
    …)."""
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
    through serialisation could in principle pick up extra entries.

    The non-function check has to enumerate ``FORBIDDEN_BUILTIN_TYPES``
    explicitly before falling through to the generic non-function
    error: every entry in that frozenset is itself a non-function
    type, so a single ``ttype != "function"`` branch would fire on
    them too — but with a generic message that hides which
    forbidden built-in was attempted. Test cases that pin the
    "forbidden built-in {name}" message would never see it.
    """
    expected_names = {s.name for s in build_tool_specs()}
    for t in tools:
        ttype = t.get("type")
        # Forbidden built-ins first, so the error names the specific
        # built-in (web_search / code_interpreter / mcp / ...) rather
        # than the generic "non-function entry" message that would
        # otherwise cover the same input.
        if ttype in FORBIDDEN_BUILTIN_TYPES:
            raise RuntimeError(
                f"Nora lockdown violation: forbidden built-in {ttype!r}"
            )
        if ttype != "function":
            raise RuntimeError(
                f"Nora lockdown violation: tools list contains "
                f"non-function entry of type {ttype!r}"
            )
        if t.get("name") not in expected_names:
            raise RuntimeError(
                f"Nora lockdown violation: unknown function tool "
                f"{t.get('name')!r}"
            )


# ---------------------------------------------------------------------------
# Few-shot turn (structural, not prompt text)
# ---------------------------------------------------------------------------


# One demonstration exchange prepended to the very first round of every
# new session, BEFORE the real user message. The Responses API treats
# these items the same as real prior conversation: the model sees
# ``submit_script`` returning a payload whose ``markdown`` field is a
# pipe table, and an assistant reply that pastes that table verbatim
# followed by a single short sentence of interpretation. The point is
# behavioral demonstration of the desired output shape, parallel to the
# tool descriptions that say "Drop the markdown directly into your
# reply" for ``compose_results`` / ``expand_result``.
#
# Why few-shot rather than another system-prompt rule. Rules describe
# the desired behavior; demonstrations are the behavior. Demonstrations
# carry stronger pull because the model has now SEEN the pattern in its
# own conversation history. The same pull is unavailable here on the
# Anthropic provider because the Claude Agent SDK doesn't expose a
# point to seed prior assistant / tool_result turns; that path uses an
# embedded example in ``_STYLE_RIDER`` instead.
#
# Token cost. ~350 input tokens on round 1 of session 1. Re-rides
# implicitly via ``previous_response_id`` on every subsequent round in
# the same session, so it pays this cost once per session, not per
# turn.
#
# Stability of call_id. The ``fewshot_call_1`` id is a literal string
# the model never produces (handler dispatch is name-keyed, not id-
# keyed). It cannot collide with a real call because real ids come
# from the OpenAI server side and use a different format.
#
# Opt out for A/B testing: ``NORA_DISABLE_FEWSHOT=1``.
_FEWSHOT_USER_TEXT = (
    "What's the breakdown of `treatment` in this sample?"
)

_FEWSHOT_TOOL_ARGS = json.dumps({
    "language": "stata",
    "code": 'nora_result_tab treatment, label("Treatment assignment")',
    "label": "Treatment frequency",
})

_FEWSHOT_TOOL_OUTPUT = json.dumps({
    "results": [
        {
            "status": "ok",
            "result_id": "r_demo_treatment_freq",
            "label": "Treatment assignment",
            "type": "frequency_table",
            "markdown": (
                "| treatment | n   | %    |\n"
                "| --------- | --- | ---- |\n"
                "| control   | 487 | 49.4 |\n"
                "| treated   | 499 | 50.6 |"
            ),
        }
    ]
})

_FEWSHOT_ASSISTANT_TEXT = (
    "| treatment | n   | %    |\n"
    "| --------- | --- | ---- |\n"
    "| control   | 487 | 49.4 |\n"
    "| treated   | 499 | 50.6 |\n"
    "\n"
    "Balanced 50/50 assignment, n=986."
)


def _build_fewshot_items() -> list[dict[str, Any]]:
    """Return the few-shot exchange as Responses-API input items.

    Order matches a real prior turn: user message, function_call,
    function_call_output, assistant message. Round-trips byte-for-byte
    with what the server would emit for the same exchange, which is
    why the demonstration registers as "real history" rather than a
    style instruction.
    """
    return [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": _FEWSHOT_USER_TEXT}],
        },
        {
            "type": "function_call",
            "call_id": "fewshot_call_1",
            "name": "submit_script",
            "arguments": _FEWSHOT_TOOL_ARGS,
        },
        {
            "type": "function_call_output",
            "call_id": "fewshot_call_1",
            "output": _FEWSHOT_TOOL_OUTPUT,
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": _FEWSHOT_ASSISTANT_TEXT}
            ],
        },
    ]


def _fewshot_enabled() -> bool:
    """``NORA_DISABLE_FEWSHOT=1`` opts out for A/B testing."""
    return os.environ.get("NORA_DISABLE_FEWSHOT") != "1"


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
        effort: str | None = None,
    ) -> None:
        from nora.provider.catalog import normalize_effort

        self.cwd = cwd
        self.model = model
        self._system_prompt = system_prompt
        # Reasoning effort — sent per request in ``reasoning.effort``
        # (see the send loop), so a change applies on the next
        # message with no client rebuild and no conversation reset.
        self.effort: str = normalize_effort(effort)
        # ``continue_conversation`` is Anthropic-specific (CLI session
        # store); accepted for interface symmetry but ignored here.
        del continue_conversation
        self._client: Any = None
        # Server-side conversation chain pointer. Updated only after a
        # turn completes cleanly (no pending tool calls); within a turn
        # we walk a local pointer through each round so a mid-turn
        # failure leaves the committed pointer at the last good turn.
        self._last_response_id: str | None = None
        # Cached tool list — same function tools for every call.
        # Built once at open() rather than per-send so the lockdown
        # check has a stable reference.
        self._tools: list[dict[str, Any]] = []

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        """Lazy-build the AsyncOpenAI client if a key is available.

        Missing-key is NOT raised here: the provider contract terminates
        a turn with an ``AuthFailure`` event, and ``open()`` runs from
        ``SessionRunner.ensure_session`` BEFORE any event stream
        exists, so a ``RuntimeError`` at this layer would surface as a
        generic ``turn_error`` instead. To stay parity with the
        Anthropic path (whose ``open()`` doesn't fail on missing key
        either), we no-op here on a missing key and let
        :meth:`send` yield ``AuthFailure`` on the first round.
        """
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
            # Defer: ``send()`` checks ``self._client`` and emits
            # ``AuthFailure`` on the first round, matching how
            # Anthropic surfaces a missing key.
            return
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

    async def set_effort(self, effort: str) -> dict[str, Any]:
        """Switch reasoning effort. Sent per request, so this is a
        field change that takes effect on the next message — the
        ``previous_response_id`` chain is untouched, no reopen."""
        from nora.provider.catalog import EFFORT_LEVELS, get_effort

        if effort not in EFFORT_LEVELS:
            return {"ok": False, "reason": f"unknown effort level: {effort}"}
        info = get_effort(effort)
        if effort == self.effort:
            return {
                "ok": True, "effort": effort, "label": info.label,
                "unchanged": True,
            }
        self.effort = effort
        return {"ok": True, "effort": effort, "label": info.label}

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
        if client is None:
            # ``open()`` declined to build a client because no API key
            # is configured. Yield the provider-neutral ``AuthFailure``
            # event so the runner emits ``auth_failure`` — same shape
            # the API-call path uses for 401s further down. Without
            # this branch a missing key crashed ``ensure_session`` and
            # surfaced as a generic ``turn_error``, breaking parity
            # with the Anthropic provider.
            yield AuthFailure(reason=(
                "no OpenAI API key configured. Add one in the auth "
                "screen or set OPENAI_API_KEY in the environment."
            ))
            return

        # Round-1 input is just the new user message. Subsequent
        # rounds inside this turn carry only the function-call outputs
        # we produce locally — the prior assistant / function_call /
        # reasoning items already live on the server, reachable via
        # the response-id chain.
        user_content = _build_user_content(prompt, images)
        pending_input: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]

        # First-turn-only: prepend the few-shot demonstration exchange
        # (see _build_fewshot_items for rationale). The chain pointer is
        # None here because no real turn has committed yet on this
        # session. On every subsequent turn the few-shot rides for free
        # via previous_response_id, so we never re-prepend it. A mid-
        # session context-chain reset (chain expiry) drops
        # _last_response_id back to None, which correctly re-injects
        # the few-shot when the next turn rebuilds the chain from a
        # fresh root.
        if self._last_response_id is None and _fewshot_enabled():
            pending_input = _build_fewshot_items() + pending_input

        # Walk a local pointer through each round of the tool loop.
        # Initialised from the last committed turn so the new turn
        # threads onto the prior conversation. Only promoted to
        # ``self._last_response_id`` after a clean turn end so a
        # mid-turn failure doesn't strand the chain on a half-done
        # response.
        turn_response_id: str | None = self._last_response_id

        # Bound the tool-loop iterations so a runaway model can't pin
        # the loop forever. 16 is generous — most analyses use 1–4.
        # Env-overridable via ``NORA_OPENAI_MAX_TOOL_ROUNDS`` for
        # researchers running parameterised batches that legitimately
        # need >16 rounds (e.g. 24 specs each requiring a
        # submit_script + a few expand_result rounds + a
        # compose_results). Bounded to [1, 64]: lower than 1 makes no
        # sense, higher than 64 is past the point where the user
        # would rather see "stop and ask for guidance" than continue
        # autonomously. Invalid env values fall back to the default
        # rather than failing the turn — a typo in the env var
        # shouldn't block a researcher mid-analysis.
        try:
            MAX_TOOL_ROUNDS = int(
                os.environ.get("NORA_OPENAI_MAX_TOOL_ROUNDS", "16")
            )
            if not 1 <= MAX_TOOL_ROUNDS <= 64:
                MAX_TOOL_ROUNDS = 16
        except (TypeError, ValueError):
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
                    # Pin to True explicitly. The Responses API has
                    # defaulted this to True historically — and the
                    # docstring above relies on that — but a silent
                    # SDK/API default change would otherwise break
                    # Nora's tool-loop ergonomics without warning.
                    # Pinning surfaces the dependency at the request
                    # boundary and the lockdown test asserts it.
                    "parallel_tool_calls": True,
                    # Reasoning controls — the OpenAI analogue of the
                    # Anthropic provider's effort +
                    # thinking.display="summarized" pinning:
                    #   - effort: the researcher's per-session pick
                    #     from the picker's Effort section
                    #     (``catalog.EFFORT_LEVELS``; default
                    #     ``xhigh``). Valid on both catalog models —
                    #     gpt-5.6-sol and gpt-5.6-terra each accept the
                    #     full none/low/medium/high/xhigh/max range per
                    #     OpenAI's model pages. If a future catalog
                    #     entry re-pins a narrower range, it would 400
                    #     here — check the model page before adding.
                    #   - summary="auto": request reasoning summaries so
                    #     the thinking trace below has something to
                    #     surface. ("concise" is NOT supported by the
                    #     gpt-5 series; "auto" lets the server pick.)
                    "reasoning": {"effort": self.effort, "summary": "auto"},
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
                    # Context-window overrun: with ``truncation="disabled"``
                    # the API returns a 400 whose message names the
                    # token cap. Translate that to actionable guidance
                    # — the generic "OpenAI request failed: …" would
                    # otherwise leave the researcher staring at an
                    # opaque error for the most-actionable failure in
                    # the system. Match on tokens semantics rather
                    # than HTTP status so SDK changes that surface
                    # the same cause through a different exception
                    # type still translate.
                    if (
                        "context_length_exceeded" in lower
                        or "maximum context length" in lower
                        or ("input" in lower and "tokens" in lower
                            and ("limit" in lower or "exceed" in lower))
                    ):
                        yield TurnError(message=(
                            "Conversation hit the model's context "
                            "window. To continue: start a new "
                            "session, or reduce earlier turns from "
                            "this session by summarizing them via "
                            "``recall_conversation`` and starting "
                            "fresh from the summary. Underlying "
                            f"error: {msg}"
                        ))
                        return
                    # Response-id chain broken: the server retention
                    # window for ``previous_response_id`` has elapsed,
                    # or the response was deleted. Without the
                    # full-replay path that used to back this up,
                    # every subsequent turn would error with the same
                    # opaque message. Surface what the researcher can
                    # actually do — a chain reset is the only
                    # in-product recovery.
                    if (
                        turn_response_id is not None
                        and (
                            "previous_response_id" in lower
                            or "response not found" in lower
                            or ("response" in lower and "404" in lower)
                            or ("response" in lower and "expired" in lower)
                        )
                    ):
                        # Reset and let the next turn rebuild the
                        # chain from a fresh root. Drop ``self.``
                        # _last_response_id so the next .send()
                        # starts without ``previous_response_id``;
                        # the conversation history isn't lost
                        # (it's in chat_history.jsonl), it just
                        # won't be cached on the server side
                        # anymore.
                        #
                        # ``context_reset=True`` tells the runner to
                        # re-arm ``needs_context_prefix`` so the
                        # next turn re-injects the warm-start
                        # prefix. Without that flag, the next turn
                        # would send a fresh prompt with no
                        # ``previous_response_id`` AND no context
                        # prefix — the model would see this session
                        # as brand new and forget every prior turn
                        # until ``recall_conversation`` is invoked.
                        # The chain-expiry path is rare but recovery
                        # must be transparent: the researcher should
                        # see "continuing the conversation" not
                        # "model forgot everything".
                        self._last_response_id = None
                        yield TurnError(
                            message=(
                                "OpenAI's server-side response chain "
                                "has expired (the previous response is "
                                "no longer retrievable). The session's "
                                "chat history is preserved on disk; "
                                "the next turn will start a new chain "
                                "from a fresh root, and the model will "
                                "see this session's prior turns "
                                "through the warm-start context prefix "
                                "+ ``recall_conversation`` as needed. "
                                f"Underlying error: {msg}"
                            ),
                            context_reset=True,
                        )
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
                    from nora.provider.usage_log import append_usage_line
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
                    append_usage_line(self.cwd, line)

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
                    elif itype == "reasoning":
                        # Surface the model's reasoning SUMMARY (requested
                        # via reasoning.summary="auto" above) as a thinking
                        # trace, mirroring the Anthropic provider's
                        # ThinkingBlock -> AssistantThinking translation so
                        # the UI's thinking panel populates on both
                        # providers. We read ``summary`` (a list of
                        # ``{type:"summary_text", text:...}`` parts), NOT the
                        # raw ``content``/``encrypted_content`` — OpenAI's
                        # policy only sanctions the summary surface, and
                        # summaries aren't emitted every round, so the
                        # strip()-guard keeps empties out. The item itself is
                        # still carried forward server-side via the
                        # response-id chain; this is display-only.
                        summary = getattr(item, "summary", None) or []
                        trace = "".join(
                            getattr(part, "text", "") or ""
                            for part in summary
                            if getattr(part, "type", None) == "summary_text"
                        )
                        if trace.strip():
                            yield AssistantThinking(text=trace)
                    # Any other item types are held server-side and
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
                        # any consumer that reads the breakdown directly.
                        # ``input_tokens`` already represents the full
                        # prompt size on the OpenAI path; ``post_turn_tokens``
                        # adds output for the canonical post-turn snapshot.
                        # ``cost_usd`` is also None — Nora doesn't compute
                        # OpenAI costs locally.
                        post_turn_tokens=last_input_tokens + last_output_tokens,
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
                    args, parse_error = _parse_tool_args(args_json)
                    if parse_error is not None:
                        # Malformed args — surface as an explicit tool
                        # error so the model fixes its JSON instead of
                        # being told "missing required arg X" by the
                        # handler's schema layer (which is what
                        # happened when we silently coerced bad JSON
                        # to ``{}``). The handler is NOT invoked in
                        # this branch.
                        out_text = json.dumps({
                            "status": "error",
                            "reason": parse_error,
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
                        result = await handler(args)
                        out_text = _mcp_payload_to_text(result)
                        is_error = False
                    except Exception as e:  # noqa: BLE001
                        # Catch-all fallback: a per-tool handler raised
                        # an unexpected exception. The exception
                        # message itself may include parser excerpts,
                        # file paths, or raw data values the tool's
                        # curated error-handling would have redacted —
                        # interpolating ``str(e)`` into the model-
                        # visible reason bypasses the per-tool
                        # disclosure contract. Log the full details
                        # locally for debugging and return only the
                        # exception CLASS (a bounded identifier) plus
                        # a generic recovery hint.
                        import traceback
                        diag = (
                            f"[nora.openai] tool {name!r} handler "
                            f"raised {e.__class__.__name__}: {e}\n"
                            + traceback.format_exc()
                        )
                        print(diag, file=sys.stderr, flush=True)
                        try:
                            from nora.provider.usage_log import (
                                append_usage_line,
                            )
                            append_usage_line(self.cwd, diag.rstrip())
                        except Exception:  # noqa: BLE001 — logging must not block
                            pass
                        out_text = json.dumps({
                            "status": "error",
                            "reason": (
                                f"tool handler failed with "
                                f"{e.__class__.__name__}. Retry with "
                                f"different arguments or fall back to "
                                f"another tool."
                            ),
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
    """Best-effort decode used for the ``ToolCall`` audit event.

    Display-side only — the audit event renders whatever args the
    model thought it was sending, and a non-empty malformed string
    degrades to ``{}`` so the chat panel still shows the call. The
    actual handler dispatch goes through :func:`_parse_tool_args`,
    which surfaces malformed JSON as an explicit tool error so the
    model isn't told "missing required arg" when its real problem
    was bad JSON.
    """
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (ValueError, TypeError):
        return {}
    if not isinstance(out, dict):
        return {}
    return out


def _parse_tool_args(s: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode a ``function_call.arguments`` string for handler dispatch.

    Returns ``(args, None)`` on success and ``(None, reason)`` when
    the string is non-empty but not a valid JSON object. The empty
    string maps to ``({}, None)`` because OpenAI emits ``""`` for
    zero-arg calls, and that's a legitimate shape for the small
    handful of Nora tools that take no arguments.

    The previous behaviour silently coerced malformed JSON to ``{}``
    and dispatched the handler anyway. The handler's required-arg
    validator then complained about missing fields, which told the
    model the wrong story: it thought it had forgotten ``code`` /
    ``language`` when the real failure was that its JSON didn't
    parse. The model retried with the same broken serialiser and
    burned a turn. Returning a parse error here lets the caller
    emit an explicit "tool arguments were not valid JSON" result so
    the model fixes the actual problem on the next round.

    Non-dict top-level values (a bare list, string, or number from
    the model's perspective is "I sent some args" without a key, so
    we still call out the shape mismatch.
    """
    if not s:
        return {}, None
    try:
        out = json.loads(s)
    except (ValueError, TypeError) as e:
        # Truncate the offending payload so a model that emitted a
        # multi-MB malformed blob doesn't blow up the error-message
        # context. ``json.JSONDecodeError`` carries position info
        # that helps the model self-correct on the retry.
        #
        # Strip non-printables BEFORE the 120-char cap so the
        # truncation cap actually bounds output size. ``repr()``
        # on bidi overrides or control chars expands each
        # character to ``\\u202e`` / ``\\x07`` (4–6 chars), so a
        # raw 120-char snippet of pathological input can render as
        # 700+ chars after ``!r`` — defeating the cap whose entire
        # job is bounding error-message size. Replacing
        # non-printables with ``?`` before the slice keeps the cap
        # honest and still lets the model see the rough shape of
        # the JSON it tried to send.
        cleaned = "".join(
            c if (c.isprintable() or c in "\t ") else "?" for c in s
        )
        snippet = cleaned[:120] + ("…" if len(cleaned) > 120 else "")
        return None, (
            f"tool arguments were not valid JSON: {e}; received "
            f"{snippet!r}"
        )
    if not isinstance(out, dict):
        return None, (
            f"tool arguments must be a JSON object, got "
            f"{type(out).__name__}"
        )
    return out, None


def _payload_has_image_block(payload: Any) -> bool:
    """Whether an MCP-shaped tool result carries an inline image block.

    ``read_attached_file`` returns ``{"type": "image", "data": ...}``
    alongside a text descriptor when the user recalls a PNG / PDF /
    EPS. The Anthropic dispatcher forwards both blocks; the OpenAI
    Responses API's ``function_call_output`` only takes a single
    string, so the image bytes can't ride with the tool result. We
    detect the image-block case so the descriptor we DO send tells
    the model definitively that pixel data was dropped on this path.
    """
    if not isinstance(payload, dict):
        return False
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False


def _mcp_payload_to_text(payload: Any) -> str:
    """Nora handlers return MCP-shaped payloads:
    ``{"content": [{"type": "text", "text": "..."}]}``. Extract the
    JSON-text body for the OpenAI tool output.

    For payloads that carry an image content block alongside a text
    descriptor (``read_attached_file`` for PNG / PDF recall), rewrite
    the text descriptor so the model knows the pixels were dropped on
    this provider — without that, the descriptor's hedged "if your
    provider doesn't support images" hint is the only signal, and the
    model may still try to reason about the (absent) bytes. The
    Anthropic path is unaffected — its content list survives intact
    on its own dispatcher.
    """
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            text_block = None
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    text_block = block
                    break
            if text_block is not None:
                if _payload_has_image_block(payload):
                    return _rewrite_for_dropped_image(
                        str(text_block.get("text", "")),
                    )
                return str(text_block.get("text", ""))
        # Already-flat dict: serialise.
        return json.dumps(payload)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _rewrite_for_dropped_image(descriptor_text: str) -> str:
    """Replace ``read_attached_file``'s hedged image descriptor with an
    OpenAI-specific reason telling the model the pixels weren't sent.

    The original descriptor (see ``read_attached_file`` in
    ``tools.py``) reads "If your provider doesn't support image tool
    results, ask the researcher to re-@mention the file …". On this
    provider it definitely doesn't, so we promote that conditional
    note to the primary status and keep the file metadata so the
    model can name the file precisely in its follow-up message to the
    researcher.

    Falls back to the original text if the descriptor isn't the JSON
    shape we expect — never raises.
    """
    try:
        meta = json.loads(descriptor_text)
    except (ValueError, TypeError):
        return descriptor_text
    if not isinstance(meta, dict):
        return descriptor_text
    name = meta.get("name") or "the file"
    rewritten = {
        "status": "image_not_supported_on_provider",
        "name": meta.get("name"),
        "kind": meta.get("kind"),
        "ext": meta.get("ext"),
        "mime": meta.get("mime"),
        "size": meta.get("size"),
        "reason": (
            "Image tool results aren't supported on this provider. "
            f"Ask the researcher to re-@mention {name!r} in their "
            "next message — that routes the bytes through the user-"
            "message vision channel, which the model can see."
        ),
    }
    # Drop None fields so the response stays tight.
    rewritten = {k: v for k, v in rewritten.items() if v is not None}
    return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)


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

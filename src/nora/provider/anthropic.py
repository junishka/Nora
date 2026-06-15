"""Anthropic-backed ``ProviderSession`` implementation.

Wraps ``ClaudeSDKClient`` with the Nora-specific configuration
(disallowed-builtins list, tool-use catch-all, MCP-server registration,
``setting_sources=[]`` lockdown) and translates SDK message blocks
into the provider-neutral ``Event`` stream defined in
``provider/base.py``.

Auth detection: ``ANTHROPIC_API_KEY`` in env → ``api_key``;
``~/.claude.json`` with an OAuth account → ``subscription``;
otherwise ``unknown``. The keyring path (added later for OpenAI) does NOT
overwrite this — the Claude SDK is the only thing that reads
``ANTHROPIC_API_KEY``, so cred storage for Anthropic continues to flow
through the env / claude-CLI surface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

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
from nora.tools import (
    ALLOWED_TOOL_NAMES,
    SERVER_NAME,
    build_server,
    friendly_tool_names,
)


PROVIDER_ID = "anthropic"


# Sentinels the Claude SDK uses on AssistantMessage.error to signal
# auth / billing trouble.
_AUTH_FAILURE = "authentication_failed"
_BILLING_FAILURE = "billing_error"


# Per-turn style rider, appended to every user message right before
# the SDK forwards it to Claude. Sits adjacent to the generation
# cursor, where Anthropic's published prompting guidance says recent
# tokens carry the most weight. The same rules in the system prompt
# (line ~39 of system_prompt.py) are diluted across ~14k cached
# tokens of tool documentation, and Claude (notably Opus) ignores
# them in practice. GPT-5.5 honors the system-prompt-only version
# fine on the OpenAI path, so this rider is intentionally
# Anthropic-only.
#
# Cost accounting. Adds ~70 uncached tokens per user turn. The
# 14k-token cached prefix (system prompt + tool schemas) stays
# untouched, so the cache-discount path is preserved.
#
# Style: bracketed framing so the model parses this as instructions
# rather than continuation of the user's question. No em-dashes
# in the rider itself (lead by example).
#
# Opt out for A/B testing: set ``NORA_DISABLE_STYLE_RIDER=1``.
_STYLE_RIDER = (
    "\n\n[Reply formatting reminder. These rules bind the response "
    "you are about to produce. Hold them through every paragraph, "
    "not just the opening.\n"
    "1. No em-dashes (—) or en-dashes (–). Use periods, commas, or "
    "parentheses.\n"
    "2. Never use semicolons. Break the clause into two sentences "
    "with a period.\n"
    "3. Use colons only to introduce a list. For an explanation or "
    "apposition, start a new sentence.\n"
    "4. One idea per sentence. Split long compound sentences.\n"
    "5. Open with the analytic point. No preamble, no restating the "
    "question, no meta-commentary on what the table shows.\n"
    "6. Reader is an applied-stats colleague. Be concise.\n"
    # Rule 7 is the table-preference demonstration. On the OpenAI
    # path this is delivered as a structural few-shot turn in the
    # message history; the Claude Agent SDK doesn't expose a seam
    # to seed prior assistant / tool_result turns, so the same
    # signal rides here as an inline example. Embedded literal
    # block: a worked exchange demonstrates the desired shape more
    # reliably than an abstract rule (Opus, in particular, follows
    # demonstrated patterns more than stated ones).
    "7. When a tool result includes a `markdown` field, paste that "
    "table verbatim into your reply and add at most one short "
    "sentence of interpretation. Do not re-narrate the numbers in "
    "prose. Example exchange:\n"
    "   User: What's the breakdown of treatment in this sample?\n"
    "   You call submit_script; result returns markdown:\n"
    "   | treatment | n   | %    |\n"
    "   | --------- | --- | ---- |\n"
    "   | control   | 487 | 49.4 |\n"
    "   | treated   | 499 | 50.6 |\n"
    "   You reply with that exact table, then one sentence "
    "(Balanced 50/50 assignment, n=986). Nothing else.]"
)


def _wrap_with_style_rider(prompt: str) -> str:
    """Append the per-turn style rider to a user prompt.

    Returns ``prompt`` unchanged if the rider is disabled via env, if
    the prompt is empty (no message to attach to), or if the prompt
    already contains the rider (defensive: prevents accidental
    double-application by upstream callers that wrap the prompt
    themselves).
    """
    if not prompt:
        return prompt
    if os.environ.get("NORA_DISABLE_STYLE_RIDER") == "1":
        return prompt
    if "[Reply formatting reminder" in prompt:
        return prompt
    return prompt + _STYLE_RIDER


# Every SDK built-in we know of. ``can_use_tool`` catches anything the
# permission layer routes through it, but several Claude Code built-ins
# (ToolSearch, Skill, ScheduleWakeup, …) bypass that hook and must be
# blocked via ``disallowed_tools`` explicitly. Lesson learned during
# step-2 testing: pair the catch-all with an explicit name list.
_DISALLOWED_BUILTINS: tuple[str, ...] = (
    # Data-touching
    "Bash",
    "BashOutput",
    "KillBash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    # Agentic / orchestration
    "Task",
    "Agent",
    "Monitor",
    # Meta / UI / harness
    "ToolSearch",
    "Skill",
    "ScheduleWakeup",
    "TodoWrite",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "SlashCommand",
)


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------


def _has_claude_subscription() -> bool:
    """Return True iff ``~/.claude.json`` carries an OAuth account.

    Factored so :func:`detect_auth` and :func:`_ensure_anthropic_env`
    agree on the subscription test. Without sharing this helper the
    injection path could (and did) silently override an active Claude
    CLI subscription with a stale keyring API key: ``detect_auth``
    reported ``subscription``, but ``_ensure_anthropic_env`` skipped
    the subscription check and copied the keyring credential into
    ``ANTHROPIC_API_KEY``, which the SDK then preferred over OAuth.
    """
    claude_json = Path.home() / ".claude.json"
    if not claude_json.is_file():
        return False
    try:
        data = json.loads(claude_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    oauth = data.get("oauthAccount")
    return isinstance(oauth, dict) and bool(oauth.get("accountUuid"))


def detect_auth() -> str:
    """Return ``'subscription'``, ``'api_key'``, or ``'unknown'``.

    Resolution order:
    1. ``ANTHROPIC_API_KEY`` env var → ``api_key``.
    2. ``~/.claude.json`` with an OAuth account → ``subscription``
       (the Claude CLI's own subscription path).
    3. Keyring-stored credential under provider id ``anthropic`` →
       ``api_key``. The session's ``open()`` will copy this into env
       so the SDK picks it up.
    4. Otherwise → ``unknown``.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api_key"
    if _has_claude_subscription():
        return "subscription"
    # Last resort: keyring. Imported here rather than at module load
    # to avoid an import cycle (nora.auth doesn't depend on this
    # module, but keeping it lazy makes the dependency direction
    # obvious).
    from nora import auth as _auth
    if _auth.has_credential("anthropic"):
        return "api_key"
    return "unknown"


# Module-level flag tracking whether ``_ensure_anthropic_env`` was the
# one that put ``ANTHROPIC_API_KEY`` into the environment. We need to
# distinguish "the researcher exported it in their shell" (don't touch
# on credential delete) from "we copied it from the keyring" (DO clear
# on delete). Without this, deleting the keyring entry leaves the
# injected env var behind and ``detect_auth()`` keeps reporting
# ``api_key`` until the app restarts.
_ENV_INJECTED_BY_NORA: bool = False


def _ensure_anthropic_env() -> None:
    """Copy a keyring-stored Anthropic API key into ``ANTHROPIC_API_KEY``
    if (a) the env var isn't already set AND (b) no Claude CLI
    subscription is configured. The Claude Agent SDK reads the env
    var at client construction; this is the bridge between Nora's
    keyring storage and the SDK's expectations.

    Subscription takes priority: when ``~/.claude.json`` carries an
    OAuth account we deliberately do NOT inject a keyring API key,
    because the SDK prefers an explicit ``ANTHROPIC_API_KEY`` over
    the OAuth path. Injecting a stale keyring credential there would
    silently override the researcher's active subscription with a
    possibly-defunct API key — and ``detect_auth`` would still
    report ``subscription`` because it checks the .claude.json file
    BEFORE the keyring, leaving the page UI and the runtime out of
    sync.
    """
    global _ENV_INJECTED_BY_NORA
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if _has_claude_subscription():
        return
    from nora import auth as _auth
    cred = _auth.get_credential("anthropic")
    if cred:
        os.environ["ANTHROPIC_API_KEY"] = cred
        _ENV_INJECTED_BY_NORA = True


def clear_injected_env() -> None:
    """Reverse what ``_ensure_anthropic_env`` did, but ONLY if we were
    the ones who set the env var. Called by the bridge when the
    researcher deletes their Anthropic keyring credential — without
    this, the in-process env var keeps the SDK happy and
    ``detect_auth()`` keeps reporting ``api_key`` so the auth screen
    refuses to admit the credential was removed.

    No-op when the user's shell exported their own
    ``ANTHROPIC_API_KEY`` (we never touched it) or when nothing was
    injected.
    """
    global _ENV_INJECTED_BY_NORA
    if _ENV_INJECTED_BY_NORA:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        _ENV_INJECTED_BY_NORA = False


# ---------------------------------------------------------------------------
# Tool-use permission gate
# ---------------------------------------------------------------------------


async def _gate_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    ctx: Any,
) -> PermissionResultAllow | PermissionResultDeny:
    """Catch-all permission hook.

    Allow only the Nora MCP tools by name. Anything else (a future
    SDK built-in, an alias, a sub-tool the disallowed list misses)
    gets a denial that names the legitimate alternatives so the model
    can recover gracefully. The list is derived from
    ``ALLOWED_TOOL_NAMES`` rather than hardcoded — earlier copies of
    this message drifted (six names vs. thirteen), which silently
    hid new tools like ``list_session_files`` /
    ``search_in_session_files`` from the model's recovery path.
    """
    del tool_input, ctx  # signature-required, unused
    if tool_name in ALLOWED_TOOL_NAMES:
        return PermissionResultAllow()
    available = ", ".join(friendly_tool_names(prefixed=False))
    return PermissionResultDeny(
        behavior="deny",
        message=(
            f"Tool '{tool_name}' is not available in Nora. Use one of the "
            f"{len(ALLOWED_TOOL_NAMES)} custom tools described in the "
            f"system prompt (all prefixed mcp__{SERVER_NAME}__): "
            f"{available}. Nora does not expose Bash, Read, Write, "
            f"Edit, Glob, Grep, or any other general tool."
        ),
        interrupt=False,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class AnthropicSession:
    """``ProviderSession`` backed by ``ClaudeSDKClient``.

    A session is bound to (cwd, model, system_prompt) at construction.
    To switch cwd, build a new session. To switch model, call
    ``set_model`` — the SDK supports in-place model swap so the
    conversation isn't reset.

    ``open()``/``close()`` are idempotent. ``send()`` opens the client
    lazily on first call so a never-used session pays no SDK cost.
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
        # Anthropic SDK can resume the CLI's own session store keyed by
        # cwd. Nora doesn't use that path — the bridge prepends its own
        # condensed history on the first turn after open — but keep
        # the parameter so a future caller could opt in.
        self._continue = continue_conversation
        self._client: ClaudeSDKClient | None = None

    # ---- lifecycle -------------------------------------------------------

    def _build_options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            model=self.model,
            # Reasoning controls. Left unset, the SDK inherits each
            # model's defaults: adaptive thinking on 4.6+ models, but a
            # "high" effort ceiling and — on Opus 4.7+ — an "omitted"
            # thinking display that strips the reasoning text down to
            # signatures (so Nora's AssistantThinking trace goes blank).
            # We pin both explicitly:
            #   - effort="xhigh": extended reasoning depth on models that
            #     support it (Opus 4.7+); falls back to "high" elsewhere
            #     (e.g. the default Sonnet 4.6), so it's safe across the
            #     whole catalog.
            #   - thinking adaptive + display="summarized": keep the model
            #     deciding how much to think, but ask for human-readable
            #     summarized traces so the thinking panel stays populated
            #     on Opus 4.8 / Fable 5, not just pre-4.7 Sonnet.
            effort="xhigh",
            thinking={"type": "adaptive", "display": "summarized"},
            continue_conversation=self._continue,
            mcp_servers={SERVER_NAME: build_server()},
            allowed_tools=list(ALLOWED_TOOL_NAMES),
            disallowed_tools=list(_DISALLOWED_BUILTINS),
            can_use_tool=_gate_tool_use,
            # `default` permission_mode routes anything outside
            # allowed/disallowed through ``can_use_tool``, which is
            # the deny-by-default catch-all we want.
            permission_mode="default",
            # Don't load CLAUDE.md / settings / project-local config.
            # Those can introduce hooks and tools we don't control.
            setting_sources=[],
            # Extend the prompt-cache TTL from the 5-minute default to
            # 1 hour. The Claude CLI auto-places a cache breakpoint at
            # the end of the tools section, covering Nora's ~14k-token
            # cached prefix (system prompt + tool schemas). With the
            # 5-minute TTL, any researcher idle gap longer than 5
            # minutes forces a full rewrite at the +25% surcharge; with
            # 1h TTL, the rewrite is deferred 12x longer at a one-time
            # write surcharge of +75% over 5min (still way under the
            # cost of repeated rewrites in a long, intermittent
            # research session). The CLI checks the env var
            # ``ENABLE_PROMPT_CACHING_1H``; ``ClaudeAgentOptions.env``
            # is forwarded to the CLI subprocess.
            env={"ENABLE_PROMPT_CACHING_1H": "1"},
        )

    async def open(self) -> None:
        if self._client is not None:
            return
        # Bridge keyring → env so the SDK sees an API key when the
        # researcher has stored one (and isn't using the Claude CLI
        # subscription path). No-op when env is already set or when
        # subscription auth is in effect.
        _ensure_anthropic_env()
        opts = self._build_options()
        self._client = await ClaudeSDKClient(options=opts).__aenter__()

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 — close-time errors aren't useful
                pass

    # ---- model swap ------------------------------------------------------

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch to a different Anthropic model without resetting the
        conversation. Returns ``{"ok": ..., ...}``.

        If the SDK rejects the new id (typo, model not in the
        researcher's plan, …) the previous client is torn down and the
        next ``send()`` will reopen with the new id, but the prior
        in-context conversation is lost — surface that to the caller
        so the UI can warn the researcher.
        """
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
                    f"{info.provider!r}, not Anthropic"
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
        # Client not yet open: just update the desired id; next send()
        # opens with it.
        if self._client is None:
            self.model = model_id
            return {
                "ok": True,
                "model": model_id,
                "label": info.label,
                "context_window": info.context_window,
            }
        try:
            await self._client.set_model(model_id)
            self.model = model_id
        except Exception as e:  # noqa: BLE001 — SDK shape varies
            await self.close()
            self.model = model_id
            return {
                "ok": False,
                "reason": (
                    f"set_model failed: {e}. Conversation reset; "
                    f"the new model will take effect on the next message."
                ),
            }
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
        events terminated by ``TurnDone`` or ``TurnError`` /
        ``AuthFailure``.

        ``images`` is an optional list of ``{"data": <base64>, "mime":
        ...}`` dicts attached as image content blocks.
        """
        await self.open()
        client = self._client
        assert client is not None

        # Wrap the user's prompt with the per-turn style rider before
        # forwarding to the SDK. This puts the formatting rules
        # adjacent to the generation cursor, where Claude weights
        # them most. See the ``_STYLE_RIDER`` block for rationale.
        wrapped_prompt = _wrap_with_style_rider(prompt)

        try:
            if images:
                await client.query(_image_message_iter(wrapped_prompt, images))
            else:
                await client.query(wrapped_prompt)
        except Exception as e:  # noqa: BLE001 — SDK may raise various things
            yield TurnError(message=f"failed to send prompt: {e}")
            return

        # Track the LAST observed ResultMessage usage, not the peak
        # across rounds. The conversation chain grows monotonically as
        # tool outputs join it, so the LAST round's prompt_total is
        # the actual context size at the end of this turn — exactly
        # what the "context occupied" chip wants to display.
        #
        # Picking MAX (the previous behavior) inflated the chip when
        # an intermediate round happened to report an unusually high
        # prompt_total — e.g., a transient retry or an SDK accounting
        # quirk where a tool result is double-counted before being
        # folded into the cache. The peak then stuck around forever
        # via the chip's high-water clamp, leaving the chip well above
        # the actual chain size. Trusting the LAST measurement matches
        # OpenAI's accounting (which uses the last round's
        # input_tokens) and gives a directly comparable number.
        last_input = 0
        last_output = 0
        last_cache_read = 0
        last_cache_creation = 0
        last_cost: float | None = None
        saw_result = False

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                err = getattr(msg, "error", None)
                if err == _AUTH_FAILURE:
                    yield AuthFailure(reason="auth failure from server")
                    return
                if err == _BILLING_FAILURE:
                    yield AuthFailure(reason="billing failure — check your account")
                    return
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        if block.text.strip():
                            yield AssistantText(text=block.text)
                    elif isinstance(block, ThinkingBlock):
                        if block.thinking.strip():
                            yield AssistantThinking(text=block.thinking)
                    elif isinstance(block, ToolUseBlock):
                        yield ToolCall(
                            name=block.name,
                            input=dict(block.input or {}),
                            call_id=block.id,
                        )
                    elif isinstance(block, ToolResultBlock):
                        yield _tool_result_event(block)
            elif isinstance(msg, UserMessage):
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            yield _tool_result_event(block)
            elif isinstance(msg, ResultMessage):
                saw_result = True
                usage = msg.usage or {}
                inp = _maybe_int(usage.get("input_tokens")) or 0
                outp = _maybe_int(usage.get("output_tokens")) or 0
                cr = _maybe_int(usage.get("cache_read_input_tokens")) or 0
                cc = _maybe_int(usage.get("cache_creation_input_tokens")) or 0
                prompt_total = inp + cr + cc
                # Diagnostic: gated by NORA_DEBUG_USAGE. Writes to
                # stderr (visible if launched from a terminal) AND
                # appends to ``<cwd>/.nora-usage.log`` (always reachable
                # by the researcher regardless of launch method —
                # pywebview swallows stderr on a double-clicked app).
                # We dump the raw usage dict so we can see whether
                # input_tokens already includes cached portions, whether
                # there are new ephemeral-cache fields we're missing,
                # and which interpretation the 1M model uses.
                if os.environ.get("NORA_DEBUG_USAGE") == "1":
                    import sys as _sys
                    from nora.provider.usage_log import append_usage_line
                    line = (
                        f"[nora.usage] round usage={dict(usage)} "
                        f"computed prompt_total={prompt_total} "
                        f"(inp={inp}, cr={cr}, cc={cc}, out={outp})"
                    )
                    print(line, file=_sys.stderr, flush=True)
                    append_usage_line(self.cwd, line)
                last_input = inp
                last_output = outp
                last_cache_read = cr
                last_cache_creation = cc
                if msg.total_cost_usd is not None:
                    last_cost = msg.total_cost_usd

        if saw_result:
            # Canonical "context occupied after this turn." Anthropic's
            # input_tokens / cache_read / cache_creation cover the
            # whole prompt-side window via the prompt cache; output
            # joins it for the post-turn snapshot the chip wants.
            post_turn = (
                (last_input or 0)
                + (last_cache_read or 0)
                + (last_cache_creation or 0)
                + (last_output or 0)
            )
            yield TurnDone(
                input_tokens=last_input,
                output_tokens=last_output,
                cache_read_input_tokens=last_cache_read,
                cache_creation_input_tokens=last_cache_creation,
                cost_usd=last_cost,
                post_turn_tokens=post_turn,
            )

    # ---- async-context-manager sugar ------------------------------------

    async def __aenter__(self) -> "AnthropicSession":
        await self.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tool_result_event(block: ToolResultBlock) -> ToolCallResult:
    text = _extract_tool_result_text(block.content)
    run_dir, language = _extract_hints(text)
    return ToolCallResult(
        call_id=block.tool_use_id,
        text=text,
        is_error=bool(block.is_error),
        run_dir=run_dir,
        language=language,
    )


def _extract_tool_result_text(
    content: str | list[dict[str, Any]] | None,
) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(parts)


def _extract_hints(text: str) -> tuple[str | None, str | None]:
    """Peel ``_run_dir`` and ``_language`` hints from a tool-result
    payload. Used by the UI to render raw R/Stata output and pick the
    "Open in …" button. Returns ``(None, None)`` for non-script results.
    """
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


async def _image_message_iter(
    text: str, images: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    """Yield one structured user message carrying text + image blocks
    in the Anthropic API shape so the SDK forwards them to vision."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("mime", "image/png"),
                "data": img.get("data", ""),
            },
        })
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def _maybe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

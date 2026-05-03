"""Anthropic-backed ``ProviderSession`` implementation.

Wraps ``ClaudeSDKClient`` with the Nora-specific configuration
(disallowed-builtins list, tool-use catch-all, MCP-server registration,
``setting_sources=[]`` lockdown) and translates SDK message blocks
into the provider-neutral ``Event`` stream defined in
``provider/base.py``.

Auth detection follows the original ``app.py`` rules: ``ANTHROPIC_API_KEY``
in env → ``api_key``; ``~/.claude.json`` with an OAuth account → ``subscription``;
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
from nora.tools import ALLOWED_TOOL_NAMES, SERVER_NAME, build_server


PROVIDER_ID = "anthropic"


# Sentinels the Claude SDK uses on AssistantMessage.error to signal
# auth / billing trouble. Kept verbatim from the original app.py.
_AUTH_FAILURE = "authentication_failed"
_BILLING_FAILURE = "billing_error"


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
    claude_json = Path.home() / ".claude.json"
    if claude_json.is_file():
        try:
            data = json.loads(claude_json.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        else:
            oauth = data.get("oauthAccount")
            if isinstance(oauth, dict) and oauth.get("accountUuid"):
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
    if the env var isn't already set. The Claude Agent SDK reads the
    env var at client construction; this is the bridge between
    Nora's keyring storage and the SDK's expectations.

    Subscription auth wins implicitly: when the env var is unset and
    ``~/.claude.json`` carries an OAuth account, the SDK uses that
    path and never consults the env var, so this function's no-op
    branch is correct.
    """
    global _ENV_INJECTED_BY_NORA
    if os.environ.get("ANTHROPIC_API_KEY"):
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

    Allow only the six Nora MCP tools by name. Anything else (a future
    SDK built-in, an alias, a sub-tool the disallowed list misses)
    gets a denial that names the legitimate alternatives so the model
    can recover gracefully.
    """
    del tool_input, ctx  # signature-required, unused
    if tool_name in ALLOWED_TOOL_NAMES:
        return PermissionResultAllow()
    return PermissionResultDeny(
        behavior="deny",
        message=(
            f"Tool '{tool_name}' is not available in Nora. Use one of the "
            f"six custom tools described in the system prompt "
            f"(mcp__{SERVER_NAME}__get_schema, request_data, submit_script, "
            f"expand_result, list_results, recall_conversation). Nora "
            f"does not expose Bash, Read, Write, Edit, Glob, Grep, or any "
            f"other general tool."
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

        try:
            if images:
                await client.query(_image_message_iter(prompt, images))
            else:
                await client.query(prompt)
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
                    line = (
                        f"[nora.usage] round usage={dict(usage)} "
                        f"computed prompt_total={prompt_total} "
                        f"(inp={inp}, cr={cr}, cc={cc}, out={outp})"
                    )
                    print(line, file=_sys.stderr, flush=True)
                    try:
                        with (self.cwd / ".nora-usage.log").open("a") as _f:
                            _f.write(line + "\n")
                    except Exception:  # noqa: BLE001 — diagnostic must never crash a turn
                        pass
                last_input = inp
                last_output = outp
                last_cache_read = cr
                last_cache_creation = cc
                if msg.total_cost_usd is not None:
                    last_cost = msg.total_cost_usd

        if saw_result:
            yield TurnDone(
                input_tokens=last_input,
                output_tokens=last_output,
                cache_read_input_tokens=last_cache_read,
                cache_creation_input_tokens=last_cache_creation,
                cost_usd=last_cost,
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

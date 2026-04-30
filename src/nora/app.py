"""Nora — step-2 app: frontier chat with the locked-down MCP interface.

What changed from step 1 (the spine):
- The Claude Agent SDK's built-in tools (Bash, Read, Write, Edit, Glob, Grep,
  WebFetch, WebSearch, etc.) are disallowed. Claude reaches the local machine
  only through the six custom tools defined in `nora.tools`.
- `can_use_tool` is a catch-all deny: any tool not on the explicit allowlist
  is rejected, including tools added by future SDK versions we haven't heard
  of yet. This is belt-and-suspenders on top of `disallowed_tools`.
- The system prompt replaces Claude Code's default with a researcher-oriented
  one that introduces the six tools and the constraints.
- The renderer now shows tool calls and tool results inline so the researcher
  can see what Claude is doing.

What has NOT changed yet (stays mocked until later steps):
- Tools return mocked structured payloads (step 2 scope).
- No real schema extraction (step 3).
- No real executor, no real sanitizer (steps 4–5).
- No local runtime library for R / Stata yet.

Auth is still inherited from the `claude` CLI state — subscription OAuth via
~/.claude.json, or `ANTHROPIC_API_KEY` env var.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, NoReturn

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from nora.config import get_cwd, set_cwd
from nora.env_detect import Environment, detect_environment
from nora.policy import (
    DEFAULT_MAX_DEPTH,
    POLICY_FILE,
    VALID_DEPTHS,
    NoraPolicy,
    DatasetPolicy,
    get_max_depth,
    has_explicit_policy,
    load_policy,
    save_policy,
)
from nora.provider import detect_auth as _provider_detect_auth
from nora.provider.catalog import (
    ANTHROPIC_MODELS,
    PROVIDER_DEFAULTS,
)
from nora.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE as _SYSTEM_PROMPT_TEMPLATE_NEW,  # noqa: F401
    build_system_prompt as _build_system_prompt,
    dataset_listing as _dataset_listing_new,
    scan_datasets as _scan_datasets_new,
)
from nora.tools import ALLOWED_TOOL_NAMES, SERVER_NAME, build_server


console = Console()

_EXIT_WORDS = {"exit", "quit", ":q"}
_AUTH_FAILURE = "authentication_failed"
_BILLING_FAILURE = "billing_error"

AuthMode = Literal["subscription", "api_key", "unknown"]

# Every SDK built-in we know of. `can_use_tool` is the catch-all for tools
# the SDK routes through its permission system; however, some Claude Code
# built-ins (ToolSearch, Skill, ScheduleWakeup, ...) BYPASS `can_use_tool`
# and must be blocked via `disallowed_tools` explicitly. Lesson learned
# during step-2 testing: assume the can_use_tool gate is incomplete and
# always pair it with a thorough disallow list.
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

# System prompt template lives in nora.system_prompt now so both providers
# can render the same text. Local alias kept for back-compat with the
# terminal entry point's _build_options() call site.
_SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_TEMPLATE_NEW


def _detect_auth_mode() -> AuthMode:
    """Anthropic auth detection. Delegates to provider/anthropic so a
    single source of truth controls how we read ``ANTHROPIC_API_KEY``
    and the Claude CLI's ``~/.claude.json`` subscription token."""
    return _provider_detect_auth("anthropic")


def _auth_mode_line(mode: AuthMode) -> str:
    if mode == "subscription":
        return "[green]signed in with Claude subscription · usage covered by your plan[/green]"
    if mode == "api_key":
        return "[yellow]using ANTHROPIC_API_KEY · you're billed per token[/yellow]"
    return "[red]auth mode unknown · run `claude` in a terminal to sign in[/red]"


def _runtimes_line(env: Environment) -> str:
    parts: list[str] = []
    if env.r:
        ver = (env.r.version or "").replace("Rscript (R) version ", "R ")
        parts.append(f"[green]{ver or 'R'}[/green]")
    else:
        parts.append("[red]R: not installed[/red]")
    if env.stata:
        # The stata-mp version string isn't cheap to probe so we just
        # show its presence; the binary path hints at edition (MP / SE).
        edition = (
            "Stata MP" if "mp" in env.stata.binary.lower()
            else "Stata SE" if "se" in env.stata.binary.lower()
            else "Stata"
        )
        parts.append(f"[green]{edition}[/green]")
    else:
        parts.append("[red]Stata: not installed[/red]")
    if env.python:
        # Show Python alongside R / Stata. Yellow when the binary is
        # there but the runtime can't actually load — gives the
        # researcher an obvious "needs ``pip install pandas``" cue
        # before they wait for the first script to fail.
        ver = env.python.version or "Python"
        hard_missing = {"pandas", "numpy"} & set(env.python.missing_packages)
        if hard_missing:
            parts.append(
                f"[yellow]{ver} (needs: "
                f"{', '.join(sorted(hard_missing))})[/yellow]"
            )
        else:
            parts.append(f"[green]{ver}[/green]")
    else:
        parts.append("[red]Python: not installed[/red]")
    sbx = "[green]sandbox on[/green]" if env.sandbox_exec else "[yellow]no sandbox[/yellow]"
    parts.append(sbx)
    return " · ".join(parts)


def _print_banner(mode: AuthMode, cwd: Path, env: Environment) -> None:
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold]Nora[/bold] [dim]· local analysis assistant · "
                "step 4, real executor[/dim]\n"
                f"{_auth_mode_line(mode)}\n"
                f"{_runtimes_line(env)}\n"
                f"[dim]working dir:[/dim] [cyan]{cwd}[/cyan]\n"
                "[dim]Type a message. 'exit' / Ctrl-D to quit.[/dim]"
            ),
            border_style="cyan",
        )
    )


# One-line descriptions of each depth tier, shown in the /policy wizard
# so researchers don't have to remember what each one means. The
# current default (policy.DEFAULT_MAX_DEPTH) is surfaced separately
# by the wizard so it doesn't have to live in this description map.
_DEPTH_DESCRIPTIONS: dict[str, str] = {
    "names_only": "variable names only",
    "names_types": "+ type per variable",
    "names_types_labels": "+ variable labels and value labels",
    "names_types_labels_summary": "+ NA counts and distinct-value counts",
}


def _run_policy_wizard(cwd: Path) -> None:
    """Interactive TUI to view and edit the schema-depth policy.

    Researchers invoke this by typing `/policy` in the chat. The
    wizard:
      1. Lists datasets in cwd with their current ceiling.
      2. Lets the researcher pick one.
      3. Lets them pick a new depth (or leave as-is).
      4. Writes the update to ``<cwd>/.nora/policy.json``.

    Nothing here touches Claude. The chat message is intercepted in
    the loop before it would otherwise be sent to the model.
    """
    datasets = _scan_datasets(cwd)
    if not datasets:
        console.print(
            Text.from_markup(
                "[yellow]No datasets (.csv / .tsv / .dta / .rds / "
                ".parquet / .jsonl) found in "
                f"[cyan]{cwd}[/cyan]. Add some and try again.[/yellow]"
            )
        )
        return

    while True:
        policy = load_policy(cwd)
        console.print()
        console.print(
            Panel.fit(
                _render_policy_table(datasets, policy),
                border_style="cyan",
                title="[bold]Schema policy[/bold]",
                title_align="left",
            )
        )
        prompt_text = (
            f"[green]pick a dataset by number (1-{len(datasets)}), "
            f"or type [cyan]q[/cyan] to return to chat[/green]"
        )
        try:
            raw = Prompt.ask(prompt_text, default="q")
        except (EOFError, KeyboardInterrupt):
            return
        raw = raw.strip()
        if raw.lower() in ("q", "quit", "exit", "done", ""):
            return
        try:
            idx = int(raw)
        except ValueError:
            console.print(Text(f"  '{raw}' isn't a number or 'q'. Try again.", style="red"))
            continue
        if not 1 <= idx <= len(datasets):
            console.print(Text("  out of range.", style="red"))
            continue
        chosen = datasets[idx - 1]
        _edit_dataset_policy(cwd, chosen, policy)


def _render_policy_table(datasets: list[Path], policy: NoraPolicy) -> Any:
    """Format the dataset×ceiling table as Rich markup."""
    lines: list[str] = []
    width = max(len(d.name) for d in datasets) + 2
    for i, ds in enumerate(datasets, start=1):
        ceiling = get_max_depth(policy, ds.name)
        source = (
            "[green]explicit[/green]"
            if has_explicit_policy(policy, ds.name)
            else "[yellow]default[/yellow]"
        )
        lines.append(
            f"  [dim]{i:2d}.[/dim]  "
            f"[cyan]{ds.name:<{width}}[/cyan]  "
            f"{ceiling:<30}  {source}"
        )
    lines.append("")
    lines.append(f"[dim]Default ceiling: [/dim]{policy.default_max_depth}")
    return Text.from_markup("\n".join(lines))


def _edit_dataset_policy(
    cwd: Path, dataset: Path, policy: NoraPolicy
) -> None:
    """Prompt the researcher for a new ceiling for one dataset, save it."""
    current = get_max_depth(policy, dataset.name)
    console.print()
    console.print(
        Text.from_markup(
            f"[bold]{dataset.name}[/bold]  [dim]·[/dim]  "
            f"current ceiling: [cyan]{current}[/cyan]"
        )
    )
    console.print()
    for i, depth in enumerate(VALID_DEPTHS, start=1):
        marker = " [dim](current)[/dim]" if depth == current else ""
        console.print(
            Text.from_markup(
                f"  [dim]{i}.[/dim]  [cyan]{depth:<30}[/cyan]  "
                f"[dim]{_DEPTH_DESCRIPTIONS.get(depth, '')}[/dim]{marker}"
            )
        )
    console.print()
    try:
        raw = Prompt.ask(
            "[green]pick a depth by number, or [cyan]Enter[/cyan] to keep current[/green]",
            default="",
        )
    except (EOFError, KeyboardInterrupt):
        return
    raw = raw.strip()
    if not raw:
        return
    try:
        idx = int(raw)
    except ValueError:
        console.print(Text(f"  '{raw}' isn't a number. Keeping current.", style="red"))
        return
    if not 1 <= idx <= len(VALID_DEPTHS):
        console.print(Text("  out of range. Keeping current.", style="red"))
        return
    new_depth = VALID_DEPTHS[idx - 1]
    if new_depth == current:
        return

    from datetime import datetime, timezone
    updated = NoraPolicy(
        version=policy.version,
        default_max_depth=policy.default_max_depth,
        datasets={
            **policy.datasets,
            dataset.name: DatasetPolicy(
                max_depth=new_depth,
                set_at=datetime.now(timezone.utc).isoformat(),
            ),
        },
    )
    try:
        save_policy(cwd, updated)
    except OSError as e:
        console.print(
            Text.from_markup(
                f"  [red]failed to write policy file: {e}[/red]"
            )
        )
        return
    console.print(
        Text.from_markup(
            f"  [green]✓[/green]  {dataset.name} → {new_depth}"
        )
    )


def _handle_slash_command(user_text: str, cwd: Path) -> bool:
    """Dispatch slash-commands typed in the chat. Returns True if the
    command was recognized and handled (so the chat loop should skip
    sending it to Claude), False if it should fall through to Claude
    as a normal message (e.g. the researcher typed '/' as part of a
    path or was quoting something).
    """
    parts = user_text[1:].split()
    if not parts:
        return False
    cmd = parts[0].lower()
    if cmd in ("policy", "policies"):
        _run_policy_wizard(cwd)
        return True
    if cmd in ("help", "?"):
        _print_slash_help()
        return True
    # Unknown slash-prefixed thing — warn but don't forward (could be a
    # typo of a known command).
    console.print(
        Text.from_markup(
            f"[yellow]unknown command[/yellow] [cyan]/{cmd}[/cyan]. "
            f"[dim]/help for the list. To send this to Claude as a "
            f"message, drop the leading '/'.[/dim]"
        )
    )
    return True


def _print_slash_help() -> None:
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold]Local commands[/bold] [dim](these never go to Claude)[/dim]\n\n"
                "  [cyan]/policy[/cyan]   view or change the schema-depth "
                "ceiling per dataset\n"
                "  [cyan]/help[/cyan]     this message\n"
                "  [cyan]exit[/cyan]      quit Nora"
            ),
            border_style="cyan",
        )
    )


# Dataset enumeration moved to nora.system_prompt so both providers can
# render the same listing without dragging in app.py's terminal scaffolding.
# Local re-exports kept so existing imports (``from nora.app import
# _scan_datasets``) — most notably ui.py at lines that predate the
# refactor — keep working without churn.
_dataset_listing = _dataset_listing_new
_scan_datasets = _scan_datasets_new


def _print_schema_policy(cwd: Path) -> None:
    """Show the researcher which schema-depth ceiling applies to each
    dataset in cwd. Silent if no datasets are present.

    This is a non-interactive notification — the researcher edits
    ``<cwd>/.nora/policy.json`` by hand to change what Claude can
    see. A proper wizard UX is a follow-on.
    """
    datasets = _scan_datasets(cwd)
    if not datasets:
        return

    policy = load_policy(cwd)
    lines: list[str] = []
    any_default = False
    for ds in datasets:
        ceiling = get_max_depth(policy, ds.name)
        if has_explicit_policy(policy, ds.name):
            source = "[green]explicit[/green]"
        else:
            source = "[yellow]default[/yellow]"
            any_default = True
        lines.append(f"  [cyan]{ds.name}[/cyan]  [dim]·[/dim]  {ceiling}  [dim]({source})[/dim]")

    body_parts = [
        "[bold]Schema policy[/bold] [dim](what Claude can see about each dataset)[/dim]",
        *lines,
    ]
    if any_default:
        body_parts.append(
            f"[dim]Edit {POLICY_FILE} to raise any dataset's ceiling. "
            f"Tiers: names_only < names_types < names_types_labels < "
            f"names_types_labels_summary.[/dim]"
        )
    console.print(
        Panel.fit(
            Text.from_markup("\n".join(body_parts)),
            border_style="yellow" if any_default else "green",
        )
    )


def _print_auth_hint() -> None:
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold red]Not signed in.[/bold red]\n\n"
                "Nora uses your Claude account. Pick one:\n\n"
                "  [bold]1.[/bold] Sign in with your Claude subscription "
                "(Pro / Max / Team):\n"
                "     Open a new Terminal and run: [cyan]claude[/cyan]\n"
                "     Follow the browser prompt, then come back.\n\n"
                "  [bold]2.[/bold] Use an Anthropic API key:\n"
                "     [cyan]export ANTHROPIC_API_KEY=sk-ant-…[/cyan]\n"
                "     Then restart nora."
            ),
            border_style="red",
            title="Authentication required",
        )
    )


# ---------------------------------------------------------------------------
# Tool-call / tool-result rendering
# ---------------------------------------------------------------------------

def _render_tool_use(block: ToolUseBlock) -> None:
    """Render Claude calling one of our tools, inline in the chat stream."""
    # Strip the mcp__nora__ prefix for readability.
    name = block.name
    short = name.split("__")[-1] if name.startswith("mcp__") else name
    inp = block.input or {}
    if short == "submit_script":
        lang = inp.get("language", "?")
        label = inp.get("label", "")
        code = inp.get("code", "")
        header = f"⚙ submit_script  [{lang}]" + (f"  {label}" if label else "")
        console.print(Text(header, style="bold cyan"))
        if code:
            # Three-way dispatch — earlier two-way "r else stata"
            # rendered Python with the Stata lexer, mangling syntax
            # highlighting on every Python script the model ran.
            lang_lower = lang.lower()
            if lang_lower == "r":
                lexer = "r"
            elif lang_lower == "python":
                lexer = "python"
            else:
                lexer = "stata"
            console.print(Syntax(code, lexer, theme="ansi_dark", line_numbers=False))
    elif short == "submit_script_file":
        lang = inp.get("language", "")
        label = inp.get("label", "")
        fname = inp.get("name", "?")
        suffix = f"  [{lang}]" if lang else ""
        header = f"⚙ submit_script_file  running {fname}{suffix}" + (
            f"  {label}" if label else ""
        )
        console.print(Text(header, style="bold cyan"))
    elif short == "get_schema":
        summary = f"dataset={inp.get('dataset', '?')!r} depth={inp.get('depth', '?')!r}"
        console.print(Text(f"⚙ get_schema  {summary}", style="cyan"))
    elif short == "search_schema":
        summary = (
            f"{inp.get('query', '?')!r} in "
            f"dataset={inp.get('dataset', '?')!r}"
        )
        console.print(Text(f"⚙ search_schema  {summary}", style="cyan"))
    elif short == "request_data":
        summary = (
            f"{inp.get('request_type', '?')} on {inp.get('variable', '?')!r} "
            f"(dataset={inp.get('dataset', '?')!r})"
        )
        console.print(Text(f"⚙ request_data  {summary}", style="cyan"))
    elif short == "expand_result":
        view = inp.get("view", "")
        view_part = f"  view={view!r}" if view else ""
        console.print(Text(
            f"⚙ expand_result  {inp.get('result_id', '?')!r}{view_part}",
            style="cyan",
        ))
    elif short == "list_results":
        limit = inp.get("limit")
        suffix = f"  limit={limit}" if limit else ""
        console.print(Text(f"⚙ list_results{suffix}", style="cyan"))
    elif short == "list_results_global":
        query = inp.get("query", "")
        suffix = f"  query={query!r}" if query else ""
        console.print(Text(f"⚙ list_results_global{suffix}", style="cyan"))
    elif short == "recall_conversation":
        bits = []
        if inp.get("query"):
            bits.append(f"query={inp.get('query')!r}")
        if inp.get("tail"):
            bits.append(f"tail={inp.get('tail')}")
        suffix = "  " + ", ".join(bits) if bits else ""
        console.print(Text(f"⚙ recall_conversation{suffix}", style="cyan"))
    elif short == "read_attached_file":
        console.print(Text(
            f"⚙ read_attached_file  {inp.get('name', '?')!r}",
            style="cyan",
        ))
    else:
        # Should not happen given the allowlist — render loudly if it does.
        console.print(Text(f"⚙ {name}  {inp!r}  [UNEXPECTED]", style="red bold"))


def _render_tool_result(block: ToolResultBlock) -> None:
    """Render the tool's response so the researcher sees what went back to Claude.

    For ``submit_script`` responses, also surface the raw R/Stata
    stdout (and non-empty stderr) from the run directory. Claude
    never sees this text — it's strictly for the researcher — so
    the conventional "Claude saw this, you see both" split is
    visible here.
    """
    text = _extract_tool_result_text(block.content)
    border = "red" if block.is_error else "green"
    if not text.strip():
        console.print(Text("  (empty tool result)", style="dim"))
        return

    # If the payload is JSON (which our tools always emit), render
    # it as JSON. Also peel off `_run_dir` so we can display the raw
    # R/Stata log the researcher actually wants to see.
    run_dir: Path | None = None
    rendered: Any
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "_run_dir" in parsed:
            run_dir = Path(parsed["_run_dir"])
            # Strip from the rendered JSON so the "what Claude saw"
            # panel doesn't clutter with the path marker.
            display_payload = {k: v for k, v in parsed.items() if k != "_run_dir"}
        else:
            display_payload = parsed
        rendered = Syntax(
            json.dumps(display_payload, indent=2),
            "json",
            theme="ansi_dark",
            line_numbers=False,
        )
    except (json.JSONDecodeError, ValueError):
        rendered = Text(text)

    # First the raw R/Stata output — what the researcher actually
    # wants to look at — then the sanitized payload that Claude saw.
    if run_dir is not None:
        _render_raw_script_output(run_dir)
    console.print(
        Panel(
            rendered,
            border_style=border,
            padding=(0, 1),
            title="[dim]what Claude saw (sanitized)[/dim]",
            title_align="left",
        )
    )


def _render_raw_script_output(run_dir: Path) -> None:
    """Display the raw R / Stata stdout + stderr from a run directory.

    These files are written by ``executor.run_script`` right after the
    subprocess returns. Claude never sees them; this is purely for
    the researcher.
    """
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_text = _read_log(stdout_path)
    stderr_text = _read_log(stderr_path)
    if not stdout_text and not stderr_text:
        return
    # Guess the lexer from the log content: Stata batch output starts
    # with lines like `. sysuse auto, clear`; R output doesn't have a
    # consistent prefix. `stata` lexer handles both better than plain
    # text but Rich may not have it — fall back to text if missing.
    body: Any
    if stdout_text:
        body = Text(stdout_text)
        # Cap very long outputs so the TUI stays readable. The full
        # log still lives at stdout_path on disk for deep inspection.
        if len(stdout_text) > 8000:
            body = Text(
                stdout_text[-8000:]
                + f"\n\n[…truncated; full log: {stdout_path}]"
            )
    else:
        body = Text("(no stdout)", style="dim")
    console.print(
        Panel(
            body,
            border_style="blue",
            padding=(0, 1),
            title="[bold blue]R / Stata output (what you see)[/bold blue]",
            title_align="left",
        )
    )
    if stderr_text.strip():
        console.print(
            Panel(
                Text(stderr_text),
                border_style="yellow",
                padding=(0, 1),
                title="[yellow]stderr[/yellow]",
                title_align="left",
            )
        )


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_tool_result_text(content: str | list[dict[str, Any]] | None) -> str:
    """ToolResultBlock.content is string | list[dict] | None; normalize to str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Assistant / result message rendering
# ---------------------------------------------------------------------------

def _render_assistant(msg: AssistantMessage) -> None:
    for block in msg.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                console.print(Markdown(block.text))
        elif isinstance(block, ThinkingBlock):
            if block.thinking.strip():
                console.print(Text(block.thinking, style="dim italic"), soft_wrap=True)
        elif isinstance(block, ToolUseBlock):
            _render_tool_use(block)
        elif isinstance(block, ToolResultBlock):
            # Unusual — tool results usually come in UserMessage — but handle it.
            _render_tool_result(block)


def _render_user_tool_results(msg: UserMessage) -> None:
    """Tool results come back as UserMessage content after Claude calls a tool."""
    if isinstance(msg.content, str):
        return  # just the researcher's input, already echoed by the prompt
    for block in msg.content:
        if isinstance(block, ToolResultBlock):
            _render_tool_result(block)


def _render_result(msg: ResultMessage, mode: AuthMode) -> None:
    if msg.is_error:
        err = ", ".join(msg.errors) if msg.errors else msg.stop_reason or "unknown"
        console.print(Text(f"  (turn ended with error: {err})", style="red dim"))
        return
    parts: list[str] = []
    usage = msg.usage or {}
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    if in_tok is not None and out_tok is not None:
        parts.append(f"{in_tok} in / {out_tok} out")
    if mode == "api_key" and msg.total_cost_usd is not None:
        parts.append(f"${msg.total_cost_usd:.4f}")
    if parts:
        console.print(Text(f"  ({' · '.join(parts)})", style="dim"))


# ---------------------------------------------------------------------------
# Tool-use permission gate — catch-all deny for anything outside our allowlist
# ---------------------------------------------------------------------------

async def _gate_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    ctx: Any,
) -> PermissionResultAllow | PermissionResultDeny:
    """Deny any tool not explicitly on the allowlist, with a helpful message.

    This is the real catch-all. `disallowed_tools` is belt-and-suspenders
    for tools we can name up front; this function catches anything we can't
    name (future SDK additions, aliases, sub-tools) and denies by default.
    """
    del tool_input, ctx  # unused; kept for signature compliance
    if tool_name in ALLOWED_TOOL_NAMES:
        return PermissionResultAllow()
    console.print(
        Text(
            f"  (blocked tool use: {tool_name} — not on Nora's allowlist)",
            style="red dim",
        )
    )
    return PermissionResultDeny(
        behavior="deny",
        message=(
            f"Tool '{tool_name}' is not available in Nora. Use one of the "
            f"ten custom tools described in the system prompt "
            f"(mcp__{SERVER_NAME}__get_schema, search_schema, "
            f"request_data, submit_script, submit_script_file, "
            f"expand_result, list_results, list_results_global, "
            f"recall_conversation, read_attached_file). Nora does not "
            f"expose Bash, Read, Write, Edit, Glob, Grep, or any other "
            f"general tool."
        ),
        interrupt=False,
    )


# ---------------------------------------------------------------------------
# Turn + loop
# ---------------------------------------------------------------------------

async def _run_turn(
    client: ClaudeSDKClient, prompt: str, mode: AuthMode
) -> bool:
    await client.query(prompt)
    saw_any = False
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            if msg.error in (_AUTH_FAILURE, _BILLING_FAILURE):
                if msg.error == _AUTH_FAILURE:
                    _print_auth_hint()
                else:
                    console.print(
                        Text("Billing error from Anthropic — check your account.",
                             style="red bold")
                    )
                return False
            _render_assistant(msg)
            saw_any = True
        elif isinstance(msg, UserMessage):
            _render_user_tool_results(msg)
        elif isinstance(msg, ResultMessage):
            _render_result(msg, mode)
            break
        # SystemMessage / StreamEvent / RateLimitEvent: ignore in v0.
    if not saw_any:
        console.print(Text("  (no response)", style="yellow dim"))
    return True


# Models the researcher can pick from the composer. The catalog moved
# to ``nora.provider.catalog`` so the OpenAI provider can extend it
# with its own models. SUPPORTED_MODELS stays here as an Anthropic-only
# back-compat dict in the original ``{id: {label, context_window}}``
# shape — both the terminal's ``_build_options`` and the web UI's
# ``list_models`` bridge call still read from it.
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    m.id: {"label": m.label, "context_window": m.context_window}
    for m in ANTHROPIC_MODELS
}
DEFAULT_MODEL = PROVIDER_DEFAULTS["anthropic"]


def _build_options(
    cwd: Path,
    model: str | None = None,
    continue_conversation: bool = False,
) -> ClaudeAgentOptions:
    server = build_server()
    # Render through the canonical builder so the runtime-environment
    # placeholder (added in the multi-provider step) is filled. The
    # raw .format() call here used to omit it and crash on launch with
    # KeyError: 'runtime_environment'; the terminal entry point is
    # Anthropic-only by design.
    system_prompt = _build_system_prompt(cwd, SERVER_NAME, provider="anthropic")
    selected_model = model if model in SUPPORTED_MODELS else DEFAULT_MODEL
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        # Sonnet 4.6 default: 1M-token window, no beta header. Opus
        # and Haiku get their standard 200k windows. Researchers can
        # switch via the composer's model chip; changes take effect
        # on the next turn (the SDK client is torn down and
        # re-opened with the new options).
        model=selected_model,
        # Pass through the caller's continue_conversation flag. The
        # bridge sets this to True when the session dir already has
        # a prior chat_history.jsonl, so Claude picks up with memory
        # of the earlier turns instead of starting fresh. First-ever
        # open of a session passes False and starts a new claude
        # CLI conversation for the cwd.
        continue_conversation=continue_conversation,
        mcp_servers={SERVER_NAME: server},
        allowed_tools=list(ALLOWED_TOOL_NAMES),
        disallowed_tools=list(_DISALLOWED_BUILTINS),
        can_use_tool=_gate_tool_use,
        # We never want filesystem / shell access from Claude via the SDK's
        # fallback paths. `default` permission_mode routes through
        # can_use_tool for tools outside allowed/disallowed lists, which is
        # what we want (deny-by-default catch-all).
        permission_mode="default",
        # Don't load the user's / project's / local CLAUDE.md or settings.
        # Those can introduce hooks, tools, and slash-commands we don't
        # control, and we want Nora's tool surface to be exactly the
        # six tools above — no more, no less, regardless of the machine.
        setting_sources=[],
    )


async def _chat_loop() -> int:
    mode = _detect_auth_mode()
    cwd = get_cwd()
    env = detect_environment()
    _print_banner(mode, cwd, env)
    _print_schema_policy(cwd)
    # Platform preflight — submit_script currently requires macOS's
    # `sandbox-exec`. Warn up-front rather than failing at the first
    # script submission; `get_schema` and `request_data` still work
    # without it, so the session isn't useless.
    if env.sandbox_exec is None:
        if sys.platform == "darwin":
            console.print(
                Text.from_markup(
                    "[red bold]/usr/bin/sandbox-exec not found on this "
                    "macOS install.[/red bold] `submit_script` will "
                    "refuse to run without it. `get_schema` and "
                    "`request_data` still work.\n"
                )
            )
        else:
            console.print(
                Text.from_markup(
                    "[yellow bold]submit_script is macOS-only in this "
                    f"version.[/yellow bold] You're on `{sys.platform}`; "
                    "scripts will refuse to run (the sandbox that "
                    "protects your data relies on macOS `sandbox-exec`). "
                    "`get_schema` and `request_data` still work — you "
                    "can inspect schema and get bounded summaries, but "
                    "Claude can't execute R/Stata.\n"
                )
            )
    if not env.has_any_runtime():
        console.print(
            Text.from_markup(
                "[red bold]No R or Stata installed on this machine.[/red bold] "
                "Claude can reason about data schema and request bounded "
                "summaries, but [bold]submit_script[/bold] will fail until "
                "you install one.\n"
            )
        )
    opts = _build_options(cwd)
    try:
        async with ClaudeSDKClient(options=opts) as client:
            while True:
                try:
                    user_text = await asyncio.to_thread(
                        Prompt.ask, "[bold green]you[/bold green]"
                    )
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]bye.[/dim]")
                    return 0
                user_text = user_text.strip()
                if not user_text:
                    continue
                if user_text.lower() in _EXIT_WORDS:
                    console.print("[dim]bye.[/dim]")
                    return 0
                # Slash-commands are local to the TUI — they never reach
                # Claude. Keeps policy management out of the chat history
                # and out of the frontier's context.
                if user_text.startswith("/"):
                    handled = await asyncio.to_thread(
                        _handle_slash_command, user_text, cwd
                    )
                    if handled:
                        continue
                console.print()
                ok = await _run_turn(client, user_text, mode)
                console.print()
                if not ok:
                    return 2
    except ClaudeSDKError as e:
        console.print(f"\n[red bold]SDK error:[/red bold] {e}")
        return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nora",
        description=(
            "Nora — a local analysis assistant. Claude drives statistical "
            "analysis against data on your own machine; data never leaves."
        ),
    )
    parser.add_argument(
        "cwd",
        nargs="?",
        default=None,
        help=(
            "Working directory — the sandbox Nora operates in. Claude can "
            "read data only from inside this directory. Defaults to the "
            "current shell directory."
        ),
    )
    return parser.parse_args(argv)


def _prompt_for_cwd() -> Path:
    """Ask the researcher where their data lives.

    Launched via `nora` with no argv (e.g., double-clicked from a
    .app launcher) there's no shell cwd that makes sense as the data
    dir — the process inherits whatever Terminal was last looking at,
    often the user's HOME. Prompt for an explicit directory instead of
    silently picking the wrong one.

    Loops until the researcher enters a valid, existing directory or
    hits Ctrl-D.
    """
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold]Welcome to Nora.[/bold]\n\n"
                "Nora reads data only from one directory you choose. "
                "That directory is the sandbox — Claude's scripts can't "
                "reach anything outside it.\n\n"
                "[dim]Where do your data files live?[/dim]"
            ),
            border_style="cyan",
        )
    )
    default = str(Path.home() / "Documents")
    while True:
        try:
            raw = Prompt.ask(
                "[bold green]data directory[/bold green]",
                default=default,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            sys.exit(0)
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except OSError as e:
            console.print(f"[red]{e}[/red]. Try again.")
            continue
        if not path.exists():
            console.print(
                f"[red]Not found:[/red] {path}. Try again or Ctrl-D to quit."
            )
            continue
        if not path.is_dir():
            console.print(
                f"[red]Not a directory:[/red] {path}. Try again."
            )
            continue
        return path


def main() -> NoReturn:  # type: ignore[misc]
    args = _parse_args()
    if args.cwd:
        cwd = Path(args.cwd).expanduser()
    elif sys.stdin.isatty():
        # No argv, interactive terminal — prompt. This is the
        # double-clicked-.app flow where no cwd can be inferred.
        cwd = _prompt_for_cwd()
    else:
        # No argv and no TTY (piped / CI) — fall back to shell cwd.
        cwd = Path.cwd()
    try:
        cwd = cwd.resolve()
        if not cwd.is_dir():
            print(f"nora: not a directory: {cwd}", file=sys.stderr)
            sys.exit(2)
        set_cwd(cwd)
    except (OSError, NotADirectoryError) as e:
        print(f"nora: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        code = asyncio.run(_chat_loop())
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()

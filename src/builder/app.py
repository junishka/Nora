"""Builder — step-2 app: frontier chat with the locked-down MCP interface.

What changed from step 1 (the spine):
- The Claude Agent SDK's built-in tools (Bash, Read, Write, Edit, Glob, Grep,
  WebFetch, WebSearch, etc.) are disallowed. Claude reaches the local machine
  only through the five custom tools defined in `builder.tools`.
- `can_use_tool` is a catch-all deny: any tool not on the explicit allowlist
  is rejected, including tools added by future SDK versions we haven't heard
  of yet. This is belt-and-suspenders on top of `disallowed_tools`.
- The system prompt replaces Claude Code's default with a researcher-oriented
  one that introduces the five tools and the constraints.
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

from builder.config import get_cwd, set_cwd
from builder.env_detect import Environment, detect_environment
from builder.policy import (
    DEFAULT_MAX_DEPTH,
    POLICY_FILE,
    VALID_DEPTHS,
    BuilderPolicy,
    DatasetPolicy,
    get_max_depth,
    has_explicit_policy,
    load_policy,
    save_policy,
)
from builder.tools import ALLOWED_TOOL_NAMES, SERVER_NAME, build_server


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

_SYSTEM_PROMPT_TEMPLATE = """\
You are the analysis assistant inside Builder, a local tool that lets a \
researcher drive statistical analysis on data that remains on their machine. \
The data never leaves this machine. You reach the researcher's data ONLY \
through the five tools below — no other tools exist in this environment.

Working directory: {cwd}
All dataset paths you pass to tools must be inside this directory. Absolute \
paths outside it, `../` traversal, and symlink escapes are denied by the \
layer with an explanatory message.

Datasets detected in this directory (filenames only — contents still gated \
by the researcher's schema-depth policy):
{datasets_list}

Use this list to find candidates when the researcher mentions a dataset by \
shorthand. Inspect any of them with `get_schema`.

Target statistical languages: **R (via Rscript) and Stata**. No other \
languages are supported. If the researcher asks for Python, SAS, Julia, or \
anything else, explain that Builder only supports R and Stata.

Your tools (all prefixed `mcp__{SERVER_NAME}__` when referenced):

1. `get_schema(dataset, depth)` — structural summary of a dataset: variable \
names, types, labels, observation count. No values. `depth` is one of \
`names_only`, `names_types`, `names_types_labels`, `names_types_labels_summary`. \
Call this first, before writing any script.

2. `request_data(dataset, request_type, variable)` — ask the layer for \
a specific, bounded piece of information about a variable. Supported \
request types:\n\
  - `categorical_levels` — the list of level names whose counts meet \
the SDC threshold. Rare levels are hidden entirely (names and counts). \
Response includes a count of hidden levels so you know the visible list \
isn't complete.\n\
  - `numeric_bounds` — 5th and 95th percentile of a numeric variable, \
rounded to 2 sig figs. NOT min / max — those are individual observations \
and are never exposed.\n\
  - `na_count` — number of missing values in a variable. Denied if the \
non-missing subgroup is below the cell-suppression threshold.\n\
Use this instead of writing a probe script when you need targeted \
information about a variable.

3. `submit_script(language, code, label, source_dataset)` — run an R or \
Stata analysis script. The script MUST emit structured results via the \
Builder runtime library (sourced automatically before your code runs):\
\n\n\
R:\
\n\
  builder$from_lm(model)                  # from an lm() fit\
\n\
  builder$from_t_test(res, n1=..., n2=...) # from a t.test() result\
\n\
  builder$from_summarize(var, n, mean, sd, missing_count)\
\n\
  builder$from_table(var, counts, ...)    # 1D freq table (named list/table)\
\n\
  builder$from_crosstab(tbl)              # 2D crosstab (a 2D R table)\
\n\
  builder$from_magnitude_table(df, group_var, value_var, aggregation="sum")\
\n\
     # sum/mean of a numeric variable by group. Applies a (1, 85%)-dominance\
\n\
     # rule: if one contributor is >85% of a group's total, the cell is\
\n\
     # suppressed even when n is large (that single contributor's value is\
\n\
     # otherwise inferable).\
\n\
  builder$result(type, ...)               # generic escape hatch\
\n\n\
Stata (the Builder runtime is already on the adopath — just call the helper):\
\n\
  builder_result_regress, label("OLS ...")     # after `regress`, `logit`, etc.\
\n\
  builder_result_ttest, label("...")           # after `ttest`\
\n\
  builder_result_sum <var>, label("...")       # after `summarize <var>`\
\n\
  builder_result_tab <var>, label("...")       # 1-way frequency_table on <var>\
\n\
  builder_result_tab <var1> <var2>, label("...") # 2-way crosstab on <var1> x <var2>\
\n\
  builder_result_magnitude <group_var> <value_var>, aggregation(sum|mean), label("...")\
\n\
     # sum or mean of value_var by group_var. Same (1, 85%)-dominance rule as R.\
\n\n\
Raw stdout/stderr is shown to the researcher but NOT returned to you. \
You receive only the sanitized structured payload plus a result ID. \
Values are precision-clamped based on sample size; forbidden fields \
(residuals, fitted values, min/max/median) are dropped.\
\n\n\
ALWAYS pass `source_dataset` when your script reads from a known file. \
Builder compares the analysis's effective N to the dataset's row count \
and flags silent row drops (NA-drop by lm()/ttest, subset/filter in the \
script, listwise deletion). This catches "I thought the regression ran \
on all 1000 rows but it actually ran on 800" — the #1 way to quietly \
change the meaning of a result. Empty string is fine when the script \
generates its own data or reads multiple files.

4. `expand_result(result_id)` — retrieve a stored sanitized payload by ID. \
Use when you need details of an earlier result without carrying the whole \
thing in context.

5. `list_results()` — list session results (id + one-line label).

How to work with the researcher:

- Do the discovery work yourself before asking. When the researcher \
mentions something by shorthand — a dataset ("let's work on 05_"), a \
variable, a concept — look at what's actually there. Match against the \
dataset list above; call `get_schema` to see what variables a file \
contains; call `list_results` to see prior analyses. Narrow the \
candidates down, then ask a concrete question with the options you \
found. *"I see three 05_ files — 05_nuevo_matched.csv, \
05_nuevo_matched_gate.csv, 05_nuevo_matched_nogate.csv; which one, or \
should I look at what's different between them?"* is useful. *"What do \
you mean by 05_?"* isn't — you can see the list.
- Research decisions are the researcher's to make. Once the dataset and \
question are clear, surface the analytical choices — outcome and \
predictors, population / subgroups, transformations, interactions, \
robust or clustered standard errors, how to handle missingness — and \
wait for their call on those. If there's a clear convention for their \
field you can suggest it; the choice is theirs. Briefly say what the \
script will do before running it.
- After a run, explain what the result means in their terms before \
asking what's next. They may not be a programmer, but they know their \
field — translate, don't simplify.
- When presenting a regression result, show the full coefficient \
table the researcher expects. For each term include Estimate, Std. \
Error, t (or z), p-value, and when space allows a 95% CI. Don't drop \
columns to save space — a table with only Estimate and SE looks \
incomplete. Also report n, R² (and adj. R²), F (or χ²), and the \
degrees of freedom as a small block under the table. For a t-test: \
means per group, difference, t, df, p, and the CI. For a frequency \
table or crosstab: counts (and proportions when natural), with any \
`<10` suppressions preserved verbatim — never silently omit rows.
- Tone: a little corny is fine. A well-placed stats pun or dad joke — \
the groan-rather-than-laugh kind — lands well in easy moments: a clean \
result, a confirmed plan, waiting on a script. Skip it when there's \
frustration, errors to fix, or a real research judgment call on the \
table. One joke per chat, not one per turn. If you can't think of one \
that fits, don't force it.

Tool use notes:

- You don't have Bash, Read, Write, Edit, Glob, Grep, or any other \
general tool — only the five above. If you think you need one, the \
right move is a custom tool call or asking the researcher.
- Keep scripts small and focused. One question per script is usually \
right.
- When you need something specific about a variable (levels, rough scale, \
missingness), `request_data` is faster and pre-approved — prefer it over \
writing a probe script.
- Don't suggest uploading data, using cloud services, or anything that \
moves data off the machine.

STAGE NOTE: step 4 is complete.\
- `get_schema` — real.\
- `submit_script` — real: R / Stata subprocess under sandbox-exec (network \
denied), runtime library injected, output routed through the real sanitizer \
and persisted to SQLite at `<cwd>/.builder/results.db`. Supports \
`linear_regression`, `t_test`, `descriptive`, `frequency_table` (primary + \
secondary cell suppression), `crosstab` (2D, cells only — no margins \
emitted), and `magnitude_table` (sum/mean by group, with a \
(1, 85%)-dominance rule).\
- `expand_result`, `list_results` — real (backed by the SQLite store).\
- `request_data` — real: three bounded query types with per-type SDC. \
More types (missingness pattern, distribution summary) land in later \
step-5 work.

Be honest with the researcher about errors or rejections. When a script fails \
or is rejected, a diagnostic row is still inserted in the store so the \
researcher can audit via `expand_result`.
"""


def _detect_auth_mode() -> AuthMode:
    """See step 1 notes — unchanged."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api_key"
    claude_json = Path.home() / ".claude.json"
    if claude_json.is_file():
        try:
            data = json.loads(claude_json.read_text())
        except (json.JSONDecodeError, OSError):
            return "unknown"
        oauth = data.get("oauthAccount")
        if isinstance(oauth, dict) and oauth.get("accountUuid"):
            return "subscription"
    return "unknown"


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
    sbx = "[green]sandbox on[/green]" if env.sandbox_exec else "[yellow]no sandbox[/yellow]"
    parts.append(sbx)
    return " · ".join(parts)


def _print_banner(mode: AuthMode, cwd: Path, env: Environment) -> None:
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold]Builder[/bold] [dim]· local analysis assistant · "
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
# so researchers don't have to remember what each one means.
_DEPTH_DESCRIPTIONS: dict[str, str] = {
    "names_only": "variable names only",
    "names_types": "+ type per variable (conservative default)",
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
      4. Writes the update to ``<cwd>/.builder/policy.json``.

    Nothing here touches Claude. The chat message is intercepted in
    the loop before it would otherwise be sent to the model.
    """
    datasets = _scan_datasets(cwd)
    if not datasets:
        console.print(
            Text.from_markup(
                "[yellow]No datasets (.csv / .dta / .rds) found in "
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


def _render_policy_table(datasets: list[Path], policy: BuilderPolicy) -> Any:
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
    cwd: Path, dataset: Path, policy: BuilderPolicy
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
    updated = BuilderPolicy(
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
                "  [cyan]exit[/cyan]      quit Builder"
            ),
            border_style="cyan",
        )
    )


def _dataset_listing(cwd: Path) -> str:
    """Render a compact dataset listing for the system prompt.

    Claude has no tool to list the working directory — the five MCP
    tools are narrow by design. Without an at-startup enumeration
    Claude can't answer "work on 05_" concretely; it has to either
    guess or ask a generic "what do you mean?" question. Dropping
    the filenames into the system prompt fixes that: Claude sees
    the list, matches candidates, and can ask a specific question
    ("I see three 05_ files; which one?").

    Returns a multi-line bullet list, or an explicit "(none)" marker
    so Claude doesn't hallucinate data that isn't there. Filenames
    are not gated by the schema-depth policy — only the *contents*
    of each dataset are. See ``policy.py`` for the depth-tier model.
    """
    datasets = _scan_datasets(cwd)
    if not datasets:
        return "  (no .csv / .dta / .rds datasets detected in this directory)"
    # One per line so the prompt stays readable even with 20+ entries.
    # Cap at a very generous limit — if a directory has 500 files we
    # truncate with a note, but 20–50 is the realistic range.
    cap = 80
    names = [d.name for d in datasets[:cap]]
    body = "\n".join(f"  - {n}" for n in names)
    if len(datasets) > cap:
        body += f"\n  … and {len(datasets) - cap} more"
    return body


def _scan_datasets(cwd: Path) -> list[Path]:
    """Return dataset files in ``cwd`` (top-level only), sorted.

    Only the three formats Builder currently supports: ``.csv``,
    ``.dta``, ``.rds``. We scan the top level only — datasets
    nested inside subdirs don't participate in the researcher's
    consent UI until they do.
    """
    results: list[Path] = []
    try:
        for child in cwd.iterdir():
            if child.is_file() and child.suffix.lower() in (".csv", ".dta", ".rds"):
                results.append(child)
    except OSError:
        return []
    results.sort()
    return results


def _print_schema_policy(cwd: Path) -> None:
    """Show the researcher which schema-depth ceiling applies to each
    dataset in cwd. Silent if no datasets are present.

    This is a non-interactive notification — the researcher edits
    ``<cwd>/.builder/policy.json`` by hand to change what Claude can
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
                "Builder uses your Claude account. Pick one:\n\n"
                "  [bold]1.[/bold] Sign in with your Claude subscription "
                "(Pro / Max / Team):\n"
                "     Open a new Terminal and run: [cyan]claude[/cyan]\n"
                "     Follow the browser prompt, then come back.\n\n"
                "  [bold]2.[/bold] Use an Anthropic API key:\n"
                "     [cyan]export ANTHROPIC_API_KEY=sk-ant-…[/cyan]\n"
                "     Then restart builder."
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
    # Strip the mcp__builder__ prefix for readability.
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
            lexer = "r" if lang.lower() == "r" else "stata"
            console.print(Syntax(code, lexer, theme="ansi_dark", line_numbers=False))
    elif short == "get_schema":
        summary = f"dataset={inp.get('dataset', '?')!r} depth={inp.get('depth', '?')!r}"
        console.print(Text(f"⚙ get_schema  {summary}", style="cyan"))
    elif short == "request_data":
        summary = (
            f"{inp.get('request_type', '?')} on {inp.get('variable', '?')!r} "
            f"(dataset={inp.get('dataset', '?')!r})"
        )
        console.print(Text(f"⚙ request_data  {summary}", style="cyan"))
    elif short == "expand_result":
        console.print(Text(f"⚙ expand_result  {inp.get('result_id', '?')!r}", style="cyan"))
    elif short == "list_results":
        console.print(Text("⚙ list_results", style="cyan"))
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
            f"  (blocked tool use: {tool_name} — not on Builder's allowlist)",
            style="red dim",
        )
    )
    return PermissionResultDeny(
        behavior="deny",
        message=(
            f"Tool '{tool_name}' is not available in Builder. Use one of the "
            f"five custom tools described in the system prompt "
            f"(mcp__{SERVER_NAME}__get_schema, request_data, submit_script, "
            f"expand_result, list_results). Builder does not expose Bash, "
            f"Read, Write, Edit, Glob, Grep, or any other general tool."
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


def _build_options(cwd: Path) -> ClaudeAgentOptions:
    server = build_server()
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        cwd=cwd,
        SERVER_NAME=SERVER_NAME,
        datasets_list=_dataset_listing(cwd),
    )
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
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
        # control, and we want Builder's tool surface to be exactly the
        # five tools above — no more, no less, regardless of the machine.
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
        prog="builder",
        description=(
            "Builder — a local analysis assistant. Claude drives statistical "
            "analysis against data on your own machine; data never leaves."
        ),
    )
    parser.add_argument(
        "cwd",
        nargs="?",
        default=None,
        help=(
            "Working directory — the sandbox Builder operates in. Claude can "
            "read data only from inside this directory. Defaults to the "
            "current shell directory."
        ),
    )
    return parser.parse_args(argv)


def _prompt_for_cwd() -> Path:
    """Ask the researcher where their data lives.

    Launched via `builder` with no argv (e.g., double-clicked from a
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
                "[bold]Welcome to Builder.[/bold]\n\n"
                "Builder reads data only from one directory you choose. "
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
            print(f"builder: not a directory: {cwd}", file=sys.stderr)
            sys.exit(2)
        set_cwd(cwd)
    except (OSError, NotADirectoryError) as e:
        print(f"builder: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        code = asyncio.run(_chat_loop())
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()

# Builder

Local privacy layer that lets Claude help a researcher analyze
sensitive data without that data leaving the researcher's machine.

Claude talks to the researcher; a narrow MCP tool interface
restricts Claude to five operations (`get_schema`, `request_data`,
`submit_script`, `expand_result`, `list_results`); scripts run
locally under macOS `sandbox-exec` with network denied and a narrow
subpath-allowlist for file reads; every output passes through a
disclosure-control sanitizer before Claude sees it. The researcher
sees raw logs. Claude only ever sees sanitized, SDC-filtered
results.

See [`docs/handoff.md`](docs/handoff.md) for the single-page
summary if you're picking the project up. [`docs/overview.md`](docs/overview.md)
has a plain-language description; [`docs/direction.md`](docs/direction.md)
has the long-form architectural direction and open-question log.

## Status

Alpha. Privacy invariants are implemented and tested (194 tests).
Two frontends ship: a terminal UI (`builder`) and a pywebview-based
web UI (`builder-ui`). The `.dmg` currently distributes the terminal
entry point only; `builder-ui` runs from source. See
[the handoff doc's status table](docs/handoff.md#where-it-stands).

## Platform requirements

**macOS-only at this stage.** The `submit_script` tool relies on
macOS `sandbox-exec` for its privacy boundary — the deny-default
subpath-allowlist profile is what enforces "Claude's scripts can
only read the researcher's cwd and a narrow set of system paths."
On other platforms `get_schema` and `request_data` still work
(schema inspection and bounded-fact queries), but scripts refuse to
execute. A Linux port using `bubblewrap` or an equivalent would be
possible but is not on the near-term roadmap.

Also needs:

- Python 3.10+.
- R (`Rscript` on PATH) to run R-language analyses, OR Stata
  (`stata-mp` / `stata-se` / `stata` on PATH, or installed at
  `/Applications/Stata`) to run Stata-language analyses. At least
  one is required for `submit_script`.
- A Claude subscription or `ANTHROPIC_API_KEY` — auth is inherited
  from the `claude` CLI or environment.

## Installing

If you have a `Builder.dmg` handed to you, see
[`docs/install.md`](docs/install.md) for double-click install
instructions and first-run Gatekeeper workaround.

If you want to build the `.dmg` yourself from the repo, or just run
from the development checkout, see "Running from source" below.

## Running from source (developer workflow)

```bash
git clone https://github.com/junishka/builder.git
cd builder
uv sync --group dev
uv run pytest                       # expect: 194 passed

# Terminal frontend
uv run builder                      # landing prompt for the data dir
uv run builder /path/to/your/data   # straight into chat

# Web-UI frontend (native WKWebView window via pywebview)
uv run builder-ui                   # landing screen: drop files or pick folder
uv run builder-ui /path/to/data     # straight into chat
```

With no path argument, `builder-ui` opens a landing screen where you
can drag `.csv` / `.dta` / `.rds` files onto a drop zone, click
**Choose files…** (native multi-select), or **Choose folder…**.
Dropped files land in `~/.builder-sessions/<timestamp>_<id>/`
which becomes the sandbox root for that session.

To rebuild the distributable `.app` and `.dmg`:

```bash
bash packaging/build_app.sh   # → dist/Builder.app  (unsigned)
bash packaging/build_dmg.sh   # → dist/Builder.dmg  (~34 MB)
```

The resulting `.app` bundles Python, all dependencies, and the
runtime libraries — researchers running it don't need Python,
`uv`, or any build tooling installed. R and (optionally) Stata are
still required separately since Builder invokes them as
subprocesses.

## Project layout

- `src/builder/` — Python source.
  - `app.py` — terminal entry point, system prompt, chat loop.
  - `ui.py` + `web/` — pywebview shell + HTML/CSS/JS frontend.
  - `tools.py` — the five MCP tools Claude sees.
  - `executor.py` — sandbox profile, R/Stata subprocess, per-run token.
  - `sanitizer.py` + `sdc.py` + `text_safety.py` — disclosure control.
  - `schema.py` — dataset metadata extraction (`.csv`/`.dta`/`.rds`).
  - `policy.py` — per-dataset schema-depth ceilings + persistence.
  - `store.py` — SQLite result store, audit log.
  - `runtime/` — R library + five Stata `.ado` files scripts call.
  - `chat_service.py` — typed event stream both frontends consume.
- `tests/` — 194 tests. `test_sanitizer.py` is the property-test
  backbone; `test_executor_*` cover the sandbox SBPL profile.
- `docs/` — handoff, overview, direction, install, verification.
- `packaging/` — PyInstaller spec + `.app`/`.dmg` build scripts.

## Security model

Three independent layers provide the privacy guarantee:

1. **Tool interface** — Claude has no general-purpose tools (no
   filesystem, no shell, no network). Only the five MCP tools, and
   they're enumerated exhaustively.
2. **Sandbox** — `(deny default)` `sandbox-exec` profile; narrow
   subpath-allowlist for reads and writes; network denied entirely.
3. **Sanitizer** — every execution result passes through
   statistical-disclosure-control rules (precision clamping,
   cell-size suppression, dominance checks) and text-safety
   sanitation before it reaches Claude.

None of these three layers depends on which of the allowed tools
Claude invokes. See [`docs/direction.md`](docs/direction.md) for
the full analysis, including what a local LLM would and would not
add on top of these layers.

## License

Proprietary. Not yet open-sourced.

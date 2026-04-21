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

See [`docs/overview.md`](docs/overview.md) for a plain-language
description and [`docs/direction.md`](docs/direction.md) for the
working architectural direction.

## Status

Alpha. Privacy invariants are implemented and tested (156 tests),
but this is not yet a product non-developers can install. See the
[remaining work section in the direction doc](docs/direction.md#whats-remaining-prioritized).

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

## Running from source (developer workflow)

Double-click install (`.dmg`) is planned but not yet done. Today:

```bash
git clone https://github.com/junishka/builder.git
cd builder
uv sync --group dev
uv run pytest              # expect: 156 passed (on macOS with R installed)
uv run python -m builder /path/to/your/data
```

The first argument is the directory containing the data files
Builder is allowed to read. Leaving it off uses the current shell
directory.

## Project layout

- `src/builder/` — Python source. `app.py` is the entry point;
  `tools.py` defines the MCP interface; `executor.py` runs scripts
  under the sandbox; `sanitizer.py` applies the SDC rules; `schema.py`
  extracts dataset metadata; runtime libraries for R and Stata live
  under `src/builder/runtime/`.
- `tests/` — pytest suite. Property tests (via Hypothesis) are the
  correctness backbone for the sanitizer.
- `docs/` — working architectural direction and plain-language
  overview.

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

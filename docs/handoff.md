# Builder — handoff

Single-page entry point for picking this project up. Current as of
the last commit on `main`. If something here disagrees with the code,
trust the code and file a patch to this doc.

## What Builder is (one paragraph)

A local macOS app that lets a researcher drive statistical analysis
(R or Stata) on their own data with Claude, without that data leaving
the machine. Claude reaches the researcher's files through a narrow
MCP interface (six tools, nothing else — no Bash, no filesystem, no
network). Scripts run under `sandbox-exec` with network denied and a
tight subpath-allowlist for reads; every output passes through a
disclosure-control sanitizer (SDC rules from Eurostat / UK ONS
guidance) before anything reaches Claude. The researcher sees raw
R/Stata output in the UI; Claude only ever sees sanitized summaries.

## Where it stands

| Layer | Status |
|---|---|
| **Data-boundary architecture** — tool interface + sandbox + sanitizer | ✅ implemented and tested |
| **Schema extraction** (.csv / .dta / .rds, four depth tiers) | ✅ done |
| **Executor** ((deny default) sandbox, R + Stata, runtime libraries, per-run token) | ✅ done, verified on real data |
| **Sanitizer** (six analysis families, SDC rules, text-safety) | ✅ done, ~3,600 Hypothesis cases |
| **Result store** (SQLite, audit log, expand_result) | ✅ done |
| **Permission policy** (per-dataset schema-depth ceiling, UI dropdowns) | ✅ done |
| **Terminal UI** (`builder`) — Rich-based chat, `/policy` wizard | ✅ done |
| **Web UI** (`builder-ui`) — pywebview shell, drag-drop upload, result panels, Open-in-R/Stata buttons | ✅ done |
| **Packaging** (`.app` + `.dmg`) — terminal `builder` only | ✅ done (unsigned; right-click → Open first time) |
| **Real-researcher pilot** (step 8) | ⏳ self-pilot in progress |
| **Cross-query composition / release ledger** | ⏭ named, deferred |
| **Web UI bundled into the .app** | ⏭ not done |
| **Dataset-picker sidebar / session browser** | ⏭ not done |

194 tests passing, covering: SDK lockdown, schema, executor SBPL
profile (pure unit tests + integration gated on Rscript), sanitizer
(property tests), policy, text-safety, row-count audit, stderr
isolation, per-run token authenticity.

## Running it

```bash
# Terminal UI — same-shell chat
uv run builder                              # opens landing prompt
uv run builder /path/to/data                # opens straight into chat

# Web UI — native WKWebView window
uv run builder-ui                           # landing: drop files or pick folder
uv run builder-ui /path/to/data             # opens straight into chat

# Tests
uv run pytest -q                            # expect 194 passing

# Build distributable .dmg (terminal only, unsigned)
bash packaging/build_app.sh                 # → dist/Builder.app (~68 MB)
bash packaging/build_dmg.sh                 # → dist/Builder.dmg (~34 MB)
```

Auth is inherited from the `claude` CLI (subscription) or
`ANTHROPIC_API_KEY` (per-token). Builder doesn't handle auth itself.

## Architecture at a glance

Three independent privacy layers. A break in any one is a bug; a
break in two at the same time is a privacy incident.

1. **Tool interface** (`src/builder/tools.py`) — Claude has exactly
   six tools: `get_schema`, `request_data`, `submit_script`,
   `expand_result`, `list_results`, `recall_conversation`. SDK
   built-ins (Bash, Read, Write, …) are disabled via
   `disallowed_tools` + `can_use_tool` catch-all + `setting_sources=[]`.
2. **Sandbox** (`src/builder/executor.py`) — `sandbox-exec` with
   `(deny default)` base, explicit subpath-allowlist for reads
   (cwd + runtime dirs + a minimal set of system paths), tighter
   allowlist for writes, network denied. Refuses to run if
   `sandbox-exec` is missing rather than falling through.
   Executor also generates a per-run HMAC-style token that the
   runtime library embeds in every emitted payload — hand-crafted
   JSON without the token is rejected.
3. **Sanitizer** (`src/builder/sanitizer.py`, `sdc.py`,
   `text_safety.py`) — allowlist of field names per analysis
   family, SDC rules (precision clamping by N, cell-size
   suppression threshold 10, secondary suppression for 1-D freq
   tables, (1, 85%)-dominance for magnitude tables), text-safety
   pass on every data-origin string.

Session model: `builder-ui` without an argv opens a landing screen;
dropped / picked files land in `~/.builder-sessions/<ts>_<id>/`
which becomes the cwd. That dir is spaces-free (Stata-safe),
outside cloud-sync roots, persistent across restarts.

## Known-real, deferred

- **Cumulative-inference / cross-query composition.** Single-query
  SDC is tight; many-query composition is not bounded.
  Acceptable for self-pilots; the blocker before wider
  distribution. Store already holds every sanitized emission, so
  a release-ledger feature has raw material to build on. See
  `docs/direction.md` §"Known-real, design-pending" for the DP /
  τ-ARGUS / release-ledger options and why naïve query counters
  are worse than nothing.
- **Web UI bundled into the .app.** `builder-ui` runs from source
  only; the `.dmg` ships the terminal `builder` only. Fold
  `src/builder/web/` into `packaging/builder.spec` as `datas` and
  update the launcher to support both entry points.
- **Stata signing / notarization.** `.dmg` is unsigned — first
  run requires right-click → Open or `xattr -cr`. Needs Apple
  Developer Program membership ($99/yr) before wider
  distribution.

## Rough edges (work, but annoy)

- Transformations log shows `"dropped unknown/forbidden field
  'label'"` on every R submit_script because the `builder$from_lm(m,
  label=…)` arg is stripped by the schema allowlist. Harmless (the
  submit_script MCP-tool label is stored separately), but noisy.
  Fix: widen the per-type string allowlist to include `label`.
- R output in the UI's result panel capped at 32 KB; larger logs
  truncate with a note pointing at the full file. Fine for
  regression tables, may cut off when a script prints per-row
  diagnostics.
- Sanitizer drops the `_cons` coefficient as empty when Stata
  reports it as "omitted" (perfect-fit edge case) — generated
  JSON becomes malformed. Only triggers on degenerate toy data;
  flagged in a test comment. Not a production blocker.

## Key files (reading order)

| File | What's there |
|---|---|
| `src/builder/app.py` | Terminal entry point, system prompt, chat loop, rendering |
| `src/builder/ui.py` | Web UI entry point, pywebview bridge, session staging |
| `src/builder/tools.py` | The six MCP tools. Start here to understand Claude's surface |
| `src/builder/executor.py` | Sandbox profile, R/Stata subprocess plumbing, per-run token |
| `src/builder/sanitizer.py` + `sdc.py` | The SDC allowlist and clamp/suppress primitives |
| `src/builder/runtime/builder.R` + `builder_result_*.ado` | Emitters the scripts call |
| `src/builder/chat_service.py` | Typed event stream both UIs consume |
| `src/builder/web/{index.html,app.js,markdown.js,style.css}` | Web frontend |
| `docs/direction.md` | Long-form architectural doc; open questions |
| `docs/overview.md` | Plain-language description for researchers |
| `docs/install.md` | Researcher-facing install flow |
| `docs/verification.md` | Manual smoke-test recipes (incl. Stata, which CI can't) |
| `tests/` | 194 tests. `test_sanitizer.py` is the property-test backbone |

## Decisions worth not re-litigating

- **No plan-submission / DSL-grammar pivot.** A reviewer proposed
  replacing `submit_script` with a constrained plan grammar +
  local-LLM translator. Rejected as over-correction — the privacy
  guarantee already rests on tool interface + sandbox + sanitizer,
  none of which depend on who authored the code. See
  `docs/direction.md` "The decision."
- **No bundled local LLM.** Install footprint (~15 GB), worse code
  generation than Claude, no privacy gain given the existing
  stack. Re-opens only when a specific use case demands it
  (stderr-based repair, text-data redaction).
- **macOS-only for now.** Sandbox relies on `sandbox-exec`. Linux
  would need `bubblewrap`/`nsjail`, Windows is on nobody's path.
- **Schema policy is a ceiling, not a fixed value.** Claude can
  request any depth ≤ ceiling. Researchers edit via the Permission
  chip (web) or `/policy` slash-command (terminal).
- **Stata batch `-b do <path>` breaks on spaces.** Executor passes
  `script.do` (bare filename) with subprocess cwd set to the run
  dir. Don't "improve" back to an absolute path.

## Next concrete step

If you're picking this up to finish and ship it, in this order:

1. **Run it yourself on your own data.** Catch the UX rough edges
   before anyone else sees them. The current author has done this
   once; a second pair of eyes surfaces new things.
2. **Bundle the web UI into the .app.** Researcher friction here is
   lower than almost anywhere else — the `.dmg` is the thing people
   install, and right now it only ships the terminal frontend.
3. **Try it on a colleague's data** (or yours via a colleague). The
   difference between "self-pilot" and "someone who didn't build
   it" is where most real UX bugs live.
4. **Then**, and only then, pick up the cumulative-inference /
   release-ledger work. It's a real project, and it matters more
   as you widen the audience — but it's the wrong thing to spend a
   week on before even one outside pilot.

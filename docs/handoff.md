# Nora — handoff

Single-page entry point for picking this project up. Last
substantive update **2026-04-25**, after the Builder → Nora rename
and the .app-launches-web-UI fix. If something here disagrees with
the code, trust the code and file a patch to this doc.

## What Nora is (one paragraph)

A local macOS app that lets a researcher drive statistical analysis
(R or Stata) on their own data with Claude as the model behind the
scenes, without that data leaving the machine. From the researcher's
point of view, Nora is one product they talk to — Claude is the
model wired into the loop, not exposed in the UI. Claude reaches
the researcher's files through a narrow MCP interface (six tools,
nothing else — no Bash, no filesystem, no network). Scripts run
under `sandbox-exec` with network denied and a tight subpath-allowlist
for reads; every output passes through a disclosure-control sanitizer
(SDC rules from Eurostat / UK ONS guidance) before anything reaches
the model. The researcher sees raw R/Stata output in the UI; the
model only ever sees sanitized summaries.

## Where it stands

| Layer | Status |
|---|---|
| **Data-boundary architecture** — tool interface + sandbox + sanitizer | ✅ implemented and tested |
| **Schema extraction** (.csv / .dta / .rds, four depth tiers) | ✅ done; default tier is `names_types_labels_summary` |
| **Executor** ((deny default) sandbox, R + Stata, runtime libraries, per-run token) | ✅ done, verified on real data |
| **Env-var allowlist** in subprocess env (no ANTHROPIC_API_KEY etc. visible to scripts) | ✅ done |
| **Sanitizer** (six analysis families, SDC rules, text-safety, structural size caps) | ✅ done, ~3,600 Hypothesis cases |
| **Result store** (SQLite, audit log, `expand_result`, **per-cwd cache**) | ✅ done; cache rebinds on session switch |
| **Permission policy** (per-dataset schema-depth ceiling, UI dropdowns) | ✅ done |
| **Filename / variable-name sanitization** at every prompt-injection surface | ✅ done |
| **Memory stack** — chat-history persisted, warm-start prefix injected on every fresh client, `recall_conversation` tool for older lookups | ✅ done |
| **Durable session state** — `.nora/session_state.json` written after each turn (last exchange, recent results, datasets, model) | ✅ done |
| **Terminal UI** (`nora`) — Rich-based chat, `/policy` wizard | ✅ done |
| **Web UI** (`nora-ui`) — pywebview shell, sessions sidebar, theme toggle, model picker, drag-drop file/image upload, typewriter, Lottie cat loading indicator, status line, Permission/Model chips with popups | ✅ done |
| **Packaging** (`.app` + `.dmg`) — bundles the **web UI**; .app launches pywebview directly with no Terminal popup | ✅ done & smoke-tested locally (unsigned — local-build only until Apple Developer Program signing) |
| **Product-identity prompt rule** — model introduces itself as Nora, not as "the assistant inside Nora" or by model name | ✅ done |
| **Real-researcher pilot** | ⏳ self-pilot in progress |
| **Cross-query composition / release ledger** | ⏭ named, deferred |
| **Apple Developer Program signing + notarization for distributable .dmg** | ⏭ blocked on $99/yr cert |
| **Stata batch wrapper around `_cons` "omitted" edge case** | ⏭ named, low-priority |

283 tests passing, covering: SDK lockdown, schema, executor SBPL
profile (pure unit tests + integration gated on Rscript), sanitizer
(property tests), policy, text-safety, row-count audit, stderr
isolation, per-run token authenticity, env-var allowlist
(subprocess can't see shell secrets), cross-session store isolation,
filename prompt-injection, OLS coefficient-key constraint,
confidence-interval length constraint, structural size caps, plus
the memory-stack tests (turn-grouping reader, warm-start prefix
generation, durable session-state writer, bridge lifecycle paths
for cwd-switch / interrupt / timestamp persistence).

## Running it

```bash
# Web UI — native WKWebView window. The recommended frontend.
uv run nora-ui                           # landing: drop files or pick folder
uv run nora-ui /path/to/data             # opens straight into chat

# Terminal UI — same-shell chat. Power-user / shell-only path.
uv run nora                              # opens landing prompt
uv run nora /path/to/data                # opens straight into chat

# Tests
uv run pytest -q                            # expect 283 passing

# Build the .app + .dmg locally. Bundles the web UI.
# Distribution to other people is blocked on Apple Developer Program
# signing — the unsigned .dmg trips Gatekeeper for anyone who didn't build it themselves.
bash packaging/build_app.sh                 # → dist/Nora.app (~70 MB)
bash packaging/build_dmg.sh                 # → dist/Nora.dmg (~35 MB)
open dist/Nora.app                       # smoke test the build
tail -F ~/Library/Logs/Nora/nora-*.log  # if it doesn't open
```

Auth is inherited from the `claude` CLI (subscription) or
`ANTHROPIC_API_KEY` (per-token). Nora doesn't handle auth itself.
The subprocess env-var allowlist (executor.py) keeps `ANTHROPIC_API_KEY`
out of script-visible env so a prompt-injected R/Stata script can't
exfiltrate it through an "allowed" numeric field.

## Architecture at a glance

Three independent privacy layers. A break in any one is a bug; a
break in two at the same time is a privacy incident.

1. **Tool interface** (`src/nora/tools.py`) — Claude has exactly
   six tools: `get_schema`, `request_data`, `submit_script`,
   `expand_result`, `list_results`, `recall_conversation`. SDK
   built-ins (Bash, Read, Write, …) are disabled via
   `disallowed_tools` + `can_use_tool` catch-all + `setting_sources=[]`.
2. **Sandbox** (`src/nora/executor.py`) — `sandbox-exec` with
   `(deny default)` base, explicit subpath-allowlist for reads
   (cwd + runtime dirs + a minimal set of system paths), tighter
   allowlist for writes, network denied. Refuses to run if
   `sandbox-exec` is missing rather than falling through.
   Executor also generates a per-run HMAC-style token that the
   runtime library embeds in every emitted payload — hand-crafted
   JSON without the token is rejected.
3. **Sanitizer** (`src/nora/sanitizer.py`, `sdc.py`,
   `text_safety.py`) — allowlist of field names per analysis
   family, SDC rules (precision clamping by N, cell-size
   suppression threshold 10, secondary suppression for 1-D freq
   tables, (1, 85%)-dominance for magnitude tables), text-safety
   pass on every data-origin string.

Session model: `nora-ui` without an argv opens a landing screen;
dropped / picked files land in `~/.nora-sessions/<ts>_<id>/`
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
- **`.app` / `.dmg` distribution to other people.** The local build
  works (`.app` launches the web UI directly via pywebview, no
  Terminal popup). What's missing is an Apple Developer Program
  signature — without it, anyone you hand the .dmg to hits a hard
  Gatekeeper warning and most users won't get past it. Right-click →
  Open / `xattr -cr` workarounds are documented in install.md but
  aren't acceptable for a "just install this" handoff. Cost: $99/yr.
  Once signed, also worth notarizing for the cleanest first-launch
  experience.

## Rough edges (work, but annoy)

- Transformations log shows `"dropped unknown/forbidden field
  'label'"` on every R submit_script because the `nora$from_lm(m,
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
| `src/nora/app.py` | Terminal entry point, system prompt, chat loop, rendering |
| `src/nora/ui.py` | Web UI entry point, pywebview bridge, session staging |
| `src/nora/tools.py` | The six MCP tools. Start here to understand Claude's surface |
| `src/nora/executor.py` | Sandbox profile, R/Stata subprocess plumbing, per-run token |
| `src/nora/sanitizer.py` + `sdc.py` | The SDC allowlist and clamp/suppress primitives |
| `src/nora/runtime/nora.R` + `nora_result_*.ado` | Emitters the scripts call |
| `src/nora/chat_service.py` | Typed event stream both UIs consume |
| `src/nora/chat_history.py` | Turn-grouped reader over the persisted chat log; warm-start prefix renderer |
| `src/nora/session_state.py` | Atomic writer / reader for `.nora/session_state.json` |
| `src/nora/web/{index.html,app.js,markdown.js,style.css}` | Web frontend |
| `src/nora/web/{cat-loading.json,lottie-player.js}` | Lottie loading-indicator asset + vendored player (MIT-licensed, pinned 2.0.12) |
| `src/nora/__main_ui__.py` | Bundle entry — calls `nora.ui:main`. The .app launches this, NOT the terminal CLI |
| `docs/direction.md` | Long-form architectural doc; open questions |
| `docs/overview.md` | Plain-language description for researchers |
| `docs/install.md` | Researcher-facing install flow |
| `docs/verification.md` | Manual smoke-test recipes (incl. Stata, which CI can't) |
| `tests/` | 283 tests. `test_sanitizer.py` is the property-test backbone |

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
- **Schema policy is a ceiling, not a fixed value.** The model can
  request any depth ≤ ceiling. Researchers edit via the Permission
  chip (web) or `/policy` slash-command (terminal).
- **Stata batch `-b do <path>` breaks on spaces.** Executor passes
  `script.do` (bare filename) with subprocess cwd set to the run
  dir. Don't "improve" back to an absolute path.
- **The product is Nora; the model behind it is Claude.** Don't
  expose "Claude" in user-facing copy. The system prompt has an
  explicit identity rule telling the model to introduce itself as
  Nora and not as "the assistant inside Nora" or any other framing.
  Same goes for docs and UI strings — when the user sees the
  product name, it's Nora.
- **GitHub repo is still named `builder`** (URL: github.com/junishka/builder).
  Renaming a GitHub repo is an out-of-band action; URLs in install
  instructions still point there. The local clone gets renamed at
  `git clone … nora` time so the on-disk dir matches the product
  name. If/when the repo gets renamed too, update the `git clone`
  URLs in README and install.md.

## Next concrete step

If you're picking this up to finish and ship it, in this order:

1. **Run it yourself on your own data.** Catch the UX rough edges
   before anyone else sees them. The current author has done this
   once; a second pair of eyes surfaces new things.
2. **Sign and notarize the .app.** The build pipeline already
   produces a working .app that launches the web UI directly with
   no Terminal popup. The blocker for handing it to colleagues is
   the missing Apple Developer Program signature — Gatekeeper
   refuses unsigned apps cleanly enough that the right-click-Open
   workaround is friction nobody should be subjected to. $99/yr.
3. **Try it on a colleague's data** (or yours via a colleague). The
   difference between "self-pilot" and "someone who didn't build
   it" is where most real UX bugs live.
4. **Then**, and only then, pick up the cumulative-inference /
   release-ledger work. It's a real project, and it matters more
   as you widen the audience — but it's the wrong thing to spend a
   week on before even one outside pilot.

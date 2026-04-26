# Nora — handoff

Single-page entry point for picking this project up. Last
substantive update **2026-04-26**, after the multi-provider /
Python-language / new-file-formats expansion. If something here
disagrees with the code, trust the code and file a patch to this
doc.

## What Nora is (one paragraph)

A local macOS app that lets a researcher drive statistical analysis
(R, Stata, or Python) on their own data with a frontier model behind
the scenes, without that data leaving the machine. From the
researcher's point of view, Nora is one product they talk to — the
underlying model (Claude or GPT-5.5) is the engine, not exposed in
the UI. The model reaches the researcher's files through a narrow
six-tool MCP interface — no Bash, no filesystem, no network. Scripts
run under `sandbox-exec` with network denied and a tight
subpath-allowlist for reads; every output passes through a
disclosure-control sanitizer (SDC rules from Eurostat / UK ONS
guidance) before anything reaches the model. The researcher sees raw
R / Stata / Python output in the UI; the model only ever sees
sanitized summaries.

## Where it stands

| Layer | Status |
|---|---|
| **Data-boundary architecture** — tool interface + sandbox + sanitizer | ✅ implemented and tested |
| **Schema extraction** — `.csv`, `.tsv`, `.dta`, `.rds`, `.parquet`, `.jsonl` / `.ndjson`, four depth tiers | ✅ done; default tier `names_types_labels_summary` |
| **Executor** — `(deny default)` sandbox, R + Stata + Python, runtime libraries, per-run token, env-var allowlist | ✅ done |
| **Sanitizer** — six analysis families, SDC rules, text-safety, structural size caps | ✅ done; ~3,600 Hypothesis cases + helper-through-sanitizer end-to-end suite |
| **Result store** — SQLite, audit log, `expand_result`, per-cwd cache | ✅ done; cache rebinds on session switch |
| **Permission policy** — per-dataset schema-depth ceiling, UI dropdowns | ✅ done |
| **Filename / variable-name sanitization** at every prompt-injection surface | ✅ done |
| **Memory stack** — chat-history persisted, warm-start prefix on every fresh session, `recall_conversation` for older lookups | ✅ done |
| **Durable session state** — `.nora/session_state.json` after each turn (last exchange, recent results, datasets, **active model**) | ✅ done |
| **Per-session model memory** — restoring on session open from constructor or _set_cwd | ✅ done |
| **Multi-provider** — Anthropic (subscription or API key) + OpenAI (API key); per-provider sessions behind a `ProviderSession` interface | ✅ done |
| **Auth screen** — first-launch flow with keyring-backed credential storage; auto-promote to whichever provider is authed | ✅ done |
| **Mid-chat uploads visible to the model** — script files travel as inline context with the next message; new datasets surface as a per-turn "newly added" notice | ✅ done |
| **Files chip** — top-right popup listing scripts/graphs/logs in the session; data files surface in the Permission chip | ✅ done |
| **Image attachments** — saved to cwd, staged for vision, rendered above the user bubble, clickable lightbox | ✅ done |
| **Terminal UI** (`nora`) — Rich-based chat, `/policy` wizard | ✅ done |
| **Web UI** (`nora-ui`) — pywebview shell, sessions sidebar, theme toggle, model picker grouped by provider with $ pricing links, drag-drop file/image upload, typewriter, Lottie cat loader, status line, per-message attachment chips | ✅ done |
| **Packaging** (`.app` + `.dmg`) — bundles the web UI; .app launches pywebview with no Terminal popup; launcher logging now resilient to unwritable log dirs | ✅ done & smoke-tested locally (unsigned) |
| **Product-identity prompt rule** — model introduces itself as Nora, uses first person ("I noticed…" not "Nora flagged…") | ✅ done |
| **Real-researcher pilot** | ⏳ self-pilot in progress |
| **Cross-query composition / release ledger** | ⏭ named, future-deployment scope |
| **Apple Developer Program signing + notarization for distributable .dmg** | ⏭ blocked on $99/yr cert |
| **Stata batch wrapper around `_cons` "omitted" edge case** | ⏭ named, low-priority |

**390 tests passing.** Coverage spans SDK lockdown (Anthropic) +
OpenAI lockdown (assert tools list never grows beyond the six
function tools), schema for all six file formats, executor SBPL
profile (unit + integration gated on Rscript), Python executor
end-to-end (gated on python3 + pandas + numpy), helper-through-
sanitizer round-trips for every `from_*` emitter, sanitizer
property tests, policy, text-safety, row-count audit, stderr
isolation, per-run token authenticity, env-var allowlist
(subprocess can't see shell secrets), cross-session store
isolation, filename prompt-injection, OLS coefficient-key
constraint, CI-length constraint, structural size caps, the
memory-stack tests, set_model rollback (failed-swap leaves the
bridge on the previous id), per-session model memory,
multi-provider reconcile, Anthropic credential-delete env
cleanup, raw-log truncation (head + tail with marker), script
attachment staging + collision refusal, system-prompt-render
(`{{}}`-escape regression), and the Stop-button hard-recover.

## Running it

```bash
# Web UI — native WKWebView window. The recommended frontend.
uv run nora-ui                           # landing: drop files or pick folder
uv run nora-ui /path/to/data             # opens straight into chat

# Terminal UI — same-shell chat. Power-user / shell-only path.
uv run nora                              # opens landing prompt
uv run nora /path/to/data                # opens straight into chat

# Tests
uv run pytest -q                         # expect 390 passing

# Build the .app + .dmg locally. Bundles the web UI.
# Distribution to other people is blocked on Apple Developer Program
# signing — the unsigned .dmg trips Gatekeeper for anyone who
# didn't build it themselves.
bash packaging/build_app.sh              # → dist/Nora.app (~70 MB)
bash packaging/build_dmg.sh              # → dist/Nora.dmg (~35 MB)
open dist/Nora.app                       # smoke test the build
tail -F ~/Library/Logs/Nora/nora-*.log   # if it doesn't open
```

### Auth

Three sources, in resolution order:

1. **Anthropic subscription** — `claude` CLI signed in
   (`~/.claude.json` carries an OAuth account). Detected
   automatically; nothing to configure.
2. **API keys via Keychain** — entered through the auth screen on
   first launch. Stored under keyring service `nora` with the
   provider id (`anthropic` / `openai`) as the username. Keychain
   prompts may require Touch ID / password.
3. **Shell environment** — `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
   exported in the user's shell. Kept out of script-visible env by
   the executor's allowlist.

The bridge auto-reconciles: a fresh researcher who only configures
OpenAI gets switched to OpenAI as the active provider before the
first turn. Deleting an Anthropic credential from the auth screen
clears any injected `ANTHROPIC_API_KEY` env so `detect_auth()`
flips back to "unknown" without an app restart.

### Models

Picker (top-right of composer, grouped by provider):

| Provider | Models | Pricing link |
|---|---|---|
| Anthropic | Sonnet 4.6, Opus 4.7 | platform.claude.com/docs/…/pricing |
| OpenAI | GPT-5.5, GPT-5.5 Pro (extended reasoning) | openai.com/api/pricing |

Each row carries a small `$` link that opens the provider's pricing
page in the system browser. Models for un-authed providers stay
listed but are dimmed; clicking them opens the auth screen
positioned on that provider.

## Architecture at a glance

Three independent privacy layers. A break in any one is a bug; a
break in two at the same time is a privacy incident.

1. **Tool interface** (`src/nora/tools.py` + `provider/tool_schemas.py`)
   — the model has exactly six tools: `get_schema`, `request_data`,
   `submit_script`, `expand_result`, `list_results`,
   `recall_conversation`. Anthropic SDK built-ins (Bash, Read,
   Write, …) are disabled via `disallowed_tools` + `can_use_tool`
   catch-all + `setting_sources=[]`. OpenAI: only the six function
   tools are passed; built-ins (`web_search`, `code_interpreter`,
   `file_search`, `image_generation`, `mcp`) are explicitly
   forbidden — verified on every request by `_verify_lockdown` and
   pinned by `test_openai_lockdown.py`.
2. **Sandbox** (`src/nora/executor.py`) — `sandbox-exec` with
   `(deny default)` base, explicit subpath-allowlist for reads
   (cwd + runtime dirs + a minimal set of system paths + the
   Python interpreter's `sys.prefix` when running Python),
   tighter allowlist for writes, network denied. Refuses to run
   if `sandbox-exec` is missing rather than falling through. Per-run
   HMAC-style token embedded in every emitted payload — hand-crafted
   JSON without the token is rejected.
3. **Sanitizer** (`src/nora/sanitizer.py`, `sdc.py`,
   `text_safety.py`) — allowlist of field names per analysis
   family, SDC rules (precision clamping by N, cell suppression
   threshold 10, secondary suppression for 1-D freq tables,
   (1, 85%)-dominance for magnitude tables), text-safety pass on
   every data-origin string.

### Provider abstraction

`src/nora/provider/` wraps both providers behind a single
`ProviderSession` Protocol. The web bridge holds one session at a
time; switching provider closes and reopens. The terminal UI is
Anthropic-only for now (the multi-provider auth screen is
web-specific).

- `provider/base.py` — `ProviderSession` Protocol + Event types
  (canonical home; `chat_service.py` re-exports for back-compat).
- `provider/catalog.py` — `ModelInfo` + per-provider model lists +
  pricing URLs.
- `provider/tool_schemas.py` — single source of truth for tool
  name, description, JSON-schema input. Both providers derive
  from here; consistency test asserts the SDK decorations agree.
- `provider/anthropic.py` — wraps `ClaudeSDKClient`. Translates
  SDK message blocks into provider-neutral events. Holds the
  `_DISALLOWED_BUILTINS` list and `_gate_tool_use` catch-all.
- `provider/openai.py` — wraps the Responses API. Maintains
  `_input` across turns (OpenAI conversation = our session). Tool
  loop dispatches via `nora.tools.HANDLERS` so behaviour is
  byte-for-byte identical regardless of which model called the
  tool.

### Mid-chat awareness

- **New datasets** dropped after session open: the bridge
  snapshots dataset names at session-open time and diffs against
  current on every turn. New names get a one-line "the researcher
  added these mid-session" notice prepended to the next prompt.
- **Script files** (.py / .do / .r / .rmd) dragged into the
  composer: the bridge stages contents in
  `_pending_script_attachments` and prepends a fenced code block
  to the next prompt. Persistent transcript chip below the user
  bubble (📎 regression.py) confirms the upload landed; the chip
  × button calls `unstage_attachment` so dismissing actually
  removes the inline content (no privacy mismatch).
- **Images**: saved to cwd alongside vision staging; rendered
  above the user bubble at thumbnail size; click → full-viewport
  lightbox.

### Session model

`nora-ui` without an argv opens a landing screen; dropped /
picked files land in `~/.nora-sessions/<ts>_<id>/` which becomes
the cwd. That dir is spaces-free (Stata-safe), outside cloud-sync
roots, persistent across restarts. Reopening a session restores
the `active_model` recorded in `.nora/session_state.json` — a
researcher who switched to Opus for one project comes back to Opus
next time, even when the launcher passes the cwd directly.

## Known-real, deferred

The actual near-term blocker is one item:

- **`.app` / `.dmg` distribution to other people.** The local build
  works (`.app` launches the web UI directly via pywebview, no
  Terminal popup). What's missing is an Apple Developer Program
  signature — without it, anyone you hand the .dmg to hits a hard
  Gatekeeper warning and most users won't get past it. Right-click →
  Open / `xattr -cr` workarounds are documented in install.md but
  aren't acceptable for a "just install this" handoff. Cost: $99/yr.
  Once signed, also worth notarizing for the cleanest first-launch
  experience.

## Longer-term: governance for wider distribution

These don't bite while Nora is being run by researchers on their
own data with their own API keys (the current and intended
near-term mode). They bite when the product moves into anything
resembling shared deployment — multi-tenant access, researchers
analysing data they don't own, regulated datasets where the threat
model includes adaptive probing. Out of scope for the current
pilot, called out here so a future maintainer doesn't rediscover
them by surprise.

- **Cumulative-inference / cross-query composition.** Single-query
  SDC is tight; the joint distribution across many queries isn't
  bounded. This is the "20 questions" attack — inherent to every
  interactive analysis system, not Nora-specific. The store
  already holds every sanitized emission, so a release-ledger
  feature has raw material to build on. When it's time, see
  `docs/direction.md` §"Known-real, design-pending" for the DP /
  τ-ARGUS / release-ledger options and why naïve query counters
  are worse than nothing. Not the right thing to spend time on
  before the product has demonstrated value with one or more
  outside pilots.
- **Bounded covert channels in regression metadata.** Documented
  in `sanitizer.py` (`predictor_variables` is script-authored;
  `nora$result(...)` allows hand-crafted payloads). Bandwidth is
  small (~hundred bytes per regression); same threat-model
  scenario as cumulative-inference above. Out of scope for
  self-pilots.
- (Other deployment-level concerns land here as they surface —
  multi-tenant key management, per-organisation policy
  enforcement, audit-log retention, etc. None are real today.)

## Rough edges (work, but annoy)

- Transformations log shows `"dropped unknown/forbidden field
  'label'"` on every R `submit_script` because the
  `nora$from_lm(m, label=…)` arg is stripped by the schema
  allowlist. Harmless (the `submit_script` MCP-tool label is
  stored separately), but noisy. Fix: widen the per-type string
  allowlist to include `label`.
- Raw-log panel cap is 32 KB per stream with head + tail
  preservation and a `[… N bytes truncated from the middle …]`
  marker. Sufficient for regression tables and helper summaries;
  larger logs survive on disk for `tail -F`.
- Sanitizer drops the `_cons` coefficient as empty when Stata
  reports it as "omitted" (perfect-fit edge case) — generated
  JSON becomes malformed. Only triggers on degenerate toy data;
  flagged in a test comment. Not a production blocker.
- OpenAI `previous_response_id` is not used; the bridge replays
  the conversation via its own `_input` array and prepends its
  context prefix on every fresh session open. Server-side
  resumption could trim per-turn payload size on long
  conversations — out of scope for now.

## Key files (reading order)

| File | What's there |
|---|---|
| `src/nora/system_prompt.py` | The model's full system prompt + dataset listing. Single source of truth for both providers |
| `src/nora/tools.py` | The six MCP tools. Start here to understand the model's surface |
| `src/nora/provider/__init__.py` + `base.py` | `ProviderSession` Protocol, Event types, `open_session(provider, …)` factory |
| `src/nora/provider/anthropic.py` + `openai.py` | Per-provider session implementations |
| `src/nora/provider/catalog.py` | Model registry + pricing URLs |
| `src/nora/provider/tool_schemas.py` | Provider-neutral tool schema source-of-truth |
| `src/nora/auth.py` | Keyring-backed credential storage |
| `src/nora/executor.py` | Sandbox profile, R/Stata/Python subprocess plumbing, per-run token |
| `src/nora/sanitizer.py` + `sdc.py` | The SDC allowlist and clamp/suppress primitives |
| `src/nora/runtime/nora.R` + `nora.py` + `nora_result_*.ado` | Runtime emitters scripts call to surface results |
| `src/nora/schema.py` | Schema extractors for all six supported file formats |
| `src/nora/ui.py` | Web UI bridge: auth, sessions, attachments, model picker, file panel |
| `src/nora/app.py` | Terminal entry point, chat loop, rendering |
| `src/nora/chat_service.py` | Back-compat re-export shim for the Event types |
| `src/nora/chat_history.py` | Turn-grouped reader; warm-start prefix renderer |
| `src/nora/session_state.py` | Atomic writer / reader for `.nora/session_state.json` (carries `active_model` for per-session memory) |
| `src/nora/web/{index.html,app.js,markdown.js,style.css}` | Web frontend |
| `src/nora/__main_ui__.py` | Bundle entry — calls `nora.ui:main`. The .app launches this, NOT the terminal CLI |
| `docs/direction.md` | Long-form architectural doc; open questions |
| `docs/overview.md` | Plain-language description for researchers |
| `docs/install.md` | Researcher-facing install flow |
| `docs/verification.md` | Manual smoke-test recipes (incl. Stata, which CI can't) |
| `tests/` | 390 tests. `test_sanitizer.py` is the property-test backbone; `test_python_runtime_sanitizer.py` covers helper-through-sanitizer for the new Python emitters; `test_openai_lockdown.py` pins the no-built-in-tools invariant on the new provider; `test_bridge_correctness.py` is the regression home for the four reviewer-flagged P1s |

## Decisions worth not re-litigating

- **No plan-submission / DSL-grammar pivot.** A reviewer proposed
  replacing `submit_script` with a constrained plan grammar +
  local-LLM translator. Rejected as over-correction — the privacy
  guarantee already rests on tool interface + sandbox + sanitizer,
  none of which depend on who authored the code. See
  `docs/direction.md` "The decision."
- **No bundled local LLM.** Install footprint (~15 GB), worse
  code generation than frontier models, no privacy gain given the
  existing stack. Re-opens only when a specific use case demands
  it (stderr-based repair, text-data redaction).
- **Multi-provider, not local model.** Anthropic + OpenAI both go
  through the same six-tool MCP surface; the privacy layers
  (sandbox + sanitizer) don't depend on which provider authored
  the call. Adding a third provider means writing a new
  `provider/foo.py` and one lockdown test; the SDC and sandbox
  layers are untouched.
- **OpenAI: API key only.** ChatGPT subscription doesn't carry
  programmatic API access; mixing browser-OAuth with API-key
  flows in one input field is a UX trap. If/when OpenAI adds a
  subscription-included API tier, revisit.
- **macOS-only for now.** Sandbox relies on `sandbox-exec`. Linux
  would need `bubblewrap`/`nsjail`, Windows is on nobody's path.
- **Schema policy is a ceiling, not a fixed value.** The model
  can request any depth ≤ ceiling. Researchers edit via the
  Permission chip (web) or `/policy` slash-command (terminal).
- **Data files in `Permission`, scripts/graphs/logs in `Files`.**
  Two surfaces, no duplication: data files have schema-depth
  policy attached, scripts/graphs/logs don't. Listing data in
  both creates two views that drift.
- **Stata batch `-b do <path>` breaks on spaces.** Executor
  passes `script.do` (bare filename) with subprocess cwd set to
  the run dir. Don't "improve" back to an absolute path.
- **Python uses system `python3`, not bundled.** Researcher needs
  `pandas` + `numpy` for the runtime to load (executor refuses
  with a `pip install` hint otherwise); `statsmodels` and
  `scipy` are needed only by `from_lm` / `from_t_test`
  respectively, so descriptive scripts work without them.
- **The product is Nora; the model is Claude or GPT-5.5.** Don't
  expose model names in user-facing copy. The system prompt has
  an explicit identity rule telling the model to introduce
  itself as Nora and to use first person ("I noticed…" not
  "Nora flagged…").
- **GitHub repo is still named `builder`** (URL:
  github.com/junishka/builder). Renaming a GitHub repo is an
  out-of-band action; URLs in install instructions still point
  there. The local clone gets renamed at `git clone … nora` time
  so the on-disk dir matches the product name.

## Next concrete step

If you're picking this up to finish and ship it, in this order:

1. **Run it yourself on your own data.** Catch the UX rough
   edges before anyone else sees them.
2. **Sign and notarize the .app.** The build pipeline already
   produces a working .app that launches the web UI directly with
   no Terminal popup. The blocker for handing it to colleagues is
   the missing Apple Developer Program signature. $99/yr.
3. **Try it on a colleague's data** (or yours via a colleague).
   The difference between "self-pilot" and "someone who didn't
   build it" is where most real UX bugs live.
4. **Distribution-mode concerns (cumulative-inference /
   release-ledger, multi-tenant policy, audit retention) wait
   until distribution itself is real.** They're meaningful work
   for a future deployment, not for the current self-pilot mode.
   See "Longer-term: governance for wider distribution" above.

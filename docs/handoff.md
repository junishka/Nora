# Nora — handoff

Single-page entry point for picking this project up. Last
substantive update **2026-04-29**, after the token-budget pass
(OpenAI `previous_response_id`, 1h Anthropic cache TTL, minified
tool JSON, leaner system prompt), per-provider system prompt +
lean OpenAI tool descriptions, regression diagnostics (vif /
condition_number / full vcov), self-contained Stata helpers
(`nora_result_sum`, `nora_ttest`), correlation_matrix sanitizer
type, two new `request_data` types (quartiles, correlation_pair),
env-gated cross-session result recall, per-variable min/max opt-in
via dataset policy, and the Stata residual-plot scaling fix. If
something here disagrees with the code, trust the code and file a
patch to this doc.

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
sanitized summaries — including plots, where a small set of opinionated
helpers (residuals, predicted-response curves, coefficient forest
plots, estimate comparisons) produce model-output figures that cross
the boundary; raw-data plots stay researcher-only by construction.
Multiple sessions run concurrently — switching the visible chat in
the sidebar is a pure focus change; long jobs in unfocused sessions
keep streaming.

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
| **Concurrent-session execution** — bridge holds `dict[cwd → SessionRunner]`; switching focus is a pure UI change, in-flight turns keep streaming in unfocused sessions, sidebar shows a busy dot per active runner | ✅ done |
| **Per-task cwd via ContextVar** — `nora.config.use_cwd` makes tool execution sandbox-safe under concurrent runners; sister tasks see their own cwd | ✅ done |
| **Plot vision (model-output only)** — runtime helpers `plot_residuals` / `plot_interaction` / `plot_coefficients` / `plot_estimate_comparison` for R, Python, Stata. Manifest-allowlisted: only files produced by these helpers cross to the model on the next turn. Raw-data plots stay researcher-only by construction | ✅ done |
| **Stata export reliability** — `_nora_export_plot` tries PDF → PNG → EPS → `.gph` so a missing `Graph2png` translator doesn't kill the script. Bridge converts PDF/EPS → PNG via `sips` for both researcher thumbnails and model-vision attachments. `nora_safe_export` is the same fallback chain for ad-hoc (non-helper) exports | ✅ done |
| **Helper-failure visibility** — plot helpers append to `_nora_plots/helper_errors.jsonl` on failure; `submit_script` summarizes succeeded + failed in the response so the model sees "matplotlib not installed; pip install matplotlib" instead of guessing | ✅ done |
| **Runtime environment in system prompt** — `env_detect` probes R packages (haven, ggplot2) and Python packages (matplotlib + the four required ones); the prompt renders a `(haven: ✓, ggplot2: ✗)` block so the model picks a language by what's actually installed | ✅ done |
| **Files chip** — top-right popup listing graphs/scripts/logs (graphs first); rows now have `[copy/send/open] [title/thumbnail] [×]`; image rows render PDF/EPS via sips sidecars; click-to-lightbox uses 96vw/96vh for sharp viewing | ✅ done |
| **delete_session_file** — Files-panel `×` deletes any file inside the session cwd, removes its PDF→PNG sidecar, drops matching pending-attachment chips | ✅ done |
| **Image attachments** — saved to cwd, staged for vision, rendered above the user bubble, clickable lightbox; persistent chip under the user bubble matches accent styling so a script attachment reads as obviously as an image thumbnail | ✅ done |
| **Cache-busted JS/CSS** — `_materialize_cache_busted_index` writes a per-launch `index.bust-<hash>.html` so WKWebView reloads frontend assets instead of serving stale cached versions on Python restart | ✅ done |
| **Terminal UI** (`nora`) — Rich-based chat, `/policy` wizard | ✅ done |
| **Web UI** (`nora-ui`) — pywebview shell, sessions sidebar, theme toggle, model picker grouped by provider with $ pricing links, drag-drop file/image upload, typewriter, Lottie cat loader, status line, per-message attachment chips, topbar visually integrated with chat surface | ✅ done |
| **Packaging** (`.app` + `.dmg`) — bundles the web UI; .app launches pywebview with no Terminal popup; launcher logging now resilient to unwritable log dirs | ✅ done & smoke-tested locally (unsigned) |
| **Product-identity prompt rule** — model introduces itself as Nora, uses first person ("I noticed…" not "Nora flagged…") | ✅ done |
| **Token-budget pass** — Anthropic 1h prompt-cache TTL via `ENABLE_PROMPT_CACHING_1H`; OpenAI uses `previous_response_id` so per-turn input is just the new content; tool-result JSON minified (~25-35% off every payload); warm-start prefix tightened (5k → 2.7k tokens on resume); `turn_done` events now persisted for cache-rate diagnostics; system-prompt content trim + STAGE NOTE deletion + em-dash dedupes | ✅ done |
| **Per-provider system prompt + lean OpenAI tool descriptions** — `build_system_prompt(cwd, server_name, provider)`; OpenAI gets a name-only tool intro instead of the Anthropic `mcp__nora__` mention; `ToolSpec.openai_description` field for the four biggest tools (recall_conversation, read_attached_file, submit_script, get_schema) cuts tool-array tokens by ~46% on the OpenAI path. Saves ~818 tokens/call, biggest wins compound across 100+ turn sessions | ✅ done |
| **Regression diagnostics** — `vif`, `condition_number`, full `vcov` (variance-covariance matrix) emitted by `from_lm` in R + Python when the design matrix is reachable. Pure aggregates from sigma² · (X'X)⁻¹; cross-field key validation mirrors the existing coefficient defense. Plus a long-standing bug fix: `Intercept` and `const` were silently dropped from statsmodels formula-fit payloads (only `(Intercept)` / `_cons` / `intercept` were in the alias list); now in | ✅ done |
| **Self-contained Stata helpers** — `nora_result_sum varname [if]` runs `summarize` itself instead of reading whatever's in `r()`. Eliminates the silent foot-gun where a second `summarize <other>` between intent and helper produced a payload labeled "age" carrying income's mean. Same for new `nora_ttest <var> [if] [, against(num) \| paired(var2) \| by(group) [unequal]]` which runs the appropriate `ttest` form itself based on mutually-exclusive shape options. Legacy `nora_result_ttest` kept for back-compat | ✅ done |
| **`correlation_matrix` sanitizer type** — pairwise correlation matrix as a first-class payload (Pearson / Spearman / Kendall), with min-N gate and per-pair value-key validation. R `nora$from_correlation` and Python `nora.from_correlation` helpers; computes complete-case N (not pairwise N) so off-diagonals draw on the same sample | ✅ done |
| **`request_data` types** — added `quartiles` (25th + 75th + IQR; median omitted as a row-level forbidden field) and `correlation_pair` (Pearson r between two variables, complete-case N). Tool schema gained an optional `variable2` field for multi-variable types | ✅ done |
| **Cross-session result recall** — `list_results_global(query?)` and `expand_result(result_id, session_path?)`. Env-gated via `NORA_ALLOW_CROSS_SESSION_RECALL=1` (default off — researcher-side project separation, NOT a privacy property; stored payloads are pre-sanitized either way). Path-confined to `~/.nora-sessions/` so prompt-injected lookups can't direct the store loader at arbitrary paths | ✅ done |
| **Per-variable min/max opt-in** — `DatasetPolicy.non_disclosive_variables` list in `.nora/policy.json`. Variables on the list (typical: `age`, `year_of_birth`, `education_years`) get `min_value` / `max_value` through descriptive payloads. Default empty: every variable's extremes still suppressed unless explicitly opted in | ✅ done |
| **Stata residual-plot scaling fix** — `nora_plot_residuals` now samples to 5000 points before `rvfplot + graph export ... as(pdf)`. PDF embeds every point as a vector path, so unsampled rendering scaled linearly with N: 200k rows took ~8s, almost entirely PDF rendering. Sampled output reduces that to ~750ms with no loss of pattern visibility. `e()` is unaffected so the downstream `nora_result_regress` still sees the full-N regression | ✅ done |
| **Allow deleting the active session** — sidebar × on the focused row now works; the bridge clears `self.cwd`, returns `was_active=True`, and the page navigates back to the landing screen. Confirm dialog adapts ("Delete the session you're currently in") | ✅ done |
| **`debug_excerpt` on script failure** — system prompt now tells the model to read it; long-standing feature was effectively invisible because the prompt didn't acknowledge it | ✅ done |
| **Real-researcher pilot** | ⏳ self-pilot in progress |
| **Cross-query composition / release ledger** | ⏭ named, future-deployment scope |
| **Apple Developer Program signing + notarization for distributable .dmg** | ⏭ blocked on $99/yr cert |
| **Stata batch wrapper around `_cons` "omitted" edge case** | ⏭ named, low-priority |

**610 tests passing.** Coverage spans SDK lockdown (Anthropic) +
OpenAI lockdown, schema for all six file formats, executor SBPL
profile, Python executor end-to-end, helper-through-sanitizer
round-trips for every `from_*` emitter, sanitizer property tests,
policy, text-safety, row-count audit, stderr isolation, per-run
token authenticity, env-var allowlist, cross-session store
isolation, filename prompt-injection, OLS / CI / structural-size
constraints, memory stack, set_model rollback, per-session model
memory, multi-provider reconcile, raw-log truncation, script
attachment staging + collision refusal, system-prompt-render,
Stop-button hard-recover — plus the new suites for **concurrent
sessions** (`test_concurrent_sessions.py`: ContextVar isolation
under concurrent asyncio tasks, two runners observing only their
own cwd, switch keeps the previous runner alive, persistence
routes by event `session_cwd`, Stop only cancels the active
runner), **plot vision** (`test_plot_vision.py`: manifest-only
allowlist, kind allowlist, path-traversal refusal, byte cap,
end-to-end capture → next-turn attachment, cancel restores
pending plots), and **plot rendering / Stata export reliability**
(`test_run_dir_plots.py`: thumbnail collector, helper diagnostic,
`png_for` PDF→PNG conversion via `sips`, helper-error
surfacing in the model-visible tool result, `_nora_export_plot`
PDF→PNG→EPS→.gph fallback order, `nora_safe_export` doesn't write
a manifest entry, runtime environment block renders in the prompt
with `✓`/`✗` package status).

## Running it

```bash
# Web UI — native WKWebView window. The recommended frontend.
uv run nora-ui                           # landing: drop files or pick folder
uv run nora-ui /path/to/data             # opens straight into chat

# Terminal UI — same-shell chat. Power-user / shell-only path.
uv run nora                              # opens landing prompt
uv run nora /path/to/data                # opens straight into chat

# Tests
uv run pytest -q                         # expect 477 passing

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
`ProviderSession` Protocol. The bridge holds **one runner per
focused-or-recently-focused session** (`dict[str, SessionRunner]`
keyed by cwd) — each runner owns its own ProviderSession plus
its own asyncio lock and turn task, so two researchers' worth
of in-flight chats can run concurrently without trampling each
other. Switching provider for a given runner closes and reopens
that runner's session; OTHER runners are untouched. The terminal
UI is Anthropic-only for now (the multi-provider auth screen is
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

### Concurrent-session execution

`src/nora/runner.py` defines `SessionRunner` — the per-cwd
execution unit. Each runner holds: its `cwd`, an asyncio
`_send_lock` (so a second send_message in the same session queues
behind the first; sends to OTHER sessions proceed in parallel),
the current `ProviderSession`, the `_current_turn_task` (so
`Stop` cancels only this runner), `needs_context_prefix`, the
active `model`/`provider`, and any pending script attachments.
`run_turn` enters `nora.config.use_cwd(self.cwd)` so tool
handlers (and any sub-tasks the SDK spawns) read THIS runner's
cwd via the ContextVar — sister tasks see their own cwd, no
trampling. Every event is stamped with `session_cwd` so
persistence routes to the correct `chat_history.jsonl` regardless
of which session the UI happens to be focused on at emit time.

### Plot vision (model-output only)

Three layers, in priority order:

1. **Allowlisted runtime helpers.** R / Python / Stata each ship
   `plot_residuals`, `plot_interaction`, `plot_coefficients`,
   `plot_estimate_comparison`. Each takes a fitted-model object as
   input and produces a canonical visualization from
   model outputs (residuals, predictions, coefficients) — never
   from the raw rows. There is no escape hatch that registers an
   arbitrary file: that would let a histogram of raw observations
   pose as a "coefficient plot" via self-attestation. The kind
   list (`residuals` / `interaction` / `coefficients` /
   `marginal_effects`) is enforced both by the helpers and by the
   runner.
2. **Manifest-gated capture.** Each helper writes its PNG/PDF/EPS
   into `<run_dir>/_nora_plots/` and appends a JSON line to
   `_nora_plots/manifest.jsonl`. The runner reads ONLY the manifest
   after every `submit_script` — files in the dir without a
   manifest entry stay invisible to the model.
3. **Format fallback + bridge conversion.** Stata's PNG export
   needs the `Graph2png` translator (often missing on macOS), so
   `_nora_export_plot.ado` tries `as(pdf)` → `as(png)` → `as(eps)`
   → `graph save .gph` and registers whichever wins. The bridge's
   `nora.plot_convert.png_for` rasterizes PDF/EPS to a sibling
   `.nora.png` via `sips` (mtime-cached). Same path for both
   model-vision attachment and researcher chat thumbnails.

Plot helper failures append to `_nora_plots/helper_errors.jsonl`
with `{helper, step, error, message, fix}` — `submit_script`'s
response includes a `plots: {succeeded, failed, note}` summary so
the model SEES "matplotlib not installed; pip install matplotlib"
and can react instead of guessing "thumbnail should be visible".

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
| `src/nora/config.py` | Process default cwd + per-asyncio-task `use_cwd` ContextVar — the gate that makes concurrent runners sandbox-safe |
| `src/nora/executor.py` | Sandbox profile, R/Stata/Python subprocess plumbing, per-run token |
| `src/nora/env_detect.py` | Probes installed runtimes + optional packages (haven, ggplot2, matplotlib) so the prompt advertises what's available |
| `src/nora/sanitizer.py` + `sdc.py` | The SDC allowlist and clamp/suppress primitives |
| `src/nora/runner.py` | `SessionRunner` — per-cwd execution unit. Owns provider session, lock, turn task, plot-vision capture, helper-error logging |
| `src/nora/plot_convert.py` | macOS `sips`-based PDF/EPS → PNG conversion with mtime-cached sidecars. Used by both the runner (model vision) and the bridge (researcher thumbnails) |
| `src/nora/runtime/nora.R` + `nora.py` + `nora_result_*.ado` + `nora_plot_*.ado` + `_nora_export_plot.ado` + `nora_safe_export.ado` | Runtime emitters: result helpers (`from_lm`, `from_t_test`, …) and plot helpers (`plot_residuals`, `plot_interaction`, `plot_coefficients`, `plot_estimate_comparison`). Stata fallback chain in `_nora_export_plot`; Stata ad-hoc safe wrapper in `nora_safe_export` |
| `src/nora/schema.py` | Schema extractors for all six supported file formats |
| `src/nora/ui.py` | Web UI bridge: runners dict, focus-only `switch_session`, plot collection + diagnostic, Files panel endpoints, cache-busted index.html, `delete_session_file` |
| `src/nora/app.py` | Terminal entry point, chat loop, rendering |
| `src/nora/chat_service.py` | Back-compat re-export shim for the Event types |
| `src/nora/chat_history.py` | Turn-grouped reader; warm-start prefix renderer |
| `src/nora/session_state.py` | Atomic writer / reader for `.nora/session_state.json` (carries `active_model` for per-session memory) |
| `src/nora/web/{index.html,app.js,markdown.js,style.css}` | Web frontend; `app.js` holds the per-session focus state + Files-panel rendering |
| `src/nora/__main_ui__.py` | Bundle entry — calls `nora.ui:main`. The .app launches this, NOT the terminal CLI |
| `docs/direction.md` | Long-form architectural doc; open questions |
| `docs/overview.md` | Plain-language description for researchers |
| `docs/install.md` | Researcher-facing install flow |
| `docs/verification.md` | Manual smoke-test recipes (incl. Stata, which CI can't) |
| `tests/` | 610 tests. `test_sanitizer.py` is the property-test backbone (now also covers vif / condition_number / vcov + correlation_matrix); `test_concurrent_sessions.py` pins the per-task ContextVar isolation; `test_plot_vision.py` pins the manifest-allowlist privacy gate; `test_run_dir_plots.py` covers thumbnail collection + PDF→PNG conversion + Stata export fallback chain; `test_openai_lockdown.py` pins the no-built-in-tools invariant; `test_cross_session_recall.py` pins the env-gated cross-session lookup + path-confinement defense; `test_bridge_lifecycle.py` covers active-session delete + landing-page navigation |

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
- **Plot vision is helper-allowlist, not file-allowlist.** A
  `register_plot(file, kind)` API was tried and removed — the
  kind label was self-attested by the script, so a histogram
  could pose as a `coefficients` plot and slip past the
  privacy line. Replacement: the only paths that surface a
  plot to the model are `plot_residuals`, `plot_interaction`,
  `plot_coefficients`, `plot_estimate_comparison`, each of which
  takes a fitted-model object and produces a canonical
  visualization from model outputs. Bespoke plots stay
  researcher-only. Don't reintroduce the file-based escape
  hatch — every gain in flexibility is a privacy-line break.
- **Switching sessions is a pure UI focus change.** The bridge
  holds runners by cwd; switching does NOT close any runner's
  SDK client. Closing on switch was the multi-session bug —
  the in-flight `receive_response()` raised mid-stream and
  surfaced as a fail bubble. Don't "helpfully tear down on
  switch" in a future refactor; the test suite catches that
  regression.
- **Per-task cwd via ContextVar, not process-global.** Tool
  handlers MUST resolve cwd through `nora.config.get_cwd()`,
  which reads the per-task ContextVar. The process-global default
  exists only for the terminal CLI / startup. Any new code that
  reaches around the ContextVar (e.g., reads a stashed cwd
  from somewhere else) breaks concurrent-runner isolation.
- **Don't carry forward script attachments on cancel/error.** An
  earlier version restored `pending_script_attachments` on the
  bridge after a failed turn, but the JS chip cleared at send
  time and the two sides drifted (the "X is already attached"
  toast for files no chip showed). If a turn fails, the user
  re-attaches. The other carry-forwards (context prefix,
  dataset diff, captured plots) stay because they don't have a
  JS chip representation that could disagree.
- **Stata `as(png)` is unreliable; export fallback chain is
  mandatory.** macOS Stata installs frequently lack the
  `Graph2png` translator; bare `graph export "x.png"` aborts the
  do-file before `nora_result_*` runs, which loses both the plot
  AND the structured result. Every Nora plot helper goes
  through `_nora_export_plot` (PDF → PNG → EPS → `.gph`); the
  `nora_safe_export` wrapper handles ad-hoc exports outside
  helpers. Don't add a new helper that calls `graph export`
  directly.
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

# Nora — handoff

Single-page entry point for picking this project up. Last
substantive update **2026-08-20**, covering the 0.11.x releases
(0.11.0 / 0.11.1 / 0.11.2, each signed and notarized).

The 0.11.x batch in brief. The model picker moved to the current
families: Sonnet 5 (default) / Opus 5 / Fable 5 on Anthropic, and
GPT-5.6 Terra / Sol (default) on OpenAI. Each new default costs
the same per token as the model it replaced. Reasoning effort,
previously fixed at `xhigh` internally, is now a per-session
setting in the model popup. Each provider's bar shows only the
levels it supports — `low` through `max` on Anthropic, `low`
through `pro` on OpenAI — the choice is saved in
`session_state.json` and restored on reopen, and switching
providers keeps you at an equivalent level. One asymmetry worth
knowing: OpenAI applies an effort change on the next request,
while Anthropic must restart its session process to apply it (the
conversation carries across via the warm-start prefix, the same
path a provider switch takes). How the `pro` rung maps onto
OpenAI's API lives in `provider/catalog.py` and
`provider/openai.py::_reasoning_params`. Smaller fixes rode
along: the Python sandbox now grants a virtualenv's base
prefixes, so uv-managed interpreters no longer die on startup;
the model popup opens leftward so it fits on screen; the
transcript follows new replies only when you are already at the
bottom, instead of yanking the view mid-read; and messages reveal
with a staggered fade rather than appearing all at once.

The previous substantive update (**2026-05-16**) was the 0.10.0
Stata-parity pass:
`nora_result_regress` now handles `mixed` / `meglm` fits
(reads `estat recovariance` for variance components, `estat icc`
for the single-grouping intercept-only case, restricts the
coefficient submatrix to the fixed-effects equation via `e(k_f)`);
new `nora_result_cluster.ado` covers `cluster kmeans` /
`cluster wardslinkage` / `cluster completelinkage` /
`cluster averagelinkage` / `cluster singlelinkage` with centroids
+ within-SS + the SS decomposition computed from the dataset
directly (Stata's clustering commands don't store centroids
natively); new `nora_result_factor.ado` covers `pca` and `factor`
(pcf / pf / ml / ipf extraction) reading loadings + eigenvalues +
explained-variance ratios from `e()`, plus ML-FA goodness-of-fit
fields. The Kaplan-Meier helper was physically present under
0.9.x but never wired into `_stage_runtime_library`'s `stata_ados`
tuple — calls to `nora_result_km` failed with "command unknown" at
runtime; fixed by adding the file to the staging tuple and the
`capture program drop` shadowing-defense list, and the
disk↔staging invariant is now pinned by a new test
(`test_every_runtime_helper_file_is_in_executor_staging_lists`)
so a future helper added to `src/nora/runtime/` without staging
fails the test. The DiD Stata path is reframed in the system
prompt as no-realistic-workflow (recommend R / Python in the same
session loading the `.dta` via `haven` / `pyreadstat`); RDD Stata
was targeted for 0.10.3 with a documented cross-language numerics
verification protocol (see [CHANGELOG.md](../CHANGELOG.md)
deferred section). The previous substantive batch (2026-05-15)
was the audit-fixes pass:
`install_packages` consent is modal-only (the system prompt and
tool description now agree that the Approve/Deny modal is the one
gate); `error_summary.py` is documented at its current stricter
redaction posture (exception bodies redacted wholesale, only
parser-anchored framing survives); schema fast paths use a shared
header peek so `load_data` and the names_only payload agree on
headerless CSV/TSV; `_names_only_payload` distinguishes
"row_count unknown" (`observation_count: null`) from "empty
dataset"; JSONL `names_only` unions keys across the file instead
of reading only the first record; malformed per-dataset policy
entries now clamp to the strictest tier instead of falling open;
`chat_history`'s lightweight readers honour their "never raises"
contract through concurrent-delete races; cancelled `submit_script`
runs raise `CancelledError` instead of returning a model-visible
"status: cancelled" tool result; `delete_credential` uses a fresh
keyring read so an external-Keychain deletion is recognised as
"already absent" rather than a false "delete failed"; session
focus switch no longer wipes captured plot images; the loop-
shutdown race that could leave a runner permanently busy is
fixed; pywebview's deprecated `OPEN_DIALOG` / `FOLDER_DIALOG`
are replaced by `FileDialog.OPEN` / `FileDialog.FOLDER`; and
`_materialize_cache_busted_index` routes its output to a temp
directory (with an injected `<base href>`) when `web_dir` is
read-only, so the cache-bust write no longer modifies the
codesigned bundle (clean-install Gatekeeper now passes).
The previous substantive batch (2026-05-12) was the release-
readiness / documentation alignment pass: multi-provider framing,
unified install docs, fourteen-tool surface including
`install_packages`, signed + notarized `.dmg`, canonical
`junishka/Nora` URLs, refreshed test counts. If something here
disagrees with the code, trust the code and file a patch to this
doc.

## What Nora is (one paragraph)

A local macOS app that lets a researcher drive statistical analysis
(R, Stata, or Python) on their own data with a frontier model behind
the scenes, without that data leaving the machine. From the
researcher's point of view, Nora is one product they talk to — the
underlying provider model is the engine, not exposed as the product.
The model reaches the researcher's files through a narrow
fourteen-tool MCP interface — no Bash, no filesystem, no network.
Scripts run under `sandbox-exec` with network denied and a tight
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
| **Sanitizer** — supported analysis families, SDC rules, text-safety, structural size caps | ✅ done; Hypothesis cases + helper-through-sanitizer end-to-end suite |
| **Result store** — SQLite, audit log, `expand_result`, per-cwd cache | ✅ done; cache rebinds on session switch |
| **Permission policy** — per-dataset schema-depth ceiling, UI dropdowns | ✅ done |
| **Filename / variable-name sanitization** at every prompt-injection surface | ✅ done |
| **Memory stack** — chat-history persisted, warm-start prefix on every fresh session, `recall_conversation` for older lookups | ✅ done |
| **Durable session state** — `.nora/session_state.json` after each turn (last exchange, recent results, datasets, **active model**) | ✅ done |
| **Per-session model memory** — restoring on session open from constructor or _set_cwd | ✅ done |
| **Per-session reasoning effort** — provider-specific ladder in the model popup (`low`…`max` Anthropic, `low`…`pro` OpenAI); persisted as `active_effort` and restored independently of the model; Anthropic re-warms the session on change, OpenAI applies per request | ✅ done (0.11.0) |
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
| **Cache-busted JS/CSS** — `_materialize_cache_busted_index` writes a per-launch `.index.bust-<hash>.html` so WKWebView reloads frontend assets instead of serving stale cached versions on Python restart. When `web_dir` isn't writable (packaged `.app` bundle, where `Resources/` is sealed by codesign), the bust file lands in a process-private temp directory with an injected `<base href>` so relative asset refs still resolve back into the bundle — preserves cache-busting in both dev and packaged builds without modifying the signed bundle | ✅ done |
| **Frontend** (`nora`) — pywebview shell, sessions sidebar, theme toggle, model picker grouped by provider with $ pricing links, drag-drop file/image upload, Lottie cat loader, status line, per-message attachment chips, topbar visually integrated with chat surface | ✅ done |
| **Packaging** (`.app` + `.dmg`) — bundles the web UI; .app launches pywebview with no Terminal popup; release `.dmg` is signed and notarized; launcher logging now resilient to unwritable log dirs | ✅ done |
| **Product-identity prompt rule** — model introduces itself as Nora, uses first person ("I noticed…" not "Nora flagged…") | ✅ done |
| **Token-budget pass** — Anthropic 1h prompt-cache TTL via `ENABLE_PROMPT_CACHING_1H`; OpenAI uses `previous_response_id` so per-turn input is just the new content; tool-result JSON minified (~25-35% off every payload); warm-start prefix tightened (5k → 2.7k tokens on resume); `turn_done` events now persisted for cache-rate diagnostics; system-prompt content trim + STAGE NOTE deletion + em-dash dedupes | ✅ done |
| **Per-provider system prompt + lean OpenAI tool descriptions** — `build_system_prompt(cwd, server_name, provider)`; OpenAI gets a name-only tool intro instead of the Anthropic `mcp__nora__` mention; `ToolSpec.openai_description` field for the four biggest tools (recall_conversation, read_attached_file, submit_script, get_schema) cuts tool-array tokens by ~46% on the OpenAI path. Saves ~818 tokens/call, biggest wins compound across 100+ turn sessions | ✅ done |
| **Regression diagnostics** — `vif`, `condition_number`, full `vcov` (variance-covariance matrix) emitted by `from_lm` in R + Python when the design matrix is reachable. Pure aggregates from sigma² · (X'X)⁻¹; cross-field key validation mirrors the existing coefficient defense. Plus a long-standing bug fix: `Intercept` and `const` were silently dropped from statsmodels formula-fit payloads (only `(Intercept)` / `_cons` / `intercept` were in the alias list); now in | ✅ done |
| **Self-contained Stata helpers** — `nora_result_sum varname [if]` runs `summarize` itself instead of reading whatever's in `r()`. Eliminates the silent foot-gun where a second `summarize <other>` between intent and helper produced a payload labeled "age" carrying income's mean. Same for new `nora_ttest <var> [if] [, against(num) \| paired(var2) \| by(group) [unequal]]` which runs the appropriate `ttest` form itself based on mutually-exclusive shape options. Legacy `nora_result_ttest` kept for back-compat | ✅ done |
| **Stata parity for the four high-usage shapes that were previously R+Python-only** (0.10.0): mixed-effects through `nora_result_regress` (extended for `mixed` / `meglm`; reads `estat recovariance` for variance components, `estat icc` for the single-grouping intercept-only case, restricts the coefficient submatrix to fixed-effects-only via `e(k_f)` so transformed variance parameters don't appear as "coefficients"); cluster analysis through new `nora_result_cluster.ado` (kmeans + hierarchical with linkage; centroids + within-SS computed from the dataset directly); factor decomposition through new `nora_result_factor.ado` (PCA + factor with pcf / pf / ml / ipf; reads loadings + eigenvalues + explained-variance ratios from `e()`); Kaplan-Meier through `nora_result_km` (the helper existed under 0.9.x but was missing from `_stage_runtime_library`'s `stata_ados` tuple — researchers calling it got "command unknown" until 0.10.0). DiD and RDD stay Stata-deferred for substantive reasons documented in the Stata coverage matrix below | ✅ done (0.10.0) |
| **Runtime-staging invariant test** — `test_executor_profile.py::test_every_runtime_helper_file_is_in_executor_staging_lists` reads `src/nora/runtime/` at test time and asserts every user-callable `.ado` / `.R` / `.py` file is referenced in the executor's `_stage_runtime_library` staging tuple AND has a `capture program drop <name>` line in the shadowing-defense list. Pins the disk↔staging invariant in both directions; pre-0.10.0 the existing one-direction check (staging-list-must-exist-on-disk) silently allowed `nora_result_km.ado` to live on disk without being staged | ✅ done (0.10.0) |
| **`correlation_matrix` sanitizer type** — pairwise correlation matrix as a first-class payload (Pearson / Spearman / Kendall), with min-N gate and per-pair value-key validation. R `nora$from_correlation` and Python `nora.from_correlation` helpers; computes complete-case N (not pairwise N) so off-diagonals draw on the same sample | ✅ done |
| **`request_data` types** — added `quartiles` (25th + 75th + IQR; median omitted as a row-level forbidden field) and `correlation_pair` (Pearson r between two variables, complete-case N). Tool schema gained an optional `variable2` field for multi-variable types | ✅ done |
| **Cross-session result recall** — `list_results_global(query?)` and `expand_result(result_id, session_path?)`. Env-gated via `NORA_ALLOW_CROSS_SESSION_RECALL=1` (default off — researcher-side project separation, NOT a privacy property; stored payloads are pre-sanitized either way). Path-confined to `~/.nora-sessions/` so prompt-injected lookups can't direct the store loader at arbitrary paths | ✅ done |
| **Per-variable min/max opt-in** — `DatasetPolicy.non_disclosive_variables` list in `.nora/policy.json`. Variables on the list (typical: `age`, `year_of_birth`, `education_years`) get `min_value` / `max_value` through descriptive payloads. Default empty: every variable's extremes still suppressed unless explicitly opted in | ✅ done |
| **Stata residual-plot scaling fix** — `nora_plot_residuals` now samples to 5000 points before `rvfplot + graph export ... as(pdf)`. PDF embeds every point as a vector path, so unsampled rendering scaled linearly with N: 200k rows took ~8s, almost entirely PDF rendering. Sampled output reduces that to ~750ms with no loss of pattern visibility. `e()` is unaffected so the downstream `nora_result_regress` still sees the full-N regression | ✅ done |
| **Allow deleting the active session** — sidebar × on the focused row now works; the bridge clears `self.cwd`, returns `was_active=True`, and the page navigates back to the landing screen. Confirm dialog adapts ("Delete the session you're currently in") | ✅ done |
| **`debug_excerpt` on script failure** — system prompt now tells the model to read it; long-standing feature was effectively invisible because the prompt didn't acknowledge it | ✅ done |
| **Multi-result `submit_script` wire format** — JSONL append-mode emit, executor parses N payloads per script with per-line token validation, response carries a `results` list with a shared `script_run_id`. Stata helpers, R `nora$.write_result`, Python `_write_result` all switched to append; sanitizer is stateless per payload so SDC stays exact across N. Solves the loss of 23-of-24 results on event-study batches | ✅ done |
| **Partial-success on script abort** — when a script aborts mid-loop, payloads emitted before the abort still surface (`status: "execution_failed_partial"`) alongside `debug_excerpt`. All-rejected-then-aborted falls to `execution_failed` so the model doesn't read disclosure rejections as usable partials. Per-result `transformations_summary` dedupes shared SDC entries across results | ✅ done |
| **`submit_script_file` tool** — read a `.do` / `.R` / `.Rmd` / `.py` from cwd by basename and forward to `submit_script`. Skips the round-trip cost of re-emitting attached scripts as inline tool input. Path-safety mirrors `read_attached_file`; language inferred from extension when omitted | ✅ done |
| **`search_schema` tool** — case-insensitive substring filter against variable names, labels, and value-label content for wide datasets. `limit` default 50, hard max 200; response carries `total_matches` and `truncated` so the model knows whether to refine | ✅ done |
| **Canonical table rendering** — `nora.result_render.render_table` produces a deterministic markdown pipe-table per analysis type (linear_regression, t_test, descriptive, frequency_table, crosstab, magnitude_table, correlation_matrix). Pure formatter; suppression markers preserved verbatim. Surfaced via `expand_result(view="markdown")` AND inline on every ok-status `submit_script` result, so the model can drop tables directly without re-deriving columns and precision per call. Web UI renders these inline on the tool-result card as a separate result panel above the native script stdout | ✅ done |
| **Inline compact result payload** — every ok-status result entry now carries a `payload` field with full sanitized data minus `vcov`/`vif` for regressions (same trim as `expand_result(view="coefficients")`), full payload for other types. The model renders coefficient tables directly from the response without N `expand_result` round-trips on parameterized batches | ✅ done |
| **Row-count audit perf fix** — `schema.row_count(path)` uses metadata-only paths where available (`.dta` via pyreadstat `metadataonly=True`, `.parquet` via pyarrow footer, `.csv` line-count). `submit_script` resolves the source row count ONCE per call and threads it through the per-payload loop, instead of re-reading the dataset on every result. ~390x faster on a 200k-row .dta benchmark; on a 3 GB file the absolute saving is on the order of a minute per call. Response also carries `_phase_timings` (executor / row_count_audit / sanitize / store seconds) so post-execution slowness can't hide behind `duration_seconds` again | ✅ done |
| **Loop-default prompt directive** — for parameterized batches (N specs / subgroups / outcomes / sensitivity sweeps), prompt now directs ONE script with a loop emitting N results, not N separate scripts. Names the costs of N-scripts directly (repeated data prep, fragmented audit, context bloat). Pinned by render-test | ✅ done |
| **Formatting rules at end of prompt** — moved formatting block past tool-use notes / honesty paragraph so it's the last thing the model reads before generating. Strengthened anti-bold rule with imperative phrasing and explicit anti-pattern call-out (`Bold sentence-leaders`); post-table interpretation is bullets, not prose paragraphs. Composite cell-format tables spell out the canonical shape `-0.013 (0.004) [0.002]` (significance stars forbidden as old convention) | ✅ done |
| **Flexible bullet count + 160-char cap** — dropped the rigid "2 to 4 bullets" rule; the model picks bullet count from what the result actually shows (one tight bullet beats four padded ones). Hard cap of 160 characters per bullet (tweet length); thoughts that genuinely need more space write a SHORT prose paragraph (3-5 sentences), not a "long bullet" that reads as prose with a dot on the front | ✅ done |
| **OpenAI cumulative context fix** — `total_input_tokens += usage.input_tokens` was multi-counting the cached prefix once per tool-loop round. Switched to last-round value (peak prompt size), since each round's `input_tokens` already includes the cached chain. Context chip is honest on OpenAI now; chip docstring describes per-provider semantics | ✅ done |
| **Context chip post-turn snapshot** — chip now sums `input + cache_read + cache_creation + output_tokens` so a long reply moves the chip immediately rather than only on the next turn. The chip's old "input only" framing made it look like context wasn't being used until the next turn folded the response back into input | ✅ done |
| **read_attached_file head+tail truncation** — scripts > 96 KB come back as 48 KB head + elision marker + 48 KB tail instead of head-only. Save calls (`df.to_parquet`, `write_dta`, `saveRDS`) at the bottom of long pipelines now visible | ✅ done |
| **scroll-to-latest button** — floating circular button above the composer fades in when the transcript is > 100 px from the bottom; click smooth-scrolls to the latest message | ✅ done |
| **Wider chat column + scroll-wrapped tables** — `--max-width` 960 → 1080 px so wide composite tables (H1a/H1b cell-format matrices) breathe. Markdown tables now render inside `<div class="md-table">` with thin custom scrollbars (Firefox `scrollbar-width: thin`, WebKit 6 px); the previous `display: block; overflow-x: auto` directly on `<table>` produced an awkward double-scrollbar above and below wide tables. Tables that fit show no scrollbar at all | ✅ done |
| **Audit fixes batch (Apr-30)** — `recall_conversation` AttributeError on multi-result tool calls (used the renamed `result_ids` list); `list_results` newest-first with `limit` (default 50, max 500) instead of unbounded ASC; recall budget includes serialized tool/result_ids size in the cap; `session_state` pairs latest user with its OWN assistant (or empty for in-flight) instead of cross-turn mismatch; `read_attached_file` plot fallback uses `is_relative_to` instead of `str.startswith` (fixes path-prefix collision); composer Send guard allows attachment-only sends; composer image drops persist to cwd via `add_files_from_blobs` AND stage for vision; cancel/error branches no longer re-prepend mentioned files (composer chip already cleared on send); per-script source row count cached once; NaN correlation on constant columns rejected with named culprit; correlation_matrix sanitizer applies `safe_key` to both sides of the cross-field check; store ordering uses `rowid` instead of lexical id sort | ✅ done |
| **Editable session names** — `SessionState.custom_name` (≤120 chars, trimmed; empty clears back to auto). `set_session_name` bridge (sandboxed to `~/.nora-sessions/`); preserved across the per-turn `write_session_state` rewrite by reading the prior file first. Topbar pill is click-to-edit (Enter/Space keyboard-accessible); each sidebar row gets a `✎` button that swaps the row's button for an edit container (avoids the invalid `<input>`-inside-`<button>` nesting); `loadSessions()` re-renders on commit/cancel so both surfaces stay in sync. Renamed sessions show the custom name as the primary line with `date · datasets · size` demoted to the meta row | ✅ done |
| **Top-anchor scroll on assistant replies** — long answers used to land the researcher at the LAST line via `scrollToBottom()`, forcing a manual scroll back to the first sentence; `append()` now top-aligns assistant wrappers via `scrollMessageToTop` (sets `messagesEl.scrollTop = wrapper.offsetTop - 16`, clamped to scroll max). Re-applied on `turn_done` after the loading-indicator removal shifts layout by ~80 px. User / system / error messages still pin to the bottom (composer / status visibility) | ✅ done |
| **List-marker selection-leak fix** — chat-output `<ul>` / `<ol>` drop the native `::marker` (which lives in the parent's padding gutter where WebKit paints text-selection background but does NOT repaint it on selection clear, leaving thin colored bars on every li after a multi-bullet drag-select). Bullets / numbers now render via `::before` inside the `<li>` content box, with a CSS counter (`chat-ol`) for ordered lists. Visual indent unchanged; selection paints/clears uniformly | ✅ done |
| **Loading-label rotation expansion + sidebar shortcut guard** — added 18 data-themed gerunds (`crunching`, `wrangling`, `polishing`, …) and 3 noun-phrase jokes (`herding outliers`, `minding the gaps`, `reticulating splines`). Sidebar arrow/Backspace shortcut handler now bails out when focus is inside an `<input>` / `<textarea>` / `[contenteditable]` so typing in the rename input doesn't fire the row's delete confirm | ✅ done |
| **Cross-query composition / release ledger** | ⏭ named, future-deployment scope |
| **Apple Developer Program signing + notarization for distributable .dmg** | ✅ done — release `.dmg` is signed (Developer ID Application) and notarized |
| **Stata batch wrapper around `_cons` "omitted" edge case** | ⏭ named, low-priority |

### Stata coverage matrix (release status)

The sanitizer recognises thirteen analysis shapes. The 0.10.0 release
ships Stata parity for the four high-usage shapes that were
previously R+Python-only (cluster, factor, mixed-effects, KM). Two
shapes remain Stata-deferred for substantive reasons (DiD: no
payload helper; RDD: numerics-unverified). The earlier-shipped KM
helper is now actually reachable — the `.ado` was in the runtime
directory under 0.9.x but never in the executor's staging list, so
`nora_result_km` failed with "command not found" on every prior
release.

| Shape / feature | R helper | Python helper | Stata helper | Status |
|---|---|---|---|---|
| `coefficient_table_with_fit_stats` (regress / GLM / Cox PH / fixest / IV-2SLS) | ✓ `from_lm` / `from_iv` | ✓ `from_lm` / `from_iv` | ✓ `nora_result_regress` (covers regress/logit/probit/poisson/stcox/xtreg fe/areg/ivregress/mixed/meglm) | shipped |
| `t_test` | ✓ `from_t_test` | ✓ `from_t_test` | ✓ `nora_ttest` (legacy `nora_result_ttest` kept) | shipped |
| `descriptive` | ✓ `from_summarize` | ✓ `from_summarize` | ✓ `nora_result_sum` | shipped |
| `frequency_table` | ✓ `from_table` | ✓ `from_table` | ✓ `nora_result_tab` (1-way) | shipped |
| `crosstab` | ✓ `from_crosstab` | ✓ `from_crosstab` | ✓ `nora_result_tab <v1> <v2>` | shipped |
| `magnitude_table` | ✓ `from_magnitude_table` | ✓ `from_magnitude_table` | ✓ `nora_result_magnitude` | shipped |
| `correlation_matrix` | ✓ `from_correlation` | ✓ `from_correlation` | ✓ `nora_result_correlation` | shipped |
| `kaplan_meier` | ✓ `from_kaplan_meier` | ✓ `from_kaplan_meier` | ✓ `nora_result_km` (also fixes the silent staging gap — the helper existed but was never in `_stage_runtime_library`'s `stata_ados` tuple under 0.9.x) | **shipped 0.10.0** (was effectively broken pre-0.10.0) |
| `cluster_analysis` | ✓ `from_cluster` (kmeans + hierarchical); DBSCAN via `nora$result(...)` | ✓ `from_cluster` (KMeans + AgglomerativeClustering); DBSCAN via `nora.result(...)` | ✓ `nora_result_cluster` (kmeans + hierarchical with linkage; centroids + within-SS computed from the dataset directly since Stata's cluster commands don't store them) | **shipped 0.10.0** |
| `factor_decomposition` | ✓ `from_pca` + `from_fa` (wraps `psych::fa` — ML / minres / pa factor analysis with rotation, communalities, RMSEA / TLI) | ✓ `from_pca` + `from_factor_analyzer` (wraps `factor_analyzer.FactorAnalyzer`) | ✓ `nora_result_factor` (PCA + factor with pcf/pf/ml/ipf extraction; reads loadings + eigenvalues from `e()`; ML-FA goodness-of-fit via `e(chi2_ms)` / `e(p_ms)` / `e(df_ms)` / `e(ll)`) | **shipped 0.10.0** |
| `marginal_effects` (per-variable AME / MEM / at-representative scalars from non-linear fits; `at_values` precision-clamped by sample N) | ✓ `from_marginal_effects` (wraps `marginaleffects::avg_slopes` / `slopes`) | ✓ `from_marginal_effects` (wraps `fit.get_margeff`) | ✗ no helper; `nora_result_margins.ado` deferred | **shipped 0.10.0** (R + Python only) |
| Mixed-effects (sub-feature of `coefficient_table_with_fit_stats`: `random_effects_variance`, `n_groups_per_level`, `icc`, `fit_method`) | ✓ via `from_lm` on `lmer` / `glmer` | ✓ via `from_lm` on `statsmodels.mixedlm` | ✓ Stata `mixed` / `meglm` now routed through `nora_result_regress` — `estat recovariance` for variance components, `estat icc` for the single-grouping intercept-only case | **shipped 0.10.0** |
| Panel-data diagnostics (sub-feature: `f_test_fe_chi2/p`, `hausman_chi2/p`, `breusch_pagan_chi2/p`, `wooldridge_ar1_chi2/p`) | ✓ R `from_lm` auto-runs `plm::pFtest` / `phtest` / `pbgtest` / `pwartest` on `plm` fits | ✓ Python `from_lm` accepts these as caller kwargs (linearmodels PanelOLS) | ✓ Stata `xtreg, fe` auto-emits `f_test_fe_chi2` + `f_test_fe_p` from `e(F_f)`; other tests pass via caller (run `xttest0` / `xtserial` in the script) | **shipped 0.10.0** |
| Cluster-robust SE + typed `robust_se_type` enum (`classical`, `hc0..hc3`, `hac_newey_west`, `cluster`, `bootstrap`) (sub-feature) | ✓ fixest `vcov=` arg auto-detected | ✓ `cov_type=` auto-mapped (`HC0..HC3`, `HAC`, `cluster`) | ✓ `vce(cluster id)` + `e(cmd)=="newey"` auto-emit cluster / hac_newey_west; `cluster_variables` + `n_clusters` populated when applicable | shipped |
| `did_event_study` | ✓ `from_callaway_santanna` + `from_sun_abraham` + `from_twfe_event_study`; de Chaisemartin via `nora$result(...)` | ✓ `from_callaway_santanna`; sun_abraham / twfe_event_study / de_chaisemartin via `nora.result(...)` | ✗ no helper; `install_packages` installs `csdid` from SSC fine, but nothing emits the payload | **deferred — no payload helper.** Hand-authoring JSON to `NORA_RESULT_PATH` is the only route from Stata, and that's a contributor escape hatch, not an end-user workflow. System prompt directs the model to recommend running CS DiD in R or Python via the same session (the `.dta` opens via `haven` / `pyreadstat`; the data stays on the machine). A real Stata helper waits on a contributor pinning the `csdid` API surface |
| `rdd` | ✓ `from_rdd` (wraps `rdrobust::rdrobust`) | ✓ `from_rdd` (wraps `rdrobust` Python) | ✗ no helper; SSC Stata `rdrobust` port has maintenance lag and numerics haven't been verified against CCT 2014 reference | **deferred — 0.10.3 shipped without it; no target release.** Go / no-go is empirical and roughly one Stata session: fit `rdrobust` in Stata and R / Python on the same simulated DGP, compare τ / SE / bandwidths at the 0.5% relative-tolerance level. If they agree, write the helper following the `nora_result_factor.ado` pattern. If they disagree, document the divergence and keep deferred. See [CHANGELOG.md](../CHANGELOG.md) deferred section for the protocol |

**Operational meaning of the two Stata-deferred entries.** Both
deferrals point a Stata-using researcher at the same fallback:
open R or Python inside the same session. Same sandbox, same
sanitizer; the `.dta` opens via `haven` (R) or `pyreadstat`
(Python) without ever leaving the machine. Only the helper lives
in a different runtime. The two deferrals differ in *why* the
Stata helper is missing:

- **RDD: numerics-unverified, still deferred.** The Stata SSC
  `rdrobust` port has known maintenance lag and we have not yet
  verified its output against the CCT 2014 reference that R and
  Python `rdrobust` reproduce. Go / no-go is empirical and roughly
  one Stata session (see [CHANGELOG.md](../CHANGELOG.md) deferred
  section for the protocol). It was targeted for 0.10.3, which
  shipped without it; the deferral is open-ended until someone
  runs the check.
- **DiD: no payload helper.** `install_packages` installs
  `csdid` from SSC fine. What's missing is a `nora_result_*`
  command to emit the payload, so the only route from Stata is
  hand-authoring JSON to `NORA_RESULT_PATH`, a contributor escape
  hatch rather than an end-user path. A real Stata helper waits on
  a contributor pinning the `csdid` API surface and following the
  `nora_result_*` ado-file convention.

**1726 pytest cases collected** via `uv run pytest --collect-only -q`
on 2026-08-20. Full pass/fail depends on local sandbox/runtime
availability. Coverage spans SDK lockdown (Anthropic) +
OpenAI lockdown, schema for all seven file formats, executor SBPL
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
pending plots), **plot rendering / Stata export reliability**
(`test_run_dir_plots.py`: thumbnail collector, helper diagnostic,
`png_for` PDF→PNG conversion via `sips`, helper-error
surfacing in the model-visible tool result, `_nora_export_plot`
PDF→PNG→EPS→.gph fallback order, `nora_safe_export` doesn't write
a manifest entry, runtime environment block renders in the prompt
with `✓`/`✗` package status), **multi-result wire format**
(`test_submit_script_partial.py`: dedup of shared transformations,
partial-success on mid-loop abort, all-rejected-then-aborted
falls to `execution_failed`, source row count resolved exactly
once per call, `_phase_timings` populated, inline compact
payload per result), **canonical-table rendering**
(`test_result_render.py`: per-type renderers, the `markdown`
view on `expand_result`, unknown-type passthrough),
**search_schema** (`test_search_schema.py`: name/label match,
case-insensitive, limit clamping, path safety),
**submit_script_file** (`test_submit_script_file.py`:
extension allowlist, language inference, empty-file refusal,
basename-only path safety),
**recall + listing fixes** (`test_recall_and_listing_fixes.py`:
recall renders multi-result tool calls, list_results bounded
newest-first, recall budget counts the tools array,
session_state pairs same-turn user/assistant), the **edge-
case audit batch** (`test_audit_fixes_2.py`: plot-fallback
path-prefix containment via `is_relative_to`, runner does
not re-prepend mentioned files on cancel/error), and the **editable
session names** suite (`test_session_state.py` additions:
`set_custom_name` round-trip, trim + 120-char cap, empty/whitespace
clears back to `None`, the critical preservation guarantee that
`custom_name` survives the per-turn `write_session_state`
rewrite, refusal on a non-existent cwd).

## Running it

```bash
# Frontend — native WKWebView window via pywebview.
uv run nora                           # landing: drop files or pick folder
uv run nora /path/to/data             # opens straight into chat

# Tests
uv run pytest -q

# Build the .app + .dmg locally. Bundles the web UI.
# A bare local build is unsigned (fine for same-machine testing).
# Set NORA_SIGN_IDENTITY before build_app.sh and NORA_NOTARIZE_PROFILE
# before build_dmg.sh to produce a release-grade signed + notarized
# .dmg; the scripts skip those steps when the env vars are unset.
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
| Anthropic | Sonnet 5 (default), Opus 5, Fable 5.1 | platform.claude.com/docs/…/pricing |
| OpenAI | GPT-5.6 Terra (cost tier), GPT-5.6 Sol (default), GPT-6 Astra (top tier) | openai.com/api/pricing |

The default on each side is the mid-priced mainstream model. The two
top tiers, Fable 5.1 and GPT-6 Astra, both bill $10/$50 per MTok and
are per-session opt-ins; Astra additionally bills 2x input / 1.5x
output on prompts over 272k input tokens.

Each row carries a small `$` link that opens the provider's pricing
page in the system browser. Models for un-authed providers stay
listed but are dimmed; clicking them opens the auth screen
positioned on that provider.

**Effort.** The same popup carries an Effort bar under the model
list, default `xhigh` (what both providers were hard-pinned to
before the dial existed). The composer chip reads `Sonnet 5 · xhigh`.

The ladders are **per provider** — the bar is rebuilt from the
selected model's provider on every render:

| Provider | Ladder |
|---|---|
| Anthropic | `low` `medium` `high` `xhigh` `max` |
| OpenAI | `low` `medium` `high` `xhigh` `pro` |

The four lower rungs are the same dial on both sides
(`output_config.effort` / `reasoning.effort`). The ceilings differ:

- Anthropic's is `max` — the Agent SDK types `EffortLevel` as
  `low…max`.
- OpenAI has no `max` its client can express (the pinned SDK, 2.41.0,
  types `ReasoningEffort` without it; OpenAI's docs *do* claim it, so
  this is the SDK lagging the API). Its ceiling is `pro`, which is a
  different knob entirely — `reasoning.mode` — that buys more model
  work per turn. It's genuinely orthogonal to effort in the API, but
  for "how hard should this try" it is the rung above `xhigh`, so
  that's where the bar puts it. `provider/openai.py::_reasoning_params`
  unpacks it back into `mode="pro"` **plus** `effort="xhigh"` — effort
  would otherwise default to `medium` in pro mode, making the top rung
  reason *less* than the one below it.

`mode` isn't in the pinned SDK's `Reasoning` TypedDict; TypedDicts
aren't runtime-enforced and the SDK's transform layer passes unknown
keys through to the JSON body (verified against
`openai._utils.maybe_transform`, and pinned by a test). Drop the
special case when the SDK types it.

Crossing providers maps by **rank**, so the two ceilings tie: a
researcher on Anthropic `max` who switches to OpenAI lands on `pro`,
not a rung below. Clamping never steps *up*.

Both settings are per-session and both persist to
`.nora/session_state.json`, but they apply differently:

| | Anthropic | OpenAI |
|---|---|---|
| Model swap | in-place via the Agent SDK, conversation kept | per-request field, conversation kept |
| Effort swap | **session closes and re-warms on the next message** | per-request field, conversation kept |

The Anthropic asymmetry is the Agent SDK: effort is the CLI's
`--effort` flag, set at launch, with no in-place control request (
unlike `set_model`). So the provider reports `requires_reopen` and
the *runner* closes the session — closing inside the provider would
let the next `send()` lazily reopen without the runner re-arming
`needs_context_prefix`, silently dropping the conversation. The
re-warm path is the same one a cross-provider model swap takes, and
the toast says so.

## Architecture at a glance

Three independent privacy layers. A break in any one is a bug; a
break in two at the same time is a privacy incident.

1. **Tool interface** (`src/nora/tools.py` + `provider/tool_schemas.py`)
   — the model has exactly fourteen tools: `get_schema`,
   `search_schema`, `request_data`, `submit_script`,
   `submit_script_file`, `expand_result`, `compose_results`,
   `list_results`, `list_results_global`, `recall_conversation`,
   `read_attached_file`, `list_session_files`,
   `search_in_session_files`, `install_packages`. Anthropic SDK
   built-ins (Bash, Read, Write, …) are disabled via
   `disallowed_tools` + `can_use_tool` catch-all +
   `setting_sources=[]`. OpenAI: only the fourteen function tools
   are passed; built-ins (`web_search`,
   `code_interpreter`, `file_search`, `image_generation`, `mcp`)
   are explicitly forbidden — verified on every request by
   `_verify_lockdown` and pinned by `test_openai_lockdown.py`.
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
that runner's session; OTHER runners are untouched.

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
- `provider/openai.py` — wraps the Responses API. Chains turns
  via `previous_response_id` so the OpenAI server holds the prior
  conversation state; the bridge only sends new content per turn
  (the user message on a fresh turn, function-call outputs
  between tool-loop rounds). Tool loop dispatches via
  `nora.tools.HANDLERS` so behaviour is byte-for-byte identical
  regardless of which model called the tool.

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

`nora` without an argv opens a landing screen; dropped /
picked files land in `~/.nora-sessions/<ts>_<id>/` which becomes
the cwd. That dir is spaces-free (Stata-safe), outside cloud-sync
roots, persistent across restarts. Reopening a session restores the
`active_model` and `active_effort` recorded in
`.nora/session_state.json` — a researcher who switched to Opus at
`max` for one project comes back to Opus at `max` next time, even
when the launcher passes the cwd directly. The two restore
independently: effort is provider-neutral, so a state file naming a
model that has since left the catalog still gets its effort back
while the model falls to the default.

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
  until concrete beta-user demand surfaces it.
- **Bounded covert channels in regression metadata.** Documented
  in `sanitizer.py` (`predictor_variables` is script-authored;
  `nora$result(...)` allows hand-crafted payloads). Bandwidth is
  small (~hundred bytes per regression); same threat-model
  scenario as cumulative-inference above. Out of scope at the
  current scale.
- (Other deployment-level concerns land here as they surface —
  multi-tenant key management, per-organisation policy
  enforcement, audit-log retention, etc. None are real today.)

## Rough edges (work, but annoy)

- Transformations log shows a single `"dropped N unknown/forbidden
  field(s)"` summary on every script call that passed a
  `label=…` arg, because `label` is omitted from the per-type
  string allowlist (most types — `_CORR_ALLOWED_STRING_FIELDS` is
  the lone outlier that lists it, which is its own minor
  inconsistency). Harmless: the field is extracted as
  `helper_label` from the raw payload BEFORE the sanitizer runs
  ([tools.py:1358-1364](../src/nora/tools.py)) and stored on the
  row's `label` column, where every model-facing surface
  (`expand_result`, `list_results`, `compose_results`) reads it
  from. The log noise just confirms the sanitizer correctly
  refuses to duplicate the field into the sanitized payload.
- Raw-log panel cap is 32 KB per stream with head + tail
  preservation and a `[… N bytes truncated from the middle …]`
  marker. Sufficient for regression tables and helper summaries;
  larger logs survive on disk for `tail -F`.
- Sanitizer drops the `_cons` coefficient as empty when Stata
  reports it as "omitted" (perfect-fit edge case) — generated
  JSON becomes malformed. Only triggers on degenerate toy data;
  flagged in a test comment. Not a production blocker.

## Key files (reading order)

| File | What's there |
|---|---|
| `src/nora/system_prompt.py` | The model's full system prompt + dataset listing. Single source of truth for both providers |
| `src/nora/tools.py` | The MCP tools (14 currently, listed in `ALLOWED_TOOL_NAMES`). Start here to understand the model's surface |
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
| `src/nora/runtime/nora.R` + `nora.py` + `nora_result_*.ado` + `nora_plot_*.ado` + `_nora_export_plot.ado` + `nora_safe_export.ado` | Runtime emitters: result helpers (`from_lm`, `from_t_test`, `from_cluster`, `from_pca`, `from_kaplan_meier`, …) and plot helpers (`plot_residuals`, `plot_interaction`, `plot_coefficients`, `plot_estimate_comparison`). Sixteen Stata `.ado` files cover every shape with Stata parity — `nora_result_regress` (extended for `mixed` / `meglm` in 0.10.0), `nora_result_cluster` (new in 0.10.0, kmeans + hierarchical with linkage), `nora_result_factor` (new in 0.10.0, PCA + factor with pcf / pf / ml / ipf), `nora_result_km` (newly staged in 0.10.0 — the helper existed earlier but the staging gap kept it unreachable), plus regress / ttest / sum / tab / magnitude / correlation. Stata fallback chain in `_nora_export_plot`; Stata ad-hoc safe wrapper in `nora_safe_export`. **The disk↔staging invariant is pinned by `test_executor_profile.py::test_every_runtime_helper_file_is_in_executor_staging_lists` — a new helper file in this directory must be wired into the executor's staging tuple or the test fails** |
| `src/nora/schema.py` | Schema extractors for all seven supported file formats |
| `src/nora/ui.py` | Web UI bridge: runners dict, focus-only `switch_session`, plot collection + diagnostic, Files panel endpoints, cache-busted index.html, `delete_session_file` |
| `src/nora/chat_service.py` | Back-compat re-export shim for the Event types |
| `src/nora/chat_history.py` | Turn-grouped reader; warm-start prefix renderer |
| `src/nora/session_state.py` | Atomic writer / reader for `.nora/session_state.json` (carries `active_model` for per-session memory) |
| `src/nora/web/{index.html,app.js,markdown.js,style.css}` | Web frontend; `app.js` holds the per-session focus state + Files-panel rendering |
| `src/nora/__main__.py` | Package entry — calls `nora.ui:main`. The .app and `python -m nora` both end up here |
| `docs/direction.md` | Long-form architectural doc; open questions |
| `docs/overview.md` | Plain-language description for researchers |
| `docs/install.md` | Researcher-facing install flow |
| `docs/verification.md` | Manual smoke-test recipes (incl. Stata, which CI can't) |
| `tests/` | 1726+ pytest cases (the 0.10.0 Stata-parity pass added cluster / factor / mixed-effects real-fit pins plus the disk↔staging invariant test; run `uv run pytest --collect-only -q` for the current count). `test_sanitizer.py` is the property-test backbone (now also covers vif / condition_number / vcov + correlation_matrix); `test_concurrent_sessions.py` pins the per-task ContextVar isolation; `test_plot_vision.py` pins the manifest-allowlist privacy gate; `test_run_dir_plots.py` covers thumbnail collection + PDF→PNG conversion + Stata export fallback chain; `test_openai_lockdown.py` pins the no-built-in-tools invariant; `test_cross_session_recall.py` pins the env-gated cross-session lookup + path-confinement defense; `test_bridge_lifecycle.py` covers active-session delete + landing-page navigation; `test_executor_profile.py` pins the staging-tuple ↔ runtime-directory invariant in both directions (every staged file exists; every file is staged) |

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
  through the same fourteen-tool MCP surface; the privacy layers
  (sandbox + sanitizer) don't depend on which provider authored
  the call. Adding a third provider means writing a new
  `provider/foo.py` and one lockdown test; the SDC and sandbox
  layers are untouched.
- **OpenAI: API key only.** ChatGPT subscription doesn't carry
  programmatic API access; mixing browser-OAuth with API-key
  flows in one input field is a UX trap. If/when OpenAI adds a
  subscription-included API tier, revisit.
- **macOS-only for now.** Script execution relies on `sandbox-exec`.
- **Schema policy is a ceiling, not a fixed value.** The model
  can request any depth ≤ ceiling. Researchers edit via the
  Permission chip in the composer row.
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
- **User-facing copy says Nora or assistant.** Provider and model
  names belong in the auth/model picker, not as the product voice.
  The system prompt tells the model to introduce itself as Nora and
  to use first person ("I noticed…" not "Nora flagged…").
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
  exists only for startup before any session is active. Any new
  code that reaches around the ContextVar (e.g., reads a stashed
  cwd from somewhere else) breaks concurrent-runner isolation.
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
- **GitHub repo is `junishka/Nora`.** Older `junishka/builder`
  URLs redirect, but docs should use the canonical repo name. The
  local clone still gets renamed to `nora` in install snippets so
  the on-disk dir matches the product name.

## Next concrete step

If you're picking this up to finish and ship it, in this order:

1. **Run it yourself on your own data.** Catch the UX rough
   edges before anyone else sees them.
2. **Try it on someone else's data** (with their permission).
   Workflows that do not match the maintainer's mental model
   surface UX bugs internal testing misses.
3. **Distribution-mode concerns (cumulative-inference /
   release-ledger, multi-tenant policy, audit retention) wait
   until distribution itself is real.** They are meaningful work
   for a future deployment, not for the current scale.
   See "Longer-term: governance for wider distribution" above.

# Nora — architectural direction

Working document. Last substantive update **2026-05-15**, after
the audit-fixes pass: install-packages consent is modal-only, the
script-failure `debug_excerpt` is documented at its stricter
redaction posture (exception bodies redacted wholesale, only
parser-anchored framing crosses), `load_data` and the schema fast
paths share a CSV/TSV header peek so they no longer disagree on
headerless files, `_names_only_payload` distinguishes
"row_count unknown" from "empty dataset", JSONL `names_only`
unions keys across the file, malformed per-dataset policy entries
clamp to the strictest tier (rather than fail open), the
`chat_history` lightweight readers honour their "never raises"
contract through concurrent-delete races, cancelled
`submit_script` runs no longer emit a model-visible cancelled
tool result, `delete_credential` notices external Keychain
deletions, the session-focus switch leaves model-captured plot
images intact, the loop-shutdown race that could leave a runner
permanently busy is fixed, pywebview's deprecated `OPEN_DIALOG` /
`FOLDER_DIALOG` are replaced by `FileDialog.OPEN` /
`FileDialog.FOLDER`, and `_materialize_cache_busted_index` routes
its bust file to a temp directory in packaged builds so the write
no longer modifies the codesigned bundle. The previous batch
(2026-05-12) was the release-readiness / documentation alignment
pass: multi-provider framing, fourteen-tool MCP surface, seven
supported data formats, R / Stata / Python as selectable analysis
runtimes, signed + notarized `.dmg`. The core decision — stay
with script-submission ("Option A") rather than pivot to
plan-submission with a bundled local LLM — still stands from
2026-04-20.

For the single-page overview aimed at someone picking this up, see
[`docs/handoff.md`](handoff.md). [`docs/overview.md`](overview.md)
is the plain-language description of what Nora is and why. This
doc is the long-form record of what the architecture is, what it
isn't, and why.

## The decision

**Keep the current architecture.** Remaining work is outside-pilot
feedback, small consent/policy UX, and future deployment governance
— not restructuring.

A recent external reviewer proposed replacing the frontier model's
direct script authorship with a plan-submission architecture where
the model writes structured plans and a bundled local LLM compiles
them into R/Stata. After working through the argument, that proposal
was rejected as over-correction.

The privacy guarantee in Nora comes from three independent
layers:

1. **The tool interface** — the model can only use the registered
   Nora tools. No filesystem, no shell, no network.
2. **The sandbox** — scripts run under macOS `sandbox-exec` with a
   `(deny default)` profile and a narrow subpath-allowlist for
   reads; network denied entirely.
3. **The sanitizer** — every output payload passes through SDC +
   text-safety checks before reaching the model.

None of those three layers depends on *who authored the code*.
The model writing R/Stata/Python is fine as long as it cannot
directly touch data — which the tool interface and sandbox together
enforce.

A local LLM in the loop would close one narrow additional channel:
an adaptive, adversarial model choosing exactly which
computations run on the data. That's not nothing — it's a real but
bounded gain. What it does *not* solve is the harder privacy
problem, which is **cumulative inference from many sanitized
queries** (the "20 questions" attack: each query is individually
compliant, but the joint distribution of answers reveals more than
any single one should). That problem is inherent to any
interactive analysis system and is addressed — imperfectly — by
session-level disclosure budgets and the sanitizer's SDC rules,
not by who authored the code.

So a local LLM offers *limited* privacy benefit (not zero, not
primary) and would add real *capability* (debugging with raw
stderr, text-data handling). It stays an optional future
enhancement, re-enters the critical path when a specific
researcher use case demands it.

## What's built

As of 2026-05-15, the implementation covers:

- Spine + full SDK lockdown (14 MCP tools — get_schema, search_schema,
  request_data, submit_script, submit_script_file, expand_result,
  compose_results, list_results, list_results_global,
  recall_conversation, read_attached_file, list_session_files,
  search_in_session_files, install_packages — every built-in disabled,
  four defense layers).
- Schema extractor for `.csv`, `.tsv`, `.dta`, `.rds`,
  `.parquet`, `.jsonl`, and `.ndjson` with four depth tiers;
  default is `names_types_labels_summary`.
- Executor with `(deny default)` subpath-allowlist sandbox AND an
  explicit subprocess env-var allowlist (PATH/HOME/LANG/LC_*/TMPDIR/
  USER/SHELL/R_LIBS — no ANTHROPIC_API_KEY, AWS creds, or other
  shell secrets visible to scripts). The profile now uses narrow
  `/private/etc` literals instead of a broad subtree read. Per-run
  HMAC token authenticates payloads from the runtime library. Pure
  unit tests lock in the SBPL profile shape; integration tests
  verify real sandbox behavior (gated on installed runtimes +
  sandbox-apply preflight).
- Sanitizer across the supported analysis families, with runtime
  emitters in R, Python, and Stata where applicable. OLS
  coefficient-key constraint (inner keys must match declared
  predictors); confidence-interval length constraint (must be
  exactly 2); structural size caps on every dict / list payload
  field; filename + variable-name sanitization at every
  prompt-injection surface.
- Runtime libraries for R, Python, and Stata with JSON-escaped
  labels and CR/LF/TAB handling.
- SQLite result store, keyed by resolved cwd (no cross-session leak
  in the same process).
- Memory stack: every turn persisted to `.nora/chat_history.jsonl`
  with timestamps; warm-start prefix injected on every fresh SDK
  client open (last ~20 turns + recent result IDs); `recall_conversation`
  tool for older lookups with neighboring-context expansion;
  durable `.nora/session_state.json` snapshot regenerated after
  each turn (last exchange, recent results, datasets, model).
- **Concurrent sessions.** Bridge holds `dict[cwd → SessionRunner]`;
  switching the visible chat is a pure UI focus change. Each
  runner has its own ProviderSession, asyncio lock, turn task,
  pending-attachment list, and active model. `run_turn` enters
  `nora.config.use_cwd(self.cwd)` so tool handlers running on
  parallel runner tasks see THIS runner's cwd via a ContextVar —
  sister tasks see their own. Stop cancels only the focused
  runner. Events are stamped with `session_cwd` so persistence
  routes correctly even when an in-flight turn outlives the focus
  switch.
- **Plot vision.** Runtime helpers in R / Python / Stata produce
  canonical model-output plots (residuals, predicted-response
  curves, coefficient forest plots, estimate comparisons) from
  fitted-model objects. Each writes a PNG/PDF/EPS into
  `<run_dir>/_nora_plots/` and appends a JSON line to
  `manifest.jsonl`. The runner reads ONLY the manifest — files
  in the dir without an entry stay invisible to the model.
  Stata's PNG export depends on the `Graph2png` translator (often
  missing), so `_nora_export_plot.ado` tries `as(pdf)` →
  `as(png)` → `as(eps)` → `graph save .gph`; the bridge
  rasterizes PDF/EPS to PNG via macOS `sips` for both
  researcher thumbnails and model-vision attachments. Plot
  helper failures append to `_nora_plots/helper_errors.jsonl`
  with a step indicator + fix hint; `submit_script`'s response
  surfaces succeeded + failed so the model knows when a
  thumbnail is missing because matplotlib isn't installed.
- **Runtime environment in the prompt.** `env_detect` probes
  installed runtimes + optional packages (R: haven, ggplot2;
  Python: matplotlib + the four required stats packages). The
  system prompt renders a per-runtime block with `✓` / `✗`
  marks so the model picks a language by what's actually
  installed instead of trial-and-error through missing-package
  failures.
- Web UI: pywebview shell, sessions sidebar with per-session
  busy dot, theme toggle, model
  picker, drag-drop file/image upload, Lottie cat loading indicator,
  status line, Permission/Model chips with popups, image-paste
  support.
- Packaging: `.app` launches the web UI directly (no Terminal popup);
  the release `.dmg` is signed and notarized.
- 1312 pytest cases collected via `uv run pytest --collect-only -q`
  on 2026-05-15, plus Hypothesis-generated adversarial cases.
  Canonical repo: [github.com/junishka/Nora](https://github.com/junishka/Nora).

## What's remaining (prioritized)

### 1. First-open policy nudge

Schema depth is already explicit researcher policy in
`<cwd>/.nora/policy.json`, with per-dataset ceilings and a
composer Permission chip. The default is
`names_types_labels_summary`; raw values, min, max, median, and
individual observations remain unavailable at every schema tier.
The remaining UX polish is an explicit first-open nudge for
un-policy'd datasets so researchers understand the default before
their first analysis.

### 2. Runtime-authenticity follow-on, only if needed

The implemented per-run token rejects trivial hand-crafted writes
to `NORA_RESULT_PATH`; tests pin that behavior. It is a cost-raising
measure, not a cryptographic proof against malicious code running
inside the interpreter. A stronger pre-opened-fd design remains
available if future pilots involve a threat model where runtime
authenticity is load-bearing.

### 3. Distribution-mode governance

Cumulative inference, release ledgers, multi-tenant policy, and
audit retention are real concerns for wider deployment. They are
not blockers for the current mode: a researcher running Nora on
their own machine, against their own data, with their own provider
credential.

## Known-real, design-pending

### Cumulative-inference / cross-query composition

**Status as of the current pilot:** named, design-pending, and
*not* the right thing to spend time on yet. The current and
intended near-term mode is a researcher running Nora against their
own data with their own API key — adversarial models and
adversarial researchers aren't part of that threat model. Where
this becomes load-bearing is wider deployment: shared instances,
researchers analysing data they don't own, regulated datasets
where the threat model includes adaptive probing. Acknowledged
here so it isn't rediscovered as a surprise; deferred until the
deployment shape that needs it actually exists.

**Re-read trigger.** Before any change that broadens the
distribution shape (a new analysis type the sanitizer accepts, a
new opinionated helper, a new field added to an existing shape's
allowlist) or that adds a cross-session channel (anything that
lets one session observe another session's results, or that lets
a session observe artifacts produced outside Nora), this section
gets revisited. The composition surface below is the baseline this
deferral is calibrated against; broaden it and the calibration is
stale.

#### Current per-shape distribution surface

Twelve analysis shapes cross the boundary today. The composition
deferral has to be evaluated against this whole surface, not
against the smaller set the older threat-model writeup assumed.
One-line SDC summary per shape:

1. **`coefficient_table_with_fit_stats`** — the regression bucket
   (OLS, GLM family, Cox PH, fixest, 2SLS, mixed-effects). SDC:
   precision clamping by N; predictor-key allowlist tied to
   declared `predictor_variables`; CI length constraint (exactly
   2); structural caps on diagnostics (`vif`, `vcov`,
   `condition_number`, `fixed_effects`, `n_clusters`,
   `random_effects_variance`, `n_groups_per_level`); rejection on
   coefficient-key collisions or unknown fields.
2. **`t_test`** — one / two-sample / Welch / paired. SDC:
   precision clamping; CI length constraint; subtype-allowlist;
   N-gated emission.
3. **`descriptive`** — mean / SD / N / missing-count for a single
   variable. SDC: precision clamping by N; min / max suppressed
   unless the variable is opted into `non_disclosive_variables`;
   median is forbidden at this shape (request via `quartiles`
   instead).
4. **`frequency_table`** — 1-D counts. SDC: cell suppression at
   threshold 10; secondary suppression so the suppressed cells'
   total can't be back-solved; per-cell precision clamping;
   structural cell cap (200).
5. **`crosstab`** — 2-D counts. SDC: cell suppression at
   threshold 10; per-cell clamp; structural cell cap (2500,
   ~50×50).
6. **`magnitude_table`** — sum / mean per group. SDC:
   (1, 85%)-dominance rule (top contributor cannot account for
   more than 85% of the cell); cell suppression; cell cap (200).
7. **`correlation_matrix`** — pairwise Pearson / Spearman /
   Kendall. SDC: complete-case N (not pairwise); min-N gate;
   per-pair value-key validation; rejection on constant-column
   NaN with named culprit; variable cap (30, →900 entries).
8. **`did_event_study`** — heterogeneous-treatment DiD
   (Callaway-Sant'Anna, Sun-Abraham, TWFE-ES, de Chaisemartin).
   SDC: cohort-N gate — cohorts below threshold are dropped whole
   (group + event-time row + SE / p / CI together), partial-cell
   publication would leak cohort size; precision clamping per
   cohort's N.
9. **`rdd`** — regression discontinuity (CCT 2014). SDC:
   precision clamping by effective N on each side of the cutoff;
   CCT three-flavor convention enforced (conventional /
   bias-corrected / robust each carry τ / SE / p / CI of the
   same length); McCrary density and binscatter are structurally
   refused — they're visual diagnostics only.
10. **`cluster_analysis`** — kmeans / hierarchical / agglomerative
    / dbscan. SDC: per-cluster precision clamping
    (`_clamp_dict_by_per_key_n`); clusters below the size
    threshold drop **whole** (size + centroid row +
    `within_cluster_ss` entry together); per-observation
    `labels_` / `cluster_membership` are structurally absent from
    the allowlist.
11. **`factor_decomposition`** — PCA / factor analysis. SDC:
    per-variable loading clamps; per-component eigenvalue and
    variance clamps; per-observation factor scores (`fit$x` /
    `fit.transform(X)`) are structurally absent from the allowlist.
12. **`kaplan_meier`** — survival. SDC: median + S(t) at preset
    horizons (1y / 3y / 5y / 10y) gated by per-horizon
    `n_at_risk_h`; the KM step function itself does not cross —
    only the horizon-summarized survival probabilities and an
    optional cross-group log-rank χ².

Plot vision adds a thirteenth surface: four model-output helpers
(`plot_coefficients`, `plot_interaction`, `plot_estimate_comparison`,
`plot_residuals`) that emit PNG/PDF/EPS via a manifest-allowlisted
capture path. Helper-allowlist gated; no file-allowlist API exists
(see "What we're not doing"). Privacy bandwidth: pixel-level
covert channels are theoretically possible inside a coefficient
plot, but the input data is already a sanitized model summary, so
the upper bound is what the SDC pass already released.

#### Cross-session channel: `list_results_global`

`list_results_global(query?)` lets one session look up stored
results from any other Nora session under `~/.nora-sessions/`.
Env-gated (`NORA_ALLOW_CROSS_SESSION_RECALL=1`, default off) and
path-confined to that prefix, so prompt-injected lookups can't
direct the loader at arbitrary on-disk paths. **It is researcher-side
project separation, not a privacy property** — every payload it
returns has already been through the sanitizer at write time, so
the bytes that cross are the same bytes that crossed when they
were first emitted. But for composition-attack accounting it is a
real cross-session channel: a session that queries the global
store can join its own emissions with prior sessions' emissions,
expanding the joint distribution beyond what a per-session
disclosure budget would bound. Named here explicitly so a future
release-ledger design knows the accounting layer has to span
sessions, not just turns within a session.

Nora's SDC rules (precision clamping, cell suppression,
dominance, text-safety) constrain what any **single** sanitized
result reveals. They do not constrain the **joint distribution of
answers across many queries**. A researcher — or an adaptive,
adversarial model — who issues 200 individually-compliant queries
can learn things about the dataset that no single query would
release. This is the "20 questions" attack, and it is inherent to
every interactive analysis system (not a Nora-specific bug).

`tools.py` (submit_script, request_data) currently serves each call
independently; `store.py` keeps every sanitized result in a single
growing SQLite table per cwd, plus the global view via
`list_results_global`. The raw material for a release ledger is
already on disk — what's missing is the accounting layer that
reads it and the policy layer that decides when to stop.

Session-level disclosure budgets are the first layer of defense,
but naïve implementations ("count calls, stop at N") do not
protect against adaptive adversaries and frustrate honest
researchers whose work is iterative by nature. Worse than nothing
if they give false confidence.

The real answers live in established SDC / DP literature:

- **Differential privacy with a per-session ε-budget.** Genuine
  guarantee, but adds calibrated noise to every result, which
  makes replication and debugging harder. Census Bureau went
  through the ergonomics pain post-2020.
- **Interactive query audit** — track every released quantity,
  block queries whose composition with prior releases would
  exceed a leakage threshold. Requires defining "composition"
  rigorously.
- **Release ledger as human-review queue** — every emission
  logged; periodic review by a disclosure officer. Organizational
  pattern, low tech cost, but requires a reviewer.

**Status: backlog. Design-pending.** Commit to reading the
literature (OpenDP, Google's DP library, the Census Bureau's
post-2020 approach, τ-ARGUS's query-log mechanisms) before
implementing. Do not ship a query counter that looks like
protection but isn't.

When does this become urgent? When Nora is used against data
where repeated-query inference is a realistic attacker scenario —
clinical trial data, HR data at the individual level, regulated
datasets. Not urgent for pilot-scale public-ish research.

## Invariants (non-negotiable)

- The model has no general-purpose tools. SDK built-ins stay
  disabled.
- The model's only interface to the machine is the MCP tools registered
  in `ALLOWED_TOOL_NAMES` (the source of truth — currently 14: the
  six original plus `search_schema`, `submit_script_file`,
  `compose_results`, `list_results_global`, `read_attached_file`,
  `list_session_files`, `search_in_session_files`, and
  `install_packages`).
- Every `submit_script` call runs under the sandbox.
- Every executor output passes through the sanitizer before
  reaching the model.
- Raw stderr / stdout never reach the model.
- Schema exposure is explicit researcher policy, bounded by the
  per-dataset ceiling.
- Researcher sees raw logs and sanitized output; the model sees
  sanitized output only.
- **Plot vision is helper-allowlist gated.** Only files produced
  by `plot_residuals` / `plot_interaction` / `plot_coefficients` /
  `plot_estimate_comparison` (each takes a fitted-model object as
  input) and registered in `<run_dir>/_nora_plots/manifest.jsonl`
  cross to the model. Bespoke plots saved via `ggsave` /
  `plt.savefig` / `graph export` stay researcher-only by
  construction. There is no file-allowlist API where a script
  self-attests "this is a coefficient plot"; that route was tried
  and removed because the kind label is unverifiable from a PNG.
- **Per-task cwd, not process-global.** Tool handlers read cwd
  through `nora.config.get_cwd()`, which honors the
  `nora.config._cwd_var` ContextVar that `SessionRunner.run_turn`
  sets via `use_cwd(self.cwd)`. Any new code path that resolves
  cwd from a stashed module global, or from `self.cwd` on a
  different object, breaks concurrent-runner isolation and is a
  bug.

## What we're not doing (and why)

- **Replacing `submit_script` with a constrained plan grammar.**
  Privacy gain is zero — the boundary is already enforced by the
  tool interface + sandbox + sanitizer stack. Capability cost is
  real (narrow grammars can't express log-transforms, interaction
  terms, clustering, fixed effects without bloating the grammar
  into a mini-language). Timeline cost is weeks. Rejected as
  over-correction.

- **Bundling a local LLM on the critical path.** 15–17 GB install
  footprint, requires 16 GB+ RAM, produces worse R/Stata than
  frontier models — especially Stata, where the open training corpus
  is thin. Privacy benefit is narrow (closes the
  frontier-authored-code channel for adaptive attackers) but does
  not address cumulative-inference / adaptive-probing risks, which
  are inherent to interactive analysis regardless of authorship
  and are handled by session-level disclosure budgets and the
  SDC rules. Stays available as a future optional helper for:
  error recovery using raw stderr (which the frontier model can't
  see), text-data redaction so free-text values can flow through the
  sanitizer, and quality improvements in specific edge cases.
  Re-enters the discussion when a real researcher's task actually
  needs one of these.

- **Mandatory safe variable IDs as the frontier-facing identity
  surface.** Schema exposure is policy, not architecture. If a
  researcher's dataset has non-sensitive variable names and they
  opt in to sharing them with the model, that's their call. Nora
  enforces conservative defaults and makes the choice visible; it
  doesn't enforce a ceiling.

- **Language-specific hybrid (frontier model writes Stata, local
  model writes R/Python).** The premise — that local models are weak
  on Stata — is true, but Option A has the frontier model writing
  all three languages directly. The hybrid solves a problem we don't
  have.

- **A `register_plot(file, kind)` API for arbitrary plot files.**
  Tried in 2026-04-26 and removed days later: the kind label was
  self-attested by the script, so a histogram of raw observations
  could pose as a `coefficients` plot and slip past the privacy
  line. The replacement is the four kind-specific helpers (each
  takes a fitted-model object); bespoke plots stay
  researcher-only. Reintroducing a file-based escape hatch
  reintroduces the privacy hole. If a researcher's workflow needs
  a plot kind we don't have, the answer is a new opinionated
  helper with model-output inputs only — not a generic file
  registrar.

- **Closing a runner's SDK client when the user switches away
  from its chat.** Doing so was the multi-session bug
  (`receive_response()` raised mid-stream and surfaced as a
  fail bubble in the new session). Switching is a pure UI focus
  change; runners stay alive until the bridge shuts down or the
  session is explicitly deleted. `test_concurrent_sessions.py`
  pins this.

- **Carrying script attachments forward across a failed turn on
  the bridge side.** Tried, removed: the JS chip cleared at send
  time and the bridge silently held the attachment, producing
  "X is already attached" toasts for files no chip showed. If a
  send fails the user re-attaches; simpler model, no drift.

## Open policy decisions

- **First-open schema-policy nudge.** The default is
  `names_types_labels_summary`; decide whether first-open should
  actively ask the researcher to confirm or lower that ceiling.
- **When to revisit local LLM.** Rule of thumb: when a real
  researcher asks for something only a local model can provide
  (text-data analysis, stderr-based repair of a specific recurring
  failure mode). Not before.
- **When to invest in pre-opened-fd result emission.** The current
  per-run token blocks trivial hand-crafted payloads. A stronger fd
  design should wait for a pilot or deployment threat model that
  needs it.

## What must not happen

- The model gains general-purpose tools at any stage.
- Sandbox or sanitizer becomes conditional on a flag.
- Raw stderr / stdout reaches the model.
- Schema-exposure defaults widen silently.
- Trivial runtime-library bypass reopens.

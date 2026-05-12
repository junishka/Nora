# Nora — architectural direction

Working document. Last substantive update **2026-05-12**, after
the release-readiness / documentation alignment pass: Nora is now
documented as a multi-provider app (Anthropic or OpenAI), the
model-facing surface is consistently fourteen MCP tools, supported
data formats are consistently seven (`.csv`, `.tsv`, `.dta`,
`.rds`, `.parquet`, `.jsonl`, `.ndjson`), install language treats
R / Stata / Python as selectable analysis runtimes, and the `.dmg`
is recorded as signed + notarized. Earlier self-pilot batches
covered concurrent sessions, plot vision, Stata export reliability,
runtime-environment probing, Files-panel polish, the Builder →
Nora rename, the web UI `.app`, the memory stack, and the
security-review fixes. The core decision — stay with
script-submission ("Option A") rather than pivot to plan-submission
with a bundled local LLM — still stands from 2026-04-20.

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

As of 2026-05-12, the implementation covers:

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
- 1121 pytest cases collected via `uv run pytest --collect-only -q`
  on 2026-05-12, plus Hypothesis-generated adversarial cases.
  Canonical repo: [github.com/junishka/Nora](https://github.com/junishka/Nora).

## What's remaining (prioritized)

### 1. Outside pilot on real data

The release path exists; the missing signal is a real outside
researcher using Nora on real data. The user self-pilot is useful,
but a colleague or two validates whether the install flow, model
choice, runtime requirements, file upload, raw-output panels,
policy chip, and result tables make sense to someone who did not
build the system.

### 2. First-open policy nudge

Schema depth is already explicit researcher policy in
`<cwd>/.nora/policy.json`, with per-dataset ceilings and a
composer Permission chip. The default is
`names_types_labels_summary`; raw values, min, max, median, and
individual observations remain unavailable at every schema tier.
The remaining UX polish is an explicit first-open nudge for
un-policy'd datasets so researchers understand the default before
their first analysis.

### 3. Runtime-authenticity follow-on, only if needed

The implemented per-run token rejects trivial hand-crafted writes
to `NORA_RESULT_PATH`; tests pin that behavior. It is a cost-raising
measure, not a cryptographic proof against malicious code running
inside the interpreter. A stronger pre-opened-fd design remains
available if future pilots involve a threat model where runtime
authenticity is load-bearing.

### 4. Distribution-mode governance

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
growing SQLite table per cwd. The raw material for a release
ledger is already on disk — what's missing is the accounting layer
that reads it and the policy layer that decides when to stop.

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

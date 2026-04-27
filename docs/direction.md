# Nora — architectural direction

Working document. Last substantive update **2026-04-27**, after
the concurrent-session refactor (per-cwd `SessionRunner`, per-task
cwd via `ContextVar`), plot vision (manifest-allowlisted helpers,
no file-based escape hatch), Stata export reliability
(PDF / PNG / EPS / .gph fallback, sips-based PDF→PNG conversion),
runtime-environment probing in the system prompt, and the
Files-panel polish (graphs first, copy/send/delete row actions,
longer dropdown). Earlier in the same self-pilot cycle: Builder
→ Nora rename, .app launches web UI, the memory stack
(warm-start prefix + recall_conversation tool + durable
session_state.json), the security-review fixes (env-var
allowlist, per-cwd store, filename sanitization, OLS coefficient-
key constraint, CI length, structural size caps), and the
product-identity prompt rule. The core decision — stay with
script-submission ("Option A") rather than pivot to plan-submission
with a bundled local LLM — still stands from 2026-04-20.

For the single-page overview aimed at someone picking this up, see
[`docs/handoff.md`](handoff.md). [`docs/overview.md`](overview.md)
is the plain-language description of what Nora is and why. This
doc is the long-form record of what the architecture is, what it
isn't, and why.

## The decision

**Keep the current architecture.** Remaining work is hardening, UX,
and real-researcher contact — not restructuring.

A recent external reviewer proposed replacing Claude's direct
script authorship with a plan-submission architecture where Claude
writes structured plans and a bundled local LLM compiles them into
R/Stata. After working through the argument, that proposal was
rejected as over-correction.

The privacy guarantee in Nora comes from three independent
layers:

1. **The tool interface** — Claude can only do six things
   (`get_schema`, `request_data`, `submit_script`, `expand_result`,
   `list_results`, `recall_conversation`). No filesystem, no shell,
   no network.
2. **The sandbox** — scripts run under macOS `sandbox-exec` with a
   `(deny default)` profile and a narrow subpath-allowlist for
   reads; network denied entirely.
3. **The sanitizer** — every output payload passes through SDC +
   text-safety checks before reaching Claude.

None of those three layers depends on *who authored the code*.
Claude writing R is fine as long as Claude cannot directly touch
data — which the tool interface and sandbox together enforce.

A local LLM in the loop would close one narrow additional channel:
an adaptive, adversarial Claude choosing exactly which
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

As of 2026-04-25, the implementation covers:

- Spine + full SDK lockdown (6 MCP tools — get_schema, request_data,
  submit_script, expand_result, list_results, recall_conversation —
  every built-in disabled, four defense layers).
- Schema extractor for `.csv` / `.dta` / `.rds` with four depth
  tiers; default is `names_types_labels_summary`.
- Executor with `(deny default)` subpath-allowlist sandbox AND an
  explicit subprocess env-var allowlist (PATH/HOME/LANG/LC_*/TMPDIR/
  USER/SHELL/R_LIBS — no ANTHROPIC_API_KEY, AWS creds, or other
  shell secrets visible to scripts). Per-run HMAC token authenticates
  payloads from the runtime library. Pure unit tests lock in the
  SBPL profile shape; integration tests verify real sandbox behavior
  (gated on `Rscript` + sandbox-apply preflight).
- Sanitizer across six analysis families (linear regression,
  t-test, descriptive, frequency table, crosstab, magnitude table)
  with full R + Stata parity. OLS coefficient-key constraint
  (inner keys must match declared predictors); confidence-interval
  length constraint (must be exactly 2); structural size caps on
  every dict / list payload field; filename + variable-name
  sanitization at every prompt-injection surface.
- Runtime libraries (R + five Stata `.ado` files) with
  JSON-escaped labels and CR/LF/TAB handling.
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
  picker, drag-drop file/image upload, typewriter assistant, Lottie
  cat loading indicator, status line, Permission/Model chips with
  popups, image-paste support.
- Packaging: `.app` launches the web UI directly (no Terminal popup),
  `.dmg` build pipeline. Local-only until Apple Developer Program
  signing is in place.
- 283 tests, ~3,600 Hypothesis-generated adversarial cases. Pushed
  to [github.com/junishka/builder](https://github.com/junishka/builder).

## What's remaining (prioritized)

### 1. Security hardening (1–2 sessions)

- **Tighten `/private/etc` reads.** Current profile allows the
  whole `/private/etc` subtree. R/Stata only actually need a small
  set of config files (`hosts`, `localtime`, `resolv.conf`,
  `protocols`). Replace the subpath with literals for those files.
  Closes reads of `/etc/passwd` and similar through the
  result-payload exfil channel.
- **Runtime-library contract.** Today a malicious script can write
  hand-crafted JSON directly to `NORA_RESULT_PATH`, bypassing
  the runtime library. Fix options, ordered by the strength of
  guarantee they actually provide:
  - **(a) Stricter sanitizer structural checks** that reject
    payloads without a runtime-library-shaped signature. Weakest:
    a determined attacker can replicate the shape.
  - **(b) Per-run token** the runtime library embeds in every
    payload; executor validates. **Raises attacker cost** — the
    trivial write-to-NORA_RESULT_PATH bypass stops working. Does
    **not** provide a strong guarantee: R closures are
    introspectable, so a script that knows the architecture can
    find the token in the library's loaded environment. Useful
    interim measure.
  - **(c) Pre-opened fd** the subprocess inherits but can't
    discover by path. Structural fix — the path Claude's code
    knows about simply isn't the fd the library writes through.
    Significantly more work in R.

  Plan: **ship (b) first** for the cost-raising benefit. Commit to
  **(c) as the follow-on** if the runtime-authenticity concern
  matters for researchers handling more sensitive data than
  current pilots.

### 2. Researcher consent UI for schema depth — *done*

Schema depth is now an explicit researcher policy in
`<cwd>/.nora/policy.json` rather than a code default. Each
dataset has a per-file `max_depth` ceiling (or inherits
`default_max_depth`); `get_schema` denies requests above the
ceiling, annotates successful responses with the current
`policy_max_depth` so Claude knows the limit without probing.
Malformed policy files fall back to the default silently — a
broken file can't lock the researcher out.

**Depths (least to most permissive):**
- `names_only` — variable names only.
- `names_types` — + a coarse type per variable.
- `names_types_labels` — + variable labels and value labels.
- `names_types_labels_summary` — + per-variable NA counts and
  distinct-value counts for categoricals. **Default.** (The
  default was raised from `names_types` once the per-dataset
  Permission UI made it cheap for researchers to dial it down
  for any dataset where the labels / counts are sensitive.)

Never at any depth: raw values, min, max, median, individual
observations. Those belong to `request_data` (with its own SDC
rules) and `submit_script` (sanitized via the result pipeline).

Interactive editing now exists in both frontends:

- **Terminal:** `/policy` slash-command (handled in `app.py`)
  opens a dataset picker + depth menu; changes persist immediately
  to `.nora/policy.json`. Startup banner still lists each
  dataset with its ceiling and `explicit` vs `default` source.
- **Web UI:** compact "Policy" chip beside the Send button; click
  unfurls a popup with per-dataset dropdowns. Changes write
  through the same bridge method (`set_dataset_policy`) as the
  terminal path.

The JSON file remains the single source of truth; both UIs just
read and write it, so a researcher who prefers hand-editing can
keep doing that. Unknown depths / malformed entries silently
fall back to the conservative default — a broken policy never
locks anyone out.

Also covered: ceiling annotation on successful responses (so
Claude learns the limit without probing), per-dataset
independence (each dataset has its own ceiling), explicit-vs-
default distinction in denial messages.

Still on the list: automatic prompt on first-open of an
un-policy'd dataset (currently the conservative default just
applies silently and the chip reflects it).

### 3. Packaging to `.dmg` (2–4 sessions)

The "install must be double-click" rule has been owed since early
in the project. Current path (`uv sync` + `uv run python -m
`nora`) is a developer workflow.

Approach:
- PyInstaller (or py2app / Briefcase — decision open) to a `.app`
  bundle.
- Notarize + sign; distribute as `.dmg`.
- R must either be bundled or instructed-to-install (Homebrew cask
  redirect). Stata stays user-installed — commercial license.
- First-run UX: pick data dir, set schema policy, configure
  Claude auth.

### 4. One real researcher on real data

The most important missing signal. The user (a quantitative
researcher) is the cheapest researcher #1 — they have real data,
real analytical questions, and they've built the thing so they can
surface UX issues in a single afternoon. A colleague or two as #2
and #3 validates whether the tool works for someone who *didn't*
build it.

### 5. Web UI polish — follow-ons from the first test run

The pywebview shell ships. Still to do, ordered by how often the
current friction bites:

- **Drag-and-drop / file upload instead of picking a directory
  path.** First researcher feedback: *"in the future we should be
  just able to upload the data instead of choosing path."*
  Implementation sketch: launch the app into a "no data yet" state
  with a drop zone; on drop, copy the files into a managed dir
  (e.g. `~/Library/Application Support/Nora/sessions/<id>/`)
  and use that as cwd. Avoids the `~/Users/bb/…` path-expansion
  class of mistake entirely, and doesn't expose a whole project
  directory to the sandbox just to give Claude two files.
- **Markdown-rendered assistant text with tables and code blocks.**
  *Done.* `src/nora/web/markdown.js` is an in-tree renderer that
  covers paragraphs, headings, fenced code, inline code, bold /
  italic, lists, blockquotes, HTTPS links, and GitHub-flavored
  pipe tables (added after researcher feedback that coefficient
  tables rendered as raw pipes). No CDN dependency — keeps
  "nothing phones home" intact.
- **Inline raw R/Stata output panel in the web UI.** *Done.*
  `tool_result` events carry the first 32 KB of `stdout.log` and
  `stderr.log`; the result panel renders them above the collapsed
  sanitized JSON, mirroring the terminal split. Action buttons
  ("Open output", "Open in Stata/R", "Show folder") let the
  researcher launch the native app on the staged script with one
  click.
- **Policy editing in the UI.** *Done.* Compact "Policy" chip in
  the composer footer unfurls a per-dataset dropdown popup. Shares
  the same `set_dataset_policy` bridge method as the terminal's
  `/policy` wizard.
- **Dataset picker sidebar / session list.** *Done.* Left rail in
  the web UI lists every session under `~/.nora-sessions/` with
  timestamp + dataset label + on-disk size; click switches into
  the session, the chat replays from `chat_history.jsonl`, and the
  warm-start prefix injects the recent turns + recent results so
  the model picks up where the conversation left off. Sidebar is
  collapsible and drag-to-resize.
- **Bundling web assets into the PyInstaller `.app`.** *Done.*
  The spec lists `src/nora/web/` (HTML / JS / CSS / Lottie / vendored
  player) as data files; the .app's bundle entry is
  `__main_ui__.py` which calls `nora.ui:main`, so a double-click
  opens the pywebview chat window directly with no Terminal popup.
  Logs go to `~/Library/Logs/Nora/nora-YYYY-MM-DD.log` for
  debugging when it fails to start. The .dmg pipeline produces a
  working bundle locally; what's still missing is the Apple
  Developer Program signature for distribution to other people.
- **Turn-state discipline in the web UI.** *Done* (after
  feedback that the Send button was re-enabling too early). The
  bridge's `send_message` is fire-and-forget by design; the web
  UI now latches `turnInFlight` on submit and only clears it
  when `turn_done` / `turn_error` / `auth_failure` arrives, so a
  quick tester can't pipeline prompts that interleave in the
  transcript.

## Known-real, design-pending

### Cumulative-inference / cross-query composition

**Status as of the current pilot:** named, design-pending, and
*not* the right thing to spend time on yet. The current and
intended near-term mode is a researcher running Nora against their
own data with their own API key — adversarial Claude and
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
adversarial Claude — who issues 200 individually-compliant queries
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

- Claude has no general-purpose tools. SDK built-ins stay
  disabled.
- Claude's only interface to the machine is the 6 MCP tools
  (`get_schema`, `request_data`, `submit_script`, `expand_result`,
  `list_results`, `recall_conversation`).
- Every `submit_script` call runs under the sandbox.
- Every executor output passes through the sanitizer before
  reaching the model.
- Raw stderr / stdout never reach the model.
- Schema exposure is explicit researcher policy, conservative by
  default.
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
  Claude — especially Stata, where the open training corpus is
  thin. Privacy benefit is narrow (closes the
  frontier-authored-code channel for adaptive attackers) but does
  not address cumulative-inference / adaptive-probing risks, which
  are inherent to interactive analysis regardless of authorship
  and are handled by session-level disclosure budgets and the
  SDC rules. Stays available as a future optional helper for:
  error recovery using raw stderr (which Claude can't see),
  text-data redaction so free-text values can flow through the
  sanitizer, and quality improvements in specific edge cases.
  Re-enters the discussion when a real researcher's task actually
  needs one of these.

- **Mandatory safe variable IDs as the frontier-facing identity
  surface.** Schema exposure is policy, not architecture. If a
  researcher's dataset has non-sensitive variable names and they
  opt in to sharing them with Claude, that's their call. Nora
  enforces conservative defaults and makes the choice visible; it
  doesn't enforce a ceiling.

- **Language-specific hybrid (Claude writes Stata, local model
  writes R/Python).** The premise — that local models are weak on
  Stata — is true, but Option A has Claude writing all three
  languages directly. The hybrid solves a problem we don't have.

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

- **Default schema depth.** names+types (safest) vs names+types+
  labels (more useful). Labels are sometimes disclosive (rare
  diagnosis codes, specific named conditions). Leaning:
  names+types default, with an opt-in prompt for labels during
  first-dataset-open.
- **Packaging framework.** PyInstaller vs py2app vs Briefcase.
  Decided when the packaging work actually starts.
- **When to revisit local LLM.** Rule of thumb: when a real
  researcher asks for something only a local model can provide
  (text-data analysis, stderr-based repair of a specific recurring
  failure mode). Not before.

## What must not happen

- Claude gains general-purpose tools at any stage.
- Sandbox or sanitizer becomes conditional on a flag.
- Raw stderr / stdout reaches Claude.
- Schema-exposure defaults widen silently.
- Runtime-library bypass remains open indefinitely.

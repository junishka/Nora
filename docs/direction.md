# Builder — architectural direction

Working document, not a locked-down spec. Captures the decision to
pivot from free-form script submission (current state) to a
plan-based architecture with a local LLM translator. Settled after
the 2026-04-20 reviewer pass revealed that the sandbox was carrying
too much of the privacy guarantee alone.

## Where we are today

The current implementation (tag-of-the-initial-commit) is a strong
privacy prototype:

- Frontier (Claude) has a 5-tool MCP surface. All SDK built-ins
  disabled. Four-layer lockdown on tool use.
- Schema extractor, executor with `(deny default)` subpath-allowlist
  sandbox, SQLite result store, SDC sanitizer across six analysis
  families (linear_regression, t_test, descriptive, frequency_table,
  crosstab, magnitude_table) with R + Stata parity.
- 152 tests, including ~3,600 Hypothesis-generated adversarial cases
  and 19 pure SBPL unit tests locking in the sandbox profile shape.

It works end-to-end against real R and Stata. Sensitive reads
(`~/.zshrc`, `/Library/Keychains/System.keychain`,
`/private/var/log/system.log`) are denied. Claude can submit R/Stata
scripts and get sanitized results back.

It has one architectural weakness: **the sandbox is the primary
privacy guarantee, not a defense-in-depth layer**. A malicious script
can:

- Bypass the runtime library by writing hand-crafted JSON directly to
  `BUILDER_RESULT_PATH`.
- Encode contents of any readable file into sanitizer-allowed fields
  (coefficient names, labels) and smuggle them out through the result
  payload that Claude sees.

Both are bounded (the sandbox narrows what's readable, the sanitizer
caps string length and strips control chars), but the trust anchor is
"the sandbox holds" rather than "Claude cannot author arbitrary
instructions." That's the wrong anchor for a privacy product.

## Where we're going

Claude stops writing code. Claude writes **detailed analysis plans**
against safe variable identifiers. A **local translator** (existing
open-weight LLM, routed through Ollama or similar) compiles the plan
into R or Stata locally. The executor runs the compiled code under
the existing sandbox. The sanitizer gates everything that returns.

The trust anchor moves from "sandbox holds" to a three-layer stack:

1. **Boundary** — Claude cannot emit executable code. The MCP
   interface accepts plans, not scripts.
2. **Sandbox** — defense in depth. Compiled code still runs under the
   `(deny default)` subpath-allowlist profile.
3. **Sanitizer** — unchanged. Every result payload still flows
   through SDC + text-safety before reaching Claude.

Each layer is independent. No single failure kills the privacy story.

## Architectural invariants (non-negotiable)

- Frontier has no general-purpose tools. SDK built-ins stay disabled.
- Frontier cannot author executable code that runs locally.
- Frontier can only submit plans through the custom MCP interface.
- Raw data, raw paths, raw variable names, raw stderr stay local.
- Frontier sees **safe variable IDs** plus sanitized display names.
  Raw column names resolve to IDs inside the local translator.
- Every execution output passes through the sanitizer before
  reaching the frontier.
- Researcher sees raw logs, compiled script, diffs, sanitized result.
  Frontier sees sanitized result only.
- Sandbox is mandatory (unchanged), but defense in depth.

## System shape

- **Frontier model (Claude).** Reads schema and bounded metadata;
  produces rich, detailed analysis plans; does not write code.
- **Schema layer.** Extracts schema locally; returns safe variable
  IDs, sanitized display names, types, labels per policy.
- **Plan interface (MCP tools).** `get_schema`, `request_data`,
  `preview_plan`, `submit_plan`, `expand_result`, `list_results`. No
  tool accepts arbitrary code.
- **Local translator.** Existing open-weight LLM (candidates:
  Qwen2.5-Coder-14B-Instruct via Ollama, DeepSeek-Coder-V2-Lite).
  Resolves safe IDs to raw names. Generates R/Stata. May do bounded
  local repair using local stderr.
- **Executor.** Unchanged. `(deny default)` subpath-allowlist
  sandbox. Refuses to run without sandbox-exec.
- **Sanitizer.** Unchanged. SDC rules + text-safety, reject by
  default.
- **Store.** Extended to persist submitted plan, compiled script,
  repair diff (if any), sanitized result, raw log path.

## Plan envelope

Structured, not tiny. Real research is not a six-enum grammar.
Envelope supports:

- `analysis_type`
- `dataset_id`
- `outcome_variables`, `predictor_variables`, `grouping_variables`
- `filters`, `transformations`, `interactions`
- `fixed_effects`, `variance_estimator`, `cluster_variables`
- `reference_levels`, `missing_data_policy`
- `table_options`, `output_requests`
- `research_intent` (prose — see decisions below)

The rule is: Claude expresses intent richly but submits no
executable code.

## Decisions (as of 2026-04-20)

- **Local LLM on the core path**, not optional. The privacy gain
  comes from moving code authorship local, not from crippling the
  plan format.
- **Use an existing model** via Ollama; do not build or bundle one.
  First candidate: Qwen2.5-Coder-14B-Instruct. Researchers can swap.
- **No plan validator at v1.** Sandbox + sanitizer are the two
  structural layers. Add a validator later if the translator's
  output surprises us in practice.
- **Script preview visible by default**, not a mandatory
  click-through. Researchers inspect plan → compiled diff as part of
  the normal audit UX. Mandatory confirmation becomes dialog-
  blindness within days.
- **`research_intent` is prose the translator reads.** Claude can
  give step-by-step instructions in natural language; the local
  model compiles them. The privacy guarantee comes from the stacked
  boundary (sandbox + sanitizer), not from restricting Claude's
  language.
- **Default schema depth: safe IDs + types.** Labels, level names,
  value labels opt-in per dataset. Sensitive domains stay tight by
  default; researchers widen explicitly.
- **Stderr data-leakage risk in local repair: defer.** Revisit only
  if the repair loop shows data values landing in stored scripts.

## Retirement rule for `submit_script`

Current free-form `submit_script` becomes a developer-only flag
during migration. Retires from production when **all four** hold:

1. `submit_plan` supports the six current analysis families in R.
2. ≥95% of tasks in a curated research-task corpus compile and run
   via `submit_plan` without researcher escape.
3. Local repair is stable — execution recovery is not materially
   worse than the free-form path today.
4. Remaining unsupported workflows are explicitly documented.

Name a specific threshold. Without one, the dev flag becomes
permanent.

## Build order

1. **Pick the model.** Qwen2.5-Coder-14B-Instruct via Ollama is the
   starting candidate. Decide this before writing any translator
   code — the product's usability floor is the model's coding
   quality.
2. **Safe variable IDs in schema extraction.** Stop emitting raw
   column names across the boundary. Schema returns `variable_id`,
   sanitized display name, type, labels per policy.
3. **`submit_plan` + `preview_plan` MCP tools.** Accept structured
   plans; store them; surface the compiled preview back to the
   researcher.
4. **Local translator, first pass, R only.** Descriptive and
   frequency_table only. Prove the shape small.
5. **30–50 task corpus.** Real plans from real research questions.
   Measure first-try success rate. Target >80% without repair. If
   <70%, the model choice needs rework before scaling.
6. **Bounded local repair** using local stderr. Bounded iteration
   count; every attempt stored in provenance.
7. **Expand to inference families.** t_test, linear_regression,
   robust SE, one-way clustering.
8. **Richer expressivity.** Transformations, interactions, filters,
   fixed effects.
9. **Stata translator.** After the R path stabilizes.
10. **Retire `submit_script`.** When the four retirement conditions
    hold.

## Load-bearing UX details

- **Script preview is critical.** With an intelligent translator,
  its output can silently deviate from Claude's plan. Preview +
  plan-to-script diff is how the researcher catches hallucinations
  and wrong variable resolution.
- **Usability floor = local model's coding quality.** Researchers
  coming from a Claude-written-R product will feel the downgrade if
  the local model generates brittle code. Repair has to absorb most
  of the gap. This is the biggest risk in the plan and the thing to
  validate first with the task corpus.

## Open questions for the build

- Which bounded-repair policy (max iterations, what counts as a
  successful repair, when to surface the repair to the researcher)?
- Where does the variable-ID ↔ raw-name mapping live in session
  state? Per-plan, per-session, per-dataset?
- How is the task corpus curated and versioned? It's the measuring
  stick for "does this work" — needs ownership.
- Which Ollama model version gets pinned at install? Model upgrades
  change output; provenance should capture it.

## What must not happen

- Frontier regains free-form code submission in production.
- Raw variable names become frontier-facing identifiers.
- Local translator edits the compiled code without the diff being
  visible to the researcher.
- Raw stderr reaches the frontier.
- Sandbox becomes the only privacy story again.
- The plan envelope is made so small that useful research workflows
  cannot be expressed.

## What this delays

Step 8 ("one real researcher on real data") pushes out by several
weeks at minimum. The current prototype is runnable from source
today; a researcher could use it this afternoon. But shipping v0 to a
real researcher and later discovering a boundary breach is a worse
outcome than delaying. The redesign is the right call. The timeline
cost is real.

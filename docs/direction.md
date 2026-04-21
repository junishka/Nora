# Builder — architectural direction

Working document. Last updated 2026-04-20 after reviewer feedback
and the decision to stay with the current script-submission
architecture ("Option A") rather than pivot to a plan-submission
architecture with a local LLM translator.

See [`docs/overview.md`](overview.md) for a plain-language
description of what Builder is and why it exists. This document is
about what the architecture is, what it isn't, and what's next.

## The decision

**Keep the current architecture.** Remaining work is hardening, UX,
and real-researcher contact — not restructuring.

A recent external reviewer proposed replacing Claude's direct
script authorship with a plan-submission architecture where Claude
writes structured plans and a bundled local LLM compiles them into
R/Stata. After working through the argument, that proposal was
rejected as over-correction.

The privacy guarantee in Builder comes from three independent
layers:

1. **The tool interface** — Claude can only do five things
   (`get_schema`, `request_data`, `submit_script`, `expand_result`,
   `list_results`). No filesystem, no shell, no network.
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

As of 2026-04-20, the implementation covers:

- Spine + full SDK lockdown (5 MCP tools, every built-in disabled,
  four defense layers).
- Schema extractor for `.csv` / `.dta` / `.rds` with configurable
  depth tiers.
- Executor with `(deny default)` subpath-allowlist sandbox. Pure
  unit tests lock in the SBPL profile shape; integration tests
  verify real sandbox behavior (gated on `Rscript` + sandbox-apply
  preflight).
- Sanitizer across six analysis families (linear regression,
  t-test, descriptive, frequency table, crosstab, magnitude table)
  with full R + Stata parity.
- Runtime libraries (R + five Stata `.ado` files) with
  JSON-escaped labels and CR/LF/TAB handling.
- SQLite result store.
- 152 tests, ~3,600 Hypothesis-generated adversarial cases. Pushed
  to [github.com/junishka/builder](https://github.com/junishka/builder).

Full implementation status in
`memory/project_builder_current_state.md`.

## What's remaining (prioritized)

### 1. Security hardening (1–2 sessions)

- **Tighten `/private/etc` reads.** Current profile allows the
  whole `/private/etc` subtree. R/Stata only actually need a small
  set of config files (`hosts`, `localtime`, `resolv.conf`,
  `protocols`). Replace the subpath with literals for those files.
  Closes reads of `/etc/passwd` and similar through the
  result-payload exfil channel.
- **Runtime-library contract.** Today a malicious script can write
  hand-crafted JSON directly to `BUILDER_RESULT_PATH`, bypassing
  the runtime library. Fix options, easiest to hardest: (a)
  stricter sanitizer structural checks that reject payloads
  without a runtime-library-shaped signature; (b) per-run token
  the runtime library embeds in every payload, executor validates;
  (c) pre-opened fd the subprocess inherits but can't discover by
  path. Pick (a) or (b) first.

### 2. Researcher consent UI for schema depth (1–2 sessions)

Currently schema depth is a code default. Move to an explicit
per-dataset policy file (`.builder/policy.json` or equivalent)
with conservative defaults:

- **Default:** variable names + types.
- **Opt-in:** variable labels.
- **Opt-in:** categorical level names.
- **Opt-in:** 5th/95th numeric bounds.
- **Never:** raw values, min, max, median, individual observations.

Surface the policy in the TUI when a dataset is first opened so
the researcher makes an explicit choice rather than inheriting a
default silently.

### 3. Packaging to `.dmg` (2–4 sessions)

The "install must be double-click" rule has been owed since early
in the project. Current path (`uv sync` + `uv run python -m
builder`) is a developer workflow.

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

## Invariants (non-negotiable)

- Claude has no general-purpose tools. SDK built-ins stay
  disabled.
- Claude's only interface to the machine is the 5 MCP tools.
- Every `submit_script` call runs under the sandbox.
- Every executor output passes through the sanitizer before
  reaching Claude.
- Raw stderr / stdout never reach Claude.
- Schema exposure is explicit researcher policy, conservative by
  default.
- Researcher sees raw logs and sanitized output; Claude sees
  sanitized output only.

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
  opt in to sharing them with Claude, that's their call. Builder
  enforces conservative defaults and makes the choice visible; it
  doesn't enforce a ceiling.

- **Language-specific hybrid (Claude writes Stata, local model
  writes R/Python).** The premise — that local models are weak on
  Stata — is true, but Option A has Claude writing all three
  languages directly. The hybrid solves a problem we don't have.

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

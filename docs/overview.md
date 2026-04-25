# Nora — overview

Plain-language description of what Nora is and the architectural
choice behind the current design. Intended for people who haven't
been in the design conversation.

## What Nora is

Nora is a tool that sits on a researcher's own computer and lets
them use Claude (or another AI assistant) to analyze sensitive data
— medical records, HR data, survey responses, IRB-restricted
research — without sending that data to any third party.

The problem it solves: AI assistants are useful for data analysis,
but using one normally means either uploading the data or
describing it in detail to the assistant. Both expose the data.
Researchers with confidential datasets legally or ethically can't
do that. Nora is a thin local layer that lets Claude help with
analysis while the actual values stay on the researcher's machine.

## How it works

The researcher either drags the data files into Nora's window
(web UI: `nora-ui`) or points Nora at a directory (terminal
UI: `nora`). Supported formats are `.csv`, `.dta` (Stata), and
`.rds` (R). A chat starts with Claude through Nora; Claude is
then restricted — no filesystem access, no shell, no network tools.

Instead, Claude has exactly six operations, through a narrow tool
interface:

1. **Ask for the schema** of a dataset — variable names, types,
   and (optionally) labels.
2. **Ask bounded questions** about a variable — how many
   categories does this have? What's the rough 5th-to-95th
   percentile range?
3. **Submit an R or Stata script** to analyze the data.
4. **See previous results.**
5. **Expand a specific result** for more detail.
6. **Recall earlier turns** of the conversation — Nora persists
   the chat log to disk and Claude can search older turns when
   the auto-loaded recent window isn't enough.

When Claude submits a script, Nora runs it locally in a
**sandbox** that blocks network access and restricts which files
the script can read (only the researcher's data directory plus the
paths R/Stata need to start up). The output of the script passes
through a **sanitizer** that applies statistical disclosure control
rules: it rounds coefficients based on sample size, suppresses
cells with fewer than 10 observations, never reveals individual
observations like min/max, and never passes through raw text values
from the data.

The researcher sees the full raw script output in the chat window
(in a result panel under each script run). Claude sees only the
sanitized version.

Nora also remembers the conversation across restarts. Every turn
is persisted to a per-session log file; when Claude opens a fresh
client (after the researcher closes and reopens, switches sessions,
or changes models), the recent turns plus a list of stored
analytical results are auto-injected as the warm-start prefix so
the conversation picks up where it left off. Older turns that have
fallen out of that window can be retrieved on demand via the
`recall_conversation` tool.

The load-bearing property: **Claude never directly touches the
data.** It writes questions about the data (as code) and gets back
privacy-filtered answers. The boundary is enforced by three
independent layers — the tool interface, the sandbox, and the
sanitizer.

## The architectural choice

A question that came up during design review: should Nora
include a **local AI model** — an open-source coder model like
Qwen3-Coder-30B running on the researcher's own laptop — as part
of the pipeline?

### Option A — no local model (current architecture)

The pipeline: Claude writes R/Stata → local sandbox runs it →
sanitizer filters the output → Claude sees the sanitized result.

**What this gets right:**
- Claude is an excellent coder. Scripts are usually correct on the
  first try.
- Installation is light. Python + whatever R or Stata the
  researcher already has. No gigabyte-scale model downloads.
- Quality is as high as Claude can make it, which is higher than
  any open-source model.

**What's missing:**
- When a script fails, Claude is debugging blind. The raw error
  message stays on the researcher's machine for privacy reasons —
  so Claude can't see exactly what went wrong. It has to guess
  from limited context.
- Nora can't yet analyze text-heavy data — survey open-ends,
  clinical notes, legal documents. The sanitizer currently rejects
  free-text values as too risky to pass through without a way to
  redact PII.
- A theoretically-adversarial Claude could encode small amounts of
  non-sensitive system info (like usernames from a system config
  file) into result fields. Low bandwidth, contained by the
  sandbox's file-access allowlist, but real.

### Option B — add a local model

The pipeline adds a step: an open-source coder model (Qwen3-Coder
or similar) running on the researcher's laptop joins the loop.

Variants:
- **B1:** The local model writes code from Claude's plans. Claude
  never writes executable code.
- **B2:** Claude still writes code; the local model handles
  recovery when execution fails (it can see raw stderr that Claude
  can't) and redacts text values so free-text data can flow
  through.

**What it gets:**
- Claude gets a debugging partner for failed scripts.
- Text data becomes analyzable via local-model redaction.
- Under B1, a narrow additional privacy property: an adaptive,
  adversarial Claude can no longer choose exactly which
  computation runs on the data. That's a real gain, but it's
  bounded — it doesn't address the harder problem of cumulative
  inference across many sanitized queries (the "20 questions"
  attack), which is inherent to any interactive analysis system
  regardless of who authored the code.

**What it costs:**
- Install gets much bigger — 15–17 GB for a reasonable model at
  4-bit quantization.
- Researcher needs a laptop that can run it (16 GB+ RAM, recent
  Apple Silicon or equivalent).
- Local models are worse coders than Claude, especially in Stata
  where the open training corpus is thin. Researchers may notice
  the quality dip.
- More moving parts to build, test, and maintain.

## Why Option A was chosen

The key realization: **the core privacy guarantee doesn't depend on
which option is chosen.** In both, Claude can't touch the data
directly — that's enforced by the tool interface, the sandbox, and
the sanitizer. Those three layers work identically either way.

A local model would close one narrower channel (an adaptive
Claude choosing exactly which computations run) and would add
capability (debugging help, text-data handling). It would not
address the harder privacy problem — cumulative inference across
many sanitized queries — which is inherent to any interactive
analysis system and is handled by session-level budgets and SDC
rules, not by code authorship.

So the privacy benefit of Option B is *limited* rather than
transformative, and the costs (install footprint, RAM, worse code
quality, maintenance) are concrete.

The pragmatic decision:

- **Ship Option A now.** The privacy properties are real, the
  architecture is tested, and it's usable by a real researcher on
  real data as soon as the remaining polish work is done.
- **Treat the local model as a future enhancement**, re-entering
  the discussion when a concrete task demands it (e.g. "I need to
  analyze my free-text survey responses" would make text-redaction
  via a local model the specific blocker).
- **Resist over-architecting.** The external reviewer's proposal
  to replace Claude's code authorship entirely with a local-model
  pipeline was a larger swing than the privacy gains justified,
  given that the sandbox and sanitizer already enforce the
  important boundary.

## What stands between Option A and a usable tool

What's done since this overview was first written:

- ✅ Security hardening: env-var allowlist (subprocess can't see
  `ANTHROPIC_API_KEY` or shell secrets), per-cwd store binding (no
  cross-session leak), filename / variable-name sanitization at
  every prompt-injection surface, OLS coefficient-key constraint,
  confidence-interval length constraint, structural size caps on
  every dict / list payload field.
- ✅ Researcher consent UI: per-dataset Permission chip in the
  web UI; `/policy` slash-command in the terminal UI. Both edit
  the same `.nora/policy.json`.
- ✅ Packaging: `.app` is built and launches the web UI directly
  (no Terminal popup). Build pipeline produces a `.dmg` too. Both
  work locally.

What's left:

1. **Apple Developer Program signing** so the .dmg can actually be
   handed to a researcher. Without it, Gatekeeper refuses unsigned
   apps cleanly enough that the right-click-Open workaround is
   real friction. $99/yr cert.
2. **One real researcher on real data.** The most important
   missing signal. Everything else is preparation for it.

The signing is paperwork-and-dollars; the pilot is where the real
feedback lives.

# Nora — overview

Plain-language description of what Nora is and the architectural
choice behind the current design. Intended for people who haven't
been in the design conversation.

## What Nora is

Nora is a tool that sits on a researcher's own computer and lets
them use an AI assistant to analyze sensitive data
— medical records, HR data, survey responses, IRB-restricted
research — without sending that data to any third party.

The problem it solves: AI assistants are useful for data analysis,
but using one normally means either uploading the data or
describing it in detail to the assistant. Both expose the data.
Researchers with confidential datasets legally or ethically can't
do that. Nora is a thin local layer that lets the assistant help
with analysis while the actual values stay on the researcher's
machine.

## How it works

The researcher drags the data files into Nora's window or points
Nora at a directory on launch. Supported formats are `.csv`, `.tsv`,
`.dta` (Stata), `.rds` (R), `.parquet`, `.jsonl`, and `.ndjson`. A
chat starts through Nora; the model is then restricted — no
filesystem access, no shell, no network tools.

Instead, the model reaches the researcher's machine through fourteen
narrow operations:

1. **Ask for the schema** of a dataset — variable names, types,
   labels, and summary metadata within the dataset's policy ceiling.
2. **Search the schema** by name or label substring on wide
   datasets, instead of pulling the full schema.
3. **Ask bounded questions** about a variable — how many
   categories does this have? What's the rough 5th-to-95th
   percentile range? What's the correlation with this other
   variable?
4. **Submit an R, Stata, or Python script** to analyze the data.
5. **Submit a script from a file** the researcher attached, so
   the script bytes don't have to round-trip through the model.
6. **Expand a stored result** for more detail (full payload,
   trimmed to coefficients, or rendered as a canonical markdown
   table).
7. **List results** in this session.
8. **Compose multiple stored results** into one canonical table.
9. **List results across sessions** when the researcher enables
   cross-session recall.
10. **Recall earlier turns** of the conversation — Nora persists
   the chat log to disk and the model can search older turns when
   the auto-loaded recent window isn't enough.
11. **Re-fetch an attached file** the researcher mentioned
    earlier (script text or image), so the model can act on it
    without asking the researcher to re-attach.
12. **List files in the session** such as scripts, logs, and plots.
13. **Search session files** without opening unrelated files.
14. **Install analysis packages** through the controlled package
    installer when the researcher approves that workflow.

When the model submits a script, Nora runs it locally in a
**sandbox** that blocks network access and restricts which files
the script can read (only the researcher's data directory plus the
paths R/Stata need to start up). The output of the script passes
through a **sanitizer** that applies statistical disclosure control
rules: it rounds coefficients based on sample size, suppresses
cells with fewer than 10 observations, never reveals individual
observations like min/max, and never passes through raw text values
from the data.

The researcher sees the full raw script output in the chat window
(in a result panel under each script run). The model sees only the
sanitized version.

Nora also remembers the conversation across restarts. Every turn
is persisted to a per-session log file; when the model opens a fresh
client (after the researcher closes and reopens, switches sessions,
or changes models), the recent turns plus a list of stored
analytical results are auto-injected as the warm-start prefix so
the conversation picks up where it left off. Older turns that have
fallen out of that window can be retrieved on demand via the
`recall_conversation` tool.

The load-bearing property: **the model never directly touches the
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

The pipeline: the model writes R/Stata/Python → local sandbox runs
it → sanitizer filters the output → the model sees the sanitized
result.

**What this gets right:**
- Frontier models are strong coders. Scripts are usually correct on the
  first try.
- Installation is light. Nora uses the analysis runtime(s) the
  researcher already has. No gigabyte-scale model downloads.
- Quality is as high as the selected model can make it, which is higher than
  any open-source model.

**What's missing:**
- When a script fails, the model is debugging blind. The raw error
  message stays on the researcher's machine for privacy reasons —
  so it can't see exactly what went wrong. It has to guess
  from limited context.
- Nora can't yet analyze text-heavy data — survey open-ends,
  clinical notes, legal documents. The sanitizer currently rejects
  free-text values as too risky to pass through without a way to
  redact PII.
- A theoretically-adversarial model could encode small amounts of
  non-sensitive system info (like usernames from a system config
  file) into result fields. Low bandwidth, contained by the
  sandbox's file-access allowlist, but real.

### Option B — add a local model

The pipeline adds a step: an open-source coder model (Qwen3-Coder
or similar) running on the researcher's laptop joins the loop.

Variants:
- **B1:** The local model writes code from the frontier model's
  plans. The frontier model never writes executable code.
- **B2:** The frontier model still writes code; the local model handles
  recovery when execution fails (it can see raw stderr that the
  frontier model can't) and redacts text values so free-text data
  can flow through.

**What it gets:**
- The frontier model gets a debugging partner for failed scripts.
- Text data becomes analyzable via local-model redaction.
- Under B1, a narrow additional privacy property: an adaptive,
  adversarial model can no longer choose exactly which
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
- Local models are worse coders than frontier models, especially in
  Stata where the open training corpus is thin. Researchers may
  notice the quality dip.
- More moving parts to build, test, and maintain.

## Why Option A was chosen

The key realization: **the core privacy guarantee doesn't depend on
which option is chosen.** In both, the model can't touch the data
directly — that's enforced by the tool interface, the sandbox, and
the sanitizer. Those three layers work identically either way.

A local model would close one narrower channel: an adaptive model
choosing exactly which computations run. It would also add
capability such as debugging help and text-data handling. It would
not address the harder privacy problem — cumulative inference
across many sanitized queries — which is inherent to any interactive
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
  to replace frontier-model code authorship entirely with a local-model
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
  composer row, editing `.nora/policy.json`.
- ✅ Packaging: `.app` is built and launches the web UI directly
  (no Terminal popup). The release `.dmg` is signed and notarized.

What's left:

1. **One real researcher on real data.** The most important
   missing signal. Everything else is preparation for it.

The pilot is where the real feedback lives.

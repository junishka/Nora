# Nora overview

Plain-language description of what Nora is and how it works.
Current version: **0.10.2** (May 2026).

## What Nora is

Nora is a macOS app that sits on a researcher's own computer and
lets them use an AI assistant to analyze sensitive data without
sending that data to any third party. Medical records, HR data,
survey responses, IRB-restricted research. The data stays on the
researcher's machine.

The problem it solves. AI assistants are useful for data analysis,
but using one normally means uploading the data or describing it
in detail. Both expose the data. Researchers with confidential
datasets often can't do that, legally or ethically. Nora is a
thin local layer that lets the assistant help with the analysis
while the actual values stay on disk.

## How it works

The researcher drags data files into Nora's window or points it
at a directory on launch. Supported formats are `.csv`, `.tsv`,
`.dta` (Stata), `.rds` (R), `.parquet`, `.jsonl`, and `.ndjson`.

A chat starts inside Nora. The researcher can use Claude
(Anthropic) or GPT (OpenAI). The model has no filesystem access,
no shell, no network. It reaches the data only through a narrow
set of operations: ask about the schema, submit an R / Stata /
Python script, read sanitized results, compose multiple results
into a comparison table.

When the model submits a script, Nora runs it locally in a
sandbox. The sandbox blocks network access and restricts which
files the script can read (the researcher's data directory plus
the paths R, Stata, or Python need to start up). The script's
output then passes through a sanitizer that applies statistical
disclosure control rules. It rounds coefficients based on sample
size, suppresses cells with fewer than 10 observations, never
reveals individual observations like min or max, and never
forwards raw text values from the data.

## What you see vs what the model sees

The researcher sees the full raw script output in the chat
window, in a result panel under each script run. The model sees
only the sanitized version.

Conversations persist across restarts. Every turn is saved to a
per-session log on disk. When the researcher reopens, the recent
turns plus a list of stored analytical results are loaded back
so the conversation picks up where it left off. Older turns that
have fallen out of that window can be retrieved on demand.

## The privacy guarantee

The model never directly touches the data. It writes questions
about the data (as code) and gets back privacy-filtered answers.
The boundary is enforced by three independent layers:

1. **The tool interface.** No filesystem, no shell, no network.
2. **The sandbox.** Network denied, file reads restricted to the
   researcher's data directory and the runtime's startup paths.
3. **The sanitizer.** Statistical disclosure rules applied to
   every output before the model sees it.

## Install and run

Download the signed and notarized `.dmg` from the
[releases page](https://github.com/junishka/Nora/releases) (you
want the latest `0.10.x`). Drag `Nora.app` to `/Applications` and
launch. First launch asks for an Anthropic or OpenAI credential
and stores it in the macOS Keychain.

R, Stata, or Python (with `pandas`) need to be installed
separately. Nora calls them as subprocesses and does not bundle
them.

## Where to read more

For implementation details, the supported analysis shapes, and
the contributor-facing architecture, see
[`docs/handoff.md`](handoff.md) and
[`docs/direction.md`](direction.md). For the full change history,
see [`CHANGELOG.md`](../CHANGELOG.md).

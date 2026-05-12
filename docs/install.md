# Installing Nora

## What you'll need

- **A Mac** — macOS 11 (Big Sur) or later, Intel or Apple Silicon.
- **Analysis runtime(s)** — install the tools you want Nora to run:
  R (`Rscript` on PATH), Stata (`stata-mp` / `stata-se` / `stata`
  on PATH or `/Applications/Stata`), or Python with the packages
  your analysis needs.
- **A model credential** — use an Anthropic account/API key or an
  OpenAI API key. The auth screen stores credentials in the system
  keyring.

## Two install paths

Nora has two ways to run: a released `.dmg` you double-click (the recommended path for researchers) and a from-source clone (for development or local builds). The released `.dmg` is signed with a Developer ID Application certificate and notarized by Apple.

## Path 1 — install the `.dmg` (recommended)

Download the latest `Nora.dmg`, open it, and drag `Nora.app` to `/Applications`. Double-click to launch — the first run goes straight to the auth screen, no right-click-Open dance and no Terminal popup. R, Stata, and Python aren't bundled; Nora invokes the runtime you choose as a subprocess.

## Path 2 — from source

```bash
git clone https://github.com/junishka/Nora.git nora
cd nora
uv sync --group dev
uv run pytest
uv run nora                   # landing screen for files / folder
uv run nora /path/to/data     # open straight into chat
```

`nora` opens a native window with a drop zone for `.csv`, `.tsv`,
`.dta`, `.rds`, `.parquet`, `.jsonl`, and `.ndjson` files. Dropped
files land in `~/.nora-sessions/<timestamp>_<id>/` — a per-session
scratch dir that becomes the sandbox root. Cleanest way to share
exactly the files you want to analyze without exposing a whole
project folder.

## Building the `.dmg` yourself

You can rebuild the bundle locally if you want to ship a custom variant or test packaging changes. The same scripts produce the released artifact when run with the right env vars set.

```bash
git clone https://github.com/junishka/Nora.git nora
cd nora
uv sync --group dev
bash packaging/build_app.sh         # → dist/Nora.app  (~70 MB)
bash packaging/build_dmg.sh         # → dist/Nora.dmg  (~35 MB)
open dist/Nora.app
```

A bare local build is unsigned — fine for testing on the same machine, but Gatekeeper will refuse it on someone else's. To produce a signed + notarized `.dmg`, set `NORA_SIGN_IDENTITY` (Developer ID Application certificate name) before `build_app.sh` and `NORA_NOTARIZE_PROFILE` (a `notarytool store-credentials` keychain profile) before `build_dmg.sh`. Both scripts skip those steps when the env vars are unset.

If you've moved an unsigned `Nora.app` somewhere persistent (`/Applications/`), the first launch hits the standard Gatekeeper dialog:

> *"Nora" cannot be opened because the developer cannot be verified.*

Two workarounds for the unsigned local-build case:
- **Right-click → Open** in Finder. macOS shows a similar dialog but with an **Open** button. Click it. From then on, double-clicking works normally.
- Or from Terminal: `xattr -cr /Applications/Nora.app` then double-click as usual.

## Running Nora

Whichever path you took, the entry point is the same: a chat window with a landing screen the first time, and your sandbox-rooted data thereafter. The first thing it asks is what files you want the assistant to use — drop them on the landing zone, or pick a folder. That selection becomes the sandbox: the model cannot read anything outside it.

Once you're in the chat, you talk. The assistant proposes; you steer. Scripts the model runs are sandboxed (no network, narrow filesystem allowlist), every result is sanitized before the model sees it, and the raw R/Stata/Python output stays visible to you in the result panels.

## Schema policy — controlling what the model sees

The model sees only structural metadata about each dataset, never the rows. The default ceiling is **`names_types_labels_summary`** — variable names, types, value labels, and per-variable NA / distinct-value counts. That's the most informative tier that doesn't leak per-observation data.

You change this via the **Permission chip** in the composer row (a dropdown per dataset). It edits `<your-data-dir>/.nora/policy.json`, which you can also hand-edit:

```json
{
  "version": 1,
  "default_max_depth": "names_types_labels_summary",
  "datasets": {
    "public-survey.csv": {
      "max_depth": "names_types_labels_summary"
    },
    "sensitive-records.dta": {
      "max_depth": "names_only"
    }
  }
}
```

Available tiers, from most private to most permissive:

1. `names_only` — variable names, nothing else.
2. `names_types` — + a coarse type per variable.
3. `names_types_labels` — + variable labels and value labels.
4. `names_types_labels_summary` — + per-variable NA count and distinct-value count for categoricals. **Default.**

**At no tier** does Nora share raw values, min, max, median, or any other per-observation value. Those require a separate tool call (`request_data`) that applies statistical disclosure control rules, or a full analysis via `submit_script` whose output is sanitized before the model sees it.

## Uninstalling

If you used the .app build: drag `Nora.app` from wherever you put it to the Trash. Nora writes per-project state into `.nora/` subdirectories inside the data directories you pointed it at — delete those if you want to remove all traces:

```bash
find ~ -type d -name .nora -print
# review, then remove
```

Plus the per-session scratch dirs the web UI creates:

```bash
rm -rf ~/.nora-sessions/
```

## Troubleshooting

**The .app launches and nothing happens.** The pywebview window failed to come up. Check the log file:

```bash
tail -F ~/Library/Logs/Nora/nora-*.log
```

Most common cause: an import error in the bundled binary. The traceback in the log says which module is missing.

**Terminal opens then closes immediately.** You're running an old build of the .app — recent builds don't open Terminal at all. Rebuild from `main` (`bash packaging/build_app.sh`) and try again.

**`sandbox-exec not found` warning.** Should never happen on macOS — `/usr/bin/sandbox-exec` is part of the OS. If you see this, file a bug.

**No analysis runtime installed warning.** Install R, Stata, or Python packages for the language you want to use, then relaunch. Schema and bounded data queries still work without them, but `submit_script` — the analysis path — won't.

**The assistant asks to see variable labels and gets denied.** That's the schema policy working as intended. If the labels aren't sensitive for this dataset, raise the ceiling via the Permission chip in the composer row.

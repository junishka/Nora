# Installing Nora

## What you'll need

- **A Mac** — macOS 11 (Big Sur) or later, Intel or Apple Silicon.
- **R installed** — download from [cran.r-project.org](https://cran.r-project.org) if you don't have it. Required if you want Nora to run R scripts.
- **Stata installed** (optional) — Nora finds Stata at `/Applications/Stata` or on your PATH.
- **A Claude account or API key** — either log in via the `claude` CLI or set `ANTHROPIC_API_KEY`. Nora inherits whichever is available.

## Two install paths

Nora has two ways to run, and right now the **from-source path is the recommended one**. The `.app` / `.dmg` build works locally, but it's unsigned — Apple Developer Program signing is required before it can be distributed cleanly to other people, and that's not in place yet. If you're picking this up as a researcher, use the from-source path until that's fixed.

## Path 1 — from source (recommended)

```bash
git clone https://github.com/junishka/builder.git nora
cd nora
uv sync --group dev
uv run pytest                       # expect ~283 passing
uv run nora-ui                   # web UI — landing screen for files / folder
uv run nora-ui /path/to/data     # web UI — open straight into chat
```

The web UI opens a native window with a drop zone for `.csv` / `.dta` / `.rds` files. Dropped files land in `~/.nora-sessions/<timestamp>_<id>/` — a per-session scratch dir that becomes the sandbox root. Cleanest way to share exactly the files you want to analyze without exposing a whole project folder.

There's also a terminal frontend — `uv run nora` — for power users who'd rather stay in the shell. Same backend, same privacy guarantees, same data; just no drag-drop and no fancy result panels.

## Path 2 — from a `.app` build (local-only for now)

You can build a real macOS `.app` and `.dmg` locally and use them on your own machine. The .app launches the web UI directly; no Terminal popup, just the chat window. **What you can't do is hand the .dmg to a colleague — without an Apple Developer Program signature it triggers Gatekeeper and most users won't get past the warning. We're not paying the $99/yr for the cert until the project is closer to wider distribution.**

Local build:
```bash
git clone https://github.com/junishka/builder.git nora
cd nora
uv sync --group dev
bash packaging/build_app.sh         # → dist/Nora.app  (~70 MB)
bash packaging/build_dmg.sh         # → dist/Nora.dmg  (~35 MB)
open dist/Nora.app
```

If you've moved `Nora.app` somewhere persistent (`/Applications/`), the first launch hits the same Gatekeeper dialog non-distributed apps do:

> *"Nora" cannot be opened because the developer cannot be verified.*

Two workarounds:
- **Right-click → Open** in Finder. macOS shows a similar dialog but with an **Open** button. Click it. From then on, double-clicking works normally.
- Or from Terminal: `xattr -cr /Applications/Nora.app` then double-click as usual.

## Running Nora

Whichever path you took, the entry point is the same: a chat window with a landing screen the first time, and your sandbox-rooted data thereafter. The first thing it asks is what files you want Claude to see — drop them on the landing zone, or pick a folder. That selection becomes the sandbox: Claude cannot read anything outside it.

Once you're in the chat, you talk. Claude proposes; you steer. Scripts that Claude runs are sandboxed (no network, narrow filesystem allowlist), every result is sanitized before Claude sees it, and the raw R/Stata output stays visible to you in the result panels.

## Schema policy — controlling what Claude sees

Claude sees only structural metadata about each dataset, never the rows. The default ceiling is **`names_types_labels_summary`** — variable names, types, value labels, and per-variable NA / distinct-value counts. That's the most informative tier that doesn't leak per-observation data.

You change this via the **Permission chip** in the web UI's composer row (a dropdown per dataset), or `/policy` in the terminal UI. Both edit `<your-data-dir>/.nora/policy.json`, which you can also hand-edit:

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

**At no tier** does Nora share raw values, min, max, median, or any other per-observation value. Those require a separate tool call (`request_data`) that applies statistical disclosure control rules, or a full analysis via `submit_script` whose output is sanitized before Claude sees it.

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

**No R or Stata installed warning.** Install R (or Stata) and relaunch. Schema and bounded data queries still work without them, but `submit_script` — the analysis path — won't.

**Claude asks to see variable labels and gets denied.** That's the schema policy working as intended. If the labels aren't sensitive for this dataset, raise the ceiling via the Permission chip (web UI) or `/policy` (terminal UI).

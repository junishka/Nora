# Installing Builder

## What you'll need

- **A Mac** — macOS 11 (Big Sur) or later, Intel or Apple Silicon.
- **R installed** — download from [cran.r-project.org](https://cran.r-project.org) if you don't have it. Required if you want Builder to run R scripts.
- **Stata installed** (optional) — Builder finds Stata at `/Applications/Stata` or on your PATH.
- **A Claude account or API key** — either log in via the `claude` CLI or set `ANTHROPIC_API_KEY`. Builder inherits whichever is available.

You do **not** need Python, `uv`, `pip`, or any other development tooling. Everything is bundled in the `.app`.

## Installing

1. **Download** `Builder.dmg` from the project (you'll receive it directly from the project maintainer; public release is not yet available).

2. **Open the `.dmg`** by double-clicking it. A window appears with `Builder.app` and a shortcut to your `Applications` folder.

3. **Drag `Builder.app` into `Applications`**.

4. **The first time you open it**, macOS will refuse and show a dialog like:

   > *"Builder" cannot be opened because the developer cannot be verified.*

   This is Gatekeeper — macOS blocks apps from unidentified developers by default. Builder is currently unsigned (signing requires an Apple Developer Program membership, which we haven't paid for at this stage).

   To work around it the first time:

   - **Option A (recommended): right-click → Open.** Open `Applications` in Finder, right-click `Builder.app`, choose **Open** from the menu. macOS will show a similar dialog but with an **Open** button alongside Cancel. Click Open. From then on, double-clicking works normally.

   - **Option B: remove the quarantine flag from Terminal.**
     ```bash
     xattr -cr /Applications/Builder.app
     ```
     Then double-click the app normally.

   If you see *"Builder is damaged and can't be opened"* instead, that's a separate Gatekeeper path — use Option B.

## Running Builder

Double-click `Builder.app` in `Applications` or Launchpad. A new Terminal window opens with the Builder chat running.

The first thing it asks is where your data lives:

```
data directory (~/Documents):
```

Enter the path to the folder containing the data files you want to analyze — absolute (`/Users/you/Project/data`) or home-prefixed (`~/Project/data`). That directory becomes Builder's sandbox for the session — Claude cannot read anything outside it.

Tab-completion works. Press **Enter** to accept the default (`~/Documents`).

After you pick a directory, Builder prints its banner — auth mode, detected runtimes (R, Stata), sandbox status, and a per-dataset permission summary — and drops you into the chat. You type; Claude responds.

**Heads-up:** the `.dmg` currently ships the **terminal** Builder. There's also a web UI (`builder-ui`) that opens a native window and supports drag-and-drop file upload, but that runs from source only right now. See "Building from source" at the bottom.

## Schema policy — controlling what Claude sees

By default, Claude sees only **variable names and types** for each dataset. It does not see labels, value labels, or any actual values.

**From inside Builder**, type `/policy` at the prompt to open an interactive wizard that lists your datasets and their current ceilings, and lets you raise or lower each one without editing any files. This is the recommended way.

Type `/help` to see the full list of local commands (they don't go to Claude).

**By hand**, you can also edit `<your-data-dir>/.builder/policy.json` directly:

```json
{
  "version": 1,
  "default_max_depth": "names_types",
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

Available depths, from most private to most permissive:

1. `names_only` — variable names, nothing else.
2. `names_types` — + a type per variable. **Default.**
3. `names_types_labels` — + variable labels and value labels.
4. `names_types_labels_summary` — + per-variable NA count and distinct-value count for categoricals.

**At no depth** does Builder share raw values, min, max, median, or any other per-observation value. Those require a separate tool call (`request_data`) that applies statistical disclosure control rules, or a full analysis via `submit_script` whose output is sanitized before Claude sees it.

## Uninstalling

Drag `Builder.app` from `Applications` to the Trash. Builder writes per-project state into `.builder/` subdirectories inside the data directories you pointed it at — delete those if you want to remove all traces:

```bash
find ~ -type d -name .builder -print
# review, then remove
```

## Troubleshooting

**Terminal opens, closes immediately.** The bundled binary is likely failing to start. From a regular Terminal, run the binary directly to see the error:

```bash
/Applications/Builder.app/Contents/Resources/builder/builder --help
```

**"sandbox-exec not found" warning.** Should never happen on macOS — `/usr/bin/sandbox-exec` is part of the OS. If you see this, file a bug.

**"No R or Stata installed" warning.** Install R (or Stata) and relaunch. Schema and bounded data queries still work without them, but `submit_script` — the analysis path — will not.

**Claude asks to see variable labels and gets denied.** That's the schema policy working as intended. If the labels aren't sensitive for this dataset, raise the ceiling in `.builder/policy.json` (see above).

## Building from source

If you'd rather run the development checkout directly — or you want the web UI with drag-and-drop file upload, which isn't in the current `.dmg`:

```bash
git clone https://github.com/junishka/builder.git
cd builder
uv sync --group dev
uv run pytest                       # expect ~194 passed

# Terminal frontend (same as the .dmg ships)
uv run builder                      # landing prompt for the data dir
uv run builder /path/to/your/data

# Web-UI frontend (native WKWebView window)
uv run builder-ui                   # landing: drop files, pick folder, etc.
uv run builder-ui /path/to/data
```

The web UI opens a chat window with a drop zone for `.csv` / `.dta` / `.rds` files. Dropped files land in `~/.builder-sessions/<timestamp>_<id>/` — a per-session scratch dir that becomes the sandbox root. This is the cleanest way to share exactly the files you want to analyze without exposing a whole project folder.

To rebuild the `.app` and `.dmg`:

```bash
bash packaging/build_app.sh   # → dist/Builder.app
bash packaging/build_dmg.sh   # → dist/Builder.dmg
```

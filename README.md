# Nora

Local research assistant for sensitive data. The data stays on the
researcher's machine. The model behind Nora is Claude (Anthropic) or
ChatGPT (OpenAI). The product the researcher talks to is Nora.

The model reaches the researcher's files through a thirteen-tool MCP
interface. No Bash. No filesystem. No network. Scripts run under
macOS `sandbox-exec` with network denied and a tight subpath
allowlist for reads. Every result passes through a disclosure-control
sanitizer before the model sees it. The researcher sees raw R,
Stata, or Python output. The model only ever sees sanitized
summaries.

For a one-page pickup, see [`docs/handoff.md`](docs/handoff.md). For
plain-language framing, [`docs/overview.md`](docs/overview.md). For
the long-form direction and open-question log,
[`docs/direction.md`](docs/direction.md).

## Status

Beta. Privacy invariants are implemented and tested. The frontend is
the pywebview-based `nora-ui`, which is also what the `.app`
launches. The released `.dmg` is signed with a Developer ID
Application certificate and notarized by Apple, so a colleague
double-clicking the bundle on their own Mac doesn't trip Gatekeeper.
See [the handoff status table](docs/handoff.md#where-it-stands) for
the layer-by-layer view.

## Platform

macOS only at this stage. The `submit_script` tool relies on
`sandbox-exec` for its privacy boundary. On other platforms,
schema inspection and bounded-fact queries still work, but
scripts refuse to execute. A Linux port via `bubblewrap` is
possible but not on the near-term roadmap.

Also needs:

- Python 3.10+
- At least one of: R (`Rscript` on PATH), Stata (`stata-mp` /
  `stata-se` / `stata` on PATH, or installed at `/Applications/Stata`),
  or Python with `pandas` + `statsmodels`.
- Anthropic credentials (Claude subscription or `ANTHROPIC_API_KEY`)
  or an OpenAI API key. The first launch shows an auth screen and
  stores the credential in the system keyring.

## Install

If you have a `Nora.dmg`, see [`docs/install.md`](docs/install.md)
for the double-click flow and the first-run Gatekeeper workaround.

To build the `.dmg` yourself or run from source, see below.

## Run from source

```bash
git clone https://github.com/junishka/builder.git nora
cd nora
uv sync --group dev
uv run pytest

uv run nora-ui                      # landing: drop files or pick folder
uv run nora-ui /path/to/data        # straight into chat
```

With no path, `nora-ui` opens a landing screen. Drop `.csv`, `.tsv`,
`.dta`, `.rds`, `.parquet`, `.jsonl`, or `.ndjson` files onto the
drop zone. Or click **Choose files…** or **Choose folder…**. Dropped
files land in `~/.nora-sessions/<timestamp>_<id>/`. That directory
becomes the sandbox root for the session.

To rebuild the bundle:

```bash
bash packaging/build_app.sh         # → dist/Nora.app   (~70 MB)
bash packaging/build_dmg.sh         # → dist/Nora.dmg   (~35 MB)
open dist/Nora.app                  # smoke test
```

A bare local build is unsigned — fine for testing on the same
machine. To produce a release-grade signed + notarized `.dmg`, set
`NORA_SIGN_IDENTITY` (Developer ID Application certificate) before
`build_app.sh` and `NORA_NOTARIZE_PROFILE` (notarytool keychain
profile) before `build_dmg.sh`. The build scripts skip those steps
when the env vars are unset.

The `.app` bundles Python, the dependencies, and the runtime
libraries. Researchers running it do not need Python, `uv`, or any
build tooling. R, Stata, and Python are still required separately
because Nora invokes them as subprocesses.

## What it can do

- **Multi-provider.** Anthropic (subscription or API key) and
  OpenAI (API key). One auth screen. Per-provider session memory.
- **Multi-session.** Each session has its own runner. Switching
  the visible chat is a pure focus change. Long jobs in unfocused
  sessions keep streaming. The sidebar shows a busy dot per
  active runner.
- **Editable session names.** Click the topbar pill or the
  per-row `✎` to set a custom label. Persists in
  `session_state.json`. Empty save reverts to the auto-derived
  dataset or timestamp label.
- **Concurrent execution with cwd isolation.** Tool execution
  picks up the focused session's cwd via a `ContextVar`, so two
  runners cannot read each other's files.
- **Plot vision (model-output only).** Helper-produced figures
  cross to the model on the next turn. Raw-data plots stay
  researcher-only by construction.
- **Memory stack.** Chat history persists per session. A warm-
  start prefix injects the recent turns and recent results when
  the conversation reopens. The `recall_conversation` tool covers
  older lookups.

## Layout

- `src/nora/`
  - `__main__.py` — entry-point shim that calls `nora.ui.main`.
  - `ui.py` and `web/` for the pywebview shell and the HTML, CSS,
    JS frontend.
  - `tools.py` for the thirteen MCP tools the model sees.
  - `executor.py` for the sandbox profile and the R, Stata,
    Python subprocess runners.
  - `sanitizer.py`, `sdc.py`, `text_safety.py` for disclosure
    control.
  - `schema.py` for dataset metadata extraction across the seven
    supported formats.
  - `policy.py` for per-dataset schema-depth ceilings.
  - `store.py` for the SQLite result store and audit log.
  - `session_state.py` for the per-session "at a glance" snapshot.
  - `runtime/` for the R library, the Python library, and the ten
    Stata `.ado` helpers scripts call.
  - `provider/` for the Anthropic and OpenAI session adapters.
  - `chat_service.py` for the typed event stream the frontend
    consumes.
- `tests/` for 775 tests. `test_sanitizer.py` is the property-test
  backbone. `test_executor_*` cover the sandbox profile.
  `test_concurrent_sessions.py` covers multi-runner isolation.
- `docs/` for handoff, overview, direction, install, verification.
- `packaging/` for the PyInstaller spec and the `.app` / `.dmg`
  build scripts.

## The thirteen tools

`get_schema`, `search_schema`, `request_data`, `submit_script`,
`submit_script_file`, `expand_result`, `compose_results`,
`list_results`, `list_results_global` (env-gated cross-session
recall), `recall_conversation`, `read_attached_file`,
`list_session_files`, `search_in_session_files`. The full
descriptions live in `src/nora/tools.py`.

## Security model

Three independent layers carry the privacy guarantee:

1. **Tool interface.** No general-purpose tools. No filesystem,
   no shell, no network. Thirteen tools, enumerated exhaustively.
2. **Sandbox.** A `(deny default)` `sandbox-exec` profile.
   Narrow subpath allowlist for reads and writes. Network
   denied entirely.
3. **Sanitizer.** Every result passes through statistical-
   disclosure-control rules (precision clamping, cell-size
   suppression, dominance checks) and text-safety sanitation
   before it reaches the model.

None of these layers depends on which allowed tool the model
invokes. The full analysis, including what a local model would and
would not add on top of these layers, lives in
[`docs/direction.md`](docs/direction.md).

## License

Proprietary. Not yet open-sourced.

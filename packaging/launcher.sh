#!/bin/bash
#
# Nora.app launcher — execs the bundled web-UI binary in place.
#
# macOS launches this script when the researcher double-clicks
# Nora.app (it's Contents/MacOS/Nora per the Info.plist set in
# build_app.sh). The script runs in the GUI session and our binary
# is the pywebview shell, which opens its own native WKWebView
# window. We do NOT spawn a Terminal — earlier versions of this
# launcher wrote a .command file and opened Terminal because the
# bundled binary used to be the curses CLI; once the bundle entry
# became the web UI (see packaging/nora.spec), the Terminal
# popup became user-hostile chrome with no purpose.
#
# stdout / stderr from the binary go to a per-day log file under
# ``~/Library/Logs/Nora/`` so debugging is still possible. The
# log is best-effort: failures to write it are silently ignored.

set -euo pipefail

# Resolve the .app's own path from wherever macOS launched us.
# $0 is Contents/MacOS/Nora (inside the app bundle); two dirs up
# gets us to the .app root.
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$LAUNCHER_DIR/../.." && pwd)"
NORA_BIN="$APP_ROOT/Contents/Resources/nora/nora"

if [[ ! -x "$NORA_BIN" ]]; then
    # Self-display dialog via osascript — doesn't need Automation
    # entitlements. Fallback for a corrupted / partial install.
    /usr/bin/osascript -e "display dialog \"Nora.app is missing its bundled binary at $NORA_BIN. The .app may have been damaged or incompletely installed.\" buttons {\"OK\"} default button \"OK\" with icon stop"
    exit 1
fi

# Best-effort log file. Per-day rotation keeps the log compact and
# the previous day's still around for incident triage. The whole
# logging path is wrapped so any failure (unwritable HOME, read-only
# Library/Logs, full disk, immutable file with the day's name) falls
# through to /dev/null instead of blowing up launch under ``set -e``.
# Earlier versions only guarded ``mkdir`` and let the ``exec`` step's
# ``>>"$LOG_FILE"`` redirection abort the launcher silently — most
# users would see nothing happen on double-click.
LOG_DIR="$HOME/Library/Logs/Nora"
LOG_FILE="$LOG_DIR/nora-$(date +%Y-%m-%d).log"
LOG_TARGET="/dev/null"

# Each step in the && chain is part of an ``if`` condition, so
# ``set -e`` does NOT abort on individual failures here — that's the
# point. Failure of any link drops us to the /dev/null fallback.
if mkdir -p "$LOG_DIR" 2>/dev/null \
        && : >>"$LOG_FILE" 2>/dev/null \
        && [[ -w "$LOG_FILE" ]]; then
    LOG_TARGET="$LOG_FILE"
fi

# exec replaces this shell with the binary so the .app's process
# tree shows ``nora`` directly (cleaner Activity Monitor entry,
# Quit/Force-Quit work as expected). Append both streams to the
# resolved log target; users who want live output can ``tail -F``
# the per-day file under ~/Library/Logs/Nora/ when logging worked,
# or run from source to debug the no-log fallback.
exec "$NORA_BIN" >>"$LOG_TARGET" 2>&1

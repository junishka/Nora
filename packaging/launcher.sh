#!/bin/bash
#
# Builder.app launcher — execs the bundled web-UI binary in place.
#
# macOS launches this script when the researcher double-clicks
# Builder.app (it's Contents/MacOS/Builder per the Info.plist set in
# build_app.sh). The script runs in the GUI session and our binary
# is the pywebview shell, which opens its own native WKWebView
# window. We do NOT spawn a Terminal — earlier versions of this
# launcher wrote a .command file and opened Terminal because the
# bundled binary used to be the curses CLI; once the bundle entry
# became the web UI (see packaging/builder.spec), the Terminal
# popup became user-hostile chrome with no purpose.
#
# stdout / stderr from the binary go to a per-day log file under
# ``~/Library/Logs/Builder/`` so debugging is still possible. The
# log is best-effort: failures to write it are silently ignored.

set -euo pipefail

# Resolve the .app's own path from wherever macOS launched us.
# $0 is Contents/MacOS/Builder (inside the app bundle); two dirs up
# gets us to the .app root.
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$LAUNCHER_DIR/../.." && pwd)"
BUILDER_BIN="$APP_ROOT/Contents/Resources/builder/builder"

if [[ ! -x "$BUILDER_BIN" ]]; then
    # Self-display dialog via osascript — doesn't need Automation
    # entitlements. Fallback for a corrupted / partial install.
    /usr/bin/osascript -e "display dialog \"Builder.app is missing its bundled binary at $BUILDER_BIN. The .app may have been damaged or incompletely installed.\" buttons {\"OK\"} default button \"OK\" with icon stop"
    exit 1
fi

# Best-effort log file. Per-day rotation keeps the log compact and
# the previous day's still around for incident triage. ``mkdir -p``
# is a no-op when the dir exists; redirecting to /dev/null keeps
# log-write failures from breaking launch.
LOG_DIR="$HOME/Library/Logs/Builder"
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="$LOG_DIR/builder-$(date +%Y-%m-%d).log"

# exec replaces this shell with the binary so the .app's process
# tree shows ``builder`` directly (cleaner Activity Monitor entry,
# Quit/Force-Quit work as expected). Append both streams to the
# log; users who want live output can ``tail -F`` that file.
exec "$BUILDER_BIN" >>"$LOG_FILE" 2>&1

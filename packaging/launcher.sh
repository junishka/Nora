#!/bin/bash
#
# Builder.app launcher — opens the bundled `builder` binary in a new
# Terminal window.
#
# macOS launches this script when the researcher double-clicks
# Builder.app (it's Contents/MacOS/Builder per the Info.plist set in
# build_app.sh). The script runs in the background GUI session — no
# terminal window of its own — so it can't just exec the binary
# directly; it needs to spawn Terminal.
#
# Why this is a shell script writing a .command file, rather than
# AppleScript telling Terminal `do script`:
#
# AppleScript cross-app events require Automation entitlements.
# macOS asks the user to grant them on first run via a Privacy &
# Security prompt — but only for *signed* apps. For unsigned apps
# (which Builder is, until we pay for the Apple Developer Program),
# macOS often blocks the event silently and the user sees
# "Not authorized to send Apple events to Terminal. (-1743)" with
# no way to grant permission through the UI.
#
# `.command` files are the way around this. When macOS opens a
# `.command` file via `open`, it's a LaunchServices file-association
# action, not AppleScript — Terminal opens and runs the script
# without needing any special permission. We write a fresh
# `.command` per launch into a temp dir so the path to the bundled
# binary stays accurate even if the researcher moves Builder.app.

set -euo pipefail

# Resolve the .app's own path from wherever macOS launched us.
# $0 is Contents/MacOS/Builder (inside the app bundle); two dirs up
# gets us to the .app root.
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$LAUNCHER_DIR/../.." && pwd)"
BUILDER_BIN="$APP_ROOT/Contents/Resources/builder/builder"

if [[ ! -x "$BUILDER_BIN" ]]; then
    # Show a dialog via osascript. This doesn't need Automation
    # entitlements — it's a self-display dialog, not an event to
    # another app. Kept as a fallback in case the bundle is somehow
    # corrupt.
    /usr/bin/osascript -e "display dialog \"Builder.app is missing its bundled binary at $BUILDER_BIN. The .app may have been damaged or incompletely installed.\" buttons {\"OK\"} default button \"OK\" with icon stop"
    exit 1
fi

# Temp .command file. The name shows up in Terminal's title bar, so
# give it something recognizable.
TMP_DIR="$(/usr/bin/mktemp -d -t builder.XXXXXXXX)"
CMD_FILE="$TMP_DIR/Builder.command"

# Write the actual script Terminal will run. On exit we clean up the
# temp dir — `trap` fires when the shell Terminal spawned terminates
# (either the user quits Builder with Ctrl-D / "exit", or Terminal
# is closed). Without cleanup, /tmp would slowly accumulate
# abandoned launchers.
cat > "$CMD_FILE" <<EOS
#!/bin/bash
trap 'rm -rf "$TMP_DIR"' EXIT
exec "$BUILDER_BIN"
EOS
chmod +x "$CMD_FILE"

# Launch via LaunchServices. `open` is happy to use Terminal as the
# .command handler without asking for Automation permission.
/usr/bin/open "$CMD_FILE"

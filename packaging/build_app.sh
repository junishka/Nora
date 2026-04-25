#!/usr/bin/env bash
#
# Build Builder.app from the PyInstaller output.
#
# The bundled binary is the WEB UI (entry point:
# src/builder/__main_ui__.py) — the pywebview shell. The terminal CLI
# is intentionally NOT bundled in the .app; double-click should give
# the chat window, not Terminal. The CLI stays available from source
# via ``uv run builder``.
#
# Pipeline:
#   1. `uv run pyinstaller packaging/builder.spec --clean --noconfirm`
#      produces `dist/builder/` (one-dir bundle).
#   2. Assemble Builder.app manually (no osacompile) so we control
#      the launcher. Contents/MacOS/Builder is the shell script from
#      packaging/launcher.sh — it just execs the bundled binary;
#      pywebview opens its own native window.
#   3. Copy the PyInstaller bundle into
#      Builder.app/Contents/Resources/builder/.
#   4. Write an Info.plist marking it a normal GUI app (LSUIElement
#      false → dock icon visible, Cmd-Q works as expected).
#
# Gatekeeper note: the resulting .app is unsigned. First-run workaround
# (right-click → Open, or `xattr -cr Builder.app`) is documented in
# docs/install.md. Proper code-signing is deferred until wider
# distribution justifies paying for the Apple Developer Program; until
# then the .dmg pipeline is mostly developer-internal.
#
# Usage (from repo root):
#   bash packaging/build_app.sh
#
# Produces:
#   dist/Builder.app      — the macOS application bundle (web UI)
#   dist/builder/         — the raw PyInstaller output (kept for
#                            direct invocation and debugging)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
PYINSTALLER_OUT="$DIST_DIR/builder"
APP_BUNDLE="$DIST_DIR/Builder.app"

echo "==> Running PyInstaller"
uv run pyinstaller packaging/builder.spec --clean --noconfirm >/dev/null

if [[ ! -x "$PYINSTALLER_OUT/builder" ]]; then
    echo "PyInstaller did not produce $PYINSTALLER_OUT/builder" >&2
    exit 1
fi

echo "==> Assembling Builder.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources/builder"

# Launcher script — the .app's executable per Info.plist.
cp "$REPO_ROOT/packaging/launcher.sh" "$APP_BUNDLE/Contents/MacOS/Builder"
chmod +x "$APP_BUNDLE/Contents/MacOS/Builder"

# PyInstaller bundle — lives under Resources/builder/.
# `cp -R` preserves the _internal/ layout PyInstaller generates.
cp -R "$PYINSTALLER_OUT/." "$APP_BUNDLE/Contents/Resources/builder/"

echo "==> Writing Info.plist"
cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Builder</string>
    <key>CFBundleDisplayName</key>
    <string>Builder</string>
    <key>CFBundleIdentifier</key>
    <string>app.junishka.builder</string>
    <key>CFBundleVersion</key>
    <string>0.0.1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.0.1</string>
    <key>CFBundleExecutable</key>
    <string>Builder</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <!-- LSUIElement=false: standard GUI app — dock icon present,    -->
    <!-- shows up in Cmd-Tab, Cmd-Q quits cleanly. Earlier comment   -->
    <!-- here referenced spawning Terminal; the launcher no longer   -->
    <!-- does that, so the only window the user sees is pywebview's. -->
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo
echo "Built: $APP_BUNDLE"
echo "Size:  $(du -sh "$APP_BUNDLE" | cut -f1)"
echo
echo "Next: bash packaging/build_dmg.sh to produce a distributable .dmg."

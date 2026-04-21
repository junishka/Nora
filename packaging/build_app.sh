#!/usr/bin/env bash
#
# Build Builder.app from the PyInstaller output.
#
# Pipeline:
#   1. `uv run pyinstaller packaging/builder.spec --clean --noconfirm`
#      produces `dist/builder/` (one-dir bundle).
#   2. `osacompile` compiles the launcher AppleScript into a .app
#      skeleton at `dist/Builder.app/`.
#   3. We copy the PyInstaller bundle into
#      `dist/Builder.app/Contents/Resources/builder/` — the launcher
#      points there at runtime.
#   4. We overwrite the generated Info.plist with our own (sets the
#      bundle identifier, version, and LSUIElement=false so the app
#      shows up in Dock / Launchpad).
#
# Gatekeeper note: the resulting .app is unsigned. On first open,
# macOS will block it with "cannot be opened because the developer
# cannot be verified." Researcher workaround is in docs/install.md —
# right-click the .app → Open, or `xattr -cr Builder.app` to clear
# the quarantine flag. Proper code-signing needs an Apple Developer
# Program membership, deferred until wider distribution.
#
# Usage (from repo root):
#   bash packaging/build_app.sh
#
# Produces:
#   dist/Builder.app      — the macOS application bundle
#   dist/builder/         — the raw PyInstaller output (used by the .app)

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

echo "==> Compiling launcher AppleScript into Builder.app"
rm -rf "$APP_BUNDLE"
/usr/bin/osacompile -o "$APP_BUNDLE" "$REPO_ROOT/packaging/launcher.applescript"

echo "==> Staging PyInstaller bundle inside Builder.app/Contents/Resources"
mkdir -p "$APP_BUNDLE/Contents/Resources/builder"
# `cp -R` preserves symlinks and the _internal/ layout PyInstaller
# generates. Copying (rather than moving) leaves the raw
# `dist/builder/` around for direct CLI invocation and debugging.
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
    <string>applet</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
PLIST

echo
echo "Built: $APP_BUNDLE"
echo "Size:  $(du -sh "$APP_BUNDLE" | cut -f1)"
echo
echo "Next: bash packaging/build_dmg.sh to produce a distributable .dmg."

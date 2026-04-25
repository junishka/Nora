#!/usr/bin/env bash
#
# Build Nora.dmg — a drag-to-/Applications installer for Nora.app.
#
# Requires `dist/Nora.app` to already exist (run build_app.sh first
# if needed). Produces `dist/Nora.dmg`.
#
# Structure the .dmg presents when mounted:
#   Nora.dmg/
#     Nora.app         # drag this...
#     Applications -> /Applications  # ...into here
#
# We could go further with a custom background image and icon
# positioning, but that needs `create-dmg` or an AppleScript to drive
# Finder after mounting. Skipping that polish until researcher #1
# gives feedback that the drag-install UX matters. The bare layout
# here works fine — it's the pattern a lot of dev tools use.
#
# Gatekeeper: the resulting .dmg contains an unsigned .app. First-run
# workaround (right-click → Open, or `xattr -cr`) is covered in
# docs/install.md.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
APP_BUNDLE="$DIST_DIR/Nora.app"
DMG_OUT="$DIST_DIR/Nora.dmg"
STAGING="$DIST_DIR/dmg-staging"

if [[ ! -d "$APP_BUNDLE" ]]; then
    echo "Nora.app not found at $APP_BUNDLE — run build_app.sh first." >&2
    exit 1
fi

echo "==> Preparing staging directory"
rm -rf "$STAGING" "$DMG_OUT"
mkdir -p "$STAGING"

# Copy the .app (don't move — we want the original to stay for
# re-runs / manual testing).
cp -R "$APP_BUNDLE" "$STAGING/"

# The drag-to-Applications symlink. macOS Finder treats this specially
# when the .dmg is opened: shows as a real Applications folder the
# user can drop the app onto.
ln -s /Applications "$STAGING/Applications"

echo "==> Building $DMG_OUT"
# hdiutil produces a compressed (UDZO) read-only .dmg. `-fs HFS+`
# avoids APFS-specific packaging which some older macOS versions
# can't mount.
/usr/bin/hdiutil create \
    -volname "Nora" \
    -srcfolder "$STAGING" \
    -fs HFS+ \
    -format UDZO \
    -ov \
    "$DMG_OUT" >/dev/null

rm -rf "$STAGING"

echo
echo "Built: $DMG_OUT"
echo "Size:  $(du -sh "$DMG_OUT" | cut -f1)"
echo
echo "Hand to a pilot researcher with docs/install.md."

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
# Gatekeeper / signing:
#   - If $NORA_SIGN_IDENTITY is set, the .dmg itself is signed too
#     (the .app inside should already be signed by build_app.sh).
#   - If $NORA_NOTARIZE_PROFILE is also set (a keychain profile name
#     stored via `xcrun notarytool store-credentials`), the .dmg is
#     submitted to Apple's notary service and the resulting ticket is
#     stapled into the .dmg so first-launch works offline.
#   - If neither is set, the .dmg contains an unsigned .app and
#     researchers need the right-click → Open workaround documented in
#     docs/install.md.

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

if [[ -n "${NORA_SIGN_IDENTITY:-}" ]]; then
    echo "==> Signing $DMG_OUT"
    /usr/bin/codesign --force --timestamp \
        --sign "$NORA_SIGN_IDENTITY" \
        "$DMG_OUT"
else
    echo "==> Skipping DMG signing (NORA_SIGN_IDENTITY unset)"
fi

if [[ -n "${NORA_NOTARIZE_PROFILE:-}" ]]; then
    if [[ -z "${NORA_SIGN_IDENTITY:-}" ]]; then
        echo "NORA_NOTARIZE_PROFILE is set but NORA_SIGN_IDENTITY is not." >&2
        echo "Notarization requires the .dmg (and its .app) to be signed first." >&2
        exit 1
    fi

    echo "==> Submitting to Apple notary service (this may take 2-15 minutes)"
    # --wait blocks until Apple returns a status. On rejection, fetch the
    # detailed log so the user doesn't have to chase the submission ID.
    if ! xcrun notarytool submit "$DMG_OUT" \
            --keychain-profile "$NORA_NOTARIZE_PROFILE" \
            --wait; then
        echo
        echo "Notarization failed. Fetching the most recent submission log:" >&2
        LATEST_ID="$(xcrun notarytool history \
                        --keychain-profile "$NORA_NOTARIZE_PROFILE" 2>/dev/null \
                    | awk '/id:/{print $2; exit}')"
        if [[ -n "$LATEST_ID" ]]; then
            xcrun notarytool log "$LATEST_ID" \
                --keychain-profile "$NORA_NOTARIZE_PROFILE" >&2 || true
        fi
        exit 1
    fi

    echo "==> Stapling notarization ticket"
    xcrun stapler staple "$DMG_OUT"
    xcrun stapler validate "$DMG_OUT"
else
    echo "==> Skipping notarization (NORA_NOTARIZE_PROFILE unset)"
fi

echo
echo "Built: $DMG_OUT"
echo "Size:  $(du -sh "$DMG_OUT" | cut -f1)"
echo
echo "Hand to a pilot researcher with docs/install.md."

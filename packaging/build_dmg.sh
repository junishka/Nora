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

# When the caller asks for a signed and/or notarized .dmg, the .app
# inside MUST already be signed. This script consumes a pre-built
# dist/Nora.app, so an easy mistake is to build the app unsigned, then
# rerun only this script with the signing variables set — Apple's
# notary service may even accept the submission, leaving you with a
# signed .dmg that wraps an unsigned app. Catch that here, before
# staging, so the failure is immediate and local. ``codesign --verify
# --deep --strict`` walks every nested Mach-O too, so a partial sign
# (where the bundle is signed but a nested binary isn't) is also
# rejected at this gate.
if [[ -n "${NORA_SIGN_IDENTITY:-}" ]] || [[ -n "${NORA_NOTARIZE_PROFILE:-}" ]]; then
    echo "==> Verifying $APP_BUNDLE is signed"
    if ! /usr/bin/codesign --verify --deep --strict "$APP_BUNDLE" 2>/dev/null; then
        echo "ERROR: $APP_BUNDLE is not signed (or its signature is invalid)." >&2
        echo "       Run build_app.sh with NORA_SIGN_IDENTITY set, e.g.:" >&2
        echo "         NORA_SIGN_IDENTITY=\"Developer ID Application: ... (TEAMID)\" \\" >&2
        echo "             bash packaging/build_app.sh" >&2
        echo "       then rerun this script. Notarizing an unsigned .app produces" >&2
        echo "       a release artifact that fails Gatekeeper at first launch." >&2
        exit 1
    fi
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

    echo "==> Submitting to Apple notary service"
    # We poll explicitly instead of using ``--wait``. ``--wait`` is a
    # tight in-process loop with no timeout; when Apple's notary queue
    # stalls (it does, occasionally — outage, accumulated backpressure,
    # something on their side), the loop hangs the build for hours
    # without ever giving up. A previous release of this DMG sat in
    # --wait for 26 hours before someone noticed. Manual polling lets
    # us bail with a recovery hint that lets the user resume from the
    # staple step once Apple eventually finishes.
    SUBMIT_OUT="$(xcrun notarytool submit "$DMG_OUT" \
                    --keychain-profile "$NORA_NOTARIZE_PROFILE")"
    echo "$SUBMIT_OUT"
    SUBMISSION_ID="$(echo "$SUBMIT_OUT" \
                     | awk -F': ' '/^[[:space:]]*id:/{print $2; exit}')"
    if [[ -z "$SUBMISSION_ID" ]]; then
        echo "Could not parse submission id from notarytool output." >&2
        exit 1
    fi

    # Poll up to 30 minutes (default; both intervals env-overridable
    # so CI can shorten on fast notarization paths and lengthen on
    # known-slow days). Apple's median is ~5min; the default absorbs
    # routine backlogs while still surfacing genuinely-stuck
    # submissions in a workday rather than a workweek.
    POLL_INTERVAL="${NORA_NOTARIZE_POLL_INTERVAL:-30}"
    POLL_TIMEOUT="${NORA_NOTARIZE_POLL_TIMEOUT:-1800}"
    SECONDS_WAITED=0
    STATUS=""
    # Poll BEFORE the first sleep — Apple sometimes flips a tiny
    # submission to Accepted within seconds, and a leading 30s wait
    # was a guaranteed floor on every release. Layout is "check,
    # then sleep, then check again" so a fast accept exits in <1s,
    # a typical accept exits at the first natural-cadence interval,
    # and slow ones still cap at POLL_TIMEOUT.
    while true; do
        # ``--output-format json`` gives a stable shape regardless
        # of Apple's free-form column layout. The previous awk
        # ``status:`` parser was one Apple cosmetic change away from
        # silently returning empty status forever (which would loop
        # the script until POLL_TIMEOUT). The python one-liner is
        # vendored here so the script keeps its single-bash-file
        # contract — bringing in jq would add a dependency that
        # release machines may not have.
        STATUS_JSON="$(xcrun notarytool info "$SUBMISSION_ID" \
                        --keychain-profile "$NORA_NOTARIZE_PROFILE" \
                        --output-format json 2>/dev/null || true)"
        STATUS="$(printf '%s' "$STATUS_JSON" \
                  | /usr/bin/python3 -c 'import json,sys
try:
  print(json.loads(sys.stdin.read()).get("status", ""))
except Exception:
  pass' 2>/dev/null)"
        echo "    [${SECONDS_WAITED}s] status: ${STATUS:-unknown}"
        case "$STATUS" in
            Accepted)
                break
                ;;
            Invalid|Rejected)
                # ``Invalid`` is Apple's documented terminal-failure
                # status. ``Rejected`` is included defensively in case
                # Apple introduces a synonym; treating it as failure
                # is correct either way.
                echo >&2
                echo "Notarization rejected (status=$STATUS). Submission log:" >&2
                xcrun notarytool log "$SUBMISSION_ID" \
                    --keychain-profile "$NORA_NOTARIZE_PROFILE" >&2 || true
                exit 1
                ;;
            "In Progress"|"")
                # Still working / transient query failure. Loop.
                ;;
            *)
                # Unknown terminal status — treat as failure rather
                # than spinning until POLL_TIMEOUT. Apple may add new
                # statuses; we'd rather fail loudly than silently
                # consume a 30-min timeout.
                echo >&2
                echo "Notarization returned unexpected status: $STATUS" >&2
                echo "Submission id: $SUBMISSION_ID" >&2
                echo "Inspect with:" >&2
                echo "  xcrun notarytool info $SUBMISSION_ID --keychain-profile $NORA_NOTARIZE_PROFILE" >&2
                exit 1
                ;;
        esac
        if [[ "$SECONDS_WAITED" -ge "$POLL_TIMEOUT" ]]; then
            break
        fi
        sleep "$POLL_INTERVAL"
        SECONDS_WAITED=$((SECONDS_WAITED + POLL_INTERVAL))
    done

    if [[ "$STATUS" != "Accepted" ]]; then
        # Apple is still chewing on it. The artifact and the
        # submission id are both salvageable — the user just runs the
        # staple step manually once `notarytool info` flips to Accepted.
        cat >&2 <<EOF

Notarization still pending after ${POLL_TIMEOUT}s. This is recoverable;
we just stopped watching. Apple's queue may be backlogged
(check https://developer.apple.com/system-status/).

Submission id: $SUBMISSION_ID

To finish the release once Apple flips it to Accepted, run:

    xcrun notarytool info $SUBMISSION_ID --keychain-profile $NORA_NOTARIZE_PROFILE
    # ...wait until status: Accepted, then:
    xcrun stapler staple "$DMG_OUT"
    xcrun stapler validate "$DMG_OUT"

EOF
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

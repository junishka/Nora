#!/usr/bin/env bash
#
# Release pipeline for Nora.
#
# Bundles the per-release flow into one command:
#   1. Pre-flight: verify signing identity + notary profile + clean tree
#   2. Build dist/Nora.app   (signed via build_app.sh)
#   3. Build dist/Nora.dmg   (signed + notarized + stapled via build_dmg.sh)
#   4. Verify the artifacts pass codesign + spctl + stapler
#   5. Install to /Applications/Nora.app (with quarantine flag stripped)
#
# Usage (run from anywhere — script resolves the repo via its own path):
#   bash packaging/release.sh                # full pipeline + install
#   bash packaging/release.sh --no-install   # build only
#   bash packaging/release.sh --app-only     # skip the .dmg + notarization
#   bash packaging/release.sh --check-only   # run pre-flight, exit
#   bash packaging/release.sh --allow-dirty  # skip the clean-tree check
#
# Required env (drop these in ~/.zshrc to make them stick):
#   NORA_SIGN_IDENTITY     e.g. "Developer ID Application: Your Name (TEAMID)"
#   NORA_NOTARIZE_PROFILE  notarytool keychain-profile name from
#                          `xcrun notarytool store-credentials`
#
# Why a wrapper at all:
#   - build_app.sh + build_dmg.sh skip signing silently when the env
#     vars are unset, producing a Gatekeeper-rejected .dmg with no error
#     message. The pre-flight catches this before the slow build starts.
#   - PyInstaller's static analysis can silently skip pulling user code
#     into the bundle if a .py file has issues. There was an incident
#     where uncommitted edits to ui.py produced a Nora.app that failed
#     to import nora.ui at startup. Bail on dirty trees by default.
#   - Apple's notary service occasionally hangs (a previous release sat
#     in --wait for 26 hours). When the notarize-poll fix lands the dmg
#     script caps at 30 min; this wrapper still verifies stapling
#     succeeded so a partial run doesn't go unnoticed.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL=true
CHECK_ONLY=false
ALLOW_DIRTY=false
APP_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-install)  INSTALL=false; shift ;;
        --check-only)  CHECK_ONLY=true; shift ;;
        --allow-dirty) ALLOW_DIRTY=true; shift ;;
        --app-only)    APP_ONLY=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# ── Pre-flight ────────────────────────────────────────────────────────

echo "==> Pre-flight"

if [[ -z "${NORA_SIGN_IDENTITY:-}" ]]; then
    cat >&2 <<EOF
  ✗ NORA_SIGN_IDENTITY is not set.
    Add this to ~/.zshrc (or set in this shell) and re-run:
      export NORA_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
EOF
    exit 1
fi

# The cert must actually exist in the login keychain. Match by the
# (TEAMID) suffix so trailing whitespace in the env value doesn't trip
# us up.
TEAM_FRAGMENT="$(echo "$NORA_SIGN_IDENTITY" \
                 | grep -oE '\([0-9A-Z]{10}\)' | head -1 || true)"
if [[ -z "$TEAM_FRAGMENT" ]] \
        || ! /usr/bin/security find-identity -v -p codesigning 2>/dev/null \
                | grep -q "$TEAM_FRAGMENT"; then
    cat >&2 <<EOF
  ✗ Signing cert for $NORA_SIGN_IDENTITY not found in login keychain.
    Run: security find-identity -v -p codesigning
    to see what's actually installed.
EOF
    exit 1
fi
echo "  ✓ Signing identity: $NORA_SIGN_IDENTITY"

if [[ "$APP_ONLY" == "false" ]]; then
    if [[ -z "${NORA_NOTARIZE_PROFILE:-}" ]]; then
        cat >&2 <<EOF
  ✗ NORA_NOTARIZE_PROFILE is not set (or pass --app-only to skip).
    Create one once with:
      xcrun notarytool store-credentials <profile-name> \\
          --apple-id you@example.com --team-id TEAMID \\
          --password APP-SPECIFIC-PASSWORD
EOF
        exit 1
    fi
    if ! xcrun notarytool history \
            --keychain-profile "$NORA_NOTARIZE_PROFILE" >/dev/null 2>&1; then
        echo "  ✗ notarytool profile '$NORA_NOTARIZE_PROFILE' is not stored." >&2
        exit 1
    fi
    echo "  ✓ Notary profile: $NORA_NOTARIZE_PROFILE"
fi

# Bail on uncommitted .py / .css / .html / .spec changes. Tracked
# changes are what break PyInstaller; untracked files (logos,
# screenshots, scratch) don't affect the bundle so we ignore them.
if [[ "$ALLOW_DIRTY" == "false" ]]; then
    DIRTY="$(/usr/bin/git status --porcelain --untracked-files=no 2>/dev/null || true)"
    if [[ -n "$DIRTY" ]]; then
        cat >&2 <<EOF
  ✗ Working tree has uncommitted changes (use --allow-dirty to override):
$DIRTY

    PyInstaller's static-analysis pass can silently skip user code if
    a tracked .py has issues, producing a bundle that fails to import
    on launch. Commit, stash, or branch the WIP first.
EOF
        exit 1
    fi
fi
echo "  ✓ Working tree clean"

BRANCH="$(/usr/bin/git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [[ "$BRANCH" != "main" ]]; then
    echo "  ⚠ Building from '$BRANCH' (not main)."
    echo -n "    Continue? [y/N] "
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi
echo "  ✓ Branch: $BRANCH"

if [[ "$CHECK_ONLY" == "true" ]]; then
    echo
    echo "Pre-flight passed. (--check-only: skipping build.)"
    exit 0
fi

# ── Build ─────────────────────────────────────────────────────────────

echo
echo "==> Building Nora.app"
bash "$REPO_ROOT/packaging/build_app.sh"

if [[ "$APP_ONLY" == "false" ]]; then
    echo
    echo "==> Building Nora.dmg"
    bash "$REPO_ROOT/packaging/build_dmg.sh"
fi

# ── Verify ────────────────────────────────────────────────────────────

echo
echo "==> Verifying artifacts"

APP="$REPO_ROOT/dist/Nora.app"
DMG="$REPO_ROOT/dist/Nora.dmg"

# codesign --verify walks every nested Mach-O when --deep is set, so a
# partial sign (where the bundle is signed but a nested binary isn't)
# fails this check. spctl --assess is what Gatekeeper itself runs at
# launch, so a pass here is a strong signal that end users won't see
# the "cannot be opened" dialog.
/usr/bin/codesign --verify --deep --strict "$APP"
echo "  ✓ codesign verify"

ASSESS="$( /usr/sbin/spctl --assess --verbose=2 --type execute "$APP" 2>&1 || true )"
if echo "$ASSESS" | grep -qE "accepted"; then
    echo "  ✓ spctl assess: $(echo "$ASSESS" | tr '\n' ' ' | sed 's/  */ /g')"
else
    echo "  ⚠ spctl assess did not accept the .app:" >&2
    echo "    $ASSESS" >&2
fi

if [[ "$APP_ONLY" == "false" ]] && [[ -f "$DMG" ]]; then
    if xcrun stapler validate "$DMG" >/dev/null 2>&1; then
        echo "  ✓ DMG notarization stapled"
    else
        echo "  ⚠ DMG is not stapled. Notarization may still be in progress on" >&2
        echo "    Apple's side. Once 'xcrun notarytool history' shows the latest" >&2
        echo "    submission as Accepted, finish with:" >&2
        echo "      xcrun stapler staple \"$DMG\"" >&2
    fi
fi

# ── Install ───────────────────────────────────────────────────────────

if [[ "$INSTALL" == "true" ]]; then
    echo
    echo "==> Installing to /Applications/Nora.app"
    # If Nora is currently running, replacing the .app underneath it
    # would leave the running process in a weird state and the next
    # launch could load mismatched resources. Quit it cleanly first.
    if pgrep -xq Nora; then
        echo "  Quitting running Nora.app first..."
        osascript -e 'tell application "Nora" to quit' 2>/dev/null \
            || pkill -x Nora 2>/dev/null || true
        sleep 1
    fi
    rm -rf /Applications/Nora.app
    cp -R "$APP" /Applications/Nora.app
    # Strip the quarantine attr in case the .app was tagged after a
    # download / move. macOS only assigns it when the file crosses an
    # internet trust boundary, so this is usually a no-op for local
    # builds — but covers the case where dist/ came from somewhere else.
    xattr -dr com.apple.quarantine /Applications/Nora.app 2>/dev/null || true
    echo "  ✓ Installed at /Applications/Nora.app"
    echo "    (run 'killall Dock Finder' if the cached Dock icon stays stale)"
fi

# ── Summary ───────────────────────────────────────────────────────────

echo
echo "Done."
echo "  App: $APP"
if [[ "$APP_ONLY" == "false" ]] && [[ -f "$DMG" ]]; then
    echo "  DMG: $DMG  ($(du -h "$DMG" | cut -f1))"
fi

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
#   bash packaging/release.sh --yes          # auto-confirm prompts (CI use)
#
# Required env (drop these in ~/.zshrc to make them stick):
#   NORA_SIGN_IDENTITY     e.g. "Developer ID Application: Your Name (TEAMID)"
#   NORA_NOTARIZE_PROFILE  notarytool keychain-profile name from
#                          `xcrun notarytool store-credentials`
#
# Optional env:
#   NORA_RELEASE_YES=1     equivalent to --yes; auto-accepts the
#                          off-main-branch and behind-origin prompts so
#                          the script can run unattended (CI, cron).
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
# Default --yes from the env so this script can run unattended (CI,
# cron). Each interactive prompt below honours this flag instead of
# blocking forever on a non-TTY ``read``.
case "${NORA_RELEASE_YES:-}" in
    1|true|yes|YES) ASSUME_YES=true ;;
    *)              ASSUME_YES=false ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-install)  INSTALL=false; shift ;;
        --check-only)  CHECK_ONLY=true; shift ;;
        --allow-dirty) ALLOW_DIRTY=true; shift ;;
        --app-only)    APP_ONLY=true; shift ;;
        --yes|-y)      ASSUME_YES=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Confirm prompts honour --yes / NORA_RELEASE_YES. When neither is set
# AND stdin isn't a TTY, refuse instead of blocking forever — CI logs
# would otherwise stall silently on a never-arriving newline.
# Both regexes match either the single-letter form (y / Y / n / N) or
# the full word (yes / YES / Yes, no / NO / No). Empty input falls
# through to whichever default the helper enforces. Explicit "no"
# typed at a default-yes prompt should mean no, not "garbage → default
# → yes" which is what an over-strict ^[Nn]$ would do.
_RE_YES='^[Yy]([Ee][Ss])?$'
_RE_NO='^[Nn]([Oo])?$'

confirm() {
    local prompt="$1"
    if [[ "$ASSUME_YES" == "true" ]]; then
        echo "    $prompt [y/N] y  (auto)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        echo "  ✗ $prompt — non-interactive shell. Re-run with --yes" >&2
        echo "    or set NORA_RELEASE_YES=1 to bypass." >&2
        return 1
    fi
    local ans=""
    echo -n "    $prompt [y/N] "
    read -r ans
    [[ "$ans" =~ $_RE_YES ]]
}

# Default-yes counterpart of ``confirm``. Use for prompts where the
# obvious / expected answer is yes (e.g. "pull now?") so the user can
# accept by just hitting Enter. Non-TTY runs silently accept — CI's
# explicit choice is to not interact, and a default-yes prompt's whole
# point is that yes is the safe path.
confirm_yes() {
    local prompt="$1"
    if [[ "$ASSUME_YES" == "true" ]]; then
        echo "    $prompt [Y/n] y  (auto)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        return 0
    fi
    local ans=""
    echo -n "    $prompt [Y/n] "
    read -r ans
    [[ ! "$ans" =~ $_RE_NO ]]
}

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
    confirm "Continue?" || exit 1
fi
echo "  ✓ Branch: $BRANCH"

# Freshness check: warn if local is behind the remote tracking branch.
# Common failure mode is "forgot to pull, built and shipped a stale
# bundle". Fetch is silent and non-merging, so this is informational —
# we don't try to pull from inside the release pipeline because pull
# failures (conflicts, network) belong upstream of the build.
#
# Surface fetch failures explicitly. The earlier ``2>/dev/null || true``
# masked offline / auth issues, so a stale build with no network would
# silently print "✓ Up to date" — exactly the case the freshness check
# was meant to catch. Now we emit a clear warning and let the caller
# decide whether to proceed.
if /usr/bin/git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    if /usr/bin/git fetch --quiet 2>/dev/null; then
        BEHIND="$(/usr/bin/git rev-list --count "HEAD..@{u}" 2>/dev/null || echo 0)"
        if [[ "$BEHIND" -gt 0 ]]; then
            echo "  ⚠ Local '$BRANCH' is $BEHIND commit(s) behind origin/$BRANCH."
            # Single-purpose prompt with the obvious default. The
            # earlier two-actions-in-one wording ("Pull first, or
            # continue anyway?") forced the user to puzzle out which
            # answer meant which action, then dumped the work back on
            # them as a manual ``git pull`` re-run. Now: just ask, just
            # do it.
            if confirm_yes "Pull now?"; then
                if /usr/bin/git pull --ff-only --quiet; then
                    NEW_HEAD="$(/usr/bin/git log -1 --format='%h %s' HEAD)"
                    echo "  ✓ Pulled — now at $NEW_HEAD"
                else
                    echo "  ✗ Pull failed (divergence or unmerged paths)." >&2
                    echo "    Resolve manually with 'git pull' and re-run." >&2
                    exit 1
                fi
            else
                echo "  ⚠ Continuing with stale source."
            fi
        else
            echo "  ✓ Up to date with origin/$BRANCH"
        fi
    else
        # Network or auth failure means we can't tell whether the local
        # tree is current. Without an explicit confirmation that's
        # acceptable, fail closed — the alternative (silently shipping
        # whatever stale ref @{u} points to) is exactly the failure
        # mode the freshness check exists to prevent.
        echo "  ⚠ Could not fetch from origin (network/auth?) — local freshness unverified."
        confirm "Continue without confirming freshness?" \
            || { echo "Restore network access and re-run." >&2; exit 1; }
    fi
fi

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

# Architecture check — the nested running binary MUST be arm64. The
# spec pins target_arch="arm64", but a future edit (or a stray
# Rosetta-installed PyInstaller) could silently revert that and ship
# an x86_64 bundle that runs fine on the build machine but triggers
# the Rosetta-deprecation banner on every user's Mac. Fail closed
# here so the bad artifact never reaches the install / DMG step.
NESTED_BIN="$APP/Contents/Resources/nora/nora"
APP_ARCHS="$(/usr/bin/lipo -archs "$NESTED_BIN" 2>/dev/null || true)"
if [[ "$APP_ARCHS" != "arm64" ]]; then
    echo "  ✗ Wrong architecture: $NESTED_BIN is '$APP_ARCHS' (want 'arm64')." >&2
    echo "    Likely cause: PyInstaller ran under an Intel Python." >&2
    echo "    Check: file \$(which python3); arch" >&2
    exit 1
fi
echo "  ✓ Architecture: arm64"

# Version-skew check: ``pyproject.toml``'s ``project.version`` must
# match the bundle's ``CFBundleShortVersionString``. ``build_app.sh``
# now derives the plist value from pyproject (so the two CAN'T diverge
# unless the derive step silently failed), but a release-time
# regression check pins the invariant — and catches a stale
# pre-derive bundle that wasn't rebuilt before this release pass.
PLIST_VERSION="$(
    /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
        "$APP/Contents/Info.plist" 2>/dev/null || true
)"
PYPROJECT_VERSION="$(
    /usr/bin/python3 - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
PY
)"
if [[ -z "$PLIST_VERSION" || -z "$PYPROJECT_VERSION" ]]; then
    echo "  ✗ Could not read both versions (plist='$PLIST_VERSION', pyproject='$PYPROJECT_VERSION')." >&2
    exit 1
fi
if [[ "$PLIST_VERSION" != "$PYPROJECT_VERSION" ]]; then
    echo "  ✗ Version skew: Info.plist='$PLIST_VERSION' but pyproject.toml='$PYPROJECT_VERSION'." >&2
    echo "    The bundle was likely built before build_app.sh's derive-from-pyproject" >&2
    echo "    step landed (or that step failed silently). Rebuild and re-run." >&2
    exit 1
fi
echo "  ✓ Version: $PLIST_VERSION (matches pyproject.toml)"

ASSESS="$( /usr/sbin/spctl --assess --verbose=2 --type execute "$APP" 2>&1 || true )"
if echo "$ASSESS" | grep -qE "accepted"; then
    echo "  ✓ spctl assess: $(echo "$ASSESS" | tr '\n' ' ' | sed 's/  */ /g')"
else
    # Gatekeeper would block this on a fresh Mac. Shipping anyway
    # defeats the wrapper's release-gating contract — the whole reason
    # to run release.sh instead of build_app.sh + build_dmg.sh directly
    # is so a rejected artifact never reaches the install / "Done" path.
    echo "  ✗ spctl assess REJECTED the .app — Gatekeeper would block this:" >&2
    echo "    $ASSESS" >&2
    echo
    echo "  Common causes:" >&2
    echo "    • Signature didn't apply (NORA_SIGN_IDENTITY unset on the" >&2
    echo "      build_app.sh run)" >&2
    echo "    • Notarization hasn't completed for the bundled .app" >&2
    echo "    • The signing identity has expired or been revoked" >&2
    exit 1
fi

if [[ "$APP_ONLY" == "false" ]] && [[ -f "$DMG" ]]; then
    if xcrun stapler validate "$DMG" >/dev/null 2>&1; then
        echo "  ✓ DMG notarization stapled"
    else
        # build_dmg.sh staples after notarytool returns Accepted, so a
        # missing staple here means either the notarize step never ran
        # (env unset) or it timed out / errored. Either way the .dmg
        # isn't ready for distribution — fail rather than print Done.
        echo "  ✗ DMG is NOT stapled — first launch on a fresh Mac will hit Gatekeeper." >&2
        echo
        echo "  Recovery if Apple is just slow:" >&2
        echo "    xcrun notarytool history --keychain-profile \"\$NORA_NOTARIZE_PROFILE\"" >&2
        echo "    # …wait for status: Accepted, then:" >&2
        echo "    xcrun stapler staple \"$DMG\"" >&2
        echo "    xcrun stapler validate \"$DMG\"" >&2
        exit 1
    fi

    # Emit a SHA-256 sidecar file alongside the .dmg. Researchers and
    # downstream packagers (homebrew-cask, internal IT distribution,
    # mirrors) need a way to verify the binary they got matches what
    # we shipped. Apple's notarization signs the bundle but doesn't
    # publish a per-release fingerprint anyone can check from the
    # outside, and Gatekeeper only catches "Apple no longer trusts
    # this developer" — not "the .dmg was modified after we built
    # it." A SHA-256 in the same dist directory is the conventional
    # fix; ``shasum -a 256`` ships with macOS so there's no toolchain
    # cost. The ``.sha256`` file format mirrors what GitHub Releases
    # accepts so users can ``shasum -a 256 -c Nora.dmg.sha256`` after
    # downloading both. Recompute on every release so the file always
    # corresponds to the .dmg next to it.
    SHA256_FILE="$DMG.sha256"
    # ``shasum`` writes ``<hash>  <path>``; rewrite to use the
    # basename so verification works regardless of where the user
    # downloaded the artifacts to.
    DMG_BASENAME="$(basename "$DMG")"
    DMG_HASH="$(/usr/bin/shasum -a 256 "$DMG" | awk '{print $1}')"
    printf '%s  %s\n' "$DMG_HASH" "$DMG_BASENAME" > "$SHA256_FILE"
    echo "  ✓ DMG SHA-256 written to $(basename "$SHA256_FILE")"
    echo "    $DMG_HASH"
fi

# ── Install ───────────────────────────────────────────────────────────

if [[ "$INSTALL" == "true" ]]; then
    echo
    echo "==> Installing to /Applications/Nora.app"
    # If Nora is currently running, replacing the .app underneath it
    # would leave the running process in a weird state and the next
    # launch could load mismatched resources. Quit it cleanly first.
    #
    # Match by the actual executable name. Even though the .app's
    # CFBundleExecutable is ``Nora``, the launcher script ``exec``s
    # the bundled PyInstaller binary at ``Contents/Resources/nora/nora``
    # — so the running process shows up as ``nora`` (lowercase) in
    # ps / pgrep. The previous ``pgrep -xq Nora`` never matched, and
    # the install path went straight to ``rm -rf /Applications/Nora.app``
    # against a live process. Both names are checked here belt-and-
    # suspenders so a future packaging change that drops the exec hand-
    # off doesn't silently re-open the bug.
    if pgrep -xq nora || pgrep -xq Nora; then
        echo "  Quitting running Nora.app first..."
        osascript -e 'tell application "Nora" to quit' 2>/dev/null \
            || pkill -x nora 2>/dev/null \
            || pkill -x Nora 2>/dev/null \
            || true
        # Loop briefly until the process actually exits — sleep 1 was
        # a guess that fails under load (heavy GC inside an Anthropic
        # SDK shutdown can take 2-3s on a busy laptop). Bail with a
        # clear error if it never quits, rather than overwriting the
        # bundle out from under it.
        for _ in 1 2 3 4 5; do
            if ! pgrep -xq nora && ! pgrep -xq Nora; then break; fi
            sleep 1
        done
        if pgrep -xq nora || pgrep -xq Nora; then
            echo "  ✗ Nora is still running. Quit it manually, then re-run." >&2
            exit 1
        fi
    fi
    rm -rf /Applications/Nora.app
    # Use ``ditto`` instead of ``cp -R``: ditto preserves resource
    # forks, ACLs, and especially extended attributes that the code
    # signature relies on. ``cp -R`` on macOS strips some xattrs in
    # certain configurations, breaking the signature in subtle ways
    # that pass spctl but fail at first launch on a stricter machine.
    /usr/bin/ditto "$APP" /Applications/Nora.app
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
    if [[ -f "$DMG.sha256" ]]; then
        echo "  SHA: $DMG.sha256"
    fi
fi

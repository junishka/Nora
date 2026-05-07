#!/usr/bin/env bash
#
# Build Nora.app from the PyInstaller output.
#
# The bundled binary is the WEB UI (entry point:
# src/nora/__main_ui__.py) — the pywebview shell. The terminal CLI
# is intentionally NOT bundled in the .app; double-click should give
# the chat window, not Terminal. The CLI stays available from source
# via ``uv run nora``.
#
# Pipeline:
#   1. `uv run pyinstaller packaging/nora.spec --clean --noconfirm`
#      produces `dist/nora/` (one-dir bundle).
#   2. Assemble Nora.app manually (no osacompile) so we control
#      the launcher. Contents/MacOS/Nora is the shell script from
#      packaging/launcher.sh — it just execs the bundled binary;
#      pywebview opens its own native window.
#   3. Copy the PyInstaller bundle into
#      Nora.app/Contents/Resources/nora/.
#   4. Write an Info.plist marking it a normal GUI app (LSUIElement
#      false → dock icon visible, Cmd-Q works as expected).
#
# Gatekeeper / signing:
#   - If $NORA_SIGN_IDENTITY is set (e.g. "Developer ID Application: ...
#     (TEAMID)"), every Mach-O inside the bundle is signed with the
#     hardened runtime, and the bundle itself is signed with the
#     entitlements in packaging/entitlements.plist. This is the
#     prerequisite for build_dmg.sh's notarization step.
#   - If unset, the .app is unsigned and researchers need the right-click
#     → Open workaround documented in docs/install.md.
#
# Usage (from repo root):
#   bash packaging/build_app.sh
#   NORA_SIGN_IDENTITY="Developer ID Application: ..." bash packaging/build_app.sh
#
# Produces:
#   dist/Nora.app      — the macOS application bundle (web UI)
#   dist/nora/         — the raw PyInstaller output (kept for
#                            direct invocation and debugging)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
PYINSTALLER_OUT="$DIST_DIR/nora"
APP_BUNDLE="$DIST_DIR/Nora.app"

echo "==> Running PyInstaller"
uv run pyinstaller packaging/nora.spec --clean --noconfirm >/dev/null

if [[ ! -x "$PYINSTALLER_OUT/nora" ]]; then
    echo "PyInstaller did not produce $PYINSTALLER_OUT/nora" >&2
    exit 1
fi

echo "==> Assembling Nora.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources/nora"

# Launcher script — the .app's executable per Info.plist.
cp "$REPO_ROOT/packaging/launcher.sh" "$APP_BUNDLE/Contents/MacOS/Nora"
chmod +x "$APP_BUNDLE/Contents/MacOS/Nora"

# PyInstaller bundle — lives under Resources/nora/.
# `cp -R` preserves the _internal/ layout PyInstaller generates.
cp -R "$PYINSTALLER_OUT/." "$APP_BUNDLE/Contents/Resources/nora/"

echo "==> Writing Info.plist"
cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Nora</string>
    <key>CFBundleDisplayName</key>
    <string>Nora</string>
    <key>CFBundleIdentifier</key>
    <string>app.junishka.nora</string>
    <key>CFBundleVersion</key>
    <string>0.0.1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.0.1</string>
    <key>CFBundleExecutable</key>
    <string>Nora</string>
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

if [[ -n "${NORA_SIGN_IDENTITY:-}" ]]; then
    ENTITLEMENTS="$REPO_ROOT/packaging/entitlements.plist"
    if [[ ! -f "$ENTITLEMENTS" ]]; then
        echo "Missing $ENTITLEMENTS — required when NORA_SIGN_IDENTITY is set." >&2
        exit 1
    fi

    echo "==> Signing nested Mach-O binaries"
    # The .app's CFBundleExecutable is the shell launcher at
    # Contents/MacOS/Nora; it ``exec``s the nested PyInstaller
    # binary at Contents/Resources/nora/nora. The nested binary is
    # what becomes the running process — and exec does NOT propagate
    # entitlements from the parent. So the hardened-runtime
    # allowances (allow-unsigned-executable-memory,
    # disable-library-validation, allow-dyld-environment-variables)
    # have to be embedded into THAT binary's signature too, not just
    # the outer bundle's. Without this, codesign verification still
    # passes but the running image launches under hardened-runtime
    # without the exemptions and aborts on the first ctypes / cffi /
    # unsigned-dylib path PyInstaller exercises at startup.
    NESTED_MAIN="$APP_BUNDLE/Contents/Resources/nora/nora"

    # `find -depth` walks deepest-first so each binary is signed before
    # its enclosing bundle. We use `file` to skip shell scripts and other
    # non-Mach-O executables that would otherwise trip codesign.
    while IFS= read -r -d '' candidate; do
        if ! /usr/bin/file -b "$candidate" | grep -q "Mach-O"; then
            continue
        fi
        if [[ "$candidate" == "$NESTED_MAIN" ]]; then
            # Sign the running process image with entitlements so
            # hardened-runtime exemptions actually apply at launch.
            /usr/bin/codesign --force --options runtime --timestamp \
                --entitlements "$ENTITLEMENTS" \
                --sign "$NORA_SIGN_IDENTITY" "$candidate"
        else
            # Dylibs / extension modules: hardened runtime + timestamp,
            # but no entitlements (they don't become processes).
            /usr/bin/codesign --force --options runtime --timestamp \
                --sign "$NORA_SIGN_IDENTITY" "$candidate"
        fi
    done < <(/usr/bin/find "$APP_BUNDLE" -depth -type f \
                \( -name "*.dylib" -o -name "*.so" -o -perm -u+x \) -print0)

    echo "==> Signing app bundle"
    # Outer bundle still gets --entitlements so codesign metadata is
    # consistent at every level a verifier might inspect (the bundle,
    # the CFBundleExecutable, and the nested running binary).
    /usr/bin/codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$NORA_SIGN_IDENTITY" \
        "$APP_BUNDLE"

    echo "==> Verifying signature"
    /usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
    # Confirm the entitlements actually landed on the nested running
    # binary — this is the guard against future refactors that move
    # the entitlements step around and silently drop them from the
    # process image. ``codesign -d --entitlements -`` prints the
    # embedded plist; we grep for one of the keys we expect to see.
    if ! /usr/bin/codesign -d --entitlements - "$NESTED_MAIN" 2>/dev/null \
            | grep -q "com.apple.security.cs.disable-library-validation"; then
        echo "ERROR: nested binary $NESTED_MAIN is missing hardened-runtime entitlements." >&2
        echo "       This means the running process won't have the exemptions and" >&2
        echo "       will abort at launch despite passing codesign --verify." >&2
        exit 1
    fi
    echo "Signed with: $NORA_SIGN_IDENTITY"
else
    echo "==> Skipping codesign (NORA_SIGN_IDENTITY unset)"
fi

echo
echo "Built: $APP_BUNDLE"
echo "Size:  $(du -sh "$APP_BUNDLE" | cut -f1)"
echo
echo "Next: bash packaging/build_dmg.sh to produce a distributable .dmg."

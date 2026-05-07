#!/usr/bin/env bash
#
# Regenerate packaging/Nora.icns from packaging/icon-source.png.
#
# build_app.sh expects Nora.icns to already exist; we keep it
# committed so the build is fully self-contained on a fresh checkout
# with no extra prerequisites. Run this script only when the source
# logo changes.
#
# Requires macOS (uses /usr/bin/sips and /usr/bin/iconutil).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$REPO_ROOT/packaging"
SRC="$PKG_DIR/icon-source.png"
OUT="$PKG_DIR/Nora.icns"
ICONSET="$PKG_DIR/Nora.iconset"

if [[ ! -f "$SRC" ]]; then
    echo "Missing $SRC" >&2
    exit 1
fi

rm -rf "$ICONSET"
mkdir "$ICONSET"

# (size in px, name-without-extension) — the names are mandated by
# iconutil's iconset format. Pairs of N and N@2x give Retina + non-Retina
# at every Finder/Dock size from 16px to 1024px.
for spec in \
    "16 16x16" \
    "32 16x16@2x" \
    "32 32x32" \
    "64 32x32@2x" \
    "128 128x128" \
    "256 128x128@2x" \
    "256 256x256" \
    "512 256x256@2x" \
    "512 512x512" \
    "1024 512x512@2x"; do
    size="${spec% *}"
    name="${spec#* }"
    /usr/bin/sips -z "$size" "$size" "$SRC" --out "$ICONSET/icon_${name}.png" >/dev/null
done

/usr/bin/iconutil -c icns "$ICONSET" -o "$OUT"
rm -rf "$ICONSET"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"

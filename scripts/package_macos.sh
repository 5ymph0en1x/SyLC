#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PACKAGER="${SYLC_PYINSTALLER:-$ROOT_DIR/.venv-release-macos/bin/pyinstaller}"
BUILD_DIR="$ROOT_DIR/build/release-macos"
DIST_DIR="$ROOT_DIR/dist/macos"
mkdir -p "$BUILD_DIR/SyLC.iconset" "$DIST_DIR"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" assets/icon.png \
        --out "$BUILD_DIR/SyLC.iconset/icon_${size}x${size}.png" >/dev/null
done
iconutil -c icns "$BUILD_DIR/SyLC.iconset" -o "$BUILD_DIR/SyLC.icns"
"$PACKAGER" --noconfirm --distpath "$DIST_DIR" \
    --workpath "$ROOT_DIR/build/pyinstaller-macos" scripts/SyLC-macos.spec
APP="$DIST_DIR/SyLC 3D Player.app"
codesign --verify --deep --strict "$APP"
STAGE_DIR="$(mktemp -d "$BUILD_DIR/dmg.XXXXXX")"
ditto "$APP" "$STAGE_DIR/SyLC 3D Player.app"
ln -s /Applications "$STAGE_DIR/Applications"
cat > "$STAGE_DIR/README.txt" <<'EOF'
SyLC 3D Player — macOS Apple Silicon

Drag SyLC 3D Player to Applications.
Includes Python, libmpv, FFmpeg, and the Small 518 model for 2D-to-3D conversion.
This build is ad hoc signed and has not been notarized by Apple.
If macOS blocks the app, try opening it, then go to System Settings >
Privacy & Security > Open Anyway.

Observed depth inference on an Apple M3 Pro: approximately 0.8 depth maps/s.
The packaged player has not been launched for functional testing.
EOF
DMG="$DIST_DIR/SyLC-7.0.0-macOS-arm64.dmg"
hdiutil create -ov -volname 'SyLC 3D Player' -srcfolder "$STAGE_DIR" \
    -format UDZO "$DMG"
hdiutil verify "$DMG"
(cd "$DIST_DIR" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256")
echo "Release: $DMG"

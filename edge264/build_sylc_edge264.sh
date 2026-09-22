#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Building edge264 for macOS..."

CC="${CC:-clang}"
CFLAGS="-O3 -fPIC -dynamiclib -Wall -Wno-initializer-overrides -Wno-unused-function"
LDFLAGS="-lpthread"

OUT_DYLIB="${SCRIPT_DIR}/libedge264.dylib"

${CC} ${CFLAGS} -o "${OUT_DYLIB}" "${SCRIPT_DIR}/src/edge264.c" ${LDFLAGS}

echo "==> Built: ${OUT_DYLIB}"

mkdir -p "${ROOT_DIR}/runtime"
cp -f "${OUT_DYLIB}" "${ROOT_DIR}/runtime/libedge264.dylib"
echo "==> Copied to ${ROOT_DIR}/runtime/libedge264.dylib"

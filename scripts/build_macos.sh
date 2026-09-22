#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================"
echo " SyLC 3D Player - macOS Native Build Script (ARM64 / x86_64)"
echo "================================================================"

# 1. Detect Python
PYTHON_BIN=""
for py in "python3.14" "python3.12" "python3" "/opt/homebrew/bin/python3.14" "/opt/homebrew/bin/python3.12"; do
    if command -v "${py}" >/dev/null 2>&1; then
        PYTHON_BIN="$(which "${py}")"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "[-] Error: Python 3 not found. Please install Python 3.14 or 3.12."
    exit 1
fi

echo "[+] Using Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

# 2. Check Homebrew dependencies
echo "[+] Checking system libraries..."
if command -v brew >/dev/null 2>&1; then
    for pkg in libmatroska libebml pybind11 mpv; do
        if ! brew list "${pkg}" >/dev/null 2>&1; then
            echo "[*] Installing missing dependency via brew: ${pkg}..."
            brew install "${pkg}" || true
        fi
    done
fi

# 3. Build edge264 (H.264 / MVC decoder dylib)
echo "[+] Step 1: Compiling edge264..."
chmod +x "${ROOT_DIR}/edge264/build_sylc_edge264.sh"
"${ROOT_DIR}/edge264/build_sylc_edge264.sh"

# 4. Build mvc_demuxer_cpp (C++ MKV demuxer and MVC decoder bindings)
echo "[+] Step 2: Compiling mvc_demuxer_cpp with CMake..."
BUILD_DIR="${ROOT_DIR}/build_macos"
mkdir -p "${BUILD_DIR}"

cmake -S "${ROOT_DIR}/mvc_realtime_demuxer" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="${PYTHON_BIN}" \
    -DPYBIND11_FINDPYTHON=ON \
    -DCMAKE_PREFIX_PATH="/opt/homebrew;/usr/local" \
    -DBUILD_NATIVE_RENDERER=OFF

cmake --build "${BUILD_DIR}" --config Release -j "$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

# 5. Copy built Python extension to runtime/ and project root
mkdir -p "${ROOT_DIR}/runtime"
find "${BUILD_DIR}" -name "mvc_demuxer_cpp*.so" -exec cp -fv {} "${ROOT_DIR}/runtime/" \;
find "${BUILD_DIR}" -name "mvc_demuxer_cpp*.so" -exec cp -fv {} "${ROOT_DIR}/" \;

echo "================================================================"
echo " [+] Build completed successfully!"
echo " [+] Dynamic libraries installed in runtime/:"
ls -lh "${ROOT_DIR}/runtime/"*.dylib "${ROOT_DIR}/runtime/"*.so 2>/dev/null || true
echo "================================================================"

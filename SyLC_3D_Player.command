#!/usr/bin/env bash
# ==============================================================================
# SyLC 3D Player - macOS Double-Click Launcher (.command)
# ==============================================================================

# Ensure working directory is the project directory
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${DIR}"

# Find the best Python 3 interpreter
PYTHON_BIN=""
for py in "python3.14" "python3.12" "python3" "/opt/homebrew/bin/python3.14" "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3" "/usr/local/bin/python3"; do
    if command -v "${py}" >/dev/null 2>&1; then
        PYTHON_BIN="$(which "${py}")"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "[!] Error: No se encontró Python 3 en el sistema."
    echo "    Por favor instala Python (brew install python@3.14) o verifica tu PATH."
    read -n 1 -s -r -p "Presiona cualquier tecla para salir..."
    exit 1
fi

# Set library and module paths for macOS
export PYTHONPATH="${DIR}/src:${DIR}/runtime:${PYTHONPATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="${DIR}/runtime:/opt/homebrew/lib:/usr/local/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

echo "[+] Iniciando SyLC 3D Player con ${PYTHON_BIN}..."
exec "${PYTHON_BIN}" "${DIR}/SyLC_3D_Player.py" "$@"

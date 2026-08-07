$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nativeBuild = Join-Path $projectRoot 'build_py314\python\Release'
$runtime = Join-Path $projectRoot 'runtime'
$ort = Join-Path $projectRoot 'ort_tensorrt'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python is missing: $python"
}
if (-not (Test-Path -LiteralPath (
        Join-Path $nativeBuild 'mvc_demuxer_cpp.cp314-win_amd64.pyd'))) {
    throw "Build the Release native module first: cmake --build build_py314 --config Release"
}

$env:SYLC_SYNTH3D_LOOKAHEAD = '1'
$env:SYLC_PROJECT_ROOT = $projectRoot
$env:PATH = "$runtime;$ort;$env:PATH"
$env:SYLC_LOOKAHEAD_PROJECT_ROOT = $projectRoot
$env:SYLC_LOOKAHEAD_NATIVE_BUILD = $nativeBuild
$env:SYLC_LOOKAHEAD_RUNTIME = $runtime
$env:SYLC_LOOKAHEAD_ORT = $ort

# Preload the freshly built extension before the normal launcher inserts the
# stable packaged runtime at sys.path[0]. Python then reuses this exact module
# from sys.modules, while every other DLL/tool still resolves from runtime/.
$bootstrap = @'
import os
import runpy
import sys

root = os.environ['SYLC_LOOKAHEAD_PROJECT_ROOT']
native_build = os.environ['SYLC_LOOKAHEAD_NATIVE_BUILD']
handles = [
    os.add_dll_directory(os.environ['SYLC_LOOKAHEAD_RUNTIME']),
    os.add_dll_directory(os.environ['SYLC_LOOKAHEAD_ORT']),
]
sys.path.insert(0, native_build)
import mvc_demuxer_cpp  # noqa: F401 -- intentional preload
runpy.run_path(os.path.join(root, 'SyLC_3D_Player.py'), run_name='__main__')
'@

& $python -c $bootstrap

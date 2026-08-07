"""Layout contract for packaged sources, assets and native runtime files."""

import os
import sys
from pathlib import Path

from sylc import runtime_paths


def test_source_checkout_resolves_project_runtime_and_assets():
    root = Path(runtime_paths.RUNTIME_DIR).resolve().parent

    assert Path(runtime_paths.__file__).resolve().parents[2] == root
    assert Path(runtime_paths.RUNTIME_DIR) == root / 'runtime'
    assert Path(runtime_paths.ASSETS_DIR) == root / 'assets'


def test_required_runtime_markers_are_grouped_outside_repository_root():
    runtime = Path(runtime_paths.RUNTIME_DIR)
    required = {
        'mpv-2.dll', 'edge264.dll', 'ffprobe.exe',
        'mvc_demuxer_cpp.cp314-win_amd64.pyd',
    }

    assert all((runtime / name).is_file() for name in required)
    root = runtime.parent
    assert all(not (root / name).exists()
               for name in required)


def test_runtime_environment_configuration_is_idempotent():
    first = runtime_paths.configure_runtime_environment()
    second = runtime_paths.configure_runtime_environment()

    assert first == second == runtime_paths.RUNTIME_DIR
    assert runtime_paths.RUNTIME_DIR in sys.path
    assert os.environ['SYLC_RUNTIME_DIR'] == runtime_paths.RUNTIME_DIR

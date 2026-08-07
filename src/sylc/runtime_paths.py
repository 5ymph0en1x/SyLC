"""Stable project, asset and native-runtime paths for source and frozen builds."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def _unique_existing(paths):
    seen = set()
    for path in paths:
        if not path:
            continue
        resolved = os.path.abspath(os.fspath(path))
        key = os.path.normcase(resolved)
        if key in seen or not os.path.isdir(resolved):
            continue
        seen.add(key)
        yield resolved


def _application_roots():
    candidates = [os.environ.get('SYLC_PROJECT_ROOT')]
    try:
        import __compiled__
        candidates.append(getattr(__compiled__, 'containing_dir', None))
    except Exception:
        pass
    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / 'src' / 'sylc').is_dir():
        candidates.append(source_checkout)
    candidates.extend((
        os.path.dirname(os.path.abspath(sys.argv[0])),
        os.path.dirname(sys.executable),
        os.getcwd(),
    ))
    return tuple(_unique_existing(candidates))


PROJECT_ROOT = next(iter(_application_roots()), os.getcwd())


def _contains_runtime_marker(directory):
    return any((
        os.path.isfile(os.path.join(directory, 'edge264.dll')),
        os.path.isfile(os.path.join(directory, 'mpv-2.dll')),
        os.path.isfile(os.path.join(directory, 'ffprobe.exe')),
        bool(glob.glob(os.path.join(directory, 'mvc_demuxer_cpp*.pyd'))),
    ))


def locate_runtime_dir():
    explicit = os.environ.get('SYLC_RUNTIME_DIR')
    candidates = [explicit]
    for root in _application_roots():
        candidates.extend((os.path.join(root, 'runtime'), root))
    existing = tuple(_unique_existing(candidates))
    for directory in existing:
        if _contains_runtime_marker(directory):
            return directory
    return existing[0] if existing else os.path.join(PROJECT_ROOT, 'runtime')


def locate_assets_dir():
    explicit = os.environ.get('SYLC_ASSETS_DIR')
    candidates = [explicit]
    for root in _application_roots():
        candidates.extend((os.path.join(root, 'assets'), root))
    existing = tuple(_unique_existing(candidates))
    for directory in existing:
        if any(os.path.isfile(os.path.join(directory, marker))
               for marker in ('icon.png', 'splash.png', 'icon.ico')):
            return directory
    for directory in existing:
        if os.path.basename(directory).lower() == 'assets':
            return directory
    return os.path.join(PROJECT_ROOT, 'assets')


RUNTIME_DIR = locate_runtime_dir()
ASSETS_DIR = locate_assets_dir()
_DLL_HANDLES = []


def configure_runtime_environment():
    """Expose bundled extensions and DLLs before importing mpv/native modules."""
    os.environ['SYLC_PROJECT_ROOT'] = PROJECT_ROOT
    os.environ['SYLC_RUNTIME_DIR'] = RUNTIME_DIR
    os.environ['SYLC_ASSETS_DIR'] = ASSETS_DIR
    if RUNTIME_DIR not in sys.path:
        sys.path.insert(0, RUNTIME_DIR)
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    if os.path.normcase(RUNTIME_DIR) not in {
            os.path.normcase(os.path.abspath(p)) for p in path_entries if p}:
        os.environ['PATH'] = RUNTIME_DIR + os.pathsep + os.environ.get('PATH', '')
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        try:
            _DLL_HANDLES.append(os.add_dll_directory(RUNTIME_DIR))
        except OSError:
            pass
    return RUNTIME_DIR


def runtime_file(name):
    return os.path.join(RUNTIME_DIR, name)


def asset_file(name):
    path = os.path.join(ASSETS_DIR, name)
    return path if os.path.exists(path) else None


__all__ = [
    'PROJECT_ROOT', 'RUNTIME_DIR', 'ASSETS_DIR',
    'configure_runtime_environment', 'runtime_file', 'asset_file',
]

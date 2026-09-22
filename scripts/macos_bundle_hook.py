"""Resolve libmpv inside the frozen application before importing python-mpv."""
import ctypes.util
import glob
import os
import sys

_original_find_library = ctypes.util.find_library


def _bundle_find_library(name):
    if name == 'mpv':
        candidate = os.path.join(sys._MEIPASS, 'runtime', 'libmpv.2.dylib')
        if os.path.isfile(candidate):
            return candidate
    if name in ('avcodec', 'avformat', 'avutil', 'avdevice', 'swscale', 'swresample'):
        candidates = glob.glob(os.path.join(sys._MEIPASS, 'lib' + name + '.*.dylib'))
        if candidates:
            return sorted(candidates)[0]
    return _original_find_library(name)


ctypes.util.find_library = _bundle_find_library

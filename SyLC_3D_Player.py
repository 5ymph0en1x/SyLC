# -*- coding: utf-8 -*-

"""
HDR/3D Video Player - Premium Edition V7b
Description: A luxurious, high-quality HDR and 3D video player using PySide6 and libmpv.
             Optimized for 3D Framepacking output with Nvidia 3D Vision support.
             Compatible with Sony VPL-HW55ES projector.
Version: V7b - V7a + CRITICAL MEMORY LEAK FIX
         - All V7a features (file switch cleanup, crash prevention)
         - CRITICAL FIX: 64GB memory leak in minutes (V7b)
         - Decoder throttling when queue is full
         - Periodic garbage collection
         - Limited presentation queue to 72 frames (~432MB max)
         - Production ready for long playback sessions
"""

import sys
import os

# Source checkout: keep the application package under src/ while preserving
# this stable launcher name for users, shortcuts and Nuitka.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PYTHON_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, 'src')
if _PYTHON_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_SOURCE_ROOT)

from sylc.runtime_paths import (
    ASSETS_DIR, RUNTIME_DIR, asset_file, configure_runtime_environment,
)

configure_runtime_environment()

# Fix encoding for Unicode characters without replacing/closing the stream object.
# Re-wrapping ``sys.stdout.buffer`` broke pytest capture and could close a host
# application's underlying descriptor when the wrapper was garbage-collected.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError):
            # GUI builds may expose None or a custom stream without reconfigure().
            pass

import math

from sylc.synth3d_aspect import (
    MIN_NATIVE_WIDE_RATIO,
    select_installed_aspect_model,
)
from sylc.synth3d_stereo_comfort import (
    StereoComfortEnvelope,
    StereoDisplayGeometry,
)
from sylc.stereo_eye_order import (
    LEFT_FIRST, RIGHT_FIRST, UNKNOWN, normalise_eye_order,
)

# CRITICAL HDR FIX: Disable Windows Fullscreen Optimizations
# This prevents Windows from detecting borderless fullscreen and switching HDR off
# Must be set BEFORE any window/graphics initialization
os.environ['__COMPAT_LAYER'] = 'DISABLEDXMAXIMIZEDWINDOWEDMODE'

# CRITICAL FIX: Ensure DLLs and modules are found for Nuitka onefile builds
def _setup_nuitka_paths():
    """Setup sys.path and DLL directories for Nuitka onefile builds.

    In Nuitka onefile mode, files are extracted to a temp directory.
    This function finds that directory and adds it to sys.path BEFORE
    any .pyd modules are imported.
    """
    import tempfile

    dirs_to_add = [('runtime', RUNTIME_DIR)]

    # PRIORITY 1: Nuitka's __compiled__ module (most reliable for onefile)
    # This gives us the ACTUAL extraction directory, not the exe location
    try:
        import __compiled__
        if hasattr(__compiled__, 'containing_dir'):
            containing_dir = __compiled__.containing_dir
            if containing_dir and os.path.isdir(containing_dir):
                dirs_to_add.append(('__compiled__.containing_dir', containing_dir))
                print(f"[NUITKA-PATH] Found __compiled__.containing_dir: {containing_dir}")
    except ImportError:
        pass

    # PRIORITY 2: Nuitka's __nuitka_binary_dir (Nuitka 1.x+)
    if hasattr(sys, '__nuitka_binary_dir'):
        nuitka_dir = sys.__nuitka_binary_dir
        if nuitka_dir and os.path.isdir(nuitka_dir):
            dirs_to_add.append(('__nuitka_binary_dir', nuitka_dir))
            print(f"[NUITKA-PATH] Found __nuitka_binary_dir: {nuitka_dir}")

    # PRIORITY 3: Search TEMP for Nuitka onefile extraction directories
    # Nuitka extracts to %TEMP%/onefile_<pid>_<timestamp>/
    # Only do this when actually running as a Nuitka compiled binary,
    # otherwise stale temp dirs with wrong Python version .pyd files cause ImportError.
    _is_nuitka = any(name in {
        '__compiled__.containing_dir', '__nuitka_binary_dir'
    } for name, _path in dirs_to_add)
    if _is_nuitka:
        try:
            temp_base = tempfile.gettempdir()
            pyd_name = 'mvc_demuxer_cpp.cp314-win_amd64.pyd'
            dll_name = 'edge264.dll'

            for entry in os.listdir(temp_base):
                if entry.startswith('onefile_'):
                    onefile_dir = os.path.join(temp_base, entry)
                    if os.path.isdir(onefile_dir):
                        # Check if our files are there
                        pyd_path = os.path.join(onefile_dir, pyd_name)
                        dll_path = os.path.join(onefile_dir, dll_name)
                        if os.path.exists(pyd_path) or os.path.exists(dll_path):
                            dirs_to_add.append(('TEMP/onefile_*', onefile_dir))
                            print(f"[NUITKA-PATH] Found onefile extraction: {onefile_dir}")
                            break
        except Exception as e:
            print(f"[NUITKA-PATH] TEMP search failed: {e}")

    # PRIORITY 4: __file__ directory (dev mode or some Nuitka configs)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir and os.path.isdir(script_dir):
            dirs_to_add.append(('__file__', script_dir))
    except Exception:
        pass

    # PRIORITY 5: Executable directory (standalone folder mode)
    try:
        exe_dir = os.path.dirname(sys.executable)
        if exe_dir and os.path.isdir(exe_dir):
            dirs_to_add.append(('sys.executable', exe_dir))
    except Exception:
        pass

    # PRIORITY 6: CWD as fallback
    try:
        cwd = os.getcwd()
        if cwd and os.path.isdir(cwd):
            dirs_to_add.append(('cwd', cwd))
    except Exception:
        pass

    # Deduplicate paths while preserving order
    seen = set()
    unique_dirs = []
    for name, path in dirs_to_add:
        if path not in seen:
            seen.add(path)
            unique_dirs.append((name, path))

    # Add all directories to sys.path for importing .pyd modules
    for name, d in unique_dirs:
        if d not in sys.path:
            sys.path.insert(0, d)
            print(f"[NUITKA-PATH] Added to sys.path: {d} ({name})")

    # Add DLL directories on Windows (Python 3.8+)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        for name, d in unique_dirs:
            try:
                os.add_dll_directory(d)
                print(f"[NUITKA-PATH] Added to DLL path: {d}")
            except Exception as e:
                print(f"[NUITKA-PATH] Failed to add DLL path {d}: {e}")

    # Return the primary directory (first valid one with our files)
    pyd_name = 'mvc_demuxer_cpp.cp314-win_amd64.pyd'
    for name, d in unique_dirs:
        if os.path.exists(os.path.join(d, pyd_name)):
            print(f"[NUITKA-PATH] Primary dir (has .pyd): {d}")
            return d

    # Fallback to first directory
    if unique_dirs:
        print(f"[NUITKA-PATH] Primary dir (fallback): {unique_dirs[0][1]}")
        return unique_dirs[0][1]

    return os.getcwd()

APP_BASE_DIR = _setup_nuitka_paths()


def _activate_pending_native_renderer():
    """Atomically promote a validated native-renderer rebuild on next launch.

    Windows locks imported ``.pyd`` files for the lifetime of the process. A
    running SyLC instance therefore cannot be patched in place; staging beside
    the module and promoting before the first import gives us an atomic,
    crash-safe hand-off after restart. Failure is harmless and retried next time.
    """
    if sys.platform != 'win32':
        return
    try:
        current = os.path.join(
            RUNTIME_DIR, 'mvc_demuxer_cpp.cp314-win_amd64.pyd')
        pending = current + '.pending'
        if not os.path.isfile(pending):
            return
        os.replace(pending, current)
        print("[STARTUP] Activated pending native renderer update")
    except OSError as exc:
        print(f"[STARTUP] Native renderer update pending (module still locked): {exc}")


_activate_pending_native_renderer()

import tempfile
import time
import shutil
import glob
import ctypes
import multiprocessing
import logging
from logging.handlers import RotatingFileHandler
import traceback
import threading
import atexit

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QFileDialog, QMessageBox, QGraphicsOpacityEffect, QStackedLayout,
)

_SYLC_LOG_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(),
    "SyLC3DPlayer", "logs")
_log_handlers = []
try:
    os.makedirs(_SYLC_LOG_DIR, exist_ok=True)
    _log_handlers.append(RotatingFileHandler(
        os.path.join(_SYLC_LOG_DIR, "sylc-player.log"),
        maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"))
except OSError:
    # Read-only/locked profile: retain console logging when one exists.
    pass
if sys.stderr is not None:
    _log_handlers.append(logging.StreamHandler())
# INFO by default; SYLC_LOG_LEVEL raises (or lowers) it without touching the code.
# Several diagnostics that answer "is data actually flowing?" are logged at DEBUG
# because they sit in per-frame hot loops -- [POLL-SUBS] on the PGS path is the
# prime example. Without this override they could only be seen by editing this
# file, which is exactly the wrong thing to ask of someone chasing a bug.
_log_level = getattr(logging, os.environ.get('SYLC_LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s',
    handlers=_log_handlers or None)
logger = logging.getLogger(__name__)

from sylc.player_constants import PRESENTATION_KEYS

# Grace period between "the viewer closed the window" and a forced process
# exit (see _arm_exit_watchdog). The honest teardown costs ~5 s worst case
# (decoder join 5 s, mpv terminate up to 2.7 s, ISO dismount) — this sits well
# above it, and well below the patience for an app that will not go away.
# Cooperative close can now spend up to 7 s joining an export and 5 s joining
# each native decoder before the last-resort exit.  Keep the watchdog outside
# that legitimate budget instead of cutting cleanup off mid-flight.
_EXIT_WATCHDOG_S = 25.0


def _hard_exit(code=0):
    """Leave the process NOW, skipping interpreter shutdown. Module level so
    the watchdog's contract is testable without killing the test runner."""
    # Do not acquire logging locks here.  This function is the last resort for
    # precisely the cases where another thread may be wedged while owning one.
    os._exit(code)


def _install_warning_filters():
    """Silence the RuntimeWarning PySide6 raises when we disconnect a signal
    that has no current connection (routine during MVC cleanup: pgsDataReady
    and friends are only connected for some sources). The failure itself is
    already caught by the try/except around each disconnect.

    The pattern must lead with `.*`: warnings anchors `message` at the START
    of the text (re.match) and PySide prefixes its own with "libpyside: ", so
    a pattern starting at "Failed to disconnect" never matched and every
    teardown printed the noise this filter exists to remove."""
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*Failed to disconnect .* from signal",
    )
if _log_level != logging.INFO:
    logger.info("[STARTUP] log level = %s (SYLC_LOG_LEVEL)",
                logging.getLevelName(_log_level))




# edge264's own parallelism runs out well before the core count does: slice and
# macroblock dependencies bound how many workers can make progress at once.
# Measured 2026-08-06 on 1080p MVC (Ryzen 9 5950X, 16C/32T, decode only, access
# units preloaded -- tools_dev/bench_decode.py):
#
#     workers    fps    p99 ms          workers    fps    p99 ms
#           0   51.5      72.2                6  144.0      10.7
#           2  104.9      29.0                8  142.6      10.3
#           3  131.7      16.8               12  145.1      10.0
#           4  136.0      14.4               16  144.7      10.4
#
# Throughput is flat past six and every count decodes bit-identical pixels.
# Above six only p99 still moves, and by less than the contention that extra
# workers add back on a machine that is also running GUI, audio and depth.


# --- FLUIDITY FIX: Force Windows High Resolution Timer (1ms) ---
if sys.platform == 'win32':
    try:
        from ctypes import windll
        timeBeginPeriod = windll.winmm.timeBeginPeriod
        timeEndPeriod = windll.winmm.timeEndPeriod
        if timeBeginPeriod(1) == 0:
            atexit.register(timeEndPeriod, 1)
            logger.info("[FLUIDITY] Windows High Resolution Timer enabled (1ms).")
        else:
            logger.warning("[FLUIDITY] Windows rejected the 1ms timer request.")
    except Exception as e:
        logger.warning(f"[FLUIDITY] Failed to set high resolution timer: {e}")

print("=" * 80)
print("[STARTUP] SyLC 3D Player V7b (Memory Leak Fix) - Initialisation...")
print("=" * 80)

import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

try:
    import cv2

    print("[STARTUP] cv2 imported")
except ImportError as e:
    print(f"[CRITICAL] Unable to import 'cv2' (OpenCV): {e}")
    sys.exit(1)

print("[STARTUP] Base imports succeeded")

from PySide6.QtCore import Qt, QTimer, Signal, QPoint, Slot, QEvent, QThread, QSettings
from PySide6.QtGui import QImage, QIcon, QCursor


# --- Human-readable track labels (audio / subtitle) -------------------------------
# mpv/MakeMKV tracks often carry placeholder titles like "TRACK_1" that mean nothing.
# Build a meaningful label from language + codec (+ channel layout for audio) instead.
from sylc.track_metadata import (
    _find_clpi_for_media, _friendly_track_label, _humanize_codec,
    _humanize_lang, _parse_clpi_languages, _track_int,
)

# The VU meter's astats filter, attached once per mpv instance by _ensure_vu_af.
#
# The measure_* restrictions are LOAD-BEARING, not tidiness. Without them astats
# computes its full statistic set -- bit depth, DC offset, crest/flat factors,
# entropy, dynamic range, zero crossings, ~30 measures -- FOR EVERY CHANNEL, and
# `reset=1` makes it redo all of that every frame. on_vu_metadata reads exactly
# four values out of that: RMS_level and Peak_level for channels 1 and 2.
#
# On stereo and 5.1 the waste is invisible. On a lossless TrueHD 7.1 track --
# 8 channels of s32 at 48 kHz -- it stalls mpv's audio filter chain outright.
# Measured on No Time To Die (TrueHD 7.1) after a seek to 600 s, sampling
# time-pos every 250 ms for 8 s:
#
#   TrueHD 7.1, no filter                  0 stalls / 31   (baseline)
#   TrueHD 7.1, unrestricted astats        8 stalls / 26   <-- ~1 s on, ~1 s off
#   TrueHD 7.1, this filter                0 stalls / 31
#   AC3 5.1,    unrestricted astats        0 stalls / 31   (why it looked 7.1-only)
#
# The restricted form also more than doubled the meter's update rate (79 -> 174
# callbacks in the same window), because the chain no longer stalls.
#
# If a future meter needs another statistic, add it HERE as well as in
# on_vu_metadata -- tests/test_vu_filter.py pins the two equal so they cannot
# drift apart, which is the failure this comment exists to prevent.
VU_ASTATS_FILTER = ('@vu:lavfi=[astats=metadata=1:reset=1'
                    ':measure_perchannel=Peak_level+RMS_level'
                    ':measure_overall=none]')










# --- Blu-ray .clpi language map (raw M2TS/SSIF carries no language tag) ------------
# Blu-ray stream_coding_type groups (per the BD spec / libbluray):






def _find_asset(name):
    """Locate a bundled asset (e.g. icon.png) across source / Nuitka --standalone / --onefile."""
    packaged = asset_file(name)
    if packaged:
        return packaged
    cands = []
    try:
        import __compiled__
        if hasattr(__compiled__, 'containing_dir'):
            cands.append(__compiled__.containing_dir)
    except Exception:
        pass
    for getter in (lambda: os.path.dirname(os.path.abspath(__file__)),
                   lambda: os.path.dirname(os.path.abspath(sys.argv[0])),
                   os.getcwd):
        try:
            cands.append(getter())
        except Exception:
            pass
    for d in cands:
        for folder in (os.path.join(d, 'assets'), d):
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
    return None
import mpv
import numpy as np
from sylc.premium_controls_overlay import PremiumControlsOverlay as ControlsOverlay
from sylc.stereo_hud import StereoHudController

print("[STARTUP] PySide6/mpv/numpy imports succeeded")

# Import MonitoringOverlay (always available, not MVC-dependent)
from sylc.monitoring_overlay import MonitoringOverlay
print("[STARTUP] OK MonitoringOverlay importe")

print("[STARTUP] Freeware build (no license system)")

# -----------------------------------------------------------------------------
# Edge264 MVC decoder integration (PRO: no mocks)
# -----------------------------------------------------------------------------
MVC_SUPPORT_AVAILABLE = False
SYNC_TRACER_AVAILABLE = False

try:
    print("[STARTUP] Attempting to import MVC modules...")

    # Ensure the canonical src package wins over stale neighbouring checkouts.
    _script_dir = _PYTHON_SOURCE_ROOT
    print(f"[STARTUP] Python source directory: {_script_dir}")
    print(f"[STARTUP] sys.path[0:5]: {sys.path[:5]}")

    # Force script directory to the very front of sys.path
    if _script_dir in sys.path:
        sys.path.remove(_script_dir)
    sys.path.insert(0, _script_dir)

    # Clear any cached import of mvc_decoder
    if 'sylc.mvc_decoder' in sys.modules:
        cached_file = getattr(sys.modules['sylc.mvc_decoder'], '__file__', 'unknown')
        if _script_dir not in str(cached_file):
            del sys.modules['sylc.mvc_decoder']

    # V42: Make mvc_demuxer_cpp optional - ctypes fallback will be used if unavailable
    try:
        import mvc_demuxer_cpp
        print("[STARTUP] OK mvc_demuxer_cpp imported")
    except ImportError as pyd_err:
        mvc_demuxer_cpp = None
        print(f"[STARTUP] mvc_demuxer_cpp not available ({pyd_err}) - ctypes fallback will be used")

    from sylc.mvc_decoder import MVCDecoderThread
    print("[STARTUP] OK MVCDecoderThread imported")

    # D3D11 NATIVE rendering for HDR preservation in fullscreen
    # Directive 2: the detached 3D window hosts the native C++ D3D11 renderer.
    # The Qt RHI widget + the OpenGL/hybrid fallbacks are gone.
    from sylc.framepacking_window_d3d11 import Framepacking3DWindow
    print("[STARTUP] OK Framepacking3DWindow (native C++ D3D11 HDR) imported")

    # Sync Tracer for pipeline diagnostics (V7 feature)
    try:
        from sync_tracer import get_tracer, SyncStage

        SYNC_TRACER_AVAILABLE = True
        print("[STARTUP] OK SyncTracer imported (V7 feature)")
    except ImportError:
        SYNC_TRACER_AVAILABLE = False
        print("[STARTUP] SyncTracer not available (optional)")

    # PGS Subtitle System for MVC mode
    try:
        from sylc.subtitle_manager import SubtitleManager
        from sylc.subtitle_extractor import SubtitleExtractor, get_pgs_tracks

        PGS_SUBTITLE_AVAILABLE = True
        print("[STARTUP] OK PGS Subtitle System imported")
    except ImportError as e:
        PGS_SUBTITLE_AVAILABLE = False
        print(f"[STARTUP] PGS Subtitle not available: {e}")

    MVC_SUPPORT_AVAILABLE = True
    print("[STARTUP] === Full MVC support available ===")
except ImportError as e:
    print(f"[CRITICAL] Failed to import MVC modules: {e}")
    traceback.print_exc()
    MVC_SUPPORT_AVAILABLE = False
    SYNC_TRACER_AVAILABLE = False
    PGS_SUBTITLE_AVAILABLE = False
    print("[STARTUP] Degraded mode: MVC support disabled.")

# Text subtitle overlay (SRT/ASS) for MVC/edge264 mode — mpv decodes the text
# track on the shared audio clock, we paint it on the native overlay.
try:
    from sylc.text_subtitle_renderer import TextSubtitleRenderer

    TEXT_SUBTITLE_AVAILABLE = True
    print("[STARTUP] OK Text Subtitle renderer imported")
except ImportError as e:
    TEXT_SUBTITLE_AVAILABLE = False
    print(f"[STARTUP] Text Subtitle renderer not available: {e}")

print(f"[STARTUP] MVC_SUPPORT_AVAILABLE = {MVC_SUPPORT_AVAILABLE}")
print(f"[STARTUP] SYNC_TRACER_AVAILABLE = {SYNC_TRACER_AVAILABLE}")
print(f"[STARTUP] PGS_SUBTITLE_AVAILABLE = {PGS_SUBTITLE_AVAILABLE if 'PGS_SUBTITLE_AVAILABLE' in dir() else False}")

# Containers edge264 can decode H.264 from. The native C++ demuxer handles
# MKV/M2TS/TS/SSIF; the libavformat-backed demuxer (lavf_h264_demuxer, task #391)
# adds MP4/AVI/MOV/FLV/WebM/raw when the bundled ffmpeg DLLs are present. Any
# edge264 failure on these degrades to mpv via _fallback_from_edge264 (#388).
try:
    from sylc import lavf_h264_demuxer as _lavf
    _LAVF_AVAILABLE = _lavf.is_available()
except Exception:
    _LAVF_AVAILABLE = False
EDGE264_CONTAINERS = ('.mkv', '.mk3d', '.m2ts', '.ts')
if _LAVF_AVAILABLE:
    EDGE264_CONTAINERS = EDGE264_CONTAINERS + ('.mp4', '.m4v', '.mov', '.avi', '.flv',
                                               '.wmv', '.webm', '.mpg', '.mpeg',
                                               '.h264', '.264', '.avc')
print(f"[STARTUP] LAVF (MP4/AVI/raw via edge264) = {_LAVF_AVAILABLE}")




# Native C++ D3D11 renderer availability — the SOLE video render path since the
# Directive 2 cutover (Qt RHI removed). edge264 routing requires it; without it,
# mpv handles everything.
try:
    import mvc_demuxer_cpp as _mdc_native
    NATIVE_RENDER_AVAILABLE = bool(getattr(_mdc_native, 'NATIVE_RENDERER_AVAILABLE', False))
except Exception:
    NATIVE_RENDER_AVAILABLE = False
print(f"[STARTUP] NATIVE_RENDER_AVAILABLE = {NATIVE_RENDER_AVAILABLE}")


from sylc.synth3d_policy import (
    SYNTH3D_DEPTH_PRESETS, SYNTH3D_DEPTH_PRESET_DEFAULT,
    SYNTH3D_ADAPTIVE_MODEL_GRIDS, SYNTH3D_SEEK_KEEP_ASPECT,
    SYNTH3D_ADVISORY_PTS_CLOCK, SYNTH3D_SETTINGS_ORG,
    SYNTH3D_SETTINGS_APP, SYNTH3D_DEPTH_PRESET_KEY,
    _SOURCE_ROOT,
    _synth3d_seek_keeps_aspect, sylc_user_data_dir,
    sylc_models_download_dir, _synth3d_models_dirs, synth3d_find_model,
    synth3d_depth_preset_entry, synth3d_depth_preset_available,
    synth3d_depth_preset_stored, synth3d_marker_attests,
)
from sylc.native_playback_policy import (
    _EDGE264_WORKER_SATURATION, _physical_core_count,
    _recommended_edge264_threads, _edge264_startup_timeout_ms,
    _select_stereo_presentation_targets,
)



# Seek state machine lives in its own timing-focused module.
# Re-export all three names here for backward-compatible imports.
from sylc.robust_seek_queue import (
    RobustSeekQueue, SeekRequest, SeekState, should_resume_after_sync,
)
# --- Style HDR Image Converter (Professional) ---
APP_STYLE = """
    QMainWindow, QWidget {
        background-color: #1e1e1e;
        color: #F0F0F0;
        font-family: 'Segoe UI', sans-serif;
    }

    QToolTip {
        color: #F5F5F5;
        background-color: #2A2A2A;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        font-family: 'Segoe UI', sans-serif;
    }

    QLabel {
        font-size: 12px;
        color: #DDDDDD;
        font-weight: 400;
    }

    QGroupBox {
        font-size: 11px;
        font-weight: 600;
        color: #AAAAAA;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 8px;
    }

    QPushButton {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 6px;
        color: #FFFFFF;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 0.15);
    }

    QSlider::groove:horizontal {
        border: none;
        height: 4px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #FFFFFF;
        border: none;
        width: 14px;
        height: 14px;
        border-radius: 7px;
        margin: -5px 0;
    }
    QSlider::handle:horizontal:hover {
        background: #007ACC;
        width: 16px;
        height: 16px;
        border-radius: 8px;
        margin: -6px 0;
    }
    QSlider::sub-page:horizontal {
        background: #007ACC;
        border-radius: 2px;
    }

    QComboBox {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 4px 10px;
        color: #E0E0E0;
        font-size: 11px;
        min-width: 60px;
    }
    QComboBox:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        image: none;
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #252525;
        color: #E0E0E0;
        selection-background-color: #007ACC;
        border: 1px solid #333;
        border-radius: 4px;
        outline: none;
    }
"""


from sylc.video_3d_analyzer import (
    Video3DAnalyzer, _FFPROBE_ANALYSIS_TIMEOUT_S, _STEREO_PRIORITY,
    _apply_filename_3d_hint, _check_ffmpeg_runtime,
    _classify_stereo_mode, _describe_windows_returncode,
    _parse_ffprobe_fps, _probe_mvc_extension, _promote_stereo_mode,
    _resolve_external_tool, configure_app_base_dir,
)
configure_app_base_dir(APP_BASE_DIR)






















# GLOBAL ThreadPool for parallel thumbnail extraction (max 2 workers)
from sylc.time_slider import (
    TimeSlider, _decide_thumbs_mode, _extract_thumbnail_ffmpeg,
)




# Re-export auxiliary widgets for compatibility with existing imports.
from sylc.player_widgets import IconButton, InfoOverlay, LoadingOverlay, PreviewTooltip












# --- NEW THREAD (V12) ---


from sylc.mvhevc_export_dialog import MVHEVCExportDialog
from sylc.player_memory_mixin import PlayerMemoryMixin
from sylc.subtitle_track_mixin import SubtitleTrackMixin, configure_subtitle_support
from sylc.cast_mixin import CastMixin
from sylc.archive_export_mixin import ArchiveExportMixin
from sylc.window_presentation_mixin import WindowPresentationMixin
from sylc.synth3d_coordination_mixin import (
    Synth3DCoordinationMixin, configure_synth3d_support,
)
from sylc.native_decoder_mixin import (
    NativeDecoderMixin, configure_native_decoder_support,
    _EYE_INHERITED_RENDER_PARAMS,
)
from sylc.media_session_mixin import (
    MediaSessionMixin, configure_media_session_support,
    _MPV_RELEASE_SETTLE_MS,
)
from sylc.playback_timeline_mixin import PlaybackTimelineMixin
from sylc.media_loading_mixin import (
    MediaLoadingMixin, configure_media_loading_support,
)

configure_subtitle_support(PGS_SUBTITLE_AVAILABLE)
configure_synth3d_support(NATIVE_RENDER_AVAILABLE)
configure_native_decoder_support(
    MVC_SUPPORT_AVAILABLE, NATIVE_RENDER_AVAILABLE,
    globals().get('MVCDecoderThread'), globals().get('Framepacking3DWindow'),
    EDGE264_CONTAINERS,
)
configure_media_session_support(mpv)
configure_media_loading_support(
    MVC_SUPPORT_AVAILABLE, NATIVE_RENDER_AVAILABLE, PGS_SUBTITLE_AVAILABLE,
    EDGE264_CONTAINERS,
)


# Per-source render parameters a Dual Projector eye window must inherit when it
# is built mid-playback: the same four attributes _try_start_hevc writes on the
# embedded/framepack pair at load time.


class PlayerWindow(
        WindowPresentationMixin, Synth3DCoordinationMixin, NativeDecoderMixin,
        MediaSessionMixin, PlaybackTimelineMixin, MediaLoadingMixin, CastMixin,
        ArchiveExportMixin, SubtitleTrackMixin,
        PlayerMemoryMixin, QMainWindow):
    """Main window."""

    # Signals for thread-safe PGS callbacks (cross-thread communication)
    # Every cross-thread payload carries the media-session token that created it.
    # ``object`` keeps the signal surface compact while allowing an immutable dict
    # with token/path/result; slots reject stale payloads before touching live UI.
    pgs_extraction_complete = Signal(object)
    pgs_load_complete = Signal(object)
    pgs_parse_complete = Signal(object)
    pgs_notification = Signal(object)
    pgs_tracks_detected = Signal(object)
    extraction_progress = Signal(object)
    # Text subtitle (SRT/ASS) overlay: mpv's 'sub-text' observer fires on the mpv
    # event thread; this signal marshals the cue text onto the Qt main thread.
    mpv_sub_text_changed = Signal(object)
    # Authored 3D depth of the active text track (disparity, measured pairs) —
    # emitted from the background analysis thread, handled on the main thread.
    text_sub_depth_ready = Signal(object)
    # python-mpv invokes observers from its event thread.  These signals are the
    # only bridge to Qt/UI work; their payload also pins the owning mpv core and
    # media session so an event queued by the previous title cannot affect the next.
    mpv_duration_event = Signal(object)
    mpv_time_event = Signal(object)
    mpv_pause_event = Signal(object)
    mpv_eof_event = Signal(object)

    def __init__(self, parent=None):
        print("[STARTUP] Initializing PlayerWindow...")
        super().__init__(parent)
        print("[STARTUP] QMainWindow.__init__() finished")
        self.setWindowTitle("SyLC 3D Player - Premium Edition")
        _icon_path = _find_asset('icon.png')
        if _icon_path:
            self.setWindowIcon(QIcon(_icon_path))
        self.resize(1280, 850)  # Increased height for better 16:9 video area ratio
        self.setStyleSheet(APP_STYLE)
        self.setAcceptDrops(True)

        # --- LAYOUT FIX (Based on V4) ---
        self.video_container = QWidget()
        self.setCentralWidget(self.video_container)
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(0)

        # Stacked layout for swapping between MPV and MVC widget without GUI shifts
        self.video_stack_container = QWidget()
        self.video_stack = QStackedLayout(self.video_stack_container)
        self.video_stack.setContentsMargins(0, 0, 0, 0)

        self.video_widget = QWidget()
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.video_widget.setStyleSheet("background-color: black;")

        self.video_stack.addWidget(self.video_widget)  # Index 0: MPV
        self.video_layout.addWidget(self.video_stack_container, 1)

        print("[STARTUP] Video widget created (Stacked)")

        self.metrics_overlay = MonitoringOverlay(self.video_widget)
        self.metrics_overlay.hide()
        self.metrics_overlay.raise_()
        print("[STARTUP] Metrics overlay created")

        self.player = None  # Will be initialized by _setup_mpv_player
        self._app_closing = False
        self._media_session_id = 0
        self._loading_session_id = None
        self._media_cancel_event = threading.Event()
        self._media_workers = set()
        self._media_workers_lock = threading.Lock()
        self._mpv_media_observers = []
        self._mpv_subtext_observer = None
        self._pgs_startup_pending_session = None
        self.mpv_duration_event.connect(self._dispatch_mpv_duration)
        self.mpv_time_event.connect(self._dispatch_mpv_time)
        self.mpv_pause_event.connect(self._dispatch_mpv_pause)
        self.mpv_eof_event.connect(self._dispatch_mpv_eof)

        # --- Controls Overlay (Floating) ---
        # Reparent to self (QMainWindow) to ensure it sits above the Central Widget
        self.controls_overlay = ControlsOverlay(self)
        # Note: We do NOT add it to the layout anymore. It will be positioned manually in resizeEvent.
        self.controls_overlay.raise_()
        # In 3D output modes the same canonical Qt controls are captured into an
        # eye-local RGBA HUD and composed by the native shader. The controller
        # also remaps clicks from either eye back to these widgets.
        self.stereo_hud = StereoHudController(self, self.controls_overlay)

        print("[STARTUP] Controls overlay created")
        self.info_overlay = InfoOverlay("Click here or drop a file", self)
        print("[STARTUP] Info overlay created")
        self.loading_overlay = LoadingOverlay(self)
        print("[STARTUP] Loading overlay created")
        # --- END LAYOUT FIX ---

        # MVC related
        self.demuxer = None
        self.mvc_decoder_thread = None
        self._mvc_leaked = []
        self._mvc_shutdown_blocked = False
        self._hevc_leaked = []
        self._hevc_shutdown_blocked = False
        self.mvc_mode_active = False
        self.framepacking_window = None  # Will be created when needed
        self.eye_windows = None   # Dual Projector: (left, right) while active

        # V14b: State flags for graceful shutdown
        self._playback_ended = False
        self._mpv_transition_in_progress = False

        # PGS Subtitle System for MVC mode
        self._subtitle_manager = None
        self._subtitle_extractor = None
        self._pgs_subtitle_tracks = []  # List of detected PGS tracks
        self._active_pgs_track_index = None  # Currently selected PGS track stream index
        self._subtitle_connected_widgets = []  # Track which widgets have subtitle signals connected
        # ========== STREAMING SUBTITLE SUPPORT ==========
        self._streaming_subtitle_tracks = []  # Tracks detected from demuxer (no extraction needed)
        self._active_streaming_track = None   # Currently active streaming track number
        # ================================================
        # ========== TEXT SUBTITLE (SRT/ASS) OVERLAY ==========
        self._text_subtitle_renderer = None
        self._text_sub_active = False          # True while a text track feeds the overlay
        self._text_sub_connected_widgets = []  # Widgets currently wired to the text renderer
        self._mpv_subtext_observer_registered = False
        self._sub_depth_cache = {}             # (filepath, sub_index) -> disparity
        self._active_text_sub_depth_key = None
        if TEXT_SUBTITLE_AVAILABLE:
            self._text_subtitle_renderer = TextSubtitleRenderer(self)
            self.mpv_sub_text_changed.connect(self._on_mpv_sub_text)
            self.text_sub_depth_ready.connect(self._on_text_sub_depth)
            print("[STARTUP] Text SubtitleRenderer initialized")
        # =====================================================
        if PGS_SUBTITLE_AVAILABLE:
            self._subtitle_manager = SubtitleManager(self)
            self._subtitle_extractor = SubtitleExtractor()
            # Connect PGS signals for thread-safe callbacks
            self.pgs_extraction_complete.connect(self._on_pgs_extraction_complete)
            self.pgs_load_complete.connect(self._finish_pgs_load)
            self.pgs_parse_complete.connect(self._on_pgs_parsed)
            self.pgs_notification.connect(self._on_pgs_notification)
            self.pgs_tracks_detected.connect(self._on_pgs_tracks_detected)
            self.extraction_progress.connect(self._on_extraction_progress)
            print("[STARTUP] PGS SubtitleManager initialized")

        # Audio synchronization based on the decoder markers
        # V7b STABILITY FIX: DISABLED - causes crashes with MPV thread safety
        # Timeline progression works without this (uses _last_mvc_timestamp instead)
        self._audio_sync_enabled = True  # SOL 2A: Re-enabled (crashes fixed by hybrid wait SOL 3A)

        # --- SYNC PARAMETERS (Tuned for V7b) ---
        self.SYNC_BIAS_WINDOW_MS = 50.0  # Window to learn constant offsets
        self.SYNC_BIAS_LEARNING_RATE = 0.05
        self.SYNC_BIAS_MAX_MS = 100.0
        
        self.SYNC_ACCEPTABLE_MS = 45   # Tightened to ~1 frame (was 200ms). Syncs sooner.
        self.SYNC_MICRO_ADJUST_MS = 250 # 45-250ms: Micro frame timing adjustment
        self.SYNC_DRIFT_THROTTLE_S = 0.5  # Min 0.5s between drift adjustments

        self._last_frame_timestamp = 0.0
        self._decoder_start_position = 0.0
        self._last_drift_adjust_time = 0.0
        self._cumulative_drift = 0.0
        self._sync_bias = 0.0  # Low-pass bias to cancel constant offset
        # Non-blocking mpv audio clock.  The observer provides sparse media-time
        # samples; _mpv_time_pos_ms extrapolates them at 1x using monotonic time.
        self._mpv_time_pos_cache = None
        self._mpv_time_pos_cache_mono = None
        self._mpv_pause_cache = True

        # V60: persisted per-install settings (currently: the A/V sync trim set
        # with [ and ] — re-applied to every new decoder thread).
        self._app_settings = self._load_app_settings()

        # 2D->3D AI synthesis (real-time depth-based stereo synthesis on the
        # native decode path): state + persisted strength/convergence.
        self._synth3d_active = False
        try:
            self._synth3d_strength = max(0.5, min(3.0,
                float(self._app_settings.get('synth3d_strength_pct', 1.5))))
            self._synth3d_convergence = max(0.0, min(1.0,
                float(self._app_settings.get('synth3d_convergence', 0.5))))
        except Exception:
            self._synth3d_strength, self._synth3d_convergence = 1.5, 0.5
        self._synth3d_auto_convergence = bool(
            self._app_settings.get('synth3d_auto_convergence', False))
        self._synth3d_temporal_fill = bool(
            self._app_settings.get('synth3d_temporal_fill', False))
        # Calibrated binocular comfort profile for the author's Full-HD
        # projector: 2.5 m image width at 3.5 m. The physical screen height is
        # retained for the future vertical/window analysis; horizontal
        # disparity depends on the illuminated image width.
        try:
            def _comfort_number(env_key, setting_key, default):
                return float(os.environ.get(
                    env_key, self._app_settings.get(setting_key, default)))

            # v5.2.1c owns the displayed disparity again. Retain this physical
            # geometry for Lab metrics, but never restore a persisted comfort
            # corrector on top of the reference warp.
            self._synth3d_comfort_enabled = False
            self._synth3d_comfort_screen_height_m = _comfort_number(
                'SYLC_COMFORT_SCREEN_HEIGHT_M',
                'synth3d_comfort_screen_height_m', 1.7)
            _geometry = StereoDisplayGeometry(
                screen_width_m=_comfort_number(
                    'SYLC_COMFORT_SCREEN_WIDTH_M',
                    'synth3d_comfort_screen_width_m', 2.5),
                horizontal_pixels=int(round(_comfort_number(
                    'SYLC_COMFORT_HORIZONTAL_PIXELS',
                    'synth3d_comfort_horizontal_pixels', 1920))),
                viewing_distance_m=_comfort_number(
                    'SYLC_COMFORT_VIEWING_DISTANCE_M',
                    'synth3d_comfort_viewing_distance_m', 3.5),
                interpupillary_distance_m=_comfort_number(
                    'SYLC_COMFORT_IPD_M', 'synth3d_comfort_ipd_m', 0.064))
            self._synth3d_comfort_envelope = StereoComfortEnvelope(
                _geometry,
                soft_vac_diopters=_comfort_number(
                    'SYLC_COMFORT_SOFT_VAC_D',
                    'synth3d_comfort_soft_vac_d', 0.18),
                hard_vac_diopters=_comfort_number(
                    'SYLC_COMFORT_HARD_VAC_D',
                    'synth3d_comfort_hard_vac_d', 0.30))
        except (TypeError, ValueError, OverflowError):
            logger.exception("[2D3D] invalid comfort profile; keeping v5.2.1c baseline")
            self._synth3d_comfort_enabled = False
            self._synth3d_comfort_screen_height_m = 1.7
            self._synth3d_comfort_envelope = StereoComfortEnvelope(
                StereoDisplayGeometry(2.5, 1920, 3.5, 0.064))
        self._synth3d_depth_view = False
        self._synth3d_diagnostics = False
        # (absolute square-model path, square side, AspectSelection). Kept for
        # the current medium so toggling AI off/on does not repeat detection;
        # cleared on file or depth-preset changes.
        self._synth3d_aspect_override = None
        self._synth3d_aspect_unavailable_key = None
        # True only inside _synth3d_set_depth_preset's toggle-off/toggle-on pair:
        # tells _on_framepacking_visibility_changed that the framepack hide it is
        # about to see is deliberate and momentary.
        self._synth3d_rearming = False
        self._synth3d_preset = str(
            self._app_settings.get('synth3d_preset', 'custom')).lower()
        if self._synth3d_preset not in {
                'comfort', 'cinema', 'immersion', 'custom'}:
            self._synth3d_preset = 'custom'
        # MatAnyone 2 is an optional, isolated CUDA worker. "auto" is the
        # production default: it starts only when the complete offline runtime
        # is installed, and otherwise leaves the depth-only renderer untouched.
        _matting_mode = os.environ.get('SYLC_MATANYONE2', 'auto').strip().lower()
        self._synth3d_matting_requested = _matting_mode not in {
            '0', 'false', 'off', 'disabled', 'no'}
        self._synth3d_matte_service = None
        self._synth3d_matte_unavailable_logged = False
        # Shot identity for every temporal matte component.  The look-ahead
        # observes a cut before presentation; the reset itself is deferred to
        # the first displayed frame at/after that absolute PTS so the dying
        # shot keeps its valid matte while T0 can never see it.
        self._synth3d_pending_cut_pts = None
        self._synth3d_matte_cut_seen_ms = -math.inf
        self._synth3d_matte_floor_pts_ms = -math.inf

        # --- SEEK / SCRUBBING STATE ---
        # Standard "Seek on Release" logic to prevent decoder saturation
        self._is_scrubbing = False         # True while user is dragging the slider
        self._was_playing_before_scrub = False # To restore playback state after seek
        self._next_seek_target = None # Keep this for safety if needed, though release logic replaces it
        # Robust seek queue (debounce/cooldown + signals)
        # NOTE: Signals are already connected in RobustSeekQueue.__init__
        # DO NOT reconnect here to avoid double execution!
        self._seek_queue = RobustSeekQueue(self)

        # Seek-race repro harness (DEV ONLY, env-gated SYLC_SEEK_STRESS=<sec>): auto-seek
        # through the real user seek path to reproduce the intermittent D3D11 render-thread
        # crash (0xe24c4a02). No-op unless the env var is set.
        self._seek_stress_n = 0
        _stress = os.environ.get('SYLC_SEEK_STRESS', '')
        if _stress:
            try:
                self._seek_stress_interval = max(1.0, float(_stress))
            except Exception:
                self._seek_stress_interval = 3.0
            self._seek_stress_timer = QTimer(self)
            self._seek_stress_timer.timeout.connect(self._seek_stress_tick)
            QTimer.singleShot(12000, lambda: self._seek_stress_timer.start(int(self._seek_stress_interval * 1000)))
            logger.warning(f"[SEEK-STRESS] enabled: auto-seek every {self._seek_stress_interval:.1f}s after 12s warmup")

        # Reload repro harness (DEV ONLY, env-gated SYLC_RELOAD_AFTER=<sec>): load a 2nd
        # file (SYLC_RELOAD_FILE, default = same file) to reproduce the black-screen-on-
        # reload bug through the real play_file path. No-op unless the env var is set.
        self._reload_done = False
        _reload = os.environ.get('SYLC_RELOAD_AFTER', '')
        if _reload:
            try:
                self._reload_after = max(5.0, float(_reload))
            except Exception:
                self._reload_after = 40.0
            self._reload_file = os.environ.get('SYLC_RELOAD_FILE', '')
            QTimer.singleShot(int(self._reload_after * 1000), self._reload_test_tick)
            logger.warning(f"[RELOAD-TEST] will load a 2nd file after {self._reload_after:.0f}s")

        # --- MVC Performance Fix: Utiliser multiprocessing.Array ---
        self.MVC_WIDTH = 1920
        self.MVC_HEIGHT = 2205
        self.MVC_CHANNELS = 3
        buffer_size = self.MVC_WIDTH * self.MVC_HEIGHT * self.MVC_CHANNELS

        try:
            self.shared_buffer = multiprocessing.Array(ctypes.c_ubyte, buffer_size)
            print("[MVC INIT] Shared memory buffer allocated.")
        except Exception as e:
            print(f"[CRIT] Failed to allocate shared memory buffer: {e}")
            self.shared_buffer = None
            self._mvc_restarting = False
            self.mvc_mode_active = False

        # Pre-allocate the numpy buffers for the BGR->RGB conversion
        self.rgb_frame_buffer = np.zeros((self.MVC_HEIGHT, self.MVC_WIDTH, self.MVC_CHANNELS), dtype=np.uint8)
        self.current_qimage_ref = None  # Reference to prevent garbage collection

        # Monitoring overlay
        self.monitoring_overlay = MonitoringOverlay(self.video_container)
        self.monitoring_overlay.hide()
        print("[STARTUP] Monitoring overlay created")
        self._last_display_frame_ts = None
        self._display_fps_avg = None
        self._framepacking_visible = False
        self._last_stats_log_ts = 0.0
        self._last_decoder_activity_ts = time.monotonic()
        self._last_watchdog_dump_ts = 0.0
        self._stall_watchdog = QTimer(self)
        self._stall_watchdog.setInterval(3000)
        self._stall_watchdog.timeout.connect(self._check_decoder_stall)
        self._edge264_startup_timer = QTimer(self)
        self._edge264_startup_timer.setSingleShot(True)
        self._edge264_startup_timer.timeout.connect(self._on_edge264_startup_timeout)
        self._edge264_waiting_for_first_frame = False
        self._edge264_mpv_handoff_done = False
        self._edge264_pre_handoff_widget = None
        # Do not start the watchdog now - it will be started when the MVC decoder starts

        # State
        self.has_media = False
        self.is_playing = False
        self.is_3d_enabled = False
        self.current_stereo_mode = 'auto'
        self.video_3d_info = None
        self._bd_eye_order = UNKNOWN
        self.current_video_fps = 24.0
        self.current_file_path = None
        self._archiving = False  # True while a disc→ISO image runs (locks playback)
        # EX-4: MV-HEVC export — a single background job at a time (NEVER touches
        # live playback: the exporter uses its own detached decode instances).
        self._export_job = None
        self._export_dialog = None
        # SyLC Cast (Task 13): the live cast session (CastController) or None when idle.
        self._cast = None
        self._cast_connected = False
        self._cast_transport = None       # 'wifi' | 'usb' while a session is up (status lights)
        self._cast_10bit_warned = False   # latch: notify once that 10-bit cast is v1-unsupported
        # EX-4 fix #2: ISO mount(s) whose dismount was skipped because a running
        # export job was still reading from that drive; retried when the job ends.
        self._deferred_iso_dismounts = []
        self.is_3d_capable = False
        self.controls_hide_timer = None  # Lazy initialization
        self._controls_timer_initialized = False
        self._is_loading_file = False  # V7a: Protection against rapid file changes
        self._pending_play_request_id = 0

        # Timer for periodic timeline updates (for MVC mode where MPV may not report time-pos)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100)  # V7b: 100ms refresh for smoother timeline progression
        self._playback_timer.timeout.connect(self._update_playback_position)

        # Audio VU meter: a 30 Hz QTimer paints the widget from _vu_cache, which is
        # filled by the af-metadata/vu OBSERVER (on_vu_metadata) on mpv's event
        # thread — the poll NEVER reads an mpv property (0xe24c4a02 / TrueHD-Atmos
        # GUI-hog rule). (level, peak) tuple + a monotonic freshness stamp.
        self._vu_cache = (0.0, 0.0)
        self._vu_cache_ts = 0.0
        self._vu_timer = QTimer(self)
        self._vu_timer.setInterval(33)
        self._vu_timer.timeout.connect(self._poll_audio_levels)
        self._last_mvc_timestamp = 0.0  # V7b: Store last MVC frame timestamp for timeline updates
        self._last_timeline_update_time = 0.0 # V7b: Store real time of last update for interpolation
        self._current_precise_time = 0.0 # V7b: High-precision float tracker for timeline
        self._last_ui_time = 0.0  # Prevent UI time from jumping backwards
        self._sync_bias = 0.0  # Low-pass bias to cancel constant drift (~200ms) without speed swings

        # V14b RENDER HEARTBEAT: Keep Qt event loop active when controls are hidden in fullscreen
        # This prevents stuttering caused by reduced Qt activity when UI elements are hidden
        self._render_heartbeat_timer = QTimer(self)
        self._render_heartbeat_timer.setTimerType(Qt.TimerType.PreciseTimer)  # Bypass Windows timer coalescing
        self._render_heartbeat_timer.setInterval(8)  # ~120Hz heartbeat for smoother timing
        self._render_heartbeat_timer.timeout.connect(self._render_heartbeat)

        # HDR FIX: Fake fullscreen state (borderless maximized preserves HDR)
        self._is_fake_fullscreen = False
        self._saved_flags = None
        self._saved_geometry = None

        # UX: Mouse tracking for auto-hide
        self._last_mouse_pos = QPoint(0, 0)
        self._mouse_outside_window = False  # Track if mouse left the playback window

        # V15: Mouse inactivity timer - hides controls after 3s of no movement
        self._mouse_inactivity_timer = QTimer(self)
        self._mouse_inactivity_timer.setSingleShot(True)
        self._mouse_inactivity_timer.setInterval(3000)  # 3 seconds
        self._mouse_inactivity_timer.timeout.connect(self._on_mouse_inactivity)

        # NAV BAR auto show/hide — SINGLE authoritative driver. Polls the GLOBAL cursor
        # (robust over every child widget, incl. the D3D11 video, where per-widget
        # mouseMoveEvent doesn't fire). Behaviour: any movement inside the window shows
        # the bar; no movement for 3 s (playing) / 5 s (paused) hides it.
        self._nav_last_cursor = QCursor.pos()
        self._nav_last_activity = time.monotonic()
        self._nav_had_media = False
        self._nav_poll = QTimer(self)
        self._nav_poll.setInterval(120)
        self._nav_poll.timeout.connect(self._nav_poll_tick)
        self._nav_poll.start()

        # SEEK STABILITY
        self._is_seeking = False
        self._was_playing_before_seek = False

        # Initialization (V4 style)
        print("[STARTUP] Calling _initialize_player()...")
        self._initialize_player()
        print("[STARTUP] _initialize_player() finished")

        print("[STARTUP] Connecting signals...")
        self._connect_signals()
        print("[STARTUP] Signals connected")

        print("[STARTUP] Updating UI...")
        self.update_ui_state()
        print("[STARTUP] UI updated")

        print("[STARTUP] Checking 3D Vision...")
        self._check_3d_vision_availability()
        print("[STARTUP] 3D Vision check finished")

        self.thumbnail_cache = {}
        QTimer.singleShot(0, self._update_monitoring_overlay_geometry)
        QTimer.singleShot(0, self._update_metrics_overlay_geometry)
        # Fix: Position floating overlays on startup
        QTimer.singleShot(0, self._update_overlays_geometry)

        # First launch on an empty models/ directory: name the action once,
        # without a modal. Someone who opened the player to watch a Blu-ray
        # should not be met by a 4.6 GB prompt.
        QTimer.singleShot(2000, self._maybe_prompt_for_models)

        print("[STARTUP] PlayerWindow initialized successfully")

    # ----- Media-session ownership ---------------------------------------

    def _maybe_prompt_for_models(self):
        try:
            settings = QSettings(SYNTH3D_SETTINGS_ORG, SYNTH3D_SETTINGS_APP)
            if settings.value("synth3d/models_prompted", False, type=bool):
                return
            manifest_path = self._synth3d_manifest_path()
            if manifest_path is None:
                return
            from sylc import model_fetcher
            manifest = model_fetcher.load_manifest(manifest_path)
            for models_dir in _synth3d_models_dirs():
                if any(s.installed for s in
                       model_fetcher.pack_status(manifest, models_dir).values()):
                    return
            settings.setValue("synth3d/models_prompted", True)
            settings.sync()
            self.show_3d_notification(
                "2D→3D AI: depth models not installed — see “Depth models” "
                "at the top of the 2D→3D menu")
        except Exception:
            logger.warning("[2D3D] first-launch model prompt failed",
                           exc_info=True)

    def _check_3d_vision_availability(self):
        """
        Force enable 3D capabilities without external checks.
        The user requested to remove 3D Vision verification entirely.
        """
        self.is_3d_capable = True
        # The 3D button AND the stereo dropdown start disabled (no media yet);
        # _update_3d_button_state() enables both only when genuine 3D content
        # (MVC / SBS / TAB / MV-HEVC) is loaded.
        self.controls_overlay.mode_3d_button.setEnabled(False)
        try:
            self.controls_overlay.stereo_mode_combo.setEnabled(False)
        except Exception:
            pass
        logger.info("[3D] 3D capabilities forced ENABLED (Validation removed).")

    def show_3d_notification(self, message, success=True, permanent=False):
        """Displays a notification about 3D mode."""
        # Update status label in controls
        status_type = 'success' if success else 'error'
        if not success and 'not detected' in message.lower():
            status_type = 'warning'

        self.controls_overlay.set_status_info(message, status_type=status_type, active=success)

        if not permanent:
            QTimer.singleShot(5000, lambda: self.controls_overlay.set_status_info("Ready"))


    def _connect_signals(self):
        """Connects UI signals to player commands."""
        self.controls_overlay.play_toggled.connect(self.toggle_play)
        self.controls_overlay.stop_clicked.connect(self.stop_playback)
        self.controls_overlay.fullscreen_toggled.connect(self.toggle_fullscreen)
        self.controls_overlay.volume_changed.connect(lambda v: setattr(self.player, 'volume', v))
        
        # --- Seek on Release Implementation ---
        # Disconnect old 'seeked' signal which fired on mouse press/click
        # self.controls_overlay.seeked.connect(self.on_seek) 
        
        # Connect standard QSlider signals directly from the widget
        slider = self.controls_overlay.time_slider
        slider.sliderPressed.connect(self._on_slider_pressed)
        slider.sliderMoved.connect(self._on_slider_moved)
        slider.sliderReleased.connect(self._on_slider_released)
        
        # Connect seek queue busy state to slider
        if hasattr(self, '_seek_queue'):
            self._seek_queue.seek_started.connect(lambda _: slider.set_busy(True))
            self._seek_queue.seek_completed.connect(lambda: slider.set_busy(False))
            
            # STABILITY: Connect logic handlers
            self._seek_queue.seek_started.connect(self._on_seek_started_logic)
            self._seek_queue.seek_completed.connect(self._on_seek_completed_logic)

        self.controls_overlay.file_opened.connect(self.open_file_dialog)
        self.controls_overlay.disc_opened.connect(self.open_disc_dialog)
        self.controls_overlay.archive_requested.connect(self.open_archive_dialog)
        # EX-4: MV-HEVC export entries of the unified « Sauvegarde / Export » menu.
        self.controls_overlay.export_mvhevc_requested.connect(self.start_mvhevc_export)
        self.controls_overlay.export_menu.aboutToShow.connect(self._update_export_menu_state)
        # SyLC Cast (Task 13): « Diffuser vers Quest » (Wi-Fi/USB-C) toggles a cast session.
        self.controls_overlay.cast_requested.connect(self._on_cast_requested)
        self.controls_overlay.mode_3d_toggled.connect(self.toggle_3d_mode)
        self.controls_overlay.stereo_mode_changed.connect(self.change_stereo_mode)
        self.controls_overlay.audio_track_changed.connect(self.change_audio_track)
        self.controls_overlay.subtitle_track_changed.connect(self.change_subtitle_track)
        # Per-file memory: an explicit eye-order pick is per-title by design
        # (the load path resets it) — remember it for this title's replays.
        try:
            self.controls_overlay.export_eye_order_group.triggered.connect(
                self._on_eye_order_picked)
        except Exception:
            pass
        # 2D->3D (AI) conversion (Task 8): sliders are single-purpose -- the
        # *_preview signals (live, valueChanged) push without persisting;
        # the *_changed signals (sliderReleased) are what persists to disk.
        self.controls_overlay.synth3d_toggled.connect(self.toggle_synth3d)
        self.controls_overlay.synth3d_depth_view_toggled.connect(self.set_synth3d_depth_view)
        self.controls_overlay.synth3d_diagnostics_toggled.connect(
            self.set_synth3d_diagnostics)
        self.controls_overlay.synth3d_preset_selected.connect(
            self.apply_synth3d_preset)
        self.controls_overlay.synth3d_depth_preset_selected.connect(
            self._synth3d_set_depth_preset)
        self.controls_overlay.synth3d_download_models_requested.connect(
            self._open_model_download_dialog)
        self.controls_overlay.synth3d_strength_preview.connect(
            lambda v: self.set_synth3d_strength(v, persist=False))
        self.controls_overlay.synth3d_strength_changed.connect(self.set_synth3d_strength)
        self.controls_overlay.synth3d_convergence_preview.connect(
            lambda v: self.set_synth3d_convergence(v, persist=False))
        self.controls_overlay.synth3d_convergence_changed.connect(self.set_synth3d_convergence)
        self.controls_overlay.synth3d_auto_convergence_toggled.connect(
            self.set_synth3d_auto_convergence)
        self.controls_overlay.synth3d_temporal_fill_toggled.connect(
            self.set_synth3d_temporal_fill)
        self.controls_overlay.synth3d_menu.aboutToShow.connect(self._update_synth3d_menu_state)
        self.info_overlay.file_clicked.connect(self.open_file_dialog)
        self.controls_overlay.installEventFilter(self)

        # V15: Install event filter on combo popup views to detect when they close
        self._setup_combo_popup_tracking()

    def _setup_combo_popup_tracking(self):
        """V15: Track combo popup visibility to restart inactivity timer when they close."""
        combos = [
            self.controls_overlay.audio_track_combo,
            self.controls_overlay.subtitle_track_combo,
            self.controls_overlay.stereo_mode_combo,
        ]
        for combo in combos:
            # Get the popup view (QAbstractItemView)
            view = combo.view()
            if view:
                view.installEventFilter(self)
                # Store reference to identify in eventFilter
                view.setProperty("is_combo_popup", True)

    def _on_combo_popup_closed(self):
        """V15: Called when a combo popup closes - check if we should start hide timer."""
        if not self.is_playing:
            return

        # Short delay to let the mouse position stabilize
        QTimer.singleShot(50, self._check_mouse_after_popup_close)

    def _check_mouse_after_popup_close(self):
        """V15: Check if mouse is still over controls after popup closed."""
        if not self.is_playing:
            return

        # Closing a popup is itself user activity. The global nav poll is the
        # sole hide authority; refresh its deadline instead of arming a second
        # timer which can later fire while another popup/drag is active.
        if not self._mouse_outside_window:
            self._mark_activity()






















    def toggle_3d_mode(self, enabled):
        """Enables or disables 3D mode."""
        if enabled and not self._effective_3d():
            # 2D content: refuse 3D — toggling it mis-drives the MVC pipeline on a
            # plain 2D stream (runaway speed + audio desync). Keep the button off.
            try:
                btn = self.controls_overlay.mode_3d_button
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            except Exception:
                pass
            self.show_3d_notification("2D video — 3D mode unavailable", success=False)
            return
        self.is_3d_enabled = enabled
        # Per-file memory: 3D auto-enables for 3D content, so the meaningful
        # preference to carry across replays is the viewer's explicit choice.
        # WHICH flag carries that choice depends on what put the 3D output on
        # screen. On a 2D title this button is only ever clickable BECAUSE the
        # AI is synthesizing (_update_3d_button_state gates it on
        # _effective_3d), so its OFF means "stop converting this title" — the
        # synthesis IS the 3D here. Recording three_d_enabled=False instead
        # built a self-contradicting pair (synth3d_enabled=True +
        # three_d_enabled=False) that the deferred restore always resolved to
        # 2D: AI back on, output off 13 ms later, replay after replay (user
        # report 2026-08-04, log 17:27:22).
        _rem = getattr(self, '_remember_for_file', None)
        if _rem is not None and self.has_media:
            if (not enabled and getattr(self, '_synth3d_active', False)
                    and not self._content_is_3d()):
                _rem(synth3d_enabled=False)
            else:
                _rem(three_d_enabled=bool(enabled))
        if self.has_media:
            self.configure_3d_output(enabled, self.current_stereo_mode)
            if self.video_3d_info and self.video_3d_info['is_3d']:
                mode_names = {'mvc': 'MultiView', 'sbs': 'Side-by-Side', 'tab': 'Top-Bottom'}
                stereo_mode = self.video_3d_info['stereo_mode']
                if enabled:
                    self.show_3d_notification(
                        f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())} - 3D Playback Active",
                        success=True, permanent=True
                    )
                else:
                    self.show_3d_notification(
                        f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())} - Downgraded to 2D",
                        success=False, permanent=True
                    )
            else:
                if enabled:
                    self.show_3d_notification("2D File - 3D Mode Enabled", success=True, permanent=True)
                else:
                    self.show_3d_notification("2D File", success=True, permanent=True)

    def _display_widgets(self):
        """Every live display widget: the embedded preview, the framepack
        window's, and — in Dual Projector mode — each eye window's.
        Several call sites used to hardcode the first two as a pair, which
        silently excluded the eye windows from subtitles and from every
        per-source render parameter.

        Pushing through here is necessary but NOT sufficient: an eye window
        built mid-playback never saw any of those pushes, so every per-source
        attribute must ALSO be listed in `_EYE_INHERITED_RENDER_PARAMS` (see
        `_seed_eye_window_params`). Adding one here and forgetting it there is
        the same trap wearing a second face."""
        seen = set()
        windows = (getattr(self, 'framepacking_window', None),
                   *(getattr(self, 'eye_windows', None) or ()))
        for widget in (getattr(self, 'mvc_embedded_widget', None),
                       *(getattr(w, 'display_widget', None) for w in windows)):
            if widget is None or id(widget) in seen:
                continue
            seen.add(id(widget))
            yield widget

    def _hevc_source_aspect_baseline(self):
        """The ordinary (non-Glasses) `source_aspect` for the current source:
        the same value `_try_start_hevc` seeds every display widget with and
        `change_stereo_mode`'s half-source recompute restores -- `mi.width /
        mi.height` for a half-packed source (the packed frame's own coded size
        IS the original, un-squeezed per-eye aspect), 0.0 (renderer derives
        from the decoded planes) otherwise. Also the correct answer for
        non-HEVC MVC/BD3D content, which never sets `hevc_media_info`/
        `_hevc_half` at all -- those getattr defaults land here at 0.0, i.e.
        "derive", which is right: MVC views are already full per-eye
        resolution, nothing to un-squeeze."""
        mi = getattr(self, 'hevc_media_info', None)
        if mi is not None and getattr(self, '_hevc_half', False):
            try:
                return float(mi.width) / float(mi.height)
            except (TypeError, ZeroDivisionError):
                pass
        return 0.0

    def _glasses_eye_aspect(self):
        """The per-eye display aspect Glasses doubles for its source_aspect
        override.

        Half-packed sources keep `_hevc_source_aspect_baseline` (mi.width /
        mi.height): the packed frame's own coded size IS the un-squeezed
        per-eye aspect, and the decoded eye PLANE for a half source is
        genuinely squeezed (e.g. 960x1080), so deriving from the plane would
        be wrong here -- same reason the existing half-source override reads
        MediaInfo instead of the planes at `:4924`/`:9546`.

        Everything else -- full-packed, MV-HEVC, native MVC/BD3D, synthesized
        -- derives from the LAST DECODED eye plane's own dimensions
        (`_glasses_eye_plane_dims`, kept current by
        `_on_mvc_frame_yuv_timed_ready`): for these classes the eye plane IS
        the display frame, so its own w/h ARE the eye's display aspect. Round
        1's 16:9 assumption was wrong for exactly this reason -- a scope
        master delivering e.g. 1920x800 per eye gives 2.4 (4.8 doubled), and
        assuming 16:9 (3.556 doubled) would have letterboxed it incorrectly.
        16:9 remains only the LAST-RESORT interim value before any frame of
        the current source has arrived to seed the cache; the first frame
        corrects it."""
        if getattr(self, '_hevc_half', False):
            return self._hevc_source_aspect_baseline()
        dims = getattr(self, '_glasses_eye_plane_dims', None)
        if dims:
            w, h = dims
            if h:
                return float(w) / float(h)
        return 16.0 / 9.0

    def _glasses_target_aspect(self, fp):
        """The `source_aspect` target for Glasses: `2 * eye_aspect`, clamped
        down to the surface's own aspect when the surface is too NARROW (in
        absolute pixels) to show the pair at full per-eye width.

        Fix round 3. The renderer pillarboxes/letterboxes to whichever side
        of `source_aspect` vs. the actual output surface is "wrong"
        (`native_renderer.cpp:688-704`), so the plain doubled value from
        round 2 is only correct when the surface is wide enough for it. On a
        surface narrower than the pair needs -- e.g. windowed, or fullscreen
        on a 1920x1080 display -- a 3.556 target against a 1.778 surface
        takes the LETTERBOX branch: each half becomes 960x540 with black
        bands above and below, throwing away half the rows for nothing.
        Filling is the right answer there instead: the glasses split
        whatever they receive at ITS OWN midpoint and stretch each half back
        out on their own panel, so on a surface too narrow for full
        per-eye width, the picture reaches the eye the same shape either way
        -- degraded resolution, but geometrically correct, exactly like
        plain SBS/framepack already is on an undersized surface.

        THE COMPARISON IS WIDTH, NOT THE FULL DOUBLED ASPECT RATIO -- a
        deliberate departure from the conceptual `min(2*eye_aspect,
        surface_aspect)`. A pure aspect-ratio clamp conflates two different
        situations that produce the SAME inequality direction: a surface
        genuinely too narrow (not enough pixels -- filling is right, per
        above), and a surface merely TALLER than 2*eye_aspect needs AT THE
        CORRECT WIDTH. The second one is not an edge case here: a scope eye
        (2.4, doubled 4.8) on the STANDARD 3840x1080 Glasses window has
        surface aspect 3.556 -- narrower than 4.8 -- so the literal
        aspect-ratio clamp would fire on the window's own INTENDED,
        correctly-sized target, filling and vertically stretching the eye's
        true 1920x800 picture up to 1920x1080. The glasses' horizontal
        split-and-restretch does nothing to correct a VERTICAL distortion,
        so that would silently undo round 2's scope fix on the ordinary
        window, not just a genuinely undersized one -- regressing a case
        this user's community actually watches, not a hypothetical.
        Comparing WIDTH against `Framepacking3DWindow.FSBS_WIDTH` (3840, the
        pair's own fixed full-resolution requirement, independent of
        content) tells the two situations apart: narrower than 3840 clamps
        (fill, regardless of aspect); at or above 3840 never clamps, so
        scope still letterboxes correctly at the window's own intended
        size, and filling only ever kicks in when the surface itself cannot
        deliver full per-eye width no matter what the content is.

        The surface size is read from `fp` itself (the live
        Framepacking3DWindow), not cached -- see `_on_framepacking_geometry_
        changed` for why a value captured once would go stale the instant
        the user toggles fullscreen."""
        from sylc.framepacking_window_d3d11 import Framepacking3DWindow
        doubled = 2.0 * self._glasses_eye_aspect()
        try:
            w, h = fp.width(), fp.height()
        except Exception:
            w = h = 0
        if w and h and w < Framepacking3DWindow.FSBS_WIDTH:
            return float(w) / float(h)
        return doubled

    def _apply_framepack_source_aspect(self, fp, stereo_mode):
        """Set the framepack widget's `source_aspect` for the presentation
        about to show in it.

        CRITICAL (fix round 1): the renderer's viewport targets `source_aspect`
        for stereo_mode 2 (SBS/Glasses) same as any other layout --
        `native_renderer.cpp:677-682` -- and that target is a SINGLE eye's
        aspect (~1.778 for 16:9), not the pair's. On the Glasses window's
        3840x1080 surface that pillarboxes: a 3.556 surface against a 1.778
        target crops to a centred 1920x1080 viewport, so EACH eye lands in a
        960-wide slice with black either side. Glasses needs the COMBINED
        (both-eyes) aspect instead, i.e. double the per-eye value (see
        `_glasses_eye_aspect`, fix round 2, for how that value is derived) --
        but CLAMPED to the actual surface aspect (see `_glasses_target_aspect`,
        fix round 3) so a surface narrower than the pair fills instead of
        letterboxing. 3.556 for 16:9 on a 3840-wide surface fills exactly;
        4.8 for a 2.39:1-ish scope source letterboxes -- bars top/bottom
        INSIDE each half, which is correct F-SBS behaviour, not a defect --
        but only when the surface is wide enough to earn that target; a
        1920x1080 surface clamps to 1.778 and fills instead.

        Only `fp.display_widget` is touched here -- never loop this over
        `_display_widgets()`, which also yields the embedded 2D preview; that
        widget reads the same field for its own (single-eye, undoubled)
        display and would be poisoned by the doubled value.

        Every other presentation gets the ordinary baseline -- this runs on
        every 'mvc'/'dual'/'glasses' reconfigure (mirroring
        `apply_output_geometry`/`set_stereo_mode` right beside it), so leaving
        Glasses always overwrites the doubled value rather than requiring a
        separate revert step."""
        widget = getattr(fp, 'display_widget', None)
        if widget is None:
            return
        try:
            if stereo_mode == 'glasses':
                widget.source_aspect = self._glasses_target_aspect(fp)
            else:
                widget.source_aspect = self._hevc_source_aspect_baseline()
        except Exception:
            pass

    def _note_decoded_eye_plane(self, left_planes):
        """Feed one decoded frame's left-eye plane dims into the Glasses
        aspect cache (`_glasses_eye_plane_dims`, read by `_glasses_eye_aspect`).

        Called from `_on_mvc_frame_yuv_timed_ready` on every valid frame. A
        no-op for half-packed HEVC -- that class derives from MediaInfo, never
        the plane, because the plane itself is squeezed (see
        `_glasses_eye_aspect`) -- and whenever the dims haven't changed, so the
        comparatively expensive push into the renderer only runs on an actual
        change, not every frame at 24fps. Torn down (reset to None) at every
        session end in `_stop_hevc_decoder`, so a new file's first frame is
        never compared against a leftover value from the one before it."""
        if getattr(self, '_hevc_half', False):
            return
        try:
            eye_h, eye_w = left_planes[0].shape[:2]
        except Exception:
            return
        if not (eye_w and eye_h):
            return
        dims = (eye_w, eye_h)
        if dims == getattr(self, '_glasses_eye_plane_dims', None):
            return
        self._glasses_eye_plane_dims = dims
        if getattr(self, 'current_stereo_mode', None) == 'glasses':
            fp = getattr(self, 'framepacking_window', None)
            if fp is not None:
                self._apply_framepack_source_aspect(fp, 'glasses')

    def change_stereo_mode(self, mode):
        self.current_stereo_mode = mode
        # Per-file memory: the presentation pick is a per-title preference.
        # (getattr idiom: fake hosts drive this method in tests.)
        _rem = getattr(self, '_remember_for_file', None)
        if (_rem is not None and self.has_media
                and mode in PRESENTATION_KEYS):
            _rem(stereo_mode=mode)
        # Dual Projector owns the detached output while selected; any other
        # presentation gives it back to the framepack/main-window paths.
        self._set_dual_projector_enabled(
            mode == 'dual' and self.has_media and self.is_3d_enabled)
        
        _mvhevc = bool(getattr(getattr(self, 'hevc_media_info', None), 'multiview', False))
        if (getattr(self, '_hevc_mode_active', False)
                and getattr(self, 'hevc_thread', None)
                and mode in ('sbs', 'tab')
                and not _mvhevc
                and not getattr(self, '_synth3d_active', False)):
            try:
                self.hevc_thread.set_mode(mode)
                mi = getattr(self, 'hevc_media_info', None)
                if mi is not None and getattr(self, '_hevc_half', False):
                    aspect = float(mi.width) / float(mi.height)
                    for _w in self._display_widgets():
                        try:
                            _w.source_aspect = aspect
                        except Exception:
                            pass
            except Exception:
                pass
        elif getattr(self, '_synth3d_active', False):
            self._synth3d_restore_mono_source()
            
        if self.has_media and self.is_3d_enabled:
            self.configure_3d_output(True, mode)
            # SUPPRESSION de _force_frame_refresh ici. C'était la cause du crash en pause.

    def _content_is_3d(self):
        """True iff the loaded media is genuinely stereoscopic (MVC / SBS / TAB),
        not a plain 2D file. Drives the 3D button's availability."""
        info = getattr(self, 'video_3d_info', None)
        if not info:
            return False
        return bool(info.get('is_3d')) or info.get('stereo_mode') not in (None, 'none')

    def _effective_3d(self):
        """3D presentation is justified: real 3D content OR AI synthesis active."""
        return self._content_is_3d() or getattr(self, '_synth3d_active', False)














































    def _update_3d_button_state(self):
        """Single authority for the availability of BOTH 3D controls (the 3D toggle button
        AND the stereo-mode dropdown). D1: enable them only for real 3D content with media
        loaded — a genuine MVC/SBS/TAB/MV-HEVC session (video_3d_info marks is_3d / a 3D
        stereo_mode; the HEVC path promotes it, a 2D HEVC avcodec session stays is_3d=False).
        For 2D files both controls are greyed OFF, so 3D can't be toggled (which mis-drives
        the MVC pipeline → runaway speed + audio desync) and the dropdown can no longer change
        picture proportions on a 2D video."""
        try:
            btn = self.controls_overlay.mode_3d_button
            combo = self.controls_overlay.stereo_mode_combo
        except Exception:
            return
        capable = self._effective_3d() and getattr(self, 'has_media', False)
        if capable:
            btn.setEnabled(True)
            btn.setToolTip("Toggle 3D mode")
            # Keep the visual check in step with reality: a synth3d enable sets
            # is_3d_enabled programmatically, which never went through the
            # button's own toggle — sync it here (single authority).
            want_checked = bool(getattr(self, 'is_3d_enabled', False))
            if btn.isChecked() != want_checked:
                btn.blockSignals(True)
                btn.setChecked(want_checked)
                btn.blockSignals(False)
            # setEnabled does not emit currentTextChanged; blockSignals is defensive.
            combo.blockSignals(True)
            combo.setEnabled(True)
            combo.blockSignals(False)
            combo.setToolTip("Stereo presentation mode")
        else:
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            btn.setEnabled(False)
            btn.setToolTip("2D video — 3D unavailable")
            combo.blockSignals(True)
            combo.setEnabled(False)
            combo.blockSignals(False)
            combo.setToolTip("2D video — stereo mode unavailable")
        # AI (2D->3D) button: same three-state cycle as the 3D button —
        # greyed when nothing can be synthesized, white letters when
        # available, blue background while the synthesis is running.
        try:
            ai = getattr(self.controls_overlay, 'synth3d_button', None)
            if ai is not None:
                active = bool(getattr(self, '_synth3d_active', False))
                supported = self._synth3d_supported()
                # The 2D->3D menu hangs off THIS button (setMenu), and it is
                # also where "Depth models..." lives -- the only in-app way to
                # acquire the models that make `supported` true. Greying the
                # button when nothing is installed therefore locks the user out
                # of the one action that would unlock it. Stay enabled while
                # unsupported so the menu opens; the individual entries inside
                # are gated separately by _update_synth3d_menu_state().
                ai.setEnabled(not supported
                              or self._synth3d_eligible() or active)
                if hasattr(ai, 'set_active_look'):
                    ai.set_active_look(active)
                if not supported:
                    # Ordered so the DEFAULT is the common, actionable case:
                    # only the two narrow causes claim their own wording.
                    reason = self._synth3d_unsupported_reason()
                    if reason == 'runtime':
                        ai.setToolTip(
                            "2D->3D AI unavailable - onnxruntime.dll is "
                            "missing from this install")
                    elif reason == 'renderer':
                        ai.setToolTip(
                            "2D->3D AI unavailable - the native renderer is "
                            "not available in this build")
                    else:
                        ai.setToolTip(
                            "2D->3D AI unavailable - open this menu and choose "
                            "“Depth models” at the top to download them")
                elif active:
                    ai.setToolTip("2D->3D AI conversion - active")
                else:
                    ai.setToolTip("2D->3D AI conversion")
        except Exception:
            pass

    def _format_badge_label(self):
        """Adaptive 3D-format label for the controls badge, or None for 2D content.
        Width/height tell full vs half packing (Full-SBS at 3840 vs SBS at 1920)."""
        info = getattr(self, 'video_3d_info', None)
        if not info:
            return None
        sm = info.get('stereo_mode')
        w = info.get('width') or 0
        h = info.get('height') or 0
        if sm == 'mvc' or info.get('has_mvc_track'):
            return "MultiView 3D"
        if sm == 'sbs':
            return "Full-SBS 3D" if w >= 2560 else "SBS 3D"
        if sm == 'tab':
            return "Full-TAB 3D" if h >= 1600 else "TAB 3D"
        return None











    # ========== TEXT SUBTITLE (SRT/ASS) OVERLAY ==========








    # =====================================================




    # ========== STREAMING SUBTITLE HANDLER ==========
    # ================================================



    # ===================== Blu-ray → ISO archiving =====================










    # ===================== MV-HEVC export (EX-4) =====================






    # ===== SyLC Cast (Task 13): « Diffuser vers Quest » ================================














    # ===== EX-4 fix #2: export-vs-ISO-dismount (dismount must never race a
    # running export reading from a mounted ISO) =====



    # ===================== Audio VU meter =====================
    def _ensure_vu_af(self):
        # See VU_ASTATS_FILTER for why the measure_* restrictions are load-bearing.
        """Attach the astats audio filter (label 'vu') so af-metadata/vu exposes
        per-channel RMS/peak for the on_vu_metadata observer. Called ONCE per mpv
        instance from _setup_observers (a controlled init point — mpv idle, no file
        loaded — so the one-off `af` read is safe; NOT the 30 Hz VU hot path). The
        af chain persists across loadfile, so one attach covers every session incl.
        the audio-only mpv used for MVC/HEVC/dual. The read-check keeps it
        idempotent (no duplicate filter) and self-healing on any mpv re-init."""
        try:
            chain = self.player._get_property('af') or []
            if not any(isinstance(f, dict) and f.get('label') == 'vu' for f in chain):
                self.player.command('af', 'add', VU_ASTATS_FILTER)
        except Exception:
            pass

    @staticmethod
    def _db_to_unit(s, floor=-50.0):
        """Map a dBFS reading (e.g. '-21.0', '-inf') to a 0..1 meter level."""
        try:
            db = float(s)
        except (TypeError, ValueError):
            return 0.0
        if db != db or db <= floor:            # NaN / -inf / below floor
            return 0.0
        return max(0.0, min(1.0, (db - floor) / (0.0 - floor)))

    def on_vu_metadata(self, _, md):
        """MPV af-metadata/vu changed — PUSHED on the mpv EVENT thread (exactly like
        on_time_update: a value mpv HANDED us, NOT a blocking read). Parse the astats
        per-channel RMS/peak into _vu_cache; the GUI-thread VU poll consumes only that
        cache. This observer→cache path is what lets the meter live in EVERY session,
        including the HEVC/MVC/dual audio-only mpv, without any synchronous mpv read on
        the GUI thread — the absolute rule from the 0xe24c4a02 crash and the 4K10
        TrueHD-Atmos GUI-hog stutter (both were mpv reads on this 30 Hz path)."""
        try:
            if not md:
                return
            rl = self._db_to_unit(md.get('lavfi.astats.1.RMS_level'))
            rr = self._db_to_unit(md.get('lavfi.astats.2.RMS_level', md.get('lavfi.astats.1.RMS_level')))
            pl = self._db_to_unit(md.get('lavfi.astats.1.Peak_level'))
            pr = self._db_to_unit(md.get('lavfi.astats.2.Peak_level', md.get('lavfi.astats.1.Peak_level')))
            # Atomic single-attribute writes under the GIL (same lock-free pattern as
            # _mpv_time_pos_cache / _mpv_pause_cache): no lock needed.
            self._vu_cache = (max(rl, rr), max(pl, pr))
            self._vu_cache_ts = time.monotonic()
        except Exception:
            pass

    def _poll_audio_levels(self):
        """Paint the VU meter from _vu_cache (~30 Hz). CACHE-ONLY: levels are pushed by
        the af-metadata/vu observer (on_vu_metadata) on mpv's event thread. This tick
        NEVER reads an mpv property — in ANY mode — so it can never block the GUI thread
        (the 0xe24c4a02 crash AND the 4K10 stutter both came from synchronous mpv reads
        here; the old HEVC dark-meter skip is gone — the meter now works there too)."""
        vu = getattr(getattr(self, 'controls_overlay', None), 'vu_meter', None)
        if vu is None:
            return
        if (self.player is None or not getattr(self, 'has_media', False)
                or getattr(self, '_archiving', False)
                or getattr(self, '_mpv_pause_cache', False)):
            vu.set_levels(0.0, 0.0)     # no media / archiving / paused → silence
            return
        lvl, pk = getattr(self, '_vu_cache', (0.0, 0.0))
        # Stale-guard: if the observer has gone quiet (seek gap, load/transition, EOF,
        # or a stream with no audio) decay smoothly to silence instead of freezing on
        # the last level. Live playback pushes ~10 Hz, so this never fires mid-play.
        if (time.monotonic() - getattr(self, '_vu_cache_ts', 0.0)) > 0.4:
            lvl *= 0.7
            pk *= 0.7
            lvl = 0.0 if lvl < 0.01 else lvl
            pk = 0.0 if pk < 0.01 else pk
            self._vu_cache = (lvl, pk)
        vu.set_levels(lvl, pk)










    # --- Per-FILE playback memory (2026-08-03, user request) ------------
    # Replaying a title restores the viewer's own tuning for it: stereo
    # presentation, 3D on/off, eye order, track picks, resume position,
    # per-title synth3d tuning. Distinct from _app_settings (global knobs):
    # these fields are exactly the ones the load path resets per file.










    # --- V60: tiny per-install settings store (JSON in the user profile) ---





    def _quit_application(self):
        """Closing the main window IS quitting this app. Qt's
        quitOnLastWindowClosed cannot be relied on here: the player keeps
        auxiliary top-level windows around (the nav bar and both overlays are
        Qt.Tool windows, plus the framepacking window and, in Dual Projector,
        two eye windows), and a single one still counted as visible leaves the
        event loop running forever — timers ticking, process alive, nothing on
        screen (user report 2026-08-04, only Ctrl-C got out)."""
        try:
            app = QApplication.instance()
            if app is not None:
                app.quit()
        except Exception:
            logger.exception("[EXIT] quit() failed")

    def _arm_exit_watchdog(self, seconds=_EXIT_WATCHDOG_S):
        """Guarantee the process leaves once the viewer has closed the window.

        The graceful path below is the normal one; this is what covers the
        case where it cannot complete — a native call that never returns
        (measured: the GUI thread wedged inside set_frame_yuv_views with the
        2D->3D AI running), or a non-daemon worker thread blocking interpreter
        shutdown after the event loop has quit. Everything worth persisting
        (the per-file resume position) is written at the top of closeEvent,
        before any of that can hang.

        Its own thread, and a daemon one: a wedged GUI thread cannot fire a
        QTimer, and a non-daemon thread would block the very exit it guards."""
        if getattr(self, '_exit_watchdog', None) is not None:
            return
        delay = float(seconds)

        def _bail():
            time.sleep(delay)
            logger.critical(
                "[EXIT] teardown did not finish %.0fs after close — forcing "
                "process exit", delay)
            _hard_exit(1)

        thread = threading.Thread(target=_bail, name='SyLC-exit-watchdog',
                                  daemon=True)
        self._exit_watchdog = thread
        thread.start()

    def closeEvent(self, event):
        # From here on, every state change is teardown, not user intent —
        # toggle_synth3d consults this flag so the framepacking-hide echo
        # can no longer overwrite the remembered per-title 2D->3D choice.
        self._app_closing = True
        # Armed before the first blocking call, so it can still fire if one of
        # them never returns.
        self._arm_exit_watchdog()
        # Vanish FIRST, tear down after: the synchronous teardown below can
        # block this thread for seconds (decoder joins up to 5 s, hot mpv
        # terminate ~0.7-2.7 s measured). With the windows still visible that
        # freeze reads as "(ne répond pas)" ghosting and files a WER
        # AppHangTransient; hidden, the same teardown is imperceptible.
        self.hide()
        _fp = getattr(self, 'framepacking_window', None)
        if _fp is not None:
            try:
                _fp.hide()
            except Exception:
                pass
        # Per-file memory: closing the app mid-film must still remember where
        # the viewer was (same contract as an explicit Stop).
        _rp = getattr(self, '_remember_position', None)
        if _rp is not None:
            _rp(final=True)
        self._invalidate_media_session('application close')
        self._is_loading_file = False
        self._cancel_media_workers(wait_ms=1000)
        # Export owns its own QThread, native decoders and possibly the mounted
        # ISO.  It must be cancelled and joined before any of those are torn down.
        if not self._stop_export_job(timeout_ms=7000):
            logger.critical(
                "[EXIT] Export remained blocked; exiting before dismounting its "
                "source or destroying its live QThread")
            _hard_exit(1)
        stop_matting = getattr(self, '_synth3d_stop_human_matting', None)
        if stop_matting is not None:
            stop_matting(timeout=1.0)
        if getattr(self, '_thumb_service', None):
            try:
                self._thumb_service.shutdown()
            except Exception:
                pass
        self._stop_mvc_decoder()
        # A Stop shortly before this close may have left a detached core
        # cooling toward its deferred terminate — its timer will never fire
        # once the app exits, so finish that release here, synchronously
        # (invisible: the windows are already hidden).
        self._drain_dying_core()
        # Release libmpv before dismounting an ISO.
        _closing_player = getattr(self, 'player', None)
        self.player = None
        if _closing_player is not None:
            try:
                try:
                    _closing_player.command('stop')
                    time.sleep(0.150)
                except Exception:
                    pass
                _closing_player.terminate()
            except Exception:
                logger.exception("[MPV] Core termination failed during window close")
        # Release any Blu-ray ISO we mounted, so no phantom drive is left behind.
        try:
            from sylc import bluray_disc
            for m in (getattr(self, '_active_iso_mount', None),
                      getattr(self, '_pending_iso_mount', None)):
                if m:
                    bluray_disc.dismount_iso(m[0])
            self._active_iso_mount = None
            self._pending_iso_mount = None
        except Exception:
            pass
        hud = getattr(self, 'stereo_hud', None)
        if hud is not None:
            hud.shutdown()
        self.controls_overlay.close()
        self.info_overlay.close()
        self.loading_overlay.close()
        self.monitoring_overlay.close()
        self.metrics_overlay.close()
        if self.framepacking_window:
            self.framepacking_window.close()
        # Dual Projector's two eye windows are parentless top-level windows too --
        # without this, closing the main window leaves them alive and able to
        # keep the process running after the user thinks they've quit.
        self._set_dual_projector_enabled(False)

        # A third-party decoder that is still executing native code must never
        # reach QObject destruction: Qt deliberately aborts the process when a
        # live QThread wrapper is deleted. Normal shutdown is graceful; this
        # last-resort branch is used only after the cooperative stop and five
        # second join above have both failed.
        self._reap_hevc_leaked()
        live_native_threads = []
        for thread in getattr(self, '_mvc_leaked', []):
            try:
                if thread.isRunning():
                    live_native_threads.append(thread)
            except RuntimeError:
                pass
        for thread, _source in getattr(self, '_hevc_leaked', []):
            try:
                if thread.isRunning():
                    live_native_threads.append(thread)
            except RuntimeError:
                pass
        if live_native_threads:
            logger.critical(
                "[NATIVE CLEANUP] Native decoder remained blocked during application "
                "shutdown; using clean process exit before Qt can destroy a live "
                "QThread")
            _hard_exit(1)
        super().closeEvent(event)
        # Teardown is done and the windows are gone: end the event loop
        # explicitly rather than hoping Qt's last-window bookkeeping agrees.
        logger.info("[EXIT] teardown complete — quitting")
        self._quit_application()









    # ==================================================================================
    # HEVC PATH (spec 2026-07-21) — avformat demux + avcodec decode (ctypes), frames
    # split/paced by HevcDecodeThread, rendered by the SAME native D3D11 widgets +
    # framepack window as MVC (frameYUVReady → _on_mvc_frame_yuv_ready), mpv audio-only
    # (vid=no/vo=null) exactly like MVC. Probed AFTER every H.264/edge264 path and BEFORE
    # the mpv 2D fallback. LavfHevcSource.open() refuses anything that is not 4:2:0
    # 8/10-bit HEVC (→ None), so H.264 never reaches this path.
    # ==================================================================================





if __name__ == "__main__":
    # Support for PyInstaller on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    # --- Build-verification hook (release smoke only) ---------------------------
    # When SYLC_EXPORT_SELFTEST names an output file, resolve the MV-HEVC export
    # tool paths in the CURRENT deployment (standalone dist / onefile launcher dir)
    # and dump the verdict as JSON, then exit before any GUI/Qt init. Lets the
    # release build prove tools\x265 + tools\gpac discovery in both layouts with
    # no window and no full export. No-op for normal launches.
    _selftest_out = os.environ.get('SYLC_EXPORT_SELFTEST')
    if _selftest_out:
        _st = {}
        try:
            from sylc import mvhevc_exporter as _me
            _st['x265'] = _me.X265
            _st['mp4box'] = _me.MP4BOX
            _st['x265_isfile'] = os.path.isfile(_me.X265)
            _st['mp4box_isfile'] = os.path.isfile(_me.MP4BOX)
            _st['tools_available'] = bool(_me.tools_available())
        except Exception as _e:
            _st['error'] = repr(_e)
        _st['argv0'] = sys.argv[0] if sys.argv else None
        _st['executable'] = sys.executable
        _st['onefile_parent'] = os.environ.get('NUITKA_ONEFILE_PARENT')
        try:
            import json as _json
            with open(_selftest_out, 'w', encoding='utf-8') as _f:
                _json.dump(_st, _f, indent=2)
        except Exception:
            pass
        raise SystemExit(0)

    # --- TensorRT engine-probe child (in-app acquisition, stage 3) ------------
    # Every TensorRT engine build runs in its OWN process, because an incomplete
    # or incompatible TensorRT assembly does not raise -- it takes the process
    # down with a hard native abort (see trt_engines.py's header and
    # tools_dev/setup_tensorrt.py's). In a frozen build there is no Python
    # interpreter to spawn: sys.executable IS this exe, so trt_engines
    # re-invokes it with this hidden flag, exactly as SYLC_EXPORT_SELFTEST above
    # re-invokes it for the release build's export smoke test.
    #
    # Handled HERE, before faulthandler and before any Qt/GUI init, so the child
    # never builds a window, never touches media and exits the moment it has a
    # verdict. The literal is duplicated from trt_engines.PROBE_ARGV_FLAG rather
    # than imported so a normal launch does not pay for the import;
    # tests/models/test_trt_engines.py pins the two equal.
    if '--sylc-trt-engine-probe' in sys.argv:
        from sylc import trt_engines
        raise SystemExit(trt_engines.probe_main(sys.argv[1:]))

    # Enable faulthandler to a file (never stderr) to capture real crashes.
    #
    # IMPORTANT — crash_log.txt is NOT, by itself, proof of a crash. faulthandler
    # installs a Windows *vectored* exception handler, which fires for EVERY SEH
    # exception in the process, INCLUDING first-chance ones that native code catches
    # and handles. mpv raises a benign, internally-handled 0xe24c4a02 during its
    # init / event-thread startup (see _setup_mpv_player ~line 2467); it carries the
    # error-severity bit, so faulthandler logs it here and then lets mpv's own
    # __try/__except swallow it and the app keeps running. It appears INTERMITTENTLY
    # (a race — hence the "delay property observers" workaround) and is unrelated to
    # the file being played or to how the process is later closed or force-killed.
    #
    # The dump is an ALL-THREADS snapshot, so it also prints whatever every OTHER
    # thread is doing at that instant — e.g. a perfectly healthy HevcDecodeThread
    # parked in its pacing time.sleep (hevc_decode_thread.py:127). Those frames are
    # bystanders, NOT the culprit. A genuine crash is one where the process actually
    # dies; a leftover 0xe24c4a02 after a run that exited/was killed cleanly is noise.
    import faulthandler
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        # Append instead of overwriting the only crash evidence on every launch,
        # and use the per-user writable log directory for installed/onefile builds.
        os.makedirs(_SYLC_LOG_DIR, exist_ok=True)
        _crash_log = open(
            os.path.join(_SYLC_LOG_DIR, "crash_log.txt"),
            "a", encoding="utf-8")
        _crash_log.write(
            f"\n=== SyLC launch {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"pid={os.getpid()} ===\n")
        _crash_log.flush()
        # all_threads=False — deliberately. faulthandler's vectored handler
        # fires on every error-severity FIRST-CHANCE SEH exception (mpv raises
        # a benign 0xe24c4a02 during command('stop') / init even with Lua
        # scripts disabled). With all_threads=True each such event walked the
        # Python frames of EVERY live thread without holding the GIL, racing
        # the running interpreter; one of those walks eventually dereferenced
        # a dead frame and turned a harmless first-chance exception into a
        # real access violation that killed the process (WER dump 04/08/2026
        # 02:21, AV in _PyObject_IsFreed inside the dump loop, 'access
        # violation' interleaved mid-dump in this very crash_log). Dumping
        # only the faulting thread keeps the useful signal — which thread
        # died where — without the lethal cross-thread traversal.
        faulthandler.enable(file=_crash_log, all_threads=False)
        print(f"[FAULTHANDLER] Enabled -> {_crash_log.name}")
    except Exception as e:
        print(f"[FAULTHANDLER] Could not enable: {e}")

    _install_warning_filters()

    print("[MAIN] Creating QApplication...")
    app = QApplication(sys.argv)
    # App / taskbar icon. On Windows, set an explicit AppUserModelID so the taskbar uses
    # our window icon (and groups correctly) even when run from source; the built .exe also
    # carries the icon via Nuitka --windows-icon-from-ico.
    try:
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('SyLC.3DPlayer.1')
        _app_icon = _find_asset('icon.png')
        if _app_icon:
            app.setWindowIcon(QIcon(_app_icon))
    except Exception as _e:
        print(f"[MAIN] icon setup skipped: {_e}")

    print("[MAIN] Creating PlayerWindow...")
    window = PlayerWindow()

    print("[MAIN] Showing window...")
    window.show()

    # V33k: Handle command-line file argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            print(f"[MAIN] V33k: Auto-loading command-line file: {file_path}")
            QTimer.singleShot(500, lambda: window.play_file(file_path))
        else:
            print(f"[MAIN] Warning: File not found: {file_path}")

    print("[MAIN] Entering event loop...")
    sys.exit(app.exec())

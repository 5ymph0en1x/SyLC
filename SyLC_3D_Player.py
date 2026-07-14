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
import io

# Fix encoding for Unicode characters
if sys.stdout:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass # sys.stdout might be None or a custom object in GUI mode

if sys.stderr:
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass # sys.stderr might be None or a custom object in GUI mode

import os
import re
import struct

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

    dirs_to_add = []

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
    _is_nuitka = dirs_to_add  # Non-empty means Priority 1 or 2 found Nuitka markers
    if _is_nuitka:
        try:
            temp_base = tempfile.gettempdir()
            pyd_name = 'mvc_demuxer_cpp.cp312-win_amd64.pyd'
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
    pyd_name = 'mvc_demuxer_cpp.cp312-win_amd64.pyd'
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

import subprocess
import json
import tempfile
import time
import shutil
import glob
import ctypes
import multiprocessing
import logging
import traceback
import threading
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider,
    QPushButton, QHBoxLayout, QLabel,
    QFileDialog, QComboBox, QMessageBox, QGraphicsOpacityEffect,
    QSizePolicy, QSplashScreen, QStackedLayout, QGraphicsDropShadowEffect
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- FLUIDITY FIX: Force Windows High Resolution Timer (1ms) ---
if sys.platform == 'win32':
    try:
        from ctypes import windll
        timeBeginPeriod = windll.winmm.timeBeginPeriod
        timeBeginPeriod(1)
        logger.info("[FLUIDITY] Windows High Resolution Timer enabled (1ms).")
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

os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"]
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QRectF, QPointF, Slot, QEvent, QObject, QThread
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPen, QBrush, QPainterPath, QBitmap, QImage, QPixmap, QIcon, QCursor


# --- Human-readable track labels (audio / subtitle) -------------------------------
# mpv/MakeMKV tracks often carry placeholder titles like "TRACK_1" that mean nothing.
# Build a meaningful label from language + codec (+ channel layout for audio) instead.
_TRACK_LANG_NAMES = {
    'eng': 'English', 'en': 'English', 'fre': 'French', 'fra': 'French', 'fr': 'French',
    'spa': 'Spanish', 'es': 'Spanish', 'ger': 'German', 'deu': 'German', 'de': 'German',
    'ita': 'Italian', 'it': 'Italian', 'jpn': 'Japanese', 'ja': 'Japanese',
    'chi': 'Chinese', 'zho': 'Chinese', 'zh': 'Chinese', 'rus': 'Russian', 'ru': 'Russian',
    'por': 'Portuguese', 'pt': 'Portuguese', 'dut': 'Dutch', 'nld': 'Dutch', 'nl': 'Dutch',
    'kor': 'Korean', 'ko': 'Korean', 'ara': 'Arabic', 'ar': 'Arabic', 'pol': 'Polish', 'pl': 'Polish',
    'swe': 'Swedish', 'dan': 'Danish', 'nor': 'Norwegian', 'fin': 'Finnish', 'cze': 'Czech', 'ces': 'Czech',
    'hun': 'Hungarian', 'tur': 'Turkish', 'tha': 'Thai', 'hin': 'Hindi', 'heb': 'Hebrew',
    'ell': 'Greek', 'gre': 'Greek', 'ukr': 'Ukrainian', 'vie': 'Vietnamese', 'ind': 'Indonesian',
}
_TRACK_CODEC_NAMES = {
    'eac3': 'Dolby Digital+', 'ac3': 'Dolby Digital', 'ac-3': 'Dolby Digital',
    'truehd': 'Dolby TrueHD', 'mlp': 'Dolby TrueHD',
    'dts': 'DTS', 'dca': 'DTS', 'dts-hd': 'DTS-HD', 'aac': 'AAC', 'flac': 'FLAC',
    'mp3': 'MP3', 'mp2': 'MP2', 'opus': 'Opus', 'vorbis': 'Vorbis',
    'pcm_bluray': 'LPCM', 'pcm_dvd': 'LPCM', 'pcm_s16le': 'PCM', 'pcm_s24le': 'PCM', 'pcm': 'PCM',
    # subtitles
    'hdmv_pgs_subtitle': 'PGS', 'pgssub': 'PGS', 'pgs': 'PGS',
    'subrip': 'SRT', 'srt': 'SRT', 'ass': 'ASS', 'ssa': 'SSA',
    'dvd_subtitle': 'VobSub', 'dvdsub': 'VobSub', 'mov_text': 'TX3G',
}
_TRACK_CHANNELS = {1: 'Mono', 2: 'Stereo', 3: '2.1', 6: '5.1', 7: '6.1', 8: '7.1'}


def _humanize_lang(code):
    if not code:
        return ''
    return _TRACK_LANG_NAMES.get(str(code).strip().lower(), str(code).upper())


def _humanize_codec(codec, profile=''):
    base = _TRACK_CODEC_NAMES.get(str(codec).strip().lower(), str(codec).strip().upper()) if codec else ''
    p = str(profile or '').strip()
    if not p or p.lower() in ('unknown', 'none'):
        return base
    # DTS family: ffmpeg/mpv profile strings are already canonical ("DTS-HD MA", "DTS-HD HRA", "DTS-ES")
    label = p if p.lower().startswith('dts') else base
    if 'atmos' in p.lower() and 'atmos' not in label.lower():
        label += ' Atmos'
    return label


def _track_int(track, *keys):
    for k in keys:
        v = track.get(k)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _friendly_track_label(track, kind='audio', lang_map=None):
    """Build a human-readable label for an mpv track-list entry (audio/sub).

    lang_map: optional {PID: 'iso639'} from a Blu-ray .clpi, used when the
    container itself carries no language tag (raw M2TS/SSIF case).
    """
    tid = track.get('id', '?')
    parts = []
    # language: prefer the container tag; fall back to the Blu-ray .clpi by PID (src-id)
    lang_code = (track.get('lang') or '').strip()
    if not lang_code and lang_map:
        lang_code = lang_map.get(track.get('src-id'), '') or ''
    lang = _humanize_lang(lang_code)
    if lang:
        parts.append(lang)
    codec = _humanize_codec(track.get('codec', ''), track.get('codec-profile', ''))
    if codec:
        parts.append(codec)
    if kind == 'audio':
        ch = _track_int(track, 'demux-channel-count', 'demux_channel_count',
                        'audio-channels', 'audio_channels')
        if ch:
            parts.append(_TRACK_CHANNELS.get(ch, f'{ch}.0'))
    # keep a real, non-placeholder title (skip MakeMKV "TRACK_1" style)
    title = (track.get('title') or '').strip()
    if title and not re.match(r'(?i)^track[\s_]*\d+$', title):
        parts.append(f'“{title}”')
    if not parts:
        parts.append(f'{"Audio" if kind == "audio" else "Subtitle"} {tid}')
    if track.get('forced'):
        parts.append('(forced)')
    elif track.get('default'):
        parts.append('(default)')
    return ' · '.join(parts)


# --- Blu-ray .clpi language map (raw M2TS/SSIF carries no language tag) ------------
# Blu-ray stream_coding_type groups (per the BD spec / libbluray):
_BD_AUDIO_CT = {0x03, 0x04, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0xA1, 0xA2}
_BD_PGIG_CT = {0x90, 0x91}   # Presentation Graphics / Interactive Graphics (PGS)
_BD_TEXT_CT = {0x92}         # Text subtitle


def _parse_clpi_languages(path):
    """Parse a Blu-ray .clpi ProgramInfo and return {PID: 'iso639'} (lowercase)."""
    out = {}
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if data[0:4] != b'HDMV':
            return out
        proginfo_addr = struct.unpack('>I', data[12:16])[0]
        q = proginfo_addr + 4   # skip ProgramInfo length
        q += 1                  # reserved (8 bits)
        num_prog = data[q]; q += 1
        for _ in range(num_prog):
            q += 6              # spn_program_sequence_start(4) + program_map_pid(2)
            num_streams = data[q]; q += 1
            q += 1              # num_groups
            for _ in range(num_streams):
                pid = struct.unpack('>H', data[q:q+2])[0]; q += 2
                sci_len = data[q]; q += 1
                sci = data[q:q+sci_len]; q += sci_len
                if not sci:
                    continue
                ct = sci[0]
                lang = ''
                if ct in _BD_AUDIO_CT and len(sci) >= 5:
                    lang = sci[2:5].decode('ascii', 'replace')   # after coding_type + format byte
                elif ct in _BD_PGIG_CT and len(sci) >= 4:
                    lang = sci[1:4].decode('ascii', 'replace')   # right after coding_type
                elif ct in _BD_TEXT_CT and len(sci) >= 5:
                    lang = sci[2:5].decode('ascii', 'replace')
                lang = lang.strip('\x00').strip()
                if lang:
                    out[pid] = lang.lower()
    except Exception:
        pass
    return out


def _find_clpi_for_media(media_path):
    """Locate the matching CLIPINF/<stem>.clpi for a BDMV STREAM file, or None."""
    try:
        stem = os.path.splitext(os.path.basename(media_path))[0]
        d = os.path.dirname(os.path.abspath(media_path))
        for _ in range(4):   # .../STREAM/SSIF -> STREAM -> BDMV (CLIPINF sits beside STREAM)
            cand = os.path.join(d, 'CLIPINF', stem + '.clpi')
            if os.path.isfile(cand):
                return cand
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    except Exception:
        pass
    return None


def _find_asset(name):
    """Locate a bundled asset (e.g. icon.png) across source / Nuitka --standalone / --onefile."""
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
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None
import mpv
import numpy as np
from premium_controls_overlay import PremiumControlsOverlay as ControlsOverlay

print("[STARTUP] PySide6/mpv/numpy imports succeeded")

# Import MonitoringOverlay (always available, not MVC-dependent)
from monitoring_overlay import MonitoringOverlay
print("[STARTUP] OK MonitoringOverlay importe")

print("[STARTUP] Freeware build (no license system)")

# -----------------------------------------------------------------------------
# Edge264 MVC decoder integration (PRO: no mocks)
# -----------------------------------------------------------------------------
MVC_SUPPORT_AVAILABLE = False
SYNC_TRACER_AVAILABLE = False

try:
    print("[STARTUP] Attempting to import MVC modules...")

    # CRITICAL FIX: Ensure we import from script directory, NOT from other SyLC_* directories
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[STARTUP] Script directory: {_script_dir}")
    print(f"[STARTUP] sys.path[0:5]: {sys.path[:5]}")

    # Force script directory to the very front of sys.path
    if _script_dir in sys.path:
        sys.path.remove(_script_dir)
    sys.path.insert(0, _script_dir)

    # Clear any cached import of mvc_decoder
    if 'mvc_decoder' in sys.modules:
        cached_file = getattr(sys.modules['mvc_decoder'], '__file__', 'unknown')
        if _script_dir not in str(cached_file):
            del sys.modules['mvc_decoder']

    # V42: Make mvc_demuxer_cpp optional - ctypes fallback will be used if unavailable
    try:
        import mvc_demuxer_cpp
        print("[STARTUP] OK mvc_demuxer_cpp imported")
    except ImportError as pyd_err:
        mvc_demuxer_cpp = None
        print(f"[STARTUP] mvc_demuxer_cpp not available ({pyd_err}) - ctypes fallback will be used")

    from mvc_decoder import MVCDecoderThread
    print("[STARTUP] OK MVCDecoderThread imported")

    # D3D11 NATIVE rendering for HDR preservation in fullscreen
    # Directive 2: the detached 3D window hosts the native C++ D3D11 renderer.
    # The Qt RHI widget + the OpenGL/hybrid fallbacks are gone.
    from framepacking_window_d3d11 import Framepacking3DWindow
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
        from subtitle_manager import SubtitleManager
        from subtitle_extractor import SubtitleExtractor, get_pgs_tracks

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
    from text_subtitle_renderer import TextSubtitleRenderer

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
    import lavf_h264_demuxer as _lavf
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


# =============================================================================
# ROBUST SEEK QUEUE - Anti-Saturation System
# =============================================================================

class SeekState(Enum):
    """States of the seek state machine."""
    IDLE = auto()  # Ready to accept a seek
    SEEKING = auto()  # Seek in progress
    COOLDOWN = auto()  # Cooldown period after a seek


@dataclass
class SeekRequest:
    """Represents a seek request."""
    target_time: float
    timestamp: float  # When the request was made
    is_mvc: bool


class RobustSeekQueue(QObject):
    """
    Robust queue for seek operations.

    THREAD-SAFETY: Uses Qt signals to guarantee that all
    MPV operations are executed on the main Qt thread.

    Features:
    - Debounce: Coalesce rapid seeks into a single one
    - Cooldown: Minimum delay between seeks
    - Timeout: Protection against stuck states
    - Thread-safe via Qt signals
    """

    # Signals for thread-safe communication with PlayerWindow
    request_mpv_pause = Signal(bool)  # True = pause, False = unpause
    request_mpv_seek = Signal(float)  # Seek MPV audio to position
    request_decoder_seek = Signal(float)  # Seek video decoder
    seek_started = Signal(float)  # Notify UI seek started
    seek_completed = Signal()  # Notify UI seek completed

    # SEEK-PERF: the decoder side of a seek measures ~25ms (headless bench) —
    # these fixed delays WERE the perceived latency floor. 60ms still coalesces
    # slider-drag events (they arrive every 10-30ms); 100ms cooldown still
    # prevents seek storms while halving chained-seek latency.
    DEBOUNCE_DELAY_MS = 60   # Wait time before executing a seek (was 150)
    COOLDOWN_PERIOD_MS = 100  # Minimum delay between consecutive seeks (was 200)
    SEEK_TIMEOUT_MS = 45000  # Timeout for a stuck seek. Raised to 45s: a COLD optical SSIF
    # seek legitimately takes a few seconds (re-pairing the interleaved base+dependent views);
    # the old 8s fired mid-seek and its forced reset (resume MPV + seek_completed) raced the
    # decode thread → hard crash. 30s only ever fires on a genuine hang, not a slow seek.

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self._parent = parent_window
        self._lock = threading.Lock()
        self._state = SeekState.IDLE
        self._pending_request: Optional[SeekRequest] = None
        self._current_seek_start: float = 0.0
        self._last_seek_completed: float = 0.0

        self._debounce_timer: Optional[QTimer] = None
        self._timeout_timer: Optional[QTimer] = None
        self._cooldown_timer: Optional[QTimer] = None

        self._seeks_requested = 0
        self._seeks_executed = 0
        self._seeks_coalesced = 0
        self._timeouts = 0

        # Connect the signals to the parent
        self.request_mpv_pause.connect(self._parent._on_seek_queue_pause_request)
        self.request_mpv_seek.connect(self._parent._on_seek_queue_mpv_seek)
        self.request_decoder_seek.connect(self._parent._on_seek_queue_decoder_seek)

        logger.info("[SEEK-QUEUE] RobustSeekQueue initialized with Qt signals")

    def _ensure_timers(self):
        if self._debounce_timer is None:
            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounce_expired)

        if self._timeout_timer is None:
            self._timeout_timer = QTimer(self)
            self._timeout_timer.setSingleShot(True)
            self._timeout_timer.timeout.connect(self._on_timeout)

        if self._cooldown_timer is None:
            self._cooldown_timer = QTimer(self)
            self._cooldown_timer.setSingleShot(True)
            self._cooldown_timer.timeout.connect(self._on_cooldown_expired)

    def request_seek(self, target_time: float, is_mvc: bool = True):
        self._seeks_requested += 1
        request = SeekRequest(
            target_time=target_time,
            timestamp=time.monotonic(),
            is_mvc=is_mvc
        )
        QTimer.singleShot(0, lambda: self._handle_request(request))

    def _handle_request(self, request: SeekRequest):
        self._ensure_timers()

        with self._lock:
            logger.info(f"[SEEK-QUEUE] Request to {request.target_time:.2f}s, state={self._state.name}")

            if self._state == SeekState.IDLE:
                self._pending_request = request
                self._state = SeekState.SEEKING
                self._debounce_timer.start(self.DEBOUNCE_DELAY_MS)

            elif self._state == SeekState.SEEKING:
                if self._pending_request:
                    self._seeks_coalesced += 1
                    logger.info(
                        f"[SEEK-QUEUE] Coalesced: {self._pending_request.target_time:.2f}s -> {request.target_time:.2f}s")
                self._pending_request = request
                if self._debounce_timer.isActive():
                    self._debounce_timer.start(self.DEBOUNCE_DELAY_MS)

            elif self._state == SeekState.COOLDOWN:
                self._pending_request = request
                logger.info(f"[SEEK-QUEUE] Queued during cooldown: {request.target_time:.2f}s")

    def _on_debounce_expired(self):
        request = None
        with self._lock:
            if self._pending_request is None:
                self._state = SeekState.IDLE
                return
            request = self._pending_request
            self._pending_request = None
            self._current_seek_start = time.monotonic()

        if request:
            self._execute_seek(request)
            self._timeout_timer.start(self.SEEK_TIMEOUT_MS)

    def _execute_seek(self, request: SeekRequest):
        self._seeks_executed += 1
        target = request.target_time
        logger.info(f"[SEEK-QUEUE] Executing seek to {target:.2f}s (#{self._seeks_executed})")

        try:
            # Notify UI
            self.seek_started.emit(target)

            if request.is_mvc and self._parent.mvc_mode_active:
                # MVC seek: pause audio, seek audio, seek decoder
                self.request_mpv_pause.emit(True)
                # Small delay to let MPV stabilize before the seek
                QTimer.singleShot(50, lambda: self._do_mvc_seek(target))
            else:
                # Simple seek: just seek MPV
                self.request_mpv_seek.emit(target)
                # V7b FIX: Increase delay to 300ms for 2D seek stability
                QTimer.singleShot(300, self.notify_seek_finished)

        except Exception as e:
            logger.error(f"[SEEK-QUEUE] Seek execution failed: {e}")
            self._force_reset_state()

    def _do_mvc_seek(self, target_time: float):
        """Executes the MVC seek after the audio pause.

        SSIF SEEK-FREEZE FIX (MEASURED 2026-06-16): do NOT seek MPV here. On a physical
        Blu-ray, MPV (audio) and the video demuxer both stream the same 45 GB .ssif from the
        single optical head. The decoder alone reads the post-seek IDR in ~0.04 s (measured) —
        the head is fine. The freeze came from making MPV seek the RAW target while the decoder
        scans forward to the actual IDR: the two readers land ~1-2 s of stream APART and the
        head thrashes between them (~0.5 MB/s, 5-15 s). The cure is to keep MPV PAUSED and idle
        at the old position during the (fast, uncontended) scan, then let seekIDRFound seek MPV
        to the EXACT IDR timestamp and resume it — so both readers are CONVERGED at the same
        spot when they finally read together (this is why steady playback never stalls).
        """
        try:
            logger.info(f"[SEEK-QUEUE] MVC seek: decoder scans (MPV stays paused/idle) at {target_time:.3f}s")

            # SSIF ANTI-THRASH (measured 2026-06-17): MPV (audio-only in MVC) and the video
            # demuxer share the single optical head. A concurrent disc reader turns the demuxer's
            # ~1.3s cold post-seek scan into 45-120s (head thrash between two far stream
            # positions — measured directly). MPV is paused here, BUT the global MPV config reads
            # 20s / 2 GB ahead, so even a PAUSED MPV pre-reads ~120 MB at the OLD position while
            # the decoder reads the NEW one → it starves the scan → timeout → _force_reset_state
            # resumes MPV → "audio plays while the image freezes". So force MPV paused and CLAMP
            # its read-ahead to a few MB right before every scan (the init-time shrink doesn't
            # reach this path reliably). seekIDRFound's ATOMIC SYNC re-seeks + resumes MPV after
            # the scan, when both readers are converged on the new region (steady playback shares
            # the OS cache, so it never thrashes).
            try:
                p = getattr(self._parent, 'player', None)
                if p is not None:
                    p.pause = True
                    p['demuxer-readahead-secs'] = 1
                    p['demuxer-max-bytes'] = '8MiB'
                    p['demuxer-max-back-bytes'] = '4MiB'
            except Exception as _e:
                logger.warning(f"[SEEK-QUEUE] MPV read-ahead clamp skipped: {_e}")

            # Seek ONLY the decoder. MPV stays paused + read-ahead-clamped (idle on the disc) so
            # the scan runs uncontended. seekIDRFound re-seeks MPV to the exact IDR timestamp and
            # resumes it (atomic handoff) → both readers converged.
            self.request_decoder_seek.emit(target_time)
        except Exception as e:
            logger.error(f"[SEEK-QUEUE] MVC seek failed: {e}")
            self._force_reset_state()

    def notify_seek_finished(self):
        """Called by the decoder when the seek is finished."""
        QTimer.singleShot(0, self._handle_seek_finished)

    def _handle_seek_finished(self):
        with self._lock:
            logger.info(f"[SEEK-QUEUE] Seek finished, state={self._state.name}")

            if self._timeout_timer and self._timeout_timer.isActive():
                self._timeout_timer.stop()

            self._last_seek_completed = time.monotonic()
            has_pending = self._pending_request is not None

        # V8 PAUSE FIX: Only resume if the decoder was NOT paused before seek
        # Check the decoder's _is_paused state which was set BEFORE seekFinished was emitted
        if self._parent.mvc_mode_active:
            decoder = getattr(self._parent, 'mvc_decoder_thread', None)
            decoder_is_paused = decoder._is_paused if decoder else False

            if decoder_is_paused:
                logger.info("[SEEK-QUEUE] Resuming playback after seek SKIPPED (decoder is paused)")
                # Keep MPV paused and don't update is_playing
                self.request_mpv_pause.emit(True)  # Ensure MPV stays paused
            else:
                logger.info("[SEEK-QUEUE] Resuming playback after seek")
                self.request_mpv_pause.emit(False)
                # Force playback timer to stay active and update is_playing flag
                if hasattr(self._parent, '_playback_timer') and self._parent._playback_timer:
                    self._parent._playback_timer.start()
                # V7b CRITICAL: Ensure is_playing flag is True to allow timeline updates
                self._parent.is_playing = True

        self.seek_completed.emit()

        with self._lock:
            if self._pending_request:
                # Start the cooldown then execute
                self._state = SeekState.COOLDOWN
                self._cooldown_timer.start(self.COOLDOWN_PERIOD_MS)
            else:
                self._state = SeekState.IDLE
                logger.info("[SEEK-QUEUE] Back to IDLE")

    def _on_cooldown_expired(self):
        request = None
        with self._lock:
            logger.info("[SEEK-QUEUE] Cooldown expired")
            if self._pending_request:
                request = self._pending_request
                self._pending_request = None
                self._state = SeekState.SEEKING
                self._current_seek_start = time.monotonic()
            else:
                self._state = SeekState.IDLE

        if request:
            self._execute_seek(request)
            self._timeout_timer.start(self.SEEK_TIMEOUT_MS)

    def _on_timeout(self):
        self._timeouts += 1
        logger.error(f"[SEEK-QUEUE] TIMEOUT! (total: {self._timeouts})")
        self._force_reset_state()

    def _force_reset_state(self):
        pending = None
        with self._lock:
            logger.warning("[SEEK-QUEUE] Forcing state reset to IDLE")

            if self._timeout_timer:
                self._timeout_timer.stop()
            if self._debounce_timer:
                self._debounce_timer.stop()
            if self._cooldown_timer:
                self._cooldown_timer.stop()

            self._state = SeekState.IDLE
            pending = self._pending_request
            self._pending_request = None

        # Resume audio playback (thread-safe via signal)
        self.request_mpv_pause.emit(False)

        # CRITICAL FIX: Unblock UI (slider) and reset seeking flags
        self.seek_completed.emit()

        # Retry the pending seek after a delay
        if pending:
            logger.info(f"[SEEK-QUEUE] Re-requesting pending seek to {pending.target_time:.2f}s")
            QTimer.singleShot(200, lambda: self.request_seek(pending.target_time, pending.is_mvc))

    def is_busy(self) -> bool:
        with self._lock:
            return self._state != SeekState.IDLE


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
        box-shadow: 0 0 5px rgba(0,0,0,0.5);
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


@lru_cache(maxsize=None)
def _resolve_external_tool(executable_name):
    """Return an absolute path to an external tool (ffmpeg/ffprobe) if available."""
    # Use APP_BASE_DIR for Nuitka compatibility
    base_dir = APP_BASE_DIR

    candidates = []
    if sys.platform == 'win32' and not executable_name.lower().endswith('.exe'):
        candidates.append(f"{executable_name}.exe")
    candidates.append(executable_name)

    # PRIORITY 1: Check local directory first (for bundled executables)
    for candidate in candidates:
        local_candidate = os.path.join(base_dir, candidate)
        if os.path.exists(local_candidate):
            return local_candidate

    # PRIORITY 2: Check system PATH
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def _describe_windows_returncode(returncode):
    """Return a human readable explanation for common Windows subprocess errors."""
    if returncode in (3221225781, -1073741515):  # 0xC0000135
        return (
            "Failed to start the executable (code 0xC0000135). "
            "This usually indicates that DLLs for ffmpeg/ffprobe are missing. "
            "Download a static build of ffmpeg from https://www.gyan.dev/ffmpeg/builds/ "
            "and place ffmpeg.exe/ffprobe.exe and their DLLs in the application's folder, "
            "or add the ffmpeg /bin folder to your PATH."
        )
    if returncode in (3221225501, -1073741795):  # 0xC0000025 or similar
        return (
            "The system prevented ffmpeg/ffprobe from running (code 0xC0000025). "
            "Check your antivirus or try running the application with sufficient privileges."
        )
    return None


def _check_ffmpeg_runtime(executable_path):
    """
    Checks if essential DLLs for ffmpeg/ffprobe are present (Windows).

    Returns:
        str | None: error message if a dependency is missing.
    """
    if sys.platform != 'win32' or not executable_path:
        return None

    # Check in multiple locations: ffprobe's folder AND APP_BASE_DIR
    folders_to_check = [os.path.dirname(executable_path)]
    if APP_BASE_DIR and APP_BASE_DIR not in folders_to_check:
        folders_to_check.append(APP_BASE_DIR)

    required_bases = ['avcodec', 'avformat', 'avutil']
    missing = []

    for base in required_bases:
        found = False
        for folder in folders_to_check:
            pattern = os.path.join(folder, f"{base}-*.dll")
            if glob.glob(pattern):
                found = True
                break
        if not found:
            missing.append(base)

    if missing:
        return (
            f"ffmpeg/ffprobe found but the following DLLs are missing in the same folder: "
            f"{', '.join(missing)}. Copy all DLLs provided with ffmpeg (from the /bin directory of the archive) "
            "next to the executables, or install a full static build."
        )

    return None


_STEREO_PRIORITY = {
    'none': 0,
    'tab': 1,
    'sbs': 2,
    'mvc': 3,
    'anaglyph': 1,
}


def _classify_stereo_mode(mode_str):
    """Normalizes a stereo_mode value to sbs/tab/mvc/anaglyph."""
    if not mode_str:
        return None

    mode = mode_str.strip().lower()
    mode = mode.replace('-', '_').replace(' ', '_')

    if mode in ('mono', 'left', 'right', 'both', '2d'):
        return None

    if any(keyword in mode for keyword in ('anaglyph', 'cyan', 'magenta', 'red_cyan', 'cyan_red')):
        return 'anaglyph'

    if any(keyword in mode for keyword in (
            'frame_altern', 'framealternate', 'frame_packing', 'frame_sequential',
            'frame_packed', 'view_packed', 'mvc', 'framepacking', 'frameinterleaved',
            'block_lr', 'block_rl', 'packed'
    )):
        return 'mvc'

    if any(keyword in mode for keyword in (
            'top_bottom', 'bottom_top', 'tab', 'over_under', 'under_over',
            'block_tb', 'block_bt', 'topbottom', 'bt', 'tb'
    )):
        return 'tab'

    if any(keyword in mode for keyword in (
            'side_by_side', 'sbs', 'left_right', 'right_left',
            'row_interleaved', 'column_interleaved'
    )):
        return 'sbs'

    return None


def _promote_stereo_mode(result_dict, mode, mark_mvc=False):
    """Updates the 3D detection result with priority."""
    if not mode:
        return

    priority = _STEREO_PRIORITY.get(mode, 0)
    current_priority = _STEREO_PRIORITY.get(result_dict.get('stereo_mode', 'none'), 0)

    if priority >= current_priority:
        result_dict['stereo_mode'] = mode

    result_dict['is_3d'] = True

    if mark_mvc or mode == 'mvc':
        result_dict['has_mvc_track'] = True


def _parse_ffprobe_fps(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split('/')
    if len(parts) == 2:
        try:
            num = float(parts[0])
            den = float(parts[1])
            if den > 0:
                fps_val = num / den
                return fps_val
        except (ValueError, ZeroDivisionError):
            return None
    else:
        try:
            fps_val = float(value)
            return fps_val
        except ValueError:
            return None
    return None


class Video3DAnalyzer:
    """
    Analyzes video files to detect 3D content.
    Uses ffprobe to extract metadata.
    """

    @staticmethod
    def analyze_file(file_path):
        """
        Analyzes a video file and returns its 3D properties.
        """
        result = {
            'is_3d': False,
            'stereo_mode': 'none',
            'has_mvc_track': False,
            'width': 0,
            'height': 0,
            'analysis_error': None,
            'duration': None,
            'fps': None,
            'codec_name': None,    # H.264 ('h264') eligible for edge264 path
            'container_ext': None, # File extension (.mkv, .mp4, .ssif, ...)
        }
        # Capture extension early — analyzer uses it for codec routing decisions.
        try:
            result['container_ext'] = os.path.splitext(file_path)[1].lower()
        except Exception:
            pass

        # Force MVC for SSIF files (Blu-ray 3D)
        # ffprobe often misidentifies them or hangs on large files
        if file_path.lower().endswith('.ssif'):
            result['is_3d'] = True
            result['stereo_mode'] = 'mvc'
            result['has_mvc_track'] = True
            # Default values, will be refined by decoder/demuxer
            result['width'] = 1920
            result['height'] = 1080
            result['fps'] = 23.976
            return result

        try:
            ffprobe_path = _resolve_external_tool('ffprobe')
            if not ffprobe_path:
                raise FileNotFoundError(
                    "ffprobe not found. Add ffprobe to the PATH or place ffprobe.exe "
                    "in the same folder as SyLC_3D_GUI.py."
                )

            runtime_issue = _check_ffmpeg_runtime(ffprobe_path)
            if runtime_issue:
                print(runtime_issue)
                result['analysis_error'] = runtime_issue
                raise FileNotFoundError(runtime_issue)

            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                file_path
            ]

            creationflags = 0
            if sys.platform == 'win32':
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    creationflags=creationflags
                )
            except PermissionError as e:
                # In sandboxed environments CreatePipe/CreateProcess can be blocked.
                result['analysis_error'] = f"ffprobe permission error: {e}"
                # Fall back to safest defaults for MVC playback.
                result['is_3d'] = True
                result['has_mvc_track'] = True
                result['stereo_mode'] = 'mvc'
                return result
            except Exception as e:
                result['analysis_error'] = f"ffprobe failed: {e}"
                return result

            data = json.loads(completed.stdout or "{}")

            format_info = data.get('format', {})
            duration_str = format_info.get('duration')
            if duration_str:
                try:
                    result['duration'] = float(duration_str)
                except ValueError:
                    pass

            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    result['width'] = stream.get('width', 0)
                    result['height'] = stream.get('height', 0)
                    fps_value = _parse_ffprobe_fps(stream.get('avg_frame_rate')) or \
                                _parse_ffprobe_fps(stream.get('r_frame_rate'))
                    if fps_value:
                        result['fps'] = fps_value
                    width = result['width']
                    height = result['height']
                    is_framepacked = (width == 1920 and height in [2205, 2160]) or (width == 3840 and height == 4320)

                    if is_framepacked:
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'

                    codec_name = (stream.get('codec_name') or '').lower()
                    profile = (stream.get('profile') or '').lower()
                    # Remember the video codec so the player can decide whether
                    # to use the edge264 path (H.264) or fall back to MPV native.
                    if codec_name and not result['codec_name']:
                        result['codec_name'] = codec_name

                    if codec_name in ('mvc', 'h264'):
                        if 'stereo' in profile or 'mvc' in profile:
                            _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    disposition = stream.get('disposition') or {}
                    if isinstance(disposition, dict) and disposition.get('dependent'):
                        _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    if not is_framepacked:
                        for side_data in stream.get('side_data_list', []):
                            side_type = (
                                    side_data.get('type')
                                    or side_data.get('side_data_type')
                                    or ''
                            ).lower()
                            if 'stereo3d' in side_type or 'stereo_3d' in side_type:
                                detected = (
                                        side_data.get('stereo_mode')
                                        or side_data.get('type')
                                        or side_data.get('layout')
                                        or side_data.get('view')
                                        or ''
                                )
                                classified = _classify_stereo_mode(detected)
                                if classified == 'mvc':
                                    _promote_stereo_mode(result, 'mvc', mark_mvc=True)
                                elif classified:
                                    _promote_stereo_mode(result, classified)

                        tags = stream.get('tags') or {}
                        for key, value in tags.items():
                            if key.lower().startswith('stereo'):
                                classified = _classify_stereo_mode(value)
                                if classified:
                                    _promote_stereo_mode(result, classified)

            if not result['has_mvc_track']:
                for stream in data.get('streams', []):
                    if stream.get('codec_name') == 'mvc':
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'
                        break

            if not result['duration']:
                for stream in data.get('streams', []):
                    dur = stream.get('duration')
                    if dur:
                        try:
                            result['duration'] = float(dur)
                            break
                        except ValueError:
                            continue

        except subprocess.CalledProcessError as e:
            error_output = (e.stderr or e.stdout or '').strip()
            message = error_output if error_output else str(e)
            print(f"Error during 3D analysis (ffprobe): {message}")
            hint = _describe_windows_returncode(e.returncode)
            if hint:
                print(hint)
                result['analysis_error'] = hint
            else:
                result['analysis_error'] = message
            filename = os.path.basename(file_path).lower()
            if '3d' in filename or 'sbs' in filename or 'hsbs' in filename:
                result['is_3d'] = True
                result['stereo_mode'] = 'sbs'
            elif '3d' in filename and ('tab' in filename or 'htab' in filename):
                result['is_3d'] = True
                result['stereo_mode'] = 'tab'
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error during 3D analysis: {e}")
            result['analysis_error'] = str(e)
            filename = os.path.basename(file_path).lower()
            if '3d' in filename or 'sbs' in filename or 'hsbs' in filename:
                result['is_3d'] = True
                result['stereo_mode'] = 'sbs'
            elif '3d' in filename and ('tab' in filename or 'htab' in filename):
                result['is_3d'] = True
                result['stereo_mode'] = 'tab'

        return result


# GLOBAL ThreadPool for parallel thumbnail extraction (max 2 workers)
_thumbnail_executor = ThreadPoolExecutor(max_workers=2)


def _extract_thumbnail_ffmpeg(video_file, time_pos):
    """Extract a thumbnail with ffmpeg (worker function for ThreadPoolExecutor)."""
    try:
        ffmpeg_path = _resolve_external_tool('ffmpeg')
        if not ffmpeg_path:
            logger.warning("[PREVIEW] ffmpeg not found. Preview thumbnails disabled.")
            return None

        temp_file = os.path.join(tempfile.gettempdir(), f"preview_{int(time.time() * 1000000)}.jpg")

        cmd = [
            ffmpeg_path,
            '-ss', str(time_pos),
            '-i', video_file,
            '-frames:v', '1',
            '-vf', 'scale=120:-1',
            '-q:v', '8',
            '-y',
            temp_file
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            creationflags=creationflags
        )

        if result.returncode == 0 and os.path.exists(temp_file):
            return temp_file
        return None
    except:
        return None


class PreviewTooltip(QLabel):
    """Widget to display the frame preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(120, 68)  # 16:9 aspect ratio
        self.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #007ACC;
                border-radius: 4px;
            }
        """)
        self.setScaledContents(True)
        self.hide()


def _decide_thumbs_mode(file_path, mounted_iso_letters, optical_letters, codec_name=None):
    """Thumbnail provider decision (spec 2026-07-14). Physical optical → 'off'
    (single head, measured 45-120s thrash with a third reader). Player-mounted
    ISO + H.264 → 'edge264' with optical guardrails. Plain H.264 file →
    'edge264'. Plain non-H.264 → 'ffmpeg'."""
    if not file_path:
        return 'off', False
    EDGE_EXTS = {'.ssif', '.m2ts', '.ts', '.mkv', '.mk3d'}
    ext = os.path.splitext(file_path)[1].lower()
    is_h264 = ext in EDGE_EXTS or (codec_name or '').lower() == 'h264'
    d = os.path.splitdrive(os.path.abspath(file_path))[0]
    letter = d[0].upper() if d else None
    if letter and letter in optical_letters:
        if letter not in mounted_iso_letters:
            return 'off', True                      # physical disc: never
        return ('edge264', True) if is_h264 else ('off', True)
    return ('edge264', False) if is_h264 else ('ffmpeg', False)


class TimeSlider(QSlider):
    """Custom slider with time preview on hover."""

    preview_requested = Signal(float)
    extraction_done = Signal(float, str)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hover_time = 0
        self._is_hovering = False
        self._player = None
        self._preview_widget = PreviewTooltip(self)
        self._last_preview_time = -99
        self._preview_cache = {}  # LRU cache (100 frames)
        self._video_file = None
        self._extraction_timer = None  # Lazy initialization
        self._timer_initialized = False
        self._pending_time = 0
        self._pending_mouse_x = 0
        self.extraction_done.connect(self._on_extraction_done)

    def _ensure_timer_initialized(self):
        """Initialize extraction timer in GUI thread when first needed"""
        if not self._timer_initialized:
            self._extraction_timer = QTimer(self)
            self._extraction_timer.setSingleShot(True)
            self._extraction_timer.timeout.connect(self._do_extraction)
            self._timer_initialized = True

    def enterEvent(self, event):
        super().enterEvent(event)
        self._is_hovering = True
        self.update()

    def set_player(self, player):
        self._player = player

    def mouseMoveEvent(self, event):
        if self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self._hover_time = max(0, min(value, self.maximum()))
            self._is_hovering = True

            s = int(self._hover_time)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            time_str = f"{h:02}:{m:02}:{s:02}"
            self.setToolTip(time_str)

            if self._video_file and abs(self._hover_time - self._last_preview_time) > 0.5:
                self._last_preview_time = self._hover_time
                self._request_on_demand_preview(self._hover_time, pos)

            self.update()
        super().mouseMoveEvent(event)

    def set_video_file(self, file_path: str, duration_seconds: float):
        """V7b+++++ PREVIEW FIX: Restore set_video_file for thumbnail preview.

        This method was mistakenly removed in a previous fix. It's required for
        the preview tooltip to work - without it, _video_file stays None and
        no thumbnails are extracted on hover.

        Args:
            file_path: Path to the video file (MKV, M2TS, etc.)
            duration_seconds: Video duration in seconds
        """
        self._video_file = file_path
        if duration_seconds > 0:
            self.setRange(0, int(duration_seconds))
        # Clear preview cache when video changes
        self._preview_cache.clear()
        self._last_preview_time = -99
        logger.info(f"[PREVIEW] Video file set: {file_path}, duration={duration_seconds:.1f}s")

    def _request_on_demand_preview(self, time_pos, mouse_x):
        cache_key = round(time_pos)
        if cache_key in self._preview_cache:
            pixmap = self._preview_cache[cache_key]
            if not pixmap.isNull():
                self._preview_widget.setPixmap(pixmap)
                self._show_preview_at(mouse_x)
                return

        self._pending_time = time_pos
        self._pending_mouse_x = mouse_x
        self._ensure_timer_initialized()  # Lazy timer creation
        self._extraction_timer.start(100)

    def _do_extraction(self):
        time_pos = self._pending_time
        mouse_x = self._pending_mouse_x
        future = _thumbnail_executor.submit(_extract_thumbnail_ffmpeg, self._video_file, time_pos)
        future.add_done_callback(lambda f: self._handle_extraction_result(f, time_pos, mouse_x))

    def _handle_extraction_result(self, future, time_pos, mouse_x):
        try:
            temp_file = future.result()
            if temp_file:
                self.extraction_done.emit(time_pos, temp_file)
        except:
            pass

    @Slot(float, str)
    def _on_extraction_done(self, time_pos, temp_file):
        try:
            cache_key = round(time_pos)
            pixmap = QPixmap(temp_file)
            if not pixmap.isNull():
                if len(self._preview_cache) > 100:
                    oldest = next(iter(self._preview_cache))
                    del self._preview_cache[oldest]
                self._preview_cache[cache_key] = pixmap
                if self._is_hovering and abs(time_pos - self._hover_time) < 3:
                    self._preview_widget.setPixmap(pixmap)
                    self._show_preview_at(self._pending_mouse_x)
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            print(f"[ERROR] {e}")

    def _show_preview_at(self, mouse_x):
        global_pos = self.mapToGlobal(QPoint(int(mouse_x), 0))
        tooltip_x = global_pos.x() - self._preview_widget.width() // 2
        tooltip_y = global_pos.y() - self._preview_widget.height() - 10

        self._preview_widget.move(tooltip_x, tooltip_y)
        self._preview_widget.show()
        self._preview_widget.raise_()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._is_hovering = False
        self.setToolTip("")
        self._preview_widget.hide()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self.setValue(max(0, min(value, self.maximum())))
            self.sliderMoved.emit(self.value())
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_hovering and self.maximum() > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            preview_x = int((self._hover_time / self.maximum()) * self.width())
            painter.setPen(QPen(QColor(0, 122, 204, 180), 2))
            painter.drawLine(preview_x, 0, preview_x, self.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 122, 204, 220)))
            painter.drawEllipse(QPointF(preview_x, self.height() // 2), 5, 5)


class IconButton(QPushButton):
    """Professional HDR Converter style button - Modern Redesign."""

    def __init__(self, icon_type, is_primary=False, parent=None):
        super().__init__(parent)
        self.icon_type = icon_type
        self.is_primary = is_primary

        if is_primary:
            self.setFixedSize(48, 48)
            self.setStyleSheet("""
                QPushButton {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #007ACC, stop:1 #0063A3);
                    border: 1px solid #0096FF;
                    border-radius: 24px;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0089E5, stop:1 #007ACC);
                    border: 1px solid #33Aaff;
                }
                QPushButton:pressed {
                    background-color: #004578;
                    margin-top: 1px; 
                }
            """)
        else:
            self.setFixedSize(38, 38)
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.15);
                    margin-top: 1px;
                }
                QPushButton:checked {
                    background-color: rgba(0, 122, 204, 0.3);
                    border: 1px solid #007ACC;
                }
            """)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Icon Color
        color = QColor(240, 240, 240)
        if not self.isEnabled():
            color = QColor(255, 255, 255, 80)

        # Thinner, more elegant stroke
        pen = QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        center_x = self.width() / 2
        center_y = self.height() / 2

        if self.icon_type == 'play':
            path = QPainterPath()
            # Refined play triangle
            path.moveTo(center_x - 3, center_y - 6)
            path.lineTo(center_x + 5, center_y)
            path.lineTo(center_x - 3, center_y + 6)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))

        elif self.icon_type == 'pause':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(center_x - 6, center_y - 6, 4, 12), 1, 1)
            painter.drawRoundedRect(QRectF(center_x + 2, center_y - 6, 4, 12), 1, 1)

        elif self.icon_type == 'stop':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(center_x - 5, center_y - 5, 10, 10), 2, 2)

        elif self.icon_type == 'folder':
            path = QPainterPath()
            path.moveTo(center_x - 8, center_y - 5)
            path.lineTo(center_x - 3, center_y - 5)
            path.lineTo(center_x - 1, center_y - 7)
            path.lineTo(center_x + 8, center_y - 7)
            path.lineTo(center_x + 8, center_y + 6)
            path.lineTo(center_x - 8, center_y + 6)
            path.closeSubpath()
            painter.strokePath(path, pen)

        elif self.icon_type == 'fullscreen':
            gap = 6
            len_ = 4
            # TL
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap + len_, center_y - gap))
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap, center_y - gap + len_))
            # TR
            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap - len_, center_y - gap))
            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap, center_y - gap + len_))
            # BL
            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap + len_, center_y + gap))
            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap, center_y + gap - len_))
            # BR
            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap - len_, center_y + gap))
            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap, center_y + gap - len_))

        elif self.icon_type == 'exit_fullscreen':
            gap = 7
            len_ = 4
            # TL (pointing in)
            painter.drawLine(QPointF(center_x - gap + len_, center_y - gap + len_),
                             QPointF(center_x - gap + len_, center_y - gap))
            painter.drawLine(QPointF(center_x - gap + len_, center_y - gap + len_),
                             QPointF(center_x - gap, center_y - gap + len_))
            # BR (pointing in)
            painter.drawLine(QPointF(center_x + gap - len_, center_y + gap - len_),
                             QPointF(center_x + gap - len_, center_y + gap))
            painter.drawLine(QPointF(center_x + gap - len_, center_y + gap - len_),
                             QPointF(center_x + gap, center_y + gap - len_))

        elif self.icon_type == '3d':
            font = QFont('Segoe UI', 9, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, '3D')

        elif self.icon_type == 'volume':
            path = QPainterPath()
            path.moveTo(center_x - 3, center_y - 2)
            path.lineTo(center_x - 1, center_y - 2)
            path.lineTo(center_x + 3, center_y - 5)
            path.lineTo(center_x + 3, center_y + 5)
            path.lineTo(center_x - 1, center_y + 2)
            path.lineTo(center_x - 3, center_y + 2)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))
            # Waves
            painter.setPen(pen)
            painter.drawArc(QRectF(center_x + 1, center_y - 3, 4, 6), -60 * 16, 120 * 16)
            painter.drawArc(QRectF(center_x + 1, center_y - 6, 8, 12), -55 * 16, 110 * 16)


class LoadingOverlay(QWidget):
    """Elegant loading animation overlay shown during file initialization."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Use Tool window to ensure it floats above native MPV window
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._status_text = "Initializing..."
        self._progress_angle = 0
        self._fade_opacity = 0.0
        self._is_showing = False
        self._progress_mode = False  # True = show progress arc, False = spinning
        self._progress_value = 0.0   # 0.0 to 1.0

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._update_animation)

        # Fade animation
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._update_fade)
        self._fade_direction = 1  # 1 = fade in, -1 = fade out

    def show_loading(self, status_text: str = "Initializing...", progress_mode: bool = False):
        """Show loading overlay with fade-in animation."""
        self._status_text = status_text
        self._is_showing = True
        self._fade_direction = 1
        self._progress_mode = progress_mode
        self._progress_value = 0.0
        self._anim_timer.start(16)  # ~60 FPS for smooth animation
        self._fade_timer.start(16)
        self.show()
        self.raise_()

    def hide_loading(self):
        """Hide loading overlay with fade-out animation."""
        self._fade_direction = -1
        self._fade_timer.start(16)
        self._progress_mode = False

    def set_status(self, text: str):
        """Update the status text."""
        self._status_text = text
        self.update()

    def set_progress(self, value: float):
        """Set progress value (0.0 to 1.0) - switches to progress mode."""
        self._progress_mode = True
        self._progress_value = max(0.0, min(1.0, value))
        self.update()

    def _update_animation(self):
        """Update spinner rotation."""
        self._progress_angle = (self._progress_angle + 6) % 360
        self.update()

    def _update_fade(self):
        """Update fade animation."""
        self._fade_opacity += self._fade_direction * 0.08

        if self._fade_opacity >= 1.0:
            self._fade_opacity = 1.0
            self._fade_timer.stop()
        elif self._fade_opacity <= 0.0:
            self._fade_opacity = 0.0
            self._fade_timer.stop()
            self._anim_timer.stop()
            self._is_showing = False
            self.hide()

        self.update()

    def paintEvent(self, event):
        if self._fade_opacity <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Semi-transparent background
        bg_alpha = int(180 * self._fade_opacity)
        painter.fillRect(self.rect(), QColor(18, 18, 18, bg_alpha))

        center_x = self.width() // 2
        center_y = self.height() // 2 - 30

        arc_alpha = int(255 * self._fade_opacity)
        arc_rect = QRectF(center_x - 25, center_y - 25, 50, 50)

        if self._progress_mode:
            # === PROGRESS MODE: Draw filling circle ===
            # Background circle (dark)
            bg_color = QColor(60, 60, 60, int(100 * self._fade_opacity))
            painter.setPen(QPen(bg_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(arc_rect)

            # Progress arc (blue) - starts at top (90°) and goes clockwise
            if self._progress_value > 0:
                arc_color = QColor(0, 122, 204, arc_alpha)
                painter.setPen(QPen(arc_color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                start_angle = 90 * 16  # Start at top (Qt uses 1/16th degree, 90° = top)
                span_angle = -int(self._progress_value * 360 * 16)  # Negative = clockwise
                painter.drawArc(arc_rect, start_angle, span_angle)

            # Percentage text in center
            percent_text = f"{int(self._progress_value * 100)}%"
            percent_font = QFont('Segoe UI', 11, QFont.Weight.Bold)
            painter.setFont(percent_font)
            painter.setPen(QColor(224, 224, 224, arc_alpha))
            fm_pct = QFontMetrics(percent_font)
            pct_width = fm_pct.horizontalAdvance(percent_text)
            pct_y = center_y + fm_pct.ascent() // 2 - 2
            painter.drawText(int(center_x - pct_width / 2), int(pct_y), percent_text)
        else:
            # === SPINNING MODE: Rotating arc ===
            arc_color = QColor(0, 122, 204, arc_alpha)
            pen = QPen(arc_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw arc spanning 270 degrees, rotating
            start_angle = self._progress_angle * 16  # Qt uses 1/16th degree
            span_angle = 270 * 16
            painter.drawArc(arc_rect, start_angle, span_angle)

            # Draw inner circle (subtle)
            inner_color = QColor(60, 60, 60, int(100 * self._fade_opacity))
            painter.setPen(QPen(inner_color, 1))
            painter.drawEllipse(QRectF(center_x - 18, center_y - 18, 36, 36))

        # Draw status text
        text_alpha = int(224 * self._fade_opacity)
        text_color = QColor(224, 224, 224, text_alpha)
        font = QFont('Segoe UI', 12, QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(text_color)

        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self._status_text)
        text_y = center_y + 60
        painter.drawText(int(center_x - text_width / 2), int(text_y), self._status_text)

        # Draw subtle hint text
        hint_text = "Please wait..."
        hint_alpha = int(140 * self._fade_opacity)
        hint_color = QColor(160, 160, 160, hint_alpha)
        hint_font = QFont('Segoe UI', 9, QFont.Weight.Normal)
        painter.setFont(hint_font)
        painter.setPen(hint_color)

        fm2 = QFontMetrics(hint_font)
        hint_width = fm2.horizontalAdvance(hint_text)
        painter.drawText(int(center_x - hint_width / 2), int(text_y + 24), hint_text)


class InfoOverlay(QWidget):
    """Elegant welcome message in the center of the window - clickable to open a file."""
    file_clicked = Signal()

    def __init__(self, text, parent=None):
        super().__init__(parent)

        # CRITICAL FIX: Use Tool window to ensure it floats above native MPV window
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pulse_timer = None  # Lazy initialization
        self._timer_initialized = False
        self._pulse_direction = -1
        self._pulse_value = 0.0
        self._hover = False
        self.setAcceptDrops(True)  # allow dropping a file onto the welcome area / icon too

    def _ensure_timer_initialized(self):
        """Initialize pulse timer in GUI thread when first needed"""
        if not self._timer_initialized:
            self._pulse_timer = QTimer(self)
            self._pulse_timer.timeout.connect(self._update_pulse)
            self._timer_initialized = True
        # Only start if visible
        if self.isVisible() and not self._pulse_timer.isActive():
            self._pulse_timer.start(30)

    def showEvent(self, event):
        """Start animation when shown."""
        super().showEvent(event)
        if self._timer_initialized and self._pulse_timer:
            self._pulse_timer.start(30)

    def hideEvent(self, event):
        """Stop animation when hidden to prevent unnecessary CPU usage."""
        if self._timer_initialized and self._pulse_timer:
            self._pulse_timer.stop()
        super().hideEvent(event)

    def _update_pulse(self):
        self._pulse_value += self._pulse_direction * 0.02
        if self._pulse_value <= 0.0:
            self._pulse_value = 0.0
            self._pulse_direction = 1
        elif self._pulse_value >= 1.0:
            self._pulse_value = 1.0
            self._pulse_direction = -1
        self.update()

    def paintEvent(self, event):
        self._ensure_timer_initialized()  # Lazy timer creation
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        center_x = self.width() // 2
        center_y = self.height() // 2 - 40

        icon_color = QColor(0, 122, 204, 200)
        path = QPainterPath()
        path.moveTo(center_x - 30, center_y - 12)
        path.lineTo(center_x - 10, center_y - 12)
        path.lineTo(center_x - 6, center_y - 20)
        path.lineTo(center_x + 30, center_y - 20)
        path.lineTo(center_x + 30, center_y + 20)
        path.lineTo(center_x - 30, center_y + 20)
        path.closeSubpath()
        # Fill the folder so its ENTIRE surface is clickable, not just the outline: a
        # translucent (WA_TranslucentBackground) window only receives mouse input on painted
        # pixels, so a hollow icon let clicks fall through its transparent interior. A gentle
        # pulse (brighter on hover) also signals that it is clickable.
        fill_alpha = 95 if self._hover else 42 + int(self._pulse_value * 26)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 122, 204, fill_alpha))
        painter.drawPath(path)
        painter.strokePath(path, QPen(icon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                      Qt.PenJoinStyle.RoundJoin))

        text_y = center_y + 60
        font = QFont('Segoe UI', 14, QFont.Weight.Normal)
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self.text)
        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - text_width / 2), int(text_y), self.text)

        subtitle = "MP4, MKV, AVI (3D & HDR)"
        subtitle_font = QFont('Segoe UI', 10, QFont.Weight.Normal)
        painter.setFont(subtitle_font)
        fm2 = QFontMetrics(subtitle_font)
        subtitle_width = fm2.horizontalAdvance(subtitle)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(int(center_x - subtitle_width / 2), int(text_y + 26), subtitle)

        app_title = "SyLC Player"
        title_font = QFont('Segoe UI', 24, QFont.Weight.Normal)
        painter.setFont(title_font)
        fm3 = QFontMetrics(title_font)
        title_width = fm3.horizontalAdvance(app_title)
        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - title_width / 2), 60, app_title)

        edition = "3D Edition"
        edition_font = QFont('Segoe UI', 9, QFont.Weight.Normal)
        painter.setFont(edition_font)
        fm4 = QFontMetrics(edition_font)
        edition_width = fm4.horizontalAdvance(edition)
        painter.setPen(QColor(0, 122, 204, 180))
        painter.drawText(int(center_x - edition_width / 2), 78, edition)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    event.acceptProposedAction()
                    parent = self.parent()
                    if parent is not None and hasattr(parent, 'play_file'):
                        parent.play_file(path)
        except Exception:
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.file_clicked.emit()
        super().mousePressEvent(event)


# --- NEW THREAD (V12) ---


class PlayerWindow(QMainWindow):
    """Main window."""

    # Signals for thread-safe PGS callbacks (cross-thread communication)
    pgs_extraction_complete = Signal(str)  # Emits file_path when extraction is done
    pgs_load_complete = Signal(str, int)  # Emits (sup_path, track_index) after extraction
    pgs_parse_complete = Signal(bool, int)  # Emits (success, track_index) after parsing
    pgs_notification = Signal(str, bool)  # Emits (message, is_success) for notifications
    pgs_tracks_detected = Signal(list)  # Emits list of detected PGS tracks
    extraction_progress = Signal(float)  # Emits progress 0.0-1.0 during subtitle extraction
    # Text subtitle (SRT/ASS) overlay: mpv's 'sub-text' observer fires on the mpv
    # event thread; this signal marshals the cue text onto the Qt main thread.
    mpv_sub_text_changed = Signal(str)
    # Authored 3D depth of the active text track (disparity, measured pairs) —
    # emitted from the background analysis thread, handled on the main thread.
    text_sub_depth_ready = Signal(float, int)

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

        # --- Controls Overlay (Floating) ---
        # Reparent to self (QMainWindow) to ensure it sits above the Central Widget
        self.controls_overlay = ControlsOverlay(self)
        # Note: We do NOT add it to the layout anymore. It will be positioned manually in resizeEvent.
        self.controls_overlay.raise_()

        print("[STARTUP] Controls overlay created")
        self.info_overlay = InfoOverlay("Click here or drop a file", self)
        print("[STARTUP] Info overlay created")
        self.loading_overlay = LoadingOverlay(self)
        print("[STARTUP] Loading overlay created")
        # --- END LAYOUT FIX ---

        # MVC related
        self.demuxer = None
        self.mvc_decoder_thread = None
        self.mvc_mode_active = False
        self.framepacking_window = None  # Will be created when needed

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
            self.pgs_notification.connect(lambda msg, ok: self.show_3d_notification(msg, success=ok))
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

        # V60: persisted per-install settings (currently: the A/V sync trim set
        # with [ and ] — re-applied to every new decoder thread).
        self._app_settings = self._load_app_settings()
        
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
        self._stall_watchdog = QTimer(self)
        self._stall_watchdog.setInterval(3000)
        self._stall_watchdog.timeout.connect(self._check_decoder_stall)
        # Do not start the watchdog now - it will be started when the MVC decoder starts

        # State
        self.has_media = False
        self.is_playing = False
        self.is_3d_enabled = False
        self.current_stereo_mode = 'auto'
        self.video_3d_info = None
        self.current_video_fps = 24.0
        self.current_file_path = None
        self._archiving = False  # True while a disc→ISO image runs (locks playback)
        self.is_3d_capable = False
        self.controls_hide_timer = None  # Lazy initialization
        self._controls_timer_initialized = False
        self._is_loading_file = False  # V7a: Protection against rapid file changes

        # Timer for periodic timeline updates (for MVC mode where MPV may not report time-pos)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100)  # V7b: 100ms refresh for smoother timeline progression
        self._playback_timer.timeout.connect(self._update_playback_position)

        # Audio VU meter: poll mpv's real audio levels (astats af-metadata) at ~30 Hz
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
        self._nav_last_activity = 0.0
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

        print("[STARTUP] PlayerWindow initialized successfully")

    def _check_3d_vision_availability(self):
        """
        Force enable 3D capabilities without external checks.
        The user requested to remove 3D Vision verification entirely.
        """
        self.is_3d_capable = True
        # The 3D button starts disabled (no media yet); _update_3d_button_state() enables
        # it only when genuine 3D content (MVC / SBS / TAB) is loaded.
        self.controls_overlay.mode_3d_button.setEnabled(False)
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

    def _initialize_player(self):
        """Configures and initializes the mpv instance with optimal settings."""
        QTimer.singleShot(100, self._setup_mpv_player)

    def _setup_mpv_player(self):
        """Advanced MPV configuration with 3D support."""
        if not self.video_widget.winId():
            logger.warning("winId not available, retrying in 100ms.")
            QTimer.singleShot(100, self._setup_mpv_player)
            return

        # V61 STABILITY: never stack instances. stop_playback arms an async re-init;
        # if a load already re-created the player synchronously (or re-inits pile up
        # after several stops), creating another MPV would LEAK the previous one with
        # live observers and its event thread — a classic source of random crashes.
        if getattr(self, 'player', None) is not None:
            logger.info("[MPV] _setup_mpv_player: instance already alive — skipping re-init")
            return

        win_id = str(int(self.video_widget.winId()))
        logger.info(f"Configuring MPV with winId: {win_id}")

        mpv_config = {
            'wid': win_id,
            # === VIDEO OUTPUT - Optimized for HDR & Fullscreen ===
            'vo': 'gpu-next',
            'gpu-api': 'd3d11',
            'hwdec': 'auto-copy',

            # === D3D11 FULLSCREEN PERFORMANCE ===
            # Triple buffering for smooth fullscreen playback
            'd3d11-flip': 'no',                     # Disable flip model for smooth windowed HDR
            'd3d11-sync-interval': 1,               # VSync on (1 frame)
            'swapchain-depth': 3,                   # Triple buffering
            'd3d11-exclusive-fs': 'no',             # CRITICAL: Disable exclusive fullscreen to preserve HDR

            # === HDR PASSTHROUGH CONFIGURATION ===
            # Force PQ swapchain for HDR preservation
            'd3d11-output-csp': 'pq',
            'target-colorspace-hint': 'yes',
            # Let MPV auto-detect HDR capabilities
            'target-trc': 'auto',
            'target-prim': 'auto',
            'target-peak': 'auto',
            # Only tone-map if display doesn't support HDR
            'tone-mapping': 'auto',
            'hdr-compute-peak': 'yes',
            'video-output-levels': 'full',
            'dither-depth': 'auto',
            # Ensure proper GPU processing for HDR
            'gpu-dumb-mode': 'no',

            # === FRAME TIMING - Smooth Playback ===
            # display-resample syncs video to display refresh rate
            'video-sync': 'display-resample',
            'interpolation': 'yes',                 # Enable for smoother motion
            'tscale': 'oversample',                 # Fast temporal scaling
            'interpolation-threshold': 0.0001,     # Lower = more interpolation

            # === RTX 4090 OPTIMIZATIONS ===
            # High-quality scaling for powerful GPUs
            'scale': 'ewa_lanczossharp',           # Best quality upscaling
            'dscale': 'mitchell',                   # Good downscaling
            'cscale': 'ewa_lanczossoft',           # Chroma upscaling
            'correct-downscaling': 'yes',           # Correct downscaling in linear light
            'linear-downscaling': 'yes',            # Linear light downscaling (HDR correct)
            'sigmoid-upscaling': 'yes',             # Better upscaling quality
            'deband': 'yes',                        # Remove banding artifacts
            'deband-iterations': 2,                 # Fast debanding
            'deband-threshold': 35,                 # Moderate threshold
            'temporal-dither': 'yes',               # Reduce dithering flicker

            # === CACHING & BUFFERING ===
            'input-default-bindings': True,
            'cache': 'yes',
            'demuxer-readahead-secs': 20,
            'demuxer-max-bytes': '2000M',
            'demuxer-max-back-bytes': '1000M',
            'stream-buffer-size': '512k',
            # V61: 'index': 'recreate' REMOVED — it made mpv ignore the MKV's own
            # Cues and linearly re-parse the file up to every deep seek target
            # (2-6s of audio silence per seek on a 26GB MKV, worse the deeper
            # the seek). Default indexing uses the container's seek index; files
            # with broken/missing Cues still fall back to mpv's own heuristics.
            'hr-seek': 'yes',

            # === DECODING ===
            'vd-lavc-threads': 0,                   # Auto-detect optimal threads

            # === UI & MISC ===
            'osc': False,
            'volume': 100,
            'mute': 'no',
            'blend-subtitles': 'video',
            'gpu-shader-cache': 'yes',
        }

        try:
            self.player = mpv.MPV(**mpv_config)
            self.player['msg-level'] = 'all=info'
            logger.info("MPV instance created successfully.")
            self._vu_timer.start()   # begin polling audio levels for the VU meter

            # FIX: Delay property observers to let MPV event thread fully initialize
            # This prevents the "Windows fatal exception: code 0xe24c4a02" error
            def _setup_observers():
                try:
                    # Check if player is still valid before observing
                    if hasattr(self, 'player') and self.player:
                        self.player.observe_property('time-pos', self.on_time_update)
                        self.player.observe_property('duration', self.on_duration_change)
                        self.player.observe_property('pause', self.on_pause_state_change)
                        self.player.observe_property('eof-reached', self.on_end_of_file)
                        self.controls_overlay.time_slider.set_player(self.player)
                        logger.info("[MPV] Property observers connected.")
                except Exception as e:
                    logger.warning(f"[MPV] Could not set up observers (safe to ignore): {e}")

            QTimer.singleShot(100, _setup_observers)  # 100ms delay
        except Exception as e:
            logger.error(f"Error initializing mpv or observers: {e}")
            # Non-fatal if observers fail, but player creation failure is fatal
            if not hasattr(self, 'player'):
                QMessageBox.critical(self, "MPV Error",
                                     f"Error initializing mpv: {e}\n\nMake sure mpv-2.dll is in the same folder.")
                sys.exit(1)

    def on_end_of_file(self, _, reached):
        """Handle end of file event."""
        # V14b: Ignore during transition
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        # Ignore once the MVC decoder's own EOS has already started teardown
        # (_on_mvc_finished sets _playback_ended=True). Without this, MPV's
        # delayed EOF event re-triggers stop_playback → _stop_mvc_decoder
        # a second time, leading to the double-cleanup pattern in the log.
        if getattr(self, '_playback_ended', False):
            return
        if reached:
            # Use singleShot to perform UI updates on the main thread
            QTimer.singleShot(0, self.stop_playback)

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
        self.controls_overlay.mode_3d_toggled.connect(self.toggle_3d_mode)
        self.controls_overlay.stereo_mode_changed.connect(self.change_stereo_mode)
        self.controls_overlay.audio_track_changed.connect(self.change_audio_track)
        self.controls_overlay.subtitle_track_changed.connect(self.change_subtitle_track)
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

        # If mouse is not over controls overlay and not outside window, start timer
        if not self.controls_overlay.underMouse() and not self._mouse_outside_window:
            self._mouse_inactivity_timer.start()

    def _ensure_controls_timer_initialized(self):
        """Initialize controls hide timer in GUI thread when first needed"""
        if not self._controls_timer_initialized:
            self.controls_hide_timer = QTimer(self)
            self.controls_hide_timer.setSingleShot(True)
            self.controls_hide_timer.timeout.connect(self.hide_controls)
            self._controls_timer_initialized = True

    def _on_slider_pressed(self):
        """User started dragging the slider. Pause playback."""
        self._is_scrubbing = True
        if self.player:
            self._was_playing_before_scrub = not self.player.pause
            if self._was_playing_before_scrub:
                self.player.pause = True
        
        # Stop auto-hiding controls while scrubbing
        self._ensure_controls_timer_initialized()
        self.controls_hide_timer.stop()

    def _on_slider_moved(self, value):
        """User is dragging. Update UI ONLY. Do NOT seek decoder."""
        if not self._is_scrubbing: return
        # Value is in ms, convert to seconds for set_time (which expects seconds)
        self.controls_overlay.set_time(float(value) / 1000.0)

    def _on_slider_released(self):
        """User released the slider. Execute the seek."""
        self._is_scrubbing = False
        # Value is in ms, convert to seconds
        target_time = float(self.controls_overlay.time_slider.value()) / 1000.0
        # PERFECT-SYNC SNAP: land exactly on the frame the tooltip promised
        # (the exact vignette's IDR), when one was shown for this position.
        snapped = self.controls_overlay.time_slider.snap_to_vignette(target_time)
        if snapped != target_time:
            logger.info(f"[THUMB] click snap: {target_time:.3f}s -> vignette IDR {snapped:.3f}s")
            target_time = snapped
        print(f"[SEEK] Slider released at {target_time:.2f}s")
        
        # Use robust seek queue to prevent freezing/race conditions
        if hasattr(self, '_seek_queue'):
            self._seek_queue.request_seek(target_time, is_mvc=self.mvc_mode_active)
        else:
            self._handle_seek_request(target_time)

        # Resume logic is handled by the seek queue or explicit pause/unpause signals

    def _seek_stress_tick(self):
        """DEV (SYLC_SEEK_STRESS): drive an auto-seek through the real seek queue to
        reproduce the intermittent seek crash. Cycles positions across the file."""
        try:
            try:
                dur = float(self.player.duration or 0.0)
            except Exception:
                dur = 0.0
            if dur <= 20.0:
                return
            self._seek_stress_n += 1
            frac = (self._seek_stress_n * 0.137) % 1.0   # spread targets across the file
            target = 5.0 + frac * (dur - 15.0)
            logger.warning(f"[SEEK-STRESS] #{self._seek_stress_n} -> {target:.2f}s (dur={dur:.1f})")
            if hasattr(self, '_seek_queue') and self._seek_queue:
                self._seek_queue.request_seek(target, is_mvc=self.mvc_mode_active)
        except Exception as e:
            logger.error(f"[SEEK-STRESS] tick error: {e}")

    def _reload_test_tick(self):
        """DEV (SYLC_RELOAD_AFTER): load a 2nd file to reproduce the reload black screen."""
        try:
            if getattr(self, '_reload_done', False):
                return
            self._reload_done = True
            f = getattr(self, '_reload_file', '') or self.current_file_path
            logger.warning(f"[RELOAD-TEST] loading 2nd file now: {f}")
            self.play_file(f)
        except Exception as e:
            logger.error(f"[RELOAD-TEST] error: {e}")

    def _handle_seek_request(self, time_pos):
        """Performs the actual seek operation."""
        if not self.current_file_path: return
        
        # STABILITY: Block re-entrant seeks immediately
        if getattr(self, '_is_seeking', False):
            return
        self._is_seeking = True
        self.controls_overlay.time_slider.set_busy(True)

        # Clear PGS subtitle during seek
        if self._subtitle_manager:
            self._subtitle_manager.on_seek()
        # Clear text subtitle overlay too (mpv re-emits sub-text after the seek)
        if self._text_sub_active and self._text_subtitle_renderer:
            self._text_subtitle_renderer.clear()

        self.show_3d_notification(f"Seeking to {time_pos:.1f}s...", success=True)

        # 1. MVC Mode Seek
        if self.mvc_mode_active:
            # Update internal trackers immediately to reflect seek target
            self._current_precise_time = time_pos
            self._last_mvc_timestamp = time_pos
            self._last_ui_time = time_pos
            self._last_timeline_update_time = time.monotonic()
            self.controls_overlay.set_time(time_pos)

            # Use robust queue if available (debounce + cooldown)
            if hasattr(self, '_seek_queue') and self._seek_queue:
                self._seek_queue.request_seek(time_pos, is_mvc=True)
                return
            # Fallback: simple in-thread seek
            try:
                if self.player:
                    self.player.pause = True
                    self.player.time_pos = time_pos
            except Exception as e:
                print(f"[MVC] mpv seek failed: {e}")

            if self.mvc_decoder_thread and self.mvc_decoder_thread.isRunning():
                print(f"[MVC] Requesting in-thread seek to {time_pos:.3f}s (fallback)")
                # V7b FIX: Prime audio clock to target to prevent false drift calc
                self.mvc_decoder_thread.update_audio_clock(time_pos)
                self.mvc_decoder_thread.seek(time_pos)
                self._decoder_start_position = time_pos
                self._sync_adjustment_count = 0
            else:
                print(f"[MVC] Hard start at {time_pos:.3f}s (fallback)")
                self._start_mvc_decoder(start_time=time_pos)

            if self.player and self._was_playing_before_scrub:
                QTimer.singleShot(100, lambda: setattr(self.player, 'pause', False))

        # 2. Standard 2D Mode Seek
        else:
            try:
                if self.player: 
                    self.player.time_pos = time_pos
                    # Ensure internal tracker is updated for 2D mode as well
                    self._decoder_start_position = time_pos
            except Exception as e:
                print(f"Error during seek: {e}")

    def on_seek(self, time_pos):
        self._handle_seek_request(time_pos)

    def _handle_mvc_seek(self, time_pos):
        self._handle_seek_request(time_pos)

    @Slot()
    def _on_mvc_seek_finished(self):
        # V8 SEEK CRASH FIX: Resume OpenGL rendering after seek completes
        # Resume framepacking window widget
        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            try:
                if hasattr(self.framepacking_window.display_widget, 'resume_rendering'):
                    self.framepacking_window.display_widget.resume_rendering()
            except Exception:
                pass
        # Resume embedded widget
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            try:
                if hasattr(self.mvc_embedded_widget, 'resume_rendering'):
                    self.mvc_embedded_widget.resume_rendering()
            except Exception:
                pass

        if hasattr(self, '_seek_queue') and self._seek_queue:
            self._seek_queue.notify_seek_finished()

    @Slot(float)
    def _on_mvc_seek_idr_found(self, cues_timestamp: float):
        """V8 INDEX-BASED SYNC: Atomic MPV ↔ Decoder synchronization.

        ╔═══════════════════════════════════════════════════════════════════╗
        ║  MATHEMATICAL FORMULA:                                            ║
        ║  T_audio = T_video = T_cues (single source of truth)              ║
        ║                                                                   ║
        ║  Before V8: T_audio ≠ T_video due to timestamp corrections        ║
        ║  After V8: T_audio = T_video = T_cues (perfect synchronization)   ║
        ╚═══════════════════════════════════════════════════════════════════╝
        """
        logger.info(f"[V8-SYNC] ========== ATOMIC SYNC: {cues_timestamp:.3f}s ==========")

        # ATOMIC STEP 1: MPV audio → T_cues
        if self.player:
            try:
                self.player.time_pos = cues_timestamp
                logger.info(f"[V8-SYNC] MPV audio seeked to {cues_timestamp:.3f}s")
            except Exception as e:
                logger.warning(f"[V8-SYNC] MPV seek warning: {e}")

        # ATOMIC STEP 2: All trackers → T_cues
        self._current_precise_time = cues_timestamp
        self._last_mvc_timestamp = cues_timestamp
        self._last_ui_time = cues_timestamp
        self._last_timeline_update_time = time.monotonic()

        # ATOMIC STEP 3: Decoder audio clock → T_cues
        if self.mvc_decoder_thread:
            self.mvc_decoder_thread.update_audio_clock(cues_timestamp)

        # ATOMIC STEP 4: UI → T_cues
        self.controls_overlay.set_time(cues_timestamp)

        # RESET sync state (clean slate)
        self._sync_bias = 0.0
        self._cumulative_drift = 0.0

        # V10 SSIF FIX: Now that both audio and video are positioned, UNPAUSE MPV
        # This is critical for SSIF files where we kept MPV paused during decoder init
        if self.player:
            try:
                self.player.pause = False
                logger.info(f"[V8-SYNC] MPV unpaused after atomic sync")
            except Exception as e:
                logger.warning(f"[V8-SYNC] MPV unpause warning: {e}")

        logger.info(f"[V8-SYNC] ATOMIC SYNC COMPLETE: T_audio = T_video = {cues_timestamp:.3f}s")

    # --- Seek Logic Handlers ---
    def _on_seek_started_logic(self, target_time):
        """Called when seek starts."""
        self._is_seeking = True
        # Capture state before seek (handled in _on_slider_pressed usually, but ensure here)
        # Actually _was_playing_before_scrub is set on slider press.
        # We can use it or check current state if not scrubbing.
        if not self._is_scrubbing:
             self._was_playing_before_seek = self.is_playing
        else:
             self._was_playing_before_seek = self._was_playing_before_scrub
             
        # Force UI to target immediately and hold
        self.controls_overlay.set_time(target_time)
        # Update internal trackers to prevent drift
        self._last_ui_time = target_time
        self._current_precise_time = target_time

    def _on_seek_completed_logic(self):
        """Called when seek finishes."""
        self._is_seeking = False
        
        # Resume if we were playing
        if self._was_playing_before_seek:
            # Force resume
            if self.player:
                self._safe_mpv_set_pause(False)
                self._handle_pause_change(False) # Update UI immediately
        
    # --- Robust seek queue handlers ---
    def _on_seek_queue_pause_request(self, pause_state: bool):
        """Pause/unpause mpv from seek queue (main thread)."""
        if not self.player:
            return
        try:
            self._safe_mpv_set_pause(pause_state)
            # STABILITY: Directly update UI/Internal state to match
            self._handle_pause_change(pause_state)
        except Exception:
            pass

    def _on_seek_queue_mpv_seek(self, target_time: float):
        """Perform MPV seek from seek queue."""
        # Update internal trackers immediately to reflect seek target (2D & MVC)
        self._current_precise_time = target_time
        self._last_ui_time = target_time
        self._last_timeline_update_time = time.monotonic()
        self.controls_overlay.set_time(target_time)
        
        if not self.player:
            return
        try:
            self.player.time_pos = target_time
        except Exception as e:
            print(f"[SEEK-QUEUE] MPV seek failed: {e}")

    def _on_seek_queue_decoder_seek(self, target_time: float):
        """Perform decoder seek from seek queue."""
        # V8 SEEK CRASH FIX: Pause OpenGL rendering BEFORE seek to prevent access violation
        # Pause framepacking window widget
        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            try:
                if hasattr(self.framepacking_window.display_widget, 'pause_rendering'):
                    self.framepacking_window.display_widget.pause_rendering()
            except Exception:
                pass
        # Pause embedded widget
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            try:
                if hasattr(self.mvc_embedded_widget, 'pause_rendering'):
                    self.mvc_embedded_widget.pause_rendering()
            except Exception:
                pass

        # Update internal trackers immediately to reflect seek target
        self._current_precise_time = target_time
        self._last_mvc_timestamp = target_time
        self._last_ui_time = target_time
        self._last_timeline_update_time = time.monotonic()

        if self.mvc_decoder_thread and self.mvc_decoder_thread.isRunning():
            print(f"[SEEK-QUEUE] Requesting decoder seek to {target_time:.3f}s")
            # CRITICAL (SSIF/M2TS): the demuxer's proportional byte seek needs the media
            # duration. mpv has it by now (the slider works) even if the async observer
            # never propagated it, so push it right before seeking (the guaranteed point).
            # Without it the C++ seek divides into 0ms and lands at byte 0 = restart.
            try:
                _dur = (self.player.duration if self.player else None) or \
                       (self.video_3d_info.get('duration') if self.video_3d_info else None)
                if _dur and _dur > 0:
                    self.mvc_decoder_thread.set_media_duration(float(_dur))
            except Exception:
                pass
            # V7b FIX: Prime audio clock to target to prevent false drift calc
            self.mvc_decoder_thread.update_audio_clock(target_time)
            self.mvc_decoder_thread.seek(target_time)
            self._decoder_start_position = target_time
            self._sync_adjustment_count = 0
        else:
            print(f"[SEEK-QUEUE] Starting decoder at {target_time:.3f}s")
            self._start_mvc_decoder(start_time=target_time)

    def update_ui_state(self):
        self.controls_overlay.show()
        self.info_overlay.setVisible(not self.has_media)
        if self.has_media:
            # Metrics overlay disabled to remove top-left artifact
            if hasattr(self, 'metrics_overlay'):
                self.metrics_overlay.hide()
            
            # if not self.metrics_overlay.isVisible() and self.metrics_overlay.has_metrics():
            #    self.metrics_overlay.show()
        else:
            if hasattr(self, 'metrics_overlay'):
                self.metrics_overlay.reset()

    def _controls_shown(self):
        """True if the nav bar is actually visible (accounts for the fullscreen opacity trick)."""
        if not self.controls_overlay.isVisible():
            return False
        eff = self.controls_overlay.graphicsEffect()
        if eff and eff.opacity() < 0.1:
            return False
        return True

    def _controls_busy(self):
        """True if the user is interacting with the bar (hovering it or an open dropdown) — never auto-hide then."""
        try:
            if self.controls_overlay.underMouse():
                return True
            for combo in (self.controls_overlay.audio_track_combo,
                          self.controls_overlay.subtitle_track_combo,
                          self.controls_overlay.stereo_mode_combo):
                if combo.view().isVisible():
                    return True
        except Exception:
            pass
        return False

    def _mark_activity(self):
        """Register mouse activity inside the window: show the bar and reset the idle clock."""
        self._nav_last_activity = time.monotonic()
        if not self._controls_shown():
            self.show_controls()

    def _nav_poll_tick(self):
        """SINGLE source of truth for the nav bar: show on movement inside the window,
        hide after 3 s (playing) / 5 s (paused) of no movement inside the window."""
        try:
            if not self.has_media:
                self._nav_had_media = False
                return  # before/after playback: leave the bar as-is
            if not self._nav_had_media:
                # Playback just started: show the bar, then let it auto-hide normally.
                self._nav_had_media = True
                self._nav_last_cursor = QCursor.pos()
                self._mark_activity()
                return
            pos = QCursor.pos()
            moved = (pos - self._nav_last_cursor).manhattanLength() > 1
            self._nav_last_cursor = pos
            inside = (self.isVisible() and not self.isMinimized()
                      and self.frameGeometry().contains(pos))
            if moved and inside:
                self._mark_activity()
                return
            if self._controls_shown():
                if self._controls_busy():
                    self._nav_last_activity = time.monotonic()  # defer while interacting
                    return
                timeout = 3.0 if self.is_playing else 5.0
                if (time.monotonic() - self._nav_last_activity) >= timeout:
                    self.hide_controls()
        except Exception:
            pass

    def show_controls(self):
        # V15: Stop old timer if it exists (for compatibility)
        self._ensure_controls_timer_initialized()
        self.controls_hide_timer.stop()

        # V14b: Restore opacity and ensure visibility
        opacity_effect = self.controls_overlay.graphicsEffect()
        if opacity_effect:
            opacity_effect.setOpacity(1.0)
        self.controls_overlay.show()
        self.controls_overlay.raise_()  # Ensure it floats on top
        self.setCursor(Qt.CursorShape.ArrowCursor)

        # V14b RENDER HEARTBEAT: Stop heartbeat when controls are visible (UI activity is sufficient)
        if self._render_heartbeat_timer.isActive():
            self._render_heartbeat_timer.stop()

        # V15: Inactivity timer is started by mouseMoveEvent, not here
        # This ensures controls only hide after mouse stops moving for 3s

    def hide_controls(self):
        # V3: auto-hide works during playback AND when paused — the timing (3 s playing /
        # 5 s paused) is enforced by _nav_poll_tick, so no play-state gate here.

        # V14b FULLSCREEN SMOOTHNESS FIX: In fullscreen, keep controls "technically visible"
        # but fully transparent. This maintains DWM compositor activity which prevents stuttering.
        # When controls are completely hidden, Windows DWM changes composition behavior.
        if self.isFullScreen():
            # Use QGraphicsOpacityEffect for child widgets
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            opacity_effect = self.controls_overlay.graphicsEffect()
            if not opacity_effect:
                opacity_effect = QGraphicsOpacityEffect(self.controls_overlay)
                self.controls_overlay.setGraphicsEffect(opacity_effect)
            opacity_effect.setOpacity(0.0)  # Invisible but still composited
        else:
            self.controls_overlay.hide()

        # Only hide cursor if we are over the video area (simplified: just hide it)
        self.setCursor(Qt.CursorShape.BlankCursor)

        # V14b RENDER HEARTBEAT: Start heartbeat when controls hide in fullscreen
        # This maintains Qt event loop activity for smooth MPV rendering
        # V7b++ STUTTER FIX: Don't start in MVC mode - D3D11 handles its own timing
        if self.isFullScreen() and not self._render_heartbeat_timer.isActive() and not self.mvc_mode_active:
            self._render_heartbeat_timer.start()

    def _on_mouse_inactivity(self):
        """Deprecated: the nav bar's auto-hide is now driven solely by _nav_poll_tick
        (single source of truth). Kept as a no-op so the legacy 3 s timer can't double-hide."""
        return

    def _render_heartbeat(self):
        """V14b: Maintain rendering smoothness when controls are hidden in fullscreen.

        When UI elements are hidden, Windows DWM may reduce compositor activity.
        Force window-level operations to keep the compositor active.
        """
        # V7b++ STUTTER FIX: Skip in MVC mode - D3D11 widget handles its own rendering
        # The heartbeat was designed for MPV rendering, not for MVC/D3D11 mode
        if self.mvc_mode_active:
            return

        if self.is_playing:
            # Force a window operation to keep DWM compositor engaged
            # This triggers the same code path as having visible UI elements
            self.video_widget.repaint()  # Immediate repaint, not deferred

            # Also process any pending events to maintain event loop cadence
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

    def on_duration_change(self, _, value):
        """MPV duration changed - called from MPV event thread!"""
        # V14b: Ignore during transition
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        # Schedule UI updates on main thread
        QTimer.singleShot(0, lambda: self._handle_duration_change(value))

    def _handle_duration_change(self, value):
        """Handle duration change in main Qt thread"""
        self.controls_overlay.set_duration(value)
        if self.current_file_path:
            self._apply_preview_thumbs_policy(self.current_file_path)
            self.controls_overlay.time_slider.set_video_file(self.current_file_path, value or 0)
            # THUMB: duration arrival = playback is up (mpv loaded the file for
            # both the 2D and MVC paths) → arm shortly after; seeks re-disarm.
            QTimer.singleShot(1000, lambda: getattr(self, '_thumb_service', None)
                              and self._thumb_service.arm())
        # CRITICAL (SSIF/M2TS seek): mpv reports duration asynchronously, usually AFTER
        # the MVC decoder + its demuxer were created, so the demuxer's proportional seek
        # never got a duration and landed at byte 0 (restart). Push it now so seeks work.
        if value and getattr(self, 'mvc_decoder_thread', None):
            try:
                self.mvc_decoder_thread.set_media_duration(float(value))
            except Exception as e:
                logger.warning(f"[MVC] set_media_duration propagation failed: {e}")

    def on_time_update(self, _, value):
        """MPV time position changed - called from MPV event thread!"""
        try:
            # V14b: Ignore during transition
            if getattr(self, '_mpv_transition_in_progress', False):
                return
            # V8 CRASH FIX: Skip entirely during seek to reduce MPV contention
            if getattr(self, '_is_seeking', False):
                return
            # Schedule UI updates on main thread
            QTimer.singleShot(0, lambda: self._handle_time_update(value))
        except OSError:
            pass # Ignore access violations during shutdown
        except Exception:
            pass

    def _set_ui_time(self, new_time: float, force: bool = False):
        """Clamp UI time to avoid small backward jumps unless forced (e.g., after seek)."""
        try:
            if new_time is None: return
            new_time = float(new_time)

            if force or self._is_scrubbing or getattr(self, '_is_seeking', False):
                self._last_ui_time = new_time
            else:
                # Prevent backward jitter (strict monotonic)
                if new_time < self._last_ui_time:
                    # Allow large jumps (seek/loop) - threshold 1.0s
                    if (self._last_ui_time - new_time) > 1.0:
                        self._last_ui_time = new_time
                    else:
                        # Jitter: ignore update, keep last time
                        pass 
                else:
                    self._last_ui_time = new_time
            
            self.controls_overlay.set_time(self._last_ui_time)

            # Update subtitle manager with current playback time
            if self._subtitle_manager and getattr(self, '_active_pgs_track_index', None) is not None:
                self._subtitle_manager.update_time(self._last_ui_time)
        except Exception:
            # Don't let UI updates crash the player
            pass

    def _handle_time_update(self, value):
        """Handle time update in main Qt thread"""
        try:
            # STABILITY: Ignore updates while seeking to prevent jitter
            if getattr(self, '_is_seeking', False):
                return

            # CRITICAL FIX: If MPV is in audio-only mode (MVC), it might not report time reliably during seek/stutter.
            # Use the decoder's estimated time if available and playing.
            current_time = value

            if current_time is None:
                current_time = self._current_mpv_time()

            self._set_ui_time(current_time)
            if self.player:
                metadata_duration = self.video_3d_info.get('duration') if self.video_3d_info else 0
                duration = self.player.duration or metadata_duration or 0
                self.metrics_overlay.update_playhead(current_time or 0, duration)
            if self.mvc_mode_active and self.mvc_decoder_thread:
                self.mvc_decoder_thread.update_audio_clock(current_time or 0.0)

            # PGS Subtitle update (MVC mode only)
            if self.mvc_mode_active and self._subtitle_manager and current_time is not None:
                self._subtitle_manager.update_time(current_time)
        except Exception as e:
            # logger.warning(f"[UI] Time update error: {e}")
            pass

    def _update_playback_position(self):
        """Periodic update of the playback position (for reliability in MVC mode)."""
        # CRITICAL FIX: Do not access player if media is not loaded or loading
        if not self.has_media or getattr(self, '_is_loading_file', False):
            return

        # V59c AUDIO-CLOCK FIX: this poller is the ONLY feed of the decoder's
        # audio clock (update_audio_clock below), and it used to early-return
        # during the whole seek window — freezing the decoder's clock by
        # construction. That frozen clock is what tripped the V12 hold storm
        # (post-seek stutter/freeze). Feed the decoder UNCONDITIONALLY: mpv's
        # time_pos is the truth of the audio clock at all times — pre-seek
        # values keep the stale-detector honest, and the jump when mpv lands
        # re-engages sync at the earliest possible moment.
        if self.mvc_mode_active and self.mvc_decoder_thread and self.player:
            try:
                _mpv_now = self.player.time_pos
                if _mpv_now is not None and _mpv_now >= 0.0:
                    self.mvc_decoder_thread.update_audio_clock(_mpv_now)
            except Exception:
                pass

        # Do not update the UI/timeline if user is scrubbing OR seeking
        if self._is_scrubbing or getattr(self, '_is_seeking', False):
            self._last_timeline_update_time = time.monotonic()
            return

        if not self.is_playing:
            self._last_timeline_update_time = time.monotonic()
            return

        try:
            new_time = None
            
            # 1. Try MVC Timestamp (most accurate for SINGLE-clip video). For a multi-segment
            # (seamless-branching) feature the decoder's frame timestamp is per-clip and snaps
            # back at each segment junction, so we use mpv's continuous GLOBAL edl:// clock
            # instead (step 2 below) — proven continuous across junctions.
            _multi_segment = bool(getattr(self, '_pending_feature_segments', None))
            if (self.mvc_mode_active and not _multi_segment
                    and hasattr(self, '_last_mvc_timestamp') and self._last_mvc_timestamp > 0.1):
                 # Check if it actually moved
                if not hasattr(self, '_prev_mvc_ts') or self._last_mvc_timestamp > self._prev_mvc_ts:
                    new_time = self._last_mvc_timestamp
                    self._prev_mvc_ts = self._last_mvc_timestamp

            # 2. Fallback to MPV time (Audio/Standard mode)
            # V7b++++++ CRITICAL FIX: Always get MPV position for audio clock sync
            mpv_pos = None
            if self.player:
                try:
                    mpv_pos = self.player.time_pos
                except:
                    pass

            # V7b++++++ CRITICAL SYNC FIX: Continuously update decoder's audio clock
            # Without this, the decoder extrapolates from wall-clock time and drifts!
            # V43 FIX: Accept mpv_pos >= 0 (not > 0.1) to keep audio clock fresh from file start.
            # The > 0.1 filter caused stale audio clock when MPV was near position 0,
            # leading to V12 sync dropping all frames (extrapolated clock raced ahead).
            if mpv_pos is not None and mpv_pos >= 0.0:
                if self.mvc_mode_active and self.mvc_decoder_thread:
                    # Update decoder's audio clock with ACTUAL MPV position
                    self.mvc_decoder_thread.update_audio_clock(mpv_pos)

            if new_time is None and mpv_pos is not None and mpv_pos > 0.1:
                new_time = mpv_pos
                # Sync internal counter to MPV time for 2D mode
                self._current_precise_time = float(mpv_pos)

            # 3. Synthetic Fallback (If backend is stuck but we are playing)
            if new_time is None:
                # Use our internal high-precision counter
                now = time.monotonic()
                delta = now - self._last_timeline_update_time
                
                # Limit delta to avoid huge jumps (e.g. after pause/lag)
                if delta > 0 and delta < 1.0:
                    self._current_precise_time += delta
                    new_time = self._current_precise_time
                
                self._last_timeline_update_time = now
            else:
                # Sync our internal counter to the authoritative source
                self._current_precise_time = float(new_time)
                self._last_timeline_update_time = time.monotonic()

            if new_time is not None:
                self._set_ui_time(new_time)

                # V14 FIX: Update streaming subtitles with current time
                # This is needed to detect when subtitles should expire and be cleared
                if self._subtitle_manager and self.mvc_mode_active:
                    self._subtitle_manager.update_time(new_time)

                # V7b DEBUG: Periodic log every 30 updates to verify progression
                if not hasattr(self, '_timeline_update_count'):
                    self._timeline_update_count = 0
                self._timeline_update_count += 1
                if self._timeline_update_count % 30 == 0:
                    logger.debug(f"[TIMELINE] Position updated: {new_time:.2f}s (MVC mode: {self.mvc_mode_active})")

        except Exception:
            pass  # Ignore errors if MPV is busy

    def on_pause_state_change(self, _, is_paused):
        """MPV pause state changed - called from MPV event thread!"""
        # Observer-side pause cache: lets GUI-thread code (VU meter poll) know
        # the pause state WITHOUT a blocking mpv property read (0xe24c4a02).
        # Updated even during transitions — it's a plain bool assign.
        self._mpv_pause_cache = bool(is_paused)
        # V14b: Ignore during transition
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        # Schedule UI updates on main thread to avoid timer threading issues
        QTimer.singleShot(0, lambda: self._handle_pause_change(is_paused))

    def _handle_pause_change(self, is_paused):
        """Handle pause state change in main Qt thread"""
        try:
            # CRITICAL FIX V2: If playback has ended, ignore MPV callbacks
            # This prevents the timer from restarting after MVC finishes
            if getattr(self, '_playback_ended', False):
                logger.info("[PAUSE CHANGE] Ignored - playback has ended")
                return

            # V14b: Ignore callbacks during MPV transition to prevent exceptions
            if getattr(self, '_mpv_transition_in_progress', False):
                logger.info("[PAUSE CHANGE] Ignored - MPV transition in progress")
                return

            # Robust boolean conversion for MPV property
            safe_is_paused = is_paused is True or is_paused == 'yes' or is_paused == 'true'

            self.is_playing = not safe_is_paused
            if safe_is_paused:
                # V15: Stop inactivity timer when paused - controls stay visible
                self._mouse_inactivity_timer.stop()
                self._ensure_controls_timer_initialized()
                self.controls_hide_timer.stop()
                # V14b RENDER HEARTBEAT: Stop heartbeat when paused
                if self._render_heartbeat_timer.isActive():
                    self._render_heartbeat_timer.stop()
                # V7b FIX: In MVC mode, keep the timer active even when paused so the cursor progresses
                if not (self.mvc_mode_active or getattr(self, '_mvc_file_detected', False)):
                    self._playback_timer.stop()  # Stop the timeline update
                    logger.info(f"[TIMELINE] Timer stopped (MVC: {self.mvc_mode_active}, detected: {getattr(self, '_mvc_file_detected', False)})")
                else:
                    logger.info(f"[TIMELINE] Timer kept active (MVC: {self.mvc_mode_active}, detected: {getattr(self, '_mvc_file_detected', False)})")
                self.show_controls()
                # Notify decoder
                if self.mvc_decoder_thread:
                    self.mvc_decoder_thread.pause()
            else:
                # V15: Start inactivity timer when playback resumes
                self._mouse_inactivity_timer.start()
                self._playback_timer.start()  # Start the timeline update
                # Notify decoder
                if self.mvc_decoder_thread:
                    self.mvc_decoder_thread.resume()

            self.controls_overlay.set_paused(safe_is_paused)
        except Exception as e:
            logger.warning(f"[UI] Error handling pause change: {e}")

    # === SAFE MPV ACCESS METHODS ===
    def _safe_mpv_command(self, *args):
        """Execute MPV command asynchronously to prevent thread crashes."""
        if not self.player:
            return None
        try:
            # V7b STABILITY FIX: Use command_async to avoid blocking main thread
            # Blocking calls to MPV from Qt thread are a major cause of 0xe24c4a02
            self.player.command_async(*args)
            return True
        except Exception as e:
            logger.warning(f"[MPV] Async command {args[0]} failed: {e}")
            return False

    def _safe_mpv_set_pause(self, paused: bool):
        """Safely set MPV pause state."""
        return self._safe_mpv_command('set', 'pause', 'yes' if paused else 'no')

    def _safe_mpv_seek(self, time_pos: float):
        """Safely seek MPV to position."""
        return self._safe_mpv_command('seek', str(time_pos), 'absolute')

    def toggle_play(self):
        if getattr(self, '_archiving', False):
            return  # playback is locked while a disc image is being written
        if self.has_media and self.player:
            try:
                # Normalize MPV property which might be string or None
                raw_pause = self.player.pause
                current_pause = raw_pause is True or raw_pause == 'yes' or raw_pause == 'true'
                
                new_pause = not current_pause
                self._safe_mpv_set_pause(new_pause)
                
                # CRITICAL FIX: Directly handle pause change to ensure immediate video stop
                # Relying solely on MPV callback can be unreliable if MPV thread is busy
                self._handle_pause_change(new_pause)
            except:
                pass

    def stop_playback(self):
        """Stops playback, resets position, and clears decoder state."""
        if not self.has_media: return

        # If MVC EOS handler already ran, the decoder is being / was already torn
        # down via its own 300ms timer. Re-running the full sequence here would
        # trigger a 2nd MVC CLEANUP (visible in the EOS logs) and attempt to
        # terminate an already-paused MPV instance, producing the "Could not
        # restore vo" warning twice.
        if getattr(self, '_playback_ended', False):
            logger.info("[PLAYER] stop_playback skipped — _on_mvc_finished already handling teardown")
            return

        print("[PLAYER] Stopping playback...")
        
        # VISUAL FIX: Hide windows IMMEDIATELY to prevent strobe of old frame
        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            self.framepacking_window.hide()
            if hasattr(self.framepacking_window.display_widget, 'clear_textures'):
                self.framepacking_window.display_widget.clear_textures()
                
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            if hasattr(self.mvc_embedded_widget, 'clear_textures'):
                self.mvc_embedded_widget.clear_textures()
                self.mvc_embedded_widget.update()

        self._playback_timer.stop()  # Stop the timeline update
        # V14b RENDER HEARTBEAT: Stop heartbeat when playback stops
        if self._render_heartbeat_timer.isActive():
            self._render_heartbeat_timer.stop()
        # Reset seek queue to prevent phantom seeks
        if hasattr(self, "_seek_queue"):
            self._seek_queue._force_reset_state()
        # V62 STOP-CRASH FIX: mark that this stop ends in terminate() so the MVC
        # cleanup SKIPS the gpu-next vo restore — rebuilding mpv's D3D11 chain an
        # instant before destroying the core was half of the 0xe24c4a02 window
        # (crash_log 2026-07-14 18:03: SEH inside mpv terminate, x3).
        self._terminating_mpv = True
        try:
            self._stop_mvc_decoder()

            if self.player:
                try:
                    # V7b STABILITY: Use terminate() instead of pause/seek 0 to fully release file handles
                    # This prevents "file in use" errors and cleans up MPV threads
                    # V61 STABILITY: stop the VU poller and NULL the reference BEFORE the
                    # async re-init — every `if self.player:` guard in the codebase then
                    # routes safely instead of poking a terminated core (access violation
                    # if a load lands inside the 500ms re-init window).
                    self._vu_timer.stop()
                    _dying = self.player
                    self.player = None
                    # V62: quiesce BEFORE destroying — same pattern as the proven
                    # V14b natural-end path (stop the core, let the event loop
                    # settle, THEN destroy). terminate() on a hot core (file
                    # loaded, vu lavfi attached) was the other half of the crash.
                    self._mpv_transition_in_progress = True
                    try:
                        _dying.command('stop')
                        time.sleep(0.150)
                    except Exception:
                        pass
                    _dying.terminate()
                    # Re-initialize player for next use after short delay
                    QTimer.singleShot(500, self._initialize_player)
                except Exception as e:
                    logger.warning(f"[MPV] Error stopping player: {e}")
        finally:
            self._terminating_mpv = False

        # V62b: a real STOP releases the disc entirely, like before the
        # preview-service era. The thumbnail service holds open handles on the
        # mounted volume (its own demuxer) — release them on its worker thread
        # first, then dismount once every reader (decoder, mpv, service) is out.
        if getattr(self, '_thumb_service', None):
            try:
                self._thumb_service.release_file()
            except Exception:
                pass
        if getattr(self, '_active_iso_mount', None) or getattr(self, '_pending_iso_mount', None):
            self.current_file_path = None      # nothing is playing anymore
            QTimer.singleShot(400, self._dismount_isos_after_stop)

        self.has_media = False
        self._update_3d_button_state()   # no media → lock the 3D button off
        self.controls_overlay.clear_format_badge()   # drop the 3D-format badge
        self.update_ui_state()
        self.controls_overlay.set_status_info("Ready")
        self.controls_overlay.set_time(0)
        self.controls_overlay.set_duration(0)
        self.setWindowTitle("SyLC 3D Player - Premium Edition")

    def toggle_fullscreen(self):
        """Toggle fullscreen using Win32 API to preserve HDR and MPV connection.

        CRITICAL: Qt's showFullScreen() triggers SDR mode on HDR displays.
        CRITICAL: Qt's setWindowFlags() recreates window and breaks MPV.
        Solution: Use Win32 API to modify window style without recreating it.
        """
        import ctypes
        from ctypes import wintypes, byref, c_void_p, c_int, c_uint

        user32 = ctypes.windll.user32
        
        # Define SetWindowPos argument types for proper casting
        user32.SetWindowPos.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_uint]
        user32.SetWindowPos.restype = ctypes.c_bool

        # Win32 constants
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        SWP_FRAMECHANGED = 0x0020
        SWP_SHOWWINDOW = 0x0040
        HWND_TOPMOST = c_void_p(-1)
        HWND_NOTOPMOST = c_void_p(-2)

        hwnd = c_void_p(int(self.winId()))

        if self._is_fake_fullscreen:
            # === EXIT FULLSCREEN ===
            if self._render_heartbeat_timer.isActive():
                self._render_heartbeat_timer.stop()

            # Restore the DWM window border + rounded corners we suppressed
            try:
                from framepacking_window_d3d11 import apply_borderless_dwm
                apply_borderless_dwm(int(self.winId()), False)
            except Exception:
                pass

            # Restore original window style
            if hasattr(self, '_saved_style'):
                user32.SetWindowLongW(int(self.winId()), GWL_STYLE, self._saved_style)
            if hasattr(self, '_saved_exstyle'):
                user32.SetWindowLongW(int(self.winId()), GWL_EXSTYLE, self._saved_exstyle)

            # Restore position and size
            if hasattr(self, '_saved_rect'):
                x, y, w, h = self._saved_rect
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, int(x), int(y), int(w), int(h), SWP_FRAMECHANGED | SWP_SHOWWINDOW)

            self._is_fake_fullscreen = False
            self.controls_overlay.set_fullscreen_icon(False)
            
            # Optimize for windowed: disable flip model to reduce compositor stuttering
            if self.player:
                try:
                    self.player['d3d11-flip'] = 'no'
                    logger.info("[HDR] Windowed: d3d11-flip=no for smooth playback")
                except Exception as e:
                    logger.warning(f"[HDR] Could not set d3d11-flip: {e}")

            # Sync framepacking window
            if self.framepacking_window and self.mvc_mode_active and self.framepacking_window.isVisible():
                self.framepacking_window.exit_fake_fullscreen()
                self.framepacking_window.raise_()

            QTimer.singleShot(100, self._apply_windowed_video_settings)
            logger.info("[FULLSCREEN-WIN32] Exited fake fullscreen")
        else:
            # === ENTER FULLSCREEN ===
            # Save current window state via Win32
            self._saved_style = user32.GetWindowLongW(int(self.winId()), GWL_STYLE)
            self._saved_exstyle = user32.GetWindowLongW(int(self.winId()), GWL_EXSTYLE)

            rect = wintypes.RECT()
            user32.GetWindowRect(int(self.winId()), byref(rect))
            self._saved_rect = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)

            # Get EXACT monitor dimensions via Win32 (avoids DPI scaling issues)
            # MonitorFromWindow + GetMonitorInfo gives us the true pixel dimensions
            MONITOR_DEFAULTTONEAREST = 2
            
            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', wintypes.DWORD),
                    ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT),
                    ('dwFlags', wintypes.DWORD),
                ]
            
            hMonitor = user32.MonitorFromWindow(int(self.winId()), MONITOR_DEFAULTTONEAREST)
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
            
            # Use rcMonitor (full monitor area) not rcWork (excludes taskbar)
            mon_x = mi.rcMonitor.left
            mon_y = mi.rcMonitor.top
            mon_w = mi.rcMonitor.right - mi.rcMonitor.left
            mon_h = mi.rcMonitor.bottom - mi.rcMonitor.top
            
            logger.info(f"[FULLSCREEN-WIN32] Monitor geometry: {mon_x},{mon_y} {mon_w}x{mon_h}")

            # Remove window decorations (borderless)
            new_style = self._saved_style & ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
            user32.SetWindowLongW(int(self.winId()), GWL_STYLE, new_style)

            # Resize to cover full monitor (HDR preserved via __COMPAT_LAYER)
            HWND_TOP = c_void_p(0)
            SWP_NOZORDER = 0x0004
            user32.SetWindowPos(
                hwnd, HWND_TOP,
                mon_x, mon_y, mon_w, mon_h,
                SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER
            )

            # Windows 11 draws a thin border + rounded corners around any
            # top-level window (the white 'liseret' all around fake-fullscreen).
            # Suppress both so the video reaches the true screen edge.
            try:
                from framepacking_window_d3d11 import apply_borderless_dwm
                apply_borderless_dwm(int(self.winId()), True)
            except Exception:
                pass

            self._is_fake_fullscreen = True
            self.controls_overlay.set_fullscreen_icon(True)
            
            # Optimize for fullscreen: flip model for best performance
            if self.player:
                try:
                    self.player['d3d11-flip'] = 'yes'
                    logger.info("[HDR] Fullscreen: d3d11-flip=yes for optimal performance")
                except Exception as e:
                    logger.warning(f"[HDR] Could not set d3d11-flip: {e}")

            # Sync framepacking window
            if self.framepacking_window and self.mvc_mode_active and self.framepacking_window.isVisible():
                self.framepacking_window.display_widget.set_stereo_mode('framepack')
                self.framepacking_window.enter_fake_fullscreen()
                self.framepacking_window.raise_()
                self.framepacking_window.activateWindow()

            logger.info(f"[FULLSCREEN-WIN32] Entered fake fullscreen {mon_w}x{mon_h} (HDR preserved)")

    def _apply_fullscreen_video_settings(self):
        """Apply optimal MPV settings for fullscreen HDR playback."""
        if not self.player:
            return
        try:
            self.player['video-sync'] = 'display-resample'
            
            # Force MPV to reset HDR/brightness settings after fullscreen transition
            # Method 1: Toggle gamma briefly
            self.player['gamma'] = 1
            self.player['gamma'] = 0
            
            # Method 2: Force video output reconfiguration
            try:
                self.player.command('vo-cmdline', 'd3d11-exclusive-fs=no')
            except:
                pass
            
            # Method 3: Re-apply HDR settings
            self.player['target-colorspace-hint'] = 'yes'
            self.player['target-trc'] = 'auto'
            self.player['target-prim'] = 'auto'
            
            # Method 4: Force DWM composition refresh
            try:
                import ctypes
                dwmapi = ctypes.windll.dwmapi
                dwmapi.DwmFlush()
                
                # Also try toggling a DWM window attribute to force HDR refresh
                hwnd = int(self.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = ctypes.c_int(1)
                dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
                value = ctypes.c_int(0)
                dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
                
                logger.info("[FULLSCREEN] Forced DWM refresh")
            except Exception as e:
                logger.warning(f"[FULLSCREEN] DWM refresh failed: {e}")
            
            logger.info("[FULLSCREEN] Applied fullscreen video settings")
        except Exception as e:
            logger.warning(f"[FULLSCREEN] Could not apply settings: {e}")

    def _refresh_windows_hdr_brightness(self):
        """Force Windows to re-apply HDR SDR brightness setting via DisplayConfig API."""
        try:
            import ctypes
            from ctypes import wintypes, Structure, byref, sizeof

            class LUID(Structure):
                _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

            class DISPLAYCONFIG_DEVICE_INFO_HEADER(Structure):
                _fields_ = [
                    ("type", wintypes.UINT),
                    ("size", wintypes.UINT),
                    ("adapterId", LUID),
                    ("id", wintypes.UINT),
                ]

            class DISPLAYCONFIG_SDR_WHITE_LEVEL(Structure):
                _fields_ = [
                    ("header", DISPLAYCONFIG_DEVICE_INFO_HEADER),
                    ("SDRWhiteLevel", wintypes.DWORD),
                ]

            class DISPLAYCONFIG_PATH_INFO(Structure):
                _fields_ = [
                    ("sourceInfo", ctypes.c_ubyte * 20),
                    ("targetInfo", ctypes.c_ubyte * 48),
                    ("flags", wintypes.UINT),
                ]

            class DISPLAYCONFIG_MODE_INFO(Structure):
                _fields_ = [
                    ("infoType", wintypes.UINT),
                    ("id", wintypes.UINT),
                    ("adapterId", LUID),
                    ("info", ctypes.c_ubyte * 64),
                ]

            QDC_ONLY_ACTIVE_PATHS = 0x00000002
            GET_SDR_WHITE_LEVEL = 0x0000000B
            SET_SDR_WHITE_LEVEL = 0x0000000C

            numPath = wintypes.UINT(0)
            numMode = wintypes.UINT(0)

            result = ctypes.windll.user32.GetDisplayConfigBufferSizes(
                QDC_ONLY_ACTIVE_PATHS, byref(numPath), byref(numMode))

            if result == 0 and numPath.value > 0:
                pathArray = (DISPLAYCONFIG_PATH_INFO * numPath.value)()
                modeArray = (DISPLAYCONFIG_MODE_INFO * numMode.value)()

                result = ctypes.windll.user32.QueryDisplayConfig(
                    QDC_ONLY_ACTIVE_PATHS, byref(numPath), pathArray,
                    byref(numMode), modeArray, None)

                if result == 0:
                    for i in range(numPath.value):
                        path_bytes = bytes(pathArray[i].targetInfo)
                        adapterId = LUID()
                        adapterId.LowPart = int.from_bytes(path_bytes[0:4], 'little')
                        adapterId.HighPart = int.from_bytes(path_bytes[4:8], 'little', signed=True)
                        targetId = int.from_bytes(path_bytes[8:12], 'little')

                        sdrLevel = DISPLAYCONFIG_SDR_WHITE_LEVEL()
                        sdrLevel.header.type = GET_SDR_WHITE_LEVEL
                        sdrLevel.header.size = sizeof(DISPLAYCONFIG_SDR_WHITE_LEVEL)
                        sdrLevel.header.adapterId = adapterId
                        sdrLevel.header.id = targetId

                        if ctypes.windll.user32.DisplayConfigGetDeviceInfo(byref(sdrLevel)) == 0:
                            currentLevel = sdrLevel.SDRWhiteLevel
                            logger.info(f"[HDR-FIX] SDR white level: {currentLevel}")
                            sdrLevel.header.type = SET_SDR_WHITE_LEVEL
                            ctypes.windll.user32.DisplayConfigSetDeviceInfo(byref(sdrLevel))
                            logger.info("[HDR-FIX] Re-applied SDR white level")
                            break

        except Exception as e:
            logger.warning(f"[HDR-FIX] Could not refresh HDR: {e}")

    def _apply_windowed_video_settings(self):
        """Apply optimal MPV settings for windowed playback."""
        if not self.player:
            return
        try:
            self.player['video-sync'] = 'display-resample'
            logger.info("[WINDOWED] Applied windowed video settings")
        except Exception as e:
            logger.warning(f"[WINDOWED] Could not apply settings: {e}")
        
        # Refresh Windows HDR brightness after exiting fullscreen
        QTimer.singleShot(200, self._refresh_windows_hdr_brightness)

    def _enter_borderless_fullscreen_win32(self):
        """Enter borderless fullscreen using Win32 API directly.
        
        This avoids Qt's setWindowFlags() which destroys and recreates the window,
        breaking the MPV player connection. Instead, we modify the window style
        directly via Windows API.
        """
        import ctypes
        from ctypes import wintypes
        
        # Win32 constants
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        WS_EX_DLGMODALFRAME = 0x00000001
        WS_EX_CLIENTEDGE = 0x00000200
        WS_EX_STATICEDGE = 0x00020000
        SWP_FRAMECHANGED = 0x0020
        SWP_NOZORDER = 0x0004
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        SWP_NOACTIVATE = 0x0010
        
        user32 = ctypes.windll.user32
        
        # Get window handle
        hwnd = int(self.winId())
        
        # Save current window state
        self._saved_style = user32.GetWindowLongW(int(self.winId()), GWL_STYLE)
        self._saved_exstyle = user32.GetWindowLongW(int(self.winId()), GWL_EXSTYLE)
        
        # Get window rect before changing
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        self._saved_rect = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        
        # Remove window decorations
        new_style = self._saved_style & ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
        new_exstyle = self._saved_exstyle & ~(WS_EX_DLGMODALFRAME | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE)
        
        user32.SetWindowLongW(int(self.winId()), GWL_STYLE, new_style)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)
        
        # Get screen geometry
        from PySide6.QtGui import QGuiApplication
        screen = self.screen() or QGuiApplication.primaryScreen()
        screen_geo = screen.geometry()
        
        # Set topmost and resize to fullscreen
        # SWP_SHOWWINDOW is critical to ensure window is visible after style change
        SWP_SHOWWINDOW = 0x0040

        x = int(screen_geo.x())
        y = int(screen_geo.y())
        w = int(screen_geo.width())
        h = int(screen_geo.height())

        logger.info(f"[FULLSCREEN-WIN32] Setting window to {x},{y} {w}x{h}")

        result = user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            x, y, w, h,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW
        )

        logger.info(f"[FULLSCREEN-WIN32] Entered borderless fullscreen {w}x{h} (SetWindowPos result={result})")

    def _exit_borderless_fullscreen_win32(self):
        """Exit borderless fullscreen using Win32 API directly."""
        import ctypes
        
        SWP_FRAMECHANGED = 0x0020
        HWND_NOTOPMOST = -2
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        
        user32 = ctypes.windll.user32
        hwnd = int(self.winId())
        
        # Restore window styles
        if hasattr(self, '_saved_style'):
            user32.SetWindowLongW(int(self.winId()), GWL_STYLE, self._saved_style)
        if hasattr(self, '_saved_exstyle'):
            user32.SetWindowLongW(int(self.winId()), GWL_EXSTYLE, self._saved_exstyle)
        
        # Restore position and size, remove topmost
        if hasattr(self, '_saved_rect'):
            x, y, w, h = self._saved_rect
            user32.SetWindowPos(
                hwnd, HWND_NOTOPMOST,
                x, y, w, h,
                SWP_FRAMECHANGED
            )
        
        logger.info("[FULLSCREEN-WIN32] Exited borderless fullscreen")

    def toggle_3d_mode(self, enabled):
        """Enables or disables 3D mode."""
        if enabled and not self._content_is_3d():
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
        if self.has_media:
            self.configure_3d_output(enabled, self.current_stereo_mode)
            if self.video_3d_info and self.video_3d_info['is_3d']:
                mode_names = {'mvc': 'MVC', 'sbs': 'Side-by-Side', 'tab': 'Top-Bottom'}
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

    def change_stereo_mode(self, mode):
        self.current_stereo_mode = mode
        if self.has_media and self.is_3d_enabled:
            self.configure_3d_output(True, mode)

    def _content_is_3d(self):
        """True iff the loaded media is genuinely stereoscopic (MVC / SBS / TAB),
        not a plain 2D file. Drives the 3D button's availability."""
        info = getattr(self, 'video_3d_info', None)
        if not info:
            return False
        return bool(info.get('is_3d')) or info.get('stereo_mode') not in (None, 'none')

    def _update_3d_button_state(self):
        """Enable the 3D button only for real 3D content with media loaded; lock it off
        for 2D files so 3D can't be toggled on a 2D video (which mis-drives the MVC
        pipeline → runaway speed + audio desync)."""
        try:
            btn = self.controls_overlay.mode_3d_button
        except Exception:
            return
        capable = self._content_is_3d() and getattr(self, 'has_media', False)
        if capable:
            btn.setEnabled(True)
            btn.setToolTip("Toggle 3D mode")
        else:
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            btn.setEnabled(False)
            btn.setToolTip("2D video — 3D unavailable")

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
            return "MVC 3D"
        if sm == 'sbs':
            return "Full-SBS 3D" if w >= 2560 else "SBS 3D"
        if sm == 'tab':
            return "Full-TAB 3D" if h >= 1600 else "TAB 3D"
        return None

    def change_audio_track(self, track_id):
        if not (self.has_media and self.player):
            return
        try:
            # mpv.command() serializes via the MPV queue instead of a direct set_property,
            # which avoids the collision with the decoder thread (SEH 0xe24c4a02).
            self.player.command('set', 'aid', str(track_id))
            print(f"Audio track changed: ID {track_id}")
        except (OSError, RuntimeError, Exception) as e:
            print(f"Error changing audio track: {e}")

    def load_audio_tracks(self):
        if not self.has_media: return
        QTimer.singleShot(500, self._fetch_audio_tracks)

    def _get_clpi_lang_map(self):
        """Return {PID: 'iso639'} from the Blu-ray .clpi for the current file (cached).

        Raw M2TS/SSIF streams carry no language tag; the per-PID language lives in
        the disc's CLIPINF/<clip>.clpi. Returns {} for non-BDMV files (e.g. MKV,
        which carry their own language tags that mpv exposes directly).
        """
        path = getattr(self, 'current_file_path', None)
        if not path:
            return {}
        if getattr(self, '_clpi_lang_map_path', None) == path:
            return self._clpi_lang_map
        clpi = _find_clpi_for_media(path)
        self._clpi_lang_map = _parse_clpi_languages(clpi) if clpi else {}
        self._clpi_lang_map_path = path
        if self._clpi_lang_map:
            logger.info(f"[CLPI] Loaded {len(self._clpi_lang_map)} stream languages from {os.path.basename(clpi)}")
        return self._clpi_lang_map

    def _fetch_audio_tracks(self):
        try:
            if not self.player: return
            track_list = self.player.track_list
            lang_map = self._get_clpi_lang_map()
            audio_tracks = []
            for track in track_list:
                if track.get('type') == 'audio':
                    track_id = track.get('id')
                    label = _friendly_track_label(track, 'audio', lang_map)
                    audio_tracks.append((track_id, label, ''))  # label is already complete
            print(f"Audio tracks found: {len(audio_tracks)}")
            self.controls_overlay.update_audio_tracks(audio_tracks)
        except Exception as e:
            print(f"Error fetching audio tracks: {e}")

    def change_subtitle_track(self, track_id):
        logger.info(f"[SUBTITLE] change_subtitle_track called with track_id={track_id}")
        logger.info(f"[SUBTITLE] has_media={self.has_media}, mvc_mode_active={self.mvc_mode_active}")
        logger.info(f"[SUBTITLE] PGS_AVAILABLE={PGS_SUBTITLE_AVAILABLE}, manager={self._subtitle_manager is not None}")
        logger.info(f"[SUBTITLE] PGS tracks detected: {len(self._pgs_subtitle_tracks)}")
        logger.info(f"[SUBTITLE] Streaming tracks detected: {len(self._streaming_subtitle_tracks)}")

        if self.has_media and self.player:
            try:
                # ========== STREAMING SUBTITLE PATH (No extraction delay!) ==========
                # Check if this is a streaming subtitle track from the MVC demuxer
                if self.mvc_mode_active and self._streaming_subtitle_tracks and self.mvc_decoder_thread:
                    # UI sends 1-based index, find the corresponding streaming track
                    # track_id 1 = first streaming track, track_id 2 = second, etc.
                    streaming_track = None
                    if track_id > 0 and track_id <= len(self._streaming_subtitle_tracks):
                        streaming_track = self._streaming_subtitle_tracks[track_id - 1]
                        logger.info(f"[STREAMING-SUBS] track_id={track_id} -> trackNumber={streaming_track.get('trackNumber')}")

                    if streaming_track and streaming_track.get('isPGS', False):
                        actual_track_number = streaming_track.get('trackNumber')
                        logger.info(f"[STREAMING-SUBS] Enabling streaming for MKV track {actual_track_number}: {streaming_track.get('name')}")

                        # Enable streaming in the decoder thread (use actual MKV track number)
                        self.mvc_decoder_thread.set_subtitle_track(actual_track_number)
                        self._active_streaming_track = actual_track_number

                        # Configure SubtitleManager for streaming
                        if self._subtitle_manager:
                            # V7b++ STUTTER FIX: Connect PGS streaming signal NOW (deferred from MVC init)
                            if not getattr(self, '_pgs_streaming_connected', False):
                                if hasattr(self.mvc_decoder_thread, 'pgsDataReady'):
                                    self.mvc_decoder_thread.pgsDataReady.connect(self._subtitle_manager.on_pgs_data)
                                    self._pgs_streaming_connected = True
                                    logger.info("[STREAMING-SUBS] Connected pgsDataReady signal (deferred)")

                            self._subtitle_manager.start_streaming()
                            video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                            video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                            self._subtitle_manager.set_video_dimensions(video_w, video_h)
                            self._subtitle_manager.set_enabled(True)

                            # Connect to display widget
                            display_widget = getattr(self, 'active_mvc_widget', None)
                            logger.info(f"[STREAMING-SUBS] active_mvc_widget = {display_widget}")
                            if not display_widget:
                                if hasattr(self, 'framepacking_window') and self.framepacking_window:
                                    display_widget = self.framepacking_window.display_widget
                                    logger.info(f"[STREAMING-SUBS] Using framepacking_window.display_widget = {display_widget}")
                                elif hasattr(self, 'mvc_embedded_widget'):
                                    display_widget = self.mvc_embedded_widget
                                    logger.info(f"[STREAMING-SUBS] Using mvc_embedded_widget = {display_widget}")
                            if display_widget:
                                logger.info(f"[STREAMING-SUBS] Connecting subtitle manager to {display_widget.__class__.__name__}")
                                self._connect_subtitle_to_widget(display_widget)
                            else:
                                logger.error("[STREAMING-SUBS] No display widget found for subtitle connection!")

                        # BD3D authored depth: route this PG stream's offset
                        # sequence (STN_table_SS) + the per-GOP OFMD depth to
                        # the overlay. No-ops outside BD3D (map empty).
                        try:
                            seq = (getattr(self, '_bd3d_pg_offset_map', None) or {}).get(actual_track_number)
                            if seq is not None and hasattr(self.mvc_decoder_thread, 'set_pg_offset_sequence'):
                                self.mvc_decoder_thread.set_pg_offset_sequence(seq)
                            if (not getattr(self, '_pg_depth_connected', False)
                                    and hasattr(self.mvc_decoder_thread, 'pgDepthChanged')):
                                self.mvc_decoder_thread.pgDepthChanged.connect(self._on_pg_depth_changed)
                                self._pg_depth_connected = True
                        except Exception as _e:
                            logger.warning(f"[BD3D-DEPTH] wiring skipped: {_e}")

                        # Disable MPV subtitles + any text overlay
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        self.show_3d_notification(f"Streaming: {streaming_track.get('name')}", success=True)
                        return

                    elif streaming_track and self._text_subtitle_renderer is not None \
                            and (str(streaming_track.get('codecId', '')).upper().startswith('S_TEXT')
                                 or str(streaming_track.get('codecId', '')).upper() in ('S_ASS', 'S_SSA')):
                        # ===== TEXT SUBTITLE PATH (SRT / ASS / SSA) =====
                        # mpv runs audio-only here (vid=no, vo=null) so it cannot draw
                        # text subs itself, but it still decodes the selected track on
                        # the shared audio clock: select it and paint 'sub-text' on the
                        # native overlay (same widget path as PGS).
                        self._enable_text_subtitle_track(track_id, streaming_track)
                        return

                    elif track_id == 0:
                        # Disable streaming
                        logger.info("[STREAMING-SUBS] Disabling subtitle streaming")
                        self.mvc_decoder_thread.set_subtitle_track(0)
                        self._active_streaming_track = None
                        if self._subtitle_manager:
                            self._subtitle_manager.set_enabled(False)
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        return
                # ====================================================================

                # Check if this is a PGS track in MVC mode (LEGACY: extraction path)
                # Only use extraction if streaming didn't handle it
                logger.info(f"[SUBTITLE] Streaming path did not handle track_id={track_id}, trying legacy extraction...")
                if self.mvc_mode_active and self._subtitle_manager and PGS_SUBTITLE_AVAILABLE:
                    # If PGS detection hasn't completed yet, do it synchronously now
                    if len(self._pgs_subtitle_tracks) == 0 and self._subtitle_extractor and self.current_file_path:
                        logger.info("[PGS] No PGS tracks cached, detecting synchronously...")
                        try:
                            self._pgs_subtitle_tracks = self._subtitle_extractor.detect_subtitle_tracks(self.current_file_path)
                            pgs_count = sum(1 for t in self._pgs_subtitle_tracks if t.is_pgs)
                            logger.info(f"[PGS] Synchronous detection found {pgs_count} PGS tracks")
                        except Exception as e:
                            logger.error(f"[PGS] Synchronous detection failed: {e}")

                    logger.info(f"[PGS] Looking for track_id={track_id} in {len(self._pgs_subtitle_tracks)} PGS tracks")
                    # Find if track_id corresponds to a PGS track
                    pgs_track = None
                    for pt in self._pgs_subtitle_tracks:
                        logger.info(f"[PGS]   - track_id={pt.track_id}, index={pt.index}, is_pgs={pt.is_pgs}, lang={pt.language}")
                        # Match by display track_id (1-based)
                        if pt.track_id == track_id:
                            pgs_track = pt
                            break

                    if pgs_track and pgs_track.is_pgs:
                        # Check if this track was pre-extracted at startup
                        cached_index = getattr(self, '_cached_pgs_track_index', None)
                        is_loaded = self._subtitle_manager.is_loaded if self._subtitle_manager else False
                        force_reparse = False  # Use cache for faster subtitle loading
                        logger.debug(f"[PGS] Cache check: cached={cached_index}, track={pgs_track.index}, loaded={is_loaded}")
                        if cached_index == pgs_track.index and is_loaded and not force_reparse:
                            # Use pre-extracted subtitles - instant activation!
                            logger.info(f"[PGS] Using PRE-EXTRACTED subtitles for track {track_id}")
                            video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                            video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                            self._subtitle_manager.set_video_dimensions(video_w, video_h)
                            self._subtitle_manager.set_enabled(True)
                            self._active_pgs_track_index = pgs_track.index
                            # Connect to the correct display widget (use active widget from decoder)
                            display_widget = getattr(self, 'active_mvc_widget', None)
                            if not display_widget:
                                if hasattr(self, 'framepacking_window') and self.framepacking_window:
                                    display_widget = self.framepacking_window.display_widget
                                elif hasattr(self, 'mvc_embedded_widget'):
                                    display_widget = self.mvc_embedded_widget
                            logger.debug(f"[PGS] Using display widget: {display_widget.__class__.__name__ if display_widget else 'None'}")
                            self._connect_subtitle_to_widget(display_widget)
                            self.show_3d_notification(f"Subtitles: {self._subtitle_manager.subtitle_count} cues", success=True)
                        else:
                            # Need to extract this track (not pre-extracted)
                            logger.info(f"[PGS] Track {pgs_track.index} not pre-extracted, extracting now...")
                            self._load_pgs_subtitle_track(pgs_track)
                        # Disable MPV's internal subtitles + any text overlay
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        logger.info(f"[PGS] Using PGS overlay for track {track_id}")
                        return

                # Default: Use MPV's subtitle handling
                if track_id == 0:
                    self.player.sid = 'no'
                    # Also disable PGS overlay + text overlay
                    if self._subtitle_manager:
                        self._subtitle_manager.set_enabled(False)
                        self._active_pgs_track_index = None
                    self._disable_text_subtitles()
                    logger.info("[SUBTITLE] Subtitles disabled")
                else:
                    self.player.sid = track_id
                    # Disable PGS overlay when using MPV subtitles
                    if self._subtitle_manager:
                        self._subtitle_manager.set_enabled(False)
                        self._active_pgs_track_index = None
                    if self.mvc_mode_active and self._text_subtitle_renderer is not None:
                        # edge264/native-renderer playback without a streaming track
                        # list (e.g. MP4 via lavf demuxer): mpv is audio-only and
                        # cannot draw its subs — mirror them onto the native overlay.
                        # Here the combo was filled from mpv's track-list, so
                        # track_id IS the mpv sid.
                        self._activate_text_overlay(track_id)
                    else:
                        self._disable_text_subtitles()
                    logger.info(f"[SUBTITLE] track changed: ID {track_id}")
            except Exception as e:
                print(f"Error changing subtitle track: {e}")

    def _load_pgs_subtitle_track(self, pgs_track):
        """Load a PGS subtitle track for MVC overlay rendering (async)."""
        if not self._subtitle_extractor or not self._subtitle_manager:
            logger.warning("[PGS] Missing extractor or manager, cannot load subtitle")
            return

        # Show notification that extraction is starting (can take 1-2 minutes)
        self.show_3d_notification("Extracting subtitles (1-2 min)...", success=True)
        logger.info(f"[PGS] Starting extraction for track {pgs_track.index}")

        # Run extraction in background thread to avoid freezing UI
        import threading
        import time as _time
        extraction_start = _time.time()

        def extract_and_load():
            try:
                logger.info(f"[PGS] Extracting track {pgs_track.index} from {self.current_file_path}")
                logger.info("[PGS] This may take 1-2 minutes for large files...")

                # Extract PGS data to temp file
                sup_path = self._subtitle_extractor.extract_pgs_track(
                    self.current_file_path,
                    pgs_track.index
                )

                elapsed = _time.time() - extraction_start
                logger.info(f"[PGS] Extraction completed in {elapsed:.1f}s, result: {sup_path}")

                # Schedule loading on main thread via signal (thread-safe)
                if sup_path:
                    self.pgs_load_complete.emit(sup_path, pgs_track.index)
                else:
                    logger.error("[PGS] Extraction returned None")
                    self.pgs_notification.emit("Subtitle extraction failed", False)
            except Exception as e:
                logger.error(f"[PGS] Error extracting subtitle track: {e}")
                import traceback
                traceback.print_exc()
                self.pgs_notification.emit("Subtitle error", False)

        thread = threading.Thread(target=extract_and_load, daemon=True)
        thread.start()

    def _finish_pgs_load(self, sup_path, track_index):
        """Load PGS file in background thread after extraction."""
        logger.info(f"[PGS] _finish_pgs_load called with {sup_path}")
        self.show_3d_notification("Parsing subtitles...", success=True)

        import threading
        def parse_pgs():
            try:
                logger.info(f"[PGS] Parsing subtitle file: {sup_path}")
                success = self._subtitle_manager.load_subtitle_file(sup_path)
                logger.info(f"[PGS] Parse result: {success}")
                # Use signal for thread-safe callback to main thread
                self.pgs_parse_complete.emit(success, track_index)
            except Exception as e:
                logger.error(f"[PGS] Error parsing subtitle file: {e}")
                import traceback
                traceback.print_exc()
                self.pgs_notification.emit("Parse error", False)

        thread = threading.Thread(target=parse_pgs, daemon=True)
        thread.start()

    def _on_pgs_parsed(self, success, track_index):
        """Called on main thread when PGS parsing completes."""
        logger.info(f"[PGS] _on_pgs_parsed called: success={success}, track_index={track_index}")
        try:
            if success:
                # Set video dimensions for coordinate normalization
                video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                logger.info(f"[PGS] Setting video dimensions: {video_w}x{video_h}")
                self._subtitle_manager.set_video_dimensions(video_w, video_h)
                self._subtitle_manager.set_enabled(True)
                self._active_pgs_track_index = track_index
                count = self._subtitle_manager.subtitle_count

                # Connect to the correct display widget (use active widget from decoder)
                display_widget = getattr(self, 'active_mvc_widget', None)
                if not display_widget:
                    if hasattr(self, 'framepacking_window') and self.framepacking_window:
                        display_widget = self.framepacking_window.display_widget
                    elif hasattr(self, 'mvc_embedded_widget'):
                        display_widget = self.mvc_embedded_widget
                logger.debug(f"[PGS] Using display widget: {display_widget.__class__.__name__ if display_widget else 'None'}")
                self._connect_subtitle_to_widget(display_widget)

                logger.info(f"[PGS] Loaded {count} subtitle cues")
                self.show_3d_notification(f"Subtitles: {count} cues", success=True)
            else:
                logger.error(f"[PGS] Failed to load subtitle track {track_index}")
                self.show_3d_notification("Failed to parse subtitles", success=False)
        except Exception as e:
            logger.error(f"[PGS] Error finishing subtitle load: {e}")
            import traceback
            traceback.print_exc()

    def _connect_subtitle_to_widget(self, widget=None):
        """Connect SubtitleManager signals to EVERY active MVC display widget — the embedded
        2D view in the main window AND the separate 3D FramePack window — so PGS subtitles
        appear on both, not only the active one (in 3D mode the embedded 2D view stays
        visible for sync, so it needs the overlay too)."""
        if not self._subtitle_manager:
            return

        try:
            # Gather every display widget that can render a subtitle overlay (dedup, keep order)
            widgets = []
            for w in (getattr(self, 'mvc_embedded_widget', None),
                      getattr(getattr(self, 'framepacking_window', None), 'display_widget', None),
                      widget):
                if (w is not None and w not in widgets
                        and hasattr(w, 'set_subtitle') and hasattr(w, 'clear_subtitle')):
                    widgets.append(w)
            if not widgets:
                logger.warning("[PGS] No display widget with subtitle methods to connect")
                return

            # Skip if already connected to exactly this set
            if getattr(self, '_subtitle_connected_widgets', None) == widgets:
                return

            # Drop any previous connections, then connect every gathered widget
            try:
                self._subtitle_manager.subtitle_changed.disconnect()
                self._subtitle_manager.subtitle_cleared.disconnect()
            except (TypeError, RuntimeError):
                pass

            def make_setter(target_widget):
                def setter(rgba, x, y, width, height, vw, vh, disparity=0.0):
                    try:
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh,
                                                   disparity)
                    except TypeError:
                        # widget predating the disparity parameter
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh)
                    except Exception as e:
                        logger.error(f"[PGS] set_subtitle error on {target_widget.__class__.__name__}: {e}")
                return setter

            for w in widgets:
                self._subtitle_manager.subtitle_changed.connect(make_setter(w))
                self._subtitle_manager.subtitle_cleared.connect(w.clear_subtitle)
            self._subtitle_connected_widgets = widgets
            logger.info(f"[PGS] CONNECTED subtitles to {[w.__class__.__name__ for w in widgets]}")
        except Exception as e:
            logger.error(f"[PGS] Error connecting subtitle manager: {e}")

    # ========== TEXT SUBTITLE (SRT/ASS) OVERLAY ==========

    def _enable_text_subtitle_track(self, ui_index, streaming_track):
        """Route a text subtitle track (S_TEXT/UTF8, S_TEXT/ASS…) to the native overlay.

        mpv plays the same file for audio and decodes the selected text track on
        that clock even without video output; its 'sub-text' property carries the
        current cue (ASS override tags already stripped, exact show/hide timing).
        """
        logger.info(f"[TEXT-SUBS] Enabling text track ui_index={ui_index}: "
                    f"{streaming_track.get('codecId')} ({streaming_track.get('name')})")

        # Text subs don't use the demuxer streaming queue nor the PGS overlay
        if self.mvc_decoder_thread:
            self.mvc_decoder_thread.set_subtitle_track(0)
        self._active_streaming_track = None
        if self._subtitle_manager:
            self._subtitle_manager.set_enabled(False)
        self._active_pgs_track_index = None

        # Map the menu index (1-based, MKV file order) to mpv's sid. mpv numbers
        # subtitle tracks in the same file order, so the Nth entry matches sid N —
        # resolved through track-list rather than assumed, when possible.
        sid = ui_index
        try:
            subs = [t for t in (self.player.track_list or []) if t.get('type') == 'sub']
            if 0 < ui_index <= len(subs):
                sid = subs[ui_index - 1].get('id', ui_index)
        except Exception as e:
            logger.warning(f"[TEXT-SUBS] track-list sid mapping failed, using index: {e}")

        self._activate_text_overlay(sid, streaming_track.get('name'))

        # Recover the authored 3D depth of this track (per-eye duplicated cues
        # encode a parallax). Cached per (file, track); analysis is a bounded
        # ffprobe sampling pass, run off the GUI thread.
        sub_index = ui_index - 1   # ffprobe s:N == Nth subtitle track in file order
        cache_key = (self.current_file_path, sub_index)
        if cache_key in self._sub_depth_cache:
            self._text_subtitle_renderer.set_disparity(self._sub_depth_cache[cache_key])
        else:
            self._text_subtitle_renderer.set_disparity(0.0)   # flat until measured
            layout = 'tab' if (self.video_3d_info or {}).get('stereo_mode') == 'tab' else 'sbs'
            filepath = self.current_file_path

            def analyze_depth():
                try:
                    from subtitle_depth_analyzer import analyze_text_track_depth
                    d, pairs = analyze_text_track_depth(filepath, sub_index, layout)
                    self._sub_depth_cache[cache_key] = d
                    self.text_sub_depth_ready.emit(d, pairs)
                except Exception as e:
                    logger.warning(f"[SUB-DEPTH] background analysis failed: {e}")

            threading.Thread(target=analyze_depth, daemon=True,
                             name="sub-depth-analyzer").start()

    def _activate_text_overlay(self, sid, name=''):
        """Select mpv sid and mirror its 'sub-text' onto the native overlay."""
        # Register the sub-text observer once (fires on mpv's event thread; the
        # Qt signal marshals onto the main thread where QPainter is legal).
        if not self._mpv_subtext_observer_registered:
            def _on_subtext(_name, value):
                try:
                    self.mpv_sub_text_changed.emit(value or '')
                except Exception:
                    pass
            try:
                self.player.observe_property('sub-text', _on_subtext)
                self._mpv_subtext_observer_registered = True
                logger.info("[TEXT-SUBS] sub-text observer registered")
            except Exception as e:
                logger.error(f"[TEXT-SUBS] Could not observe sub-text: {e}")
                return

        try:
            self.player['sub-visibility'] = True
        except Exception:
            pass
        self.player['sid'] = sid
        self._text_sub_active = True
        self._connect_text_subtitle_to_widget()
        logger.info(f"[TEXT-SUBS] Active: mpv sid={sid}")
        if name:
            self.show_3d_notification(f"Subtitles: {name}", success=True)

    def _disable_text_subtitles(self):
        """Stop feeding the text overlay and clear any cue still on screen."""
        if self._text_sub_active:
            logger.info("[TEXT-SUBS] Disabled")
        self._text_sub_active = False
        if self._text_subtitle_renderer:
            self._text_subtitle_renderer.clear()

    @Slot(str)
    def _on_mpv_sub_text(self, text):
        """Main-thread handler for mpv 'sub-text' changes."""
        if self._text_sub_active and self._text_subtitle_renderer:
            self._text_subtitle_renderer.set_text(text)

    @Slot(float)
    def _on_pg_depth_changed(self, disparity):
        """Apply the BD3D per-GOP authored PG depth to every display widget."""
        for w in (getattr(self, 'active_mvc_widget', None),
                  getattr(self, 'mvc_embedded_widget', None),
                  getattr(getattr(self, 'framepacking_window', None), 'display_widget', None)):
            if w is not None and hasattr(w, 'set_subtitle_depth'):
                w.set_subtitle_depth(disparity)
        if not getattr(self, '_pg_depth_logged', False):
            self._pg_depth_logged = True
            logger.info(f"[BD3D-DEPTH] Authored PG depth active (first value: {disparity:+.4f})")

    @Slot(float, int)
    def _on_text_sub_depth(self, disparity, pairs):
        """Apply the measured authored subtitle depth (main thread)."""
        if not self._text_sub_active or not self._text_subtitle_renderer:
            return
        self._text_subtitle_renderer.set_disparity(disparity)
        if disparity:
            logger.info(f"[SUB-DEPTH] Applying authored depth: {disparity:+.4f} "
                        f"eye-width ({pairs} pairs)")
            self.show_3d_notification(
                f"3D subtitle depth: {disparity * 100:+.1f}% (authored)", success=True)

    def _connect_text_subtitle_to_widget(self):
        """Wire the text renderer to every active MVC display widget (same set as PGS)."""
        if not self._text_subtitle_renderer:
            return
        try:
            widgets = []
            for w in (getattr(self, 'active_mvc_widget', None),
                      getattr(self, 'mvc_embedded_widget', None),
                      getattr(getattr(self, 'framepacking_window', None), 'display_widget', None)):
                if (w is not None and w not in widgets
                        and hasattr(w, 'set_subtitle') and hasattr(w, 'clear_subtitle')):
                    widgets.append(w)
            if not widgets:
                logger.warning("[TEXT-SUBS] No display widget with subtitle methods to connect")
                return
            if self._text_sub_connected_widgets == widgets:
                return

            try:
                self._text_subtitle_renderer.subtitle_changed.disconnect()
                self._text_subtitle_renderer.subtitle_cleared.disconnect()
            except (TypeError, RuntimeError):
                pass

            def make_setter(target_widget):
                def setter(rgba, x, y, width, height, vw, vh, disparity):
                    try:
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh,
                                                   disparity)
                    except TypeError:
                        # widget predating the disparity parameter
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh)
                    except Exception as e:
                        logger.error(f"[TEXT-SUBS] set_subtitle error on {target_widget.__class__.__name__}: {e}")
                return setter

            for w in widgets:
                self._text_subtitle_renderer.subtitle_changed.connect(make_setter(w))
                self._text_subtitle_renderer.subtitle_cleared.connect(w.clear_subtitle)
            self._text_sub_connected_widgets = widgets
            logger.info(f"[TEXT-SUBS] CONNECTED to {[w.__class__.__name__ for w in widgets]}")
        except Exception as e:
            logger.error(f"[TEXT-SUBS] Error connecting text renderer: {e}")

    # =====================================================

    def load_subtitle_tracks(self):
        logger.info(f"[SUBTITLE] load_subtitle_tracks called, has_media={self.has_media}")
        if not self.has_media: return
        QTimer.singleShot(500, self._fetch_subtitle_tracks)

    def _fetch_subtitle_tracks(self):
        logger.info("[SUBTITLE] _fetch_subtitle_tracks called")
        try:
            if not self.player:
                logger.info("[SUBTITLE] player is None, returning")
                return
            track_list = self.player.track_list
            logger.info(f"[SUBTITLE] track_list has {len(track_list)} tracks")
            lang_map = self._get_clpi_lang_map()
            subtitle_tracks = []
            for track in track_list:
                logger.debug(f"[SUBTITLE] track type={track.get('type')}")
                if track.get('type') == 'sub':
                    track_id = track.get('id')
                    label = _friendly_track_label(track, 'sub', lang_map)
                    subtitle_tracks.append((track_id, label, ''))  # label is already complete
                    logger.info(f"[SUBTITLE]   Found subtitle: id={track_id}, label={label}")
            logger.info(f"[SUBTITLE] Subtitle tracks found: {len(subtitle_tracks)}")
            self.controls_overlay.update_subtitle_tracks(subtitle_tracks)

            # Also detect PGS tracks for MVC mode overlay (async to avoid blocking)
            if PGS_SUBTITLE_AVAILABLE and self._subtitle_extractor and self.current_file_path:
                import threading
                filepath = self.current_file_path
                def detect_pgs():
                    try:
                        tracks = self._subtitle_extractor.detect_subtitle_tracks(filepath)
                        self.pgs_tracks_detected.emit(tracks)
                    except Exception as e:
                        logger.error(f"[PGS] Detection error: {e}")
                threading.Thread(target=detect_pgs, daemon=True).start()
        except Exception as e:
            logger.error(f"Error fetching subtitle tracks: {e}")

    def _on_pgs_tracks_detected(self, tracks):
        """Called when PGS track detection completes."""
        self._pgs_subtitle_tracks = tracks
        pgs_count = sum(1 for t in tracks if t.is_pgs)
        if pgs_count > 0:
            logger.info(f"[PGS] Detected {pgs_count} PGS subtitle tracks")

    # ========== STREAMING SUBTITLE HANDLER ==========
    def _on_subtitle_tracks_detected(self, tracks):
        """Called when MVC decoder detects subtitle tracks (streaming mode).

        These tracks can be streamed in real-time without extraction delay.
        """
        if not tracks:
            return

        self._streaming_subtitle_tracks = tracks
        pgs_tracks = [t for t in tracks if t.get('isPGS', False)]

        # Raw Blu-ray M2TS/SSIF carries no language tag in the PMT — enrich the PID-only PGS
        # tracks with the language from the .clpi ProgramInfo (cached), so the menu shows the
        # language (e.g. "French (PID 0x1200)") instead of just the raw PID.
        try:
            lang_map = self._get_clpi_lang_map()
            if lang_map:
                seen = {}
                for i, t in enumerate(tracks):
                    lang = lang_map.get(t.get('trackNumber'), '')
                    if lang:
                        t['language'] = lang
                        base = _humanize_lang(lang) or lang.upper()
                    else:
                        base = f"Subtitle {i + 1}"
                    # label by language only — no PID; a counter disambiguates same-language dupes
                    seen[base] = seen.get(base, 0) + 1
                    t['name'] = base if seen[base] == 1 else f"{base} {seen[base]}"
        except Exception as e:
            logger.warning(f"[STREAMING-SUBS] CLPI language label skipped: {e}")

        logger.info(f"[STREAMING-SUBS] Detected {len(tracks)} subtitle tracks ({len(pgs_tracks)} PGS)")
        for t in tracks:
            logger.info(f"  - {t.get('name')} (lang={t.get('language') or '?'})")
        for t in tracks:
            logger.info(f"  - Track {t.get('trackNumber')}: {t.get('name')} (PGS={t.get('isPGS')})")

        # Update subtitle track menu in controls overlay
        if hasattr(self, 'controls_overlay') and hasattr(self.controls_overlay, 'update_subtitle_tracks_streaming'):
            self.controls_overlay.update_subtitle_tracks_streaming(tracks)

        # Show notification
        if pgs_tracks:
            self.show_3d_notification(f"Streaming: {len(pgs_tracks)} PGS tracks", success=True)
    # ================================================

    def open_file_dialog(self):
        # MKV (MVC) + Blu-ray 3D raw streams (SSIF/M2TS) via the native demuxer.
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open a video or Blu-ray ISO",
            "",
            "Video / Blu-ray (*.mkv *.mk3d *.ssif *.m2ts *.ts *.mp4 *.avi *.iso);;"
            "Blu-ray disc image (*.iso);;"
            "Blu-ray streams (*.ssif *.m2ts *.ts);;All files (*.*)"
        )
        if file_path:
            self.play_file(file_path)

    def open_disc_dialog(self):
        """Open a Blu-ray 3D disc/folder: pick a drive letter (e.g. J:\\) or a BDMV folder.
        The feature film's SSIF is auto-detected (duration-based main-title detection)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open a Blu-ray 3D — pick the drive (e.g. J:\\) or the BDMV folder", ""
        )
        if folder:
            self.play_file(folder)  # play_file auto-detects the feature SSIF on a disc/folder

    # ===================== Blu-ray → ISO archiving =====================
    def open_archive_dialog(self):
        """Open the disc-imaging dialog (configure → live throughput/ETA/animation → STOP)."""
        if getattr(self, '_archiving', False):
            return
        try:
            from disc_archiver import DiscArchiveDialog
            DiscArchiveDialog(self, parent=self).exec()
        except Exception as e:
            logger.error(f"[ARCHIVE] dialog error: {e}")
            import traceback
            traceback.print_exc()
            self.show_3d_notification(f"Archive error: {e}", success=False)

    def _mounted_iso_letters(self):
        """Drive letters of ISOs WE mounted (excluded from archiving — you already have them)."""
        out = set()
        for m in (getattr(self, '_active_iso_mount', None), getattr(self, '_pending_iso_mount', None)):
            if m and m[1]:
                L = str(m[1]).rstrip('\\').rstrip(':')[:1].upper()
                if L:
                    out.add(L)
        return out

    def _apply_preview_thumbs_policy(self, file_path):
        """Pick the thumbnail provider for this source (spec 2026-07-14).
        Physical optical → off (measured 45-120s head thrash). Player-mounted
        ISO → in-process edge264 with guardrails (a concurrent ffmpeg probe
        broke demuxer init on 2026-07-14 — the service is disarmed outside
        steady playback instead). Plain files → edge264 (H.264) or ffmpeg."""
        try:
            import disc_archiver as da
            optical = set(da.list_optical_drives())
        except Exception:
            optical = set()
        codec = None
        try:
            codec = (self.video_3d_info or {}).get('codec_name')
        except Exception:
            pass
        mode, is_optical = _decide_thumbs_mode(
            file_path, self._mounted_iso_letters(), optical, codec)
        svc = self._ensure_thumbnail_service()
        dur = 0.0
        try:
            dur = float(self.controls_overlay.time_slider.maximum()) / 1000.0
        except Exception:
            pass
        # Packed-stereo sources: thumbnails show a SINGLE eye (sbs → left half,
        # tab → top half). MVC/2D need no crop (base view is one eye already).
        _sm = None
        try:
            _sm = (self.video_3d_info or {}).get('stereo_mode')
        except Exception:
            pass
        layout = _sm if _sm in ('sbs', 'tab') else None
        svc.configure(file_path, dur, mode, optical=is_optical, layout=layout)
        slider = self.controls_overlay.time_slider
        slider.set_thumbnail_provider(svc if mode == 'edge264' else None)
        slider.set_thumbnails_allowed(mode != 'off')
        logger.info(f"[THUMB] provider={mode} optical={is_optical} for {file_path}")

    def _ensure_thumbnail_service(self):
        """Create the (single) ThumbnailService lazily and wire its lifecycle:
        disarmed around seeks via the seek queue, thumbnails routed to the
        slider cache."""
        if getattr(self, '_thumb_service', None) is None:
            from thumbnail_service import ThumbnailService
            svc = ThumbnailService()
            svc.thumbnailReady.connect(
                self.controls_overlay.time_slider._on_service_thumbnail)
            svc.start(QThread.Priority.LowPriority)
            self._thumb_service = svc
            if getattr(self, '_seek_queue', None):
                self._seek_queue.seek_started.connect(lambda _t: svc.disarm())
                self._seek_queue.seek_completed.connect(
                    lambda: QTimer.singleShot(500, svc.arm))
        return self._thumb_service

    def _is_physical_bluray(self, letter):
        """True iff `letter` is a physical optical drive holding a Blu-ray (BDMV) — i.e. not a
        mounted ISO and not a non-BD disc. This is the ONLY thing we allow imaging."""
        if not letter:
            return False
        try:
            import disc_archiver as da
            import bluray_disc
            letter = letter.upper()
            if letter in self._mounted_iso_letters():
                return False
            if letter not in da.list_optical_drives():
                return False
            return bool(bluray_disc.is_bluray_path(f"{letter}:\\"))
        except Exception:
            return False

    def _archivable_disc_drive(self):
        """Letter of the physical Blu-ray currently loaded, else None. Drives the archive
        button's enabled state so it lights up ONLY for a Blu-ray source (never MKV/ISO)."""
        if not getattr(self, 'current_file_path', None):
            return None
        d = os.path.splitdrive(os.path.abspath(self.current_file_path))[0]
        letter = d[0].upper() if d else None
        return letter if self._is_physical_bluray(letter) else None

    def _update_archive_button_state(self):
        """Enable the archive button only when a Blu-ray disc is the active source."""
        try:
            ok = self._archivable_disc_drive() is not None and not getattr(self, '_archiving', False)
            self.controls_overlay.archive_button.setEnabled(ok)
        except Exception:
            pass

    def _resolve_archive_source(self):
        """Blu-ray discs ONLY. Returns {found, ready, kind:'volume', drive, label, length, error}.
        Imaging is offered solely for a physical Blu-ray optical disc — never a mounted ISO,
        an MKV, or any other source."""
        import disc_archiver as da

        def vol(letter):
            info = da.probe_volume(letter)
            return {"found": True, "ready": bool(info.get("ok")), "kind": "volume",
                    "drive": letter, "label": info.get("label", ""),
                    "length": info.get("length", 0), "iso_path": "",
                    "error": info.get("error", "")}

        # the disc currently playing, if it's a physical Blu-ray
        cur = self._archivable_disc_drive()
        if cur:
            return vol(cur)
        # otherwise a physical Blu-ray sitting in a drive (with ready media)
        bd = [c for c in da.list_optical_drives() if self._is_physical_bluray(c)]
        for c in bd:
            r = vol(c)
            if r["ready"]:
                return r
        if bd:
            return vol(bd[0])
        return {"found": False, "ready": False, "kind": "volume", "drive": "", "label": "",
                "length": 0, "iso_path": "",
                "error": "ISO saving is only possible from a Blu-ray disc."}

    def _begin_archive_lock(self):
        """Lock playback while imaging: stop the player (releases disc handles) and disable
        the transport so the drive is dedicated to a clean sequential read."""
        self._archiving = True
        try:
            if getattr(self, 'has_media', False):
                self.stop_playback()
        except Exception as e:
            logger.warning(f"[ARCHIVE] stop during lock: {e}")
        for attr in ('play_pause_button', 'archive_button', 'skip_back_button', 'skip_forward_button'):
            try:
                getattr(self.controls_overlay, attr).setEnabled(False)
            except Exception:
                pass
        logger.info("[ARCHIVE] playback locked for disc imaging")

    def _end_archive_lock(self):
        """Release the playback lock after imaging finishes/cancels."""
        self._archiving = False
        for attr in ('play_pause_button', 'skip_back_button', 'skip_forward_button'):
            try:
                getattr(self.controls_overlay, attr).setEnabled(True)
            except Exception:
                pass
        self._update_archive_button_state()
        logger.info("[ARCHIVE] playback lock released")

    # ===================== Audio VU meter =====================
    def _ensure_vu_af(self):
        """Attach the astats audio filter to mpv (label 'vu') if absent, so
        af-metadata/vu exposes per-channel RMS/peak. Idempotent + self-healing."""
        try:
            chain = self.player._get_property('af') or []
            if not any(isinstance(f, dict) and f.get('label') == 'vu' for f in chain):
                self.player.command('af', 'add', '@vu:lavfi=[astats=metadata=1:reset=1]')
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

    def _poll_audio_levels(self):
        """Drive the VU meter from mpv's real-time audio levels (~30 Hz)."""
        vu = getattr(getattr(self, 'controls_overlay', None), 'vu_meter', None)
        if vu is None:
            return
        p = self.player
        if p is None or not getattr(self, 'has_media', False) or getattr(self, '_archiving', False):
            vu.set_levels(0.0, 0.0)
            return
        # 0xe24c4a02 HARDENING (2026-07-14, crash_log.txt): this 30Hz tick made a
        # BLOCKING p.pause read on the GUI thread and crashed inside
        # mpv_get_property while mpv rebuilt its audio chain (MVC init window).
        # Rule: no synchronous mpv reads from the GUI thread during loads,
        # transitions or seeks — and `pause` comes from the observer cache,
        # never from a property read.
        if getattr(self, '_mpv_transition_in_progress', False) or getattr(self, '_is_loading_file', False):
            vu.set_levels(0.0, 0.0)
            return
        sq = getattr(self, '_seek_queue', None)
        if sq is not None and sq.is_busy():
            return                              # keep last levels through a seek
        if getattr(self, '_mpv_pause_cache', False):  # paused → bars fall to silence
            vu.set_levels(0.0, 0.0)
            return
        try:
            md = p._get_property('af-metadata/vu')
        except Exception:
            md = None
            self._ensure_vu_af()                # filter missing (new file / new mpv) → (re)attach
        if not md:
            vu.set_levels(0.0, 0.0)
            return
        rl = self._db_to_unit(md.get('lavfi.astats.1.RMS_level'))
        rr = self._db_to_unit(md.get('lavfi.astats.2.RMS_level', md.get('lavfi.astats.1.RMS_level')))
        pl = self._db_to_unit(md.get('lavfi.astats.1.Peak_level'))
        pr = self._db_to_unit(md.get('lavfi.astats.2.Peak_level', md.get('lavfi.astats.1.Peak_level')))
        vu.set_levels(max(rl, rr), max(pl, pr))   # overall level (+ transient peak)

    def dragEnterEvent(self, event):
        """Accept drag of files/folders (including a Blu-ray drive or BDMV folder)."""
        try:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
        except Exception:
            pass

    def dropEvent(self, event):
        """Play a dropped file, or auto-detect the 3D feature from a dropped folder/disc."""
        try:
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    event.acceptProposedAction()
                    self.play_file(path)  # smart: file, folder, drive root, or BDMV
        except Exception as e:
            logger.warning(f"[DROP] {e}")

    def resizeEvent(self, event):
        """Repositions overlays on window resize."""
        super().resizeEvent(event)
        self._update_overlays_geometry()

        # Fix: Safely handle MPV resize commands
        if self.player:
            try:
                self.player.command_async('auto', ['set', 'video-zoom', 0])
                self.player.command_async('auto', ['set', 'video-pan-x', 0])
                self.player.command_async('auto', ['set', 'video-pan-y', 0])
            except Exception:
                pass

    def moveEvent(self, event):
        """Repositions overlays on window move."""
        super().moveEvent(event)
        self._update_overlays_geometry()

    def _update_overlays_geometry(self):
        """Updates the geometry of all floating overlays."""
        if not self.isVisible(): return

        # Calculate global geometry for the video area
        # We want overlays to cover the video_container
        global_pos = self.video_container.mapToGlobal(QPoint(0, 0))
        w = self.video_container.width()
        h = self.video_container.height()

        # Info Overlay (Full Screen)
        self.info_overlay.move(global_pos)
        self.info_overlay.resize(w, h)

        # Loading Overlay (Full Screen)
        self.loading_overlay.move(global_pos)
        self.loading_overlay.resize(w, h)

        # Controls Overlay (Bottom Floating)
        ctrl_h = self.controls_overlay.sizeHint().height()
        margin_bottom = 20
        margin_side = 10

        # Center the controls horizontally with margins
        ctrl_w = max(600, w - (margin_side * 2))
        ctrl_x = global_pos.x() + (w - ctrl_w) // 2
        ctrl_y = global_pos.y() + h - ctrl_h - margin_bottom

        self.controls_overlay.move(ctrl_x, ctrl_y)
        self.controls_overlay.resize(ctrl_w, ctrl_h)

        # Monitoring Overlay (Top Right)
        self._update_monitoring_overlay_geometry()
        self._update_metrics_overlay_geometry()

        # Ensure visibility/z-order
        if self.controls_overlay.isVisible():
            self.controls_overlay.raise_()
        if self.info_overlay.isVisible():
            self.info_overlay.raise_()

    def mouseMoveEvent(self, event):
        """Nav bar: any movement over the window counts as activity (moves over child
        widgets like the D3D11 video are caught globally by _nav_poll_tick)."""
        self._mark_activity()
        super().mouseMoveEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                self.controls_overlay.hide()
                self.info_overlay.hide()
                self.monitoring_overlay.hide()
            elif not self.isHidden():
                # Restore visibility based on state
                if self.has_media:
                    self.show_controls()
                    # V15: Start inactivity timer if playing
                    if self.is_playing:
                        self._mouse_inactivity_timer.start()
                else:
                    self.info_overlay.show()
                self._refresh_monitoring_overlay()
        super().changeEvent(event)

    def enterEvent(self, event):
        """V15: Mouse entered the main window - cancel hide timer."""
        self._mouse_outside_window = False
        self._mouse_inactivity_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """V15: Mouse left the main window - start 3s timer to hide controls."""
        self._mouse_outside_window = True
        if self.is_playing and self.controls_overlay.isVisible():
            # Check if mouse is over a popup (ComboBox dropdown)
            audio_combo = self.controls_overlay.audio_track_combo
            subtitle_combo = self.controls_overlay.subtitle_track_combo
            stereo_combo = self.controls_overlay.stereo_mode_combo

            for combo in [audio_combo, subtitle_combo, stereo_combo]:
                if combo.view().isVisible():
                    # Mouse is in a popup - don't start hide timer
                    return

            # Start 3s timer to hide controls
            self._mouse_inactivity_timer.start()
        super().leaveEvent(event)

    # --- V60: tiny per-install settings store (JSON in the user profile) ---

    def _app_settings_path(self):
        return os.path.join(os.path.expanduser('~'), '.sylc3d_player.json')

    def _load_app_settings(self):
        try:
            with open(self._app_settings_path(), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_app_settings(self):
        try:
            with open(self._app_settings_path(), 'w', encoding='utf-8') as f:
                json.dump(self._app_settings, f, indent=2)
        except Exception as e:
            logger.warning(f"[SETTINGS] Could not save {self._app_settings_path()}: {e}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        # Space -> Play/Pause
        if key == Qt.Key.Key_Space:
            self.toggle_play()
            event.accept()
            return

        # Escape -> Exit Fullscreen
        if key == Qt.Key.Key_Escape:
            if self._is_fake_fullscreen:
                self.toggle_fullscreen()
            event.accept()
            return

        # [ / ] -> adjust A/V sync offset live (delay video to match the heard audio)
        if key in (Qt.Key.Key_BracketRight, Qt.Key.Key_BracketLeft):
            if self.mvc_decoder_thread and self.mvc_mode_active and hasattr(self.mvc_decoder_thread, 'adjust_av_offset'):
                delta = 0.05 if key == Qt.Key.Key_BracketRight else -0.05
                off = self.mvc_decoder_thread.adjust_av_offset(delta)
                # V60: persist the trim so every future decoder thread starts with it
                self._app_settings['av_sync_offset_s'] = off
                self._save_app_settings()
                if off >= 0:
                    self.show_3d_notification(f"A/V sync — video delayed by {off*1000:.0f} ms", success=True)
                else:
                    self.show_3d_notification(f"A/V sync — video advanced by {-off*1000:.0f} ms", success=True)
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        if getattr(self, '_thumb_service', None):
            try:
                self._thumb_service.shutdown()
            except Exception:
                pass
        self._stop_mvc_decoder()
        # Release any Blu-ray ISO we mounted, so no phantom drive is left behind.
        try:
            import bluray_disc
            for m in (getattr(self, '_active_iso_mount', None),
                      getattr(self, '_pending_iso_mount', None)):
                if m:
                    bluray_disc.dismount_iso(m[0])
            self._active_iso_mount = None
            self._pending_iso_mount = None
        except Exception:
            pass
        self.controls_overlay.close()
        self.info_overlay.close()
        self.loading_overlay.close()
        self.monitoring_overlay.close()
        self.metrics_overlay.close()
        if self.framepacking_window:
            self.framepacking_window.close()
        super().closeEvent(event)

    def eventFilter(self, watched, event):
        # V15: Handle combo popup visibility changes
        if watched.property("is_combo_popup") and event.type() == QEvent.Type.Hide:
            self._on_combo_popup_closed()
            return super().eventFilter(watched, event)

        if watched is self.controls_overlay:
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
                # V15: Mouse is on controls - stop inactivity timer
                self._mouse_inactivity_timer.stop()
                # Also stop old timer for compatibility
                self._ensure_controls_timer_initialized()
                self.controls_hide_timer.stop()
            elif event.type() == QEvent.Type.Leave and self.is_playing:
                # V15: Mouse left controls overlay
                # Check if it went to a popup (ComboBox dropdown)
                audio_combo = self.controls_overlay.audio_track_combo
                subtitle_combo = self.controls_overlay.subtitle_track_combo
                stereo_combo = self.controls_overlay.stereo_mode_combo

                for combo in [audio_combo, subtitle_combo, stereo_combo]:
                    if combo.view().isVisible():
                        # Mouse is in a popup - don't start hide timer
                        return super().eventFilter(watched, event)

                # Mouse left to main window area - start inactivity timer
                if not self._mouse_outside_window:
                    self._mouse_inactivity_timer.start()
        return super().eventFilter(watched, event)

    def _update_metrics_overlay_geometry(self):
        if not hasattr(self, 'metrics_overlay') or not self.metrics_overlay.parent():
            return
        self.metrics_overlay.adjustSize()
        margin = 20
        self.metrics_overlay.move(margin, margin)

    def _update_monitoring_overlay_geometry(self):
        if not hasattr(self, 'monitoring_overlay'):
            return
        margin = 20
        width = self.monitoring_overlay.width()
        height = self.monitoring_overlay.height()

        # Convert local position to global screen coordinates for the Tool window
        local_pos = QPoint(self.width() - width - margin, margin)
        global_pos = self.mapToGlobal(local_pos)

        self.monitoring_overlay.move(global_pos)

    def _dismount_pending_iso(self):
        """Dismount an ISO we just mounted but couldn't use (best-effort)."""
        try:
            import bluray_disc
            pending = getattr(self, '_pending_iso_mount', None)
            if pending:
                logger.info(f"[DISC] Dismounting unused ISO: {pending[0]}")
                bluray_disc.dismount_iso(pending[0])
                self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] Dismount (pending) failed: {e}")

    def _dismount_isos_after_stop(self):
        """STOP released all readers (decoder, mpv terminate, thumbnail service
        release_file) — return the mounted ISO(s) to the system."""
        try:
            import bluray_disc
            for m in (getattr(self, '_active_iso_mount', None),
                      getattr(self, '_pending_iso_mount', None)):
                if m:
                    logger.info(f"[DISC] Dismounting ISO on stop: {m[0]}")
                    bluray_disc.dismount_iso(m[0])
            self._active_iso_mount = None
            self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] Dismount on stop failed: {e}")

    def _promote_iso_mount(self):
        """After the previous file's handles are released: dismount the previously
        active ISO (when switching away from it) and promote the just-mounted one to
        active. Best-effort — a stuck mount must never block playback."""
        try:
            import bluray_disc
            pending = getattr(self, '_pending_iso_mount', None)
            active = getattr(self, '_active_iso_mount', None)
            if active and (not pending or active[0] != pending[0]):
                # NEVER dismount the ISO that hosts the file being (re)loaded —
                # the fresh-mount retry replays the SAME D:\...ssif path with no
                # pending mount, and dismounting here killed the retry
                # ("Failed to open SSIF file", measured 2026-07-14 on Avatar).
                cur = getattr(self, 'current_file_path', None) or ''
                mnt_letter = str(active[1] or '').rstrip('\\').rstrip(':')[:1].upper()
                cur_drive = os.path.splitdrive(os.path.abspath(cur))[0] if cur else ''
                cur_letter = cur_drive[:1].upper() if cur_drive else ''
                if mnt_letter and cur_letter == mnt_letter:
                    logger.info(f"[DISC] Keeping ISO mounted (current file lives on {mnt_letter}:)")
                    self._active_iso_mount = active
                    self._pending_iso_mount = None
                    return
                logger.info(f"[DISC] Dismounting previous ISO: {active[0]}")
                bluray_disc.dismount_iso(active[0])
                active = None
            self._active_iso_mount = pending or active
            self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] ISO promote failed: {e}")

    def play_file(self, file_path):
        """Loads and starts playing a video file - V7a Enhanced with cleanup delay.

        Also accepts a Blu-ray 3D disc/folder: a drive letter (J:\\), a BDMV folder,
        an index.bdmv, or any folder containing a BDMV. In that case the feature film
        ("main title") SSIF is auto-detected (duration-based, robust to decoy playlists).
        """
        if getattr(self, '_archiving', False):
            self.show_3d_notification("ISO copy in progress — playback unavailable", success=False)
            return
        # Reset multi-segment (seamless-branching) feature state for every load; set below
        # only when a disc feature spans several SSIF segments (an edl:// URI, no temp file).
        self._pending_feature_segments = None
        self._feature_edl_uri = None
        try:
            import bluray_disc
            from PySide6.QtWidgets import QApplication
            self._pending_iso_mount = None
            # Blu-ray ISO: mount it (no admin needed) and treat the mount as the disc.
            if bluray_disc.is_iso(file_path):
                try:
                    self.show_3d_notification("Mounting Blu-ray ISO…", success=True)
                    QApplication.processEvents()
                except Exception:
                    pass
                drive = bluray_disc.mount_iso(file_path)
                if drive and bluray_disc.is_bluray_path(drive):
                    self._pending_iso_mount = (file_path, drive)
                    logger.info(f"[DISC] Mounted ISO {file_path} -> {drive}")
                    file_path = drive  # detect the feature on the mounted drive
                elif drive:
                    bluray_disc.dismount_iso(file_path)
                    self.show_3d_notification("ISO has no Blu-ray (BDMV) structure", success=False)
                    return
                else:
                    self.show_3d_notification("Could not mount the ISO", success=False)
                    return
            if os.path.isdir(file_path) or bluray_disc.is_bluray_path(file_path):
                try:
                    self.show_3d_notification("Detecting the feature on disc…", success=True)
                    QApplication.processEvents()
                except Exception:
                    pass
                feat, info = bluray_disc.find_feature(file_path)
                # Freshly-mounted ISO second chance: a large UDF volume can become
                # browsable a moment after the mount readiness wait gives up —
                # one bounded retry beats dismounting a perfectly good disc.
                if not feat and getattr(self, '_pending_iso_mount', None):
                    logger.info("[DISC] No feature on first scan of fresh ISO mount — retrying in 2s")
                    time.sleep(2.0)
                    feat, info = bluray_disc.find_feature(file_path)
                if feat:
                    kind = info.get('kind')
                    mins = (info.get('duration_s') or 0) / 60.0
                    logger.info(f"[DISC] Feature ({kind}): {feat} | method={info.get('method')} "
                                f"dur={info.get('duration_s', 0):.0f}s playlist={info.get('playlist')} "
                                f"clip={info.get('clip')}")
                    label = "3D feature" if kind == 'ssif' else "2D feature"
                    self.show_3d_notification(f"{label}: {os.path.basename(feat)} ({mins:.0f} min)", success=True)
                    file_path = feat
                    # BD3D authored subtitle depth: PG PID -> offset_sequence_id
                    # from the playlist's STN_table_SS (drives the OFMD depth).
                    self._bd3d_pg_offset_map = {}
                    try:
                        _pl = info.get('playlist')
                        if _pl:
                            # find_feature reports the playlist BASENAME (e.g.
                            # '00852.mpls') — resolve it under <BDMV>/PLAYLIST.
                            if not os.path.isabs(_pl) and info.get('bdmv'):
                                _pl = os.path.join(info['bdmv'], 'PLAYLIST', _pl)
                            if os.path.isfile(_pl):
                                from bd3d_offset_metadata import parse_mpls_pg_offsets
                                self._bd3d_pg_offset_map = parse_mpls_pg_offsets(_pl)
                            else:
                                logger.warning(f"[BD3D-DEPTH] playlist not found: {_pl}")
                    except Exception as _e:
                        logger.warning(f"[BD3D-DEPTH] PG offset map unavailable: {_e}")
                    # Seamless-branching 3D feature spanning several SSIF segments: build a
                    # matching mpv EDL (continuous audio on one timeline) + remember the
                    # ordered segment sequence so the decoder plays them as one film.
                    _segs = info.get('segments') or []
                    if kind == 'ssif' and len(_segs) > 1:
                        try:
                            _uri = bluray_disc.build_feature_edl(_segs)
                            if _uri:
                                self._pending_feature_segments = _segs
                                self._feature_edl_uri = _uri
                                logger.info(f"[DISC] Multi-segment feature: {len(_segs)} SSIF segments via edl:// ({mins:.0f} min)")
                        except Exception as _e:
                            logger.warning(f"[DISC] Multi-segment EDL build failed ({_e}); first segment only")
                            self._pending_feature_segments = None
                            self._feature_edl_uri = None
                else:
                    logger.info(f"[DISC] No feature found under: {file_path}")
                    self.show_3d_notification("No playable feature found on this disc/folder", success=False)
                    self._dismount_pending_iso()
                    return
        except Exception as e:
            logger.warning(f"[DISC] BDMV detection failed: {e}")

        # If detection threw/failed and left a directory or drive root, abort cleanly
        # instead of trying to "play" a folder.
        if os.path.isdir(file_path):
            logger.warning(f"[DISC] No playable feature resolved from: {file_path}")
            self._dismount_pending_iso()
            self.show_3d_notification("No playable feature found on this disc/folder", success=False)
            return

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        # SSIF/M2TS (Blu-ray 3D raw streams) are now supported via the native demuxer.
        file_ext = os.path.splitext(file_path)[1].lower()

        if not self.player:
            print("Player not ready, retrying...")
            QTimer.singleShot(100, lambda: self.play_file(file_path))
            return

        # CRITICAL: Prevent multiple simultaneous file loads
        if hasattr(self, '_is_loading_file') and self._is_loading_file:
            logger.warning("[LOAD] File load already in progress, ignoring request")
            return

        # CRITICAL FIX V2: Reset playback ended flag when loading new file
        # This allows MPV callbacks to work normally again
        self._playback_ended = False

        # V14b: Reset MPV transition flag - new file load starts fresh
        self._mpv_transition_in_progress = False

        self._is_loading_file = True
        self._current_precise_time = 0.0 # Reset precise timeline tracker
        logger.info(f"[LOAD] Loading file: {file_path}")

        # THUMB: no thumbnail I/O may overlap the load/demuxer-init window
        if getattr(self, '_thumb_service', None):
            self._thumb_service.disarm()

        # Show loading overlay with animation (hide welcome screen to avoid text overlap)
        self.info_overlay.hide()
        self._update_overlays_geometry()
        self.loading_overlay.show_loading("Initializing playback...")

        # CRITICAL: Reset subtitles from previous file to prevent carryover
        if self._subtitle_manager:
            logger.info("[LOAD] Clearing previous subtitles")
            self._subtitle_manager.clear()
            self._subtitle_manager.set_enabled(False)
        self._cached_pgs_sup_path = None
        self._cached_pgs_track_index = None
        self._active_pgs_track_index = None
        self._pgs_subtitle_tracks = []

        # CRITICAL STABILITY: Stop MPV first to release file handles and video output
        # V61 STABILITY: if a previous stop_playback terminated mpv and its async
        # re-init hasn't fired yet, re-create it SYNCHRONOUSLY now — the whole load
        # path assumes a live player (the duplicate-init guard in _setup_mpv_player
        # makes the pending timer a no-op afterwards).
        if self.player is None:
            logger.info("[LOAD] MPV instance not alive (post-stop) — synchronous re-init")
            self._setup_mpv_player()
        if self.player:
            try:
                self.player.stop()
            except:
                pass

        # Stop decoder and wait for cleanup to complete
        self._stop_mvc_decoder()

        # The previous file's handles are now released — dismount the previously
        # active ISO (if we're switching away) and promote the just-mounted one.
        self._promote_iso_mount()

        # CRITICAL FIX: Add delay after cleanup to ensure all threads are stopped
        # and resources are released before starting new file
        logger.info("[LOAD] Waiting 500ms for cleanup to complete...")
        QTimer.singleShot(500, lambda: self._continue_play_file(file_path))

    def _continue_play_file(self, file_path):
        """Continue loading file after cleanup delay."""
        self.current_file_path = file_path
        self._edge264_consecutive_crashes = 0  # fresh source: reset edge264 crash streak
        self._update_archive_button_state()  # archive button lights up only for a Blu-ray disc

        # V7b CRITICAL FIX: Reset timeline IMMEDIATELY to prevent stale duration from previous video
        # This ensures the slider maximum() doesn't contain old values during seek calculations
        self.controls_overlay.set_duration(0)
        self.controls_overlay.time_slider.setRange(0, 1)  # Temporary range until MPV provides duration
        self.controls_overlay.time_slider.setEnabled(False)

        # STEP 1: Quick analysis to detect if MVC (DON'T start decoder yet!)
        self.loading_overlay.set_status("Analyzing 3D structure...")
        self.video_3d_info = Video3DAnalyzer.analyze_file(file_path)
        is_mvc = self.video_3d_info.get('stereo_mode') == 'mvc'
        self._mvc_file_detected = is_mvc

        logger.info(f"[LOAD] MVC detected: {is_mvc}, PGS available: {PGS_SUBTITLE_AVAILABLE}")

        # STEP 2: For MVC files, extract PGS subtitles BEFORE starting anything
        # EXCEPTION: SSIF files use streaming mode for subtitles (no pre-extraction needed)
        is_ssif = file_path.lower().endswith('.ssif')
        is_mkv = file_path.lower().endswith(('.mkv', '.mk3d'))  # mk3d = MKV 3D variant

        # ========== STREAMING SUBTITLE OPTIMIZATION ==========
        # For MKV files: Skip extraction! The MVC decoder will stream subtitles in real-time.
        # This eliminates the 2-5 minute startup delay.
        # For SSIF/M2TS: Still use extraction (streaming not yet implemented for these formats)
        if is_mkv and is_mvc:
            logger.info("[PGS] MKV detected - using streaming subtitles (no extraction delay!)")
            self._configure_and_start_playback(file_path)
            return
        # =====================================================

        if PGS_SUBTITLE_AVAILABLE and self._subtitle_extractor and is_mvc and not is_ssif:
            self._extract_pgs_at_startup(file_path)
            return  # _extract_pgs_at_startup will call _finalize_play_file_mvc when done

        # Non-MVC files: Configure 3D and continue directly
        self._configure_and_start_playback(file_path)

    def _extract_pgs_at_startup(self, file_path):
        """Extract PGS subtitles at startup before playback begins."""
        logger.info("[PGS] Starting pre-extraction of PGS subtitles...")
        self.loading_overlay.show_loading("Extracting subtitles...", progress_mode=True)

        # Clear any previously cached extraction
        self._cached_pgs_sup_path = None
        self._pgs_subtitle_tracks = []

        # Progress callback (emits Qt signal for thread-safe UI update)
        def on_progress(progress_value: float):
            self.extraction_progress.emit(progress_value)

        import threading
        def extract_thread():
            try:
                # Step 1: Detect PGS tracks
                logger.info(f"[PGS STARTUP] Detecting subtitle tracks in {file_path}")
                tracks = self._subtitle_extractor.detect_subtitle_tracks(file_path)
                pgs_tracks = [t for t in tracks if t.is_pgs]

                if not pgs_tracks:
                    logger.info("[PGS STARTUP] No PGS tracks found, skipping extraction")
                    self.pgs_extraction_complete.emit(file_path)
                    return

                # Store detected tracks
                self._pgs_subtitle_tracks = tracks

                # Step 2: Extract first PGS track (usually the main one)
                pgs_track = pgs_tracks[0]
                logger.info(f"[PGS STARTUP] Extracting track {pgs_track.index} ({pgs_track.language})...")

                import time
                start_time = time.time()

                # Pass progress callback for real-time UI updates
                sup_path = self._subtitle_extractor.extract_pgs_track(
                    file_path, pgs_track.index, progress_callback=on_progress
                )

                elapsed = time.time() - start_time
                logger.info(f"[PGS STARTUP] Extraction completed in {elapsed:.1f}s: {sup_path}")

                if sup_path:
                    # Step 3: Parse the extracted file
                    logger.info("[PGS STARTUP] Parsing subtitle file...")
                    # Note: Can't update UI directly from thread, but extraction is the main wait
                    success = self._subtitle_manager.load_subtitle_file(sup_path)

                    if success:
                        self._cached_pgs_sup_path = sup_path
                        self._cached_pgs_track_index = pgs_track.index
                        count = self._subtitle_manager.subtitle_count
                        logger.info(f"[PGS STARTUP] Loaded {count} subtitle cues, cached_track_index={pgs_track.index}, is_loaded={self._subtitle_manager.is_loaded}")
                    else:
                        logger.warning("[PGS STARTUP] Failed to parse subtitle file")
                else:
                    logger.warning("[PGS STARTUP] Extraction returned no file")

                # CRITICAL: Use Qt Signal for thread-safe callback (NOT QTimer.singleShot!)
                # QTimer.singleShot from background thread causes freezes/crashes
                logger.info("[PGS STARTUP] Emitting pgs_extraction_complete signal...")
                self.pgs_extraction_complete.emit(file_path)
                logger.info("[PGS STARTUP] Signal emitted, thread completing")

            except Exception as e:
                logger.error(f"[PGS STARTUP] Error during extraction: {e}")
                import traceback
                traceback.print_exc()
                # Continue to playback anyway via signal
                self.pgs_extraction_complete.emit(file_path)

        thread = threading.Thread(target=extract_thread, daemon=True)
        thread.start()

    def _on_extraction_progress(self, progress: float):
        """Called on main thread when extraction progress updates (via Qt Signal)."""
        self.loading_overlay.set_progress(progress)

    def _on_pgs_extraction_complete(self, file_path):
        """Called on main thread when PGS extraction is complete (via Qt Signal)."""
        logger.info(f"[PGS STARTUP] Signal received on main thread, file: {file_path}")
        if file_path == self.current_file_path:
            self._configure_and_start_playback(file_path)
        else:
            logger.warning(f"[PGS STARTUP] File path mismatch: expected {self.current_file_path}, got {file_path}")

    def _configure_and_start_playback(self, file_path):
        """Configure 3D mode and start playback (called after PGS extraction for MVC files)."""
        logger.info(f"[LOAD] _configure_and_start_playback called for {file_path}")
        self.loading_overlay.set_status("Starting playback...")
        try:
            # Configure 3D mode (this starts MVC decoder if needed)
            # video_3d_info was already set in _continue_play_file
            logger.info(f"[LOAD] video_3d_info before configure: {self.video_3d_info}")
            self.analyze_and_configure_3d(file_path)
            logger.info("[LOAD] analyze_and_configure_3d completed")

            self.has_media = True
            self._update_3d_button_state()   # 3D button enabled only for genuine 3D content
            # self.metrics_overlay.show() # Disabled to remove top-left artifact
            # Multi-segment feature: mpv plays the EDL (continuous audio across all segments
            # on one timeline); the decoder plays the matching SSIF sequence (SequenceDemuxer).
            _mpv_src = getattr(self, '_feature_edl_uri', None) or file_path
            self.player.play(_mpv_src)
            self.player.pause = True
            # V7b FIX: FORCE the timer to stay active even when paused for MVC mode
            # This lets the slider progress immediately
            if self.mvc_mode_active or getattr(self, "_mvc_file_detected", False):
                self._playback_timer.start()  # Override pause behavior
            self.update_ui_state()

            QTimer.singleShot(500, self.load_audio_tracks)
            QTimer.singleShot(500, self.load_subtitle_tracks)

            # V7b TIMELINE FIX: Sync timeline BEFORE starting playback
            # Update timeline with MPV duration and THEN start playback to ensure correct scale
            def _update_timeline_and_start_playback(retry_count=0):
                try:
                    mpv_duration = 0
                    if hasattr(self, 'player') and self.player:
                        try:
                            mpv_duration = self.player.duration
                        except:
                            pass # Property access failed

                    if mpv_duration and mpv_duration > 0 and self.current_file_path:
                        # Update BOTH duration label and slider range FIRST
                        self.controls_overlay.set_duration(mpv_duration)
                        self.controls_overlay.time_slider.set_video_file(self.current_file_path, mpv_duration)
                        logger.info(f"[TIMELINE] Updated range from MPV: {mpv_duration}s")
                        # THUMB: playback is up on this path too → arm shortly after
                        if getattr(self, '_thumb_service', None):
                            self._thumb_service.set_duration(float(mpv_duration))
                            QTimer.singleShot(1000, self._thumb_service.arm)

                        # SSIF/M2TS seek needs this duration inside the demuxer (proportional
                        # byte seek). This is where mpv's duration reliably lands at startup.
                        if getattr(self, 'mvc_decoder_thread', None):
                            try:
                                self.mvc_decoder_thread.set_media_duration(float(mpv_duration))
                            except Exception:
                                pass

                        # NOW start playback with correct timeline scale
                        # Explicitly force UI to playing state (Pause Icon) immediately
                        self.controls_overlay.set_paused(False)

                        def _safe_start():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    # V10 SSIF FIX: For MVC mode, DON'T unpause here.
                                    # Wait for decoder to emit seekIDRFound with actual start position.
                                    # This prevents audio from playing ahead of video during decoder init.
                                    if self.mvc_mode_active or getattr(self, "_mvc_file_detected", False):
                                        logger.info("[TIMELINE] MVC mode: keeping MPV paused until decoder ready")
                                        # Just set position to 0, but keep paused
                                        self.player.command_async('set', 'time-pos', '0')
                                    else:
                                        self.player.command_async('set', 'time-pos', '0')
                                        self.player.command_async('set', 'pause', 'no')
                                except:
                                    pass

                        QTimer.singleShot(50, _safe_start)
                        logger.info("[TIMELINE] Playback started with correct scale")
                        return

                    # Retry if duration is still missing (up to 8 times with shorter intervals)
                    if retry_count < 8:
                        delay = 150 if retry_count < 3 else 500  # Fast retries first, then slower
                        QTimer.singleShot(delay, lambda: _update_timeline_and_start_playback(retry_count + 1))
                    else:
                        # Fallback: start playback anyway with ffprobe duration
                        logger.warning("[TIMELINE] MPV duration not available, starting with ffprobe duration")
                        # Explicitly force UI to playing state (Pause Icon) immediately
                        self.controls_overlay.set_paused(False)
                        
                        def _safe_fallback_start():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    self.player.command_async('set', 'time-pos', '0')
                                    self.player.command_async('set', 'pause', 'no')
                                except:
                                    pass

                        QTimer.singleShot(50, _safe_fallback_start)

                except Exception as e:
                    logger.debug(f"[TIMELINE] Could not update from MPV: {e}")
                    # Fallback: start playback anyway
                    try:
                        self.controls_overlay.set_paused(False)
                        def _safe_last_resort():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    self.player.command_async('set', 'time-pos', '0')
                                    self.player.command_async('set', 'pause', 'no')
                                except:
                                    pass
                        QTimer.singleShot(50, _safe_last_resort)
                    except:
                        pass

            QTimer.singleShot(200, _update_timeline_and_start_playback)

            if self.is_3d_enabled:
                self.configure_3d_output(True, self.current_stereo_mode)

            logger.info("[LOAD] File loaded successfully")
            # Hide loading overlay with fade-out animation
            self.loading_overlay.hide_loading()
        finally:
            # Re-enable file loading
            self._is_loading_file = False

    def analyze_and_configure_3d(self, file_path):
        """Analyzes the file and automatically configures the 3D mode."""
        # Skip re-analysis if already done (e.g., in _continue_play_file for PGS extraction)
        if not self.video_3d_info:
            self.video_3d_info = Video3DAnalyzer.analyze_file(file_path)

        # SOL 1A: Set MVC flag IMMEDIATELY (before _start_mvc_decoder)
        # Allows the timer to stay active as soon as player.pause = True.
        # Packed-stereo H.264 (SBS/TAB) is also edge264-decoded, so it keeps the
        # flag set too (timeline timer / pause / 3D-button gating treat it alike).
        _sm = self.video_3d_info.get('stereo_mode')
        _cn = (self.video_3d_info.get('codec_name') or '').lower()
        _cx = (self.video_3d_info.get('container_ext') or '').lower()
        self._mvc_file_detected = (
            _sm == 'mvc'
            or (_sm in ('sbs', 'tab') and _cn == 'h264'
                and _cx in EDGE264_CONTAINERS)
        )
        self._update_3d_button_state()

        # V7b CRITICAL FIX: DO NOT update the timeline with the ffprobe duration
        # The timeline will be updated ONLY by _update_timeline_and_start_playback with MPV
        # This avoids scale conflicts between ffprobe and MPV that cause incorrect seeks

        # BUT we keep the FPS and the file name for the previews
        self._apply_preview_thumbs_policy(file_path)
        self.controls_overlay.time_slider.set_video_file(file_path, 0)  # Duration=0 for now
        fps_val = self.video_3d_info.get('fps')
        if fps_val:
            self.current_video_fps = fps_val

        # NOTE: The duration will be updated by _update_timeline_and_start_playback after MPV loads

        if self.video_3d_info.get('analysis_error'):
            self.show_3d_notification("3D analysis via ffprobe failed.", success=False)

        # Always enable 3D controls if MVC support is available
        # This allows manual override for 2D->3D conversion or misidentified files
        if MVC_SUPPORT_AVAILABLE:
            self.controls_overlay.enable_3d_controls(True)

        if self.video_3d_info['is_3d'] and self.video_3d_info['stereo_mode'] != 'none':
            stereo_mode = self.video_3d_info['stereo_mode']
            # Index mapping: 0=MVC, 1=Side-by-Side, 2=Top-Bottom
            mode_index = {'mvc': 0, 'sbs': 1, 'tab': 2}.get(stereo_mode, 0)
            self.controls_overlay.stereo_mode_combo.setCurrentIndex(mode_index)
            mode_names = {'mvc': 'MVC', 'sbs': 'Side-by-Side', 'tab': 'Top-Bottom'}
            self.show_3d_notification(f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())}", success=True,
                                      permanent=True)

            # Always start MVC decoder for MVC content (required for SSIF 2D playback)
            if stereo_mode == 'mvc':
                # V7b CRITICAL FIX: Force decoder to start at 0s, not at current MPV time
                # This prevents the "21.955s drift" bug where decoder starts at wrong timestamp
                if MVC_SUPPORT_AVAILABLE:
                    try:
                        self._start_mvc_decoder(start_time=0.0)
                    except Exception as e:
                        # edge264 first, mpv only on failure: don't crash the load —
                        # degrade to mpv and stop configuring the 3D window.
                        self._fallback_from_edge264(reason=f"MVC decoder init failed: {e}")
                        return
                else:
                    logger.warning("[MVC] MVC content detected but decoder support is unavailable; using mpv fallback.")
                    self._fallback_to_mpv_mvc()
                # V7b CRITICAL FIX: Framepacking window should ALWAYS use framepack mode
                # It's specifically designed for 1920x2205 framepack output!
                if self.framepacking_window:
                    self.framepacking_window.display_widget.set_stereo_mode('framepack')
                # Reassure the user: edge264 recognised & adapted to this 3D stream.
                self.controls_overlay.set_format_badge(self._format_badge_label())

                # 3D button starts OFF - user must manually enable 3D mode
                # (Previously auto-enabled if Nvidia 3D Vision was active)
            elif stereo_mode in ('sbs', 'tab'):
                # Packed-stereo H.264 (Full-SBS / Full-TAB): edge264 decodes EVERY
                # H.264 stream. The decoded frame carries BOTH eyes;
                # _on_mvc_frame_yuv_ready splits it into L (base eye) + R, so the
                # player drives it EXACTLY like MVC:
                #   - main window default = the BASE view (left/top eye), '2d' mode
                #   - SBS/TAB combo       = the main view's layout (L|R or L/R)
                #   - FramePack window    = the two views stacked at full resolution
                # Same containers as the 2D edge264 path; mpv fallback only on failure.
                codec = (self.video_3d_info.get('codec_name') or '').lower()
                ext = (self.video_3d_info.get('container_ext') or '').lower()
                if (MVC_SUPPORT_AVAILABLE and codec == 'h264'
                        and ext in EDGE264_CONTAINERS
                        and NATIVE_RENDER_AVAILABLE):
                    logger.info(f"[PACKED-3D] {stereo_mode.upper()} H.264 ({ext}) via edge264 (split L/R, like MVC)")
                    try:
                        self._start_mvc_decoder(start_time=0.0)
                        # Main window shows the BASE view (left/top eye) by default.
                        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
                            self.mvc_embedded_widget.set_stereo_mode('2d')
                        # FramePack window ready to stack BOTH views (shown on 3D toggle).
                        if self.framepacking_window:
                            self.framepacking_window.display_widget.set_stereo_mode('framepack')
                        # Reassure the user: edge264 recognised & adapted to this stream.
                        self.controls_overlay.set_format_badge(self._format_badge_label())
                    except Exception as e:
                        self._fallback_from_edge264(reason=f"{stereo_mode} edge264 init failed: {e}")
                        return
                else:
                    # Non-h264 packed stereo, or unsupported container -> mpv (original).
                    self._present_via_mpv_native()
        else:
            # 2D content detected
            self.controls_overlay.clear_format_badge()  # no 3D badge for 2D content
            self.show_3d_notification("2D content detected", success=True, permanent=True)

            # === 2D-via-edge264 path ===
            # For H.264 in MKV containers, route through MVCDecoderThread anyway.
            # The C++ demuxer (mvc_matroska_demuxer.cpp:214) sets hasMVC=true
            # optimistically for any single-track AVC, and the Python decoder
            # duplicates the left view when samples_mvc[0] is NULL. So a 2D H.264
            # file flows through the same pipeline, just with both eyes identical.
            # This gives us: HDR via D3D11 widget, consistent codec path, and
            # MPV stays audio-only (no more MPV vo glitches on 2D files).
            codec = (self.video_3d_info.get('codec_name') or '').lower()
            ext = (self.video_3d_info.get('container_ext') or '').lower()
            # Containers edge264 can demux: MKV/M2TS/TS via the C++ demuxer; MP4/AVI/
            # MOV/FLV/WebM/raw via lavf_h264_demuxer (task #391). Any edge264 failure
            # degrades to mpv via _fallback_from_edge264 (the 2D try/except below).
            eligible_2d = (MVC_SUPPORT_AVAILABLE
                           and codec == 'h264'
                           and ext in EDGE264_CONTAINERS
                           and NATIVE_RENDER_AVAILABLE)
            if eligible_2d:
                logger.info(f"[2D-EDGE264] Routing 2D H.264 ({ext}) through edge264 decoder")
                try:
                    self._start_mvc_decoder(start_time=0.0)
                    # Force every render target to 2D mode (left eye only,
                    # right plane upload skipped by our optimization in widget).
                    if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
                        self.mvc_embedded_widget.set_stereo_mode('2d')
                    if self.framepacking_window:
                        self.framepacking_window.display_widget.set_stereo_mode('2d')
                    self.show_3d_notification("2D edge264 decoder active", success=True, permanent=True)
                except Exception as e:
                    # edge264 first, mpv only on failure (unified fallback path).
                    self._fallback_from_edge264(reason=f"2D edge264 init failed: {e}")
            else:
                # 2D not edge264-eligible (non-h264, or non-.mkv: .mp4/.avi/.m2ts/VC-1…):
                # MPV plays it natively. If a 3D/MVC file was loaded before, MPV video was
                # disabled and the stack shows the MVC widget -> black unless we restore both.
                self._present_via_mpv_native()

    def configure_3d_output(self, enable_3d=True, stereo_mode='auto'):
        """Configures the 3D output of mpv."""
        if not self.player: return

        if not enable_3d:
            # Switch to Embedded 2D Mode
            if self.mvc_mode_active and hasattr(self, 'mvc_embedded_widget'):
                try:
                    self.framepacking_window.hide()

                    # Show embedded widget in stack
                    self.video_stack.setCurrentWidget(self.mvc_embedded_widget)
                    self.mvc_embedded_widget.set_stereo_mode('2d')

                    # Switch decoder target
                    self.active_mvc_widget = self.mvc_embedded_widget
                    if self.mvc_decoder_thread:
                        self.mvc_decoder_thread.set_display_widget(self.mvc_embedded_widget)

                    self.show_3d_notification("2D Mode (Left View)", success=True)
                except Exception as e:
                    print(f"Error switching to 2D: {e}")
                return

            self._stop_mvc_decoder()
            try:
                self.player['lavfi-complex'] = ''
                self.player['video'] = 'auto'
                self.video_stack.setCurrentWidget(self.video_widget)
            except:
                pass
            return

        # Resolve 'auto' to the actually detected mode BEFORE any branching.
        # Without this, the routing branches further down (`if stereo_mode == 'mvc':`
        # and `elif stereo_mode in ('sbs', 'tab'):`) test the literal string and
        # silently do nothing when the user has never manually picked a mode —
        # the stereo combo defaults to index 0 (MVC), which doesn't fire the
        # change signal if it was already at 0, so current_stereo_mode stays 'auto'.
        if stereo_mode == 'auto':
            detected = (self.video_3d_info.get('stereo_mode') if self.video_3d_info else None)
            if detected and detected != 'none':
                stereo_mode = detected
            else:
                stereo_mode = 'mvc'  # safe default for MVC content

        # is_sbs/is_tab are read further down in the non-MVC fallback branch
        # for native SBS/TAB notifications; is_mvc is implicit via stereo_mode.
        is_sbs = stereo_mode == 'sbs'
        is_tab = stereo_mode == 'tab'

        # Packed-stereo H.264 (SBS/TAB) is edge264-decoded but DISPLAYED as the full
        # anamorphic frame in the MAIN window (never framepack — that is MVC only).
        _detected_in = (self.video_3d_info.get('stereo_mode') if self.video_3d_info else None)
        _pcodec = (self.video_3d_info.get('codec_name') or '').lower() if self.video_3d_info else ''
        _pext = (self.video_3d_info.get('container_ext') or '').lower() if self.video_3d_info else ''
        packed_input = (_detected_in in ('sbs', 'tab') and _pcodec == 'h264'
                        and _pext in EDGE264_CONTAINERS)

        # Use the edge264 decoder for MVC content (any output) AND packed-stereo H.264.
        use_mvc_decoder = (MVC_SUPPORT_AVAILABLE and self.current_file_path and
                           (self.video_3d_info.get('stereo_mode') == 'mvc' or
                            self.video_3d_info.get('has_mvc_track') or
                            packed_input))

        if use_mvc_decoder:
            try:
                # Only start decoder if not already running
                if not self.mvc_mode_active:
                    # V7b FIX: Preserve current playback position when toggling 3D mode
                    # Use _last_ui_time (reliable) instead of _current_mpv_time (can fail and return 0)
                    current_pos = getattr(self, '_last_ui_time', 0.0) or self._current_mpv_time() or 0.0
                    logger.info(f"[3D TOGGLE] Starting MVC decoder at position: {current_pos:.3f}s")
                    self._start_mvc_decoder(start_time=current_pos)

                # Configure output based on requested mode
                if stereo_mode == 'mvc':
                    # --- Detached 3D FramePack Mode ---
                    if hasattr(self, 'mvc_embedded_widget') and self.framepacking_window:
                        # V7b SYNC: Embedded stays in 2D (left eye), Framepack window in framepack mode
                        # Both receive same frames for timing sync, but render differently
                        self.mvc_embedded_widget.set_stereo_mode('2d')
                        self.framepacking_window.display_widget.set_stereo_mode('framepack')

                        # V7b CRITICAL SYNC FIX: Keep embedded widget VISIBLE so it continues rendering!
                        # If we hide it, Qt won't render it and it will freeze on last frame = desync
                        # Show embedded widget so it keeps rendering frames for perfect sync
                        self.video_stack.setCurrentWidget(self.mvc_embedded_widget)

                        # Switch decoder target to detached widget
                        self.active_mvc_widget = self.framepacking_window.display_widget
                        if self.mvc_decoder_thread:
                            self.mvc_decoder_thread.set_display_widget(self.framepacking_window.display_widget)

                        # Connect PGS subtitles to framepacking widget
                        self._connect_subtitle_to_widget(self.framepacking_window.display_widget)
                        if self._text_sub_active:
                            self._connect_text_subtitle_to_widget()

                        self.framepacking_window.showNormal()
                        self.framepacking_window.activateWindow()

                elif stereo_mode in ('sbs', 'tab'):
                    # --- SBS/TAB Mode in MAIN WINDOW ---
                    # Packed-stereo (FSBS) reaches here too: edge264 split gives L/R,
                    # so the embedded 'sbs'/'tab' shader lays the base+right eye out
                    # as the requested main-view layout. (combo 'mvc' -> FramePack above.)
                    # User preference: SBS/TAB displays in main window, only MVC uses detached window
                    if hasattr(self, 'mvc_embedded_widget'):
                        # Hide framepacking window if visible
                        if self.framepacking_window and self.framepacking_window.isVisible():
                            self.framepacking_window.hide()

                        # Configure embedded widget for SBS/TAB rendering
                        self.mvc_embedded_widget.set_stereo_mode(stereo_mode)
                        self.video_stack.setCurrentWidget(self.mvc_embedded_widget)

                        # Switch decoder target to embedded widget
                        self.active_mvc_widget = self.mvc_embedded_widget
                        if self.mvc_decoder_thread:
                            self.mvc_decoder_thread.set_display_widget(self.mvc_embedded_widget)

                        # Connect PGS subtitles to embedded widget
                        self._connect_subtitle_to_widget(self.mvc_embedded_widget)
                        if self._text_sub_active:
                            self._connect_text_subtitle_to_widget()

                        self.show_3d_notification(f"3D Mode: {stereo_mode.upper()} (Main Window)", success=True)

            except Exception as e:
                print(f"Error configuring MVC decoder: {e}")
                self._fallback_to_mpv_mvc()
        else:
            # --- Native SBS/TAB files (non-MVC) - use MPV in main window ---
            if is_sbs or is_tab:
                self._restore_mpv_video_output()
                self.video_stack.setCurrentWidget(self.video_widget)
                # Hide framepacking window if visible
                if self.framepacking_window and self.framepacking_window.isVisible():
                    self.framepacking_window.hide()
                mode_name = "Side-by-Side" if is_sbs else "Top-Bottom"
                self.show_3d_notification(f"3D Mode: {mode_name} (Native)", success=True)
            else:
                self._fallback_to_mpv_mvc()

    def _make_display_widget(self):
        """Create the video display widget — the native C++ D3D11 renderer, the SOLE
        render path since the Directive 2 cutover (Qt RHI removed). It self-queries
        the display SDR white level for HDR. If it can't be created, the caller's
        try/except degrades to mpv (#388)."""
        from native_renderer.native_framepack_widget import NativeFramepackWidget
        logger.info("[RENDER] display widget = NativeFramepackWidget (C++ D3D11)")
        return NativeFramepackWidget()

    def _start_mvc_decoder(self, start_time=None):
        if getattr(self, "_mvc_restarting", False):
            logger.info("[MVC INIT] Skipped: _mvc_restarting is True (init in progress)")
            return
        # V33j FIX: Also check if decoder is already running - no need to restart
        if self.mvc_mode_active and self.mvc_decoder_thread and self.mvc_decoder_thread.isRunning():
            logger.info("[MVC INIT] Skipped: decoder already running")
            return
        self._mvc_restarting = True
        print(f"[MVC INIT] V33j: Starting decoder (start_time={start_time})")
        if not MVC_SUPPORT_AVAILABLE or not NATIVE_RENDER_AVAILABLE:
            logger.warning("[MVC] Decoder start requested but MVC support is unavailable. Falling back to mpv.")
            self._mvc_restarting = False
            self._fallback_to_mpv_mvc()
            return
        requested_start = self._current_mpv_time() if start_time is None else start_time
        actual_start_time = float(requested_start or 0.0)

        # Store the start position for audio synchronization
        self._decoder_start_position = actual_start_time
        self._sync_adjustment_count = 0  # Reset the counter
        # V7b FIX: Reset timeline trackers to ensure cursor movement
        self._last_mvc_timestamp = actual_start_time
        self._current_precise_time = actual_start_time
        print(f"[SYNC] Decoder start position: {actual_start_time:.3f}s")

        self._stop_mvc_decoder()

        print(f"[MVC INIT] Starting MVC decoder initialization")

        # SSIF SEEK-FREEZE FIX: in MVC mode MPV is AUDIO-ONLY (video is decoded by the
        # demuxer+edge264). The global config gives MPV a 20s / 2 GB read-ahead — fine for
        # 2D, but on a physical Blu-ray it makes MPV pre-read tens of MB of the 45 GB .ssif,
        # which fights the video demuxer for the single optical head on every seek (→ 10-20s
        # freezes). Audio needs only a small buffer, so shrink MPV's cache here (restored to
        # the generous defaults for 2D playback in _present_via_mpv_native / on 2D load).
        if self.player:
            try:
                # Modest, CHUNKED read-ahead: MPV (audio-only here) reads the disc in a few
                # 16 MB sequential chunks (not 2 GB of constant pre-read, and NOT cache=no which
                # made it do tiny head-seeking reads) so it shares the optical head with the
                # video demuxer with minimal thrash. 2D playback keeps the generous default.
                self.player['cache'] = 'yes'
                self.player['demuxer-readahead-secs'] = 1
                self.player['demuxer-max-bytes'] = '16MiB'
                self.player['demuxer-max-back-bytes'] = '8MiB'
                logger.info("[MVC INIT] MPV cache set to modest chunked read-ahead for MVC mode (anti seek-freeze)")
            except Exception as e:
                logger.warning(f"[MVC INIT] Could not adjust MPV cache: {e}")

        # Demuxer initialization moved to thread to avoid blocking GUI

        # REMOVED: Do not seek MPV here. 
        # We rely on _on_frame_timestamp to sync MPV to the exact IDR timestamp 
        # of the first decoded frame. This prevents race conditions and double-seeks.
        # try:
        #    if self.player:
        #        target_seek = actual_start_time
        #        self.player.seek(target_seek, 'absolute')
        #        print(f"[MVC INIT] Seeked mpv to {target_seek}")
        # except Exception as e:
        #    print(f"Could not seek mpv: {e}")

        if not self.shared_buffer:
            raise RuntimeError("Shared memory buffer not allocated.")

        # GPU YUV->RGB + frame_struct active
        USE_GPU_YUV_CONVERSION = True
        STORE_FRAME_STRUCT_FOR_GPU = True

        logger.info(f"[MVC] Initializing decoder (GPU YUV Conversion: {USE_GPU_YUV_CONVERSION})")

        mpv_video_disabled = False
        decoder_started = False

        try:
            # 1. Prepare Embedded Widget (for 2D)
            if not hasattr(self, 'mvc_embedded_widget'):
                self.mvc_embedded_widget = self._make_display_widget()

            # Ensure it's in the stack
            if self.video_stack.indexOf(self.mvc_embedded_widget) == -1:
                self.video_stack.addWidget(self.mvc_embedded_widget)

            # 2. Prepare Detached Window (for 3D FramePack)
            if not self.framepacking_window:
                # Directive 2: the detached 3D window is rendered by the native C++
                # D3D11 renderer (sole render path; the Qt RHI widget was removed).
                # The widget self-queries the display SDR white level (HDR).
                _dw = self._make_display_widget()
                _dw.set_stereo_mode('framepack')
                self.framepacking_window = Framepacking3DWindow(
                    parent=None,
                    use_yuv_shader=USE_GPU_YUV_CONVERSION,
                    display_widget=_dw
                )
                self.framepacking_window.visibilityChanged.connect(self._on_framepacking_visibility_changed)

            # 3. Initial State: 2D Embedded (Show MVC widget in stack)
            if self.mvc_embedded_widget.parent() != self.video_stack_container:
                # If it was detached, bring it back
                self.video_stack.addWidget(self.mvc_embedded_widget)

            self.video_stack.setCurrentWidget(self.mvc_embedded_widget)
            self.mvc_embedded_widget.set_stereo_mode('2d')
            self.active_mvc_widget = self.mvc_embedded_widget

            # V57 BLACK-SCREEN-ON-RELOAD FIX: _stop_mvc_decoder() (called above) paused
            # the REUSED display widgets (pause_rendering → _rendering_paused=True, which
            # makes set_frame_yuv_views drop EVERY frame). On a fresh file load there is no
            # seek, so seekFinished → _on_mvc_seek_finished → resume_rendering never fires,
            # leaving the widget paused → black screen until the whole app is restarted
            # (which builds fresh, unpaused widgets). Explicitly resume here so the new
            # decoder's frames are actually painted.
            try:
                if hasattr(self.mvc_embedded_widget, 'resume_rendering'):
                    self.mvc_embedded_widget.resume_rendering()
                if self.framepacking_window and hasattr(self.framepacking_window.display_widget, 'resume_rendering'):
                    self.framepacking_window.display_widget.resume_rendering()
                logger.info("[MVC INIT] V57: rendering resumed for new file (un-pause reused widgets)")
            except Exception as e:
                logger.warning(f"[MVC INIT] V57 resume_rendering failed: {e}")

            # Don't connect subtitle signals at MVC init - deferred until user selects track
            # The connection will be made when user actually selects a subtitle track
            # This prevents stuttering caused by idle subtitle signal processing when window has focus
            # self._connect_subtitle_to_widget(self.mvc_embedded_widget)  # DEFERRED

            # Target the embedded widget initially
            # Demuxer is now initialized inside the thread
            self.mvc_decoder_thread = MVCDecoderThread(
                self.current_file_path,
                self.shared_buffer,
                parent=self,
                use_gpu_yuv_conversion=USE_GPU_YUV_CONVERSION,
                store_frame_struct_for_gpu=STORE_FRAME_STRUCT_FOR_GPU,
                start_position=actual_start_time,
                threads=4,  # V7b FIX: Reduced to 4 to prevent edge264 deadlock (starvation)
                media_duration=(self.player.duration or self.video_3d_info.get('duration') if self.video_3d_info else None),
                feature_segments=getattr(self, '_pending_feature_segments', None)
            )
            self.mvc_decoder_thread.set_target_fps(self._get_effective_video_fps())
            # V60: re-apply the persisted A/V sync trim (new thread = default 0.0)
            try:
                _av_trim = float(self._app_settings.get('av_sync_offset_s', 0.0))
                self.mvc_decoder_thread._av_sync_offset_s = max(-1.0, min(2.0, _av_trim))
                if _av_trim:
                    logger.info(f"[V60-SYNC] Applied persisted A/V trim: {_av_trim*1000:+.0f} ms")
            except Exception:
                pass
            # Push initial clock
            self.mvc_decoder_thread.update_audio_clock(actual_start_time)

            # Set initial target
            self.mvc_decoder_thread.set_display_widget(self.mvc_embedded_widget)

            # mpv_video_disabled = self._disable_mpv_video_output() # MOVED TO DELAYED START

            # Fallback / monitoring
            self.mvc_decoder_thread.frameDecoded.connect(self._on_mvc_frame_decoded_optimized)
            self.mvc_decoder_thread.frameReady.connect(self._on_mvc_frame_ready)
            # CRITICAL: Force QueuedConnection for cross-thread signal delivery
            from PySide6.QtCore import Qt
            self.mvc_decoder_thread.frameYUVReady.connect(self._on_mvc_frame_yuv_ready, Qt.QueuedConnection)
            logger.info("[MVC INIT] frameYUVReady connected with Qt.QueuedConnection")
            self.mvc_decoder_thread.error.connect(self._on_mvc_error)
            self.mvc_decoder_thread.fps_update.connect(self._on_mvc_fps_update)
            self.mvc_decoder_thread.decodingFinished.connect(self._on_mvc_finished)
            self.mvc_decoder_thread.stats_update.connect(self._on_mvc_stats_update)
            self.mvc_decoder_thread.decoderCrashed.connect(self._on_mvc_decoder_crashed)
            # New: Audio synchronization based on the decoder markers
            self.mvc_decoder_thread.frameTimestampReady.connect(self._on_frame_timestamp)
            # Smart Queue Signal
            self.mvc_decoder_thread.seekFinished.connect(self._on_mvc_seek_finished)
            # V7b+ SYNC FIX: Connect seekIDRFound to sync MPV audio with actual IDR timestamp
            self.mvc_decoder_thread.seekIDRFound.connect(self._on_mvc_seek_idr_found)

            # PGS Subtitle Streaming: DEFERRED INITIALIZATION
            # V7b++ STUTTER FIX: Don't initialize subtitle streaming at MVC init
            # This was causing stuttering when window had focus, even with no subtitles enabled
            # The streaming infrastructure will be set up when user actually selects a subtitle track
            self._pgs_streaming_connected = False
            if self._subtitle_manager and hasattr(self.mvc_decoder_thread, 'pgsDataReady'):
                # Store video dimensions for later use
                video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                self._subtitle_manager.set_video_dimensions(video_w, video_h)
                logger.info("[MVC INIT] PGS subtitle streaming DEFERRED (will connect on track selection)")

            # ========== STREAMING SUBTITLE TRACK DETECTION ==========
            # Connect subtitle track detection signal
            if hasattr(self.mvc_decoder_thread, 'subtitleTracksDetected'):
                self.mvc_decoder_thread.subtitleTracksDetected.connect(self._on_subtitle_tracks_detected)
                logger.info("[MVC INIT] Subtitle track detection signal connected")

            # THUMB HARVEST: decoded-frame captures (every ~10s + seek landings)
            # flow into the slider preview cache — zero extra I/O on the source.
            # Packed sources (sbs/tab): harvest crops to a single eye.
            try:
                _sm_thumb = (self.video_3d_info or {}).get('stereo_mode')
                self.mvc_decoder_thread._thumb_layout = _sm_thumb if _sm_thumb in ('sbs', 'tab') else None
                self.mvc_decoder_thread.thumbnailHarvested.connect(
                    self.controls_overlay.time_slider._on_harvest_thumbnail)
            except Exception:
                pass
            # ========================================================

            # CRITICAL: Let OpenGL initialize before starting decoding
            # Start the thread after a short delay to avoid race conditions
            print(f"[MVC INIT] Starting decoder thread in 100ms...")
            QTimer.singleShot(100, lambda: self._delayed_start_decoder(disable_mpv=True))

            # SYNC TIMER: Periodically push audio clock to decoder thread
            self._sync_timer = QTimer(self)
            self._sync_timer.setInterval(50)  # 20Hz update rate
            self._sync_timer.timeout.connect(self._update_decoder_audio_clock)
            # V9 SSIF FIX: Start sync timer immediately instead of 1000ms delay
            # The decoder needs audio clock during init for SSIF sync
            # Use a short delay (100ms) just to let MPV initialize
            QTimer.singleShot(100, self._sync_timer.start)

            self.mvc_mode_active = True
            # V7b+++ STUTTER FIX: Use set_mvc_active() to stop ALL animations
            # Previously we only set time_slider._is_mvc_active directly
            # Now we call the overlay method which also stops button animations
            self.controls_overlay.set_mvc_active(True)

            # Framepacking window is NOT shown here anymore.
            # It is only shown when 3D mode is explicitly enabled via configure_3d_output.
            # The decoder output is directed to the embedded widget in 2D mode by default.

            print(f"[MVC INIT] Framepacking window shown: visible={self.framepacking_window.isVisible()}")

            self.monitoring_overlay.update_window_state(self.framepacking_window.isVisible())
            self._framepacking_visible = self.framepacking_window.isVisible()
            self._refresh_monitoring_overlay()

            # Force one frame delivery so shader uploads valid texture data before glasses prompt
            self.mvc_decoder_thread.frameReady.emit()

            # V7c FIX: Hide loading overlay when MVC decoder is ready
            if hasattr(self, 'loading_overlay') and self.loading_overlay:
                self.loading_overlay.hide_loading()
                print("[MVC INIT] Loading overlay hidden")

            self.show_3d_notification(
                "Edge264 MVC Decoder Active - Put on your 3D glasses",
                success=True,
                permanent=True
            )
        except Exception:
            if decoder_started:
                self._stop_mvc_decoder()
            else:
                if mpv_video_disabled:
                    self._restore_mpv_video_output()
                if self.mvc_decoder_thread:
                    self.mvc_decoder_thread = None
                if self.demuxer:
                    try:
                        self.demuxer.close()
                    except Exception:
                        pass
                    self.demuxer = None
                self.mvc_mode_active = False
            raise
        finally:
            # V33j FIX: Do NOT reset _mvc_restarting here!
            # The flag must stay True until _delayed_start_decoder completes.
            # This prevents race conditions where a second _start_mvc_decoder call
            # could kill the decoder during the 100ms delay before it actually starts.
            # _mvc_restarting is reset in _delayed_start_decoder after thread.start()
            pass

    def _on_mvc_decoder_crashed(self):
        """Slot triggered when the decoder thread signals an unrecoverable crash.

        edge264 restart-on-crash is *intentional* resilience against transient
        source corruption (historical root cause = a flaky optical drive). We
        keep that recovery, but cap consecutive crashes that produce no good
        frame in between: a persistently unusable stream then degrades to mpv
        instead of looping forever.
        """
        if not self.mvc_mode_active:
            return  # Avoid restart loops if we already exited MVC mode

        # Get the current audio/video time from the main player to resume from that point
        resume_time = self._current_mpv_time()

        self._edge264_consecutive_crashes = getattr(self, '_edge264_consecutive_crashes', 0) + 1
        cap = getattr(self, '_EDGE264_CRASH_CAP', 3)
        if self._edge264_consecutive_crashes > cap:
            logger.error(f"[PLAYER] edge264 crashed {self._edge264_consecutive_crashes}x "
                         f"consecutively (cap {cap}) with no recovery — degrading to mpv.")
            self._fallback_from_edge264(
                reason=f"{self._edge264_consecutive_crashes} consecutive crashes")
            return

        # The _start_mvc_decoder method already handles stopping the old thread.
        # We call it again to create a fresh decoder instance.
        # A short delay is added to prevent rapid-fire crash loops if the source is persistently corrupt.
        # Increased to 500ms to allow MPV internals to stabilize (fixes 0xe24c4a02 exception).
        logger.warning(f"[PLAYER] MVC decoder crash {self._edge264_consecutive_crashes}/{cap}; "
                       f"restarting for transient-corruption recovery...")
        self.show_3d_notification("Decoder recovering...", success=False)
        QTimer.singleShot(500, lambda: self._start_mvc_decoder(start_time=resume_time))

    def _fallback_to_mpv_mvc(self):
        """Fallback to mpv native MVC handling"""
        self._restore_mpv_video_output()
        try:
            if not self.player: return
            self.player['hwdec'] = 'no'
            self.player['override-display-fps'] = self._get_effective_video_fps()
            self.player['vf'] = 'scale=1920:2205'
            try:
                self.player['video-sync'] = 'display-resample'
            except Exception:
                pass
            self.show_3d_notification("3D MVC mode (mpv fallback)", success=True)
        except:
            pass

    def _disable_mpv_video_output(self):
        """Force mpv into audio-only mode. Returns True if state changed."""
        if not self.player:
            logger.info("[MVC INIT] _disable_mpv_video_output: player is None!")
            return False
        try:
            # Use 'vid' property (not 'video') - this is the track selection property
            # vid=no means no video track, vid=auto means auto-select
            try:
                current_val = self.player['vid']
            except Exception:
                current_val = 'unknown'

            logger.info(f"[MVC INIT] _disable_mpv_video_output: current vid = {current_val}")

            if current_val == 'no' or current_val is False:
                logger.info("[MVC INIT] _disable_mpv_video_output: already disabled")
                return False

            # Disable video track
            self.player['vid'] = 'no'

            # CRITICAL FIX: Switch video output to null to release D3D11 context
            # This prevents GPU contention between MPV's D3D11 and QRhiWidget's D3D11
            # when the window has focus (both would try to present frames)
            try:
                self._saved_vo = self.player['vo']
                self.player['vo'] = 'null'
                logger.info(f"[MVC INIT] Switched MPV vo from {self._saved_vo} to null (D3D11 released)")
            except Exception as e:
                logger.warning(f"[MVC INIT] Could not switch vo to null: {e}")

            # Also try to set audio sync
            try:
                self.player['video-sync'] = 'audio'
            except Exception:
                pass

            # Verify it worked
            try:
                new_val = self.player['vid']
            except Exception:
                new_val = 'unknown'

            logger.info(f"[MVC INIT] Disabled mpv video output. Before={current_val}, After={new_val}")
            return True
        except Exception as e:
            # Catch ALL exceptions, including Windows fatal exceptions if they propagate
            logger.error(f"[MVC INIT] Warning: Could not disable mpv video: {e}")
            return False

    def _present_via_mpv_native(self):
        """Route presentation to MPV's OWN video output, for 2D files not decoded by
        edge264 (non-h264 or non-.mkv: .mp4/.avi/.m2ts/VC-1…). Idempotent.

        BLACK-SCREEN-ON-RELOAD (2D-after-3D) FIX: a prior 3D/MVC file put MPV in
        audio-only mode (vid=no, vo=null via _disable_mpv_video_output) AND left
        video_stack showing the MVC widget. On a fresh load there is no seek/3D path to
        undo that, so a subsequent 2D MPV-native file plays audio with a black picture.
        Restore BOTH: MPV video output (vo+vid) and the on-screen MPV video widget."""
        try:
            self._restore_mpv_video_output()  # restores vo + sets video='auto'
            try:
                self.player['vid'] = 'auto'   # belt-and-suspenders: ensure track re-selected
            except Exception:
                pass
            if getattr(self, 'video_widget', None) is not None:
                self.video_stack.setCurrentWidget(self.video_widget)
            logger.info("[2D-MPV] Restored MPV native video output + switched to video_widget")
        except Exception as e:
            logger.warning(f"[2D-MPV] present-via-mpv failed: {e}")

    def _fallback_from_edge264(self, reason=""):
        """edge264 could not handle this H.264 stream -> degrade gracefully to mpv.

        Implements the architecture rule "edge264 first, mpv only on failure":
        tear down the edge264 pipeline and hand the source to mpv's own video
        output. We do NOT assume mpv can decode it -- a raw .ssif, for instance,
        may have no mpv-demuxable video -- so _confirm_mpv_fallback_video() checks
        for a real decoded video track afterwards and reports honestly (2D
        playback vs audio-only) instead of silently showing a black frame.
        """
        if reason:
            logger.warning(f"[EDGE264-FALLBACK] {reason}")
        self.mvc_mode_active = False
        try:
            self.controls_overlay.clear_format_badge()  # edge264 didn't adapt → drop the badge
        except Exception:
            pass
        try:
            self._stop_mvc_decoder()
        except Exception:
            pass
        # Degrading to 2D: the framepack 3D window must not linger on a frozen frame.
        fp = getattr(self, 'framepacking_window', None)
        if fp is not None:
            try:
                if fp.isVisible():
                    fp.hide()
            except Exception:
                pass
        # Hand the file to mpv's native video output (idempotent). mpv has been
        # decoding audio all along, so its position is already correct -- no seek.
        self._present_via_mpv_native()
        # mpv reconfigures its video chain asynchronously; verify before claiming.
        QTimer.singleShot(700, self._confirm_mpv_fallback_video)

    def _confirm_mpv_fallback_video(self, attempt=0):
        """Honest post-fallback status: did mpv actually land a decoded video track?"""
        try:
            has_video = bool(self.player and self.player.video_params)
        except Exception:
            has_video = False
        if has_video:
            self.show_3d_notification(
                "edge264 couldn't decode this stream — playing via mpv.", success=False)
        elif attempt < 1:
            # video chain may still be reconfiguring; re-check once before concluding.
            QTimer.singleShot(700, lambda: self._confirm_mpv_fallback_video(attempt + 1))
        else:
            self.show_3d_notification(
                "edge264 failed and mpv has no video for this source — audio only.",
                success=False)

    def _restore_mpv_video_output(self):
        """Restore mpv video output after MVC playback failures/stop."""
        if not self.player:
            return
        try:
            # Restore video output backend (D3D11) if we saved it.
            # MPV's 'vo' property returns a list of dicts when read
            #   (e.g. [{'name': 'gpu-next', 'enabled': True, 'params': {}}])
            # but the *set* path requires a plain string (e.g. 'gpu-next').
            # Without normalization, restore fails with the 'wrong format' MPV error.
            if hasattr(self, '_saved_vo') and self._saved_vo:
                if isinstance(self._saved_vo, list) and self._saved_vo:
                    first = self._saved_vo[0]
                    vo_str = first.get('name', 'gpu-next') if isinstance(first, dict) else 'gpu-next'
                elif isinstance(self._saved_vo, str):
                    vo_str = self._saved_vo
                else:
                    vo_str = 'gpu-next'
                try:
                    self.player['vo'] = vo_str
                    logger.info(f"[MVC] Restored MPV vo to {vo_str}")
                except Exception as e:
                    logger.warning(f"[MVC] Could not restore vo: {e}")
                    # Fallback to gpu-next
                    try:
                        self.player['vo'] = 'gpu-next'
                    except Exception:
                        pass

            self.player['video'] = 'auto'
            try:
                self.player['video-sync'] = 'display-resample'
            except Exception:
                pass
        except Exception:
            pass

    def _update_decoder_audio_clock(self):
        """
        Called periodically by _sync_timer to push the current audio clock
        from the main GUI thread to the decoder thread safely.
        """
        if not self.mvc_mode_active or not self.mvc_decoder_thread:
            if hasattr(self, '_sync_timer') and self._sync_timer.isActive():
                self._sync_timer.stop()
            return

        # V9 SSIF FIX: Update audio clock even when paused
        # The decoder needs the current position for initial sync
        if self.player:
            try:
                pos = self.player.time_pos
                if pos is not None:
                    self.mvc_decoder_thread.update_audio_clock(pos)
            except Exception:
                pass  # Ignore transient errors

    def _stop_mvc_decoder(self):
        """Stop edge264 MVC decoder and cleanup - V7a Enhanced"""
        logger.info("[MVC CLEANUP] Starting complete decoder shutdown...")

        # THUMB: stop thumbnail I/O before any decoder/demuxer teardown
        if getattr(self, '_thumb_service', None):
            self._thumb_service.disarm()

        # V7b FIX: Stop control overlay animations to prevent paintEvent crashes during cleanup
        if hasattr(self, 'controls_overlay') and self.controls_overlay:
            try:
                self.controls_overlay.stop_all_animations()
                logger.debug("[MVC CLEANUP] Controls overlay animations stopped")
            except Exception:
                pass

        # Stop PGS subtitle streaming before decoder cleanup
        if self._subtitle_manager and self._subtitle_manager.is_streaming:
            self._subtitle_manager.stop_streaming()
            logger.info("[MVC CLEANUP] PGS subtitle streaming stopped")

        # ========== CLEANUP STREAMING SUBTITLE STATE ==========
        self._streaming_subtitle_tracks = []
        self._active_streaming_track = None
        self._disable_text_subtitles()
        self._text_sub_connected_widgets = []
        # BD3D depth: drop the dynamic override + per-file state
        self._pg_depth_connected = False
        self._pg_depth_logged = False
        for _w in (getattr(self, 'active_mvc_widget', None),
                   getattr(self, 'mvc_embedded_widget', None),
                   getattr(getattr(self, 'framepacking_window', None), 'display_widget', None)):
            if _w is not None and hasattr(_w, 'set_subtitle_depth'):
                _w.set_subtitle_depth(None)
        logger.info("[MVC CLEANUP] Streaming subtitle state cleared")
        # ======================================================

        # V13 CRASH FIX: Set cleanup flag IMMEDIATELY to stop all memory access
        # This must be done BEFORE anything else to give decoder thread time to notice
        if self.mvc_decoder_thread:
            self.mvc_decoder_thread._cleanup_in_progress = True
            # Break any in-flight C++ SSIF read NOW so the thread can promptly see the stop
            # flags. read_next_*() releases the GIL, so this cross-thread abort lands mid-read;
            # without it a slow cold/contended dependent-extent read pins the thread for tens of
            # seconds and we fall through to the force-terminate path below (which can crash
            # inside the C extension). The flag stays set; the demuxer is recreated on restart.
            try:
                _dmx = getattr(self.mvc_decoder_thread, 'demuxer', None)
                if _dmx is not None and hasattr(_dmx, 'request_abort'):
                    _dmx.request_abort()
                    logger.info("[MVC CLEANUP] Demuxer read abort requested")
            except Exception:
                pass
            # Brief pause to allow decoder thread to see the flag and abort operations
            import time
            time.sleep(0.050)  # 50ms

        # Stop ALL timers (not just _sync_timer and watchdog)
        timer_names = ['_sync_timer', '_stall_watchdog', '_playback_timer', 'controls_hide_timer', '_render_heartbeat_timer']
        for timer_name in timer_names:
            timer = getattr(self, timer_name, None)
            if timer and hasattr(timer, 'isActive') and timer.isActive():
                timer.stop()
                logger.debug(f"[MVC CLEANUP] {timer_name} stopped")

        # CRITICAL: Pause rendering to avoid concurrent access during cleanup.
        # NOTE: pause_rendering is a *method* on the widgets — assigning True to it
        # silently shadows the method with a bool, breaking all future seek protection
        # on the same widget instance (widgets are created once and reused).
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            try:
                self.mvc_embedded_widget.pause_rendering()
                logger.info("[MVC CLEANUP] Embedded widget rendering paused")
            except Exception:
                pass

        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            try:
                self.framepacking_window.display_widget.pause_rendering()
                logger.info("[MVC CLEANUP] Framepacking widget rendering paused")
            except Exception:
                pass

        if self.mvc_decoder_thread:
            # STEP 1: Disconnect all signals to prevent callbacks during shutdown
            try:
                self.mvc_decoder_thread.frameReady.disconnect()
                self.mvc_decoder_thread.frameDecoded.disconnect()
                self.mvc_decoder_thread.frameYUVReady.disconnect()
                self.mvc_decoder_thread.error.disconnect()
                self.mvc_decoder_thread.fps_update.disconnect()
                self.mvc_decoder_thread.decodingFinished.disconnect()
                self.mvc_decoder_thread.stats_update.disconnect()
                self.mvc_decoder_thread.decoderCrashed.disconnect()
                self.mvc_decoder_thread.frameTimestampReady.disconnect()
                self.mvc_decoder_thread.seekFinished.disconnect()
                # Streaming subtitle signals
                if hasattr(self.mvc_decoder_thread, 'subtitleTracksDetected'):
                    try:
                        self.mvc_decoder_thread.subtitleTracksDetected.disconnect()
                    except:
                        pass
                if hasattr(self.mvc_decoder_thread, 'pgsDataReady'):
                    try:
                        self.mvc_decoder_thread.pgsDataReady.disconnect()
                    except:
                        pass
                logger.info("[MVC CLEANUP] All decoder signals disconnected")
            except Exception as e:
                logger.warning(f"[MVC CLEANUP] Error disconnecting signals: {e}")

            # STEP 2: Clear internal buffers/queues
            # DISABLED: Race condition with decoder thread cleanup!
            # try:
            #    if hasattr(self.mvc_decoder_thread, 'presentation_queue'):
            #        self.mvc_decoder_thread.presentation_queue.clear()
            #    if hasattr(self.mvc_decoder_thread, 'frame_buffer'):
            #        self.mvc_decoder_thread.frame_buffer.clear()
            #    logger.info("[MVC CLEANUP] Decoder buffers cleared")
            # except Exception as e:
            #    logger.warning(f"[MVC CLEANUP] Error clearing buffers: {e}")

            # STEP 3: Signal thread to stop
            self.mvc_decoder_thread._stop_requested = True
            self.mvc_decoder_thread.requestInterruption()
            logger.info("[MVC CLEANUP] Stop signal sent to decoder thread")

            # STEP 4: Wait for thread to finish (with timeout)
            if not self.mvc_decoder_thread.wait(5000):  # 5s timeout (increased from 3s)
                logger.error("[MVC CLEANUP] Thread did not stop in 5s! Force terminating...")
                # CRITICAL FIX: NEVER call terminate() on a thread inside a C-extension!
                # Calling terminate() causes "Windows fatal exception: access violation" (0xc0000005)
                # because it kills the thread while it's holding a lock or inside edge264.dll.
                # Last resort: try terminate() with additional wait
                try:
                    self.mvc_decoder_thread.terminate()
                    if not self.mvc_decoder_thread.wait(1000):  # 1s for terminate
                        logger.critical("[MVC CLEANUP] Thread still alive after terminate! App may hang...")
                except:
                    pass  # Terminate may fail, we continue anyway

            logger.info("[MVC CLEANUP] Decoder thread stopped successfully")
            self.mvc_decoder_thread = None
            self._last_display_frame_ts = None
            self._display_fps_avg = None

        # STEP 5: Clear framepacking windows and flush OpenGL textures
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            try:
                # V7b FIX: Clear OpenGL textures to prevent stale frames during seek
                if hasattr(self.mvc_embedded_widget, 'clear_textures'):
                    self.mvc_embedded_widget.clear_textures()
                logger.info("[MVC CLEANUP] Embedded widget cleared")
            except Exception as e:
                logger.warning(f"[MVC CLEANUP] Error clearing embedded widget: {e}")

        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            try:
                # V7b FIX: Clear OpenGL textures to prevent stale frames during seek
                if hasattr(self.framepacking_window.display_widget, 'clear_textures'):
                    self.framepacking_window.display_widget.clear_textures()
                logger.info("[MVC CLEANUP] Framepacking window cleared")
            except Exception as e:
                logger.warning(f"[MVC CLEANUP] Error clearing framepacking window: {e}")

        # STEP 6: Close demuxer
        if self.demuxer:
            try:
                self.demuxer.close()
            except:
                pass
            self.demuxer = None

        # STEP 7: Reset seek queue
        if hasattr(self, '_seek_queue') and self._seek_queue:
             try:
                 self._seek_queue._force_reset_state()
             except:
                 pass

        # Clear MVC mode flag and references
        self.mvc_mode_active = False
        self.active_mvc_widget = None  # CRITICAL: Release reference to widget

        # V7c FIX: Do NOT clear PGS subtitles during MVC decoder restart
        # The subtitle parser should persist across decoder stop/start cycles
        # Only clear the connection state, not the parsed subtitle data
        if self._subtitle_manager:
            # self._subtitle_manager.clear()  # REMOVED - preserves loaded subtitles
            self._subtitle_connected_widgets = []  # Will be reconnected when decoder starts
            # Keep _active_pgs_track_index so we know subtitles were previously selected

        # Force GC to clean up ctypes objects from decoder thread
        import gc
        gc.collect()

        # V14b: Small delay to let MPV event loop settle after all the cleanup
        # This prevents Windows threading exceptions when MPV's C code interacts with Python
        import time
        time.sleep(0.100)  # 100ms settling time

        logger.info("[MVC CLEANUP] Complete decoder shutdown finished")

        # V14b: Clear transition flag - MPV should now be safe
        self._mpv_transition_in_progress = False

        self.mvc_mode_active = False
        # V7b+++ STUTTER FIX: Use proper method call for consistency
        self.controls_overlay.set_mvc_active(False)
        self.controls_overlay.time_slider.setEnabled(True)
        self.monitoring_overlay.reset()
        self.monitoring_overlay.hide()

        # Only restore video if we are NOT restarting (e.g. real stop) and NOT
        # about to terminate mpv (V62: restoring gpu-next rebuilds the D3D11
        # chain on a core that terminate() destroys 50ms later — crash window)
        if not getattr(self, '_mvc_restarting', False) and not getattr(self, '_terminating_mpv', False):
            self._restore_mpv_video_output()
            # Reset stack to MPV widget only on full stop
            if hasattr(self, 'video_stack'):
                self.video_stack.setCurrentWidget(self.video_widget)

    @Slot()
    def _on_mvc_frame_ready(self):
        """
        DEPRECATED: This slot is bypassed when direct widget path is active.
        Kept only for fallback compatibility when _display_widget is None.

        PERFORMANCE: When direct path is active, this entire function is skipped,
        saving ~5-8ms per frame (shared memory read + QImage creation overhead).
        """
        # CRITICAL OPTIMIZATION: Skip entirely if direct widget path is active
        # Direct path handles frame delivery via set_frame_fast() with zero overhead
        if self.mvc_decoder_thread and self.mvc_decoder_thread._display_widget:
            # Frame already delivered directly via _write_frame_to_shared_memory
            # No need to read from shared memory or create QImage
            # Stats update happens in _on_mvc_frame_decoded_optimized
            return

        # FALLBACK PATH: Legacy signal-based delivery (only if no direct widget)
        if not self.shared_buffer:
            return

        try:
            with self.shared_buffer.get_lock():
                buffer_view = np.frombuffer(self.shared_buffer.get_obj(), dtype=np.uint8)
                np.copyto(
                    self.rgb_frame_buffer,
                    buffer_view.reshape((self.MVC_HEIGHT, self.MVC_WIDTH, self.MVC_CHANNELS))
                )
        except Exception as e:
            logger.error(f"Error reading from shared buffer: {e}")
            return

        self._record_display_frame_stats()
        bytes_per_line = self.MVC_WIDTH * self.MVC_CHANNELS

        qimage = QImage(
            self.rgb_frame_buffer.data,
            self.MVC_WIDTH,
            self.MVC_HEIGHT,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        self.current_qimage_ref = qimage

        if self.framepacking_window:
            self.framepacking_window.display_frame(qimage)

    @Slot(object)
    def _on_mvc_frame_decoded_optimized(self, frame_array):
        """
        OPTIMIZED signal handler for numpy array frames.
        Only used as fallback when direct widget call is not available.
        """
        if not self.framepacking_window:
            return

        # Direct numpy array to widget - no QImage conversion needed
        self.framepacking_window.display_widget.set_frame_fast(frame_array)
        self._record_display_frame_stats()

    @staticmethod
    def _split_packed_stereo(planes, mode):
        """Split one packed-stereo YUV420 frame (Full-SBS / Full-TAB) into (L, R).

        edge264 decodes a packed-stereo H.264 stream as a single view, so both
        eyes live in one frame. SBS = left|right halves (split on width) → L is the
        LEFT (base) eye; TAB = top/bottom halves (split on height) → L is the TOP
        (base) eye. Chroma is half-resolution, so it splits at half the luma
        boundary. Returns two (Y, U, V) tuples, each made contiguous for upload.
        With this split the player drives the FSBS exactly like MVC: base view in
        the main window, SBS/TAB combo = main-view layout, FramePack = L+R stacked.
        """
        y, u, v = planes
        if mode == 'sbs':
            wy, wc = y.shape[1] // 2, u.shape[1] // 2
            left = (y[:, :wy], u[:, :wc], v[:, :wc])
            right = (y[:, wy:wy * 2], u[:, wc:wc * 2], v[:, wc:wc * 2])
        else:  # 'tab'
            hy, hc = y.shape[0] // 2, u.shape[0] // 2
            left = (y[:hy], u[:hc], v[:hc])
            right = (y[hy:hy * 2], u[hc:hc * 2], v[hc:hc * 2])
        # ZERO-COPY: return the raw views. The decoded planes are Python-owned
        # buffers (copied out of the edge264 DPB at decode time), so the views
        # stay valid, and the native renderer's set_yuv_frame uploads
        # row-contiguous strided views in place (it honors the row stride).
        # TAB slices are even fully contiguous. Saves ~5.7 MB of memcpy per
        # 3840x1012 frame (~140 MB/s at 24 fps).
        return left, right

    @Slot(object, object)
    def _on_mvc_frame_yuv_ready(self, left_planes, right_planes):
        """Dispatch one decoded MVC frame to every visible render target.

        Both numpy tuples are passed by reference — no extra copy made here.
        The embedded widget and the detached framepack window can render the
        same frame simultaneously (kept in sync) without doubling decode cost.
        """
        # Validate planes — bail out silently on malformed input (decoder transient)
        if (not left_planes or not right_planes
                or len(left_planes) != 3 or len(right_planes) != 3):
            return
        for plane in (*left_planes, *right_planes):
            if plane is None or not isinstance(plane, np.ndarray):
                return

        # A valid frame means edge264 is healthy again — clear the crash streak so
        # transient (recoverable) crashes never accumulate toward the fallback cap.
        if getattr(self, '_edge264_consecutive_crashes', 0):
            self._edge264_consecutive_crashes = 0

        # Packed-stereo (Full-SBS / Full-TAB): edge264 delivered both eyes in ONE
        # frame. Split into separate L (base) / R views so every target renders it
        # like MVC — embedded '2d' shows the base eye, '2d'/'sbs'/'tab' set the main
        # layout, and the framepack window stacks L+R.
        _sm = self.video_3d_info.get('stereo_mode') if isinstance(self.video_3d_info, dict) else None
        if _sm in ('sbs', 'tab'):
            try:
                left_planes, right_planes = self._split_packed_stereo(left_planes, _sm)
            except Exception as e:
                logger.error(f"[PACKED-3D] frame split failed: {e}")
                return

        # Enumerate currently visible targets, deduplicated by identity
        targets = []
        seen = set()

        def _add(widget):
            if widget is None or id(widget) in seen:
                return
            seen.add(id(widget))
            targets.append(widget)

        embedded = getattr(self, 'mvc_embedded_widget', None)
        fp_window = getattr(self, 'framepacking_window', None)
        if embedded is not None and embedded.isVisible():
            _add(embedded)
        if fp_window is not None and fp_window.isVisible():
            _add(fp_window.display_widget)

        # Fallback: during init/transitions no widget may be visible yet but
        # the active target still needs the frame to keep its texture fresh.
        if not targets:
            _add(getattr(self, 'active_mvc_widget', None))

        for target in targets:
            try:
                target.set_frame_yuv_views(left_planes, right_planes)
            except Exception as e:
                logger.error(f"[FRAME-ROUTE] delivery to {type(target).__name__} failed: {e}")

        # Native renderer A/B tap (Tokyo #3, S4): diagnostic only, env-gated by
        # SYLC_NATIVE_TAP=1. Mirrors the same frame into a separate native-D3D11
        # window for live parity comparison. Zero impact when the flag is unset.
        if os.environ.get("SYLC_NATIVE_TAP") == "1":
            tap = getattr(self, '_native_tap', None)
            if tap is None:
                try:
                    from native_renderer.native_tap import NativeRendererTap
                    self._native_tap = tap = NativeRendererTap()
                except Exception as _e:
                    logger.warning(f"[NATIVE-TAP] init failed: {_e}")
                    self._native_tap = tap = False
            if tap:
                sm = 1 if (fp_window is not None and fp_window.isVisible()) else 0
                # Mirror the SAME SDR white level the Qt widget feeds the shader,
                # so the native window matches brightness/saturation exactly.
                ref = targets[0] if targets else getattr(self, 'active_mvc_widget', None)
                sdr = getattr(ref, '_sdr_white_level', None)
                tap.push(left_planes, right_planes, sm, sdr)

        self._record_display_frame_stats()

    @Slot(str)
    def _on_mvc_error(self, error_msg):
        """Slot: MVC decoder error - immediate stop and cleanup"""
        logger.error(f"[MVC ERROR] {error_msg}")

        # CRITICAL: Stop the watchdog IMMEDIATELY BEFORE everything else
        if hasattr(self, '_stall_watchdog') and self._stall_watchdog.isActive():
            self._stall_watchdog.stop()
            logger.info("[MVC ERROR] Watchdog stopped immediately")

        # CRITICAL: Disable MVC mode immediately to prevent the watchdog from restarting
        self.mvc_mode_active = False

        # edge264 first, mpv only on failure: hand the source to mpv's native
        # video output rather than going dark. The helper stops the decoder,
        # restores mpv video, and reports honestly (2D playback vs audio-only).
        self._fallback_from_edge264(reason=f"fatal decoder error: {error_msg}")

    @Slot(float)
    def _on_mvc_fps_update(self, fps):
        """Slot: Update FPS display"""
        self.controls_overlay.set_status_info(f"MVC @ {fps:.1f} fps")
        self.monitoring_overlay.update_decoder_fps(fps)
        self.metrics_overlay.update_decoder_fps(fps)

    @Slot(int, int)
    def _on_mvc_stats_update(self, buffer_size, drop_count):
        self.monitoring_overlay.update_buffer(buffer_size, drop_count)
        self.metrics_overlay.update_buffer(buffer_size, drop_count)
        self._refresh_monitoring_overlay()
        now = time.monotonic()
        if now - self._last_stats_log_ts >= 1.0:
            logger.info(f"[MVC] Stats: buffer={buffer_size}, drops={drop_count}, active={self.mvc_mode_active}")
            self._last_stats_log_ts = now

    @Slot(int, float, int)
    def _on_frame_timestamp(self, frame_id, timestamp, poc):
        """
        Audio synchronization based on the decoder markers.
        DISABLED by default - requires thorough thread-safety testing.

        New synchronization system where the decoder generates a precise
        timestamp for each frame based on the PictureOrderCnt (POC).

        Args:
            frame_id: Unique identifier of the frame
            timestamp: Timestamp computed in seconds
            poc: Picture Order Count of the frame
        """
        raw_timestamp = timestamp

        # DISABLED by default to avoid thread-safety crashes
        if not self._audio_sync_enabled:
            self._last_mvc_timestamp = raw_timestamp
            return

        # Safety checks
        if not self.player or not self.mvc_mode_active:
            self._last_mvc_timestamp = raw_timestamp
            return

        try:
            # Timestamp is already absolute since the decoder thread fix
            absolute_timestamp = raw_timestamp

            # Get the current MPV audio position (thread-safe?)
            # V7b STABILITY FIX: Protect MPV access from crashes
            try:
                audio_pos = self.player.time_pos
            except (RuntimeError, OSError):
                # MPV access failed from decoder thread
                return
            if audio_pos is None or not isinstance(audio_pos, (int, float)):
                return

            # === SMART AUDIO SYNC ALGORITHM ===
            # Velvet #9: make this corrector aware of the V58 video-delay offset so it targets the
            # SAME delayed position the decoder's V12 sync does, instead of fighting it. When it was
            # V58-unaware it read raw_timestamp-audio_pos as a ~610ms "lag" and railed the decoder
            # PI to -250ms -> video paced ~25% fast -> a constant brake/race judder. Targeting the
            # offset lets both correctors cooperate -> stable, smooth playback.
            av_offset = 0.0
            if self.mvc_decoder_thread is not None:
                av_offset = getattr(self.mvc_decoder_thread, '_av_sync_offset_s', 0.0) or 0.0

            # 1. Low-pass bias filter: Cancel constant offsets without abrupt skips
            raw_error_ms = (raw_timestamp - audio_pos + av_offset) * 1000.0  # Convert to ms

            if abs(raw_error_ms) < self.SYNC_BIAS_WINDOW_MS:
                # Learn bias from small errors (adaptive low-pass filter)
                self._sync_bias += raw_error_ms * self.SYNC_BIAS_LEARNING_RATE
                self._sync_bias = max(min(self._sync_bias, self.SYNC_BIAS_MAX_MS), -self.SYNC_BIAS_MAX_MS)

            # 2. Apply bias correction to video timestamp
            absolute_timestamp = raw_timestamp - (self._sync_bias / 1000.0)  # Back to seconds
            self._last_mvc_timestamp = absolute_timestamp

            # 3. Calculate residual error after bias correction
            sync_error_ms = (absolute_timestamp - audio_pos + av_offset) * 1000.0  # ms for comparison
            now = time.monotonic()

            # 4. Progressive drift correction (3-tier strategy)
            if abs(sync_error_ms) < self.SYNC_ACCEPTABLE_MS:
                # Tier 1: Perfect sync - Reset drift accumulator
                self._cumulative_drift = 0.0
                if self.mvc_decoder_thread and hasattr(self.mvc_decoder_thread, 'adjust_timing_drift'):
                    self.mvc_decoder_thread.adjust_timing_drift(0.0)

            elif abs(sync_error_ms) < self.SYNC_MICRO_ADJUST_MS:
                # Tier 2: Small drift (200-500ms) - Micro timing adjustments
                if (now - self._last_drift_adjust_time) > self.SYNC_DRIFT_THROTTLE_S:
                    self._cumulative_drift = sync_error_ms / 1000.0  # seconds
                    if self.mvc_decoder_thread and hasattr(self.mvc_decoder_thread, 'adjust_timing_drift'):
                        self.mvc_decoder_thread.adjust_timing_drift(self._cumulative_drift)
                        self._last_drift_adjust_time = now

                    # Adaptive logging: Only every 48 frames (2s @ 24fps)
                    if frame_id % 48 == 0:
                        logger.debug(f"[AUDIO SYNC] Micro-adjust: {sync_error_ms:.1f}ms "
                                    f"(bias={self._sync_bias:.1f}ms, video={absolute_timestamp:.3f}s)")

            else:
                # Tier 3: Large drift (>500ms) - Gentle timing adjustment only
                if (now - self._last_drift_adjust_time) > self.SYNC_DRIFT_THROTTLE_S:
                    if self.mvc_decoder_thread and hasattr(self.mvc_decoder_thread, 'adjust_timing_drift'):
                        try:
                            self.mvc_decoder_thread.adjust_timing_drift(sync_error_ms / 1000.0)
                            self._last_drift_adjust_time = now

                            # Adaptive logging: Every 120 frames (~5s @ 24fps)
                            if frame_id % 120 == 0:
                                direction = "ahead" if sync_error_ms > 0 else "behind"
                                logger.warning(f"[AUDIO SYNC] Large drift {sync_error_ms:.1f}ms ({direction}) "
                                             f"(bias={self._sync_bias:.1f}ms, gentle timing)")
                        except Exception as e:
                            logger.error(f"[AUDIO SYNC] Timing adjust failed: {e}")

            self._last_frame_timestamp = absolute_timestamp
            
            # 5. Direct UI update with bias-corrected timestamp
            # This ensures the timeline matches the actual sync-adjusted playback
            if absolute_timestamp > 0:
                self._set_ui_time(absolute_timestamp)

        except AttributeError as e:
            logger.error(f"[SYNC] Player attribute error: {e}")
        except Exception as e:
            logger.error(f"[SYNC] Error during synchronization: {e}")

    @Slot()
    def _on_mvc_finished(self):
        """Slot: MVC decoding finished"""
        logger.info("MVC playback finished")

        # V14 GRACEFUL ENDING: Set cleanup flag IMMEDIATELY to stop decoder memory access
        # This must be the VERY FIRST action to prevent Windows threading exceptions
        # The decoder thread checks this flag before every memory operation
        if self.mvc_decoder_thread:
            self.mvc_decoder_thread._cleanup_in_progress = True
            logger.info("[MVC FINISHED] V14: Cleanup flag set - decoder thread notified")

        # V14b MPV TRANSITION GUARD: Prevent MPV event loop exceptions
        # Set flag to block any MPV interactions during shutdown transition
        self._mpv_transition_in_progress = True

        # V7b FIX: Stop control overlay animations IMMEDIATELY to prevent paintEvent crash
        # This must happen BEFORE any async delays
        if hasattr(self, 'controls_overlay') and self.controls_overlay:
            try:
                self.controls_overlay.stop_all_animations()
            except Exception:
                pass

        self.show_3d_notification("MVC playback finished", success=True)

        # CRITICAL FIX V2: Set flag to block MPV callbacks from restarting playback
        # This must be set BEFORE any other operations
        self._playback_ended = True

        # CRITICAL FIX: Stop timeline timer IMMEDIATELY to prevent continued updates
        # This must happen BEFORE any async operations
        if hasattr(self, '_playback_timer') and self._playback_timer.isActive():
            self._playback_timer.stop()
            logger.info("[MVC FINISHED] Playback timer stopped")

        # CRITICAL FIX: Set is_playing to False immediately
        # The MPV pause callback may not fire reliably
        self.is_playing = False

        # CRITICAL FIX V2: Clear MVC file detection flag to prevent timer restart
        # The _handle_pause_change callback keeps timer active if _mvc_file_detected is True
        self._mvc_file_detected = False
        self.mvc_mode_active = False

        # V14b MPV QUIET: Stop MPV completely to calm event loop before cleanup
        # This reduces the chance of MPV event thread throwing exceptions
        try:
            if self.player:
                # First pause
                self.player.pause = True
                logger.info("[MVC FINISHED] MPV paused")
                # Then seek to start to stop any buffering activity
                try:
                    self.player.command('stop')
                    logger.info("[MVC FINISHED] V14b: MPV stopped (event loop calmed)")
                except Exception:
                    pass  # stop may fail if no file loaded, that's OK
        except Exception as e:
            logger.warning(f"[MVC FINISHED] Could not pause/stop MPV: {e}")

        # Update UI to show paused state
        self.controls_overlay.set_paused(True)

        # V14b GRACEFUL ENDING: Increase delay to 300ms for MPV event loop to settle
        # The decoder needs time to exit AND MPV event thread needs to calm down
        QTimer.singleShot(300, self._stop_mvc_decoder)

    def _record_display_frame_stats(self):
        now = time.perf_counter()
        self._last_decoder_activity_ts = time.monotonic()
        if self._last_display_frame_ts is not None:
            delta = now - self._last_display_frame_ts
            if delta > 0:
                fps = 1.0 / delta
                if self._display_fps_avg is None:
                    self._display_fps_avg = fps
                else:
                    self._display_fps_avg = (self._display_fps_avg * 0.8) + (fps * 0.2)
                self.monitoring_overlay.update_display_fps(self._display_fps_avg)
                self.metrics_overlay.update_display_fps(self._display_fps_avg)
        self._last_display_frame_ts = now

    def _get_effective_video_fps(self):
        """
        Determine target FPS without querying mpv properties (which can crash when mpv is mid-transition).
        Prefer metadata from the analyzer, then fall back to last known fps or 24.
        """
        fps_candidates = [
            self.current_video_fps,
            self.video_3d_info.get('fps') if self.video_3d_info else None,
            24.0,
        ]

        fps = 24.0
        for candidate in fps_candidates:
            if candidate and candidate > 1e-3:
                fps = float(candidate)
                break

        # CRITICAL FIX: If detected FPS is suspiciously low (e.g. < 20), force 23.976.
        # This fixes stuttering on files where ffprobe reports wrong/low FPS (e.g. 7.5 fps).
        if fps < 20.0:
            logger.warning(f"[MVC] Detected FPS {fps:.2f} is too low. Forcing 23.976 fps.")
            fps = 23.976
        else:
            logger.info(f"[MVC] Using target FPS: {fps:.3f}")

        fps = max(12.0, min(120.0, fps))
        return fps

    def _current_mpv_time(self):
        """Current playback time (seconds) from mpv, with a UI fallback.

        Sole definition: a second, shadowing copy of this method previously
        lived earlier in the class and silently won method resolution. This
        version is the robust one — it tolerates mpv returning None and, when
        mpv has no position, falls back to the time-slider value instead of 0.0.
        """
        if self.player:
            try:
                pos = self.player.time_pos
                if pos is not None:
                    return float(pos)
            except Exception:
                pass
        return float(self.controls_overlay.time_slider.value()) / 1000.0

    def _on_framepacking_visibility_changed(self, visible):
        self._framepacking_visible = visible
        self.monitoring_overlay.update_window_state(visible)
        self._refresh_monitoring_overlay()

        # V9 FIX: Update active widget and stereo mode when framepacking window visibility changes
        # This ensures frames go to the correct widget with correct stereo mode
        if visible and self.framepacking_window:
            # Switch to framepack mode when window becomes visible
            self.framepacking_window.display_widget.set_stereo_mode('framepack')
            self.active_mvc_widget = self.framepacking_window.display_widget
            if self.mvc_decoder_thread:
                self.mvc_decoder_thread.set_display_widget(self.framepacking_window.display_widget)
            logger.info("[VISIBILITY] Framepacking window visible: switched to framepack mode")
        elif not visible and hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            # Switch back to embedded 2D mode when window is hidden
            self.mvc_embedded_widget.set_stereo_mode('2d')
            self.active_mvc_widget = self.mvc_embedded_widget
            if self.mvc_decoder_thread:
                self.mvc_decoder_thread.set_display_widget(self.mvc_embedded_widget)
            logger.info("[VISIBILITY] Framepacking window hidden: switched to embedded 2D mode")

            # Auto-deactivate the 3D button so the UI matches reality when the
            # user closes the framepacking window (X / Alt-F4 / exit fullscreen).
            # blockSignals avoids re-triggering toggle_3d_mode → configure_3d_output
            # → another hide attempt (idempotent here, but cleaner without the bounce).
            try:
                btn = self.controls_overlay.mode_3d_button
                if btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                    logger.info("[3D-BUTTON] Auto-deactivated (framepacking window closed)")
                self.is_3d_enabled = False
            except Exception as e:
                logger.warning(f"[3D-BUTTON] Could not auto-deactivate: {e}")

    def _refresh_monitoring_overlay(self):
        # Show overlay if MVC decoder is active and we have a file loaded
        # DEBUG: Disabled by default to prevent UI pollution
        should_show = False # self.mvc_mode_active and self.has_media

        # Only show if not explicitly hidden (future feature maybe)
        self.monitoring_overlay.setVisible(should_show)
        if should_show:
            self.monitoring_overlay.raise_()

    def _check_decoder_stall(self):
        # Check whether MVC mode is active
        if not self.mvc_mode_active:
            return

        # Check whether the thread exists and is still alive
        if not self.mvc_decoder_thread or not self.mvc_decoder_thread.isRunning():
            # Thread stopped, stop the watchdog
            if self._stall_watchdog.isActive():
                self._stall_watchdog.stop()
                logger.info("[WATCHDOG] Decoder thread stopped, watchdog disabled")
            return

        # Check for the stall only if the thread is active
        
        # CRITICAL FIX: Do NOT check for stalls if paused!
        if not self.is_playing:
            self._last_decoder_activity_ts = time.monotonic()
            return

        now = time.monotonic()
        if now - self._last_decoder_activity_ts > 5.0:
            logger.error("[WATCHDOG] MVC decoder stalled for >5s. Dumping stack traces...")
            try:
                import faulthandler
                faulthandler.dump_traceback()
            except Exception as e:
                logger.error(f"[WATCHDOG] Failed to dump traceback: {e}")
            if self.mvc_decoder_thread:
                try:
                    self.mvc_decoder_thread.dump_debug_state()
                except Exception as e:
                    logger.error(f"[WATCHDOG] Could not dump decoder state: {e}")
                    self._last_decoder_activity_ts = now

    def _delayed_start_decoder(self, disable_mpv=False):
        """Start MVC decoder thread after OpenGL is fully initialized"""
        try:
            if disable_mpv:
                self._disable_mpv_video_output()

            if self.mvc_decoder_thread:
                # Initialize activity timestamp and start watchdog
                self._last_decoder_activity_ts = time.monotonic()
                self._stall_watchdog.start()

                # Connect subtitle streaming to display widget (for SSIF streaming mode)
                if self._subtitle_manager and self._subtitle_manager.is_streaming:
                    display_widget = getattr(self, 'active_mvc_widget', None)
                    if not display_widget:
                        if hasattr(self, 'framepacking_window') and self.framepacking_window:
                            display_widget = self.framepacking_window.display_widget
                        elif hasattr(self, 'mvc_embedded_widget'):
                            display_widget = self.mvc_embedded_widget
                    if display_widget:
                        self._connect_subtitle_to_widget(display_widget)
                        logger.info(f"[MVC INIT] Streaming subtitles connected to {display_widget.__class__.__name__}")

                self.mvc_decoder_thread.start()
                logger.info("[MVC INIT] Decoder thread started")
            else:
                logger.error("[MVC INIT] mvc_decoder_thread is None!")
                self._mvc_restarting = False
        except Exception as e:
            logger.error(f"[MVC INIT] Failed to start decoder thread: {e}")
            import traceback
            traceback.print_exc()
            self._mvc_restarting = False
        else:
            # Reset restart guard after successful start
            self._mvc_restarting = False


if __name__ == "__main__":
    # Support for PyInstaller on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    # Enable faulthandler to a file (never stderr) to capture
    # real crashes without polluting the console with the SEH 0xe24c4a02
    # transients (cross-thread MPV/decoder) that are already handled by try/except.
    import faulthandler
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _crash_log = open(os.path.join(_script_dir, "crash_log.txt"), "w", encoding="utf-8")
        faulthandler.enable(file=_crash_log, all_threads=True)
        print(f"[FAULTHANDLER] Enabled -> {_crash_log.name}")
    except Exception as e:
        print(f"[FAULTHANDLER] Could not enable: {e}")

    # Suppress PySide6's noisy RuntimeWarning when we disconnect a signal
    # that has no current connection (happens legitimately during MVC cleanup
    # when subtitle streaming was never enabled — pgsDataReady etc.). The
    # actual disconnect failure is already caught by try/except, but PySide6
    # also emits a Python warning at the C++ level before raising.
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r"Failed to disconnect .* from signal",
    )

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

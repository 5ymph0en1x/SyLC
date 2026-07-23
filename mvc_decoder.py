#!/usr/bin/env python3
"""
MVC Decoder Thread - Memory Leak Fix Edition (V5.7)
Version: 5.7 - CRITICAL FIX for 64GB memory leak in minutes
               - Added maxlen to presentation_queue (prevents unlimited growth)
               - Added decoder throttling when queue is full
               - Added periodic garbage collection
               Now relies on edge264's built-in emergency SPS->SSPS copy logic.
               Fixes access violation crashes on NAL 20 (MVC coded slices).
"""

import ctypes
import sys
import os
import struct
import bisect
import heapq
import time
import logging
import threading
import gc
import weakref
import numpy as np
try:
    import velvet_probe  # read-only timing probe; no-op unless SYLC_VELVET_PROBE=1
except Exception:  # pragma: no cover - keep player runnable if probe is absent
    class _VelvetNoop:
        ENABLED = False
        @staticmethod
        def _noop(*a, **k):
            return None
        on_emit = on_present = on_drop = on_hold = on_bulkdrop = record = tick = incr = _noop
        now = time.perf_counter
    velvet_probe = _VelvetNoop()
from collections import deque
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage


# -----------------------------------------------------------------------------
# Blu-ray .clpi EP_map (PTS -> byte seek index for the base view)
# Lets the SSIF/M2TS dual-file demuxer land exactly on an IDR for frame-accurate
# seeking (instead of a slow byte binary-search). Validated against real discs.
# -----------------------------------------------------------------------------
class _ClpiBits:
    __slots__ = ('d', 'pos')

    def __init__(self, data, byte_pos=0):
        self.d = data
        self.pos = byte_pos * 8

    def u(self, n):
        v = 0
        d = self.d
        p = self.pos
        for _ in range(n):
            v = (v << 1) | ((d[p >> 3] >> (7 - (p & 7))) & 1)
            p += 1
        self.pos = p
        return v

    def skip(self, n):
        self.pos += n

    def bytepos(self):
        return self.pos >> 3


def _parse_clpi_epmap(path):
    """Parse a Blu-ray .clpi CPI/EP_map. Returns (pts_ms_list, byte_list) for the first
    stream PID (the base view), RAW timestamps (90kHz/90), bytes = SPN*192. ([],[]) on failure."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if data[0:4] != b'HDMV':
            return [], []
        cpi_start = struct.unpack('>I', data[16:20])[0]
        bs = _ClpiBits(data, cpi_start)
        cpi_len = bs.u(32)
        if cpi_len == 0:
            return [], []
        bs.skip(12)            # reserved
        bs.u(4)                # CPI_type
        ep_map_start = bs.bytepos()
        bs.skip(8)             # reserved_for_future_use
        num_pid = bs.u(8)
        entries = []
        for _ in range(num_pid):
            pid = bs.u(16)
            bs.skip(10)
            bs.u(4)                       # EP_stream_type
            num_coarse = bs.u(16)
            num_fine = bs.u(18)
            start_addr = bs.u(32)
            entries.append((pid, num_coarse, num_fine, start_addr))
        if not entries:
            return [], []
        pid, num_coarse, num_fine, start_addr = entries[0]
        base = ep_map_start + start_addr
        b = _ClpiBits(data, base)
        ep_fine_table_start = b.u(32)
        coarse = []
        for _ in range(num_coarse):
            ref_fine = b.u(18); pts_c = b.u(14); spn_c = b.u(32)
            coarse.append((ref_fine, pts_c, spn_c))
        bf = _ClpiBits(data, base + ep_fine_table_start)
        fine = []
        for _ in range(num_fine):
            bf.u(1); bf.u(3)             # is_angle_change_point, I_end_position_offset
            pts_f = bf.u(11); spn_f = bf.u(17)
            fine.append((pts_f, spn_f))
        pts_ms, byte_off = [], []
        for ci, (ref_fine, pts_c, spn_c) in enumerate(coarse):
            nxt = coarse[ci + 1][0] if ci + 1 < len(coarse) else len(fine)
            for fi in range(ref_fine, min(nxt, len(fine))):
                pf, sf = fine[fi]
                pts90k = (((pts_c & ~0x01) << 18) + (pf << 8)) * 2   # 45kHz -> 90kHz
                spn = (spn_c & ~0x1FFFF) + sf
                pts_ms.append(pts90k // 90)
                byte_off.append(spn * 192)
        return pts_ms, byte_off
    except Exception:
        return [], []


def _find_clpi_for_media(media_path):
    """Locate CLIPINF/<stem>.clpi for a BDMV STREAM file (.ssif/.m2ts), or None."""
    try:
        stem = os.path.splitext(os.path.basename(media_path))[0]
        d = os.path.dirname(os.path.abspath(media_path))
        for _ in range(4):   # .../STREAM/SSIF -> STREAM -> BDMV (CLIPINF beside STREAM)
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


# -----------------------------------------------------------------------------
# Blu-ray 3D Extent Start Point map (the interleaving map -> EXACT .ssif byte).
#
# A .ssif file carries NO embedded extent table; the interleaving map lives in the .clpi
# ExtensionData under ID1=0x0002, ID2=0x0004 (verified on real discs). Combined with the base
# EP_map it yields the EXACT .ssif byte of the interleaved-unit boundary that contains any base
# IDR, so a seek lands on a clean RAPI with BOTH views present -- no size-ratio estimate, no
# disc binary-search. Validated to ~3 ms (sub-frame) vs ~7.5 s for the old ratio heuristic.
# -----------------------------------------------------------------------------
def _parse_clpi_extent_start_points(path):
    """Return the extent-start SPN list (longest monotonic table) from a .clpi ExtensionData,
    or [] if absent. Block layout: [block_len u32][num_point u32][SPN u32 x num_point]."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except Exception:
        return []
    if len(data) < 28 or data[0:4] != b'HDMV':
        return []
    try:
        ext_start = struct.unpack('>I', data[24:28])[0]
        if ext_start == 0 or ext_start + 12 > len(data):
            return []
        if struct.unpack('>I', data[ext_start:ext_start + 4])[0] == 0:  # ExtensionData length
            return []
        p = ext_start + 4
        p += 4 + 3                       # data_block_start_address (u32) + reserved (24 bits)
        n_entries = data[p]; p += 1
        best = []
        for _ in range(n_entries):
            if p + 12 > len(data):
                break
            ext_off = struct.unpack('>I', data[p + 4:p + 8])[0]   # rel. to ExtensionData start
            p += 12
            block_off = ext_start + ext_off
            # The block is length-prefixed, so num_point sits at +4; tolerate a missing prefix.
            for num_off in (block_off + 4, block_off):
                if num_off + 4 > len(data):
                    continue
                num = struct.unpack('>I', data[num_off:num_off + 4])[0]
                spn_off = num_off + 4
                if num < 2 or num > 2_000_000 or spn_off + num * 4 > len(data):
                    continue
                spns = list(struct.unpack('>%dI' % num, data[spn_off:spn_off + num * 4]))
                if all(spns[i] <= spns[i + 1] for i in range(num - 1)):
                    if len(spns) > len(best):
                        best = spns
                    break
        return best
    except Exception:
        return []


def _derive_bd_companion_paths(base_clpi):
    """From a base .clpi, derive (dep_clpi, base_m2ts, dep_m2ts). The dependent clip is the
    base stream number + 1 (00001 -> 00002). Returns existing paths or None per slot."""
    try:
        clipinf = os.path.dirname(base_clpi)             # .../CLIPINF
        bdmv = os.path.dirname(clipinf)                  # .../BDMV
        stream = os.path.join(bdmv, 'STREAM')
        stem = os.path.splitext(os.path.basename(base_clpi))[0]
        depnum = '%05d' % (int(stem) + 1)
    except (ValueError, TypeError):
        return None, None, None
    def existing(p):
        return p if os.path.isfile(p) else None
    return (existing(os.path.join(clipinf, depnum + '.clpi')),
            existing(os.path.join(stream, stem + '.m2ts')),
            existing(os.path.join(stream, depnum + '.m2ts')))


def _build_ssif_seek_table(base_clpi, dep_clpi, base_m2ts, dep_m2ts):
    """Combine the base EP_map with the base+dependent Extent Start Point tables into a
    frame-accurate SSIF seek table: parallel (pts_ms[], ssif_byte[]) where ssif_byte is the
    EXACT byte offset of the interleaved-unit boundary (dependent-first; clean RAPI, both views)
    that contains each base IDR. Returns ([], []) when the extent tables are unavailable."""
    idr_pts_ms, idr_byte = _parse_clpi_epmap(base_clpi)
    if not idr_pts_ms:
        return [], []
    base_starts = _parse_clpi_extent_start_points(base_clpi)
    dep_starts = _parse_clpi_extent_start_points(dep_clpi) if dep_clpi else []
    if len(base_starts) < 2 or len(dep_starts) < 2:
        return [], []
    n = min(len(base_starts), len(dep_starts))
    try:
        base_total = (os.path.getsize(base_m2ts) // 192) if (base_m2ts and os.path.isfile(base_m2ts)) else None
        dep_total = (os.path.getsize(dep_m2ts) // 192) if (dep_m2ts and os.path.isfile(dep_m2ts)) else None
    except OSError:
        base_total = dep_total = None

    def ext_len(starts, j, total):
        nxt = starts[j + 1] if j + 1 < len(starts) else (total if total else starts[j])
        return max(0, nxt - starts[j])

    # Cumulative .ssif source-packet offset of the start of interleaved unit j (dependent-first:
    # each unit = [dependent extent j][base extent j]).
    cum = [0] * n
    for j in range(1, n):
        cum[j] = cum[j - 1] + ext_len(dep_starts, j - 1, dep_total) + ext_len(base_starts, j - 1, base_total)

    pts_out, byte_out = [], []
    for pts_ms, byte_off in zip(idr_pts_ms, idr_byte):
        spn = byte_off // 192                          # base-view source packet number
        j = bisect.bisect_right(base_starts, spn) - 1
        if 0 <= j < n:
            pts_out.append(int(pts_ms))
            byte_out.append(int(cum[j]) * 192)         # .ssif byte of the unit boundary
    return pts_out, byte_out


# -----------------------------------------------------------------------------
# Nuitka Onefile Support - MUST be before importing .pyd modules
# -----------------------------------------------------------------------------
def _get_nuitka_data_dir():
    """Get the directory where Nuitka extracts data files (onefile mode)."""
    candidates = []

    # Method 1: Nuitka's __compiled__ module (most reliable for onefile)
    try:
        import __compiled__
        if hasattr(__compiled__, 'containing_dir'):
            candidates.append(('__compiled__.containing_dir', __compiled__.containing_dir))
    except ImportError:
        pass

    # Method 2: Nuitka's __nuitka_binary_dir (Nuitka 1.x+)
    if hasattr(sys, '__nuitka_binary_dir'):
        candidates.append(('__nuitka_binary_dir', sys.__nuitka_binary_dir))

    # Method 3: Check __file__ of main module (points to extraction dir in onefile)
    try:
        import __main__
        if hasattr(__main__, '__file__') and __main__.__file__:
            main_dir = os.path.dirname(os.path.abspath(__main__.__file__))
            candidates.append(('__main__.__file__', main_dir))
    except:
        pass

    # Method 4: sys.path[0] (often set by Nuitka)
    if sys.path and sys.path[0] and os.path.isdir(sys.path[0]):
        candidates.append(('sys.path[0]', sys.path[0]))

    # Method 5: Directory of executable
    exe_dir = os.path.dirname(sys.executable)
    candidates.append(('sys.executable', exe_dir))

    # Method 6: TEMP directory pattern for Nuitka onefile
    # Nuitka extracts to %TEMP%/onefile_<pid>_<timestamp>/
    try:
        import tempfile
        temp_base = tempfile.gettempdir()
        if os.path.isdir(temp_base):
            for entry in os.listdir(temp_base):
                if entry.startswith('onefile_'):
                    onefile_dir = os.path.join(temp_base, entry)
                    if os.path.isdir(onefile_dir):
                        # Check if our files are there
                        if os.path.exists(os.path.join(onefile_dir, 'edge264.dll')) or \
                           os.path.exists(os.path.join(onefile_dir, 'mvc_demuxer_cpp.cp312-win_amd64.pyd')):
                            candidates.append(('TEMP/onefile_*', onefile_dir))
                            break
    except:
        pass

    # Debug: print all candidates
    for name, path in candidates:
        exists = os.path.isdir(path) if path else False
        pyd_exists = os.path.exists(os.path.join(path, 'mvc_demuxer_cpp.cp312-win_amd64.pyd')) if path else False
        dll_exists = os.path.exists(os.path.join(path, 'edge264.dll')) if path else False

    # Return first valid path that contains our files
    for name, path in candidates:
        if path and os.path.isdir(path):
            # Check if our required files are there
            if os.path.exists(os.path.join(path, 'edge264.dll')) or \
               os.path.exists(os.path.join(path, 'mvc_demuxer_cpp.cp312-win_amd64.pyd')):
                return path

    # Fallback: first existing directory
    for name, path in candidates:
        if path and os.path.isdir(path):
            return path

    return exe_dir

# Ensure Nuitka data directory is in sys.path for .pyd imports
try:
    _nuitka_dir = _get_nuitka_data_dir()
    if _nuitka_dir and os.path.isdir(_nuitka_dir) and _nuitka_dir not in sys.path:
        sys.path.insert(0, _nuitka_dir)

    # Also add to DLL search path on Windows (for edge264.dll)
    if sys.platform == 'win32' and _nuitka_dir:
        os.add_dll_directory(_nuitka_dir)
except Exception:
    pass

try:
    import mvc_demuxer_cpp  # Optional fast path
except ImportError as e:
    # Try to find the .pyd file manually
    pyd_name = 'mvc_demuxer_cpp.cp312-win_amd64.pyd'
    for p in sys.path:
        pyd_path = os.path.join(p, pyd_name)
        if os.path.exists(pyd_path):
            break
    else:
        mvc_demuxer_cpp = None
except Exception:
    mvc_demuxer_cpp = None

# -----------------------------------------------------------------------------
# Configuration & Library Loading (Edge264)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    lib_name = 'edge264.dll'
else:
    lib_name = 'libedge264.so'

def _find_dll(dll_name):
    """Find DLL in multiple locations - works with Nuitka onefile."""
    search_dirs = []

    # Priority 1: Nuitka data directory (for onefile mode)
    try:
        nuitka_dir = _get_nuitka_data_dir()
        if nuitka_dir and os.path.isdir(nuitka_dir):
            search_dirs.append(nuitka_dir)
    except Exception:
        pass

    # Priority 2: Directory of the executable (standalone mode)
    try:
        exe_dir = os.path.dirname(sys.executable)
        if exe_dir and exe_dir not in search_dirs:
            search_dirs.append(exe_dir)
    except Exception:
        pass

    # Priority 3: Directory of this script (development mode)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir and script_dir not in search_dirs:
            search_dirs.append(script_dir)
    except Exception:
        pass

    # Priority 4: Current working directory
    try:
        cwd = os.getcwd()
        if cwd and cwd not in search_dirs:
            search_dirs.append(cwd)
    except Exception:
        pass

    # Search in all directories
    for search_dir in search_dirs:
        dll_path = os.path.join(search_dir, dll_name)
        if os.path.exists(dll_path):
            return dll_path

    # Not found, return just the name for system PATH
    return dll_name

edge264 = None
try:
    dll_path = _find_dll(lib_name)
    edge264 = ctypes.CDLL(dll_path)
    logger.info(f"[MVC-DLL] Loaded from: {dll_path}")
except OSError as e:
    logger.critical(f"[MVC-DLL] CRITICAL ERROR: Unable to load {lib_name}. {e}")

# -----------------------------------------------------------------------------
# C Structures & Constants
# -----------------------------------------------------------------------------
NAL_TYPE_SLICE = 1
NAL_TYPE_IDR = 5
NAL_TYPE_SEI = 6
NAL_TYPE_SPS = 7
NAL_TYPE_PPS = 8
NAL_TYPE_AUD = 9
NAL_TYPE_PREFIX = 14
NAL_TYPE_SUBSET_SPS = 15
NAL_TYPE_SLICE_EXT = 20

def is_mvc_idr_nal(nal_data: bytes, sc_len: int) -> bool:
    """
    Check if a NAL type 20 (coded slice extension) is an IDR frame.
    In MVC, the IDR flag is in the NAL unit header extension.
    Format after NAL header byte:
      - svc_extension_flag (1 bit) = 0 for MVC
      - non_idr_flag (1 bit) - if 0, this is an IDR
      - priority_id (6 bits)
      - ...
    """
    if len(nal_data) < sc_len + 2:
        return False
    nal_type = nal_data[sc_len] & 0x1F
    if nal_type != NAL_TYPE_SLICE_EXT:
        return False
    # Extension header byte
    ext_byte = nal_data[sc_len + 1]
    # svc_extension_flag is the MSB (bit 7)
    svc_flag = (ext_byte >> 7) & 1
    if svc_flag == 1:
        return False  # SVC, not MVC
    # non_idr_flag is bit 6
    non_idr_flag = (ext_byte >> 6) & 1
    return non_idr_flag == 0  # IDR if non_idr_flag is 0

class _PlanePool:
    """Recycles the DPB-exit plane copies (6 large numpy allocs per frame,
    ~144/s at 24fps stereo) through per-size freelists. Buffers return to the
    pool automatically when the LAST numpy reference dies (weakref.finalize;
    numpy views keep their base alive), so frames sitting in the reorder or
    presentation queues can never be overwritten. Thread-safe (deque)."""

    def __init__(self, max_per_size=96):
        self._free = {}          # nbytes -> deque of bytearray
        self._max = max_per_size

    def copy(self, src):
        """Pooled equivalent of src.copy() for a 2-D uint8 array/view."""
        n = src.nbytes
        lst = self._free.get(n)
        try:
            buf = lst.pop() if lst else bytearray(n)
        except IndexError:
            buf = bytearray(n)
        # Finalize the frombuffer array itself: numpy view-collapsing makes ALL
        # downstream views point at it as their base (not at the reshape below),
        # so it is the last ndarray to die — recycling can never race a view.
        flat = np.frombuffer(buf, dtype=np.uint8, count=n)
        weakref.finalize(flat, self._recycle, n, buf)
        arr = flat.reshape(src.shape)
        np.copyto(arr, src)
        return arr

    def _recycle(self, n, buf):
        lst = self._free.get(n)
        if lst is None:
            lst = self._free.setdefault(n, deque())
        if len(lst) < self._max:
            lst.append(buf)


_PLANE_POOL = _PlanePool()


# EDGE264 SESSION LOCK (2026-07-14, watchdog stall at post-seek decode_NAL):
# two edge264 sessions now coexist (playback + ThumbnailService). Steady-state
# concurrent DECODING is fine (measured), but alloc/free racing a decode in the
# other session wedges the DLL. Rule: every edge264_alloc/edge264_free takes
# this lock, and the ThumbnailService holds it across each of its edge264 call
# sections. The playback per-NAL hot path stays UNLOCKED (zero fluidity cost).
edge264_session_lock = threading.RLock()


class Edge264Frame(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_uint8) * 3),
        ("samples_mvc", ctypes.POINTER(ctypes.c_uint8) * 3),
        ("mb_errors", ctypes.c_void_p),
        ("bit_depth_Y", ctypes.c_int8), ("bit_depth_C", ctypes.c_int8),
        ("width_Y", ctypes.c_int16), ("width_C", ctypes.c_int16),
        ("height_Y", ctypes.c_int16), ("height_C", ctypes.c_int16),
        ("stride_Y", ctypes.c_int16), ("stride_C", ctypes.c_int16), ("stride_mb", ctypes.c_int16),
        ("FrameId", ctypes.c_int32), ("FrameId_mvc", ctypes.c_int32),
        ("PictureOrderCnt", ctypes.c_int32), ("PictureOrderCnt_mvc", ctypes.c_int32),
        ("frame_crop_offsets", ctypes.c_int16 * 4), ("return_arg", ctypes.c_void_p),
    ]

if edge264:
    edge264.edge264_alloc.restype = ctypes.c_void_p
    edge264.edge264_alloc.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_void_p]
    edge264.edge264_free.restype = None
    edge264.edge264_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    edge264.edge264_decode_NAL.restype = ctypes.c_int
    edge264.edge264_decode_NAL.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ]
    edge264.edge264_get_frame.restype = ctypes.c_int
    edge264.edge264_get_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(Edge264Frame), ctypes.c_int]
    edge264.edge264_bump_frames.restype = None
    edge264.edge264_bump_frames.argtypes = [ctypes.c_void_p]
    # V33 FIX: Polling functions to wait for decode tasks
    edge264.edge264_get_busy_tasks.restype = ctypes.c_uint
    edge264.edge264_get_busy_tasks.argtypes = [ctypes.c_void_p]
    edge264.edge264_bump_and_get_busy.restype = ctypes.c_uint
    edge264.edge264_bump_and_get_busy.argtypes = [ctypes.c_void_p]
    edge264.edge264_is_frame_ready.restype = ctypes.c_int
    edge264.edge264_is_frame_ready.argtypes = [ctypes.c_void_p]
    edge264.edge264_flush.restype = None
    edge264.edge264_flush.argtypes = [ctypes.c_void_p]
    # CRITICAL: Add return_frame for proper buffer borrowing (fixes 6-band color artifact)
    edge264.edge264_return_frame.restype = None
    edge264.edge264_return_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

def find_nal_units(data: bytes):
    """A generator that finds NAL units in a byte stream."""
    SC3 = b'\x00\x00\x01'
    SC4 = b'\x00\x00\x00\x01'

    start = 0

    # Find the first start code
    pos4 = data.find(SC4, start)
    pos3 = data.find(SC3, start)

    if pos4 != -1 and (pos4 < pos3 or pos3 == -1):
        start = pos4
    elif pos3 != -1:
        start = pos3
    else:
        # No start codes at all
        return

    # Loop to find all subsequent NAL units
    while True:
        next_start = -1

        # Find the next start code after the current one
        pos4 = data.find(SC4, start + 3)
        pos3 = data.find(SC3, start + 3)

        if pos4 != -1 and (pos4 < pos3 or pos3 == -1):
            next_start = pos4
        elif pos3 != -1:
            next_start = pos3

        if next_start == -1:
            # This is the last NAL unit
            yield data[start:]
            return

        # Yield the current NAL unit
        yield data[start:next_start]

        # Move to the next one
        start = next_start

# -----------------------------------------------------------------------------
def _apply_bd_seek_tables(demuxer, filepath):
    """Load Blu-ray frame-accurate seek tables (CLPI EP_map + SSIF exact extent map)
    onto `demuxer` for the clip at `filepath`. Best-effort — the dual-file PTS
    binary-search seek works without it. Shared by the single-clip path and by
    SequenceDemuxer (which calls it for each segment at a junction)."""
    try:
        clpi = _find_clpi_for_media(filepath) if hasattr(demuxer, "set_base_seek_table") else None
        if clpi:
            pts_ms, byte_off = _parse_clpi_epmap(clpi)
            if pts_ms:
                demuxer.set_base_seek_table(pts_ms, byte_off)
                logger.info(f"[BD-SEEK] EP_map seek table: {len(pts_ms)} entries from {os.path.basename(clpi)}")
            # BD3D EXACT seek map: combine the EP_map with the CLPI Extent Start Point
            # tables to land on the precise interleaved-unit boundary in the .ssif (clean
            # RAPI, both views present). Preferred over the EP_map size-ratio estimate;
            # falls back to it silently when the extent tables aren't on disc.
            if hasattr(demuxer, "set_ssif_seek_table"):
                dep_clpi, base_m2ts, dep_m2ts = _derive_bd_companion_paths(clpi)
                s_pts, s_byte = _build_ssif_seek_table(clpi, dep_clpi, base_m2ts, dep_m2ts)
                if s_pts:
                    demuxer.set_ssif_seek_table(s_pts, s_byte)
                    logger.info(f"[BD-SEEK] SSIF exact seek map: {len(s_pts)} IDR->unit entries (byte-exact)")
                else:
                    logger.info("[BD-SEEK] SSIF exact map unavailable (no extent tables) — EP_map ratio fallback in use")
    except Exception as e:
        logger.warning(f"[BD-SEEK] EP_map/SSIF seek-map load skipped: {e}")


def create_demuxer(filepath):
    """Demuxer selection shared by MVCDecoderThread and ThumbnailService.
    Returns (unopened_demuxer, effective_path). effective_path differs from
    filepath only for .m2ts files with an SSIF companion (BD3D layout)."""
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == '.ssif':
        return mvc_demuxer_cpp.MVCSSIFDemuxer(), filepath
    if ext in ('.m2ts', '.ts'):
        try:
            if hasattr(mvc_demuxer_cpp, 'SSIFParser') and mvc_demuxer_cpp.SSIFParser.has_ssif(filepath):
                ssif_path = mvc_demuxer_cpp.SSIFParser.detect_ssif_path(filepath)
                if ssif_path:
                    return mvc_demuxer_cpp.MVCSSIFDemuxer(), ssif_path
        except Exception:
            pass
        return mvc_demuxer_cpp.MVCM2TSDemuxer(), filepath
    if ext in ('.mkv', '.mk3d'):
        return mvc_demuxer_cpp.MVCMatroskaDemuxer(), filepath
    try:
        import lavf_h264_demuxer
        if lavf_h264_demuxer.is_available():
            return lavf_h264_demuxer.LavfH264Demuxer(), filepath
    except Exception:
        pass
    return mvc_demuxer_cpp.MVCMatroskaDemuxer(), filepath


def convert_avcc_to_annexb(codec_private):
    """Annex-B passthrough; avcC (MKV CodecPrivate) converted to Annex-B NALs.
    Delegates to MVCDecoderThread._convert_avcc_to_annexb, whose body is
    self-free (verified) — behavior is byte-identical to the playback path."""
    if not codec_private:
        return b''
    codec_private = bytes(codec_private)
    if codec_private[:4] == b'\x00\x00\x00\x01' or codec_private[:3] == b'\x00\x00\x01':
        return codec_private
    out = MVCDecoderThread._convert_avcc_to_annexb(None, codec_private)
    return out or b''


class SequenceDemuxer:
    """Plays a Blu-ray feature split into several seamless-branching segments
    (e.g. 00016->00020->00022->00026->00028) as ONE continuous MVC stream, so the
    decoder/player can treat a multi-clip feature exactly like a single clip.

    Wraps one MVCSSIFDemuxer at a time (the 'current' segment) and:
      - read_next_frame_pair() delegates to the current segment; at that segment's EOS
        it transparently opens the NEXT segment and continues, returning success=False
        only after the LAST segment. Every returned timestamp is shifted by the
        segment's global start so the decoder sees one global timeline (matching the
        mpv EDL audio clock).
      - seek(global_ms) maps the global time to (segment, local) and switches segments.
      - get_external_duration / cues are made sequence-safe; any other attribute is
        delegated to the current segment's demuxer.
    Each segment gets its own CLPI EP_map / SSIF exact seek table and its own clip
    duration, so frame-accurate seeking keeps working per clip.
    """

    def __init__(self, segments, make_demuxer):
        # segments: ordered [{'path': ssif, 'duration_s': float, ...}]
        self._segments = list(segments)
        self._make = make_demuxer                 # callable () -> a fresh MVCSSIFDemuxer
        self._offsets_ms = []                     # cumulative global start (ms) per segment
        acc = 0
        for s in self._segments:
            self._offsets_ms.append(acc)
            acc += int(round(float(s.get('duration_s', 0.0)) * 1000))
        self._total_ms = acc
        self._idx = 0
        self._inner = None
        self._offset_ms = 0
        self._primed = False                      # current inner has read >=1 frame (seek-ready)
        self.clip_changed = False                 # True right after an auto-advance (junction hook)

    def _open_segment(self, i):
        seg = self._segments[i]
        inner = self._make()
        if not inner.open(seg['path']):
            logger.error(f"[SEQ-DEMUX] failed to open segment {i+1}: {seg['path']}")
            return False
        try:
            if hasattr(inner, 'set_external_duration_ms') and seg.get('duration_s'):
                inner.set_external_duration_ms(int(round(float(seg['duration_s']) * 1000)))
        except Exception:
            pass
        _apply_bd_seek_tables(inner, seg['path'])
        if self._inner is not None and self._inner is not inner:
            try: self._inner.close()
            except Exception: pass
        self._inner = inner
        self._idx = i
        self._offset_ms = self._offsets_ms[i]
        self._primed = False                      # fresh demuxer: must read >=1 frame before seek
        logger.info(f"[SEQ-DEMUX] Segment {i+1}/{len(self._segments)}: {os.path.basename(seg['path'])} "
                    f"@ global {self._offset_ms/1000.0:.1f}s")
        return True

    def _shift(self, d):
        """Shift a frame dict's timestamp into the global timeline."""
        if d is not None:
            try:
                ts = d.get('timestamp')
                if ts is not None:
                    d['timestamp'] = ts + self._offset_ms
            except Exception:
                pass
        return d

    # ---- demuxer interface the decoder thread relies on ----
    def open(self, _path=None):
        return self._open_segment(0)

    def read_next_frame_pair(self):
        while True:
            success, base, dep = self._inner.read_next_frame_pair()
            if success:
                self._primed = True               # this inner is now seek-ready
                return True, self._shift(base), self._shift(dep)
            if self._idx + 1 < len(self._segments):
                logger.info(f"[SEQ-DEMUX] junction: segment {self._idx+1} EOS -> next")
                if self._open_segment(self._idx + 1):
                    self.clip_changed = True
                    continue
            return False, None, None  # true end-of-feature

    def seek(self, global_ms):
        global_ms = max(0, int(global_ms))
        i = 0
        for k in range(len(self._segments)):
            if self._offsets_ms[k] <= global_ms:
                i = k
            else:
                break
        local_ms = global_ms - self._offsets_ms[i]
        if i != self._idx or self._inner is None:
            if not self._open_segment(i):
                return False
        # A freshly-(re)opened SSIF demuxer only seeks correctly after reading >=1 frame
        # (the first read initialises its interleaved-stream / PTS state); otherwise seek()
        # snaps back to the clip start. Prime it once — the frame is discarded since we seek
        # away immediately. (Junctions don't hit this: they read from the start sequentially.)
        if not self._primed:
            try:
                self._inner.read_next_frame_pair()
            except Exception:
                pass
            self._primed = True
        try:
            return bool(self._inner.seek(int(local_ms))) if hasattr(self._inner, 'seek') else True
        except Exception as e:
            logger.warning(f"[SEQ-DEMUX] inner seek error: {e}")
            return False

    def set_external_duration_ms(self, _ms):
        # per-clip durations are applied when each segment opens; ignore the global value
        return None

    def getCuesTimestamps(self):
        # force the robust exact-table + linear-scan seek path per clip (per-clip cues
        # would be local, mismatching the global seek target)
        return []

    def getLastCueTimestamp(self):
        # disable the cue-based base_timestamp path: the wrapper returns GLOBAL timestamps,
        # so the seek handler must anchor on the (global) target, not a per-clip local cue.
        return -1

    def close(self):
        if self._inner is not None:
            try: self._inner.close()
            except Exception: pass
            self._inner = None

    def __getattr__(self, name):
        # delegate anything not overridden to the current segment's demuxer
        inner = self.__dict__.get('_inner')
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


# MVC Decoding Thread
# -----------------------------------------------------------------------------
class MVCDecoderThread(QThread):
    frameReady = Signal()
    frameDecoded = Signal(object)
    frameYUVReady = Signal(object, object)
    error = Signal(str)
    fps_update = Signal(float)
    stats_update = Signal(int, int)
    decodingFinished = Signal()
    decoderCrashed = Signal()
    seekFinished = Signal()  # Signal emitted when the seek is fully complete (IDR found + primed)
    # V7b+ SYNC FIX: Signal emitted with the EXACT timestamp of the IDR found after seek
    # The GUI must seek MPV to this timestamp to guarantee audio/video synchronization
    seekIDRFound = Signal(float)  # (idr_timestamp_seconds)
    thumbnailHarvested = Signal(float, QImage)  # (pts_seconds, 320x180 RGB) — zero-I/O preview harvest
    # New signal for audio synchronization based on the decoder's markers
    frameTimestampReady = Signal(int, float, int)  # (frame_id, timestamp_seconds, poc)
    # PGS subtitle streaming - emits raw PGS data for real-time parsing
    pgsDataReady = Signal(bytes, float)  # (pgs_data, pts_seconds)
    # BD3D authored graphics depth: emitted when the OFMD offset metadata (SEI of
    # the dependent view, per GOP) changes the active PG plane disparity.
    pgDepthChanged = Signal(float)  # normalized eye-width disparity (>0 = in front)
    # Subtitle tracks detected - emits list of available subtitle tracks
    subtitleTracksDetected = Signal(list)  # [{trackNumber, codecId, language, name, isPGS}, ...]

    def __init__(self, filepath, shared_buffer, parent=None,
                 use_gpu_yuv_conversion=True, store_frame_struct_for_gpu=True,
                 start_position=0.0, threads=4, media_duration=None,
                 feature_segments=None, dual_pair=None):
        super().__init__(parent)
        self.filepath = filepath
        self.demuxer = None  # Lazy init in run()
        # Multi-segment (seamless-branching) Blu-ray feature: ordered
        # [{'path','m2ts','duration_s'}]. None or a single segment => normal single clip.
        self._feature_segments = feature_segments if (feature_segments and len(feature_segments) > 1) else None
        # BD3D backup WITHOUT SSIF interleave (MakeMKV: base + dependent views in
        # SEPARATE .m2ts files). (base_path, dep_path) -> _init_demuxer opens an
        # MVCSSIFDemuxer via open_dual() instead of open(); the demuxer then delivers
        # the IDENTICAL base/dep pairs as a real SSIF disc, so the whole downstream
        # pipeline (edge264, POC re-pair, frameYUVReady, seek, abort) is unchanged.
        self._dual_pair = tuple(dual_pair) if (dual_pair and len(dual_pair) == 2) else None
        self.shared_buffer = shared_buffer
        self._media_duration = media_duration  # seconds (float) or None
        
        # THREAD SAFETY FIX: Remove direct player access
        # self.player = player  <-- REMOVED to prevent 0xe24c4a02 crashes
        # THUMB HARVEST throttle: anchored at init time so the FIRST harvest
        # fires ~10s into playback — never inside the first-frame/native-init
        # window (keep the critical startup path free of extra work).
        self._last_thumb_harvest = time.time()
        # Packed-stereo layout for thumbnails ('sbs'|'tab'|None): a thumbnail is
        # a single eye, so packed frames are cropped before downscale. Set by
        # the player right after thread creation (video_3d_info stereo_mode).
        self._thumb_layout = None
        
        self.decoder = None
        self._stop_requested = False
        self._seek_requested = False
        self._seek_target = 0.0
        self._display_widget = None

        # V13 CRASH FIX: Cleanup flag - set BEFORE any cleanup starts
        # This flag is checked before ALL memory access operations
        self._cleanup_in_progress = False

        # V7c FIX: EOS flag - decodingFinished is emitted AFTER cleanup completes
        self._eos_reached = False

        # V14 CRASH FIX: Frame delivery lock - prevents race condition during seek
        # This lock protects memory access in get_plane during frame delivery.
        # Seek must wait for any in-progress frame delivery to complete before freeing decoder.
        self._frame_delivery_lock = threading.Lock()
        self._frame_delivery_active = False  # True while inside _deliver_frame_to_gpu

        self.target_fps = 23.976
        self.target_frame_time = 1.0 / 23.976
        self.num_threads = threads
        self.au_buffers = deque(maxlen=2000)
        self.frame_count = 0
        self.start_time = 0
        self.last_stats_time = 0
        self.base_timestamp = start_position # Base timestamp for frame calculation
        self.mutex = QMutex()
        
        # External Audio Clock Synchronization (Thread-Safe)
        self._audio_mutex = QMutex()
        self._external_audio_clock = 0.0
        self._external_audio_clock_update_ts = 0.0   # liveness: bumped on EVERY push (post-seek acquisition)
        self._external_audio_clock_value_ts = 0.0    # V53: when the VALUE last CHANGED (extrapolation base)
        # Velvet "liquid" pacing: replace the binary V12 early-HOLD (which FREEZES the
        # picture for a cycle = the measured dense-scene judder) with a gentle per-frame
        # deceleration that NEVER blanks a cycle. Root cause (probe-confirmed): in dense
        # scenes the video-ahead diff parks on the +150ms hold trigger and normal jitter
        # dithers it across the line -> hold bursts -> ~90ms emit gaps. _liquid_stretch is
        # the one-shot seconds added to the NEXT frame's schedule while easing back to sync.
        # Default ON (probe-proven on Ruin.mkv: dense-scene emit-std 8-15ms->3-4ms, holds
        # 348->0, ~90ms freezes gone, calm scenes byte-identical). SYLC_LIQUID=0 reverts to
        # the legacy freeze-hold for A/B.
        self._liquid_pacing = os.environ.get('SYLC_LIQUID', '1') not in ('0', 'false', 'False', 'no', 'off')
        self._liquid_stretch = 0.0
        self._enable_audio_sync = True
        # V58/V60: A/V sync trim in seconds. Positive = delay the video (audio chain
        # slower than mpv reports: BT/eARC/soundbar); negative = advance the video
        # (display chain slower: projector/TV processing latency). mpv's time-pos
        # ALREADY subtracts the WASAPI output-buffer delay, so the correct default
        # is 0.0 — the old 0.75 was tuned by ear against the pre-V53/V59c
        # half-frozen audio clock and became a pure 0.75s picture lag once the
        # clock was fixed. Tunable live with [ and ]; the GUI persists the value
        # per install (~/.sylc3d_player.json) and re-applies it on every load.
        self._av_sync_offset_s = 0.0

        self.decoder_ready = False
        self._fatal_error = False
        self._subset_sps_seen = False
        self._first_poc = None
        self._efault_count = 0
        self._efault_limit = 10
        self._consecutive_errors = 0
        self._priming_in_progress = False  # V16 FIX: Flag for increased ENOBUFS tolerance during priming
        self._needs_full_reset = False
        self.frame_buffer = []

        # V60 SYNC-PTS: TRUE container PTS routed to presentation. Every fed frame
        # pair pushes its demuxer timestamp here (a min-heap); every frame that
        # enters the presentation queue (display order — exact since the
        # monotone-POC fix) pops the smallest one. Frames drained-and-DISCARDED
        # during priming consumed the earliest display slots, so their count
        # (_pts_discard_credit) is burned off the heap first. Replaces the
        # arrival-order synthetic base+n*frametime stamps whose bias (priming
        # discards = whole missing slots) parked a constant lip-sync error.
        self._pts_pending = []
        self._pts_discard_credit = 0
        self._pts_last_emit_ts = None
        self._emit_seq = 0
        # V60 SYNC-METER: rolling A/V diff samples for the periodic report
        self._sync_meter = deque(maxlen=512)
        self._sync_meter_count = 0

        # V10 SSIF FIX: Flag to emit seekIDRFound after priming when IDR is not at start
        self._emit_ssif_sync_after_priming = False

        # CRITICAL FIX V7b: Header Caching for Seek Recovery
        # Store the last seen SPS, PPS, and SSPS (MVC) to re-inject them after a seek/flush.
        self._cached_nal_sps = None
        self._cached_nal_pps = None
        self._cached_nal_ssps = None

        # V7b FLUIDITY FIX: Increase to 4 (aligned with ultimate_mvc_player buffer >= 3)
        # Improves smoothness with B-frames, latency +83ms acceptable
        self.REORDER_DEPTH = 4
        self.epoch = 0
        self.last_poc_ordered = -100
        self.force_next_epoch = False
        # MVC RE-PAIRING: edge264 can emit base[X] glued with dependent[X-1] (the two
        # per-view output queues drift by one on B-frame GOPs; fixing it inside the decoder
        # deadlocks the DPB). We pool each dependent picture by its own PictureOrderCnt_mvc
        # and re-associate base[X] with dep[X] (same instant) when queueing for display.
        self._dep_pool = {}

        # V7c STARTUP FIX: Counter to track frames emitted since startup/seek
        # First REORDER_DEPTH+2 frames bypass reordering to establish visual flow
        self._startup_frames_emitted = 0

        # V43 FIX: V12 sync startup grace period and hold timeout
        self._v12_frames_emitted = 0  # Skip V12 sync for first 30 frames
        self._v12_hold_start = None  # Track hold duration for hold timeout
        self._v12_audio_stale_freerun = False  # V59: emit wall-clock paced while audio clock is stale
        self._v12_freerun_audio0 = None        # V59: audio clock value when free-run engaged
        self._v12_freerun_from_seek = False    # V61: free-run entered at seek (vs mid-play stall)
        self._audio_gate_near_since = None     # V61: audio-gate settle tracker

        # MEMORY LEAK FIX V5.7: Add maxlen to presentation_queue
        # Limit to ~3 seconds of frames at 24fps = 72 frames
        # Each frame ~6MB (left+right YUV) = max 432MB for queue
        MAX_PRESENTATION_QUEUE_SIZE = 72
        self.presentation_queue = deque(maxlen=MAX_PRESENTATION_QUEUE_SIZE)
        # V54: dedicated PRESENTER thread decouples presentation from decode. A heavy
        # I-frame decode (~100ms, every ~1s GOP) used to block the single-threaded
        # decode+present loop → a ~100ms video freeze per keyframe (the "stutter
        # every second"). With a presenter thread, the buffer is drained at a
        # steady cadence WHILE the decode thread absorbs the I-frame spike. Backpressure
        # (_await_queue_space, applied at the decode entry OUTSIDE the delivery lock)
        # stops the freed-up decoder from over-filling and evicting unshown frames.
        self._presenter_thread = None
        self._presenter_active = False
        self._present_high_water = MAX_PRESENTATION_QUEUE_SIZE - 6  # backpressure trigger

        # Fix for missing attribute error (legacy from old debug code)
        self._ppq_diag_count = 0

        # State flags
        self._is_paused = False
        self._pause_locked = False  # V8 PAUSE FIX: Prevents GUI from resuming after paused seek
        self._pause_timestamp = 0.0  # V8 PAUSE FIX: Track when pause() was last called
        self._waiting_for_idr = False
        self._first_frame_after_seek = True  # FORCE display of first frame after seek to resync Audio
        self._audio_gate_locked = False      # V7b: Smart Audio Gate (Wait for actual audio start)
        self._audio_gate_timeout_start = 0
        self._audio_gate_start_time = None   # V7b: Capture exact audio start time for sync
        self._sync_pending = False           # V7b: Pending Hard Resync flag
        self._waiting_for_audio_edge = False # V7b: Show & Hold flag
        self._catchup_mode = False           # V7b: Catch-Up Dropping flag
        self._pll_integral = 0.0             # V7b: PLL Integral term (Speed learning)
        self._seek_in_progress = False
        self._pending_seek_target = None

        # ========== SUBTITLE STREAMING SUPPORT ==========
        self._subtitle_tracks = []           # List of detected subtitle tracks
        self._active_subtitle_track = None   # Currently active track number
        self._subtitle_streaming_enabled = False
        # ================================================
        self._decoder_stabilizing = False  # V7b++++++ CRASH FIX: Post-flush stabilization flag
        self._sync_gate_active = False      # V8 SYNC GATE: Blocks presentation but allows decoding/drain
        self._sync_gate_start_time = 0      # V8 SYNC GATE: When the gate was activated
        self._sync_gate_target = 0.0        # V8 SYNC GATE: Target timestamp to sync to

        # V8 IDR INDEX CACHE: Store discovered IDR frames for faster repeat seeks
        # Format: sorted list of (timestamp_seconds, was_successful)
        # When we find an IDR during scan, we cache it for future use
        self._idr_index_cache = []
        self._idr_index_lock = QMutex()

        # MEMORY LEAK FIX V5.7: Garbage collection counter
        # GC at 100 frames = every ~4s at 24fps → potentially perceptible 20-50ms pause.
        # 500 frames = every ~20s; still catches refcycles but spreads the cost.
        self._gc_counter = 0
        self._gc_interval = 500  # Force GC every 500 frames

        # SOL 2D: Progressive timing adjustment
        self._timing_drift_ms = 0.0  # Cumulative drift in milliseconds
        self._timing_adjustment_mutex = QMutex()

        # Native zero-copy ring buffer (C++ side) and decoder fast-path
        self._native_ring = None
        self._native_decoder = None
        self._use_native_pipeline = False
        if mvc_demuxer_cpp:
            try:
                if hasattr(mvc_demuxer_cpp, "FrameRingBuffer"):
                    if isinstance(shared_buffer, mvc_demuxer_cpp.FrameRingBuffer):
                        self._native_ring = shared_buffer
                    else:
                        capacity = max(48, int(self.target_fps * 6))
                        self._native_ring = mvc_demuxer_cpp.FrameRingBuffer(capacity=capacity)
                if hasattr(mvc_demuxer_cpp, "MVCDecoder"):
                    self._native_decoder = mvc_demuxer_cpp.MVCDecoder()
                
                # Enable native pipeline only if both components are ready
                if self._native_ring and self._native_decoder:
                    self._use_native_pipeline = True
                    logger.info("[MVC-THREAD] Native C++ pipeline enabled (RingBuffer + MVCDecoder)")
                else:
                    self._use_native_pipeline = False
                    logger.warning("[MVC-THREAD] Native pipeline incomplete. Fallback to ctypes.")
            except Exception as e:
                logger.warning(f"[MVC-THREAD] Native pipeline setup failed: {e}")
                self._native_ring = None
                self._native_decoder = None
                self._use_native_pipeline = False

    def set_media_duration(self, seconds):
        """Update the media duration (seconds) so proportional SSIF/M2TS seek works.

        mpv reports duration asynchronously, often AFTER this thread and its demuxer
        were created, so the startup set_external_duration_ms() can be skipped (the
        proportional C++ seek then divides into a zero duration and lands at byte 0 =
        restart). The GUI calls this when mpv's duration arrives; the value is also
        re-applied to the demuxer right before each seek, on this thread.
        """
        try:
            if not seconds or float(seconds) <= 0:
                return
            self._media_duration = float(seconds)
        except (TypeError, ValueError):
            pass

    def seek(self, time_pos):
        """Request a seek to the specified time (seconds)."""
        self.mutex.lock()
        if self._seek_in_progress:
            # Queue only the latest target; drop older ones to avoid rapid-fire seeks
            self._pending_seek_target = time_pos
            self.mutex.unlock()
            return
        self._seek_requested = True
        self._seek_target = time_pos
        self._first_frame_after_seek = True
        self._valid_frames_received = 0  # V39: Reset cold start counter on seek

        # V7b STRATEGY CHANGE: Fire on Sight + HOLD.
        # Show first frame, then wait for audio to catch up.
        self.mutex.unlock()
        self._audio_mutex.lock()
        self._audio_gate_locked = False
        self._waiting_for_audio_edge = True # Enable "Show & Hold"
        self._audio_gate_timeout_start = time.time()
        self._audio_gate_near_since = None  # V61: settle tracker for the release test
        self._sync_pending = True
        self._audio_mutex.unlock()
        return

    def set_display_widget(self, widget):
        self._display_widget = widget

    def pause(self):
        """Pauses the decoder clock and presentation."""
        self.mutex.lock()
        # V8 PAUSE FIX: Only update timestamp on REAL transition from playing to paused
        # If already paused, don't update timestamp (GUI may call pause() multiple times)
        if not self._is_paused:
            self._pause_timestamp = time.time()
            logger.info("[MVC-THREAD] Paused (transition from PLAYING).")
        else:
            logger.info("[MVC-THREAD] Paused (already paused, timestamp preserved).")
        self._is_paused = True
        self.mutex.unlock()

    def resume(self):
        """Resumes the decoder clock and presentation."""
        self.mutex.lock()
        # V8 PAUSE FIX: Don't resume if pause is locked (after paused seek)
        if self._pause_locked:
            self.mutex.unlock()
            logger.info("[MVC-THREAD] Resume BLOCKED (pause locked after seek)")
            return
        # V8 CRASH FIX: Don't resume if already playing (prevents double resume race)
        if not self._is_paused:
            self.mutex.unlock()
            logger.debug("[MVC-THREAD] Resume SKIPPED (already playing)")
            return
        self._is_paused = False
        # Reset update timestamp to prevent huge jump in extrapolation
        # The next update_audio_clock from GUI will re-sync perfectly
        self._audio_mutex.lock()
        _now = time.time()
        self._external_audio_clock_update_ts = _now
        self._external_audio_clock_value_ts = _now   # V53: reset extrapolation base too
        self._audio_mutex.unlock()
        self.mutex.unlock()
        logger.info("[MVC-THREAD] Resumed.")

    def set_target_fps(self, fps):
        if fps > 0:
            self.target_fps = fps
            self.target_frame_time = 1.0 / fps

    def update_audio_clock(self, clock_time):
        """Thread-safe update of the external audio clock."""
        new_clock = float(clock_time)
        now = time.time()
        self._audio_mutex.lock()
        # Liveness timestamp: bumped on EVERY push (post-seek acquisition keys off this).
        self._external_audio_clock_update_ts = now

        # V53 1Hz-STUTTER FIX: only move the EXTRAPOLATION base when the value actually
        # advances. MPV runs audio-only in MVC mode (vid=no), so its time-pos steps in
        # coarse ~1Hz increments while the GUI re-pushes the SAME value every 100ms.
        # The old code reset the extrapolation base on EVERY push, so `elapsed` never
        # grew past ~100ms — the sync clock stayed pinned to the stale value for ~1s
        # then jumped +1s, and V12 sync held/dropped a frame every second (visible 1Hz
        # stutter). Keeping the original value-change timestamp lets _get_audio_clock
        # extrapolate smoothly across the gap; when MPV finally steps, the extrapolated
        # value already matches it → no discontinuity, no stutter.
        if new_clock != self._external_audio_clock or self._external_audio_clock_value_ts == 0.0:
            self._external_audio_clock = new_clock
            self._external_audio_clock_value_ts = now

            # V7b+ SYNC FIX: Use base_timestamp (IDR position) instead of _seek_target.
            # The GUI now syncs MPV to base_timestamp, so audio should be near base_timestamp.
            if self._audio_gate_locked:
                # If audio clock is close to base_timestamp (within 500ms), open the gate.
                if abs(self._external_audio_clock - self.base_timestamp) < 0.5:
                    self._audio_gate_locked = False
                    self._audio_gate_start_time = self._external_audio_clock  # Capture exact audio start

        self._audio_mutex.unlock()

    # ========== SUBTITLE STREAMING PUBLIC API ==========

    def set_subtitle_track(self, track_number):
        """Enable streaming for a specific subtitle track.

        Args:
            track_number: Track number to enable (0 or None to disable)
        """
        self.mutex.lock()
        try:
            if track_number is None or track_number == 0:
                self._active_subtitle_track = None
                self._subtitle_streaming_enabled = False
                if self.demuxer:
                    if hasattr(self.demuxer, 'set_subtitle_track'):
                        self.demuxer.set_subtitle_track(0)  # Disable in demuxer
                    elif hasattr(self.demuxer, 'set_subtitle_pid'):
                        self.demuxer.set_subtitle_pid(0)
                logger.info("[MVC-THREAD] Subtitle streaming DISABLED")
            else:
                self._active_subtitle_track = track_number
                self._subtitle_streaming_enabled = True
                if self.demuxer:
                    if hasattr(self.demuxer, 'set_subtitle_track'):
                        self.demuxer.set_subtitle_track(track_number)
                    elif hasattr(self.demuxer, 'set_subtitle_pid'):
                        self.demuxer.set_subtitle_pid(track_number)
                logger.info(f"[MVC-THREAD] Subtitle streaming ENABLED for track {track_number}")
        finally:
            self.mutex.unlock()

    def get_subtitle_tracks(self):
        """Return detected subtitle tracks."""
        return self._subtitle_tracks.copy()

    def _poll_subtitles(self):
        """Poll for subtitle blocks and emit them via pgsDataReady signal.

        Called from the main decode loop when subtitle streaming is enabled.
        """
        if not self._subtitle_streaming_enabled or not self._active_subtitle_track:
            return

        if not self.demuxer:
            return

        try:
            # MKV: Use has_subtitle_data() and read_subtitle_block()
            if hasattr(self.demuxer, 'has_subtitle_data'):
                has_data = self.demuxer.has_subtitle_data()

                while has_data:
                    success, block = self.demuxer.read_subtitle_block()
                    if not success or block is None:
                        break

                    pts_seconds = block['timestampMs'] / 1000.0
                    data = bytes(block['data'])

                    # DEBUG level: this runs in the decode hot loop — INFO here
                    # spams the log at every PGS cue during playback.
                    logger.debug(f"[POLL-SUBS] Got subtitle block: PTS={pts_seconds:.3f}s, size={len(data)} bytes")

                    # Emit PGS data for real-time parsing by SubtitleManager
                    self.pgsDataReady.emit(data, pts_seconds)

                    has_data = self.demuxer.has_subtitle_data()

        except Exception as e:
            logger.warning(f"[POLL-SUBS] Error: {e}")
            pass

    # ===================================================

    def adjust_timing_drift(self, drift_seconds):
        """SOL 2D: Adjust frame timing to compensate for A/V drift.
        Drift is negative if video lags behind audio (most common case).
        """
        self._timing_adjustment_mutex.lock()
        # Convert to milliseconds and clamp to avoid brutal speed swings
        drift_ms = max(min(drift_seconds * 1000.0, 250.0), -250.0)
        self._timing_drift_ms = drift_ms
        self._timing_adjustment_mutex.unlock()

    def seek_to_timestamp(self, timestamp_seconds):
        """SOL 2E: Gentle seek to resync with audio."""
        logger.info(f"[MVC-THREAD] Gentle seek requested to {timestamp_seconds:.3f}s")
        self.seek(timestamp_seconds)

    def adjust_av_offset(self, delta):
        """V58/V60: change the A/V sync trim. Positive = delay the video (audio chain
        latency); negative = advance the video (display chain latency). Returns new value."""
        self._av_sync_offset_s = max(-1.0, min(2.0, self._av_sync_offset_s + delta))
        return self._av_sync_offset_s

    def _push_pair_pts(self, base, dep):
        """V60 SYNC-PTS: register the container PTS (display time, ms in the demuxer
        dicts) of a frame pair that was actually FED to the decoder."""
        try:
            ts = None
            if base and 'timestamp' in base:
                ts = float(base['timestamp']) / 1000.0
            elif dep and 'timestamp' in dep:
                ts = float(dep['timestamp']) / 1000.0
            if ts is not None:
                heapq.heappush(self._pts_pending, ts)
        except Exception:
            pass

    def _push_pts_value(self, ts_seconds):
        """V60 SYNC-PTS: register an already-known PTS (seconds)."""
        try:
            if ts_seconds is not None:
                heapq.heappush(self._pts_pending, float(ts_seconds))
        except Exception:
            pass

    def _assign_emit_pts(self, data):
        """V60 SYNC-PTS: stamp a frame entering the presentation queue with its TRUE
        container PTS. Emission happens in display order (monotone-POC), and PTS
        sorted ascending IS display order, so popping the smallest pending PTS per
        emitted frame pairs them exactly. Self-healing: entries whose frames never
        delivered (corrupt, seek edges) are dropped once stale; if the heap is
        empty or implausible, fall back to the last stamp + one frame time."""
        ft = self.target_frame_time
        last = self._pts_last_emit_ts
        # First frame after (re)start: anchor on base_timestamp with a wide window
        # (recovery-point seeks land up to ~2-3s from the requested target).
        expected = self.base_timestamp if last is None else last + ft
        window = 3.0 if last is None else 0.5
        ts = None
        try:
            # Frames drained-and-discarded during priming consumed the earliest
            # display slots: burn their PTS entries first.
            while self._pts_discard_credit > 0 and self._pts_pending:
                heapq.heappop(self._pts_pending)
                self._pts_discard_credit -= 1
            # Drop stale entries (their frame will never be presented).
            while self._pts_pending and self._pts_pending[0] < expected - window:
                heapq.heappop(self._pts_pending)
            if self._pts_pending and self._pts_pending[0] <= expected + window:
                ts = heapq.heappop(self._pts_pending)
        except Exception:
            ts = None
        if ts is None:
            ts = expected
        data['timestamp'] = ts
        self._pts_last_emit_ts = ts
        self._emit_seq += 1
        return ts

    def _reset_pts_pipeline(self):
        """V60 SYNC-PTS: clear PTS state (seek / full reset)."""
        self._pts_pending.clear()
        self._pts_discard_credit = 0
        self._pts_last_emit_ts = None
        self._emit_seq = 0

    def _get_audio_clock(self):
        """Returns the extrapolated audio clock."""
        if self._is_paused:
            # Return last known clock without extrapolation
            self._audio_mutex.lock()
            base_clock = self._external_audio_clock
            self._audio_mutex.unlock()
            return base_clock

        self._audio_mutex.lock()
        base_clock = self._external_audio_clock
        # V53: extrapolate from when the VALUE last changed, not the last push.
        # MPV (audio-only) steps time-pos ~1Hz; extrapolating from the value-change
        # timestamp bridges those gaps smoothly instead of pinning to a stale value.
        value_ts = self._external_audio_clock_value_ts
        self._audio_mutex.unlock()

        if value_ts == 0.0:
            return None

        # V43 FIX: Cap extrapolation to prevent the audio clock from racing ahead
        # when updates stop arriving (e.g., MPV stalled). V53: cap raised 1.0->1.5s
        # so a normal coarse ~1Hz time-pos step (plus jitter) is still bridged by
        # extrapolation; only a genuine multi-second stall falls back to base_clock.
        elapsed = time.time() - value_ts
        if elapsed > 1.5:
            # Stale value (>1.5s old) - return base clock without extrapolation
            # to prevent wild divergence. V12 sync will handle the drift.
            return base_clock

        # Extrapolate based on time elapsed since the value last changed
        return base_clock + elapsed

    def _setup_seek_tables(self):
        """Load the Blu-ray frame-accurate seek tables for the current single clip.
        (Multi-segment features set their tables per clip inside SequenceDemuxer.)"""
        _apply_bd_seek_tables(self.demuxer, self.filepath)

    def _open_demuxer(self):
        """Open self.demuxer for the current source. BD3D dual-file backups use
        open_dual(base, dep) (base + dependent m2ts read in parallel, paired by PTS);
        every other source uses the normal open(path). Kept separate so the
        fresh-mount settle-retry re-runs the correct open call."""
        if self._dual_pair:
            return self.demuxer.open_dual(self._dual_pair[0], self._dual_pair[1])
        return self.demuxer.open(self.filepath)

    def _init_demuxer(self):
        """Initializes the appropriate demuxer based on file extension."""
        if self.demuxer: return True

        try:
            _, ext = os.path.splitext(self.filepath)
            ext = ext.lower()

            logger.info(f"[MVC-THREAD] Initializing demuxer for {ext}...")

            if self._dual_pair:
                # BD3D backup with no SSIF interleave: base + dependent views live in
                # SEPARATE .m2ts files. The C++ dual-source demuxer (open_dual) reads
                # both and matches AUs by PTS, yielding the SAME base/dep pairs an SSIF
                # disc would — the downstream MVC pipeline is untouched.
                base_p, dep_p = self._dual_pair
                logger.info(f"[MVC] dual-file: base={os.path.basename(base_p)} dep={os.path.basename(dep_p)}")
                self.demuxer = mvc_demuxer_cpp.MVCSSIFDemuxer()
                self.filepath = base_p  # base clip = seek-table / logging anchor
            elif self._feature_segments and ext == '.ssif':
                # Seamless-branching feature: play the ordered SSIF segments as one
                # continuous stream (timestamps shifted to a global timeline matching
                # the mpv EDL). The decoder loop sees a normal demuxer.
                logger.info(f"[MVC-THREAD] Multi-segment feature: {len(self._feature_segments)} SSIF "
                            f"segments -> SequenceDemuxer (seamless branching)")
                self.demuxer = SequenceDemuxer(self._feature_segments, mvc_demuxer_cpp.MVCSSIFDemuxer)
            else:
                # Selection factored into create_demuxer() — shared with the
                # ThumbnailService (spec 2026-07-14). Same demuxers as before.
                self.demuxer, _eff = create_demuxer(self.filepath)
                if _eff != self.filepath:
                    logger.info(f"[MVC-THREAD] SSIF companion detected: {_eff}")
                    self.filepath = _eff  # Use SSIF path for demuxer
                logger.info(f"[MVC-THREAD] Using {type(self.demuxer).__name__} (create_demuxer)")

            if not self._open_demuxer():
                # FRESH-MOUNT SETTLE RETRY (2026-07-14, Avatar ISO): right after
                # an ISO mount, reads can transiently return nothing — the
                # codec-private extraction failed at T+0 while the IDENTICAL
                # open succeeded 2s later. One retry with a fresh instance.
                logger.warning("[MVC-THREAD] Demuxer open failed — fresh-mount settle retry in 1.5s...")
                for _ in range(15):
                    if self._stop_requested:
                        return False
                    time.sleep(0.1)
                try:
                    if self._dual_pair:
                        self.demuxer = mvc_demuxer_cpp.MVCSSIFDemuxer()
                    elif self._feature_segments and ext == '.ssif':
                        self.demuxer = SequenceDemuxer(self._feature_segments, mvc_demuxer_cpp.MVCSSIFDemuxer)
                    else:
                        self.demuxer, _eff_r = create_demuxer(self.filepath)
                        self.filepath = _eff_r
                except Exception as e:
                    logger.warning(f"[MVC-THREAD] Settle-retry re-create failed: {e}")
                if not self._open_demuxer():
                    self.error.emit(f"Unable to open file: {self.filepath}")
                    return False
                logger.info("[MVC-THREAD] Settle retry succeeded — demuxer open on 2nd attempt")

            # Provide duration hint when available (helps seek positioning when MKV header lacks Duration)
            try:
                if hasattr(self.demuxer, "set_external_duration_ms") and self._media_duration:
                    self.demuxer.set_external_duration_ms(int(float(self._media_duration) * 1000))
                    logger.info(f"[MVC-THREAD] External duration hint set: {self._media_duration:.3f}s")
            except Exception as e:
                logger.warning(f"[MVC-THREAD] Could not set external duration hint: {e}")

            # Frame-accurate seeking: a single clip loads its EP_map/SSIF map here; a
            # multi-segment SequenceDemuxer sets each clip's tables itself per segment.
            if not isinstance(self.demuxer, SequenceDemuxer):
                self._setup_seek_tables()

            logger.info("[MVC-THREAD] Demuxer opened successfully")

            # ========== SUBTITLE TRACK DETECTION ==========
            try:
                # MKV files: Use get_subtitle_tracks() for track enumeration
                if hasattr(self.demuxer, 'get_subtitle_tracks'):
                    tracks = self.demuxer.get_subtitle_tracks()
                    if tracks:
                        self._subtitle_tracks = tracks
                        pgs_tracks = [t for t in tracks if t.get('isPGS', False)]
                        logger.info(f"[MVC-THREAD] Detected {len(tracks)} subtitle tracks ({len(pgs_tracks)} PGS)")
                        for t in tracks:
                            logger.info(f"  - Track {t.get('trackNumber')}: {t.get('codecId')} (PGS={t.get('isPGS')})")
                        # Emit signal to GUI with track list
                        self.subtitleTracksDetected.emit(tracks)
                    else:
                        logger.info("[MVC-THREAD] No subtitle tracks detected in MKV")
                # M2TS/SSIF files: Use get_subtitle_pids() for PID-based detection
                elif hasattr(self.demuxer, 'get_subtitle_pids'):
                    subtitle_pids = self.demuxer.get_subtitle_pids()
                    if subtitle_pids:
                        # Convert PIDs to track-like format for consistent API. The language is
                        # filled in by the GUI (_on_subtitle_tracks_detected) from the Blu-ray
                        # .clpi — raw M2TS/SSIF carries no language tag in the PMT.
                        tracks = [{'trackNumber': pid, 'name': f'PGS Track (PID 0x{pid:04X})',
                                   'language': '', 'codecId': 'S_HDMV/PGS', 'isPGS': True}
                                  for pid in subtitle_pids]
                        self._subtitle_tracks = tracks
                        logger.info(f"[MVC-THREAD] PGS subtitle PIDs detected: {[hex(p) for p in subtitle_pids]}")
                        self.subtitleTracksDetected.emit(tracks)
                    else:
                        logger.info("[MVC-THREAD] No PGS subtitle tracks detected in M2TS")
            except Exception as e:
                logger.warning(f"[MVC-THREAD] Subtitle track detection failed: {e}")
            # ================================================

            return True

        except Exception as e:
            logger.error(f"[MVC-THREAD] Demuxer init exception: {e}")
            self.error.emit(f"Demuxer error: {str(e)}")
            return False

    def _cache_idr_position(self, timestamp_seconds):
        """V8 IDR INDEX: Cache a successfully found IDR position."""
        self._idr_index_lock.lock()
        try:
            # Avoid duplicates (within 0.5s tolerance)
            for ts in self._idr_index_cache:
                if abs(ts - timestamp_seconds) < 0.5:
                    return  # Already cached
            self._idr_index_cache.append(timestamp_seconds)
            self._idr_index_cache.sort()
            logger.info(f"[V8-IDR-INDEX] Cached IDR at {timestamp_seconds:.3f}s (total: {len(self._idr_index_cache)})")
        finally:
            self._idr_index_lock.unlock()

    def _find_nearest_cached_idr(self, target_timestamp):
        """V8 IDR INDEX: Find the nearest cached IDR at or before target_timestamp.
        Returns timestamp or None if no suitable IDR is cached."""
        self._idr_index_lock.lock()
        try:
            if not self._idr_index_cache:
                return None
            # Find the largest cached IDR that is <= target
            best = None
            for ts in self._idr_index_cache:
                if ts <= target_timestamp:
                    best = ts
                else:
                    break  # List is sorted, no need to continue
            return best
        finally:
            self._idr_index_lock.unlock()

    def scan_for_idr_via_cues(self, target_s: float, max_keyframes: int = 30):
        """
        V8 SEEK OPTIMIZATION: Navigate between keyframes using Cues index.
        This is MUCH faster than reading every frame for distant forward seeks.

        Strategy:
        1. Check IDR cache first - if we have a cached IDR near target, use it
        2. Get sorted list of keyframe timestamps from Cues index
        3. Jump directly to each keyframe using seekToCue() and check for IDR NAL
           Also checks for MVC IDR (type 20 with non_idr_flag=0) and recovery points.

        Returns (au_data, timestamp_seconds, is_recovery_only) or (None, None, False).
        is_recovery_only=True means the frame has SPS/PPS but no IDR slice (needs extended priming).
        """
        # V8 FAST PATH: Check IDR cache first
        cached_idr = self._find_nearest_cached_idr(target_s)
        if cached_idr is not None and (target_s - cached_idr) < 10.0:
            # We have a cached IDR within 10 seconds before target
            # Seek directly to it without Cues navigation
            logger.info(f"[MVC-THREAD] V8: Using cached IDR at {cached_idr:.3f}s (target: {target_s:.3f}s)")
            cached_idr_ms = int(cached_idr * 1000)
            if hasattr(self.demuxer, 'seek') and self.demuxer.seek(cached_idr_ms):
                try:
                    success, base, dep = self.demuxer.read_next_frame_pair()
                    if success:
                        au_data = bytearray()
                        if base and 'data' in base: au_data.extend(bytes(base['data']))
                        # Verify it's still an IDR (sanity check) - check type 5 AND MVC IDR (type 20)
                        for nal_data in find_nal_units(au_data):
                            sc_len = 4 if nal_data.startswith(b'\x00\x00\x00\x01') else 3
                            if len(nal_data) > sc_len:
                                nal_type = nal_data[sc_len] & 0x1F
                                if nal_type == NAL_TYPE_IDR or is_mvc_idr_nal(nal_data, sc_len):
                                    timestamp = cached_idr
                                    if base and 'timestamp' in base:
                                        timestamp = float(base['timestamp']) / 1000.0
                                    logger.info(f"[MVC-THREAD] V8: *** IDR from cache at {timestamp:.3f}s (FAST PATH) ***")
                                    full_au_data = bytearray()
                                    if base and 'data' in base: full_au_data.extend(bytes(base['data']))
                                    if dep and 'data' in dep: full_au_data.extend(bytes(dep['data']))
                                    return full_au_data, timestamp, False  # True IDR
                except Exception as e:
                    logger.warning(f"[MVC-THREAD] V8: Cache seek failed: {e}")
            # If cached seek failed, fall through to Cues navigation

        if not hasattr(self.demuxer, 'getCuesTimestamps') or not hasattr(self.demuxer, 'seekToCue'):
            logger.info("[MVC-THREAD] V8: Cues navigation not available, falling back to linear scan")
            return None, None, False

        try:
            cues_timestamps_ms = self.demuxer.getCuesTimestamps()
            if not cues_timestamps_ms or len(cues_timestamps_ms) == 0:
                logger.info("[MVC-THREAD] V8: Empty Cues index, falling back to linear scan")
                return None, None, False

            target_ms = int(target_s * 1000)
            cues_timestamps_ms = list(cues_timestamps_ms)  # Ensure it's a list
            logger.info(f"[MVC-THREAD] V8: Cues index has {len(cues_timestamps_ms)} keyframes, target={target_ms}ms")

            # Find starting index: closest keyframe at or before target
            start_idx = 0
            for i, ts in enumerate(cues_timestamps_ms):
                if ts <= target_ms:
                    start_idx = i
                else:
                    break

            # Try keyframes starting from start_idx, going forward
            keyframes_checked = 0
            for idx in range(start_idx, min(start_idx + max_keyframes, len(cues_timestamps_ms))):
                if self._stop_requested:
                    return None, None, False

                cue_ts_ms = cues_timestamps_ms[idx]
                keyframes_checked += 1

                # Jump directly to this keyframe
                if not self.demuxer.seekToCue(cue_ts_ms):
                    logger.warning(f"[MVC-THREAD] V8: seekToCue({cue_ts_ms}ms) failed")
                    continue

                # Read the frame at this position and check for IDR
                # Read up to 3 frames per keyframe to find IDR within cluster
                for frame_in_cluster in range(3):
                    if self._stop_requested:
                        return None, None, False
                    try:
                        success, base, dep = self.demuxer.read_next_frame_pair()
                        if not success:
                            break
                    except Exception as e:
                        logger.warning(f"[MVC-THREAD] V8: read_next_frame_pair error at cue {cue_ts_ms}ms: {e}")
                        break

                    au_data = bytearray()
                    if base and 'data' in base:
                        au_data.extend(bytes(base['data']))

                    if not au_data:
                        continue

                    # Check if this frame contains an IDR NAL, MVC IDR, or recovery point
                    has_sps = False
                    has_pps = False
                    has_subset_sps = False
                    is_true_idr = False

                    for nal_data in find_nal_units(au_data):
                        sc_len = 4 if nal_data.startswith(b'\x00\x00\x00\x01') else 3
                        if len(nal_data) > sc_len:
                            nal_type = nal_data[sc_len] & 0x1F
                            # Track parameter sets for recovery point detection
                            if nal_type == 7:  # SPS
                                has_sps = True
                            elif nal_type == 8:  # PPS
                                has_pps = True
                            elif nal_type == 15:  # Subset SPS (MVC)
                                has_subset_sps = True
                            # Check for standard IDR (type 5) OR MVC IDR (type 20 with idr flag)
                            if nal_type == NAL_TYPE_IDR or is_mvc_idr_nal(nal_data, sc_len):
                                is_true_idr = True

                    # Accept true IDR OR recovery point (SPS+PPS+SubsetSPS with keyframe flag)
                    is_keyframe = base.get('isKeyframe', False) if base else False
                    is_recovery_point = is_keyframe and has_sps and has_pps and has_subset_sps
                    is_idr = is_true_idr or is_recovery_point

                    if is_idr:
                        timestamp = cue_ts_ms / 1000.0
                        if base and 'timestamp' in base:
                            timestamp = float(base['timestamp']) / 1000.0

                        is_recovery_only = (is_recovery_point and not is_true_idr)
                        idr_type = "recovery point" if is_recovery_only else "IDR"
                        logger.info(f"[MVC-THREAD] V8: *** {idr_type} found via Cues at keyframe #{keyframes_checked} (ts={timestamp:.3f}s) ***")

                        # Cache this IDR position (only for true IDR)
                        if not is_recovery_only:
                            self._cache_idr_position(timestamp)

                        # Return full AU data
                        full_au_data = bytearray()
                        if base and 'data' in base: full_au_data.extend(bytes(base['data']))
                        if dep and 'data' in dep: full_au_data.extend(bytes(dep['data']))
                        return full_au_data, timestamp, is_recovery_only

            logger.warning(f"[MVC-THREAD] V8: No IDR found in {keyframes_checked} keyframes via Cues")
            return None, None, False

        except Exception as e:
            logger.error(f"[MVC-THREAD] V8: Cues navigation error: {e}")
            return None, None, False

    def scan_for_idr(self, max_frames=1000, tolerance=5.0, skip_timestamp_check=False):
        """
        Scans the demuxer stream to find the first Access Unit containing an IDR slice
        OR a valid random access point (SPS+PPS+Subset SPS with isKeyframe=True).
        V7b FIX: After a seek, this will skip IDR frames until finding one >= base_timestamp.
        V7b+ CUES FIX: If skip_timestamp_check=True, accept the first IDR without timestamp validation.
        This is crucial for Cues-based seeks where the demuxer already positioned us correctly.
        Returns (au_data, timestamp_seconds, is_recovery_only) or (None, None, False).
        is_recovery_only=True means the frame has SPS/PPS but no IDR slice (needs extended priming).
        """
        logger.info(f"[MVC-THREAD] Scanning for IDR frame (max {max_frames} frames, tol={tolerance}s, skip_ts_check={skip_timestamp_check})...")
        skipped_idr_count = 0
        _scan_diag = os.environ.get('SYLC_SCAN_DIAG')
        _scan_t0 = time.time() if _scan_diag else 0.0

        for i in range(max_frames):
            if self._stop_requested or getattr(self, '_cleanup_in_progress', False):
                logger.info("[MVC-THREAD] Scan aborted by stop request")
                return None, None, False

            try:
                _c0 = time.time() if _scan_diag else 0.0
                success, base, dep = self.demuxer.read_next_frame_pair()
                if _scan_diag:
                    _dt = time.time() - _c0
                    if _dt > 1.0 or i % 100 == 0:
                        logger.error(f"[SCAN-DIAG] read#{i} dt={_dt:.2f}s total={time.time()-_scan_t0:.1f}s "
                                     f"base={len(base['data']) if base and 'data' in base else 0} "
                                     f"dep={len(dep['data']) if dep and 'data' in dep else 0} ok={success}")
                if not success:
                    logger.warning("[MVC-THREAD] End of stream reached while scanning for IDR.")
                    return None, None, False
            except Exception as e:
                logger.error(f"[MVC-THREAD] Demuxer error while scanning for IDR: {e}")
                return None, None, False

            au_data = bytearray()
            base_size = len(base['data']) if base and 'data' in base else 0
            dep_size = len(dep['data']) if dep and 'data' in dep else 0
            if _scan_diag and i < 5:  # Log first 5 frames for diagnostic (env-gated: console I/O is slow)
                logger.info(f"[MVC-DIAG] Scan frame {i}: base={base_size} bytes, dep={dep_size} bytes")
                # Dump first bytes to see if Annex B start codes are present
                if base and 'data' in base and base_size > 0:
                    raw = bytes(base['data'])
                    hex_preview = raw[:32].hex() if len(raw) >= 32 else raw.hex()
                    # Count start codes
                    sc_count = raw.count(b'\x00\x00\x00\x01') + raw.count(b'\x00\x00\x01')
                    logger.info(f"[MVC-DIAG] base first 32 bytes: {hex_preview} (start codes: {sc_count})")
                if dep and 'data' in dep and dep_size > 0:
                    raw = bytes(dep['data'])
                    hex_preview = raw[:32].hex() if len(raw) >= 32 else raw.hex()
                    sc_count = raw.count(b'\x00\x00\x00\x01') + raw.count(b'\x00\x00\x01')
                    logger.info(f"[MVC-DIAG] dep first 32 bytes: {hex_preview} (start codes: {sc_count})")
            if base and 'data' in base:
                au_data.extend(bytes(base['data']))
            # GRAVITY FIX: SubsetSPS (NAL 15) lives in the dep view for some MVC
            # encoders (Gravity 3D BD). To detect recovery points correctly we must
            # scan NALs from BOTH views, otherwise has_subset_sps stays False and
            # the recovery point check fails.
            if dep and 'data' in dep:
                au_data.extend(bytes(dep['data']))

            if au_data:
                # Check for IDR frames: NAL type 5 (standard H.264 IDR)
                # OR NAL type 20 with non_idr_flag=0 (MVC IDR)
                # OR recovery point frames: SPS+PPS+SubsetSPS with container keyframe flag
                has_sps = False
                has_pps = False
                has_subset_sps = False
                is_true_idr = False

                nal_type_summary = []
                for nal_data in find_nal_units(au_data):
                    sc_len = 4 if nal_data.startswith(b'\x00\x00\x00\x01') else 3
                    if len(nal_data) > sc_len:
                        nal_type = nal_data[sc_len] & 0x1F
                        if _scan_diag and i < 5:
                            nal_type_summary.append(nal_type)
                        # Track parameter sets for recovery point detection
                        if nal_type == 7:  # SPS
                            has_sps = True
                        elif nal_type == 8:  # PPS
                            has_pps = True
                        elif nal_type == 15:  # Subset SPS (MVC)
                            has_subset_sps = True
                        # Check for standard IDR (type 5) OR MVC IDR (type 20 with idr flag)
                        if (nal_type == NAL_TYPE_IDR) or is_mvc_idr_nal(nal_data, sc_len):
                            is_true_idr = True
                if _scan_diag and i < 5:
                    logger.info(f"[MVC-IDR-DIAG] Frame {i}: NALs={nal_type_summary[:30]}... sps={has_sps} pps={has_pps} subset_sps={has_subset_sps} keyframe={base.get('isKeyframe', False) if base else False} true_idr={is_true_idr}")

                # Accept true IDR OR recovery point (SPS+PPS+SubsetSPS with keyframe flag)
                is_keyframe = base.get('isKeyframe', False) if base else False
                is_recovery_point = is_keyframe and has_sps and has_pps and has_subset_sps
                # GRAVITY FIX: For Cues-based seeks (skip_timestamp_check=True), the demuxer
                # has already positioned us at a valid restart point. Some BD MVC encoders
                # (Gravity 3D BD) use open GOPs with recovery points but Matroska doesn't
                # always propagate the isKeyframe flag. In seek mode, accept SPS+PPS+SubsetSPS
                # as a recovery point regardless of isKeyframe — the decoder can restart from
                # this point even without a true IDR.
                if not is_recovery_point and skip_timestamp_check and has_sps and has_pps and has_subset_sps:
                    is_recovery_point = True
                    logger.info(f"[MVC-THREAD] Frame {i}: Lenient recovery point (SPS+PPS+SubsetSPS, isKeyframe={is_keyframe}) — accepted because of seek mode")
                is_idr = is_true_idr or is_recovery_point

                if is_recovery_point and not is_true_idr:
                    logger.info(f"[MVC-THREAD] Frame {i}: Recovery point detected (SPS+PPS+SubsetSPS, keyframe={is_keyframe})")

                if is_idr:
                    # Extract timestamp BEFORE checking
                    timestamp = None
                    if base and 'timestamp' in base:
                        timestamp = float(base['timestamp']) / 1000.0
                    elif dep and 'timestamp' in dep:
                        timestamp = float(dep['timestamp']) / 1000.0

                    # V7b+ CUES FIX: If skip_timestamp_check is True, accept the first IDR immediately
                    # This is used after Cues-based seeks where the C++ demuxer already positioned us correctly
                    # and timestamps may be corrupted in the cluster
                    if skip_timestamp_check:
                        is_recovery_only = (is_recovery_point and not is_true_idr)
                        idr_type = "recovery point" if is_recovery_only else "IDR"
                        logger.info(f"[MVC-THREAD] *** {idr_type} found at scan position {i} (TS: {timestamp}) - ACCEPTED (Cues-based seek, timestamp check skipped) ***")
                        # V8 IDR INDEX: Cache this IDR for future seeks
                        if timestamp is not None:
                            self._cache_idr_position(timestamp)
                        # Found suitable IDR. Return the full AU data (base + dependent) for priming.
                        full_au_data = bytearray()
                        if base and 'data' in base: full_au_data.extend(bytes(base['data']))
                        if dep and 'data' in dep: full_au_data.extend(bytes(dep['data']))
                        return full_au_data, timestamp, is_recovery_only

                    # V7b CRITICAL FIX: Skip IDR frames that are WAY before base_timestamp
                    # This happens if demuxer seek failed or reset to 0.
                    # We allow a tolerance (default 5.0s) because seek lands on keyframe BEFORE target.
                    if timestamp is not None and timestamp < (self.base_timestamp - tolerance):
                        skipped_idr_count += 1
                        if skipped_idr_count == 1 or skipped_idr_count % 50 == 0:
                            logger.info(f"[MVC-THREAD] Skipping IDR at {timestamp:.3f}s (target: {self.base_timestamp:.3f}s)")
                        continue  # Continue to next frame

                    is_recovery_only = (is_recovery_point and not is_true_idr)
                    idr_type = "recovery point" if is_recovery_only else "IDR"
                    logger.info(f"[MVC-THREAD] *** {idr_type} found at scan position {i} (TS: {timestamp}) ***")
                    # V8 IDR INDEX: Cache this IDR for future seeks
                    if timestamp is not None:
                        self._cache_idr_position(timestamp)
                    # Found suitable IDR. Return the full AU data (base + dependent) for priming.
                    full_au_data = bytearray()
                    if base and 'data' in base: full_au_data.extend(bytes(base['data']))
                    if dep and 'data' in dep: full_au_data.extend(bytes(dep['data']))

                    return full_au_data, timestamp, is_recovery_only

        logger.error(f"[MVC-THREAD] No IDR frame found after scanning {max_frames} frames.")
        return None, None, False

    def run(self):
        # Initialize demuxer first
        if not self._init_demuxer():
            return

        try:
            if self._use_native_pipeline:
                # Check if demuxer supports native ring buffer
                if not hasattr(self.demuxer, "read_next_into_ring"):
                    logger.warning(
                        "[MVC-THREAD] Demuxer instance does not expose read_next_into_ring, falling back to Python path.")
                    self._use_native_pipeline = False
                else:
                    success = False
                    try:
                        success = self._run_native_pipeline()
                    except Exception as e:
                        logger.error(f"[MVC-THREAD] Native pipeline crashed: {e}", exc_info=True)
                        success = False
                    if success:
                        return  # Finally block will still execute
                    logger.warning("[MVC-THREAD] Native pipeline unavailable or failed, falling back to ctypes path.")

            self._run_ctypes_pipeline()
        finally:
            # V54: stop the presenter thread before tearing down the decoder/demuxer.
            self._stop_presenter()
            # NEW: Close demuxer INSIDE the thread to avoid race conditions
            if self.demuxer:
                try:
                    self.demuxer.close()
                    logger.info("[MVC-THREAD] Demuxer closed in thread")
                except:
                    pass  # Ignore close errors
                self.demuxer = None

    def _run_ctypes_pipeline(self):
        if not edge264:
            self.error.emit("edge264 library not loaded.")
            return

        # Demuxer is already initialized by _init_demuxer() in run()

        try:
            # SYNC FIX: Use n_threads=0 (fully synchronous) for MVC decode
            # n_threads=1 creates an async worker thread that causes DPB overflow (ENOBUFS)
            # because frames aren't drained fast enough between NAL submissions.
            # With n_threads=0, each NAL is processed inline and frames are immediately
            # available for draining, preventing DPB saturation. Verified: 2000 AUs,
            # 2000 MVC frames, 0 errors with n_threads=0.
            # MULTITHREADED DECODE (2026-07-13): enabled after fixing the three MT
            # bugs in edge264 (MT-fatal DPB assert; unset_currPic force-complete
            # racing the workers' remaining_mbs accounting; edge264_free teardown
            # deadlock on winpthread). Validated on real content: bit-exact output
            # (per-POC sha over 192 frames), 80 -> 146 fps at 4 threads. Workers
            # deliver frames with POC-order jitter — absorbed by the sorted
            # frame_buffer reorder (sort_key, REORDER_DEPTH). The existing
            # predrain/ENOBUFS machinery matches the MT drain-in-feeder contract.
            # 0 = legacy synchronous decode (env override to compare/rollback).
            try:
                alloc_threads = int(os.environ.get('SYLC_EDGE264_THREADS', '4'))
            except ValueError:
                alloc_threads = 4
            alloc_threads = max(0, min(16, alloc_threads))
            self._alloc_threads = alloc_threads  # V51: remember threading mode to skip pointless async drain-retries
            logger.info(f"[MVC-THREAD] Alloc with {alloc_threads} thread(s)")
            with edge264_session_lock:
                ptr = edge264.edge264_alloc(alloc_threads, None, None, 0, None, None, None)
            if not ptr:
                raise MemoryError("Alloc failed")
            self.decoder = ctypes.c_void_p(ptr)
            logger.info("[MVC-THREAD] Decoder allocated")

            # CRITICAL FIX: Increase buffer pool size for MVC multi-threading
            # MVC with 8 threads needs: 8 threads x ~50 frames DPB x ~20 NALs/frame = ~1000 buffers
            # Old value of 64 caused premature garbage collection -> EDGE264 use-after-free -> EFAULT crash
            self.au_buffers = deque(maxlen=2000)

            # CRITICAL FIX: Initial seek if restarting/seeking to a specific time
            # The decoder thread needs to position the demuxer before scanning for IDR
            if self.base_timestamp > 0.0:
                logger.info(f"[MVC-THREAD] Initial seek to {self.base_timestamp:.3f}s...")
                try:
                    if hasattr(self.demuxer, 'seek'):
                        self.demuxer.seek(int(self.base_timestamp * 1000))
                except Exception as e:
                    logger.warning(f"[MVC-THREAD] Initial seek failed: {e}")

            # 1. Scan for IDR
            # V7b CRITICAL FIX: If we performed an initial seek (base_timestamp > 0),
            # we MUST skip the timestamp check. The demuxer (using Cues) is already
            # at the correct cluster, but the cluster timestamp might be garbage
            # (e.g. negative/uninitialized), causing scan_for_idr to reject valid IDRs.
            skip_check = (self.base_timestamp > 0.0)
            idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(skip_timestamp_check=skip_check)
            if not idr_au_data:
                self.error.emit("Unable to find an IDR frame to initialize the decoder.")
                return

            # V7b CRITICAL FIX: Only update base_timestamp from IDR if we're NOT at initial load (0.0)
            # At initial load, keep base_timestamp at 0.0 to ensure playback starts from beginning
            # After a seek, base_timestamp is already set to the seek target and should stay there
            #
            # V10 SSIF FIX: For Blu-ray SSIF files, the first IDR may be several seconds into the stream
            # (due to M2TS GOP structure). Since we can't decode frames before the IDR, video playback
            # actually starts at the IDR timestamp. We MUST seek MPV audio to match.
            is_ssif = self.filepath.lower().endswith('.ssif')

            if idr_timestamp is not None and self.base_timestamp > 0.1:
                # This is a seek (not initial load), log the IDR position for debug
                logger.info(f"[MVC-THREAD] IDR found at {idr_timestamp:.3f}s (keeping base_timestamp at {self.base_timestamp:.3f}s)")
            elif idr_timestamp is not None and idr_timestamp > 0.5:
                if is_ssif:
                    # V10 SSIF FIX: IDR is not at start - we MUST seek MPV audio to match
                    # The video can only be decoded from the IDR onwards, so audio must start there too
                    logger.info(f"[MVC-THREAD] SSIF V10: IDR at {idr_timestamp:.3f}s - will seek MPV audio to match")
                    self.base_timestamp = idr_timestamp
                    self._emit_ssif_sync_after_priming = True  # Flag to emit signal after priming
                else:
                    # Non-SSIF (MKV): IDR is after start, keep base_timestamp for normal sync
                    logger.info(f"[MVC-THREAD] IDR at {idr_timestamp:.3f}s, base_timestamp unchanged (non-SSIF)")
            else:
                # Initial load with IDR at ~0: keep base_timestamp at 0.0
                logger.info(f"[MVC-THREAD] Initializing base_timestamp to 0.000s (IDR found at {idr_timestamp if idr_timestamp else 'unknown'}s)")

            # 2. Inject codec private (SPS/PPS) BEFORE priming
            # CRITICAL FIX: Without SPS/PPS, edge264 cannot decode slices and will crash
            # V33h FIX: Set priming flag BEFORE header injection to prevent pre-drain
            # Pre-drain during header injection causes get_frame calls before Subset SPS is decoded,
            # resulting in ssps.BitDepth_Y=0 and no MVC frames.
            self._priming_in_progress = True
            try:
                if hasattr(self.demuxer, 'get_codec_private'):
                    codec_private = self.demuxer.get_codec_private()
                    if codec_private:
                        codec_private_bytes = bytes(codec_private)
                        logger.info(f"[MVC-THREAD] Injecting CodecPrivate ({len(codec_private_bytes)} bytes) before priming...")

                        # Check if already Annex-B format (starts with start code)
                        is_annexb = (codec_private_bytes[:4] == b'\x00\x00\x00\x01' or
                                     codec_private_bytes[:3] == b'\x00\x00\x01')

                        if is_annexb:
                            # Already Annex-B (from SSIF streaming) - use directly
                            logger.info("[MVC-THREAD] CodecPrivate is already Annex-B format")
                            annexb_headers = codec_private_bytes
                        else:
                            # AVCC format (from MKV) - needs conversion
                            logger.info("[MVC-THREAD] CodecPrivate is AVCC format, converting...")
                            annexb_headers = self._convert_avcc_to_annexb(codec_private_bytes)

                        if annexb_headers:
                            injected_count = 0
                            for h_nal in find_nal_units(annexb_headers):
                                sc = 4 if h_nal.startswith(b'\x00\x00\x00\x01') else 3
                                nal_type = h_nal[sc] & 0x1F if len(h_nal) > sc else 0
                                logger.debug(f"[MVC-THREAD] Injecting NAL type {nal_type} ({len(h_nal)} bytes)")
                                self._push_nal_direct(h_nal, sc, force=True)
                                injected_count += 1
                            logger.info(f"[MVC-THREAD] CodecPrivate injected successfully ({injected_count} NALs)")
                        else:
                            logger.warning("[MVC-THREAD] CodecPrivate conversion returned empty")
                    else:
                        logger.warning("[MVC-THREAD] No CodecPrivate available from demuxer")
            except Exception as e:
                logger.warning(f"[MVC-THREAD] CodecPrivate injection error: {e}")

            # 3. Prime decoder
            # MVC DPB FILL FIX: MVC decoder needs ~20+ frames to fill the Decoded Picture Buffer
            # before outputting valid frames. With insufficient priming, first frames have Y=0 (black).
            # Diagnostic showed frames 1-19 had Y=0, frame 20+ had actual content.
            # True IDR: 25 AUs (fills DPB), Recovery Point: 100 AUs (~4 seconds at 24fps)
            # V33u FIX: DPB can only hold ~16 MVC frame pairs. Priming > 15 overflows.
            # Recovery point needs more priming but NOT 100 - that just causes ENOBUFS deadlock.
            if is_recovery_only:
                logger.warning("[MVC-THREAD] RECOVERY POINT (no IDR) - using extended priming (15 AUs, V33u cap)")
                PRIME_AU_COUNT = 15  # V33u: Reduced from 100 - DPB limit
            else:
                PRIME_AU_COUNT = 10  # V33q FIX: Reduced from 25 to prevent DPB overflow - leave room for main loop

            logger.info(f"[MVC-THREAD] Priming decoder with {PRIME_AU_COUNT} AUs...")
            self._priming_in_progress = True  # V16 FIX: Mark priming phase - increased ENOBUFS tolerance
            prime_aus = [idr_au_data]
            # V60 SYNC-PTS: the scan-returned IDR AU is the first fed display frame
            self._push_pts_value(idr_timestamp if idr_timestamp is not None else self.base_timestamp)
            for _ in range(PRIME_AU_COUNT - 1):
                if self._stop_requested: return
                success, base, dep = self.demuxer.read_next_frame_pair()
                if not success: break
                au_data = bytearray()
                base_size = len(base['data']) if base and 'data' in base else 0
                dep_size = len(dep['data']) if dep and 'data' in dep else 0
                logger.debug(f"[MVC-DIAG] Frame pair: base={base_size} bytes, dep={dep_size} bytes")
                if base and 'data' in base: au_data.extend(bytes(base['data']))
                if dep and 'data' in dep: au_data.extend(bytes(dep['data']))
                if au_data:
                    prime_aus.append(au_data)
                    self._push_pair_pts(base, dep)  # V60 SYNC-PTS

            priming_drain_total = 0
            for i, au_data in enumerate(prime_aus):
                if self._stop_requested or getattr(self, '_needs_full_reset', False): 
                    if getattr(self, '_needs_full_reset', False):
                        logger.error("[MVC-THREAD] V30: Reset flagged during priming, stopping.")
                    return
                logger.info(f"[MVC-THREAD] Priming with AU #{i + 1}/{len(prime_aus)} ({len(au_data)} bytes)...")
                self._process_au_data(bytes(au_data))

                # V33k FIX: Simplified priming drain - matches working test script
                # Previous code used edge264_is_frame_ready polling which doesn't work
                # in synchronous mode. Just bump and get_frame like the test script.
                if self.decoder:
                    try:
                        edge264.edge264_bump_frames(self.decoder)
                        tmp_frame = Edge264Frame()
                        drain_count = 0
                        while not self._stop_requested:
                            ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(tmp_frame), 0)
                            if ret != 0:
                                break  # No more frames available
                            drain_count += 1
                            priming_drain_total += 1
                        if drain_count > 0:
                            logger.info(f"[MVC-THREAD] Priming AU #{i+1}: Drained {drain_count} frames")
                    except (OSError, RuntimeError) as e:
                        logger.warning(f"[MVC-THREAD] V33k: Priming drain error (ignored): {e}")

            logger.info(f"[MVC-THREAD] Priming complete: {len(prime_aus)} AUs processed, {priming_drain_total} frames drained")
            # V60 SYNC-PTS: the drained frames were DISCARDED (never presented) —
            # they consumed the earliest display slots, so burn their PTS entries.
            self._pts_discard_credit += priming_drain_total
            self._priming_in_progress = False  # V16 FIX: End priming phase

            if self._fatal_error:
                logger.error("[MVC-THREAD] Fatal error during priming. Stopping.")
                return

            # V33i FIX: REMOVED V31 flush - it was draining all frames from DPB after priming
            # The flush put the decoder into a "needs more AUs" state, returning 122 (ENOBUFS)
            # in the main loop, which prevented any frames from being output.
            # Priming already fills the DPB sufficiently - no flush needed.
            # OLD CODE (V31):
            # logger.info("[MVC-THREAD] V31: Flushing to force frame output...")
            # if self.decoder:
            #     try:
            #         flush_ret = edge264.edge264_flush(self.decoder)
            #         logger.info(f"[MVC-THREAD] V31: edge264_flush returned {flush_ret}")
            #     except Exception as e:
            #         logger.warning(f"[MVC-THREAD] V31: flush exception: {e}")

            # V11 UNIFIED SYNC: Same logic for both SSIF and MKV
            # Key insight: Audio is the master clock. Video must follow audio.
            # Don't try to command MPV where to play - instead, read where audio IS
            # and set video timestamps to match.
            if getattr(self, '_emit_ssif_sync_after_priming', False):
                self._emit_ssif_sync_after_priming = False
                logger.info(f"[MVC-THREAD] SSIF V11: IDR at {self.base_timestamp:.3f}s - will align video to audio")

            # Wait for audio clock from GUI thread
            sync_wait_start = time.time()
            current_audio = 0.0
            wait_iterations = 0
            last_update_ts = 0.0

            # Phase 1: Wait for any valid audio clock
            # V43 FIX: Accept audio clock >= 0 (not > 0.05) when update_ts is recent.
            # The previous > 0.05 threshold rejected position 0.0, causing misalignment
            # when files start from the beginning. A recent update_ts proves the clock is live.
            while (time.time() - sync_wait_start) < 2.0:
                self._audio_mutex.lock()
                current_audio = self._external_audio_clock
                audio_update_ts = self._external_audio_clock_update_ts
                self._audio_mutex.unlock()

                wait_iterations += 1
                if audio_update_ts > 0 and (time.time() - audio_update_ts) < 0.5:
                    # V43: Accept clock at position 0 if update is recent
                    if current_audio >= 0.0:
                        last_update_ts = audio_update_ts
                        logger.info(f"[MVC-THREAD] V43 SYNC: Audio clock found after {wait_iterations} iterations: {current_audio:.3f}s")
                        break
                try:
                    time.sleep(0.05)
                except (OSError, Exception):
                    pass  # V13: Ignore Windows threading exceptions

            # Phase 2: Wait for ONE more fresh update to get the latest value
            if last_update_ts > 0:
                fresh_wait_start = time.time()
                while (time.time() - fresh_wait_start) < 0.2:  # Max 200ms
                    self._audio_mutex.lock()
                    new_audio = self._external_audio_clock
                    new_update_ts = self._external_audio_clock_update_ts
                    self._audio_mutex.unlock()

                    if new_update_ts > last_update_ts:
                        current_audio = new_audio
                        logger.info(f"[MVC-THREAD] V43 SYNC: Fresh audio clock: {current_audio:.3f}s")
                        break
                    try:
                        time.sleep(0.010)  # Poll quickly
                    except (OSError, Exception):
                        pass  # V13: Ignore Windows threading exceptions

            # V43 FIX: Accept audio clock at position 0 (file start).
            # Only reject if we never got any update at all (last_update_ts == 0).
            if last_update_ts > 0:
                self.base_timestamp = current_audio
                logger.info(f"[MVC-THREAD] V43 SYNC: Aligned base_timestamp to audio position: {current_audio:.3f}s")
            else:
                logger.warning(f"[MVC-THREAD] V43 SYNC: No audio clock after {wait_iterations} iterations "
                             f"(audio={current_audio:.3f}s, update_age={(time.time() - last_update_ts) if last_update_ts > 0 else -1:.3f}s)")
                logger.info(f"[MVC-THREAD] V43 SYNC: Using base_timestamp={self.base_timestamp:.3f}s")

            # Set start_time and emit signal to unpause MPV
            self.start_time = time.time() - self.base_timestamp
            logger.info(f"[MVC-THREAD] V11 SYNC: Emitting seekIDRFound({self.base_timestamp:.3f}s) to unpause MPV")
            self.seekIDRFound.emit(self.base_timestamp)
            try:
                time.sleep(0.05)  # Brief delay for signal to be processed
            except (OSError, Exception):
                pass  # V13: Ignore Windows threading exceptions

            # 3. Main decoding loop
            logger.info("[MVC-THREAD] Entering main decode loop...")
            # V54: hand presentation to the dedicated presenter thread for steady-state
            # playback. From here the decode loop only PRODUCES frames; the presenter
            # DRAINS them, so an I-frame decode spike no longer freezes the picture.
            self._ensure_presenter()
            frame_struct = Edge264Frame()
            get_frame_call_count = 0

            # Flag for aggressive keyframe draining
            should_decode_next = True

            # V33i DIAG: Track loop iterations
            _diag_loop_iter = 0

            while not self._stop_requested and not self._cleanup_in_progress:
                _diag_loop_iter += 1
                if _diag_loop_iter <= 10 or _diag_loop_iter % 100 == 0:
                    logger.debug(f"[V33o] Main loop iter={_diag_loop_iter} frame_count={self.frame_count}")

                # V14 GRACEFUL ENDING: Check cleanup flag
                if self._cleanup_in_progress:
                    logger.info("[MVC-THREAD] V14: Cleanup in progress, exiting ctypes loop")
                    break

                # --- SEEK HANDLING ---
                self.mutex.lock()
                if self._seek_requested:
                    # Mark seek as in progress
                    self._seek_in_progress = True
                    target = self._seek_target
                    self._seek_requested = False
                    # V8 PAUSE FIX: Remember pause state BEFORE seek to restore it after
                    was_paused_before_seek = self._is_paused

                    # V8 PAUSE FIX: Detect if this pause was the GUI's pre-seek pause
                    # If pause() was called very recently (within 200ms), it's the GUI preparing for seek
                    # In that case, the user was actually PLAYING, not paused
                    if was_paused_before_seek and self._pause_timestamp > 0:
                        time_since_pause = time.time() - self._pause_timestamp
                        if time_since_pause < 0.200:  # 200ms threshold
                            was_paused_before_seek = False
                            logger.info(f"[MVC-THREAD] V8: Detected GUI pre-seek pause ({time_since_pause*1000:.0f}ms ago), user was PLAYING")

                    # V8 PAUSE FIX: Clear any stale pause lock from previous crashed seek
                    self._pause_locked = False
                    self.mutex.unlock()

                    logger.info(f"[MVC-THREAD] ========== V8 SEEK START: {target:.3f}s (was_paused={was_paused_before_seek}) ==========")
                    self._seek_perf_t0 = time.time()  # SEEK-PERF: per-phase timing breakdown

                    # ╔═══════════════════════════════════════════════════════════════════╗
                    # ║  V8 INDEX-BASED SYNC: RECREATE decoder instead of FLUSH           ║
                    # ║  Mathematically: RECREATE guarantees a clean initial state        ║
                    # ║  flush() leaves internal threads in an indeterminate state        ║
                    # ╚═══════════════════════════════════════════════════════════════════╝

                    # STEP 1: DESTROY - Free the current decoder cleanly
                    if self.decoder:
                        logger.info("[MVC-THREAD] V8: Destroying old decoder (clean slate)...")

                        # V14 CRASH FIX: Wait for any in-progress frame delivery to complete
                        # This prevents the 0xe24c4a02 crash caused by accessing freed memory
                        if self._frame_delivery_active:
                            logger.info("[MVC-THREAD] V14: Waiting for frame delivery to complete...")
                        self._frame_delivery_lock.acquire()  # Block until delivery completes
                        try:
                            # V8 FIX: self.decoder is already c_void_p, pass byref directly
                            # edge264_free expects POINTER(c_void_p) which is &decoder
                            with edge264_session_lock:
                                edge264.edge264_free(ctypes.byref(self.decoder))
                            logger.info("[MVC-THREAD] V8: edge264_free successful")
                        except Exception as e:
                            logger.warning(f"[MVC-THREAD] V8: edge264_free warning (ignored): {e}")
                        finally:
                            self.decoder = None
                            self._frame_delivery_lock.release()

                    # STEP 2: CLEAR - Empty all buffers (clean state)
                    self.frame_buffer.clear()
                    self.presentation_queue.clear()
                    self.au_buffers.clear()
                    self._reset_pts_pipeline()  # V60 SYNC-PTS

                    # STEP 3: RESET - Reset the counters
                    self.frame_count = 0
                    self.epoch += 1
                    self.last_poc_ordered = -100
                    self.force_next_epoch = False
                    self._consecutive_errors = 0
                    self._efault_count = 0
                    self._startup_frames_emitted = 0  # V7c: Reset startup bypass counter
                    self._v12_frames_emitted = 0  # V43: Reset V12 startup grace period
                    self._v12_hold_start = None  # V43: Reset hold timeout
                    # V59b: START in free-run — right after a seek the mpv audio
                    # clock is stale by construction (seek+restart latency); holding
                    # frames against it froze the picture for the hold duration.
                    # Free-run emits wall-clock paced immediately.
                    # V61: mark it as FROM-SEEK — the disengage test is then
                    # "raw clock landed near base_timestamp", NOT "clock moved
                    # 200ms" (the V53 extrapolation creeps even on a frozen mpv
                    # and faked the movement, re-engaging sync against a dead
                    # clock -> 0.6s hold-timeout churn after every deep seek).
                    self._v12_audio_stale_freerun = True
                    self._v12_freerun_audio0 = None
                    self._v12_freerun_from_seek = True
                    self._v12_freerun_t0 = time.time()
                    # V59b: drop pre-seek PLL state — stale drift distorted the
                    # post-seek pacing (up to ±20% frame time for no reason).
                    self._timing_adjustment_mutex.lock()
                    self._timing_drift_ms = 0.0
                    self._timing_adjustment_mutex.unlock()
                    self._pll_integral = 0.0

                    # ╔═══════════════════════════════════════════════════════════════════╗
                    # ║  V8 SEEK OPTIMIZATION: Use cached IDR positions when available     ║
                    # ║  This speeds up seeks to previously visited areas significantly    ║
                    # ╚═══════════════════════════════════════════════════════════════════╝
                    cached_idr = self._find_nearest_cached_idr(target)
                    optimized_target = target
                    if cached_idr is not None:
                        # Use cached IDR position if it's close enough (within 10s before target)
                        if target - cached_idr <= 10.0:
                            optimized_target = cached_idr
                            logger.info(f"[MVC-THREAD] V8: Using cached IDR at {cached_idr:.3f}s (requested: {target:.3f}s)")

                    # STEP 4: SEEK DEMUXER - Position at the nearest Cue
                    # The Cues timestamp is THE SOURCE OF TRUTH (no correction)
                    cues_timestamp_ms = None
                    success = False
                    try:
                        if hasattr(self.demuxer, 'seek'):
                            # SSIF/M2TS proportional seek needs the duration; mpv may have
                            # reported it only AFTER demuxer init, so re-apply it here (on
                            # this thread) right before seeking. Without it the C++ seek
                            # divides into a zero duration and lands at byte 0 (restart).
                            if hasattr(self.demuxer, 'set_external_duration_ms') and self._media_duration:
                                try:
                                    self.demuxer.set_external_duration_ms(int(float(self._media_duration) * 1000))
                                except Exception:
                                    pass
                            success = self.demuxer.seek(int(optimized_target * 1000))
                            # Retrieve the actual Cues timestamp if available
                            if hasattr(self.demuxer, 'getLastCueTimestamp'):
                                cues_timestamp_ms = self.demuxer.getLastCueTimestamp()
                    except Exception as e:
                        logger.error(f"[MVC-THREAD] V8: Demuxer seek error: {e}")

                    # STEP 5: DETERMINE THE BASE TIMESTAMP
                    # Use the Cues timestamp if available, otherwise target
                    if cues_timestamp_ms is not None and cues_timestamp_ms > 0:
                        self.base_timestamp = cues_timestamp_ms / 1000.0
                        logger.info(f"[MVC-THREAD] V8: Using Cues timestamp: {self.base_timestamp:.3f}s")
                    else:
                        self.base_timestamp = target
                        logger.info(f"[MVC-THREAD] V8: Using target timestamp: {self.base_timestamp:.3f}s")

                    self.start_time = time.time() - self.base_timestamp

                    if success:
                        logger.info("[MVC-THREAD] Demuxer seek successful")
                        self.start_time = time.time() - target

                        # ╔═══════════════════════════════════════════════════════════════════╗
                        # ║  V8 SEEK OPTIMIZATION: Try Cues navigation first (fast path)      ║
                        # ║  This jumps directly between keyframes instead of reading every   ║
                        # ║  frame, making distant forward seeks much faster.                 ║
                        # ╚═══════════════════════════════════════════════════════════════════╝
                        idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr_via_cues(target)

                        # Fallback to linear scan if Cues navigation failed
                        if not idr_au_data:
                            logger.info("[MVC-THREAD] V8: Cues navigation failed, falling back to linear scan")
                            # Re-seek to target for linear scan
                            if hasattr(self.demuxer, 'seek'):
                                self.demuxer.seek(int(target * 1000))
                            # V7b+ CUES FIX: Skip timestamp check on first scan after seek
                            # The C++ demuxer uses Cues index to position directly at the correct cluster
                            # and timestamps may be corrupted, so we accept the first IDR found
                            idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(max_frames=500, skip_timestamp_check=True)

                        # If we reached EOF without an IDR, rewind a bit and retry
                        if not idr_au_data:
                            backoff_ms = 5000  # rewind 5s
                            # Prefer C++ helper if available
                            if hasattr(self.demuxer, "rewind_after_failed_seek_ms"):
                                logger.warning(f"[MVC-THREAD] No IDR after seek. Rewinding {backoff_ms}ms via demuxer helper...")
                                try:
                                    if self.demuxer.rewind_after_failed_seek_ms(int(target * 1000), backoff_ms):
                                        # V7b+ CUES FIX: Also skip timestamp check on fallback (Cues still used)
                                        idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(max_frames=800, tolerance=10.0, skip_timestamp_check=True)
                                except Exception as e:
                                    logger.warning(f"[MVC-THREAD] Backoff seek failed: {e}")
                            elif hasattr(self.demuxer, "rewind_after_failed_seek"):
                                logger.warning(f"[MVC-THREAD] No IDR after seek. Rewinding {backoff_ms}ms and retrying...")
                                try:
                                    if self.demuxer.rewind_after_failed_seek(int(target * 1000), backoff_ms):
                                        # V7b+ CUES FIX: Also skip timestamp check on fallback (Cues still used)
                                        idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(max_frames=800, tolerance=10.0, skip_timestamp_check=True)
                                except Exception as e:
                                    logger.warning(f"[MVC-THREAD] Backoff seek failed: {e}")

                        # One last-chance extended backoff (e.g., 30s) if still nothing
                        if not idr_au_data and target > 0:
                            extended_backoff_ms = min(30000, int(target * 1000))  # up to 30s
                            if hasattr(self.demuxer, "rewind_after_failed_seek_ms"):
                                logger.warning(f"[MVC-THREAD] Still no IDR. Extended rewind {extended_backoff_ms}ms...")
                                try:
                                    if self.demuxer.rewind_after_failed_seek_ms(int(target * 1000), extended_backoff_ms):
                                        # V7b+ CUES FIX: Also skip timestamp check on extended fallback (Cues still used)
                                        idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(max_frames=1200, tolerance=40.0, skip_timestamp_check=True)
                                except Exception as e:
                                    logger.warning(f"[MVC-THREAD] Extended backoff failed: {e}")
                            elif hasattr(self.demuxer, "rewind_after_failed_seek"):
                                logger.warning(f"[MVC-THREAD] Still no IDR. Extended rewind {extended_backoff_ms}ms (py helper)...")
                                try:
                                    if self.demuxer.rewind_after_failed_seek(int(target * 1000), extended_backoff_ms):
                                        # V7b+ CUES FIX: Also skip timestamp check on extended fallback (Cues still used)
                                        idr_au_data, idr_timestamp, is_recovery_only = self.scan_for_idr(max_frames=1200, tolerance=40.0, skip_timestamp_check=True)
                                except Exception as e:
                                    logger.warning(f"[MVC-THREAD] Extended backoff failed: {e}")
                        
                        if idr_au_data:
                            logger.debug(f"[SEEK-PERF] scan(+seek) done +{time.time() - self._seek_perf_t0:.2f}s")
                            # ╔═══════════════════════════════════════════════════════════════════╗
                            # ║  V8 STEP 6: RECREATE - Allocate a fresh clean decoder            ║
                            # ║  Mathematically: alloc() = deterministic initial state           ║
                            # ╚═══════════════════════════════════════════════════════════════════╝
                            logger.info("[MVC-THREAD] V8: Allocating fresh decoder...")
                            try:
                                # SEEK-PERF FIX: this used to hardcode 0 threads, silently
                                # DOWNGRADING the decoder to synchronous mode after the first
                                # seek (and priming single-threaded). Recreate with the same
                                # threading mode as the initial alloc (SYLC_EDGE264_THREADS).
                                alloc_threads = getattr(self, '_alloc_threads', 0)
                                with edge264_session_lock:
                                    ptr = edge264.edge264_alloc(alloc_threads, None, None, 0, None, None, None)
                                if not ptr:
                                    raise MemoryError("V8: Decoder alloc failed")
                                self.decoder = ctypes.c_void_p(ptr)
                                self.au_buffers = deque(maxlen=2000)
                                logger.info("[MVC-THREAD] V8: Fresh decoder allocated successfully")
                            except Exception as e:
                                logger.error(f"[MVC-THREAD] V8: Failed to allocate decoder: {e}")
                                self.seekFinished.emit()
                                self._seek_in_progress = False
                                continue

                            # ╔═══════════════════════════════════════════════════════════════════╗
                            # ║  V8 STEP 7: INJECT HEADERS - CodecPrivate = source of truth      ║
                            # ╚═══════════════════════════════════════════════════════════════════╝
                            try:
                                if hasattr(self.demuxer, 'get_codec_private'):
                                    codec_private = self.demuxer.get_codec_private()
                                    if codec_private:
                                        cp_bytes = bytes(codec_private)
                                        logger.info("[MVC-THREAD] V8: Injecting CodecPrivate headers...")
                                        # SSIF codec_private is already Annex-B; only MKV is AVCC.
                                        # (The seek path used to assume AVCC unconditionally → the SSIF
                                        # Annex-B header was rejected with "Unexpected AVCC version: 0",
                                        # leaving the fresh post-seek decoder with NO SPS/PPS.)
                                        if cp_bytes[:4] == b'\x00\x00\x00\x01' or cp_bytes[:3] == b'\x00\x00\x01':
                                            annexb_headers = cp_bytes
                                        else:
                                            annexb_headers = self._convert_avcc_to_annexb(cp_bytes)
                                        if annexb_headers:
                                            for h_nal in find_nal_units(annexb_headers):
                                                sc = 4 if h_nal.startswith(b'\x00\x00\x00\x01') else 3
                                                self._push_nal_direct(h_nal, sc, force=True)
                            except Exception as e:
                                logger.warning(f"[MVC-THREAD] V8: CodecPrivate injection warning: {e}")

                            # ╔═══════════════════════════════════════════════════════════════════╗
                            # ║  V8 STEP 8: PRIME - Fill DPB with initial frames                 ║
                            # ║  RECOVERY POINT FIX: When seeking to a non-IDR keyframe, we     ║
                            # ║  need MANY MORE frames for P-slices to build correct references.║
                            # ║  True IDR: 25 AUs (DPB fill), Recovery: 100 AUs (~4s at 24fps)   ║
                            # ╚═══════════════════════════════════════════════════════════════════╝
                            # V33u FIX: DPB can only hold ~16 MVC frame pairs.
                            if is_recovery_only:
                                logger.warning("[MVC-THREAD] V8: RECOVERY POINT (no IDR) - using extended priming (15 AUs, V33u cap)")
                                extra_prime_count = 14  # 14 extra = 15 total (V33u: DPB limit)
                            else:
                                extra_prime_count = 9   # 9 extra = 10 total (V33q: DPB fill)

                            prime_aus = [idr_au_data]
                            # V60 SYNC-PTS: first fed display frame = the scan-returned IDR AU
                            self._push_pts_value(idr_timestamp if idr_timestamp is not None else self.base_timestamp)
                            for _ in range(extra_prime_count):
                                if self._stop_requested: break
                                try:
                                    ok, base, dep = self.demuxer.read_next_frame_pair()
                                    if not ok: break
                                    au_data = bytearray()
                                    if base and 'data' in base: au_data.extend(bytes(base['data']))
                                    if dep and 'data' in dep: au_data.extend(bytes(dep['data']))
                                    if au_data:
                                        prime_aus.append(au_data)
                                        self._push_pair_pts(base, dep)  # V60 SYNC-PTS
                                except Exception as e:
                                    logger.warning(f"[MVC-THREAD] V8: Priming AU read error: {e}")
                                    break

                            logger.info(f"[MVC-THREAD] V8: Priming {len(prime_aus)} AUs...")
                            # SEEK-PERF FIX: the old drain retried 5x with 1ms sleeps per AU
                            # (~50-75ms of pure sleep per seek). Feed back-to-back — with
                            # worker threads the decode overlaps the feeding — and drain
                            # non-blockingly between AUs (no retry sleeps): DPB pressure is
                            # already handled by the ENOBUFS machinery in _push_nal_direct.
                            tmp_frame = Edge264Frame()
                            _seek_prime_discards = 0  # V60 SYNC-PTS
                            for i, au_data in enumerate(prime_aus):
                                if self._stop_requested: break
                                self._process_au_data(bytes(au_data), force=True)
                                if self.decoder:
                                    try:
                                        edge264.edge264_bump_frames(self.decoder)
                                        while not self._stop_requested and \
                                                edge264.edge264_get_frame(self.decoder, ctypes.byref(tmp_frame), 0) == 0:
                                            _seek_prime_discards += 1  # drain only, do not present
                                    except (OSError, RuntimeError) as e:
                                        logger.warning(f"[MVC-THREAD] V8: Seek priming drain error (ignored): {e}")
                            # Final settle: give in-flight worker tasks a bounded chance to
                            # land (two consecutive empty passes or ~30ms, whichever first).
                            if self.decoder and not self._stop_requested:
                                try:
                                    empty_passes = 0
                                    settle_deadline = time.time() + 0.030
                                    while empty_passes < 2 and time.time() < settle_deadline:
                                        edge264.edge264_bump_frames(self.decoder)
                                        drained_any = False
                                        while edge264.edge264_get_frame(self.decoder, ctypes.byref(tmp_frame), 0) == 0:
                                            drained_any = True
                                            _seek_prime_discards += 1  # V60 SYNC-PTS
                                        empty_passes = 0 if drained_any else empty_passes + 1
                                        if not drained_any:
                                            time.sleep(0.002)
                                    logger.debug(f"[SEEK-PERF] prime settle done +{time.time() - self._seek_perf_t0:.2f}s")
                                except (OSError, RuntimeError):
                                    pass
                            # V60 SYNC-PTS: drained-and-discarded frames consumed the
                            # earliest display slots — burn their PTS entries.
                            self._pts_discard_credit += _seek_prime_discards

                            # ╔═══════════════════════════════════════════════════════════════════╗
                            # ║  V8 STEP 9: UPDATE base_timestamp with ACTUAL IDR timestamp      ║
                            # ║  The Cues may point to a non-IDR keyframe, so we must use the    ║
                            # ║  timestamp of the actual IDR found by scan_for_idr()             ║
                            # ╚═══════════════════════════════════════════════════════════════════╝
                            # V8 FIX: Use actual IDR timestamp, not Cues timestamp
                            if idr_timestamp is not None and idr_timestamp > 0:
                                old_base = self.base_timestamp
                                self.base_timestamp = idr_timestamp
                                self.start_time = time.time() - self.base_timestamp
                                if abs(old_base - idr_timestamp) > 0.5:
                                    logger.warning(f"[MVC-THREAD] V8: IDR timestamp ({idr_timestamp:.3f}s) differs from Cues ({old_base:.3f}s) - using IDR")
                            logger.info(f"[MVC-THREAD] V8: SEEK COMPLETE - base_timestamp={self.base_timestamp:.3f}s")
                            logger.info(f"[SEEK-PERF] ===== TOTAL decoder seek +{time.time() - self._seek_perf_t0:.2f}s (scan+alloc+prime) =====")
                            logger.info(f"[MVC-THREAD] ========== V8 SEEK END ==========")
                            # V8 CRASH FIX: DON'T emit seekIDRFound here!
                            # Moving to AFTER stabilization to prevent race condition with MPV seek
                            # self.seekIDRFound.emit(self.base_timestamp)  # MOVED TO LINE ~1048
                        else:
                            # CRITICAL FIX: If still no IDR, signal end but don't crash
                            logger.warning("[MVC-THREAD] No IDR found after seek (likely near EOF). Treating as end of stream.")
                            # V7b++++++ CRASH FIX: Clear stabilizing flag even on failure
                            self._decoder_stabilizing = False
                            self.seekFinished.emit()
                            self.decodingFinished.emit()
                            break  # Exit main loop cleanly instead of continuing in invalid state
                    else:
                        logger.warning("[MVC-THREAD] Demuxer seek failed or unsupported")
                        # V7b++++++ CRASH FIX: Clear stabilizing flag on demuxer failure
                        self._decoder_stabilizing = False

                    # ╔═══════════════════════════════════════════════════════════════════╗
                    # ║  V8 POST-SEEK: Simple stabilization (SYNC GATE DISABLED)           ║
                    # ║  SYNC GATE was causing deadlocks and timeouts.                     ║
                    # ║  Instead, we rely on the existing PLL for resynchronization.       ║
                    # ╚═══════════════════════════════════════════════════════════════════╝

                    # V8: Clear stale buffers BEFORE releasing seek mode
                    self.au_buffers.clear()
                    self.presentation_queue.clear()
                    self.frame_buffer.clear()

                    # V8: Small stabilization delay to let audio player catch up
                    # V8 CRASH FIX: Use time.sleep instead of QThread.msleep for stability
                    try:
                        time.sleep(0.100)  # 100ms for audio catchup
                    except Exception:
                        pass  # Windows exception caught

                    # Release seek mode to allow normal decoding (check pending seek first)
                    self.mutex.lock()
                    if self._pending_seek_target is not None:
                        # Another seek was queued
                        self._seek_target = self._pending_seek_target
                        self._pending_seek_target = None
                        self._seek_requested = True
                        # V8 PAUSE FIX: Preserve the original pause state for chained seeks
                        # The GUI may have called resume() via seekFinished signal, corrupting _is_paused
                        # We restore it here so the next seek iteration captures the correct state
                        self._is_paused = was_paused_before_seek
                        self.mutex.unlock()
                        # V8 FIX: Still emit seekFinished so GUI can resume rendering
                        self.seekFinished.emit()
                        # V8 CRASH FIX: Reset frame_struct even for chained seeks
                        frame_struct = Edge264Frame()
                        continue
                    else:
                        self._seek_in_progress = False
                    self.mutex.unlock()

                    # V8 PAUSE FIX: Set pause state BEFORE emitting signals
                    # This allows the GUI to check _is_paused and decide whether to resume MPV
                    self.mutex.lock()
                    if was_paused_before_seek:
                        self._pause_locked = True
                        self._is_paused = True  # Pre-set to paused
                        self.mutex.unlock()
                        logger.info("[MVC-THREAD] V8: PAUSE LOCKED before signals (was paused before)")
                    else:
                        self._is_paused = False  # Pre-set to playing (GUI's pre-seek pause was temporary)
                        self.mutex.unlock()
                        logger.info("[MVC-THREAD] V8: PLAY state set before signals (was playing before)")

                    # V8 CRASH FIX: Emit seekIDRFound AFTER stabilization is complete
                    # This prevents race condition where MPV seek triggers callbacks while
                    # the decoder is still clearing buffers. The atomic sync now happens
                    # when the decoder is fully stabilized.
                    if idr_au_data and idr_timestamp is not None:
                        self.seekIDRFound.emit(self.base_timestamp)

                    # V8 FIX: Emit seekFinished AFTER stabilization is complete
                    # This ensures resume_rendering() is called after buffers are cleared
                    # and _seek_in_progress is False
                    self.seekFinished.emit()
                    logger.info(f"[MVC-THREAD] V8: POST-SEEK stabilization complete")

                    # V13 CRASH FIX: Use time.sleep for GUI wait
                    # Despite V8 comment, threading.Event causes more crashes than time.sleep
                    # based on 0xe24c4a02 exception analysis
                    try:
                        for _ in range(10):  # 10 x 10ms = 100ms total
                            if self._stop_requested or self._seek_requested:
                                break
                            time.sleep(0.010)
                    except (OSError, Exception):
                        pass  # Windows exception caught, continue execution

                    # V8 PAUSE FIX: Restore the EXACT pre-seek state
                    # Whether we were paused or playing, restore that state
                    self.mutex.lock()
                    self._pause_locked = False  # Always unlock to allow future user interaction
                    self._is_paused = was_paused_before_seek  # Restore exact pre-seek state
                    self.mutex.unlock()

                    if was_paused_before_seek:
                        logger.info("[MVC-THREAD] V8: PAUSE state RESTORED after seek (staying paused)")
                        # V8 PAUSE FRAME FIX: Force decode and display ONE frame at the new position
                        # Without this, the screen stays frozen at the old position after pause→seek→pause
                        try:
                            if self.decoder and not self._stop_requested:
                                # Bump and get one frame from the primed decoder
                                edge264.edge264_bump_frames(self.decoder)
                                pause_frame = Edge264Frame()
                                # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                                ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(pause_frame), 1)
                                if ret == 0 and pause_frame.samples[0]:
                                    # Extract YUV planes from the frame (same logic as _deliver_frame_to_gpu_stabilization)
                                    w, h = pause_frame.width_Y, pause_frame.height_Y
                                    if w > 0 and h > 0:
                                        # Use actual chroma dimensions from frame
                                        cw, ch = pause_frame.width_C, pause_frame.height_C
                                        if cw <= 0 or ch <= 0:
                                            cw, ch = w // 2, h // 2
                                        sy, sc = pause_frame.stride_Y, pause_frame.stride_C

                                        def get_plane_copy(base_ptr, pw, ph, ps):
                                            # V13 CRASH FIX: Check cleanup flag
                                            if self._cleanup_in_progress or self._stop_requested:
                                                return np.zeros((ph, pw), dtype=np.uint8)
                                            if not base_ptr or pw <= 0 or ph <= 0 or ps < pw:
                                                return np.zeros((ph, pw), dtype=np.uint8)
                                            return _PLANE_POOL.copy(np.ctypeslib.as_array(base_ptr, shape=(ph, ps))[:, :pw])

                                        y_l = get_plane_copy(pause_frame.samples[0], w, h, sy)

                                        # V44: Unified UV extraction (matches _deliver_frame_to_gpu V38)
                                        # samples[1] = U plane, samples[2] = V plane (side-by-side in memory)
                                        if pause_frame.samples[1] and pause_frame.samples[2]:
                                            try:
                                                u_l = get_plane_copy(pause_frame.samples[1], cw, ch, sc)
                                                v_l = get_plane_copy(pause_frame.samples[2], cw, ch, sc)
                                            except Exception:
                                                u_l = np.full((ch, cw), 128, dtype=np.uint8)
                                                v_l = np.full((ch, cw), 128, dtype=np.uint8)
                                        else:
                                            u_l = np.full((ch, cw), 128, dtype=np.uint8)
                                            v_l = np.full((ch, cw), 128, dtype=np.uint8)

                                        if pause_frame.samples_mvc[0]:
                                            y_r = get_plane_copy(pause_frame.samples_mvc[0], w, h, sy)
                                            if pause_frame.samples_mvc[1] and pause_frame.samples_mvc[2]:
                                                try:
                                                    u_r = get_plane_copy(pause_frame.samples_mvc[1], cw, ch, sc)
                                                    v_r = get_plane_copy(pause_frame.samples_mvc[2], cw, ch, sc)
                                                except Exception:
                                                    u_r = np.full((ch, cw), 128, dtype=np.uint8)
                                                    v_r = np.full((ch, cw), 128, dtype=np.uint8)
                                            else:
                                                u_r = np.full((ch, cw), 128, dtype=np.uint8)
                                                v_r = np.full((ch, cw), 128, dtype=np.uint8)
                                        else:
                                            y_r, u_r, v_r = y_l.copy(), u_l.copy(), v_l.copy()

                                        frame_data = {'left': (y_l, u_l, v_l), 'right': (y_r, u_r, v_r)}
                                        self._emit_single_frame(frame_data)
                                        logger.info("[MVC-THREAD] V8: Pause frame displayed at new position")
                                        # THUMB HARVEST at seek landing: free cache
                                        # fill at the landed IDR (base_timestamp).
                                        try:
                                            from thumbnail_service import planes_to_qimage_320
                                            _timg = planes_to_qimage_320(y_l, u_l, v_l, layout=self._thumb_layout)
                                            if _timg is not None and self.base_timestamp is not None:
                                                self.thumbnailHarvested.emit(float(self.base_timestamp), _timg)
                                        except Exception:
                                            pass
                                    # CRITICAL: Return frame AFTER all data is copied
                                    if self.decoder and pause_frame.return_arg:
                                        try:
                                            edge264.edge264_return_frame(self.decoder, pause_frame.return_arg)
                                        except (OSError, RuntimeError):
                                            pass
                        except Exception as e:
                            logger.warning(f"[MVC-THREAD] V8: Could not display pause frame: {e}")
                    else:
                        logger.info("[MVC-THREAD] V8: PLAY state RESTORED after seek (resuming playback)")

                    # V8 CRASH FIX: Check if a new seek was requested during sleep
                    if self._seek_requested or self._stop_requested:
                        continue  # Skip to handle new seek

                    # V8 CRASH FIX: CRITICAL - Reset frame_struct after decoder recreation!
                    # The old frame_struct contains dangling pointers (samples[0], etc.)
                    # from the FREED old decoder. Using these causes 0xe24c4a02 crash.
                    # We MUST create a fresh Edge264Frame to avoid accessing freed memory.
                    frame_struct = Edge264Frame()
                    logger.debug("[MVC-THREAD] V8: frame_struct reset after SEEK (dangling pointer fix)")

                    continue  # Go back to main loop

                self.mutex.unlock()

                # V8 PAUSE FIX: Check pause state at the START of each loop iteration
                # When paused, we should NOT decode new frames, just wait
                if self._is_paused:
                    # Process presentation queue (which will also pause and wait)
                    if not self._presenter_active:  # V54: presenter thread handles it
                        self._process_presentation_queue()
                    # V13 CRASH FIX: Use time.sleep instead of Event().wait()
                    try:
                        time.sleep(0.010)
                    except (OSError, Exception):
                        pass
                    continue  # Skip all decode work

                loop_start_time = time.time()

                # DIAG: Log main loop iterations
                if not hasattr(self, '_main_loop_iter'):
                    self._main_loop_iter = 0
                self._main_loop_iter += 1
                if self._main_loop_iter <= 20 or self._main_loop_iter % 100 == 0:
                    logger.debug(f"[DIAG] Main loop iter #{self._main_loop_iter}, frame_count={self.frame_count}, fb={len(self.frame_buffer)}, pq={len(self.presentation_queue)}")

                # MEMORY LEAK FIX V5.7 + V7b FLUIDITY: Decoder throttling
                # If presentation queue is almost full, skip decoding to prevent memory buildup
                queue_full_threshold = self.presentation_queue.maxlen * 0.9 if self.presentation_queue.maxlen else 60
                if len(self.presentation_queue) >= queue_full_threshold:
                    # Queue is 90% full, just process frames without decoding more
                    if not self._presenter_active:  # V54: presenter thread handles it
                        self._process_presentation_queue()
                    if self._seek_requested or self._stop_requested:
                        continue
                    # V13 CRASH FIX: Use time.sleep instead of Event().wait()
                    try:
                        if hasattr(self, '_last_display_time'):
                            sleep_time = self.target_frame_time - (time.time() - self._last_display_time)
                            if sleep_time > 0:
                                time.sleep(min(sleep_time, 0.010))
                        else:
                            time.sleep(0.001)
                    except (OSError, Exception):
                        pass
                    continue

                # 1. PRE-DRAIN: Bump and retrieve available frames BEFORE reading next AU
                # This prevents DPB overflow and ensures flow
                # V8 CRASH FIX: Check BOTH seek flags AND decoder validity before ALL edge264 calls
                # V33o: Reduced diagnostic output for PRE-DRAIN
                if should_decode_next and not self._seek_requested and not self._seek_in_progress and self.decoder:
                    try:
                        edge264.edge264_bump_frames(self.decoder)
                    except (OSError, RuntimeError) as e:
                        logger.warning(f"[MVC-THREAD] V8: edge264_bump_frames error (ignored): {e}")

                    # V12 NON-BLOCKING FIX: Add retry loop for non-blocking decode
                    get_frame_attempts = 0
                    pre_drain_retry = 0
                    max_pre_drain_retry = 3  # Less aggressive than POST-DRAIN since this is from previous iteration
                    while not self._stop_requested and not self._seek_requested and not self._seek_in_progress and not self._cleanup_in_progress and self.decoder:
                        try:
                            # Use borrow=1 to keep buffer valid while copying to GPU
                            ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                            get_frame_attempts += 1
                        except (OSError, RuntimeError) as e:
                            logger.warning(f"[MVC-THREAD] edge264_get_frame error: {e}")
                            break

                        if ret == 0:
                            get_frame_call_count += 1
                            pre_drain_retry = 0  # Reset on success
                            try:
                                self._deliver_frame_to_gpu(frame_struct)
                            finally:
                                # V33r: Return frame AFTER copying (borrow=1 requires return)
                                if self.decoder and frame_struct.return_arg:
                                    try:
                                        edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                    except (OSError, RuntimeError):
                                        pass
                            self.frame_count += 1

                            # MEMORY LEAK FIX V5.7 / V52 SMOOTHNESS: periodic gen-0 GC.
                            # Downgraded from full gc.collect() (whole-heap ~70ms spike)
                            # to gc.collect(0) — frames are refcounted, not GC-managed.
                            self._gc_counter += 1
                            if self._gc_counter >= self._gc_interval:
                                gc.collect(0)
                                self._gc_counter = 0
                        else:
                            # V51 PERF: synchronous decode (n_threads=0) produces frames inline,
                            # so a non-zero get_frame here means nothing is pending — break instead
                            # of burning 0.5ms sleeps per retry (per-AU overhead → stutter).
                            if getattr(self, '_alloc_threads', 0) == 0:
                                break
                            # V12 NON-BLOCKING FIX (threaded only): Retry with yield
                            pre_drain_retry += 1
                            if pre_drain_retry >= max_pre_drain_retry:
                                if self.frame_count <= 10 or self.frame_count % 100 == 0:
                                    logger.debug(f"[DIAG] edge264_get_frame ret={ret} after {get_frame_attempts} attempts, total_frames={self.frame_count}")
                                break
                            try:
                                time.sleep(0.0005)  # 0.5ms yield (shorter for PRE-DRAIN)
                                if self.decoder and not self._seek_requested and not self._seek_in_progress:
                                    edge264.edge264_bump_frames(self.decoder)
                            except (OSError, RuntimeError):
                                break
                    
                    # V33o: Removed verbose PRE-DRAIN logging

                # 2. Read next AU
                # V8 CRASH FIX: Check BOTH seek flags before demuxer access
                if self._seek_requested or self._seek_in_progress or self._stop_requested:
                    try:
                        time.sleep(0.005)  # 5ms - use time.sleep for stability
                    except Exception:
                        pass
                    continue

                # V46 EOF GUARD: Prevent access violation in C++ demuxer near end-of-stream.
                # The libmatroska EBML parser can crash when trying to read past the last
                # cluster in the MKV file. Detect EOF proactively using frame count and duration.
                # frame_count only counts main-loop frames; the demuxer has also read
                # ~10 priming + 1 IDR-scan frames before the main loop started.
                _DEMUXER_PREREAD = 12
                if self._media_duration and self._media_duration > 0 and self.frame_count > 0:
                    estimated_pos = (self.frame_count + _DEMUXER_PREREAD) * self.target_frame_time
                    if estimated_pos >= self._media_duration - (2 * self.target_frame_time):
                        logger.info(f"[MVC-THREAD] V46 EOF: position {estimated_pos:.3f}s >= duration {self._media_duration:.3f}s")
                        success, base, dep = False, None, None
                    else:
                        try:
                            success, base, dep = self.demuxer.read_next_frame_pair()
                        except (RuntimeError, OSError) as e:
                            if self._seek_requested or self._seek_in_progress:
                                try:
                                    time.sleep(0.005)
                                except Exception:
                                    pass
                                continue
                            logger.error(f"[MVC-THREAD] Demuxer read failed: {e}")
                            break
                        except Exception as e:
                            logger.error(f"[MVC-THREAD] Demuxer read failed: {e}")
                            break
                else:
                    try:
                        success, base, dep = self.demuxer.read_next_frame_pair()
                    except (RuntimeError, OSError) as e:
                        if self._seek_requested or self._seek_in_progress:
                            try:
                                time.sleep(0.005)
                            except Exception:
                                pass
                            continue
                        logger.error(f"[MVC-THREAD] Demuxer read failed: {e}")
                        break
                    except Exception as e:
                        logger.error(f"[MVC-THREAD] Demuxer read failed: {e}")
                        break

                # V7b FIX: Ensure 1:1 pairing for MVC to prevent queue desync in edge264
                # If we have Base but NO Dependent in MVC mode, drop the frame.
                # This prevents "queue starvation" where Base frames fill the DPB waiting for non-existent Deps.
                # EXCEPTION: In combined/interleaved MVC mode, both views are in the "base" data,
                # so we should NOT drop frames even if "dependent" is empty.
                if success and hasattr(self.demuxer, 'get_video_info'):
                    try:
                        vinfo = self.demuxer.get_video_info()
                        has_mvc = getattr(vinfo, 'hasMVC', False)

                        # Check for combined MVC mode (both views in same PID - M2TS)
                        base_pid = getattr(vinfo, 'baseVideoPid', 0)
                        mvc_pid = getattr(vinfo, 'mvcVideoPid', 0)
                        is_combined_mvc_m2ts = (base_pid == mvc_pid and base_pid != 0)

                        # Check for single-track interleaved MVC (MKV) - mvcTrackNumber=0 means
                        # no separate MVC track, so both views are interleaved in base track
                        mvc_track = getattr(vinfo, 'mvcTrackNumber', -1)
                        is_interleaved_mvc_mkv = (mvc_track == 0 and has_mvc)

                        is_combined_mvc = is_combined_mvc_m2ts or is_interleaved_mvc_mkv

                        dep_empty = not dep or 'data' not in dep or len(dep['data']) == 0
                        base_ok = base and 'data' in base and len(base['data']) > 0

                        # Skip dropping in combined/interleaved MVC mode - both views are in the base data
                        if has_mvc and base_ok and dep_empty and not is_combined_mvc:
                            # Only drop if we are actually deep in the stream (avoid dropping initial frames if startup is weird)
                            # But actually we must drop ALL such frames to keep sync.
                            # Frame 4 was the culprit in logs.
                            logger.warning(f"[MVC-THREAD] Dropping frame (Base-only) to maintain MVC sync. (Base: {len(base['data'])} bytes)")
                            continue
                    except Exception:
                        pass

                # Process PGS subtitle packets (streaming mode)
                if success:
                    try:
                        # NEW API: Poll subtitle blocks from demuxer queue
                        self._poll_subtitles()
                    except Exception as e:
                        pass  # Don't let subtitle errors break video playback

                if not success:
                    # DF-FINAL FIX 1: a False read mid-stream is NOT always genuine
                    # end-of-stream. The dual-file/SSIF demuxer can also bail out on
                    # MAX_ITERS / no-progress (corrupt dependent-view PTS class) far
                    # short of the real end of the disc. Treat that as an error so
                    # the player falls back to 2D instead of settling into a fake
                    # "playback finished" freeze. Genuine near-end EOS (or unknown
                    # duration) is unaffected.
                    _last_known_ts = self._pts_last_emit_ts
                    if _last_known_ts is None:
                        _last_known_ts = self.base_timestamp + (self.frame_count * self.target_frame_time)
                    _known_duration = self._media_duration
                    if _known_duration and _known_duration > 0 and (_known_duration - _last_known_ts) > 10.0:
                        logger.error(
                            f"[MVC] fin prematuree a {_last_known_ts:.1f}s/{_known_duration:.1f}s -> fallback 2D"
                        )
                        self.error.emit(
                            f"Premature end of stream at {_last_known_ts:.1f}s of {_known_duration:.1f}s "
                            "(dependent-view desync or corrupt disc read) -- falling back to 2D playback."
                        )
                        break

                    logger.info("[MVC-THREAD] EOS. Flushing final frames.")

                    # SOL CRASH FIX: Protect flush from 0xe24c4a02
                    try:
                        # V8 CRASH FIX: Use time.sleep for stability
                        # Small delay to let MPV thread settle
                        time.sleep(0.050)  # 50ms

                        # V8 CRASH FIX: Check decoder validity before flush
                        if self.decoder:
                            edge264.edge264_flush(self.decoder)

                            # Small delay after flush
                            time.sleep(0.020)  # 20ms

                            # V12 NON-BLOCKING FIX: Add retry counter for final flush
                            flush_drain_retry = 0
                            max_flush_drain_retry = 5
                            while not self._stop_requested and not self._cleanup_in_progress and self.decoder:
                                try:
                                    # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                                    ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                                except (OSError, RuntimeError):
                                    break
                                if ret == 0:
                                    flush_drain_retry = 0  # V12: Reset on success
                                    try:
                                        self._deliver_frame_to_gpu(frame_struct)
                                    finally:
                                        # CRITICAL: Return frame AFTER all data is copied
                                        if self.decoder and frame_struct.return_arg:
                                            try:
                                                edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                            except (OSError, RuntimeError):
                                                pass
                                else:
                                    # V12 NON-BLOCKING FIX: Retry with yield
                                    flush_drain_retry += 1
                                    if flush_drain_retry >= max_flush_drain_retry:
                                        break
                                    time.sleep(0.001)  # 1ms yield
                                    if self.decoder:
                                        try:
                                            edge264.edge264_bump_frames(self.decoder)
                                        except (OSError, RuntimeError):
                                            break

                        # Flush remaining buffered frames
                        while self.frame_buffer:
                            self._repair_and_queue(self.frame_buffer.pop(0)['data'])

                        while self.presentation_queue:
                            self._emit_single_frame(self.presentation_queue.popleft())
                            self._precise_wait(self.target_frame_time)  # Use hybrid wait instead of sleep

                    except Exception as e:
                        logger.error(f"[MVC-THREAD] Error during final flush: {e}")
                    finally:
                        # V7c FIX: Don't emit decodingFinished here - set flag for later
                        # This prevents GUI cleanup racing with our own cleanup
                        self._eos_reached = True
                    # break OUTSIDE the finally: same control flow (the except above
                    # already handles every Exception), but a break inside finally
                    # would swallow KeyboardInterrupt/SystemExit (Py3.14 warns).
                    break

                # 3. Aggressive Keyframe Draining
                # If new frame is a keyframe, drain DPB aggressively to make space
                # V8 CRASH FIX: Check BOTH seek flags AND decoder validity before ALL edge264 calls
                is_keyframe = base.get('isKeyframe', False) if base else False
                if is_keyframe and self.frame_count > 0 and not self._seek_requested and not self._seek_in_progress and self.decoder:
                    for _ in range(10):  # Aggressive bump
                        if self._seek_requested or self._seek_in_progress or not self.decoder:
                            break
                        try:
                            edge264.edge264_bump_frames(self.decoder)
                        except (OSError, RuntimeError):
                            break
                        # V12 NON-BLOCKING FIX: Add retry counter for keyframe drain
                        kf_drain_retry = 0
                        max_kf_drain_retry = 3
                        while not self._seek_requested and not self._seek_in_progress and self.decoder:
                            try:
                                # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                                ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                            except (OSError, RuntimeError):
                                break
                            if ret == 0:
                                kf_drain_retry = 0  # V12: Reset on success
                                try:
                                    self._deliver_frame_to_gpu(frame_struct)
                                finally:
                                    # CRITICAL: Return frame AFTER all data is copied
                                    if self.decoder and frame_struct.return_arg:
                                        try:
                                            edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                        except (OSError, RuntimeError):
                                            pass
                                self.frame_count += 1
                            else:
                                # V12 NON-BLOCKING FIX: Retry with yield
                                kf_drain_retry += 1
                                if kf_drain_retry >= max_kf_drain_retry:
                                    break
                                time.sleep(0.0005)  # 0.5ms yield
                                if self.decoder and not self._seek_requested and not self._seek_in_progress:
                                    try:
                                        edge264.edge264_bump_frames(self.decoder)
                                    except (OSError, RuntimeError):
                                        break

                # 4. Process AU
                # V33m DIAG: Log AU processing entry
                if _diag_loop_iter <= 10:
                    pass  # V33o: Removed verbose AU logging
                au_data = bytearray()
                # MVC-DIAG: Log base/dep sizes every 30 frames
                if self.frame_count < 10 or self.frame_count % 30 == 0:
                    base_size = len(base['data']) if base and 'data' in base else 0
                    dep_size = len(dep['data']) if dep and 'data' in dep else 0
                    logger.debug(f"[MVC-DIAG] Frame {self.frame_count}: base={base_size} bytes, dep={dep_size} bytes")
                if base and 'data' in base: au_data.extend(bytes(base['data']))
                if dep and 'data' in dep: au_data.extend(bytes(dep['data']))

                if au_data:
                    # V8 CRASH FIX: Check BOTH seek flags AND decoder validity before processing
                    if not self._seek_requested and not self._seek_in_progress and self.decoder:
                        try:
                            self._process_au_data(bytes(au_data))
                            self._push_pair_pts(base, dep)  # V60 SYNC-PTS
                        except (OSError, RuntimeError) as e:
                            logger.warning(f"[MVC-THREAD] _process_au_data exception (ignored): {e}")

                if self._needs_full_reset:
                    # V47 FIX (replaces V44): The previous V44 unconditionally suppressed
                    # the reset whenever frame_count > 0, which masked GENUINE stuck states
                    # — Gravity 3D after priming produces 10+1 frames then deadlocks forever,
                    # but frame_count stays > 0 so V44 swept the reset under the rug.
                    #
                    # New logic: track whether frame_count is still GROWING between reset
                    # attempts. If yes → transient warmup, suppress. If frames haven't
                    # moved in 3 reset windows → real deadlock, escalate to crash signal
                    # (which triggers _start_mvc_decoder restart on the GUI side).
                    if not hasattr(self, '_frames_at_last_reset_check'):
                        self._frames_at_last_reset_check = -1
                        self._reset_suppress_count = 0
                    if self.frame_count > self._frames_at_last_reset_check:
                        # Frame count is growing → decoder is making progress, just slow
                        logger.warning(
                            f"[MVC-THREAD] V47: Reset suppressed — frame_count growing "
                            f"({self._frames_at_last_reset_check} → {self.frame_count})"
                        )
                        self._frames_at_last_reset_check = self.frame_count
                        self._reset_suppress_count = 0
                        self._needs_full_reset = False
                        self._consecutive_errors = 0
                    else:
                        # No new frames since last reset attempt → genuinely stuck
                        self._reset_suppress_count += 1
                        if self._reset_suppress_count >= 3:
                            logger.error(
                                f"[MVC-THREAD] V47: STUCK at frame {self.frame_count} for "
                                f"{self._reset_suppress_count} reset windows. Forcing restart."
                            )
                            self.decoderCrashed.emit()
                            break
                        else:
                            logger.warning(
                                f"[MVC-THREAD] V47: Reset deferred ({self._reset_suppress_count}/3) — "
                                f"no new frames since count={self._frames_at_last_reset_check}"
                            )
                            self._needs_full_reset = False
                            self._consecutive_errors = 0

                if self._fatal_error: break

                # V8 CRASH FIX: Check BOTH seek flags during decode
                if self._seek_requested or self._seek_in_progress:
                    try:
                        time.sleep(0.005)  # 5ms - use time.sleep for stability
                    except Exception:
                        pass
                    continue

                # 5. Post-feed Bumping (Aggressive)
                # Force frames out after feeding data
                # V8 CRASH FIX: Check BOTH seek flags and decoder validity before ALL edge264 calls
                # V51 PERF: synchronous decode (n_threads=0) needs only ONE bump — frames are
                # already fully decoded inline, so the extra 4 bumps were pure per-AU waste.
                _bump_rounds = 1 if getattr(self, '_alloc_threads', 0) == 0 else 5
                for _ in range(_bump_rounds):
                    if self._seek_requested or self._seek_in_progress or self._stop_requested or self._cleanup_in_progress or not self.decoder:
                        break
                    try:
                        edge264.edge264_bump_frames(self.decoder)
                    except (OSError, RuntimeError):
                        break

                # 6. Post-feed Drain
                # V8 CRASH FIX: Check BOTH seek flags and decoder validity before ALL edge264 calls
                # V14 GRACEFUL ENDING: Also check cleanup flag
                # V12 NON-BLOCKING FIX: With non_blocking=1 in edge264_decode_NAL, frames may not be
                # immediately available. Add a retry loop with yields to give decoder threads time.
                drain_retry_count = 0
                max_drain_retries = 5  # Max retries before giving up (frames will be caught in next PRE-DRAIN)
                frames_extracted_this_round = 0
                
                # V33o: Removed verbose POST-DRAIN logging

                while not self._stop_requested and not self._seek_requested and not self._seek_in_progress and not self._cleanup_in_progress and self.decoder:
                    try:
                        # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                        # V33r: Use borrow=1 to keep buffer valid while copying
                        ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                    except (OSError, RuntimeError):
                        break
                    if ret == 0:
                        get_frame_call_count += 1
                        frames_extracted_this_round += 1
                        drain_retry_count = 0  # Reset retry count on success
                        
                        
                        try:
                            self._deliver_frame_to_gpu(frame_struct)
                        finally:
                            # V33r: Return frame AFTER copying
                            if self.decoder and frame_struct.return_arg:
                                try:
                                    edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                except (OSError, RuntimeError):
                                    pass
                        self.frame_count += 1
                    else:
                        # V51 PERF: with synchronous decode (n_threads=0) all frames are produced
                        # inline by edge264_decode_NAL, so a non-zero get_frame means there is
                        # genuinely nothing more to drain — retrying with 1ms sleeps just burned
                        # ~5ms per AU (the dominant per-frame overhead that caused frame drops /
                        # stutter on dense scenes). Break immediately. The async retry path is kept
                        # only for n_threads>0.
                        if getattr(self, '_alloc_threads', 0) == 0:
                            break
                        # V12 NON-BLOCKING FIX (threaded only): Frame not ready yet — yield and retry
                        drain_retry_count += 1
                        if drain_retry_count >= max_drain_retries:
                            break
                        try:
                            time.sleep(0.001)  # 1ms yield
                        except Exception:
                            pass
                        if self.decoder and not self._seek_requested and not self._seek_in_progress:
                            try:
                                edge264.edge264_bump_frames(self.decoder)
                            except (OSError, RuntimeError):
                                break

                # V12 DIAG: Log drain statistics
                if self.frame_count <= 30 or self.frame_count % 100 == 0:
                    logger.debug(f"[DRAIN-DIAG] frame_count={self.frame_count}, extracted_this_round={frames_extracted_this_round}, retries={drain_retry_count}")
                
                # V33o: POST-DRAIN logging kept minimal (see V12 DIAG above)

                # Stats and Sync
                if loop_start_time - self.last_stats_time > 1.0:
                    fps = self.frame_count / (loop_start_time - self.start_time) if (
                                                                                                loop_start_time - self.start_time) > 0 else 0
                    self.fps_update.emit(fps)
                    self.last_stats_time = loop_start_time

                # --- PRESENTATION / PACING LOGIC ---
                # Decoupled from decoding loop to handle bursts (like scene changes) smoothly.
                # V8 SYNC GATE check is now INSIDE _process_presentation_queue()
                # V54: once the presenter thread is running it owns presentation; this
                # inline call only serves any path before the presenter starts.
                if not self._presenter_active:
                    self._process_presentation_queue()

                # Calculate time spent processing/decoding
                # We do NOT sleep here anymore because _process_presentation_queue()
                # handles the precise AV-sync sleeping.
                # If we sleep here again, we double-sleep and kill the framerate.
                
            else:
                # Buffer underrun or startup phase:
                # Don't sleep the full frame time, loop quickly to fill the buffer!
                time.sleep(0.001)

        except Exception as e:
            logger.error(f"[MVC-THREAD] CRASH in run loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(f"Error: {str(e)}")
        finally:
            # V7c CRASH FIX: Give GUI time to set cleanup flags before we touch memory
            # This prevents race between GUI's _on_mvc_finished and our cleanup
            time.sleep(0.100)  # 100ms settling time

            if self.decoder:
                if self._efault_count > 0:
                    logger.warning(f"[MVC-THREAD] Skipping edge264_free() due to {self._efault_count} EFAULT errors.")
                elif self._cleanup_in_progress:
                    # V7c CRASH FIX: If GUI cleanup is in progress, let GUI handle cleanup timing
                    # We just null out our reference
                    logger.info("[MVC-THREAD] V7c: Cleanup in progress from GUI, skipping edge264_free")
                else:
                    # V14 CRASH FIX: Wait for any in-progress frame delivery before cleanup
                    if self._frame_delivery_active:
                        logger.info("[MVC-THREAD] V14: Waiting for frame delivery before cleanup...")

                    # V7c: Extra settling time before touching C memory
                    time.sleep(0.050)

                    if not self._frame_delivery_lock.acquire(timeout=1.0):
                        logger.warning("[MVC-THREAD] Could not acquire frame_delivery_lock for cleanup")
                    else:
                        try:
                            # V7b STABILITY FIX: Extra checks before free
                            if self.decoder and self.decoder.value and not self._cleanup_in_progress:
                                decoder_ptr = ctypes.c_void_p(self.decoder.value)
                                with edge264_session_lock:
                                    edge264.edge264_free(ctypes.byref(decoder_ptr))
                                logger.info("[MVC-THREAD] edge264_free() completed successfully")
                        except OSError:
                            # Catch Windows fatal exceptions (access violation) safely
                            logger.warning("[MVC-THREAD] Access violation during edge264_free (ignored/safe).")
                        except Exception as e:
                            logger.error(f"[MVC-THREAD] Error during edge264_free(): {e}")
                        finally:
                            self._frame_delivery_lock.release()

            self.decoder = None

            # V7c: Small delay before clearing Python objects
            time.sleep(0.050)

            try:
                self.au_buffers.clear()
                self.frame_buffer.clear()
                self.presentation_queue.clear()
            except Exception as e:
                logger.error(f"[MVC-THREAD] Error clearing buffers: {e}")

            # V7c: GC after a delay to avoid racing with GUI
            try:
                time.sleep(0.050)
                gc.collect()
            except Exception as e:
                pass  # Ignore GC errors

            # V7c FIX: Emit decodingFinished AFTER all cleanup is complete
            # This prevents GUI cleanup from racing with our cleanup
            if self._eos_reached:
                logger.info("[MVC-THREAD] V7c: Emitting decodingFinished after cleanup complete")
                self.decodingFinished.emit()

    def _convert_avcc_to_annexb(self, avcc_data):
        """Converts AVCC extradata to Annex B NALs. Includes extension/SubsetSPS logic."""
        try:
            if not avcc_data or len(avcc_data) < 7:  # 6 bytes header + at least 1 for num pps
                logger.error("[AVCC] CodecPrivate data is too short.")
                return None

            if avcc_data[0] != 1:
                logger.error(f"[AVCC] Unexpected AVCC version: {avcc_data[0]}")
                return None

            blob = bytearray()
            offset = 5  # After configurationVersion, AVCProfileIndication, profile_compatibility, AVCLevelIndication, lengthSizeMinusOne

            # 1. SPS
            if offset >= len(avcc_data):
                logger.error("[AVCC] CodecPrivate data is too short to contain numOfSequenceParameterSets.")
                return None
            num_sps = avcc_data[offset] & 0x1F
            offset += 1
            for i in range(num_sps):
                if offset + 2 > len(avcc_data):
                    logger.error(f"[AVCC] Incomplete SPS length data (SPS #{i + 1})")
                    return None
                size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                offset += 2
                if offset + size > len(avcc_data):
                    logger.error(f"[AVCC] Incomplete SPS NAL data (SPS #{i + 1})")
                    return None
                blob.extend(b'\x00\x00\x00\x01')
                blob.extend(avcc_data[offset:offset + size])
                offset += size

            # 2. PPS
            if offset >= len(avcc_data):
                logger.warning("[AVCC] CodecPrivate ended after SPS, no PPS found.")
                return bytes(blob)  # Still valid if it only contains SPS

            num_pps = avcc_data[offset]
            offset += 1
            for i in range(num_pps):
                if offset + 2 > len(avcc_data):
                    logger.error(f"[AVCC] Incomplete PPS length data (PPS #{i + 1})")
                    return None
                size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                offset += 2
                if offset + size > len(avcc_data):
                    logger.error(f"[AVCC] Incomplete PPS NAL data (PPS #{i + 1})")
                    return None
                blob.extend(b'\x00\x00\x00\x01')
                blob.extend(avcc_data[offset:offset + size])
                offset += size

            # 3. MVC Extension (Subset SPS for MVC) - ISO/IEC 14496-15 section 5.3.4.2.1
            # For profiles >= 100 (High, High 10, etc.), extension data follows PPS
            if offset < len(avcc_data):
                remaining = len(avcc_data) - offset
                logger.info(f"[MVC-THREAD] AVCC has {remaining} bytes of MVC extension data")

                # Hex dump first 20 bytes for diagnostics
                ext_preview = avcc_data[offset:offset+min(20, remaining)]
                hex_str = ' '.join(f'{b:02X}' for b in ext_preview)
                logger.info(f"[MVC-THREAD] AVCC ext hex preview: {hex_str}")

                # Parse High profile extension (chroma_format, bit_depth fields)
                # Byte 1: reserved(6) + chroma_format(2)
                # Byte 2: reserved(5) + bit_depth_luma_minus8(3)
                # Byte 3: reserved(5) + bit_depth_chroma_minus8(3)
                if offset + 3 <= len(avcc_data):
                    chroma_format = avcc_data[offset] & 0x03
                    bit_depth_luma = (avcc_data[offset + 1] & 0x07) + 8
                    bit_depth_chroma = (avcc_data[offset + 2] & 0x07) + 8
                    offset += 3
                    logger.info(f"[MVC-THREAD] AVCC ext: chroma={chroma_format}, luma_depth={bit_depth_luma}, chroma_depth={bit_depth_chroma}")

                    # Byte 4: numOfSequenceParameterSetExt (regular SPS extensions, usually 0)
                    if offset + 1 <= len(avcc_data):
                        num_sps_ext = avcc_data[offset]
                        offset += 1
                        logger.info(f"[MVC-THREAD] AVCC ext: {num_sps_ext} SPS extensions")

                        # Skip SPS extensions (if any)
                        for i in range(num_sps_ext):
                            if offset + 2 > len(avcc_data):
                                break
                            size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                            offset += 2 + size

                        # Check for "mvcC" box (ISO Base Media File Format)
                        # Format: 4 bytes size + 4 bytes "mvcC" (0x6D766343)
                        if offset + 8 <= len(avcc_data):
                            box_size = (avcc_data[offset] << 24) | (avcc_data[offset+1] << 16) | (avcc_data[offset+2] << 8) | avcc_data[offset+3]
                            box_type = avcc_data[offset+4:offset+8]

                            if box_type == b'mvcC':
                                logger.info(f"[MVC-THREAD] Found mvcC box ({box_size} bytes)")
                                offset += 8  # Skip box header

                                # mvcC box contains a mini-AVCC structure for MVC layer
                                # Byte 1: configurationVersion (1)
                                # Byte 2: complete_representation(1) + reserved(7)
                                # Skip 2 more bytes (reserved/profile info)
                                # Byte 5: reserved(6) + lengthSizeMinusOne(2)
                                # Byte 6: reserved(3) + numOfSequenceParameterSets(5)
                                if offset + 6 <= len(avcc_data):
                                    mvc_version = avcc_data[offset]
                                    complete_rep = (avcc_data[offset + 1] >> 7) & 0x01
                                    logger.info(f"[MVC-THREAD] mvcC version={mvc_version}, complete_representation={complete_rep}")
                                    offset += 4  # Skip version, complete_rep, reserved bytes
                                    length_size = (avcc_data[offset] & 0x03) + 1
                                    offset += 1

                                    # numOfSequenceParameterSets (lower 5 bits)
                                    num_mvc_sps = avcc_data[offset] & 0x1F
                                    offset += 1
                                    logger.info(f"[MVC-THREAD] mvcC: {num_mvc_sps} SPS NALs, length_size={length_size}")

                                    for i in range(num_mvc_sps):
                                        if offset + 2 > len(avcc_data):
                                            break
                                        size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                                        offset += 2
                                        if offset + size > len(avcc_data):
                                            break
                                        # Inject NAL with start code
                                        blob.extend(b'\x00\x00\x00\x01')
                                        blob.extend(avcc_data[offset:offset + size])
                                        nal_type = avcc_data[offset] & 0x1F if size > 0 else -1
                                        logger.info(f"[MVC-THREAD] Injected mvcC SPS #{i+1} (NAL type {nal_type}, {size} bytes)")
                                        offset += size

                                    # numOfPictureParameterSets
                                    if offset + 1 <= len(avcc_data):
                                        num_mvc_pps = avcc_data[offset]
                                        offset += 1
                                        logger.info(f"[MVC-THREAD] mvcC: {num_mvc_pps} PPS NALs")

                                        for i in range(num_mvc_pps):
                                            if offset + 2 > len(avcc_data):
                                                break
                                            size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                                            offset += 2
                                            if offset + size > len(avcc_data):
                                                break
                                            blob.extend(b'\x00\x00\x00\x01')
                                            blob.extend(avcc_data[offset:offset + size])
                                            nal_type = avcc_data[offset] & 0x1F if size > 0 else -1
                                            logger.info(f"[MVC-THREAD] Injected mvcC PPS #{i+1} (NAL type {nal_type}, {size} bytes)")
                                            offset += size
                            else:
                                # Not a mvcC box - try flat format parsing
                                logger.info(f"[MVC-THREAD] No mvcC box found, trying flat MVC format")
                                mvc_remaining = len(avcc_data) - offset
                                logger.info(f"[MVC-THREAD] MVC extension: {mvc_remaining} bytes remaining for Subset SPS/PPS")

                                # numOfSequenceParameterSetMVC (Subset SPS count)
                                if offset + 1 <= len(avcc_data):
                                    num_subset_sps = avcc_data[offset]
                                    offset += 1
                                    logger.info(f"[MVC-THREAD] AVCC extension: {num_subset_sps} Subset SPS NALs")

                                    for i in range(num_subset_sps):
                                        if offset + 2 > len(avcc_data):
                                            logger.warning(f"[AVCC] Incomplete Subset SPS length data at index {i}")
                                            break
                                        size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                                        offset += 2
                                        if offset + size > len(avcc_data):
                                            logger.warning(f"[AVCC] Incomplete Subset SPS NAL data at index {i} (need {size}, have {len(avcc_data) - offset})")
                                            break
                                        # Add Subset SPS with start code
                                        blob.extend(b'\x00\x00\x00\x01')
                                        blob.extend(avcc_data[offset:offset + size])
                                        nal_type = avcc_data[offset] & 0x1F if size > 0 else -1
                                        logger.info(f"[MVC-THREAD] Injected Subset SPS #{i+1} (NAL type {nal_type}, {size} bytes)")
                                        offset += size

                                    # numOfPictureParameterSetsMVC (MVC PPS count)
                                    if offset + 1 <= len(avcc_data):
                                        num_mvc_pps = avcc_data[offset]
                                        offset += 1
                                        logger.info(f"[MVC-THREAD] AVCC extension: {num_mvc_pps} MVC PPS NALs")

                                        for i in range(num_mvc_pps):
                                            if offset + 2 > len(avcc_data):
                                                break
                                            size = (avcc_data[offset] << 8) | avcc_data[offset + 1]
                                            offset += 2
                                            if offset + size > len(avcc_data):
                                                break
                                            # Add MVC PPS with start code
                                            blob.extend(b'\x00\x00\x00\x01')
                                            blob.extend(avcc_data[offset:offset + size])
                                            logger.info(f"[MVC-THREAD] Injected MVC PPS #{i+1} ({size} bytes)")
                                            offset += size

            return bytes(blob)
        except Exception as e:
            logger.error(f"[AVCC] EXCEPTION during AVCC->AnnexB conversion: {e}")
            return None

    def _prime_decoder(self):
        for i in range(2):  # Reduced from 4 to 2 to minimize I/O spike
            if self._stop_requested: return False
            success, base, dep = self.demuxer.read_next_frame_pair()
            if not success: return False
            if base and 'data' in base: self._process_au_data(bytes(base['data']))
            if dep and 'data' in dep: self._process_au_data(bytes(dep['data']))
        return True

    def _decode_loop(self):
        frame_struct = Edge264Frame()
        last_loop_time = time.time()
        gc_timer = 0
        frame_count_local = 0

        logger.info("[MVC-THREAD] Entering decode loop...")

        while True:
            self.mutex.lock()
            if self._stop_requested:
                self.mutex.unlock()
                logger.info("[MVC-THREAD] Stop requested")
                break
            self.mutex.unlock()

            if self._fatal_error:
                logger.error("[MVC-THREAD] Fatal error (EFAULT). Stopping.")
                self.error.emit("Fatal decoder error (140).")
                break

            loop_start = time.time()

            # DRAIN
            # V12 NON-BLOCKING FIX: Add retry counter for main pipeline drain
            main_drain_retry = 0
            max_main_drain_retry = 3
            while True:
                # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                if ret == 0:
                    main_drain_retry = 0  # V12: Reset on success
                    try:
                        self._deliver_frame_to_gpu(frame_struct)
                    finally:
                        # CRITICAL: Return frame AFTER all data is copied
                        if self.decoder and frame_struct.return_arg:
                            try:
                                edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                            except (OSError, RuntimeError):
                                pass
                    self.frame_count += 1
                    frame_count_local += 1
                else:
                    # V12 NON-BLOCKING FIX: Retry with yield instead of breaking immediately
                    main_drain_retry += 1
                    if main_drain_retry >= max_main_drain_retry:
                        if ret != 122:  # Only log if not EAGAIN
                            logger.warning(f"[MVC-THREAD] edge264_get_frame returned {ret}")
                        break
                    edge264.edge264_bump_frames(self.decoder)
                    time.sleep(0.0005)  # 0.5ms yield

            # STATS
            if loop_start - self.last_stats_time > 1.0:
                fps = self.frame_count / (loop_start - self.start_time) if (loop_start - self.start_time) > 0 else 0
                self.fps_update.emit(fps)
                self.last_stats_time = loop_start
                gc_timer += 1
                if gc_timer > 5:
                    # V52 SMOOTHNESS: gen-0 only. A full gc.collect() sweeps all 3
                    # generations (whole-heap scan ~70ms) and was the regular ~5s
                    # playback spike. Per-frame numpy arrays are freed by refcounting
                    # (not GC), so gen-0 reclaims young cycles cheaply; automatic GC
                    # + the full collect at seek/reset remain the safety net for cycles.
                    gc.collect(0)
                    gc_timer = 0

            # SYNC
            elapsed = time.time() - last_loop_time
            if elapsed < (self.target_frame_time * 0.5):
                try:
                    time.sleep(0.001)
                except Exception:
                    pass

            # READ
            try:
                self.mutex.lock()
                success, base, dep = self.demuxer.read_next_frame_pair()
                self.mutex.unlock()
            except Exception as e:
                logger.error(f"[MVC-THREAD] Demuxer read failed: {e}")
                break

            if not success:
                logger.info("[MVC-THREAD] EOS.")
                self.decodingFinished.emit()
                break

            # CRITICAL FIX: Concatenate base + dependent into single stream
            # edge264 expects a continuous stream where it can detect the transition
            # from base to dependent view and call unset_currPic() to update basePic.
            # Injecting them separately breaks this logic, causing basePic=-1.
            au_data = bytearray()
            if base and 'data' in base:
                au_data.extend(bytes(base['data']))
            if dep and 'data' in dep:
                au_data.extend(bytes(dep['data']))

            if os.environ.get("SYLC_DECODE_DIAG") == "1":
                _bs = len(base['data']) if base and 'data' in base else 0
                _ds = len(dep['data']) if dep and 'data' in dep else 0
                import sys as _sys; _sys.stderr.write(f"[PY-AU] base={_bs} dep={_ds} total={len(au_data)}\n")

            if au_data:
                self._process_au_data(bytes(au_data))

            # BUMP frames after complete access unit
            edge264.edge264_bump_frames(self.decoder)

            last_loop_time = loop_start
            process_dur = time.time() - loop_start
            rem = self.target_frame_time - process_dur
            if rem > 0.002:
                try:
                    time.sleep(rem * 0.9)
                except Exception:
                    pass

    def set_pg_offset_sequence(self, seq_id):
        """Select which BD3D offset sequence drives the PG subtitle depth
        (from the playlist's STN_table_SS mapping for the chosen PG PID)."""
        self._pg_offset_seq_id = int(seq_id)
        self._pg_last_disparity = None   # force re-emit with the new sequence
        logger.info(f"[BD3D-DEPTH] PG offset sequence -> {seq_id}")

    def _scan_bd3d_offset_metadata(self, au_data):
        """Extract the authored PG depth from OFMD SEIs (GOP-start dep AUs)."""
        try:
            from bd3d_offset_metadata import ofmd_scan, offset_to_disparity
            res = ofmd_scan(au_data)
            if res is None:
                return
            frames, seqs = res
            seq_id = getattr(self, '_pg_offset_seq_id', 0)
            if not (0 <= seq_id < len(seqs)):
                seq_id = 0
            row = seqs[seq_id]
            # offsets are constant within a GOP in practice -> use the median
            off = sorted(row)[len(row) // 2]
            disp = offset_to_disparity(off)
            if disp != getattr(self, '_pg_last_disparity', None):
                self._pg_last_disparity = disp
                self.pgDepthChanged.emit(disp)
        except Exception:
            pass  # depth is best-effort; never disturb the decode loop

    def _process_au_data(self, au_data, force=False):
        """Process Access Unit data. force=True allows processing during seek (for re-priming)."""
        if not au_data: return
        self._scan_bd3d_offset_metadata(au_data)

        # V54: backpressure — wait here (outside the delivery lock) if the presenter
        # hasn't drained the buffer yet, so decode doesn't evict unshown frames.
        if not force:
            self._await_queue_space()

        # V7b++++++ CRASH FIX: Convert to bytes immediately to stabilize data
        # This prevents memoryview slices from referencing freed memory
        try:
            if not isinstance(au_data, bytes):
                au_data = bytes(au_data)
        except (TypeError, ValueError, MemoryError) as e:
            logger.warning(f"[MVC-THREAD] Failed to convert au_data to bytes: {e}")
            return

        # DIAGNOSTIC: Log AU info for debugging combined MVC SSIF issues
        # V7c CRASH FIX: Wrap hex preview in try/except - genexpr can crash on invalid memory
        if not hasattr(self, '_au_count'):
            self._au_count = 0
        self._au_count += 1
        if self._au_count <= 10 or self._au_count % 100 == 0:
            try:
                au_len = len(au_data) if au_data else 0
                if au_len >= 32:
                    hex_preview = ' '.join(f'{b:02x}' for b in bytes(au_data[:32]))
                elif au_len > 0:
                    hex_preview = ' '.join(f'{b:02x}' for b in bytes(au_data))
                else:
                    hex_preview = "(empty)"
                logger.debug(f"[MVC-DIAG] AU #{self._au_count}: {au_len} bytes, first 32 bytes: {hex_preview}")
            except Exception as e:
                logger.debug(f"[MVC-DIAG] AU #{self._au_count}: (hex preview failed: {e})")

        _vp_t0 = velvet_probe.now() if velvet_probe.ENABLED else 0.0
        try:
            # 1. Collect all NAL units
            nal_units = []
            for nal_data in find_nal_units(au_data):
                sc_len = 3
                if nal_data.startswith(b'\x00\x00\x00\x01'):
                    sc_len = 4
                elif nal_data.startswith(b'\x00\x00\x01'):
                    sc_len = 3

                if len(nal_data) > sc_len:
                    header_byte = nal_data[sc_len]
                    nal_type = header_byte & 0x1F
                    nal_units.append((nal_type, nal_data, sc_len))

            # DIAGNOSTIC: Log NAL types in this AU (DEBUG level to avoid lock contention)
            if self._au_count <= 10 or self._au_count % 100 == 0:
                try:
                    nal_summary = ', '.join(f'type{nt}({len(nd)}B)' for nt, nd, _ in nal_units)
                    logger.debug(f"[MVC-DIAG] AU #{self._au_count}: {len(nal_units)} NALs: {nal_summary}")
                except Exception:
                    pass

            # MVC FIX: DO NOT sort NAL units!
            # For single-track interleaved MVC (like Gravity.mkv), the original NAL order is CRITICAL.
            # edge264 uses the NAL sequence to detect view transitions:
            #   [base_SPS, base_PPS, base_slices] -> [SubsetSPS, mvc_PPS, mvc_slices]
            # Reordering breaks edge264's internal state machine (basePic association).
            # Parameter sets have unique IDs, so edge264 can look them up regardless of position.
            # Previously this sorted by type which broke MVC view transition detection.
            pass  # Preserve original NAL order

            # Check for IDR in this batch
            has_idr = any(nt == NAL_TYPE_IDR for nt, _, _ in nal_units)

            # SAFETY: If we are waiting for an IDR (e.g. after a failed seek), drop everything else
            if self._waiting_for_idr:
                if not has_idr:
                    return
                else:
                    logger.info("[MVC-THREAD] IDR found! Resuming decoding.")
                    self._waiting_for_idr = False

            if has_idr and self.decoder:
                # (MONOTONE-POC: no epoch signalling anymore — the decoder's
                # IDR-floored POC keeps display order sortable by POC alone,
                # see _queue_frame_for_display.)
                # Mimic ultimate_mvc_player.py: Drain aggressive before IDR
                # This ensures old scene frames are out before new scene starts
                # V8 CRASH FIX: Check seek flags before ANY edge264 call in drain loop
                for _ in range(5):  # Bump a few times
                    # V8 CRASH FIX: Early exit if seek started (decoder may be destroyed)
                    # V14 GRACEFUL ENDING: Also check cleanup flag
                    if self._seek_requested or self._seek_in_progress or self._stop_requested or self._cleanup_in_progress or not self.decoder:
                        break
                    try:
                        edge264.edge264_bump_frames(self.decoder)
                    except (OSError, RuntimeError):
                        break

                    # Drain loop
                    # V14 GRACEFUL ENDING: Also check cleanup flag
                    # V12 NON-BLOCKING FIX: Add retry counter for non-blocking frame extraction
                    idr_drain_retry = 0
                    max_idr_drain_retry = 3
                    while not self._seek_requested and not self._seek_in_progress and not self._stop_requested and not self._cleanup_in_progress and self.decoder:
                        try:
                            frame_struct = Edge264Frame()
                            # CRITICAL FIX: Use borrow=1 to prevent buffer reuse during copy
                            ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                        except (OSError, RuntimeError):
                            break
                        if ret == 0:
                            idr_drain_retry = 0  # V12: Reset on success
                            # V8 CRASH FIX: Double-check before delivering
                            if self._seek_requested or self._seek_in_progress or self._stop_requested:
                                # CRITICAL: Still need to return borrowed frame
                                if self.decoder and frame_struct.return_arg:
                                    try:
                                        edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                    except (OSError, RuntimeError):
                                        pass
                                break
                            try:
                                self._deliver_frame_to_gpu(frame_struct)
                            finally:
                                # CRITICAL: Return frame AFTER all data is copied
                                if self.decoder and frame_struct.return_arg:
                                    try:
                                        edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                    except (OSError, RuntimeError):
                                        pass
                            self.frame_count += 1
                        else:
                            # V12 NON-BLOCKING FIX: Retry with yield instead of breaking immediately
                            idr_drain_retry += 1
                            if idr_drain_retry >= max_idr_drain_retry:
                                break
                            time.sleep(0.0005)  # 0.5ms yield
                            if self.decoder and not self._seek_requested and not self._seek_in_progress:
                                try:
                                    edge264.edge264_bump_frames(self.decoder)
                                except (OSError, RuntimeError):
                                    break

            # 3. Filter and Feed
            for nal_type, nal_data, sc_len in nal_units:
                # V48 NAL ORDER FIX: Previously SEI (6) and AUD (9) were filtered.
                # But edge264 uses these to:
                #   - AUD: detect access-unit boundaries (frame transitions)
                #   - SEI: read MVC-specific metadata (view dependencies, etc.)
                # Filtering them out broke the decoder's CABAC alignment state
                # because slice header offsets depend on what came before.
                # Keep them — edge264 will skip ones it doesn't need.
                # Only filter STAP-A (type 24) which is invalid for Annex B.
                if nal_type == 24:
                    continue

                # Track if we see a real Subset SPS (NAL 15) from the stream
                if nal_type == NAL_TYPE_SUBSET_SPS:
                    self._subset_sps_seen = True
                    self._cached_nal_ssps = bytes(nal_data)  # Cache deep copy
                elif nal_type == NAL_TYPE_SPS:
                    self._cached_nal_sps = bytes(nal_data)   # Cache deep copy
                elif nal_type == NAL_TYPE_PPS:
                    self._cached_nal_pps = bytes(nal_data)   # Cache deep copy

                # V7b+++++ SEEK FIX: Pass force flag to allow re-priming during seek
                self._push_nal_direct(nal_data, sc_len, force=force)

        except Exception as e:
            logger.error(f"[MVC-THREAD] Parse error: {e}")
        finally:
            if velvet_probe.ENABLED and _vp_t0:
                velvet_probe.record('au_ms', (velvet_probe.now() - _vp_t0) * 1000.0)

    def _push_nal_direct(self, nal_data, start_code_len=3, force=False, _nal_retry=0):
        # V30 FIX: Bail out immediately if reset flagged to prevent infinite loop
        if getattr(self, '_needs_full_reset', False):
            return
        
        # V8 CRASH FIX: Check BOTH seek flags BEFORE any ctypes call to avoid 0xe24c4a02
        # V7b+++++ SEEK FIX: Allow forced injection during seek for CodecPrivate headers
        if not force and (self._seek_requested or self._seek_in_progress or self._stop_requested):
            return

        if not nal_data or not self.decoder: return

        # V7b++++++ CRASH FIX: Immediately convert to stable bytes to avoid memoryview invalidation
        # The nal_data might be a slice/memoryview from demuxer buffers that can be freed
        try:
            if not isinstance(nal_data, bytes):
                nal_data = bytes(nal_data)
        except (TypeError, ValueError, MemoryError) as e:
            logger.warning(f"[MVC-THREAD] Failed to convert nal_data to bytes: {e}")
            return

        if len(nal_data) <= start_code_len:
            return

        # V8 STABILITY: Create a DEEP copy of nal_content as bytes BEFORE any ctypes operations
        # This ensures the data is pinned and won't be garbage collected during memmove
        try:
            nal_content = bytes(nal_data[start_code_len:])
        except (TypeError, ValueError, MemoryError) as e:
            logger.warning(f"[MVC-THREAD] Failed to copy nal_content: {e}")
            return

        data_len = len(nal_content)
        if data_len == 0: return

        # V7b++++++ CRASH FIX: Sanity check data length (H.264 NAL max practical size ~10MB)
        if data_len > 10 * 1024 * 1024:
            logger.warning(f"[MVC-THREAD] NAL too large ({data_len} bytes), skipping")
            return

        # V8 CRASH FIX: Double-check BOTH flags before ctypes call (but allow forced)
        if not force and (self._seek_requested or self._seek_in_progress or self._stop_requested):
            return

        # V7c CRASH FIX: Validate nal_content is actually bytes before using ctypes
        if not isinstance(nal_content, (bytes, bytearray)):
            try:
                nal_content = bytes(nal_content)
                data_len = len(nal_content)  # Update data_len after conversion
            except (TypeError, ValueError, MemoryError) as e:
                logger.warning(f"[MVC-THREAD] V7c: Invalid nal_content type, cannot convert: {e}")
                return

        # Final sanity check: data_len must match actual content length
        if data_len != len(nal_content):
            data_len = len(nal_content)

        # V45 FIX (restored): edge264's get_bytes reads 2 bytes BEFORE the
        # buffer pointer for emulation-prevention detection. If those 2 bytes
        # are 0x00 and the NAL header is <= 0x03 (e.g. NAL ref_idc=0, type=3),
        # a false `00 00 N` (N<=3) escape pattern is detected, gb->end is set
        # to a value < gb->CPB, and the bitstream is treated as empty.
        # Prefix with `FF FF` to ensure the 2 leading bytes are non-zero.
        PREFIX_PAD = 2
        padded_len = PREFIX_PAD + data_len + 64
        # V7c CRASH FIX: Sanity check padded_len range
        if padded_len <= 0 or padded_len > 15 * 1024 * 1024:
            logger.warning(f"[MVC-THREAD] V7c: Invalid padded_len={padded_len}, skipping")
            return

        try:
            c_buffer = ctypes.create_string_buffer(padded_len)
            c_buffer[0] = b'\xFF'
            c_buffer[1] = b'\xFF'
            ctypes.memmove(ctypes.addressof(c_buffer) + PREFIX_PAD, nal_content, data_len)
        except (OSError, MemoryError, ValueError, OverflowError) as e:
            logger.error(f"[MVC-THREAD] ctypes buffer/memmove failed: {e}")
            return
        self.au_buffers.append(c_buffer)

        p_header = ctypes.cast(ctypes.addressof(c_buffer) + PREFIX_PAD, ctypes.POINTER(ctypes.c_uint8))
        p_end = ctypes.cast(ctypes.addressof(c_buffer) + PREFIX_PAD + data_len, ctypes.POINTER(ctypes.c_uint8))

        # V8 CRASH FIX: Final check just before edge264 call (but allow forced)
        if not force and (self._seek_requested or self._seek_in_progress or self._stop_requested):
            return

        # V7b++++++ CRASH FIX: Validate NAL type before calling edge264
        # Skip potentially corrupt NALs that could crash the decoder
        if data_len > 0:
            nal_type = nal_content[0] & 0x1F
            # Valid H.264 NAL types are 0-31, but practical range is 0-20
            # NAL types > 20 are rare or indicate corruption
            if nal_type > 20 and nal_type != 31:  # 31 is sometimes used
                logger.warning(f"[MVC-THREAD] V7b++++++ Skipping suspicious NAL type {nal_type}")
                if os.environ.get("SYLC_DECODE_DIAG") == "1":
                    import sys as _sys; _sys.stderr.write(f"[PY-SKIP] suspicious NAL type {nal_type}\n")
                return

            # SSIF FIX: For NAL type 14 (Prefix) and 20 (MVC slice extension),
            # validate svc_extension_flag BEFORE sending to edge264
            # svc_extension_flag is bit 7 of the byte AFTER the NAL header
            # If svc_extension_flag=1, it's SVC (not supported), skip it
            if nal_type in (14, 20) and data_len >= 2:
                ext_byte = nal_content[1]
                svc_extension_flag = (ext_byte >> 7) & 1
                if svc_extension_flag == 1:
                    # Log hex dump for diagnosis
                    hex_preview = ' '.join(f'{b:02x}' for b in nal_content[:8])
                    logger.warning(f"[MVC-THREAD] Skipping NAL type {nal_type} with svc_extension_flag=1 "
                                   f"(SVC not supported) - first 8 bytes: {hex_preview}")
                    return

                # SSIF COMBINED MVC FIX: Skip NAL type 20 slices that are too small to be valid
                # A valid MVC slice needs: 1 byte NAL header + 3 bytes MVC extension + at least 4 bytes slice data
                # Minimum = 8 bytes. Smaller NALs are truncated/malformed (common in combined MVC streams
                # where we're only getting dependent view without base view)
                if nal_type == 20 and data_len < 8:
                    if not hasattr(self, '_warned_small_mvc_nal'):
                        hex_preview = ' '.join(f'{b:02x}' for b in nal_content[:data_len])
                        logger.warning(f"[MVC-THREAD] Skipping truncated NAL type 20 ({data_len} bytes < 8 minimum) - "
                                       f"bytes: {hex_preview}")
                        logger.warning(f"[MVC-THREAD] This may indicate missing base view data (SSIF combined MVC issue)")
                        self._warned_small_mvc_nal = True
                    if os.environ.get("SYLC_DECODE_DIAG") == "1":
                        self._diag_trunc20 = getattr(self, '_diag_trunc20', 0) + 1
                        import sys as _sys; _sys.stderr.write(f"[PY-SKIP] truncated NAL20 ({data_len}B) #{self._diag_trunc20}\n")
                    return

                # Log first valid MVC NAL type 20 for diagnosis
                if nal_type == 20 and not hasattr(self, '_logged_first_mvc_nal20'):
                    hex_preview = ' '.join(f'{b:02x}' for b in nal_content[:8])
                    logger.info(f"[MVC-THREAD] First valid MVC NAL type 20 - first 8 bytes: {hex_preview}")
                    self._logged_first_mvc_nal20 = True

        # V7b++++++ CRASH FIX: Additional pointer validation
        try:
            # Verify we can read the first byte (catches invalid pointers)
            _ = p_header[0]
        except (OSError, ValueError) as e:
            logger.error(f"[MVC-THREAD] V7b++++++ Invalid pointer detected: {e}")
            return

        if not hasattr(self, '_nal_decode_count'):
            self._nal_decode_count = 0
        self._nal_decode_count += 1
        nal_type = nal_content[0] & 0x1F if nal_content else -1

        # V29 ENOBUFS PREVENTION: Pre-drain DPB before pushing NAL to prevent saturation
        # The edge264 DPB has 32 slots max. When it fills up, ENOBUFS is returned.
        # For complex MVC streams (like Gravity with large frames), the DPB fills faster.
        # By pre-draining every N NALs, we keep the DPB from saturating.
        if not hasattr(self, '_predrain_counter'):
            self._predrain_counter = 0
        self._predrain_counter += 1
        
        # V48 ALWAYS-PREDRAIN: Drain before EVERY NAL to prevent DPB saturation
        # in heavy B-pyramid streams (Gravity 3D). Previously draining only every 4
        # NALs let pressure build up to 32 → ENOBUFS → infinite retry loop. Draining
        # every NAL keeps the queue moving in sync with worker progress.
        should_predrain = True
        force_aggressive_drain = (self._consecutive_errors > 0) or (self._predrain_counter <= 30)

        # V33g FIX: Skip pre-drain during priming to prevent MVC deadlock
        if self._priming_in_progress:
            should_predrain = False
        
        # V33o FIX: DISABLE V29 PREDRAIN entirely - it corrupts MVC decoder state
        # In synchronous mode (n_threads=0), calling bump_frames + get_frame during
        # NAL processing disrupts the decoder's MVC view pairing state machine.
        # Frames are correctly extracted in POST-DRAIN after all NALs are decoded.
        should_predrain = False

        if should_predrain and self.decoder and not self._seek_requested and not self._seek_in_progress:
            try:
                edge264.edge264_bump_frames(self.decoder)
                predrain_frame = Edge264Frame()
                predrain_count = 0
                predrain_retry = 0
                max_predrain_retry = 8 if force_aggressive_drain else 3  # V29: More retries during priming
                
                while not self._seek_requested and not self._seek_in_progress and not self._stop_requested and self.decoder:
                    try:
                        # Use borrow=0 for immediate DPB slot release
                        ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(predrain_frame), 0)
                    except (OSError, RuntimeError):
                        break
                    
                    if ret == 0:
                        predrain_retry = 0  # Reset on success
                        if self._seek_requested or self._seek_in_progress or self._stop_requested:
                            break
                        # Deliver the frame
                        if self._sync_gate_active or self._seek_in_progress:
                            self._deliver_frame_to_gpu_stabilization(predrain_frame)
                        else:
                            self._deliver_frame_to_gpu(predrain_frame)
                        self.frame_count += 1
                        predrain_count += 1
                    else:
                        predrain_retry += 1
                        if predrain_retry >= max_predrain_retry:
                            break
                        time.sleep(0.0005)  # 0.5ms yield
                        if self.decoder:
                            try:
                                edge264.edge264_bump_frames(self.decoder)
                            except (OSError, RuntimeError):
                                break
                
                if predrain_count > 0:
                    if self._nal_decode_count <= 50 or predrain_count > 2:
                        logger.info(f"[PREDRAIN] Pre-drained {predrain_count} frames before NAL #{self._nal_decode_count}")
                elif force_aggressive_drain and self._nal_decode_count <= 25:
                    logger.info(f"[V29-PREDRAIN] No frames ready at NAL #{self._nal_decode_count}")
                    # Reset consecutive errors since we successfully drained
                    self._consecutive_errors = max(0, self._consecutive_errors - predrain_count)
            except Exception as e:
                logger.warning(f"[V29-PREDRAIN] Exception: {e}")

        # V15 FIX: Revert to BLOCKING mode (non_blocking=0)
        # Non-blocking mode was causing NALs to be dropped when EWOULDBLOCK was returned.
        # The original "watchdog stall" issue at AU #17+ was likely caused by a different bug
        # that has since been fixed. Blocking mode works correctly in all tests.
        # Key insight: With threads=1, there's no background worker to process NALs,
        # so non_blocking=1 always returns EWOULDBLOCK.
        try:
            ret = edge264.edge264_decode_NAL(self.decoder, p_header, p_end, 0, None, None, None)
        except (OSError, RuntimeError, Exception) as e:
            logger.error(f"[MVC-THREAD] Exception in edge264_decode_NAL (ignoring to keep thread alive): {e}")
            return

        # MinGW Errno Values:
        # 11  = EAGAIN (Try again)
        # 140 = EWOULDBLOCK (Operation would block / Busy) - Now handled with retry above!
        # 119 = ENOBUFS (No buffer space available) - Used for Deadlock detection in our patch
        # 104 = EBADMSG (Bad message) - Dependency error

        if ret != 0 and ret != 11 and ret != 140 and ret != 122:  # V31: 122 is EWOULDBLOCK on some Windows builds
            nal_type = nal_content[0] & 0x1F

            # Error 104 (EBADMSG): Dependency not met.
            # Error 119 (ENOBUFS): Deadlock detected (buffer full, no tasks ready).
            if ret == 104 or ret == 119:
                self._consecutive_errors += 1
                if ret == 119:
                    logger.warning(f"[MVC-THREAD] Deadlock (ENOBUFS/119) on NAL {nal_type}. Attempting to drain...")
                    # Attempt to clear deadlock by bumping and draining
                    # V8 CRASH FIX: Check seek flags before edge264 calls in deadlock drain
                    # V14 GRACEFUL ENDING: Also check cleanup flag
                    if self._seek_requested or self._seek_in_progress or self._stop_requested or self._cleanup_in_progress or not self.decoder:
                        return
                    try:
                        edge264.edge264_bump_frames(self.decoder)
                        frame_struct = Edge264Frame()
                        drained_count = 0
                        # V26 ENOBUFS FIX: Increased drain retries and added NAL resubmit
                        enobufs_drain_retry = 0
                        max_enobufs_drain_retry = 20  # V26: Increased from 5 to 20
                        while not self._seek_requested and not self._seek_in_progress and not self._stop_requested and not self._cleanup_in_progress and self.decoder:
                            try:
                                # V33r: Use borrow=1 to keep buffer valid for pixel copy
                                drain_ret = edge264.edge264_get_frame(self.decoder, ctypes.byref(frame_struct), 1)
                            except (OSError, RuntimeError):
                                break
                            if drain_ret == 0:
                                enobufs_drain_retry = 0  # V12: Reset on success
                                # V8 CRASH FIX: Double-check before delivering
                                if self._seek_requested or self._seek_in_progress or self._stop_requested:
                                    # V33r: Still need to return borrowed frame
                                    if self.decoder and frame_struct.return_arg:
                                        try:
                                            edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                        except:
                                            pass
                                    break
                                # V8 SYNC GATE FIX: Use stabilization drain when in sync gate or seek mode
                                try:
                                    if self._sync_gate_active or self._seek_in_progress:
                                        self._deliver_frame_to_gpu_stabilization(frame_struct)
                                    else:
                                        self._deliver_frame_to_gpu(frame_struct)
                                finally:
                                    # V33r: Return frame AFTER copying
                                    if self.decoder and frame_struct.return_arg:
                                        try:
                                            edge264.edge264_return_frame(self.decoder, frame_struct.return_arg)
                                        except:
                                            pass
                                self.frame_count += 1
                                drained_count += 1
                            else:
                                # V12 NON-BLOCKING FIX: Retry with yield for deadlock recovery
                                enobufs_drain_retry += 1
                                if enobufs_drain_retry >= max_enobufs_drain_retry:
                                    break
                                time.sleep(0.001)  # 1ms yield
                                if self.decoder and not self._seek_requested and not self._seek_in_progress:
                                    try:
                                        edge264.edge264_bump_frames(self.decoder)
                                    except (OSError, RuntimeError):
                                        break
                        logger.info(f"[MVC-THREAD] Drained {drained_count} frames to relieve pressure (retries={enobufs_drain_retry}).")

                        # V27 BEHAVIOR (RESTORED): When drain extracted frames, retry NAL
                        # to push it again. When drain failed, SKIP the NAL — re-pushing
                        # a slice that already partially decoded would corrupt the CABAC
                        # state and cause runaway slice decoding (CurrMbAddr=8160 for a
                        # 1200-MB slice). The skip is now safe because edge264 has
                        # FORCE-COMPLETE which prevents the stuck-frame deadlock.
                        max_nal_retries = 3
                        if drained_count > 0:
                            self._consecutive_errors = 0
                            if _nal_retry < max_nal_retries:
                                return self._push_nal_direct(nal_data, start_code_len, force, _nal_retry + 1)
                            return
                        else:
                            if _nal_retry < max_nal_retries:
                                time.sleep(0.005)
                                return self._push_nal_direct(nal_data, start_code_len, force, _nal_retry + 1)
                            return

                    except Exception as e:
                        logger.error(f"[MVC-THREAD] Error during deadlock recovery: {e}")
                else:
                    pass
                    # logger.warning(
                    #    f"[MVC-THREAD] Dependency error (104) on NAL {nal_type}. (Consecutive: {self._consecutive_errors})")

                # Aggressive reset for deadlock or persistent dependency errors
                # V7b: Increased tolerance for 119 if we just drained
                # V16 FIX: Use priming-aware tolerance
                # V44 FIX: Increased tolerances significantly - transient ENOBUFS during DPB warmup
                # are normal after priming. The main loop check now guards against false resets.
                enobufs_tolerance = 100 if self._priming_in_progress else 100  # V44: Unified at 100
                if ret == 119 and self._consecutive_errors < enobufs_tolerance:
                     return

                # V16 FIX: Apply priming-aware tolerance to dependency errors too
                dep_error_tolerance = 200 if self._priming_in_progress else 200  # V44: Unified at 200
                if ret == 119 or self._consecutive_errors > dep_error_tolerance:
                    logger.error(f"[MVC-THREAD] Unrecoverable state ({ret}, errors={self._consecutive_errors}). Flagging for full decoder reset.")
                    self._needs_full_reset = True
                # Return here is okay because these are fatal/reset conditions
                return
            else:
                self._consecutive_errors = 0

            # Log warning but DON'T return for other errors (like 129/ENOTSUP)
            # This allows processing subsequent NALs in the same AU
            logger.warning(f"decode_NAL returned {ret} for NAL type {nal_type}")

            # 140 is EWOULDBLOCK, NOT EFAULT. We handle it above, but keep this check
            # if any other code creeps in. Real EFAULT is 14.
            if ret == 14:
                self._efault_count += 1
                if not self._fatal_error:
                    self._fatal_error = True
                    logger.critical(f"EFAULT (14) on NAL {nal_type}. BUG in edge264.", exc_info=True)
                    self.error.emit("CRITICAL BUG in EDGE264 decoder.")
        else:
            self._consecutive_errors = 0

    def dump_debug_state(self):
        pass

    def _emit_single_frame(self, frame_data):
        """Helper to emit signals for a single frame dictionary."""
        try:
            # V14 GRACEFUL ENDING: Do not emit during cleanup
            if self._cleanup_in_progress:
                return

            left_planes = frame_data['left']
            right_planes = frame_data['right']

            # Swapchain commit barrier: validate left/right plane shape coherence
            # under the frame_delivery lock before signaling the GUI thread. This
            # prevents the GUI from reading a half-built frame (partial copy or
            # mid-cleanup zeroed-out plane mistakenly published to the queue).
            # Cheap O(1) check — numpy shape access is a Python attribute read.
            try:
                yl, ul, vl = left_planes
                yr, ur, vr = right_planes
                if (yl.shape != yr.shape or ul.shape != ur.shape or
                    vl.shape != vr.shape or yl.size == 0 or ul.size == 0):
                    if not hasattr(self, '_swapchain_drop_count'):
                        self._swapchain_drop_count = 0
                    self._swapchain_drop_count += 1
                    if self._swapchain_drop_count <= 5:
                        logger.warning(
                            f"[SWAPCHAIN] frame {self.frame_count}: plane-shape "
                            f"mismatch (L={yl.shape}/{ul.shape}/{vl.shape}, "
                            f"R={yr.shape}/{ur.shape}/{vr.shape}) — dropping"
                        )
                    return
            except (KeyError, ValueError, AttributeError, TypeError) as _e:
                logger.warning(f"[SWAPCHAIN] unpack/shape error on emit: {_e}")
                return

            # Emit generic ready signal AFTER barrier validation
            if velvet_probe.ENABLED:
                velvet_probe.on_emit(len(self.presentation_queue))
            self.frameReady.emit()

            # V7b SYNC FIX: ALWAYS emit YUV data to allow GUI to route to multiple widgets
            # References are safe because get_plane created new objects that won't be reused by decoder.
            # DIAG: Log frame emission
            if self.frame_count <= 10 or self.frame_count % 100 == 0:
                logger.debug(f"[DIAG] EMIT frameYUVReady #{self.frame_count}")
            self.frameYUVReady.emit(left_planes, right_planes)

            # Legacy signal for backward compatibility (not used in V7b)
            if not self._display_widget:
                # self.frameDecoded.emit((*left_copies, *right_copies)) # Disable legacy copy too
                pass

            # V7b TIMELINE FIX: ALWAYS emit timestamp for timeline progression
            # Timeline needs this signal even when audio sync is disabled
            # V60: 'timestamp' is now always set (SYNC-PTS stamps at emission), but
            # 'id' only exists when _enable_audio_sync populated it — use .get().
            if 'timestamp' in frame_data:
                self.frameTimestampReady.emit(frame_data.get('id', 0), frame_data['timestamp'], frame_data.get('poc', 0))

        except Exception as e:
            logger.error(f"[MVC-THREAD] Error emitting frame: {e}")

    def _queue_frame_for_display(self, frame_data):
        """Reordering buffer shared by both ctypes and pybind decoding paths."""
        try:
            current_poc = frame_data.get('poc', 0)

            # MONOTONE-POC (2026-07-13): edge264 now floors every IDR's POC
            # strictly above the previous GOP's max (IDR-floor patch in
            # edge264_headers.c) on top of its continuous cross-IDR POC
            # chaining, so POC alone is a globally valid display key for
            # every stream it decodes. The former epoch heuristics
            # (force_next_epoch one-shot + ±50 POC-step thresholds) are gone:
            # under MT delivery the one-shot flag could land on an in-flight
            # OLD frame (held ~100 emissions = a multi-second-late frame) and
            # the +50 "straggler" rule mis-tagged whole GOPs after a rebased
            # IDR jump (the one-frame backward jump at scene cuts). Measured
            # against the single-thread delivery order on Avatar BD3D
            # (3000 frames, 18 IDRs, MT=4): epoch logic = 30 inversions with
            # the old DLL / 2 with the patched one; pure-POC sort = 0
            # inversions, 0 displacement, at REORDER_DEPTH=4.
            self.frame_buffer.append({
                'sort_key': (0, current_poc),
                'data': frame_data
            })

            # Sort by POC (globally display-monotone, see above)
            self.frame_buffer.sort(key=lambda x: x['sort_key'])

            # DIAG: Log frame_buffer state periodically
            if len(self.frame_buffer) <= 10 or len(self.frame_buffer) % 50 == 0:
                logger.debug(f"[DIAG] frame_buffer: {len(self.frame_buffer)} frames, REORDER_DEPTH={self.REORDER_DEPTH}, startup_emitted={self._startup_frames_emitted}")

            # V7c STARTUP FIX: Bypass REORDER_DEPTH for first several frames after seek
            # This establishes visual flow before B-frame reordering kicks in
            startup_bypass = self._startup_frames_emitted < (self.REORDER_DEPTH + 2)

            # V59b MT-ORDER GUARD: with worker threads, frames ARRIVE with POC
            # jitter; blind arrival-order bypass scrambled the first post-seek
            # frames (visible strobe). Emit the very first frame immediately
            # (visual feedback), then require a small sorted lookahead (3) so
            # bypass frames leave in display order. Costs ~2 frames of decode
            # time (~15ms at MT speed), not perceptible.
            _bypass_min = 1 if self._startup_frames_emitted == 0 else 3
            if startup_bypass and len(self.frame_buffer) >= _bypass_min:
                item = self.frame_buffer.pop(0)
                self._repair_and_queue(item['data'])
                self._startup_frames_emitted += 1
                if self._startup_frames_emitted <= 10:
                    logger.debug(f"[DIAG] STARTUP: Frame #{self._startup_frames_emitted} pushed immediately (bypassing reorder)")
            # Enforce reordering depth after startup is complete
            # Only push to presentation if we have enough depth or if we're flushing
            elif len(self.frame_buffer) > self.REORDER_DEPTH:
                # Pop the oldest frame (lowest epoch, lowest POC)
                item = self.frame_buffer.pop(0)
                self._repair_and_queue(item['data'])
                # DIAG: Log when frames move to presentation_queue
                if len(self.presentation_queue) <= 10 or len(self.presentation_queue) % 50 == 0:
                    logger.debug(f"[DIAG] -> presentation_queue: {len(self.presentation_queue)} frames")

        except Exception as exc:
            logger.error(f"[MVC-THREAD] Failed to queue frame: {exc}")

    def _deliver_frame_to_gpu_stabilization(self, frame):
        """V8 STABILIZATION: Buffer frames during post-seek stabilization.
        This is called WHILE _seek_in_progress=True, so we skip those checks.
        Frames are only buffered, not displayed."""
        # V14 CRASH FIX: Acquire frame delivery lock
        if not self._frame_delivery_lock.acquire(timeout=0.1):
            return

        try:
            self._frame_delivery_active = True

            # V14 GRACEFUL ENDING: Check cleanup flag FIRST
            if self._cleanup_in_progress or self._stop_requested or not self.decoder:
                return

            w, h = frame.width_Y, frame.height_Y
            if w <= 0 or h <= 0:
                return

            # Use actual chroma dimensions from frame
            cw, ch = frame.width_C, frame.height_C
            if cw <= 0 or ch <= 0:
                cw, ch = w // 2, h // 2
            sy, sc = frame.stride_Y, frame.stride_C

            def get_plane_safe(base_ptr, w, h, s):
                """Safe plane copy for stabilization - V13: Added cleanup check."""
                try:
                    # V13 CRASH FIX: Check cleanup flag before any memory access
                    if self._cleanup_in_progress or self._stop_requested:
                        return np.zeros((h, w), dtype=np.uint8)
                    if not base_ptr or w <= 0 or h <= 0 or s < w:
                        return np.zeros((h, w), dtype=np.uint8)
                    return _PLANE_POOL.copy(np.ctypeslib.as_array(base_ptr, shape=(h, s))[:, :w])
                except Exception:
                    return np.zeros((h, w), dtype=np.uint8)

            y_l = get_plane_safe(frame.samples[0], w, h, sy)

            # Read U and V planes directly from edge264.
            # edge264 layout: samples[1] = pointer to U[0][0], samples[2] = pointer to V[0][0]
            # samples[2] - samples[1] == stride_C / 2 == cw (U and V live in the same row,
            # U at offset 0..cw-1, V at offset cw..2*cw-1). Reading samples[1] or samples[2]
            # with stride sc gives the correct plane.
            u_l = get_plane_safe(frame.samples[1], cw, ch, sc) if frame.samples[1] else np.full((ch, cw), 128, dtype=np.uint8)
            v_l = get_plane_safe(frame.samples[2], cw, ch, sc) if frame.samples[2] else np.full((ch, cw), 128, dtype=np.uint8)

            if frame.samples_mvc[0]:
                y_r = get_plane_safe(frame.samples_mvc[0], w, h, sy)
                u_r = get_plane_safe(frame.samples_mvc[1], cw, ch, sc) if frame.samples_mvc[1] else np.full((ch, cw), 128, dtype=np.uint8)
                v_r = get_plane_safe(frame.samples_mvc[2], cw, ch, sc) if frame.samples_mvc[2] else np.full((ch, cw), 128, dtype=np.uint8)
            else:
                y_r, u_r, v_r = y_l, u_l, v_l

            poc_base = frame.PictureOrderCnt
            poc_mvc = frame.PictureOrderCnt_mvc
            current_poc = max(poc_base, poc_mvc) if frame.samples_mvc[0] else poc_base

            # Calculate timestamp for this frame
            frame_timestamp = self.base_timestamp + (self.frame_count * self.target_frame_time)

            # DIAG: log first 30 stabilization-delivered frames pixel stats
            if not hasattr(self, '_pixstat_stab_count'):
                self._pixstat_stab_count = 0
            if self._pixstat_stab_count < 30:
                self._pixstat_stab_count += 1
                try:
                    y_avg = float(y_l.mean()) if y_l.size else -1.0
                    y_min = int(y_l.min()) if y_l.size else -1
                    y_max = int(y_l.max()) if y_l.size else -1
                    u_avg = float(u_l.mean()) if u_l.size else -1.0
                    v_avg = float(v_l.mean()) if v_l.size else -1.0
                    y_first = ' '.join(f'{b:02x}' for b in y_l[0, :16].tolist()) if y_l.size else ''
                    logger.info(f"[PIXSTAT-STAB #{self._pixstat_stab_count}] POC={current_poc} Y_avg={y_avg:.2f} Y_min={y_min} Y_max={y_max} U_avg={u_avg:.2f} V_avg={v_avg:.2f} Y[0..15]={y_first}")
                except Exception as _e:
                    logger.info(f"[PIXSTAT-STAB #{self._pixstat_stab_count}] dump error: {_e}")

            frame_data = {
                'poc': current_poc,
                'left': (y_l, u_l, v_l),
                'right': (y_r, u_r, v_r),
                'timestamp': frame_timestamp,
                'id': frame.FrameId,
            }

            # V60 SYNC-PTS: stamp with the true container PTS (keeps the PTS heap
            # accounting consistent even for stabilization-path frames).
            try:
                self._assign_emit_pts(frame_data)
            except Exception:
                pass
            # Queue directly to presentation queue (skip reordering during stabilization)
            self.presentation_queue.append(frame_data)
            self.frame_count += 1

        except Exception as e:
            logger.warning(f"[MVC-THREAD] Stabilization frame copy error: {e}")
        finally:
            # V14 CRASH FIX: Always release the frame delivery lock
            self._frame_delivery_active = False
            self._frame_delivery_lock.release()

    def _repair_and_queue(self, data):
        """Push a frame to the presentation queue, first re-pairing its dependent (right-eye)
        view by POC: replace the frame's dependent with the pooled dependent whose POC equals
        this frame's base POC, so the two views show the SAME instant. The reorder buffer
        depth absorbs the ~1-frame delay before the matching dependent is decoded. Falls back
        to the frame's own dependent when the match isn't pooled (e.g. startup / genuine gap)."""
        try:
            bp = data.get('base_poc')
            if bp is not None:
                rep = self._dep_pool.get(bp)
                if rep is not None:
                    data['right'] = rep
        except Exception:
            pass
        # V60 SYNC-PTS: overwrite the arrival-order synthetic stamp with the true
        # container PTS matching this frame's display slot.
        try:
            self._assign_emit_pts(data)
        except Exception:
            pass
        # THUMB HARVEST: one 320x180 copy every ~10s of playback. Zero disk I/O
        # (planes are already numpy copies) — works on ALL sources incl. discs.
        # SYLC_THUMB_HARVEST=0 disables (diagnostic kill-switch).
        try:
            if (time.time() - self._last_thumb_harvest >= 10.0 and data.get('left')
                    and os.environ.get("SYLC_THUMB_HARVEST") != "0"):
                self._last_thumb_harvest = time.time()
                if os.environ.get("SYLC_THUMB_DIAG") == "1":
                    import sys as _sys
                    _l = data.get('left')
                    _sys.stderr.write(f"[THUMB-DIAG] harvest firing: types={[type(x).__name__ for x in _l]} "
                                      f"shapes={[getattr(x, 'shape', None) for x in _l]} "
                                      f"contig={[getattr(getattr(x, 'flags', None), 'c_contiguous', None) for x in _l]} "
                                      f"ts={data.get('timestamp')}\n")
                from thumbnail_service import planes_to_qimage_320
                _timg = planes_to_qimage_320(*data['left'], layout=self._thumb_layout)
                if _timg is not None and data.get('timestamp') is not None:
                    self.thumbnailHarvested.emit(float(data['timestamp']), _timg)
                    if os.environ.get("SYLC_THUMB_DIAG") == "1":
                        import sys as _sys
                        _sys.stderr.write(f"[THUMB-DIAG] harvest emitted ts={data.get('timestamp'):.3f}\n")
        except Exception:
            pass
        self.presentation_queue.append(data)

    def _deliver_frame_to_gpu(self, frame):
        # self.frameReady.emit()  <-- MOVED to _emit_single_frame
        if not hasattr(self, '_valid_frames_received'):
            self._valid_frames_received = 0

        # V14 CRASH FIX: Acquire frame delivery lock to prevent race with seek
        # This ensures the decoder memory isn't freed while we're accessing it
        if not self._frame_delivery_lock.acquire(timeout=0.1):
            # Timeout - seek is probably waiting, abort frame delivery
            return

        try:
            self._frame_delivery_active = True
            _vp_dt0 = velvet_probe.now() if velvet_probe.ENABLED else 0.0

            # V14 GRACEFUL ENDING: Check cleanup flag FIRST
            if self._cleanup_in_progress:
                return

            # V8 CRASH FIX: Check BOTH _seek_requested AND _seek_in_progress
            # _seek_requested is set by GUI thread immediately
            # _seek_in_progress is set by decoder thread when handling the seek
            # There's a window where _seek_requested=True but _seek_in_progress=False
            if self._seek_requested or self._seek_in_progress or self._stop_requested or not self.decoder:
                return

            # CRITICAL: Verify frame validity FIRST
            w, h = frame.width_Y, frame.height_Y
            if w <= 0 or h <= 0:
                logger.error(f"[FRAME-DELIVERY] Invalid frame dimensions: {w}x{h}")
                return

            # FIX: Use actual chroma dimensions from frame, not assumed 4:2:0
            # This fixes potential issues with non-standard chroma subsampling
            cw, ch = frame.width_C, frame.height_C
            if cw <= 0 or ch <= 0:
                # Fallback to 4:2:0 assumption if chroma dims not set
                cw, ch = w // 2, h // 2
            sy, sc = frame.stride_Y, frame.stride_C


            # Audio synchronization (disabled by default to avoid crashes)
            frame_timestamp_info = {}
            if self._enable_audio_sync:
                try:
                    poc = frame.PictureOrderCnt
                    frame_id = frame.FrameId

                    if self._first_poc is None:
                        self._first_poc = poc
                        logger.info(f"[SYNC] First reference POC: {poc}")

                    # Compute the relative timestamp in seconds
                    # CRITICAL FIX: Use frame count instead of POC.
                    # POC gap varies (2 for 24fps, 1 for others), causing 2x speedup if assumed 2.
                    # Using frame_count is robust and matches target FPS perfectly.
                    frame_timestamp = self.base_timestamp + (self.frame_count * self.target_frame_time)

                    # Store for emission later
                    frame_timestamp_info = {
                        'timestamp': frame_timestamp,
                        'id': frame_id,
                        'poc': poc
                    }

                    if not hasattr(self, '_sync_logged'):
                        self._sync_logged = True
                except Exception as e:
                    logger.error(f"[SYNC] Error calculating timestamp: {e}")

            def get_plane(base_ptr, w, h, s):
                try:
                    if self._cleanup_in_progress:
                        return np.zeros((h, w), dtype=np.uint8)

                    if self._seek_requested or self._seek_in_progress or self._stop_requested or not self.decoder:
                        return np.zeros((h, w), dtype=np.uint8)

                    if not base_ptr:
                        return np.zeros((h, w), dtype=np.uint8)

                    if w <= 0 or h <= 0 or s < w:
                        logger.error(f"[get_plane] Invalid dimensions: w={w}, h={h}, s={s}")
                        return np.zeros((h, w), dtype=np.uint8)

                    if self._cleanup_in_progress or self._seek_requested or self._seek_in_progress or self._stop_requested:
                        return np.zeros((h, w), dtype=np.uint8)

                    arr = _PLANE_POOL.copy(np.ctypeslib.as_array(base_ptr, shape=(h, s))[:, :w])
                    return arr
                except OSError as e:
                    logger.error(f"[get_plane] Fatal error accessing memory: {e}")
                    return np.zeros((h, w), dtype=np.uint8)
                except Exception as e:
                    logger.error(f"[get_plane] Error copying plane: {e}")
                    return np.zeros((h, w), dtype=np.uint8)

            # V8 CRASH FIX: Check seek state before accessing frame.samples
            # Check BOTH _seek_requested (set by GUI) AND _seek_in_progress (set by decoder)
            if self._seek_requested or self._seek_in_progress or self._stop_requested or not self.decoder:
                return

            # V8 CRASH FIX: Validate samples[0] is not NULL (freshly initialized frame_struct)
            # After frame_struct = Edge264Frame(), all samples are NULL until edge264_get_frame fills them
            if not frame.samples[0]:
                logger.warning("[FRAME-DELIVERY] V8: frame.samples[0] is NULL - skipping frame")
                return

            y_l = get_plane(frame.samples[0], w, h, sy)
            
            # V36 FIX: edge264 outputs U and V as SEPARATE planes (not side-by-side!)
            # samples[1] = U plane only, samples[2] = V plane only
            # stride_C is just alignment padding, NOT an indicator of NV12 format
            # V36b: Add same safety checks as get_plane to prevent crash during seek/cleanup
            uv_safe = (not self._cleanup_in_progress and
                       not self._seek_requested and
                       not self._seek_in_progress and
                       not self._stop_requested and
                       self.decoder and
                       frame.samples[1] and frame.samples[2] and
                       cw > 0 and ch > 0 and sc >= cw)

            if uv_safe:
                try:
                    # V38 CORRECT FIX: edge264 outputs U and V as SEPARATE planes
                    # samples[1] = pointer to U plane (cw bytes per row, stride=sc)
                    # samples[2] = pointer to V plane (cw bytes per row, stride=sc)
                    # V is located at samples[1] + stride_C/2, but samples[2] already points there

                    # Read U from samples[1]
                    u_l = get_plane(frame.samples[1], cw, ch, sc)

                    # Read V from samples[2]
                    v_l = get_plane(frame.samples[2], cw, ch, sc)

                except Exception as e:
                    logger.warning(f"[UV-V38] Error reading UV: {e}")
                    u_l = np.full((ch, cw), 128, dtype=np.uint8)
                    v_l = np.full((ch, cw), 128, dtype=np.uint8)
            else:
                u_l = np.full((ch, cw), 128, dtype=np.uint8)
                v_l = np.full((ch, cw), 128, dtype=np.uint8)

            # V8 CRASH FIX: Check seek state before accessing frame.samples_mvc
            if self._seek_requested or self._seek_in_progress or self._stop_requested or not self.decoder:
                return

            if frame.samples_mvc[0]:
                y_r = get_plane(frame.samples_mvc[0], w, h, sy)
                # V38 FIX: Read U and V from SEPARATE planes for MVC view
                mvc_uv_safe = (not self._cleanup_in_progress and
                               not self._seek_requested and
                               not self._seek_in_progress and
                               not self._stop_requested and
                               self.decoder and
                               frame.samples_mvc[1] and frame.samples_mvc[2])

                if mvc_uv_safe:
                    try:
                        # V38: Read U and V separately using get_plane
                        u_r = get_plane(frame.samples_mvc[1], cw, ch, sc)
                        v_r = get_plane(frame.samples_mvc[2], cw, ch, sc)
                    except Exception as e:
                        logger.warning(f"[UV-V38] Error reading MVC UV: {e}")
                        u_r = np.full((ch, cw), 128, dtype=np.uint8)
                        v_r = np.full((ch, cw), 128, dtype=np.uint8)
                else:
                    # Fallback: duplicate left view UV
                    u_r, v_r = u_l.copy(), v_l.copy()
            else:
                # If no MVC data, duplicate left view to avoid black screen on right
                y_r, u_r, v_r = y_l, u_l, v_l

            # V8 CRASH FIX: Final check before queuing frame
            # All planes have been copied, verify we should still deliver
            if self._seek_requested or self._seek_in_progress or self._stop_requested:
                return

            # --- REORDERING LOGIC ---
            # Buffer the frame data

            # V7b FIX: Use max(base, mvc) for POC like ultimate_mvc_player.py
            # This ensures correct ordering if views have slightly different POCs
            poc_base = frame.PictureOrderCnt
            poc_mvc = frame.PictureOrderCnt_mvc

            # Reorder by the BASE view's POC (display order), NOT max(base, dep): with the
            # decoder's off-by-one view pairing a B-frame's dependent POC equals the anchor's,
            # so max() collides B-frames with anchors and scrambles the reorder. The dependent
            # view is re-associated by POC separately (_dep_pool / _repair_and_queue).
            current_poc = poc_base


            # Pool this dependent picture by its OWN POC so a later base of the same POC
            # can re-pair with it (see _repair_and_queue). Only real dependent views.
            if frame.samples_mvc[0]:
                self._dep_pool[poc_mvc] = (y_r, u_r, v_r)
                if len(self._dep_pool) > 16:
                    for _k in sorted(self._dep_pool)[:-12]:
                        del self._dep_pool[_k]

            frame_data = {
                'poc': current_poc,
                'base_poc': poc_base,
                'left': (y_l, u_l, v_l),
                'right': (y_r, u_r, v_r),
                **frame_timestamp_info
            }

            self._queue_frame_for_display(frame_data)

        except Exception as e:
            logger.error(f"[MVC-THREAD] Deliver error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if velvet_probe.ENABLED and _vp_dt0:
                velvet_probe.record('deliver_ms', (velvet_probe.now() - _vp_dt0) * 1000.0)
            # V14 CRASH FIX: Always release the frame delivery lock
            self._frame_delivery_active = False
            self._frame_delivery_lock.release()

    def _deliver_pybind_frame(self, frame, timestamp_seconds=None):
        """Fast-path delivery using the C++ decoder bindings."""
        try:
            # V14 GRACEFUL ENDING: Check cleanup flag before memory access
            if self._cleanup_in_progress:
                return

            w = getattr(frame.base_view, "width", 0)
            h = getattr(frame.base_view, "height", 0)
            if w <= 0 or h <= 0:
                return

            def to_np(buf, width, height, stride):
                if buf is None:
                    return np.zeros((height, width), dtype=np.uint8)
                arr = np.array(buf, copy=False)
                if arr.size == 0:
                    return np.zeros((height, width), dtype=np.uint8)
                arr = arr.reshape((height, stride))
                return np.array(arr[:, :width], copy=True)

            def extract_uv_planes(view, width, height):
                """Extract U and V planes from pybind view, handling side-by-side UV format.

                edge264 stores UV in side-by-side format: [U0...U(w/2-1)][V0...V(w/2-1)] per row
                stride_c = 2 * (width/2) = width, cb_plane and cr_plane point to adjacent halves.

                CRITICAL: cr_plane has shape (h/2, stride_c) but starts at offset width/2 into the
                UV buffer. Creating an array of that size would read past the end of the buffer!
                Solution: Read from cb_plane (properly sized) and split into U and V halves.
                """
                cw = width // 2  # Chroma width
                ch = height // 2  # Chroma height
                sc = getattr(view, 'stride_c', 0)

                # Check for side-by-side format: stride_c = 2 * chroma_width
                if sc == 2 * cw:
                    # Read full UV plane from cb_plane (which is properly sized)
                    cb_plane = view.cb_plane
                    if cb_plane is not None:
                        uv_arr = np.array(cb_plane, copy=False)
                        if uv_arr.size >= ch * sc:
                            uv_arr = uv_arr.reshape((ch, sc))
                            # Split: first half = U (Cb), second half = V (Cr)
                            u_raw = np.array(uv_arr[:, :cw], copy=True)
                            v_raw = np.array(uv_arr[:, cw:], copy=True)
                            
                            return u_raw, v_raw

                # Fallback: standard I420 format with separate planes
                u_plane = to_np(view.cb_plane, cw, ch, sc)
                v_plane = to_np(view.cr_plane, cw, ch, sc)
                return u_plane, v_plane

            y_l = to_np(frame.base_view.y_plane, w, h, frame.base_view.stride_y)
            u_l, v_l = extract_uv_planes(frame.base_view, w, h)

            if getattr(frame, "has_mvc", False):
                y_r = to_np(frame.dependent_view.y_plane, w, h, frame.dependent_view.stride_y)
                u_r, v_r = extract_uv_planes(frame.dependent_view, w, h)
            else:
                y_r, u_r, v_r = y_l, u_l, v_l

            if y_l.size > 0 and y_l.sum() == 0:
                logger.warning("[MVC-THREAD] Left Y-plane is all black (sum is 0).")
            if y_r.size > 0 and y_r.sum() == 0:
                logger.warning("[MVC-THREAD] Right Y-plane is all black (sum is 0).")

            frame_data = {
                'poc': getattr(frame, "frame_id", self.frame_count),
                'left': (y_l, u_l, v_l),
                'right': (y_r, u_r, v_r)
            }

            if timestamp_seconds is not None:
                frame_data.update({
                    'timestamp': timestamp_seconds,
                    'id': getattr(frame, "frame_id", self.frame_count),
                    'poc': frame_data['poc']
                })

            self._queue_frame_for_display(frame_data)
        except Exception as exc:
            logger.error(f"[MVC-THREAD] Pybind delivery failed: {exc}")

    def _precise_wait(self, duration):
        """Minimal wait to pace frame display without risking crashes.
        V13 CRASH FIX: Use time.sleep instead of threading.Event().wait()
        Creating new Event objects triggers Windows 0xe24c4a02 exceptions.
        """
        if duration <= 0:
            return

        # Check flags first (without mutex to avoid potential deadlock)
        if self._seek_requested or self._seek_in_progress or self._stop_requested:
            return

        # V13 CRASH FIX: Use simple time.sleep - it's more stable on Windows
        # than creating new threading.Event objects repeatedly
        try:
            if duration > 0.001:
                time.sleep(min(duration, 0.005))  # Max 5ms single sleep
        except (OSError, Exception):
            pass  # Ignore any exceptions during sleep

    def _ensure_presenter(self):
        """V54: start the dedicated presenter thread once (idempotent). Decouples
        presentation pacing from decode so an I-frame decode spike can't freeze video."""
        t = getattr(self, '_presenter_thread', None)
        if self._presenter_active and t is not None and t.is_alive():
            return
        self._presenter_active = True
        # V56: tighten the GIL switch interval (default 5ms). With decode/presenter/GUI
        # all contending for the single GIL, a 5ms interval let the decode/GUI threads
        # hold the GIL long enough to starve the real-time presenter for tens of ms
        # (the residual ~80ms freezes). A short interval lets the presenter reclaim the
        # GIL promptly so it can keep its frame cadence.
        try:
            import sys as _sys
            _sys.setswitchinterval(0.0005)
        except Exception:
            pass
        import threading
        self._presenter_thread = threading.Thread(
            target=self._presenter_loop, name="MVC-Presenter", daemon=True)
        self._presenter_thread.start()
        logger.info("[PRESENTER] V54 presenter thread started (present decoupled from decode)")

    def _presenter_loop(self):
        """V54: drain the presentation queue at a steady cadence, independent of the
        decode thread. _process_presentation_queue self-paces (gates on target_frame_time
        and precise_waits when not due), so a short sleep here avoids a busy-spin."""
        try:
            while self._presenter_active and not self._stop_requested:
                try:
                    if not self._cleanup_in_progress:
                        self._process_presentation_queue()
                except Exception:
                    pass
                try:
                    time.sleep(0.001)
                except Exception:
                    pass
        finally:
            logger.info("[PRESENTER] V54 presenter thread exiting")

    def _stop_presenter(self):
        """V54: stop the presenter thread (best-effort join)."""
        self._presenter_active = False
        t = getattr(self, '_presenter_thread', None)
        if t is not None:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
            self._presenter_thread = None

    def _await_queue_space(self):
        """V54 backpressure: pause decode while the presentation buffer is full, so the
        decode thread (now free of the present cadence) can't out-run the presenter and
        evict frames it hasn't shown yet (which would fast-forward the video). Bounded
        so a seeking/stalled presenter can never deadlock decode. MUST be called from the
        decode path OUTSIDE _frame_delivery_lock (else it deadlocks the presenter's emit)."""
        if not self._presenter_active:
            return
        start = time.time()
        while (len(self.presentation_queue) >= self._present_high_water
               and not self._seek_requested and not self._seek_in_progress
               and not self._stop_requested and not self._cleanup_in_progress):
            if (time.time() - start) > 0.5:
                break
            try:
                time.sleep(0.003)
            except Exception:
                break

    def _process_presentation_queue(self):
        """V7b SOL 2B: Hybrid timing FPS + audio check."""
        # V14 GRACEFUL ENDING: Check cleanup flag before anything
        # DIAG: Track calls to _process_presentation_queue
        if not hasattr(self, '_ppq_diag_count'):
            self._ppq_diag_count = 0
        self._ppq_diag_count += 1
        if self._ppq_diag_count <= 20 or self._ppq_diag_count % 500 == 0:
            logger.debug(f"[DIAG] _process_presentation_queue call #{self._ppq_diag_count}, queue_len={len(self.presentation_queue)}, fb_len={len(self.frame_buffer)}")

        if self._cleanup_in_progress:
            return

        # V8 CRASH FIX: Use mutex to safely check BOTH seek flags
        # _seek_requested is set by GUI immediately, _seek_in_progress by decoder later
        self.mutex.lock()
        is_seek_requested = self._seek_requested
        is_seeking = self._seek_in_progress
        is_stopped = self._stop_requested
        self.mutex.unlock()

        if is_seek_requested or is_seeking or is_stopped:
            return

        # CRITICAL FIX: Catch 0xe24c4a02 crashes during seek
        try:
            # Double-check after mutex release (defensive)
            if self._seek_requested or self._seek_in_progress:
                return

            if not self.presentation_queue:
                self._precise_wait(0.001)
                return

            # Pause handling
            if self._is_paused:
                self._precise_wait(0.010)
                return

            # Check timing
            current_time = time.time()
            if not hasattr(self, '_last_display_time'):
                self._last_display_time = current_time

            time_since_last = current_time - self._last_display_time

            # Get drift
            self._timing_adjustment_mutex.lock()
            drift_ms = self._timing_drift_ms
            self._timing_adjustment_mutex.unlock()

            # V7b HARD RESYNC (Fire & Correct Strategy)
            # DEPRECATED: Replaced by "Hold & Catch-Up" strategy.
            # We disable this to prevent fighting with the new logic.
            if self._sync_pending:
                 self._sync_pending = False

            # PLL Logic (PI Controller)
            # Dead-zone: drift < 10ms is below perception threshold; correcting it
            # creates continuous micro-jitter in frame pacing.
            adjusted_frame_time = self.target_frame_time
            if abs(drift_ms) > 10.0:
                Kp = 0.05
                Ki = 0.001
                max_integral = self.target_frame_time * 0.10
                self._pll_integral += (drift_ms / 1000.0) * Ki
                self._pll_integral = max(min(self._pll_integral, max_integral), -max_integral)
                p_term = (drift_ms / 1000.0) * Kp
                correction = p_term + self._pll_integral
                max_correction = self.target_frame_time * 0.2
                correction = max(min(correction, max_correction), -max_correction)
                adjusted_frame_time += correction

            # DIAG: Log timing state
            if self._ppq_diag_count <= 50 or self._ppq_diag_count % 500 == 0:
                logger.debug(f"[DIAG] timing: time_since_last={time_since_last*1000:.1f}ms, adj_frame_time={adjusted_frame_time*1000:.1f}ms, first_frame={self._first_frame_after_seek}, waiting_audio={self._waiting_for_audio_edge}")

            if time_since_last >= adjusted_frame_time:
                # V7b STRATEGY: Show & Hold
                # 1. Show the FIRST frame immediately (Visual Feedback)
                if self._first_frame_after_seek:
                    frame = self.presentation_queue.popleft()

                    try:
                        self._emit_single_frame(frame)
                        self._last_display_time = current_time
                    except Exception as e:
                        logger.error(f"[MVC-THREAD] Error emitting frame: {e}")

                    self._first_frame_after_seek = False
                    return # Exit to hold this frame on screen

                # 2. HOLD logic: Wait for Audio Edge
                # V7b+ SYNC FIX: Use base_timestamp (the actual IDR timestamp) instead of _seek_target
                # The GUI now syncs MPV to base_timestamp via seekIDRFound signal
                if self._waiting_for_audio_edge:
                    self._audio_mutex.lock()
                    audio_raw = self._external_audio_clock
                    self._audio_mutex.unlock()

                    # V7b+ SYNC FIX: Compare with base_timestamp (IDR position), not _seek_target (user target)
                    diff = audio_raw - self.base_timestamp

                    # V38 BUG FIX: Calculate timeout FIRST to avoid infinite block
                    # Previous bug: abs(diff) > 5.0 returned without checking timeout
                    # V61: 2.0 -> 5.0s — a deep seek in a big file can need a few
                    # seconds of demux/cache work before audio actually flows; a
                    # premature release marched the video ahead of dead audio and
                    # V12 then burned the excess in 0.6s hold cycles (the post-seek
                    # freeze-churn). Held video (first frame shown) is the right UX.
                    is_timeout = (time.time() - self._audio_gate_timeout_start) > 5.0

                    # FILTER STALE TIMESTAMPS (> 5s away from IDR position)
                    # BUT always check timeout first!
                    if abs(diff) > 5.0:
                        if not is_timeout:
                            self._precise_wait(0.015)  # SEEK-PERF: finer poll = faster release
                            return
                        else:
                            # V38 FIX: Timeout! Release hold even with stale timestamp
                            self._waiting_for_audio_edge = False
                            self._catchup_mode = True
                            logger.warning(f"[V38-SYNC] Stale timestamp ({diff:.1f}s) + timeout. Force-releasing video.")
                            # Fall through to emit frame

                    # Check if audio has moved significantly (> 250ms from IDR position)
                    # V7b+ SYNC FIX: Now audio should be at base_timestamp since GUI seeked MPV there
                    has_audio_moved = abs(diff) < 1.0 and audio_raw > 0.1

                    # V61 SETTLE: mpv reports the target position BEFORE audio actually
                    # flows (and the ATOMIC SYNC pre-pushes it) — position alone is not
                    # proof of playing audio. Require the clock to sit near base for
                    # 250ms before releasing; the small backlog this builds is erased
                    # instantly by the V50 bulk-drop on the first synced frame.
                    if has_audio_moved and not is_timeout:
                        _near_since = getattr(self, '_audio_gate_near_since', None)
                        if _near_since is None:
                            self._audio_gate_near_since = time.time()
                            self._precise_wait(0.015)
                            return
                        if (time.time() - _near_since) < 0.250:
                            self._precise_wait(0.015)
                            return
                    else:
                        self._audio_gate_near_since = None

                    if not has_audio_moved and not is_timeout:
                        self._precise_wait(0.015)  # SEEK-PERF: finer poll = faster release
                        return

                    # Audio started OR Timeout! Release the Hold.
                    logger.debug(f"[SEEK-PERF] audio-hold released after {time.time() - self._audio_gate_timeout_start:.2f}s "
                                f"(diff={diff:.2f}s timeout={is_timeout})")
                    self._waiting_for_audio_edge = False
                    self._catchup_mode = True

                    # V7b+ SYNC FIX: DO NOT recalculate base_timestamp here!
                    # The GUI already synced MPV to our base_timestamp via seekIDRFound signal.
                    # Any recalculation here would UNDO that sync and cause desync.
                    if not is_timeout:
                        # Audio is live and should be at base_timestamp. Trust it.
                        pass
                    else:
                        # Timeout case: Audio is dead/stale.
                        # Trust our IDR-based timestamp.
                        logger.warning("[SYNC] Audio Hold Timeout. Releasing video without alignment.")

                # Normal playback continues here...
                frame = self.presentation_queue.popleft()

                # V12 CONTINUOUS SYNC: Always sync video to audio
                # Key insight: Never exit catchup mode. Always compare with audio.
                # This prevents drift accumulation that caused SSIF desync.
                audio_smooth = self._get_audio_clock()
                # V58: shift the audio reference back by the output-latency offset, so the
                # presenter shows the frame matching the HEARD audio (not MPV's fed position).
                if audio_smooth is not None and self._av_sync_offset_s:
                    audio_smooth -= self._av_sync_offset_s

                # V43 STARTUP GRACE: Skip V12 sync for the first 30 frames (~1.25s at 24fps)
                # This ensures initial visual feedback appears quickly without being
                # dropped or held by sync logic that may not be stable yet.
                v12_frames_emitted = getattr(self, '_v12_frames_emitted', 0)
                v12_active = (audio_smooth is not None and audio_smooth > 0.1
                              and v12_frames_emitted >= 30)

                if v12_active:
                    # Get frame timestamp
                    frame_ts = frame.get('timestamp', 0.0)
                    if frame_ts == 0.0:
                        frame_ts = self.base_timestamp + (self.frame_count * self.target_frame_time)

                    diff = frame_ts - audio_smooth

                    # V59 STALE-AUDIO FREE-RUN: after a seek, mpv's audio clock can
                    # stay frozen for a while (seek+restart latency). The old behavior
                    # re-armed a 2s hard hold on EVERY frame against that dead clock —
                    # a 0.5fps slideshow until audio revived (the post-seek stutter).
                    # Once a hold times out we stop trusting the clock: emit paced by
                    # wall time, and re-engage sync only when the clock has MOVED
                    # substantially (it is live again). Normal sync then reconciles
                    # (bulk-drop if video is behind, liquid/hold if ahead).
                    if getattr(self, '_v12_audio_stale_freerun', False):
                        # V61: judge liveness on the RAW clock — the V53 extrapolation
                        # inside audio_smooth creeps forward even when mpv is frozen,
                        # which faked the old "moved 200ms" test and re-engaged sync
                        # against a dead clock (0.6s hold-timeout churn after seeks).
                        self._audio_mutex.lock()
                        _audio_raw = self._external_audio_clock
                        self._audio_mutex.unlock()
                        audio0 = getattr(self, '_v12_freerun_audio0', None)
                        if audio0 is None:
                            # baseline = first raw sample after free-run engaged
                            self._v12_freerun_audio0 = audio0 = _audio_raw
                        if getattr(self, '_v12_freerun_from_seek', False):
                            # post-seek: live means the raw clock LANDED near the seek
                            # base (mpv finished its seek and audio is at position).
                            # 10s cap = pathological mpv (no audio track / dead seek).
                            _live = ((abs(_audio_raw - self.base_timestamp) < 1.0 and _audio_raw > 0.1)
                                     or (time.time() - getattr(self, '_v12_freerun_t0', 0)) > 10.0)
                        else:
                            # mid-play stall: live means the raw VALUE actually changed
                            _live = abs(_audio_raw - audio0) > 0.200
                        if _live:
                            self._v12_audio_stale_freerun = False
                            self._v12_freerun_from_seek = False
                            logger.info(f"[V59-SYNC] Audio clock live again (raw={_audio_raw:.3f}s, "
                                        f"base={self.base_timestamp:.3f}s) — resuming sync")
                            # V61 SNAP: if free-run marched the video ahead while audio
                            # restarted silently, don't burn the gap in 0.6s hold cycles
                            # (frozen picture) — jump the AUDIO to the video position via
                            # the existing atomic-sync path. Last resort: with the audio
                            # gate + fast seeks this should rarely exceed a few 100ms.
                            if diff > 0.6:
                                logger.warning(f"[V61-SYNC] Free-run left video {diff*1000:.0f}ms "
                                               f"ahead of audio — snapping audio to video position")
                                try:
                                    self.seekIDRFound.emit(frame_ts)
                                except Exception:
                                    pass
                                diff = 0.0
                        else:
                            self._v12_hold_start = None
                            diff = 0.0  # free-run: bypass Case 1/2, emit paced below

                    # V60 SYNC-METER: record the true A/V diff (skip free-run's fake 0)
                    if not getattr(self, '_v12_audio_stale_freerun', False):
                        self._sync_meter.append(diff * 1000.0)
                        self._sync_meter_count += 1
                        if self._sync_meter_count % 240 == 0 and len(self._sync_meter) >= 48:
                            _s = sorted(self._sync_meter)
                            _n = len(_s)
                            logger.info(f"[SYNC-METER] video-audio diff: median={_s[_n//2]:+.0f}ms "
                                        f"p5={_s[max(0, _n//20)]:+.0f}ms p95={_s[min(_n-1, _n*19//20)]:+.0f}ms "
                                        f"(n={_n}, av_offset={self._av_sync_offset_s*1000:+.0f}ms)")

                    if velvet_probe.ENABLED:
                        velvet_probe.record('v12diff_ms', diff * 1000.0)
                        velvet_probe.record('adjframe_ms', adjusted_frame_time * 1000.0)
                        velvet_probe.record('drift_ms', drift_ms)

                    # V12 DEBUG: Log drift periodically
                    if not hasattr(self, '_v12_drift_log_counter'):
                        self._v12_drift_log_counter = 0
                    self._v12_drift_log_counter += 1
                    if self._v12_drift_log_counter % 100 == 0:  # Every 100 frames (~4 seconds at 24fps)
                        logger.debug(f"[V12-SYNC] Drift: {diff*1000:.0f}ms (video={frame_ts:.3f}s, audio={audio_smooth:.3f}s)")

                    # Case 1: Video is LATE (Behind Audio) -> DROP FRAME
                    # V12b: Increased threshold to 120ms for smoother playback
                    # Only drop when seriously behind to avoid stuttering
                    if diff < -0.120:  # 120ms threshold
                        # V50 BULK-SKIP CATCH-UP: when a BACKLOG of already-decoded late
                        # frames is queued (typical right after a seek: the post-seek audio
                        # hold lets video fall behind, then per-frame dropping only nets
                        # decode_fps - realtime_fps ~6fps catch-up = 5-10s of stutter),
                        # discard ALL consecutive late frames from the queue in ONE pass.
                        # They are already decoded, so discarding is instant — this collapses
                        # the catch-up to a single jump to the audio position. For a transient
                        # single-frame lateness (no backlog) the while-loop finds the next
                        # frame in-sync and breaks immediately, so normal playback is unchanged.
                        # Only pops already-queued frames (same op as normal popleft) — does NOT
                        # touch frame_count, so it cannot trip the post-seek render race.
                        dropped = 1
                        while self.presentation_queue:
                            nxt_ts = self.presentation_queue[0].get('timestamp', 0.0)
                            if nxt_ts == 0.0 or (nxt_ts - audio_smooth) >= -0.120:
                                break
                            self.presentation_queue.popleft()
                            dropped += 1
                        if velvet_probe.ENABLED:
                            velvet_probe.on_bulkdrop(dropped)
                        if self._v12_drift_log_counter % 10 == 0:
                            logger.debug(f"[V12-SYNC] Video Late ({diff*1000:.1f}ms). Bulk-dropped {dropped} frame(s).")
                        # V55 SMOOTH CATCH-UP: the old code `return`ed here, emitting NOTHING
                        # this cycle — so every time the video was behind (which, on a stream
                        # the player can't quite sustain in real time, is most cycles) there
                        # was a visible emit GAP = the residual stutter. Instead, present the
                        # now-in-sync front frame THIS cycle: forward progress every cycle (no
                        # freeze), catching up by skipping stale CONTENT (one extra dropped
                        # frame ≈ invisible) rather than by stalling the picture.
                        if self.presentation_queue:
                            frame = self.presentation_queue.popleft()
                            # fall through to emit `frame` (it is now within sync tolerance)
                        else:
                            return  # buffer drained — nothing to show this cycle

                    # Case 2: Video is EARLY (Ahead of Audio) -> HOLD FRAME
                    # V43 FIX: Added 2-second timeout to prevent infinite video freeze.
                    # Without this timeout, if audio clock is stale or stuck, video would
                    # freeze permanently (the frame gets put back and retried forever).
                    elif diff > 0.150:  # 150ms threshold: video ahead of audio
                        if self._liquid_pacing and diff < 0.350:
                            # LIQUID PACING: gentle decel instead of a freeze. Probe-measured
                            # root cause of the dense-scene wobble: diff parks on this +150ms
                            # line and the binary hold (precise_wait+appendleft+return) blanks
                            # the cycle in bursts (=judder, ~90ms emit gaps). Instead, EMIT this
                            # frame but lengthen the NEXT interval ∝ the excess (capped +12ms),
                            # so audio eases back into sync with forward progress every cycle.
                            # Self-limiting: as diff falls under 150 the stretch returns to 0.
                            # The hard freeze below is kept for a genuine runaway (>350ms) only.
                            self._liquid_stretch = min((diff - 0.150) * 0.30, 0.012)
                            if velvet_probe.ENABLED:
                                velvet_probe.record('stretch_ms', self._liquid_stretch * 1000.0)
                            if hasattr(self, '_v12_hold_start'):
                                self._v12_hold_start = None
                            # fall through to emit `frame`
                        else:
                            # Hard hold (legacy freeze): SYLC_LIQUID=0, or runaway (>350ms) desync.
                            # Track hold start time for timeout
                            if not hasattr(self, '_v12_hold_start') or self._v12_hold_start is None:
                                self._v12_hold_start = time.time()
                            hold_duration = time.time() - self._v12_hold_start
                            # V59: 0.6s cap (was 2.0s) — long enough for a normal mpv
                            # audio restart, short enough not to feel like a freeze.
                            if hold_duration < 0.6:
                                if velvet_probe.ENABLED:
                                    velvet_probe.on_hold()
                                self._precise_wait(0.010)
                                self.presentation_queue.appendleft(frame)  # Put back to retry
                                return
                            else:
                                # V59: the clock did not catch up during the hold — it is
                                # stale/stuck. Enter free-run (wall-clock paced emission)
                                # instead of re-arming a fresh hold on every frame.
                                logger.warning(f"[V12-SYNC] Hold timeout ({hold_duration:.1f}s), "
                                             f"diff={diff*1000:.0f}ms — audio clock stale, "
                                             f"entering free-run until it moves")
                                self._v12_hold_start = None
                                self._v12_audio_stale_freerun = True
                                self._v12_freerun_from_seek = False
                                # V61: baseline must be the RAW clock (the moved-test
                                # now compares raw values; an extrapolated baseline
                                # would fake up to +1.5s of movement instantly).
                                self._audio_mutex.lock()
                                self._v12_freerun_audio0 = self._external_audio_clock
                                self._audio_mutex.unlock()
                    else:
                        # Frame is synced - reset hold timer
                        if hasattr(self, '_v12_hold_start'):
                            self._v12_hold_start = None
                        # V60 MICRO-SYNC: inside the old -120..+150ms dead zone NOTHING
                        # pulled diff toward 0, so a constant lip-sync error of up to
                        # ~±5 frames could park there forever. Steer pacing gently and
                        # SYMMETRICALLY toward diff=0: video early (+) -> stretch the
                        # next interval, video late (-) -> shrink it. Capped at ±8ms
                        # per frame (~±19% of 24fps pacing): a 100ms offset is absorbed
                        # in ~0.5s with no drop, no hold, no visible speed change.
                        # Below 25ms we leave it alone (mpv clock granularity floor).
                        if self._liquid_pacing and abs(diff) > 0.025:
                            self._liquid_stretch = max(-0.008, min(0.008, diff * 0.25))

                    # Case 3: SYNCED (within -120ms to +150ms) -> Emit normally
                    # V12b: Wider tolerance for smoother playback, continuous sync forever

                # Sync check counter
                if hasattr(self, '_sync_check_counter'):
                    self._sync_check_counter += 1
                else:
                    self._sync_check_counter = 0

                try:
                    self._emit_single_frame(frame)
                    # Velvet #9: phase-accurate schedule — advance by the target, not to
                    # current_time, so per-frame overhead (~2ms) doesn't accumulate into a slow
                    # drift (which made the video fall behind -> bulk-drops). Resync on a real
                    # stall (>1 frame behind) to avoid a catch-up burst.
                    # Velvet "liquid": _liquid_stretch (>=0, one-shot) gently lengthens THIS
                    # interval to ease a dense-scene video-ahead diff back into sync without
                    # ever blanking a cycle. It is 0 in baseline (SYLC_LIQUID=0).
                    self._last_display_time += adjusted_frame_time + self._liquid_stretch
                    self._liquid_stretch = 0.0
                    if current_time - self._last_display_time > adjusted_frame_time:
                        self._last_display_time = current_time
                    # V43: Track frames emitted for startup grace period
                    self._v12_frames_emitted = getattr(self, '_v12_frames_emitted', 0) + 1
                except Exception as e:
                    logger.error(f"[MVC-THREAD] Error emitting frame: {e}")
            else:
                sleep_time = adjusted_frame_time - time_since_last
                self._precise_wait(sleep_time)

        except OSError:
            # Catch Windows fatal exception 0xe24c4a02 (Access Violation in Threading)
            # This happens when accessing queue while it's being cleared by seek.
            pass
        except Exception as e:
            logger.error(f"[MVC-THREAD] Error in presentation queue: {e}")

    def _run_native_pipeline(self):
        """Zero-copy path: demux -> C++ ring buffer -> C++ decoder."""
        if not self._native_decoder or not self._native_ring:
            logger.warning("[MVC-THREAD] Native decoder pipeline unavailable.")
            return False

        try:
            if not self._native_decoder.init(self.num_threads):
                err_msg = ""
                try:
                    err_msg = self._native_decoder.get_last_error()
                except Exception:
                    err_msg = ""
                logger.error(f"[MVC-THREAD] Native decoder init failed: {err_msg}")
                return False
        except Exception as exc:
            logger.error(f"[MVC-THREAD] Native decoder init failed: {exc}")
            return False

        eos = False
        self.start_time = time.time()
        self.last_stats_time = time.time()

        # Sync variables
        video_clock = 0.0
        last_frame_time = 0.0

        while not self._stop_requested and not self._cleanup_in_progress:
            loop_start = time.time()

            # V14 GRACEFUL ENDING: Check cleanup flag at start of each loop
            if self._cleanup_in_progress:
                logger.info("[MVC-THREAD] V14: Cleanup in progress, exiting decode loop")
                break

            # --- 1. DECODING PHASE ---
            # Feed the ring buffer and decode if we have space in presentation queue
            # Don't decode too far ahead to save memory
            if len(self.presentation_queue) < 10:
                # Feed C++ ring buffer
                if not eos:
                    try:
                        # Try to read a few packets to keep ring full
                        for _ in range(2):
                            if not self.demuxer.read_next_into_ring(self._native_ring):
                                eos = True
                                logger.info("[MVC-THREAD] End of stream detected from demuxer")
                                break
                    except Exception as exc:
                        logger.error(f"[MVC-THREAD] Demuxer error (native path): {exc}")
                        eos = True

                # Pop from ring and decode
                # Process up to 2 frames per loop to catch up if needed, but usually 1
                for _ in range(2):
                    success, base_mv, dep_mv, ts_ms, is_keyframe, seq = self._native_ring.pop()
                    if not success:
                        if eos:
                            # Handle EOS flush
                            try:
                                self._native_decoder.flush()
                            except Exception as exc:
                                logger.error(f"[MVC-THREAD] Native flush failed: {exc}")

                            # Drain remaining frames
                            while True:
                                got, frame_obj = self._native_decoder.get_frame()
                                if not got: break
                                self._deliver_pybind_frame(frame_obj, (ts_ms or 0) / 1000.0)
                                self.frame_count += 1

                            # Move buffer to presentation
                            while self.frame_buffer:
                                self._repair_and_queue(self.frame_buffer.pop(0)['data'])

                            # Emit remaining
                            # V14 GRACEFUL ENDING: Also check cleanup flag to exit early
                            while self.presentation_queue and not self._stop_requested and not self._cleanup_in_progress:
                                self._emit_single_frame(self.presentation_queue.popleft())
                                try:
                                    time.sleep(self.target_frame_time)
                                except Exception:
                                    pass

                            self.decodingFinished.emit()
                            return True
                        break  # Ring empty, stop trying to decode this loop

                    # CRITICAL FIX: Use synthetic timestamp to avoid aliasing input timestamps to output frames
                    # The demuxer returns the timestamp of the NAL unit just read (Decode Order).
                    # The decoder outputs frames in Display Order.
                    # Due to H.264 reordering (B-frames), the input NAL timestamp != output frame timestamp.
                    # Passing the input timestamp causes "future" timestamps to be assigned to current frames,
                    # leading to false "video ahead" detection and permanent slowdown (drift).
                    # We use the robust synthetic calculation: base + count * duration.
                    synthetic_timestamp = self.base_timestamp + (self.frame_count * self.target_frame_time)

                    # Retrieve decoded frames
                    got_any = False
                    try:
                        while True:
                            got, frame_obj = self._native_decoder.get_frame()
                            # logger.debug(f"[MVC-THREAD] get_frame returned: {got}")
                            if not got: break
                            got_any = True
                            
                            # Use synthetic timestamp for sync
                            self._deliver_pybind_frame(frame_obj, synthetic_timestamp)
                            self.frame_count += 1
                            
                            # Increment timestamp for next frame in this batch (rare but possible)
                            synthetic_timestamp += self.target_frame_time
                            
                    except Exception as exc:
                        logger.error(f"[MVC-THREAD] Native get_frame exception: {exc}")

                    # FORCE TRANSFER: Ensure frames move from buffer to presentation queue immediately
                    # The reordering logic in _queue_frame_for_display buffers frames.
                    # We must ensure that if the buffer has enough frames, they are moved.
                    # _queue_frame_for_display already handles this if REORDER_DEPTH is reached.
                    # However, if we are stuck, we might need to force it.
                    # BUT WAIT, checking _queue_frame_for_display implementation:
                    # It only pops if len > REORDER_DEPTH.
                    # If REORDER_DEPTH is high (e.g. 4) and we have 3 frames, nothing happens.
                    # We should probably not change this behavior unless we want low latency.
                    # The issue might be REORDER_DEPTH is too high or frames aren't reaching it.
                    # Let's double check if _deliver_pybind_frame calls _queue_frame_for_display.
                    # Yes it does.
                    # So frames ARE going into frame_buffer.
                    # If frame_buffer grows indefinitely, it means they are not being popped.
                    # They are popped if len > REORDER_DEPTH.
                    # So if we are looping, we must have > REORDER_DEPTH frames.
                    # Unless REORDER_DEPTH is huge.
                    # Let's verify REORDER_DEPTH.

                    # If we got a frame, we can stop decoding for this iteration to check presentation
                    if got_any:
                        break

            # --- 2. PRESENTATION PHASE ---
            self._process_presentation_queue()

            # Stats update
            current_sys_time = time.time()
            if current_sys_time - self.last_stats_time > 1.0:
                fps = self.frame_count / (current_sys_time - self.start_time) if (
                                                                                             current_sys_time - self.start_time) > 0 else 0
                self.fps_update.emit(fps)
                self.last_stats_time = current_sys_time

        return True

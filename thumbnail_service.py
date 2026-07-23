# -*- coding: utf-8 -*-
"""In-process hover-thumbnail extraction (spec 2026-07-14).

Owns its OWN demuxer + single-threaded edge264 session — never the playback
ones. Disarmed by default: while disarmed it performs ZERO I/O (the Avatar
ISO lesson: a concurrent reader during demuxer init broke the mounted UDF
volume's reads). The player arms it only in steady playback and disarms it
around loads and seeks.
"""
import ctypes
import logging
import os
import threading
import time

import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from mvc_decoder import (edge264, Edge264Frame, find_nal_units, create_demuxer,
                         convert_avcc_to_annexb, _apply_bd_seek_tables,
                         edge264_session_lock)

logger = logging.getLogger(__name__)

# Pre-warm cv2's lazy internals (thread pool) at import time — imports happen on
# the GUI thread at service creation — so the first real resize never has to
# initialize them inside the decoder thread mid-playback.
try:
    cv2.resize(np.zeros((4, 4), np.uint8), (2, 2), interpolation=cv2.INTER_AREA)
except Exception:
    pass

THUMB_W, THUMB_H = 320, 180
READ_BUDGET_BYTES = 32 * 1024 * 1024   # abandon a request beyond this
OPTICAL_MIN_INTERVAL_S = 0.3           # extraction spacing on optical-class volumes
BLACKLIST_S = 10.0                     # failed keys are not retried for this long
EXTRACT_DEADLINE_S = 3.0
MAX_PAIR_SCAN = 600


def planes_to_qimage_320(y, u, v, layout=None, half=False):
    """Downscale numpy Y/U/V planes (I420 layout) to a 320x180 RGB QImage,
    PRESERVING the display aspect ratio (letterboxed with black bars inside the
    fixed box). Resize planes first (cheap), then one small I420->RGB conversion.
    Also used by the decoder-thread harvest tap (zero-I/O cache fill).

    The consumer (PreviewTooltip) blits the pixmap into a FIXED 320x180 rect, so
    a non-16:9 frame must be letterboxed HERE or it would be re-stretched on
    screen (e.g. a 2.40:1 scope master looked squished). We always return a
    320x180 image; off-aspect content is centred with black bars.

    layout: 'sbs' → keep the LEFT half (packed side-by-side sources),
            'tab' → keep the TOP half (top-and-bottom), else full frame.
    half:   half-packed source (HSBS/HTAB): the cropped eye is spatially
            squeezed, so its DISPLAY aspect is un-squeezed before letterboxing
            (×2 wide for sbs, ×2 tall for tab) — an HSBS eye 960x1080 shows 16:9.
    A thumbnail is a single eye — never the packed pair."""
    try:
        if layout == 'sbs':
            y = y[:, :y.shape[1] // 2]
            u = u[:, :u.shape[1] // 2]
            v = v[:, :v.shape[1] // 2]
        elif layout == 'tab':
            y = y[:y.shape[0] // 2]
            u = u[:u.shape[0] // 2]
            v = v[:v.shape[0] // 2]
        src_h, src_w = int(y.shape[0]), int(y.shape[1])
        if src_w <= 0 or src_h <= 0:
            return None
        # Display aspect ratio of the (cropped, un-squeezed) eye.
        dar = src_w / float(src_h)
        if half and layout == 'sbs':
            dar *= 2.0
        elif half and layout == 'tab':
            dar *= 0.5
        # Fit within the 320x180 box preserving DAR (letterbox / pillarbox).
        if dar >= (THUMB_W / float(THUMB_H)):
            fw, fh = THUMB_W, int(round(THUMB_W / dar))
        else:
            fw, fh = int(round(THUMB_H * dar)), THUMB_H
        fw = max(2, min(THUMB_W, fw - (fw % 2)))    # even width  (I420 chroma)
        fh = max(4, min(THUMB_H, fh - (fh % 4)))    # mult-of-4 height (I420 pack)
        y_s = cv2.resize(y, (fw, fh), interpolation=cv2.INTER_AREA)
        u_s = cv2.resize(u, (fw // 2, fh // 2), interpolation=cv2.INTER_AREA)
        v_s = cv2.resize(v, (fw // 2, fh // 2), interpolation=cv2.INTER_AREA)
        i420 = np.empty((fh * 3 // 2, fw), np.uint8)
        i420[:fh] = y_s
        i420[fh:fh + fh // 4] = u_s.reshape(fh // 4, fw)
        i420[fh + fh // 4:] = v_s.reshape(fh // 4, fw)
        rgb = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
        if fw == THUMB_W and fh == THUMB_H:
            canvas = np.ascontiguousarray(rgb)
        else:
            canvas = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)      # black bars
            ox, oy = (THUMB_W - fw) // 2, (THUMB_H - fh) // 2
            canvas[oy:oy + fh, ox:ox + fw] = rgb
        img = QImage(canvas.data, THUMB_W, THUMB_H, THUMB_W * 3, QImage.Format.Format_RGB888)
        return img.copy()   # detach from the numpy buffer (mandatory)
    except Exception:
        return None


def frame_to_qimage_320(frame, layout=None, half=False):
    """Downscale one Edge264Frame (base view) to a 320x180 RGB QImage
    (display-aspect preserved, letterboxed — see planes_to_qimage_320)."""
    w, h = frame.width_Y, frame.height_Y
    if w <= 0 or h <= 0 or frame.bit_depth_Y != 8 or not frame.samples[0]:
        return None
    cw, ch = frame.width_C, frame.height_C
    if cw <= 0 or ch <= 0:
        cw, ch = w // 2, h // 2
    sy, sc = frame.stride_Y, frame.stride_C
    if sy < w or sc < cw or not frame.samples[1] or not frame.samples[2]:
        return None
    y = np.ctypeslib.as_array(frame.samples[0], shape=(h, sy))[:, :w]
    u = np.ctypeslib.as_array(frame.samples[1], shape=(ch, sc))[:, :cw]
    v = np.ctypeslib.as_array(frame.samples[2], shape=(ch, sc))[:, :cw]
    return planes_to_qimage_320(y, u, v, layout, half)


class ThumbnailService(QThread):
    # (t_requested_s, idr_s, image): idr_s is the container PTS of the IDR the
    # thumbnail was decoded from — i.e. the frame a seek to t_requested lands
    # on. The slider stores it so a click can SNAP to it (perfect preview/seek
    # sync: landing frame == tooltip image, by construction).
    thumbnailReady = Signal(float, float, QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending_t = None          # newest-wins slot (seconds)
        self._filepath = None
        self._duration_s = 0.0
        self._mode = 'off'              # 'edge264' | 'avcodec' | 'off'
        self._optical = False
        self._layout = None             # 'sbs' | 'tab' | None
        self._half = False              # half-packed (HSBS/HTAB) → un-squeeze aspect
        self._armed = False
        self._stopping = False
        self._demuxer = None
        self._decoder = None            # ctypes.c_void_p edge264 session
        self._hevc_source = None        # LavfHevcSource (mode 'avcodec', HEVC) — worker-owned
        self._headers_fed = False
        self._blacklist = {}            # round(t) -> time.monotonic() deadline
        self._last_extract = 0.0
        self._release = False           # worker-side pipeline close request
        self._arm_logged = False        # one [THUMB] armed log per configure

    # ---- GUI-thread API -----------------------------------------------
    def configure(self, filepath, duration_s, mode, optical=False, layout=None, half=False):
        with self._lock:
            if filepath != self._filepath:
                self._close_pipeline_locked()
                self._blacklist.clear()
            self._filepath = filepath
            self._duration_s = float(duration_s or 0.0)
            self._mode = mode
            self._optical = bool(optical)
            self._layout = layout        # 'sbs' | 'tab' | None (single-eye crop)
            self._half = bool(half)      # half-packed source → un-squeeze aspect
            self._armed = False          # every (re)configure starts disarmed
            self._arm_logged = False
            self._pending_t = None

    def set_duration(self, duration_s):
        """Late duration push (mpv reports it asynchronously). Propagated to the
        demuxer on the next pipeline (re)open via _open_pipeline."""
        with self._lock:
            self._duration_s = float(duration_s or 0.0)
            d = self._demuxer
        if d is not None and duration_s and hasattr(d, 'set_external_duration_ms'):
            try:
                d.set_external_duration_ms(int(float(duration_s) * 1000))
            except Exception:
                pass

    def set_layout(self, layout, half=None):
        """Late single-eye crop update. The HEVC stereo layout ('sbs'/'tab') is
        only finalised by the SEI/side-data detector in _try_start_hevc, which
        runs AFTER configure() (which took the ffprobe analyzer's earlier, maybe
        absent, verdict). Refresh it so packed-3D HEVC thumbs show one eye — and
        the half/full verdict too, so a half-packed eye is un-squeezed."""
        with self._lock:
            self._layout = layout if layout in ('sbs', 'tab') else None
            if half is not None:
                self._half = bool(half)

    def arm(self):
        log_mode = None
        with self._lock:
            if self._filepath and self._mode in ('edge264', 'avcodec'):
                if not self._armed and not self._arm_logged:
                    self._arm_logged = True
                    log_mode = self._mode
                self._armed = True
        if log_mode:
            logger.info(f"[THUMB] armed (mode={log_mode})")

    def disarm(self):
        with self._lock:
            self._armed = False
            self._pending_t = None
            d = self._demuxer
        if d is not None and hasattr(d, 'request_abort'):
            try:
                d.request_abort()        # cut short any in-flight read
            except Exception:
                pass

    def request(self, t_seconds):
        with self._lock:
            if not self._armed or self._mode not in ('edge264', 'avcodec') or self._stopping:
                return
            key = round(t_seconds)
            if self._blacklist.get(key, 0) > time.monotonic():
                return
            self._pending_t = float(t_seconds)
        self._wake.set()

    def release_file(self):
        """Close the demuxer/decoder to free the file handles (e.g. so an ISO
        dismount can succeed after STOP). The close runs on the WORKER thread —
        it can never race an in-flight extraction. The service stays alive and
        reopens lazily on the next armed request."""
        self.disarm()
        with self._lock:
            self._release = True
        self._wake.set()

    def shutdown(self):
        with self._lock:
            self._stopping = True
            self._pending_t = None
        self.disarm()
        self._wake.set()
        self.wait(3000)
        with self._lock:
            self._close_pipeline_locked()

    # ---- worker thread --------------------------------------------------
    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            with self._lock:
                if self._release:
                    self._close_pipeline_locked()
                    self._release = False
            while True:
                with self._lock:
                    t = self._pending_t
                    self._pending_t = None
                    armed = self._armed and self._mode in ('edge264', 'avcodec')
                    optical = self._optical
                if t is None or self._stopping:
                    break
                if not armed:
                    continue
                if optical:
                    wait_s = OPTICAL_MIN_INTERVAL_S - (time.monotonic() - self._last_extract)
                    if wait_s > 0:
                        time.sleep(wait_s)
                res = None
                try:
                    res = self._extract(t)
                except Exception as e:
                    logger.warning(f"[THUMB] extraction failed at {t:.1f}s: {e}")
                    with self._lock:
                        self._close_pipeline_locked()   # self-heal for next request
                self._last_extract = time.monotonic()
                if res is not None:
                    img, idr_s = res
                    if os.environ.get("SYLC_THUMB_DIAG") == "1":
                        import sys as _sys
                        _sys.stderr.write(f"[THUMB-DIAG] extract t={t:.3f}s snap={idr_s if idr_s is None else round(idr_s, 3)}\n")
                    self.thumbnailReady.emit(
                        float(t), float(idr_s) if idr_s is not None else float(t), img)
                else:
                    with self._lock:
                        still_armed = self._armed
                    if still_armed:     # disarm-abort is not a failure: no blacklist
                        self._blacklist[round(t)] = time.monotonic() + BLACKLIST_S

    def _open_pipeline(self):
        if self._demuxer is not None and self._decoder is not None:
            return True
        with self._lock:
            filepath, duration_s = self._filepath, self._duration_s
        if not filepath:
            return False
        demuxer, eff_path = create_demuxer(filepath)
        if not hasattr(demuxer, 'read_next_frame_pair'):
            return False
        if not demuxer.open(eff_path):
            logger.info(f"[THUMB] demuxer open failed: {eff_path}")
            return False
        if duration_s > 0 and hasattr(demuxer, 'set_external_duration_ms'):
            demuxer.set_external_duration_ms(int(duration_s * 1000))
        try:
            _apply_bd_seek_tables(demuxer, eff_path)
        except Exception:
            pass
        # DOMAIN ANCHOR (SSIF/M2TS): these demuxers normalize timestamps to the
        # instance's FIRST-read PTS ("[SSIF] PTS normalization"). If our first
        # read happened at the first hover position, every reported timestamp
        # (and thus the click-snap target) would live in a shifted domain.
        # Seek to title 0 and read one pair so OUR offset = title start —
        # same 0-based domain as the playback timeline. Matroska needs none
        # (absolute container timestamps).
        if not hasattr(demuxer, 'getLastCueTimestamp'):
            try:
                demuxer.seek(0)
                demuxer.read_next_frame_pair()
            except Exception:
                pass
        with edge264_session_lock:
            ptr = edge264.edge264_alloc(0, None, None, 0, None, None, None)  # single-thread
        if not ptr:
            demuxer.close()
            return False
        with self._lock:
            self._demuxer = demuxer
            self._decoder = ctypes.c_void_p(ptr)
            self._headers_fed = False
        return True

    def _close_pipeline_locked(self):
        if self._decoder is not None:
            try:
                with edge264_session_lock:
                    edge264.edge264_free(ctypes.byref(self._decoder))
            except (OSError, RuntimeError):
                pass
            self._decoder = None
        if self._demuxer is not None:
            try:
                self._demuxer.close()
            except Exception:
                pass
            self._demuxer = None
        if self._hevc_source is not None:
            try:
                self._hevc_source.close()
            except Exception:
                pass
            self._hevc_source = None
        self._headers_fed = False

    def _open_hevc(self):
        """Open ONE software LavfHevcSource for the armed HEVC file, kept alive
        across extractions (like the edge264 session). SW only (allow_hw=False):
        hover thumbnails are sporadic single frames — a D3D11VA device per hover
        is pure overhead. Created + used + closed on THIS worker thread only."""
        if self._hevc_source is not None:
            return True
        with self._lock:
            filepath = self._filepath
        if not filepath:
            return False
        try:
            from lavf_hevc_source import LavfHevcSource
        except Exception as e:
            logger.info(f"[THUMB] lavf_hevc_source unavailable: {e}")
            return False
        src = LavfHevcSource()
        try:
            mi = src.open(filepath, allow_hw=False)
        except Exception as e:
            logger.info(f"[THUMB] HEVC open raised: {e}")
            try:
                src.close()
            except Exception:
                pass
            return False
        if mi is None:
            try:
                src.close()
            except Exception:
                pass
            logger.info(f"[THUMB] HEVC open refused (out of scope): {filepath}")
            return False
        with self._lock:
            self._hevc_source = src
        logger.info(f"[THUMB] avcodec pipeline open: {mi.width}x{mi.height} "
                    f"{mi.bit_depth}-bit {mi.pix_fmt_name}")
        return True

    def _extract_avcodec(self, t_seconds):
        """HEVC hover extraction via avcodec (SW): seek -> one read_frame ->
        (Y,U,V) numpy -> 320x180 RGB QImage. 10-bit planes are LSB-aligned
        (yuv420p10le, allow_hw=False) so a >>2 downshift yields 8-bit. Bounded
        (single seek + single decoded frame, <100ms @ 4K10 SW): its boundedness
        stands in for the abort hook LavfHevcSource lacks."""
        if not self._open_hevc():
            return None
        src = self._hevc_source
        if not src.seek(int(max(0.0, t_seconds) * 1000)):
            return None
        with self._lock:
            if not self._armed or self._stopping:
                return None
        res = src.read_frame()          # None => EOF (t past end): no crash, no image
        if res is None:
            return None
        (y, u, v), pts_ms = res
        if y.dtype != np.uint8:         # 10-bit yuv420p10le (LSB-aligned 0..1023)
            y = (y >> 2).astype(np.uint8)
            u = (u >> 2).astype(np.uint8)
            v = (v >> 2).astype(np.uint8)
        with self._lock:
            layout, half = self._layout, self._half
        img = planes_to_qimage_320(y, u, v, layout, half)   # same crop + I420->RGB as edge264
        if img is None:
            return None
        # Snap target: the frame's own container PTS (post-BACKWARD-seek this is
        # the keyframe a click-seek would also land on). +10ms rounding safety so
        # the seek resolves at-or-after this entry (mirrors the edge264 path).
        snap_s = (float(pts_ms) / 1000.0) if pts_ms is not None and pts_ms >= 0 else float(t_seconds)
        snap_s += 0.010
        return (img, snap_s)

    def _extract(self, t_seconds):
        with self._lock:
            mode = self._mode
        if mode == 'avcodec':
            return self._extract_avcodec(t_seconds)
        if not self._open_pipeline():
            return None
        d = self._demuxer
        if hasattr(d, 'clear_abort'):
            d.clear_abort()
        d.seek(int(max(0.0, t_seconds) * 1000))
        try:
            with edge264_session_lock:
                edge264.edge264_flush(self._decoder)
        except (OSError, RuntimeError):
            pass
        self._headers_fed = False
        spent = 0
        deadline = time.monotonic() + EXTRACT_DEADLINE_S
        au = None
        idr_s = None
        warmup = 0
        for _ in range(MAX_PAIR_SCAN):
            with self._lock:
                if not self._armed or self._stopping:
                    return None
            ok, base, _dep = d.read_next_frame_pair()
            if not ok:
                return None
            data = bytes(base['data'])
            spent += len(data)
            sync = self._classify_sync(data)
            if base.get('isKeyframe') or sync == 'idr':
                au = data
                warmup = 0
            elif sync == 'recovery':
                # Open-GOP recovery point (BD SSIF): accept it like the playback
                # seek does, and warm the decoder with following AUs so the
                # picture is fully refreshed (mirrors the extended priming).
                au = data
                warmup = 10
            if au is not None:
                ts_ms = base.get('timestamp')
                idr_s = (float(ts_ms) / 1000.0) if ts_ms is not None else None
                if os.environ.get("SYLC_THUMB_DIAG") == "1":
                    import sys as _sys
                    _sys.stderr.write(f"[THUMB-DIAG] sync accept: {'idr' if warmup == 0 else 'recovery'} at ts={idr_s}\n")
                break
            if spent > READ_BUDGET_BYTES or time.monotonic() > deadline:
                return None
        if au is None:
            return None

        # SNAP TARGET: the time a click must seek to in order to land on THIS
        # frame. For Matroska, the Cues timestamp is authoritative and can sit
        # slightly AFTER the block PTS (cluster ts) — seeking to the raw block
        # PTS would resolve to the PREVIOUS cue (one GOP early, measured on
        # Top Gun: cue=336.330 vs block PTS=336.294). Use the cue when exposed.
        snap_s = idr_s
        try:
            if hasattr(d, 'getLastCueTimestamp'):
                cue_ms = d.getLastCueTimestamp()
                if cue_ms is not None and cue_ms >= 0:
                    cue_s = float(cue_ms) / 1000.0
                    if idr_s is None or abs(cue_s - idr_s) < 5.0:
                        snap_s = cue_s
        except Exception:
            pass
        if snap_s is not None:
            snap_s += 0.010     # ms-rounding safety: stay at-or-after the entry

        def _next_au():
            ok2, base2, _dep2 = d.read_next_frame_pair()
            return bytes(base2['data']) if ok2 else None

        img = self._decode_one(au, _next_au, warmup=warmup)
        return (img, snap_s) if img is not None else None

    def _decode_one(self, au, next_au_fn=None, warmup=0):
        if not self._headers_fed:
            cp = b''
            if hasattr(self._demuxer, 'get_codec_private'):
                try:
                    cp = bytes(self._demuxer.get_codec_private() or b'')
                except Exception:
                    cp = b''
            headers = convert_avcc_to_annexb(cp)
            if headers:
                self._feed(headers)
            self._headers_fed = True
        self._feed(au)
        # Recovery-point sync (warmup>0): feed following AUs so the picture is
        # fully refreshed before we grab it — mirrors the playback seek's
        # extended priming, so vignette and landing show the same instant.
        for _ in range(warmup):
            with self._lock:
                if not self._armed or self._stopping:
                    return None
            nxt = next_au_fn() if next_au_fn else None
            if nxt is None:
                break
            self._feed(nxt)
        frame = Edge264Frame()
        best = None
        # The frame may not pop until the DPB is bumped or a following AU
        # arrives (reorder delay) — feed up to 4 extra AUs before giving up.
        # Drain everything available and keep the LAST (most recovered) frame.
        for attempt in range(5):
            # FAST-ABORT: a starting seek disarms us and will alloc/free the
            # playback session — leave the DLL immediately.
            with self._lock:
                if not self._armed or self._stopping:
                    return best
            with edge264_session_lock:
                edge264.edge264_bump_frames(self._decoder)
                while True:
                    ret = edge264.edge264_get_frame(self._decoder, ctypes.byref(frame), 1)  # borrow
                    if ret != 0 or not frame.samples[0]:
                        break
                    img = frame_to_qimage_320(frame, self._layout, self._half)
                    if img is not None:
                        best = img
                    if frame.return_arg:
                        try:
                            edge264.edge264_return_frame(self._decoder, frame.return_arg)
                        except (OSError, RuntimeError):
                            pass
            if best is not None:
                return best
            nxt = next_au_fn() if next_au_fn else None
            if nxt is None:
                return None
            self._feed(nxt)
        return best

    def _feed(self, data):
        # Mirror the playback contract exactly (mvc_decoder._decode_single_nal):
        # bare NAL bytes (start code STRIPPED), 2-byte 0xFF prefix pad (V45:
        # edge264 reads 2 bytes BEFORE the pointer for emulation-prevention
        # detection) and 64-byte tail pad.
        for nal in find_nal_units(data):
            off = 4 if nal[:4] == b'\x00\x00\x00\x01' else 3
            content = nal[off:]
            n = len(content)
            if n == 0:
                continue
            ntype = content[0] & 0x1F
            if ntype > 20 and ntype != 31:
                continue
            buf = ctypes.create_string_buffer(2 + n + 64)
            buf[0] = b'\xFF'
            buf[1] = b'\xFF'
            ctypes.memmove(ctypes.addressof(buf) + 2, bytes(content), n)
            start = ctypes.cast(ctypes.addressof(buf) + 2, ctypes.POINTER(ctypes.c_uint8))
            end = ctypes.cast(ctypes.addressof(buf) + 2 + n, ctypes.POINTER(ctypes.c_uint8))
            try:
                with edge264_session_lock:
                    edge264.edge264_decode_NAL(self._decoder, start, end, 0, None, None, None)
            except (OSError, RuntimeError):
                return

    @staticmethod
    def _classify_sync(data):
        """Classify an AU as a stream sync point: 'idr' (NAL 5), 'recovery'
        (in-band SPS+PPS without IDR — the open-GOP recovery points Blu-ray
        SSIF uses, same lenient rule as the playback seek scan), or None."""
        has_sps = has_pps = has_idr = False
        for nal in find_nal_units(data):
            off = 4 if nal[:4] == b'\x00\x00\x00\x01' else 3
            if off >= len(nal):
                continue
            ntype = nal[off] & 0x1F
            if ntype == 5:
                has_idr = True
            elif ntype == 7:
                has_sps = True
            elif ntype == 8:
                has_pps = True
        if has_idr:
            return 'idr'
        if has_sps and has_pps:
            return 'recovery'
        return None

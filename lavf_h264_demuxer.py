#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""libavformat-backed H.264 demuxer (task #391).

Lets edge264 handle H.264 in ANY container ffmpeg can open (.mp4/.avi/.ts/.flv/
raw .h264/.264, …) — not just the MKV/M2TS/SSIF the native C++ demuxer covers.
It is a thin ctypes wrapper over the ffmpeg 8.0 DLLs already shipped beside the
app (avformat-62/avcodec-62/avutil-60) — NO .pyd rebuild, NO extra binary.

It exposes the SAME interface MVCDecoderThread expects from a demuxer:
    open(path) -> bool
    get_codec_private() -> bytes        (the avcC; the decoder converts to Annex-B)
    read_next_frame_pair() -> (ok, base, dep)   base={'data','timestamp','isKeyframe'}
    seek(timestamp_ms) -> bool
    close()
read_next_frame_pair returns Annex-B NALs in base['data'] (the pipeline runs
find_nal_units on it), single-view (dep=None) — the decoder duplicates the left
view, exactly like a 2D MKV.

NOTE on struct offsets: ffmpeg keeps AVPacket / the first AVCodecParameters fields
/ the top AVFormatContext fields ABI-stable; only AVStream.codecpar moves between
majors. AVStream.codecpar is at offset 16 in libavformat 62 (ffmpeg 8.0), which we
VERIFY at open() (codec_id must read back as H.264) and refuse the file otherwise,
so a future ffmpeg ABI change degrades to the mpv fallback instead of crashing.
"""
import os
import ctypes
import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- ffmpeg constants ---
_AVMEDIA_TYPE_VIDEO = 0
_AV_CODEC_ID_H264 = 27
_AVSEEK_FLAG_BACKWARD = 1
_AV_PKT_FLAG_KEY = 0x0001
_AV_NOPTS = -9223372036854775808  # AV_NOPTS_VALUE (INT64_MIN)

# AVStream field offsets for libavformat 62 (ffmpeg 8.0), x64. Verified at runtime.
_OFF_CODECPAR = 16
_OFF_TIME_BASE = 32


class AVRational(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int), ("den", ctypes.c_int)]


class AVFormatContext(ctypes.Structure):       # partial — these top fields are ABI-stable
    _fields_ = [
        ("av_class", ctypes.c_void_p), ("iformat", ctypes.c_void_p),
        ("oformat", ctypes.c_void_p), ("priv_data", ctypes.c_void_p),
        ("pb", ctypes.c_void_p), ("ctx_flags", ctypes.c_int),
        ("nb_streams", ctypes.c_uint), ("streams", ctypes.POINTER(ctypes.c_void_p)),
    ]


class AVCodecParameters(ctypes.Structure):     # partial — first fields are ABI-stable
    _fields_ = [
        ("codec_type", ctypes.c_int), ("codec_id", ctypes.c_int),
        ("codec_tag", ctypes.c_uint32),
        ("extradata", ctypes.POINTER(ctypes.c_uint8)), ("extradata_size", ctypes.c_int),
    ]


class AVPacket(ctypes.Structure):              # ABI-stable since ffmpeg 5.0
    _fields_ = [
        ("buf", ctypes.c_void_p), ("pts", ctypes.c_int64), ("dts", ctypes.c_int64),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_int), ("stream_index", ctypes.c_int), ("flags", ctypes.c_int),
        ("side_data", ctypes.c_void_p), ("side_data_elems", ctypes.c_int),
        ("duration", ctypes.c_int64), ("pos", ctypes.c_int64),
        ("opaque", ctypes.c_void_p), ("opaque_ref", ctypes.c_void_p),
        ("time_base", AVRational),
    ]


_AVFORMAT = None
_AVCODEC = None
_AVUTIL = None


def _load():
    """Load + sign the bundled ffmpeg DLLs once. Raises on failure."""
    global _AVFORMAT, _AVCODEC, _AVUTIL
    if _AVFORMAT is not None:
        return
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        os.add_dll_directory(d)
    except (OSError, AttributeError):
        pass
    avutil = ctypes.CDLL(os.path.join(d, 'avutil-60.dll'))
    ctypes.CDLL(os.path.join(d, 'swresample-6.dll'))   # avcodec dependency
    avcodec = ctypes.CDLL(os.path.join(d, 'avcodec-62.dll'))
    avformat = ctypes.CDLL(os.path.join(d, 'avformat-62.dll'))

    avformat.avformat_open_input.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
    avformat.avformat_open_input.restype = ctypes.c_int
    avformat.avformat_find_stream_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    avformat.avformat_find_stream_info.restype = ctypes.c_int
    avformat.av_find_best_stream.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    avformat.av_find_best_stream.restype = ctypes.c_int
    avformat.av_read_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    avformat.av_read_frame.restype = ctypes.c_int
    avformat.av_seek_frame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int]
    avformat.av_seek_frame.restype = ctypes.c_int
    avformat.avformat_close_input.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    avcodec.av_packet_alloc.restype = ctypes.c_void_p
    avcodec.av_packet_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    avcodec.av_packet_unref.argtypes = [ctypes.c_void_p]

    _AVUTIL, _AVCODEC, _AVFORMAT = avutil, avcodec, avformat


def is_available():
    try:
        _load()
        return True
    except Exception as e:
        logger.warning(f"[LAVF] ffmpeg DLLs unavailable: {e}")
        return False


# Containers ffmpeg should demux for us that the native C++ demuxer does NOT cover.
SUPPORTED_EXTS = ('.mp4', '.m4v', '.mov', '.avi', '.flv', '.wmv', '.webm',
                  '.ts', '.mpg', '.mpeg', '.mpv', '.h264', '.264', '.avc')


class LavfH264Demuxer:
    def __init__(self):
        self._ctx = ctypes.c_void_p()
        self._pkt = None
        self._pkt_view = None
        self._vidx = -1
        self._avcc = b''
        self._nls = 4
        self._tb_num, self._tb_den = 1, 1000
        self._opened = False

    # ---- interface expected by MVCDecoderThread ----
    def open(self, path):
        try:
            _load()
        except Exception as e:
            logger.error(f"[LAVF] load failed: {e}")
            return False
        if self._opened:
            return True
        if _AVFORMAT.avformat_open_input(ctypes.byref(self._ctx),
                                         str(path).encode('utf-8'), None, None) < 0:
            logger.warning(f"[LAVF] avformat_open_input failed: {path}")
            return False
        if _AVFORMAT.avformat_find_stream_info(self._ctx, None) < 0:
            logger.warning("[LAVF] find_stream_info failed")
            self.close()
            return False
        self._vidx = _AVFORMAT.av_find_best_stream(self._ctx, _AVMEDIA_TYPE_VIDEO, -1, -1, None, 0)
        if self._vidx < 0:
            logger.warning("[LAVF] no video stream")
            self.close()
            return False
        fmt = ctypes.cast(self._ctx, ctypes.POINTER(AVFormatContext))
        if self._vidx >= fmt.contents.nb_streams:
            self.close()
            return False
        sptr = fmt.contents.streams[self._vidx]
        # AVStream.codecpar @16 — VERIFY (codec_id must be H.264) or refuse → mpv fallback.
        cp_addr = ctypes.c_void_p.from_address(sptr + _OFF_CODECPAR).value
        if not cp_addr:
            logger.warning("[LAVF] codecpar NULL")
            self.close()
            return False
        cp = AVCodecParameters.from_address(cp_addr)
        if cp.codec_type != _AVMEDIA_TYPE_VIDEO or cp.codec_id != _AV_CODEC_ID_H264:
            logger.info(f"[LAVF] not H.264 (type={cp.codec_type} id={cp.codec_id}) → mpv fallback")
            self.close()
            return False
        esz = cp.extradata_size
        if 0 < esz < 1_000_000 and cp.extradata:
            self._avcc = bytes((ctypes.c_uint8 * esz).from_address(
                ctypes.cast(cp.extradata, ctypes.c_void_p).value))
        else:
            self._avcc = b''
        # NAL length size from the avcC (byte 4, low 2 bits) + 1; default 4.
        if len(self._avcc) > 4 and self._avcc[0] == 1:
            self._nls = (self._avcc[4] & 0x03) + 1
        # Stream time_base @32 — validate; fall back to {1,1000} (treat pts as ms).
        tb = AVRational.from_address(sptr + _OFF_TIME_BASE)
        if 0 < tb.den <= 1_000_000_000 and 0 < tb.num <= 1_000_000_000:
            self._tb_num, self._tb_den = tb.num, tb.den
        else:
            logger.warning(f"[LAVF] implausible time_base {tb.num}/{tb.den}; using 1/1000")
            self._tb_num, self._tb_den = 1, 1000
        self._pkt = _AVCODEC.av_packet_alloc()
        self._pkt_view = ctypes.cast(self._pkt, ctypes.POINTER(AVPacket))
        self._opened = True
        logger.info(f"[LAVF] opened {os.path.basename(str(path))}: vidx={self._vidx} "
                    f"nls={self._nls} tb={self._tb_num}/{self._tb_den} avcC={len(self._avcc)}B")
        return True

    def get_codec_private(self):
        return self._avcc

    def _avcc_to_annexb(self, data):
        """Replace each NAL's length prefix with a 4-byte Annex-B start code."""
        nls = self._nls
        out = bytearray()
        i, n = 0, len(data)
        while i + nls <= n:
            ln = int.from_bytes(data[i:i + nls], 'big')
            i += nls
            if ln <= 0 or i + ln > n:
                break
            out += b'\x00\x00\x00\x01'
            out += data[i:i + ln]
            i += ln
        return bytes(out)

    def read_next_frame_pair(self):
        if not self._opened:
            return (False, None, None)
        while True:
            if _AVFORMAT.av_read_frame(self._ctx, self._pkt) < 0:
                return (False, None, None)   # EOF / error
            pk = self._pkt_view.contents
            if pk.stream_index != self._vidx or pk.size <= 0:
                _AVCODEC.av_packet_unref(self._pkt)
                continue
            raw = bytes((ctypes.c_uint8 * pk.size).from_address(
                ctypes.cast(pk.data, ctypes.c_void_p).value))
            pts = pk.pts if pk.pts != _AV_NOPTS else pk.dts
            ts_ms = int(pts * 1000 * self._tb_num / self._tb_den) if pts != _AV_NOPTS else 0
            key = bool(pk.flags & _AV_PKT_FLAG_KEY)
            _AVCODEC.av_packet_unref(self._pkt)
            annexb = self._avcc_to_annexb(raw)
            base = {'data': np.frombuffer(annexb, dtype=np.uint8),
                    'timestamp': ts_ms, 'isKeyframe': key}
            return (True, base, None)

    def seek(self, timestamp_ms):
        if not self._opened:
            return False
        # stream_index = -1 → timestamp is in AV_TIME_BASE (microseconds); BACKWARD
        # lands on/at-or-before the nearest keyframe so the decoder can prime.
        ts = int(max(0.0, float(timestamp_ms)) * 1000.0)
        return _AVFORMAT.av_seek_frame(self._ctx, -1, ts, _AVSEEK_FLAG_BACKWARD) >= 0

    def close(self):
        try:
            if self._pkt:
                p = ctypes.c_void_p(self._pkt)
                _AVCODEC.av_packet_free(ctypes.byref(p))
                self._pkt = None
        except Exception:
            pass
        try:
            if self._ctx:
                _AVFORMAT.avformat_close_input(ctypes.byref(self._ctx))
        except Exception:
            pass
        self._opened = False

    # ---- optional methods the pipeline probes via hasattr ----
    def getCuesTimestamps(self):
        return []          # no cue index → the decoder uses seek()

    def get_subtitle_tracks(self):
        return []          # subtitle streaming not provided by this demuxer

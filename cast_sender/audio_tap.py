#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SyLC Cast -- AudioTap: independent avcodec + swresample audio decode (Task 8).

In MVC/HEVC mode mpv plays the movie's audio straight to the sound card and there
is NO PCM tap. To stream audio to the Quest we therefore run a SEPARATE, independent
decode of the SAME media file here: demux (avformat) -> decode the best audio stream
(avcodec) -> resample to s16le INTERLEAVED PCM (swresample) -> emit (pts_ms, pcm,
sample_rate, channels) via on_pcm, PACED to stay ~200ms ahead of a playback clock.
Task 9 wires this into the transport + receiver A/V sync (the Quest's AudioSink becomes
the master clock); this module is standalone and pure-Python (ctypes over the bundled
ffmpeg 8.0 DLLs -- CPU only, no GPU).

Design choices (documented in the Task 8 report):
  * Output = AV_SAMPLE_FMT_S16 interleaved (s16le on x64), DOWNMIXED to STEREO (2 ch)
    at Android's native 48 kHz rate. Stereo/48 kHz gives the transport and Quest
    AudioTrack one fixed, predictable format without device-side resampling.
  * The sample format is resolved by name via av_get_sample_fmt (never a hardcoded enum).
  * sample_rate / channel count are read from AVCodecParameters by PINNED OFFSET (major-62
    only) and CROSS-CHECKED at runtime (plausible ranges + NATIVE-order popcount(mask)==
    channels); a failed cross-check refuses the audio cleanly rather than emitting garbage.

Reuses lavf_h264_demuxer._load() (loads avutil-60 / swresample-6 / avcodec-62 /
avformat-62 + signs avformat + av_packet_*) and its ABI-stable structs; the audio
decoder + swresample symbols are bound here following the same _sign() convention.
"""
import os
import ctypes
import logging
import threading
import time

import lavf_h264_demuxer as _lavf          # _load() + DLL handles + ABI structs (DRY)
from lavf_h264_demuxer import AVFormatContext, AVCodecParameters, AVRational, AVPacket

logger = logging.getLogger(__name__)

# --- ffmpeg constants ---
_AVMEDIA_TYPE_AUDIO = 1
_AVSEEK_FLAG_BACKWARD = 1
_AV_NOPTS = -9223372036854775808            # AV_NOPTS_VALUE (INT64_MIN)
_AVERROR_EAGAIN = -11                        # AVERROR(EAGAIN)
_AVERROR_EOF = -541478725                    # FFERRTAG('E','O','F',' ')

# Custom-AVIO read callback: int (*read_packet)(void *opaque, uint8_t *buf, int size).
_AVIO_READ_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_uint8), ctypes.c_int)
# AVIOContext head (avformat 62, x64): { const AVClass *av_class @0;
# unsigned char *buffer @8; ... } — ffmpeg may av_realloc the buffer after
# avio_alloc_context, so teardown must free AVIOContext.buffer (read at this
# offset), NOT the pointer originally passed in.
_OFF_AVIO_BUFFER = 8
_AVIO_BUF_SIZE = 64 * 1024

# AVCodecParameters audio-field offsets for LIBAVCODEC major 62 (ffmpeg 8.0), x64.
# PROBED empirically against the bundled avcodec-62.dll (a known 48000/stereo AAC read
# back ch_layout.order=1 @128, nb_channels=2 @132, mask=0x3 @136, sample_rate=48000 @152)
# and CROSS-CHECKED at runtime in open(); the color offsets (@100..112) that Task 9 probed
# with a compiled program corroborate this same layout. Guarded by avcodec_version()>>16==62.
_OFF_CP_CH_ORDER = 128
_OFF_CP_CH_NB_CHANNELS = 132
_OFF_CP_CH_MASK = 136
_OFF_CP_SAMPLE_RATE = 152

# AVFrame head-field offsets (avutil 60), x64 -- the SAME fields the HEVC path validates
# at runtime (lavf_hevc_source.AVFrame): confirmed here by the same probe (nb_samples=1024
# @112, format=8/fltp @116, pts=0 @136, extended_data non-NULL @96).
_OFF_FR_EXTENDED_DATA = 96
_OFF_FR_NB_SAMPLES = 112
_OFF_FR_FORMAT = 116
_OFF_FR_PTS = 136


class AVChannelLayout(ctypes.Structure):
    """avutil 60 AVChannelLayout: { AVChannelOrder order; int nb_channels;
    union { uint64_t mask; AVChannelCustom *map; } u; void *opaque; } -- 24 bytes on x64
    (confirmed: av_channel_layout_default(2) -> order=1 nb_channels=2 mask=0x3, sizeof=24)."""
    _fields_ = [("order", ctypes.c_int), ("nb_channels", ctypes.c_int),
                ("mask", ctypes.c_uint64), ("opaque", ctypes.c_void_p)]


_ASIGNED = False
_SWR = None                                  # libswresample-6 handle (bound here)


def _sign_audio():
    """Load + sign the audio-decode / swresample symbols once (idempotent). Reuses the
    avformat + av_packet_* signatures already installed by _lavf._load(); binds only the
    additional avcodec / avutil / swresample symbols this module needs."""
    global _ASIGNED, _SWR
    _lavf._load()                            # avformat_open_input/find_stream_info/av_find_best_stream/
                                             # av_read_frame/av_seek_frame/close_input + av_packet_* signed here
    if _ASIGNED:
        return
    ac, au = _lavf._AVCODEC, _lavf._AVUTIL
    d = os.path.dirname(os.path.abspath(_lavf.__file__))
    swr = ctypes.CDLL(os.path.join(d, 'swresample-6.dll'))   # already loaded by _load(); handle for swr_*

    ac.avcodec_version.restype = ctypes.c_uint
    ac.avcodec_find_decoder.argtypes = [ctypes.c_int]
    ac.avcodec_find_decoder.restype = ctypes.c_void_p
    ac.avcodec_alloc_context3.argtypes = [ctypes.c_void_p]
    ac.avcodec_alloc_context3.restype = ctypes.c_void_p
    ac.avcodec_free_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    ac.avcodec_parameters_to_context.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_parameters_to_context.restype = ctypes.c_int
    ac.avcodec_open2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_open2.restype = ctypes.c_int
    ac.avcodec_send_packet.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_send_packet.restype = ctypes.c_int
    ac.avcodec_receive_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_receive_frame.restype = ctypes.c_int
    ac.avcodec_flush_buffers.argtypes = [ctypes.c_void_p]
    ac.avcodec_get_name.argtypes = [ctypes.c_int]
    ac.avcodec_get_name.restype = ctypes.c_char_p

    # Custom-AVIO (stream-source mode): demux from bytes a caller feeds us —
    # e.g. the SSIF demuxer's stream tap — instead of opening the file again.
    af = _lavf._AVFORMAT
    af.avformat_alloc_context.restype = ctypes.c_void_p
    af.avio_alloc_context.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, _AVIO_READ_CB,
                                      ctypes.c_void_p, ctypes.c_void_p]
    af.avio_alloc_context.restype = ctypes.c_void_p
    af.avio_context_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    au.av_malloc.argtypes = [ctypes.c_size_t]
    au.av_malloc.restype = ctypes.c_void_p
    au.av_free.argtypes = [ctypes.c_void_p]
    au.av_dict_set.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                               ctypes.c_char_p, ctypes.c_int]
    au.av_dict_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    au.av_frame_alloc.restype = ctypes.c_void_p
    au.av_frame_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    au.av_get_sample_fmt.argtypes = [ctypes.c_char_p]
    au.av_get_sample_fmt.restype = ctypes.c_int
    au.av_channel_layout_default.argtypes = [ctypes.POINTER(AVChannelLayout), ctypes.c_int]
    # NB: av_frame_unref (avcodec_receive_frame unrefs the frame internally) and
    # av_channel_layout_uninit (the local layouts in _setup_swr own no heap) are deliberately
    # NOT bound -- there is no caller for either.

    _PU8 = ctypes.POINTER(ctypes.c_uint8)
    _PPU8 = ctypes.POINTER(_PU8)
    swr.swr_alloc_set_opts2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),                       # SwrContext **ps
        ctypes.POINTER(AVChannelLayout), ctypes.c_int, ctypes.c_int,   # out layout, fmt, rate
        ctypes.POINTER(AVChannelLayout), ctypes.c_int, ctypes.c_int,   # in  layout, fmt, rate
        ctypes.c_int, ctypes.c_void_p]                         # log_offset, log_ctx
    swr.swr_alloc_set_opts2.restype = ctypes.c_int
    swr.swr_init.argtypes = [ctypes.c_void_p]
    swr.swr_init.restype = ctypes.c_int
    swr.swr_convert.argtypes = [ctypes.c_void_p, _PPU8, ctypes.c_int, _PPU8, ctypes.c_int]
    swr.swr_convert.restype = ctypes.c_int
    swr.swr_get_delay.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    swr.swr_get_delay.restype = ctypes.c_int64
    swr.swr_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    _SWR = swr
    _ASIGNED = True


def is_available():
    """True iff the ffmpeg DLLs load + sign (avcodec/avutil/avformat/swresample)."""
    try:
        _sign_audio()
        return True
    except Exception as e:
        logger.warning("[ATAP] ffmpeg DLLs unavailable: %s", e)
        return False


class AudioTap:
    """Independent, paced audio decode of a media file's best audio stream.

        tap = AudioTap(path, on_pcm)              # on_pcm(pts_ms:int, pcm:bytes, sr:int, ch:int)
        tap.start(clock_ms)                       # clock_ms: callable()->int|None (playback ms)
        tap.seek(pts_ms); tap.stop()              # stop() is idempotent + frees every context

    A single worker thread demuxes + decodes + resamples and calls on_pcm, sleeping in
    small slices while it is more than LOOKAHEAD_MS ahead of clock_ms() (free-running at
    real time from decode start when clock_ms() returns None). All libav* calls -- decode,
    seek, flush, free -- happen on that one thread (they are not re-entrant); seek() and
    stop() only signal it. Output is s16le interleaved stereo at 48 kHz.
    """

    LOOKAHEAD_MS = 200                       # emit while decoded pts <= clock + this
    _PACE_SLICE_S = 0.005                    # re-check the clock this often while far ahead
    _JOIN_TIMEOUT_S = 5.0
    _OUT_CHANNELS = 2                        # downmix to stereo
    # Stream mode: decoded-PCM buffer depth (seconds). Ingestion must follow
    # the SOURCE's rhythm, not the emission clock: the SSIF demuxer reads ~6s
    # ahead of playback in bursts, so a tap that only consumed at realtime let
    # the tee overflow every ~3s -> mpegts discontinuity -> h264 parser error
    # spam + free/copy bursts = periodic hitches. 30s of decoded stereo PCM is
    # ~5.8 MB and comfortably swallows any read-ahead the demuxer can build.
    STREAM_BUFFER_S = 30.0

    def __init__(self, path, on_pcm):
        # `path` is either a filesystem path (str/PathLike, the classic mode)
        # or a STREAM SOURCE object exposing read(n)->bytes (b'' = no data yet,
        # never blocking) and optionally close(). Stream mode demuxes from
        # those bytes over a custom AVIO — used for optical discs, where the
        # SSIF demuxer tees the bytes it already reads so this tap never opens
        # a second reader on the optical head.
        self._stream_src = None if isinstance(path, (str, os.PathLike)) else path
        self._path = path
        self._label = (getattr(path, 'name', None) or os.path.basename(str(path))
                       if self._stream_src is not None else os.path.basename(str(path)))
        self._on_pcm = on_pcm
        self._avio = None                    # AVIOContext* (stream mode)
        self._avio_read_cb = None            # CFUNCTYPE kept alive for the C side
        # native contexts (owned by the worker; freed in stop() after join)
        self._ctx = None                     # AVFormatContext*  (c_void_p)
        self._dec = None                     # AVCodecContext*   (int)
        self._pkt = None                     # AVPacket*         (int)
        self._pkt_view = None
        self._frame = None                   # AVFrame*          (int)
        self._swr = None                     # SwrContext*       (c_void_p)
        self._aidx = -1
        self._tb_num, self._tb_den = 1, 1000
        self._in_rate = 0
        self._in_channels = 0
        self._in_order = 0                    # probed AVChannelLayout order/mask -- used verbatim
        self._in_mask = 0                     # by swr, not just the channel count (see _setup_swr)
        self._out_rate = 0
        # threading / control
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()         # guards _seek_req (the worker acquires it each loop)
        self._stop_lock = threading.Lock()    # serializes stop() teardown (NOT _lock -> no join deadlock)
        self._seek_req = None                # ms (guarded by _lock)
        self._started = False
        self._eof_sent = False
        self._synthetic_samples = 0          # fallback pts when a frame carries AV_NOPTS
        self._t0 = 0.0                        # free-run clock origin

    # ------------------------------------------------------------------ public
    def start(self, clock_ms):
        """Spawn the decode worker; the file OPEN happens on that worker, not
        here. No-op if already started; if the file has no decodable audio it
        logs and simply produces no PCM (never raises).

        The open moved off the caller's thread deliberately: CastController
        starts the tap on the GUI thread, and avformat's open+probe on a
        Blu-ray .ssif measured 14 SECONDS (mpegts probesize on an optical
        head) — the whole UI froze for the duration. Even plain files can
        stall a probe for hundreds of ms; the caller never needs the result
        synchronously (a failed open just means no PCM ever arrives)."""
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._eof_sent = False
        self._synthetic_samples = 0
        self._thread = threading.Thread(target=self._worker_main, args=(clock_ms,),
                                        name="SyLC-AudioTap", daemon=True)
        self._thread.start()

    def _worker_main(self, clock_ms):
        """Worker entry: open (possibly slow — see start()), then decode."""
        if not self._open():
            return
        if self._stop.is_set():
            return                            # stopped while the open was in flight
        self._run(clock_ms)

    def seek(self, pts_ms):
        """Reposition the audio decode to ~pts_ms. Recorded for the worker to apply at a
        safe point (av_seek_frame + avcodec_flush_buffers on the decode thread)."""
        with self._lock:
            self._seek_req = max(0, int(pts_ms))

    def stop(self):
        """Stop the worker and free every avcodec/swr context. Idempotent AND safe under
        concurrent callers -- a dedicated _stop_lock serializes teardown so two threads can't
        both pass the join and double-free. Never hangs (bounded join). Safe to call from
        within on_pcm: the worker-self check runs BEFORE the lock (else the worker would
        deadlock on its own join)."""
        self._stop.set()
        if self._thread is threading.current_thread():
            return                            # called from the worker (e.g. via on_pcm) -> just signal
        with self._stop_lock:
            t = self._thread
            if t is None:
                return                        # already torn down (idempotent)
            t.join(self._JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning("[ATAP] worker join timed out; native contexts left intact (still in use)")
                return                        # do NOT free under a live worker (would be use-after-free)
            self._thread = None
            self._free_native()
            # Stream mode: release the source (e.g. disable the demuxer's tee)
            # only once the worker is provably gone — its read callback polls
            # the source until then.
            if self._stream_src is not None:
                try:
                    close = getattr(self._stream_src, 'close', None)
                    if close is not None:
                        close()
                except Exception:
                    logger.exception("[ATAP] stream source close failed")

    # ------------------------------------------------------------------ open / free
    def _open(self):
        try:
            _sign_audio()
        except Exception as e:
            logger.error("[ATAP] DLL load/sign failed: %s", e)
            return False
        af, ac, au = _lavf._AVFORMAT, _lavf._AVCODEC, _lavf._AVUTIL
        if (ac.avcodec_version() >> 16) != 62:
            logger.warning("[ATAP] avcodec major != 62 -> audio offsets unverified; refusing")
            return False
        if self._stream_src is not None:
            if not self._open_custom_io(af, au):
                return False
        else:
            self._ctx = ctypes.c_void_p()
            if af.avformat_open_input(ctypes.byref(self._ctx),
                                      str(self._path).encode('utf-8'), None, None) < 0:
                logger.warning("[ATAP] avformat_open_input failed: %s", self._path)
                self._ctx = None
                return False
        try:
            if af.avformat_find_stream_info(self._ctx, None) < 0:
                raise RuntimeError("find_stream_info")
            self._aidx = af.av_find_best_stream(self._ctx, _AVMEDIA_TYPE_AUDIO, -1, -1, None, 0)
            if self._aidx < 0:
                raise RuntimeError("no audio stream")
            fmt = ctypes.cast(self._ctx, ctypes.POINTER(AVFormatContext))
            if self._aidx >= fmt.contents.nb_streams:
                raise RuntimeError("audio index out of range")
            sptr = fmt.contents.streams[self._aidx]
            cp_addr = ctypes.c_void_p.from_address(sptr + _lavf._OFF_CODECPAR).value
            if not cp_addr:
                raise RuntimeError("codecpar NULL")
            cp = AVCodecParameters.from_address(cp_addr)
            # stream time_base @32 -> pts scaling; fall back to 1/1000 if implausible
            tb = AVRational.from_address(sptr + _lavf._OFF_TIME_BASE)
            if 0 < tb.den <= 1_000_000_000 and 0 < tb.num <= 1_000_000_000:
                self._tb_num, self._tb_den = tb.num, tb.den
            else:
                self._tb_num, self._tb_den = 1, 1000
            # audio params via pinned offsets + runtime cross-check (major 62 already gated)
            order = ctypes.c_int.from_address(cp_addr + _OFF_CP_CH_ORDER).value
            nb_ch = ctypes.c_int.from_address(cp_addr + _OFF_CP_CH_NB_CHANNELS).value
            mask = ctypes.c_uint64.from_address(cp_addr + _OFF_CP_CH_MASK).value
            rate = ctypes.c_int.from_address(cp_addr + _OFF_CP_SAMPLE_RATE).value
            if not (1 <= nb_ch <= 64 and 8000 <= rate <= 384000 and 0 <= order <= 3):
                raise RuntimeError(f"audio param cross-check failed (ch={nb_ch} rate={rate} order={order})")
            if order == 1 and bin(mask).count("1") != nb_ch:     # AV_CHANNEL_ORDER_NATIVE
                raise RuntimeError(f"channel mask/count mismatch (mask={mask:#x} ch={nb_ch})")
            self._in_channels = nb_ch
            self._in_rate = rate
            self._in_order = order                               # keep the ACTUAL probed layout
            self._in_mask = mask                                 # (order+mask), not just the count
            # Perfectionist: Force 48kHz output. Quest 3 (and most Android) uses 48kHz
            # as native rate. Resampling on the PC (via swresample) is higher quality
            # than letting the Android mixer resample 44.1 or others.
            self._out_rate = 48000
            # open decoder
            codec = ac.avcodec_find_decoder(cp.codec_id)
            if not codec:
                raise RuntimeError("decoder not found")
            self._dec = ac.avcodec_alloc_context3(codec)
            if not self._dec:
                raise RuntimeError("avcodec_alloc_context3")
            if ac.avcodec_parameters_to_context(self._dec, cp_addr) < 0:
                raise RuntimeError("avcodec_parameters_to_context")
            if ac.avcodec_open2(self._dec, codec, None) < 0:
                raise RuntimeError("avcodec_open2")
            self._pkt = ac.av_packet_alloc()
            if not self._pkt:
                raise RuntimeError("av_packet_alloc")
            self._pkt_view = ctypes.cast(self._pkt, ctypes.POINTER(AVPacket))
            self._frame = au.av_frame_alloc()
            if not self._frame:
                raise RuntimeError("av_frame_alloc")
            logger.info("[ATAP] opened %s: aidx=%d codec=%s in=%dHz/%dch tb=%d/%d -> out=s16/%dHz/%dch",
                        self._label, self._aidx,
                        ac.avcodec_get_name(cp.codec_id).decode('ascii', 'replace'),
                        self._in_rate, self._in_channels, self._tb_num, self._tb_den,
                        self._out_rate, self._OUT_CHANNELS)
            return True
        except Exception as e:
            logger.warning("[ATAP] open refused: %s", e)
            self._free_native()
            return False

    def _avio_read(self, _opaque, buf, buf_size):
        """Custom-AVIO read callback (runs on the worker thread, inside a
        libav call). Blocks in small slices until the stream source has data
        or the tap is stopped: mpegts probing/reading expects a blocking
        read; a permanent b'' here (source torn down) resolves as EOF."""
        while not self._stop.is_set():
            try:
                data = self._stream_src.read(buf_size)
            except Exception:
                return _AVERROR_EOF
            if data:
                n = min(len(data), buf_size)
                ctypes.memmove(buf, data, n)
                return n
            time.sleep(0.005)
        return _AVERROR_EOF

    def _open_custom_io(self, af, au):
        """Build AVIOContext(read=self._avio_read) + AVFormatContext with pb
        set, then avformat_open_input over it (format probed from the bytes;
        mpegts locks onto the 0x47 cadence from any mid-stream position)."""
        buf = au.av_malloc(_AVIO_BUF_SIZE)
        if not buf:
            logger.warning("[ATAP] av_malloc(avio buffer) failed")
            return False
        self._avio_read_cb = _AVIO_READ_CB(self._avio_read)   # keep alive on self
        avio = af.avio_alloc_context(buf, _AVIO_BUF_SIZE, 0, None,
                                     self._avio_read_cb, None, None)
        if not avio:
            au.av_free(buf)
            logger.warning("[ATAP] avio_alloc_context failed")
            return False
        self._avio = ctypes.c_void_p(avio)
        ctx = af.avformat_alloc_context()
        if not ctx:
            self._free_avio()
            logger.warning("[ATAP] avformat_alloc_context failed")
            return False
        ctypes.cast(ctx, ctypes.POINTER(AVFormatContext)).contents.pb = avio
        self._ctx = ctypes.c_void_p(ctx)
        # Bound the probe: we only need PAT/PMT + the AUDIO stream's params.
        # The defaults (5 MB probesize, long analyzeduration) make open+
        # find_stream_info decode video mid-GOP for seconds, spewing h264
        # "non-existing PPS" on stderr the whole time.
        au = _lavf._AVUTIL
        opts = ctypes.c_void_p()
        au.av_dict_set(ctypes.byref(opts), b"probesize", b"2500000", 0)
        au.av_dict_set(ctypes.byref(opts), b"analyzeduration", b"2000000", 0)
        rc = af.avformat_open_input(ctypes.byref(self._ctx), b"", None, ctypes.byref(opts))
        au.av_dict_free(ctypes.byref(opts))
        if rc < 0:
            logger.warning("[ATAP] avformat_open_input(stream) failed: %s", self._label)
            self._ctx = None                  # open_input freed the context on failure
            self._free_avio()
            return False
        return True

    def _free_avio(self):
        """Free the custom AVIOContext + its (possibly reallocated) buffer.
        avformat_close_input does NOT free a caller-provided pb."""
        try:
            if self._avio is not None and self._avio.value:
                af, au = _lavf._AVFORMAT, _lavf._AVUTIL
                inner = ctypes.c_void_p.from_address(self._avio.value + _OFF_AVIO_BUFFER)
                if inner.value:
                    au.av_free(inner)
                af.avio_context_free(ctypes.byref(self._avio))
        except Exception:
            logger.exception("[ATAP] avio teardown failed")
        self._avio = None
        self._avio_read_cb = None

    def _free_native(self):
        """Free every avcodec/swr/avformat context. Each free is wrapped in its OWN try/except
        (ctypes can surface a Windows access violation as a Python exception) so a raise on one
        never skips the rest -- and each pointer is nulled afterwards for idempotency. Same
        convention as lavf_h264_demuxer.close()."""
        af, ac, au = (getattr(_lavf, n, None) for n in ("_AVFORMAT", "_AVCODEC", "_AVUTIL"))
        try:
            if self._frame and au:
                p = ctypes.c_void_p(self._frame)
                au.av_frame_free(ctypes.byref(p))
        except Exception:
            logger.exception("[ATAP] av_frame_free failed")
        self._frame = None
        try:
            if self._swr is not None and _SWR is not None:
                s = self._swr if isinstance(self._swr, ctypes.c_void_p) else ctypes.c_void_p(self._swr)
                if s.value:
                    _SWR.swr_free(ctypes.byref(s))
        except Exception:
            logger.exception("[ATAP] swr_free failed")
        self._swr = None
        try:
            if self._pkt and ac:
                p = ctypes.c_void_p(self._pkt)
                ac.av_packet_free(ctypes.byref(p))
        except Exception:
            logger.exception("[ATAP] av_packet_free failed")
        self._pkt = None
        self._pkt_view = None
        try:
            if self._dec and ac:
                p = ctypes.c_void_p(self._dec)
                ac.avcodec_free_context(ctypes.byref(p))
        except Exception:
            logger.exception("[ATAP] avcodec_free_context failed")
        self._dec = None
        try:
            if self._ctx and af:
                c = self._ctx if isinstance(self._ctx, ctypes.c_void_p) else ctypes.c_void_p(self._ctx)
                if c.value:
                    af.avformat_close_input(ctypes.byref(c))
        except Exception:
            logger.exception("[ATAP] avformat_close_input failed")
        self._ctx = None
        self._free_avio()                     # no-op in path mode

    # ------------------------------------------------------------------ worker
    def _run(self, clock_ms):
        if self._stream_src is not None:
            return self._run_stream(clock_ms)
        return self._run_path(clock_ms)

    def _run_stream(self, clock_ms):
        """Stream-mode loop: INGESTION DECOUPLED FROM EMISSION.

        The path-mode loop paces BEFORE decoding the next frame, which stalls
        av_read_frame whenever we are ahead of the clock — correct for a
        seekable file, fatal for a live tee: the upstream demuxer keeps
        producing and the bounded tee wraps (discontinuities, parser spam,
        hitches). Here decode always follows the SOURCE's rhythm into a
        bounded local PCM queue, and only the EMISSION of that queue is paced
        against the clock (the Quest buffers ~130ms of PCM, drop-oldest — it
        must never be flooded)."""
        from collections import deque
        pcm_q = deque()                       # (pts_ms, pcm)
        q_ms = 0.0                            # decoded duration queued
        chunk_ms = lambda pcm: (len(pcm) / (2 * self._OUT_CHANNELS)) * 1000.0 / self._out_rate
        try:
            self._t0 = time.monotonic()
            while not self._stop.is_set():
                sk = None
                with self._lock:
                    if self._seek_req is not None:
                        sk, self._seek_req = self._seek_req, None
                if sk is not None:
                    if self._do_seek(sk):
                        self._t0 = time.monotonic() - (sk / 1000.0)
                    pcm_q.clear()             # pre-seek PCM is stale
                    q_ms = 0.0
                    continue

                # 1) EMIT everything due (never blocks).
                c = None
                if clock_ms is not None:
                    try:
                        c = clock_ms()
                    except Exception:
                        c = None
                if c is None:
                    c = (time.monotonic() - self._t0) * 1000.0
                while pcm_q and pcm_q[0][0] <= c + self.LOOKAHEAD_MS:
                    pts_ms, pcm = pcm_q.popleft()
                    q_ms -= chunk_ms(pcm)
                    if self._stop.is_set():
                        return
                    try:
                        self._on_pcm(int(pts_ms), pcm, self._out_rate, self._OUT_CHANNELS)
                    except Exception:
                        logger.exception("[ATAP] on_pcm callback raised")

                # 2) INGEST at the source's rhythm while the local buffer has room.
                if q_ms < self.STREAM_BUFFER_S * 1000.0:
                    if not self._pull_frame():
                        # Stream EOF only happens on stop/teardown; otherwise the
                        # avio callback blocks until bytes arrive.
                        self._stop.wait(self._PACE_SLICE_S)
                        continue
                    chunk = self._convert_current()
                    if chunk is not None:
                        pcm_q.append(chunk)
                        q_ms += chunk_ms(chunk[1])
                else:
                    self._stop.wait(self._PACE_SLICE_S)   # absurdly far ahead: idle
        except Exception:
            logger.exception("[ATAP] stream worker crashed")
        finally:
            logger.debug("[ATAP] stream worker exit")

    def _run_path(self, clock_ms):
        try:
            self._t0 = time.monotonic()
            while not self._stop.is_set():
                # apply a pending seek (all libav* calls stay on this thread)
                sk = None
                with self._lock:
                    if self._seek_req is not None:
                        sk, self._seek_req = self._seek_req, None
                if sk is not None:
                    if self._do_seek(sk):                            # rebase ONLY when the seek landed
                        self._t0 = time.monotonic() - (sk / 1000.0)  # (else the position didn't move)
                    continue
                if not self._pull_frame():
                    break                                            # EOF
                chunk = self._convert_current()
                if chunk is None:
                    continue
                pts_ms, pcm = chunk
                if not self._pace(pts_ms, clock_ms):
                    continue                                         # stop / seek interrupted pacing -> drop
                with self._lock:
                    stale = self._seek_req is not None
                if stale or self._stop.is_set():
                    continue
                try:
                    self._on_pcm(int(pts_ms), pcm, self._out_rate, self._OUT_CHANNELS)
                except Exception:
                    logger.exception("[ATAP] on_pcm callback raised")
        except Exception:
            logger.exception("[ATAP] worker crashed")
        finally:
            logger.debug("[ATAP] worker exit")

    def _pace(self, pts_ms, clock_ms):
        """Block until it is time to emit a chunk with timestamp pts_ms: return True when
        pts_ms <= clock + LOOKAHEAD_MS, or False if stop/seek intervened. Free-runs at real
        time from decode start when clock_ms() returns None (or raises)."""
        while not self._stop.is_set():
            with self._lock:
                if self._seek_req is not None:
                    return False
            c = None
            if clock_ms is not None:
                try:
                    c = clock_ms()
                except Exception:
                    c = None
            if c is None:
                c = (time.monotonic() - self._t0) * 1000.0
            if pts_ms <= c + self.LOOKAHEAD_MS:
                return True
            self._stop.wait(self._PACE_SLICE_S)
        return False

    def _pull_frame(self):
        """Advance the decoder until a frame is ready in self._frame. Returns True (frame
        ready) or False (clean EOF / unrecoverable error). Mirrors the HEVC decode loop."""
        ac, af = _lavf._AVCODEC, _lavf._AVFORMAT
        while True:
            if self._stop.is_set():
                return False                                         # bail promptly out of a long decode/drain loop
            r = ac.avcodec_receive_frame(self._dec, self._frame)
            if r == 0:
                return True
            if r == _AVERROR_EOF:
                return False
            if r != _AVERROR_EAGAIN:
                logger.warning("[ATAP] receive_frame err %d -> stop", r)
                return False
            if self._eof_sent:
                return False                                         # drain complete, nothing more
            # feed one audio packet (skip other streams / empty packets)
            if af.av_read_frame(self._ctx, self._pkt) < 0:
                ac.avcodec_send_packet(self._dec, None)              # enter drain mode
                self._eof_sent = True
                continue
            pk = self._pkt_view.contents
            if pk.stream_index != self._aidx or pk.size <= 0:
                ac.av_packet_unref(self._pkt)
                continue
            s = ac.avcodec_send_packet(self._dec, self._pkt)
            ac.av_packet_unref(self._pkt)
            if s < 0 and s != _AVERROR_EAGAIN:
                logger.warning("[ATAP] send_packet err %d", s)

    def _convert_current(self):
        """Resample the frame in self._frame to interleaved s16le stereo. Returns
        (pts_ms, pcm_bytes) or None (no output / not convertible)."""
        if self._swr is None:
            if not self._setup_swr():
                self._stop.set()                                     # fatal: cannot convert -> stop cleanly
                return None
        nb = ctypes.c_int.from_address(self._frame + _OFF_FR_NB_SAMPLES).value
        if nb <= 0:
            return None
        ext = ctypes.c_void_p.from_address(self._frame + _OFF_FR_EXTENDED_DATA).value
        if not ext:
            return None
        # Capacity must include swresample's accumulated fractional delay and the
        # input/output rate ratio. `nb + 256` happened to cover 44.1->48 kHz movie
        # audio, but truncated valid output for low-rate sources and made the
        # resampler retain an ever-growing tail.
        delay_in_samples = max(0, int(_SWR.swr_get_delay(self._swr, self._in_rate)))
        out_max = max(
            1,
            ((delay_in_samples + nb) * self._out_rate + self._in_rate - 1)
            // self._in_rate,
        )
        out_buf = (ctypes.c_uint8 * (out_max * self._OUT_CHANNELS * 2))()
        out_planes = (ctypes.POINTER(ctypes.c_uint8) * 1)()
        out_planes[0] = ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_uint8))
        in_pp = ctypes.cast(ext, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)))
        n = _SWR.swr_convert(self._swr, out_planes, out_max, in_pp, nb)
        if n <= 0:
            return None
        pcm = ctypes.string_at(out_buf, n * self._OUT_CHANNELS * 2)   # exact bytes, no intermediate list
        pts = ctypes.c_int64.from_address(self._frame + _OFF_FR_PTS).value
        if pts != _AV_NOPTS:
            pts_ms = (pts * 1000 * self._tb_num) // self._tb_den
        else:
            pts_ms = (self._synthetic_samples * 1000) // self._out_rate
        # Synthetic PTS lives in the OUTPUT time base. Advancing by input samples
        # drifts whenever resampling (for example 44.1 -> 48 kHz).
        self._synthetic_samples += n
        return (pts_ms, pcm)

    def _setup_swr(self):
        """Lazily build the SwrContext once the first frame reveals the real input sample
        format. in layout/rate come from the (cross-checked) codecpar; out = s16 stereo."""
        au = _lavf._AVUTIL
        in_fmt = ctypes.c_int.from_address(self._frame + _OFF_FR_FORMAT).value
        if in_fmt < 0:
            logger.warning("[ATAP] invalid input sample_fmt %d", in_fmt)
            return False
        out_fmt = au.av_get_sample_fmt(b"s16")
        if out_fmt < 0:
            logger.warning("[ATAP] av_get_sample_fmt(s16) failed")
            return False
        in_layout = AVChannelLayout()
        out_layout = AVChannelLayout()
        # Input layout: use the ACTUAL probed layout (order + mask), NOT just the channel
        # count -- av_channel_layout_default(count) substitutes the STANDARD arrangement for
        # that count and would silently mislabel a real 5.1/7.1 stream whose mask differs.
        # For a validated NATIVE layout we copy order+mask verbatim; for the rare UNSPEC/CUSTOM
        # (no usable mask) we fall back to a synthesized standard layout so swr can still
        # downmix. Neither path allocates heap (default() uses a mask; our copy is shallow and
        # a CUSTOM map, if any, belongs to codecpar), and swr deep-copies both internally --
        # so there is nothing to uninit here.
        if self._in_order == 1 and bin(self._in_mask).count("1") == self._in_channels:
            in_layout.order = 1                                    # AV_CHANNEL_ORDER_NATIVE
            in_layout.nb_channels = self._in_channels
            in_layout.mask = self._in_mask
        else:
            au.av_channel_layout_default(ctypes.byref(in_layout), self._in_channels)
        au.av_channel_layout_default(ctypes.byref(out_layout), self._OUT_CHANNELS)
        self._swr = ctypes.c_void_p()
        r = _SWR.swr_alloc_set_opts2(ctypes.byref(self._swr),
                                     ctypes.byref(out_layout), out_fmt, self._out_rate,
                                     ctypes.byref(in_layout), in_fmt, self._in_rate, 0, None)
        ok = (r == 0 and bool(self._swr.value))
        if ok:
            ok = (_SWR.swr_init(self._swr) == 0)
        if not ok:
            logger.warning("[ATAP] swr init failed (r=%s)", r)
            if self._swr and self._swr.value:
                p = self._swr
                _SWR.swr_free(ctypes.byref(p))
            self._swr = None
            return False
        logger.info("[ATAP] swr ready: in fmt=%d %dHz/%dch (order=%d mask=%#x) -> out s16 %dHz/%dch",
                    in_fmt, self._in_rate, self._in_channels, self._in_order, self._in_mask,
                    self._out_rate, self._OUT_CHANNELS)
        return True

    def _do_seek(self, ms):
        """av_seek_frame (BACKWARD, to a keyframe at/before ms) + flush the decoder, and drop
        swresample's residual so post-seek audio never concatenates with pre-seek. Returns
        True iff the seek landed (the caller rebases the free-run clock only on success)."""
        af, ac = _lavf._AVFORMAT, _lavf._AVCODEC
        ts = int(max(0.0, float(ms)) * 1000.0)                      # stream -1 => AV_TIME_BASE microseconds
        if self._stream_src is None:
            if af.av_seek_frame(self._ctx, -1, ts, _AVSEEK_FLAG_BACKWARD) < 0:
                logger.warning("[ATAP] seek(%sms) refused", ms)
                return False
        # Stream mode: the source is not seekable — the UPSTREAM demuxer's own
        # seek already repositioned the byte stream and cleared its tee, so
        # here we only flush local decode state; mpegts resynchronizes on the
        # discontinuity by itself.
        ac.avcodec_flush_buffers(self._dec)
        self._eof_sent = False
        if self._swr is not None and self._swr.value:
            # A seek is a discontinuity, not end-of-stream: draining once does not
            # reset swresample's fractional delay/filter history. Destroy it and
            # lazily build a pristine context from the first post-seek frame.
            swr = self._swr
            _SWR.swr_free(ctypes.byref(swr))
            self._swr = None
        self._synthetic_samples = (ts * self._out_rate) // 1_000_000    # rebase fallback pts
        return True

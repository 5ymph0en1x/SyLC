"""Loopback receiver for SyLC Cast (PC stand-in for the Quest app).

Receives the cast stream a PC sender emits (Task 5's WifiTransport), decodes the HEVC
Annex-B video units with SyLC's own no-demuxer avcodec path (lavf_hevc_source, added by
Task 3), splits each decoded 3840x1080 frame into its two SBS eyes, and hands them to a
callback (and/or an optional cv2 preview window) -- proving the whole sender pipeline
works end-to-end WITHOUT a headset. Pure Python, CPU decode: no GPU needed here.

Real UDP is adversarial (reordering, loss, duplication, noise from anywhere else that
hits this port), so every inbound datagram is parsed and reassembled defensively: a
malformed datagram, or a corrupted fragment join, is dropped/logged and never crashes
the endpoint. A video unit that fails to decode is likewise dropped, not fatal.

The live stream is IDR (frame 0, forced) then P-frames (gopLength=INFINITE,
frameIntervalP=1), so each unit is streamed through ONE PERSISTENT AnnexbStreamDecoder
(created in start(), drained + freed in stop()) whose reference-frame state carries
across units -- that is what lets the P-frames decode. The standalone decode_and_split
(a fresh decoder per call, only good for a self-contained IDR unit) stays as-is for the
single-IDR fixture unit test.

Task 14 (robustness): as the Quest stand-in, this receiver also pings a lightweight
PT_CONTROL "keepalive" back to the sender every _KEEPALIVE_PERIOD_S (see
_schedule_keepalive) -- the real Quest app does the same. WifiTransport has no other
way to know a UDP peer is still there; this inbound traffic is what keeps its
liveness timeout (transport_wifi.WifiTransport._check_liveness) from firing and
pausing the sender's CastController while a receiver is actually still listening.
"""

import asyncio
import logging

from sylc.lavf_hevc_source import AnnexbStreamDecoder, decode_annexb_frames

from . import protocol as P

logger = logging.getLogger(__name__)


def decode_and_split(es: bytes):
    """Generator: decode a raw Annex-B HEVC access unit (bytes) via lavf_hevc_source's
    no-demuxer avcodec helper (decode_annexb_frames -- reused, not reimplemented), and for
    each decoded 3840x1080 SBS frame yield ((Yl,Ul,Vl),(Yr,Ur,Vr), pts_ms). Left/right
    split (done by lavf_hevc_source._extract_yuv420, split_sbs=True): Y[:, :1920]/[:, 1920:],
    U/V[:, :960]/[:, 960:].

    pts_ms is a SYNTHETIC 0-based decode-order counter, not a real timestamp: this direct
    avcodec path carries none -- lavf_hevc_source._decode_annexb sends every packet with
    pts=AV_NOPTS_VALUE by construction, so there is nothing real to read back off the
    decoded AVFrame. LoopbackReceiver has the REAL pts_ms from the wire protocol header
    for the unit it decodes and uses that instead of this generator's pts_ms when it
    calls on_frame; this synthetic counter exists only so a standalone caller (this
    module's own unit tests, e.g.) always gets a non-None value.
    """
    frames = decode_annexb_frames([bytes(es)], split_sbs=True)
    for i, (left, right) in enumerate(frames):
        yield left, right, i


class _Endpoint(asyncio.DatagramProtocol):
    """Forwards inbound datagrams to the owning LoopbackReceiver, which holds all the
    actual parsing/dispatch logic (same idiom as transport_wifi._Endpoint)."""

    def __init__(self, owner: "LoopbackReceiver"):
        self._owner = owner

    def datagram_received(self, data, addr):
        self._owner._on_datagram(data, addr)


class LoopbackReceiver:
    """UDP endpoint that reassembles + decodes + splits the cast video stream.

    Display is OPTIONAL: passing on_frame=None and never calling enable_preview() means
    pure decode + no callback at all -- safe for headless test runs. enable_preview()
    lazily imports cv2 INSIDE itself (never at module top level), so importing this
    module never requires cv2 to be installed.

    A/V sync (Task 9) is OPT-IN via av_sync_clock, a callable ()->int (ms), and is OFF
    by default -- with av_sync_clock=None every decoded eye-pair is still delivered to
    _emit IMMEDIATELY, exactly Task 7's original behavior (test_skeleton_e2e.py depends
    on this and is unaffected). When av_sync_clock IS supplied: decoded eye-pairs go into
    a small pts-sorted jitter buffer (_buffer_video) instead, and a presentation step
    (_present_due) releases every buffered frame whose pts_ms <= av_sync_clock() -- the
    spec's audio-is-the-master-clock model. PT_AUDIO packets (parsed-and-ignored in
    immediate mode) become that clock's data feed in sync mode: see _handle_audio_packet/
    audio_clock(). av_sync_clock itself is the actual "now" (on the Quest, real
    AudioTrack/Oboe playback position; here, whatever the caller/test injects) -- audio
    packets only report what pts range is available, they do not drive the gate directly.
    """

    _VIDEO_BUFFER_CAP = 8       # sync mode: drop-oldest-with-log past this many buffered frames
    _DRAIN_PERIOD_S = 0.02      # sync mode: periodic _present_due() so wall-clock-only progress
                                # (no new audio/video packet to trigger it) can't strand a frame
    _KEEPALIVE_PERIOD_S = 0.5   # Task 14: liveness ping cadence back to the sender

    def __init__(self, on_frame=None, av_sync_clock=None):
        self._on_frame = on_frame
        self._av_sync_clock = av_sync_clock   # None = T7 immediate mode; callable ()->ms = A/V sync (T9)
        self._transport = None
        self._peer = None
        self._active = False
        self._seq = 0
        self._video_reasm = P.Reassembler()   # one instance for the video stream (Task 4 contract)
        self._stream_decoder = None           # persistent AnnexbStreamDecoder (created in start)
        self._cv2 = None                      # set by enable_preview(); None = no display
        self._video_buffer = []               # sync mode only: [(pts_ms, left, right), ...] sorted by pts_ms
        self._audio_clock_ms = None           # sync mode only: latest audio pts_ms seen (see audio_clock())
        self._drain_handle = None             # sync mode only: the periodic _schedule_drain's call_later handle
        self._keepalive_handle = None         # Task 14: the periodic _schedule_keepalive's call_later handle

    def enable_preview(self):
        """Opt-in live preview window (cv2 imshow of the left eye), OFF by default. cv2
        is imported HERE, lazily, so this module stays importable -- and every
        non-preview test runnable -- on a machine without cv2 or a display. Composes
        with on_frame (both fire per decoded pair); safe to call before or after start()."""
        import cv2   # lazy: see docstring -- never at module top level
        self._cv2 = cv2

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def start(self, listen_host: str, listen_port: int, pc_host: str, pc_port: int) -> None:
        if self._active:
            return
        # Persistent decode session, split_sbs=True (each 3840x1080 frame -> two eyes).
        # Created BEFORE binding so a decode-unavailable host fails here without leaking
        # a socket; torn down again if the bind itself fails.
        self._stream_decoder = AnnexbStreamDecoder(split_sbs=True)
        try:
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(
                lambda: _Endpoint(self), local_addr=(listen_host, listen_port),
            )
        except Exception:
            self._stream_decoder.close()
            self._stream_decoder = None
            raise
        self._transport = transport
        self._peer = (pc_host, pc_port)
        self._active = True
        self._transport.sendto(P.pack_hello(self._next_seq()), self._peer)
        self._schedule_keepalive()
        if self._av_sync_clock is not None:
            self._schedule_drain()

    def _on_datagram(self, data: bytes, addr) -> None:
        try:
            msg = P.parse(data)
        except ValueError:
            return   # malformed datagram -- drop silently, keep serving

        if msg["type"] == P.PT_VIDEO:
            try:
                done = self._video_reasm.push(msg)
            except Exception:
                logger.warning("[cast-loopback] reassembly join failed, dropping fragment", exc_info=True)
                return
            for unit in done:
                self._handle_video_unit(unit)
        elif msg["type"] == P.PT_AUDIO and self._av_sync_clock is not None:
            self._handle_audio_packet(msg)
        # PT_AUDIO in immediate mode (no av_sync_clock): parsed-and-ignored, same as before
        # Task 9. PT_HELLO_ACK/PT_CONTROL/PT_BYE: no handler yet, defensively ignored.

    def _handle_video_unit(self, unit: dict) -> None:
        # Stream each completed PT_VIDEO unit through the PERSISTENT decoder: its
        # reference-frame state carries across units, so the live IDR-then-P stream
        # decodes fully (a fresh-per-unit decoder would drop every P-frame). The unit's
        # real wire pts_ms is stamped on the packet and rides through to its decoded
        # frame, so each on_frame carries that frame's OWN true pts even across the
        # codec's output delay.
        if self._stream_decoder is None:
            return
        try:
            pairs = self._stream_decoder.push(unit["payload"], unit["pts_ms"])
        except Exception:
            # A malformed / undecodable unit must never take the endpoint down with it.
            logger.warning("[cast-loopback] decode failed for a video unit, dropping", exc_info=True)
            return
        for left, right, pts_ms in pairs:
            if self._av_sync_clock is not None:
                self._buffer_video(left, right, pts_ms)
                self._present_due()
            else:
                self._emit(left, right, pts_ms)

    def _handle_audio_packet(self, msg: dict) -> None:
        """Sync-mode only (Task 9): PT_AUDIO packets are the clock's data feed, not the
        clock itself -- track the latest audio pts_ms seen (monotonic: real UDP can
        reorder/duplicate, so a regressing pts_ms is ignored rather than rewinding the
        tracked timeline backwards) and take a presentation step, since fresh audio
        arriving is one of the two arrival events (the other is a video unit decoding)
        that can make an already-buffered video frame due."""
        pts_ms = msg["pts_ms"]
        if self._audio_clock_ms is None or pts_ms > self._audio_clock_ms:
            self._audio_clock_ms = pts_ms
        self._present_due()

    def audio_clock(self) -> int:
        """Sync-mode bookkeeping: the latest audio pts_ms observed from PT_AUDIO packets --
        i.e. how far the audio TIMELINE has been fed, not a wall clock -- 0 before any audio
        has arrived. Distinct from av_sync_clock(), which is the actual playback-position
        "now" that _present_due gates against; this is diagnostic/available-range
        bookkeeping only, not itself a presentation gate."""
        return self._audio_clock_ms if self._audio_clock_ms is not None else 0

    def _buffer_video(self, left, right, pts_ms) -> None:
        """Sync-mode seam (Task 9): insert one decoded eye-pair into the jitter buffer,
        kept sorted by pts_ms (target depth 1-2 frames; network jitter/decoder handling
        could still hand frames to us slightly out of order). Capped at
        _VIDEO_BUFFER_CAP -- if a stuck/slow clock ever let the buffer grow past that, the
        OLDEST (smallest-pts) frame is dropped with a log rather than growing unbounded;
        in normal operation _present_due drains buffered frames long before the cap is
        ever reached."""
        entry = (pts_ms, left, right)
        idx = 0   # buffer is small (target depth 1-2) -> a linear insert is fine
        while idx < len(self._video_buffer) and self._video_buffer[idx][0] <= pts_ms:
            idx += 1
        self._video_buffer.insert(idx, entry)
        if len(self._video_buffer) > self._VIDEO_BUFFER_CAP:
            dropped_pts = self._video_buffer.pop(0)[0]
            logger.warning("[cast-loopback] video jitter buffer overflow (cap=%d), dropped pts_ms=%s",
                           self._VIDEO_BUFFER_CAP, dropped_pts)

    def _present_due(self) -> None:
        """Sync-mode seam (Task 9): release every buffered eye-pair whose pts_ms <=
        av_sync_clock() ("now"), oldest first. A frame far behind the clock is still
        presented -- present-late beats drop for a movie, so there is no separate
        staleness/drop check here beyond the cap _buffer_video already enforces. No-op in
        immediate mode (av_sync_clock is None) or while the buffer is empty."""
        if self._av_sync_clock is None or not self._video_buffer:
            return
        now = self._av_sync_clock()
        while self._video_buffer and self._video_buffer[0][0] <= now:
            pts_ms, left, right = self._video_buffer.pop(0)
            self._emit(left, right, pts_ms)

    def _flush_video_buffer(self) -> None:
        """Sync mode only, called at stop(): unconditionally emit every remaining
        buffered frame in pts order, ignoring av_sync_clock. Once the endpoint is torn
        down nothing will ever call _present_due again, so holding a late frame back here
        would silently lose it instead of just presenting it late."""
        while self._video_buffer:
            pts_ms, left, right = self._video_buffer.pop(0)
            self._emit(left, right, pts_ms)

    def _schedule_keepalive(self) -> None:
        """Task 14: periodic proof-of-life back to the sender. A real
        WifiTransport peer only knows the far end is alive from INBOUND
        traffic (its own liveness timeout -- see
        transport_wifi.WifiTransport._check_liveness), so this receiver --
        the Quest stand-in -- pings a lightweight PT_CONTROL "keepalive" on a
        fixed cadence; the real Quest app does the same. Self-rescheduling
        (same idiom as _schedule_drain); cancelled in stop()."""
        if not self._active:
            return
        self._transport.sendto(P.pack_control("keepalive"), self._peer)
        loop = asyncio.get_running_loop()
        self._keepalive_handle = loop.call_later(self._KEEPALIVE_PERIOD_S, self._schedule_keepalive)

    def _schedule_drain(self) -> None:
        """Sync mode only: a lightweight periodic _present_due() so a frame that only
        becomes due because WALL TIME passed -- with no new audio/video packet arriving
        to trigger the check directly -- is never stranded in the jitter buffer.
        Self-reschedules on the running loop; cancelled in stop()."""
        if not self._active:
            return
        self._present_due()
        loop = asyncio.get_running_loop()
        self._drain_handle = loop.call_later(self._DRAIN_PERIOD_S, self._schedule_drain)

    def _emit(self, left, right, pts_ms) -> None:
        """Deliver one decoded eye-pair to on_frame and/or the preview window. In immediate
        mode (no av_sync_clock), called directly from the live push path
        (_handle_video_unit) and the end-of-stream tail flush (stop); in sync mode, called
        only from _present_due / _flush_video_buffer once a buffered frame's pts_ms is due
        or the stream is ending."""
        if self._on_frame is not None:
            self._on_frame(left, right, pts_ms)
        if self._cv2 is not None:
            y, _u, _v = left
            self._cv2.imshow("SyLC Cast Loopback (L eye)", y)
            self._cv2.waitKey(1)

    async def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._keepalive_handle is not None:
            self._keepalive_handle.cancel()
            self._keepalive_handle = None
        if self._drain_handle is not None:
            # Sync mode: stop the periodic drain first. _schedule_drain also no-ops once
            # self._active is False, but cancelling avoids one dead wakeup after teardown.
            self._drain_handle.cancel()
            self._drain_handle = None
        # Drain the persistent decoder's retained tail frame(s) -- the codec can hold up
        # to one frame past its last input -- so no decoded frame is lost at end-of-stream,
        # then free it. Each flushed frame still carries its own true pts (reorder queue).
        if self._stream_decoder is not None:
            try:
                for left, right, pts_ms in self._stream_decoder.flush():
                    if self._av_sync_clock is not None:
                        self._buffer_video(left, right, pts_ms)
                    else:
                        self._emit(left, right, pts_ms)
            except Exception:
                logger.warning("[cast-loopback] tail-flush failed", exc_info=True)
            self._stream_decoder.close()
            self._stream_decoder = None
        if self._av_sync_clock is not None:
            # End of stream: nothing will call _present_due again, so present whatever is
            # still buffered right now rather than let av_sync_clock gating silently lose it.
            self._flush_video_buffer()
        if self._transport is not None:
            # Windows ProactorEventLoop gotcha (same one found+fixed in
            # transport_wifi.WifiTransport.stop): closing while a write (here, the
            # PT_HELLO sent back in start() -- there is no guaranteed intervening await
            # between that send and an immediate stop(), e.g. test_receiver_stop_idempotent)
            # is still in flight leaves its socket un-nulled until __del__ runs, which
            # surfaces as an "unclosed transport" ResourceWarning. A real (if tiny) slice
            # of wall-clock time before AND after close() lets both the pending write's
            # completion callback and close's own teardown callback run first.
            await asyncio.sleep(0.01)
            self._transport.close()
            await asyncio.sleep(0.01)
            self._transport = None
        self._peer = None
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()

"""SyLC Cast -- CastController: the one session brain (Task 12).

This is the INTEGRATION seam. It ties four already-built, already-reviewed pieces
into a single cast session and -- crucially -- it is where the cross-thread
transport hazard flagged by the Task 9 & 10 reviews is FIXED for real:

  * the native renderer cast pipeline (cast_start / cast_encode / cast_reconfigure
    / cast_stop) -- Tasks 3 & 11;
  * a TransportServer (WifiTransport / UsbTransport) -- Tasks 5 & 10;
  * the independent AudioTap PCM decode -- Task 8;
  * the pure FallbackLadder quality policy -- Task 11.

THE THREADING MODEL (the whole point of this task)
--------------------------------------------------
Three threads are in play and getting the marshaling right is the core of the
work:

  * GUI/render thread (Qt main thread): owns the NativeRenderer and its D3D11
    device. ALL renderer calls happen here -- cast_start, cast_encode (packs SBS +
    NVENC-encodes on the device), cast_reconfigure, cast_stop. `push()` is called
    on this thread (from the player's frame tap).
  * Cast IO thread (dedicated, created here): runs an asyncio event loop that owns
    the transport. The transport's sockets / _seq / sendto|write may be touched
    ONLY on this thread.
  * AudioTap worker thread: AudioTap decodes audio and fires `on_pcm(...)` on its
    own thread.

The invariant: the transport is only ever touched on the IO thread; the renderer
is only ever touched on the GUI thread. Marshaling bridges them, both directions:

  * GUI -> IO (video): `push()` encodes on the renderer (GUI thread) then, for
    each NAL, `self._loop.call_soon_threadsafe(self._send_video_io, ...)` so the
    actual `transport.send_video` runs on the IO thread. NEVER call send_video
    directly from the GUI thread.
  * AUDIO -> IO (audio): AudioTap's `on_pcm` (audio thread) ->
    `self._loop.call_soon_threadsafe(self._transport.send_audio, pts_ms, pcm)`.
  * IO -> GUI (control): the transport fires `on_control` on the IO thread; we
    turn client seek/pause into Qt signals (`seekRequested` / `pauseRequested`).
    Because this QObject lives on the GUI thread, emitting from the IO thread
    delivers queued to the main thread.
  * IO -> GUI (reconfigure): bwfeedback (IO thread) folds through the pure ladder
    (any thread), but APPLYING the step (`renderer.cast_reconfigure`) must be on
    the GUI thread -- so it is marshaled via an internal queued Qt signal
    (`_applyReconfigure` -> `_do_reconfigure`).

Session discipline: one session at a time (a 2nd start while active is refused);
stop() is idempotent and cancel-safe -- no zombie IO thread, no leaked AudioTap
worker, no half-open transport, and a start()->stop() with zero frames in between
must not throw. `push()` returns fast and never blocks the GUI thread: if the IO
send queue is backed up it drops the frame and counts it.

Design notes captured here (ambiguities the brief left to a clean call):
  * push() can reuse the shared display renderer's just-uploaded planes. The
    shipping integration verifies that the upload succeeded on that exact
    renderer before opting in; standalone callers keep the safe upload-then-encode
    behavior. This removes a redundant six-plane CPU->GPU transfer per cast frame
    without ever encoding stale textures after a display failure/transition.
  * `stereo_mode` / `sdr_white` are accepted for tap-signature parity with
    NativeRendererTap.push but are unused by the cast path: the NV12-SBS pack is a
    fixed L|R lossless copy of the YUV planes; brightness/stereo uniforms only
    matter for the display composite.
  * `cbr_lowres` rung (Task 11): cast_reconfigure keeps 3840x1080, so v1 applies
    cbr_lowres as the lowest CBR bitrate and LOGS that the true resolution drop is
    deferred -- we do not silently claim a resolution drop that is not happening.
  * PC-side send-queue depth is used only for local frame-drop backpressure; the
    FallbackLadder is driven by the client's bwfeedback (per the brief's v1 note).

ROBUSTNESS (Task 14): the transport now exposes on_client_lost -- Wi-Fi via an
inbound-liveness timeout, USB-C via the TCP reader's own EOF/reset -- wired here
to `_on_client_lost`, which PAUSES the session (a `_paused` flag under
`_state_lock`; `push()` checks it and returns before touching the renderer or the
IO loop at all) instead of tearing anything down: the renderer stays cast_start'd,
the transport + IO thread stay up, so a later reconnect can resume cleanly. A
fresh handshake (`_on_client` firing again) clears the pause and arms `_due_idr`
so the reconnected client's very next frame opens on an IDR. `_on_client_lost`
also no-ops while `_stopping` is set, so a real client-lost racing a
user-initiated `stop()` can't emit a spurious pause status mid-teardown.
`pause_on_disconnect` (constructor kwarg, default True) opts out.

`_due_idr` cross-thread safety (FIX PASS -- the first cut of this had a real
TOCTOU, see below): `_on_client`/`_on_client_lost` fire on the IO thread; `push()`
runs on the GUI thread but `cast_encode` RELEASES THE GIL for its native
GPU-pack+NVENC body (python_bindings.cpp: `py::gil_scoped_release` around
`r.cast_encode(...)`), so the IO thread genuinely runs concurrently with an
in-flight push(), not merely "interleaved by scheduling". `push()` therefore reads
`_due_idr` and, if it was True, clears it to False in ONE step under
`_state_lock` -- BEFORE calling cast_encode, not after. The original version of
this code read the flag early but only cleared it, unconditionally, AFTER
cast_encode returned; that was a classic TOCTOU -- a reconnect's `_due_idr=True`,
set by `_on_client` WHILE the encode was in flight, could be silently clobbered by
that same push's own unconditional post-encode clear, permanently losing the
reconnected client's forced IDR with no retry. Clearing adjacent to the read fixes
this: nothing later in push() does an unconditional write, so every early-return
failure path between the read-clear and success instead RESTORES `_due_idr=True`
if (and only if) this push was the one that claimed it -- preserving the existing
"soft failure keeps the IDR armed" contract. `_on_client`'s own `_due_idr=True`
write is inside the SAME `with self._state_lock:` block as its
`_paused`/`_connected` writes, so it can never interleave with push()'s
read-clear either. (`_do_reconfigure`'s `_due_idr=True` is intentionally NOT
lock-guarded: it runs on the GUI thread via a queued Qt connection, which cannot
be dispatched while that same thread is synchronously blocked inside push()'s
cast_encode call -- there is no actual concurrency to guard against there.)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

from PySide6.QtCore import QObject, Signal, Qt

from . import protocol as P
from .audio_tap import AudioTap
from .fallback import FallbackLadder
from .transport_base import TransportServer
from .transport_usb import UsbTransport
from .transport_wifi import WifiMediaTransport

logger = logging.getLogger(__name__)


def _make_transport(kind: str) -> TransportServer:
    """Resolve the transport implementation for a session kind. Kept a module
    function (not a method) so tests can swap it without touching __init__.

    "wifi" is TCP media + a UDP discovery responder (WifiMediaTransport): the
    Quest receiver's only media path is TCP (SyLcTcpClient) and its discovery
    is a UDP broadcast HELLO -- the old pure-UDP WifiTransport satisfied only
    the discovery half, so Wi-Fi sessions could never actually connect."""
    if kind == "usb":
        return UsbTransport()
    return WifiMediaTransport()


class CastController(QObject):
    """One cast session: renderer encode -> transport, audio -> transport,
    client feedback -> quality ladder, client control -> player signals.

    Parameters
    ----------
    renderer :
        The NativeRenderer exposing cast_start / cast_encode / cast_reconfigure /
        cast_stop / cast_available / set_yuv_frame. Touched ONLY on the GUI thread.
    media_path_provider : callable ()->str|None
        Returns the current media file path for the independent AudioTap decode.
    clock_ms : callable ()->int|None
        The playback audio clock in ms (the player's _mpv_time_pos_ms): the PTS
        source for encoded frames and AudioTap's pacing reference.
    pause_on_disconnect : bool
        Task 14: when a connected client's liveness times out (Wi-Fi) or its TCP
        connection drops (USB-C), pause the session (stop encoding/sending, keep
        everything else up) instead of ignoring it. Default True; an explicit
        False makes on_client_lost a no-op (push() keeps encoding/sending as if
        nothing happened).
    reuse_uploaded_frame : bool
        Default frame-upload policy. True is valid only when the caller shares
        this renderer with the display and invokes push immediately after a
        successful display upload. It may be overridden per push.
    """

    # {"connected":bool,"mode":str,"bitrate":int,"fps":float,"dropped":int,"error":str}
    statusChanged = Signal(dict)
    seekRequested = Signal(int)          # ms -- from a client control message
    pauseRequested = Signal(bool)        # True = pause, False = resume/play
    # internal IO->GUI marshal: apply a ladder step on the GUI thread (queued).
    _applyReconfigure = Signal(object)

    DEFAULT_FPS = 24
    # v1 encodes a fixed side-by-side canvas: cast_reconfigure changes the
    # bitrate rung, never the geometry. Announced to the receiver rather
    # than assumed by it.
    CAST_WIDTH = 3840
    CAST_HEIGHT = 1080
    LISTEN_HOST = "0.0.0.0"
    DEFAULT_PORT = 47420
    # Bound GUI -> IO latency. The encoder currently emits one access unit per
    # frame, so twelve callbacks are roughly 500 ms at 24 fps. Keeping the old
    # two-second allowance let stale frames accumulate long after the receiver
    # had stopped consuming them.
    MAX_INFLIGHT = 12
    _IO_JOIN_TIMEOUT = 5.0
    _IO_SHUTDOWN_TIMEOUT = 5.0
    _STATUS_EMIT_INTERVAL = 1.0          # throttle continuous (per-frame) status emits

    def __init__(self, renderer, media_path_provider, clock_ms,
                 pause_on_disconnect: bool = True, reuse_uploaded_frame: bool = False):
        super().__init__()
        self._renderer = renderer
        self._media_path_provider = media_path_provider
        self._clock_ms = clock_ms
        self._pause_on_disconnect = pause_on_disconnect
        self._reuse_uploaded_frame = bool(reuse_uploaded_frame)

        # cross-thread guards
        self._state_lock = threading.Lock()   # _active/_stopping/_connected/_paused/_mode/_bitrate/_error
        self._q_lock = threading.Lock()        # _inflight/_dropped backpressure accounting

        # session state
        self._active = False
        self._stopping = False
        self._paused = False                    # Task 14: client presumed gone -- push() no-ops
        self._main10 = False                    # HEVC Main10/P010 HDR session (set by start())
        self._cast_started = False
        self._transport_kind = None
        self._transport = None                 # set + touched on the IO thread
        self._audio = None
        self._ladder = None

        # IO thread / loop
        self._loop = None                      # asyncio loop owned by the IO thread
        self._thread = None
        self._loop_ready = None                # threading.Event: loop up + transport started
        self._io_error = None

        # encode/stream state (GUI thread)
        self._due_idr = False
        self._synth_pts_ms = 0
        self._pts_cursor_ms = None
        self._last_clock_ms = None
        self._last_push_mono = None
        self._fps_ema = 0.0

        # backpressure / status
        self._inflight = 0
        self._dropped = 0
        self._connected = False
        self._mode = None
        self._bitrate = 0
        self._fps = self.DEFAULT_FPS
        self._error = ""
        self._last_status_emit = 0.0
        self._last_feedback_log = 0.0
        self._last_receiver_idr_request = 0.0
        self._transport_pressure_since_feedback = False

        # Apply reconfigure ALWAYS on the GUI thread via a queued connection, so it
        # can never run on the IO thread regardless of which thread emitted it.
        self._applyReconfigure.connect(self._do_reconfigure, Qt.QueuedConnection)

    # -- introspection ------------------------------------------------------- #
    @property
    def is_active(self) -> bool:
        with self._state_lock:
            return self._active

    @property
    def is_paused(self) -> bool:
        """Task 14: True while the session is paused (on_client_lost fired) but
        not torn down -- the renderer/transport/IO thread all stay up; a
        reconnect (on_client firing again) clears this and forces the next
        push()'s encode to open on a fresh IDR."""
        with self._state_lock:
            return self._paused

    @property
    def dropped(self) -> int:
        with self._q_lock:
            return self._dropped

    @property
    def audio_active(self) -> bool:
        """True while the independent AudioTap is running (False = the session
        streams video-only, e.g. no media path / tap start failed)."""
        return self._audio is not None

    @property
    def is_main10(self) -> bool:
        """True while the session encodes HEVC Main10/P010 (HDR cast): the
        frame tap may then push 10-bit (uint16) planes."""
        return bool(getattr(self, '_main10', False))

    # -- lifecycle ----------------------------------------------------------- #
    def start(self, transport: str, quality: str, main10: bool = False) -> None:
        """Bring up one cast session. transport "wifi"|"usb"; quality
        "auto"|"lossless"|"cbr". main10=True encodes HEVC Main10/P010 with PQ
        BT.2020 signalling (HDR sources); the receiver learns it from the
        HELLO_ACK "hdr" field AND from the in-bitstream VUI. Refuses a 2nd
        start while a session is active."""
        with self._state_lock:
            if self._active:
                logger.warning("[CAST] start ignored: a session is already active")
                return
            self._active = True
            self._stopping = False
            self._error = ""
            self._connected = False
            self._paused = False
        self._main10 = bool(main10)

        with self._q_lock:
            self._inflight = 0
            self._dropped = 0

        self._due_idr = True
        self._synth_pts_ms = 0
        self._pts_cursor_ms = None
        self._last_clock_ms = None
        self._last_push_mono = None
        self._fps_ema = 0.0
        self._cast_started = False
        self._transport_kind = transport
        self._last_feedback_log = 0.0
        self._last_receiver_idr_request = 0.0
        self._transport_pressure_since_feedback = False

        wired = (transport == "usb")
        if quality == "lossless":
            start_mode = "lossless"
        elif quality == "cbr":
            start_mode = "cbr"
        else:                                   # "auto"
            # A 4K-SBS lossless stream can consume gigabits and monopolize the
            # render/IO threads. Auto starts at the visually-lossless 500 Mbps
            # rung and the receiver's 2 Hz pressure feedback steps it down before
            # its four-AU MediaCodec pool saturates. Lossless remains explicit.
            start_mode = "balanced"

        self._ladder = FallbackLadder(start_mode, wired=wired)
        rung = self._ladder.current
        with self._state_lock:
            self._mode = rung["mode"]
            self._bitrate = rung["bitrate_bps"]

        # 1) Bring up the IO thread + asyncio loop + transport (transport is
        #    created + started on that thread; it is never touched from here).
        if not self._start_io():
            logger.error("[CAST] transport failed to start: %s", self._io_error)
            self._error = "transport start failed: %s" % (self._io_error,)
            self._shutdown()
            return

        # 2) Start the renderer cast pipeline (GUI thread -- its D3D11 device).
        try:
            ok = bool(self._renderer.cast_start(self._mode, self._fps, int(self._bitrate),
                                                self._main10))
        except TypeError:
            # Older pyd without the main10 parameter: SDR session only.
            if self._main10:
                logger.warning("[CAST] renderer lacks main10 support; falling back to 8-bit")
                self._main10 = False
            try:
                ok = bool(self._renderer.cast_start(self._mode, self._fps, int(self._bitrate)))
            except Exception:
                logger.exception("[CAST] renderer.cast_start raised")
                ok = False
        except Exception:
            logger.exception("[CAST] renderer.cast_start raised")
            ok = False
        if not ok:
            self._error = "renderer cast_start failed"
            logger.error("[CAST] %s", self._error)
            self._shutdown()
            return
        self._cast_started = True

        # 3) Independent audio decode -> marshaled send_audio.
        self._start_audio()

        logger.info("[CAST] session up: transport=%s mode=%s bitrate=%d fps=%d",
                    transport, self._mode, self._bitrate, self._fps)
        self._emit_status()

    def push(self, left, right, stereo_mode, sdr_white,
             frame_already_uploaded: bool | None = None) -> None:
        """Tap entry (mirrors NativeRendererTap.push): upload one stereo YUV frame,
        encode it on the renderer (GUI thread), and marshal each NAL to the IO
        thread for sending. Returns fast; never blocks the GUI thread. Drops (and
        counts) the frame when the IO send queue is backed up. No-ops -- without
        touching the renderer, encoding nothing, sending nothing, and NOT counted
        as a drop -- while paused (Task 14: the client is presumed gone; a
        reconnect clears this)."""
        if not self._active:
            return
        with self._state_lock:
            if self._paused:
                return
        loop = self._loop
        if loop is None:
            return

        # Backpressure: if the IO send queue is backed up, drop -- do not encode,
        # do not block. (stereo_mode / sdr_white are unused by the cast pack.)
        # NB: decide + count UNDER the lock, but emit status OUTSIDE it --
        # _emit_status() re-acquires _q_lock, so calling it while held would
        # self-deadlock (non-reentrant Lock) the GUI thread on a sustained stall.
        with self._q_lock:
            backed_up = self._inflight >= self.MAX_INFLIGHT
            if backed_up:
                self._dropped += 1
                self._transport_pressure_since_feedback = True
        if backed_up:
            self._maybe_emit_status()
            return

        pts = self._next_pts()
        # Read-then-clear _due_idr in ONE step, under _state_lock, BEFORE the
        # (possibly slow, GIL-released) cast_encode call below -- see the module
        # docstring's ROBUSTNESS note (FIX PASS): cast_encode releases the GIL for
        # its native GPU-pack+NVENC body, so the IO thread's _on_client can run
        # truly concurrently with an in-flight push(). Clearing here, adjacent to
        # the read, means nothing LATER in this function does an unconditional
        # write -- every failure path below restores _due_idr=True ONLY IF this
        # push was the one that claimed it, so a concurrent reconnect's own
        # _due_idr=True (set on the IO thread while we're inside cast_encode) can
        # never be clobbered by this push's completion.
        with self._state_lock:
            force_idr = self._due_idr
            if force_idr:
                self._due_idr = False
        try:
            # The shipping integration calls push immediately after the shared
            # display renderer uploaded these exact planes. Re-uploading all six
            # planes here cost ~6 MiB/frame of CPU->GPU traffic and serialized the
            # render thread. Standalone/test callers retain the safe upload path.
            reuse_upload = (
                self._reuse_uploaded_frame
                if frame_already_uploaded is None
                else bool(frame_already_uploaded)
            )
            # 10-bit (uint16) planes must go through set_yuv_frame16: the plain
            # set_yuv_frame binding forcecasts to uint8, which TRUNCATES every
            # P010 MSB-aligned value (k << 6) to its low byte — usually zero.
            _upload = self._renderer.set_yuv_frame
            if getattr(left[0], 'dtype', None) is not None and left[0].dtype.itemsize == 2:
                _upload = getattr(self._renderer, 'set_yuv_frame16', _upload)
            if (not reuse_upload and
                    not _upload(*left, *right)):
                with self._q_lock:
                    self._dropped += 1
                logger.warning("[CAST] set_yuv_frame failed; dropping frame")
                if force_idr:
                    with self._state_lock:
                        self._due_idr = True    # restore what we claimed but didn't deliver
                return
            nals = self._renderer.cast_encode(int(pts), force_idr)
        except Exception:
            logger.exception("[CAST] encode failed; dropping frame")
            with self._q_lock:
                self._dropped += 1
            if force_idr:
                with self._state_lock:
                    self._due_idr = True        # restore -> retry IDR next frame
            return

        # An EMPTY return is the native soft-error signal (native_renderer.h:
        # "empty on error, see last_error()"), distinct from a raised exception.
        # Count it and restore the IDR claim (if we made one) so a soft failure
        # right after start/reconfigure never loses the "stream opens on an IDR"
        # guarantee.
        if not nals:
            with self._q_lock:
                self._dropped += 1
            logger.warning("[CAST] cast_encode returned no packets; dropping frame")
            if force_idr:
                with self._state_lock:
                    self._due_idr = True        # restore -> retry IDR next frame
            return

        # Encode produced packets: the stream now carries this frame. _due_idr was
        # already cleared above (if we were the one who claimed it) -- do NOT
        # unconditionally clear it again here: that unconditional post-encode clear
        # is exactly the TOCTOU this function's opening block exists to avoid.
        for nal in nals:
            with self._q_lock:
                self._inflight += 1
            try:
                loop.call_soon_threadsafe(self._send_video_io, int(pts), nal, force_idr)
            except RuntimeError:
                # loop stopped between the guard and here -> undo + count a drop.
                with self._q_lock:
                    self._inflight = max(0, self._inflight - 1)
                    self._dropped += 1

        self._update_fps()
        self._maybe_emit_status()

    def stop(self) -> None:
        """Tear the session down. Idempotent and safe to call twice; a
        start()->stop() with no frames must not throw. No zombie IO thread, no
        leaked AudioTap worker, no half-open transport."""
        self._shutdown()

    def seek_audio(self, pts_ms: int) -> None:
        """Reposition the independent cast-audio decoder.

        AudioTap owns its libavcodec contexts on its worker thread, so seek() only
        posts a request and is safe from the GUI/player seek path. Keeping this
        explicit prevents AudioTap from decoding every intervening packet at full
        speed after a large playhead jump.
        """
        audio = self._audio
        if audio is None:
            return
        try:
            audio.seek(max(0, int(pts_ms)))
        except Exception:
            logger.exception("[CAST] AudioTap seek failed")

    def on_control(self, msg: dict) -> None:
        """Handle a PT_CONTROL dict from the transport (fired on the IO thread).
        Client seek/pause become Qt signals (marshaled to the GUI thread);
        bwfeedback folds through the ladder and (if a step results) is marshaled to
        the GUI thread for application."""
        ctrl = msg.get("control") if isinstance(msg, dict) else None
        if not ctrl:
            return
        kind = ctrl.get("kind")
        if kind == "seek":
            try:
                self.seekRequested.emit(int(ctrl.get("pos_ms", 0)))
            except Exception:
                logger.exception("[CAST] seek emit failed")
        elif kind == "pause":
            self.pauseRequested.emit(True)
        elif kind == "play":
            self.pauseRequested.emit(False)
        elif kind == "bwfeedback":
            ladder = self._ladder
            if ladder is None:
                return
            depth = max(0, int(ctrl.get("queue_depth", 0)))
            receiver_pressure = bool(ctrl.get("underrun", False))
            needs_idr = bool(ctrl.get("needs_idr", False))
            with self._q_lock:
                transport_pressure = self._transport_pressure_since_feedback
                self._transport_pressure_since_feedback = False
            pressure = receiver_pressure or transport_pressure

            # A dropped compressed AU can break the reference chain even after
            # the queue has drained. Ask the next accepted frame to be an IDR,
            # but rate-limit recovery IDRs so sustained pressure cannot turn the
            # stream into an expensive all-intra sequence.
            if needs_idr:
                now = time.monotonic()
                with self._state_lock:
                    if now - self._last_receiver_idr_request >= 2.0:
                        self._due_idr = True
                        self._last_receiver_idr_request = now

            now = time.monotonic()
            if (now - self._last_feedback_log >= 5.0 or pressure or needs_idr):
                with self._q_lock:
                    inflight, dropped = self._inflight, self._dropped
                logger.info(
                    "[CAST] feedback depth=%d receiver_pressure=%s "
                    "transport_pressure=%s needs_idr=%s "
                    "sender_inflight=%d sender_dropped=%d bitrate=%d",
                    depth, receiver_pressure, transport_pressure, needs_idr,
                    inflight, dropped, self._bitrate,
                )
                self._last_feedback_log = now
            try:
                step = ladder.on_feedback(depth, pressure)
            except Exception:
                logger.exception("[CAST] ladder.on_feedback raised")
                return
            if step:
                self._applyReconfigure.emit(step)   # -> _do_reconfigure on the GUI thread

    # -- IO thread bring-up / tear-down -------------------------------------- #
    def _start_io(self) -> bool:
        """Spawn the IO thread, wait until its loop is up and the transport has
        started (or startup failed). Returns True iff the transport is serving."""
        self._io_error = None
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._io_main, name="SyLC-Cast-IO",
                                        daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(self._IO_JOIN_TIMEOUT):
            self._io_error = "loop did not become ready in %ss" % self._IO_JOIN_TIMEOUT
            return False
        return self._io_error is None

    def _io_main(self) -> None:
        """IO-thread entry: own an asyncio loop, start the transport on it, then
        serve until stop() stops the loop. All transport touches live here."""
        try:
            loop = asyncio.new_event_loop()
        except Exception as e:                  # pragma: no cover - loop alloc is reliable
            self._io_error = "new_event_loop: %s" % e
            self._loop_ready.set()
            return
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._io_startup())
        except Exception as e:
            logger.exception("[CAST] IO startup failed")
            self._io_error = str(e) or repr(e)
            self._close_loop(loop)
            self._loop = None
            self._loop_ready.set()
            return

        # Transport is serving: signal ready, then run until stop() stops us.
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            self._close_loop(loop)

    async def _io_startup(self) -> None:
        """Create + start the transport ON the IO loop and wire its callbacks."""
        transport = _make_transport(self._transport_kind)
        transport.on_control = self.on_control
        transport.on_client = self._on_client
        transport.on_client_lost = self._on_client_lost
        # Announced in PT_HELLO_ACK so the receiver configures its decoder and
        # screen from what we send rather than assuming a fixed geometry.
        transport.stream_format = P.stream_announcement(
            self.CAST_WIDTH, self.CAST_HEIGHT, self._fps, stereo="lr",
            hdr=("pq" if getattr(self, '_main10', False) else None),
        )
        await transport.start(self.LISTEN_HOST, self.DEFAULT_PORT)
        self._transport = transport

    async def _io_shutdown(self) -> None:
        """Stop the transport ON the IO loop (its sockets/tasks release here)."""
        t = self._transport
        if t is not None:
            await t.stop()

    @staticmethod
    def _close_loop(loop) -> None:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass

    def _stop_io(self) -> None:
        """From the GUI thread: stop the transport on its loop, stop the loop, and
        join the IO thread. Never called from the IO thread (would self-join)."""
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None
        if loop is not None and thread is not None and thread.is_alive():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._io_shutdown(), loop)
                fut.result(timeout=self._IO_SHUTDOWN_TIMEOUT)
            except Exception:
                logger.exception("[CAST] transport shutdown error/timeout")
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            thread.join(self._IO_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning("[CAST] IO thread did not exit within %ss",
                               self._IO_JOIN_TIMEOUT)
        elif thread is not None and thread.is_alive():  # pragma: no cover - odd path
            thread.join(self._IO_JOIN_TIMEOUT)
        self._transport = None

    def _shutdown(self) -> None:
        """The single idempotent teardown path (GUI thread only). Reused by stop(),
        by start()'s failure branches, and by the reconfigure-failure contract."""
        # Defensive guard (for Tasks 13/14): teardown joins the IO thread and calls
        # renderer.cast_stop -- both illegal from the IO thread itself (a self-join
        # would raise the cryptic "cannot join current thread"). Fail loudly and
        # clearly instead, so a stop() accidentally wired into an IO-thread callback
        # is caught at the call site rather than corrupting teardown.
        io_thread = self._thread
        if io_thread is not None and io_thread is threading.current_thread():
            raise RuntimeError(
                "CastController.stop() must be called from the GUI thread, not the "
                "cast IO thread (teardown joins that thread and calls cast_stop on "
                "the renderer's device). Marshal the stop to the GUI thread instead.")
        with self._state_lock:
            if self._stopping:
                return                          # a teardown is already in progress
            if not self._active and self._thread is None and self._audio is None:
                return                          # nothing to tear down (idempotent)
            self._stopping = True

        # 1) Stop the audio worker first (no more send_audio marshaling).
        audio = self._audio
        self._audio = None
        if audio is not None:
            try:
                audio.stop()
            except Exception:
                logger.exception("[CAST] AudioTap.stop failed")

        # 2) Stop the renderer cast pipeline (GUI thread) -- only if it started.
        if self._cast_started:
            try:
                self._renderer.cast_stop()
            except Exception:
                logger.exception("[CAST] renderer.cast_stop failed")
        self._cast_started = False

        # 3) Stop the transport on its loop, stop the loop, join the IO thread.
        self._stop_io()

        with self._state_lock:
            self._active = False
            self._connected = False
            self._paused = False
            self._stopping = False
        self._emit_status()

    # -- audio --------------------------------------------------------------- #
    def _start_audio(self) -> None:
        """Start the independent AudioTap; its on_pcm marshals send_audio to the IO
        loop. Missing media / ffmpeg is non-fatal -- the session runs video-only."""
        path = None
        try:
            path = self._media_path_provider() if self._media_path_provider else None
        except Exception:
            logger.exception("[CAST] media_path_provider raised")
        if not path:
            logger.info("[CAST] no media path -> audio disabled")
            return
        try:
            self._audio = AudioTap(path, self._on_pcm)
            self._audio.start(self._clock_ms)
        except Exception:
            logger.exception("[CAST] AudioTap start failed (continuing video-only)")
            self._audio = None

    def _on_pcm(self, pts_ms, pcm, sample_rate, channels) -> None:
        """AudioTap callback (fires on AudioTap's worker thread). Marshal the send
        onto the IO loop -- the exact call_soon_threadsafe fix the Task-9 review
        prescribed; self._loop was captured at start()."""
        loop = self._loop
        transport = self._transport
        if loop is None or transport is None or not self._active:
            return
        try:
            loop.call_soon_threadsafe(transport.send_audio, int(pts_ms), pcm)
        except RuntimeError:
            pass                                # loop stopped mid-teardown -> drop

    # -- video send (IO thread) ---------------------------------------------- #
    def _send_video_io(self, pts, nal, keyframe) -> None:
        """Runs ON the IO thread (scheduled by push via call_soon_threadsafe): the
        only place transport.send_video is ever called."""
        try:
            t = self._transport
            if t is not None:
                accepted = t.send_video(pts, nal, keyframe)
                if accepted is False:
                    with self._q_lock:
                        self._dropped += 1
                        self._transport_pressure_since_feedback = True
                    # The receiver did not get this reference frame. Arm an IDR
                    # immediately so the first frame accepted after pressure
                    # clears is independently decodable.
                    with self._state_lock:
                        self._due_idr = True
        except Exception:
            logger.exception("[CAST] transport.send_video failed")
        finally:
            with self._q_lock:
                self._inflight = max(0, self._inflight - 1)

    # -- reconfigure (GUI thread, queued) ------------------------------------ #
    def _do_reconfigure(self, step) -> None:
        """Apply a FallbackLadder step on the GUI thread. Honors the Task-11
        contract: a False return means the session is unhealthy -> stop + error."""
        with self._state_lock:
            if not self._active or self._stopping:
                return
        mode = step.get("mode")
        bitrate = int(step.get("bitrate_bps", 0))
        # cbr_lowres keeps 3840x1080 in v1 -> apply as plain lowest-CBR and say so.
        call_mode = "cbr" if mode == "cbr_lowres" else mode
        if mode == "cbr_lowres":
            logger.warning("[CAST] cbr_lowres rung: applying lowest CBR bitrate (%d bps) at "
                           "full 3840x1080 -- true resolution drop is deferred (v1)", bitrate)

        try:
            ok = bool(self._renderer.cast_reconfigure(call_mode, bitrate))
        except Exception:
            logger.exception("[CAST] renderer.cast_reconfigure raised")
            ok = False

        if not ok:
            # Task-11 handoff: False = driver rejected AND the encoder-reopen's
            # registerInput failed -> the pipeline is inactive with a
            # possibly-live-but-unregistered session. Treat as unhealthy: stop.
            logger.error("[CAST] cast_reconfigure(%s,%d) returned False -> session "
                         "unhealthy, stopping", call_mode, bitrate)
            self._error = "reconfigure failed (session unhealthy)"
            self._shutdown()
            return

        with self._state_lock:
            self._mode = mode
            self._bitrate = bitrate
        self._due_idr = True                    # fresh IDR after a rung change
        logger.info("[CAST] reconfigured -> mode=%s bitrate=%d (next frame forced IDR)",
                    mode, bitrate)
        self._emit_status()

    # -- client presence / liveness (Task 14) --------------------------------- #
    def _on_client(self, addr, hello) -> None:
        """Transport handshake callback (IO thread): a client is present --
        either the initial connect, or a RECONNECT after _on_client_lost paused
        the session. A reconnect clears the pause and arms a forced IDR for the
        very next push(), so the (re)connected client always gets a clean start
        regardless of whatever the encoder's last state was."""
        with self._state_lock:
            self._connected = True
            was_paused = self._paused
            self._paused = False
            if was_paused:
                # Inside the SAME lock acquisition as the _paused/_connected writes
                # above (see the module docstring's ROBUSTNESS note): this is the
                # same _state_lock push() takes to read-then-clear _due_idr, so this
                # write can never interleave with that critical section.
                self._due_idr = True
        if was_paused:
            logger.info("[CAST] client reconnected: %s -- resuming (forced IDR)", addr)
        else:
            logger.info("[CAST] client connected: %s", addr)
        self._emit_status()

    def _on_client_lost(self) -> None:
        """Transport liveness callback (Task 14; fires on the IO thread -- Wi-Fi's
        own inbound timeout, or USB-C's TCP reader hitting EOF/reset): the
        connected client is presumed gone. Pauses the session -- push() stops
        encoding/sending -- WITHOUT tearing anything down: the renderer stays
        cast_start'd, the transport + IO thread stay up, so a later reconnect
        (_on_client above) can resume cleanly. Never touches the renderer or the
        IO loop itself. No-op if the caller opted out via
        pause_on_disconnect=False, if the session already ended or a teardown is
        already in progress (_stopping -- FIX PASS: a real client-lost racing a
        user-initiated stop() must not fight _shutdown() with a spurious pause +
        status emit), or if already paused (fires the pause transition at most
        once per silence episode)."""
        if not self._pause_on_disconnect:
            return
        with self._state_lock:
            if not self._active or self._paused or self._stopping:
                return
            self._paused = True
            self._connected = False
        logger.info("[CAST] client lost -- pausing session (transport/IO thread stay up)")
        self._emit_status()

    # -- status -------------------------------------------------------------- #
    def _next_pts(self) -> int:
        """Return a phase-locked, strictly progressing video PTS.

        mpv's cached audio clock can update more slowly than the 24 Hz video,
        so copying it verbatim gives several consecutive access units the same
        timestamp. The Quest now schedules Surface presentation against its
        AudioTrack clock, which needs a real per-frame timeline. Advance at the
        nominal cadence, gently pull toward live mpv samples, and snap only on a
        genuine discontinuity such as seek.
        """
        c = None
        if self._clock_ms is not None:
            try:
                c = self._clock_ms()
            except Exception:
                c = None
        try:
            c = None if c is None else float(c)
        except (TypeError, ValueError):
            c = None

        frame_ms = 1000.0 / max(1.0, float(self._fps))
        cursor = self._pts_cursor_ms
        previous_clock = self._last_clock_ms
        clock_changed = (
            c is not None and
            (previous_clock is None or abs(c - previous_clock) > 0.001)
        )
        if clock_changed:
            self._last_clock_ms = c
        if cursor is None:
            cursor = c if c is not None else 0.0
        else:
            expected = cursor + frame_ms
            if c is None or not clock_changed:
                cursor = expected
            else:
                error = c - expected
                if abs(error) > 250.0:
                    cursor = c                       # seek / source discontinuity
                else:
                    cursor = expected + max(-1.0, min(1.0, error * 0.25))

        self._pts_cursor_ms = cursor
        self._synth_pts_ms = int(round(cursor + frame_ms))
        return int(round(cursor))

    def _update_fps(self) -> None:
        now = time.monotonic()
        last = self._last_push_mono
        self._last_push_mono = now
        if last is not None:
            dt = now - last
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema <= 0 else (0.9 * self._fps_ema + 0.1 * inst)

    def _maybe_emit_status(self) -> None:
        now = time.monotonic()
        if now - self._last_status_emit >= self._STATUS_EMIT_INTERVAL:
            self._emit_status()

    def _emit_status(self) -> None:
        self._last_status_emit = time.monotonic()
        with self._q_lock:
            dropped = self._dropped
        with self._state_lock:
            connected = self._connected
            mode = self._mode
            bitrate = self._bitrate
            error = self._error
        fps = round(self._fps_ema, 1) if self._fps_ema > 0 else float(self._fps)
        status = {
            "connected": bool(connected),
            "mode": mode,
            "bitrate": int(bitrate) if bitrate is not None else 0,
            "fps": fps,
            "dropped": int(dropped),
            "error": error or "",
        }
        try:
            self.statusChanged.emit(status)
        except Exception:
            logger.exception("[CAST] statusChanged emit failed")

    # -- test / teardown aid ------------------------------------------------- #
    def _wait_io_idle(self, timeout: float = 2.0) -> None:
        """Block until the IO loop has processed every callback scheduled so far.
        Schedules a sentinel via call_soon_threadsafe (FIFO after all pending
        sends) and waits for it -- deterministic, no polling."""
        loop = self._loop
        if loop is None:
            return
        ev = threading.Event()
        try:
            loop.call_soon_threadsafe(ev.set)
        except RuntimeError:
            return                              # loop not running
        ev.wait(timeout)

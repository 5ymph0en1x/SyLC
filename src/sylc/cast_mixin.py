# -*- coding: utf-8 -*-
"""Quest Cast session coordination for PlayerWindow."""

import logging
import os

from sylc.time_slider import _decide_thumbs_mode


logger = logging.getLogger(__name__)


class CastMixin:
    class _DemuxerStreamTapSource:
        """Non-blocking audio source backed by the native demuxer's byte tap."""

        def __init__(self, demuxer, label):
            self._demuxer = demuxer
            self.name = label

        def read(self, n):
            try:
                return self._demuxer.read_stream_tap(n)
            except Exception:
                return b""

        def close(self):
            try:
                self._demuxer.disable_stream_tap()
            except Exception:
                pass

    def _is_optical_class_source(self, path):
        """True when `path` sits on an optical-class volume — physical disc OR
        player-mounted ISO. The class where ONE MORE concurrent reader breaks
        playback: the single optical head already serves the video demuxer AND
        mpv's audio; a third reader causes the measured 45-120s seek thrash
        (same policy, same detection as the thumbnail gate)."""
        if not path:
            return False
        try:
            from sylc import disc_archiver as da
            optical = set(da.list_optical_drives())
        except Exception:
            optical = set()
        try:
            _mode, is_optical = _decide_thumbs_mode(
                path, self._mounted_iso_letters(), optical, None)
            return bool(is_optical)
        except Exception:
            return False

    def _cast_media_path(self):
        """Audio source for the cast's independent AudioTap decode.

        Regular files: the file path (AudioTap opens its own avformat reader —
        harmless on HDD/SSD). Optical-class sources (physical disc / mounted
        ISO): NEVER a path — a third concurrent reader on the single optical
        head froze playback (8.4s+ single demux reads measured). Instead, the
        SSIF demuxer TEES the bytes it already reads (enable_stream_tap) and
        AudioTap demuxes the audio from that stream: full disc audio on the
        headset, zero additional disc I/O. If no tappable demuxer is up
        (shouldn't happen during 3D disc playback), the cast degrades to
        video-only rather than ever touching the head."""
        p = getattr(self, 'current_file_path', None)
        if not p:
            return None
        if not self._is_optical_class_source(p):
            return p
        demux = getattr(getattr(self, 'mvc_decoder_thread', None), 'demuxer', None)
        if demux is not None and hasattr(demux, 'enable_stream_tap'):
            try:
                if demux.enable_stream_tap(32 * 1024 * 1024):
                    logger.info("[CAST] optical source: cast audio demuxed from the "
                                "demuxer's stream tap (no extra disc reader)")
                    return self._DemuxerStreamTapSource(demux, os.path.basename(p))
            except Exception:
                logger.exception("[CAST] enable_stream_tap failed")
        logger.info("[CAST] optical-class source without a tappable demuxer "
                    "-> video-only cast")
        return None

    def _cast_renderer(self):
        """The NativeRenderer the cast encodes from.

        Preferred source: the framepack DISPLAY widget's renderer (Task 13, the
        user's « option a »: share the display's renderer/upload — the framepack
        surface is guaranteed genuine L+R stereo, so the zero-re-upload path
        applies). FALLBACK: the embedded preview widget's renderer, so casting
        no longer requires the framepack window to have been opened. On that
        renderer the frame handler's identity check naturally reports
        frame_already_uploaded=False (the preview only holds one eye), so
        CastController.push() uploads BOTH eyes itself — full stereo on the
        headset either way, at the cost of the ~6 MiB/frame upload the shared
        framepack path avoids. Subtitles follow too: the preview widget receives
        the same PGS/uniform state as every display widget."""
        for holder in (getattr(getattr(self, 'framepacking_window', None),
                               'display_widget', None),
                       getattr(self, 'mvc_embedded_widget', None)):
            r = getattr(holder, '_r', None) if holder is not None else None
            if r is not None and r is not False:
                return r
        return None

    def _on_cast_requested(self, transport):
        """« Diffuser vers Quest » clicked (transport 'wifi'|'usb'). Toggles the cast
        session: a request while already casting tears the current session down."""
        # Already casting -> toggle OFF (idempotent stop; clear the reference).
        if self._cast is not None:
            try:
                self._cast.stop()
            except Exception:
                logger.exception("[CAST] stop failed")
            self._cast = None
            self._cast_connected = False
            self._cast_transport = None
            self.controls_overlay.set_cast_transport_state(None)
            self.show_3d_notification("Streaming to Quest stopped.", success=True)
            return

        # Defense in depth (the menu gate already enforces this): a live 3D session
        # and an NVENC-capable renderer are required.
        renderer = self._cast_renderer()
        session_3d = (getattr(self, 'mvc_mode_active', False)
                      or getattr(self, '_hevc_mode_active', False))
        if renderer is None or not session_3d:
            self.show_3d_notification(
                "Streaming unavailable — start 3D playback first.", success=False)
            return
        try:
            if not renderer.cast_available():
                self.show_3d_notification(
                    "Streaming unavailable — no NVENC encoder (NVIDIA GPU) was found.",
                    success=False)
                return
        except Exception:
            logger.exception("[CAST] cast_available() check failed")
            self.show_3d_notification("Streaming unavailable.", success=False)
            return
        try:
            from sylc.cast_sender.cast_controller import CastController
        except Exception as e:
            logger.exception("[CAST] CastController import failed")
            self.show_3d_notification(f"Streaming unavailable: {e}", success=False)
            return

        self._cast_connected = False
        self._cast_10bit_warned = False   # fresh session -> allow the 10-bit notice once
        cast = CastController(renderer,
                              media_path_provider=self._cast_media_path,
                              clock_ms=self._mpv_time_pos_ms,
                              reuse_uploaded_frame=True)
        cast.seekRequested.connect(self._on_cast_seek)
        cast.pauseRequested.connect(self._on_cast_pause)
        cast.statusChanged.connect(self._on_cast_status)
        # Publish BEFORE start(): start() brings up the IO thread and start() itself is
        # slow-ish; a frame tap firing meanwhile calls push(), which no-ops safely while
        # the loop is still coming up. If start() fails it self-tears-down -> is_active
        # is False and we drop the dead reference.
        self._cast = cast
        # HDR cast: a 10-bit PQ HEVC source goes out as Main10/P010 with its PQ
        # signalling; everything else keeps the 8-bit SDR session. (The Quest's
        # panels are not true HDR, but its compositor's contrast enhancement
        # reads the PQ declaration — and correctness beats guessing.)
        main10 = bool(getattr(self, '_hevc_mode_active', False)
                      and getattr(self, '_hevc_pq10', False))
        if main10:
            logger.info("[CAST] 10-bit PQ source -> HEVC Main10 HDR cast session")
        cast.start(transport, quality="auto", main10=main10)
        if not cast.is_active:
            self._cast = None      # start() failed; _on_cast_status already surfaced why
            return
        self._cast_transport = transport
        # Status light: session up on this transport, Quest not attached yet
        # (orange). Turns green from _on_cast_status when the client connects.
        self.controls_overlay.set_cast_transport_state(transport, False)
        label = "Wi-Fi" if transport == "wifi" else "USB-C"
        if cast.audio_active:
            self.show_3d_notification(f"Streaming to Quest ({label})…", success=True)
        else:
            self.show_3d_notification(
                f"Streaming to Quest ({label})… video only (no audio source).",
                success=True)

    def _on_cast_seek(self, pos_ms):
        """Cast client asked to seek: drive the player's normal seek path (ms -> s)."""
        try:
            self.on_seek(int(pos_ms) / 1000.0)
        except Exception:
            logger.exception("[CAST] seek from client failed")

    def _on_cast_pause(self, paused):
        """Cast client asked to pause/resume: mirror the play/pause button path so mpv,
        the decoder and the UI all follow (same two calls toggle_play() makes)."""
        try:
            self._safe_mpv_set_pause(bool(paused))
            self._handle_pause_change(bool(paused))
        except Exception:
            logger.exception("[CAST] pause from client failed")

    def _on_cast_status(self, status):
        """Light cast status handler: surface errors + the first client connect, log
        the rest. Called ~1/s while casting (throttled by the controller)."""
        try:
            err = status.get('error') if isinstance(status, dict) else None
            if err:
                self.show_3d_notification(f"Streaming to Quest: {err}", success=False)
                logger.warning("[CAST] %s", err)
                # An errored session tears itself down -> lights off.
                self._cast_transport = None
                self.controls_overlay.set_cast_transport_state(None)
                return
            connected = bool(status.get('connected'))
            if connected and not self._cast_connected:
                self.show_3d_notification("Quest connected — streaming.", success=True)
            self._cast_connected = connected
            # Status light: green while the Quest is attached, orange while the
            # session waits for it (incl. the paused-on-client-lost state), off
            # once no session is active.
            active = (self._cast is not None
                      and getattr(self._cast, 'is_active', False))
            self.controls_overlay.set_cast_transport_state(
                self._cast_transport if active else None, connected)
            logger.debug("[CAST] status: %s", status)
        except Exception:
            pass


__all__ = ['CastMixin']

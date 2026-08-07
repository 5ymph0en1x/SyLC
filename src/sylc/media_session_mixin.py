"""Ownership and lifecycle of media sessions and the persistent mpv core."""

import logging
import threading
import time

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

_MPV_RELEASE_SETTLE_MS = 1500
MPV_MODULE = None


def configure_media_session_support(mpv_module):
    """Inject python-mpv after the entry point has prepared native DLL paths."""
    global MPV_MODULE
    MPV_MODULE = mpv_module


class MediaSessionMixin:
    def _session_is_current(self, session_id, *, core=None):
        """Cheap cross-thread-safe ownership test (plain Python references only)."""
        if getattr(self, '_app_closing', False):
            return False
        if session_id != getattr(self, '_media_session_id', None):
            return False
        if core is not None and core is not getattr(self, 'player', None):
            return False
        return True

    def _native_signal_is_current(self):
        """Reject a queued signal emitted by a decoder from an older session."""
        sender = self.sender()
        if sender is None:  # direct calls and focused unit tests
            return True
        if getattr(sender, '_sylc_session_id', None) != self._media_session_id:
            return False
        return sender in (
            getattr(self, 'mvc_decoder_thread', None),
            getattr(self, 'hevc_thread', None),
        )

    def _begin_media_session(self, requested_path):
        """Invalidate all deferred work from the previous title and mint a token."""
        old_cancel = getattr(self, '_media_cancel_event', None)
        if old_cancel is not None:
            old_cancel.set()
        seek_queue = getattr(self, '_seek_queue', None)
        if seek_queue is not None:
            seek_queue.invalidate_session()
        self._remove_mpv_media_observers()
        self._remove_mpv_subtext_observer()
        self._media_session_id += 1
        self._media_cancel_event = threading.Event()
        self._loading_session_id = self._media_session_id
        self._pgs_startup_pending_session = None
        self._mpv_transition_in_progress = True
        logger.info("[SESSION] begin #%d: %s",
                    self._media_session_id, requested_path)
        return self._media_session_id

    def _invalidate_media_session(self, reason):
        """Make every queued callback/worker result from the active title stale."""
        cancel = getattr(self, '_media_cancel_event', None)
        if cancel is not None:
            cancel.set()
        seek_queue = getattr(self, '_seek_queue', None)
        if seek_queue is not None:
            seek_queue.invalidate_session()
        self._remove_mpv_media_observers()
        self._remove_mpv_subtext_observer()
        self._media_session_id += 1
        self._media_cancel_event = threading.Event()
        self._media_cancel_event.set()  # no active media owns this generation
        self._loading_session_id = None
        self._pgs_startup_pending_session = None
        logger.info("[SESSION] invalidated -> #%d (%s)",
                    self._media_session_id, reason)
        return self._media_session_id

    def _abort_media_load(self, session_id, message=None):
        """Release the load gate only if it still belongs to this attempt."""
        if not self._session_is_current(session_id):
            return
        self._is_loading_file = False
        self._loading_session_id = None
        self._pgs_startup_pending_session = None
        self._mpv_transition_in_progress = False
        if self.player is not None and self.current_file_path:
            self._install_mpv_media_observers(session_id, self.player)
        if message:
            self.show_3d_notification(message, success=False)

    def _media_single_shot(self, delay_ms, callback, session_id=None):
        """Run a timer callback only while its originating media session owns it."""
        owner = (self._media_session_id if session_id is None else session_id)

        def guarded():
            if self._session_is_current(owner):
                callback()

        QTimer.singleShot(int(delay_ms), guarded)

    def _start_media_worker(self, target, *, session_id, name):
        """Track a daemon worker and prevent it from publishing after cancellation."""
        cancel_event = self._media_cancel_event

        def run_owned():
            try:
                if not cancel_event.is_set() and self._session_is_current(session_id):
                    target(cancel_event)
            finally:
                with self._media_workers_lock:
                    self._media_workers.discard(threading.current_thread())

        worker = threading.Thread(target=run_owned, daemon=True, name=name)
        with self._media_workers_lock:
            self._media_workers.add(worker)
        worker.start()
        return worker

    def _cancel_media_workers(self, wait_ms=750):
        cancel = getattr(self, '_media_cancel_event', None)
        if cancel is not None:
            cancel.set()
        with self._media_workers_lock:
            workers = list(self._media_workers)
        deadline = time.monotonic() + max(0, wait_ms) / 1000.0
        for worker in workers:
            if worker is threading.current_thread() or not worker.is_alive():
                continue
            worker.join(max(0.0, deadline - time.monotonic()))

    def _initialize_player(self):
        """Configures and initializes the mpv instance with optimal settings."""
        QTimer.singleShot(100, self._setup_mpv_player)

    def _release_mpv_core(self, dying):
        """Begin releasing a DETACHED core (self.player already nulled by the
        caller): async unload now, deferred terminate once cooled."""
        self._remove_mpv_media_observers(dying)
        self._remove_mpv_subtext_observer(dying)
        self._mpv_dying = dying
        try:
            dying.command('stop')
        except Exception:
            pass
        QTimer.singleShot(_MPV_RELEASE_SETTLE_MS, self._finish_mpv_release)

    def _drain_dying_core(self):
        """Terminate a pending detached core NOW (cheap once it has settled;
        correct even when it hasn't — just slower). Idempotent."""
        dying = getattr(self, '_mpv_dying', None)
        if dying is None:
            return
        self._mpv_dying = None
        t0 = time.perf_counter()
        try:
            dying.terminate()
        except Exception:
            logger.exception("[MPV] deferred core release failed")
        else:
            logger.info("[MPV] detached core released in %.0f ms",
                        (time.perf_counter() - t0) * 1000.0)

    def _finish_mpv_release(self):
        """Timer tail of _release_mpv_core: drain, then restore the ordering
        the old synchronous flow guaranteed (ISO dismount and player re-init
        both strictly AFTER the core is gone)."""
        self._drain_dying_core()
        if getattr(self, '_dismount_after_release', False):
            self._dismount_after_release = False
            QTimer.singleShot(400, self._dismount_isos_after_stop)
        # No re-init while the app is going down (a Stop right before close
        # would otherwise resurrect a core mid-exit), and none either when a
        # quick re-load already built one.
        if self.player is None and not getattr(self, '_app_closing', False):
            self._initialize_player()

    def _remove_mpv_media_observers(self, core=None):
        """Detach session observers, optionally only those owned by ``core``."""
        keep = []
        for owner, name, handler in list(getattr(self, '_mpv_media_observers', [])):
            if core is not None and owner is not core:
                keep.append((owner, name, handler))
                continue
            try:
                owner.unobserve_property(name, handler)
            except Exception:
                pass
        self._mpv_media_observers = keep

    def _remove_mpv_subtext_observer(self, core=None):
        entry = getattr(self, '_mpv_subtext_observer', None)
        if entry is None:
            self._mpv_subtext_observer_registered = False
            return
        owner, handler = entry
        if core is not None and owner is not core:
            return
        try:
            owner.unobserve_property('sub-text', handler)
        except Exception:
            pass
        self._mpv_subtext_observer = None
        self._mpv_subtext_observer_registered = False

    def _install_mpv_media_observers(self, session_id, core=None):
        """Attach one exact observer set to one core and one media session."""
        core = core or getattr(self, 'player', None)
        if core is None or not self._session_is_current(session_id, core=core):
            return False
        self._remove_mpv_media_observers()
        handlers = (
            ('time-pos', lambda name, value, s=session_id, c=core:
             self.on_time_update(name, value, s, c)),
            ('duration', lambda name, value, s=session_id, c=core:
             self.on_duration_change(name, value, s, c)),
            ('pause', lambda name, value, s=session_id, c=core:
             self.on_pause_state_change(name, value, s, c)),
            ('eof-reached', lambda name, value, s=session_id, c=core:
             self.on_end_of_file(name, value, s, c)),
        )
        installed = []
        try:
            for name, handler in handlers:
                core.observe_property(name, handler)
                installed.append((core, name, handler))
        except Exception:
            for owner, name, handler in installed:
                try:
                    owner.unobserve_property(name, handler)
                except Exception:
                    pass
            logger.exception("[MPV] Could not install media-session observers")
            return False
        self._mpv_media_observers = installed
        logger.info("[MPV] Media observers attached to session #%d", session_id)
        return True

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
            # No Lua at all: mpv loads its BUILTIN scripts (stats, console,
            # select) even with osc=no, and their LuaJIT VM is one source of
            # the benign first-chance SEH 0xE24C4A02 exceptions that trip
            # faulthandler into a traceback dump (see the faulthandler
            # enable() comment near the end of this file — mpv also raises
            # that code from command('stop') with scripts disabled, so this
            # option alone does NOT silence it). SyLC draws its own UI, so
            # mpv-side scripts are dead weight; dropping them removes their
            # exception traffic and their startup cost.
            'load-scripts': 'no',
            'volume': 100,
            'mute': 'no',
            # The subtitle combo starts on "None". Prevent mpv from silently
            # auto-selecting an embedded track behind that UI state.
            'sid': 'no',
            'blend-subtitles': 'video',
            'gpu-shader-cache': 'yes',
        }

        # A core detached by Stop may still be cooling toward its deferred
        # terminate. Two cores must never overlap on the same wid HWND —
        # finish the old release NOW (cheap once settled) before creating.
        self._drain_dying_core()

        try:
            if MPV_MODULE is None:
                raise RuntimeError("python-mpv support has not been configured")
            self.player = MPV_MODULE.MPV(**mpv_config)
            self.player['msg-level'] = 'all=info'
            logger.info("MPV instance created successfully.")
            self._vu_timer.start()   # begin polling audio levels for the VU meter

            # FIX: Delay property observers to let MPV event thread fully initialize
            # This prevents the "Windows fatal exception: code 0xe24c4a02" error
            core = self.player

            def _setup_observers(owner=core):
                try:
                    # Check that this delayed setup still belongs to the created core.
                    if owner is self.player:
                        # VU meter (observer→cache, the RIGHT way): attach the astats
                        # filter (label 'vu') ONCE and observe af-metadata/vu. The
                        # observer (on_vu_metadata) runs on mpv's EVENT thread and
                        # pushes parsed levels into _vu_cache; the GUI-thread VU poll
                        # only reads that plain attribute, never mpv. One persistent
                        # mpv instance → the filter covers EVERY session incl.
                        # audio-only mpv (MVC/HEVC/dual). Verified: the observer fires
                        # ~10 Hz with real per-channel RMS/peak on a vid=no instance.
                        self._ensure_vu_af()
                        owner.observe_property('af-metadata/vu', self.on_vu_metadata)
                        self.controls_overlay.time_slider.set_player(owner)
                        self._mpv_subtext_observer_registered = False
                        self._mpv_subtext_observer = None
                        logger.info("[MPV] Persistent core observers connected.")
                except Exception as e:
                    logger.warning(f"[MPV] Could not set up observers (safe to ignore): {e}")

            QTimer.singleShot(100, _setup_observers)  # 100ms delay
        except Exception as e:
            logger.error(f"Error initializing mpv or observers: {e}")
            # Never leave a partially configured core behind: the duplicate-core
            # guard would otherwise preserve it forever without observers.
            failed_core = getattr(self, 'player', None)
            self.player = None
            if failed_core is not None:
                try:
                    failed_core.terminate()
                except Exception:
                    pass
            if not getattr(self, '_app_closing', False):
                QMessageBox.critical(self, "MPV Error",
                                     f"Error initializing mpv: {e}\n\nMake sure runtime/mpv-2.dll is installed.")

    def on_end_of_file(self, _, reached, session_id=None, core=None):
        """mpv-thread callback: marshal EOF with immutable ownership metadata."""
        owner = self._media_session_id if session_id is None else session_id
        source = self.player if core is None else core
        self.mpv_eof_event.emit((owner, source, bool(reached)))

    @Slot(object)
    def _dispatch_mpv_eof(self, payload):
        session_id, core, reached = payload
        if not self._session_is_current(session_id, core=core):
            return
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        # Ignore once the MVC decoder's own EOS has already started teardown
        # (_on_mvc_finished sets _playback_ended=True). Without this, MPV's
        # delayed EOF event re-triggers stop_playback → _stop_mvc_decoder
        # a second time, leading to the double-cleanup pattern in the log.
        if getattr(self, '_playback_ended', False):
            return
        if reached:
            self.stop_playback()


__all__ = [
    'MediaSessionMixin', 'configure_media_session_support',
    '_MPV_RELEASE_SETTLE_MS',
]

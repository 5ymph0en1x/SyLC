# -*- coding: utf-8 -*-
"""Native MVC/HEVC decoder lifecycle and stereo frame presentation."""

import faulthandler
import gc
import logging
import os
import time
import traceback

import numpy as np
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QImage

from sylc.native_playback_policy import (
    _edge264_startup_timeout_ms, _recommended_edge264_threads,
    _select_stereo_presentation_targets,
)
from sylc.stereo_eye_order import UNKNOWN


logger = logging.getLogger(__name__)

MVC_SUPPORT_AVAILABLE = False
NATIVE_RENDER_AVAILABLE = False
MVCDecoderThread = None
Framepacking3DWindow = None
EDGE264_CONTAINERS = ()

_EYE_INHERITED_RENDER_PARAMS = (
    'plane_scale', 'source_aspect', 'yuv_matrix_sel', 'transfer_sel',
    'synth3d_enabled', 'synth3d_strength', 'synth3d_convergence',
    'synth3d_depth_view', 'synth3d_diagnostics', 'synth3d_model_path',
    'synth3d_ort_dir', 'synth3d_side', 'synth3d_grid_width',
    'synth3d_grid_height', 'synth3d_crop_top', 'synth3d_crop_bottom',
)


def configure_native_decoder_support(
        mvc_available, native_render_available, mvc_decoder_class,
        framepacking_window_class, edge264_containers):
    """Inject optional native components discovered by the application boot."""
    global MVC_SUPPORT_AVAILABLE, NATIVE_RENDER_AVAILABLE
    global MVCDecoderThread, Framepacking3DWindow, EDGE264_CONTAINERS
    MVC_SUPPORT_AVAILABLE = bool(mvc_available)
    NATIVE_RENDER_AVAILABLE = bool(native_render_available)
    MVCDecoderThread = mvc_decoder_class
    Framepacking3DWindow = framepacking_window_class
    EDGE264_CONTAINERS = tuple(edge264_containers or ())


class NativeDecoderMixin:
    def configure_3d_output(self, enable_3d=True, stereo_mode='auto'):
        """Configures the 3D output of mpv."""
        # HEVC path: frames already flow to every visible native widget via
        # _on_mvc_frame_yuv_ready. 3D = show the framepack window (L+R stacked); 2D =
        # hide it and keep the base view in the embedded widget. mpv stays audio-only —
        # the native SBS/TAB branch below would (wrongly) restore its non-existent video.
        # M3: this MUST run BEFORE the `if not self.player` bail-out below — the HEVC path
        # owns its own native widgets, so the 3D toggle has to work even if mpv has died.
        if getattr(self, '_hevc_mode_active', False):
            self._configure_3d_output_hevc(enable_3d, stereo_mode)
            return

        # Fix round 1 (bidirectional): enforce the `eye_windows is non-None iff
        # current_stereo_mode == 'dual'` invariant on EVERY call, in both directions --
        # closing the pair when 3D goes off (the 3D button bypasses the presentation
        # combo entirely) and REOPENING it when 3D comes back on while the combo still
        # reads 'dual' (a one-directional close-only helper left the combo showing
        # "Dual Projector" while the screen silently fell back to MultiView). Placed
        # BEFORE the `if not self.player` guard below on purpose: that guard is itself a
        # second way to leak the pair open (3D-off returning early because mpv already
        # died during a dual session), and this call touches no mpv state, so running it
        # first closes that hole too. _set_dual_projector_enabled is idempotent in both
        # directions, so calling it unconditionally here is safe.
        self._set_dual_projector_enabled(enable_3d and stereo_mode == 'dual')

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
                
                # Restore 2D navigation bar UI - AFTER stack switch and OS compositor updates
                QTimer.singleShot(250, lambda: (self._update_overlays_geometry(), self.show_controls()))
                return

            self._stop_mvc_decoder()
            try:
                self.player['lavfi-complex'] = ''
                self.player['video'] = 'auto'
                self.video_stack.setCurrentWidget(self.video_widget)
            except:
                pass
                
            # Restore 2D navigation bar UI - AFTER stack switch and OS compositor updates
            QTimer.singleShot(250, lambda: (self._update_overlays_geometry(), self.show_controls()))
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
                            packed_input or
                            getattr(self, '_synth3d_active', False)))

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
                # 'dual' routes into the SAME branch as 'mvc' (fix round 1, Finding 3):
                # Dual Projector needs the identical decoder configuration as MultiView --
                # the decoder must deliver an L/R pair to the routing dispatch in
                # _on_mvc_frame_yuv_ready. Only the PRESENTATION differs, and that is
                # already handled separately by eye_windows (which _select_stereo_
                # presentation_targets prioritises over the framepack window) plus the
                # _show_framepacking_output guard, so no duplicate branch is needed here.
                if stereo_mode in ('mvc', 'dual', 'glasses'):
                    # --- Detached 3D FramePack Mode ---
                    # 'glasses' joins 'mvc'/'dual' here (not the SBS/TAB branch below) for
                    # the same reason as the HEVC path's equivalent branch: it needs the
                    # DETACHED window, sized to F-SBS, not the main window at desktop size.
                    if hasattr(self, 'mvc_embedded_widget') and self.framepacking_window:
                        # V7b SYNC: Embedded stays in 2D (left eye), Framepack window in framepack mode
                        # Both receive same frames for timing sync, but render differently
                        self.mvc_embedded_widget.set_stereo_mode('2d')
                        self.framepacking_window.apply_output_geometry(stereo_mode)
                        self._apply_framepack_source_aspect(self.framepacking_window, stereo_mode)
                        self.framepacking_window.display_widget.set_stereo_mode(
                            'glasses' if stereo_mode == 'glasses' else 'framepack')

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

                        self._show_framepacking_output()
                        if stereo_mode == 'glasses':
                            self.show_3d_notification("3D Mode: Glasses F-SBS", success=True)

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
            elif stereo_mode == 'glasses':
                # Glasses needs the native decoder's detached F-SBS window, same as
                # Dual Projector needs it for the eye pair below -- mpv has no
                # equivalent output and cannot be steered into producing one.
                # (fix round 1, Minor 5: this used to fall through silently.)
                self.show_3d_notification(
                    "Glasses needs the native decoder — not available for this "
                    "source", success=False)
                self._fallback_to_mpv_mvc()
            else:
                # Dual Projector is fed exclusively by the native decoder's frame
                # dispatch (_on_mvc_frame_yuv_ready). mpv renders into its own
                # window and cannot feed the eye windows at all, so a source that
                # lands here would leave two permanently black rectangles on the
                # projectors with nothing in the log to explain them. Close the
                # pair and say why. The invariant enforcement at the top of this
                # function may reopen it on the next call, but that call ends here
                # again, so the settled state is always "closed" for such a source.
                if getattr(self, 'eye_windows', None):
                    logger.warning(
                        "[DUAL-PROJECTOR] this source plays through mpv, which cannot "
                        "feed the eye windows - closing them")
                    self._set_dual_projector_enabled(False)
                    self.show_3d_notification(
                        "Dual Projector needs the native decoder — not available "
                        "for this source", success=False)
                self._fallback_to_mpv_mvc()

    def _make_display_widget(self):
        """Create the video display widget — the native C++ D3D11 renderer, the SOLE
        render path since the Directive 2 cutover (Qt RHI removed). It self-queries
        the display SDR white level for HDR. If it can't be created, the caller's
        try/except degrades to mpv (#388)."""
        from sylc.native_renderer.native_framepack_widget import NativeFramepackWidget
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
        if (getattr(self, '_mvc_shutdown_blocked', False)
                or getattr(self, '_hevc_shutdown_blocked', False)):
            logger.error(
                "[MVC INIT] Previous native decoder still owns resources; refusing "
                "to start a concurrent edge264 instance and falling back to mpv."
            )
            self._mvc_restarting = False
            self._fallback_to_mpv_mvc()
            return

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
                self.framepacking_window.geometryChanged.connect(self._on_framepacking_geometry_changed)

            # 3. Initial State: prepare the native widget, but keep MPV visible
            # until edge264 has produced a validated first frame. This makes the
            # renderer handoff transactional: a failed probe never creates a
            # black-screen gap and MPV remains an immediately usable fallback.
            if self.mvc_embedded_widget.parent() != self.video_stack_container:
                # If it was detached, bring it back
                self.video_stack.addWidget(self.mvc_embedded_widget)

            try:
                self._edge264_pre_handoff_widget = self.video_stack.currentWidget()
            except Exception:
                self._edge264_pre_handoff_widget = getattr(self, 'video_widget', None)
            self.mvc_embedded_widget.set_stereo_mode('2d')
            self.active_mvc_widget = self.mvc_embedded_widget
            self._edge264_waiting_for_first_frame = True
            self._edge264_mpv_handoff_done = False

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
                threads=_recommended_edge264_threads(),
                media_duration=(self.player.duration or self.video_3d_info.get('duration') if self.video_3d_info else None),
                feature_segments=getattr(self, '_pending_feature_segments', None),
                # DF-4: dual-file BD3D pair -> MVCSSIFDemuxer.open_dual(base, dep).
                # Gated on _bd_dual_active so SSIF discs / 2D files are unaffected.
                # Persists across seek/crash-restart re-inits (correct: each re-open
                # must re-establish the dual demuxer).
                dual_pair=(getattr(self, '_bd_dual_file_pair', None)
                           if getattr(self, '_bd_dual_active', False) else None)
            )
            self.mvc_decoder_thread._sylc_session_id = self._media_session_id
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
            # Timed delivery carries the exact media PTS with the pixels. The
            # legacy two-argument signal remains available to external callers,
            # but using it here would force Synth3D to confuse GUI/compute time
            # with video time.
            self.mvc_decoder_thread.frameYUVTimedReady.connect(
                self._on_mvc_frame_yuv_timed_ready, Qt.QueuedConnection)
            logger.info("[MVC INIT] frameYUVTimedReady connected with Qt.QueuedConnection")
            # Export MV (04/08) : indices de mouvement du décodeur vers le
            # service de profondeur (3e candidat de la fusion). getattr :
            # threads/fakes sans ce signal restent valides.
            _mh = getattr(self.mvc_decoder_thread, 'motionHintsReady', None)
            if _mh is not None:
                _mh.connect(self._forward_motion_hints, Qt.QueuedConnection)
            _cut = getattr(self.mvc_decoder_thread, 'lookaheadCutReady', None)
            if _cut is not None:
                _cut.connect(
                    self._forward_lookahead_cut_boundary,
                    Qt.QueuedConnection)
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
            _session = self._media_session_id
            self._media_single_shot(
                100,
                lambda: self._delayed_start_decoder(disable_mpv=False),
                _session)

            # SYNC TIMER: Periodically push audio clock to decoder thread
            self._sync_timer = QTimer(self)
            self._sync_timer.setInterval(50)  # 20Hz update rate
            self._sync_timer.timeout.connect(self._update_decoder_audio_clock)
            # V9 SSIF FIX: Start sync timer immediately instead of 1000ms delay
            # The decoder needs audio clock during init for SSIF sync
            # Use a short delay (100ms) just to let MPV initialize
            self._media_single_shot(100, self._sync_timer.start, _session)

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

            self.show_3d_notification("Initializing edge264 video...", success=True)
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
        if not self._native_signal_is_current():
            return
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
        self._media_single_shot(
            500,
            lambda: self._start_mvc_decoder(start_time=resume_time),
            self._media_session_id)

    def _fallback_to_mpv_mvc(self):
        """Fallback to mpv native MVC handling"""
        # F1a: this is a genuine loss of the native decode path (mpv takes over
        # MVC presentation) -- synthesis cannot keep running, same as the
        # _stop_*_decoder call sites' guard just below.
        if hasattr(self, '_synth3d_on_native_path_lost'):
            self._synth3d_on_native_path_lost()
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

    def _commit_edge264_video_handoff(self):
        """Atomically replace MPV video only after edge264 delivered a real frame."""
        if not getattr(self, '_edge264_waiting_for_first_frame', False):
            return True
        if not self.mvc_mode_active or not self.mvc_decoder_thread:
            return False

        timer = getattr(self, '_edge264_startup_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()

        # Release MPV's D3D11 chain immediately before the first native upload.
        # Until this point MPV remained visible and fully recoverable.
        if not self._disable_mpv_video_output():
            self._edge264_waiting_for_first_frame = False
            self._fallback_from_edge264(
                reason="could not release MPV video output for the edge264 renderer")
            return False

        try:
            self.video_stack.setCurrentWidget(self.mvc_embedded_widget)
        except Exception as e:
            self._edge264_waiting_for_first_frame = False
            self._fallback_from_edge264(reason=f"native video handoff failed: {e}")
            return False

        self._edge264_waiting_for_first_frame = False
        self._edge264_mpv_handoff_done = True
        self._edge264_pre_handoff_widget = None
        logger.info("[EDGE264-HANDOFF] First decoded frame validated; MPV -> native D3D11 committed")
        if isinstance(self.video_3d_info, dict) and self.video_3d_info.get('is_3d'):
            self.show_3d_notification(
                "Edge264 MultiView decoder active", success=True, permanent=True)
        else:
            self.show_3d_notification(
                "Edge264 H.264 decoder active", success=True, permanent=True)
        return True

    def _on_edge264_startup_timeout(self):
        """Fail open to MPV when edge264 cannot produce its first frame in time."""
        if (not getattr(self, '_edge264_waiting_for_first_frame', False)
                or not self.mvc_mode_active):
            return
        timeout_ms = _edge264_startup_timeout_ms(self.video_3d_info)
        logger.error(
            f"[EDGE264-STARTUP] No valid frame within {timeout_ms / 1000:.0f}s; "
            "restoring MPV video")
        self._edge264_waiting_for_first_frame = False
        self._fallback_from_edge264(
            reason=f"no decoded frame within {timeout_ms / 1000:.0f}s")

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
        timer = getattr(self, '_edge264_startup_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._edge264_waiting_for_first_frame = False
        self._edge264_mpv_handoff_done = False
        self._edge264_pre_handoff_widget = None
        # Review fix #2: capture BEFORE any teardown/state change below, mirroring
        # _on_hevc_failed's `was_multiview` gate (~L6785, same idea, edge264 side). A
        # genuinely-3D session (MVC/SBS/TAB via edge264) that silently degrades to mpv's
        # 2D-only video previously left video_3d_info['is_3d']/['stereo_mode'] at their 3D
        # values and never called _update_3d_button_state() -> the 3D button/dropdown stayed
        # stale-enabled on a now-2D-only session.
        was_3d = bool(isinstance(self.video_3d_info, dict) and self.video_3d_info.get('is_3d'))
        self.mvc_mode_active = False
        self._synth3d_on_native_path_lost()
        # DF-FINAL FIX 2: clear the dual-file BD3D session flags too, so a fallen-back
        # pipeline can't be mistaken for a live dual-file session by later code
        # (e.g. a queued decoder-seek or a subsequent load-state check).
        self._bd_dual_active = False
        self._bd_dual_file_pair = None
        self._bd_eye_order = UNKNOWN
        try:
            self.controls_overlay.clear_format_badge()  # edge264 didn't adapt → drop the badge
        except Exception:
            pass
        try:
            self._stop_mvc_decoder()
        except Exception:
            pass
        if was_3d:
            try:
                if isinstance(self.video_3d_info, dict):
                    self.video_3d_info['stereo_mode'] = 'none'
                    self.video_3d_info['is_3d'] = False
            except Exception:
                pass
        try:
            self._update_3d_button_state()
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
        self._media_single_shot(
            700, self._confirm_mpv_fallback_video, self._media_session_id)

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
            self._media_single_shot(
                700,
                lambda: self._confirm_mpv_fallback_video(attempt + 1),
                self._media_session_id)
        else:
            self.show_3d_notification(
                "edge264 failed and mpv has no video for this source — audio only.",
                success=False)

    def _mpv_time_pos_ms(self):
        """Master clock for the HEVC decode thread: mpv's audio time-pos in ms, read
        FROM A CACHED value only (populated by the existing on_time_update observer). A
        blocking mpv property read on this hot cross-thread path is exactly the
        0xe24c4a02 FSBS-crash pattern (project memory) — never do it here. Samples are
        extrapolated at 1x while playing; None keeps an audio-backed decoder behind its
        startup barrier, while a proven video-only session uses PTS/wall-clock pacing."""
        pos = getattr(self, '_mpv_time_pos_cache', None)
        if pos is None:
            return None
        stamp = getattr(self, '_mpv_time_pos_cache_mono', None)
        if stamp is not None and not getattr(self, '_mpv_pause_cache', True):
            pos = float(pos) + max(0.0, time.monotonic() - float(stamp))
        return float(pos) * 1000.0

    def _arm_hevc_audio_start(self, target_s=0.0):
        """Request a clocked HEVC start without trusting one mpv observer.

        Normally the first advancing time-pos releases the decoder.  If Python
        never receives that callback, a GUI-thread timer reconstructs where the
        1x audio clock should be and releases it anyway.  The captured thread
        identity prevents an old timer from waking a later media session.
        """
        th = getattr(self, 'hevc_thread', None)
        if (th is None or not getattr(self, '_hevc_mode_active', False)
                or not getattr(self, '_hevc_clocked', False)):
            return
        request = (float(target_s), time.monotonic(), th)
        self._hevc_start_request = request
        self._media_single_shot(
            350,
            lambda owner=th: self._release_hevc_audio_start(
                observed_s=None, expected_thread=owner))

    def _release_hevc_audio_start(self, observed_s=None, expected_thread=None):
        """Idempotently release the startup barrier from audio evidence/fallback."""
        request = getattr(self, '_hevc_start_request', None)
        if request is None:
            return False
        target_s, requested_at, owner = request
        current = getattr(self, 'hevc_thread', None)
        if (current is not owner
                or (expected_thread is not None and owner is not expected_thread)):
            return False
        now = time.monotonic()
        if observed_s is None:
            # No callback: audio was commanded to start at requested_at, so its
            # best non-blocking estimate is target + elapsed wall time.
            clock_s = target_s + max(0.0, now - requested_at)
            source = 'timeout'
        else:
            clock_s = float(observed_s)
            source = 'time-pos'
        self._prime_mpv_time_pos(clock_s)
        self._mpv_pause_cache = False
        self._hevc_start_request = None
        owner.set_paused(False)
        logger.info("[HEVC-SYNC] startup barrier released by %s at %.3fs",
                    source, clock_s)
        return True

    def _try_start_hevc(self, file_path):
        """Probe + start the HEVC path. Returns True when it took the file (the caller
        must NOT fall back to mpv), False to let the normal mpv 2D path play it.

        Contract: LavfHevcSource().open() returns MediaInfo for an in-scope HEVC stream
        or None (H.264 / 12-bit / 4:2:2 …). On None → close + False. On MediaInfo → start
        the HEVC pipeline; ANY startup exception → clean teardown → False (the plan's
        "jamais de crash": mpv, which .play()s this same file right after, takes over)."""
        try:
            from sylc import lavf_hevc_source
        except Exception as e:
            logger.info(f"[HEVC] module unavailable: {e}")
            return False
        if not lavf_hevc_source.is_available():
            return False
        if self._reap_hevc_leaked():
            logger.error(
                "[HEVC] A previous native decoder is still blocked; refusing "
                "to stack another native owner and falling back to mpv")
            return False
        src = lavf_hevc_source.LavfHevcSource()
        try:
            mi = src.open(file_path, allow_hw=True)
        except Exception as e:
            logger.info(f"[HEVC] open raised: {e}")
            try:
                src.close()
            except Exception:
                pass
            return False
        if mi is None:
            # Not HEVC / out of perimeter — open() already logged the refusal and closed.
            try:
                src.close()
            except Exception:
                pass
            return False

        # Genuine in-scope HEVC stream from here on.
        try:
            from sylc.hevc_stereo_detect import detect
            from sylc.hevc_decode_thread import HevcDecodeThread
            from PySide6.QtCore import Qt

            # MV-4: MV-HEVC multiview (spec §5) PRIME sur toute autre detection. Quand la
            # source a ete re-ouverte en multiview (mi.multiview), les DEUX vues decodees
            # SONT la paire stereo -> mode fige a 'mvhevc', jamais de split (half=False),
            # oeil gauche deja assigne COTE SOURCE via left_view_id ; l'inversion eventuelle
            # vient de la side-data Stereo3D (mi.stereo_inverted, la meme valeur "mi-based"
            # que detect() applique au chemin mono-vue), la plomberie swap_eyes de configure()
            # restant applicable par-dessus. On court-circuite detect()/analyzer-fallback/
            # UI-override ET tout le traitement 2D (le fichier est promu 3D comme un MVC).
            if getattr(mi, 'multiview', False):
                mode, half, inverted = 'mvhevc', False, bool(mi.stereo_inverted)
            else:
                mode, half, inverted = detect(file_path, mi)
                # M1: the HEVC stereo detector (SEI / container side-data) can return mode=None
                # even when the upstream ffprobe analysis already classified the layout (e.g. a
                # filename/container tag hint). Fall back to that analyzer verdict so an SBS/TAB
                # HEVC clip the detector missed still plays stereo. (UI override still wins below.)
                if mode is None and isinstance(self.video_3d_info, dict):
                    _an = (self.video_3d_info.get('stereo_mode') or '').lower()
                    if _an in ('sbs', 'tab'):
                        logger.info(f"[HEVC] mode from analyzer video_3d_info: {_an} "
                                    f"(detector returned None)")
                        mode = _an
                # UI override: an explicit SBS/TAB pick in the stereo combo wins over
                # auto-detection (same precedence as configure_3d_output's 'auto' resolve).
                ui_mode = getattr(self, 'current_stereo_mode', 'auto')
                if ui_mode in ('sbs', 'tab') and ui_mode != mode:
                    logger.info(f"[HEVC] UI stereo override: {ui_mode} (detected {mode})")
                    mode = ui_mode

            # DIRECTIVE (native = avcodec, mpv = secours): le chemin avcodec-62 +
            # renderer natif D3D11 est la lecture NATIVE de TOUT le HEVC, 2D inclus —
            # son image est meilleure que celle de mpv; mpv n'est que le chemin de
            # SECOURS (sur echec). On ne bifurque donc PAS vers mpv sur un flux 2D:
            # mode=None poursuit le chemin HEVC et le thread duplique la vue (L=R),
            # comme a l'origine. Tout repli sur echec (open/startup/decode) reste
            # intact plus bas et rend la main a mpv.

            # Refresh the hover-thumbnail single-eye crop with the DEFINITIVE HEVC
            # stereo layout: the SEI/side-data detector above can find sbs/tab that
            # the ffprobe analyzer (already consumed by _apply_preview_thumbs_policy)
            # missed, so the thumb service was configured with the wrong/absent crop.
            # A packed-3D thumbnail must show a SINGLE eye.
            if getattr(self, '_thumb_service', None) is not None:
                try:
                    self._thumb_service.set_layout(mode, half=bool(half))
                except Exception:
                    pass

            # Widgets: reuse the MVC embedded widget + detached framepack window (the sole
            # native D3D11 render path). Created on first use, like _start_mvc_decoder.
            if not hasattr(self, 'mvc_embedded_widget') or self.mvc_embedded_widget is None:
                self.mvc_embedded_widget = self._make_display_widget()
            if self.video_stack.indexOf(self.mvc_embedded_widget) == -1:
                self.video_stack.addWidget(self.mvc_embedded_widget)
            if not self.framepacking_window:
                _dw = self._make_display_widget()
                _dw.set_stereo_mode('framepack')
                # M5: mirror the MVC site's GPU-YUV flag (name it, don't hardcode True) so
                # both framepack windows are constructed from the same source of truth.
                USE_GPU_YUV_CONVERSION = True
                self.framepacking_window = Framepacking3DWindow(
                    parent=None, use_yuv_shader=USE_GPU_YUV_CONVERSION, display_widget=_dw)
                self.framepacking_window.visibilityChanged.connect(
                    self._on_framepacking_visibility_changed)
                self.framepacking_window.geometryChanged.connect(
                    self._on_framepacking_geometry_changed)

            # 10-bit frames arrive as uint16 planes → the widget's R16 path needs a
            # plane_scale; HW D3D11VA copy-back (P010) is MSB-aligned (65535/65472),
            # SW yuv420p10le is LSB-aligned (65535/1023). 8-bit uint8 ignores it (1.0).
            # Reset on teardown.
            if mi.bit_depth == 10:
                _scale = (65535.0 / 65472.0) if src.hw_active() else (65535.0 / 1023.0)
            else:
                _scale = 1.0
            for _w in self._display_widgets():
                try:
                    _w.plane_scale = _scale
                except Exception:
                    pass

            # C2: display-aspect override for HALF formats. For half-SBS/half-TAB the packed
            # frame keeps the ORIGINAL 2D dimensions, so each squeezed eye (e.g. 960x1080)
            # must still display at the packed frame's own W/H aspect — frame W/H IS the
            # correct per-eye display aspect. Full formats have correct eye dims already
            # (0.0 = derive from planes). Stored MediaInfo/half feed I3's live mode switch.
            self.hevc_media_info = mi
            self._hevc_half = bool(half)
            _src_aspect = (float(mi.width) / float(mi.height)) if half else 0.0
            for _w in self._display_widgets():
                try:
                    _w.source_aspect = _src_aspect
                except Exception:
                    pass
            if half:
                logger.info(f"[HEVC] source_aspect={_src_aspect:.3f} (half format "
                            f"{mi.width}x{mi.height})")

            # HDR10/PQ color path: map the resolved color metadata (lavf_hevc_source) to
            # the widgets' shader selectors. matrix_sel is display-independent; transfer_sel
            # depends on each widget's HDR-display flag (_hdr, decided at widget construction).
            #   yuv_matrix_sel: bt2020nc/bt2020c -> 2; bt709 (or the HEVC width-heuristic
            #     '709') -> 1; else legacy BT.601 (0).
            #   transfer_sel: smpte2084 -> 1 (HDR display) / 2 (SDR fallback); HLG -> 0 (legacy,
            #     logged); else 0. 0/0 keeps the legacy render (byte-identical).
            _cs = (mi.color_space or '').lower()
            if _cs in ('bt2020nc', 'bt2020c'):
                _matrix_sel = 2
            elif _cs in ('bt709', '709'):
                _matrix_sel = 1
            else:
                _matrix_sel = 0
            _trc = (mi.color_trc or '').lower()
            if _trc in ('arib-std-b67', 'hlg'):
                logger.info('[HEVC] HLG non gere (legacy render)')
            # HDR cast eligibility: a 10-bit PQ source can be cast as HEVC
            # Main10/P010 with its PQ signalling intact (_on_cast_requested).
            self._hevc_pq10 = bool(mi.bit_depth == 10 and _trc == 'smpte2084')
            for _w in self._display_widgets():
                try:
                    _w.yuv_matrix_sel = _matrix_sel
                    if _trc == 'smpte2084':
                        _w.transfer_sel = 1 if getattr(_w, '_hdr', False) else 2
                    else:
                        _w.transfer_sel = 0
                except Exception:
                    pass
            # Summary transfer_sel for the startup log (embedded widget's HDR decision).
            _sel_t = ((1 if getattr(self.mvc_embedded_widget, '_hdr', False) else 2)
                      if _trc == 'smpte2084' else 0)

            # Initial on-screen state: embedded widget shows the base (left/top) eye in
            # '2d'; the framepack window stacks L+R when the user toggles 3D. Un-pause the
            # reused widgets (V57 black-screen-on-reload fix) so new frames are painted.
            self.video_stack.setCurrentWidget(self.mvc_embedded_widget)
            self.mvc_embedded_widget.set_stereo_mode('2d')
            self.active_mvc_widget = self.mvc_embedded_widget
            self.framepacking_window.display_widget.set_stereo_mode(
                'framepack' if mode in ('sbs', 'tab', 'mvhevc') else '2d')
            for _w in self._display_widgets():
                try:
                    if hasattr(_w, 'resume_rendering'):
                        _w.resume_rendering()
                except Exception:
                    pass

            # mpv audio-only, exactly like MVC (vid=no, vo=null). mpv .play()s this same
            # file right after analyze_and_configure_3d returns → it decodes only audio.
            self._disable_mpv_video_output()

            # Fix-1 (MV-5 final review, defense in depth): flip this BEFORE the
            # promotion/UI block below (was previously set just before hevc_thread.start(),
            # ~10 lines later) so that if the blockSignals guard above is ever bypassed
            # (e.g. a future direct emit of stereo_mode_changed), any reentrant
            # configure_3d_output/_on_hevc_failed sees the HEVC path as already active
            # instead of False. Traced: nothing between here and the old position reads
            # _hevc_mode_active expecting False — the only consumers reachable in this
            # span are _update_3d_button_state()/_format_badge_label() (never touch it)
            # and the (now signal-blocked) combo setCurrentIndex; hevc_thread/hevc_source
            # themselves aren't constructed until after this block, so a reentrant
            # configure_3d_output(...) -> _configure_3d_output_hevc(...) here would only
            # show/hide the framepack window early (idempotent, harmless) rather than the
            # pre-fix hazard of launching _start_mvc_decoder().
            self._hevc_mode_active = True

            # Reflect the stereo nature in the UI (combo / badge / 3D button) when a stereo
            # layout was detected, like the MVC/FSBS path in analyze_and_configure_3d.
            # MV-4: MV-HEVC is promoted 3D EXACTLY like an MVC disc — combo defaults to
            # index 0 (MVC = framepack presentation), video_3d_info marked 'mvc'+is_3d so
            # the 3D button arms and the badge reads "MVC 3D". The two views ARE the stereo
            # pair; the combo switches the RENDERER presentation only (change_stereo_mode
            # skips hevc_thread.set_mode in mvhevc). SBS/TAB keep their own combo index and
            # stereo_mode verbatim (packed-3D path byte-identical).
            #
            # Review fix #1a: stash the mode _try_start_hevc actually resolved (mvhevc /
            # sbs / tab / None for plain 2D HEVC) so _configure_3d_output_hevc can later
            # resolve a literal 'auto' stereo_mode to the REAL session layout instead of
            # testing the string 'auto' against ('sbs','tab') and silently falling into the
            # framepack branch (the pre-D2 bug, reachable again via 'auto' since that
            # function's own dispatcher in configure_3d_output() returns BEFORE the 'auto'
            # resolution at L5908-5913 — see _configure_3d_output_hevc for the consumer
            # side). Set unconditionally (covers 2D HEVC too, mode=None); harmless there
            # since the 3D toggle is never reachable while is_3d stays False.
            self._hevc_detected_mode = mode
            _promote = None
            if isinstance(self.video_3d_info, dict):
                if mode == 'mvhevc':
                    _promote = ('mvc', 0)
                elif mode in ('sbs', 'tab'):
                    _promote = (mode, 1 if mode == 'sbs' else 2)
            if _promote is not None:
                _sm_ui, _combo_ix = _promote
                self.video_3d_info['stereo_mode'] = _sm_ui
                self.video_3d_info['is_3d'] = True
                # Review fix #1b: latch current_stereo_mode to the same value the combo is
                # about to show (below), by direct attribute assignment (no signal fire, so
                # it can't re-enter change_stereo_mode/configure_3d_output mid-startup —
                # same rationale as blockSignals just below). Without this,
                # current_stereo_mode is left at whatever _continue_play_file's per-file
                # reset set it to ('auto', L5449) — order proof: that reset runs earlier in
                # THIS SAME synchronous load chain (_continue_play_file L5449 ->
                # _configure_and_start_playback L5585 analyze_and_configure_3d ->
                # _try_start_hevc, all direct calls, no event-loop turn in between), so this
                # assignment is never clobbered by it for the file being promoted right now.
                # The user's first 3D-button click then calls
                # configure_3d_output(True, self.current_stereo_mode) with an attribute that
                # already agrees with the UI/video_3d_info, instead of a stale 'auto'.
                self.current_stereo_mode = _sm_ui
                # Fix round 1, Finding 2: this direct assignment bypasses
                # change_stereo_mode (deliberately -- see the blockSignals below), so it
                # must carry its own Dual Projector teardown. _sm_ui is always 'mvc'/
                # 'sbs'/'tab', never 'dual' -- if eye_windows survived from a PREVIOUS
                # file's Dual Projector session, this promotion would otherwise leave it
                # open while current_stereo_mode disagrees with it.
                self._set_dual_projector_enabled(False)
                # Fix-1 (MV-5 final review): setCurrentIndex() fires currentTextChanged
                # SYNCHRONOUSLY (same-thread direct connection) -> _on_stereo_mode_changed
                # -> change_stereo_mode('mvc') re-enters mid-startup, before hevc_thread
                # even exists. With a prior 3D session's has_media/is_3d_enabled still True
                # and video_3d_info['stereo_mode'] just set to 'mvc' two lines above,
                # change_stereo_mode's tail (`if self.has_media and self.is_3d_enabled:
                # configure_3d_output(True, 'mvc')`) resolves use_mvc_decoder=True and can
                # call _start_mvc_decoder() on the MV-HEVC file. blockSignals suppresses the
                # re-entrant call entirely (same pattern as the 3D button at ~3894/~7643).
                try:
                    combo = self.controls_overlay.stereo_mode_combo
                    combo.blockSignals(True)
                    try:
                        combo.setCurrentIndex(_combo_ix)
                    finally:
                        combo.blockSignals(False)
                except Exception:
                    pass
                try:
                    self.controls_overlay.set_format_badge(self._format_badge_label())
                except Exception:
                    pass
                self._update_3d_button_state()

            # I2: clear any stale mpv time-pos cache left over from the previous file /
            # session BEFORE the thread starts.  Audio-backed HEVC starts PAUSED and
            # consumes no frame while this cache is None; video-only HEVC retains its
            # proven PTS/wall-clock pacing.  A stale prior-session value can therefore
            # neither release the startup barrier nor trigger a false catch-up burst.
            self._prime_mpv_time_pos(None)

            # Decode thread: same frame contract as MVC → the SAME slot. Master clock =
            # mpv audio time-pos FROM CACHE ONLY (never a blocking read; crash 0xe24c4a02).
            self.hevc_source = src
            self.hevc_thread = HevcDecodeThread()
            self.hevc_thread._sylc_session_id = self._media_session_id
            _has_audio = (self.video_3d_info or {}).get('has_audio')
            _clocked = _has_audio is not False
            self._hevc_clocked = _clocked
            self._hevc_start_request = None
            try:
                _av_trim = float(self._app_settings.get('av_sync_offset_s', 0.0))
            except (TypeError, ValueError):
                _av_trim = 0.0
            self.hevc_thread.configure(
                src, mode=mode, half=half, inverted=inverted,
                start_paused=_clocked,
                require_master_clock=_clocked,
                bounded_delivery=True,
                av_sync_offset_s=_av_trim)
            self.hevc_thread.clock_offset_provider = self._mpv_time_pos_ms
            # HEVC cut identity is emitted before the matching paced frame.
            # Both connections target this PlayerWindow with queued delivery,
            # preserving boundary-before-frame order across the thread hop.
            self.hevc_thread.lookaheadCutReady.connect(
                self._forward_lookahead_cut_boundary, Qt.QueuedConnection)
            self.hevc_thread.frameYUVTimedReady.connect(
                self._on_hevc_frame_yuv_timed_ready, Qt.QueuedConnection)
            self.hevc_thread.decodeFailed.connect(self._on_hevc_failed)
            self.hevc_thread.endOfStream.connect(self._on_mvc_finished)  # shared EOS handler
            # PGS streaming (miroir du chemin MVC) : la source lavf collecte les
            # blocs au fil du demux — plus d'extraction ffmpeg 1-2 min. Le combo
            # se remplit via le MEME handler que MVC ; pgsDataReady est connecte
            # a la selection de piste (change_subtitle_track), comme en MVC.
            self._streaming_subtitle_tracks = []
            self._pgs_streaming_connected = False
            self.hevc_thread.subtitleTracksDetected.connect(
                self._on_subtitle_tracks_detected, Qt.QueuedConnection)
            # Timeline position from the decode thread itself (mirror of the MVC
            # per-frame timestamp slot): for a no-audio file the mpv shell never
            # reports time-pos, so without this the slider has no position source.
            self.hevc_thread.positionChanged.connect(
                self._on_hevc_position, Qt.QueuedConnection)
            self.hevc_thread.set_lookahead_enabled(
                bool(getattr(self, '_synth3d_active', False)))
            self.hevc_thread.start()

            # MV-4/Fix-4 (MV-5 final review): expose the two view_ids + left-eye mapping in
            # the startup log for multiview files (spec §5); mono-view keeps the exact
            # original shape. Fix-4: print the SAME actual probed ids the source re-opened
            # with (src._mv_view_ids, set by LavfHevcSource.open()/_probe_multiview) instead
            # of a hardcoded "0,1" literal. Smallest clean route: the source instance
            # already carries the value it used for its own re-open, so a source attribute
            # was plumbed rather than adding a MediaInfo field just for this log line.
            _mv_ids_str = (getattr(src, '_mv_view_ids', None) or b'0,1').decode('ascii', 'replace')
            _mv_seg = (f"views={_mv_ids_str} left={mi.left_view_id} "
                       if getattr(mi, 'multiview', False) else "")
            logger.info(f"[HEVC] lecture via avcodec: mode={mode} {_mv_seg}half={half} "
                        f"inverted={inverted} {mi.width}x{mi.height} {mi.bit_depth}-bit "
                        f"hw={int(src.hw_active())} matrix={mi.color_space} trc={mi.color_trc} "
                        f"sel={_matrix_sel}/{_sel_t} ({os.path.basename(str(file_path))})")
            return True
        except Exception as e:
            logger.error(f"[HEVC] startup failed → mpv fallback: {e}")
            try:
                self._stop_hevc_decoder()
            except Exception:
                pass
            try:
                src.close()
            except Exception:
                pass
            return False

    def _on_hevc_failed(self, msg):
        """decodeFailed slot: the HEVC decode thread hit an unrecoverable error → degrade
        to mpv (same "edge264 first, mpv only on failure" pattern as _fallback_from_edge264).
        Tear down the HEVC path and hand this file to mpv's own video output — mpv has been
        decoding its audio all along, so its position is already correct (no seek)."""
        if not self._native_signal_is_current():
            return
        logger.warning(f"[HEVC] decode failed → mpv fallback: {msg}")
        # Fix-2 (MV-5 final review): capture BEFORE _stop_hevc_decoder(), which nulls
        # self.hevc_media_info (C2 comment there). A mid-play decodeFailed on an MV-HEVC
        # file had left video_3d_info['stereo_mode']=='mvc'/is_3d=True promoted (Fix in
        # _try_start_hevc / MV-4) and the "MVC 3D" badge up, both now stale — mpv is only
        # falling back to the 2D base view, not a stereo pair.
        was_multiview = bool(getattr(self.hevc_media_info, 'multiview', False))
        try:
            self._stop_hevc_decoder()
        except Exception:
            pass
        if was_multiview:
            try:
                if isinstance(self.video_3d_info, dict):
                    self.video_3d_info['stereo_mode'] = 'none'
                    self.video_3d_info['is_3d'] = False
            except Exception:
                pass
            try:
                self.controls_overlay.clear_format_badge()  # mirror _fallback_from_edge264
            except Exception:
                pass
            try:
                self._update_3d_button_state()
            except Exception:
                pass
        fp = getattr(self, 'framepacking_window', None)
        if fp is not None:
            try:
                if fp.isVisible():
                    fp.hide()
            except Exception:
                pass
        self._present_via_mpv_native()
        self._media_single_shot(
            700, self._confirm_mpv_fallback_video, self._media_session_id)

    def _reap_hevc_leaked(self):
        """Close native HEVC owners once their formerly wedged thread has exited."""
        survivors = []
        for th, src in list(getattr(self, '_hevc_leaked', [])):
            try:
                running = bool(th.isRunning())
            except RuntimeError:
                running = False
            if running:
                survivors.append((th, src))
                continue
            if src is not None:
                try:
                    src.close()
                except Exception:
                    logger.exception("[HEVC] Could not close a reaped leaked source")
            try:
                th.deleteLater()
            except RuntimeError:
                pass
        self._hevc_leaked = survivors
        self._hevc_shutdown_blocked = bool(survivors)
        return survivors

    def _stop_hevc_decoder(self, restarting=False):
        """Symmetric teardown of the HEVC path (mirror of the MVC cleanup): disconnect the
        decode thread's signals (pattern of _stop_mvc_decoder), stop + join it, close the
        avformat/avcodec source, and reset the widgets' plane_scale to 1.0 (8-bit default)
        so a subsequent 8-bit MVC/H.264 file renders correctly. Idempotent / no-op when no
        HEVC session is active, so it is safe to call from every MVC teardown site.

        `restarting`: forwarded to `_synth3d_handle_decoder_stop` (default False — a
        genuine loss). `_stop_mvc_decoder` passes its own `_mvc_restarting` state through
        here, since it calls this unconditionally on every teardown, including a
        same-session restart (edge264 crash recovery, seek-queue restart) -- without
        that, this method's own synth3d notification would fire the false "native
        decoder lost" alarm mid-restart even after `_stop_mvc_decoder`'s tail check is
        fixed to skip it.

        request_stop() only flips a flag checked between frames -- it cannot interrupt a
        blocking read_frame() (GIL released inside av_read_frame/avcodec_receive_frame).
        If the thread is wedged past the wait() timeout, closing hevc_source would free
        _ctx/_dec out from under a live native call -> use-after-free crash. So the source
        is only closed once the thread has actually terminated; otherwise it is
        deliberately leaked (leak-over-UAF, matches the project's "jamais de crash" rule).
        The thread keeps its own reference to the source (HevcDecodeThread._src, set via
        configure()), so it stays alive even though self.hevc_source is nulled below."""
        self._reap_hevc_leaked()
        th = getattr(self, 'hevc_thread', None)
        thread_dead = True
        if th is not None:
            try:
                for _sig in ('frameYUVReady', 'frameYUVTimedReady',
                             'decodeFailed', 'endOfStream',
                             'pgsDataReady', 'subtitleTracksDetected'):
                    try:
                        getattr(th, _sig).disconnect()
                    except Exception:
                        pass
                # Le connect pgsDataReady est par-thread : un nouveau thread doit
                # pouvoir se reconnecter (meme regle que le teardown MVC).
                self._pgs_streaming_connected = False
                th.request_stop()
                if not th.wait(5000):
                    thread_dead = False
                    # I1: dropping the last Python ref to a still-RUNNING QThread makes Qt
                    # qFatal (abort the whole process) when it is later garbage-collected.
                    # Keep the wedged thread AND its source alive in a leak list instead —
                    # leak-over-abort, the same rationale as the leak-over-use-after-free
                    # on the source below.
                    if not hasattr(self, '_hevc_leaked'):
                        self._hevc_leaked = []
                    try:
                        th.setParent(None)
                    except RuntimeError:
                        pass
                    self._hevc_leaked.append((th, getattr(self, 'hevc_source', None)))
                    logger.error("[HEVC] teardown: thread non termine apres 5 s - "
                                 "source NON fermee (fuite volontaire, anti use-after-free)")
            except Exception as e:
                logger.warning(f"[HEVC] thread teardown error: {e}")
                try:
                    thread_dead = not bool(th.isRunning())
                except RuntimeError:
                    thread_dead = True
                if not thread_dead and not any(
                        old_th is th for old_th, _ in self._hevc_leaked):
                    try:
                        th.setParent(None)
                    except RuntimeError:
                        pass
                    self._hevc_leaked.append(
                        (th, getattr(self, 'hevc_source', None)))
            try:
                stats = th.sync_stats()
                logger.info("[HEVC-SYNC] teardown: sync_drops=%d backpressure_drops=%d",
                            stats['sync_drops'], stats['backpressure_drops'])
            except Exception:
                pass
            if thread_dead:
                try:
                    th.deleteLater()
                except RuntimeError:
                    pass
            self.hevc_thread = None
        src = getattr(self, 'hevc_source', None)
        if src is not None:
            if thread_dead:
                try:
                    src.close()
                except Exception:
                    pass
            # sinon: le thread wedge garde sa reference native active; fermer ici serait
            # un use-after-free. self.hevc_source est quand meme mis a None juste en
            # dessous -- la reference du thread (self._src) suffit a garder l'objet vivant.
            self.hevc_source = None
        if getattr(self, '_hevc_mode_active', False):
            logger.info("[HEVC] path torn down")
        self._hevc_mode_active = False
        self._hevc_shutdown_blocked = bool(self._reap_hevc_leaked())
        self._hevc_clocked = False
        self._hevc_start_request = None
        # C2: forget the stored MediaInfo/half so a stale aspect can't leak into the next file.
        self.hevc_media_info = None
        self._hevc_half = False
        # Fix round 2: same reasoning for the Glasses per-eye plane-dims cache --
        # this is the single shared teardown every session end routes through
        # (see _stop_mvc_decoder's docstring), HEVC or plain MVC/edge264 alike,
        # so it is the right place to drop it for both.
        self._glasses_eye_plane_dims = None
        # Reset the 10-bit rescale AND the C2 display-aspect override so the reused widgets
        # render subsequent 8-bit / full-format content correctly.
        for _w in self._display_widgets():
            try:
                _w.plane_scale = 1.0
            except Exception:
                pass
            try:
                # Fix round 3: this reset is what makes a mid-session decoder
                # restart (e.g. edge264 crash recovery) self-healing for
                # Glasses -- it MUST stay paired with the
                # `_glasses_eye_plane_dims = None` reset above (:9994). If a
                # restart resumes the same file at the same resolution, the
                # first post-restart frame's dims equal the OLD cached dims,
                # so _note_decoded_eye_plane sees no change and skips
                # re-applying -- UNLESS the cache was also cleared, which
                # forces that first frame to register as a change regardless.
                # Without both together, this line alone would leave the
                # widget pillarboxed (0.0 = "derive", i.e. single-eye, not
                # doubled) until an explicit mode switch fixes it by hand.
                _w.source_aspect = 0.0
            except Exception:
                pass
            # HDR10/PQ: back to the legacy 0/0 color path so a subsequent 8-bit
            # MVC/H.264/SDR source renders byte-identically.
            try:
                _w.yuv_matrix_sel = 0
                _w.transfer_sel = 0
            except Exception:
                pass
        if hasattr(self, '_synth3d_handle_decoder_stop'):
            self._synth3d_handle_decoder_stop(restarting)

    def _show_framepacking_output(self):
        """Show the detached 3D output without stealing the main UI's focus."""
        fp = getattr(self, 'framepacking_window', None)
        if fp is None:
            return
        # Dual Projector owns the detached output while it is active. The mode
        # switch hides this window, but configure_3d_output runs afterwards and
        # would otherwise put it straight back on top of a projector.
        if getattr(self, 'eye_windows', None):
            return
        fp.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        fp.showNormal()
        fp.raise_()

    def _make_eye_window(self, eye):
        """One Dual Projector output window. Split out so the lifecycle below
        stays testable without a real D3D11 widget."""
        from sylc.framepacking_window_d3d11 import EyeOutputWindow
        return EyeOutputWindow(eye, parent=None)

    def _seed_eye_window_params(self, widget):
        """Give a freshly built eye widget the render parameters of the source
        that is ALREADY playing.

        Extending the per-source sites to _display_widgets() is necessary but not
        sufficient: those sites (_try_start_hevc, change_stereo_mode) run at load
        time, and every path that reaches them clears the pair first, so a window
        built afterwards — the normal case, the user picks Dual Projector during
        playback — would keep its constructor defaults. plane_scale alone is the
        difference between a correct picture and a blown-out white one on a 10-bit
        D3D11VA source (65535/65472 vs the default 65535/1023).

        The embedded preview is the reference: it is the widget those same sites
        wrote to, and it is never destroyed.
        """
        ref = getattr(self, 'mvc_embedded_widget', None)
        if widget is None or ref is None:
            return
        for name in _EYE_INHERITED_RENDER_PARAMS:
            try:
                setattr(widget, name, getattr(ref, name))
            except Exception:
                pass
        # BD3D authored PG depth is pushed per GOP; without this the new window
        # renders at zero depth until the next one arrives.
        depth = getattr(self, '_pg_depth_last', None)
        if depth is not None and hasattr(widget, 'set_subtitle_depth'):
            try:
                widget.set_subtitle_depth(depth)
            except Exception:
                pass

    def _refresh_subtitle_targets(self):
        """Re-point both subtitle renderers at the CURRENT display-widget set.

        Opening or closing the eye-window pair changes that set, and both
        renderers cache the widgets they are connected to. Without this, a pair
        opened after the last connect never receives a cue (subtitles on the
        control monitor, none on either projector) and a closed pair leaves two
        dead widgets wired to the renderers.
        """
        if getattr(self, '_subtitle_manager', None):
            self._connect_subtitle_to_widget()
        if getattr(self, '_text_sub_active', False):
            self._connect_text_subtitle_to_widget()

    def _set_dual_projector_enabled(self, enable):
        """Open or close the two one-eye output windows.

        Asymmetry with the framepack window is deliberate: that one is a
        singleton we HIDE so returning to MultiView is instant and its renderer
        survives, while the eye windows are built on demand and CLOSED on exit
        -- their closeEvent already releases the D3D11 resources.
        """
        if enable:
            if self.eye_windows is None:
                # Fix round 1, Finding 4: build both defensively. eye_windows was
                # previously assigned only after BOTH constructors returned -- if the
                # second one raised (D3D11 init can fail), the first window was alive
                # but unreferenced (leaked) and the exception escaped with
                # current_stereo_mode already 'dual'. Close whatever was already
                # built and leave eye_windows None, so the caller sees a clean failure
                # instead of an orphaned window plus a crash.
                left = None
                try:
                    left = self._make_eye_window('left')
                    right = self._make_eye_window('right')
                except Exception:
                    logger.exception("[DUAL-PROJECTOR] failed to open the eye-window pair")
                    if left is not None:
                        try:
                            left.close()
                        except Exception:
                            pass
                    return
                self.eye_windows = (left, right)
                for window in self.eye_windows:
                    self._seed_eye_window_params(
                        getattr(window, 'display_widget', None))
            fp = getattr(self, 'framepacking_window', None)
            if fp is not None:
                fp.hide()
            for window in self.eye_windows:
                window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
                window.showNormal()
                window.raise_()
            # The pair is now part of the display-widget set the subtitle
            # renderers must feed. Idempotent: both connect helpers no-op when
            # the set they are already wired to is unchanged.
            self._refresh_subtitle_targets()
            logger.info("[DUAL-PROJECTOR] two eye outputs open")
            return

        pair, self.eye_windows = self.eye_windows, None
        for window in (pair or ()):
            try:
                window.close()
            except Exception:
                logger.exception("[DUAL-PROJECTOR] closing an eye window failed")
        if pair:
            # Drop the two closed widgets from the renderers' target set.
            self._refresh_subtitle_targets()
            logger.info("[DUAL-PROJECTOR] eye outputs closed")

    def _configure_3d_output_hevc(self, enable_3d, stereo_mode='auto'):
        """3D toggle for the HEVC path: show / hide the framepack window (which already
        receives L+R via _on_mvc_frame_yuv_ready → visible targets). mpv stays audio-only
        throughout — unlike the native SBS/TAB branch we NEVER restore mpv video (it has
        none). Keeps the embedded widget visible+rendering for timing parity.

        An explicit SBS/TAB choice lays the available pair out in the MAIN window
        instead of the detached framepack window. Genuine packed/multiview sources
        already provide L/R; Synth3D reaches the same presentation branch with its
        generated pair while the original HEVC input remains strictly mono."""
        fp = getattr(self, 'framepacking_window', None)
        emb = getattr(self, 'mvc_embedded_widget', None)
        # Review fix #1a: resolve 'auto' to the actual HEVC session mode BEFORE branching.
        # configure_3d_output() (the sibling entry point) does this same kind of resolution
        # at L5908-5913, but only AFTER its own "if _hevc_mode_active: self.
        # _configure_3d_output_hevc(...); return" early-out (L5867-5869) -- so THIS function
        # never got the benefit: a literal 'auto' (e.g. the user's first 3D-button click,
        # which never touches the combo) tested false against `stereo_mode in ('sbs','tab')`
        # below and fell through to the framepack branch even for a packed SBS/TAB file that
        # belongs in the main-window branch -- the pre-D2 bug, reachable again via 'auto'.
        # Source of truth: _hevc_detected_mode, latched in _try_start_hevc at promotion time
        # (~L6694) from the same `mode` local ('mvhevc'/'sbs'/'tab') that drove the promotion
        # itself, so it can't disagree with what video_3d_info/the combo were set to. mvhevc
        # (or an unset/unexpected value) resolves to the framepack/'mvc' branch below -- same
        # safe default as configure_3d_output's own 'auto' resolve ("mvc" for MVC content).
        if stereo_mode == 'auto':
            stereo_mode = getattr(self, '_hevc_detected_mode', None) or 'mvhevc'
        # Fix round 1 (bidirectional): same invariant enforcement as configure_3d_output's
        # own copy of this line -- placed AFTER the 'auto' resolution above so stereo_mode
        # holds a real value here ('auto' never equals 'dual', so ordering only matters for
        # correctness of the resolved value, not for safety).
        self._set_dual_projector_enabled(enable_3d and stereo_mode == 'dual')
        # D2: SBS/TAB always present in the MAIN window with the framepack window HIDDEN — for
        # EVERY HEVC 3D source (packed SBS/TAB and MV-HEVC multiview alike), not just multiview.
        # Only the MultiView ('mvc') selection opens the detached framepack window (below). The
        # HevcDecodeThread already splits packed SBS/TAB into L/R per its live _mode, so the
        # embedded widget's 'sbs'/'tab' shader lays those two views out as the requested
        # main-window layout (no source re-split here). MV-HEVC's two views are the stereo pair.
        if enable_3d and stereo_mode in ('sbs', 'tab') and emb is not None:
            # fp.hide() fires _on_framepacking_visibility_changed; that handler now IGNORES the
            # hide while current_stereo_mode is sbs/tab (D3 fix), so is_3d_enabled is no longer
            # flipped off underneath us. The explicit restore below stays as defense in depth.
            if fp is not None and fp.isVisible():
                fp.hide()
            emb.set_stereo_mode(stereo_mode)
            self.video_stack.setCurrentWidget(emb)
            self.active_mvc_widget = emb
            self.is_3d_enabled = True
            self._connect_subtitle_to_widget(emb)
            if self._text_sub_active:
                self._connect_text_subtitle_to_widget()
            self.show_3d_notification(f"3D Mode: {stereo_mode.upper()} (HEVC main window)",
                                      success=True)
            return
        if enable_3d and fp is not None:
            fp.apply_output_geometry(stereo_mode)
            self._apply_framepack_source_aspect(fp, stereo_mode)
            fp.display_widget.set_stereo_mode(
                'glasses' if stereo_mode == 'glasses' else 'framepack')
            self.active_mvc_widget = fp.display_widget
            if emb is not None:
                # Keep the normal UI page selected and live. Frame delivery gives
                # this 2D preview the left eye only with non-blocking Present(0),
                # while the detached framepack renderer remains the timing authority.
                self.video_stack.setCurrentWidget(emb)
                emb.set_stereo_mode('2d')
            self._connect_subtitle_to_widget(fp.display_widget)
            if self._text_sub_active:
                self._connect_text_subtitle_to_widget()
            self._show_framepacking_output()
            if stereo_mode == 'glasses':
                self.show_3d_notification("3D Mode: Glasses F-SBS (HEVC)", success=True)
            else:
                self.show_3d_notification("3D Mode (HEVC framepack)", success=True)
        else:
            if fp is not None and fp.isVisible():
                fp.hide()
            if emb is not None:
                self.video_stack.setCurrentWidget(emb)
                emb.set_stereo_mode('2d')
                self.active_mvc_widget = emb
            self.show_3d_notification("2D Mode (HEVC base view)", success=True)

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

        # Reap native owners retained by an earlier timeout once they have finally
        # returned on their own. Running entries must remain strongly referenced:
        # destroying a live QThread is a process-aborting Qt error.
        _mvc_survivors = []
        for _old_thread in getattr(self, '_mvc_leaked', []):
            try:
                if _old_thread.isRunning():
                    _mvc_survivors.append(_old_thread)
                else:
                    _old_thread.deleteLater()
            except RuntimeError:
                pass
        self._mvc_leaked = _mvc_survivors
        self._mvc_shutdown_blocked = bool(_mvc_survivors)

        # SyLC Cast (Task 13): the cast session is fed by _on_mvc_frame_yuv_ready, so it
        # cannot outlive the decoder. EVERY session-end routes through here (stop /
        # load-new / close / edge264-fallback / EOS), so this single point guarantees no
        # cast session leaks across a file change. stop() is idempotent and must run on
        # the GUI thread — which every caller of this method does.
        _cast = getattr(self, '_cast', None)
        if _cast is not None:
            try:
                _cast.stop()
            except Exception:
                logger.exception("[CAST] stop during MVC teardown failed")
            self._cast = None
            self._cast_connected = False

        # HEVC path teardown (symmetric, idempotent no-op when inactive): every MVC
        # teardown site (stop / load-new / close / edge264-fallback / EOS) routes through
        # here, so it also releases the HEVC decode thread + source when one is active.
        # restarting=... forwarded so its own synth3d-lost check doesn't fire the false
        # alarm mid same-session restart (see _stop_hevc_decoder's docstring).
        self._stop_hevc_decoder(restarting=getattr(self, '_mvc_restarting', False))

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
        # Mirror of _on_pg_depth_changed's latch: an eye window opened after this
        # cleanup must not inherit the previous file's authored depth.
        self._pg_depth_last = None
        for _w in (getattr(self, 'active_mvc_widget', None), *self._display_widgets()):
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
            try:
                _native = getattr(
                    self.mvc_decoder_thread, '_native_decoder', None)
                if _native is not None and hasattr(_native, 'request_abort'):
                    _native.request_abort()
                    logger.info("[MVC CLEANUP] Native decode abort requested")
            except Exception:
                pass
            # Brief pause to allow decoder thread to see the flag and abort operations
            import time
            time.sleep(0.050)  # 50ms

        # Stop ALL timers (not just _sync_timer and watchdog)
        timer_names = ['_sync_timer', '_stall_watchdog', '_edge264_startup_timer',
                       '_playback_timer', 'controls_hide_timer', '_render_heartbeat_timer']
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
                self.mvc_decoder_thread.frameYUVTimedReady.disconnect()
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
                if (hasattr(self.mvc_decoder_thread, 'pgsDataReady') and
                        getattr(self, '_pgs_streaming_connected', False)):
                    try:
                        if self._subtitle_manager:
                            self.mvc_decoder_thread.pgsDataReady.disconnect(
                                self._subtitle_manager.on_pgs_data)
                        else:
                            self.mvc_decoder_thread.pgsDataReady.disconnect()
                    except (RuntimeError, TypeError):
                        pass
                    self._pgs_streaming_connected = False
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

            # STEP 4: Wait for thread to finish (with timeout).
            #
            # NEVER call QThread.terminate() here: the worker can be inside edge264
            # or the native demuxer while holding locks. Killing it would make every
            # subsequent cleanup operation a possible use-after-free. If native I/O
            # does not return, retain the QThread and all of its native ownership in a
            # leak list. This mirrors the HEVC leak-over-UAF policy.
            _mvc_thread = self.mvc_decoder_thread
            thread_dead = bool(_mvc_thread.wait(5000))
            self._mvc_shutdown_blocked = (not thread_dead) or bool(_mvc_survivors)
            if not thread_dead:
                if not hasattr(self, '_mvc_leaked'):
                    self._mvc_leaked = []
                try:
                    # The QThread object normally has the main window as QObject
                    # parent. Detach it or closing the window would delete the live
                    # child despite the Python leak-list reference and trigger
                    # "QThread: Destroyed while thread is still running".
                    _mvc_thread.setParent(None)
                except RuntimeError:
                    pass
                self._mvc_leaked.append(_mvc_thread)
                logger.critical(
                    "[MVC CLEANUP] Decoder thread did not stop in 5 s; it was NOT "
                    "force-terminated. Native ownership is retained to prevent a "
                    "use-after-free. A second MVC decoder will not be started."
                )
            else:
                logger.info("[MVC CLEANUP] Decoder thread stopped successfully")
                _mvc_thread.deleteLater()
            self.mvc_decoder_thread = None
            self._last_display_frame_ts = None
            self._display_fps_avg = None

        # STEP 5: Clear framepacking windows and flush OpenGL textures.
        # EVERY display widget, not the embedded+framepack pair this used to name
        # one by one. stop_playback's own clearing loop does not cover this: the
        # natural end of a film reaches THIS method directly (_on_mvc_finished
        # sets _playback_ended and posts _stop_mvc_decoder on a 300 ms timer,
        # while stop_playback early-returns on that same flag), and so does the
        # decoder-error path via _fallback_from_edge264. An unlisted widget
        # therefore keeps the last frame frozen at exactly the moment a Dual
        # Projector user is looking at the screen -- on both projectors.
        # The two pre-existing log lines are reproduced verbatim per widget.
        _emb = getattr(self, 'mvc_embedded_widget', None)
        _fpw = getattr(getattr(self, 'framepacking_window', None),
                       'display_widget', None)
        for _w in self._display_widgets():
            if _w is _emb:
                _label = "Embedded widget"
            elif _w is _fpw:
                _label = "Framepacking window"
            else:
                _label = f"Eye window ({getattr(_w, 'eye_view', None) or '?'})"
            try:
                # V7b FIX: Clear OpenGL textures to prevent stale frames during seek
                if hasattr(_w, 'clear_textures'):
                    _w.clear_textures()
                logger.info(f"[MVC CLEANUP] {_label} cleared")
            except Exception as e:
                logger.warning(f"[MVC CLEANUP] Error clearing {_label.lower()}: {e}")

        # STEP 6: Close demuxer
        if self.demuxer and not getattr(self, '_mvc_shutdown_blocked', False):
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
        self._edge264_waiting_for_first_frame = False
        self._edge264_mpv_handoff_done = False
        self._edge264_pre_handoff_widget = None

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

        if hasattr(self, '_synth3d_handle_decoder_stop'):
            self._synth3d_handle_decoder_stop(getattr(self, '_mvc_restarting', False))

    @Slot()
    def _on_mvc_frame_ready(self):
        """
        DEPRECATED: This slot is bypassed when direct widget path is active.
        Kept only for fallback compatibility when _display_widget is None.

        PERFORMANCE: When direct path is active, this entire function is skipped,
        saving ~5-8ms per frame (shared memory read + QImage creation overhead).
        """
        if not self._native_signal_is_current():
            return
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
        if not self._native_signal_is_current():
            return
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

    @staticmethod
    def _planes_for_target(target, left_planes, right_planes):
        """Which eye(s) this presentation target wants.

        A Dual Projector window shows ONE eye through the renderer's '2d' mode,
        which samples only the first three textures -- so its own eye's planes
        go in the FIRST slot and the second is left empty. Every other target
        (framepack, embedded preview) keeps receiving the full pair."""
        eye = getattr(target, 'eye_view', None)
        if eye == 'right':
            return (right_planes, None)
        if eye == 'left':
            return (left_planes, None)
        return (left_planes, right_planes)

    @Slot(object, object)
    def _on_mvc_frame_yuv_ready(self, left_planes, right_planes):
        """Compatibility entry point for callers without a media timestamp."""
        if not self._native_signal_is_current():
            return
        return self._on_mvc_frame_yuv_timed_ready(
            left_planes, right_planes, -1.0)

    @Slot(object, object, object)
    def _on_hevc_frame_yuv_timed_ready(
            self, left_planes, right_planes, video_time_ms):
        """Bounded HEVC delivery wrapper.

        The decoder marks one presentation pending before emitting.  Acknowledge
        it only after the GUI has uploaded/presented this frame, even when the
        shared presentation slot rejects malformed data or raises.
        """
        try:
            if not self._native_signal_is_current():
                return
            return self._on_mvc_frame_yuv_timed_ready(
                left_planes, right_planes, video_time_ms)
        finally:
            try:
                owner = self.sender()
            except Exception:
                owner = None
            if owner is None:
                owner = getattr(self, 'hevc_thread', None)
            if owner is not None and hasattr(owner, 'presentation_consumed'):
                owner.presentation_consumed()

    @Slot(object, object, object)
    def _on_mvc_frame_yuv_timed_ready(
            self, left_planes, right_planes, video_time_ms):
        """Dispatch one decoded stereo frame to every visible presentation target.

        Both numpy tuples are passed by reference — no extra copy made here.
        The detached framepack window owns vsync while visible; the embedded
        main-window preview is updated with a non-blocking Present(0), and its 2D
        renderer uploads only the left-eye planes. This keeps both windows alive
        without the old 2x-six-plane upload / two-blocking-vsync performance trap.

        In Dual Projector mode the two eye windows REPLACE the framepack window
        as the targets (same picture, cut in two): the left eye becomes the
        timing authority, the right eye and the preview follow with Present(0),
        and each eye window receives only its own eye's planes.
        """
        if not self._native_signal_is_current():
            return
        # Validate planes — bail out silently on malformed input (decoder transient)
        if (not left_planes or not right_planes
                or len(left_planes) != 3 or len(right_planes) != 3):
            return
        for plane in (*left_planes, *right_planes):
            if plane is None or not isinstance(plane, np.ndarray):
                return

        # Startup transaction: MPV stays on-screen while demux/edge264 warms up.
        # The first structurally valid decoded frame is the commit point; only
        # then do we release MPV's D3D11 output and expose the native widget.
        if (not getattr(self, '_hevc_mode_active', False)
                and getattr(self, '_edge264_waiting_for_first_frame', False)):
            try:
                signal_owner = self.sender()
            except Exception:
                signal_owner = None
            if (signal_owner is not None and self.mvc_decoder_thread is not None
                    and signal_owner is not self.mvc_decoder_thread):
                logger.debug("[EDGE264-HANDOFF] Ignoring a queued frame from a stale decoder")
                return
            if not self._commit_edge264_video_handoff():
                return

        # A valid frame means edge264 is healthy again — clear the crash streak so
        # transient (recoverable) crashes never accumulate toward the fallback cap.
        if getattr(self, '_edge264_consecutive_crashes', 0):
            self._edge264_consecutive_crashes = 0

        # Packed-stereo (Full-SBS / Full-TAB): edge264 delivered both eyes in ONE
        # frame. Split into separate L (base) / R views so every target renders it
        # like MVC — embedded '2d' shows the base eye, '2d'/'sbs'/'tab' set the main
        # layout, and the framepack window stacks L+R.
        # HEVC guard: the HevcDecodeThread ALREADY splits packed SBS/TAB into separate
        # L/R views before emitting, so it must NOT be re-split here (unlike the edge264
        # packed-H.264 path, which emits one packed frame). _hevc_mode_active suppresses it.
        _sm = self.video_3d_info.get('stereo_mode') if isinstance(self.video_3d_info, dict) else None
        if _sm in ('sbs', 'tab') and not getattr(self, '_hevc_mode_active', False):
            try:
                left_planes, right_planes = self._split_packed_stereo(left_planes, _sm)
            except Exception as e:
                logger.error(f"[PACKED-3D] frame split failed: {e}")
                return

        embedded = getattr(self, 'mvc_embedded_widget', None)
        fp_window = getattr(self, 'framepacking_window', None)

        # Fix round 2 (Critical 1, completion): left_planes here is already the
        # real per-eye plane for every source class (the HEVC decode thread and
        # the _split_packed_stereo call above both split BEFORE this point) --
        # keep the Glasses aspect cache current from it.
        self._note_decoded_eye_plane(left_planes)

        presentations = _select_stereo_presentation_targets(
            embedded, fp_window, getattr(self, 'active_mvc_widget', None),
            getattr(self, 'eye_windows', None))
        targets = [widget for widget, _ in presentations]

        # One shared asynchronous matte for every surface. Submission is
        # latest-only and returns immediately; the result selected here always
        # belongs to this media timeline (stale masks are rejected by PTS).
        matte_for_frame = getattr(self, '_synth3d_human_matte_for_frame', None)
        human_matte = (matte_for_frame(left_planes, video_time_ms)
                       if matte_for_frame is not None else None)

        eye_active = bool(getattr(self, 'eye_windows', None))
        if len(presentations) > 1 and not getattr(self, '_dual_output_logged', False):
            self._dual_output_logged = True
            if eye_active:
                logger.info("[FRAME-ROUTE] Dual Projector active: left eye=vsync, "
                            "right eye + main preview=non-blocking, one eye each")
            else:
                logger.info("[FRAME-ROUTE] dual presentation active: framepack=vsync, "
                            "main preview=non-blocking left eye")

        cast = getattr(self, '_cast', None)
        cast_frame_uploaded = False
        for target, use_vsync in presentations:
            try:
                # NativeFramepackWidget reads this immediately in the same GUI
                # thread. An attribute keeps the call contract compatible with
                # older/test widgets while the new renderer honours Present(0).
                target.present_vsync = bool(use_vsync)
                # Exact media PTS travels beside the planes without widening
                # set_frame_yuv_views(), whose compatibility surface is used by
                # test and fallback widgets. NativeFramepackWidget consumes it
                # immediately before uploading this same frame.
                target.video_time_ms = float(video_time_ms)
                try:
                    target.set_synth3d_human_matte(human_matte)
                except (AttributeError, TypeError):
                    pass
                _first, _second = self._planes_for_target(
                    target, left_planes, right_planes)
                delivered = target.set_frame_yuv_views(_first, _second)
                if (delivered is not False and cast is not None and
                        getattr(target, '_r', None) is getattr(cast, '_renderer', None)
                        and getattr(target, 'current_stereo_mode', 1) != 0):
                    cast_frame_uploaded = True
            except Exception as e:
                logger.error(f"[FRAME-ROUTE] delivery to {type(target).__name__} failed: {e}")

        # Native renderer A/B tap (Tokyo #3, S4): diagnostic only, env-gated by
        # SYLC_NATIVE_TAP=1. Mirrors the same frame into a separate native-D3D11
        # window for live parity comparison. Zero impact when the flag is unset.
        if os.environ.get("SYLC_NATIVE_TAP") == "1":
            tap = getattr(self, '_native_tap', None)
            if tap is None:
                try:
                    from sylc.native_renderer.native_tap import NativeRendererTap
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

        # SyLC Cast (Task 13): feed the SAME decoded stereo frame to an active cast
        # session. Fires AFTER the display's own set_frame_yuv_views above (same GUI
        # thread, same planes), so the cast's upload + NVENC-encode never disturbs the
        # on-screen picture; push() returns fast and self-drops frames under backpressure.
        if cast is not None:
            # 8-bit sessions cannot take uint16 planes (silent low-byte truncation +
            # R16<->R8 texture thrash on the shared renderer). A Main10 session CAN:
            # the pack reads the same R16 planes the display uploaded (plane_scale
            # aligns SW/HW layouts) into a P010 surface -- so only gate the 10-bit
            # push when the session is NOT main10.
            if left_planes[0].dtype != np.uint8 and not getattr(cast, 'is_main10', False):
                if not getattr(self, '_cast_10bit_warned', False):
                    self._cast_10bit_warned = True
                    logger.warning("[CAST] 10-bit 3D source (uint16 planes) not supported by "
                                   "the v1 cast path; skipping cast push")
                    self.show_3d_notification(
                        "Streaming to Quest: 10-bit video is not supported in v1.",
                        success=False)
            else:
                # Dual Projector presents the SAME stereo pair through two one-eye
                # windows instead of the framepack window, which it hides -- so the
                # frame being cast is still stereo and this gate must say so, or the
                # Quest silently receives mono for the whole session.
                csm = 1 if (eye_active or
                            (fp_window is not None and fp_window.isVisible())) else 0
                cref = targets[0] if targets else getattr(self, 'active_mvc_widget', None)
                csdr = getattr(cref, '_sdr_white_level', None)
                cast.push(
                    left_planes,
                    right_planes,
                    csm,
                    csdr,
                    frame_already_uploaded=cast_frame_uploaded,
                )

        self._record_display_frame_stats()

    @Slot(str)
    def _on_mvc_error(self, error_msg):
        """Slot: MVC decoder error - immediate stop and cleanup"""
        if not self._native_signal_is_current():
            return
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
        if not self._native_signal_is_current():
            return
        self.controls_overlay.set_status_info(f"MVC @ {fps:.1f} fps")
        self.monitoring_overlay.update_decoder_fps(fps)
        self.metrics_overlay.update_decoder_fps(fps)

    @Slot(int, int)
    def _on_mvc_stats_update(self, buffer_size, drop_count):
        if not self._native_signal_is_current():
            return
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
        if not self._native_signal_is_current():
            return
        raw_timestamp = timestamp

        # V62 SYNC-FIX: Disable the external drift corrector. The decoder's
        # internal V12 loop is now the master authority; fighting it from here
        # with a stale MPV clock was the cause of the 19.98 fps cap.
        self._last_mvc_timestamp = raw_timestamp
        if raw_timestamp > 0:
            self._set_ui_time(raw_timestamp)
        return

    @Slot(float)
    def _on_hevc_position(self, t_s):
        """HEVC-native timeline feed (HevcDecodeThread.positionChanged, ~4 Hz):
        same contract as the MVC per-frame timestamp slot above, and the ONLY
        position clock in HEVC mode (_update_playback_position stands down).
        Vital for no-audio media, where the mpv shell never produces a
        time-pos."""
        if not self._native_signal_is_current():
            return
        self._last_mvc_timestamp = t_s
        # Keep the relative-seek base in sync now that the position poller no
        # longer maintains it in HEVC mode.
        self._current_precise_time = float(t_s)
        if t_s > 0 and not self._is_scrubbing and not getattr(self, '_is_seeking', False):
            self._set_ui_time(t_s)
        
        if self._subtitle_manager is not None:
            self._subtitle_manager.update_time(t_s)

    @Slot()
    def _on_mvc_finished(self):
        """Slot: MVC decoding finished"""
        if not self._native_signal_is_current():
            return
        logger.info("MVC playback finished")

        # Per-file memory: a NATURAL end clears the resume point — replaying a
        # finished film starts at the beginning, not at the credits.
        _rem = getattr(self, '_remember_for_file', None)
        if _rem is not None:
            _rem(resume_pos=None)

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

        # EOS toast: _on_mvc_finished is the SHARED end-of-stream handler (MVC + HEVC),
        # so label it for the path that actually finished instead of always saying "MVC".
        _finished_msg = ("HEVC playback finished"
                         if getattr(self, '_hevc_mode_active', False)
                         else "MultiView playback finished")
        self.show_3d_notification(_finished_msg, success=True)

        # CRITICAL FIX V2: Set flag to block MPV callbacks from restarting playback
        # This must be set BEFORE any other operations
        self._playback_ended = True

        # CRITICAL FIX: Stop timeline timer IMMEDIATELY to prevent continued updates
        # This must happen BEFORE any async operations
        if hasattr(self, '_playback_timer') and self._playback_timer.isActive():
            self._playback_timer.stop()
            logger.info("[MVC FINISHED] Playback timer stopped")

        # MPV callbacks are suppressed during this transition; reset the
        # transport synchronously instead.
        self._set_playback_stopped_ui()

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

        # V14b GRACEFUL ENDING: Increase delay to 300ms for MPV event loop to settle
        # The decoder needs time to exit AND MPV event thread needs to calm down
        self._media_single_shot(
            300, self._stop_mvc_decoder, self._media_session_id)

    def _record_display_frame_stats(self):
        now = time.perf_counter()
        self._last_decoder_activity_ts = time.monotonic()
        self._last_watchdog_dump_ts = 0.0
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

    def _on_framepacking_geometry_changed(self, width, height):
        """Fix round 3: keep the Glasses aspect clamp (`_glasses_target_
        aspect`) live across every real size change of the detached window --
        an ordinary Qt resize AND fake-fullscreen enter/exit alike, both of
        which deliver a ordinary Qt resizeEvent (see Framepacking3DWindow.
        resizeEvent). `_apply_framepack_source_aspect` itself re-reads the
        window's current size, so this slot's job is only to know WHEN to
        call it again -- a value computed once at configure time would be
        stale the instant the user toggles fullscreen, exactly when it
        matters most. A no-op for every presentation but Glasses: the plain
        baseline the others use doesn't depend on the surface size."""
        if getattr(self, 'current_stereo_mode', None) != 'glasses':
            return
        fp = getattr(self, 'framepacking_window', None)
        if fp is not None:
            self._apply_framepack_source_aspect(fp, 'glasses')

    def _on_framepacking_visibility_changed(self, visible):
        self._framepacking_visible = visible
        self.monitoring_overlay.update_window_state(visible)
        self._refresh_monitoring_overlay()

        # V9 FIX: Update active widget and stereo mode when framepacking window visibility changes
        # This ensures frames go to the correct widget with correct stereo mode
        if visible and self.framepacking_window:
            # Switch to framepack mode when window becomes visible -- except while
            # Glasses is the active presentation (fix round 1, Critical 2). This
            # handler used to stomp 'framepack' back on unconditionally on every
            # show, including a plain resize-back-to-windowed while fullscreen
            # (Framepacking3DWindow.enter_fake_fullscreen emits visibilityChanged
            # (True) unconditionally) -- clobbering the Glasses layout at exactly
            # the moment the user goes fullscreen on their glasses, and on every
            # SBS/off/Dual -> Glasses transition that re-shows this window.
            if getattr(self, 'current_stereo_mode', None) != 'glasses':
                self.framepacking_window.display_widget.set_stereo_mode('framepack')
            self.active_mvc_widget = self.framepacking_window.display_widget
            if self.mvc_decoder_thread:
                self.mvc_decoder_thread.set_display_widget(self.framepacking_window.display_widget)
            logger.info("[VISIBILITY] Framepacking window visible: switched to framepack mode")
        elif not visible and hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            # D3 ROOT-CAUSE FIX: a hidden framepack window means "user closed the 3D output"
            # ONLY when the framepack window IS the active 3D presentation — i.e. the MultiView
            # ('mvc'/'auto') mode. For SBS/TAB the framepack window is DELIBERATELY hidden by
            # the player (presentation runs in the MAIN window), so a visibilityChanged(False)
            # here is the normal expected state, NOT a user close. Treating it as a close
            # flipped is_3d_enabled off unconditionally — correction (review pass): the
            # emit is SYNCHRONOUS (fp.hide() -> hideEvent -> visibilityChanged.emit(False) ->
            # this slot, all in the same call stack; verified against
            # framepacking_window_d3d11.py's hideEvent/exit_fake_fullscreen). The bug was
            # never timing/ordering — it was that this handler had no way to tell "window
            # hidden because the player switched to SBS/TAB main-window presentation" apart
            # from "window hidden because the user closed it", so every SBS/TAB switch's own
            # fp.hide() re-entered here and flipped is_3d_enabled off with nothing to restore
            # it, permanently gating out change_stereo_mode() so subsequent SBS<->TAB
            # switches silently stopped taking effect (the picture froze mis-scaled). The fix
            # below removes the ambiguity by mode-gating: current_stereo_mode in ('sbs','tab')
            # IS that missing signal, so the hide is recognized and ignored, not raced.
            # 'dual' added (fix round 1, Finding 1): _set_dual_projector_enabled(True)
            # hides framepacking_window as part of ENTERING Dual Projector, which fires
            # this same handler synchronously (fp.hide() -> hideEvent -> visibilityChanged
            # -> here) -- exactly the same ambiguity SBS/TAB already solved, just with a
            # third deliberate-hide reason.
            if getattr(self, 'current_stereo_mode', 'mvc') in ('sbs', 'tab', 'dual'):
                logger.info("[VISIBILITY] Framepacking window hidden during SBS/TAB "
                            "presentation — expected (main-window 3D); is_3d preserved")
                return
            # Fourth deliberate-hide reason (round 4, depth presets): changing the
            # depth preset re-arms synthesis with toggle_synth3d(False) then (True),
            # and the off leg's configure_3d_output(False) hides this window for the
            # microseconds until the on leg re-shows it. Reading that as a user close
            # would tear 3D down mid-re-arm and, worse, auto-stop the cast session
            # below — which the re-arm is required to preserve. The flag is set only
            # around that one toggle pair (see _synth3d_set_depth_preset).
            if getattr(self, '_synth3d_rearming', False):
                logger.info("[VISIBILITY] Framepack hidden during a depth-preset "
                            "re-arm — 3D and any cast session preserved")
                return
            # FSBS/edge264 MultiView blind spot (fix-fsbs-multiview): a
            # visibilityChanged(False) does NOT always mean the window was hidden. The
            # framepack window's exit_fake_fullscreen() (F key / double-click / Esc)
            # emits visibilityChanged(False) when LEAVING fullscreen while the window
            # stays VISIBLE (just resized back to windowed) — and in the MultiView
            # ('mvc') presentation that window IS the active 3D output. Treating that as
            # a user-close tore 3D down (is_3d_enabled off, 3D button unchecked, decoder
            # retargeted to the 2D embedded widget); the still-open framepack window then
            # froze and every later MultiView/SBS/TAB combo pick became a no-op (the
            # `has_media and is_3d_enabled` gate in change_stereo_mode) — i.e. "MultiView
            # ne marche plus" after the first fullscreen session. The D3 gate above only
            # covered the sbs/tab presentation; the packed-'mvc' framepack flow (edge264
            # AND mv-hevc alike) was never accounted for. Only a GENUINE hide — the window
            # actually not visible (X / Alt-F4 / the player's own fp.hide()) — is a close.
            fpw = getattr(self, 'framepacking_window', None)
            if fpw is not None and fpw.isVisible():
                logger.info("[VISIBILITY] Framepack visibilityChanged(False) but window "
                            "still visible (fullscreen-exit) — MultiView 3D preserved, "
                            "not a user close")
                return
            # Switch back to embedded 2D mode when window is hidden
            self.mvc_embedded_widget.set_stereo_mode('2d')
            self.active_mvc_widget = self.mvc_embedded_widget
            if self.mvc_decoder_thread:
                self.mvc_decoder_thread.set_display_widget(self.mvc_embedded_widget)
            logger.info("[VISIBILITY] Framepacking window hidden: switched to embedded 2D mode")

            # F1c: a genuine close must also tear down synth3d -- otherwise the
            # embedded '2d' preview just switched to above shows the WARPED left
            # eye (the shader-side effect of synth3d_enabled) and the depth model
            # keeps running with nothing left to display its output. toggle_synth3d
            # is idempotent-safe here: its own configure_3d_output(False) repeats
            # the same 2D-switch already done just above (redundant but harmless,
            # since mvc_mode_active is still True at this point -- same branch,
            # same idempotent calls), and additionally does the widget/notify/menu
            # teardown (_push_synth3d_to_widgets, renderer set_synth3d(False),
            # show_3d_notification, _update_3d_button_state/_update_synth3d_menu_state)
            # this handler doesn't otherwise perform. Placed BEFORE the 3D-button
            # uncheck below so its "2D->3D AI off" notification doesn't race the
            # button-state update.
            if getattr(self, '_synth3d_active', False):
                # remember=False: dismantling the pipeline because its output
                # window went away is not the viewer renouncing 2D->3D for
                # this title — see the persistence comment in toggle_synth3d.
                self.toggle_synth3d(False, remember=False)

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

                # Restore 2D navigation bar UI - AFTER 3D state is fully disabled
                QTimer.singleShot(250, lambda: (self._update_overlays_geometry(), self.show_controls()))
                
                # SyLC Cast: Auto-stop cast if running, because the 3D renderer is shutting down
                if getattr(self, '_cast', None) is not None:
                    self._on_cast_requested(None)
                    logger.info("[CAST] Auto-stopped because framepacking window closed")
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

        # The transactional edge264 startup guard owns this phase and has a
        # topology-aware deadline (12 s mono, 25 s MVC). Dumping all process
        # stacks every three seconds before the first frame is both misleading
        # and expensive on the very systems we are trying to diagnose.
        if getattr(self, '_edge264_waiting_for_first_frame', False):
            return

        now = time.monotonic()
        last_dump = getattr(self, '_last_watchdog_dump_ts', 0.0)
        if (now - self._last_decoder_activity_ts > 5.0 and
                (not last_dump or now - last_dump >= 15.0)):
            self._last_watchdog_dump_ts = now
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
        """Start edge264 while retaining MPV video until the first native frame."""
        try:
            if disable_mpv:
                # Kept for call-site compatibility. Early disable used to turn a
                # demuxer probe failure into a black screen; handoff now happens
                # transactionally in _commit_edge264_video_handoff().
                logger.warning("[EDGE264-HANDOFF] Ignoring unsafe early MPV-disable request")

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
                if getattr(self, '_edge264_waiting_for_first_frame', False):
                    timeout_ms = _edge264_startup_timeout_ms(self.video_3d_info)
                    self._edge264_startup_timer.start(timeout_ms)
                    logger.info(
                        f"[EDGE264-STARTUP] First-frame guard armed for "
                        f"{timeout_ms / 1000:.0f}s; MPV video retained")
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


__all__ = [
    'NativeDecoderMixin', 'configure_native_decoder_support',
    '_EYE_INHERITED_RENDER_PARAMS',
]

"""Seek coordination, playback clock and pause-state orchestration."""

import logging
import time

from PySide6.QtCore import QTimer, Slot

from sylc.robust_seek_queue import should_resume_after_sync

logger = logging.getLogger(__name__)


class PlaybackTimelineMixin:
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
        self._mark_activity()
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
            self._seek_queue.request_seek(
                target_time,
                is_mvc=self.mvc_mode_active,
                resume_after=self._was_playing_before_scrub)
        else:
            self._handle_seek_request(target_time)

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

        # The cast audio is decoded independently from the same media. Seek it
        # through its thread-safe request API at the same instant as mpv/video;
        # otherwise it would race through every intermediate audio packet and
        # flood the receiver after a large timeline jump.
        _cast = getattr(self, '_cast', None)
        if _cast is not None and getattr(_cast, 'is_active', False):
            try:
                _cast.seek_audio(round(float(time_pos) * 1000.0))
            except Exception:
                logger.exception("[CAST] Unable to align AudioTap with player seek")

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
                owner = self.player

                def resume_owned_player():
                    if owner is self.player:
                        owner.pause = False

                self._media_single_shot(100, resume_owned_player)

        # 2. Standard 2D Mode Seek (also the HEVC path — mvc_mode_active stays False)
        else:
            # Ride the robust queue exactly like the MVC branch above: its
            # simple (is_mvc=False) executor already drives mpv AND the HEVC
            # decode thread (_on_seek_queue_mpv_seek), and — decisively — it
            # emits seek_started/seek_completed, the ONLY signals that ever
            # clear _is_seeking and the slider's busy state. The direct seek
            # below never signalled completion, so the first resume/seek of
            # an HEVC session latched _is_seeking forever: _on_hevc_position
            # then refused every _set_ui_time and the timeline stayed frozen
            # at its pre-seek value while the film played on (2026-08-04).
            if hasattr(self, '_seek_queue') and self._seek_queue:
                self._seek_queue.request_seek(time_pos, is_mvc=False)
                return
            # Fallback (no queue built yet): direct seek, kept for parity
            # with the MVC fallback above.
            try:
                if self.player:
                    self.player.time_pos = time_pos
                    # Ensure internal tracker is updated for 2D mode as well
                    self._decoder_start_position = time_pos
            except Exception as e:
                print(f"Error during seek: {e}")
            # HEVC path: seek the decode thread alongside the mpv audio seek.
            if getattr(self, '_hevc_mode_active', False) and getattr(self, 'hevc_thread', None):
                try:
                    # I2: prime the master-clock cache to the seek TARGET (seconds — the
                    # same unit the on_time_update observer stores; _mpv_time_pos_ms
                    # multiplies by 1000) BEFORE seeking, so the decode thread re-anchors
                    # to the target rather than the stale pre-seek position.
                    self._prime_mpv_time_pos(time_pos)
                    self.hevc_thread.seek_to(time_pos * 1000.0)
                except Exception:
                    pass
            # Without the queue there is no seek_completed to clear the
            # latch — reopen the position gate here or the bar freezes.
            self._is_seeking = False
            self.controls_overlay.time_slider.set_busy(False)

    def on_seek(self, time_pos):
        self._handle_seek_request(time_pos)

    def _handle_mvc_seek(self, time_pos):
        self._handle_seek_request(time_pos)

    @Slot()
    def _on_mvc_seek_finished(self):
        if not self._native_signal_is_current():
            return
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
        if not self._native_signal_is_current():
            return
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

        # Restore the state captured before the seek. The SSIF scan pause is
        # technical and must never become a new user-visible pause. This same
        # callback also performs the INITIAL MVC/Framepack handshake at 0s;
        # there is no seek intent then, so preserve the historical behaviour
        # and start playback unconditionally.
        if self.player:
            try:
                is_user_seek = bool(getattr(self, '_is_seeking', False))
                should_resume = should_resume_after_sync(
                    is_user_seek,
                    getattr(self, '_was_playing_before_seek', False))
                self.player.pause = not should_resume
                logger.info(
                    "[V8-SYNC] MPV state after atomic sync: %s (%s)",
                    "playing" if should_resume else "paused",
                    "user seek" if is_user_seek else "initial handshake")
            except Exception as e:
                logger.warning(f"[V8-SYNC] MPV state restore warning: {e}")

        logger.info(f"[V8-SYNC] ATOMIC SYNC COMPLETE: T_audio = T_video = {cues_timestamp:.3f}s")

    def _on_seek_started_logic(self, target_time):
        """Called when seek starts."""
        self._is_seeking = True
        # The queue captured this BEFORE its technical pause. Reading mpv or
        # decoder state here is too late: their pause callback may already have
        # arrived and would turn a playing seek into a persistent pause.
        self._was_playing_before_seek = self._seek_queue.resume_after_seek
             
        # Force UI to target immediately and hold
        self.controls_overlay.set_time(target_time)
        # Update internal trackers to prevent drift
        self._last_ui_time = target_time
        self._current_precise_time = target_time

        # PGS subtitles: EVERY queue-driven seek must reset the subtitle state
        # (visible set + streaming buffer + the parser's _fed_until watermark).
        # This used to live only in _handle_seek_request, which the seek QUEUE
        # path never traverses — after a BACKWARD disc seek the max-updated
        # watermark stayed at the pre-seek position (log: 'fed=173.76s' frozen
        # while clock replayed 138→170s), silently disabling the anti-flicker
        # bridge for that whole window and briefly showing a pre-seek set.
        if self._subtitle_manager:
            try:
                self._subtitle_manager.on_seek()
            except Exception:
                logger.exception("[SEEK] subtitle on_seek failed")
        self._synth3d_notify_seek_widgets()
        if self._text_sub_active and self._text_subtitle_renderer:
            try:
                self._text_subtitle_renderer.clear()
            except Exception:
                pass

    def _on_seek_completed_logic(self):
        """Called when seek finishes."""
        self._is_seeking = False
        
        # Restore exactly the transport state from before the seek.
        self._on_seek_queue_pause_request(
            not bool(self._was_playing_before_seek))

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
        # HEVC path uses the simple (is_mvc=False) seek route: drive the decode thread's
        # seek alongside the mpv audio seek. No IDR handshake — the thread re-anchors to
        # the mpv clock via clock_offset_provider.
        if getattr(self, '_hevc_mode_active', False) and getattr(self, 'hevc_thread', None):
            try:
                # I2: prime the master-clock cache to the seek TARGET (seconds — the same
                # unit the on_time_update observer stores; _mpv_time_pos_ms multiplies by
                # 1000) BEFORE seeking, so the decode thread re-anchors to the target
                # rather than the stale pre-seek position.
                self._prime_mpv_time_pos(target_time)
                self.hevc_thread.seek_to(target_time * 1000.0)
            except Exception:
                pass

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
        elif self.mvc_mode_active:
            print(f"[SEEK-QUEUE] Starting decoder at {target_time:.3f}s")
            self._start_mvc_decoder(start_time=target_time)
        else:
            # DF-FINAL FIX 2: mvc_mode_active is cleared by _fallback_from_edge264
            # (and _on_mvc_error before it). A decoder-seek queued before the
            # fallback must not resurrect the MVC pipeline after we've already
            # degraded to 2D mpv.
            print(f"[SEEK-QUEUE] Ignoring stale decoder-start at {target_time:.3f}s "
                  f"(mvc_mode_active=False, pipeline fell back to 2D)")

    def update_ui_state(self):
        self.show_controls()
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

    def on_duration_change(self, _, value, session_id=None, core=None):
        """mpv-thread callback: carry the owning session onto Qt's main thread."""
        owner = self._media_session_id if session_id is None else session_id
        source = self.player if core is None else core
        self.mpv_duration_event.emit((owner, source, value))

    @Slot(object)
    def _dispatch_mpv_duration(self, payload):
        session_id, core, value = payload
        if not self._session_is_current(session_id, core=core):
            return
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        self._handle_duration_change(value, session_id)

    def _handle_duration_change(self, value, session_id=None):
        """Handle duration change in main Qt thread"""
        owner = self._media_session_id if session_id is None else session_id
        if not self._session_is_current(owner):
            return
        self.controls_overlay.set_duration(value)
        if self.current_file_path:
            self._apply_preview_thumbs_policy(self.current_file_path)
            self.controls_overlay.time_slider.set_video_file(self.current_file_path, value or 0)
            # THUMB: duration arrival = playback is up (mpv loaded the file for
            # both the 2D and MVC paths) → arm shortly after; seeks re-disarm.
            self._media_single_shot(
                1000,
                lambda: getattr(self, '_thumb_service', None)
                and self._thumb_service.arm(),
                owner)
        # CRITICAL (SSIF/M2TS seek): mpv reports duration asynchronously, usually AFTER
        # the MVC decoder + its demuxer were created, so the demuxer's proportional seek
        # never got a duration and landed at byte 0 (restart). Push it now so seeks work.
        if value and getattr(self, 'mvc_decoder_thread', None):
            try:
                self.mvc_decoder_thread.set_media_duration(float(value))
            except Exception as e:
                logger.warning(f"[MVC] set_media_duration propagation failed: {e}")

    def _prime_mpv_time_pos(self, value):
        """Install an exact media-time anchor without reading an mpv property."""
        if value is None:
            self._mpv_time_pos_cache = None
            self._mpv_time_pos_cache_mono = None
            return
        self._mpv_time_pos_cache = float(value)
        self._mpv_time_pos_cache_mono = time.monotonic()

    def _cache_mpv_time_pos(self, value):
        """Merge one pushed mpv sample into a smooth, monotonic audio clock.

        python-mpv callbacks can be delayed while the GUI owns the GIL.  Treating
        every late callback as a fresh wall-clock anchor made the apparent audio
        clock run at ~0.4x on 4K TrueHD.  Between explicit seeks the real audio
        device and media timeline are 1x, so never pull an established estimate
        backwards; a newer sample may still advance it immediately.
        """
        if value is None:
            return
        sample = float(value)
        now = time.monotonic()
        previous = getattr(self, '_mpv_time_pos_cache', None)
        previous_mono = getattr(self, '_mpv_time_pos_cache_mono', None)
        if (previous is not None and previous_mono is not None
                and not getattr(self, '_mpv_pause_cache', True)
                and not getattr(self, '_is_seeking', False)):
            predicted = float(previous) + max(0.0, now - float(previous_mono))
            sample = max(sample, predicted)
        self._mpv_time_pos_cache = sample
        self._mpv_time_pos_cache_mono = now

    def on_time_update(self, _, value, session_id=None, core=None):
        """MPV time position changed - called from MPV event thread!"""
        owner = self._media_session_id if session_id is None else session_id
        source = self.player if core is None else core
        if not self._session_is_current(owner, core=source):
            return
        try:
            # HEVC master clock: cache mpv's pushed time-pos (seconds) for the decode
            # thread's clock_offset_provider (_mpv_time_pos_ms). This is a plain attribute
            # store of a value mpv HANDED us — NOT a blocking mpv read (the 0xe24c4a02
            # hot-path pattern). Kept above the guards so the cache never goes stale.
            if value is not None:
                self._cache_mpv_time_pos(value)
                # Startup handshake: do not depend on the separate pause observer
                # to wake HEVC.  The first advancing audio timestamp proves mpv is
                # running; release the paused decoder directly from this callback
                # (only atomic Python state changes, no GUI/mpv calls).
                request = getattr(self, '_hevc_start_request', None)
                if request is not None:
                    target_s = request[0]
                    if (not getattr(self, '_mpv_pause_cache', True)
                            or float(value) > float(target_s) + 0.001):
                        self._release_hevc_audio_start(float(value))
            # V14b: Ignore during transition
            if getattr(self, '_mpv_transition_in_progress', False):
                return
            # V8 CRASH FIX: Skip entirely during seek to reduce MPV contention
            if getattr(self, '_is_seeking', False):
                return
            self.mpv_time_event.emit((owner, source, value))
        except OSError:
            pass # Ignore access violations during shutdown
        except Exception:
            pass

    @Slot(object)
    def _dispatch_mpv_time(self, payload):
        session_id, core, value = payload
        if not self._session_is_current(session_id, core=core):
            return
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        self._handle_time_update(value)

    def _set_ui_time(self, new_time: float, force: bool = False):
        """Clamp UI time to avoid small backward jumps unless forced (e.g., after seek)."""
        try:
            if new_time is None: return
            new_time = float(new_time)

            # While the user owns the timeline gesture, the slider already
            # displays the requested position. Late mpv/decoder clock callbacks
            # must not overwrite its time label (the slider value itself is
            # protected by isSliderDown(), the label was not). The release/seek
            # path installs the final target immediately afterwards.
            if getattr(self, '_is_scrubbing', False) and not force:
                return

            if force or getattr(self, '_is_seeking', False):
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
        # Per-file memory: keep the resume position fresh (throttled to 15s
        # inside; a crash or kill then costs at most 15s of resume accuracy).
        _rp = getattr(self, '_remember_position', None)
        if _rp is not None:
            _rp()
        # Look-ahead advisory pump (no-op unless a synth3d session runs).
        _pump = getattr(self, '_pump_lookahead_advisory', None)
        if _pump is not None:
            _pump()
        if getattr(self, '_hevc_mode_active', False):
            # Mise à jour du temps du sous-titre via la dernière position connue
            if self._subtitle_manager is not None and self._last_mvc_timestamp > 0:
                self._subtitle_manager.update_time(self._last_mvc_timestamp)
            self._last_timeline_update_time = time.monotonic()
            return

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

        # HEVC native: _on_hevc_position (the decode thread's own pts) is the
        # ONE authoritative UI clock. The mpv time-pos cache is frozen for
        # no-audio media and the synthetic counter runs on a different origin;
        # feeding either alongside the pts made _set_ui_time alternate between
        # clocks more than 1 s apart (the backward filter treats >1 s as a
        # seek) — the slider snapped back and forth at up to ~8 Hz.
        if getattr(self, '_hevc_mode_active', False):
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
                if getattr(self, '_hevc_mode_active', False):
                    # GUI-HOG FIX (py-spy indicted, 2026-07): a synchronous
                    # self.player.time_pos here is a BLOCKING mpv-core read that stalls the
                    # GUI thread ~200 ms while mpv decodes TrueHD Atmos — 44.7% of the GUI
                    # thread's time, one of the two co-causes of the [HEVC-METER] widget
                    # slot-to-slot bursts (p99 ~200 ms). In HEVC mode the time-pos observer
                    # CACHE (_mpv_time_pos_cache, pushed non-blocking by on_time_update on the
                    # mpv event thread) already carries the same value — the HEVC decode thread
                    # itself clocks off it (clock_offset_provider). Read the cache, never mpv.
                    mpv_pos = getattr(self, '_mpv_time_pos_cache', None)
                else:
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

    def on_pause_state_change(self, _, is_paused, session_id=None, core=None):
        """MPV pause state changed - called from MPV event thread!"""
        owner = self._media_session_id if session_id is None else session_id
        source = self.player if core is None else core
        if not self._session_is_current(owner, core=source):
            return
        # Observer-side pause cache: lets GUI-thread code (VU meter poll) know
        # the pause state WITHOUT a blocking mpv property read (0xe24c4a02).
        # Updated even during transitions — it's a plain bool assign.
        new_paused = (is_paused is True or is_paused == 'yes'
                      or is_paused == 'true')
        old_paused = bool(getattr(self, '_mpv_pause_cache', True))
        now = time.monotonic()
        # Freeze the extrapolated clock at the pause edge; restart its monotonic
        # anchor at resume.  No synchronous mpv access occurs on this event thread.
        if new_paused and not old_paused:
            estimate_ms = self._mpv_time_pos_ms()
            if estimate_ms is not None:
                self._mpv_time_pos_cache = estimate_ms / 1000.0
                self._mpv_time_pos_cache_mono = now
        elif not new_paused and old_paused:
            if getattr(self, '_mpv_time_pos_cache', None) is not None:
                self._mpv_time_pos_cache_mono = now
        self._mpv_pause_cache = new_paused
        self.mpv_pause_event.emit((owner, source, is_paused))

    @Slot(object)
    def _dispatch_mpv_pause(self, payload):
        session_id, core, is_paused = payload
        if not self._session_is_current(session_id, core=core):
            return
        if getattr(self, '_mpv_transition_in_progress', False):
            return
        self._handle_pause_change(is_paused, from_observer=True)

    def _handle_pause_change(self, is_paused, from_observer=False):
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

            # HEVC native + a file mpv has NOTHING of (no audio track, vid=no):
            # the mpv shell then pauses BY ITSELF a few seconds in, and that
            # spontaneous report used to freeze the whole transport (is_playing
            # False, timeline timer stopped, hevc thread paused) while the
            # native video kept playing. mpv's own reports carry no meaning for
            # such a file — drop them; user intent still arrives through the
            # direct calls (toggle_play, seek queue, cast).
            if (from_observer and getattr(self, '_hevc_mode_active', False)
                    and getattr(self, '_mpv_shell_inert', False)):
                logger.info("[PAUSE CHANGE] Ignored - inert mpv shell (no audio), "
                            "HEVC native drives the transport")
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
                # V7b FIX: In MVC mode — and HEVC native mode, same native-transport
                # situation — keep the timer active even when paused so the cursor progresses
                if not (self.mvc_mode_active or getattr(self, '_mvc_file_detected', False)
                        or getattr(self, '_hevc_mode_active', False)):
                    self._playback_timer.stop()  # Stop the timeline update
                    logger.info(f"[TIMELINE] Timer stopped (MVC: {self.mvc_mode_active}, detected: {getattr(self, '_mvc_file_detected', False)})")
                else:
                    logger.info(f"[TIMELINE] Timer kept active (MVC: {self.mvc_mode_active}, detected: {getattr(self, '_mvc_file_detected', False)}, hevc: {getattr(self, '_hevc_mode_active', False)})")
                self.show_controls()
                # Notify decoder
                if self.mvc_decoder_thread:
                    self.mvc_decoder_thread.pause()
            else:
                # Resume is a user-visible navigation transition. Refresh the
                # single poller's deadline instead of starting the retired
                # inactivity timer (whose callback is intentionally a no-op).
                self._mark_activity()
                self._playback_timer.start()  # Start the timeline update
                # Notify decoder
                if self.mvc_decoder_thread:
                    self.mvc_decoder_thread.resume()

            # HEVC path: mirror the pause state onto the decode thread (like MVC above).
            if getattr(self, 'hevc_thread', None) is not None:
                try:
                    self.hevc_thread.set_paused(safe_is_paused)
                except Exception:
                    pass

            self.controls_overlay.set_paused(safe_is_paused)
        except Exception as e:
            logger.warning(f"[UI] Error handling pause change: {e}")

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
        # REPLAY: after an end of stream the pipelines are torn down and a bare
        # un-pause has nothing to drive (_handle_pause_change ignores it while
        # _playback_ended is set; the 2D mpv EOF goes through stop_playback,
        # which drops has_media). In both end states the last file path is
        # still known — pressing play relaunches it from the start instead of
        # doing nothing. Also covers play-after-Stop, the same dead state.
        last = getattr(self, 'current_file_path', None)
        if last and (getattr(self, '_playback_ended', False) or not self.has_media):
            if getattr(self, '_is_loading_file', False):
                return  # a (re)load is already in flight — ignore double-clicks
            logger.info(f"[REPLAY] play pressed after end/stop — reloading {last}")
            self.play_file(last)
            return
        if self.has_media and self.player:
            try:
                # self.is_playing is the pause state's single source of truth, not
                # self.player.pause. In MVC mode the native decoder drives the
                # picture while mpv is deliberately kept paused (see "keeping MPV
                # paused until decoder ready"), so mpv.pause reads True during
                # playback. Deriving the toggle from it computed the wrong
                # direction on the first click -- it "un-paused" an already
                # playing stream (the visible micro-glitch is resume()'s clock
                # reset), and only the second click actually paused. is_playing is
                # kept correct across the 2D (mpv observer), MVC and HEVC paths by
                # _handle_pause_change, so it toggles right in every mode.
                new_pause = self.is_playing

                self._safe_mpv_set_pause(new_pause)

                # CRITICAL FIX: Directly handle pause change to ensure immediate video stop
                # Relying solely on MPV callback can be unreliable if MPV thread is busy
                self._handle_pause_change(new_pause)
            except:
                pass

    def _set_playback_stopped_ui(self):
        """Synchronously reset transport state; never wait for an MPV callback."""
        self.is_playing = False
        overlay = getattr(self, 'controls_overlay', None)
        if overlay is not None:
            overlay.set_playback_stopped()

    def stop_playback(self):
        """Stops playback, resets position, and clears decoder state."""
        # Do this before every early return. During teardown MPV pause callbacks
        # are intentionally ignored, so they cannot be the source of truth for
        # the button's stopped appearance.
        self._set_playback_stopped_ui()
        loading = bool(getattr(self, '_is_loading_file', False))
        if not self.has_media and not loading:
            return

        # If MVC EOS handler already ran, the decoder is being / was already torn
        # down via its own 300ms timer. Re-running the full sequence here would
        # trigger a 2nd MVC CLEANUP (visible in the EOS logs) and attempt to
        # terminate an already-paused MPV instance, producing the "Could not
        # restore vo" warning twice.
        if getattr(self, '_playback_ended', False) and not loading:
            logger.info("[PLAYER] stop_playback skipped — _on_mvc_finished already handling teardown")
            return

        # Per-file memory: an explicit Stop is the classic "I'll finish later"
        # — record where they were so the next load offers to resume there.
        _rp = getattr(self, '_remember_position', None)
        if _rp is not None:
            _rp(final=True)

        # Cancel workers and make every queued callback from this title stale
        # before touching mpv/native ownership.
        self._invalidate_media_session('stop')
        self._is_loading_file = False
        self._cancel_media_workers()

        print("[PLAYER] Stopping playback...")
        
        # VISUAL FIX: Hide windows IMMEDIATELY to prevent strobe of old frame
        if hasattr(self, 'framepacking_window') and self.framepacking_window:
            self.framepacking_window.hide()

        # Clear EVERY display widget, not just the framepack/preview pair: Dual
        # Projector's eye windows are not hidden by Stop (they stay open until
        # the user leaves the mode), so an uncleared one holds the last decoded
        # frame frozen on its projector indefinitely.
        for _w in self._display_widgets():
            if hasattr(_w, 'clear_textures'):
                _w.clear_textures()
        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
            if hasattr(self.mvc_embedded_widget, 'clear_textures'):
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
        _released_async = False
        try:
            self._stop_mvc_decoder()

            if self.player:
                try:
                    # V7b STABILITY: Use terminate() instead of pause/seek 0 to fully release file handles
                    # This prevents "file in use" errors and cleans up MPV threads
                    # V61 STABILITY: stop the VU poller and NULL the reference BEFORE the
                    # async re-init — every `if self.player:` guard in the codebase then
                    # routes safely instead of poking a terminated core (access violation
                    # if a load lands inside the re-init window).
                    self._vu_timer.stop()
                    _dying = self.player
                    self.player = None
                    # V62 quiesce-before-destroy, now DEFERRED: 'stop' returns
                    # in ~0 ms and unloads asynchronously; terminate() runs
                    # ~1.5 s later on a cooled core (3 ms measured, vs 737 ms+
                    # hot inline — the GUI-freeze half of AppHangTransient).
                    # Re-init is chained AFTER that terminate instead of the
                    # old fixed 500 ms guess (see _release_mpv_core).
                    self._mpv_transition_in_progress = True
                    self._release_mpv_core(_dying)
                    _released_async = True
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
            if _released_async:
                # Keep the proven ordering: dismount only once the deferred
                # terminate has fully released the core (see _finish_mpv_release).
                self._dismount_after_release = True
            else:
                QTimer.singleShot(400, self._dismount_isos_after_stop)

        self.has_media = False
        self._update_3d_button_state()   # no media → lock the 3D button off
        self.controls_overlay.clear_format_badge()   # drop the 3D-format badge
        self.update_ui_state()
        self.controls_overlay.set_status_info("Ready")
        self.controls_overlay.set_time(0)
        self.controls_overlay.set_duration(0)
        self.setWindowTitle("SyLC 3D Player - Premium Edition")


__all__ = ['PlaybackTimelineMixin']

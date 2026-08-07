# -*- coding: utf-8 -*-
"""Audio-track and subtitle coordination for PlayerWindow."""

import logging
import os
import traceback

from PySide6.QtCore import Slot

from sylc.track_metadata import (
    _find_clpi_for_media, _friendly_track_label, _humanize_lang,
    _parse_clpi_languages,
)


logger = logging.getLogger(__name__)
PGS_SUBTITLE_AVAILABLE = False


def configure_subtitle_support(available):
    """Mirror the optional PGS capability detected by the application boot."""
    global PGS_SUBTITLE_AVAILABLE
    PGS_SUBTITLE_AVAILABLE = bool(available)


class SubtitleTrackMixin:
    def change_audio_track(self, track_id):
        if not (self.has_media and self.player):
            return
        try:
            # mpv.command() serializes via the MPV queue instead of a direct set_property,
            # which avoids the collision with the decoder thread (SEH 0xe24c4a02).
            self.player.command('set', 'aid', str(track_id))
            _rem = getattr(self, '_remember_for_file', None)
            if _rem is not None:
                _rem(audio_track=int(track_id))
            print(f"Audio track changed: ID {track_id}")
        except (OSError, RuntimeError, Exception) as e:
            print(f"Error changing audio track: {e}")

    def load_audio_tracks(self, session_id=None):
        if not self.has_media: return
        owner = self._media_session_id if session_id is None else session_id
        self._media_single_shot(500, lambda: self._fetch_audio_tracks(owner), owner)

    def _get_clpi_lang_map(self):
        """Return {PID: 'iso639'} from the Blu-ray .clpi for the current file (cached).

        Raw M2TS/SSIF streams carry no language tag; the per-PID language lives in
        the disc's CLIPINF/<clip>.clpi. Returns {} for non-BDMV files (e.g. MKV,
        which carry their own language tags that mpv exposes directly).
        """
        path = getattr(self, 'current_file_path', None)
        if not path:
            return {}
        if getattr(self, '_clpi_lang_map_path', None) == path:
            return self._clpi_lang_map
        clpi = _find_clpi_for_media(path)
        self._clpi_lang_map = _parse_clpi_languages(clpi) if clpi else {}
        self._clpi_lang_map_path = path
        if self._clpi_lang_map:
            logger.info(f"[CLPI] Loaded {len(self._clpi_lang_map)} stream languages from {os.path.basename(clpi)}")
        return self._clpi_lang_map

    def _fetch_audio_tracks(self, session_id=None):
        owner = self._media_session_id if session_id is None else session_id
        if not self._session_is_current(owner):
            return
        try:
            if not self.player: return
            track_list = self.player.track_list
            lang_map = self._get_clpi_lang_map()
            audio_tracks = []
            for track in track_list:
                if track.get('type') == 'audio':
                    track_id = track.get('id')
                    label = _friendly_track_label(track, 'audio', lang_map)
                    audio_tracks.append((track_id, label, ''))  # label is already complete
            print(f"Audio tracks found: {len(audio_tracks)}")
            # In the native modes mpv exists ONLY to play the audio. With no
            # audio track it has literally nothing to do (vid=no) and its
            # spontaneous pause/EOF reports are noise — flag it so
            # _handle_pause_change ignores observer-side reports.
            self._mpv_shell_inert = not audio_tracks
            if self._mpv_shell_inert and (self.mvc_mode_active
                                          or getattr(self, '_hevc_mode_active', False)):
                logger.info("[TRANSPORT] no audio track: mpv shell marked inert "
                            "(native pipeline drives the transport)")
                if getattr(self, '_hevc_mode_active', False):
                    th = getattr(self, 'hevc_thread', None)
                    if th is not None:
                        th.set_master_clock_required(False)
                        if getattr(self, 'is_playing', False):
                            th.set_paused(False)
            self.controls_overlay.update_audio_tracks(audio_tracks)
            # Per-file memory: re-apply the remembered audio pick (combo is
            # synced silently; mpv is driven through the normal handler).
            _art = getattr(self, '_apply_remembered_track', None)
            aid = _art(self.controls_overlay.audio_track_combo,
                       'audio_track') if _art else None
            if aid is not None:
                self.change_audio_track(aid)
        except Exception as e:
            print(f"Error fetching audio tracks: {e}")

    def _disable_all_subtitle_outputs(self):
        """Make the UI's ``None`` state true in every subtitle backend.

        Populating a combo with signals blocked does not invoke
        ``change_subtitle_track(0)``.  mpv can therefore retain its automatic
        subtitle choice while the combo visibly says ``None``.  Native PGS and
        text overlays have independent state as well, so switching off only
        mpv is insufficient.  This helper deliberately does not update file
        memory: it is also used as the neutral state while a remembered choice
        is waiting to be restored.
        """
        seen = set()
        for thread in (getattr(self, 'mvc_decoder_thread', None),
                       getattr(self, 'hevc_thread', None)):
            if thread is None or id(thread) in seen:
                continue
            seen.add(id(thread))
            try:
                if hasattr(thread, 'set_subtitle_track'):
                    thread.set_subtitle_track(0)
            except Exception:
                logger.exception("[SUBTITLE] Could not disable native subtitle track")

        self._active_streaming_track = None
        manager = getattr(self, '_subtitle_manager', None)
        if manager is not None:
            try:
                manager.set_enabled(False)
            except Exception:
                logger.exception("[SUBTITLE] Could not disable PGS overlay")
        self._active_pgs_track_index = None
        self._disable_text_subtitles()

        player = getattr(self, 'player', None)
        if player is not None:
            try:
                player['sid'] = 'no'
            except Exception:
                try:
                    player.sid = 'no'
                except Exception:
                    logger.exception("[SUBTITLE] Could not set mpv sid=no")

    def change_subtitle_track(self, track_id):
        logger.info(f"[SUBTITLE] change_subtitle_track called with track_id={track_id}")
        # Per-file memory: the pick is stable per title (same enumeration on
        # every load of the same media). 0 = subtitles off, also remembered.
        # The id space differs between the streaming combo (1-based index) and
        # the plain mpv combo (sid), so the kind is stored with the id and each
        # restore site only applies its own kind.
        _rem = getattr(self, '_remember_for_file', None)
        if _rem is not None and self.has_media:
            try:
                _rem(subtitle_track=int(track_id),
                     subtitle_track_kind=('streaming'
                                          if getattr(self, '_streaming_subtitle_tracks', None)
                                          else 'mpv'))
            except (TypeError, ValueError):
                pass
        hevc_mode_active = bool(getattr(self, '_hevc_mode_active', False))
        native_overlay_mode = bool(self.mvc_mode_active or hevc_mode_active)
        logger.info(f"[SUBTITLE] has_media={self.has_media}, mvc_mode_active={self.mvc_mode_active}, "
                    f"hevc_mode_active={hevc_mode_active}")
        logger.info(f"[SUBTITLE] PGS_AVAILABLE={PGS_SUBTITLE_AVAILABLE}, manager={self._subtitle_manager is not None}")
        logger.info(f"[SUBTITLE] PGS tracks detected: {len(self._pgs_subtitle_tracks)}")
        logger.info(f"[SUBTITLE] Streaming tracks detected: {len(self._streaming_subtitle_tracks)}")

        if self.has_media and self.player:
            try:
                # ``None`` in the combo is a real state, not merely its label.
                # Handle it before track detection/extraction so selecting it
                # cannot accidentally wake the first PGS track.
                if track_id == 0:
                    self._disable_all_subtitle_outputs()
                    logger.info("[SUBTITLE] Subtitles disabled")
                    return

                # ========== STREAMING SUBTITLE PATH (No extraction delay!) ==========
                # Streaming tracks come from the MVC demuxer (MKV/M2TS) OR from the
                # lavf HEVC source (same signal contract) — one branch serves both.
                native_thread = (self.mvc_decoder_thread
                                 if (self.mvc_mode_active and self.mvc_decoder_thread)
                                 else (getattr(self, 'hevc_thread', None)
                                       if hevc_mode_active else None))
                if native_thread is not None and self._streaming_subtitle_tracks:
                    # UI sends 1-based index, find the corresponding streaming track
                    # track_id 1 = first streaming track, track_id 2 = second, etc.
                    streaming_track = None
                    if track_id > 0 and track_id <= len(self._streaming_subtitle_tracks):
                        streaming_track = self._streaming_subtitle_tracks[track_id - 1]
                        logger.info(f"[STREAMING-SUBS] track_id={track_id} -> trackNumber={streaming_track.get('trackNumber')}")

                    if streaming_track and streaming_track.get('isPGS', False):
                        actual_track_number = streaming_track.get('trackNumber')
                        logger.info(f"[STREAMING-SUBS] Enabling streaming for track {actual_track_number}: {streaming_track.get('name')}")

                        # Enable streaming in the native decode thread (MVC or HEVC)
                        native_thread.set_subtitle_track(actual_track_number)
                        self._active_streaming_track = actual_track_number

                        # Configure SubtitleManager for streaming
                        if self._subtitle_manager:
                            # V7b++ STUTTER FIX: Connect PGS streaming signal NOW (deferred from MVC init)
                            if not getattr(self, '_pgs_streaming_connected', False):
                                if hasattr(native_thread, 'pgsDataReady'):
                                    native_thread.pgsDataReady.connect(self._subtitle_manager.on_pgs_data)
                                    self._pgs_streaming_connected = True
                                    logger.info("[STREAMING-SUBS] Connected pgsDataReady signal (deferred)")

                            self._subtitle_manager.start_streaming()
                            video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                            video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                            self._subtitle_manager.set_video_dimensions(video_w, video_h)
                            self._subtitle_manager.set_enabled(True)

                            # Connect to display widget
                            display_widget = getattr(self, 'active_mvc_widget', None)
                            logger.info(f"[STREAMING-SUBS] active_mvc_widget = {display_widget}")
                            if not display_widget:
                                if hasattr(self, 'framepacking_window') and self.framepacking_window:
                                    display_widget = self.framepacking_window.display_widget
                                    logger.info(f"[STREAMING-SUBS] Using framepacking_window.display_widget = {display_widget}")
                                elif hasattr(self, 'mvc_embedded_widget'):
                                    display_widget = self.mvc_embedded_widget
                                    logger.info(f"[STREAMING-SUBS] Using mvc_embedded_widget = {display_widget}")
                            if display_widget:
                                logger.info(f"[STREAMING-SUBS] Connecting subtitle manager to {display_widget.__class__.__name__}")
                                self._connect_subtitle_to_widget(display_widget)
                            else:
                                logger.error("[STREAMING-SUBS] No display widget found for subtitle connection!")

                        # BD3D authored depth: route this PG stream's offset
                        # sequence (STN_table_SS) + the per-GOP OFMD depth to
                        # the overlay. No-ops outside BD3D (map empty).
                        try:
                            seq = (getattr(self, '_bd3d_pg_offset_map', None) or {}).get(actual_track_number)
                            if seq is not None and hasattr(native_thread, 'set_pg_offset_sequence'):
                                native_thread.set_pg_offset_sequence(seq)
                            if (not getattr(self, '_pg_depth_connected', False)
                                    and hasattr(native_thread, 'pgDepthChanged')):
                                native_thread.pgDepthChanged.connect(self._on_pg_depth_changed)
                                self._pg_depth_connected = True
                        except Exception as _e:
                            logger.warning(f"[BD3D-DEPTH] wiring skipped: {_e}")

                        # Disable MPV subtitles + any text overlay
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        self.show_3d_notification(f"Streaming: {streaming_track.get('name')}", success=True)
                        return

                    elif streaming_track and self._text_subtitle_renderer is not None \
                            and (str(streaming_track.get('codecId', '')).upper().startswith('S_TEXT')
                                 or str(streaming_track.get('codecId', '')).upper() in ('S_ASS', 'S_SSA')):
                        # ===== TEXT SUBTITLE PATH (SRT / ASS / SSA) =====
                        # mpv runs audio-only here (vid=no, vo=null) so it cannot draw
                        # text subs itself, but it still decodes the selected track on
                        # the shared audio clock: select it and paint 'sub-text' on the
                        # native overlay (same widget path as PGS).
                        self._enable_text_subtitle_track(track_id, streaming_track)
                        return

                    elif track_id == 0:
                        # Disable streaming
                        logger.info("[STREAMING-SUBS] Disabling subtitle streaming")
                        native_thread.set_subtitle_track(0)
                        self._active_streaming_track = None
                        if self._subtitle_manager:
                            self._subtitle_manager.set_enabled(False)
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        return
                # ====================================================================

                # Check if this is a PGS track on either native-video path. mpv is
                # audio-only in both MVC/edge264 and avcodec HEVC modes, so selecting
                # its sid cannot draw bitmap subtitles; extract and feed the existing
                # high-quality native overlay instead.
                # Only use extraction if MVC streaming didn't handle it.
                logger.info(f"[SUBTITLE] Streaming path did not handle track_id={track_id}, trying legacy extraction...")
                if native_overlay_mode and self._subtitle_manager and PGS_SUBTITLE_AVAILABLE:
                    # If PGS detection hasn't completed yet, do it synchronously now
                    if len(self._pgs_subtitle_tracks) == 0 and self._subtitle_extractor and self.current_file_path:
                        logger.info("[PGS] No PGS tracks cached, detecting synchronously...")
                        try:
                            self._pgs_subtitle_tracks = self._subtitle_extractor.detect_subtitle_tracks(self.current_file_path)
                            pgs_count = sum(1 for t in self._pgs_subtitle_tracks if t.is_pgs)
                            logger.info(f"[PGS] Synchronous detection found {pgs_count} PGS tracks")
                        except Exception as e:
                            logger.error(f"[PGS] Synchronous detection failed: {e}")

                    logger.info(f"[PGS] Looking for track_id={track_id} in {len(self._pgs_subtitle_tracks)} PGS tracks")
                    # Find if track_id corresponds to a PGS track
                    pgs_track = None
                    for pt in self._pgs_subtitle_tracks:
                        logger.info(f"[PGS]   - track_id={pt.track_id}, index={pt.index}, is_pgs={pt.is_pgs}, lang={pt.language}")
                        # Match by display track_id (1-based)
                        if pt.track_id == track_id:
                            pgs_track = pt
                            break

                    if pgs_track and pgs_track.is_pgs:
                        # Check if this track was pre-extracted at startup
                        cached_index = getattr(self, '_cached_pgs_track_index', None)
                        is_loaded = self._subtitle_manager.is_loaded if self._subtitle_manager else False
                        force_reparse = False  # Use cache for faster subtitle loading
                        logger.debug(f"[PGS] Cache check: cached={cached_index}, track={pgs_track.index}, loaded={is_loaded}")
                        if cached_index == pgs_track.index and is_loaded and not force_reparse:
                            # Use pre-extracted subtitles - instant activation!
                            logger.info(f"[PGS] Using PRE-EXTRACTED subtitles for track {track_id}")
                            video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                            video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                            self._subtitle_manager.set_video_dimensions(video_w, video_h)
                            self._subtitle_manager.set_enabled(True)
                            self._active_pgs_track_index = pgs_track.index
                            # Connect to the correct display widget (use active widget from decoder)
                            display_widget = getattr(self, 'active_mvc_widget', None)
                            if not display_widget:
                                if hasattr(self, 'framepacking_window') and self.framepacking_window:
                                    display_widget = self.framepacking_window.display_widget
                                elif hasattr(self, 'mvc_embedded_widget'):
                                    display_widget = self.mvc_embedded_widget
                            logger.debug(f"[PGS] Using display widget: {display_widget.__class__.__name__ if display_widget else 'None'}")
                            self._connect_subtitle_to_widget(display_widget)
                            self.show_3d_notification(f"Subtitles: {self._subtitle_manager.subtitle_count} cues", success=True)
                        else:
                            # Need to extract this track (not pre-extracted)
                            logger.info(f"[PGS] Track {pgs_track.index} not pre-extracted, extracting now...")
                            self._load_pgs_subtitle_track(pgs_track)
                        # Disable MPV's internal subtitles + any text overlay
                        self._disable_text_subtitles()
                        self.player.sid = 'no'
                        logger.info(f"[PGS] Using PGS overlay for track {track_id}")
                        return

                # Default: Use MPV's subtitle handling
                if track_id == 0:
                    self.player.sid = 'no'
                    # Also disable PGS overlay + text overlay
                    if self._subtitle_manager:
                        self._subtitle_manager.set_enabled(False)
                        self._active_pgs_track_index = None
                    self._disable_text_subtitles()
                    logger.info("[SUBTITLE] Subtitles disabled")
                else:
                    self.player.sid = track_id
                    # Disable PGS overlay when using MPV subtitles
                    if self._subtitle_manager:
                        self._subtitle_manager.set_enabled(False)
                        self._active_pgs_track_index = None
                    if native_overlay_mode and self._text_subtitle_renderer is not None:
                        # edge264/native-renderer playback without a streaming track
                        # list (e.g. MP4 via lavf demuxer): mpv is audio-only and
                        # cannot draw its subs — mirror them onto the native overlay.
                        # Here the combo was filled from mpv's track-list, so
                        # track_id IS the mpv sid.
                        self._activate_text_overlay(track_id)
                    else:
                        self._disable_text_subtitles()
                    logger.info(f"[SUBTITLE] track changed: ID {track_id}")
            except Exception as e:
                print(f"Error changing subtitle track: {e}")

    def _load_pgs_subtitle_track(self, pgs_track):
        """Load a PGS subtitle track for MVC overlay rendering (async)."""
        if not self._subtitle_extractor or not self._subtitle_manager:
            logger.warning("[PGS] Missing extractor or manager, cannot load subtitle")
            return

        # Show notification that extraction is starting (can take 1-2 minutes)
        self.show_3d_notification("Extracting subtitles (1-2 min)...", success=True)
        logger.info(f"[PGS] Starting extraction for track {pgs_track.index}")

        # Capture every input now.  A worker must never consult mutable
        # ``current_file_path`` after another title has begun loading.
        import time as _time
        extraction_start = _time.time()
        session_id = self._media_session_id
        file_path = self.current_file_path
        extractor = self._subtitle_extractor

        def extract_and_load(cancel_event):
            try:
                logger.info(f"[PGS] Extracting track {pgs_track.index} from {file_path}")
                logger.info("[PGS] This may take 1-2 minutes for large files...")

                # Extract PGS data to temp file
                sup_path = extractor.extract_pgs_track(file_path, pgs_track.index)

                elapsed = _time.time() - extraction_start
                logger.info(f"[PGS] Extraction completed in {elapsed:.1f}s, result: {sup_path}")
                if cancel_event.is_set() or not self._session_is_current(session_id):
                    return

                # Schedule loading on main thread via signal (thread-safe)
                if sup_path:
                    self.pgs_load_complete.emit({
                        'session': session_id,
                        'file_path': file_path,
                        'sup_path': sup_path,
                        'track_index': pgs_track.index,
                    })
                else:
                    logger.error("[PGS] Extraction returned None")
                    self.pgs_notification.emit({
                        'session': session_id,
                        'message': "Subtitle extraction failed",
                        'success': False,
                    })
            except Exception as e:
                logger.error(f"[PGS] Error extracting subtitle track: {e}")
                import traceback
                traceback.print_exc()
                if not cancel_event.is_set():
                    self.pgs_notification.emit({
                        'session': session_id,
                        'message': "Subtitle error",
                        'success': False,
                    })

        self._start_media_worker(
            extract_and_load, session_id=session_id, name='pgs-track-extractor')

    @Slot(object)
    def _finish_pgs_load(self, payload):
        """Load PGS file in background thread after extraction."""
        session_id = payload.get('session')
        if not self._session_is_current(session_id):
            return
        sup_path = payload.get('sup_path')
        track_index = payload.get('track_index')
        file_path = payload.get('file_path')
        logger.info(f"[PGS] _finish_pgs_load called with {sup_path}")
        self.show_3d_notification("Parsing subtitles...", success=True)

        def parse_pgs(cancel_event):
            try:
                from sylc.pgs_subtitle_parser import PGSSubtitleParser
                logger.info(f"[PGS] Parsing subtitle file: {sup_path}")
                parser = PGSSubtitleParser()
                success = parser.load_from_file(sup_path)
                logger.info(f"[PGS] Parse result: {success}")
                if cancel_event.is_set() or not self._session_is_current(session_id):
                    return
                self.pgs_parse_complete.emit({
                    'session': session_id,
                    'file_path': file_path,
                    'sup_path': sup_path,
                    'track_index': track_index,
                    'parser': parser if success else None,
                })
            except Exception as e:
                logger.error(f"[PGS] Error parsing subtitle file: {e}")
                import traceback
                traceback.print_exc()
                if not cancel_event.is_set():
                    self.pgs_notification.emit({
                        'session': session_id,
                        'message': "Parse error",
                        'success': False,
                    })

        self._start_media_worker(
            parse_pgs, session_id=session_id, name='pgs-track-parser')

    @Slot(object)
    def _on_pgs_parsed(self, payload):
        """Called on main thread when PGS parsing completes."""
        session_id = payload.get('session')
        if not self._session_is_current(session_id):
            return
        parser = payload.get('parser')
        track_index = payload.get('track_index')
        success = parser is not None
        logger.info(f"[PGS] _on_pgs_parsed called: success={success}, track_index={track_index}")
        try:
            if success:
                success = self._subtitle_manager.install_parser(
                    parser, payload.get('sup_path') or '<background>')
            if success:
                # Set video dimensions for coordinate normalization
                video_w = self.video_3d_info.get('width', 1920) if self.video_3d_info else 1920
                video_h = self.video_3d_info.get('height', 1080) if self.video_3d_info else 1080
                logger.info(f"[PGS] Setting video dimensions: {video_w}x{video_h}")
                self._subtitle_manager.set_video_dimensions(video_w, video_h)
                self._subtitle_manager.set_enabled(True)
                self._active_pgs_track_index = track_index
                count = self._subtitle_manager.subtitle_count

                # Connect to the correct display widget (use active widget from decoder)
                display_widget = getattr(self, 'active_mvc_widget', None)
                if not display_widget:
                    if hasattr(self, 'framepacking_window') and self.framepacking_window:
                        display_widget = self.framepacking_window.display_widget
                    elif hasattr(self, 'mvc_embedded_widget'):
                        display_widget = self.mvc_embedded_widget
                logger.debug(f"[PGS] Using display widget: {display_widget.__class__.__name__ if display_widget else 'None'}")
                self._connect_subtitle_to_widget(display_widget)

                logger.info(f"[PGS] Loaded {count} subtitle cues")
                self.show_3d_notification(f"Subtitles: {count} cues", success=True)
            else:
                logger.error(f"[PGS] Failed to load subtitle track {track_index}")
                self.show_3d_notification("Failed to parse subtitles", success=False)
        except Exception as e:
            logger.error(f"[PGS] Error finishing subtitle load: {e}")
            import traceback
            traceback.print_exc()

    @Slot(object)
    def _on_pgs_notification(self, payload):
        if not isinstance(payload, dict):
            return
        if not self._session_is_current(payload.get('session')):
            return
        self.show_3d_notification(
            payload.get('message') or 'Subtitle error',
            success=bool(payload.get('success')))

    def _connect_subtitle_to_widget(self, widget=None):
        """Connect SubtitleManager signals to EVERY active MVC display widget — the embedded
        2D view in the main window, the separate 3D FramePack window AND, in Dual Projector
        mode, each eye window — so PGS subtitles appear on all of them, not only the active
        one (in 3D mode the embedded 2D view stays visible for sync, so it needs the overlay
        too)."""
        if not self._subtitle_manager:
            return

        try:
            # Gather every display widget that can render a subtitle overlay (dedup, keep order)
            widgets = []
            for w in (*self._display_widgets(), widget):
                if (w is not None and w not in widgets
                        and hasattr(w, 'set_subtitle') and hasattr(w, 'clear_subtitle')):
                    widgets.append(w)
            if not widgets:
                logger.warning("[PGS] No display widget with subtitle methods to connect")
                return

            # Skip if already connected to exactly this set
            if getattr(self, '_subtitle_connected_widgets', None) == widgets:
                return

            # Drop any previous connections, then connect every gathered widget.
            # PySide6 signale un disconnect() sans connexion par un RuntimeWarning
            # (pas une exception) : ce cas est ATTENDU ici — premier branchement,
            # ou reconnexion après le reset de _subtitle_connected_widgets qui
            # laisse volontairement les connexions en place pour cette purge.
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    self._subtitle_manager.subtitle_changed.disconnect()
                    self._subtitle_manager.subtitle_cleared.disconnect()
                except (TypeError, RuntimeError):
                    pass

            def make_setter(target_widget):
                def setter(rgba, x, y, width, height, vw, vh, disparity=0.0):
                    try:
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh,
                                                   disparity)
                    except TypeError:
                        # widget predating the disparity parameter
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh)
                    except Exception as e:
                        logger.error(f"[PGS] set_subtitle error on {target_widget.__class__.__name__}: {e}")
                return setter

            for w in widgets:
                self._subtitle_manager.subtitle_changed.connect(make_setter(w))
                self._subtitle_manager.subtitle_cleared.connect(w.clear_subtitle)
            self._subtitle_connected_widgets = widgets
            logger.info(f"[PGS] CONNECTED subtitles to {[w.__class__.__name__ for w in widgets]}")
        except Exception as e:
            logger.error(f"[PGS] Error connecting subtitle manager: {e}")

    def _enable_text_subtitle_track(self, ui_index, streaming_track):
        """Route a text subtitle track (S_TEXT/UTF8, S_TEXT/ASS…) to the native overlay.

        mpv plays the same file for audio and decodes the selected text track on
        that clock even without video output; its 'sub-text' property carries the
        current cue (ASS override tags already stripped, exact show/hide timing).
        """
        logger.info(f"[TEXT-SUBS] Enabling text track ui_index={ui_index}: "
                    f"{streaming_track.get('codecId')} ({streaming_track.get('name')})")

        # Text subs don't use the demuxer streaming queue nor the PGS overlay
        if self.mvc_decoder_thread:
            self.mvc_decoder_thread.set_subtitle_track(0)
        self._active_streaming_track = None
        if self._subtitle_manager:
            self._subtitle_manager.set_enabled(False)
        self._active_pgs_track_index = None

        # Map the menu index (1-based, MKV file order) to mpv's sid. mpv numbers
        # subtitle tracks in the same file order, so the Nth entry matches sid N —
        # resolved through track-list rather than assumed, when possible.
        sid = ui_index
        try:
            subs = [t for t in (self.player.track_list or []) if t.get('type') == 'sub']
            if 0 < ui_index <= len(subs):
                sid = subs[ui_index - 1].get('id', ui_index)
        except Exception as e:
            logger.warning(f"[TEXT-SUBS] track-list sid mapping failed, using index: {e}")

        self._activate_text_overlay(sid, streaming_track.get('name'))

        # Recover the authored 3D depth of this track (per-eye duplicated cues
        # encode a parallax). Cached per (file, track); analysis is a bounded
        # ffprobe sampling pass, run off the GUI thread.
        sub_index = ui_index - 1   # ffprobe s:N == Nth subtitle track in file order
        cache_key = (self.current_file_path, sub_index)
        self._active_text_sub_depth_key = cache_key
        if cache_key in self._sub_depth_cache:
            self._text_subtitle_renderer.set_disparity(self._sub_depth_cache[cache_key])
        else:
            self._text_subtitle_renderer.set_disparity(0.0)   # flat until measured
            layout = 'tab' if (self.video_3d_info or {}).get('stereo_mode') == 'tab' else 'sbs'
            filepath = self.current_file_path

            session_id = self._media_session_id

            def analyze_depth(cancel_event):
                try:
                    from sylc.subtitle_depth_analyzer import analyze_text_track_depth
                    d, pairs = analyze_text_track_depth(filepath, sub_index, layout)
                    if cancel_event.is_set() or not self._session_is_current(session_id):
                        return
                    self.text_sub_depth_ready.emit({
                        'session': session_id,
                        'file_path': filepath,
                        'sub_index': sub_index,
                        'disparity': d,
                        'pairs': pairs,
                    })
                except Exception as e:
                    logger.warning(f"[SUB-DEPTH] background analysis failed: {e}")

            self._start_media_worker(
                analyze_depth, session_id=session_id, name='sub-depth-analyzer')

    def _activate_text_overlay(self, sid, name=''):
        """Select mpv sid and mirror its 'sub-text' onto the native overlay."""
        # Register the sub-text observer once (fires on mpv's event thread; the
        # Qt signal marshals onto the main thread where QPainter is legal).
        if not self._mpv_subtext_observer_registered:
            session_id = self._media_session_id
            core = self.player

            def _on_subtext(_name, value, owner=session_id, source=core):
                try:
                    self.mpv_sub_text_changed.emit((owner, source, value or ''))
                except Exception:
                    pass
            try:
                core.observe_property('sub-text', _on_subtext)
                self._mpv_subtext_observer_registered = True
                self._mpv_subtext_observer = (core, _on_subtext)
                logger.info("[TEXT-SUBS] sub-text observer registered")
            except Exception as e:
                logger.error(f"[TEXT-SUBS] Could not observe sub-text: {e}")
                return

        try:
            self.player['sub-visibility'] = True
        except Exception:
            pass
        self.player['sid'] = sid
        self._text_sub_active = True
        self._connect_text_subtitle_to_widget()
        logger.info(f"[TEXT-SUBS] Active: mpv sid={sid}")
        if name:
            self.show_3d_notification(f"Subtitles: {name}", success=True)

    def _disable_text_subtitles(self):
        """Stop feeding the text overlay and clear any cue still on screen."""
        if self._text_sub_active:
            logger.info("[TEXT-SUBS] Disabled")
        self._text_sub_active = False
        self._active_text_sub_depth_key = None
        if self._text_subtitle_renderer:
            self._text_subtitle_renderer.clear()

    @Slot(object)
    def _on_mpv_sub_text(self, payload):
        """Main-thread handler for mpv 'sub-text' changes."""
        session_id, core, text = payload
        if not self._session_is_current(session_id, core=core):
            return
        if self._text_sub_active and self._text_subtitle_renderer:
            self._text_subtitle_renderer.set_text(text)

    @Slot(float)
    def _on_pg_depth_changed(self, disparity):
        """Apply the BD3D per-GOP authored PG depth to every display widget."""
        # Latched so an eye window opened LATER (Dual Projector is picked mid-film)
        # starts at the depth currently in force instead of waiting for the next GOP.
        self._pg_depth_last = disparity
        for w in (getattr(self, 'active_mvc_widget', None), *self._display_widgets()):
            if w is not None and hasattr(w, 'set_subtitle_depth'):
                w.set_subtitle_depth(disparity)
        if not getattr(self, '_pg_depth_logged', False):
            self._pg_depth_logged = True
            logger.info(f"[BD3D-DEPTH] Authored PG depth active (first value: {disparity:+.4f})")

    @Slot(object)
    def _on_text_sub_depth(self, payload):
        """Apply the measured authored subtitle depth (main thread)."""
        session_id = payload.get('session')
        if not self._session_is_current(session_id):
            return
        result_key = (payload.get('file_path'), payload.get('sub_index'))
        if result_key != getattr(self, '_active_text_sub_depth_key', None):
            return
        disparity = payload.get('disparity', 0.0)
        pairs = payload.get('pairs', 0)
        self._sub_depth_cache[result_key] = disparity
        if not self._text_sub_active or not self._text_subtitle_renderer:
            return
        self._text_subtitle_renderer.set_disparity(disparity)
        if disparity:
            logger.info(f"[SUB-DEPTH] Applying authored depth: {disparity:+.4f} "
                        f"eye-width ({pairs} pairs)")
            self.show_3d_notification(
                f"3D subtitle depth: {disparity * 100:+.1f}% (authored)", success=True)

    def _connect_text_subtitle_to_widget(self):
        """Wire the text renderer to every active MVC display widget (same set as PGS)."""
        if not self._text_subtitle_renderer:
            return
        try:
            widgets = []
            for w in (getattr(self, 'active_mvc_widget', None), *self._display_widgets()):
                if (w is not None and w not in widgets
                        and hasattr(w, 'set_subtitle') and hasattr(w, 'clear_subtitle')):
                    widgets.append(w)
            if not widgets:
                logger.warning("[TEXT-SUBS] No display widget with subtitle methods to connect")
                return
            if self._text_sub_connected_widgets == widgets:
                return

            try:
                self._text_subtitle_renderer.subtitle_changed.disconnect()
                self._text_subtitle_renderer.subtitle_cleared.disconnect()
            except (TypeError, RuntimeError):
                pass

            def make_setter(target_widget):
                def setter(rgba, x, y, width, height, vw, vh, disparity):
                    try:
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh,
                                                   disparity)
                    except TypeError:
                        # widget predating the disparity parameter
                        target_widget.set_subtitle(rgba, x, y, width, height, vw, vh)
                    except Exception as e:
                        logger.error(f"[TEXT-SUBS] set_subtitle error on {target_widget.__class__.__name__}: {e}")
                return setter

            for w in widgets:
                self._text_subtitle_renderer.subtitle_changed.connect(make_setter(w))
                self._text_subtitle_renderer.subtitle_cleared.connect(w.clear_subtitle)
            self._text_sub_connected_widgets = widgets
            logger.info(f"[TEXT-SUBS] CONNECTED to {[w.__class__.__name__ for w in widgets]}")
        except Exception as e:
            logger.error(f"[TEXT-SUBS] Error connecting text renderer: {e}")

    def load_subtitle_tracks(self, session_id=None):
        logger.info(f"[SUBTITLE] load_subtitle_tracks called, has_media={self.has_media}")
        if not self.has_media: return
        owner = self._media_session_id if session_id is None else session_id
        self._media_single_shot(
            500, lambda: self._fetch_subtitle_tracks(owner), owner)

    def _fetch_subtitle_tracks(self, session_id=None):
        logger.info("[SUBTITLE] _fetch_subtitle_tracks called")
        owner = self._media_session_id if session_id is None else session_id
        if not self._session_is_current(owner):
            return
        try:
            if not self.player:
                logger.info("[SUBTITLE] player is None, returning")
                return
            track_list = self.player.track_list
            logger.info(f"[SUBTITLE] track_list has {len(track_list)} tracks")
            lang_map = self._get_clpi_lang_map()
            subtitle_tracks = []
            for track in track_list:
                logger.debug(f"[SUBTITLE] track type={track.get('type')}")
                if track.get('type') == 'sub':
                    track_id = track.get('id')
                    label = _friendly_track_label(track, 'sub', lang_map)
                    subtitle_tracks.append((track_id, label, ''))  # label is already complete
                    logger.info(f"[SUBTITLE]   Found subtitle: id={track_id}, label={label}")
            logger.info(f"[SUBTITLE] Subtitle tracks found: {len(subtitle_tracks)}")
            self.controls_overlay.update_subtitle_tracks(subtitle_tracks)
            # Per-file memory: mpv-sid picks are restored here; streaming picks
            # belong to the streaming population site (different id space).
            _art = getattr(self, '_apply_remembered_track', None)
            sid = None
            if _art and (getattr(self, '_file_memory', None) or {}).get(
                    'subtitle_track_kind') == 'mpv':
                sid = _art(self.controls_overlay.subtitle_track_combo,
                           'subtitle_track')
                if sid:
                    self.change_subtitle_track(sid)
            if not sid:
                # update_subtitle_tracks() blocks combo signals, so index 0
                # cannot disable mpv by itself. Keep backend and UI aligned.
                self._disable_all_subtitle_outputs()

            # Also detect PGS tracks for the native MVC/HEVC overlay
            # (async to avoid blocking the UI).
            if PGS_SUBTITLE_AVAILABLE and self._subtitle_extractor and self.current_file_path:
                filepath = self.current_file_path
                extractor = self._subtitle_extractor

                def detect_pgs(cancel_event):
                    try:
                        tracks = extractor.detect_subtitle_tracks(filepath)
                        if cancel_event.is_set() or not self._session_is_current(owner):
                            return
                        self.pgs_tracks_detected.emit({
                            'session': owner,
                            'file_path': filepath,
                            'tracks': tracks,
                        })
                    except Exception as e:
                        logger.error(f"[PGS] Detection error: {e}")
                self._start_media_worker(
                    detect_pgs, session_id=owner, name='pgs-track-detector')
        except Exception as e:
            logger.error(f"Error fetching subtitle tracks: {e}")

    @Slot(object)
    def _on_pgs_tracks_detected(self, payload):
        """Called when PGS track detection completes."""
        if not self._session_is_current(payload.get('session')):
            return
        tracks = payload.get('tracks') or []
        self._pgs_subtitle_tracks = tracks
        pgs_count = sum(1 for t in tracks if t.is_pgs)
        if pgs_count > 0:
            logger.info(f"[PGS] Detected {pgs_count} PGS subtitle tracks")

    def _on_subtitle_tracks_detected(self, tracks):
        """Called when MVC decoder detects subtitle tracks (streaming mode).

        These tracks can be streamed in real-time without extraction delay.
        """
        if not self._native_signal_is_current():
            return
        if not tracks:
            return

        self._streaming_subtitle_tracks = tracks
        pgs_tracks = [t for t in tracks if t.get('isPGS', False)]

        # Raw Blu-ray M2TS/SSIF carries no language tag in the PMT — enrich the PID-only PGS
        # tracks with the language from the .clpi ProgramInfo (cached), so the menu shows the
        # language (e.g. "French (PID 0x1200)") instead of just the raw PID.
        try:
            lang_map = self._get_clpi_lang_map()
            if lang_map:
                seen = {}
                for i, t in enumerate(tracks):
                    lang = lang_map.get(t.get('trackNumber'), '')
                    if lang:
                        t['language'] = lang
                        base = _humanize_lang(lang) or lang.upper()
                    else:
                        base = f"Subtitle {i + 1}"
                    # label by language only — no PID; a counter disambiguates same-language dupes
                    seen[base] = seen.get(base, 0) + 1
                    t['name'] = base if seen[base] == 1 else f"{base} {seen[base]}"
        except Exception as e:
            logger.warning(f"[STREAMING-SUBS] CLPI language label skipped: {e}")

        logger.info(f"[STREAMING-SUBS] Detected {len(tracks)} subtitle tracks ({len(pgs_tracks)} PGS)")
        for t in tracks:
            logger.info(f"  - {t.get('name')} (lang={t.get('language') or '?'})")
        for t in tracks:
            logger.info(f"  - Track {t.get('trackNumber')}: {t.get('name')} (PGS={t.get('isPGS')})")

        # Update subtitle track menu in controls overlay
        if hasattr(self, 'controls_overlay') and hasattr(self.controls_overlay, 'update_subtitle_tracks_streaming'):
            self.controls_overlay.update_subtitle_tracks_streaming(tracks)
            # Per-file memory: re-apply the remembered subtitle pick for this
            # title (streaming enumeration is stable per media).
            _art = getattr(self, '_apply_remembered_track', None)
            sid = None
            if _art and (getattr(self, '_file_memory', None) or {}).get(
                    'subtitle_track_kind') == 'streaming':
                sid = _art(self.controls_overlay.subtitle_track_combo,
                           'subtitle_track')
                if sid:
                    self.change_subtitle_track(sid)
            if not sid:
                # The streaming combo is also populated with signals blocked.
                # Explicitly disable the decoder, PGS manager, text renderer
                # and mpv whenever the visible selection remains None.
                self._disable_all_subtitle_outputs()

        # Show notification
        if pgs_tracks:
            self.show_3d_notification(f"Streaming: {len(pgs_tracks)} PGS tracks", success=True)



__all__ = ['SubtitleTrackMixin', 'configure_subtitle_support']

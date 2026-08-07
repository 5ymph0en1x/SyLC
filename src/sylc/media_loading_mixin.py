"""Transactional media loading, source resolution and startup analysis."""

import logging
import os
import time
import traceback

from PySide6.QtCore import QEventLoop, QTimer, Slot
from PySide6.QtWidgets import QApplication

from sylc.stereo_eye_order import UNKNOWN, normalise_eye_order
from sylc.video_3d_analyzer import Video3DAnalyzer

logger = logging.getLogger(__name__)

MVC_SUPPORT_AVAILABLE = False
NATIVE_RENDER_AVAILABLE = False
PGS_SUBTITLE_AVAILABLE = False
EDGE264_CONTAINERS = ()


def configure_media_loading_support(
        mvc_available, native_render_available, pgs_available,
        edge264_containers):
    """Inject optional runtime capabilities discovered by the entry point."""
    global MVC_SUPPORT_AVAILABLE, NATIVE_RENDER_AVAILABLE
    global PGS_SUBTITLE_AVAILABLE, EDGE264_CONTAINERS
    MVC_SUPPORT_AVAILABLE = bool(mvc_available)
    NATIVE_RENDER_AVAILABLE = bool(native_render_available)
    PGS_SUBTITLE_AVAILABLE = bool(pgs_available)
    EDGE264_CONTAINERS = tuple(edge264_containers or ())


class MediaLoadingMixin:
    def _dismount_pending_iso(self):
        """Dismount an ISO we just mounted but couldn't use (best-effort)."""
        try:
            from sylc import bluray_disc
            pending = getattr(self, '_pending_iso_mount', None)
            if pending:
                logger.info(f"[DISC] Dismounting unused ISO: {pending[0]}")
                bluray_disc.dismount_iso(pending[0])
                self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] Dismount (pending) failed: {e}")

    def _dismount_isos_after_stop(self):
        """STOP released all readers (decoder, mpv terminate, thumbnail service
        release_file) — return the mounted ISO(s) to the system. EX-4 fix #2: a
        mount still being read by a running export job is left mounted (deferred
        via _dismount_iso_or_defer) instead of being yanked out from under it."""
        try:
            for m in (getattr(self, '_active_iso_mount', None),
                      getattr(self, '_pending_iso_mount', None)):
                self._dismount_iso_or_defer(m, label='ISO on stop')
            self._active_iso_mount = None
            self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] Dismount on stop failed: {e}")

    def _promote_iso_mount(self):
        """After the previous file's handles are released: dismount the previously
        active ISO (when switching away from it) and promote the just-mounted one to
        active. Best-effort — a stuck mount must never block playback. EX-4 fix #2:
        if a running export job is still reading the previous ISO, its dismount is
        deferred (not lost) until the job ends."""
        try:
            pending = getattr(self, '_pending_iso_mount', None)
            active = getattr(self, '_active_iso_mount', None)
            if active and (not pending or active[0] != pending[0]):
                # NEVER dismount the ISO that hosts the file being (re)loaded —
                # the fresh-mount retry replays the SAME D:\...ssif path with no
                # pending mount, and dismounting here killed the retry
                # ("Failed to open SSIF file", measured 2026-07-14 on Avatar).
                cur = getattr(self, 'current_file_path', None) or ''
                mnt_letter = str(active[1] or '').rstrip('\\').rstrip(':')[:1].upper()
                cur_drive = os.path.splitdrive(os.path.abspath(cur))[0] if cur else ''
                cur_letter = cur_drive[:1].upper() if cur_drive else ''
                if mnt_letter and cur_letter == mnt_letter:
                    logger.info(f"[DISC] Keeping ISO mounted (current file lives on {mnt_letter}:)")
                    self._active_iso_mount = active
                    self._pending_iso_mount = None
                    return
                self._dismount_iso_or_defer(active, label='previous ISO')
                active = None
            self._active_iso_mount = pending or active
            self._pending_iso_mount = None
        except Exception as e:
            logger.warning(f"[DISC] ISO promote failed: {e}")

    def play_file(self, file_path):
        """Exception boundary for the multi-stage media load transaction."""
        self._pending_play_request_id += 1
        request_id = self._pending_play_request_id
        try:
            return self._play_file_impl(file_path, request_id)
        except Exception as exc:
            session_id = getattr(self, '_loading_session_id', None)
            logger.exception("[LOAD] Unhandled load failure for %s", file_path)
            if session_id is not None:
                self._abort_media_load(
                    session_id, f"Could not load this source: {exc}")
            try:
                self._dismount_pending_iso()
            except Exception:
                pass
            try:
                self.loading_overlay.hide_loading()
            except Exception:
                pass
            return None

    def _retry_play_file_when_ready(self, file_path, request_id, attempt):
        """Retry one pre-session request without letting it outlive a newer one."""
        if (getattr(self, '_app_closing', False)
                or request_id != self._pending_play_request_id):
            return
        if self.player:
            self.play_file(file_path)
            return
        if attempt >= 100:  # 10 seconds: mpv initialization should take ~100 ms.
            logger.error("[LOAD] mpv was not ready after 10 seconds")
            self.show_3d_notification(
                "The playback engine did not initialize.", success=False)
            return
        QTimer.singleShot(
            100,
            lambda: self._retry_play_file_when_ready(
                file_path, request_id, attempt + 1))

    def _play_file_impl(self, file_path, request_id):
        """Loads and starts playing a video file - V7a Enhanced with cleanup delay.

        Also accepts a Blu-ray 3D disc/folder: a drive letter (J:\\), a BDMV folder,
        an index.bdmv, or any folder containing a BDMV. In that case the feature film
        ("main title") SSIF is auto-detected (duration-based, robust to decoy playlists).
        """
        original_request = file_path
        if getattr(self, '_archiving', False):
            self.show_3d_notification("ISO copy in progress — playback unavailable", success=False)
            return
        # The gate is deliberately BEFORE every reset, mount and processEvents()
        # call.  A rejected re-entrant request must be a true no-op.
        if getattr(self, '_is_loading_file', False):
            logger.warning("[LOAD] File load already in progress, ignoring request")
            return
        # Retry the ORIGINAL request before mounting anything. The former check
        # happened after ISO resolution and retried the temporary D:\... feature
        # path, losing ownership of the mount on the second call.
        if not self.player:
            print("Player not ready, retrying...")
            QTimer.singleShot(
                100,
                lambda: self._retry_play_file_when_ready(
                    file_path, request_id, 1))
            return
        self._is_loading_file = True
        session_id = self._begin_media_session(original_request)
        # Reset multi-segment (seamless-branching) feature state for every load; set below
        # only when a disc feature spans several SSIF segments (an edl:// URI, no temp file).
        self._pending_feature_segments = None
        self._feature_edl_uri = None
        # Reset the "authored-3D disc whose SSIF interleave was never materialized"
        # flag (MakeMKV backup: .ssif.smap sidecars but no .ssif). Set below when
        # find_feature reports it; routes the base m2ts to reliable 2D (mpv) instead of
        # the MVC demuxer, which would otherwise hang on the single-view stream.
        self._bd_ssif_interleave_missing = False
        # BD3D dual-file backup pair (base.m2ts, dep.m2ts) when the SSIF interleave
        # is missing but find_feature paired the separate views; routes the MVC
        # pipeline through open_dual instead of the mpv-2D fallback. Reset per load.
        self._bd_dual_file_pair = None
        self._bd_eye_order = UNKNOWN
        resolved_disc_feature = None
        try:
            from sylc import bluray_disc
            from PySide6.QtWidgets import QApplication
            self._pending_iso_mount = None
            # Blu-ray ISO: mount it (no admin needed) and treat the mount as the disc.
            if bluray_disc.is_iso(file_path):
                already_mounted = bluray_disc.get_iso_mount_drive(file_path)
                try:
                    self.show_3d_notification("Mounting Blu-ray ISO…", success=True)
                    QApplication.processEvents()
                except Exception:
                    pass
                drive = bluray_disc.mount_iso(file_path)
                if drive and bluray_disc.is_bluray_path(drive):
                    self._pending_iso_mount = (file_path, drive)
                    logger.info(f"[DISC] Mounted ISO {file_path} -> {drive}")
                    file_path = drive  # detect the feature on the mounted drive
                    # A drive letter is published before a large UDF filesystem has
                    # necessarily enumerated all PLAYLIST/STREAM entries. Accepting
                    # the first visible playlist selected an 8-second title on the
                    # Avatar ISO; manually reopening the settled drive then found the
                    # 9702-second SSIF. Require a stable inventory and retain the best
                    # candidate observed during the bounded settling window.
                    try:
                        from PySide6.QtCore import QEventLoop

                        def _pump_iso_mount_ui():
                            QApplication.processEvents(
                                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

                        resolved_disc_feature = bluray_disc.find_feature_stable(
                            drive,
                            fresh_mount=not bool(already_mounted),
                            timeout_s=30.0,
                            poll_s=1.0,
                            stable_samples=3,
                            on_poll=_pump_iso_mount_ui,
                        )
                        _stable_feat, _stable_info = resolved_disc_feature
                        logger.info(
                            "[DISC] Stable ISO inventory: playlists=%s, "
                            "feature=%s, duration=%.0fs",
                            (_stable_info or {}).get('candidates_playlists'),
                            _stable_feat,
                            float((_stable_info or {}).get('duration_s') or 0.0),
                        )
                    except Exception as stable_error:
                        resolved_disc_feature = None
                        logger.warning(
                            f"[DISC] Stable ISO scan failed; using regular scan: "
                            f"{stable_error}")
                elif drive:
                    bluray_disc.dismount_iso(file_path)
                    self.show_3d_notification("ISO has no Blu-ray (BDMV) structure", success=False)
                    self._abort_media_load(session_id)
                    return
                else:
                    self.show_3d_notification("Could not mount the ISO", success=False)
                    self._abort_media_load(session_id)
                    return
            if os.path.isdir(file_path) or bluray_disc.is_bluray_path(file_path):
                try:
                    self.show_3d_notification("Detecting the feature on disc…", success=True)
                    QApplication.processEvents()
                except Exception:
                    pass
                if resolved_disc_feature is not None:
                    feat, info = resolved_disc_feature
                else:
                    feat, info = bluray_disc.find_feature(file_path)
                # Freshly-mounted ISO second chance: a large UDF volume can become
                # browsable a moment after the mount readiness wait gives up —
                # one bounded retry beats dismounting a perfectly good disc.
                if not feat and getattr(self, '_pending_iso_mount', None):
                    logger.info("[DISC] No feature on first scan of fresh ISO mount — retrying in 2s")
                    time.sleep(2.0)
                    feat, info = bluray_disc.find_feature(file_path)
                if feat:
                    kind = info.get('kind')
                    mins = (info.get('duration_s') or 0) / 60.0
                    logger.info(f"[DISC] Feature ({kind}): {feat} | method={info.get('method')} "
                                f"dur={info.get('duration_s', 0):.0f}s playlist={info.get('playlist')} "
                                f"clip={info.get('clip')}")
                    # Authored-3D disc whose interleaved SSIF was never written (MakeMKV
                    # backup: only .ssif.smap sidecars). We cannot pair the separate
                    # base/dependent m2ts, so play the base view in 2D — and say so plainly
                    # instead of mislabelling it a "2D feature" or hanging the MVC demuxer.
                    self._bd_ssif_interleave_missing = bool(info.get('ssif_interleave_missing'))
                    # DF-4: MakeMKV BD3D backup whose separate base/dependent m2ts were
                    # paired by find_feature -> play as REAL 3D via the dual-source
                    # demuxer (open_dual). Only honoured when the interleave is missing.
                    self._bd_dual_file_pair = info.get('dual_file_pair') if self._bd_ssif_interleave_missing else None
                    self._bd_eye_order = normalise_eye_order(info.get('eye_order'))
                    # DF-FINAL FIX 4: only claim the dual-file MVC label when the
                    # same gate analyze_and_configure_3d uses to actually promote
                    # the session (MVC_SUPPORT_AVAILABLE and NATIVE_RENDER_AVAILABLE)
                    # is confirmed -- otherwise the pipeline silently falls back to
                    # the base view in 2D and this toast would have over-promised.
                    if (self._bd_ssif_interleave_missing and self._bd_dual_file_pair
                            and MVC_SUPPORT_AVAILABLE and NATIVE_RENDER_AVAILABLE):
                        label = "3D backup (dual-file MVC)"
                    elif self._bd_ssif_interleave_missing:
                        label = "3D disc without SSIF interleave — base view in 2D"
                    else:
                        label = "3D feature" if kind == 'ssif' else "2D feature"
                    self.show_3d_notification(f"{label}: {os.path.basename(feat)} ({mins:.0f} min)", success=True)
                    file_path = feat
                    # BD3D authored subtitle depth: PG PID -> offset_sequence_id
                    # from the playlist's STN_table_SS (drives the OFMD depth).
                    self._bd3d_pg_offset_map = {}
                    try:
                        _pl = info.get('playlist')
                        if _pl:
                            # find_feature reports the playlist BASENAME (e.g.
                            # '00852.mpls') — resolve it under <BDMV>/PLAYLIST.
                            if not os.path.isabs(_pl) and info.get('bdmv'):
                                _pl = os.path.join(info['bdmv'], 'PLAYLIST', _pl)
                            if os.path.isfile(_pl):
                                from sylc.bd3d_offset_metadata import parse_mpls_pg_offsets
                                self._bd3d_pg_offset_map = parse_mpls_pg_offsets(_pl)
                            else:
                                logger.warning(f"[BD3D-DEPTH] playlist not found: {_pl}")
                    except Exception as _e:
                        logger.warning(f"[BD3D-DEPTH] PG offset map unavailable: {_e}")
                    # Seamless-branching 3D feature spanning several SSIF segments: build a
                    # matching mpv EDL (continuous audio on one timeline) + remember the
                    # ordered segment sequence so the decoder plays them as one film.
                    _segs = info.get('segments') or []
                    if kind == 'ssif' and len(_segs) > 1:
                        try:
                            _uri = bluray_disc.build_feature_edl(_segs)
                            if _uri:
                                self._pending_feature_segments = _segs
                                self._feature_edl_uri = _uri
                                logger.info(f"[DISC] Multi-segment feature: {len(_segs)} SSIF segments via edl:// ({mins:.0f} min)")
                        except Exception as _e:
                            logger.warning(f"[DISC] Multi-segment EDL build failed ({_e}); first segment only")
                            self._pending_feature_segments = None
                            self._feature_edl_uri = None
                else:
                    logger.info(f"[DISC] No feature found under: {file_path}")
                    self.show_3d_notification("No playable feature found on this disc/folder", success=False)
                    self._dismount_pending_iso()
                    self._abort_media_load(session_id)
                    return
        except Exception as e:
            logger.warning(f"[DISC] BDMV detection failed: {e}")

        # If detection threw/failed and left a directory or drive root, abort cleanly
        # instead of trying to "play" a folder.
        if os.path.isdir(file_path):
            logger.warning(f"[DISC] No playable feature resolved from: {file_path}")
            self._dismount_pending_iso()
            self.show_3d_notification("No playable feature found on this disc/folder", success=False)
            self._abort_media_load(session_id)
            return

        if not os.path.exists(file_path):
            logger.error(f"[LOAD] Resolved feature disappeared or is not ready: {file_path}")
            self._dismount_pending_iso()
            self.show_3d_notification(
                "The Blu-ray feature is not readable after mounting the ISO.",
                success=False,
            )
            self._abort_media_load(session_id)
            return

        # SSIF/M2TS (Blu-ray 3D raw streams) are now supported via the native demuxer.
        file_ext = os.path.splitext(file_path)[1].lower()

        # CRITICAL FIX V2: Reset playback ended flag when loading new file
        # This allows MPV callbacks to work normally again
        self._playback_ended = False

        # New file: trust mpv's pause reports again until _fetch_audio_tracks
        # re-establishes whether the shell actually has audio to play.
        self._mpv_shell_inert = False

        self._current_precise_time = 0.0 # Reset precise timeline tracker
        logger.info(f"[LOAD] Loading file: {file_path}")

        # THUMB: no thumbnail I/O may overlap the load/demuxer-init window
        if getattr(self, '_thumb_service', None):
            self._thumb_service.disarm()

        # Show loading overlay with animation (hide welcome screen to avoid text overlap)
        self.info_overlay.hide()
        self._update_overlays_geometry()
        self.loading_overlay.show_loading("Initializing playback...")

        # CRITICAL: Reset subtitles from previous file to prevent carryover
        if self._subtitle_manager:
            logger.info("[LOAD] Clearing previous subtitles")
            self._subtitle_manager.clear()
            self._subtitle_manager.set_enabled(False)
        self._cached_pgs_sup_path = None
        self._cached_pgs_track_index = None
        self._active_pgs_track_index = None
        self._pgs_subtitle_tracks = []

        # CRITICAL STABILITY: Stop MPV first to release file handles and video output
        # V61 STABILITY: if a previous stop_playback terminated mpv and its async
        # re-init hasn't fired yet, re-create it SYNCHRONOUSLY now — the whole load
        # path assumes a live player (the duplicate-init guard in _setup_mpv_player
        # makes the pending timer a no-op afterwards).
        if self.player is None:
            logger.info("[LOAD] MPV instance not alive (post-stop) — synchronous re-init")
            self._setup_mpv_player()
        if self.player:
            try:
                self.player.stop()
            except:
                pass

        # Stop decoder and wait for cleanup to complete
        self._stop_mvc_decoder()

        # The previous file's handles are now released — dismount the previously
        # active ISO (if we're switching away) and promote the just-mounted one.
        self._promote_iso_mount()

        # CRITICAL FIX: Add delay after cleanup to ensure all threads are stopped
        # and resources are released before starting new file
        logger.info("[LOAD] Waiting 500ms for cleanup to complete...")
        self._media_single_shot(
            500, lambda: self._continue_play_file(file_path, session_id), session_id)

    def _continue_play_file(self, file_path, session_id):
        """Exception boundary for the deferred half of a load transaction."""
        if not self._session_is_current(session_id):
            return
        try:
            self._continue_play_file_owned(file_path, session_id)
        except Exception as exc:
            logger.exception("[LOAD] Deferred load failed for %s", file_path)
            self._abort_media_load(
                session_id, f"Could not initialize this source: {exc}")
            try:
                self.loading_overlay.hide_loading()
            except Exception:
                pass

    def _continue_play_file_owned(self, file_path, session_id):
        """Continue loading file after cleanup delay."""
        if not self._session_is_current(session_id):
            return
        self.current_file_path = file_path
        # Per-FILE memory: recall this title's remembered tuning right at the
        # canonical per-file reset point, so every reset below can prefer the
        # viewer's own choice over the blanket default.
        _recall = getattr(self, '_recall_for_file', None)
        if _recall is not None:
            _recall()
        # Encoded mattes are per-medium. A rectangular choice from the previous
        # title must never crop a new source before its own detector has spoken.
        self._synth3d_aspect_override = None
        self._synth3d_aspect_unavailable_key = None
        # I3: a manual SBS/TAB pick in the stereo combo is per-file — don't let it stick
        # to the NEXT file (which would override that file's own auto-detection, e.g. force
        # a 2D or MVC clip into SBS). Reset to 'auto' so detection wins on a fresh load.
        self.current_stereo_mode = 'auto'
        # This reset bypasses change_stereo_mode (direct attribute assignment), so it
        # must carry its own Dual Projector teardown: without this, a pair left open
        # from the PREVIOUS file survives into the new one, desyncing eye_windows from
        # (now non-'dual') current_stereo_mode -- the new file's combo shows MultiView/
        # SBS/TAB but the old projector windows keep receiving whatever the decoder
        # emits next.
        self._set_dual_projector_enabled(False)
        # Eye-order overrides are per source.  A right-first choice for one MVC
        # must never silently carry into the next movie — unless it was made
        # for THIS movie: the per-file memory restores it.
        try:
            _eye = (getattr(self, '_file_memory', None) or {}).get('eye_order')
            ov = self.controls_overlay
            (ov.export_eye_left_action if _eye == 'left'
             else ov.export_eye_right_action if _eye == 'right'
             else ov.export_eye_auto_action).setChecked(True)
        except Exception:
            pass
        self._edge264_consecutive_crashes = 0  # fresh source: reset edge264 crash streak
        self._update_archive_button_state()  # archive button lights up only for a Blu-ray disc

        # V7b CRITICAL FIX: Reset timeline IMMEDIATELY to prevent stale duration from previous video
        # This ensures the slider maximum() doesn't contain old values during seek calculations
        self.controls_overlay.set_duration(0)  # range -> (0,0), label 00:00, disabled
        # TIMELINE-RESET FIX (the "bar stays PLEINE on file change" bug): keep the max at 0
        # until mpv reports the REAL duration — do NOT install a temporary max=1. With max=1
        # the very first position tick (a small value, e.g. 0.08s) is clamped by Qt to
        # maximum()=1, so the seek bar paints value/maximum = 100% FULL, and it STAYS full for
        # the whole new file whenever mpv's duration is slow to arrive or never widens the max
        # (large MKV / MVC / SSIF over HDD, damaged streams, or the 8-retry fallback in
        # _update_timeline_and_start_playback that skips set_duration). With max=0 every draw in
        # the slider paintEvent is guarded by `if maximum() > 0`, so the bar is simply EMPTY
        # until the real duration lands (set_duration(0) already left the range at (0,0)).
        self.controls_overlay.time_slider.setEnabled(False)
        # Zero the value AND clear every stale monotonic timeline tracker so a queued/late
        # poller tick or mpv time-pos observer can't re-assert the PREVIOUS file's near-end
        # position onto the fresh bar (the _set_ui_time monotonic guard always re-issues
        # set_time(_last_ui_time); _prev_mvc_ts was never reset per-file, blocking the MVC
        # timeline until it exceeded the old file's end). This is the canonical per-file reset
        # point (every load path traverses it, right beside the current_stereo_mode='auto'
        # reset above) — no per-path duplicates needed.
        self.controls_overlay.set_time(0)
        self._last_ui_time = 0.0
        self._prev_mvc_ts = 0.0
        self._last_mvc_timestamp = 0.0
        self._current_precise_time = 0.0
        self._prime_mpv_time_pos(None)

        # STEP 1: Quick analysis to detect if MVC (DON'T start decoder yet!)
        self.loading_overlay.set_status("Analyzing 3D structure...")
        self.video_3d_info = Video3DAnalyzer.analyze_file(file_path)
        is_mvc = self.video_3d_info.get('stereo_mode') == 'mvc'
        self._mvc_file_detected = is_mvc

        logger.info(f"[LOAD] MVC detected: {is_mvc}, PGS available: {PGS_SUBTITLE_AVAILABLE}")

        # STEP 2: For MVC files, extract PGS subtitles BEFORE starting anything
        # EXCEPTION: SSIF files use streaming mode for subtitles (no pre-extraction needed)
        is_ssif = file_path.lower().endswith('.ssif')
        is_mkv = file_path.lower().endswith(('.mkv', '.mk3d'))  # mk3d = MKV 3D variant

        # ========== STREAMING SUBTITLE OPTIMIZATION ==========
        # For MKV files: Skip extraction! The MVC decoder will stream subtitles in real-time.
        # This eliminates the 2-5 minute startup delay.
        # For SSIF/M2TS: Still use extraction (streaming not yet implemented for these formats)
        if is_mkv and is_mvc:
            logger.info("[PGS] MKV detected - using streaming subtitles (no extraction delay!)")
            self._configure_and_start_playback(file_path, session_id)
            return
        # =====================================================

        if PGS_SUBTITLE_AVAILABLE and self._subtitle_extractor and is_mvc and not is_ssif:
            self._extract_pgs_at_startup(file_path, session_id)
            return  # _extract_pgs_at_startup will call _finalize_play_file_mvc when done

        # Non-MVC files: Configure 3D and continue directly
        self._configure_and_start_playback(file_path, session_id)

    def _extract_pgs_at_startup(self, file_path, session_id):
        """Extract PGS subtitles at startup before playback begins."""
        if not self._session_is_current(session_id):
            return
        logger.info("[PGS] Starting pre-extraction of PGS subtitles...")
        self.loading_overlay.show_loading("Extracting subtitles...", progress_mode=True)
        self._pgs_startup_pending_session = session_id

        # Clear any previously cached extraction
        self._cached_pgs_sup_path = None
        self._pgs_subtitle_tracks = []

        # Progress callback (emits Qt signal for thread-safe UI update)
        def on_progress(progress_value: float):
            if self._session_is_current(session_id):
                self.extraction_progress.emit({
                    'session': session_id,
                    'progress': progress_value,
                })

        extractor = self._subtitle_extractor

        def extract_thread(cancel_event):
            payload = {
                'session': session_id,
                'file_path': file_path,
                'tracks': [],
                'sup_path': None,
                'track_index': None,
                'parser': None,
            }
            try:
                # Step 1: Detect PGS tracks
                logger.info(f"[PGS STARTUP] Detecting subtitle tracks in {file_path}")
                tracks = extractor.detect_subtitle_tracks(file_path)
                if cancel_event.is_set():
                    return
                pgs_tracks = [t for t in tracks if t.is_pgs]
                payload['tracks'] = tracks

                if not pgs_tracks:
                    logger.info("[PGS STARTUP] No PGS tracks found, skipping extraction")
                    if self._session_is_current(session_id):
                        self.pgs_extraction_complete.emit(payload)
                    return

                # Step 2: Extract first PGS track (usually the main one)
                pgs_track = pgs_tracks[0]
                logger.info(f"[PGS STARTUP] Extracting track {pgs_track.index} ({pgs_track.language})...")

                import time
                start_time = time.time()

                # Pass progress callback for real-time UI updates
                sup_path = extractor.extract_pgs_track(
                    file_path, pgs_track.index, progress_callback=on_progress
                )

                elapsed = time.time() - start_time
                logger.info(f"[PGS STARTUP] Extraction completed in {elapsed:.1f}s: {sup_path}")
                if cancel_event.is_set() or not self._session_is_current(session_id):
                    return

                if sup_path:
                    # Step 3: Parse the extracted file
                    logger.info("[PGS STARTUP] Parsing subtitle file...")
                    from sylc.pgs_subtitle_parser import PGSSubtitleParser
                    parser = PGSSubtitleParser()
                    success = parser.load_from_file(sup_path)

                    if success:
                        payload.update({
                            'sup_path': sup_path,
                            'track_index': pgs_track.index,
                            'parser': parser,
                        })
                        logger.info(
                            "[PGS STARTUP] Parsed %d subtitle cues for track %s",
                            len(parser.display_sets), pgs_track.index)
                    else:
                        logger.warning("[PGS STARTUP] Failed to parse subtitle file")
                else:
                    logger.warning("[PGS STARTUP] Extraction returned no file")

                # CRITICAL: Use Qt Signal for thread-safe callback (NOT QTimer.singleShot!)
                # QTimer.singleShot from background thread causes freezes/crashes
                logger.info("[PGS STARTUP] Emitting pgs_extraction_complete signal...")
                if not cancel_event.is_set() and self._session_is_current(session_id):
                    self.pgs_extraction_complete.emit(payload)
                logger.info("[PGS STARTUP] Signal emitted, thread completing")

            except Exception as e:
                logger.error(f"[PGS STARTUP] Error during extraction: {e}")
                import traceback
                traceback.print_exc()
                # Continue to playback anyway via signal
                if not cancel_event.is_set() and self._session_is_current(session_id):
                    payload['error'] = str(e)
                    self.pgs_extraction_complete.emit(payload)

        self._start_media_worker(
            extract_thread, session_id=session_id, name='pgs-startup-extractor')
        # The extractor's subprocess paths are bounded, but its pure-Python last
        # resort is not.  Release the UI transaction after six minutes and make
        # its eventual result stale instead of leaving the player locked forever.
        self._media_single_shot(
            360000,
            lambda: self._on_pgs_startup_timeout(file_path, session_id),
            session_id)

    @Slot(object)
    def _on_extraction_progress(self, payload):
        """Called on main thread when extraction progress updates (via Qt Signal)."""
        if not self._session_is_current(payload.get('session')):
            return
        self.loading_overlay.set_progress(float(payload.get('progress') or 0.0))

    @Slot(object)
    def _on_pgs_extraction_complete(self, payload):
        """Called on main thread when PGS extraction is complete (via Qt Signal)."""
        session_id = payload.get('session')
        file_path = payload.get('file_path')
        logger.info(f"[PGS STARTUP] Signal received on main thread, file: {file_path}")
        if not self._session_is_current(session_id):
            return
        if file_path != self.current_file_path:
            logger.warning(f"[PGS STARTUP] File path mismatch: expected {self.current_file_path}, got {file_path}")
            return
        if self._pgs_startup_pending_session != session_id:
            return
        self._pgs_startup_pending_session = None
        self._pgs_subtitle_tracks = payload.get('tracks') or []
        parser = payload.get('parser')
        if parser is not None and self._subtitle_manager.install_parser(
                parser, payload.get('sup_path') or '<startup>'):
            self._cached_pgs_sup_path = payload.get('sup_path')
            self._cached_pgs_track_index = payload.get('track_index')
        self._configure_and_start_playback(file_path, session_id)

    def _on_pgs_startup_timeout(self, file_path, session_id):
        if (not self._session_is_current(session_id)
                or self._pgs_startup_pending_session != session_id):
            return
        logger.error("[PGS STARTUP] Timed out for %s", file_path)
        self._pgs_startup_pending_session = None
        self._media_cancel_event.set()
        self._abort_media_load(
            session_id,
            "Subtitle extraction timed out; playback was not started.")
        self.loading_overlay.hide_loading()

    def _mpv_source_for(self, file_path):
        """Return the source MPV should open for this file.

        Multi-segment features play through an EDL so audio stays continuous on
        one timeline; everything else plays the file directly.
        """
        return getattr(self, '_feature_edl_uri', None) or file_path
    def _configure_and_start_playback(self, file_path, session_id=None):
        """Configure 3D mode and start playback (called after PGS extraction for MVC files)."""
        owner = self._media_session_id if session_id is None else session_id
        if not self._session_is_current(owner):
            return
        logger.info(f"[LOAD] _configure_and_start_playback called for {file_path}")
        self.loading_overlay.set_status("Starting playback...")
        try:
            # Configure 3D mode (this starts MVC decoder if needed)
            # video_3d_info was already set in _continue_play_file
            logger.info(f"[LOAD] video_3d_info before configure: {self.video_3d_info}")
            self.analyze_and_configure_3d(file_path)
            logger.info("[LOAD] analyze_and_configure_3d completed")

            self.has_media = True
            self._update_3d_button_state()   # 3D button enabled only for genuine 3D content
            # self.metrics_overlay.show() # Disabled to remove top-left artifact
            # Multi-segment feature: mpv plays the EDL (continuous audio across all segments
            # on one timeline); the decoder plays the matching SSIF sequence (SequenceDemuxer).
            _mpv_src = self._mpv_source_for(file_path)
            if self.player is None:
                raise RuntimeError("mpv core is unavailable")
            playback_core = self.player
            if not self._install_mpv_media_observers(owner, playback_core):
                raise RuntimeError("mpv media observers could not be installed")
            # Old-core observers were detached at _begin_media_session; from this
            # point every observer belongs to this exact core/session pair.
            self._mpv_transition_in_progress = False
            self.player.play(_mpv_src)
            # A new playlist entry may run mpv's track selection again. Keep
            # every subtitle backend neutral until the remembered/user choice
            # is explicitly applied after track enumeration.
            self._disable_all_subtitle_outputs()
            self.player.pause = True
            # The async pause observer can legitimately arrive late (notably
            # while TrueHD initializes).  Latch the command-side truth now so
            # the HEVC audio clock cannot extrapolate during the load barrier.
            self._mpv_pause_cache = True
            # V7b FIX: FORCE the timer to stay active even when paused for MVC mode
            # This lets the slider progress immediately
            if self.mvc_mode_active or getattr(self, "_mvc_file_detected", False):
                self._playback_timer.start()  # Override pause behavior
            self.update_ui_state()

            self._media_single_shot(500, lambda: self.load_audio_tracks(owner), owner)
            self._media_single_shot(500, lambda: self.load_subtitle_tracks(owner), owner)

            # V7b TIMELINE FIX: Sync timeline BEFORE starting playback
            # Update timeline with MPV duration and THEN start playback to ensure correct scale
            def _update_timeline_and_start_playback(retry_count=0):
                if (not self._session_is_current(owner, core=playback_core)
                        or playback_core is not self.player):
                    return
                try:
                    # PROVISIONAL RANGE, first tick: give the slider the ffprobe
                    # duration IMMEDIATELY instead of leaving an empty bar for
                    # the up-to-4 s mpv-duration retry dance (a no-audio file
                    # never provides one at all). mpv's value, when it lands,
                    # simply overwrites this with the same number.
                    if retry_count == 0:
                        try:
                            _ffd = float((self.video_3d_info or {}).get('duration') or 0)
                        except (TypeError, ValueError):
                            _ffd = 0.0
                        if _ffd > 0 and self.current_file_path:
                            self.controls_overlay.set_duration(_ffd)
                            self.controls_overlay.time_slider.set_video_file(self.current_file_path, _ffd)
                            logger.info(f"[TIMELINE] Provisional range from ffprobe: {_ffd}s")
                    mpv_duration = 0
                    if hasattr(self, 'player') and self.player:
                        try:
                            mpv_duration = self.player.duration
                        except:
                            pass # Property access failed

                    if mpv_duration and mpv_duration > 0 and self.current_file_path:
                        # Update BOTH duration label and slider range FIRST
                        self.controls_overlay.set_duration(mpv_duration)
                        self.controls_overlay.time_slider.set_video_file(self.current_file_path, mpv_duration)
                        logger.info(f"[TIMELINE] Updated range from MPV: {mpv_duration}s")
                        # THUMB: playback is up on this path too → arm shortly after
                        if getattr(self, '_thumb_service', None):
                            self._thumb_service.set_duration(float(mpv_duration))
                            self._media_single_shot(
                                1000, self._thumb_service.arm, owner)

                        # SSIF/M2TS seek needs this duration inside the demuxer (proportional
                        # byte seek). This is where mpv's duration reliably lands at startup.
                        if getattr(self, 'mvc_decoder_thread', None):
                            try:
                                self.mvc_decoder_thread.set_media_duration(float(mpv_duration))
                            except Exception:
                                pass

                        # NOW start playback with correct timeline scale
                        # Explicitly force UI to playing state (Pause Icon) immediately
                        self.controls_overlay.set_paused(False)

                        def _safe_start():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    # V10 SSIF FIX: For MVC mode, DON'T unpause here.
                                    # Wait for decoder to emit seekIDRFound with actual start position.
                                    # This prevents audio from playing ahead of video during decoder init.
                                    if self.mvc_mode_active or getattr(self, "_mvc_file_detected", False):
                                        logger.info("[TIMELINE] MVC mode: keeping MPV paused until decoder ready")
                                        # Just set position to 0, but keep paused
                                        self.player.command_async('set', 'time-pos', '0')
                                    else:
                                        self.player.command_async('set', 'time-pos', '0')
                                        self.player.command_async('set', 'pause', 'no')
                                        self._arm_hevc_audio_start(0.0)
                                except:
                                    pass

                        self._media_single_shot(50, _safe_start, owner)
                        # Per-file memory: apply the deferred fields (resume
                        # position, 3D-off, synth3d re-enable) once the
                        # pipeline is actually running.
                        self._media_single_shot(
                            1800, self._apply_deferred_file_memory, owner)
                        logger.info("[TIMELINE] Playback started with correct scale")
                        return

                    # Retry if duration is still missing (up to 8 times with shorter intervals)
                    if retry_count < 8:
                        delay = 150 if retry_count < 3 else 500  # Fast retries first, then slower
                        self._media_single_shot(
                            delay,
                            lambda: _update_timeline_and_start_playback(retry_count + 1),
                            owner)
                    else:
                        # Fallback: start playback anyway with ffprobe duration.
                        # ACTUALLY APPLY it: this branch used to only log the
                        # intent — with a no-audio file (mpv shell has nothing
                        # to play, duration never appears) the slider range
                        # stayed at 0 and the timeline was an empty bar.
                        logger.warning("[TIMELINE] MPV duration not available, starting with ffprobe duration")
                        ff_dur = 0
                        try:
                            ff_dur = float((self.video_3d_info or {}).get('duration') or 0)
                        except (TypeError, ValueError):
                            pass
                        if ff_dur > 0 and self.current_file_path:
                            self.controls_overlay.set_duration(ff_dur)
                            self.controls_overlay.time_slider.set_video_file(self.current_file_path, ff_dur)
                            logger.info(f"[TIMELINE] Updated range from ffprobe: {ff_dur}s")
                            if getattr(self, '_thumb_service', None):
                                self._thumb_service.set_duration(ff_dur)
                                self._media_single_shot(
                                    1000, self._thumb_service.arm, owner)
                            if getattr(self, 'mvc_decoder_thread', None):
                                try:
                                    self.mvc_decoder_thread.set_media_duration(ff_dur)
                                except Exception:
                                    pass
                        # Explicitly force UI to playing state (Pause Icon) immediately
                        self.controls_overlay.set_paused(False)
                        
                        def _safe_fallback_start():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    self.player.command_async('set', 'time-pos', '0')
                                    self.player.command_async('set', 'pause', 'no')
                                    self._arm_hevc_audio_start(0.0)
                                except:
                                    pass

                        self._media_single_shot(50, _safe_fallback_start, owner)
                        # Per-file memory: same deferred application as the
                        # mpv-duration branch (no-audio files land here).
                        self._media_single_shot(
                            1800, self._apply_deferred_file_memory, owner)

                except Exception as e:
                    logger.debug(f"[TIMELINE] Could not update from MPV: {e}")
                    # Fallback: start playback anyway
                    try:
                        self.controls_overlay.set_paused(False)
                        def _safe_last_resort():
                            if hasattr(self, 'player') and self.player:
                                try:
                                    self.player.command_async('set', 'time-pos', '0')
                                    self.player.command_async('set', 'pause', 'no')
                                    self._arm_hevc_audio_start(0.0)
                                except:
                                    pass
                        self._media_single_shot(50, _safe_last_resort, owner)
                    except:
                        pass

            self._media_single_shot(
                200, _update_timeline_and_start_playback, owner)

            if self.is_3d_enabled:
                self.configure_3d_output(True, self.current_stereo_mode)

            logger.info("[LOAD] File loaded successfully")
            # Hide loading overlay with fade-out animation
            self.loading_overlay.hide_loading()
        except Exception as exc:
            logger.exception("[LOAD] Playback configuration failed for %s", file_path)
            self.has_media = False
            self._abort_media_load(owner, f"Playback initialization failed: {exc}")
            try:
                self.loading_overlay.hide_loading()
            except Exception:
                pass
        finally:
            # A stale completion must never unlock a newer transaction.
            if (self._session_is_current(owner)
                    and self._loading_session_id == owner):
                self._is_loading_file = False
                self._loading_session_id = None

    def analyze_and_configure_3d(self, file_path):
        """Analyzes the file and automatically configures the 3D mode."""
        # Skip re-analysis if already done (e.g., in _continue_play_file for PGS extraction)
        if not self.video_3d_info:
            self.video_3d_info = Video3DAnalyzer.analyze_file(file_path)

        # DF-4: BD3D backup WITHOUT SSIF interleave (MakeMKV: base + dependent views in
        # SEPARATE .m2ts, no interleaved .ssif). find_feature paired them into
        # info['dual_file_pair']; the C++ dual-source demuxer (open_dual) reads both and
        # emits the SAME base/dep pairs as an SSIF disc. The base m2ts alone probes as
        # flat 2D, so PROMOTE the session to a real MVC 3D session here — it then flows
        # through the identical MVC UI + pipeline path below (MultiView combo, framepack,
        # badge, _start_mvc_decoder). dual_pair is plumbed to the decoder in
        # _start_mvc_decoder, gated on _bd_dual_active. Any failure (open_dual/decode)
        # degrades to the existing 2D mpv fallback via _fallback_from_edge264.
        self._bd_dual_active = False
        _dual_pair = getattr(self, '_bd_dual_file_pair', None)
        if (getattr(self, '_bd_ssif_interleave_missing', False) and _dual_pair
                and MVC_SUPPORT_AVAILABLE and NATIVE_RENDER_AVAILABLE
                and isinstance(self.video_3d_info, dict)):
            try:
                self.video_3d_info['is_3d'] = True
                self.video_3d_info['stereo_mode'] = 'mvc'
                self._bd_dual_active = True
                logger.info(f"[MVC] dual-file BD3D session: base={os.path.basename(_dual_pair[0])} "
                            f"dep={os.path.basename(_dual_pair[1])} -> MVC pipeline (open_dual)")
            except Exception as e:
                logger.warning(f"[MVC] dual-file promotion failed, staying on 2D fallback: {e}")
                self._bd_dual_active = False

        # SOL 1A: Set MVC flag IMMEDIATELY (before _start_mvc_decoder)
        # Allows the timer to stay active as soon as player.pause = True.
        # Packed-stereo H.264 (SBS/TAB) is also edge264-decoded, so it keeps the
        # flag set too (timeline timer / pause / 3D-button gating treat it alike).
        _sm = self.video_3d_info.get('stereo_mode')
        _cn = (self.video_3d_info.get('codec_name') or '').lower()
        _cx = (self.video_3d_info.get('container_ext') or '').lower()
        self._mvc_file_detected = (
            _sm == 'mvc'
            or (_sm in ('sbs', 'tab') and _cn == 'h264'
                and _cx in EDGE264_CONTAINERS)
        )
        self._update_3d_button_state()

        # V7b CRITICAL FIX: DO NOT update the timeline with the ffprobe duration
        # The timeline will be updated ONLY by _update_timeline_and_start_playback with MPV
        # This avoids scale conflicts between ffprobe and MPV that cause incorrect seeks

        # BUT we keep the FPS and the file name for the previews
        self._apply_preview_thumbs_policy(file_path)
        self.controls_overlay.time_slider.set_video_file(file_path, 0)  # Duration=0 for now
        fps_val = self.video_3d_info.get('fps')
        if fps_val:
            self.current_video_fps = fps_val

        # NOTE: The duration will be updated by _update_timeline_and_start_playback after MPV loads

        if self.video_3d_info.get('analysis_error'):
            self.show_3d_notification("3D analysis via ffprobe failed.", success=False)

        # D1: 3D controls (button + dropdown) are gated by _update_3d_button_state — enabled
        # only for genuine 3D content, greyed for 2D. (Previously force-enabled here to allow a
        # manual 2D→3D override; that override is removed by design — forcing 3D on a 2D file
        # mis-drives the pipeline, and the dropdown must not alter a 2D video's proportions.)
        self._update_3d_button_state()

        if self.video_3d_info['is_3d'] and self.video_3d_info['stereo_mode'] != 'none':
            stereo_mode = self.video_3d_info['stereo_mode']
            # Index mapping: 0=MVC, 1=Side-by-Side, 2=Top-Bottom, 3=Dual Projector, 4=Glasses
            # The COMBO starts on the viewer's remembered presentation for this
            # file when one exists (per-file memory); the decoder configuration
            # below keeps following the detected CONTENT mode regardless.
            _choose = getattr(self, '_choose_initial_stereo_mode', None)
            ui_mode = _choose(stereo_mode) if _choose else stereo_mode
            mode_index = {'mvc': 0, 'sbs': 1, 'tab': 2, 'dual': 3,
                          'glasses': 4}.get(ui_mode, 0)
            self.controls_overlay.stereo_mode_combo.setCurrentIndex(mode_index)
            # A REMEMBERED pick must be latched even when the combo was already
            # showing it (setCurrentIndex emits nothing then — measured: the
            # replay booted with the right combo but current_stereo_mode stuck
            # at 'auto'). Detection-only loads keep the historical behavior.
            if ((getattr(self, '_file_memory', None) or {}).get('stereo_mode') == ui_mode
                    and self.current_stereo_mode != ui_mode):
                self.change_stereo_mode(ui_mode)
            mode_names = {'mvc': 'MultiView', 'sbs': 'Side-by-Side', 'tab': 'Top-Bottom'}
            self.show_3d_notification(f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())}", success=True,
                                      permanent=True)

            # Always start MVC decoder for MVC content (required for SSIF 2D playback)
            if stereo_mode == 'mvc':
                # V7b CRITICAL FIX: Force decoder to start at 0s, not at current MPV time
                # This prevents the "21.955s drift" bug where decoder starts at wrong timestamp
                if MVC_SUPPORT_AVAILABLE:
                    try:
                        self._start_mvc_decoder(start_time=0.0)
                    except Exception as e:
                        # edge264 first, mpv only on failure: don't crash the load —
                        # degrade to mpv and stop configuring the 3D window.
                        self._fallback_from_edge264(reason=f"MVC decoder init failed: {e}")
                        return
                else:
                    logger.warning("[MVC] MVC content detected but decoder support is unavailable; using mpv fallback.")
                    self._fallback_to_mpv_mvc()
                # V7b CRITICAL FIX: Framepacking window should ALWAYS use framepack mode
                # It's specifically designed for 1920x2205 framepack output!
                if self.framepacking_window:
                    self.framepacking_window.display_widget.set_stereo_mode('framepack')
                # Reassure the user: edge264 recognised & adapted to this 3D stream.
                self.controls_overlay.set_format_badge(self._format_badge_label())

                # 3D button starts OFF - user must manually enable 3D mode
                # (Previously auto-enabled if Nvidia 3D Vision was active)
            elif stereo_mode in ('sbs', 'tab'):
                # Packed-stereo H.264 (Full-SBS / Full-TAB): edge264 decodes EVERY
                # H.264 stream. The decoded frame carries BOTH eyes;
                # _on_mvc_frame_yuv_ready splits it into L (base eye) + R, so the
                # player drives it EXACTLY like MVC:
                #   - main window default = the BASE view (left/top eye), '2d' mode
                #   - SBS/TAB combo       = the main view's layout (L|R or L/R)
                #   - FramePack window    = the two views stacked at full resolution
                # Same containers as the 2D edge264 path; mpv fallback only on failure.
                codec = (self.video_3d_info.get('codec_name') or '').lower()
                ext = (self.video_3d_info.get('container_ext') or '').lower()
                if (MVC_SUPPORT_AVAILABLE and codec == 'h264'
                        and ext in EDGE264_CONTAINERS
                        and NATIVE_RENDER_AVAILABLE):
                    logger.info(f"[PACKED-3D] {stereo_mode.upper()} H.264 ({ext}) via edge264 (split L/R, like MVC)")
                    try:
                        self._start_mvc_decoder(start_time=0.0)
                        # Main window shows the BASE view (left/top eye) by default.
                        if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
                            self.mvc_embedded_widget.set_stereo_mode('2d')
                        # FramePack window ready to stack BOTH views (shown on 3D toggle).
                        if self.framepacking_window:
                            self.framepacking_window.display_widget.set_stereo_mode('framepack')
                        # Reassure the user: edge264 recognised & adapted to this stream.
                        self.controls_overlay.set_format_badge(self._format_badge_label())
                    except Exception as e:
                        self._fallback_from_edge264(reason=f"{stereo_mode} edge264 init failed: {e}")
                        return
                else:
                    # Non-h264 packed stereo, or unsupported container. HEVC (spec
                    # 2026-07-21) is probed here, AFTER the H.264 paths and BEFORE the mpv
                    # fallback; open() refuses non-HEVC so this never diverts H.264.
                    if not self._try_start_hevc(file_path):
                        self._present_via_mpv_native()
        else:
            # 2D content detected
            self.controls_overlay.clear_format_badge()  # no 3D badge for 2D content
            self.show_3d_notification("2D content detected", success=True, permanent=True)

            # === 2D-via-edge264 path ===
            # Route supported H.264 containers through the shared edge264 thread.
            # Each demuxer must report MVC from explicit metadata/bitstream evidence;
            # an ordinary M2TS runs in base-view-only mode, and the Python decoder
            # duplicates the decoded left view only at presentation time.
            # This gives us: HDR via D3D11 widget, consistent codec path, and
            # MPV stays audio-only (no more MPV vo glitches on 2D files).
            codec = (self.video_3d_info.get('codec_name') or '').lower()
            ext = (self.video_3d_info.get('container_ext') or '').lower()
            # Containers edge264 can demux: MKV/M2TS/TS via the C++ demuxer; MP4/AVI/
            # MOV/FLV/WebM/raw via lavf_h264_demuxer (task #391). Any edge264 failure
            # degrades to mpv via _fallback_from_edge264 (the 2D try/except below).
            eligible_2d = (MVC_SUPPORT_AVAILABLE
                           and codec == 'h264'
                           and ext in EDGE264_CONTAINERS
                           and NATIVE_RENDER_AVAILABLE
                           # A MakeMKV 3D backup whose eyes live in separate .m2ts files
                           # is handled by the explicit dual-file route above. If the pair
                           # could not be resolved, do not silently decode only one eye here.
                           and not getattr(self, '_bd_ssif_interleave_missing', False))
            if eligible_2d:
                logger.info(f"[2D-EDGE264] Routing 2D H.264 ({ext}) through edge264 decoder")
                try:
                    self._start_mvc_decoder(start_time=0.0)
                    # Force every render target to 2D mode (left eye only,
                    # right plane upload skipped by our optimization in widget).
                    if hasattr(self, 'mvc_embedded_widget') and self.mvc_embedded_widget:
                        self.mvc_embedded_widget.set_stereo_mode('2d')
                    if self.framepacking_window:
                        self.framepacking_window.display_widget.set_stereo_mode('2d')
                    self.show_3d_notification("2D edge264 decoder active", success=True, permanent=True)
                except Exception as e:
                    # edge264 first, mpv only on failure (unified fallback path).
                    self._fallback_from_edge264(reason=f"2D edge264 init failed: {e}")
            else:
                # 2D not edge264-eligible (non-H.264 or unsupported container).
                # HEVC (spec 2026-07-21) is probed here, AFTER the H.264 paths and BEFORE
                # the mpv fallback; open() refuses non-HEVC so H.264/VC-1 fall straight
                # through to mpv, which plays them natively as before.
                if not self._try_start_hevc(file_path):
                    self._present_via_mpv_native()


__all__ = ['MediaLoadingMixin', 'configure_media_loading_support']

# -*- coding: utf-8 -*-
"""Media opening, disc archiving, MV-HEVC export and ISO lifecycle."""

import logging
import os
import shutil
import traceback

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QFileDialog, QMessageBox

from sylc.mvhevc_export_dialog import MVHEVCExportDialog
from sylc.stereo_eye_order import LEFT_FIRST, RIGHT_FIRST, UNKNOWN, normalise_eye_order
from sylc.time_slider import _decide_thumbs_mode


logger = logging.getLogger(__name__)


class ArchiveExportMixin:
    def open_file_dialog(self):
        # MKV (MVC) + Blu-ray 3D raw streams (SSIF/M2TS) via the native demuxer.
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open a video or Blu-ray ISO",
            "",
            "Video / Blu-ray (*.mkv *.mk3d *.ssif *.m2ts *.ts *.mp4 *.avi *.iso);;"
            "Blu-ray disc image (*.iso);;"
            "Blu-ray streams (*.ssif *.m2ts *.ts);;All files (*.*)"
        )
        if file_path:
            self.play_file(file_path)

    def open_disc_dialog(self):
        """Open a Blu-ray 3D disc/folder: pick a drive letter (e.g. J:\\) or a BDMV folder.
        The feature film's SSIF is auto-detected (duration-based main-title detection)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open a Blu-ray 3D — pick the drive (e.g. J:\\) or the BDMV folder", ""
        )
        if folder:
            self.play_file(folder)

    def open_archive_dialog(self):
        """Open the disc-imaging dialog (configure → live throughput/ETA/animation → STOP)."""
        if getattr(self, '_archiving', False):
            return
        try:
            from sylc.disc_archiver import DiscArchiveDialog
            DiscArchiveDialog(self, parent=self).exec()
        except Exception as e:
            logger.error(f"[ARCHIVE] dialog error: {e}")
            import traceback
            traceback.print_exc()
            self.show_3d_notification(f"Archive error: {e}", success=False)

    def _mounted_iso_letters(self):
        """Drive letters of ISOs WE mounted (excluded from archiving — you already have them)."""
        out = set()
        for m in (getattr(self, '_active_iso_mount', None), getattr(self, '_pending_iso_mount', None)):
            if m and m[1]:
                L = str(m[1]).rstrip('\\').rstrip(':')[:1].upper()
                if L:
                    out.add(L)
        return out

    def _apply_preview_thumbs_policy(self, file_path):
        """Pick the thumbnail provider for this source (spec 2026-07-14).
        Physical optical → off (measured 45-120s head thrash). Player-mounted
        ISO → in-process edge264 with guardrails (a concurrent ffmpeg probe
        broke demuxer init on 2026-07-14 — the service is disarmed outside
        steady playback instead). Plain files → edge264 (H.264) or avcodec
        (in-process bundled avcodec for HEVC/other; NO ffmpeg.exe)."""
        try:
            from sylc import disc_archiver as da
            optical = set(da.list_optical_drives())
        except Exception:
            optical = set()
        codec = None
        try:
            codec = (self.video_3d_info or {}).get('codec_name')
        except Exception:
            pass
        mode, is_optical = _decide_thumbs_mode(
            file_path, self._mounted_iso_letters(), optical, codec)
        svc = self._ensure_thumbnail_service()
        dur = 0.0
        try:
            dur = float(self.controls_overlay.time_slider.maximum()) / 1000.0
        except Exception:
            pass
        # Packed-stereo sources: thumbnails show a SINGLE eye (sbs → left half,
        # tab → top half). MVC/2D need no crop (base view is one eye already).
        _sm = None
        _w = _h = 0
        try:
            _info = self.video_3d_info or {}
            _sm = _info.get('stereo_mode')
            _w, _h = _info.get('width') or 0, _info.get('height') or 0
        except Exception:
            pass
        layout = _sm if _sm in ('sbs', 'tab') else None
        # Half-packed detection (same thresholds as the format badge: Full-SBS at
        # ≥2560px wide, Full-TAB at ≥1600px tall). A half eye is spatially squeezed,
        # so the thumbnail un-squeezes its display aspect. HEVC refines this later
        # via set_layout(half=) once the SEI/side-data verdict lands.
        half = (_sm == 'sbs' and 0 < _w < 2560) or (_sm == 'tab' and 0 < _h < 1600)
        svc.configure(file_path, dur, mode, optical=is_optical, layout=layout, half=half)
        slider = self.controls_overlay.time_slider
        # 'avcodec' now = the in-process ThumbnailService too (HEVC via LavfHevcSource,
        # bundled avcodec-62.dll — NO ffmpeg.exe). Only 'off' leaves the provider unset.
        slider.set_thumbnail_provider(svc if mode in ('edge264', 'avcodec') else None)
        slider.set_thumbnails_allowed(mode != 'off')
        logger.info(f"[THUMB] provider={mode} optical={is_optical} for {file_path}")

    def _ensure_thumbnail_service(self):
        """Create the (single) ThumbnailService lazily and wire its lifecycle:
        disarmed around seeks via the seek queue, thumbnails routed to the
        slider cache."""
        if getattr(self, '_thumb_service', None) is None:
            from sylc.thumbnail_service import ThumbnailService
            svc = ThumbnailService()
            svc.thumbnailReady.connect(
                self.controls_overlay.time_slider._on_service_thumbnail)
            svc.start(QThread.Priority.LowPriority)
            self._thumb_service = svc
            if getattr(self, '_seek_queue', None):
                self._seek_queue.seek_started.connect(lambda _t: svc.disarm())
                self._seek_queue.seek_completed.connect(
                    lambda: self._media_single_shot(500, svc.arm))
        return self._thumb_service

    def _is_physical_bluray(self, letter):
        """True iff `letter` is a physical optical drive holding a Blu-ray (BDMV) — i.e. not a
        mounted ISO and not a non-BD disc. This is the ONLY thing we allow imaging."""
        if not letter:
            return False
        try:
            from sylc import disc_archiver as da
            from sylc import bluray_disc
            letter = letter.upper()
            if letter in self._mounted_iso_letters():
                return False
            if letter not in da.list_optical_drives():
                return False
            return bool(bluray_disc.is_bluray_path(f"{letter}:\\"))
        except Exception:
            return False

    def _archivable_disc_drive(self):
        """Letter of the physical Blu-ray currently loaded, else None. Drives the archive
        button's enabled state so it lights up ONLY for a Blu-ray source (never MKV/ISO)."""
        if not getattr(self, 'current_file_path', None):
            return None
        d = os.path.splitdrive(os.path.abspath(self.current_file_path))[0]
        letter = d[0].upper() if d else None
        return letter if self._is_physical_bluray(letter) else None

    def _update_archive_button_state(self):
        """EX-4: the former ISO button is now the « Sauvegarde / Export » MENU
        button — it must stay reachable for MV-HEVC export on ANY 3D source, not
        just a Blu-ray disc. Its MENU entries self-gate (the ISO entry only for a
        physical Blu-ray, the export entry only for a 3D source with tools), so
        the button itself is only disabled while a disc→ISO image runs."""
        try:
            self.controls_overlay.archive_button.setEnabled(not getattr(self, '_archiving', False))
        except Exception:
            pass

    def _resolve_archive_source(self):
        """Blu-ray discs ONLY. Returns {found, ready, kind:'volume', drive, label, length, error}.
        Imaging is offered solely for a physical Blu-ray optical disc — never a mounted ISO,
        an MKV, or any other source."""
        from sylc import disc_archiver as da

        def vol(letter):
            info = da.probe_volume(letter)
            return {"found": True, "ready": bool(info.get("ok")), "kind": "volume",
                    "drive": letter, "label": info.get("label", ""),
                    "length": info.get("length", 0), "iso_path": "",
                    "error": info.get("error", "")}

        # the disc currently playing, if it's a physical Blu-ray
        cur = self._archivable_disc_drive()
        if cur:
            return vol(cur)
        # otherwise a physical Blu-ray sitting in a drive (with ready media)
        bd = [c for c in da.list_optical_drives() if self._is_physical_bluray(c)]
        for c in bd:
            r = vol(c)
            if r["ready"]:
                return r
        if bd:
            return vol(bd[0])
        return {"found": False, "ready": False, "kind": "volume", "drive": "", "label": "",
                "length": 0, "iso_path": "",
                "error": "ISO saving is only possible from a Blu-ray disc."}

    def _begin_archive_lock(self):
        """Lock playback while imaging: stop the player (releases disc handles) and disable
        the transport so the drive is dedicated to a clean sequential read."""
        self._archiving = True
        try:
            if getattr(self, 'has_media', False):
                self.stop_playback()
        except Exception as e:
            logger.warning(f"[ARCHIVE] stop during lock: {e}")
        for attr in ('play_pause_button', 'archive_button', 'skip_back_button', 'skip_forward_button'):
            try:
                getattr(self.controls_overlay, attr).setEnabled(False)
            except Exception:
                pass
        logger.info("[ARCHIVE] playback locked for disc imaging")

    def _end_archive_lock(self):
        """Release the playback lock after imaging finishes/cancels."""
        self._archiving = False
        for attr in ('play_pause_button', 'skip_back_button', 'skip_forward_button'):
            try:
                getattr(self.controls_overlay, attr).setEnabled(True)
            except Exception:
                pass
        self._update_archive_button_state()
        logger.info("[ARCHIVE] playback lock released")

    def _export_eye_order_override(self):
        """Return the explicit export-menu choice, or ``unknown`` for Auto."""
        ov = getattr(self, 'controls_overlay', None)
        try:
            if ov.export_eye_left_action.isChecked():
                return LEFT_FIRST
            if ov.export_eye_right_action.isChecked():
                return RIGHT_FIRST
        except Exception:
            pass
        return UNKNOWN

    def _export_desc_with_eye_order(self, desc, detected_order, source):
        """Attach an auditable eye-order decision to an exporter descriptor."""
        override = self._export_eye_order_override()
        detected = normalise_eye_order(detected_order)
        if override != UNKNOWN:
            desc['eye_order'] = override
            desc['eye_order_source'] = 'user override'
        else:
            desc['eye_order'] = detected
            desc['eye_order_source'] = source if detected != UNKNOWN else 'not signalled'
        # The exporter probes duration itself, but the player already has a
        # container-duration verdict.  Preserve it as a fallback for optical or
        # unusual demuxers whose ffprobe AVStream fields are empty.
        info = getattr(self, 'video_3d_info', None) or {}
        try:
            duration_hint = float(info.get('duration') or 0.0)
        except (TypeError, ValueError):
            duration_hint = 0.0
        if duration_hint > 0.0:
            desc['duration_s'] = duration_hint
        return desc

    def _is_half_packed_source(self):
        """True if the currently loaded packed-stereo (sbs/tab) source is
        HALF-packed (each eye squeezed anamorphically into half the packed frame,
        no PAR signaling) rather than Full-SBS/Full-TAB. v1 MV-HEVC export spec
        is Full only — a half source exported per-view with no PAR metadata would
        play back squeezed on Apple devices.

        Detection: the HEVC avcodec path resolves this authoritatively via SEI/
        side-data and stores it in `_hevc_half` (set in _try_start_hevc, ~L7231).
        For H.264 packed content there is no equivalent stored verdict — fall
        back to the same width/height thresholds the format badge / thumbnail
        crop use (~L5040): Full-SBS is >=2560px wide, Full-TAB is >=1600px tall;
        anything narrower/shorter at that stereo_mode is half."""
        info = getattr(self, 'video_3d_info', None) or {}
        sm = info.get('stereo_mode')
        if sm not in ('sbs', 'tab'):
            return False
        if getattr(self, '_hevc_half', False):
            return True
        w = info.get('width') or 0
        h = info.get('height') or 0
        return (sm == 'sbs' and 0 < w < 2560) or (sm == 'tab' and 0 < h < 1600)

    def _current_export_source_desc(self):
        """Derive the MV-HEVC exporter `source_desc` for the media currently
        loaded, or None if it is not an exportable 3D source.

        Precedence MATTERS: native MV-HEVC and a BD dual-file session both
        masquerade as stereo_mode=='mvc' in video_3d_info, so MV-HEVC
        (hevc_media_info.multiview) is resolved BEFORE the generic MVC branch,
        and the dual-file pair before the single-file MVC branch. Shapes mirror
        mvhevc_exporter.build_adapter / tests\\export\\test_exporter.py:
          packed  -> {'path','kind':'packed','packing':'sbs'|'tab','eye_order'}
          mvhevc  -> {'path','kind':'mvhevc','eye_order'}
          mvc     -> {'path','kind':'mvc','eye_order'
                      [, 'mvc_container':'dual','dep_path']}
        """
        # 2D->3D AI (spec round-5c §0bis): the offline synthesis export is
        # available exactly WHEN the viewer has the AI active on a 2D file —
        # never systematically. The descriptor carries every synth3d parameter
        # so the exporter's detached renderer reproduces the live tuning at
        # offline (per-frame inference) quality.
        if (getattr(self, '_synth3d_active', False)
                and not self._content_is_3d()
                and getattr(self, 'current_file_path', None)):
            try:
                model, side = self._synth3d_model_path()
            except Exception:
                model, side = None, 0
            return {
                'path': self.current_file_path,
                'kind': 'synth3d',
                'eye_order': LEFT_FIRST,      # synthesized: base view IS left
                'eye_order_source': '2D->3D AI synthesis',
                'duration_s': (getattr(self, 'video_3d_info', None) or {}).get('duration'),
                'synth3d': {
                    'strength_pct': float(getattr(self, '_synth3d_strength', 1.5)),
                    'convergence': float(getattr(self, '_synth3d_convergence', 0.5)),
                    'auto_convergence': bool(getattr(self, '_synth3d_auto_convergence', False)),
                    'temporal_fill': bool(getattr(self, '_synth3d_temporal_fill', False)),
                    'model_path': str(model or ''),
                    'ort_dir': str(self._synth3d_ort_dir(model) or '') if model else '',
                    'side': int(side or 0),
                },
            }
        if not self._content_is_3d():
            return None
        path = getattr(self, 'current_file_path', None)
        info = getattr(self, 'video_3d_info', None) or {}

        # 1) native MV-HEVC (avcodec / LavfHevcSource multiview) — remux fast-path
        mi = getattr(self, 'hevc_media_info', None)
        if mi is not None and getattr(mi, 'multiview', False):
            if not path:
                return None
            # LavfHevcSource already returns a semantic (left,right) pair using
            # left_view_id.  "right first" here is therefore only a deliberate
            # user override and forces the exporter out of the remux fast-path.
            return self._export_desc_with_eye_order(
                {'path': path, 'kind': 'mvhevc'}, LEFT_FIRST,
                'MV-HEVC left_view_id mapping')

        sm = info.get('stereo_mode')
        # 2) MVC (BD dual-file / SSIF / single-file MKV-M2TS)
        if sm == 'mvc' or info.get('has_mvc_track'):
            mvc_order = normalise_eye_order(
                getattr(self, '_bd_eye_order', UNKNOWN))
            mvc_source = 'Blu-ray MPLS MVC_Base_view_R_flag'
            if mvc_order == UNKNOWN:
                mvc_order = normalise_eye_order(info.get('eye_order'))
                mvc_source = info.get('eye_order_source') or 'Matroska StereoMode'
            pair = getattr(self, '_bd_dual_file_pair', None)
            if getattr(self, '_bd_dual_active', False) and pair and len(pair) == 2:
                base, dep = pair[0], pair[1]
                if base and dep and os.path.isfile(base) and os.path.isfile(dep):
                    return self._export_desc_with_eye_order(
                        {'path': base, 'kind': 'mvc',
                         'mvc_container': 'dual', 'dep_path': dep},
                        mvc_order, mvc_source)
            if not path:
                return None
            return self._export_desc_with_eye_order(
                {'path': path, 'kind': 'mvc'}, mvc_order, mvc_source)

        # 3) packed Full-SBS / Full-TAB (H.264 or HEVC) — HALF-packed sources are
        # refused (v1 export spec is Full only; see _is_half_packed_source).
        if sm in ('sbs', 'tab'):
            if self._is_half_packed_source():
                return None
            if not path:
                return None
            packed_order = normalise_eye_order(info.get('eye_order'))
            packed_source = info.get('eye_order_source') or 'container metadata'
            # The avcodec HEVC path has the authoritative AVStereo3D flag even
            # when this ffprobe build omitted it from Video3DAnalyzer.
            if (mi is not None and not getattr(mi, 'multiview', False)
                    and getattr(mi, 'stereo_hint', None) == sm):
                packed_order = (RIGHT_FIRST if mi.stereo_inverted else LEFT_FIRST)
                packed_source = 'AVStereo3D side data'
            return self._export_desc_with_eye_order(
                {'path': path, 'kind': 'packed', 'packing': sm},
                packed_order, packed_source)

        return None

    def _export_tools_missing(self):
        """List the export tools that are absent (empty list = all present).
        Used for the explicit disabled-entry tooltip and the launch guard."""
        try:
            from sylc import mvhevc_exporter as me
        except Exception as e:
            return [f'mvhevc_exporter ({e})']
        missing = []
        if not os.path.isfile(me.X265):
            missing.append('x265.exe')
        if not os.path.isfile(me.MP4BOX):
            missing.append('mp4box.exe')
        if not shutil.which('ffmpeg'):
            missing.append('ffmpeg')
        if not shutil.which('ffprobe'):
            missing.append('ffprobe')
        return missing

    def _update_export_menu_state(self):
        """Refresh the « Sauvegarde / Export » menu entries just before it opens
        (wired to export_menu.aboutToShow). ISO entry follows the exact same rule
        as the former archive button; the MV-HEVC submenu is enabled only for an
        exportable 3D source with the tools present, and its title/tooltip reflect
        the source (remux « copie sans réencodage » vs re-encode) or the reason
        it is disabled."""
        ov = self.controls_overlay
        # --- « Créer un ISO du disque » : physical Blu-ray only (unchanged rule) ---
        try:
            iso_ok = (self._archivable_disc_drive() is not None
                      and not getattr(self, '_archiving', False))
            ov.iso_action.setEnabled(iso_ok)
            ov.iso_action.setToolTip(
                "" if iso_ok else "Available only for an inserted Blu-ray disc.")
        except Exception:
            pass
        # --- « Exporter en MV-HEVC » ---
        try:
            sub = ov.export_mvhevc_menu
            desc = self._current_export_source_desc()
            missing = self._export_tools_missing()
            busy = self._export_job is not None and self._export_job.isRunning()
            title = "Export to MV-HEVC (.mov)"
            if desc is None:
                enabled = False
                if self._is_half_packed_source():
                    tip = ("Half-resolution formats cannot be exported to MV-HEVC "
                           "(v1 supports Full-SBS/Full-TAB only).")
                else:
                    tip = ("No exportable 3D source — load an MVC, MV-HEVC, "
                           "Full-SBS or Full-TAB file, or enable 2D→3D AI on "
                           "a 2D video to export an AI conversion.")
            elif missing:
                enabled = False
                tip = ("Missing export tool: " + ", ".join(missing)
                       + " (tools\\x265 + tools\\gpac; ffmpeg/ffprobe on PATH).")
            elif busy:
                enabled = False
                tip = "An MV-HEVC export is already running."
            else:
                enabled = True
                tip = ""
                if (desc.get('kind') == 'mvhevc'
                        and normalise_eye_order(desc.get('eye_order')) != RIGHT_FIRST):
                    title = "Export to MV-HEVC (no re-encoding)"
                elif desc.get('kind') == 'synth3d':
                    title = "Export to MV-HEVC (2D→3D AI)"
            sub.setTitle(title)
            sub.setEnabled(enabled)
            eye_menu = getattr(ov, 'export_eye_order_menu', None)
            if eye_menu is not None:
                order = normalise_eye_order(
                    desc.get('eye_order') if desc else UNKNOWN)
                order_label = {
                    LEFT_FIRST: "Left first",
                    RIGHT_FIRST: "Right first",
                    UNKNOWN: "Ask on export",
                }[order]
                eye_menu.setTitle(f"Source eye order  ·  {order_label}")
                eye_menu.setEnabled(desc is not None and not busy)
            ma = sub.menuAction()
            if ma is not None:
                ma.setEnabled(enabled)
                ma.setToolTip(tip)
        except Exception as e:
            logger.debug(f"[EXPORT] menu state refresh failed: {e}")
        # --- « Diffuser vers Quest » : live 3D session + NVENC cast pipeline ---
        try:
            cast_sub = getattr(ov, 'cast_menu', None)
            if cast_sub is not None:
                if self._cast is not None:
                    # Already casting -> keep it enabled so the user can toggle it off.
                    cast_ok, ctip = True, ""
                else:
                    renderer = self._cast_renderer()
                    # A 3D session is either the classic edge264/MVC path (mvc_mode_active)
                    # OR the HEVC 3D path (_hevc_mode_active) — the latter never sets
                    # mvc_mode_active, so gate on both (repo idiom, see :3356). _cast_renderer()
                    # resolves the framepack display renderer for both.
                    session_3d = (getattr(self, 'mvc_mode_active', False)
                                  or getattr(self, '_hevc_mode_active', False))
                    if not session_3d or renderer is None:
                        cast_ok = False
                        ctip = ("Available during 3D playback "
                                "(MultiView / Framepack mode).")
                    else:
                        try:
                            avail = bool(renderer.cast_available())
                        except Exception:
                            avail = False
                        cast_ok = avail
                        ctip = "" if avail else (
                            "Streaming unavailable — no NVENC encoder "
                            "(NVIDIA GPU) was found.")
                cast_sub.setEnabled(cast_ok)
                cma = cast_sub.menuAction()
                if cma is not None:
                    cma.setEnabled(cast_ok)
                    cma.setToolTip(ctip)
        except Exception as e:
            logger.debug(f"[CAST] menu state refresh failed: {e}")

    def _resolve_export_out_path(self, src_path):
        """Default output = <source_basename>_MVHEVC.mov beside the source. For a
        disc/ISO (optical / non-writable) source, ask the user for a destination
        directory via QFileDialog (patron disc_archiver). Returns the .mov path,
        or None if the user cancelled the destination dialog."""
        base = os.path.splitext(os.path.basename(src_path))[0] or 'export'
        name = base + '_MVHEVC.mov'
        src_dir = os.path.dirname(os.path.abspath(src_path)) or '.'
        on_optical = False
        try:
            from sylc import disc_archiver as da
            drv = os.path.splitdrive(os.path.abspath(src_path))[0].rstrip(':').upper()[:1]
            on_optical = bool(drv) and drv in set(da.list_optical_drives())
        except Exception:
            on_optical = False
        writable = os.path.isdir(src_dir) and os.access(src_dir, os.W_OK)
        if writable and not on_optical:
            return os.path.join(src_dir, name)
        out_dir = QFileDialog.getExistingDirectory(
            self, "Export to MV-HEVC — destination folder", os.path.expanduser('~'))
        if not out_dir:
            return None
        return os.path.join(out_dir, name)

    def _resolve_unknown_export_eye_order(self, desc):
        """Require an explicit decision when the source carries no eye mapping."""
        if normalise_eye_order(desc.get('eye_order')) != UNKNOWN:
            return True
        if desc.get('kind') == 'mvc':
            subject = "the MVC base view"
        elif desc.get('packing') == 'tab':
            subject = "the first (top) packed image"
        else:
            subject = "the first (left) packed image"
        reply = QMessageBox.question(
            self, "Source eye order",
            "This source does not contain trustworthy left/right eye metadata.\n\n"
            f"Is {subject} the LEFT eye?\n\n"
            "Yes = left eye first   •   No = right eye first",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if reply not in (QMessageBox.StandardButton.Yes,
                          QMessageBox.StandardButton.No):
            return False
        order = (LEFT_FIRST if reply == QMessageBox.StandardButton.Yes
                 else RIGHT_FIRST)
        desc['eye_order'] = order
        desc['eye_order_source'] = 'user confirmation (metadata absent)'
        try:
            action = (self.controls_overlay.export_eye_left_action
                      if order == LEFT_FIRST
                      else self.controls_overlay.export_eye_right_action)
            action.setChecked(True)
        except Exception:
            pass
        logger.info("[EXPORT] ambiguous source eye order resolved by user: %s", order)
        return True

    def start_mvhevc_export(self, quality='quality'):
        """EX-4 entry point: launch a background MV-HEVC export of the CURRENT 3D
        media. Isolation invariant: this NEVER stops/pauses playback — the
        exporter runs on the file's paths with its own detached decode instances,
        so playback of the same (or any) file continues undisturbed. One export
        job at a time (a second request is politely refused)."""
        if self._export_job is not None and self._export_job.isRunning():
            self.show_3d_notification("An MV-HEVC export is already running.", success=False)
            return
        desc = self._current_export_source_desc()
        if desc is None:
            # Fix #1 defense-in-depth: the menu entry is already disabled for a
            # half-packed source (see _update_export_menu_state), but start_mvhevc_export
            # can in principle be invoked directly — refuse it here too with the
            # same specific reason rather than the generic "no source" message.
            if self._is_half_packed_source():
                self.show_3d_notification(
                    "Half-resolution formats cannot be exported to MV-HEVC "
                    "(v1 supports Full-SBS/Full-TAB only).", success=False)
            else:
                self.show_3d_notification(
                    "No exportable 3D source (MVC / MV-HEVC / Full-SBS / Full-TAB).",
                    success=False)
            return
        missing = self._export_tools_missing()
        if missing:
            self.show_3d_notification(
                "Cannot export — missing tool: " + ", ".join(missing), success=False)
            return
        if not self._resolve_unknown_export_eye_order(desc):
            return
        out_path = self._resolve_export_out_path(desc['path'])
        if not out_path:
            return  # user cancelled the destination dialog
        if os.path.exists(out_path):
            # Fix #3: the export silently overwrote an existing out_path — confirm first.
            reply = QMessageBox.question(
                self, "Export to MV-HEVC",
                f"Overwrite {os.path.basename(out_path)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            from sylc import mvhevc_exporter as me
        except Exception as e:
            self.show_3d_notification(f"Export unavailable: {e}", success=False)
            return
        opts = {'quality': 'fast' if quality == 'fast' else 'quality'}
        job = me.MVHEVCExporter(desc, out_path, opts)
        self._export_job = job
        reencode_hint = desc.get('kind') != 'mvhevc'
        dlg = MVHEVCExportDialog(self, out_path, reencode_hint=reencode_hint, parent=self)
        self._export_dialog = dlg
        job.progress.connect(dlg.on_progress)
        job.exportFinished.connect(self._on_export_finished)
        job.failed.connect(self._on_export_failed)
        job.finished.connect(lambda j=job: self._export_cleanup(j))
        dlg.show()          # NON-modal: playback + the rest of the UI stay live
        job.start()
        logger.info(
            "[EXPORT] started: kind=%s quality=%s eye_order=%s eye_source=%s -> %s",
            desc.get('kind'), opts['quality'], desc.get('eye_order'),
            desc.get('eye_order_source'), out_path)
        self._update_export_menu_state()

    def _on_export_finished(self, out_path):
        if self._export_dialog is not None:
            try:
                self._export_dialog.set_finished(out_path)
            except Exception:
                pass
        job = getattr(self, '_export_job', None)
        if getattr(job, 'audio_status', None) == 'failed':
            # Video-only is a deliberate, non-fatal fallback in the exporter —
            # but it must never be silent, or the user discovers a mute movie
            # after an hour-long encode they were told had succeeded.
            self.show_3d_notification(
                f"MV-HEVC export complete, but WITHOUT audio (the source track "
                f"could not be converted): {out_path}", success=False)
        else:
            self.show_3d_notification(f"MV-HEVC export complete: {out_path}", success=True)
        logger.info(f"[EXPORT] finished -> {out_path}")

    def _on_export_failed(self, reason):
        if self._export_dialog is not None:
            try:
                self._export_dialog.set_failed(reason)
            except Exception:
                pass
        if reason == 'annule':
            self.show_3d_notification("MV-HEVC export cancelled.", success=False)
        else:
            self.show_3d_notification(f"MV-HEVC export failed: {reason}", success=False)
        logger.info(f"[EXPORT] failed: {reason}")

    def _export_cleanup(self, job):
        """QThread.finished slot: drop the job reference only once run() has fully
        returned (the QThread object is kept alive until then via this connection's
        closure, so it is never GC'd mid-run). Frees a new export to start."""
        try:
            if self._export_job is job:
                self._export_job = None
        finally:
            try:
                job.deleteLater()
            except Exception:
                pass
            self._update_export_menu_state()
            # Fix #2: the job just ended — retry any ISO dismount that was
            # deferred while it was reading from a mounted volume.
            self._retry_deferred_iso_dismounts()

    def _stop_export_job(self, timeout_ms=7000):
        """Cancel and join the detached export before sources/Qt are destroyed."""
        job = getattr(self, '_export_job', None)
        if job is None:
            return True
        try:
            running = bool(job.isRunning())
        except RuntimeError:
            self._export_job = None
            return True
        if running:
            logger.info("[EXPORT] Cancelling active export for application shutdown")
            try:
                job.cancel()
            except Exception:
                logger.exception("[EXPORT] cancel() failed during shutdown")
            try:
                running = not bool(job.wait(int(timeout_ms)))
            except RuntimeError:
                running = False
        if running:
            logger.critical(
                "[EXPORT] Export thread did not stop within %.1fs",
                timeout_ms / 1000.0)
            return False
        self._export_job = None
        try:
            job.deleteLater()
        except RuntimeError:
            pass
        return True

    def _export_job_source_drives(self):
        """Drive letters used by the CURRENTLY RUNNING export job's source
        paths (base + dependent view for a BD dual-file source). Empty set
        when no export is running."""
        job = getattr(self, '_export_job', None)
        if job is None:
            return set()
        try:
            if not job.isRunning():
                return set()
        except Exception:
            return set()
        desc = getattr(job, 'source_desc', None) or {}
        drives = set()
        for key in ('path', 'dep_path'):
            p = desc.get(key)
            if not p:
                continue
            try:
                drv = os.path.splitdrive(os.path.abspath(p))[0].rstrip(':').upper()[:1]
            except Exception:
                drv = ''
            if drv:
                drives.add(drv)
        return drives

    def _dismount_iso_or_defer(self, mount, label='ISO'):
        """Dismount ONE (iso_path, drive) mount tuple, unless a running export
        job is still reading from that drive — in which case remember it in
        `_deferred_iso_dismounts` and retry once the job ends (see
        _retry_deferred_iso_dismounts, hooked to _export_cleanup). Returns True
        if dismounted (or nothing to do), False if the dismount was deferred."""
        if not mount:
            return True
        letter = str(mount[1] or '').rstrip('\\').rstrip(':')[:1].upper()
        if letter and letter in self._export_job_source_drives():
            logger.info(f"[EXPORT] demontage differe: export en cours depuis {letter}:")
            if not any(m[0] == mount[0] for m in self._deferred_iso_dismounts):
                self._deferred_iso_dismounts.append(mount)
            return False
        from sylc import bluray_disc
        logger.info(f"[DISC] Dismounting {label}: {mount[0]}")
        bluray_disc.dismount_iso(mount[0])
        return True

    def _retry_deferred_iso_dismounts(self):
        """Called when an export job ends (finished or failed, via
        _export_cleanup): retry any ISO dismount(s) that were skipped while the
        job was reading from that volume. Re-deferred (e.g. a new export
        started immediately) mounts stay queued for the next job to end."""
        pending = getattr(self, '_deferred_iso_dismounts', None)
        if not pending:
            return
        self._deferred_iso_dismounts = []
        for m in pending:
            self._dismount_iso_or_defer(m, label='deferred ISO')


__all__ = ['ArchiveExportMixin']

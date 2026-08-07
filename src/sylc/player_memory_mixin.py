# -*- coding: utf-8 -*-
"""Per-title playback memory and lightweight application settings."""

import json
import logging
import os
import time

from sylc.player_constants import PRESENTATION_KEYS


logger = logging.getLogger(__name__)


class PlayerMemoryMixin:
    def _playback_memory_store(self):
        store = getattr(self, '_playback_memory', None)
        if store is None:
            from sylc.playback_memory import PlaybackMemory
            # SYLC_PLAYBACK_MEMORY: test isolation — pytest points this at a
            # temp file so tests can never rewrite the user's real memory.
            store = PlaybackMemory(
                os.environ.get('SYLC_PLAYBACK_MEMORY')
                or os.path.join(os.path.expanduser('~'),
                                '.sylc3d_playback_memory.json'))
            self._playback_memory = store
        return store

    def _memory_key_path(self):
        """The stable identity of the current media: for a file living on a
        mounted ISO, the ISO itself — mount letters change between sessions
        (D: today, E: tomorrow) and must not fragment the memory."""
        path = getattr(self, 'current_file_path', None)
        if not path:
            return None
        try:
            drive = os.path.splitdrive(os.path.abspath(path))[0][:1].upper()
            for m in (getattr(self, '_active_iso_mount', None),
                      getattr(self, '_pending_iso_mount', None)):
                if not m:
                    continue
                letter = str(m[1] or '').rstrip('\\').rstrip(':')[:1].upper()
                if letter and drive == letter:
                    return m[0]
        except Exception:
            pass
        return path

    def _recall_for_file(self):
        """Load the current file's remembered fields; called once per load at
        the canonical per-file reset point."""
        self._file_memory = {}
        self._file_memory_applied = set()
        self._file_memory_last_pos_save = 0.0
        try:
            path = self._memory_key_path()
            if path:
                self._file_memory = self._playback_memory_store().recall(path)
                if self._file_memory:
                    logger.info(f"[FILE-MEMORY] recalled {sorted(self._file_memory)} "
                                f"for {os.path.basename(path)}")
        except Exception:
            logger.exception("[FILE-MEMORY] recall failed (ignored)")

    def _remember_for_file(self, **fields):
        """Persist per-file fields for the current media. Never raises."""
        try:
            path = self._memory_key_path()
            if not path:
                return
            self._playback_memory_store().remember(path, **fields)
            mem = getattr(self, '_file_memory', None)
            if isinstance(mem, dict):
                for name, value in fields.items():
                    if value is None:
                        mem.pop(name, None)
                    else:
                        mem[name] = value
        except Exception:
            logger.exception("[FILE-MEMORY] remember failed (ignored)")

    def _choose_initial_stereo_mode(self, detected_mode):
        """The presentation the combo should start on: the viewer's remembered
        pick for THIS file when valid, else the detected content mode."""
        remembered = (getattr(self, '_file_memory', None) or {}).get('stereo_mode')
        if remembered in PRESENTATION_KEYS:
            return remembered
        return detected_mode

    def _remember_position(self, final=False):
        """Record the resume position (throttled unless `final`). Positions in
        the first 30s or the last 7% mean 'no resume' — restarting there is
        better than resuming. Periodic ticks only ever UPGRADE the position
        and only after the deferred restore ran: the early ticks of a fresh
        session sit near 0s and would otherwise erase the very resume point
        the restore is about to apply (measured race, 2026-08-03 smoke)."""
        try:
            if not (self.has_media and getattr(self, 'current_file_path', None)):
                return
            now = time.monotonic()
            if not final:
                if 'deferred' not in getattr(self, '_file_memory_applied', set()):
                    return
                if now - getattr(self, '_file_memory_last_pos_save', 0.0) < 15.0:
                    return
            self._file_memory_last_pos_save = now
            pos = float(getattr(self, '_current_precise_time', 0.0) or 0.0)
            try:
                dur = float(self.controls_overlay.time_slider.maximum() or 0) / 1000.0
            except Exception:
                dur = 0.0
            in_resume_range = pos >= 30.0 and (dur <= 0 or pos <= 0.93 * dur)
            if in_resume_range:
                self._remember_for_file(resume_pos=round(pos, 3),
                                        duration=round(dur, 3) if dur > 0 else None)
            elif final:
                self._remember_for_file(resume_pos=None)
        except Exception:
            pass

    def _apply_deferred_file_memory(self):
        """Apply the remembered fields that need a RUNNING pipeline: resume
        position, 3D-off, per-title synth3d tuning. Scheduled once shortly
        after playback starts; every step is individually best-effort."""
        mem = getattr(self, '_file_memory', None) or {}
        applied = getattr(self, '_file_memory_applied', set())
        if not mem or 'deferred' in applied or not self.has_media:
            return
        applied.add('deferred')

        # Per-title synth3d tuning first (values only touch an active synthesis).
        try:
            if 'synth3d_strength' in mem:
                self.set_synth3d_strength(float(mem['synth3d_strength']) * 10.0,
                                          persist=False)
            if 'synth3d_convergence' in mem:
                self.set_synth3d_convergence(float(mem['synth3d_convergence']) * 100.0,
                                             persist=False)
        except Exception:
            logger.exception("[FILE-MEMORY] synth3d tuning restore failed")
        try:
            if (mem.get('synth3d_enabled') and not getattr(self, '_synth3d_active', False)
                    and self._synth3d_supported() and self._synth3d_eligible()):
                logger.info("[FILE-MEMORY] re-enabling 2D->3D AI (remembered)")
                self.toggle_synth3d(True)
                self._update_synth3d_menu_state()
        except Exception:
            logger.exception("[FILE-MEMORY] synth3d re-enable failed")

        # The viewer had turned 3D OFF for this title (ON is the content
        # default). NOT when the synthesis above just restored itself: on a 2D
        # title the AI owns is_3d_enabled (toggle_synth3d sets it), so applying
        # a remembered 3D-off here disabled the very output it had just brought
        # back — the 2026-08-04 report, where every replay of a title holding
        # both flags played flat. A remembered 3D-off speaks about CONTENT the
        # viewer downgraded; older stores still carry the contradictory pair,
        # so the guard reads the live synthesis state, not the stored flags.
        try:
            if (mem.get('three_d_enabled') is False and self.is_3d_enabled
                    and not getattr(self, '_synth3d_active', False)):
                logger.info("[FILE-MEMORY] disabling 3D output (remembered)")
                try:
                    btn = self.controls_overlay.mode_3d_button
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                except Exception:
                    pass
                self.toggle_3d_mode(False)
        except Exception:
            logger.exception("[FILE-MEMORY] 3D-off restore failed")

        # Resume position last (through the production seek path).
        try:
            pos = float(mem.get('resume_pos') or 0.0)
            try:
                dur = float(self.controls_overlay.time_slider.maximum() or 0) / 1000.0
            except Exception:
                dur = 0.0
            if pos > 30.0 and (dur <= 0 or pos < 0.93 * dur):
                logger.info(f"[FILE-MEMORY] resuming at {pos:.1f}s")
                self._handle_seek_request(pos)
                self.show_3d_notification(
                    f"Resumed at {self.controls_overlay._format_time(pos)}",
                    success=True)
        except Exception:
            logger.exception("[FILE-MEMORY] resume failed")

    def _apply_remembered_track(self, combo, field):
        """Select the remembered track in `combo` (by itemData) and return the
        track id to apply, or None. Marks the field applied exactly once."""
        mem = getattr(self, '_file_memory', None) or {}
        applied = getattr(self, '_file_memory_applied', set())
        if field in applied or field not in mem:
            return None
        applied.add(field)
        track_id = mem[field]
        try:
            idx = combo.findData(track_id)
            if idx < 0:
                return None
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            logger.info(f"[FILE-MEMORY] restoring {field}={track_id}")
            return track_id
        except Exception:
            logger.exception(f"[FILE-MEMORY] {field} restore failed")
            return None

    def _on_eye_order_picked(self, action):
        """Persist the export eye-order pick for this title."""
        if not self.has_media:
            return
        try:
            ov = self.controls_overlay
            order = ('left' if action is ov.export_eye_left_action
                     else 'right' if action is ov.export_eye_right_action
                     else 'auto')
            # 'auto' is the default: remembering it just deletes the override.
            self._remember_for_file(eye_order=None if order == 'auto' else order)
        except Exception:
            pass

    def _app_settings_path(self):
        return os.path.join(os.path.expanduser('~'), '.sylc3d_player.json')

    def _load_app_settings(self):
        try:
            with open(self._app_settings_path(), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_app_settings(self):
        try:
            with open(self._app_settings_path(), 'w', encoding='utf-8') as f:
                json.dump(self._app_settings, f, indent=2)
        except Exception as e:
            logger.warning(f"[SETTINGS] Could not save {self._app_settings_path()}: {e}")


__all__ = ['PlayerMemoryMixin']


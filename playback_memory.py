# playback_memory.py
"""Per-file playback memory for SyLC (user request 2026-08-03).

Stores the viewer's per-title choices so replaying a file restores the session
exactly as they tuned it: stereo presentation mode, 3D on/off, eye order,
audio/subtitle track, resume position, per-title synth3d tuning. One JSON map
in the user profile, keyed by the normalized absolute media path.

Design constraints:
  - never raise into the player: any I/O or schema problem degrades to an
    empty recall / skipped save (a broken memory must not break playback);
  - atomic writes (temp + os.replace) — a crash mid-save keeps the old file;
  - bounded: least-recently-updated records are pruned past `max_entries`.
"""
import json
import os
import tempfile
import time


class PlaybackMemory:
    def __init__(self, path, max_entries=400):
        self.path = path
        self.max_entries = int(max_entries)
        self._data = self._load()

    # --- keys -----------------------------------------------------------
    @staticmethod
    def _key(media_path):
        try:
            return os.path.normcase(os.path.normpath(os.path.abspath(media_path)))
        except Exception:
            return str(media_path)

    # --- persistence ----------------------------------------------------
    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self):
        try:
            directory = os.path.dirname(self.path) or '.'
            fd, tmp = tempfile.mkstemp(prefix='.playback_memory-',
                                       suffix='.tmp', dir=directory)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            # Losing one memory update is always preferable to interfering
            # with playback; the next successful save catches up.
            pass

    def _prune(self):
        if len(self._data) <= self.max_entries:
            return
        by_age = sorted(self._data.items(),
                        key=lambda kv: kv[1].get('updated', 0.0))
        for key, _ in by_age[:len(self._data) - self.max_entries]:
            del self._data[key]

    # --- API ------------------------------------------------------------
    def recall(self, media_path):
        """The remembered fields for this media, or {}. Always a copy."""
        rec = self._data.get(self._key(media_path))
        if not isinstance(rec, dict):
            return {}
        out = dict(rec)
        out.pop('updated', None)
        return out

    def remember(self, media_path, **fields):
        """Merge fields into the media's record. A None value deletes the
        field; a record left empty is dropped entirely."""
        key = self._key(media_path)
        rec = self._data.get(key)
        if not isinstance(rec, dict):
            rec = {}
        for name, value in fields.items():
            if value is None:
                rec.pop(name, None)
            else:
                rec[name] = value
        meaningful = [k for k in rec if k != 'updated']
        if not meaningful:
            self._data.pop(key, None)
        else:
            rec['updated'] = time.time()
            self._data[key] = rec
        self._prune()
        self._save()

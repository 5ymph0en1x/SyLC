# -*- coding: utf-8 -*-
"""Progress dialog for MV-HEVC exports."""

from sylc import disc_archiver
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)

class MVHEVCExportDialog(QDialog):
    """EX-4 non-modal progress window for a background MV-HEVC export job.

    NON-MODAL on purpose: playback and the rest of the UI stay interactive while
    the export runs (the isolation invariant — an export must never freeze or
    lock the player, unlike the disc→ISO archiver which does lock playback). It
    surfaces the running step + frames/fps/ETA (re-encode path) or the step name
    (remux fast-path), a Cancel button wired to the job's cancel(), and a
    completion/failure end-state with a Close button. It only READS/updates its
    own widgets from the exporter's progress/exportFinished/failed signals; the
    host (PlayerWindow) owns the MVHEVCExporter and the completion/failure toasts.
    """

    _STEP_LABELS = {
        'probe': "Analyzing source…",
        'copy': "Copying without re-encoding…",
        'extracting': "Extracting video stream…",
        'audio': "Processing audio track…",
        'muxing': "Building .mov container…",
        'validating': "Validating output…",
        'encoding': "Encoding with x265…",
        'encoded': "Encoding complete, finalizing…",
        'done': "Done.",
    }

    def __init__(self, player, out_path, reencode_hint=True, parent=None):
        super().__init__(parent)
        self.player = player
        self.out_path = out_path
        self._done = False
        self._total_frames = 0
        self._audio_status = None
        self.setWindowTitle("Export to MV-HEVC")
        self.setModal(False)
        self.setMinimumWidth(460)
        try:
            from sylc import disc_archiver
            self.setStyleSheet(disc_archiver._DIALOG_QSS)
        except Exception:
            pass
        self._build_ui(reencode_hint)

    def _build_ui(self, reencode_hint):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        title = QLabel("Export MV-HEVC (.mov)")
        title.setObjectName("title")
        root.addWidget(title)

        self.subtitle = QLabel("Re-encoding with x265…" if reencode_hint
                               else "Copying without re-encoding…")
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setWordWrap(True)
        root.addWidget(self.subtitle)

        self.status = QLabel("Preparing…")
        self.status.setObjectName("source")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)      # busy/indeterminate until an encode % arrives
        self.bar.setTextVisible(False)
        root.addWidget(self.bar)

        self.detail = QLabel("")
        self.detail.setObjectName("stat")
        root.addWidget(self.detail)

        dest = QLabel(f"→ {self.out_path}")
        dest.setObjectName("subtitle")
        dest.setWordWrap(True)
        root.addWidget(dest)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self._cancel)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.hide()
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

    @staticmethod
    def _fmt_eta(s):
        try:
            s = int(s)
        except Exception:
            return "—"
        if s < 0:
            return "—"
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    def on_progress(self, d):
        """Slot for MVHEVCExporter.progress(dict). Re-encode dicts carry
        frames_done/total_frames/fps/eta_s; remux dicts are step-based only."""
        if self._done or not isinstance(d, dict):
            return
        step = d.get('step', '')
        mode = d.get('mode')
        if d.get('audio_status'):
            # Emitted with the terminal 'done' payload; kept so set_finished()
            # can warn when the soundtrack silently didn't make it (the export
            # deliberately continues video-only rather than losing the encode).
            self._audio_status = d['audio_status']
        if mode == 'reencode':
            self.subtitle.setText("Re-encoding with x265…")
        elif mode in ('remux-tier1', 'remux-tier2'):
            self.subtitle.setText("Copying without re-encoding…")
        self.status.setText(self._STEP_LABELS.get(step, step or "…"))
        if step == 'encoding':
            reported_total = int(d.get('total_frames') or 0)
            if reported_total > 0:
                self._total_frames = reported_total
            total = self._total_frames
            done = int(d.get('frames_done') or 0)
            if total > 0:
                self.bar.setRange(0, total)
                self.bar.setValue(min(done, total))
            else:
                self.bar.setRange(0, 0)
            fps = d.get('fps') or 0.0
            eta = d.get('eta_s', -1)
            # Defense in depth: older/background exporters may omit eta_s while
            # still supplying total+fps.  The dialog can derive the same value.
            if (eta is None or eta < 0) and total > done and fps > 0:
                eta = int((total - done) / fps)
            self.detail.setText(
                f"{done}/{total if total else '?'} frames · {fps:.1f} fps · "
                f"{self._fmt_eta(eta)} remaining")
        else:
            self.bar.setRange(0, 0)     # step-based (remux) / non-frame steps: busy
            self.detail.setText("")

    def set_finished(self, out_path):
        self._done = True
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self.status.setText("Export complete.")
        self.subtitle.setText(f"File created: {out_path}")
        self.detail.setText(
            "⚠ The audio track could not be converted — the file is video-only."
            if self._audio_status == 'failed' else "")
        self.cancel_btn.hide()
        self.close_btn.show()

    def set_failed(self, reason):
        self._done = True
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        if reason == 'annule':
            self.status.setText("Export cancelled.")
            self.subtitle.setText("Temporary files removed.")
        else:
            self.status.setText("Export failed.")
            self.subtitle.setText(str(reason))
        self.detail.setText("")
        self.cancel_btn.hide()
        self.close_btn.show()

    def _cancel(self):
        job = getattr(self.player, '_export_job', None)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")
        self.status.setText("Cancellation in progress…")
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass

    def closeEvent(self, event):
        # Closing the window mid-export cancels the job (an export must not
        # outlive its dialog silently); a finished/failed job just closes.
        job = getattr(self.player, '_export_job', None)
        if not self._done and job is not None and job.isRunning():
            try:
                job.cancel()
            except Exception:
                pass
        super().closeEvent(event)


__all__ = ['MVHEVCExportDialog']


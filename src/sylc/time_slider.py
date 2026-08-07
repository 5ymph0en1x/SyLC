# -*- coding: utf-8 -*-
"""Timeline slider and its isolated thumbnail extraction helpers."""

import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSlider

from sylc.player_widgets import PreviewTooltip
from sylc.video_3d_analyzer import _resolve_external_tool


logger = logging.getLogger(__name__)

_thumbnail_executor = ThreadPoolExecutor(max_workers=2)


def _extract_thumbnail_ffmpeg(video_file, time_pos):
    """Extract a thumbnail with ffmpeg (worker function for ThreadPoolExecutor)."""
    try:
        ffmpeg_path = _resolve_external_tool('ffmpeg')
        if not ffmpeg_path:
            logger.warning("[PREVIEW] ffmpeg not found. Preview thumbnails disabled.")
            return None

        temp_file = os.path.join(tempfile.gettempdir(), f"preview_{int(time.time() * 1000000)}.jpg")

        cmd = [
            ffmpeg_path,
            '-ss', str(time_pos),
            '-i', video_file,
            '-frames:v', '1',
            '-vf', 'scale=120:-1',
            '-q:v', '8',
            '-y',
            temp_file
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            creationflags=creationflags
        )

        if result.returncode == 0 and os.path.exists(temp_file):
            return temp_file
        return None
    except:
        return None


def _decide_thumbs_mode(file_path, mounted_iso_letters, optical_letters, codec_name=None):
    """Thumbnail provider decision. Physical optical → 'off' (single head,
    measured 45-120s thrash with a third reader). Player-mounted ISO → 'off'
    too: the in-process thumbnail path opens its OWN demuxer + edge264 session
    and reads the same mounted-UDF volume concurrently with playback, which
    corrupts the volume's reads (Avatar 2026-07-14; GITS dual-file BD3D crash
    2026-07-22: provider=edge264 armed on D:\\BDMV\\STREAM\\00005.m2ts →
    0xe24c4a02 fault cascade). Optical-class = OFF, physical AND mounted ISO,
    regardless of codec — the proven rule. Plain H.264 file → 'edge264'. Plain
    non-H.264 → 'avcodec' (the in-process bundled-avcodec path; NO ffmpeg.exe)."""
    if not file_path:
        return 'off', False
    EDGE_EXTS = {'.ssif', '.m2ts', '.ts', '.mkv', '.mk3d'}
    ext = os.path.splitext(file_path)[1].lower()
    codec = (codec_name or '').lower()
    # Codec verdict wins over the extension shortcut: HEVC lives in .mkv/.ts too,
    # so `ext in EDGE_EXTS` would (wrongly) route HEVC to edge264 (H.264-only) and
    # leave HEVC files with NO hover preview. edge264 only when the stream is
    # genuinely H.264; HEVC -> 'avcodec' (in-process LavfHevcSource, spec 2026-07-21).
    if codec == 'hevc':
        is_h264 = False
    elif codec == 'h264':
        is_h264 = True
    else:
        is_h264 = ext in EDGE_EXTS
    d = os.path.splitdrive(os.path.abspath(file_path))[0]
    letter = d[0].upper() if d else None
    if letter and letter in optical_letters:
        if letter in (mounted_iso_letters or set()):
            # Player-mounted ISO: a virtual volume has no optical head to
            # thrash. The in-process service reads it with its OWN demuxer,
            # DISARMED around init/seeks and throttled by the optical pacing
            # (OPTICAL_MIN_INTERVAL_S) — the guardrails built after the
            # 2026-07-14/22 concurrent-read breaks. Author decision
            # 2026-08-02: only a PHYSICAL disc skips thumbnails; ISOs must
            # produce them. is_optical=True keeps those guardrails engaged.
            return ('edge264', True) if is_h264 else ('avcodec', True)
        # Physical optical drive: the single head is already shared by mpv
        # (audio) and the video demuxer — a third reader causes measured
        # 45-120 s head thrash. The ONE case with no thumbnails.
        return 'off', True
    return ('edge264', False) if is_h264 else ('avcodec', False)


class TimeSlider(QSlider):
    """Custom slider with time preview on hover."""

    preview_requested = Signal(float)
    extraction_done = Signal(float, str)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hover_time = 0
        self._is_hovering = False
        self._player = None
        self._preview_widget = PreviewTooltip(self)
        self._last_preview_time = -99
        self._preview_cache = {}  # LRU cache (100 frames)
        self._video_file = None
        self._extraction_timer = None  # Lazy initialization
        self._timer_initialized = False
        self._pending_time = 0
        self._pending_mouse_x = 0
        self.extraction_done.connect(self._on_extraction_done)

    def _ensure_timer_initialized(self):
        """Initialize extraction timer in GUI thread when first needed"""
        if not self._timer_initialized:
            self._extraction_timer = QTimer(self)
            self._extraction_timer.setSingleShot(True)
            self._extraction_timer.timeout.connect(self._do_extraction)
            self._timer_initialized = True

    def enterEvent(self, event):
        super().enterEvent(event)
        self._is_hovering = True
        self.update()

    def set_player(self, player):
        self._player = player

    def mouseMoveEvent(self, event):
        if self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self._hover_time = max(0, min(value, self.maximum()))
            self._is_hovering = True

            s = int(self._hover_time)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            time_str = f"{h:02}:{m:02}:{s:02}"
            self.setToolTip(time_str)

            if self._video_file and abs(self._hover_time - self._last_preview_time) > 0.5:
                self._last_preview_time = self._hover_time
                self._request_on_demand_preview(self._hover_time, pos)

            self.update()
        super().mouseMoveEvent(event)

    def set_video_file(self, file_path: str, duration_seconds: float):
        """V7b+++++ PREVIEW FIX: Restore set_video_file for thumbnail preview.

        This method was mistakenly removed in a previous fix. It's required for
        the preview tooltip to work - without it, _video_file stays None and
        no thumbnails are extracted on hover.

        Args:
            file_path: Path to the video file (MKV, M2TS, etc.)
            duration_seconds: Video duration in seconds
        """
        self._video_file = file_path
        if duration_seconds > 0:
            self.setRange(0, int(duration_seconds))
        # Clear preview cache when video changes
        self._preview_cache.clear()
        self._last_preview_time = -99
        logger.info(f"[PREVIEW] Video file set: {file_path}, duration={duration_seconds:.1f}s")

    def _request_on_demand_preview(self, time_pos, mouse_x):
        cache_key = round(time_pos)
        if cache_key in self._preview_cache:
            pixmap = self._preview_cache[cache_key]
            if not pixmap.isNull():
                self._preview_widget.setPixmap(pixmap)
                self._show_preview_at(mouse_x)
                return

        self._pending_time = time_pos
        self._pending_mouse_x = mouse_x
        self._ensure_timer_initialized()  # Lazy timer creation
        self._extraction_timer.start(100)

    def _do_extraction(self):
        time_pos = self._pending_time
        mouse_x = self._pending_mouse_x
        future = _thumbnail_executor.submit(_extract_thumbnail_ffmpeg, self._video_file, time_pos)
        future.add_done_callback(lambda f: self._handle_extraction_result(f, time_pos, mouse_x))

    def _handle_extraction_result(self, future, time_pos, mouse_x):
        try:
            temp_file = future.result()
            if temp_file:
                self.extraction_done.emit(time_pos, temp_file)
        except:
            pass

    @Slot(float, str)
    def _on_extraction_done(self, time_pos, temp_file):
        try:
            cache_key = round(time_pos)
            pixmap = QPixmap(temp_file)
            if not pixmap.isNull():
                if len(self._preview_cache) > 100:
                    oldest = next(iter(self._preview_cache))
                    del self._preview_cache[oldest]
                self._preview_cache[cache_key] = pixmap
                if self._is_hovering and abs(time_pos - self._hover_time) < 3:
                    self._preview_widget.setPixmap(pixmap)
                    self._show_preview_at(self._pending_mouse_x)
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            print(f"[ERROR] {e}")

    def _show_preview_at(self, mouse_x):
        # FIX VIGNETTE : Lecture persistante de la propriété définie par le HUD
        if self.property('hud_mode'):
            self._preview_widget.hide()
            self.update()
            return

        global_pos = self.mapToGlobal(QPoint(int(mouse_x), 0))
        tooltip_x = global_pos.x() - self._preview_widget.width() // 2
        tooltip_y = global_pos.y() - self._preview_widget.height() - 10

        self._preview_widget.move(tooltip_x, tooltip_y)
        self._preview_widget.show()
        self._preview_widget.raise_()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._is_hovering = False
        self.setToolTip("")
        self._preview_widget.hide()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self.setValue(max(0, min(value, self.maximum())))
            self.sliderMoved.emit(self.value())
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_hovering and self.maximum() > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            preview_x = int((self._hover_time / self.maximum()) * self.width())
            painter.setPen(QPen(QColor(0, 122, 204, 180), 2))
            painter.drawLine(preview_x, 0, preview_x, self.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 122, 204, 220)))
            painter.drawEllipse(QPointF(preview_x, self.height() // 2), 5, 5)

__all__ = [
    'TimeSlider', '_decide_thumbs_mode', '_extract_thumbnail_ffmpeg',
]


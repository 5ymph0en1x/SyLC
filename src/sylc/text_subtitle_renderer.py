"""
TextSubtitleRenderer - Paints text subtitles (SRT/ASS) as an RGBA overlay.

In MVC/edge264 mode the video is presented by the native D3D11 renderer, not by
mpv (mpv runs audio-only: vid=no, vo=null), so mpv cannot draw text subtitles
itself. mpv still DECODES the selected text track against the shared audio
clock and exposes the current cue through its 'sub-text' property — this class
turns that text into an (H, W, 4) RGBA numpy image and emits it through the
same signal shape as SubtitleManager, so the existing set_subtitle()/
clear_subtitle() widget overlay path (built for PGS) renders it.
"""

import logging
import re

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen

logger = logging.getLogger(__name__)

# All coordinates are authored against this reference canvas; the display
# widget normalizes against it (same convention as PGS composition size).
REF_W = 1920
REF_H = 1080

# Residual markup defense: mpv's sub-text is already plain text, but some
# files carry literal HTML-ish tags in the cue payload itself.
_TAG_RE = re.compile(r'</?[a-zA-Z][^>{}]*>|\{\\[^}]*\}')


def _dedup_stereo_lines(lines):
    """Collapse per-eye duplicated subtitles to a single copy.

    3D SBS releases ship text tracks where EVERY cue exists twice — one event
    confined to the left half of the frame (MarginR=width/2), one to the right
    half (MarginL=width/2) — so raw-SBS playback shows one copy per eye. mpv's
    sub-text drops the margins and joins the simultaneous events with '\\n',
    which would paint the same text twice, stacked. Our overlay is already
    drawn once per eye by the renderer, so only one copy is wanted: when the
    line list is exactly a doubled sequence, keep the first half.
    """
    n = len(lines)
    if n >= 2 and n % 2 == 0 and lines[:n // 2] == lines[n // 2:]:
        return lines[:n // 2]
    return lines


class TextSubtitleRenderer(QObject):
    """Renders subtitle text to an RGBA overlay image (Qt main thread only)."""

    # SubtitleManager's signature + stereoscopic disparity:
    # (rgba_array, x, y, w, h, ref_w, ref_h, disparity) — disparity is the
    # authored horizontal parallax normalized to eye width (>0 = in front of
    # the screen), forwarded to the native renderer's per-eye overlay shift.
    subtitle_changed = Signal(object, int, int, int, int, int, int, float)
    subtitle_cleared = Signal()

    FONT_FAMILY = 'Segoe UI'
    FONT_PIXEL_SIZE = 58          # on the 1080-tall reference canvas
    OUTLINE_WIDTH = 6             # stroke diameter around glyphs
    BOTTOM_MARGIN = 52            # distance from canvas bottom to text block
    PADDING = 10                  # image padding so the outline is not clipped

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_text = None
        self._disparity = 0.0
        self._last_emit = None   # (img, x, y, w, h) — re-emit when depth arrives late

        self._font = QFont(self.FONT_FAMILY)
        self._font.setPixelSize(self.FONT_PIXEL_SIZE)
        self._font.setWeight(QFont.Weight.DemiBold)

    def set_disparity(self, disparity):
        """Set the stereoscopic depth (normalized eye-width, >0 = pop-out).

        The depth analysis runs asynchronously at track selection; if a cue is
        already on screen when the result lands, re-emit it with the new depth.
        """
        d = float(disparity)
        if d == self._disparity:
            return
        self._disparity = d
        if self._last_emit is not None:
            img, x, y, w, h = self._last_emit
            self.subtitle_changed.emit(img, x, y, w, h, REF_W, REF_H, d)

    @Slot(str)
    def set_text(self, text):
        """Display `text` (multi-line, '\\n' separated); empty/None clears."""
        text = _TAG_RE.sub('', text or '').strip()
        if text == self._last_text:
            return
        self._last_text = text

        if not text:
            self._last_emit = None
            self.subtitle_cleared.emit()
            return

        try:
            img, x, y = self._render(text)
        except Exception as e:
            logger.error(f"[TextSubtitle] render failed: {e}")
            self._last_emit = None
            self.subtitle_cleared.emit()
            return

        h, w = img.shape[:2]
        self._last_emit = (img, x, y, w, h)
        self.subtitle_changed.emit(img, x, y, w, h, REF_W, REF_H, self._disparity)

    def clear(self):
        self._last_text = None
        self._last_emit = None
        self.subtitle_cleared.emit()

    def _render(self, text):
        """Rasterize `text` -> (rgba HxWx4 uint8, x, y on the reference canvas)."""
        lines = [ln.strip() for ln in text.split('\n')]
        lines = [ln for ln in lines if ln] or ['']
        lines = _dedup_stereo_lines(lines)

        fm = QFontMetrics(self._font)
        line_h = fm.lineSpacing()
        widths = [max(1, fm.horizontalAdvance(ln)) for ln in lines]
        text_w = min(max(widths), REF_W - 2 * self.PADDING)
        img_w = text_w + 2 * self.PADDING
        img_h = line_h * len(lines) + 2 * self.PADDING

        qimg = QImage(img_w, img_h, QImage.Format.Format_RGBA8888)
        qimg.fill(Qt.GlobalColor.transparent)

        painter = QPainter(qimg)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            path = QPainterPath()
            for i, ln in enumerate(lines):
                lw = min(widths[i], text_w)
                lx = self.PADDING + (text_w - lw) / 2.0
                baseline = self.PADDING + i * line_h + fm.ascent()
                path.addText(lx, baseline, self._font, ln)
            pen = QPen(QColor(0, 0, 0, 235), self.OUTLINE_WIDTH,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(path, pen)
            painter.fillPath(path, QColor(255, 255, 255))
        finally:
            painter.end()

        # QImage rows are padded to bytesPerLine — slice the stride off.
        # .copy() is mandatory: without it the array can stay a VIEW over the
        # QImage buffer, which is freed when qimg goes out of scope.
        bpl = qimg.bytesPerLine()
        buf = np.frombuffer(qimg.constBits(), dtype=np.uint8,
                            count=img_h * bpl).reshape(img_h, bpl)
        rgba = buf[:, :img_w * 4].reshape(img_h, img_w, 4).copy()

        x = (REF_W - img_w) // 2
        y = REF_H - self.BOTTOM_MARGIN - img_h
        return rgba, x, max(0, y)

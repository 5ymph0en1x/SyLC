# -*- coding: utf-8 -*-
"""Small reusable Qt widgets used by the SyLC player window."""

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

class PreviewTooltip(QLabel):
    """Widget to display the frame preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(120, 68)  # 16:9 aspect ratio
        self.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #007ACC;
                border-radius: 4px;
            }
        """)
        self.setScaledContents(True)
        self.hide()


class IconButton(QPushButton):
    """Professional HDR Converter style button - Modern Redesign."""

    def __init__(self, icon_type, is_primary=False, parent=None):
        super().__init__(parent)
        self.icon_type = icon_type
        self.is_primary = is_primary

        if is_primary:
            self.setFixedSize(48, 48)
            self.setStyleSheet("""
                QPushButton {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #007ACC, stop:1 #0063A3);
                    border: 1px solid #0096FF;
                    border-radius: 24px;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0089E5, stop:1 #007ACC);
                    border: 1px solid #33Aaff;
                }
                QPushButton:pressed {
                    background-color: #004578;
                    margin-top: 1px; 
                }
            """)
        else:
            self.setFixedSize(38, 38)
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.15);
                    margin-top: 1px;
                }
                QPushButton:checked {
                    background-color: rgba(0, 122, 204, 0.3);
                    border: 1px solid #007ACC;
                }
            """)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Icon Color
        color = QColor(240, 240, 240)
        if not self.isEnabled():
            color = QColor(255, 255, 255, 80)

        # Thinner, more elegant stroke
        pen = QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        center_x = self.width() / 2
        center_y = self.height() / 2

        if self.icon_type == 'play':
            path = QPainterPath()
            # Refined play triangle
            path.moveTo(center_x - 3, center_y - 6)
            path.lineTo(center_x + 5, center_y)
            path.lineTo(center_x - 3, center_y + 6)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))

        elif self.icon_type == 'pause':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(center_x - 6, center_y - 6, 4, 12), 1, 1)
            painter.drawRoundedRect(QRectF(center_x + 2, center_y - 6, 4, 12), 1, 1)

        elif self.icon_type == 'stop':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(center_x - 5, center_y - 5, 10, 10), 2, 2)

        elif self.icon_type == 'folder':
            path = QPainterPath()
            path.moveTo(center_x - 8, center_y - 5)
            path.lineTo(center_x - 3, center_y - 5)
            path.lineTo(center_x - 1, center_y - 7)
            path.lineTo(center_x + 8, center_y - 7)
            path.lineTo(center_x + 8, center_y + 6)
            path.lineTo(center_x - 8, center_y + 6)
            path.closeSubpath()
            painter.strokePath(path, pen)

        elif self.icon_type == 'fullscreen':
            gap = 6
            len_ = 4
            # TL
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap + len_, center_y - gap))
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap, center_y - gap + len_))
            # TR
            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap - len_, center_y - gap))
            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap, center_y - gap + len_))
            # BL
            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap + len_, center_y + gap))
            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap, center_y + gap - len_))
            # BR
            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap - len_, center_y + gap))
            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap, center_y + gap - len_))

        elif self.icon_type == 'exit_fullscreen':
            gap = 7
            len_ = 4
            # TL (pointing in)
            painter.drawLine(QPointF(center_x - gap + len_, center_y - gap + len_),
                             QPointF(center_x - gap + len_, center_y - gap))
            painter.drawLine(QPointF(center_x - gap + len_, center_y - gap + len_),
                             QPointF(center_x - gap, center_y - gap + len_))
            # BR (pointing in)
            painter.drawLine(QPointF(center_x + gap - len_, center_y + gap - len_),
                             QPointF(center_x + gap - len_, center_y + gap))
            painter.drawLine(QPointF(center_x + gap - len_, center_y + gap - len_),
                             QPointF(center_x + gap, center_y + gap - len_))

        elif self.icon_type == '3d':
            font = QFont('Segoe UI', 9, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, '3D')

        elif self.icon_type == 'volume':
            path = QPainterPath()
            path.moveTo(center_x - 3, center_y - 2)
            path.lineTo(center_x - 1, center_y - 2)
            path.lineTo(center_x + 3, center_y - 5)
            path.lineTo(center_x + 3, center_y + 5)
            path.lineTo(center_x - 1, center_y + 2)
            path.lineTo(center_x - 3, center_y + 2)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))
            # Waves
            painter.setPen(pen)
            painter.drawArc(QRectF(center_x + 1, center_y - 3, 4, 6), -60 * 16, 120 * 16)
            painter.drawArc(QRectF(center_x + 1, center_y - 6, 8, 12), -55 * 16, 110 * 16)


class LoadingOverlay(QWidget):
    """Elegant loading animation overlay shown during file initialization."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Use Tool window to ensure it floats above native MPV window
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._status_text = "Initializing..."
        self._progress_angle = 0
        self._fade_opacity = 0.0
        self._is_showing = False
        self._progress_mode = False  # True = show progress arc, False = spinning
        self._progress_value = 0.0   # 0.0 to 1.0

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._update_animation)

        # Fade animation
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._update_fade)
        self._fade_direction = 1  # 1 = fade in, -1 = fade out

    def show_loading(self, status_text: str = "Initializing...", progress_mode: bool = False):
        """Show loading overlay with fade-in animation."""
        self._status_text = status_text
        self._is_showing = True
        self._fade_direction = 1
        self._progress_mode = progress_mode
        self._progress_value = 0.0
        self._anim_timer.start(16)  # ~60 FPS for smooth animation
        self._fade_timer.start(16)
        self.show()
        self.raise_()

    def hide_loading(self):
        """Hide loading overlay with fade-out animation."""
        self._fade_direction = -1
        self._fade_timer.start(16)
        self._progress_mode = False

    def set_status(self, text: str):
        """Update the status text."""
        self._status_text = text
        self.update()

    def set_progress(self, value: float):
        """Set progress value (0.0 to 1.0) - switches to progress mode."""
        self._progress_mode = True
        self._progress_value = max(0.0, min(1.0, value))
        self.update()

    def _update_animation(self):
        """Update spinner rotation."""
        self._progress_angle = (self._progress_angle + 6) % 360
        self.update()

    def _update_fade(self):
        """Update fade animation."""
        self._fade_opacity += self._fade_direction * 0.08

        if self._fade_opacity >= 1.0:
            self._fade_opacity = 1.0
            self._fade_timer.stop()
        elif self._fade_opacity <= 0.0:
            self._fade_opacity = 0.0
            self._fade_timer.stop()
            self._anim_timer.stop()
            self._is_showing = False
            self.hide()

        self.update()

    def paintEvent(self, event):
        if self._fade_opacity <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Semi-transparent background
        bg_alpha = int(180 * self._fade_opacity)
        painter.fillRect(self.rect(), QColor(18, 18, 18, bg_alpha))

        center_x = self.width() // 2
        center_y = self.height() // 2 - 30

        arc_alpha = int(255 * self._fade_opacity)
        arc_rect = QRectF(center_x - 25, center_y - 25, 50, 50)

        if self._progress_mode:
            # === PROGRESS MODE: Draw filling circle ===
            # Background circle (dark)
            bg_color = QColor(60, 60, 60, int(100 * self._fade_opacity))
            painter.setPen(QPen(bg_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(arc_rect)

            # Progress arc (blue) - starts at top (90°) and goes clockwise
            if self._progress_value > 0:
                arc_color = QColor(0, 122, 204, arc_alpha)
                painter.setPen(QPen(arc_color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                start_angle = 90 * 16  # Start at top (Qt uses 1/16th degree, 90° = top)
                span_angle = -int(self._progress_value * 360 * 16)  # Negative = clockwise
                painter.drawArc(arc_rect, start_angle, span_angle)

            # Percentage text in center
            percent_text = f"{int(self._progress_value * 100)}%"
            percent_font = QFont('Segoe UI', 11, QFont.Weight.Bold)
            painter.setFont(percent_font)
            painter.setPen(QColor(224, 224, 224, arc_alpha))
            fm_pct = QFontMetrics(percent_font)
            pct_width = fm_pct.horizontalAdvance(percent_text)
            pct_y = center_y + fm_pct.ascent() // 2 - 2
            painter.drawText(int(center_x - pct_width / 2), int(pct_y), percent_text)
        else:
            # === SPINNING MODE: Rotating arc ===
            arc_color = QColor(0, 122, 204, arc_alpha)
            pen = QPen(arc_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw arc spanning 270 degrees, rotating
            start_angle = self._progress_angle * 16  # Qt uses 1/16th degree
            span_angle = 270 * 16
            painter.drawArc(arc_rect, start_angle, span_angle)

            # Draw inner circle (subtle)
            inner_color = QColor(60, 60, 60, int(100 * self._fade_opacity))
            painter.setPen(QPen(inner_color, 1))
            painter.drawEllipse(QRectF(center_x - 18, center_y - 18, 36, 36))

        # Draw status text
        text_alpha = int(224 * self._fade_opacity)
        text_color = QColor(224, 224, 224, text_alpha)
        font = QFont('Segoe UI', 12, QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(text_color)

        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self._status_text)
        text_y = center_y + 60
        painter.drawText(int(center_x - text_width / 2), int(text_y), self._status_text)

        # Draw subtle hint text
        hint_text = "Please wait..."
        hint_alpha = int(140 * self._fade_opacity)
        hint_color = QColor(160, 160, 160, hint_alpha)
        hint_font = QFont('Segoe UI', 9, QFont.Weight.Normal)
        painter.setFont(hint_font)
        painter.setPen(hint_color)

        fm2 = QFontMetrics(hint_font)
        hint_width = fm2.horizontalAdvance(hint_text)
        painter.drawText(int(center_x - hint_width / 2), int(text_y + 24), hint_text)


class InfoOverlay(QWidget):
    """Elegant welcome message in the center of the window - clickable to open a file."""
    file_clicked = Signal()

    def __init__(self, text, parent=None):
        super().__init__(parent)

        # CRITICAL FIX: Use Tool window to ensure it floats above native MPV window
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pulse_timer = None  # Lazy initialization
        self._timer_initialized = False
        self._pulse_direction = -1
        self._pulse_value = 0.0
        self._hover = False
        self.setAcceptDrops(True)  # allow dropping a file onto the welcome area / icon too

    def _ensure_timer_initialized(self):
        """Initialize pulse timer in GUI thread when first needed"""
        if not self._timer_initialized:
            self._pulse_timer = QTimer(self)
            self._pulse_timer.timeout.connect(self._update_pulse)
            self._timer_initialized = True
        # Only start if visible
        if self.isVisible() and not self._pulse_timer.isActive():
            self._pulse_timer.start(30)

    def showEvent(self, event):
        """Start animation when shown."""
        super().showEvent(event)
        if self._timer_initialized and self._pulse_timer:
            self._pulse_timer.start(30)

    def hideEvent(self, event):
        """Stop animation when hidden to prevent unnecessary CPU usage."""
        if self._timer_initialized and self._pulse_timer:
            self._pulse_timer.stop()
        super().hideEvent(event)

    def _update_pulse(self):
        self._pulse_value += self._pulse_direction * 0.02
        if self._pulse_value <= 0.0:
            self._pulse_value = 0.0
            self._pulse_direction = 1
        elif self._pulse_value >= 1.0:
            self._pulse_value = 1.0
            self._pulse_direction = -1
        self.update()

    def paintEvent(self, event):
        self._ensure_timer_initialized()  # Lazy timer creation
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        center_x = self.width() // 2
        center_y = self.height() // 2 - 40

        icon_color = QColor(0, 122, 204, 200)
        path = QPainterPath()
        path.moveTo(center_x - 30, center_y - 12)
        path.lineTo(center_x - 10, center_y - 12)
        path.lineTo(center_x - 6, center_y - 20)
        path.lineTo(center_x + 30, center_y - 20)
        path.lineTo(center_x + 30, center_y + 20)
        path.lineTo(center_x - 30, center_y + 20)
        path.closeSubpath()
        # Fill the folder so its ENTIRE surface is clickable, not just the outline: a
        # translucent (WA_TranslucentBackground) window only receives mouse input on painted
        # pixels, so a hollow icon let clicks fall through its transparent interior. A gentle
        # pulse (brighter on hover) also signals that it is clickable.
        fill_alpha = 95 if self._hover else 42 + int(self._pulse_value * 26)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 122, 204, fill_alpha))
        painter.drawPath(path)
        painter.strokePath(path, QPen(icon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                      Qt.PenJoinStyle.RoundJoin))

        text_y = center_y + 60
        font = QFont('Segoe UI', 14, QFont.Weight.Normal)
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self.text)
        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - text_width / 2), int(text_y), self.text)

        subtitle = "MP4, MKV, AVI (3D & HDR)"
        subtitle_font = QFont('Segoe UI', 10, QFont.Weight.Normal)
        painter.setFont(subtitle_font)
        fm2 = QFontMetrics(subtitle_font)
        subtitle_width = fm2.horizontalAdvance(subtitle)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(int(center_x - subtitle_width / 2), int(text_y + 26), subtitle)

        app_title = "SyLC Player"
        title_font = QFont('Segoe UI', 24, QFont.Weight.Normal)
        painter.setFont(title_font)
        fm3 = QFontMetrics(title_font)
        title_width = fm3.horizontalAdvance(app_title)
        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - title_width / 2), 60, app_title)

        edition = "3D Edition"
        edition_font = QFont('Segoe UI', 9, QFont.Weight.Normal)
        painter.setFont(edition_font)
        fm4 = QFontMetrics(edition_font)
        edition_width = fm4.horizontalAdvance(edition)
        painter.setPen(QColor(0, 122, 204, 180))
        painter.drawText(int(center_x - edition_width / 2), 78, edition)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    event.acceptProposedAction()
                    parent = self.parent()
                    if parent is not None and hasattr(parent, 'play_file'):
                        parent.play_file(path)
        except Exception:
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.file_clicked.emit()
        super().mousePressEvent(event)


__all__ = [
    'IconButton', 'InfoOverlay', 'LoadingOverlay', 'PreviewTooltip',
]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring Overlay Widget for MAGMA Player
Displays real-time statistics with lazy timer initialization to prevent threading issues.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush


class MonitoringOverlay(QWidget):
    """
    Displays monitoring information over the video.
    Shows FPS, codec, bitrate, buffer status, etc.
    
    THREAD-SAFE: All timers are lazily initialized in the GUI thread.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Optimization: Use native window for overlay to bypass Qt software composition overhead
        # This allows the OpenGL widget underneath to render at full speed (VSync)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # self.setStyleSheet("background: transparent;") # Handled by paintEvent

        # Monitoring data
        self.decoder_fps = 0.0
        self.display_fps = 0.0
        self.buffer_size = 0
        self.drop_count = 0
        self.window_visible = False
        
        # No timers created in __init__ - lazy initialization
        self._update_timer = None
        self._timer_initialized = False

    def _ensure_timer_initialized(self):
        """Initialize update timer in GUI thread when first needed"""
        if not self._timer_initialized:
            self._update_timer = QTimer(self)
            self._update_timer.timeout.connect(self.update)
            self._timer_initialized = True

    def update_decoder_fps(self, fps):
        """Update decoder FPS"""
        self.decoder_fps = fps
        self._ensure_timer_initialized()
        self.update()

    def update_display_fps(self, fps):
        """Update display FPS"""
        self.display_fps = fps
        self._ensure_timer_initialized()
        self.update()

    def update_buffer(self, size, drops):
        """Update buffer statistics"""
        self.buffer_size = size
        self.drop_count = drops
        self._ensure_timer_initialized()
        self.update()

    def update_window_state(self, visible):
        """Update window visibility state"""
        self.window_visible = visible
        self._ensure_timer_initialized()
        self.update()

    def has_metrics(self):
        """Check if there are any metrics to display"""
        return self.decoder_fps > 0 or self.display_fps > 0 or self.buffer_size > 0

    def reset(self):
        """Reset all monitoring data"""
        self.decoder_fps = 0.0
        self.display_fps = 0.0
        self.buffer_size = 0
        self.drop_count = 0
        self.window_visible = False
        self.update()

    def paintEvent(self, event):
        """Draw monitoring overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Semi-transparent background
        bg_rect = QRectF(10, 10, 280, 140)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(43, 43, 43, 220)))
        painter.drawRoundedRect(bg_rect, 8, 8)

        # Border
        painter.setPen(QPen(QColor(0, 122, 204, 200), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bg_rect, 8, 8)

        # Text
        painter.setPen(QColor(224, 224, 224))
        font = QFont('Consolas', 10, QFont.Weight.Normal)
        painter.setFont(font)

        y_offset = 30
        painter.drawText(20, y_offset, "=== MVC DECODER ===")
        y_offset += 25
        painter.drawText(20, y_offset, f"Decoder FPS: {self.decoder_fps:.1f}")
        y_offset += 20
        painter.drawText(20, y_offset, f"Display FPS: {self.display_fps:.1f}")
        y_offset += 20
        painter.drawText(20, y_offset, f"Buffer: {self.buffer_size} frames")
        y_offset += 20
        painter.drawText(20, y_offset, f"Drops: {self.drop_count}")
        y_offset += 20
        
        # Window state indicator
        if self.window_visible:
            painter.setPen(QColor(0, 200, 100))
            painter.drawText(20, y_offset, "🎬 3D Window ACTIVE")
        else:
            painter.setPen(QColor(200, 100, 0))
            painter.drawText(20, y_offset, "⏸ 3D Window HIDDEN")
